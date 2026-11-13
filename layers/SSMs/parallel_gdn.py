import torch
import torch.nn.functional as F
from torch import nn

# Parallel deltaNet implementation, with parallel chunked blocks. Uses WY form + UT transform to fully parallelise
# Vanilla DeltaNet, not a GDN

class ParallelGatedDeltaNet(nn.Module):
    def __init__(self, D_input, D_output, D_qk, D_v) -> None:
        super().__init__()
        # Can specify qk, v, params differnetly for sake of curiosity
        self.D_input = D_input
        self.D_output = D_output

        self.D_qk = D_qk
        self.D_v = D_v

        # Initialise W_k, W_q and W_v
        self.W_q = nn.Parameter(torch.randn((D_input, D_qk)) * (1.0 / (D_input ** 0.5)))
        self.W_k = nn.Parameter(torch.randn((D_input, D_qk)) * (1.0 / (D_input ** 0.5)))
        self.W_v = nn.Parameter(torch.randn((D_input, D_v)) * (1.0 / (D_input ** 0.5)))

        # Output projection
        self.W_o = nn.Parameter(torch.randn((D_v, D_output)) * (1.0 / (D_v ** 0.5)))

        # Write strength - beta(common to all)
        self.W_b = nn.Parameter(torch.randn((D_input, 1))) # Takes (B,T,C) -> (B,T), which is then broadcast

        # Alpha for decay
        self.W_a = nn.Parameter(torch.randn((D_input)))

    def legacy_out(self, x):
        # S_t = alpha * S_(t-1) + beta * k_t * Error  // Write the error to correct address, (K) @ (V)-> (K, V) correct
        # Error = v_t - k_t . S_(t-1)
        # Solve this equation you get(equivalent)
        # S_(t+1) = alpha * (I - beta * k_t^T * k_t) * S_t + beta * k_t * v_t
        B,T,C = x.shape
        S = torch.zeros((B, self.D_qk, self.D_v), device=x.device, dtype=x.dtype)
        self.decay = "scalar"
        output = []
        for t in range(T):
            input = x[:,t,:]
            q = F.normalize(torch.einsum('bc,cq->bq', input, self.W_q))
            k = F.normalize(torch.einsum('bc,ck->bk', input, self.W_k))
            v = torch.einsum('bc,cv->bv', input, self.W_v)

            # alpha and beta
            if self.decay != None:
                alpha = torch.sigmoid(input @ self.W_a) # (B,1) for scalar or (B,K) for vector
            beta = (torch.sigmoid(input @ self.W_b)).view(B,1,1) # So we can multiply cleanly by (B,K,K)

            # Find A
            A = (torch.eye(self.D_qk, device=x.device, dtype=x.dtype).unsqueeze(dim=0) - beta * torch.einsum('bk,bs->bks', k, k))
            if self.decay == 'scalar':
                # alpha is scalar
                A = alpha.view(B,1,1) * A # type:ignore

            # Find B
            B_in = beta * torch.einsum('bk,bv->bkv', k, v)

            # Write state update
            if self.decay == 'vector':
                S = alpha.unsqueeze(dim=-1) * torch.einsum('bsk,bkv->bsv', A, S) + B_in # type:ignore
            else:
                S = torch.einsum('bsk,bkv->bsv', A, S) + B_in

            # Output proj
            O = torch.einsum('bkv,bk->bv', S, q)

            output.append(O @ self.W_o)
        return torch.stack(output, dim=1)

    def forward(self, x, chunk_size = 32, is_grad_check = False):
        # Some operations aren't very good for kernels, but just implement a parallel version rn
        bsize,T,_ = x.shape
        S_in = torch.zeros((bsize, self.D_qk, self.D_v), device=x.device, dtype=x.dtype)
        outputs = []
        for cs in range(0, T, chunk_size):
            input = x[:, cs:cs+chunk_size]
            lc = input.shape[1]
            # 0. q,k,v
            Q = F.normalize(torch.einsum('bsc,cq->bsq', input, self.W_q), dim=-1)
            K = F.normalize(torch.einsum('bsc,ck->bsk', input, self.W_k), dim=-1)
            V = torch.einsum('bsc,cv->bsv', input, self.W_v)
            # 1. Get D
            D = torch.sigmoid(torch.einsum('bsc,ck->bsk', input, self.W_b)) # (B, C, 1)
            # 2. Alpha, g and G
            # Use log form because products underflow to zero, add clamps if it still breaks
            Alpha = torch.sigmoid(input @ self.W_a) # (B, C, D_input) @ (D_input) -> B, C
            log_alpha = torch.log(Alpha)
            log_g = torch.cumsum(log_alpha, dim=-1)
            g = torch.exp(log_g) # (B, C)
            G = (log_g.unsqueeze(dim=-1) - log_g.unsqueeze(dim=-2)) # (B, C, 1) / (B, 1, C) -> B, C, C
            G = torch.tril(G) # B, C, C
            G = torch.exp(G)
            # 3. W and U
            # W is same as before
            g_matrix = torch.tril(torch.einsum('bsk, btk->bst', K, K), diagonal=-1) # OG matrix named after Scandinavian guy
            A_w = torch.eye(lc, device=x.device, dtype=x.dtype) + torch.einsum('bs,bst->bst', D.squeeze(dim=-1), g_matrix)
            if is_grad_check:
                cast_type = torch.float64
            else:
                cast_type = torch.float32
            # MPS triangular solve is fp32-only. Accumulate there, store in x's dtype.
            W = torch.linalg.solve_triangular(
                A_w.to(cast_type), (D * K).to(cast_type), upper=False, unitriangular=True,
            ).to(dtype=x.dtype)
            # U is different
            decayed_G = G * g_matrix # Both are (C,C)
            A_u = torch.eye(lc, device=x.device, dtype=x.dtype) + torch.einsum('bs,bst->bst', D.squeeze(dim=-1), decayed_G)
            U = torch.linalg.solve_triangular(
                A_u.to(cast_type), (D * V).to(cast_type), upper=False, unitriangular=True,
            ).to(dtype=x.dtype)
            # 4. Output projection
            E = U - torch.einsum('bck,bkv->bcv', g.unsqueeze(-1) * W, S_in)
            chunk_scores = G * torch.tril(torch.einsum('bck,bsk->bcs', Q, K))
            carry_out = torch.einsum('bsk,bkv->bsv',(g.unsqueeze(dim=-1) * Q), S_in)
            O = carry_out + chunk_scores @ E
            # 5. Carry state
            E_carry = G[:, -1].unsqueeze(dim=-1) * E
            S_in = g[:, -1].view(bsize, 1, 1) * S_in + torch.einsum('bsk, bsv->bkv',K, E_carry)

            # Append to outputs
            outputs.append(torch.einsum('bcv,vo->bco', O, self.W_o))

        return torch.cat(outputs, dim=1)


    def grad_check(self, o1:torch.Tensor, o2:torch.Tensor):
        from torch.autograd.gradcheck import gradcheck
        l1 = o1.view(-1).sum()
        l1.backward()
        params = [i for i in self.parameters()]
        gradients1 = [x.grad for x in params]
        for param in self.parameters():
            assert type(param) == nn.Parameter
            param.grad = None
        l2 = o2.view(-1).sum()
        l2.backward()
        gradients2 = [x.grad for x in params]
        for i in range(len(gradients1)):
            assert gradients1[i] is not None and gradients2[i] is not None
            assert torch.allclose(gradients1[i], gradients2[i], atol=1e-4, rtol=1e-3) # type:ignore
            print(i,torch.abs(gradients1[i] - gradients2[i]).max() , "Yay!") # type:ignore
        print("Gradients match")
        ## Do gradcheck
        gradcheck(get_forward(), (torch.randn((1, 3, 5), dtype=torch.float64, requires_grad=True )))
        print("Gradcheck passed")

def get_forward():
    D_net = ParallelGatedDeltaNet(D_input = 5, D_output = 32, D_qk = 32, D_v = 32).to(torch.float64)
    return lambda data: D_net(data, is_grad_check=True)

if __name__ == '__main__':
    import time
    data = torch.randn(4, 64, 128)
    D_net = ParallelGatedDeltaNet(D_input = 128, D_output = 32, D_qk = 32, D_v = 32)
    t1 = time.time()
    outputs = D_net(data)
    t2 = time.time()
    ref_outs = D_net.legacy_out(data)
    t3 = time.time()
    # print(f"time taken by sequential: {t3-t2}, time taken by parallel chunked: {t2 - t1}, speedup: {(t3-t2)/(t2-t1)}")
    # print(outputs.shape, ref_outs.shape)
    # print(torch.allclose(outputs, ref_outs, atol=1e-4, rtol=1e-3), torch.abs(outputs-ref_outs).max())

    # grad
    D_net.grad_check(outputs, ref_outs)
