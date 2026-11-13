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
        self.W_a = nn.Parameter(torch.randn((D_input)) / (D_input ** 0.5))

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

    def forward(self, x, chunk_size = 32):
        # Some operations aren't very good for kernels, but just implement a parallel version rn
        # Accumulate in at least fp32 (fp64 stays fp64 for gradcheck); cast back on the way out
        out_dtype = x.dtype
        cdtype = torch.promote_types(x.dtype, torch.float32)
        x = x.to(cdtype)
        W_q, W_k, W_v, W_b, W_a, W_o = (p.to(cdtype) for p in (self.W_q, self.W_k, self.W_v, self.W_b, self.W_a, self.W_o))
        bsize,T,_ = x.shape
        S_in = torch.zeros((bsize, self.D_qk, self.D_v), device=x.device, dtype=x.dtype)
        outputs = []
        for cs in range(0, T, chunk_size):
            input = x[:, cs:cs+chunk_size]
            lc = input.shape[1]
            # 0. q,k,v
            Q = F.normalize(torch.einsum('bsc,cq->bsq', input, W_q), dim=-1)
            K = F.normalize(torch.einsum('bsc,ck->bsk', input, W_k), dim=-1)
            V = torch.einsum('bsc,cv->bsv', input, W_v)
            # 1. Get D
            D = torch.sigmoid(torch.einsum('bsc,ck->bsk', input, W_b)) # (B, C, 1)
            # 2. Alpha, g and G
            # Use log form because products underflow to zero, add clamps if it still breaks
            Alpha = input @ W_a # (B, C, D_input) @ (D_input) -> B, C
            log_alpha = torch.nn.functional.logsigmoid(Alpha)
            log_g = torch.cumsum(log_alpha, dim=-1)
            g = torch.exp(log_g) # (B, C)
            G = (log_g.unsqueeze(dim=-1) - log_g.unsqueeze(dim=-2)) # (B, C, 1) / (B, 1, C) -> B, C, C
            G = torch.tril(G) # B, C, C
            G = torch.exp(G.clamp(-88.0, 88.0))
            # 3. W and U
            # W is same as before
            g_matrix = torch.tril(torch.einsum('bsk, btk->bst', K, K), diagonal=-1) # OG matrix named after Scandinavian guy
            A_w = torch.eye(lc, device=x.device, dtype=x.dtype) + torch.einsum('bs,bst->bst', D.squeeze(dim=-1), g_matrix)
            # MPS triangular solve is fp32-only; x is already at least fp32 here
            W = torch.linalg.solve_triangular(A_w, D * K, upper=False, unitriangular=True)
            # U is different
            decayed_G = G * g_matrix # Both are (C,C)
            A_u = torch.eye(lc, device=x.device, dtype=x.dtype) + torch.einsum('bs,bst->bst', D.squeeze(dim=-1), decayed_G)
            U = torch.linalg.solve_triangular(A_u, D * V, upper=False, unitriangular=True)
            # 4. Output projection
            E = U - torch.einsum('bck,bkv->bcv', g.unsqueeze(-1) * W, S_in)
            chunk_scores = G * torch.tril(torch.einsum('bck,bsk->bcs', Q, K))
            carry_out = torch.einsum('bsk,bkv->bsv',(g.unsqueeze(dim=-1) * Q), S_in)
            O = carry_out + chunk_scores @ E
            # 5. Carry state
            E_carry = G[:, -1].unsqueeze(dim=-1) * E
            S_in = g[:, -1].view(bsize, 1, 1) * S_in + torch.einsum('bsk, bsv->bkv',K, E_carry)

            # Append to outputs
            outputs.append(torch.einsum('bcv,vo->bco', O, W_o))

        return torch.cat(outputs, dim=1).to(out_dtype)


def grad_check():
    D_output = 8
    alltype = torch.float64
    x = torch.randn((1,64,64), dtype=alltype, requires_grad=True)
    model = ParallelGatedDeltaNet(D_input = 64, D_output = D_output, D_qk = 4, D_v = 4).to(alltype)
    get_grads = lambda :[p.grad for p in [x, *model.parameters()]] # input grad too
    random_output_weights = torch.randn((1,64, D_output), dtype=alltype)
    (model(x, chunk_size = 8) * random_output_weights).view(-1).sum().backward()
    grads1 = get_grads()
    model.zero_grad()
    x.grad = None
    (model.legacy_out(x) * random_output_weights).view(-1).sum().backward()
    grads2 = get_grads()
    # Manual
    for i in range(len(grads1)):
        assert torch.allclose(grads1[i],grads2[i], rtol=1e-8, atol=1e-10) # type:ignore
    print("Gradients match")
    # Now, for the grad_check
    from torch.autograd.gradcheck import gradcheck
    forward = lambda x: model(x)
    gradcheck(forward, x)
    print("gradcheck passed!")

def finite_check(T = 4096, D_input = 1024, trials = 3):
    # fp32 gradients stay finite over long sequences (ID1-279). Large D_input is the case that broke log(sigmoid)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    D_output = 64
    model = ParallelGatedDeltaNet(D_input = D_input, D_output = D_output, D_qk = 128, D_v = 128).to(device)
    for _ in range(trials):
        x = torch.randn((2, T, D_input), device=device, requires_grad=True)
        random_output_weights = torch.randn((2, T, D_output), device=device)
        (model(x) * random_output_weights).view(-1).sum().backward()
        for p in [x, *model.parameters()]:
            assert torch.isfinite(p.grad).all() # type:ignore
            p.grad = None
    print(f"fp32 grads finite at T={T} on {device}")

if __name__ == '__main__':
    grad_check()
    finite_check()
