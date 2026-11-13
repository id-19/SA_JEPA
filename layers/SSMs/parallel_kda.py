import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.linalg import solve_triangular as tri_solve

# Parallel deltaNet implementation, with parallel chunked blocks. Uses WY form + UT transform to fully parallelise
# Vanilla DeltaNet, not a GDN

class ParallelKDA(nn.Module):
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
        self.W_b = nn.Parameter(torch.randn(D_input) / (D_input ** 0.5)) # Takes (B,T,C) -> (B,T), which is then broadcast

        # Alpha for decay
        self.W_a = nn.Parameter(torch.randn((D_input, D_qk))/ (D_input**0.5))

    def legacy_out(self, x):
        # S_t = alpha * S_(t-1) + beta * k_t * Error  // Write the error to correct address, (K) @ (V)-> (K, V) correct
        # Error = v_t - k_t . S_(t-1)
        # Solve this equation you get(equivalent)
        # S_(t+1) = alpha * (I - beta * k_t^T * k_t) * S_t + beta * k_t * v_t
        B,T,C = x.shape
        S = torch.zeros((B, self.D_qk, self.D_v), device=x.device, dtype=x.dtype)
        output = []
        for t in range(T):
            input = x[:,t,:]
            q = Fn.normalize(torch.einsum('bc,cq->bq', input, self.W_q))
            k = Fn.normalize(torch.einsum('bc,ck->bk', input, self.W_k))
            v = torch.einsum('bc,cv->bv', input, self.W_v)

            # alpha and beta
            alpha = torch.sigmoid(input @ self.W_a) # (B,1) for scalar or (B,K) for vector
            beta = (torch.sigmoid(input @ self.W_b)).view(B,1,1) # So we can multiply cleanly by (B,K,K)

            # Find A
            A = (torch.eye(self.D_qk, device=x.device, dtype=x.dtype).unsqueeze(dim=0) - beta * torch.einsum('bk,bs->bks', k, k))

            # Write state update
            S_decay = alpha.unsqueeze(-1) * S

            error = v - torch.einsum('bk,bkv->bv', k, S_decay)

            S = S_decay + beta * (
                k.unsqueeze(-1) * error.unsqueeze(1)
            )

            # Output proj
            O = torch.einsum('bkv,bk->bv', S, q)

            output.append(O @ self.W_o)
        return torch.stack(output, dim=1)

    def forward(self, x, chunk_size = 32):
        # Numerically stable chunked parallel form.
        # Decay ratios are computed as exp(clamped log-differences) instead of
        # dividing by a possibly underflowed-to-zero cumulative decay D.
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
            Qraw = Fn.normalize(torch.einsum('bsc,cq->bsq', input, W_q), dim=-1)
            K = Fn.normalize(torch.einsum('bsc,ck->bsk', input, W_k), dim=-1)
            V = torch.einsum('bsc,cv->bsv', input, W_v)
            # 1. B
            B = torch.sigmoid(input @ W_b)  # (B, L)
            # 2. Alpha / cumulative decay in log space
            raw_alpha = input @ W_a  # (B, L, K)
            log_alpha = Fn.logsigmoid(raw_alpha)
            log_d = torch.cumsum(log_alpha, dim=1)  # (B, L, K)
            D = torch.exp(log_d.clamp(min=-88.0))  # (B, L, K)

            # exp(log_d[t] - log_d[s]), clamped for stability
            diff_ls = (log_d.unsqueeze(2) - log_d.unsqueeze(1)).clamp(min=-88.0, max=0.0)  # (B, L, L, K)
            weight = torch.exp(diff_ls)

            # 3. G for the triangular solve
            G = torch.einsum('blk,bsk,blsk->bls', K, K, weight)  # (B, L, L)

            Kdash = K * D  # (B, L, K)
            M_minus = torch.tril(torch.ones((lc, lc), device=x.device, dtype=x.dtype), diagonal=-1)
            Lmat = torch.eye(lc, device=x.device, dtype=x.dtype) + M_minus.view(1, lc, lc) * (B.view(bsize, lc, 1) * G)

            # 4. Solve for W and U
            rhs = B.unsqueeze(dim=-1) * torch.cat((Kdash, V), dim=-1)  # (B, L, K+V)
            # MPS triangular solve is fp32-only; x is already at least fp32 here
            solve = tri_solve(Lmat, rhs, upper=False, unitriangular=True)
            W, U = solve.split([self.D_qk, self.D_v], dim=-1)

            # 5. Scores
            scores = torch.einsum('blk,bsk,blsk->bls', Qraw, K, weight)  # (B, L, L)
            M = torch.tril(torch.ones((lc, lc), device=x.device, dtype=x.dtype))
            scores = M.view(1, lc, lc) * scores

            E = U - torch.einsum('btk,bkv->btv', W, S_in)  # (B, L, V)
            Qsc = D * Qraw  # (B, L, K)
            O = torch.einsum('btq,bqv->btv', Qsc, S_in) + scores @ E  # (B, L, V)

            # 6. Carry state (stable)
            last_log = log_d[:, -1, :]  # (B, K)
            w = torch.exp((last_log.unsqueeze(1) - log_d).clamp(min=-88.0, max=0.0))  # (B, L, K)
            S_in = D[:, -1, :].view(bsize, self.D_qk, 1) * S_in + torch.einsum('blk,blv->bkv', K * w, E)

            # Append output
            outputs.append(torch.einsum('bcv,vo->bco', O, W_o))

        return torch.cat(outputs, dim=1).to(out_dtype)


def grad_check():
    D_output = 8
    alltype = torch.float64
    x = torch.randn((1,64,64), dtype=alltype, requires_grad=True)
    model = ParallelKDA(D_input = 64, D_output = D_output, D_qk = 4, D_v = 4).to(alltype)
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
        assert torch.allclose(grads1[i],grads2[i], rtol=1e-4, atol=1e-10) # type:ignore
    # Now, for the grad_check
    from torch.autograd.gradcheck import gradcheck
    forward = lambda x: model(x)
    gradcheck(forward, x)
    print("gradcheck passed!")


def finite_check(T = 4096, D_input = 1024, trials = 3):
    # fp32 gradients stay finite over long sequences (ID1-279)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    D_output = 64
    model = ParallelKDA(D_input = D_input, D_output = D_output, D_qk = 128, D_v = 128).to(device)
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
