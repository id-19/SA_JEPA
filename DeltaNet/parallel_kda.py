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
        self.W_b = nn.Parameter(torch.randn(D_input)) # Takes (B,T,C) -> (B,T), which is then broadcast

        # Alpha for decay
        self.W_a = nn.Parameter(torch.randn((D_input, D_qk))/ (D_input**0.5))

    def legacy_out(self, x):
        # S_t = alpha * S_(t-1) + beta * k_t * Error  // Write the error to correct address, (K) @ (V)-> (K, V) correct
        # Error = v_t - k_t . S_(t-1)
        # Solve this equation you get(equivalent)
        # S_(t+1) = alpha * (I - beta * k_t^T * k_t) * S_t + beta * k_t * v_t
        B,T,C = x.shape
        S = torch.zeros((B, self.D_qk, self.D_v))
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
            A = (torch.eye(self.D_qk).unsqueeze(dim=0) - beta * torch.einsum('bk,bs->bks', k, k))

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

    def forward(self, x, chunk_size = 16):
        # Numerically stable chunked parallel form.
        # Decay ratios are computed as exp(clamped log-differences) instead of
        # dividing by a possibly underflowed-to-zero cumulative decay D.
        bsize,T,_ = x.shape
        S_in = torch.zeros((bsize, self.D_qk, self.D_v))
        outputs = []
        for cs in range(0, T, chunk_size):
            input = x[:, cs:cs+chunk_size]
            lc = input.shape[1]
            # 0. q,k,v
            Qraw = Fn.normalize(torch.einsum('bsc,cq->bsq', input, self.W_q), dim=-1)
            K = Fn.normalize(torch.einsum('bsc,ck->bsk', input, self.W_k), dim=-1)
            V = torch.einsum('bsc,cv->bsv', input, self.W_v)
            # 1. B
            B = torch.sigmoid(input @ self.W_b)  # (B, L)
            # 2. Alpha / cumulative decay in log space
            raw_alpha = input @ self.W_a  # (B, L, K)
            log_alpha = Fn.logsigmoid(raw_alpha)
            log_d = torch.cumsum(log_alpha, dim=1)  # (B, L, K)
            D = torch.exp(log_d.clamp(min=-88.0))  # (B, L, K)

            # exp(log_d[t] - log_d[s]), clamped for stability
            diff_ls = (log_d.unsqueeze(2) - log_d.unsqueeze(1)).clamp(min=-88.0, max=0.0)  # (B, L, L, K)
            weight = torch.exp(diff_ls)

            # 3. G for the triangular solve
            G = (K.unsqueeze(2) * K.unsqueeze(1) * weight).sum(dim=-1)  # (B, L, L)

            Kdash = K * D  # (B, L, K)
            M_minus = torch.tril(torch.ones((lc, lc)), diagonal=-1)
            Lmat = torch.eye(lc) + M_minus.view(1, lc, lc) * (B.view(bsize, lc, 1) * G)

            # 4. Solve for W and U
            rhs = B.unsqueeze(dim=-1) * torch.cat((Kdash, V), dim=-1)  # (B, L, K+V)
            solve = tri_solve(Lmat, rhs, upper=False, unitriangular=True)
            W, U = solve.split([self.D_qk, self.D_v], dim=-1)

            # 5. Scores
            scores = (Qraw.unsqueeze(2) * K.unsqueeze(1) * weight).sum(dim=-1)  # (B, L, L)
            M = torch.tril(torch.ones((lc, lc)))
            scores = M.view(1, lc, lc) * scores

            E = U - torch.einsum('btk,bkv->btv', W, S_in)  # (B, L, V)
            Qsc = D * Qraw  # (B, L, K)
            O = torch.einsum('btq,bqv->btv', Qsc, S_in) + scores @ E  # (B, L, V)

            # 6. Carry state (stable)
            last_log = log_d[:, -1, :]  # (B, K)
            w = torch.exp((last_log.unsqueeze(1) - log_d).clamp(min=-88.0, max=0.0))  # (B, L, K)
            S_in = D[:, -1, :].view(bsize, self.D_qk, 1) * S_in + torch.einsum('blk,blv->bkv', K * w, E)

            # Append output
            outputs.append(torch.einsum('bcv,vo->bco', O, self.W_o))

        return torch.cat(outputs, dim=1)


if __name__ == '__main__':
    import time
    data = torch.randn(2, 3, 5)
    D_net = ParallelKDA(D_input=5, D_output=7, D_qk=11, D_v=13)
    t1 = time.time()
    outputs = D_net(data)
    t2 = time.time()
    ref_outs = D_net.legacy_out(data)
    t3 = time.time()
    print(f"time taken by sequential: {t3-t2}, time taken by parallel chunked: {t2 - t1}, speedup: {(t3-t2)/(t2-t1)}")
    print(outputs.shape, ref_outs.shape)
    print(torch.allclose(outputs, ref_outs))
