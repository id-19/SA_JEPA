import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.linalg import solve_triangular as tri_solve

# [1.2] ID1-280: multi-head chunked KDA. Starts as a copy of single_headed/parallel_kda.py;
# add the head axis and nothing else. Contract the tests copy weights through (D_qk, D_v are total widths,
# head h owns columns h*D_qk//n_heads : (h+1)*D_qk//n_heads, likewise for v and the rows of W_o):
#   W_q, W_k, W_a: (D_input, D_qk)    W_v: (D_input, D_v)    W_o: (D_v, D_output)
#   W_b:           (D_input, n_heads)
# Tests: eval_test/tests/test_multihead_kda.py

class MultiHeadParallelKDA(nn.Module):
    def __init__(self, D_input, D_output, D_qk, D_v, n_heads, kernel_size = 4) -> None:
        super().__init__()
        # Can specify qk, v, params differnetly for sake of curiosity
        self.D_input = D_input
        self.D_output = D_output
        self.n_heads = n_heads

        self.D_qk = D_qk
        self.D_v = D_v
        self.hdim_qk = D_qk // n_heads
        self.hdim_v = D_v // n_heads

        self.kernel_size = kernel_size

        # Initialise W_k, W_q and W_v
        self.W_q = nn.Parameter(torch.randn((D_input, D_qk)) * (1.0 / (D_input ** 0.5)))
        self.W_k = nn.Parameter(torch.randn((D_input, D_qk)) * (1.0 / (D_input ** 0.5)))
        self.W_v = nn.Parameter(torch.randn((D_input, D_v)) * (1.0 / (D_input ** 0.5)))

        # Initialise convs
        self.Q_conv = nn.Conv1d(D_qk, D_qk, groups=D_qk, kernel_size=kernel_size)
        self.K_conv = nn.Conv1d(D_qk, D_qk, groups=D_qk, kernel_size=kernel_size)
        self.V_conv = nn.Conv1d(D_v, D_v, groups=D_v, kernel_size=kernel_size)

        # SiLU
        self.silu = nn.SiLU()

        # Output projection
        self.W_o = nn.Parameter(torch.randn((D_v, D_output)) * (1.0 / (D_v ** 0.5)))

        # Write strength - beta, one per head
        self.W_b = nn.Parameter(torch.randn((D_input, n_heads)) / (D_input ** 0.5)) # Takes (B,T,C) -> (B,T,H)

        # Alpha for decay, one per key channel; split into (H, hdim_qk) like K
        self.W_a = nn.Parameter(torch.randn((D_input, D_qk))/ (D_input**0.5))


    def legacy_out(self, x, return_state = False, q_conv_in=None, k_conv_in = None, v_conv_in = None):
        # S_t = alpha * S_(t-1) + beta * k_t * Error  // Write the error to correct address, (K) @ (V)-> (K, V) correct
        # Error = v_t - k_t . S_(t-1)
        # Solve this equation you get(equivalent)
        # S_(t+1) = alpha * (I - beta * k_t^T * k_t) * S_t + beta * k_t * v_t
        B,T,_ = x.shape
        S = torch.zeros((B, self.n_heads, self.hdim_qk, self.hdim_v), device=x.device, dtype=x.dtype) # (B, H, hdim_qk, hdim_v)
        Q,K,V,q_conv_in, k_conv_in, v_conv_in = self.get_qkv(x, q_conv_in, k_conv_in, v_conv_in) # (B, H, T, hdim)
        output = []
        for t in range(T):
            input = x[:,t,:]
            q = Q[:, :, t]
            k = K[:, :, t]
            v = V[:, :, t]
            # alpha and beta
            alpha = torch.sigmoid(input @ self.W_a).view(B, self.n_heads, self.hdim_qk)
            beta = torch.sigmoid(input @ self.W_b) # (B, H), unsqueezed where used

            # Write state update, decaying each key row of S by its own alpha
            S_decay = alpha.unsqueeze(-1) * S

            error = v - torch.einsum('bhk,bhkv->bhv', k, S_decay)

            S = S_decay + beta.view(B, self.n_heads, 1, 1) * torch.einsum('bhk,bhv->bhkv', k, error)

            # Output proj
            O = torch.einsum('bhkv,bhk->bhv', S, q)
            O = O.view(B, self.D_v)

            output.append(O @ self.W_o)
        return (torch.stack(output, dim=1), S, q_conv_in, k_conv_in, v_conv_in) if return_state else torch.stack(output, dim=1)

    def get_qkv(self, input, q_conv_in=None, k_conv_in = None, v_conv_in = None):
        bsize, T, _ = input.shape
        # Q and K are entered as (B, T, d_qk) here, before the whole thing
        q = input @ self.W_q
        k = input @ self.W_k
        v = input @ self.W_v
        # First transpose
        q = q.transpose(-1, -2)
        # print("Q dims after transpose:", q.shape)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        # Cat with input conv
        # Along the time dimension(last dimension)
        if q_conv_in is not None:
            q = torch.cat((q_conv_in, q), dim=-1)
        if k_conv_in is not None:
            k = torch.cat((k_conv_in, k), dim=-1)
        if v_conv_in is not None:
            v = torch.cat((v_conv_in, v), dim=-1)
        # Left pad for conv
        # q,k,v have same timesteps
        # print("q dims after some messery:", q.shape)
        padding_needed = (T + self.kernel_size - 1) - q.shape[-1]
        # print(padding_needed)
        q = Fn.pad(q, (padding_needed, 0))
        # print("q dims after messery:", q.shape)
        k = Fn.pad(k, (padding_needed, 0))
        v = Fn.pad(v, (padding_needed, 0))
        # Get carry
        q_conv_carry = q[:, :, T:]
        k_conv_carry = k[:, :, T:]
        v_conv_carry = v[:, :, T:]
        # Apply conv
        q = self.Q_conv(q)
        k = self.K_conv(k)
        v = self.V_conv(v)

        # Transpose back
        q = q.transpose(-1, -2)
        k = k.transpose(-1, -2)
        v = v.transpose(-1, -2)
        # Apply SiLU() - goes on last dim, saves computation
        q = self.silu(q)
        k = self.silu(k)
        v = self.silu(v)
        # Reshape and transpose for head dim - B, H, T, hdim
        q = q.view(bsize, T, self.n_heads, self.hdim_qk).transpose(-2, -3)
        k = k.view(bsize, T, self.n_heads, self.hdim_qk).transpose(-2, -3)
        v = v.view(bsize, T, self.n_heads, self.hdim_v).transpose(-2, -3)
        # Apply L2 norm to q, k- q / max(|q|)
        q = Fn.normalize(q, dim=-1)
        k = Fn.normalize(k, dim=-1)

        return (q,k,v, q_conv_carry, k_conv_carry, v_conv_carry)

    def forward(self, x, state=None, return_state=False, chunk_size = 32):
        # state: None for a fresh sequence, else the tuple (S, q_conv_in, k_conv_in, v_conv_in) a previous call returned
        bsize,T,_ = x.shape

        # Numerically stable chunked parallel form.
        # Decay ratios are computed as exp(clamped log-differences) instead of
        # dividing by a possibly underflowed-to-zero cumulative decay D.
        # Everything runs in x.dtype (the layer's weights must be in the same dtype) except three places, which
        # run in at least fp32 (fp64 stays fp64 for gradcheck):
        #   1. the log-decay cumsum and what is built from it (D, weight): bf16 rounds 0.999 to 1.0 and the cumsum drifts
        #   2. the triangular solve (Lmat, rhs): ill-conditioning feeds errors into the state; MPS solve is fp32-only
        #   3. the carried state S_in and everything that reads or writes it (E, O)
        # The Q side (Qraw, scores) stays in x.dtype. einsum and @ do not promote mixed dtypes, so casts are explicit
        dt = x.dtype
        ft = torch.promote_types(dt, torch.float32)

        if state is None:
            S_in = torch.zeros((bsize, self.n_heads, self.hdim_qk, self.hdim_v), device=x.device, dtype=ft) # (B, H, hdim_qk, hdim_v)
            q_conv_in = k_conv_in = v_conv_in = None
        else:
            S_in, q_conv_in, k_conv_in, v_conv_in = state
            assert S_in.shape == (bsize, self.n_heads, self.hdim_qk, self.hdim_v), S_in.shape
            S_in = S_in.to(ft)
        outputs = []
        for cs in range(0, T, chunk_size):
            input = x[:, cs:cs+chunk_size]
            lc = input.shape[1]
            # 0. q,k,v, split into heads as (B, H, T, hdim) before normalising
            # Also, get the carry convs for the next layer as well
            Qraw,K,V,q_conv_in, k_conv_in, v_conv_in = self.get_qkv(input, q_conv_in, k_conv_in, v_conv_in)

            # print(f"For cs={cs}: Qraw:{Qraw}")
            # 1. B, one write strength per head
            B = torch.sigmoid(input @ self.W_b).transpose(1, 2).to(ft)  # (B, H, T), feeds the solve
            # 2. Alpha / cumulative decay in log space
            raw_alpha = (input @ self.W_a).view(bsize, lc, self.n_heads, self.hdim_qk).transpose(1, 2)  # (B, H, T, hdim_qk)
            log_alpha = Fn.logsigmoid(raw_alpha.to(ft))  # [1] cumsum in low precision drifts
            log_d = torch.cumsum(log_alpha, dim=2)  # (B, H, L, K)
            D = torch.exp(log_d.clamp(min=-88.0))  # (B, H, L, K)

            # exp(log_d[t] - log_d[s]), clamped for stability
            diff_ls = (log_d.unsqueeze(3) - log_d.unsqueeze(2)).clamp(min=-88.0, max=0.0)  # (B, H, L, L, K)
            weight = torch.exp(diff_ls)

            # 3. G for the triangular solve
            Kf = K.to(ft)
            G = torch.einsum('bhlk,bhsk,bhlsk->bhls', Kf, Kf, weight)  # [2] (B, H, L, L)

            Kdash = Kf * D  # (B, H, L, hdim_qk)
            M_minus = torch.tril(torch.ones((lc, lc), device=x.device, dtype=ft), diagonal=-1).view(1, 1, lc, lc)
            Lmat = torch.eye(lc, device=x.device, dtype=ft).view(1, 1, lc, lc) + M_minus * (B.unsqueeze(-1) * G)

            # 4. Solve for W and U
            rhs = B.unsqueeze(dim=-1) * torch.cat((Kdash, V.to(ft)), dim=-1)  # [2] (B, H, lc, K+V)
            solve = tri_solve(Lmat, rhs, upper=False, unitriangular=True)
            W, U = solve.split([self.hdim_qk, self.hdim_v], dim=-1)

            # 5. Scores: the Q side, in x.dtype
            scores = torch.einsum('bhlk,bhsk,bhlsk->bhls', Qraw, K, weight.to(dt))  # (B, H, L, L)
            M = torch.tril(torch.ones((lc, lc), device=x.device, dtype=dt))
            scores = M.view(1, 1, lc, lc) * scores

            E = U - torch.einsum('bhtk,bhkv->bhtv', W, S_in)  # [3] (B, H, L, hdim_v)
            Qsc = D * Qraw  # (B, H, L, hdim_qk), promotes to ft through D
            O = torch.einsum('bhtq,bhqv->bhtv', Qsc, S_in) + scores.to(ft) @ E  # [3] (B, H, L, hdim_v)
            O = O.transpose(1, 2).reshape(bsize, lc, self.D_v).to(dt)  # (B, L, D_v)

            # 6. Carry state (stable)
            last_log = log_d[:, :, -1, :]  # (B, H, hdim_qk)
            w = torch.exp((last_log.unsqueeze(2) - log_d).clamp(min=-88.0, max=0.0))  # (B, H, L, hdim_qk)
            S_in = D[:, :, -1, :].view(bsize, self.n_heads, self.hdim_qk, 1) * S_in + torch.einsum('bhlk,bhlv->bhkv', Kf * w, E)  # [3]

            # Append output
            outputs.append(torch.einsum('bcv,vo->bco', O, self.W_o))


        y = torch.cat(outputs, dim=1)
        return (y, (S_in, q_conv_in, k_conv_in, v_conv_in)) if return_state else y

if __name__ == '__main__':
    data = torch.randn(1, 4, 3)
    m = MultiHeadParallelKDA(D_input = 3, D_output = 7, D_qk = 8, D_v = 12, n_heads = 4)
    # Full data
    out, s = m(data[:, 0:2], return_state=True)
    # print("For full run, state dims:", [x.shape for x in s])
    # print("Q_conv_in for run from 0:2:", s[1])
    out, s = m(data[:, 2:4], state=s, return_state=True)
    k = 2
    outs = []
    state = None
    for i in range(0, 4, k):
        out1, state = m(data[:, i:i+k], state=state, return_state=True)
        # print(f"Q_conv_in for i:{i}:", state[1])
        # print("state dimensions are", [x.shape for x in state])
        outs.append(out1)
    out2 = torch.cat(outs, dim=1)
    # print(out.shape, out2.shape)
    # print(out2 - out)
