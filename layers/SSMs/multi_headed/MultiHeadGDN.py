import torch
from torch import nn
from torch.linalg import solve_triangular as tri_solve
from torch.nn import functional as Fn


class MultiHeadGDN(nn.Module):
    def __init__(self, d_input, d_output, d_model, n_heads, rank = 4, kernel_size = 4):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_model = d_model
        self.n_heads = n_heads
        self.rank = rank
        self.kernel_size = kernel_size
        # Head dims with heads
        assert d_model % n_heads == 0, "The model dim is not divisble by the number of heads!"
        self.head_dim = d_model // n_heads
        # Initialise W_k, W_q, W_v and W_o. No low rank since these genuinely remember features
        self.W_q = nn.Parameter(torch.randn((d_input, d_model)) * (1.0 / (d_input ** 0.5)))
        self.W_k = nn.Parameter(torch.randn((d_input, d_model)) * (1.0 / (d_input ** 0.5)))
        self.W_v = nn.Parameter(torch.randn((d_input, d_model)) * (1.0 / (d_input ** 0.5)))
        # Init the conv and SiLU
        self.q_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, groups=d_model)
        self.k_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, groups=d_model)
        self.v_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, groups=d_model)

        # Output - (d_model, d_output)
        self.W_o = nn.Parameter(torch.randn((d_model, d_output)) / (d_model ** 0.5))
        ## Low rank stuff: Alpha, Beta,
        # Split all of them into (d1, rank), (rank, d2)
        # Alpha - GDN so (d_input, n_heads)
        self.W_a_high = nn.Parameter(torch.randn((d_input, rank)) / (d_input**0.5))
        self.W_a_low = nn.Parameter(torch.randn((rank, n_heads)))
        # Beta - (d_input, n_heads)
        self.W_b_high = nn.Parameter(torch.randn((d_input, rank)) / (d_input**0.5))
        self.W_b_low = nn.Parameter(torch.randn((rank, n_heads)))

        # SiLU
        self.silu = nn.SiLU()


    def forward(self, x, S = None, chunk_size = 32):
        B,T,_ = x.shape
        def _apply_causal_conv(input, is_q = True):
            input = input.transpose(1, 2) # Expects (B, C, T)
            input = Fn.pad(input, pad=(self.kernel_size-1, 0)) # left pad
            if is_q:
                return self.q_conv(input).transpose(1,2)
            else:
                return self.k_conv(input).transpose(1,2)
        if S is None:
            S = torch.zeros(B, self.n_heads, self.d_model // self.n_heads, self.d_model // self.n_heads, dtype=x.dtype, device=x.device) # (B, K, V)
        else:
            assert S.shape == (B, self.n_heads, self.d_model // self.n_heads, self.d_model // self.n_heads)

        outputs = []
        for cs in range(0, T, chunk_size):
            input = x[:, cs:cs + chunk_size]
            lc = input.shape[1]

            ## 1. Q, K, V
            # Get Q, K - W, conv, SiLU, then L2 normalises
            Q = Fn.normalize(self.silu(_apply_causal_conv(input @ self.W_q).view(B, T, self.n_heads, self.head_dim).transpose(1,2)), dim = -1)
            K = Fn.normalize(self.silu(_apply_causal_conv(input @ self.W_k, is_q = False).view(B, T, self.n_heads, self.head_dim).transpose(1,2)), dim = -1)
            # Get V - W, conv SiLU
            V = self.silu(self.v_conv(input @ self.W_v).view(B, T, self.n_heads, self.head_dim).transpose(1,2))

            # 2. Get alpha, beta - both are scalars for GDN
            # Alpha
            log_alpha = Fn.logsigmoid((input @ self.W_a_high) @  self.W_a_low).transpose(1,2).to(torch.promote_types(x.dtype, torch.float32)) # (B, n_heads, T)
            g = torch.exp(torch.cumsum(log_alpha, dim=-1).clamp(-88.0, 88.0)).to(x.dtype) # (B, H, T)
            G = torch.cumsum(log_alpha, dim=-1) # (B, n_heads, T)
            G = torch.exp(torch.tril(G.unsqueeze(dim=-1) - G.unsqueeze(dim=-2)).clamp(-88.0, 88.0)).to(x.dtype) # (B, n_heads, T, T) | (i,j) -> g_i / g_j
            # Beta
            Beta = Fn.sigmoid((input @ self.W_b_high) @ self.W_b_low).transpose(1,2) # (B, n_heads, T)


            #  3. Get W and U
            # W is normal, the decay is applied to the chunk scores later
            cast_type = torch.promote_types(x.dtype, torch.float32)
            g_matrix = Beta.unsqueeze(dim=-1) * G * torch.tril(torch.einsum('bhtk, bhsk->bhts', K, K), diagonal = -1)
            A_w = torch.eye(lc, device=x.device, dtype=x.dtype) + g_matrix # Solve for W, promote to fp32 if need be
            Beta = Beta.unsqueeze(dim=-1)
            W = torch.linalg.solve_triangular(A_w.to(cast_type), (Beta * K).to(cast_type), upper=False, unitriangular=True)

            # U is different
            decayed_G = G * g_matrix # Both are (B, H, T, T)
            A_u = torch.eye(lc, device=x.device, dtype=x.dtype) + Beta * decayed_G
            U = torch.linalg.solve_triangular(A_u, Beta * V, upper=False, unitriangular=True) # (B, H, T, V)


            # 4. Output projection
            E = U - torch.einsum('bhtk,bhkv->bhtv', W * G, S)
            chunk_scores = G * torch.tril(torch.einsum('bhck,bhsk->bhcs', Q, K))
            carry_out = torch.einsum('bhsk,bhkv->bhsv',(g.unsqueeze(dim=-1) * Q), S) # ((B, H, T, 1) * (B, H, T, D)) .
            O = carry_out + chunk_scores @ E # (B, H, T, D)
            O = O.transpose(1,2).reshape(B, lc, self.d_model)


            # 5. Carry state
            E_carry = G[:, -1].unsqueeze(dim=-1) * E
            S = g[:,:, -1, None] * S + torch.einsum('bhsk, bhsv->bhkv',K, E_carry)

            # Append to outputs
            outputs.append(torch.einsum('bcv,vo->bco', O, self.W_o))
        return torch.stack(outputs, dim=1)

def recurrent_reference(model, x):
    # Token-by-token oracle: features over the whole sequence, then the single-head legacy
    # update (parallel_gdn.py legacy_out) run on each head separately, concat, W_o
    B, T, _ = x.shape
    H, d, k = model.n_heads, model.head_dim, model.kernel_size

    def feature(W, conv):
        h = (x @ W).transpose(1, 2)                                    # (B, D, T) for conv1d
        h = Fn.conv1d(Fn.pad(h, (k - 1, 0)), conv.weight, conv.bias, groups=model.d_model)  # causal
        return Fn.silu(h).transpose(1, 2).reshape(B, T, H, d)

    q = Fn.normalize(feature(model.W_q, model.q_conv), dim=-1)
    kk = Fn.normalize(feature(model.W_k, model.k_conv), dim=-1)
    v = feature(model.W_v, model.v_conv)
    alpha = torch.sigmoid((x @ model.W_a_high) @ model.W_a_low)       # (B, T, H)
    beta = torch.sigmoid((x @ model.W_b_high) @ model.W_b_low)
    I = torch.eye(d, dtype=x.dtype, device=x.device)

    heads = []
    for h in range(H):
        S = x.new_zeros(B, d, d)
        outs = []
        for t in range(T):
            kt, vt, qt = kk[:, t, h], v[:, t, h], q[:, t, h]
            a, b = alpha[:, t, h].view(B, 1, 1), beta[:, t, h].view(B, 1, 1)
            S = a * (I - b * kt[:, :, None] * kt[:, None, :]) @ S + b * kt[:, :, None] * vt[:, None, :]
            outs.append(torch.einsum('bkv,bk->bv', S, qt))
        heads.append(torch.stack(outs, dim=1))                         # (B, T, d)
    return torch.cat(heads, dim=-1) @ model.W_o                        # head h -> channels h*d:(h+1)*d


if __name__ == '__main__':
    torch.manual_seed(0)
    dt = torch.float64
    B, T, D_in, D_model, D_out = 2, 50, 16, 16, 8

    for H in (4, 1):
        model = MultiHeadGDN(D_in, D_out, D_model, H).to(dt)
        x = torch.randn(B, T, D_in, dtype=dt, requires_grad=True)
        w = torch.randn(B, T, D_out, dtype=dt)

        ref = recurrent_reference(model, x)
        (ref * w).sum().backward()
        ref_grads = [p.grad.clone() for p in (x, *model.parameters())]

        # 8, 16: T=50 leaves a ragged last chunk. 64 > T: a single chunk
        for cs in (8, 16, 64):
            x.grad = None
            model.zero_grad()
            out = model(x, chunk_size=cs)
            assert out.shape == ref.shape, f"shape {tuple(out.shape)} != {tuple(ref.shape)}"
            print(f"H={H} chunk={cs}: max |out - ref| = {(out - ref).abs().max().item():.2e}")
            assert torch.allclose(out, ref, rtol=1e-8, atol=1e-10)
            (out * w).sum().backward()
            for g1, g2 in zip([p.grad for p in (x, *model.parameters())], ref_grads):
                assert torch.allclose(g1, g2, rtol=1e-8, atol=1e-10)
            print(f"H={H} chunk={cs}: gradients match")
    print("All checks passed")
