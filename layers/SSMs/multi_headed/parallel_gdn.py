import torch
import torch.nn.functional as F
from torch import nn

# [1.2] ID1-280: multi-head chunked Gated DeltaNet. Starts as a copy of single_headed/parallel_gdn.py;
# add the head axis and nothing else. Contract the tests copy weights through (D_qk, D_v are total widths,
# head h owns columns h*D_qk//n_heads : (h+1)*D_qk//n_heads, likewise for v and the rows of W_o):
#   W_q, W_k: (D_input, D_qk)    W_v: (D_input, D_v)    W_o: (D_v, D_output)
#   W_a, W_b: (D_input, n_heads)
# output_gate=True adds (applied per head to O before W_o: O = rms_norm(O) * silu(x @ W_g)):
#   o_norm_weight: (head_dim_v,) shared across heads, init ones    W_g: (D_input, D_v)
# Tests: eval_test/tests/test_multihead_gdn.py

class MultiHeadParallelGDN(nn.Module):
    def __init__(self, D_input, D_output, D_qk, D_v, n_heads, output_gate=True) -> None:
        super().__init__()
        # Can specify qk, v, params differnetly for sake of curiosity
        self.D_input = D_input
        self.D_output = D_output
        self.n_heads = n_heads
        self.output_gate = output_gate

        self.D_qk = D_qk
        self.D_v = D_v

        # head dims
        assert D_qk % n_heads == 0
        assert D_v % n_heads == 0
        self.head_dim_qk = D_qk // n_heads
        self.head_dim_v = D_v // n_heads

        # Initialise W_k, W_q and W_v
        self.W_q = nn.Parameter(torch.randn((D_input, D_qk)) * (1.0 / (D_input ** 0.5)))
        self.W_k = nn.Parameter(torch.randn((D_input, D_qk)) * (1.0 / (D_input ** 0.5)))
        self.W_v = nn.Parameter(torch.randn((D_input, D_v)) * (1.0 / (D_input ** 0.5)))

        # Output projection
        self.W_o = nn.Parameter(torch.randn((D_v, D_output)) * (1.0 / (D_v ** 0.5)))

        # Write strength - beta(common to all)
        self.W_b = nn.Parameter(torch.randn((D_input, n_heads)) / (D_input ** 0.5)) # Takes (B,T,C) -> (B,T, 1), which is then broadcast

        # Alpha for decay
        self.W_a = nn.Parameter(torch.randn(D_input, n_heads) / (D_input ** 0.5))

        # Output norm (per head, over head_dim_v, weight shared across heads) and gate
        if output_gate:
            self.o_norm_weight = nn.Parameter(torch.ones(self.head_dim_v))
            self.W_g = nn.Parameter(torch.randn((D_input, D_v)) * (1.0 / (D_input ** 0.5)))

    def legacy_out(self, x, state=None, return_state=False):
        # S_t = alpha * S_(t-1) + beta * k_t * Error  // Write the error to correct address, (K) @ (V)-> (K, V) correct
        # Error = v_t - k_t . S_(t-1)
        # Solve this equation you get(equivalent)
        # S_(t+1) = alpha * (I - beta * k_t^T * k_t) * S_t + beta * k_t * v_t
        # state: None for a fresh sequence, else the tensor S that forward or legacy_out returned
        B,T,_ = x.shape
        H = self.n_heads
        d_qk = self.head_dim_qk
        d_v = self.head_dim_v
        if state is None:
            S = torch.zeros((B, H, d_qk, d_v), device=x.device, dtype=x.dtype)
        else:
            assert state.shape == (B, H, d_qk, d_v), state.shape
            S = state.to(x.dtype)
        output = []
        for t in range(T):
            input = x[:,t,:]
            q = F.normalize(torch.einsum('bc,cq->bq', input, self.W_q).view(B, H, d_qk), dim=-1)
            k = F.normalize(torch.einsum('bc,ck->bk', input, self.W_k).view(B, H, d_qk), dim=-1)
            v = torch.einsum('bc,cv->bv', input, self.W_v).view(B, H, d_v)

            # alpha and beta
            alpha = torch.sigmoid(input @ self.W_a) # (B,H)
            beta = (torch.sigmoid(input @ self.W_b)) # (B,H)

            # Find A - B,H, K, K
            A = (torch.eye(d_qk, device=x.device, dtype=x.dtype).view(1, 1, d_qk, d_qk) - beta.view(B, H, 1, 1) * torch.einsum('bhk,bhs->bhks', k, k))
            A = alpha.view(B,H,1,1) * A

            # Find B
            B_in = beta.view(B, H, 1, 1) * torch.einsum('bhk,bhv->bhkv', k, v)

            # Write state update
            S = torch.einsum('bhsk,bhkv->bhsv', A, S) + B_in

            # Output proj
            O = torch.einsum('bhk,bhkv->bhv', q, S)
            if self.output_gate:
                O = F.rms_norm(O, (d_v,), self.o_norm_weight.to(O.dtype), 1e-5)
                O = O * F.silu(input @ self.W_g).view(B, H, d_v)
            O = O.reshape(B, self.D_v)

            output.append(O @ self.W_o)
        y = torch.stack(output, dim=1)
        return (y, S) if return_state else y

    def forward(self, x, state=None, return_state=False, chunk_size = 32):
        # state: None for a fresh sequence, else the tensor S_in (B, H, head_dim_qk, head_dim_v) a previous call returned
        H = self.n_heads
        d_qk = self.head_dim_qk
        d_v = self.head_dim_v
        # Projections and score matmuls run in x.dtype (bf16/fp16 ok). The decay cumsum, both triangular
        # solves and the carried state S run in at least fp32 (fp64 stays fp64 for gradcheck)
        dt = x.dtype
        ft = torch.promote_types(dt, torch.float32)
        W_q, W_k, W_v, W_b, W_a, W_o = (p.to(dt) for p in (self.W_q, self.W_k, self.W_v, self.W_b, self.W_a, self.W_o))
        if self.output_gate:
            W_g = self.W_g.to(dt)
        bsize,T,_ = x.shape
        B = bsize
        if state is None:
            S_in = torch.zeros((bsize, self.n_heads, self.head_dim_qk, self.head_dim_v), device=x.device, dtype=ft)
        else:
            assert state.shape == (bsize, self.n_heads, self.head_dim_qk, self.head_dim_v), state.shape
            S_in = state.to(ft)
        outputs = []
        for cs in range(0, T, chunk_size):
            input = x[:, cs:cs+chunk_size]
            lc = input.shape[1]
            # 0. q,k,v - (B, n_heads, lc, head_dim<qk/v>)
            Q = torch.einsum('bsc,cq->bsq', input, W_q)
            Q = Q.view(bsize, lc, self.n_heads, self.head_dim_qk).transpose(1,2)
            Q = F.normalize(Q, dim=-1)
            K = torch.einsum('bsc,ck->bsk', input, W_k)
            K = K.view(bsize, lc, self.n_heads, self.head_dim_qk).transpose(1,2)
            K = F.normalize(K, dim=-1)
            V = torch.einsum('bsc,cv->bsv', input, W_v)
            V = V.view(bsize, lc, self.n_heads, self.head_dim_v).transpose(1,2)

            # 1. Get D - Unchanged for heads
            D = torch.sigmoid(torch.einsum('bsc,ch->bsh', input, W_b)).transpose(1,2) # (B, H, lc)

            # 2. Alpha, g and G
            # Use log form because products underflow to zero, add clamps if it still breaks
            Alpha = (input @ W_a).transpose(1,2) # (B, lc, D_input) @ (D_input, n_heads) -> (B, lc, h) -> (B, H, lc)
            log_alpha = torch.nn.functional.logsigmoid(Alpha.to(ft)) # cumsum in low precision drifts
            log_g = torch.cumsum(log_alpha, dim=-1)
            g = torch.exp(log_g) # (B, H, lc)
            G = (log_g.unsqueeze(dim=-1) - log_g.unsqueeze(dim=-2)) # (B, H, lc, 1) / (B, H, 1, lc) -> (B, H, lc, lc)
            G = torch.tril(G) # B, H, lc, lc
            G = torch.exp(G.clamp(-88.0, 88.0)) # (B, H, lc, lc)


            # 3. W and U
            # W is same as before
            g_matrix = torch.tril(torch.einsum('bhsk, bhtk->bhst', K, K), diagonal=-1).to(ft) # OG matrix named after Scandinavian guy
            D = D.to(ft)
            A_w = torch.eye(lc, device=x.device, dtype=ft).view(1, 1, lc, lc) + D.unsqueeze(-1) * g_matrix # D(B, H, lc, 1) broadcasts to (B, H, lc, lc)
            # Solves in fp32: ill-conditioning feeds errors into the state, and MPS triangular solve is fp32-only
            W = torch.linalg.solve_triangular(A_w, (D.unsqueeze(-1) * K).to(ft), upper=False, unitriangular=True) # (B, H, lc, K)

            # U is different
            decayed_G = G * g_matrix # Both are (B, H, lc, lc)
            A_u = torch.eye(lc, device=x.device, dtype=ft).view(1, 1, lc, lc) + D.unsqueeze(-1) * decayed_G # D broadcasts again (B, H, lc, 1) to (B, H, lc, lc)
            U = torch.linalg.solve_triangular(A_u, (D.unsqueeze(-1) * V).to(ft), upper=False, unitriangular=True) # (B, H, lc, V)

            # 4. Output projection (fp32 from here: everything below touches S)
            E = U - torch.einsum('bhsk,bhkv->bhsv', g.unsqueeze(-1) * W, S_in)
            chunk_scores = G * torch.tril(torch.einsum('bhtk,bhsk->bhts', Q, K)) # (Both are (B, H, lc, lc))
            carry_out = torch.einsum('bhsk,bhkv->bhsv',(g.unsqueeze(-1) * Q), S_in)
            O = carry_out + torch.einsum('bhts,bhsv->bhtv', chunk_scores, E)
            if self.output_gate:
                # per-head RMSNorm in ft, then gate from this chunk's input, split into heads like V
                O = F.rms_norm(O, (d_v,), self.o_norm_weight.to(ft), 1e-5)
                gate = F.silu(torch.einsum('bsc,cv->bsv', input, W_g)).view(bsize, lc, self.n_heads, self.head_dim_v).transpose(1,2)
                O = O * gate
            O = O.transpose(1,2).reshape(bsize, lc, self.D_v) # (B, H, T, head_v) -> (B, T, H, *) -> (B, T, V)

            # 5. Carry state
            E_carry = G[:, :, -1].view(B, H, lc, 1) * E # E_carry is (B, H, lc, d_v)
            S_in = g[:,:, -1].view(B, H, 1, 1) * S_in + torch.einsum('bhsk, bhsv->bhkv',K.to(ft), E_carry)

            # Append to outputs
            outputs.append(torch.einsum('bcv,vo->bco', O.to(dt), W_o))

        y = torch.cat(outputs, dim=1)
        return (y, S_in) if return_state else y
