import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import solve_triangular

# Parallel deltaNet implementation, with parallel chunked blocks. Uses WY form + UT transform to fully parallelise
# Vanilla DeltaNet, not a GDN

class ParallelDeltaNet(nn.Module):
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

    def legacy_forward(self, x):
        # S_t = alpha * S_(t-1) + beta * k_t * Error  // Write the error to correct address, (K) @ (V)-> (K, V) correct
        # Error = v_t - k_t . S_(t-1)
        # Solve this equation you get(equivalent)
        # S_(t+1) = alpha * (I - beta * k_t^T * k_t) * S_t + beta * k_t * v_t
        bsize,T,_ = x.shape
        S = torch.zeros((bsize, self.D_qk, self.D_v), device=x.device, dtype=x.dtype)
        output = []
        for t in range(T):
            input = x[:,t,:]
            q = F.normalize(torch.einsum('bc,cq->bq', input, self.W_q))
            k = F.normalize(torch.einsum('bc,ck->bk', input, self.W_k))
            v = torch.einsum('bc,cv->bv', input, self.W_v)

            # beta
            beta = (torch.sigmoid(input @ self.W_b)).view(bsize,1,1) # So we can multiply cleanly by (B,K,K)

            # Find A
            A = (torch.eye(self.D_qk, device=x.device, dtype=x.dtype).unsqueeze(dim=0) - beta * torch.einsum('bk,bs->bks', k, k))

            # Find B
            B_in = beta * torch.einsum('bk,bv->bkv', k, v)

            S = torch.einsum('bsk,bkv->bsv', A, S) + B_in
            # Output proj
            O = torch.einsum('bkv,bk->bv', S, q)

            output.append(O @ self.W_o)
        return torch.stack(output, dim=1)

    def forward(self, x, chunk_size = 32):
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
            # 2. Get U and W
            g_matrix = torch.einsum('bsk, btk->bst', K, K) # OG matrix named after Scandinavian guy
            A = torch.eye(lc, device=x.device, dtype=x.dtype) + torch.tril(torch.einsum('bs,bst->bst', D.squeeze(dim=-1), g_matrix), diagonal=-1)
            # K for W, V for U - both mult by D
            RHS = torch.cat([D * K, D * V], dim=-1) # (B, C, K + V) Beta being (B, C, 1) broadcasts perfectly
            # Don't be lazy, solve the triangular
            # MPS triangular solve is fp32-only. Accumulate there, store in x's dtype.
            solve = solve_triangular(
                A.float(), RHS.float(), upper=False, unitriangular=True,
            ).to(dtype=x.dtype)
            W,U = solve.split([self.D_qk, self.D_v], dim=-1)

            # 4. Get output projection!
            M = torch.tril(torch.ones((lc, lc), device=x.device, dtype=x.dtype))
            carry = torch.einsum('bsq,bqv->bsv', Q, S_in)
            intra_chunk_scores = (M * (Q @ K.transpose(-1,-2))) # (B, C, C)
            value_error = (U - W @ S_in) # (B, C, V)
            O = carry + intra_chunk_scores @ value_error
            final_out = torch.einsum('bsv,vo->bso', O, self.W_o)

            outputs.append(final_out)
            # The statement below is summing over timesteps in-chunk, not over steps
            # The einsum sums over token contributions, not states. S_in = carry + intra-chunk contributions
            # Yes, in the DeltaNet vanilla version, S_in carries over(it's crazy)
            S_in = S_in + torch.einsum('bsk, bsv->bkv', K, value_error)

        return torch.cat(outputs, dim=1)


if __name__ == '__main__':
    data = torch.randn(3, 5, 7)
    d_net = ParallelDeltaNet(D_input = 7, D_output = 1, D_qk = 8, D_v = 8)
    outputs = d_net(data)
    print(outputs)
