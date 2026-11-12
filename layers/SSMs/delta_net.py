import torch
from torch import nn
import torch.nn.functional as F

class UnifiedDeltaNet(nn.Module):
    def __init__(self, d_input, d_output, d_qk, d_v=None, decay=None):
        # decay sets the kind, none = vanilla D-net, scalar = GDN, vector = KDA(per-channel decay)
        super().__init__()
        # Can specify qk, v, params differnetly for sake of curiosity
        self.d_input = d_input
        self.d_output = d_output

        self.d_qk = d_qk
        d_v = d_v if d_v != None else d_qk
        self.d_v = d_v

        # Initialise W_k, W_q and W_v
        self.W_q = nn.Parameter(torch.randn((d_input, d_qk)) * (1.0 / (d_input ** 0.5)))
        self.W_k = nn.Parameter(torch.randn((d_input, d_qk)) * (1.0 / (d_input ** 0.5)))
        self.W_v = nn.Parameter(torch.randn((d_input, d_v)) * (1.0 / (d_input ** 0.5)))

        # Output projection
        self.W_o = nn.Parameter(torch.randn((d_v, d_output)) * (1.0 / (d_v ** 0.5)))

        # Write strength - beta(common to all)
        self.W_b = nn.Parameter(torch.randn((d_input, 1))) # Takes (B,T,C) -> (B,T), which is then broadcast

        # Alpha - decay term
        self.decay = decay
        if decay == 'scalar':
            self.W_a = nn.Parameter(torch.randn((d_input, 1)))
        elif decay == 'vector':
            self.W_a = nn.Parameter(torch.randn((d_input, d_qk))) # (B,T,C) -> (B, T, K) then (B,T,K) * (B, T, K, V) ----(broadcast)-->  (BTKV)


    def forward(self, x):
        # S_t = alpha * S_(t-1) + beta * k_t * Error  // Write the error to correct address, (K) @ (V)-> (K, V) correct
        # Error = v_t - k_t . S_(t-1)
        # Solve this equation you get(equivalent)
        # S_(t+1) = alpha * (I - beta * k_t^T * k_t) * S_t + beta * k_t * v_t
        B,T,C = x.shape
        S = torch.zeros((B, self.d_qk, self.d_v), device=x.device, dtype=x.dtype)
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
            A = (torch.eye(self.d_qk, device=x.device, dtype=x.dtype).unsqueeze(dim=0) - beta * torch.einsum('bk,bs->bks', k, k))
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

    def chunked_forward(self, x, chunk_size=64):
        """Serial oracle inside chunk boundaries: runs the exact recurrent
        update token by token, but bounds bookkeeping/vectorisation to chunks."""
        B, T, C = x.shape
        S = torch.zeros((B, self.d_qk, self.d_v), device=x.device, dtype=x.dtype)
        outs = []
        I = torch.eye(self.d_qk, device=x.device, dtype=x.dtype)
        for cs in range(0, T, chunk_size):
            chunk = x[:, cs:cs + chunk_size, :]      # (B, Lc, C)
            for t in range(chunk.shape[1]):
                inp = chunk[:, t, :]                  # (B, C)
                q = F.normalize(torch.einsum('bc,cq->bq', inp, self.W_q), dim=-1)
                k = F.normalize(torch.einsum('bc,ck->bk', inp, self.W_k), dim=-1)
                v = torch.einsum('bc,cv->bv', inp, self.W_v)
                # alpha/beta mirror ::forward()
                if self.decay is not None:
                    alpha = torch.sigmoid(inp @ self.W_a)
                beta = (torch.sigmoid(inp @ self.W_b)).view(B, 1, 1)
                A = I.expand(B, self.d_qk, self.d_qk).clone()
                A = A - beta * torch.einsum('bk,bs->bks', k, k)
                if self.decay == 'scalar':
                    A = alpha.view(B, 1, 1) * A
                B_in = beta * torch.einsum('bk,bv->bkv', k, v)
                if self.decay == 'vector':
                    S = (alpha.unsqueeze(-1)
                         * torch.einsum('bsk,bkv->bsv', A, S)
                         + B_in)
                else:
                    S = torch.einsum('bsk,bkv->bsv', A, S) + B_in
                O = torch.einsum('bkv,bk->bv', S, q) @ self.W_o
                outs.append(O)
        out = torch.stack(outs, dim=1)            # (B, T, d_output)
        return out

if __name__ == '__main__':
    data = torch.randn(3, 5, 7)
    for decay in (None, "scalar", "vector"):
        print(decay)
        d_net = UnifiedDeltaNet(d_input = 7, d_output = 1, d_qk = 8)
        outputs = d_net.chunked_forward(data)
        print(outputs)
    legacy_outs = former.legacy_call(data)
    print(legacy_outs)
    is_correct = torch.allclose(outputs, legacy_outs)
    print(is_correct)
