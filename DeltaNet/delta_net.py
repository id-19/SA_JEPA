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
        self.W_q = nn.Parameter(torch.randn((d_input, d_qk)) / (1.0 / (d_input ** 0.5)))
        self.W_k = nn.Parameter(torch.randn((d_input, d_qk)) / (1.0 / (d_input ** 0.5)))
        self.W_v = nn.Parameter(torch.randn((d_input, d_v)) / (1.0 / (d_input ** 0.5)))

        # Output projection
        self.W_o = nn.Parameter(torch.randn((d_v, d_output)) / (1.0 / (d_v ** 0.5)))

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
        S = torch.zeros((B, self.d_qk, self.d_v))
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
            A = (torch.eye(self.d_qk).unsqueeze(dim=0) - beta * torch.einsum('bk,bs->bks', k, k))
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

if __name__ == '__main__':
    data = torch.randn(3, 5, 7)
    for decay in (None, "scalar", "vector"):
        print(decay)
        d_net = UnifiedDeltaNet(d_input = 7, d_output = 1, d_qk = 8)
        outputs = d_net(data)
        print(outputs)
    # legacy_outs = former.legacy_call(data)
    # print(legacy_outs)
    # is_correct = torch.allclose(outputs, legacy_outs)
    # print(is_correct)
