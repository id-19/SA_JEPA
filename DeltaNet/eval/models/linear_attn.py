# Implement a simple linear attention layer
import torch
from torch import nn
import torch.nn.functional as F


class LinFormer(nn.Module):
    def __init__(self, D_input, D_output, D_qk = 0, D_v = 0, model_dim = 0):
        super().__init__()
        # Can specify qk, v, params differnetly for sake of curiosity
        self.D_input = D_input
        self.D_output = D_output
        self.model_dim = model_dim

        self.D_qk = D_qk if D_qk else model_dim
        self.D_v = D_v if D_v else model_dim

        # Initialise W_k, W_q and W_v
        self.W_q = nn.Parameter(torch.randn((D_input, D_qk if D_qk else model_dim)))
        self.W_k = nn.Parameter(torch.randn((D_input, D_qk if D_qk else model_dim)))
        self.W_v = nn.Parameter(torch.randn((D_input, D_v if D_v else model_dim)))

        # Output projection
        self.W_o = nn.Parameter(torch.randn((D_v if D_v else model_dim, D_output)))

    def forward(self, x, chunk_size = 128):
        if x.ndim == 2:
            x = torch.unsqueeze(x, dim=0)
        B,T,C = x.shape
        ## Parallel + chunked version
        S = torch.zeros((B, self.D_qk, self.D_v)) # (B, T, K, V)
        final_pre = None
        for chunk_start in range(0, T, chunk_size):
            input = x[:, chunk_start:chunk_start + chunk_size, :]
            q = torch.einsum('btc, cq->btq', input, self.W_q)
            q = F.normalize(q, eps=1e-6, dim=-1)
            k = torch.einsum('btc, cq->btq', input, self.W_k)
            k = F.normalize(k, eps=1e-6, dim=-1)
            v = torch.einsum('btc, cv->btv', input, self.W_v)
            # Find KV and cumsum it
            chunk_state = torch.cumsum(torch.einsum('bck,bcv->bckv', k, v), dim=1)
            # Add previous S to chunk state i.e. final state of prevuous chunk added
            # [:, None, :, :] inserts a new dim along C dimension(syntatic sugar for unsqueeze)
            chunk_state = S[:, None, :, :]  + chunk_state
            # Get pre-proj output
            chunk_pre = torch.einsum('bckv, bck->bcv', chunk_state, q)
            if chunk_start == 0:
                final_pre = chunk_pre
            else:
                assert(final_pre != None), "Final pre is None"
                final_pre = torch.cat([final_pre, chunk_pre], dim=1)
            # Add back to chunk-global state
            S = chunk_state[:, -1, :, :]
        return torch.einsum('btv, vo -> bto', final_pre, self.W_o)

    def legacy_call(self, x):
        if x.ndim == 2:
            x = torch.unsqueeze(x, dim=0)
        B,T,C = x.shape
        state = torch.zeros((B, self.D_qk, self.D_v))
        outputs = []
        for t in range(T):
            input = x[:, t, :]
            q,k,v = input @ self.W_q, input @ self.W_k, input @ self.W_v
            q = F.normalize(q, eps=1e-6, dim=-1)
            k = F.normalize(k, eps=1e-6, dim=-1)
            state = state + torch.einsum('bk, bv->bkv',k, v)
            output_pre = torch.einsum('bqv, bq -> bv', state, q)
            output = torch.einsum('bv, vo -> bo', output_pre, self.W_o)
            outputs.append(output)
        return torch.stack(outputs, dim=1)



if __name__ == '__main__':
    data = torch.randn(1, 4, 16)
    former = LinFormer(D_input = 16, D_output = 1, model_dim = 8)
    outputs = former(data)
    print(outputs)
    legacy_outs = former.legacy_call(data)
    print(legacy_outs)
    is_correct = torch.allclose(outputs, legacy_outs)
    print(is_correct)
