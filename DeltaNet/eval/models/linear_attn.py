# Implement a simple linear attention layer
import torch
from torch import nn


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
        self.W_q = nn.Linear(D_input, D_qk if D_qk else model_dim)
        self.W_k = nn.Linear(D_input, D_qk if D_qk else model_dim)
        self.W_v = nn.Linear(D_input, D_v if D_v else model_dim)

        # Output projection
        self.W_o = nn.Linear(D_v if D_v else model_dim, D_output)

    def forward(self, x):
        if x.ndim == 2:
            x = torch.unsqueeze(x, dim=0)
        B,T,C = x.shape
        state = torch.zeros((B, self.D_qk, self.D_v))
        outputs = []
        for timestep in range(T):
            input = x[:, timestep, :]
            q_t,k_t,v_t = self.W_q(input), self.W_k(input), self.W_v(input)
            # Norm on k,q
            q_t = q_t / torch.sqrt(torch.einsum("bq, bq -> b", q_t, q_t).unsqueeze(dim=-1)+1e-6)
            k_t = k_t / torch.sqrt(torch.einsum("bk, bk -> b", k_t, k_t).unsqueeze(dim=-1)+1e-6)
            # state update (B, k, v)
            # (bk, bv) -> bkv [OUTER PRODUCT]
            state = state + torch.einsum('bki,ibv -> bkv',k_t.unsqueeze(dim=-1), v_t.unsqueeze(dim=0))
            # Output projection
            out_raw = torch.einsum('bkv,bk->bv', state, q_t)
            output = self.W_o(out_raw)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

if __name__ == '__main__':
    data = torch.randn(1, 4, 16)
    former = LinFormer(D_input = 16, D_output = 1, model_dim = 8)
    outputs = former(data)
    print(outputs)
