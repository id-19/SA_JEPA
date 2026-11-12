import torch
import torch.nn as nn

class LonghornArm(nn.Module):
    def __init__(self, d_model, d_out):
        super().__init__()
        self.cell = nn.GRUCell(d_model, d_model)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, x.size(-1), device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            h = self.cell(x[:, t], h)
            ys.append(self.out(h))
        return torch.stack(ys, dim=1)
