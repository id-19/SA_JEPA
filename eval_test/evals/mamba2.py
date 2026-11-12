"""Fallback Mamba2-like arm (pure PyTorch; no mamba-ssm dependency)."""
import torch
import torch.nn as nn

class Mamba2Arm(nn.Module):
    def __init__(self, d_model, d_out, d_state=8):
        super().__init__()
        # Simplified diagonal SSM + gated MLP; NOT full Mamba2.
        self.in_proj = nn.Linear(d_model, 2 * d_model)
        self.A = nn.Parameter(torch.ones(d_model) * 0.9)
        self.B = nn.Linear(d_model, d_model)
        self.C = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_out)

    def forward(self, x):
        B, T, _ = x.shape
        h = torch.zeros(B, x.size(-1), device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            xt = x[:, t]
            a, g = self.in_proj(xt).chunk(2, dim=-1)
            a = torch.nn.functional.silu(a)
            inp = self.B(xt)
            h = self.A * h + inp * a
            y = self.C(xt) * torch.nn.functional.silu(h)
            ys.append(self.out(y))
        return torch.stack(ys, dim=1)
