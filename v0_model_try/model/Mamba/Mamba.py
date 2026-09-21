import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal Mamba 1 block. The one idea that separates this from DiscreteSSM:
# delta, B and C are FUNCTIONS OF THE INPUT, so the model can choose what to
# remember and forget per timestep. That is the "selective" in selective SSM.
class Mamba(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)   # inner width the SSM operates on

        # Input projection: produces x AND z (the gating branch), so 2 * d_inner.
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner)

        # Depthwise causal conv1d over time: lets neighbouring tokens mix, which is
        # what lets the per-token delta approximate the whole discretisation step.
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, groups=self.d_inner,
            padding=d_conv - 1,
        )

        # Input-dependent delta: low-rank projection d_inner -> dt_rank -> d_inner.
        self.dt_rank = max(1, d_model // 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is diagonal, stored as log: A = -exp(A_log) -> strictly negative -> decay.
        # Init log(1..d_state) gives A = -1..-d_state, the standard HiPPO-ish spread.
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Input-independent bias for delta, initialised so softplus(dt_bias) spans
        # a range of timescales (~0.001 to ~0.1).
        dt = torch.exp(torch.rand(self.d_inner) * (torch.log(torch.tensor(0.1)) - torch.log(torch.tensor(0.001)))
                       + torch.log(torch.tensor(0.001)))
        self.dt_bias = nn.Parameter(dt)

        self.D = nn.Parameter(torch.ones(self.d_inner))   # skip connection
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, _ = x.shape

        # Split the projection into the SSM branch (x) and the gate branch (z).
        xz = self.in_proj(x)                                  # (B, T, 2*d_inner)
        x_in, z = xz.chunk(2, dim=-1)                         # each (B, T, d_inner)

        # Causal depthwise conv over time, then SiLU.
        x_conv = x_in.transpose(1, 2)                         # (B, d_inner, T)
        x_conv = self.conv1d(x_conv)[:, :, :T]                # trim the right padding
        x_conv = F.silu(x_conv.transpose(1, 2))               # (B, T, d_inner)

        # --- Selection: build delta, B, C from the input ---
        ssm_par = self.x_proj(x_conv)                         # (B, T, dt_rank + 2*d_state)
        delta_in, B_in, C_in = ssm_par.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        # delta must be POSITIVE (it is a step size), so softplus. (B, T, d_inner)
        delta = F.softplus(self.dt_proj(delta_in) + self.dt_bias)

        # Discretise A per timestep: A_bar = exp(delta * A), elementwise on the diagonal.
        A = -torch.exp(self.A_log)                            # (d_inner, d_state)
        # delta: (B, T, d_inner) -> (B, T, d_inner, 1)  |  A: (d_inner, d_state)
        deltaA = torch.exp(delta.unsqueeze(-1) * A)           # (B, T, d_inner, d_state)

        # B_bar = ((A_bar - 1) / A) * B_t. With diagonal A this is elementwise.
        # B_in is (B, T, d_state) -> (B, T, 1, d_state) to broadcast over d_inner.
        scale = (deltaA - 1.0) / A                            # (B, T, d_inner, d_state)
        b_t = B_in.unsqueeze(2)                               # (B, T, 1, d_state)
        deltaB_u = scale * b_t * x_conv.unsqueeze(-1)         # (B, T, d_inner, d_state)

        # --- Sequential scan of the recurrence ---
        h = torch.zeros(B, self.d_inner, self.d_state, dtype=x.dtype, device=x.device)
        ys = []
        for t in range(T):
            h = deltaA[:, t] * h + deltaB_u[:, t]             # (B, d_inner, d_state)
            y_t = (h * C_in[:, t].unsqueeze(1)).sum(dim=-1)   # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                            # (B, T, d_inner)

        # Skip connection, then gate with SiLU(z), then project back to d_model.
        y = y + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)                               # (B, T, d_model)


if __name__ == "__main__":
    torch.manual_seed(0)
    m = Mamba(d_model=32, d_state=8)
    x = torch.randn(2, 16, 32)

    y = m(x)
    print(f"input  {tuple(x.shape)} -> output {tuple(y.shape)}")
    print(f"output finite: {bool(torch.isfinite(y).all())}")

    # Every parameter trains
    y.sum().backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    print(f"parameters: {len(list(m.parameters()))}, without grad: {missing}")

    # Selection: delta should differ across timesteps and across batch items
    xz = m.in_proj(x).chunk(2, dim=-1)[0].transpose(1, 2)
    xc = F.silu(m.conv1d(xz)[:, :, :x.shape[1]].transpose(1, 2))
    d = F.softplus(m.dt_proj(m.x_proj(xc).split([m.dt_rank, m.d_state, m.d_state], dim=-1)[0]) + m.dt_bias)
    print(f"delta varies over time:   {not torch.allclose(d[:, 0], d[:, -1])}")
    print(f"delta varies over batch:  {not torch.allclose(d[0], d[1])}")

    # Stability over a much longer sequence
    y_long = m(torch.randn(1, 500, 32))
    print(f"T=500: |out|max={y_long.abs().max().item():.3f} finite={bool(torch.isfinite(y_long).all())}")
