# my_sel_s4d.py — rung 3: S4D with input-dependent Delta (selective)
#
# SIMPLE FORM FIRST. a = -rate is REAL. State is 1-D, no omega, no 2-vector.
# Train this arm, then restore the complex diagonal form (omega comes back: B and C
# grow a trailing 2, and the angle omega*delta replaces the hoisted cos/sin).
#
# ZOH of  h' = a*h + B*x  over one ragged step delta_k gives
#     h_k = e^{a*delta_k} * h_{k-1}  +  (1 - e^{a*delta_k}) / a  *  B * x_k
# with a = -rate (real).  PARENTHESES MATTER: (1 - e^{a*d}) / a,  NOT  1 - a_bar/a.
#   check: delta->0 must give b_bar->0 ("no time passed, no input").
# delta_k is the gap between step k-1 and step k. Input-dependent delta is the
# selective part: x decides how much time passes, so the kick lands differently.
#
# contract:
#   in   (B, T, d_input)          out (B, T, d_output)
#   state   h = (B, d_state)
#   A parts log_rate (d_state,)          rate = exp(log_rate)
#   B       (d_input, d_state)          C (d_state, d_output)
#   new     W_delta (d_state, d_input)  -- separate from B/C, no bias
#   delta is NOT a parameter. Computed per step from x: shape (B, d_state).
#
# pass-bars before this counts (the six):
#   1 determinism   2 shape across several (B,T)   3 no param with grad None
#   4 BIBO on the TRAINED r                       5 T=2 by hand   6 causality
# watch 3: drop the bias and delta ~ 0.693 at init (softplus(0)), so e^{a*d} != 1
#          and A still matters -- but as W_delta -> 0, delta becomes the SAME
#          constant for every channel and every step. The arm then tests nothing.
#          Signature to look for: delta losing its per-channel and per-step variation.
#
# board: wall 0.20245 / real-pole arm 0.053 / S4D 0.02874 (03 Oct, best).
# pre-register before running: the loss band, AND whether delta_t moves at all on
# 25 stationary sines.
import torch
from torch import nn


class mySelS4D(nn.Module):
    def __init__(self, d_input=1, d_state=16, d_output=1):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_output

        # transition: one real pole per channel (unchanged from my_s4d)
        self.log_rate = nn.Parameter(torch.log(torch.arange(1., d_state + 1)))

        # input / output (no trailing 2 -> the complex form is deferred)
        self.B = nn.Parameter(torch.randn(d_input, d_state))
        self.C = nn.Parameter(torch.randn(d_state, d_output))

        # the dial: x -> one delta per (batch, channel). separate, no bias.
        self.W_delta = nn.Parameter(torch.randn(d_state, d_input))

    def forward(self, x):
        # TODO(me): the selective scan.
        # Del calculated at each timestep
        a = -torch.exp(self.log_rate)
        B, T, C = x.shape
        assert C == self.d_input, "Input of wrong num channels"
        h = torch.zeros((B, self.d_state))
        outputs = []
        for t in range(T):
            delta_raw = torch.einsum('bc,sc->bs', x[:,t,:], self.W_delta)
            delta = torch.log(1+ torch.exp(delta_raw)) # Softplus
            A_bar = torch.exp(a * delta)
            B_bar = ((1 - A_bar) / a) * self.B
            # print(f"A bar: {A_bar.shape}")
            # print(f"B bar: {B_bar.shape}")
            h = A_bar * h + B_bar * x[:,t,:]
            # print(h.shape)
            # Calculate y
            y = h @ self.C
            # print(y.shape)
            outputs.append(y)
        outputs = torch.stack(outputs, dim=1)
        return outputs


if __name__ == '__main__':
    m = mySelS4D(d_input=1, d_state=16, d_output=1)
    for name, p in m.named_parameters():
        print(f"{name:10s} {tuple(p.shape)}")
    dummy_in = torch.sin(torch.randn((10,16,1)))
    outs = m(dummy_in)
    print(outs.shape, (outs**2).mean())
