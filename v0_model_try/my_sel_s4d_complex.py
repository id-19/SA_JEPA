# my_sel_s4d_complex.py — selective Delta, complex pole: a = -rate + i*omega
#
# contract:
#   in   (B, T, d_input)          out (B, T, d_output)
#   state    h = (B, d_state, 2)              re, im per channel
#   log_rate (d_state,)   rate = exp(log_rate)
#   omega    (d_state,)
#   B        (d_input, d_state, 2)     C (d_state, 2, d_output)
#   W_delta  (d_state, d_input)        REAL -- delta is a duration
#   delta per step from x: (B, d_state).  rho = exp(-rate*delta), theta = omega*delta
#
# W_delta -> 0 makes delta constant: every channel shares one rotation.
#
# board: wall 0.20245 / real-pole par 0.053 / S4D 0.02874 / selective real-pole 0.02070
import torch
from torch import nn


class mySelS4DComplex(nn.Module):
    def __init__(self, d_input=1, d_state=16, d_output=1):
        super().__init__()
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_output

        # one COMPLEX pole per channel
        self.log_rate = nn.Parameter(torch.log(torch.arange(1., d_state + 1)))
        self.omega = nn.Parameter(torch.arange(-3.14, 3.14, 6.28 / d_state))

        self.B = nn.Parameter(torch.randn(d_input, d_state, 2))
        self.C = nn.Parameter(torch.randn(d_state, 2, d_output))

        self.W_delta = nn.Parameter(torch.randn(d_state, d_input))

    def forward(self, x):
        pass
