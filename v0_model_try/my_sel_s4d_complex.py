# my_sel_s4d_complex.py — selective Delta, complex pole: a = -rate + i*omega
#
# contract:
#   in   (B, T, d_input)                    out (B, T, d_output)   REAL
#   state  h = (B, d_state)                COMPLEX64 -- one complex number per channel
#   log_rate (d_state,)  rate = exp(log_rate)
#   omega    (d_state,)  one rotation per channel
#   a        = -rate + i*omega             COMPLEX64
#   B        (d_input, d_state)  COMPLEX64  C (d_state, d_output) COMPLEX64
#   W_delta  (d_state, d_input)  REAL       delta is a duration, not a rotation
#   delta per step from x: (B, d_state), real, = softplus(x_t @ W_delta)
#
# per step:  A_bar = exp(a*delta)   B_bar = (1 - A_bar)/a
#            kick  = x_t @ B        h = h*A_bar + B_bar*kick
#            y     = Re(<h, conj(C)>)          <- conj is load-bearing
#
# watch: W_delta -> 0 makes delta the SAME constant for every channel and every
#        step -- one shared rotation, and the arm tests nothing.
#
# board: wall 0.20245 / real-pole par 0.053 / S4D 0.02874 / selective real-pole 0.019658
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

        self.B = nn.Parameter(torch.randn(d_input, d_state, dtype=torch.complex64))
        self.C = nn.Parameter(torch.randn(d_state, d_output, dtype=torch.complex64))

        self.W_delta = nn.Parameter(torch.randn(d_state, d_input))

    def forward(self, x):
        a = torch.complex(-torch.exp(self.log_rate), self.omega)
        a = a.to(torch.complex64)
        B,T,C = x.shape
        h = torch.zeros((B, self.d_state), dtype=torch.complex64)
        outputs = []
        for t in range(T):
            delta_raw = torch.einsum('bc,sc->bs', x[:,t,:], self.W_delta)
            delta = nn.functional.softplus(delta_raw) # torch.log(1+ torch.exp(delta_raw)) # Softplus
            A_bar = torch.exp(a * delta)
            B_bar = (1 - A_bar) / a
            kick = torch.einsum('bc,cs->bs', x[:,t, :].to(self.B.dtype), self.B)
            h = h * A_bar + B_bar * kick

            y = torch.einsum('bs,so->bo', h, self.C.conj()).real
            outputs.append(y)
        return torch.stack(outputs, dim=1) # (B, T, C)

if __name__ == '__main__':
    d_input = 80
    d_output = 5
    m = mySelS4DComplex(d_input=d_input, d_output=d_output)
    dummy_in = torch.sin(torch.randn((10,16,d_input)))
    outs = m(dummy_in)
    print(outs.shape, (outs**2).mean())
