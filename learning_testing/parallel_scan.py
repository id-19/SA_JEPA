## How the parallel scan works
# 1. Calculate delta for every step - Can probably express this as a single mat mul?
# 2. Use delta at every step to find Abar and Bgain - M_step, V_step
# 3. Each state update is expressed as h_i = Abar.h_(i-1) + Bgain.u - where u is B.x
# Now the composition rule for L -> k -> R: (M_R, V_R).(M_L, V_L) = (M_R.M_L, M_R.V_L + V_R)
# Calculate in a tree structure, bottoms-up, the transition matrices M and V for every combo of steps
# Make an array of size T, and for each round(zero indexed), i*(2**round)->i*(2**(round+1)) is stored at index i
# Use these to find state, and hence output

# Put everything on MPS, without a loop it's way faster
# For v0, do naively, for v1 can parallelise, for v2 can compute output and state together to avoid storing matrices
import torch
from torch import nn
import math

class ParallelScanSSM(nn.Module):
    def __init__(self, d_input=1, d_state=16, d_output=1):
        super().__init__()
        self.d_input, self.d_state, self.d_output = d_input, d_state, d_output

        self.log_rate = nn.Parameter(torch.log(torch.arange(1., d_state + 1)))      # (d_state,)  REAL
        self.omega    = nn.Parameter(torch.arange(-3.14, 3.14, 6.28 / d_state))     # (d_state,)  REAL
        self.W_delta  = nn.Parameter(torch.randn(d_input, d_state))                 # (d_in, d_s) REAL
        self.B        = nn.Parameter(torch.randn(d_state, d_input, dtype=torch.complex64))
        self.C        = nn.Parameter(torch.randn(d_state, d_output, dtype=torch.complex64))
        self.device = torch.backends.mps


    def forward(self, x):
        def compose_M_V(r:int, l:int):
            r,l = max(r,l), min(r,l)
            M_R, V_R = Abar[:,r,:],U[:,r,:]
            M_L, V_L = Abar[:,l,:],U[:,l,:]
            # same dims, just multiply
            return (M_R * M_L, M_R*V_L + V_R)
        x = x.to(self.device)
        # 1. Calculate delta fro every step
        B, T, C = x.shape
        delta_allt_raw = torch.einsum('btc,cs->bts', x, self.W_delta)
        delta_allt = torch.nn.functional.softplus(delta_allt_raw)

        # 2. Use delta to calculate Abar and Bbar
        # abar
        a = torch.complex(-torch.exp(self.log_rate), self.omega)
        Abar = torch.exp(a * delta_allt).to(torch.complex64) # (s * bts) -> bts, broadcasted
        # bbar
        Bgain = (1 - Abar) / a # see as * 1/a. Dims: bts * s -> bts with broadcast
        # calculate kicks
        U = Bgain * torch.einsum('btc,sc->bts', x, self.B)

        # 3. Calculate state updates and outputs
        # Make an array of size T, and for each round(zero indexed), i*(2**round)->i*(2**(round+1)) is stored at index i
        outputs = torch.zeros((B, T)) # [:, 0] is already all zeroes
        total_rounds = math.ceil(math.log2(T))
        for round in range(total_rounds):
            # Abar is M, U is V
            # Populate leftmost node to outputs
            curr_leftmost = 1 * min(round**2, T)
            outputs[:,curr_leftmost] =
