# Define a very basic Mamba block in pytorch
# Mamba 1 implementation
from this import d

import torch
import torch.nn as nn

# VERY BASIC SSM IMPLEMENTATION
# A, B, C, delta
class SSM(nn.Module):
    def __init__(self, d_input:int, d_latent:int, d_state:int, d_output:int):
        super().__init__()
        # Record the dimensions
        self.d_input = d_input
        self.d_state = d_state
        self.d_output = d_output

        # Define state
        self.state:torch.Tensor = torch.zeros(d_state)

        # Record the parameters
        self.A = torch.randn((d_state, d_state))
        self.B = torch.randn((d_input, d_latent))
        self.C = torch.randn((d_state, d_output))
