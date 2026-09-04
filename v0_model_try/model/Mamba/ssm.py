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
        self.B = torch.randn((d_latent, d_state))
        self.C = torch.randn((d_state, d_output))
        # Latent projection as well
        self.proj = nn.Linear(d_input, d_latent)


    def forward(self, x: torch.Tensor):
        if x.dim == 3:
            B,T,C = x.shape
        else:
            T,C = x.shape
            B = 1
        x = x.view(B*T, C)

        if x.shape[-1] != self.d_input:
            raise Exception("Input to SSM is not of the expected dimensions")

        # Project input
        x = self.proj(x)

        # Update state
        # print("Initial shape of state", self.state.shape)
        for eff_timestep in range(x.shape[0]):
            # print(f"State at timestep, {eff_timestep} is: {self.state}")
            self.state = self.A @ self.state # (d_state, d_state) @ (d_state, 1) = (d_state)
            self.state += x[eff_timestep] @ self.B # (1, d_latent) @ (d_latent, d_state) = (d_state)
            # print("Output at ", eff_timestep, ": ", self.state @ self.C)
        # print("Final shape of state", self.state.shape)

        # Calculate output
        output = self.state @ self.C
        return output


def test_const_delta_ssm(d_input = 10, d_latent = 5, d_state = 10, d_output = 5, inputs = torch.randn((10, 10))):
    ssm = SSM(d_input=d_input, d_latent=d_latent, d_state=d_state, d_output=d_output)
    # Go thru the inputs one by one
    print("Shape of state initially", ssm.state.shape)
    output = ssm(inputs)
    print("Final state", ssm.state.shape)
    print("Output", output)

if __name__ == "__main__":
    test_const_delta_ssm()
