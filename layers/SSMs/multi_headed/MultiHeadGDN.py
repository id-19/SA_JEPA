import torch
from torch import nn
from torch.linalg import solve_triangular as tri_solve


class MultiHeadGDN(nn.Module):
    def __init__(self, d_input, d_output, d_model, n_heads, rank=4):
        super().__init__()
        self.d_input = d_input
        self.d_output = d_output
        self.d_model = d_model
        self.n_heads = n_heads
        self.rank = rank
        # Initialise W_k, W_q and W_v. No low rank since these genuinely remember features
        self.W_q = nn.Parameter(torch.randn((d_input, d_model)) * (1.0 / (d_input ** 0.5)))
        self.W_k = nn.Parameter(torch.randn((d_input, d_model)) * (1.0 / (d_input ** 0.5)))
        self.W_v = nn.Parameter(torch.randn((d_input, d_model)) * (1.0 / (d_input ** 0.5)))
        ## Low rank stuff: Alpha, Beta, output proj
        # Split all of them into (d1, rank), (rank, d2)
        # Alpha - GDN so (d_input, n_heads)
        self.W_a_high = nn.Parameter(torch.randn((d_input, rank)) / (d_input**0.5))
        self.W_a_low = nn.Parameter(torch.randn((rank, n_heads)))
        # Beta - (d_input, n_heads)
        self.W_b_high = nn.Parameter(torch.randn((d_input, rank)) / (d_input**0.5))
        self.W_b_low = nn.Parameter(torch.randn((rank, n_heads)))
        # Output - (d_model, d_output)
        self.W_o_high = nn.Parameter(torch.randn((d_model, rank)) / (d_model ** 0.5))
        self.W_o_low = nn.Parameter(torch.randn((rank, d_output)))

    def forward(self, x):
        B,T,_ = x.shape
