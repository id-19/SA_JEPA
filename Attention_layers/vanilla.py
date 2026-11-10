# Implementing a vanilla attention layer
import torch
from torch import nn

class VanillaAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(VanillaAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, x):