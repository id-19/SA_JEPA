"""Rotary position embedding, one or several axes. AI-written reference (13 Nov 2026): a
correctness oracle and speed baseline, not the learning implementation.

    positions   (T,) or (B, T)            one axis
                (T, A) or (B, T, A)       A axes (axis_dims has A entries); int or float
    cos, sin    (..., T, dim // 2)        float32
    x           (B, H, T, dim) or (B, T, dim)

    angle[..., i] = positions[..., axis(i)] * inv_freq[i],   inv_freq[i] = base ** (-2j / d_axis)
    (x[2i], x[2i+1]) -> (x[2i] cos - x[2i+1] sin,  x[2i] sin + x[2i+1] cos)

Each axis owns a slice of the pairs and its own frequency ladder (j counts inside the slice), so
for images (row, col) or video (time, row, col) each coordinate is rotated independently. The
rotation makes <rot(q, p), rot(k, p')> depend only on p - p' on every axis.
"""

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000.0, axis_dims=None):
        super().__init__()
        axis_dims = tuple(axis_dims) if axis_dims is not None else (dim,)
        assert sum(axis_dims) == dim and all(d % 2 == 0 for d in axis_dims), "axis_dims: even, sum to dim"
        self.dim, self.axis_dims = dim, axis_dims
        ladders = [base ** (-torch.arange(0, d, 2, dtype=torch.float32) / d) for d in axis_dims]
        axis_of_pair = torch.cat([torch.full((d // 2,), a) for a, d in enumerate(axis_dims)])
        self.register_buffer("inv_freq", torch.cat(ladders), persistent=False)
        self.register_buffer("axis_of_pair", axis_of_pair, persistent=False)

    def forward(self, positions):
        pos = positions.to(self.inv_freq.device).float()    # angles in float32 even for half models
        if len(self.axis_dims) == 1:
            pos = pos.unsqueeze(-1)                         # single axis: add the axis dim
        angle = pos[..., self.axis_of_pair] * self.inv_freq
        return angle.cos(), angle.sin()


def apply_rope(x, cos, sin):
    if x.ndim == 4 and cos.ndim == 3:
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)       # broadcast over heads
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


def default_positions(positions, T, offset, device, rope):
    """Caller's positions, else arange(T) shifted by the tokens already in the cache."""
    if positions is not None:
        return positions
    assert len(rope.axis_dims) == 1, "multi-axis RoPE needs explicit positions"
    return torch.arange(T, device=device) + offset
