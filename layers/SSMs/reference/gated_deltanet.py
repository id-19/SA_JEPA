"""Reference Gated DeltaNet layer (token mixer only, no residual) for MPS/CPU.

Reference implementation written by an AI agent (13 Nov 2026), following the Gated
DeltaNet block (Yang et al. 2024, arXiv 2412.06464) and the structure of the fla
reference layer. It is a correctness oracle and a speed baseline for the user's own
layer in layers/SSMs/multi_headed/, not a replacement for it.

Block, bottom to top (the paper's figure):
    q, k: Linear -> causal depthwise conv -> SiLU -> split heads -> L2 per head
    v:    Linear -> causal depthwise conv -> SiLU -> split heads
    alpha = exp(-exp(A_log) * softplus(x W_a + dt_bias))   per head, in (0, 1]
    beta  = sigmoid(x W_b)                                  per head, in (0, 1)
    o     = gated delta rule(q, k, v, alpha, beta)
    o     = RMSNorm_per_head(o) * SiLU(x W_g)               output gate
    y     = o W_o

The residual (x + mixer(norm(x))) belongs to the block that wraps this layer.
Streaming: forward(x, state, return_state=True) carries the whole memory of the past (the
delta-rule matrix and the conv tails), so a sequence fed in pieces equals one call.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .gated_delta_rule import chunk_gated_delta_rule, recurrent_gated_delta_rule


class CausalDepthwiseConv(nn.Module):
    """Each channel gets its own k-tap filter over the last k tokens (groups = channels)."""

    def __init__(self, dim, kernel_size=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(dim, dim, kernel_size, groups=dim, bias=False)

    def forward(self, x, tail=None):                        # x: (B, T, D)
        """tail: the last k-1 inputs of the previous call (B, k-1, D), None at stream start.
        Returns (y, new_tail)."""
        pad = self.kernel_size - 1
        if tail is None:
            tail = x.new_zeros(x.shape[0], pad, x.shape[2])     # left pad only: causal
        x = torch.cat([tail.to(x.dtype), x], dim=1)
        new_tail = x[:, x.shape[1] - pad:]                      # also right if T < k-1
        if x.device.type != "cpu":
            return self.conv(x.transpose(1, 2)).transpose(1, 2), new_tail
        # CPU: conv1d with groups == channels runs one slow conv per channel (measured at
        # D=1024: 12.7 ms for a single decode token, 51 ms fwd+bwd at T=2048). The same k-tap
        # filter as a sum of k shifted, scaled copies is 0.02 ms and 13 ms. On MPS it is the
        # other way round (conv1d 8 ms, taps 22 ms), hence the switch.
        w = self.conv.weight                                    # (D, 1, k)
        T = x.shape[1] - pad
        y = x[:, 0:T] * w[:, 0, 0]
        for j in range(1, self.kernel_size):
            y = y + x[:, j:j + T] * w[:, 0, j]
        return y, new_tail


class GatedDeltaNetRef(nn.Module):
    """forward(x, state=None, return_state=False) -> y, or (y, state) with return_state.

    state is a plain dict, None meaning a fresh stream:
        "S":      (B, H, Dk, Dv)        delta-rule matrix
        "conv_q": (B, k-1, key_dim)     last k-1 conv inputs of q, k, v
        "conv_k": (B, k-1, key_dim)
        "conv_v": (B, k-1, value_dim)
    mode "auto" uses the recurrent op for pieces of T <= 8 tokens (decode), else chunked.
    """

    def __init__(self, d_model, n_heads, head_k_dim=None, head_v_dim=None,
                 conv_size=4, chunk_size=64, mode="auto", inverse="doubling", norm_eps=1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.dk = head_k_dim or d_model // n_heads
        self.dv = head_v_dim or d_model // n_heads
        self.chunk_size = chunk_size
        self.conv_size = conv_size
        self.mode = mode
        self.inverse = inverse
        key_dim, value_dim = n_heads * self.dk, n_heads * self.dv

        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, value_dim, bias=False)
        self.q_conv = CausalDepthwiseConv(key_dim, conv_size)
        self.k_conv = CausalDepthwiseConv(key_dim, conv_size)
        self.v_conv = CausalDepthwiseConv(value_dim, conv_size)

        self.a_proj = nn.Linear(d_model, n_heads, bias=False)
        self.b_proj = nn.Linear(d_model, n_heads, bias=False)
        # Mamba-2 style decay. A in (0, 16): per-head forgetting rate.
        self.A_log = nn.Parameter(torch.empty(n_heads).uniform_(0, 16).log())
        # dt_bias = inverse-softplus of a step size drawn log-uniformly in [1e-3, 1e-1],
        # so heads start with a spread of memory lengths.
        dt = torch.exp(torch.rand(n_heads) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        dt = dt.clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.g_proj = nn.Linear(d_model, value_dim, bias=False)
        self.o_norm_weight = nn.Parameter(torch.ones(self.dv))
        self.norm_eps = norm_eps
        self.o_proj = nn.Linear(value_dim, d_model, bias=False)

    def state_floats(self):
        """Floats of state per batch element: the layer's whole memory, independent of T."""
        tails = (self.conv_size - 1) * (2 * self.n_heads * self.dk + self.n_heads * self.dv)
        return self.n_heads * self.dk * self.dv + tails

    def forward(self, x, state=None, return_state=False):
        B, T, _ = x.shape
        H, dk, dv = self.n_heads, self.dk, self.dv
        state = state or {}

        q, tq = self.q_conv(self.q_proj(x), state.get("conv_q"))
        k, tk = self.k_conv(self.k_proj(x), state.get("conv_k"))
        v, tv = self.v_conv(self.v_proj(x), state.get("conv_v"))
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        # (B, T, H*d) -> (B, H, T, d); L2 per head, after the split
        q = F.normalize(q.view(B, T, H, dk), dim=-1).transpose(1, 2)
        k = F.normalize(k.view(B, T, H, dk), dim=-1).transpose(1, 2)
        v = v.view(B, T, H, dv).transpose(1, 2)

        beta = torch.sigmoid(self.b_proj(x)).transpose(1, 2)                    # (B, H, T)
        g = -self.A_log.float().exp() * F.softplus(self.a_proj(x).float() + self.dt_bias)
        g = g.transpose(1, 2)                                                   # log alpha, <= 0

        mode = self.mode
        if mode == "auto":
            mode = "recurrent" if T <= 8 else "chunk"
        if mode == "chunk":
            o, S = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=state.get("S"),
                                          chunk_size=self.chunk_size, inverse=self.inverse)
        elif mode == "recurrent":
            o, S = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=state.get("S"))
        else:
            raise ValueError(f"unknown mode {self.mode!r}")

        o = o.transpose(1, 2)                                                   # (B, T, H, dv)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.o_norm_weight
        o = o * F.silu(self.g_proj(x).view(B, T, H, dv))
        y = self.o_proj(o.reshape(B, T, H * dv))
        if return_state:
            return y, {"S": S, "conv_q": tq, "conv_k": tk, "conv_v": tv}
        return y
