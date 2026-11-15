"""Bug hunt for [1.2] (ID1-280). See QUIZ.md, part 2.

A copy of layers/SSMs/reference/kda.py with ONE planted bug. The block description below is
the spec and is correct; the code departs from it in one place. Do not diff this file
against the original.

Block, bottom to top:
    q, k: Linear -> causal depthwise conv -> SiLU -> split heads -> L2 per head
    v:    Linear -> causal depthwise conv -> SiLU -> split heads
    g     = -exp(A_log) * softplus(f_proj(x) + dt_bias)   log decay, (n_v_heads, dk) per token, <= 0
    beta  = sigmoid(x W_b)                                (times 2 with allow_neg_eigval)
    o     = KDA(q, k, v, g, beta)
    o     = RMSNorm_per_head(o) * sigmoid(g_proj(x))       low-rank output gate
    y     = o W_o

With n_v_heads > n_heads, each q/k head is shared by a group of value heads; the state has
one (dk, dv) matrix per value head.

The residual (x + mixer(norm(x))) belongs to the block that wraps this layer.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from layers.SSMs.reference.gated_deltanet import CausalDepthwiseConv
from layers.SSMs.reference.kda_ops import chunk_kda, recurrent_kda


class KimiDeltaAttentionBugged(nn.Module):
    """forward(x, state=None, return_state=False) -> y, or (y, state) with return_state.

    state is a plain dict, None meaning a fresh stream:
        "S":      (B, n_v_heads, dk, dv)   delta-rule matrix, one per value head
        "conv_q": (B, k-1, key_dim)        last k-1 conv inputs of q, k, v
        "conv_k": (B, k-1, key_dim)
        "conv_v": (B, k-1, value_dim)
    mode "auto" uses the recurrent op for pieces of T <= 8 tokens (decode), else chunked.
    """

    def __init__(self, d_model, n_heads, head_dim=None, expand_v=1.0, n_v_heads=None,
                 conv_size=4, chunk_size=64, mode="auto", inverse="doubling", gram="auto",
                 allow_neg_eigval=False, norm_eps=1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.n_v_heads = n_v_heads or n_heads
        if self.n_v_heads % n_heads:
            raise ValueError("n_v_heads must be a multiple of n_heads")
        self.dk = head_dim or d_model // n_heads
        self.dv = int(self.dk * expand_v)
        self.conv_size = conv_size
        self.chunk_size = chunk_size
        self.mode = mode
        self.inverse = inverse
        self.gram = gram
        self.allow_neg_eigval = allow_neg_eigval
        self.norm_eps = norm_eps
        key_dim = n_heads * self.dk
        value_dim = self.n_v_heads * self.dv
        gate_dim = self.n_v_heads * self.dk

        self.q_proj = nn.Linear(d_model, key_dim, bias=False)
        self.k_proj = nn.Linear(d_model, key_dim, bias=False)
        self.v_proj = nn.Linear(d_model, value_dim, bias=False)
        self.q_conv = CausalDepthwiseConv(key_dim, conv_size)
        self.k_conv = CausalDepthwiseConv(key_dim, conv_size)
        self.v_conv = CausalDepthwiseConv(value_dim, conv_size)

        # Low-rank decay gate: one logit per value head per key channel.
        self.f_proj = nn.Sequential(nn.Linear(d_model, self.dv, bias=False),
                                    nn.Linear(self.dv, gate_dim, bias=False))
        self.A_log = nn.Parameter(torch.empty(self.n_v_heads).uniform_(1, 16).log())
        # dt_bias = inverse-softplus of a step size drawn log-uniformly in [1e-3, 1e-1].
        dt = torch.exp(torch.rand(gate_dim) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        dt = dt.clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))

        self.b_proj = nn.Linear(d_model, self.n_v_heads, bias=False)
        self.g_proj = nn.Sequential(nn.Linear(d_model, self.dv, bias=False),
                                    nn.Linear(self.dv, value_dim, bias=True))
        self.o_norm_weight = nn.Parameter(torch.ones(self.dv))
        self.o_proj = nn.Linear(value_dim, d_model, bias=False)

    def state_floats(self):
        """Floats of state per batch element: the layer's whole memory, independent of T."""
        key_dim, value_dim = self.n_heads * self.dk, self.n_v_heads * self.dv
        return (self.n_v_heads * self.dk * self.dv
                + (self.conv_size - 1) * (2 * key_dim + value_dim))

    def gates(self, x):
        """Log decay g (B, Hv, T, dk), <= 0, and write strength beta (B, Hv, T)."""
        B, T, _ = x.shape
        f = self.f_proj(x).float().view(B, T, self.n_v_heads, self.dk)
        g = -self.A_log.float().exp()[:, None] * F.softplus(f + self.dt_bias.view(self.n_v_heads, self.dk))
        beta = torch.sigmoid(self.b_proj(x).float())
        if self.allow_neg_eigval:
            beta = beta * 2.0
        return g.transpose(1, 2), beta.transpose(1, 2)

    def forward(self, x, state=None, return_state=False):
        B, T, _ = x.shape
        H, Hv, dk, dv = self.n_heads, self.n_v_heads, self.dk, self.dv
        state = state or {}

        q, tq = self.q_conv(self.q_proj(x), state.get("conv_q"))
        k, tk = self.k_conv(self.k_proj(x), state.get("conv_k"))
        v, tv = self.v_conv(self.v_proj(x), state.get("conv_v"))
        q, k, v = F.silu(q), F.silu(k), F.silu(v)
        # (B, T, H*d) -> (B, H, T, d)
        q = F.normalize(q, dim=-1).view(B, T, H, dk).transpose(1, 2)
        k = F.normalize(k, dim=-1).view(B, T, H, dk).transpose(1, 2)
        v = v.view(B, T, Hv, dv).transpose(1, 2)
        if Hv > H:                       # each q/k head serves a group of value heads
            q = q.repeat_interleave(Hv // H, dim=1)
            k = k.repeat_interleave(Hv // H, dim=1)

        g, beta = self.gates(x)

        mode = self.mode
        if mode == "auto":
            mode = "recurrent" if T <= 8 else "chunk"
        if mode == "chunk":
            o, S = chunk_kda(q, k, v, g, beta, initial_state=state.get("S"),
                             chunk_size=self.chunk_size, inverse=self.inverse, gram=self.gram)
        elif mode == "recurrent":
            o, S = recurrent_kda(q, k, v, g, beta, initial_state=state.get("S"))
        else:
            raise ValueError(f"unknown mode {self.mode!r}")

        o = o.transpose(1, 2)                                                   # (B, T, Hv, dv)
        o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + self.norm_eps) * self.o_norm_weight
        o = o * torch.sigmoid(self.g_proj(x).view(B, T, Hv, dv))
        y = self.o_proj(o.reshape(B, T, Hv * dv))
        if return_state:
            return y, {"S": S, "conv_q": tq, "conv_k": tk, "conv_v": tv}
        return y
