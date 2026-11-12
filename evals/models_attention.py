"""Attention family arms for the shared eval runner.

vanilla is VanillaAttention (scaled-dot-product forward).
gqa is VanillaGQA through efficient_simple_forward.
"""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch.nn as nn

from Attention_layers.gqa import VanillaGQA
from Attention_layers.vanilla import VanillaAttention

ARMS = ["vanilla", "gqa"]


def _w(d_in, loss):
    return 64 if loss == "ce" else d_in


def _n_heads(d_model):
    for h in (8, 4, 2, 1):
        if d_model % h == 0:
            return h
    return 1


def _grouping(n_heads):
    for g in (4, 2, 1):
        if n_heads % g == 0:
            return g
    return 1


class EfficientGQA(nn.Module):
    """Runner calls forward(x). The timed path is efficient_simple_forward."""

    def __init__(self, d_in, d_out, n_heads):
        super().__init__()
        head = d_in // n_heads
        self.gqa = VanillaGQA(
            d_in, d_out, head, head, n_heads, _grouping(n_heads),
        )

    def forward(self, x):
        return self.gqa.efficient_simple_forward(x)


class ArmsModel(nn.Module):
    def __init__(self, arm, d_in, d_out, loss):
        super().__init__()
        self.emb = nn.Embedding(d_in, 64) if loss == "ce" else None
        w = _w(d_in, loss)
        n_heads = _n_heads(w)
        if arm == "vanilla":
            # Block width stays w. self.out maps w -> d_out when the task is wider (CE).
            self.net = VanillaAttention(w, w, w, n_heads)
        elif arm == "gqa":
            self.net = EfficientGQA(w, w, n_heads)
        else:
            raise ValueError(f"unknown arm: {arm}")
        self.out = nn.Identity() if w == d_out else nn.Linear(w, d_out)

    def forward(self, x):
        if self.emb is not None:
            x = self.emb(x)
        return self.out(self.net(x))
