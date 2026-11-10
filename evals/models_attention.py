"""Attention family arms for the shared eval runner."""

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import torch.nn as nn

ARMS = ["vanilla", "gqa", "mla"]


def _w(d_in, loss):
    return 64 if loss == "ce" else d_in


def _n_heads(d_model):
    for h in (8, 4, 2, 1):
        if d_model % h == 0:
            return h
    return 1


class ArmsModel(nn.Module):
    def __init__(self, arm, d_in, d_out, loss):
        super().__init__()
        self.emb = nn.Embedding(d_in, 64) if loss == "ce" else None
        w = _w(d_in, loss)
        n_heads = _n_heads(w)
        if arm == "vanilla":
            from Attention_layers.vanilla import VanillaAttention
            layer = VanillaAttention(w, n_heads)
        elif arm == "gqa":
            from Attention_layers.gqa import GQA
            layer = GQA(w, n_heads)
        elif arm == "mla":
            from Attention_layers.mla import MLA
            layer = MLA(w, n_heads)
        else:
            raise ValueError(f"unknown arm: {arm}")
        self.net = layer
        self.out = nn.Identity() if w == d_out else nn.Linear(w, d_out)

    def forward(self, x):
        if self.emb is not None:
            x = self.emb(x)
        return self.out(self.net(x))
