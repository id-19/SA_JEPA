"""DeltaNet family arms for the shared eval runner."""

import sys
from pathlib import Path

import torch.nn as nn

_evals = Path(__file__).resolve().parent
_dn = _evals.parent / "DeltaNet"
for p in (_evals, _dn):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

from linear_attn import LinFormer
from delta_net import UnifiedDeltaNet
from parallel_dn import ParallelDeltaNet
from parallel_gdn import ParallelGatedDeltaNet
from longhorn import LonghornArm
from mamba2 import Mamba2Arm

ARMS = ["la", "mamba2", "longhorn", "deltanet", "gdn", "kda"]


def _w(d_in, loss):
    return 64 if loss == "ce" else d_in


class ArmsModel(nn.Module):
    def __init__(self, arm, d_in, d_out, loss):
        super().__init__()
        self.emb = nn.Embedding(d_in, 64) if loss == "ce" else None
        w = _w(d_in, loss)
        if arm == "la":
            self.net = LinFormer(D_input=w, D_output=d_out, model_dim=w)
        elif arm == "deltanet":
            self.net = ParallelDeltaNet(w, d_out, w, w)
        elif arm == "gdn":
            self.net = ParallelGatedDeltaNet(w, d_out, w, w)
        elif arm == "kda":
            self.net = UnifiedDeltaNet(w, d_out, w, w, "vector")
        elif arm == "mamba2":
            self.net = Mamba2Arm(w, d_out)
        elif arm == "longhorn":
            self.net = LonghornArm(w, d_out)
        else:
            raise ValueError(f"unknown arm: {arm}")

    def forward(self, x):
        return self.net(self.emb(x) if self.emb is not None else x)
