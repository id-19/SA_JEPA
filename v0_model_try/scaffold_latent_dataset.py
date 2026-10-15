# SA-CE v0 — SCAFFOLD for Ishaan's model: dataset + board plumbing ONLY.
# The model class (hybrid delta-net layers, heads) is HIS file, HIS forward pass — nothing here trains anything.
# Contract your model must satisfy (from v2_unfrozen/DESIGN_v0.md):
#   input : (B, T, 1536) float32          — cached USM frame latents (6.5 fps)
#   mask  : (B, T) bool                   — padding (truncated chunks pad at the END)
#   output: predictions at delta in {1, 4, 18, 24} — shapes (B, T, 1536) each, valid-masked in the loss
# Board line your trainer must print beside every loss:
#   step | losses(Δ1,Δ4,Δ18,Δ24) | ΔS(Δ24) | blind | memless | linear
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parent.parent

# ---- the four prediction horizons (design doc §3) ----
DELTAS = (1, 4, 18, 24)
WEIGHTS = {1: 0.3, 4: 0.5, 18: 1.5, 24: 1.0}  # 4D weighting — single-horizon ablation arm is the falsification test


class LatentChunkDataset(Dataset):
    """Manifest-driven chunks of cached USM latents, padded/truncated to fixed T.
    Applies per-dim standardization from data/latents/norm_stats.json — the SAME
    transform floors_v0.py applies; different transforms => incomparable boards."""

    def __init__(self, kind: str = "dev", T: int = 195, limit: int = 0):
        self.kind, self.T = kind, T
        stats_p = REPO / "data" / "latents" / "norm_stats.json"
        if stats_p.exists():
            st = json.load(open(stats_p))
            self.mu = torch.tensor(st["mean"], dtype=torch.float32)
            self.sd = torch.tensor(st["std"], dtype=torch.float32)
        else:
            self.mu, self.sd = None, None
        self.rows: list[tuple[str, str, int]] = []
        with open(REPO / "data" / "latents" / f"{kind}-clean" / "manifest.csv") as f:
            for r in csv.DictReader(f):
                self.rows.append((r["utt_id"], r["chunk"], int(r["n_frames"])))
        if limit:
            self.rows = self.rows[:limit]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        utt, chunk, nf = self.rows[i]
        z = np.load(REPO / "data" / "latents" / f"{self.kind}-clean" / f"{utt}_{int(chunk):03d}.npy")  # (nf, 1536) fp16
        z = torch.from_numpy(z).float()
        if self.mu is not None:
            z = (z - self.mu) / self.sd
        T = self.T
        if z.shape[0] >= T:
            z, mask = z[:T], torch.ones(T, dtype=torch.bool)
        else:
            pad = torch.zeros(T - z.shape[0], 1536)
            z = torch.cat([z, pad], dim=0)
            mask = torch.cat([torch.ones(nf, dtype=torch.bool),
                              torch.zeros(T - nf, dtype=torch.bool)])
        return z, mask


def batch_collate(batch):
    zs = torch.stack([b[0] for b in batch])
    ms = torch.stack([b[1] for b in batch])
    return zs, ms


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, delta: int) -> torch.Tensor:
    """MSE over positions where BOTH t and t+delta are unmasked. pred/target: (B, T, 1536)."""
    B, T, _ = pred.shape
    valid = mask.clone()
    valid[:, T - delta:] = False  # future target must exist inside the window
    err = (pred - target) ** 2
    return err[valid].mean() if valid.any() else pred.sum() * 0.0


def load_floors() -> dict:
    p = REPO / "data" / "latents" / "floors_v0.json"
    if not p.exists():
        return {}
    import json
    return json.load(open(p))


def print_board(step: int, losses: dict, floors: dict):
    b = floors.get("delta_24", {})
    line = f"step {step:5d} | " + " ".join(f"L{d}={losses.get(d, float('nan')):.4f}" for d in DELTAS)
    if b:
        best = min(b['blind'], b['memoryless_mlp'], b['linear'])
        line += f" | floors(Δ24): blind {b['blind']:.4f} memless {b['memoryless_mlp']:.4f} linear {b['linear']:.4f}"
        if losses.get(24) is not None and not math.isnan(losses[24]):
            ds = (best - losses[24]) / best
            line += f" | ΔS24 = {ds:+.3f}"
    print(line)


import math  # noqa: E402  (used in print_board)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--T", type=int, default=195)
    args = ap.parse_args()

    ds = LatentChunkDataset("dev", T=args.T, limit=args.limit)
    dl = DataLoader(ds, batch_size=4, shuffle=False, collate_fn=batch_collate)
    z, m = next(iter(dl))
    print(f"dataset rows: {len(ds)} | batch z: {tuple(z.shape)} {z.dtype} | mask: {tuple(m.shape)}, mean {m.float().mean():.3f}")
    print(f"horizons {DELTAS} (weights {WEIGHTS}) | masked_mse smoke: {masked_mse(z, z, m, 24).item():.4f} (same-tensor => 0.0000)")
    fl = load_floors()
    print_board(0, {d: float('nan') for d in DELTAS}, fl)
    print("--- scaffold OK. the forward pass is yours. ---")