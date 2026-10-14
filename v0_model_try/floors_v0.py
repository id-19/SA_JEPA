# SA-CE v0 — FLOORS for the frozen-latent dataset (board discipline: floors BEFORE models).
# Three floor predictors on cached USM latents:
#   L  = closed-form linear lstsq   (z_t -> z_{t+delta})
#   M  = best memoryless nonlinear  (small MLP on z_t -> z_{t+delta})
#   B  = blind                      (predict the per-dim mean of the target)
# Board line (per delta):  B | M | L | <your model>
# Different dataset => different board; these numbers NEVER mix with the sine board.
import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
DEVICE = "cpu"


def load_manifest(kind: str) -> list[tuple[str, str, int, float]]:
    rows = []
    with open(REPO / "data" / "latents" / f"{kind}-clean" / "manifest.csv") as f:
        for r in csv.DictReader(f):
            rows.append((r["utt_id"], r["chunk"], int(r["n_frames"]), float(r["dur"])))
    return rows


def _norm():
    p = REPO / "data" / "latents" / "norm_stats.json"
    if not p.exists():
        return None, None
    st = json.load(open(p))
    return np.array(st["mean"], dtype=np.float32), np.array(st["std"], dtype=np.float32)


def sample_pairs(rows, delta: int, n: int, seed: int = 0, split: str = "train", mu=None, sd=None):
    """(z_t, z_{t+delta}) pairs sampled inside-chunk; split='train'|'eval' picks utterance subset.
    Standardization (mu, sd) identical to the dataset scaffold's — one transform, one board."""
    rng = random.Random(seed)
    rows = sorted(rows)
    n_all = len(rows)
    idx_eval = set(range(int(n_all * 0.9), n_all))
    pool = [r for i, r in enumerate(rows) if (i in idx_eval) == (split == "eval")]
    pairs = []
    tries = 0
    while len(pairs) < n and tries < n * 20:
        tries += 1
        utt, chunk, nf, _ = pool[rng.randrange(len(pool))]
        t = rng.randrange(0, max(1, nf - delta))
        p = REPO / "data" / "latents" / "dev-clean" / f"{utt}_{int(chunk):03d}.npy"  # v0: dev-clean only
        z = np.load(p).astype(np.float32)
        if mu is not None:
            z = (z - mu) / sd
        if t + delta < z.shape[0]:
            pairs.append((z[t], z[t + delta]))
    X = torch.from_numpy(np.stack([p[0] for p in pairs])).float()
    Y = torch.from_numpy(np.stack([p[1] for p in pairs])).float()
    return X, Y


def floor_linear(Xtr, Ytr, Xev, Yev):
    # closed-form ridge-free lstsq on the train pairs, scored on eval pairs
    Xb = torch.cat([Xtr, torch.ones(Xtr.shape[0], 1)], dim=1)
    W = torch.linalg.lstsq(Xb, Ytr).solution
    Xe = torch.cat([Xev, torch.ones(Xev.shape[0], 1)], dim=1)
    pred = Xe @ W
    return ((pred - Yev) ** 2).mean().item()


def floor_blind(Ytr, Yev):
    mu = Ytr.mean(dim=0)
    return ((Yev - mu) ** 2).mean().item()


class TinyMLP(nn.Module):
    def __init__(self, d_in=1536, d_hidden=512, d_out=1536):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.SiLU(), nn.Linear(d_hidden, d_out))

    def forward(self, x):
        return self.net(x)


def floor_memoryless(Xtr, Ytr, Xev, Yev, steps=300, seed=0):
    torch.manual_seed(seed)
    m = TinyMLP().to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    ds = torch.utils.data.TensorDataset(Xtr, Ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    for step in range(steps):
        for xb, yb in dl:
            loss = ((m(xb) - yb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = m(Xev)
    return ((pred - Yev) ** 2).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-eval", type=int, default=4000)
    ap.add_argument("--deltas", default="1,4,18,24")
    ap.add_argument("--out", default="data/latents/floors_v0.json")
    args = ap.parse_args()

    rows = load_manifest("dev")
    mu, sd = _norm()
    out = {}
    for delta in [int(d) for d in args.deltas.split(",")]:
        Xtr, Ytr = sample_pairs(rows, delta, args.n_train, seed=0, split="train", mu=mu, sd=sd)
        Xev, Yev = sample_pairs(rows, delta, args.n_eval, seed=1, split="eval", mu=mu, sd=sd)
        res = {
            "blind": floor_blind(Ytr, Yev),
            "linear": floor_linear(Xtr, Ytr, Xev, Yev),
            "memoryless_mlp": floor_memoryless(Xtr, Ytr, Xev, Yev),
        }
        out[f"delta_{delta}"] = res
        print(f"delta={delta:2d}  blind={res['blind']:.4f}  linear={res['linear']:.4f}  memless={res['memoryless_mlp']:.4f}")

    (REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(REPO / args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()