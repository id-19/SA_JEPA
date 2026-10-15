# SA-JEPA v2 one-shot — floors on the MEL substrate (board bar for the learned model).
# Run AFTER build_mel_cache.py. New substrate = new board; never compared to USM-latent boards.
# Floors: blind (predict batch/global mean) and closed-form linear (lstsq) at d ∈ {1,10,50,150}.
# NOTE: linear floor here is LINEAR IN MEL FRAME (not in any learned space) — it is the honest bar.
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

SR, FPS, NMEL = 16000, 50, 80
T_WIN = 500  # 10 s window

HERE = Path(__file__).resolve().parent
CACHE = HERE / "mel_cache"
DELTAS = [1, 10, 50, 150]  # frames @50fps = 20ms/0.2s/1s/3s


def load_windows(limit: int):
    rows = list(csv.DictReader(open(CACHE / "windows.csv")))
    if limit:
        rows = rows[:limit]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-windows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    rows = load_windows(args.n_windows + 50)
    picks = rng.choice(len(rows), size=min(args.n_windows, len(rows)), replace=False)

    feats, targs = [], []
    for W in DELTAS:
        X, Y = [], []
        for j in picks:
            r = rows[j]
            mel = torch.load(CACHE / f"{r['utt']}.pt").float().numpy()  # (nf, 80)
            t0 = int(r["t0"])
            # pairs: (t, t+d) inside the window, stride 5 to decorrelate
            for t in range(t0, t0 + int(r["n_frames"]) - W, 5):
                X.append(mel[t]); Y.append(mel[t + W])
            if len(X) > 4000:
                break
        X = np.stack(X); Y = np.stack(Y)
        split = int(0.8 * len(X))
        Xtr, Xev = X[:split], X[split:]
        Ytr, Yev = Y[:split], Y[split:]

        mu = Xtr.mean(0)
        blind = ((Yev - mu) ** 2).mean()  # blind predicts the marginal mean
        Wls, *_ = np.linalg.lstsq(Xtr, Ytr, rcond=None)
        pred = Xev @ Wls
        linear = ((pred - Yev) ** 2).mean()
        print(f"d={W:3d} ({W/FPS:.2f}s): blind {blind:.5f} | linear {linear:.5f} | variance ratio {linear/blind:.3f}")
        feats.append((W, blind, linear))

    print("\nBOARD BAR (mel substrate, dB-squared units):")
    print("| d | blind | linear |")
    print("|---|---|---|")
    for W, b, l in feats:
        print(f"| {W} | {b:.4f} | {l:.4f} |")


if __name__ == "__main__":
    main()