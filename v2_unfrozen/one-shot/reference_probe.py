# SA-JEPA v2 one-shot — REFERENCE PROBE on the frozen-USM cache (the number to beat).
# Speaker-ID linear probe, pooled per-chunk latents, dev-clean USM cache (certified last night).
# This is the learned-space target: v2's speaker probe must eventually beat this acc.
# One-off, CPU, minutes. Reuses data/latents/dev-clean read-only.
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

REPO = Path(__file__).resolve().parent.parent.parent
LAT = REPO / "data" / "latents" / "dev-clean"


def main():
    rows = list(csv.DictReader(open(LAT / "manifest.csv")))
    utts = sorted({r["utt_id"] for r in rows})
    X, y, names = [], [], []
    for u in utts:
        chunks = [r for r in rows if r["utt_id"] == u]
        vecs = [np.load(LAT / f"{u}_{int(r['chunk']):03d}.npy").astype(np.float32).mean(0) for r in chunks]
        X.append(np.stack(vecs)); y.extend([u.split("-")[0]] * len(vecs)); names.append(u)
    X = np.concatenate(X)
    y = np.array(y)
    print(f"pooled latents: {X.shape}, speakers: {len(set(y))}, chunks: {len(y)}")
    accs = []
    for seed in range(3):
        Xtr, Xev, ytr, yev = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
        # multiclass logistic on standardized features
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=1)
        clf.fit((Xtr - mu) / sd, ytr)
        accs.append(clf.score((Xev - mu) / sd, yev))
    print(f"speaker probe acc (USM frozen space): {np.mean(accs):.4f} ± {np.std(accs):.4f}  (3 seeds, chunk-level)")
    print("This is the REFERENCE number for v2's linear probe on pooled learned latents.")


if __name__ == "__main__":
    main()