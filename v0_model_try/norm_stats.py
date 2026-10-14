# SA-CE v0 — latent normalization stats.
# WHY: the frozen USM tower emits 2 near-constant giant dims (+1203, -1202) that poison cosine
# geometry entirely (all raw cos ≈ 1.0). Every consumer (floors, dataset, model in-proj) must
# apply THE SAME per-dim standardization, computed HERE once and stored.
# Usage: uv run python v0_model_try/norm_stats.py [--k 40]
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
GIANT_FALLBACK = [500, 1193]  # found by usm_giant_probe; recomputed-and-checked below anyway


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=40, help="chunks sampled for stats")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    lat_root = REPO / "data" / "latents" / "dev-clean"
    man = lat_root / "manifest.csv"
    if not man.exists():
        sys.exit("no manifest — run encode_usm.py first")
    rows = list(csv.DictReader(open(man)))
    files = [lat_root / f"{r['utt_id']}_{int(r['chunk']):03d}.npy" for r in rows]
    files = [f for f in files if f.exists()]
    rng = np.random.default_rng(0)
    k = min(args.k, len(files))
    picks = [files[i] for i in rng.choice(len(files), size=k, replace=False)]

    chunks = []
    for f in picks:
        chunks.append(np.load(f).astype(np.float32))
    A = np.concatenate(chunks, axis=0)  # (N_frames, 1536)

    giant = np.abs(A).max(0) > 100
    giant_idx = giant.nonzero()[0].tolist()
    print(f"giant dims (>100 abs): {len(giant_idx)} at {giant_idx[:8]}{'…' if len(giant_idx) > 8 else ''}")

    mean = A.mean(0).astype(np.float32)
    std = A.std(0).astype(np.float32)
    # safety: never divide by ~0 even if a dim looks clean in the sample
    std = np.maximum(std, 1e-3)

    out_path = REPO / "data" / "latents" / "norm_stats.json"
    json.dump(
        {
            "mean": mean.tolist(),
            "std": std.tolist(),
            "giant_dims": giant_idx,
            "n_files_sampled": k,
            "note": "apply as (z - mean) / std at EVERY consumer; keep giant dims in place, standardization neutralizes them",
        },
        open(out_path, "w"),
    )
    print(f"saved {out_path} from {k} chunks, {A.shape} frames")


if __name__ == "__main__":
    main()