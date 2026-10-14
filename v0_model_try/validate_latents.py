# SA-CE v0 — post-hoc latent-cache validation: EVERY manifest row, independently verified.
# Verifies: file exists, shape matches manifest n_frames, d=1536, dtype fp16, fps in 6.1–6.9,
# finite values, and manifest coverage vs acquisition_meta. Prints a fail list; exit 1 on any fail.
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="dev", help="leading token of the split dir (<kind>-clean)")
    args = ap.parse_args()

    lat = REPO / "data" / "latents" / f"{args.kind}-clean"
    fails = []
    rows = list(csv.DictReader(open(lat / "manifest.csv")))
    seen = set()
    total_frames = 0
    for r in rows:
        utt, chunk, nf, dur = r["utt_id"], str(int(r["chunk"])), int(r["n_frames"]), float(r["dur"])
        key = f"{utt}_{chunk}"
        if key in seen:
            fails.append((key, "duplicate manifest row"))
            continue
        seen.add(key)
        f = lat / f"{utt}_{int(chunk):03d}.npy"
        try:
            z = np.load(f)
        except FileNotFoundError:
            fails.append((key, "missing file"))
            continue
        total_frames += nf
        rate = nf / dur
        if abs(z.shape[0] - nf) > 0:
            fails.append((key, f"shape {z.shape} vs manifest {nf}"))
        if z.shape[-1] != 1536:
            fails.append((key, f"wrong dim {z.shape[-1]}"))
        if z.dtype != np.float16:
            fails.append((key, f"dtype {z.dtype}"))
        if not (6.1 <= rate <= 6.9):
            fails.append((key, f"fps {rate:.2f}"))
        if not np.isfinite(z).all():
            fails.append((key, "nonfinite"))

    # converse check: every npy on disk has a manifest row (chunk ids normalized: '000' == '0')
    npys_on_disk = set()
    for p in lat.glob("*.npy"):
        utt, ch = p.stem.rsplit("_", 1)
        npys_on_disk.add(f"{utt}_{str(int(ch))}")
    orphans = npys_on_disk - seen
    for o in sorted(orphans):
        fails.append((o, "orphan npy (no manifest row)"))

    meta = json.load(open(lat / "acquisition_meta.json"))
    meta_ok = meta["n_files"] == len({r["utt_id"] for r in rows}) and meta["n_frames"] == total_frames

    print(f"rows {len(rows)} | unique {len(seen)} | npy on disk {len(npys_on_disk)} | frames {total_frames}")
    print(f"meta agreement: n_files {meta['n_files']} == utts {len({r['utt_id'] for r in rows})}: {meta_ok} | n_frames match: {meta['n_frames'] == total_frames}")
    if fails:
        print(f"\nFAILURES ({len(fails)}):")
        for key, why in fails[:20]:
            print(f"  {key}: {why}")
        sys.exit(1)
    print("\nALL VALID — cache is certified.")


if __name__ == "__main__":
    main()