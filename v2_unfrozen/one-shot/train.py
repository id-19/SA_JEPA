# SA-JEPA v2 one-shot — trainer: gates in order, board rows, checkpoints. Per DESIGN_v0 §5.
# Gates: (1) overfit-one-batch -> loss ~ 0   (2) seeded determinism -> identical losses
#        (3) collapse dial: alpha=0 -> eff-rank craters vs alpha>0 -> rank survives
#        (4) real run: full cache, 5k steps, bs 8 -> board + checkpoint
# Usage: uv run python v2_unfrozen/one-shot/train.py --gate 1|2|3|4 [--alpha A] [--steps N] [--device mps]
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import SAJePAV0, D_MODEL  # noqa: E402
from board import board_row  # noqa: E402


def get_window_batch(bs, T=500, seed=None, device="cpu"):
    """Sample (bs) windows of T mel frames from the certified cache. Random utt + offset.
    T=500 for real runs; gates 1/3 can use shorter T for speed (model is T-agnostic)."""
    rng = np.random.default_rng(seed)
    rows = _windows()
    idx = rng.integers(0, len(rows), size=bs)
    batch = torch.zeros(bs, T, 80)
    speakers = []
    for bi, i in enumerate(idx):
        r = rows[i]
        mel = torch.load(HERE / "mel_cache" / f"{r['utt']}.pt").float()  # (nf, 80)
        nf = mel.shape[0]
        t0 = rng.integers(0, max(1, nf - T)) if T < nf else 0
        batch[bi] = mel[t0:t0 + T]
        speakers.append(r["utt"].split("-")[0])
    return batch.to(device), speakers


_WIN = None


def _windows():
    global _WIN
    import csv
    if _WIN is None:
        _WIN = list(csv.DictReader(open(HERE / "mel_cache" / "windows.csv")))
    return _WIN


def run(gate, alpha, steps, bs, device, seed=0, T=500):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SAJePAV0().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)  # explicit wd=0
    floors = {1: (293.9, 81.7), 10: (292.4, 283.6), 50: (260.3, 274.2), 150: (280.4, 292.9)}

    if gate == 1:
        # overfit ONE batch to ~0
        print(f"GATE 1: bs 4, T {T}, device {device}, alpha {alpha}", flush=True)
        x, spk1 = get_window_batch(bs=4, T=T, seed=0, device=device)
        for it in range(300):
            loss, parts, e = model.loss(x, alpha=alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if it % 25 == 0 or it == 299:
                print(f"it {it:3d} | loss {loss.item():.5f} | " +
                      " ".join(f"d{k}:{v.item():.4f}" for k, v in parts.items() if k.startswith("d")), flush=True)
        final = loss.item()
        print(f"GATE 1 {'PASS' if final < 0.05 else 'FAIL'}: final loss {final:.5f} (start ~9)")
        return

    if gate == 2:
        # determinism: same seed twice -> identical step-0 losses
        l1 = None
        for trial in range(2):
            torch.manual_seed(seed); np.random.seed(seed)
            m = SAJePAV0().to(device)
            x, _ = get_window_batch(bs=4, seed=0, device=device)
            loss, _, _ = m.loss(x, alpha=alpha)
            print(f"trial {trial}: step-0 loss {loss.item():.7f}")
            if l1 is None:
                l1 = loss.item()
        print(f"GATE 2 {'PASS' if abs(loss.item() - l1) < 1e-6 else 'FAIL'} (bit-identical init + data)")

    if gate == 3:
        # collapse dial: alpha=0 vs alpha=alpha_arg, few steps each, compare eff-rank trajectory.
        # MEASUREMENT OBJECT: per-frame z (B*T, D) — the exact thing sigreg regularizes.
        # (Time-pooling e.mean(1) measures a different object and hid the dial: both arms
        # looked collapsed while per-frame stats differed.)
        floors = None
        for a in [0.0, alpha if alpha > 0 else 0.1]:
            torch.manual_seed(seed)
            m = SAJePAV0().to(device)
            o = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.0)
            print(f"--- alpha={a}", flush=True)
            for it in range(steps):
                x, _ = get_window_batch(bs=8, T=T, device=device)
                loss, parts, e = m.loss(x, alpha=a)
                o.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                o.step()
                if it % max(1, steps // 4) == 0 or it == steps - 1:
                    with torch.no_grad():
                        zflat = e.reshape(-1, e.shape[-1])
                        sub = zflat[torch.randperm(zflat.shape[0])[:2048]]
                        print(f"  it {it} | {board_row(it, {k: v.item() for k, v in parts.items() if k.startswith('d')}, parts['sigreg'].item(), sub)}", flush=True)
            print()

    if gate == 4:
        # real run
        print(f"REAL RUN: {steps} steps, bs {bs}, T {T}, alpha {alpha}, device {device}", flush=True)
        t0 = time.time()
        log = []
        for it in range(steps):
            x, spk = get_window_batch(bs=bs, T=T, device=device)
            loss, parts, e = model.loss(x, alpha=alpha)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if it % 100 == 0 or it == steps - 1:
                pooled = e.mean(1)
                row = board_row(it, {k: v.item() for k, v in parts.items() if k.startswith("d")},
                                parts["sigreg"].item(), pooled, y=spk, floors=floors)
                print(row, flush=True)
                log.append(row)
        dt = time.time() - t0
        print(f"\n{steps} steps in {dt/60:.1f} min ({dt/steps:.2f} s/step)")
        ckpt = HERE / "runs" / f"v0_a{alpha}_{steps}.pt"
        ckpt.parent.mkdir(exist_ok=True)
        torch.save({"model": model.state_dict(), "alpha": alpha, "steps": steps, "log": log}, ckpt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, required=True, choices=[1, 2, 3, 4])
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=500 if False else 200)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--T", type=int, default=500, help="window length (gates 1/3 can use e.g. 120)")
    a = ap.parse_args()
    import torch  # noqa: F401
    run(a.gate, a.alpha, a.steps, a.bs, a.device, T=a.T)