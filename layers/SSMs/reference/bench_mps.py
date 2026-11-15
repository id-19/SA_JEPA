"""Speed and memory of the reference Gated DeltaNet layer, forward + backward.

    python layers/SSMs/reference/bench_mps.py                 # default sweep on MPS
    python layers/SSMs/reference/bench_mps.py --device cpu
    python layers/SSMs/reference/bench_mps.py --d-model 512 --heads 8 --seq 2048 --batch 4

Reports tokens/s for fwd+bwd, per chunk size and inverse method, plus the recurrent op
at short lengths for comparison. MPS memory is the driver-allocated peak seen after a step.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")   # fail loudly instead of silently using CPU
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from layers.SSMs.reference import GatedDeltaNetRef


def sync(device):
    if device == "mps":
        torch.mps.synchronize()


def bench(layer, x, device, iters, warmup=2):
    for _ in range(warmup):
        layer(x).pow(2).mean().backward()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        layer(x).pow(2).mean().backward()
    sync(device)
    dt = (time.perf_counter() - t0) / iters
    mem = torch.mps.driver_allocated_memory() / 2**20 if device == "mps" else float("nan")
    return dt, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, nargs="+", default=[512, 2048, 8192])
    ap.add_argument("--chunks", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--recurrent-max-seq", type=int, default=512)
    a = ap.parse_args()

    dev = a.device
    print(f"device={dev} torch={torch.__version__} d_model={a.d_model} heads={a.heads} batch={a.batch}")
    print(f"{'T':>6} {'mode':>10} {'chunk':>5} {'inverse':>8} {'ms/step':>9} {'tok/s':>10} {'MiB':>8}")
    for T in a.seq:
        x = torch.randn(a.batch, T, a.d_model, device=dev)
        runs = [("chunk", C, inv) for C in a.chunks for inv in ("doubling", "solve")]
        if T <= a.recurrent_max_seq:
            runs.append(("recurrent", 0, "-"))
        for mode, C, inv in runs:
            torch.manual_seed(0)
            layer = GatedDeltaNetRef(a.d_model, a.heads, chunk_size=max(C, 1), mode=mode,
                                     inverse=inv if inv != "-" else "doubling").to(dev)
            try:
                dt, mem = bench(layer, x, dev, a.iters)
                print(f"{T:>6} {mode:>10} {C:>5} {inv:>8} {dt*1e3:>9.1f} {a.batch*T/dt:>10.0f} {mem:>8.0f}")
            except RuntimeError as e:
                print(f"{T:>6} {mode:>10} {C:>5} {inv:>8}  failed: {str(e).splitlines()[0][:80]}")
            del layer
            if dev == "mps":
                torch.mps.empty_cache()


if __name__ == "__main__":
    main()
