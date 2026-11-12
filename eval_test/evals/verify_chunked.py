"""One-shot verification + benchmark for ParallelKDA (chunked vs token-by-token).

Correctness: for several (B, T) shapes and seeds, compare the chunked forward against
legacy_out (the sequential reference) across a sweep of chunk sizes. Fails on any NaN/Inf
in either reference or chunked output, or on an allclose mismatch.

Benchmark: time legacy_out vs forward on a fixed set of shapes/chunk sizes.

    python3 eval_test/evals/verify_chunked.py
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
torch_threads_set = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "DeltaNet"))

import torch

torch.set_num_threads(1)

from parallel_kda import ParallelKDA


SEEDS = (0, 1, 2)
VERIFY_SHAPES = (
    (1, 32),
    (1, 64),
    (2, 31),
    (3, 33),
    (8, 128),
    (4, 1024),
    (32, 1024 * 8),
    (61, 1023 * 17),
)
# CHUNK_SIZES = (1, 2, 4, 8, 16, 32, 64, 128)
# CHUNK_SIZES = (1, 2, 4, 8, 16, 32, 64)
# CHUNK_SIZES = (1, 2, 4, 8, 16, 32, 48)
CHUNK_SIZES = (1, 2, 4, 8, 16, 32)
BENCH_SHAPES = (
    (1, 1024, 128),
    (1, 1024*2, 128),
    (4, 1024*4, 128),
)
BENCH_CHUNKS = (8, 16, 24, 32, 48)
D_INPUT = 80
D_QK = 128
D_V = 128
D_OUTPUT = 80


def _check_finite(name, t):
    if torch.isnan(t).any():
        raise AssertionError(f"{name} contains NaN")
    if torch.isinf(t).any():
        raise AssertionError(f"{name} contains Inf")


def verify():
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = ParallelKDA(D_input=D_INPUT, D_output=D_OUTPUT, D_qk=D_QK, D_v=D_V).eval()
        for (B, T) in VERIFY_SHAPES:
            torch.manual_seed(seed)
            x = torch.randn(B, T, D_INPUT)
            with torch.no_grad():
                ref = model.legacy_out(x)
            _check_finite("reference", ref)
            for c in CHUNK_SIZES:
                if c > T:
                    continue
                with torch.no_grad():
                    got = model(x, chunk_size=c)
                _check_finite(f"chunked (c={c})", got)
                if not torch.allclose(ref, got, atol=1e-4, rtol=1e-3):
                    maxerr = (ref - got).abs().max().item()
                    raise AssertionError(
                        f"seed={seed} B={B} T={T} c={c}: maxerr={maxerr:.3g}"
                    )
        print(f"OK seed={seed}: all shapes/chunk sizes match, outputs finite")
    print("VERIFY PASS")


def bench():
    print("\nBENCH")
    print(f"{'B':>4} {'T':>6} {'chunk':>6} {'legacy(s)':>11} {'chunked(s)':>11} {'speedup':>9} {'maxerr':>10} {'finite':>7}")
    print("-" * 72)
    for (B, T, D) in BENCH_SHAPES:
        torch.manual_seed(0)
        model = ParallelKDA(D_input=D, D_output=D, D_qk=D, D_v=D).eval()
        x = torch.randn(B, T, D)
        with torch.no_grad():
            ref = model.legacy_out(x)
        t0 = time.perf_counter()
        with torch.no_grad():
            model.legacy_out(x)
        # more reps for stable timing
        reps = 3
        for _ in range(reps - 1):
            with torch.no_grad():
                model.legacy_out(x)
        legacy_time = (time.perf_counter() - t0) / reps

        for c in BENCH_CHUNKS:
            if c > T:
                continue
            t0 = time.perf_counter()
            for _ in range(reps):
                with torch.no_grad():
                    got = model(x, chunk_size=c)
            chunk_time = (time.perf_counter() - t0) / reps
            finite = (not torch.isnan(got).any().item()) and (not torch.isinf(got).any().item())
            maxerr = (ref - got).abs().max().item() if finite else float("nan")
            speedup = legacy_time / chunk_time if chunk_time > 0 else float("inf")
            print(f"{B:>4} {T:>6} {c:>6} {legacy_time:>11.4f} {chunk_time:>11.4f} {speedup:>8.2f}x {maxerr:>10.3g} {'Y' if finite else 'N':>7}")


if __name__ == "__main__":
    verify()
    bench()
