"""Three VanillaGQA forwards on audio-eval frames. CPU, then MPS.

One table at the end. Times are ms per forward.

    uv run eval_test/evals/bench_gqa_audio.py
"""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import torch

import task_audio
from Attention_layers.gqa import VanillaGQA

# (B, T, C). T has to fit the train split of the cached clip (419 frames).
SHAPES = (
    (1, 64, 32),
    (1, 256, 80),
    (1, 419, 128),
    (4, 128, 80),
    (16, 64, 32),
    (16, 256, 80),
    (16, 1024, 256),
    (16, 1024, 512),
    (2, 1024*8, 512),
    (64, 1024*16, 2048),
)
WARMUP = 2
REPS = 5

data = task_audio.make_data()
frames = data["train"][:, :-1].contiguous()
width = frames.shape[-1]
rows = []

for B, T, C in SHAPES:
    feat = frames[:, :T]
    if C < width:
        feat = feat[..., :C]
    elif C > width:
        reps = (C + width - 1) // width
        feat = feat.repeat(1, 1, reps)[..., :C]
    x = feat.expand(B, -1, -1).contiguous()
    torch.manual_seed(0)
    model = VanillaGQA(
        d_input=C, d_output=C, hdim_qk=128, hdim_v=128, n_heads=16, grouping=4,
    ).eval()
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x, use_flash=True)
        t0 = time.perf_counter()
        for _ in range(REPS):
            model(x, use_flash=True)
        flash_ms = (time.perf_counter() - t0) / REPS * 1e3

        for _ in range(WARMUP):
            model(x, use_flash=False)
        t0 = time.perf_counter()
        for _ in range(REPS):
            model(x, use_flash=False)
        einsum_ms = (time.perf_counter() - t0) / REPS * 1e3

        for _ in range(WARMUP):
            model.efficient_simple_forward(x)
        t0 = time.perf_counter()
        for _ in range(REPS):
            model.efficient_simple_forward(x)
        eff_ms = (time.perf_counter() - t0) / REPS * 1e3
    rows.append(("cpu", B, T, C, flash_ms, einsum_ms, eff_ms))

for B, T, C in SHAPES:
    feat = frames[:, :T]
    if C < width:
        feat = feat[..., :C]
    elif C > width:
        reps = (C + width - 1) // width
        feat = feat.repeat(1, 1, reps)[..., :C]
    x = feat.expand(B, -1, -1).contiguous().to("mps")
    torch.manual_seed(0)
    model = VanillaGQA(
        d_input=C, d_output=C, hdim_qk=256, hdim_v=768, n_heads=8, grouping=4,
    ).eval().to("mps")
    with torch.inference_mode():
        for _ in range(WARMUP):
            model(x, use_flash=True)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            model(x, use_flash=True)
        torch.mps.synchronize()
        flash_ms = (time.perf_counter() - t0) / REPS * 1e3

        for _ in range(WARMUP):
            model(x, use_flash=False)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            model(x, use_flash=False)
        torch.mps.synchronize()
        einsum_ms = (time.perf_counter() - t0) / REPS * 1e3

        for _ in range(WARMUP):
            model.efficient_simple_forward(x)
        torch.mps.synchronize()
        t0 = time.perf_counter()
        for _ in range(REPS):
            model.efficient_simple_forward(x)
        torch.mps.synchronize()
        eff_ms = (time.perf_counter() - t0) / REPS * 1e3
    rows.append(("mps", B, T, C, flash_ms, einsum_ms, eff_ms))

header = f"{'dev':<4} {'B':>4} {'T':>5} {'C':>5} {'flash':>8} {'einsum':>8} {'eff':>8}"
print(header)
print("-" * len(header))
for dev, B, T, C, flash_ms, einsum_ms, eff_ms in rows:
    print(f"{dev:<4} {B:>4} {T:>5} {C:>5} {flash_ms:>8.2f} {einsum_ms:>8.2f} {eff_ms:>8.2f}")
