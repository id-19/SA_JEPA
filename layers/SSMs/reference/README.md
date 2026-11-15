# Reference Gated DeltaNet (MPS / CPU)

AI-written, 13 Nov 2026. This folder is **not** the learning implementation. The user's own
GDN lives in `layers/SSMs/multi_headed/`. This one exists for two jobs:

1. **Correctness oracle.** Same weights, same input → the user's layer should match this
   one with `torch.testing.assert_close`. That is a pass gate that doesn't hand over the code.
2. **MPS speed baseline.** fla's fast kernels are Triton (CUDA only), so on a Mac the
   question is how far pure PyTorch gets before a custom Metal kernel is worth it.

## Files

- `gated_delta_rule.py`: the ops. `recurrent_gated_delta_rule` (token by token, the ground
  truth) and `chunk_gated_delta_rule` (chunks in parallel, a short loop over chunks). The
  docstring derives the chunked form from the recurrence.
- `gated_deltanet.py`: `GatedDeltaNetRef`, the full token mixer from the paper's figure:
  Linear → causal depthwise conv → SiLU on q/k/v, L2 per head on q/k, Mamba-2-style α,
  sigmoid β, per-head RMSNorm × SiLU output gate, output Linear. No residual inside.
- `bench_mps.py`: fwd+bwd speed and memory sweep.
- Tests: `eval_test/tests/test_gdn_reference.py` (27 tests: chunk vs recurrent at several
  lengths and chunk sizes, state hand-over, exact recall of a stored pair, causality,
  gradients chunk vs recurrent, MPS vs CPU).

```bash
uv run pytest eval_test/tests/test_gdn_reference.py
```

```bash
uv run python layers/SSMs/reference/bench_mps.py
```

## Results on this Mac (13 Nov 2026)

torch 2.13, MPS, d_model 512, 4 heads (head dim 128), batch 4, fwd+bwd, fp32:

| T | chunk 64, doubling | chunk 64, solve | recurrent |
|---|---|---|---|
| 512 | 25.7k tok/s | 20.8k | 1.5k |
| 2048 | 24.7k | 19.2k | – |
| 8192 | 24.7k | 19.2k | – |

- Throughput is flat in T, as it should be for a linear-time layer.
- `doubling` (inverse of the unit lower-triangular matrix by log₂C rounds of matmuls) beats
  `solve_triangular` on MPS. Both run natively; the benchmark sets
  `PYTORCH_ENABLE_MPS_FALLBACK=0` so a silent CPU fallback would show up as an error.
- Chunk 32–64 is the sweet spot here. 128 is slower and uses more memory.
- Chunked is ~17× the token-by-token loop at T=512 and the gap grows with T.

One bug found while benchmarking: indexing `x[:, :, i]` inside the chunk loop made the
backward pass O(n²) in the number of chunks (each index op's backward allocates a
full-size zero gradient). `unbind` fixes it; 8192 tokens went from 2983 ms to 1329 ms
per step. Worth knowing for your own chunk loop.

## Not done

- bf16 / fp16. Everything is fp32.
- A Metal (MLX or custom) kernel. Decide from the numbers above whether it's needed.
