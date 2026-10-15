# BOARD — v2 one-shot (mel substrate)

*Different substrate = different board. Never compare against USM-latent boards.*

## Floors (blind / closed-form linear, dB² units; mel frame → mel frame)

| d | 0.20s/1s? | blind | linear | read |
|---|---|---|---|---|
| 1 | 20 ms | 293.8535 | **81.7146** | adjacent mel frames highly predictible |
| 10 | 0.2 s | 292.3764 | 283.6227 | linear ≈ blind already |
| 50 | 1 s | 260.2917 | 274.2166 | **linear WORSE than blind** (fits noise) |
| 150 | 3 s | **280.3918** | 292.9808 | linear anti-helps; mean is the bar |

**Board read:** on raw mel, memoryless-linearizable signal dies by ~0.2 s and lstsq actively overfits past 1 s. Only a stateful model can get under the blind bar at d ≥ 50 — that is the entire v2 bet restated on this substrate. (Blind's non-monotonicity across d reflects marginal-mel variance by position, not signal.)

## Reference (frozen-USM space, dev-clean — the learned space's target)

- Speaker probe (linear, pooled chunk latents, chunk-level): **0.9996 ± 0.0006** (3 seeds).
- v0's real comparison is the *failure set*: which probes v2 fails that USM passes.

## v2 model rows — BLANK until the first trainer run

`step | L_pred d1 d10 d50 d150 | sigreg | eff-rank | var[min,mean,max] | spk | floors bar`

### Gate 1 (overfit-one-batch, chunkwise core, first entry)

- 2026-10-15: PASS, final loss **0.02026** (bar 0.05), 120 iters / 19 s CPU (147 ms/step at bs=4 T=120).
- α=0.1 failed this gate (loss stuck 0.124, per-d losses re-rising in sync with sigreg pressure):
  sigreg variance-pressure fights memorization on padded windows. α=0.01 + warmup → PASS.
  Keep α=0.01 for gate 2/4; revisit at scale for gate 3 (collapse demo needs real α).

### Gate rows (chunkwise core, real mel windows)

- **Gate 2 PASS** — bit-identical step-0 losses on reseed.
- **Gate 1 PASS** at α=0.01 (0.0201 < 0.05 bar, real mel batch, CPU 147 ms/step at bs=4/T=120).
  α=0.1 FAIL confirmed again on real windows (0.143, sigreg fights memorization).
- **Gate 3 PASS** after re-measure fix: dial now reads per-frame z (time-pooled means hid it).
  α=0: var craters to 0.00. α=1.0: var mean ~0.9 held, eff-rank ~13.
- **Gate 4 (T=250, bs=8, α=0.01): FAIL — full collapse.** Per-frame var → 0, trivial loss 1e-4,
  spk probe 0.000. α=0.01 has no anti-collapse power at scale; fine only for gate-1 memorization.
  *[15 Oct: the "collapsed" verdict stands on var→0 + trivial loss (both real computations), but
  `spk 0.000` was **never a measurement** — that probe returned a fabricated 0.0 from an
  `except ValueError` and could not run at all. See `POST_MORTEM.md` §2.3.]*
- **Gate 4 (α=1.0, base sigreg): slow spectrum collapse.** Estimator satisfied (sigreg ~0.21,
  per-dim var mean ~0.8) while eff-rank 10.9 → 6.0 (step 300) → 4-5 (step 1200), var[min] → 0,
  spk 0.000. Two-moment mean-variance per random direction cannot see fat-dim concentration.
- **Isotropy fix (iso=1.0, α=1.0, 600 steps): directionally right.** eff-rank floors ~7.5
  (baseline: keeps decaying), var[min] held 0.12-0.18, window-identity probe 0.987.
  Pre-registered ≥15 NOT met — magnitude insufficient. Next: iso=4.0 arm, kurtosis term.

## Chunkwise scan (in place of per-step loop)

- Core: canonical decay-free DeltaNet `S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ`, chunkwise
  UT-transform (Yang et al. 2406.06484 Eqs. 5–11). Loop trips: T → 64 + T/64, carries batched
  over B·H. Verified vs per-step ref: outputs AND states < 1e-5 (read-after-update, diag-incl).
- **Closed-form chunk inverse does NOT exist for decayed (a<1) delta rule** — verified three
  candidate M-forms, all fail; error-correcting writes make transitions state-dependent
  (unlike linear attention). Gated DeltaNet's chunk form = separate follow-up with real math.
- Speed: CPU bs=4 T=120 **147 ms/step (was ~1300)**.
- **CORRECTED 15 Oct (real sweep, bs 8/16/32, T=250 — the earlier "MPS at bs≥32" note was wrong):**
  MPS throughput is *flat* at **1.1 kframe/s for bs 8 and 16** (launch-bound: constant per-frame
  cost as batch grows) and collapses to 0.3 kframe/s at bs=32 (24.3 s/step, swap → 8.5 GB, the
  8 GB machine thrashes). CPU bs=8 fp32 = **609 ms/step, 3.3 kframe/s, 1.13 GB peak** — the best
  configuration on this machine and the one v0 trains with. CPU bs=16/32 are 3806/8839 ms/step
  (memory-pressure cliff). bf16 autocast on CPU is 19× *slower* (12.1 s/step) — never use it there.

## Run gates (from DESIGN_v0 §5)

1. overfit-one-batch → loss → ~0
2. seeded determinism → bit-identical
3. α=0 collapse demo (eff-rank craters) vs α=1.0 stable
4. first real run: 5k steps, bs 8, 10 s windows

## Parallel scan (log-depth) — 15 Oct

- Chunk carry is **affine in the state**: `S_out = S_in·M_c + b_c`, `M_c = I − W_cᵀK_c`,
  `b_c = U_cᵀK_c`, computable for every chunk independently. Affine maps compose associatively
  (`(S·M₁+b₁)·M₂+b₂ = S·(M₁M₂) + (b₁M₂+b₂)`), so chunk incoming states come from a
  **Hillis–Steele prefix scan**: sequential depth `N_chunks → log₂(N_chunks)`.
  T=500: 8 → 3 sequential steps; T=1024: 16 → 4.
- Verified three ways: parallel vs sequential loop ≤ **7.2e-07**, and both vs the per-step
  definition ~6e-07 (fp32 rounding). Smoke 6/6. `_chunk_scan_parallel` in `model.py`,
  dispatch flag `DeltaCore.parallel_scan`; sequential loop kept as the reference.
- **Profiled — and the prefix scan is NOT faster than the CHUNKWISE loop it replaced (15 Oct,
  `_profile_scan_depth.py`, bs=4, fwd+bwd):** the scan issues **more ops** (+12% at T=100, +18% at
  T=250) and buys ≤7% wall-clock on MPS, ~0% on CPU, and **−10% on CPU at T=500**. Per-frame cost
  is flat in T for BOTH variants (cpu 0.36/0.29/0.30 ms per frame at T=100/250/500) and the
  dominant ops are `as_strided`/`select`/`resolve_conj` views, so the binding cost is per-frame
  indexing work. **Baseline matters**: this compares prefix-scan vs *chunkwise*; the earlier
  per-step → chunkwise change was the real win (≈9× CPU at bs=4/T=120: 1300 → 147 ms/step; the
  board's bare-cell numbers were 8.4–22.3× CPU and 160× MPS fwd). Depth 8 → 3 is real and stays;
  the payoff needs `torch.compile`/fusion, graph capture, or a bare cell at large T — untested.
  Label: **more parallel, not yet faster**. See `POST_MORTEM.md` §1.1.
- **Prior art existed in this repo the whole time**: `v0_model_try/parallel_scan.py`
  (untracked, mtime 13 Oct 09:00 — two days before the sequential v2 scan) already stated the
  affine composition rule and the doubling scan plan, with the note that it removes the loop
  and is much faster. It was never consulted; now tracked. A third source predates it: the
  coaching skill's own `references/parallel-scan.md` + `scripts/parallel_scan_ref.py`
  (11 Oct) carry the identical merge rule and a runnable verified implementation. See
  `POST_MORTEM.md`.

## v0 run — first real training, 2000 steps (15 Oct, the v0 deliverable)

`cpu fp32, bs=8, T=250, 6 Δ-blocks + 2 GQA, d=256, α=1.0 + isotropy, 28.3 min`

| step | held-out recon mel-MSE | eff-rank | var[min,mean,max] |
|---|---|---|---|
| 0 | 764 dB² | 8.6 | — |
| 400 | 43.7 | 11.7 | [0.31, 0.81, 2.37] |
| 1200 | 24.1 | 18.6 | [0.32, 0.81, 4.98] |
| 2000 | **17.1** | **20.8** | [0.31, 0.72, 11.28] |

**No collapse**: eff-rank *grows* 8.6 → 20.8 and var[min] stays ~0.31 across 2000 steps — the
isotropy fix holds at scale where the plain α=1.0 arm decayed 10.9 → 4-5. Reconstruction improves
monotonically (764 → 17.1 dB²); var[max] 11.3 shows a few fat dims the two-moment estimator still
doesn't see (the kurtosis term remains open).

## Evidence suite (`evidence_v0.py`, ckpt `runs/v0`) — 15 Oct

| probe | result | reference / floor |
|---|---|---|
| Speaker, frozen latents, linear | **0.991 ± 0.003** | blind 0.040 (37 classes, n=599); USM ref 0.9996 |
| Speech-vs-music, frozen latents | **1.000 ± 0.000** | 0.500 floor (n=96, *synthetic* music — easy) |
| Decoder autoencode | mel-MSE **19.5 dB², corr 0.964** | — |
| Decoder trained on *frozen* latents (ablation) | **15.9 dB², corr 0.971** | beats the joint decoder → the repr is genuinely decodable, not decoder co-adaptation |
| 1 s-ahead: latent-MSE / mel-MSE | 0.667 / **224.7 dB², corr 0.440** | **FAILS** — flat, structureless spectrogram (see `mel_reconstruction.png`) |
| Geometry | eff-rank 21.9 frames / 6.7 utt-pooled | var[0.29, 0.74, 12.22] |

**Honest read: classification and reconstruction work; prediction does not.** The 1 s-ahead panel
is a mean-ish band with no harmonics or formants — 224.7 dB² is ~13× worse than the 17.1 dB²
autoencode and well above the 260 dB² blind bar's *frame-level* analogue. v0 (2000 steps, 8 GB,
30 min) shows the encoder/decoder stack is sound and the predictor needs real work.

Artifacts: `runs/evidence/{latent_pca,state_trajectories,mel_reconstruction}.png`,
`runs/evidence/*.wav`, `runs/evidence/evidence.json`, `runs/evidence/deploy.pt`.