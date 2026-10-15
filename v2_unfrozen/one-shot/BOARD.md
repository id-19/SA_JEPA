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
- Speed: CPU bs=4 T=120 **147 ms/step (was ~1300)**. MPS still launch-bound at small batch
  (496 ms — use CPU for toy gates, MPS only at bs≥32 where 3.5 s/step ≈ 13× CPU throughput).

## Run gates (from DESIGN_v0 §5)

1. overfit-one-batch → loss → ~0
2. seeded determinism → bit-identical
3. α=0 collapse demo (eff-rank craters) vs α=1.0 stable
4. first real run: 5k steps, bs 8, 10 s windows