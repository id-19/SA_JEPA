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

## Run gates (from DESIGN_v0 §5)

1. overfit-one-batch → loss → ~0
2. seeded determinism → bit-identical
3. α=0 collapse demo (eff-rank craters) vs α=1.0 stable
4. first real run: 5k steps, bs 8, 10 s windows