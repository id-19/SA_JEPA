# SA-JEPA v2/v0 — delta-net + attention + JEPA-SIGREG (design)

*Written 15 Oct 2026 (session still live). Per the coaching contract this is a SPEC, not code: every line of implementation is Ishaan's. Repo of record: `v2_unfrozen/`. Thesis anchors that ARE his committed docs: `manifesto.md` (the bet), `goal.md` (tiers).*

## 0. One-sentence thesis

A ~12M-param causal hybrid encoder (delta-rule memory + grouped attention) over mel frames, trained **end-to-end with JE prediction + SIGREG** — no EMA, no stop-grad, no frozen teacher — on a laptop, evaluated by loss profile + linear probes + non-collapse metrics.

## 1. What v0 is NOT

- **No EMA target encoder anywhere.** Objective is UniJEPA-style: encoder-predictor trained end-to-end, anti-collapse by Gaussian regularizer alone (`thesis` line: "SIGREG, not just EMA" — and here: *not EMA at all*).
- **No MoE.** Manifesto's own rule: parked below 50M active.
- **No vocoder, no decoder, no text.** Pure representation rung. Audio generation is Phase 2.
- **No USM.** The frozen tower is demoted to a *reference probe target set* (§7), never a teacher.
- **No multi-day infra.** 10h cache, one trainer file, one board file.

## 2. Data (mel, floors before models)

- Source: LibriSpeech **train-clean-100 subset (~10 h first)**, 16 kHz → 80-mel log-mel, **50 fps** (25 ms win / 10 ms hop). fp16 on disk.
- Cache size (verified by arithmetic): **275 MB** — fits current free disk (1.8 GB). 100h = 1.3 GB when disk is cleared.
- Windows: 10 s = **T = 500** frames, random offset per utterance per epoch. 3,600 windows; **bs 8 → 450 steps/epoch** (5k steps ≈ 11 epochs; one epoch/night is the manifesto bar — this beats it).
- **Floors computed BEFORE any model, on this exact cache** (board discipline; new substrate = new board, never compared to USM-latent boards): blind (predict-mean) and closed-form linear on mel-frame prediction, printed as the board bar.

## 3. Architecture (all counts computed, not estimated)

```
mel (80) ──causal Conv1d k=4──> D=256 ──> [ 6× DeltaNet block  |  2× GQA block ] ──> ẽ (B,500,256)
                                              │
                                predictor: 2× DeltaNet (no MLP) + head ──> ẑ_{t+d} for d ∈ grid
```

| component | params (printed by script) |
|---|---|
| patch embed (causal conv 80→256, k=4) | 82,176 |
| DeltaNet block ×6 (causal-conv qkv k=4, per-head β + v-gate, 64×64 per-head state, MLP×4) | 1,379,072 each |
| GQA block ×2 (4 q heads / 2 kv heads, RoPE, out-gate, MLP×4) | 723,456 each |
| **encoder total** | **9,803,520** |
| predictor (2 delta layers + input proj + 4 horizon heads) | 1,773,056 |
| **MODEL TOTAL** | **11,576,576** |

- GQA vs MLA at this scale: **723,456 vs 723,680 params — a tie.** MLA's real advantage (KV-cache compression) is a deploy-time concern; v0 picks **GQA for simplicity**, MLA revisited only if/when scale grows.
- Training-resident memory: fp32 weights 44 MB; +grads+AdamW = **177 MB**. Attn scores (8,4,500,500) ≈ 61 MB, delta states ≈ 4 MB. Peak « 4 GB working ceiling. Laptop stays usable.
- Causality: every block causal (delta-net natively; attention via RoPE + causal mask) → streaming at inference for free.

## 4. Objective (the SIGREG setup)

For each window, targets are the **live encoder's** own future latents:

- `ẑ_{t+d} = Pred_d(ẽ_{≤t})` for d ∈ {1, 10, 50, 150} frames = 20 ms / 0.2 s / 1 s / 3 s @ 50 fps.
- `L_pred = Σ_d w_d · MSE(ẑ_{t+d}, ẽ_{t+d})`  — gradients flow into encoder AND predictor **and through the targets** (end-to-end, no stop-grad).
- `L_SIGREG = E_{z,u}[ χ²₁((uᵀz)²) ]` — pull the batch's latent distribution toward N(0,I) along random unit directions u (512 random projections per step, UniJEPA Eq. 3).
- `L = L_pred + α · L_SIGREG`, **single scalar α** — the one loss hyperparameter.
- Weight init note for the bands: UniJEPA's theorem needs a bounded-Lipschitz predictor; the 2-layer delta predictor with a 4-head head satisfies it structurally (no unbounded growth op). Pre-register what happens to L_pred if α=0 on the same run — that number is the collapse demonstration.

## 5. Training loop (simple on purpose)

- AdamW, constant LR (start 3e-4), **weight_decay=0.0 explicitly** (the stored-log trap from the coaching skill), bf16 autocast on MPS, grad-clip 1.0, seed pinned.
- Gates in order, each with a printed bar: (1) overfit-one-batch → loss →~0, (2) seeded determinism → bit-identical, (3) α-sweep sanity: α=0 run shows rank collapse, α=1.0 run doesn't, (4) first real run.
- First real run: 10 h cache, 5k steps, bs 8. Output: board row + checkpoint.

## 6. Board v0 (what a run prints)

`step | L_pred per-d | SIGREG | eff-rank(Ẽ) | per-dim var | speaker-probe acc | floors bar`

- **Non-collapse set:** effective rank of batch covariance, per-dim variance vs N(0,1).
- **Probe set:** pooled ẽ → speaker ID (linear, labels from file IDs); the **Δ-profile** of L_pred is the memory story: in USM space memoryless signal died by Δ≈4 (0.64 s). Pre-register whether the learned space shows the same death or the delta-net keeps improving past 0.2 s.
- **Reference probes:** the same speaker probe + memory probe run on cached USM latents = the number the learned space must eventually beat. One-off, CPU, minutes, reuses `data/latents/dev-clean` — nothing new to compute except the probe itself.

## 7. One-line-diff ablation (pre-registered, run after first real run)

Arm B: identical model/seed/steps, delta blocks → plain gated-attention blocks (the hybrid loses its memory mechanism, keeps parameter count). Band: mechanism reasoning says arm B collapses memory-driven gains at d ≥ 10. Fill your bands in §8 **before** either arm runs.

## 8. Falsification table — bands BLANK, Ishaan fills before the first run

| # | claim | falsified if | band |
|---|---|---|---|
| A | SIGREG prevents collapse at α=1.0 | eff-rank < 0.3·256 at 5k steps | ____ |
| B | α=0 collapses (the theorem demonstrated) | α=0 run keeps rank > 0.6·256 | ____ |
| C | delta-net memory earns d ≥ 10 (vs arm B) | hybrid ≤ arm B at d=50/150 | ____ |
| D | speaker probe beats USM-latent reference eventually | never exceeds it | ____ |
| E | L_pred at d=1 lands where? (vs floors bar) | — | ____ |

## 9. Forks (decisions are yours; each changes the design)

1. **GQA (recommended: tie in params, simpler) vs MLA** — cost of MLA ≈ nothing in params; its payoff is deploy-time KV cache, not v0 training.
2. **Horizon grid {1,10,50,150} (recommended — spans 20 ms→3 s on 50 fps mel) vs 4-horizon grid inherited from the dead doc** — the old grid assumed 6.27 fps; this substrate is 50 fps, the grids are not comparable.
3. **Δ-state per head (4×64×64, recommended — matches Qwen3-Next style, 4 MB states) vs one shared 256×256 matrix** — the shared version triples state memory and is the untested design.
4. **First overnight = 10 h / 5k steps (recommended)** vs waiting for the full 100 h cache (needs disk cleanup first — ~12 GB reclaimable, your word).
5. **α (SIGREG weight)** — decide with band B: pre-register what α=0 does, then pick α accordingly.

## 10. Who writes what

- This doc + boards + floors + probes-harness specs: Hermes. Probes RUN by Ishaan.
- `v2_unfrozen/src/`: encoder, predictor, blocks, trainer, board printer — **all Ishaan's code.** Files land blank-first with pass-bars, per the coaching contract.