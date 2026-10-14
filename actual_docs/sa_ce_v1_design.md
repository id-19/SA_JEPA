# SA-CE v1 — the semantic predictor (design doc)

*Written 14 Oct 2026, session-directed. Companion to `thesis.md` (the why) and `latest_plan.md` (the board). This is the WHAT: architecture + training loop for a model that trains on a 6GB-RAM laptop while staying usable. Every line of code here is Ishaan's to write; this doc fixes contracts, shapes, numbers, and the order of gates.*

**Name:** SA-CE — Semantic-Continuum Encoder. One sentence: **a causal, stateful encoder over frozen USM frame latents that maintains a running semantic state of the audio stream and is trained to predict that state's future — on a 4D time grid (0.15s → 3.7s) — with a text head and an embedding-alignment loss bolted to the same states.**

---

## 0. What v1 is NOT (deliberate exclusions)

- **No trainable audio encoder.** The USM tower (300M) is frozen, run once, cached to disk. Collapse machinery (EMA, SIGREG, stop-grad design) is therefore *parked*, not skipped — it returns the day the encoder unfreezes (v2). Pre-registered fact: with a frozen encoder, representation collapse is impossible; only the predictor can waste capacity.
- **No MoE.** Thesis §5 already rules it out below ~50M active. FFNs are dense and small.
- **No vocoder / no audio decoder.** Speech-to-speech is NOT v1. The frozen latents already solve "understand audio"; generating audio needs a vocoder rung that belongs after the semantic core exists (Phase 2, thesis diagram).
- **No SSD paging, no recurrent-depth loop, no quantized training.** Params are small enough that fp32 AdamW fits in RAM with room to spare (audit below). These are v2/v3 rungs, entered one at a time with their own measurements.

## 1. Data pipeline (build first)

- Input: `data/LibriSpeech/train-clean-100` (~100h, 16 kHz wav) + dev-clean/test-clean for eval splits.
- Frozen encode: USM (Gemma-3n audio encoder, `transformers` first-class, or `n0mad-0/gemma3n-usm-rip`). Frame latents `(T_f, 1536)` at **6.27 fps verified** (188 frames / 30 s; the 6.5 figure that seeded this doc was approximate).
- Contract: chunk each file into 30 s windows (USM's native chunk), drop remainders < 2 s, write `data/latents/train-clean-100/{utt_id}_{chunk:03d}.npy` + a manifest CSV (`utt_id, chunk, n_frames, dur`). Validation: `n_frames ≈ dur × 6.5` within ±2 frames for every row; log drifts.
- Budget: ~100h → **~7 GB fp16 on disk** (disk, not RAM — check the drive has it). Encode runs once; the training loop never touches audio or the encoder again.
- **Floors BEFORE any model (board discipline — same rule as the sine board):** on the cached-latent dataset, compute:
  - `B_lat`: blind = per-dim variance of z (predict-mean MSE),
  - `M_lat`: best memoryless predictor (small MLP on z_t → z_{t+k}, or ridge),
  - `L_lat`: closed-form linear (lstsq z_t → z_{t+k}).
  Different dataset ⇒ different floors; they never mix with the sine board.

## 2. Architecture

```
z_t (1536) ──> in-proj ──> [ L = 10 hybrid layers ] ──> s_t (384) ──┬─> latent head ──> ẑ_{t+k}   (primary loss)
                            8 delta-net : 2 gated-attn              ├─> text head (causal LM over latents)
                            d_model=384, 4 heads, dk=dv=128         └─> pooled (mean over T) → alignment
                            FFN 384→1024→384, SiLU                       vs EmbGemma2 text emb (frozen)
```

| component | params (M) |
|---|---|
| in-proj 1536→384 | 0.59 |
| 10 hybrid layers | 15.3 |
| latent head 384→1536 | 0.59 |
| text head (2-layer transformer + 8k BPE vocab) | 4.3 |
| **total (trainable)** | **~21 M** |

- Delta-net layer (Gated DeltaNet form, the thesis engine): per head, state `S ∈ (128, 128)` fp32; update
  `S ← α·S·(I − β·k kᵀ) + β·v kᵀ`, with `k, q, v = d·x` projections, `α = exp(softplus(aₐ)) ∈ (0,1)` per channel, `β = σ(σ_b)`.
  Readout `o = S · q`, per-head concat → out-proj → SiLU gate. Contraction over d_model: `x @ W_kk` style — **write the expected shape beside every einsum** (T-loop lessons apply verbatim).
- Gated-attention layer: standard MHA (4 heads, dk=128) + short causal conv patch (time order) — the recall channel.
- Context length: `T = 248` frames ≈ 39.6 s per training window (scaffold default T=195 ≈ 31 s until full-corpus cache lands). Streaming is native (delta-net state), exact-lookup comes from the 2 attention layers over the window.
- Param rule enforced by the table: every module that exists must change a board number or it gets deleted before it lands.

## 3. The objective — prediction on a 4D time grid

From each position `t`, the latent head predicts the future frozen latent at **four horizons**:

| Δ (frames) | Δ (s) @6.27fps | weight | what it tests |
|---|---|---|---|
| 1 | 0.16 | 0.3 | is there even signal to predict (sanity) |
| 4 | 0.64 | 0.5 | local phonetic transition |
| 18 | 2.87 | 1.5 | word/phrase trajectory — the "semantic" band |
| 24 | 3.83 | 1.0 | mid-phrase horizon (memory horizon ceiling) |

Loss: `L = Σ Δ w_Δ · MSE(ẑ_{t+Δ}, z_{t+Δ})` on valid (in-window) targets. This **subsumes the masked latent + time-shift targets of thesis §3** (unmasked causal variant: nothing is masked, everything is future — the streaming world-model move). The future-target z is a **cached, non-trainable tensor**: no EMA needed, stop-grad is automatic.

- **ΔS (the semantic probe):** ΔS(Δ) = (M_lat − SA-CE) / M_lat. Report `ΔS₂₄` as the headline.
- **Secondary reads** (every 1000 steps): per-horizon ΔS curve; **text WER** (greedy decode, LibriSpeech refs) — the *is semantics actually inside* meter; **alignment CCA**: 20k 10 s windows, latent-pool vs text-prompt EmbGemma2 embeddings, cosine-CCA@16 before training (pre-init baseline!) and at each eval — answers "does the 768-d semantic space inherit into frame states".

## 4. Training loop (the 6GB contract)

- **Batch = 32 windows × 248 frames.** Precomputed latents ⇒ activations are the only scaling cost: B=32/T=248 ≈ 1.2–1.5 GB activations + 0.33 GB optimizer residency + ~1.5–2 GB macOS headroom ≈ **~4 GB peak**. T=496 fits at B=16 (~5 GB) for the rare long-context run. Nothing else is resident.
- Overnight unit: 500 steps × 32 × 38 s ≈ **4.7 h audio/night**. Full schedule: **~3000 steps (≈ 9 passes over 100 h)**; eval every 250 (4D board + ΔS), full secondary board + checkpoint every 1000.
- Optimizer AdamW lr 3e-4, cosine to 3e-5, warmup 150, weight-decay **0.0 on all log/softplus-parameterized gates** (log-space trap, on record), betas (0.9, 0.95), grad-clip 1.0. fp32 weights+grads+Adam on 21M ≈ 0.33 GB.
- Gates (unchanged house rules): **overfit-one-batch → loss → ~0**; **seeded run bit-identical**; determinism, shape, grad-flow+**Δ-spread** (α, β must vary across channels AND steps — the degenerate-constant-Δ mode), causality (perturb z[t+5:], outputs z[:t] unchanged), BIBO on the state under zero-input (α < 1 ⇒ ‖S‖ shrinks — but verify, don't assume), known-answer T=2 by hand.
- **Board print beside every loss**, one row: `step | 4D losses | ΔS₂₄ | floors B/M/L`.
- **Streaming checkpoints** (the thesis §4.3 row, start measuring day one): save/restore (state-only) vs (state + 24-frame warmup window); metric = 4D-board delta between the two; prediction: the warmup gap is nonzero but small — write the band down before measuring.
- 6GB discipline: USM encode (1.2 GB weights, no_grad) runs in a **separate session** from training; close it first. Nothing else on the machine during runs until the numbers say otherwise.

## 5. v1 → the rest of the thesis (hooks, not promises)

- **v2 — unfreeze + real JEPA:** thaw the encoder (or train a smaller replacement), EMA + SIGREG return, collapse instrumentation per PhyLatient. The predictor built here is exactly the module that loop wants.
- **v3 — SA-LM:** autoregressive LM over quantized (FSQ) s_t — speech-to-text and *semantic-token* generation. Speculative decoding falls out free: SA-CE's own 24-step rollout is the draft; the SA-LM verifies in semantic space. (That is the natural home of the semantic→text speculative-decoding idea; nothing in v1 blocks it.)
- **MoE** re-enters only as a residency mechanism once something must be paged (thesis §5's condition).
- **Research hooks carried, not run now** (they need a trained SA-CE): *safety of prediction-in-semantics* (does the semantic circuit inherit the text-tower's refusals/steering directions? do refusal-like features exist in audio semantics? — probe protocol after v1) and *reasoning in semantic space* (the 24-step prediction IS chain-of-latent; test whether interpolation/mixing in s-space behaves like intermediate reasoning states).

## 6. Falsification table (pre-registered, graded like the manifesto)

| claim | falsified if |
|---|---|
| A frozen-pretrained-latenent space is a learnable target for a 21M causal predictor | ΔS₂₄ ≤ 0 at step 3000 (predictor never beats memoryless) |
| Semantics live in frame states, not just the pooled vector | text-WER plateau while word-alignment CCA stays at pre-init baseline |
| The 4-horizon weighting helps over single-horizon | 4D board ≤ per-horizon single-loss boards at 3000 steps |
| Delta-net beats the same-size SSM arm | gated-delta layer ≤ Mamba-block equivalent at matched params/steps |

**Pre-registered bands (fill BEFORE the first run, he writes them):** ΔS₂₄ @ 3k: ____ · first-night ΔS₂₄: ____ · WER @ 3k: ____