# Thesis: The 8GB Audio World Model
## why a JEPA + delta-rule + recurrent-depth + quantized + memory-hierarchied audio model can be state-of-the-art on a laptop, not despite the constraint but because of it

*Written 07 Oct 2026, Bangalore. The companion to `manifesto.md` (the bet) and the working plans (`latest_plan.md` and the rung ladder) — this is the *argument* those documents are executing. Committed before a single line of v4.1 code exists, so it can be graded against what we actually build.*

---

## 1. The claim

**We can build, train, and run a state-of-the-art general-purpose audio model on an 8GB M1 laptop — and the 8GB ceiling is the reason the architecture is right, not a handicap we're apologizing for.**

The frontier labs train audio models with quadratic attention on GPU clusters with 80GB of memory and treat capacity as free. Every choice they make is downstream of "we can afford it." Almost none of it is *necessary*. A model designed from scratch under a hard memory-and-FLOPs ceiling is not a shrunk copy of a big model — it is a different design problem, and in 2024–2026 the literature quietly assembled the answer without anyone bolting the pieces together for audio, under constraint, end-to-end. That gap is where this model lives.

The claim is not "good for a laptop." The claim is: **the constraint selects the architecture, and the architecture is a better fit for audio than the datacenter default.** 8GB is the lab. The laptop is just where the proof ships.

## 2. Why now — the four pieces exist, unbolted

In the last eighteen months, four results landed that, read together, describe a model nobody has published for audio:

1. **Latent prediction beats signal reconstruction.** Audio-JEPA (2025) showed masked-latent prediction on spectrograms is competitive with wav2vec 2.0 / data2vec on *a fifth of the training data* [1]. V-JEPA 2 (2025) showed latent prediction learns *transition structure* — a world model, not a feature extractor [4]. The objective that saves data, learns semantics, and is provably non-collapsing with a *single* Gaussian regularizer (a 15M-param model training in hours on one GPU [2]) is exactly the objective an 8GB machine can afford.

2. **Memory management grew up one generation past Mamba.** The delta rule (Gated DeltaNet, RWKV-7, TTT) is an *error-correcting* state update: the state can delete a specific stored value at a specific key, not just decay everything uniformly [6][7]. Constant memory per token, natively streaming, parallel trainable. Qwen3-Next showed this hybrid (Gated DeltaNet + gated attention, 3:1) beats monolithic architectures at scale [12] — and the ratio at *our* scale is a genuinely open question.

3. **Depth no longer costs parameters.** Looped/recurrent-depth transformers apply one shared block R times: effective depth = compute, not weights — up to ~100× parameter efficiency, depth as a new scaling axis [19]. The loop transition itself is a fast-weight update that rhymes with the delta rule. The honest 2026 warning (CART [22]) is that a naive loop can be vestigial — so the loop must earn its place, which is a *design* problem we're equipped to solve.

4. **The memory wall is a hierarchy, not a wall.** MoE sparsity + quantization + SSD offload makes "8GB" mean "8GB of *latency-critical* memory," not "8GB of parameters." 8-bit Adam matches 32-bit at a fraction of optimizer state [17]; 16-bit→1.58-bit continual pretraining beats 1.58-bit from scratch [18]; KV cache and cold experts page to NVMe with real bandwidth headroom (the workload is 128KiB reads and the software, not the flash, is the bottleneck [21]).

**Nobody has combined these for audio.** Each exists in someone else's paper. The *combination*, under a memory ceiling, for audio, end-to-end — that is the contribution.

## 3. The architecture, argued

```
                    ┌─ target encoder (EMA) ──────────┐
   audio ──> mel patch embed                           │ latent targets (masked regions + futures)
     │                                                  │
     └──> context encoder ──────────────────┐           │
            hybrid stack, N blocks:         │           │
            ├─ [ delta-rule × 3 ] ─┐        │           │
            └─ [ gated attn × 1 ] ─┘        │           │
            ALL RECURRENT-DEPTH (shared     ├─> predictor
            block, looped R times)          │    (small delta-rule stack)
            dense quantized FFNs            │    predicts masked-region latents
            ~12–30M active params           │
                    │                        │
                    ├─> probe  (linear probes → understanding)
                    └─> decoder (Phase 2: FSQ latents → vocoder → generation)
```

**JEPA / the objective.** Predict latent representations of masked regions and future states, not waveforms. Anti-collapse by provable Gaussian regularizer + SIGREG, not just EMA [2]. The catch that makes this a thesis and not a recipe: global non-collapse is *not* sufficient — invariance, identifiability, and counterfactual separation can each collapse silently [3]. So the encoder must be designed to keep the latent space a *reliable state space*, not just a non-degenerate one. That's the first thing we test for.

**Delta-rule memory / the engine.** The state update `S_t = S_{t-1}(I − β k kᵀ) + β v kᵀ` is the whole show [6][7]. It is the one piece of the architecture whose memory does **not** grow with sequence length — and that property is what makes the memory hierarchy viable. A pure-attention model cannot cleanly say "this state is tiny, keep it in RAM, page the rest." This one can. The September ladder (full-matrix → diagonal → selective) was tuition for this exact mechanism; the wrapper (conv1d + SiLU gate) is shared with the Mamba block, so nothing built is wasted.

**Hybrid 3:1 / the recall channel.** Three delta-rule blocks : one gated-attention block. The delta-rule stack handles streaming, long context, constant-memory state; the attention block does the exact token-mixing lookups a low-rank recurrent state can't [12]. *The ratio is ours to measure, not to transplant:* 3:1 was validated at 80B/3B-active scale, and the right ratio at 10–30M is an open, publishable question.

**Recurrent depth / the free axis.** One shared hybrid block, looped R times. On an 8GB budget, parameters are precious and depth is the scaling axis you *don't* pay for in weights [19]. The mechanism is not decoration: the loop transition is a fast-weight update with explicit weight decay and adaptive step size [19] — formally adjacent to the delta rule, which is why they compose.

**Quantization + offload / the systems thesis.** This is the piece that turns the laptop from a limit into a design driver. The model is split by what every token touches versus what grows with time:

- **RAM (hot, latency-critical):** the delta-rule recurrent state (tiny, always resident), the current attention layer's KV window, the optimizer state (8-bit only) on the params being touched, activations for the current tokens.
- **External NVMe SSD (cold, bandwidth-tolerant):** unused experts (if ever added, int4, chunked ≥128KiB), the paged KV cache, checkpoints.

The research edge is real: **speech is temporally smooth**, so a prefetch predictor for "which KV entries / which experts you'll need next" is unusually accurate — more so than for text. Nobody has built this for an audio streaming model. And a dedicated external SSD (cheap, replaceable) means the write-heavy KV traffic never touches the drive macOS lives on.

## 4. The thesis, falsifiable

This is a bet, and a bet you can lose. Written now, graded at the end:

1. **The constraint thesis** — an architecture selected by an 8GB budget (delta-rule + hybrid + recurrent-depth + quantized + offloaded) *outperforms* the naive "shrink a transformer" baseline at the same active-param count on audio.
2. **The combination thesis** — no single piece is enough; the ablation grid must show the whole is better than the sum of any subset. If hybrid-or-loop-or-quantization alone wins, one of the five pieces was decoration and we cut it honestly.
3. **The streaming thesis** — because the only thing that grows unboundedly (KV cache) is the thing that lives on disk, RAM stays constant as context length grows. **This is the axis where a laptop model can beat a datacenter model outright**, not asymptotically approach it.
4. **The scale-transfer thesis** — the hybrid ratio and loop depth that win at 12M keep winning when compute is rented once (the ₹10k run). If the architecture only wins at toy scale, it was a toy.

If any of these fails, the right move is to say so, name the number that falsified it, and keep the strongest surviving piece. The thesis earns its existence by surviving attempts to break it, not by being believed.

## 5. What this is not

- **Not a shrunk transformer.** The design starts from the constraint and the modality (audio, streaming, temporally smooth), not from a big model with layers removed.
- **Not a MoE capacity play.** MoE below ~50M active parameters loses to dense on the current evidence [7][8][9 in manifesto review]; here it appears, if at all, only as a *residency* mechanism once the memory hierarchy exists. Capacity comes from depth and quantization, not from inactive experts.
- **Not a claim that SSD makes training free.** Offload + quantization get you *inference* and *context length* on flash; training the dense core stays in RAM. The SSD buys reach, not a free pass.
- **Not a religion.** A pre-registered, falsifiable bet with numbers attached — and a standing offer to be wrong in a way that teaches something.

## 6. Why this matters beyond the model

The audio model is the artifact. **The epistemics is the portfolio.** Designing under a hard constraint, predicting behavior before running it, owning the full cost accounting of your architecture, ablation-honesty, and the discipline to cut what doesn't earn its place — these are precisely the skills the AI-safety and alignment research path screens for. The thesis is "here is the full accounting, here is the prediction, here is the result, here is what I was wrong about," executed in public, one graded number at a time.

And there is a claim worth taking seriously on its own: **the next wave of useful models will not come from scaling what we have — it will come from re-deriving architectures under constraints the giants don't share.** Memory-constrained, streaming-native, temporally-local prediction with a flash-backed hierarchy is that re-derivation, in a domain where nobody has staked the claim yet.

We are not building a toy that fits in 8GB. We are building the argument that 8GB is enough — and shipping the proof.

---

### Sources
[1] Tuncay et al., *Audio-JEPA*, arXiv:2507.02915 — JEPA for audio representation learning, competitive with wav2vec 2.0 / data2vec on 1/5th the data.
[2] *UniJEPA*, arXiv:2608.07409 — single Gaussian regularizer, provable anti-collapse, 15M model trains in hours on one GPU.
[3] *PhyLatent*, arXiv:2608.05720 — global non-collapse is insufficient: invariance / identifiability / counterfactual collapse.
[4] Assran et al., *V-JEPA 2*, arXiv:2506.09985 — latent prediction learns transition structure (understanding, prediction, planning).
[6] Peng et al., *RWKV-7*, arXiv:2503.14456 — generalized delta rule, vector gating, constant memory.
[7] *Comba*, arXiv:2506.02475 — delta-rule family as bilinear RNNs / state-feedback control.
[12] Alibaba, *Qwen3-Next* — Gated DeltaNet + Gated Attention 3:1, ultra-sparse MoE (the ratio we re-derive, not transplant).
[17] Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861 — 8-bit Adam matches 32-bit at a fraction of optimizer memory.
[18] *Continual Quantization-Aware Pre-Training*, arXiv:2502.11895 — 16-bit → 1.58-bit continual pretraining beats 1.58-bit from scratch.
[19] *Looped Transformers as Optimizers*, arXiv:2609.37379 — loop transitions as fast-weight updates; depth as a parameter-free scaling axis (~100× parameter efficiency).
[21] *An I/O Characterizing Study of Offloading LLM Models and KV Caches to NVMe SSD*, ACM 2025 — workload dominated by 128KiB reads; NVMe not saturated by naive offload.
[22] *CART*, arXiv:2606.01495 — honest negative result: recurrent-core machinery individually vestigial in their config; the warning we watch for.
