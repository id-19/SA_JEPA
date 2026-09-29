# The 8GB Manifesto
## SA-JEPA v4.1 — building the best audio model that fits on a laptop

*Written 28 Sep 2026, Bangalore. Committed before a single line of v4.1 code exists — so it can be graded against what we actually build.*

---

### 0. The bet

We are going to build a **general-purpose audio model that trains and runs on an 8GB M1 laptop** — and we claim that the constraint is not our weakness. It is our thesis.

The giants train on GPU clusters with 80GB of memory and quadratic attention over million-token spectrogram dumps. Every one of their choices is downstream of "we can afford it." Almost nothing in their architecture is *necessary* — it is what falls out of excess. A model designed from scratch under a hard memory-and-FLOPs ceiling is not a shrunk copy of a big model. It is a different design problem, and the literature has been quietly assembling the answer for three years without anyone bolting the pieces together for audio.

Our bet: the pieces are

1. **JEPA** — predict representations, not signals,
2. **delta-rule recurrent memory** — Mamba's descendants, constant memory per step,
3. **hybrid non-quadratic attention** — a 3:1 recurrent:gated-attention layout,
4. **ultra-sparse MoE** — capacity without activation cost,

and that **no published model combines all four for audio.** That gap is ours. If the bet is right, the result is not "good for a laptop model." It is a new point on the efficiency frontier, and the laptop is just where it happens to live.

---

### 1. The enemies

Name them, because every design decision is a strike at one of them:

- **O(T²) attention.** On audio spectrograms, T is brutal — 50Hz frames mean 500 tokens per 10 seconds, thousands for minutes-long clips. Audio Spectrogram Transformers pay quadratic rent on every layer [1]. We pay linear, or we don't pay at all.
- **The memory wall.** 8GB unified. Every MB of optimizer state, every KV cache, every dense FFN activation is a weapon taken from us. Total params ≠ active params ≠ resident memory. We exploit all three seams.
- **Pixel-space prediction.** Reconstruction wastes capacity on signal details no downstream task needs. JEPA already proved this in vision [2] and audio [3]: predict the *representation* of what's masked, and you learn semantics on a fifth of the data.
- **Scaffolding creep.** The old plan spent September on toy sines, diagonal-arm theory, and renorm bounds. That was the tuition. The education is paid for. From here, every rung must move the actual model or it doesn't exist.

### 2. The architecture — SA-JEPA v4.1

One sentence: **a JEPA-trained, ultra-sparse-MoE, hybrid-attention encoder over spectrogram frames, where the "attention" is mostly a delta-rule recurrent memory, plus a lightweight recurrent-attention predictor — encoder-decoder for anything that emits.**

**Objective: JEPA, always.** Context encoder sees visible spectrogram patches; EMA target encoder sees all; predictor predicts masked-region latents; stop-grad + SIGREG against collapse [3][4]. No waveform or spectrogram reconstruction, ever. Prediction targets are *masked regions* and *time-shifted futures* (the world-model move: V-JEPA 2 shows latent prediction is what makes representations robust and plannable [4]).

**Backbone: hybrid, not monolithic.** The layout is Qwen3-Next's [5], transplanted to audio: 75% delta-rule linear-attention layers (Gated DeltaNet / RWKV-7-style generalized delta rule [6][7]) interleaved 3:1 with a small number of gated-attention layers for exact token-mixing lookups. Audio Mamba already proved SSMs beat transformer baselines for self-supervised audio [8]; we go one generation up the ladder — from Mamba's selective decay to *delta-rule memory management*: the state can delete a specific stored value, not just decay everything uniformly [7]. Our Sep-23 diagonal-arm theory (real `a` hosts only ω=0 or ω=π; oscillation must be synthesized from a real-pole bank) is exactly the kind of first-principles liability we now pay attention to when choosing parameterizations.

**Capacity: ultra-sparse MoE.** Fine-grained experts, tiny activation ratio, one shared expert. The MoE scaling laws say the leverage is real and predictable — Ling-mini-beta matched a 6.1B dense model with 0.85B active, >7× compute leverage [9]; memory-constrained settings *favor* MoE over dense [10]. For us the knob isn't just FLOPs-per-token: **experts are our substitute for the parameters we can't afford to keep active.** The router also gets an audio-specific job: frequency-band and phonetic-context routing, so the model learns *which* part of its capacity handles sibilants vs vowels vs silence.

**Output: encoder-decoder where it matters.** For representation tasks, encoder + probe. For anything generative or sequential (transcription, streaming prediction), a small autoregressive decoder over the latents — Moonshine-sized (34M–123M parameters) is the calibration for what "edge-class" means [11]. Recurrent attention in the predictor: the predictor itself is a cheap delta-rule stack, constant memory, so long rollout horizons cost nothing quadratic.

**Streaming is a first-class property, not a bolt-on.** Delta-rule layers are natively streaming; the few attention layers use bounded windows. This is the one axis where the laptop model can *beat* the giants outright, not just approach them: constant-memory, constant-latency, arbitrarily-long audio.

### 3. The numbers we are chasing

Pre-registered now, graded later:

| Claim | Target | Falsified if |
|---|---|---|
| Params | 50–100M total, ≤15M active (MoE) | can't hold in <2GB at bf16 with optimizer state offloaded |
| Throughput | trains ≥1h of LibriSpeech audio per night on M1 | <30min/night sustained |
| Representation quality | beats SSAST-baseline linear probes on ≥3/10 tasks [8] | loses head-to-head at matched active params |
| Streaming | constant memory/latency at 10min context | memory grows with T |
| The novel bit | hybrid 3:1 + MoE + JEPA beats each axis removed | ablations show no combination effect |

The last row is the whole point. Every piece exists in someone else's paper. **The combination, under a memory ceiling, for audio, with JEPA targets — that is the contribution.** The ablation grid isn't a nicety; it's how the claim earns its existence.

### 4. Built backwards: the learning plan

Every item either feeds the model or gets cut. No more "future ideas to try (and validate)" parking lots — ideas enter the plan only when they are a milestone with a kill criterion.

**Track A — the memory core (continues from rungs 1–4, timeboxed)**
1. Finish the SSM ladder *fast*: diagonal arm head-to-head vs 0.0299, selective Δ, Mamba block wrapper. Timebox: this is the last week of scaffolding. Each rung ends with a board number, not a journal entry.
2. Delta-rule (Gated DeltaNet-style) mini-implementation on the sine board — this is *the* new mechanism, worth the depth Mamba got. The state-matrix-update view is the unification of everything built so far (SSM = rank-1 memory write; delta rule = error-correcting write).

**Track B — the objective (in parallel, cheap)**
3. JEPA loop on toy data with a *tiny* transformer pair — stop-grad, EMA, SIGREG. Learn collapse dynamics on a model that trains in seconds.
4. Predictor-as-recurrent-model experiment: does a delta-rule predictor match a transformer predictor at a fraction of memory? This is a genuinely open question and cheap to test.

**Track C — the model (from ~2 weeks in, the only track that matters after)**
5. v4.1-skeleton: patch embed → 1 hybrid block (3 delta-rule : 1 gated attention) → probe. Overfit-one-batch gate, seeded determinism gate.
6. Add MoE to the FFN slots, fine-grained, 1 shared expert, activation ratio ≤15%. Load-balancing loss from day one.
7. Real data: LibriSpeech mels through the full stack. Board: blind / probe / v4.1.
8. Scale-up ladder: 8GB M1 → 16GB → rented GPU *only* for the one run that tests whether the architecture keeps winning as compute grows (goal.md's ₹10k, spent once, on purpose).
9. Ablation grid → the table in §3, filled in, graded against this manifesto.

**Cut on sight:** LoRA-on-big-model fantasies, GQA taxonomies for their own sake, more theory arms, any experiment that cannot change a §3 number.

### 5. The standards

- **Pre-register every guess, grade every run.** A run whose outcome we can't predict teaches nothing.
- **Board numbers over prose.** blind 0.4835 → wall 0.20245 → mine 0.0299 → ... every milestone is a number on a scoreboard.
- **One copy of the plan, in order.** No parallel versions drifting.
- **The 8GB machine is the lab, not the limit.** Ideas too big for the M1 get tested at M1-scale first, then rented. Never the other way around.
- **Honest falsification.** If the 3:1 hybrid shows no combination effect on audio, we say so, kill it, and keep the strongest surviving piece. The manifesto is a bet, not a religion.

### 6. Why this matters beyond audio

This is the same muscle the alignment path needs: **designing under constraint, predicting behavior before running, ablation-honesty.** A researcher who can hold "here is the full cost accounting of my architecture, here is the prediction, here is the result, here is what I was wrong about" is doing the *exact* skill MATS screens for. The audio model is the artifact. The epistemics is the portfolio.

And there's a claim hiding in here worth taking seriously: **the next wave of useful models will not come from scaling what we have — it will come from re-deriving architectures under constraints the giants don't share.** Memory-constrained JEPA + delta-rule + sparse experts is that re-derivation, in a domain (audio) where nobody has staked the claim yet.

We are not building a toy that fits in 8GB. We are building the argument that 8GB is enough — and shipping the proof.

---

*Grade this document the way we grade a pre-registered guess: at the end of Track C, against the filled §3 table.*

**Sources:**
[1] Audio Spectrogram Transformer (Gong et al.) — quadratic attention baseline for audio classification.
[2] I-JEPA (Assran et al., 2023) — masked latent prediction in vision. arXiv:2301.08243.
[3] Audio-JEPA (Tuncay et al., 2025) — JEPA for audio, competitive with wav2vec 2.0 / data2vec on 1/5th the data. arXiv:2507.02915.
[4] V-JEPA 2 (Assran et al., 2025) — latent prediction → understanding, prediction, planning; world-model move. arXiv:2506.09985.
[5] Qwen3-Next (Alibaba, 2025) — Gated DeltaNet + Gated Attention 3:1 hybrid, ultra-sparse MoE (80B/3B active). https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct.
[6] RWKV-7 "Goose" (Peng et al., 2025) — generalized delta rule, vector gating, constant memory. arXiv:2503.14456.
[7] Comba (2025) — bilinear RNNs; delta-rule family (Gated DeltaNet, TTT, RWKV-7) as state-feedback control. arXiv:2506.02475.
[8] Audio Mamba (Yadav & Tan, INTERSPEECH 2024) — SSAM beats SSAST on 10 downstream tasks, ~30% aggregate gain. arXiv:2406.02178.
[9] Ling-mini-beta / MoE scaling laws (2025) — 0.85B active ≈ 6.1B dense, >7× compute leverage; activation ratio is the primary driver. arXiv:2507.17702.
[10] Joint MoE scaling laws (Ludziejewski et al., ICML 2025) — MoE can be *more* memory-efficient than dense under fixed budgets. https://proceedings.mlr.press/v267/ludziejewski25a.html.