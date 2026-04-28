# 0. Project charter (boundaries)

**Goal:** Build a research vehicle, not a product or a giant foundation model.

Broad semantic model, not "just emotion classification," not a single-task classifier.

Centered on latent semantic prediction (JEPA-style) and hybrid attention + Mamba ideas.

Under hard constraints: limited Mac, daily free Colab (~3h), maybe $50–100 only after you deeply understand the pipeline.

## Key research axes you care about:

- **Multimodal understanding:** different modalities mapping into a shared semantic space.
- **Prediction in semantic space** instead of token/waveform space.
- **Decoding from that semantic space** into various output formats (audio, text, maybe more).
- **Hybrid architectures:** attention layers plus Mamba/SSM layers, especially for long context.
- **Getting larger effective context** than the parameter count would suggest.
- **Practical usefulness:** something you can actually run and use, not just stare at loss curves.

## Key personal learning goals:

- **Training a JEPA-style student–predictor–teacher setup** (V-/VL-/Audio-JEPA lineage).
- **How VL/Video JEPA variants use teachers** (EMA vs frozen) and masking.
- **Training a Mamba-based model** (possibly with your own implementation if libraries are buggy).
- **Audio preprocessing + audio model basics.**
- **Pretraining and optimization basics** you're rusty on.
- **Low-level training/inference and compute-constrained tricks.**
- **How to architect experiments** so you don't waste your own time or money.

## Success for Stage 0:

A 1–2 page internal charter: what the project is, what it is not, what counts as "useful," and which research questions are primary vs "only if there's time."

# 1. Data & preprocessing pipeline

**Objective:** Lock the data stack so you're not debugging training and data simultaneously.

## Choose core datasets (likely):

- 1–2 speech/audio corpora (e.g., LibriSpeech subset, similar to your original plan).
- Possibly one “emotion / paralinguistic / non-ASR” dataset (RAVDESS or similar) purely as one downstream probe, not the project’s identity.
- Optional: some music/environmental audio to keep the semantics broad (as in your SA-JEPA plan).

## Implement audio preprocessing:

- Resample, mono, log-Mel spectrograms, consistent window/hop/mel bins, normalization.

## Define clip segmentation (e.g., 5–20s), bucketing, shuffling, and data-loader behavior on:

- Local Mac.
- Colab free tier.

## Success for Stage 1:

- Datasets chosen and written down (with sizes and usage rationale).
- Preprocessing parameters fixed.
- A single data pipeline that:
  - Loads batches on local and Colab.
  - Can be profiled for speed/memory.
  - Can serve multiple tasks (self-supervised training + small labeled probes).

**Ambiguity allowed:** exact datasets and sizes may change early, but the pipeline design (how you plug new datasets in) is stable.

# 2. Semantic-core experimentation (JEPA + architecture)

**Objective:** Decide how you do JEPA and what your backbone/predictor looks like.

## This is where you test:

### JEPA mechanics

- Student encoder, predictor, teacher targets.
- EMA teacher vs frozen teacher variants (Rethinking-JEPA-style).
- Masking or context cropping strategies (predict masked/future regions in semantic space).

### Backbone family

- Tiny pretrained attention model as backbone (or as feature extractor).
- Insert Mamba layers in the middle or at specific depths.
- Compare: pure attention vs hybrid attention+Mamba vs (maybe) mostly-Mamba with minimal attention.

### Semantic prediction objective

- Direct future embedding prediction (JEPA-style).
- Possibly small variations (multi-step horizon, time-offset conditioning, etc.).

### Context behavior

- How far into the future you can stably predict at small scale.
- How the hybrid structure affects effective context length vs cost.

## This is also the stage where you explicitly pursue your learning goals:

- Implement at least one student–teacher JEPA loop from scratch or near-scratch.
- Get a Mamba block actually training in practice, understand stability issues.
- Refresh pretraining practices: warmup, LR schedules, gradient clipping, logging, etc.
- Do tiny local/Colab runs to measure failure modes: collapse, exploding loss, degenerate embeddings.

## Success for Stage 2:

- One chosen JEPA scheme (student+predictor+teacher, with specific teacher strategy).
- One chosen encoder blueprint (where attention goes, where Mamba goes).
- One chosen prediction objective and masking strategy.
- A documented basic training loop you understand line-by-line.

**Ambiguity allowed:** exact depth, exact width, exact horizon can still be tuned later.

# 3. First serious semantic runs (local + Colab)

**Objective:** Stop living in toy-land and see whether the latent space starts to look non-trivial.

## Run 1–N “proper” JEPA training jobs:

- On your Mac (likely smaller batch, more gradient accumulation).
- On free Colab (~3 hours per day), possibly in short increments with frequent checkpointing.

## Start checking behavioral signals, not just scalar loss:

- Embedding norms, temporal coherence, basic cluster structure.
- Simple linear probes or tiny classifiers on a few downstream labels (e.g., speaker ID, rough class labels, maybe emotion as one probe).

## Success for Stage 3:

- At least one model that:
  - Trains without collapsing.
  - Produces embeddings that are clearly better than random on simple probes.
- You have scripts and configs for:
  - Resuming from checkpoints.
  - Running on both local and Colab.

**Ambiguity:** how many runs exactly; this is where you’ll learn how fragile your setup is.

# 4. Semantic ablations at small scale

**Objective:** Test your research ideas on the semantic backbone itself.

## Questions you explicitly want to probe here:

### Hybrid vs non-hybrid:

- Does inserting Mamba actually give you better effective context per FLOP than pure attention at the same scale?

### Teacher design:

- Does EMA vs frozen teacher materially affect stability/quality under your compute constraints?

### Semantic prediction specifics:

- Horizon length vs stability.
- Masked prediction vs pure future prediction.

### Multimodal potential (latent only):

- Even if you don’t fully train multimodal decoders yet, can the latent space align different audio-derived signals in a way that looks promising for future multimodal extension?

## You run multiple (not one) local+Colab experiments here, explicitly to kill bad ideas:

- 3–5 “biggish” runs is a reasonable expectation, not a hard cap.

## Success for Stage 4:

- You can argue from evidence which backbone + JEPA combo you’re taking forward.
- You have at least one model whose embeddings seem usable and stable across tasks.
- You understand what Mamba is actually buying you (or not buying you) at this scale.

**Ambiguity:** some questions may remain open; you only need enough clarity to justify scaling.

# 5. Final paid semantic run

**Objective:** Spend up to ~$50–100 only when it buys you a clear jump in capability and you aren’t still debugging fundamentals.

- Use the best recipe from Stage 4.
- Run on rented GPU(s) or paid Colab/other cloud, with:
  - Longer training time.
  - Larger batch/sequence lengths (within reason).
  - More data from your existing pipeline, not new mystery datasets.

## Success for Stage 5:

- Your strongest semantic backbone:
  - Trained on as much data as you sensibly can.
  - Showing clear gains over the local-only runs.
- Clear logs, checkpoints, and notes so you can reproduce or extend this later.

**Ambiguity:** exact hardware provider, exact training length, exact dollar spend inside that $50–100 envelope.

# 6. Decoder track: problem definition + first runs

**Objective:** Treat decoders as a separate research axis, plugged onto the fixed semantic space.

## Decide which outputs to care about first:

- Audio reconstruction (audio→semantic→audio), as a sanity check.
- Audio→text (using a small decoder or leveraging pretrained Whisper-style components in some hybrid way).
- Maybe a simple semantic tagger (intent/emotion/command labels) as a light decoder.

## Keep the semantic backbone frozen at first.

## Do initial local/Colab runs of small decoders to see:

- Whether your latent space is actually decodable.
- What level of complexity you need in decoders to get anything non-trivial.

## Success for Stage 6:

- At least one decoder that clearly benefits from the learned semantic space (compared to a trivial baseline).
- A clear sense of which decoder direction is most promising for your final system.

**Ambiguity:** you might discover that some outputs are too hard under your constraints, and that’s fine.

# 7. Decoder ablations and scaling

**Objective:** Put the same discipline you had for the backbone into the decoders.

## Test:

- Different decoder sizes.
- Different conditioning schemes (e.g., compressed sequence vs full sequence).
- Different training strategies (pure supervised vs semi-supervised using your JEPA latent).

## Possibly experiment with multimodal alignment:

- E.g., audio semantic space aligned with text semantic space in a VL-JEPA-inspired way, where decoding to text is just a lightweight head operating on shared latent embeddings.

- Use multiple local+Colab runs (again, not fixed to one).

## Success for Stage 7:

- A reasoned choice of decoder designs that you know you can train and run under your constraints.
- At least one configuration that looks practically useful (e.g., a model that can “talk” in some limited, realistic sense, possibly by leveraging pretrained text/speech modules around your semantic core).

# 8. Pre-final local + Colab optimization (full system)

**Objective:** This is the missing step you pointed out.

## Take the full stack you intend to keep:

- Fixed semantic backbone (from Stage 5).
- Chosen decoders (from Stage 7).

## Run several (not just one) optimized local+Colab experiments to:

- Tighten training efficiency (throughput, memory).
- Clean up logging, checkpointing, failure recovery.
- Do any deep local inference optimization you feel like (quantization, fp8/fp16, caching, etc.), if your excitement and energy justify it.

This is where you try to squeeze every bit of performance out of free/local resources before any final paid full-system runs.

## Success for Stage 8:

- A full-system training script that “just works” on your Mac + Colab.
- Reasonable inference performance locally (e.g., fp8/fp16 on your laptop).
- No major unknowns remaining in the training/inference loop.

**Ambiguity:** exact number of runs; you stop when marginal learning from more runs becomes small.

# 9. Final paid full-system run (if justified)

**Objective:** Only if Stage 8 looks strong do you spend more money on a full-system training run.

## Run the full model (semantic backbone + decoders) under your best recipe.

## Aim for:

- Better benchmarks on chosen tasks (audio-audio, audio-text, etc.).
- Clearly demonstrable real-world behavior: “this thing can talk in X way,” or “this thing can do Y useful semantic mapping,” even if not at SOTA.

## Success for Stage 9:

- A system you can actually use and show: some combination of:
  - Reasonable audio understanding.
  - Some form of text or speech output.
  - A latent space that is meaningfully multimodal-friendly.
- Honest accounting of what comes from your training vs what is piggybacked on pretrained components.

**Ambiguity:** you may decide that the marginal gain from this stage isn’t worth the money; that’s allowed.

# 10. Finalization and publication

**Objective:** Turn this from a personal science project into a shareable artifact.

## Clean code, configs, documentation.

## Archive checkpoints you’re willing to share.

## Write a serious paper/writeup:

- Problem, constraints, architecture.
- JEPA + hybrid Mamba experiments.
- What worked and what didn’t.
- Lessons for constrained training.

## Success for Stage 10:

- A repo and paper you’re not embarrassed by.
- Enough detail that someone else on a low budget could reproduce or extend key parts.
