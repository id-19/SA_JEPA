Anything with // is meant to be a side note/comment type thing
/This is a basic scope definition and conceptual summary of the project
//Written by AI

## What this project is
A compute‑constrained research vehicle to explore JEPA/VL‑JEPA style semantic prediction with hybrid attention + Mamba backbones, starting from audio and extending toward multimodal understanding and speculative decoding, using as much pre‑made backbone structure as possible to avoid wasting time on low‑value pretraining boilerplate.

The core object is a small but strong semantic model that takes in audio (and later other modalities) and learns to predict and manipulate latent semantic states, which can then be decoded into multiple output formats (audio, text, etc.) using lightweight decoders and/or pretrained components.

## What this project is not
1. A benchmark buster: I'm not going to be beating any benchmarks, this is not a full-blown SOTA on benchmark project
2. Not focused on any narrow functionality: Test performance on a variety of tasks that show true semantic understanding and prediction.
3. **4 weeks of beating up my laptop with size: Scale is not the most important thing, this is meant to be a POC at small scale to verify research ideas.
4. An all-in-one audio pre-trained model: I don't have the data budget, memory or the compute to train a full audio pre-trained model from scratch.

## Important research ideas I want to explore in this project
This is actual research stuff I want to explore in this project, the hypotheses and methods I want to test.

1. Training JEPA models
2. Integrating Mamba with JEPA
3. Integrating Mamba with transformers
4. VL-JEPA-type advances
5. Re-using backbones
6. Training in constrained regimes
7. Optional: auto-research loop
8. Practical targets

**Training JEPA models:** Learn proper JEPA training with student, predictor, teacher architecture. Train by predicting semantic embeddings, not raw signals or tokens. Understand masking, future prediction, EMA teacher vs frozen teacher. Explore multimodal embeddings in a shared latent space, not just unimodal representations.

**Integrating Mamba-type models with JEPA:** Combine JEPA training with stateful Mamba / SSM-style models. Test whether JEPA + Mamba gives better long-context semantic prediction than JEPA + plain attention. Understand how state, recurrence, and latent prediction interact.

**Integrating Mamba-type models with transformers:** Build and test hybrid attention + Mamba architectures. Figure out where Mamba layers should go, what they replace, and what they improve. Main target: better effective context window, memory behavior, and efficiency for small models.

**VL-JEPA-type advances:** Use the latent-first, decode-later idea. Keep decoding lightweight and only invoke it when needed. Explore speculative/selective decoding in some form, especially if latent prediction can help reduce decoding work. VL-JEPA reports selective decoding reducing decoding operations by about 2.85x while keeping similar performance.

**Re-using backbones:** Avoid full from-scratch pretraining whenever possible. Reuse tiny pretrained backbones / premade encoders, then add JEPA training structure, predictor heads, Mamba layers, hybrid blocks. Study what can be gained by repurposing existing models instead of rebuilding everything.

**Training in constrained regimes:** Treat constrained training as a core problem, not just an annoyance. Learn how to make useful progress with Mac local training, free Colab, small paid budget later. Design experiments so that cheap runs still teach something real.

**Integrate JEPA model with an MoE type architecture:** Try this in the toy runs, see if it helps with training or performance.

**Have recurrent attention layers:** This is really hot rn, probably will be a pain to figure out but solid if you can make it work. Valuable shit on a research and learning basis, will likely get you outsized performance too.

**Optional: auto-research / small agentic loop:** Later, maybe add a small loop that helps with experiment selection, hyperparameter search, architecture search, training analysis. This is optional and comes after the core system is real.

## Practical capabilities

Primary practical target: a small semantic model that powers at least one useful local audio workflow.

**Most realistic workflows:**
1. **Offline personal STT that is actually usable** — Record speech, get decent text locally, no API bill, no cloud dependency. Bonus if it handles timestamps or rough segmentation.
2. **Audio search / recall using semantic embeddings** — Query old audio notes by meaning. Example: "find the clip where I talked about optimizer bugs" and retrieve the right region.
3. **Basic speaker-aware transcription** — Enough to separate "who said what" in short conversations. Whisper-based tiny diarization prototypes show this is possible on consumer hardware.
4. **Small audio-to-audio semantic transform** — Simple but real behavior where the model processes audio semantically. Examples: denoised speech representation, compressed semantic reconstruction, or latent-driven audio rewrite.

**Stretch workflow:**
5. **Simple voice interface on top of the semantic core** — Audio in -> semantic model -> text or speech response. Small local voice-assistant projects show this is practical on-device.

## Key Learnings

### Essential
- Train JEPA properly: student, predictor, teacher, masking, future prediction, latent objectives.
- Understand teacher strategy in JEPA variants: EMA vs frozen teacher, and why/when each works.
- Train Mamba-based and hybrid transformer+Mamba models without fooling myself.
- Learn to run real pretraining under hard compute constraints without wasting runs or money.

### Basic
- Audio preprocessing and audio modeling basics for clean pipelines.
- Optimization fundamentals: warmup, clipping, logging, checkpointing, mixed precision.
- Low-level training/inference discipline and experiment design so every run teaches something.

### Stage-based Outcomes
- **Stage 1-2:** Lock data pipeline, implement JEPA mechanics, get Mamba training stably.
- **Stage 3-4:** Produce non-collapsing models with usable embeddings, understand hybrid architecture tradeoffs.
- **Stage 5:** Execute one serious paid run with clear gains over local-only training.
- **Stage 6-7:** Build decoders on frozen semantic space, align audio-text in latent space.
- **Stage 8-9:** Optimize full system for local inference, potentially demonstrate practical voice interface.
- **Stage 10:** Document and publish reproducible work for others on constrained budgets.
