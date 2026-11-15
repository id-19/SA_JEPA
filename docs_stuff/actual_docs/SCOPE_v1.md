# SA-JEPA v1 — scope list

*Started 13 Nov 2026. This is a running list. It is not a design.*

**What this is.** A list of what is in scope for SA-JEPA v1, and of things that were explicitly put off past v0. "v0" is the current text-only ~100M SSM-hybrid JEPA build ([DESIGN_v0.md](DESIGN_v0.md), [BUILD_v0.md](BUILD_v0.md)). "v1" is the NEXT model version after v0.

**Careful with names.** The change log in DESIGN_v0.md has entries called "v1" and "v1.1". Those are revisions of the v0 design doc, not the v1 model.

**Where a remote v1 list also exists.** Linear has a project called "SA-JEPA v1" (P-ID1-20, status Backlog, created 9 Nov 2026, inside the HIM initiative). It holds two tickets, listed in section 7. It has no description and no documents. This file was not copied into it. Linear and Drive were read only.

## How to read each item

- **Who said it.** "USER" means the user decided or said it directly. "Proposal (AI)" means an AI agent wrote it in a doc, ticket or chat. AI text is not a user decision.
- **Status.**
  - *In v1 (user decided)*: the user put it in v1 scope.
  - *Deferred past v0*: someone put it after v0. It has not been assigned to v1 specifically.
- Nothing here has a date, budget, estimate or threshold unless the source gave one.
- Items where v0 versus v1 is not clear are in section 8, not in the groups above it.

---

## 1. Architecture

### 1.1 Choose the W_o rank empirically (output projection of MultiHeadGDN)

- **Status:** In v1 (user decided, 13 Nov 2026).
- **Who:** Decision is USER. The analysis and the numbers below were produced by an AI agent in the same chat.
- **Where:** `layers/SSMs/multi_headed/MultiHeadGDN.py`. In commit `3891bb5`, W_o is factorised as `W_o_high (d_model, rank) @ W_o_low (rank, d_output)` with rank=4. (The working tree on 13 Nov, read later the same day, shows a plain full `W_o (d_model, d_output)` instead. The file may have been changed after the chat. Check before running the experiment.)
- **Why it is in scope.** Every token's layer output is a mix of the same `rank` fixed vectors (the rows of W_o_low). So the layer can only write a rank-4 update into the residual stream. Checked numerically in the chat: 10k random inputs with d_model=256 gave output rank 4, against about 256 with a full matrix. The token itself is not capped, because the residual carries it. The layer's contribution is.
- Low rank only saves parameters when r < d/2 for a square matrix. Rank 4 at d=256 is 3% of the full parameter count.
- **Experiment.**
  1. Train a small model with a full-rank W_o.
  2. Take `torch.linalg.svdvals(W_o)`. Count the singular values that hold about 90 to 95% of the energy. That is the rank the layer actually uses.
  3. Ablate low-rank against full on loss.
- **Precedents where low rank works:** ALBERT factorised embeddings, LoRA (a low-rank update on top of a full matrix), DeepSeek MLA (KV-cache compression).
- **Not part of this item:** low rank for alpha and beta (per-head scalars). That is fine as it is.
- **Related, from the same week (Proposal (AI)):** the Gated DeltaNet paper (arXiv 2412.06464, pages 1 to 8 read by an AI agent) factorises nothing on the output side. It has a full gate projection and a full W_o. The agent also noted that W_o does not touch the memory state, so low-ranking it limits only the directions the layer writes back to the residual stream. Source: session `5af73c0f`, 13 Nov 2026, 11:12. The user's question that started it was "we can low rank this no" (10:53 same session).
- **Source:** today's chat, session `0fc48b60`, 13 Nov 2026 (decision). Supporting: session `5af73c0f`, 13 Nov 2026; session `052df7ae`, 13 Nov 2026, 10:41 (an AI agent called the evidence for low-rank gates "thin" and said low rank suits control signals better than content).

### 1.2 Multimodality, starting with audio

- **Status:** Deferred past v0.
- **Who:** USER said it was left for after. The user's words, 12 Nov 2026: "Some things like multimodality, MLA, MoE, latent space reasoning, sparse attention implementations were left for after." The audio target is also in the user's 2026 annual goals (a text-and-audio model on 8 GB).
- **Why.** v0 is text-only so the two core convictions can be tested first (hybrids beat attention on memory and compute; semantic prediction is the way forward). Audio comes second ("train with text first, audio second" is in `IDEA.md`, the user's older doc).
- **Proposal (AI) on how:** Balestriero and LeCun's result says JEPAs trained on good views of the same thing learn the same space up to one rotation. So an audio encoder trained later could land in the same space as the text one. DESIGN_v0.md section 5 says this is what makes audio "a rotation, not a rebuild". This is an argument, not a result for this model.
- **Source:** sessions `f067672d` and `51d27fbe`, 12 Nov 2026, 11:38 (first message). `docs_stuff/actual_docs/IDEA.md`. `/Users/id19/Desktop/Core_Document/Actual_Docs/Annual/2026/02_2026.md`. DESIGN_v0.md sections 1 and 5.

### 1.3 MLA, MoE and sparse attention

- **Status:** Deferred past v0.
- **Who:** USER, same quote as 1.2.
- **Notes.**
  - Linear ID1-152, "Read and implement Latent Sparse Attention (Whittington et al., 2026)", is in the v0 project, status Backlog, not archived. Description: read the paper and implement GQA/MLA with it. The ticket text looks AI-written ("Estimated effort: 4 hours"); the user created the ticket.
  - ID1-162 (learn and implement sparse attention) is Canceled.
  - ID1-167 "Implement basic GQA/MLA for SA-JEPA" is Done. So some MLA work exists. What "MLA left for after" covers beyond that is unclear. See section 8.
- **Source:** as 1.2. Linear ID1-152, ID1-162, ID1-167.

### 1.4 Dynamic stopping for the predictor loop, and carrying hidden state between passes

- **Status:** Deferred past v0.
- **Who:** The idea of a loop with "some fixed or dynamic" number of steps is USER (12 Nov 2026, 13:34). Deferring the dynamic part is Proposal (AI), in DESIGN_v0.md.
- **Why.** v0 runs a fixed number of passes first (R=4 as a pilot switch). Stopping early is awkward in a sequence model, because a later chunk may need a memory state an earlier chunk never computed. DESIGN_v0.md says it is built only if most chunks turn out to settle early. Open question 2 in that doc: is the prediction alone enough scratch space, or must the hidden state also pass between passes?
- **Source:** session `f067672d`, 12 Nov 2026, 13:34. DESIGN_v0.md sections 5 and 10.

### 1.5 Memory inside the loop (stored latents fetched as extra chunks)

- **Status:** Deferred past v0.
- **Who:** Proposal (AI). DESIGN_v0.md section 7. It came from an AI-written free-run draft. The user has not said they want it.
- **Why (as argued in the doc).** Stored latents are fetched and fed to the predictor as extra chunks tagged as memories. With the loop it becomes think, recall, think. A chunk is written to the store only if the predictor got it badly wrong. RETRO matched GPT-3 with 25 times fewer parameters this way. The predictor leaves an unused argument for it now.
- **Source:** DESIGN_v0.md section 7.

### 1.6 Choosing a thought instead of averaging (sampled predictions)

- **Status:** Deferred past v0.
- **Who:** Proposal (AI) in DESIGN_v0.md section 7. USER reaction on 12 Nov 2026: first "this is a weak idea since it basically replicates a distributional loss", then 13:44: it is not covered by SIGReg, because SIGReg avoids collapse across data and this avoids it across predictions for one context. The doc records it as a later idea.
- **Why (as argued in the doc).** MSE gives the average of the plausible next latents. With two opposite good answers the average is neither. Give the predictor head a noise input, draw several predictions per context, pull them toward the target, push them apart. It should be the same function as SIGReg with a different reference. CALM trains this way. A cheap half-step (SIGReg on the predictions, no sampling) is already a v0 pilot switch.
- **Related ticket:** ID1-278 (read and implement CALM; minimal toy reproduction only; full-scale training "NOT approved scope"), v0 project, Backlog. The ticket text is likely AI-written.
- **Source:** DESIGN_v0.md sections 7 and 9 (risk 2). Session `f067672d`, 12 Nov 2026, 13:43 to 13:44. Linear ID1-278.

### 1.7 Bytes to words to thoughts (drop the tokenizer)

- **Status:** Deferred past v0.
- **Who:** Proposal (AI), DESIGN_v0.md section 7.
- **Why (as argued in the doc).** A tiny byte-level model emits one vector per word. The encoder reads word vectors. A tiny byte model spells the output. It removes the embedding table and the 32,000-way output layer, makes a typo a nearby input, suits romanized Hindi, and is a route to audio as one more bottom level. H-Net and the Byte Latent Transformer match tokenized models at a billion parameters or more. Nobody has shown it at 100M. It comes last because a bug at the bottom hides everything above it.
- **Source:** DESIGN_v0.md section 7.

### 1.8 Predictor ablations after v0

- **Status:** Deferred past v0.
- **Who:** Proposal (AI). From the Drive doc "SA-JEPA v0: 100M text run plan, independent (revised)" (id `19WaZPZCNTgT087YX_ZzHLWeXfuZ4o4LvPCmrX614qoU`) section 7: "Ablations, only after v0". The user said earlier plans written by AI are not signed off.
- **Items that still fit the agreed design:** SWA against GQA as the attention layer; GDN against KDA against Mamba-3 in the predictor; Hymba-style parallel branches; more than one latent per chunk.
- **Items that do not fit:** the doc's A2 (block size 16 against 64) assumes fixed blocks. The user rejected fixed-size blocks on 12 Nov 2026.
- **Source:** Drive doc `19WaZPZCNTgT087YX_ZzHLWeXfuZ4o4LvPCmrX614qoU`, section 7.

### 1.9 Short conv on the gates (α, β), not only on q, k, v

- **Status:** The user wants it run ("we'll do the ablation", 15 Nov 2026). When is not decided. The AI suggested a cheap pilot arm.
- **Who:** The question is USER's, asked during the [1.3a] GDN conv quiz (session `26e01eb6`, 15 Nov 2026). The background below is AI.
- **Why.** GDN and KDA in FLA and in `layers/SSMs/reference/` run the depthwise causal conv on q, k and v only. α and β come from the raw x_t. Practice is not uniform: in Mamba-1 the decay step Δ is computed from the conv output, so the gate sees the last k tokens. In Mamba-2 the conv covers x, B and C but not Δ. The argument for leaving the gates alone is that they are per-moment decisions (forget now, write now), and x_t already carries context from earlier layers. Neither the user nor the AI knows of a clean ablation. Arms: conv on q, k, v (as now) against conv on q, k, v, α, β, with everything else equal.
- **Related:** the α gate in the user's KDA has no bias, so at init α averages 0.5 and the layer forgets within a few tokens (measured 15 Nov: the last output depends on a token 30 back at about 1e-7). The reference initialises slow forgetting with `A_log` and `dt_bias`. That is the "decay bias" item in open question 6. Settle it before this ablation, or the gates being compared start from a layer with almost no memory.
- **Source:** session `26e01eb6`, 15 Nov 2026. `layers/SSMs/reference/gated_deltanet.py` (conv on q, k, v only, line 37 onward).

---

## 2. Training and losses

### 2.1 Stage 2: learn to speak (decoder, then whole stack), and chat fine-tune

- **Status:** Deferred past v0 by BUILD_v0.md ("Deliberately not ticketed yet ... They come after [7.2]"). Whether this is v1 or an in-between step is in section 8.
- **Who:** The order (encoder and predictor first on latents, decoder and the intermediate layer later, then the whole stack trained with the decoder) is USER, 12 Nov 2026, 12:14. The exact stage-2 recipe is Proposal (AI) in DESIGN_v0.md section 4.
- **Why.** A randomly initialised decoder sends large, meaningless gradients, so it trains first on frozen latents. Then the whole stack trains with the token loss added and the stage-1 losses still on, so the space stays Gaussian and predictable. The retrieval readouts are re-run afterwards to measure drift.
- **Source:** session `f067672d`, 12 Nov 2026, 12:14. DESIGN_v0.md section 4. BUILD_v0.md "Deliberately not ticketed yet".

### 2.2 Weight editing by a model like Claude

- **Status:** Deferred past v0 (after stage 2 and the chat fine-tune).
- **Who:** The idea is USER's (DESIGN_v0.md says "your idea"). The caveat is AI: it says it knows of no evidence this works, and that the edits it would trust first are closed-form ones (fold an exact whitening into the latent projection, re-solve a linear head by least squares, calibrate a temperature). Judged by before-and-after readouts.
- **Source:** DESIGN_v0.md section 4.

### 2.3 Extra control arms for stage 2

- **Status:** Deferred past v0.
- **Who:** Proposal (AI), DESIGN_v0.md section 7.
- **Why.** Two controls: the same architecture trained on cross-entropy from scratch (does latent pretraining earn its place for generation?), and an ordinary token-level hybrid language model at matched compute (what does the chunk design cost?).
- **Source:** DESIGN_v0.md section 7.

### 2.4 Fixes held in reserve for v0 failures

- **Status:** Contingent. Only if v0 shows the matching failure.
- **Who:** Proposal (AI), DESIGN_v0.md section 9.
- **Items.** Decoder sees the last few tokens of the previous chunk (if names do not survive a chunk boundary). Narrower latent (if dimensions turn into filler). Lower learning rate on encoder and predictor, or a longer frozen phase (if stage 2 damages the semantic space). Overlap on the decoder side only (if chunk boundaries hurt; see 8.5).
- **Source:** DESIGN_v0.md sections 5, 6 and 9.

---

## 3. Data

### 3.1 Noisy and clean text pairs for autocorrect

- **Status:** Deferred past v0.
- **Who:** Proposal (AI). Drive doc `19WaZPZCNTgT087YX_ZzHLWeXfuZ4o4LvPCmrX614qoU`, sections 2 and 7: autocorrect needs a decoder that reads a noisy block and writes clean tokens, "which needs noisy/clean pairs and is v0.5 data work". DESIGN_v0.md section 5 says typo-damaged copies are not good views, but are useful as input noise for robustness.
- **Source:** the Drive doc above. DESIGN_v0.md section 5.

---

## 4. Evaluation

### 4.1 Generation-side evaluation (chat quality, autocomplete, reasoning benchmarks)

- **Status:** Deferred past v0.
- **Who:** Proposal (AI). DESIGN_v0.md section 1: "Chat and autocomplete come in the stage after that." The v0 gate is a retrieval test over the user's own notes. The Drive doc said GSM8k "belongs here or later, if at all", and that a latent-only model cannot produce responses until a decoder exists.
- **Why.** v0 has no decoder trained, so generation metrics have nothing to measure yet. See also 8.2.
- **Source:** DESIGN_v0.md section 1. Drive doc `19WaZPZCNTgT087YX_ZzHLWeXfuZ4o4LvPCmrX614qoU`, sections 1 and 7.

### 4.2 Evals that beat similar-size models, and a paper

- **Status:** Not assigned to a version. See 8.1.
- **Who:** USER, in the 2026 annual goals: "SA-JEPA beats on some evaluation, a similar model of its own size"; a paper on OpenReview and arXiv; a real demo.
- **Source:** `/Users/id19/Desktop/Core_Document/Actual_Docs/Annual/2026/02_2026.md`, sections 1 and 2.

---

## 5. Infrastructure and inference

### 5.1 On-device measurements and inference speed

- **Status:** Deferred past v0.
- **Who:** BUILD_v0.md (AI-written) lists "on-device measurements" as not ticketed yet, after [7.2]. The USER's annual goals ask for an inference engine at a real-time speed (the goal text says 5+ tok/s on the laptop with almost the same performance). The USER also said on 28 Oct 2026: "I need to highly optimise inference later so the design does matter that way."
- **Source:** BUILD_v0.md. `02_2026.md`. Session `2c010190` (project dir `-Users-id19-Programming-AI-research-SA-JEPA-SA-JEPA`), 28 Oct 2026, 17:05.

### 5.2 Quantisation for more efficient inference

- **Status:** Deferred past v0 (backlog).
- **Who:** USER-created ticket. The description is one line, with no reasoning.
- **Source:** Linear ID1-172, v0 project, Backlog, not archived.

---

## 6. Product (HIM)

### 6.1 HIM role split and audio-native perception

- **Status:** Deferred past v0. This note was written for the older audio-first model, so parts may be stale.
- **Who:** Proposal (AI). `/Users/id19/Desktop/Imp personal stuff/HIM on-device redesign (JEPA).md`, dated 14 Oct 2026. It says SA-JEPA v0 is not a chat model, so HIM splits roles: SA-JEPA encoder for perception, memory and confidence; a small local LLM for language and tool calls; a thin orchestrator.
- **Items in that note that depend on a later model:**
  - A transcription head on the latents that replaces Whisper.
  - A confidence router that uses a latent density score to decide when to escalate (the note says it is only trustworthy after silent-collapse checks land).
  - Generation (FSQ latents to a vocoder to audio replies) as "Phase 2", downstream of a validated encoder.
  - Persona behaviour later as a small adapter on a decode head.
- **Why it matters for v1.** v0's latent store and retrieval readouts (BUILD_v0.md [5.4], [5.3]) are the v0 piece HIM would use.
- **Source:** the HIM note above. The user's own words on what v0 is for (12 Nov 2026, 11:38): v0 is "the POC or bare minimum usable version for a future model that powers a powerful, constantly improving on-device agent, namely the HIM application."

---

## 7. Other

### 7.1 Linear project "SA-JEPA v1" (P-ID1-20): two reading tickets

- **Status:** In the project named v1. Not archived. Both in Backlog.
- **Who:** Both tickets were created by Ishaan Datta (USER). They were updated on 9 Nov 2026, 13:20, minutes after the project was created (13:05). Linear does not show who moved them in.
- **ID1-120, "Study activation plateaus further".** Read three papers (2024 stable regions across models, arXiv 2409.17113; 2025 where and how plateaus emerge, LessWrong; 2026 feature-specific error correction, arXiv 2606.24964), then run an own experiment, scope to be decided.
- **ID1-47, "Read and internalise Towards a Science of Scaling Agent Systems".** arXiv 2512.08296.
- **Why this needs a look.** Both are reading tickets, with no stated link to a model change. Whether they are v1 scope, or only a parking place, is not stated.

### 7.2 Safety work on the model

- **Status:** Not assigned to a version. See 8.1.
- **Who:** USER, annual goals: mech interp and fundamentals, tracing through the model, testing on harmful prompts, LoRA fine-tuning.
- **Source:** `02_2026.md`, goal 2.

---

## 8. Unclear — needs the user's call

1. **Is v1 the 1B text-and-audio model, or a step before it?** The annual goals (USER) name a ~1B text-and-audio model on 8 GB. BUILD_v0.md (AI) says "the 1B text-plus-audio model is the run after the architecture has won at 100M". Nothing says whether that run is v1. This decides where the paper, the safety work and the "beats a similar-size model" eval (4.2, 7.2) belong.
2. **Chat and autocomplete: v0 or v1?** The user's first message of 12 Nov 2026 asked for "a better pure text model, for chat, basic autocomplete, indexing, retrieval". DESIGN_v0.md limits the v0 gate to retrieval and puts chat and autocomplete "in the stage after that". The doc is AI-written.
3. **Stage 2 (decoder, whole-stack training, chat fine-tune): v0 or v1?** DESIGN_v0.md is titled v0 and describes stage 2 in section 4. BUILD_v0.md stops at the stage-1 run [7.2] and says stage 2 comes after. A Drive draft (AI) called the decoder step "v0.5".
4. **Latent-space reasoning.** The user listed it as left "for after" (12 Nov 2026, 11:38). The same day the user proposed the predictor loop, and it is now a v0 pilot switch (R=4, ticket ID1-303). Which part of latent reasoning the user meant as later is not clear. The loop is in v0. Dynamic stopping (1.4) and memory in the loop (1.5) are the candidates.
5. **Overlapping chunks.** The user proposed it on 12 Nov 2026 at 13:34, since chunk boundaries may turn into information breaks. DESIGN_v0.md marks it "Not adopted": overlap makes part of the target copyable. It says to measure whether boundaries hurt and, if so, overlap on the decoder side only. Not decided whether it is v1.
6. **KDA and GDN block extras: conv, SiLU, output norm, output gate, low-rank alpha, decay bias, shared q/k across value heads, per-value-head beta and alpha.** An AI agent said on 13 Nov 2026 (session `052df7ae`, 10:17 and 10:29) that each is its own small ablation "later", not part of ID1-280. The user said "we don't do the conv or swish" at 10:29. But the working-tree `MultiHeadGDN.py` already has `q_conv` and `k_conv`. So conv may be moving into v0. Which of these are v0, which are v1, and which are dropped is not stated. The agent also said multi-head state is H times smaller than one wide head, a real trade worth ablating.
7. **Kernel speed.** Ticket ID1-253 ("Make the blocks faster: optimise for faster training"), Backlog, v0 project. It quotes the user (11 Nov 2026): "I'll make them faster, optimise for training (if that exists, I have no clue tbh, I just want faster training)." It says to do it after the architecture is written and before finalising a starting arch (ID1-179). DESIGN_v0.md risk 1 says speed on a rented GPU is unmeasured and `flash-linear-attention` has kernels. Not stated whether this is v0 or v1.
8. **Study tickets in the v0 project with no version.** ID1-74 (study Qwen3-Next's Gated DeltaNet and attention hybrid, and "what it suggests testing for SA-JEPA"), ID1-278 (CALM, see 1.6). Both Backlog. Learning tasks, not model changes.
9. **Linear tickets ID1-120 and ID1-47** (7.1): are they v1 scope at all?
10. **W_o item (1.1):** the file in the working tree no longer has the factorised W_o. If the user already switched to full rank, the experiment is still the plan, but the baseline in step 1 is already in place.

---

## Checked and left out

- `eval_test/scripts/learning_testing/parallel_scan.py` line 11 says "For v0, do naively, for v1 can parallelise, for v2 ...". That is the learning script's own versioning, not the SA-JEPA v1 model.
- `latest_plan.md`, `manifesto.md` and the 15 Oct 2026 journal are audio-era and AI-written. The user said on 12 Nov 2026 not to over-index on that run, so nothing is taken from them.
- Archived Linear tickets (ID1-215, 217, 168, 178, 219, 254, 225, 176, 252, 251, 221, 224, 216 and others) are dead. None is cited.
- The Drive doc "SA-JEPA v0 - Text-First Architecture Plan (draft)" (id `17LrEXpB-sh8WVWwYqfv2SY5TrwiK1I5vS3Es9vLMed8`) has open decisions for v0 only. Nothing deferred to v1 was found in it.
- Not searched: Notion, Gmail. Past Claude Code transcripts in other project folders had no SA-JEPA mentions.
