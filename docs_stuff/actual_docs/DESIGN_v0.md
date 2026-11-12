# SA-JEPA v0 — design

**Version 1, 12 Nov 2026. Tentative.** This is the one design doc. It merges the agreed plan with the ideas from the free-run doc that survived discussion. It will change as we build; changes go in the log at the bottom.

**[H]** marks a hypothesis nobody has tested. Results from other people's papers are cited under Sources.

## 1. What we are building

A text model of about 100M parameters that runs on an 8 GB laptop and is useful for something real. It is the proof of concept for the model behind HIM, and the first step toward the 1B text-and-audio model in the 2026 goals.

**The bar for "useful":** a question asked over your own notes returns the right passages, scored next to BM25 and an off-the-shelf embedding model of the same size. Chat and autocomplete come in the stage after that.

**The two convictions it tests:**

1. SSM + attention hybrids do the job of attention for less memory and compute.
2. Predicting meaning, not tokens, is the right training objective.

**The resource bet behind the design:** on a laptop, RAM is scarce, compute is tight, and disk is nearly free. So parameters are for thinking, disk is for knowing, and tokens are only for speaking.

## 2. The design in one picture

```
text ─► CHUNKER      cuts at sentence and clause boundaries
          │
          ▼
        ENCODER      sees one chunk only ─► one latent z
          │            first dimensions = meaning, later ones = wording
          │
          ├──────────► STORE on disk: a short prefix of every z ever read
          ▼
        PREDICTOR    causal over chunk latents, 3 KDA : 1 attention
          │  ▲         may loop: each pass sees its own previous prediction
          │  └──┘
          ▼
        ẑ            predicted latent of the next chunk
          │
          ▼
        DECODER      small; writes the tokens of one chunk from the latent alone
```

## 3. The parts

| Part | What it does | Decided by |
|---|---|---|
| **Chunker** | A rule: a chunk ends at sentence-final punctuation or a newline; long sentences split at a clause; tiny fragments merge. Same function in training and generation | you (variable chunks); the rule is mine |
| **Encoder** | Bidirectional attention over one chunk, pooled to one latent `z`. It never sees another chunk, so `z` can be computed once and stored | me |
| **Latent** | Ordered: the first 64 dimensions carry meaning, the rest carry wording and style. Kept Gaussian by SIGReg | you (Matryoshka, SIGReg) |
| **Predictor** | The hybrid stack over chunk latents; predicts the next chunk's latent. Its layer group can run R times, each pass refining the previous prediction. R=1 is the plain predictor | hybrid: you. Loop: your variant of my idea |
| **Decoder** | A small causal transformer over one chunk's tokens. Its only input is a latent. Trained after the latents are formed | you |
| **Store** | Chunk latents on disk at one byte per dimension, searched exactly. One million chunks at 64 dimensions is 64 MB | me |

Starting sizes for the ~100M model: tied token embedding 14M, encoder 15M, predictor 59M (12 layers, width 640), decoder 10M. These are a starting point.

## 4. Training

**Stage 1: form the latents.** Encoder and predictor only. No token loss.

| Loss | What it compares | What it is for |
|---|---|---|
| Prediction | `ẑ` for the next chunk against that chunk's `z`, weighted more on early dimensions | teaches the predictor to move through the space; the weights sort the latent by predictability |
| Good-view agreement | the first 64 dimensions of two different texts that say the same thing | defines "meaning" as what survives rewording |
| SIGReg | all `z` in the batch against a Gaussian | stops collapse; no EMA, no stop-gradient |

A probe decoder trains alongside with its gradient cut. It measures what the latents hold and cannot shape them.

**Stage 2: learn to speak.** First the decoder trains on frozen latents. Then the whole stack trains with the token loss added and the stage-1 losses still on.

**After stage 2:** chat fine-tune; then, your idea, hand the weights to a model like me to edit directly, judged by before-and-after readouts.

## 5. Why each choice

**Chunks, not tokens.** A latent one token ahead is nearly that token's embedding, so predicting it duplicates ordinary language modelling. A sentence's latent has to summarise.

**Chunks follow meaning.** A fixed window ends mid-clause, so its latent would hold the tail of one thought and the head of the next.

**The encoder sees one chunk only.** Otherwise it could make the target a function of the past: trivially predictable, and still varied enough to pass SIGReg.

**Latent-first.** Cross-entropy is a far denser signal than a latent loss. Trained together from scratch, it would shape the latents first and we could never tell which loss made them useful.

**Good views.** Balestriero and LeCun prove that JEPAs trained on two observations of the same thing learn the same representation whatever the modality, up to one rotation. Their 109M text JEPA aligned with image models better than language models up to 72B, with 42 times fewer samples than next-token prediction. That is our size, and it is what makes adding audio later a rotation, not a rebuild. They also argue that a prefix and its continuation share the author's wording and style, so a model trained only on that pair spends latent directions on style. Our prediction loss is that kind of pair at sentence scale. Whether the argument carries over to sentences is my inference, which is why good views are a switch to test and not an assumption.

**The loop.** The predictor runs once per chunk, roughly once per twenty tokens, so depth is almost free there. Looping on the prediction itself (your variant) keeps the loop variable a real latent: every pass can be scored, probed or used as a search query, and "stop when it stops moving" needs nothing extra. Geiping et al. show a looped language model improving with more passes at test time.

**Hybrid layout.** Kimi Linear's own ablation picked 3 KDA layers to 1 attention layer over pure attention, and reports 75% less cache. That is at 48B parameters; nothing shows it at 100M. Since the predictor works on chunks, training sequences must be long in chunks, around a thousand, for the hybrid to matter. **[H]**

**Decoder on the latent alone.** One channel: if the decoder needs something, the encoder and predictor must learn to put it in the latent. The risk is exact wording across a sentence boundary, such as spelling a name the same way twice. The fallback is to let the decoder see the last few tokens of the previous chunk.

**Simple beats clever.** Every addition below is a switch on the base design, off by default, judged alone.

## 6. What is in, what is a switch, what is later

| Piece | Status for the first runs |
|---|---|
| Chunker, encoder, predictor, prediction loss, SIGReg, detached probe | **Base.** Always on |
| Good-view agreement loss | **Switch.** Pilot arm |
| Predictor loop, R=4 | **Switch.** Pilot arm |
| SIGReg on the predictions as well | **Switch.** Pilot arm. Costs nothing; stops predictions shrinking toward zero |
| Prediction loss allowed to push on the encoder | **Switch.** On by default; tested off once good views exist |
| Latent store used as an index | **Built for the usefulness test.** No training change |
| Stage 2, chat fine-tune | **Later**, after the 100M stage-1 run |
| Store latents fed to the predictor as extra chunks; write only what surprised the predictor | **Later.** The predictor leaves an unused argument for it now |
| Several predictions per context, so the model chooses a thought instead of averaging | **Later.** It is SIGReg pointed at a different reference; the loss is written to allow it |
| Bytes → words → thoughts, no tokenizer | **Later.** The endgame: typo-proof, right for romanized Hindi, and the route to audio |
| Overlapping chunks | **Not adopted.** Overlap makes the target copyable. Measure whether boundaries hurt first; if they do, overlap on the decoder side |

## 7. The experiment that decides

A 25M pilot. Every arm differs from the base by one change and sees the same tokens.

| Arm | Change | Question it answers |
|---|---|---|
| Base | none | the control |
| Attention-only | no KDA layers in the predictor | does the hybrid earn its place? |
| Good views | agreement loss on | does it make the latents more semantic? |
| Loop | R=4 | does thinking longer help? |
| Prediction SIGReg | SIGReg on `ẑ` too | does it help to keep predictions full-length? |

Readouts for every arm, with floors from an untrained model: where the true next chunk ranks among distractors; how many latent dimensions the predictor explains; recall of a cue stated many chunks earlier, at and beyond the training length; and what the probe can reconstruct from a prefix. You write your predicted ordering down before the run. The 100M config is the base plus the switches that won.

## 8. Build plan

Tickets are in Linear, nested under seven tickets that already existed. **You write all model, loss and training code. I do the plumbing that would only cost you time**: data shards, paired text, the store, and readouts that call your code without changing it.

| Phase | Tickets | Owner |
|---|---|---|
| 1. Layers | [1.1] gradient check · [1.2] multi-head · [1.3] carried state · [1.4] single step | you |
| 2. Model | [2.1] block · [2.2] encoder · [2.3] predictor · [2.4] probe decoder · [2.5] predictor loop · assembly (ID1-179) | you |
| 3. Data | [3.1] tokenizer · [3.2] chunker · [3.3] collation | you |
| | [3.4] corpus shards · [3.5] paired text | me |
| 4. Training | [4.1] SIGReg · [4.2] prediction loss · [4.3] stage-1 loop · [4.4] detached probe · [4.5] good-view loss · [4.6] switches | you |
| 5. Readouts | [5.1] metrics and floors · [5.2] cue recall · [5.3] usefulness test · [5.4] latent store | me |
| 6. Cloud | [6.1] Modal basics · [6.2] smoke model on Modal · [6.3] throughput and cost | you |
| 7. Runs | [7.1] 25M pilot · [7.2] 100M stage-1 run | you |

**Order.** [1.1] first: it checks the gradients everything trains through. [4.1], [4.2], [3.1] and [6.1] depend on nothing and can fill gaps. Phases 1 to 4 meet at [4.3], the first time one batch trains end to end. The loop, good-view loss and switches ([2.5], [4.5], [4.6]) come after that works, and before the pilot.

**Scale ladder.** 5M on the Mac to overfit and smoke-test, 25M on Modal for the pilot, about 100M on Modal for the run this plan ends in. Run lengths come from measured throughput ([6.3]), not from estimates.

## 9. Open questions

1. The chunk rule: is the starting rule good enough? [3.2] ends with you looking at real chunk lengths.
2. Do the two pressures on the first 64 dimensions fight? The prediction weights want the most predictable features there; the agreement loss wants only what survives rewording. **[H]**
3. Is the prediction alone enough scratch space for the loop, or must the hidden state be carried between passes too?
4. Where does paired text come from at sentence scale, in enough volume? [3.5] starts by listing candidates for you to choose from.

## Sources

- Balestriero and LeCun, *LeJEPA* — https://arxiv.org/abs/2511.08544
- Balestriero and LeCun, *Proving and Improving the Platonic Representation Hypothesis with JEPAs* — https://arxiv.org/abs/2610.39302
- Moonshot AI, *Kimi Linear* — https://arxiv.org/abs/2510.26692 (ratio numbers read from a review, not the paper's own table)
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning* — https://arxiv.org/abs/2502.05171
- Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks* — https://arxiv.org/abs/2510.04871
- Borgeaud et al., *RETRO* — https://arxiv.org/abs/2112.04426
- Shao et al., *Continuous Autoregressive Language Models* — https://arxiv.org/abs/2510.27688
- Ho et al., *Block Transformer* — https://arxiv.org/abs/2406.02657
- Hwang, Wang, Gu, *H-Net* — https://arxiv.org/abs/2507.07955
- Pagnoni et al., *Byte Latent Transformer* — https://arxiv.org/abs/2412.09871

## Change log

- **v1, 12 Nov 2026.** First version. Merges the agreed plan (chunks, latent-first, Matryoshka latent, decoder on the latent alone, detached probe) with good views, the predictor loop, SIGReg on predictions and the latent store.
