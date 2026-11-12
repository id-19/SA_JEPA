# SA-JEPA v0 — design

**Version 1.1, 12 Nov 2026. Tentative.** This is the one design doc. The build plan and ticket tree are in [BUILD_v0.md](BUILD_v0.md). It will change as we build; changes go in the log at the bottom.

**[H]** marks a hypothesis nobody has tested. Results from other people's papers are cited under Sources.

## 1. What we are building

A text model of about 100M parameters that runs on an 8 GB laptop and is useful for something real. It is the proof of concept for the model behind HIM, and the first step toward the 1B text-and-audio model in the 2026 goals.

**The bar for "useful":** a question asked over your own notes returns the right passages, scored next to BM25 and an off-the-shelf embedding model of the same size. Chat and autocomplete come in the stage after that.

**The two convictions it tests:**

1. SSM + attention hybrids do the job of attention for less memory and compute.
2. Predicting meaning, not tokens, is the right training objective.

**The resource bet behind the design:** on a laptop, RAM is scarce, compute is tight, and disk is nearly free. So parameters are for thinking, disk is for knowing, and tokens are only for speaking. The demo to aim at is the thing this allows that bigger models cannot do: a plain-language assistant with a long memory and a library on disk, in 8 GB.

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
| **Chunker** | A rule: a chunk ends at sentence-final punctuation or a newline; a sentence over 64 tokens splits at a clause; a fragment under 4 tokens merges. Same function in training and generation | you (variable chunks); the rule is mine |
| **Encoder** | Bidirectional attention over one chunk, pooled to one latent `z`. It never sees another chunk, so `z` can be computed once and stored | me |
| **Latent** | Ordered: the first 64 dimensions carry meaning, the rest carry wording and style. Kept Gaussian by SIGReg | you (Matryoshka, SIGReg) |
| **Predictor** | The hybrid stack over chunk latents; predicts the next chunk's latent. Its layer group can run R times, each pass refining the previous prediction. R=1 is the plain predictor | hybrid: you. Loop: your variant of my idea |
| **Decoder** | A small causal transformer over one chunk's tokens. Its only input is a latent. Trained after the latents are formed | you |
| **Store** | Chunk latents on disk at one byte per dimension, searched exactly. One million chunks at 64 dimensions is 64 MB | me |

Starting sizes for the ~100M model: tied token embedding 14M, encoder 15M (6 layers, width 448), predictor 59M (12 layers, width 640), decoder 10M (4 layers, width 448), latent width 512. A starting point, to be changed freely.

## 4. Training

**Stage 1: form the latents.** Encoder and predictor only. No token loss.

| Loss | What it compares | What it is for |
|---|---|---|
| Prediction | `ẑ` for the next chunk against that chunk's `z`, weighted more on early dimensions | teaches the predictor to move through the space; the weights sort the latent by predictability |
| Good-view agreement | the first 64 dimensions of two different texts that say the same thing | defines "meaning" as what survives rewording |
| SIGReg | all `z` in the batch against a Gaussian | stops collapse; no EMA, no stop-gradient |

A probe decoder trains alongside with its gradient cut. It measures what the latents hold and cannot shape them.

**Stage 2: learn to speak.** First the decoder trains on frozen latents, because a randomly initialised decoder sends large, meaningless gradients. Then the whole stack trains with the token loss added and the stage-1 losses still on, so the space stays Gaussian and predictable while it adapts. The stage-1 retrieval readouts are run again afterwards to measure how far the space drifted.

**After stage 2:** chat fine-tune. Then, your idea, hand the weights to a model like me to edit directly, judged by before-and-after readouts. I know of no evidence that this works. The edits I would trust first are the closed-form ones: folding an exact whitening into the latent projection, re-solving a linear head by least squares, calibrating a temperature.

## 5. Why each choice

**Chunks, not tokens.** A latent one token ahead is nearly that token's embedding, so predicting it duplicates ordinary language modelling. A sentence's latent has to summarise.

**Chunks follow meaning.** A fixed window ends mid-clause, so its latent would hold the tail of one thought and the head of the next. A rule is used instead of learned boundaries because it depends only on the tokens: training and generation agree with nothing extra to learn.

**The encoder sees one chunk only.** Otherwise it could make the target a function of the past: trivially predictable, and still varied enough to pass SIGReg.

**Latent-first.** Cross-entropy is a far denser signal than a latent loss. Trained together from scratch, it would shape the latents first and we could never tell which loss made them useful.

**The Matryoshka latent is one weight vector.** SIGReg does not care how the latent is rotated; a prediction error weighted more on early dimensions does. To minimise it the encoder must put its most predictable features where the weights are largest, so the latent sorts itself. One honest gap: in stage 1 nothing teaches the tail what to hold, because nothing yet asks for wording. It gets real content in stage 2.

**MSE and SIGReg, no contrastive loss.** MSE gives the average of the plausible next latents. For retrieval that average is the best single query point. Because SIGReg makes targets unit-variance Gaussian, the length of `ẑ` reads as confidence: long means one clear continuation, short means many. **[H]** A contrastive loss needs negatives, a temperature and large batches. One caveat on the evidence: LeJEPA validated SIGReg on two views of an image, not on next-chunk text.

**Good views.** Balestriero and LeCun prove that JEPAs trained on two observations of the same thing learn the same representation whatever the modality, up to one rotation. Their 109M text JEPA aligned with image models better than language models up to 72B, with 42 times fewer samples than next-token prediction. That is our size, and it is what would make adding audio later a rotation, not a rebuild. They also argue that a prefix and its continuation share the author's wording and style, so a model trained only on that pair spends latent directions on style. Our prediction loss is that kind of pair at sentence scale. In their closest experiment, a JEPA with latent targets but views cut from one caption aligned about as poorly as next-token prediction (.047 and .042 against .062 for good views). Whether the argument carries over to next-sentence prediction is my inference, and the paper itself calls even the next-token case a prediction to test. So good views are a switch to test, not an assumption. A typo-damaged copy of a chunk is not a good view: it only removes what the damage touches. Typos are useful as input noise for robustness.

**The loop.** The predictor runs once per chunk, roughly once per twenty tokens, so depth is almost free there. Looping on the prediction itself (your variant) keeps the loop variable a real latent: every pass can be scored, probed or used as a search query, and "stop when it stops moving" needs nothing extra. Geiping et al. show a looped language model improving with more passes at test time. Fixed R comes first. Stopping early is awkward in a sequence model, because a later chunk may need a memory state that an earlier chunk never computed, so it is built only if most chunks turn out to settle early.

**Hybrid layout.** Kimi Linear's own ablation picked 3 KDA layers to 1 attention layer over pure attention, and reports 75% less cache. That is at 48B parameters; nothing shows it at 100M. Since the predictor works on chunks, training sequences must be long in chunks, around a thousand, for the hybrid to matter. **[H]** What v0 can honestly show: equal or better quality than an attention-only predictor at matched parameters, memory that stays nearly flat as context grows, and no cliff past the training length. By design arithmetic, 100k tokens of context costs about 40 MB of attention cache here against about 3.7 GB for a dense 12-layer transformer.

**Decoder on the latent alone.** One channel: if the decoder needs something, the encoder and predictor must learn to put it in the latent. The risk is exact wording across a sentence boundary, such as spelling a name the same way twice. The fallback is to let the decoder see the last few tokens of the previous chunk.

**Simple beats clever.** Every addition is a switch on the base design, off by default, judged alone.

## 6. What is in, what is a switch, what is later

| Piece | Status for the first runs |
|---|---|
| Chunker, encoder, predictor, prediction loss, SIGReg, detached probe | **Base.** Always on |
| Good-view agreement loss | **Switch.** Pilot arm |
| Predictor loop, R=4 | **Switch.** Pilot arm |
| SIGReg on the predictions as well | **Switch.** Pilot arm |
| Prediction loss allowed to push on the encoder | **Switch.** On by default; tested off once good views exist |
| Latent store used as an index | **Built for the usefulness test.** No training change |
| Everything in section 7 | **Later** |
| Overlapping chunks | **Not adopted.** Overlap makes part of the target copyable, and the weighting would put the shared tokens first. Measure whether boundaries hurt; if they do, overlap on the decoder side only |

## 7. Later, with the argument for each

**Memory inside the loop.** Stored latents are fetched and fed to the predictor as extra chunks, tagged as memories. With the loop it becomes think → recall → think: one pass's guess is the query for what the next pass sees. A chunk is written to the store only if the predictor got it badly wrong, so the store holds what the weights lack. RETRO matched GPT-3 with 25 times fewer parameters this way, and the gain held from 150M to 7B. The predictor leaves an unused argument for it now.

**Choosing a thought instead of averaging.** Give the predictor head a noise input and draw several predictions for one context; pull them toward the target and push them apart from each other. This is SIGReg turned sideways: SIGReg stops every input collapsing to one latent, across the batch; this stops every prediction collapsing to one average, within a context. Neither covers the other. It should be the same function with a different reference, which is why the SIGReg rewrite is written against a reference. CALM trains a next-vector language model this way. The cheap half-step, SIGReg on the predictions with no sampling, is already a pilot switch.

**Bytes → words → thoughts.** Drop the tokenizer: a tiny byte-level model emits one vector per word, the encoder reads word vectors, and a tiny byte model spells the output. It removes the embedding table and the 32,000-way output layer, makes a typo a nearby input instead of a different token, suits romanized Hindi where spelling is not standardised, and is the route to audio as one more bottom level. H-Net and the Byte Latent Transformer match tokenized models at a billion parameters or more; nobody has shown it at 100M. It comes last because a bug at the bottom hides everything above it.

**Further control arms for stage 2.** The same architecture trained on cross-entropy from scratch (does latent pretraining earn its place for generation?), and an ordinary token-level hybrid language model at matched compute (what does the chunk design cost?).

## 8. The experiment that decides

A 25M pilot. Every arm differs from the base by one change and sees the same tokens.

| Arm | Change | Question it answers |
|---|---|---|
| Base | none | the control |
| Attention-only | no KDA layers in the predictor | does the hybrid earn its place? |
| Good views | agreement loss on | does it make the latents more semantic? |
| Loop | R=4 | does thinking longer help? |
| Prediction SIGReg | SIGReg on `ẑ` too | does it help to keep predictions full-length? |

Readouts for every arm, with floors from an untrained model and from "copy the previous latent": where the true next chunk ranks among distractors; how many latent dimensions the predictor explains; recall of a cue stated many chunks earlier, at and beyond the training length; and what the probe can reconstruct from a prefix. You write your predicted ordering down before the run. The 100M config is the base plus the switches that won.

## 9. Risks, most likely first **[H]**

1. **KDA and GDN too slow at scale.** A first check on small shapes shows the chunked backward already agrees with the recurrent reference to single precision. Speed on a rented GPU is unmeasured; `flash-linear-attention` has kernels if needed.
2. **A mean is a poor thing to speak from.** For a question with two opposite good answers, the average latent is neither. Seen as vague generations in stage 2. The fix is "choosing a thought" in section 7.
3. **Latent filler.** SIGReg forces every dimension to carry variance, but the prediction loss only shapes the dimensions the predictor can predict. The rest hold anything. Seen as the variance-explained curve dropping to zero early. Fix: a narrower latent.
4. **Wording does not survive the latent.** Seen as poor name-copy across a sentence boundary. Fix: the decoder fallback in section 5.
5. **The two pressures on the first 64 dimensions fight.** The weights want the most predictable features there; the agreement loss wants only what survives rewording.
6. **Stage 2 damages the semantic space.** Seen as the retrieval readouts falling. Fix: lower learning rate on encoder and predictor, or a longer frozen phase.

## 10. Open questions

1. Is the starting chunk rule good enough? Ticket [3.2] ends with you looking at real chunk lengths.
2. Is the prediction alone enough scratch space for the loop, or must the hidden state be carried between passes too?
3. Where does paired text come from at sentence scale, in enough volume? Ticket [3.5] starts by listing candidates for you to choose from.

## Sources

- Balestriero and LeCun, *LeJEPA* — https://arxiv.org/abs/2511.08544
- Balestriero and LeCun, *Proving and Improving the Platonic Representation Hypothesis with JEPAs* — https://arxiv.org/abs/2610.39302 (read through section 3.4 and appendix A)
- Moonshot AI, *Kimi Linear* — https://arxiv.org/abs/2510.26692 (ratio numbers read from a review, not the paper's own table)
- Geiping et al., *Scaling up Test-Time Compute with Latent Reasoning* — https://arxiv.org/abs/2502.05171
- Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks* — https://arxiv.org/abs/2510.04871
- Borgeaud et al., *RETRO* — https://arxiv.org/abs/2112.04426
- Shao et al., *Continuous Autoregressive Language Models* — https://arxiv.org/abs/2510.27688
- Ho et al., *Block Transformer* — https://arxiv.org/abs/2406.02657
- Hwang, Wang, Gu, *H-Net* — https://arxiv.org/abs/2507.07955
- Pagnoni et al., *Byte Latent Transformer* — https://arxiv.org/abs/2412.09871

## Change log

- **v1.1, 12 Nov 2026.** Coalesced. The longer reasoning from the agreed plan and the free-run doc is folded in (Matryoshka mechanism, MSE and confidence, the later ideas with their arguments, risks). The build plan moved to BUILD_v0.md. Four draft docs removed.
- **v1, 12 Nov 2026.** First version. Merges the agreed plan (chunks, latent-first, Matryoshka latent, decoder on the latent alone, detached probe) with good views, the predictor loop, SIGReg on predictions and the latent store.
