# SA-JEPA v0 — build plan

*12 Nov 2026. What has to be built for the design in [DESIGN_v0.md](DESIGN_v0.md), in order. The tickets are in Linear, project "SA-JEPA v0: SSM-based hybrid JEPA for text, audio": 29 work tickets under seven live parents. Each ticket has what to do, what to read first, done-when gates, and what it needs.*

**Who does what.** You write all model, loss and training code. I do the plumbing that would only cost you time: data shards, paired text, the latent store, and readouts that call your code without changing it. Dataset choices and any spend come to you first.

Numbers on the right are point estimates on your scale (1, 2, 4, 8, 16).

## The tree, in build order

```
1  Layers ready ............................ ID1-265  Fix up layers + Build out SWA        (under ID1-179)
   [1.1] Verify KDA and GDN gradients ...... ID1-279   4   ← start here
   [1.2] Multi-head KDA and GDN ............ ID1-280   8
   [1.3] Carry state across calls .......... ID1-281   2
   [1.4] Single-step path for generation ... ID1-282   2

2  Model parts ............................. ID1-148  Build out encoder/predictor backbones (under ID1-179)
   [2.1] Residual block and layout string .. ID1-283   2
   [2.2] Chunk encoder ..................... ID1-284   4
   [2.3] Predictor ......................... ID1-285   4
   [2.4] Probe decoder ..................... ID1-286   4
   [2.5] Predictor loop .................... ID1-303   4   (after [4.3] works)

3  Data .................................... ID1-308  [3] Data: tokenizer, chunker, shards
   [3.1] Tokenizer and reserved IDs ........ ID1-287   1
   [3.2] Chunker ........................... ID1-288   4
   [3.3] Collation ......................... ID1-289   2
   [3.4] Corpus shards (Fable) ............. ID1-290   2
   [3.5] Paired text, good views (Fable) ... ID1-306   2

4  Training loop ........................... ID1-218  Get the training loop set up
   [4.1] Rewrite SIGReg .................... ID1-291   4   ← alternate with 1.x
   [4.2] Weighted prediction loss .......... ID1-292   1
   [4.3] Stage-1 loop, overfit one batch ... ID1-293   8
   [4.4] Detached probe .................... ID1-294   2
   [4.5] Good-view agreement loss .......... ID1-304   2
   [4.6] Experiment switches ............... ID1-305   4

5  Readouts ................................ ID1-309  [5] Readouts and validation
   [5.1] Floors and core readouts (Fable) .. ID1-295   2   (metric functions written)
   [5.2] Cue-recall test (Fable) ........... ID1-296   2
   [5.3] Usefulness test (Fable) ........... ID1-297   2
   [5.4] Latent store (Fable) .............. ID1-307   2

6  Modal ................................... ID1-177  Modal cloud run
   [6.1] Modal basics ...................... ID1-298   2   ← no dependencies
   [6.2] 5M smoke model on Modal ........... ID1-299   4
   [6.3] Throughput, cost, run lengths ..... ID1-300   2

7  Runs .................................... ID1-310  [7] Pilot and 100M stage-1 run
   [7.1] 25M pilot, five arms + floors ..... ID1-301   8
   [7.2] 100M stage-1 run .................. ID1-302   8   ← the end of this tree
```

**Yours:** 23 tickets, 86 points. **Mine:** 6 tickets, 12 points, unassigned in Linear.

## What can run in parallel

```
[1.1] → [1.2] → [1.3] → [1.4] ─┐
                 [1.2] → [2.1] → [2.2], [2.3], [2.4] ─┐
[3.1] → [3.2] → [3.3] ────────────────────────────────┼→ [4.3] → [4.4] → [6.2] → [6.3] → [7.1] → [7.2]
[4.1], [4.2] ─────────────────────────────────────────┘      └→ [2.5], [4.5] → [4.6] ──────┘
[6.1] any time
```

[1.1] comes first because it checks the gradients everything trains through. [4.1], [4.2], [3.1] and [6.1] depend on nothing and can fill gaps. Phases 1 to 4 meet at [4.3], the first time one batch trains end to end. The loop, the good-view loss and the switches come after that works, and before the pilot.

## Scale

| Size | Where | Purpose |
|---|---|---|
| 5M | the Mac | overfit one batch, smoke test |
| 25M | Modal | the pilot: which switches go into the 100M run, and is it worth running |
| ~100M | Modal | the end of this tree |

Run lengths come from measured throughput ([6.3]), not from estimates.

Why 100M stays the finish line even though it was picked arbitrarily: it is the size with published reference points. Balestriero and LeCun's text JEPA is 109M, and "beats a similar model of its size", from the annual goals, needs a size where similar models exist. The 1B text-plus-audio model is the run after the architecture has won at 100M.

The finish gate on [7.2] is a usefulness test, not a loss curve: a query over your own notes returns sensible passages, with the score reported next to BM25 and a same-size embedding model.

## What already exists

- `layers/` — KDA, GDN, GQA, SWA and plain attention. KDA and GDN are single-head, start from a zero state and return none. That is phase 1.
- `eval_test/readouts/metrics.py` with tests — the tensor-level part of [5.1]: retrieval rank, effective rank, variance explained per dimension, and the copy-previous floor.
- `v0_model_try/training/SIGREG.py` — has four defects, listed in [4.1].
- `v0_model_try/model/encoder.py`, `patch_layer.py`, `config.py` — audio-era; they do not carry over.

## Deliberately not ticketed yet

Stage 2 (training the decoder with the whole stack), the chat fine-tune, the later ideas in section 7 of the design doc, and on-device measurements. They come after [7.2].

## Archived tickets this tree replaces

ID1-215, ID1-217, ID1-168, ID1-178, ID1-219, ID1-254 and ID1-225 were archived before this tree was made and stay archived. Their work is covered by phases 3, 4, 5 and 7.
