# SA-JEPA v0 — build plan

*12 Nov 2026, re-sorted 14 Nov 2026. What has to be built for the design in [DESIGN_v0.md](DESIGN_v0.md), in order, and who does which part. The tickets are in Linear, project "SA-JEPA v0: SSM-based hybrid JEPA for text, audio": 29 work tickets under seven live parents. Each ticket now opens with its mode.*

## How the work is split (changed 14 Nov)

Until 14 Nov you wrote all model code by hand. After multi-head GDN you said it "felt fairly menial": the hard ideas had been learned single-head, and the head axis added typing, not understanding. You asked me to design the workflow instead of adopting your suggestion or mine from earlier that day. This is it. The short form is in `CLAUDE.md`.

**One rule: who writes a piece depends on the stage of the skill it needs.**

| Stage | Meaning | Who builds | Your part |
|---|---|---|---|
| 1, new | You have never done it | You, the smallest version, by hand | All of it. Scaffold and tests from me, no hints |
| 2, done once | You have built one like it | An agent | Spec before, check after |
| 3, proven | You have passed checks on it without a miss | An agent | Look at the result |

- A missed check moves the skill down one stage.
- A piece that **fails silently** never goes past stage 2. Wrong code there still trains and still draws a nice curve, so you cannot "jump in when it breaks": you would not know. `SIGREG.py` is the example in this repo: four defects, and it ran.
- **Plumbing** that is not about the model and fails loudly (shard files, logging, checkpoint files) starts at stage 3.

Why this and not a fixed split. A fixed "you design, agents code" split is wrong for a beginner: you cannot spec or check what you have never built once. A fixed "you code everything" split is what made multi-head KDA a wasted afternoon. The stage rule gives both: mechanical work is yours the first time and never again. Head axes are the case in point: GDN by hand was stage 1, KDA is stage 2, and Mamba later would be stage 3.

**Three rules that keep it honest.**

- **The constraint.** Given 100 to 200 times the time an agent took, you could do the same work without AI help beyond autocomplete. This is measured below, piece by piece, not assumed.
- **The oracle rule.** Every agent-built piece ships with a slow version that is obviously correct and short enough to read in one sitting, and a test that the fast version equals it. Your duty to understand attaches to the slow version and the invariant.
- **The queue rule.** Agents may build ahead of you, but nothing gets a paid run until its check has passed. Proposed cap, yours to change: no more than four built pieces waiting for a check at once. Past that the agents stop and you catch up.

**What every ticket tells you.** Each ticket opens with the same block, in plain words: what you should be able to do after it, why you need that, your exact steps in order (files, commands, and the words to say to Claude), and what you do not do. Go into a ticket by reading the first two parts; they are what you are there to get. The older "What", "Learn first" and "Done when" sections sit below as reference, and "Done when" is the gate for the finished piece, whoever builds it.

**Where things go.**

| What | Where |
|---|---|
| Your invariants and written predictions | in the ticket, under "Your invariants" |
| Model code | `v0_model_try/model/` (`block.py`, `pooling.py`, `chunk_encoder.py`, `predictor.py`, `predictor_loop.py`, `probe_decoder.py`) |
| Losses and the loop | `v0_model_try/training/` (`SIGREG.py`, `losses.py`, `train_stage1.py`) |
| Tokenizer, chunker, collation | `v0_model_try/data/` |
| Tests | `eval_test/tests/` |
| Checks | `eval_test/checks/<name>/QUIZ.md` |
| Small experiments by hand | `learn.ipynb` |

The file names are mine, chosen so the tickets can be exact. The audio-era files in `v0_model_try/model/` (`encoder.py`, `patch_layer.py`) stay untouched; `predictor.py` there is an 11-line stub that [2.3] replaces.

**The words that start each kind of work.** "scaffold and tests for [x]" before a hand-build. "build [x] from my invariants" for an agent build. "make the check for [x]" after it. "do [x]" for delegated work. "review [x]" when a hand-build passes its tests.

## Skill ledger

The stages are my reading of the repo and its history. Correct any I have wrong; you know what you have built before better than git does.

| Skill | Stage | Evidence | Where it comes up next |
|---|---|---|---|
| Attention: MHA, GQA, SWA | 2 | hand-built (ID1-167) | attention inside [2.2]–[2.4]: agent |
| Delta rule and the chunked form | 2 | hand-built single-head DN, GDN, KDA | – |
| Checking gradients against an oracle | 2 | [1.1] | every agent-built layer: agent |
| A head axis on a layer | 2 | multi-head GDN, 14 Nov | KDA heads [1.2]: agent, you check |
| Carrying state between calls | 2 | your chunk loop already carries `S_in` | [1.3], [1.4]: agent, you check |
| Residual block, pre-norm stack | 1 | none found | [2.1]: you write the block |
| Padding masks and masked pooling | 1 | none found | [2.2] pooling, [3.3] one toy batch |
| A distribution loss (SIGReg) | 1 | first attempt has four defects | [4.1] |
| Weighted and paired losses | 1 | none | [4.2], [4.5] |
| A training step: optimiser, schedule, clipping | 1 | the October run was AI-led | [4.3] |
| Cutting a gradient and proving it is cut | 1 | none | [4.4]: you write the test |
| Looping a block on its own output | 1 | new idea | [2.5] |
| Using a tokenizer | 1 | none found | [3.1]: round-trip a paragraph |
| Starting, stopping and pricing a cloud run | 1 | none | [6.1] |
| Designing a readout and its controls | 1 | none | [5.1]–[5.3] |
| Designing an experiment that can fail | 1 | none | [7.1] |
| Shards, logging, checkpoint files | 3 | plumbing | agent |

Silent-failure skills in this list: the three loss rows, masks, the gradient cut, readouts, experiment design. They stop at stage 2 however often you pass.

## The constraint, measured (14 Nov)

**AI time.** Everything in `layers/SSMs/reference/`, `layers/Attention_layers/reference/` and their tests (3,568 lines) was built on 13 Nov: the GDN reference in one session, and the rest in one evening session with subagents working in parallel between about 22:56 and 23:26. Summed subagent time from the transcripts is about 26 minutes. Per-piece times below are from those transcripts and are rough.

**Your time.** The only measured hand-build of the same kind is multi-head GDN: first commit 13 Nov 16:25, tests passing 14 Nov 14:35. Git does not show how many of those hours you worked. Your session activity across both days is about five hours in total, on everything. **Fill in your real number**; the ratio below assumes 3 to 6 hours.

| Piece | AI time | Your budget (×100 to ×200) | Evidence that you could | Holds? |
|---|---|---|---|---|
| GDN ops and layer, with heads | ~10 min | 17 to 33 h | you built it by hand in an assumed 3 to 6 h, about 20 to 40 times the AI's time | **yes** |
| KDA ops (plain path) and layer, with heads | under 10 min | up to 33 h | you built single-head chunked KDA; heads are the GDN skill again | yes, once the check is passed |
| Carried state and streaming | part of the above | – | your own chunk loop carries state | yes, once checked |
| Attention reference: GQA, MLA, RoPE, KV cache | ~14 min | 23 to 47 h | you built GQA and SWA; MLA and the cache are new to you | probably; MLA is unproven |
| Divide-and-conquer Gram with a hand-written backward | under 10 min | up to 33 h | nothing comparable | **no** |
| Metal kernels | ~2 min | 3 to 7 h | none; a language you have not used | **no** |

**What follows from the two failures.** Neither piece is in the model. `chunk_kda` now has `gram="naive"`: the definition written out, the same way your own `parallel_kda.py` builds it. It is tested against the token-by-token op for outputs and gradients (72 tests in `test_kda_ops.py` pass). v0 configs set `gram="naive"`. The faster paths come back only if [6.3] shows the speed matters, and then only after you have learned the trick and passed a check on it. The Metal step still switches itself on for no-gradient decode on the Mac; stage 1 has no decode, so it does not touch training.

**Going forward.** Each agent-built piece gets a row here when it is built: the agent's time, your budget, and the evidence. Each check records how long it took you. A piece whose row says "no" does not enter a run.

## The tree, in build order

"You" is your time in points on your scale (0, 1, 2, 4, 8, 16), and is what the Linear estimate now holds for tickets assigned to you. "Was" is the 12 Nov estimate, when you were to write the code.

```
                                                         mode             you  was
1  Layers ready ............................ ID1-265
   [1.1] Verify KDA and GDN gradients ...... ID1-279     done              –    4
   [1.2] Multi-head KDA and GDN ............ ID1-280     spec and check    1    8   ← the check is ready
   [1.3] Carry state across calls .......... ID1-281     spec and check    1    2   ← the check is ready (with [1.4])
   [1.4] Single-step path for generation ... ID1-282     spec and check    1    2

2  Model parts ............................. ID1-148
   [2.1] Residual block and layout string .. ID1-283     spec and check    1    2   first time: you write the block
   [2.2] Chunk encoder ..................... ID1-284     spec and check    2    4   first time: pooling; you write the leak test
   [2.3] Predictor ......................... ID1-285     spec and check    2    4   you write the causality test
   [2.4] Probe decoder ..................... ID1-286     spec and check    1    4
   [2.5] Predictor loop .................... ID1-303     HAND-BUILD        4    4   your idea, no reference

3  Data .................................... ID1-308
   [3.1] Tokenizer and reserved IDs ........ ID1-287     delegate          1    1   you choose; first time: round-trip a paragraph
   [3.2] Chunker ........................... ID1-288     spec and check    2    4   the rule is yours
   [3.3] Collation ......................... ID1-289     delegate          1    2   first time: one toy batch by hand
   [3.4] Corpus shards ..................... ID1-290     delegate          0    –   you choose the dataset
   [3.5] Paired text, good views ........... ID1-306     delegate          0    –   you choose the sources

4  Training loop ........................... ID1-218
   [4.1] Rewrite SIGReg .................... ID1-291     HAND-BUILD        4    4   ← your next hand-build
   [4.2] Weighted prediction loss .......... ID1-292     HAND-BUILD        1    1
   [4.3] Stage-1 loop, overfit one batch ... ID1-293     HAND-BUILD        4    8   step, optimiser, schedule; agent adds checkpoints
   [4.4] Detached probe .................... ID1-294     spec and check    1    2   you write the no-gradient test
   [4.5] Good-view agreement loss .......... ID1-304     HAND-BUILD        2    2
   [4.6] Experiment switches ............... ID1-305     spec and check    1    4

5  Readouts ................................ ID1-309
   [5.1] Floors and core readouts .......... ID1-295     spec and check    1    –   you pick the floors, read the table
   [5.2] Cue-recall test ................... ID1-296     spec and check    2    –   you design the test and its controls
   [5.3] Usefulness test ................... ID1-297     spec and check    1    –   you judge it on your own notes
   [5.4] Latent store ...................... ID1-307     delegate          0    –

6  Modal ................................... ID1-177
   [6.1] Modal basics ...................... ID1-298     FIRST TIME        1    2   do it yourself once
   [6.2] 5M smoke model on Modal ........... ID1-299     delegate          0    4
   [6.3] Throughput, cost, run lengths ..... ID1-300     delegate          1    2   you set the spend cap
   
7  Runs .................................... ID1-310
   [7.1] 25M pilot, five arms + floors ..... ID1-301     YOURS ALONE       8    8
   [7.2] 100M stage-1 run .................. ID1-302     YOURS ALONE       8    8   ← the end of this tree
```

**Your remaining time: 52 points, against 82 on the old split.** Of the 52, 15 are hand-builds, 16 are the two runs, and 21 are specs, checks, first-time parts and choices. These are estimates, not measurements; the first few checks will show whether a check really costs one point.

Readouts moved toward you, not away. On 12 Nov they were mine end to end. They are the measuring instruments, and the claim that this model is easier to interpret rests on them, so their design and their reading are yours even though an agent types them.

## Order: two tracks

You and the agents no longer work on the same ticket at the same time.

```
Your track:    [1.2 check] → [4.1] → [4.2] → [4.5] → specs for [2.1]–[2.4] → [4.3 bare step] → [2.5] → [7.1] → [7.2]
                     choices, early: tokenizer [3.1], chunk rule [3.2], dataset [3.4], pair sources [3.5]

Agent track:   [1.3], [1.4] exist → [3.1], [3.3] → [2.1]–[2.4] as each spec lands → [3.4], [3.5] → [4.3 plumbing],
               [4.4], [4.6] → [5.x] → [6.1]–[6.3]
                     each one stops in the check queue until you pass it
```

The tracks meet at [4.3], the first time one batch trains end to end. That step is yours.

The choices come early because they are the only things that block the agent track, and each takes minutes, not days. You are away 20 Nov to 4 Dec. Agents cannot choose a dataset or approve spend for you, so whatever is undecided on 20 Nov waits until you are back.

## Which layers the model uses

The model imports the reference layers in `layers/SSMs/reference/` and `layers/Attention_layers/reference/`, once you have passed their checks. They are AI-written (13 Nov), have heads, carried state and token-by-token streaming, and 118 tests pass. Your own files in `layers/SSMs/single_headed/` and `multi_headed/` stay as the record of what you built by hand: the delta rule, the chunked form, the gradient checks and multi-head GDN.

Two things to know before trusting them:

- The KDA check found a gap. No test checks that heads are independent or that keys are unit length per head. A layer with that bug passes all 31 layer tests. Part 4 of the check is about exactly this; the missing test gets added to the main suite once you have found it.
- The reference KDA is more elaborate than the design needs: a divide-and-conquer Gram, a hand-written backward, Metal kernels. Two of those fail the constraint (see "The constraint, measured"), so v0 runs on `gram="naive"`.

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

- `layers/` — your KDA, GDN (single-head, and multi-head GDN), GQA, SWA and plain attention; and the reference layers described above.
- `eval_test/checks/kda_multihead/` — the check that closes [1.2].
- `eval_test/checks/state_streaming/` — the check that closes [1.3] and [1.4].
- `eval_test/readouts/metrics.py` with tests — the tensor-level part of [5.1]: retrieval rank, effective rank, variance explained per dimension, and the copy-previous floor.
- `v0_model_try/training/SIGREG.py` — has four defects, listed in [4.1].
- `v0_model_try/model/encoder.py`, `patch_layer.py`, `config.py` — audio-era; they do not carry over.

## Deliberately not ticketed yet

Stage 2 (training the decoder with the whole stack), the chat fine-tune, the later ideas in section 7 of the design doc, and on-device measurements. They come after [7.2].

## Archived tickets this tree replaces

ID1-215, ID1-217, ID1-168, ID1-178, ID1-219, ID1-254 and ID1-225 were archived before this tree was made and stay archived. Their work is covered by phases 3, 4, 5 and 7.
