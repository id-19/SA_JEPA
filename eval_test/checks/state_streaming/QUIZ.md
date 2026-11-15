# Check for [1.3] and [1.4]: carried state and one-token steps (ID1-281, ID1-282)

*14 Nov 2026. Second check in the new workflow. It replaces hand-building `forward(x, state)` and the step path. About an hour.*

You did not write the state-carrying code. An agent did, inside `layers/SSMs/reference/gated_deltanet.py` and `kda.py`. This check is how you find out whether you understand it well enough to own it.

**What you should be able to do after this**
Say exactly what a memory layer hands over between calls, how big that is, and why it must not grow with the text. Say what "reading in pieces equals reading in one go" means for outputs and for gradients, and how close "equal" is in float32. Say which code runs when the model generates one token at a time.

**Why you need it**
Generation, long documents and save-and-resume all run the model in pieces. Training runs it on whole sequences. If the two disagree, the model you trained is not the model you run, and nothing crashes. The size of the hand-over is also the model's memory cost on a laptop or phone.

**What you do**
The four parts below, in order. Each part says exactly what to write and what to run.

**Before you start, two facts the tickets got wrong.** The agent did not build the interface the tickets asked for:

- It is `forward(x, state=None, return_state=False)`, which returns `y`, or `(y, state)` when `return_state=True`. It is not `forward(x, state) -> (y, state)`.
- There is no `step()`. One-token generation is `forward(x_t, state, return_state=True)` with a piece one token long. Q5 is about what that does.

Whether that is acceptable is your call, in Q6.

**Rules**

- Do the parts in order. Write each answer in this file before you look anything up or run anything.
- A prediction is a number, a list of positions or a yes/no, not a paragraph. Wrong predictions are the useful ones; leave them in.
- `ANSWER_KEY.md` stays closed until all four parts are done.
- Note your start and end time here, so the cost of a check is measured, not guessed. Start: ____  End: ____
- To grade: open a fresh Fable or Opus session, give it this file and `ANSWER_KEY.md`, and ask it to mark strictly.

**Passing**

- Part 2: Q2 and Q4 all right. Q3: at most one row wrong in the "exactly equal" column.
- Part 4: a test that fails on the bugged layer and passes on the reference.
- If you miss, do what [1.3] says: add `forward(x, state=None) -> (y, state)` by hand to your own `layers/SSMs/multi_headed/parallel_gdn.py`, with the conv tails, and make your own test from part 4 pass on it.

---

## Part 1. The spec (before opening the reference code)

Copy your [1.3] invariants from the ticket here. That's all. Part 1 is not graded.

An invariant is just "a thing that must always be true, written so a test can check it". Example: "two calls with the carried state give the same output as one call, to within 1e-6."

Your [1.3] invariants:

1.
2.
3.

---

## Part 2. Predictions

Read `CausalDepthwiseConv` and `GatedDeltaNetRef.forward` in `layers/SSMs/reference/gated_deltanet.py` (about 70 lines), and `recurrent_gated_delta_rule` in `layers/SSMs/reference/gated_delta_rule.py` (about 30 lines). Then answer. Run nothing yet.

**Q1. Size of the hand-over.** `GatedDeltaNetRef(d_model=512, n_heads=4)`, conv kernel 4. Floats per batch element in the returned state:

| | after T = 100 | after T = 10,000 |
|---|---|---|
| `S` | | |
| `conv_q` + `conv_k` + `conv_v` | | |
| total | | |

A `GQARef(512, 8 heads, 2 kv heads)` layer (head dim 64) caches its keys and values for every past token. After how many tokens does its cache hold more floats than the GDN state? ____

**Q2. The conv tails.**

a. A stream is fed 5 tokens, then 1 token. After the second call, `conv_k` holds the projected inputs of which positions (0-indexed)? ____

b. A 60-token sequence is split at 30. The second call gets only `{"S": state["S"]}`, with the conv tails dropped. (You worked out in chat that every token from 30 on is wrong.) How big is the error at token 59, the last one: about 1e-6, 1e-3 or 1e-1? ____

A test compares only token 59 with the one-call output. Does it catch this bug with `atol=1e-5`? With `atol=0.1`? ____ / ____

**Q3. How equal is equal.** One call on T = 1024 (float32, `chunk_size` 64, `mode="auto"`) against the same sequence fed in pieces with the state carried. For each split, are the outputs and final `S` **exactly** equal (difference 0.0)? If not, roughly how far apart: 1e-7, 1e-5 or 1e-3?

| Pieces | exactly equal? | if not, about |
|---|---|---|
| 512 + 512 | | |
| 1000 + 24 | | |
| 17 + 1 + 1 + 1005 | | |
| 1024 pieces of 1 token | | |

**Q4. Gradients across the hand-over.** Small GDN layer, 60 tokens, split at 30.

a. State passed as returned, loss on both pieces. Are the gradients for `x` and for every parameter equal to the one-call gradients, up to float error? yes / no

b. `state["S"]` is detached before the second call; the conv tails are not. Loss on the **second piece only**. Which tokens of the first piece (0 to 29) get a nonzero gradient? ____

c. `S` and all three conv tails are detached. Same loss. Which tokens of the first piece get a nonzero gradient? ____

**Q5. The one-token path.**

a. With `mode="auto"`, which op runs for a piece of T tokens?

| T | 1 | 8 | 9 | 64 |
|---|---|---|---|---|
| `chunk_gated_delta_rule` or `recurrent_gated_delta_rule` | | | | |

b. Force `mode="chunk"` and feed 60 tokens one at a time. Does the output still match one call, up to float error? yes / no

c. On the Mac (MPS), a one-token piece under `torch.no_grad()`. Which function computes the state update? And with gradients on?

**Q6. Judgment (no right answer, not graded pass/fail).**

a. The interface is `return_state=True` instead of always returning `(y, state)`, and there is no `step()`. Accept it, or have it changed? One line why.

b. On the Mac, generation without gradients runs a Metal kernel that failed the 100x to 200x constraint (BUILD_v0.md, "The constraint, measured"), and training never runs it. Is that acceptable for v0? What would you require before trusting text generated on the Mac?

**Now run it:**

```bash
uv run python eval_test/checks/state_streaming/reveal.py
```

Write next to each prediction whether it held. For any that did not, write what you had wrong.

---

## Part 3. Your spec against the tests

Open `eval_test/tests/test_gdn_reference.py`, `eval_test/tests/test_kda_layer.py` and `eval_test/tests/test_hybrid_streaming.py`. For each invariant you wrote in part 1, [1.3] and [1.4], name the test that checks it, or write "not tested". Then do the same for the two Done-when lists in tickets ID1-281 and ID1-282.

| Invariant or Done-when item | Test that checks it |
|---|---|
| | |

---

## Part 4. Bug hunt

`eval_test/checks/state_streaming/gdn_layer_bugged.py` is a copy of the reference GDN layer with one planted bug.

What you are told:

- The docstring at the top of the file is the spec, and it is correct. The code departs from it in one place.
- All 42 tests in `test_gdn_reference.py` and `test_hybrid_streaming.py` pass with the bugged layer swapped in. This was checked.
- Every output of the bugged layer equals the reference's, for any split. Nothing crashes and nothing goes to NaN.
- Do not diff the file against `layers/SSMs/reference/gated_deltanet.py`.

Done when you have:

1. The line, and one sentence on what it does, and to which training setup it would matter.
2. A test, in `eval_test/checks/state_streaming/test_found_it.py`, that fails on `GatedDeltaNetBugged` and passes on `GatedDeltaNetRef`. You may state the invariant in words and have an agent write the test. Stating the invariant is the work.

Your line and sentence:
