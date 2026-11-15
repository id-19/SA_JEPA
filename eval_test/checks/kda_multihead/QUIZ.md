# Check for [1.2]: multi-head KDA (ID1-280)

*14 Nov 2026. First check in the new workflow (see `CLAUDE.md`, "The workflow"). It replaces hand-building multi-head KDA. About an hour.*

You did not write the multi-head KDA layer. An agent did: `layers/SSMs/reference/kda.py` and `kda_ops.py`. This check is how you find out whether you understand it well enough to own it.

**What you should be able to do after this**
Say what splitting a memory into heads does to it: how big it is, what each head can see, and how many things it can remember. Spot a layer where heads leak into each other.

**Why you need it**
Every memory layer in the model has heads, and you will pick how many for the pilot. A multi-head layer with a mistake in it still trains and still gives a nice curve, so you have to be able to tell a broken one from a working one yourself.

**What you do**
The four parts below, in order. Each part says exactly what to write and what to run.

**Rules**

- Do the parts in order. Write each answer in this file before you look anything up or run anything.
- A prediction is a number or a yes/no, not a paragraph. Wrong predictions are the useful ones; leave them in.
- `ANSWER_KEY.md` stays closed until all four parts are done.
- Note your start and end time here, so the cost of a check is measured, not guessed. Start: ____  End: ____
- To grade: open a fresh Fable or Opus session, give it this file and `ANSWER_KEY.md`, and ask it to mark strictly.

**Passing**

- Part 2: Q1 to Q3 predicted right. Q4: you put the breaking point in the right column for every H.
- Part 4: a test that fails on the bugged layer and passes on the reference.
- If you miss Q4, write the capacity experiment yourself from scratch. If you miss part 4, hand-build multi-head KDA after all, in `layers/SSMs/multi_headed/parallel_kda.py`.

---

## Part 1. The spec (before opening the reference code)

List the invariants that would prove a multi-head KDA layer is right. One line each. Think of what you would demand before trusting someone else's layer in a training run.

Your invariants:

1.
2.
3.

(add more)

---

## Part 2. Predictions

Read `kda_step` in `layers/SSMs/reference/kda_ops.py` (14 lines) and `forward` in `layers/SSMs/reference/kda.py`. Then answer. Run nothing yet.

**Q1. State size.** `KimiDeltaAttentionRef(d_model=256, n_heads=H)`. How many floats are in the delta-rule state `S` for one batch element?

| H | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| floats | | | | |

**Q2. Who can see whom.**

a. At the op level (H=4, T=20), the key of head 2 is replaced at token 10. Which heads' outputs change, and at which tokens?

b. At the layer level, the input `x` is changed at token 10. Which tokens of the layer output `y` change? Do all four heads' contributions change, or one?

c. Without running anything: after the output projection, can what head 0 remembered affect a number that head 1 also affects? Before the output projection?

**Q3. KDA against GDN.**

a. Give KDA the same decay on every key channel. Is its output equal to GDN's with that scalar decay? Exactly, roughly, or not at all?

b. In GDN, the decay inside a chunk is a `(C, C)` table multiplied onto `K Kᵀ`. Why can KDA not do that? What shape is the object it needs if you build it the direct way?

**Q4. Capacity.** Keys and values are 64 wide in total, split across H heads (head h owns slice h of both). Write N pairs with no decay and full write strength, then read every key back. Keys are made as orthogonal as each head allows.

For each H, what is the largest N in the list at which recall is still exact?

| H | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| largest exact N, from 4, 8, 16, 32, 64, 128 | | | | |

One sentence: why does it break there?

Is there any way for a 4-head layer to recall more items than your number for H=4? What would it give up?

**Q5. Judgment (no right answer, not graded pass/fail).** Given your answer to Q4, what do heads buy? Would you want an arm in the pilot that settles it, and what would the arm be?

**Now run it:**

```bash
uv run python eval_test/checks/kda_multihead/reveal.py
```

Write next to each prediction whether it held. For any that did not, write what you had wrong.

---

## Part 3. Your spec against the tests

Open `eval_test/tests/test_kda_layer.py` and `eval_test/tests/test_kda_ops.py`. For each invariant you wrote in part 1, name the test that checks it, or write "not tested".

| Your invariant | Test that checks it |
|---|---|
| | |

---

## Part 4. Bug hunt

`eval_test/checks/kda_multihead/kda_layer_bugged.py` is a copy of the reference layer with one planted bug.

What you are told:

- The docstring at the top of the file is the spec, and it is correct. The code departs from it in one place.
- All 31 existing tests in `test_kda_layer.py` and `test_hybrid_streaming.py` pass against the bugged layer. This was checked.
- The model would train with it. Nothing crashes and nothing goes to NaN.
- Do not diff the file against `layers/SSMs/reference/kda.py`.

Done when you have:

1. The line, and one sentence on what it does to the memory.
2. A test, in `eval_test/checks/kda_multihead/test_found_it.py`, that fails on `KimiDeltaAttentionBugged` and passes on `KimiDeltaAttentionRef`. You may state the invariant in words and have an agent write the test. Stating the invariant is the work.

Your line and sentence:

