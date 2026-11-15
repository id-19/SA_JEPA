# Answer key for QUIZ.md — closed until all four parts are done

*For the grading session. Mark strictly: a prediction is right or wrong, and a vague answer is wrong. Numbers below come from `reveal.py`, run on 14 Nov 2026.*

## Part 1. Invariants a good answer contains

Not every one is needed. The first two are the ones that matter most.

1. **Heads are independent inside the op.** Changing head h's q, k, v, decay or write strength changes only head h's output. Equivalently: H heads equal H single-head modules holding the same weights.
2. **Each head's keys and queries have unit length**, taken over that head's own slice.
3. Chunked equals token-by-token, for outputs and for gradients, including a ragged last chunk.
4. Causal: token t's output does not depend on tokens after t.
5. Splitting a sequence and carrying the state gives the same output as one call.
6. With H=1 it equals the single-head layer.
7. The same decay on every channel reduces to GDN.

## Part 2

**Q1.** 65536, 32768, 16384, 8192. The state is H matrices of (d/H) by (d/H), so d²/H floats. More heads means a smaller memory.

**Q2.**
a. Only head 2, tokens 10 to 19. Nothing in heads 0, 1, 3.
b. Tokens 10 to 19. All heads change, because `x` feeds every head's projections. (The causal conv also spreads token 10 into the q, k, v of tokens 11 to 13, but the state already carries the change forward.)
c. After the output projection, yes: `o_proj` mixes all heads into every output channel. Before it, no: the op, the per-head RMSNorm and the output gate all act head by head. So heads are separate memories that are only combined at the very end.

**Q3.**
a. Exactly equal (`max |KDA − GDN| = 0`).
b. In GDN the decay ratio between tokens i and j is one number, so it comes out of the sum over key channels: `(k_i · k_j) × ratio_ij`. In KDA the ratio differs per channel and stays inside the sum: `Σ_d k_id k_jd ratio_ijd`. Built directly that is a `(C, C, Dk)` tensor per head per chunk. (The reference avoids it with a divide-and-conquer; knowing that is not required.)

**Q4.** Largest exact N: H=1 → 64, H=2 → 32, H=4 → 16, H=8 → 8. That is d/H.

Why: a head's keys live in d/H dimensions, so at most d/H of them can be orthogonal. One more key must overlap the others, and its write disturbs what they return.

More than 16 items with 4 heads: only by storing an item in some heads and not others (write strength near zero elsewhere). That gives up part of each item's value, since each head holds its own slice of the value. The budget is the state size, d²/H floats, and an item stored in full costs d of them. This last paragraph is reasoning, not something `reveal.py` measures; accept any answer that identifies the state size as the budget.

With random keys there is no sharp edge, but the same ordering holds at every N: more heads, worse recall.

**Q5.** Not graded. Reasonable points: heads make the state and the per-chunk work smaller; each head gets its own decay and write gates, so heads can run on different timescales; whether that is worth an H-fold cut in raw capacity is an empirical question. A good arm: same d_model, H=1 against H=4, on the cue-recall test [5.2] at increasing gap lengths.

## Part 3

Invariants 1 and 2 are **not tested** anywhere in `test_kda_layer.py` or `test_kda_ops.py`. The suite checks chunk against recurrent, causality, streaming, gradients, shapes, gate ranges and devices. All of those compare the layer with itself, so a mistake made before the op, the same way in both modes, passes every one. Credit for finding either gap.

## Part 4

The bug is in `forward`:

```python
q = F.normalize(q, dim=-1).view(B, T, H, dk).transpose(1, 2)
k = F.normalize(k, dim=-1).view(B, T, H, dk).transpose(1, 2)
```

It normalises across all heads together, then splits. The spec says split, then L2 per head. With H=1 the two are the same.

What it does to the memory: each head's key now has squared length about 1/H instead of 1. The delta rule's write at a key of squared length s corrects only a fraction `beta × s` of the error, so every write is about H times weaker than the gate asked for, and how hard a head writes depends on how large the other heads' keys happen to be. Heads are no longer independent.

A passing test is any of:

- Capture the `k` passed to the op (wrap `chunk_kda` or `recurrent_kda`) and assert `k.norm(dim=-1)` is 1 for every head.
- Scale the rows of `k_proj.weight` that belong to head 0 by 10. On the reference, the other heads' outputs before `o_proj` do not change. On the bugged layer they do.
- Build H single-head layers from the weights and compare.
