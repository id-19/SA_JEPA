# Answer key for QUIZ.md — closed until all four parts are done

*For the grading session. Mark strictly: a prediction is right or wrong, and a vague answer is wrong. Numbers below come from `reveal.py`, run on 14 Nov 2026 on an M1, torch on CPU (MPS for Q5c).*

## Part 1. Invariants a good answer contains

Not graded. Use this to point out gaps.

[1.3]
1. **Everything that carries the past crosses the boundary.** For this layer that is the delta-rule matrix `S` and the last k−1 inputs of each causal conv (q, k, v). Nothing else depends on past tokens.
2. Pieces with carried state equal one call, outputs and final state, **within a stated tolerance** (float32: about 1e-6), at awkward splits: pieces of 1 and 2 tokens (shorter than the conv), splits that are not multiples of the chunk size, many pieces.
3. Gradients for inputs and parameters through a carried state equal the one-call gradients. Equality, not just "nonzero".
4. Detaching the state stops the gradient exactly: with every part detached, no token of the earlier piece gets gradient from a loss on the later piece.
5. The state size is fixed: it does not grow with T.
6. `state=None` equals an explicit zero state; batch elements' states do not mix.

[1.4]
1. Feeding one token at a time equals the full call position by position, from a zero state and from a carried one.
2. Mixed use: a prefix in one call, then one-token steps.
3. It holds over long runs (thousands of steps), not just 20.
4. It holds on the code path actually used for generation, including the no-gradient Metal path on the Mac.

## Part 2

**Q1.** `S`: 65,536 (4 heads × 128 × 128). Tails: 4,608 (3 positions × (512 + 512 + 512)). Total: 70,144. Same after T = 100 and T = 10,000. The tails are 6.6% of the state. GQA passes it after **274 tokens** (70,144 / 256, where 256 = 2 for K and V × 2 kv heads × 64).

Grade: the totals and "same at both T" must be right; the crossover within ±10%.

**Q2.**
a. Positions **3, 4, 5**. The tail is the last k−1 = 3 inputs of the stream, so a 1-token piece pushes out the oldest one. The code builds it from `cat(old tail, new piece)`, which is why it is right even when the piece is shorter than 3.
b. **About 1e-1** (measured 5.1e-2 at token 59; 0.2 to 0.3 at tokens 30–32). The wrong k and v at tokens 30–32 are written into S, and decay shrinks the error slowly. Catches it with `atol=1e-5`: **yes**. With `atol=0.1`: **no** (0.05 < 0.1). A loose tolerance on late tokens hides a real bug, and one far above float rounding (about 1e-6). (The question of which tokens go wrong was discussed in chat on 14 Nov and is not graded here.)

**Q3.**

| Pieces | exactly equal? | about |
|---|---|---|
| 512 + 512 | **yes** (0.0, GDN and KDA) | – |
| 1000 + 24 | no | 3e-7 to 4e-7 (round to 1e-7) |
| 17 + 1 + 1 + 1005 | no | 4e-7 to 6e-7 (1e-7) |
| 1024 × 1 token | no | 5e-7 to 7e-7 (1e-7) |

Why 512 + 512 is exact: 512 is a multiple of the chunk size 64, so the two calls run exactly the same chunks, in the same order, with the same arithmetic. Every other split either changes where chunks start or switches pieces of 8 tokens or fewer to the token-by-token op, which adds the same numbers in a different order. That is float rounding, around 1e-7 in float32, not an error. So a test of "pieces equal one call" needs a tolerance, and the right one is the chunked-against-recurrent error, not zero. After 1024 one-token steps the error has not grown past 1e-6: no visible drift.

**Q4.**
a. **Yes.** Measured: `x` gradients 5e-7 apart, worst parameter gradient 1e-5 apart (absolute, on gradients of order 1 to 10; float error).
b. **Tokens 27, 28, 29**: the last k−1 = 3 tokens of the first piece, through the conv tails. Detaching `S` alone does not cut the gradient.
c. **None.**

To pass Q4, b must name exactly 27–29 (or "the last three", with the reason).

**Q5.**
a. T = 1 and 8: `recurrent_gated_delta_rule`. T = 9 and 64: `chunk_gated_delta_rule`. The switch is `T <= 8` in `forward`.
b. **Yes**, 3e-7. The chunked op is correct on a 1-token piece (one partial chunk of length 1); "auto" picks the recurrent op for speed, not correctness.
c. Without gradients: `metal_kernels.kda_step_metal`, a fused Metal kernel (the KDA kernel; a scalar decay is the same value on every channel). With gradients on: the PyTorch loop in `recurrent_gated_delta_rule`. So the decode path on the Mac is **not** the code checked in this quiz. It is covered only by `test_stack_mps_prefill_and_decode_matches_cpu` (20 decode steps, against CPU).

**Q6.** Not graded. Reasonable points:
a. `return_state` is fine and keeps training calls short; a `step()` adds nothing that a one-token `forward` does not already do. Asking for the ticket's exact interface is also defensible, if the block in [2.1] will always want the state back.
b. Acceptable for v0, since stage 1 has no generation. Before trusting generated text: a long test (thousands of steps) comparing Metal decode with the CPU loop, or turning Metal off until [6.3] shows the speed matters.

## Part 3. What the tests do and do not cover

| Item | Test |
|---|---|
| Pieces equal one call, outputs, awkward splits (17, 1, 1, 31) | `test_layer_streaming_equals_single_call` (GDN), `test_streaming_equals_single_call` (KDA), hybrid `test_stack_streaming_matches_single_call` |
| Conv tails correct, pieces shorter than the kernel | `test_conv_tails_match_single_call` |
| Final `S` equal at the split (op level) | `test_state_carries_across_calls` in `test_gdn_reference.py` and `test_kda_ops.py` |
| One token at a time from zero | `test_token_by_token_streaming` (KDA, 20 tokens). GDN: only via the 1-token pieces in the streaming test |
| Prefix then one-token steps | hybrid streaming test, pieces `(40,) + (1,) * 20` |
| State size fixed in T | hybrid `test_delta_rule_state_is_constant_and_attention_cache_grows`, `test_layer_state_floats` |
| Save to disk and resume | hybrid `test_stack_state_survives_save_and_load` |
| Metal decode path | hybrid `test_stack_mps_prefill_and_decode_matches_cpu`, 20 steps |
| **Gradients through a carried state equal one call** | **not tested** |
| **Detaching the state stops the gradient** | **not tested** |
| Long one-token runs (thousands of steps) | not tested |
| Batch elements' states independent | not tested directly |

The two bold rows are both Done-when items on ID1-281, and the planted bug lives in that gap. A good part 3 finds them before part 4.

## Part 4

**The line:** in `CausalDepthwiseConv.forward`,

```python
new_tail = x.data[:, x.shape[1] - pad:]
```

The reference has `x[:, ...]`. `.data` returns the same numbers **cut off from autograd**, so the returned conv tails carry the right values and no gradient. Outputs are identical for any split. Only gradients differ, and only when a loss on a later piece should reach the last k−1 = 3 tokens of an earlier piece through the conv. `S` still carries gradient, so a test that only asks whether gradients "flow" passes.

**When it matters:** any training that backpropagates across calls: training on a long sequence in pieces without detaching, or a loop that feeds the layer its own state, as [2.5] might. Stage-1 training reads whole sequences in one call, so it would never notice. That is what makes it silent.

**A test that catches it:** split a sequence (e.g. 60 tokens at 30), loss on both pieces, compare `x.grad` (or loss on the second piece only, and compare the gradient at tokens 27–29) with the one-call gradient, within 1e-5. Measured: the bugged layer differs by 0.49 at tokens 27, 28, 29; the reference by 5e-7.

The test must fail on `GatedDeltaNetBugged` and pass on `GatedDeltaNetRef`. A test that only checks the gradient is nonzero does not count: it passes on both.

Once found, the test goes into `eval_test/tests/test_gdn_reference.py` and `test_kda_layer.py`, closing the gap in part 3.
