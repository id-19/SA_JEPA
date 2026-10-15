# Critical evidence for the post-mortem (added mid-run, 2026-10-15)

## Primary root cause: prior art already in the working tree, never consulted

File: `v0_model_try/parallel_scan.py` — **untracked**, mtime **Oct 13 09:00:49 2026**.
Authored by the repo owner (Ishaan), two days BEFORE the v2 sequential chunkwise scan work
(`v2_unfrozen/one-shot/`, Oct 14-15).

Its header comments already state:

- the affine recurrence: each state update is `h_i = Abar * h_(i-1) + Bgain * u`
- the **affine composition rule** for chaining two steps/ranges:
  `(M_R, V_R) . (M_L, V_L) = (M_R * M_L,  M_R * V_L + V_R)`
- a **tree / doubling (Hillis–Steele) scan plan**, with the index arithmetic spelled out:
  "Make an array of size T, and for each round, i*(2**round) -> i*(2**(round+1)) is stored at
  index i"
- the explicit remark: "Put everything on MPS, without a loop it's way faster"
- an explicit staging plan: "For v0, do naively, for v1 can parallelise, for v2 can compute
  output and state together to avoid storing matrices"

This is the same mathematical content as `_chunk_scan_parallel` implemented on Oct 15 in
`v2_unfrozen/one-shot/model.py` (chunk maps `M_c = I - W_c^T K_c`, `b_c = U_c^T K_c`, composed by
a Hillis–Steele prefix scan).

### Why this matters
The sequential Python chunk loop that survived into v2 was not an unknown-hard problem: the
correct, parallel formulation was already written down in the repository's own working tree
before the sequential version was built. Nobody searched for it.

### Contributing factor
The file is **untracked**, so `git log`/`git grep` on tracked history would never surface it —
only a whole-working-tree file search would. Any future "look for prior art" rule must include
untracked scratch directories (`v0_model_try/`, `_control_*.py`, etc.), not just git history.

## Secondary root cause: "matches the reference numerically" treated as done

The implementation path was: derive the chunkwise algorithm from the paper (arXiv:2406.06484),
then verify outputs/state against a per-step loop to < 1e-5. That verification answered
"is it correct?" and was treated as "is it finished?". No step ever asked:

- Is this the most parallel formulation available for this recurrence?
- What is the sequential depth / number of kernel launches, and is that acceptable on the
  target device?
- What prior art exists in this repo for this exact recurrence?

Proof that the question was never asked: the first performance response to slow MPS numbers was
"the device is launch-bound, so use CPU" — i.e. adapt the workload to the device rather than
remove the sequential dependency. The correct response to launch-bound behaviour is to reduce
the number of sequential launches.

## Prevention rules proposed from this evidence

1. **Prior-art search before any perf-critical kernel**: grep the ENTIRE working tree (including
   untracked scratch dirs) for the recurrence/kernel name and for words like scan, recurrence,
   parallel, compose, doubling; cite what was found in the commit message. Cheap, would have
   caught this exactly.
2. **Depth budget in the design note**: every recurrent/sequential component states its
   sequential depth and its launch count at design time, alongside its FLOPs.
3. **Device conclusion requires an implementation fix first**: a benchmark may only conclude
   "don't use device X" after the launch count has been reduced; otherwise record it as
   "implementation-bound, unresolved".
4. **Perf gate before long runs**: a tiny throughput check (bs=8, T=250, N steps) recorded in the
   board before any multi-thousand-step run, with ms/step and peak RSS.
5. **Track scratch that matters**: promote design notes like `parallel_scan.py` into the repo
   (`actual_docs/`) so they are discoverable by ordinary search.
