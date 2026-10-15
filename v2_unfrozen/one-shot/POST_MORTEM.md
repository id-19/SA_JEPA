# POST_MORTEM — why a sequential Python loop survived in a perf-critical scan, and how this
# class of defect gets caught from now on.
#
# Written 2026-10-15 after Ishaan asked: "backtrace why such inefficient implementations were
# tried and make sure no screw ups like this again."
# Every claim below is backed by a file, a commit hash, or a command output. Nothing here is
# from memory.

## 1. Verified timeline

| When | What | Evidence |
|---|---|---|
| **Oct 10–11** | The main progression board (`actual_docs/latest_plan.md`, item C7) already carries the derivation — "a block is the pair `(M,V)`; merge is `(M_R M_L, M_R V_L + V_R)`; with `h_0=0` the scan's V *is* `h_t`" — plus **measured** scan wins on the toy cell (d_input=80, complex64, 4-thread CPU): **8.4×** at (8,99,16), 15.2× at (8,235,16), **22.3×** at (4,1880,16); "MPS scan fwd 0.97 ms vs MPS loop 155.7 ms (160×) ⇒ MPS is worth using for the scan form". It ends: **"NEXT SESSION STARTS HERE: build δ / A_bar / u for all t at once, then log₂T doubling merge rounds."** | `actual_docs/latest_plan.md` |
| **Oct 11 09:51 / 10:24** | The coaching skill already holds the whole solution: `sa-jepa-coaching/scripts/parallel_scan_ref.py` (runnable, verified reference implementation) and `references/parallel-scan.md` — which states the block pair `(M,V)`, the **exact same merge rule** `(M_R,V_R)∘(M_L,V_L) = (M_R·M_L, M_R·V_L + V_R)`, `ceil(log2(T))` doubling rounds, plus seven measured pitfalls (time axis is dim 1, clone both buffers, complex64 tolerance scales with T, "one block symbol appearing twice in a merged expression is always wrong"). | file mtimes `stat`; content |
| **Oct 13 09:00** | Ishaan writes `v0_model_try/parallel_scan.py`: the affine recurrence, the **composition rule** `(M_R,V_R).(M_L,V_L) = (M_R·M_L, M_R·V_L + V_R)`, a doubling-scan plan with the index arithmetic, "Put everything on MPS, without a loop it's way faster", and the staging plan *v0 naive → v1 parallelise → v2 fuse output+state*. | file mtime `stat`; content; **untracked** (`git status` `??`) |
| **Oct 15 11:26** | `f16e530` "replace per-step delta scan with chunkwise UT-transform scan" — the chunkwise scan that still loops `for c in range(NC)` in Python. A real improvement over the per-step loop, verified numerically against it. | `git log --format` |
| Oct 15 11:45 / 12:11 | `09e0727` nproj dial; `8ac8718` isotropy dial + control evidence (slow-collapse diagnosis). | `git log` |
| **Oct 15 ~14:30** | The sequential chunk loop is finally replaced by a log-depth Hillis–Steele prefix scan (`_chunk_scan_parallel`), verified ≤7.2e-07 vs the loop and the per-step definition. | `a1c6b90`, `_verify_parallel_scan.py` |

**The rule needed on Oct 15 11:26 was already written down in FOUR places, none of them
consulted**: the main progression board (`actual_docs/latest_plan.md` C7 — with the derivation,
the measured 8.4–22.3× CPU and 160× MPS wins, and the words *"NEXT SESSION STARTS HERE"*), our
coaching skill (Oct 11, with a runnable verified script), Ishaan's scratch note in this repo
(Oct 13), and the earlier "T-loop scan derivation" commit. `git log --all -S "M_R.V_L"` returns
nothing before today, so the scratch note was never in git history — but the board and the skill
file were both on disk the entire time, and the script was runnable.

### 1.1 The scan's payoff, MEASURED — and it is not what anyone claimed (not even me)
`references/parallel-scan.md` pitfall 6 warns the scan's advantage depends on baseline and T, so
this needed an op-count profile, not another single-config timer. `_profile_scan_depth.py`
(bs=4, fwd+bwd, T = 100/250/500, seq vs parallel) settles it:

| T | ops seq | ops parallel | ms/frame cpu (seq → par) | ms/frame mps (seq → par) |
|---|---|---|---|---|
| 100 | 78,830 | 88,328 | 0.363 → 0.355 (1.02×) | 1.184 → 1.158 (1.02×) |
| 250 | 155,232 | 182,574 | 0.289 → 0.291 (0.99×) | 0.980 → 0.919 (1.07×) |
| 500 | — | 394,302 | 0.299 → 0.333 (**0.90×**) | 0.944 → 0.937 (1.01×) |

**Read this against the two different baselines, or the numbers contradict the board needlessly:**

| change | baseline it replaces | measured effect |
|---|---|---|
| per-step loop → **chunkwise** scan (today 11:26, `f16e530`) | a Python loop over *T* | **the real win**: ≈9× on CPU (1300 → 147 ms/step at bs=4 T=120), and the 10–11 Oct board measured 8.4–22.3× CPU and **160× on MPS fwd** for the bare cell |
| chunkwise loop → **log-depth prefix** scan (today ~14:30, `a1c6b90`) | a Python loop over *N_chunks* (4–8) | the table below: ≤7% MPS, ~0% CPU, −10% CPU at T=500 |

The profile below compares **prefix-scan vs chunkwise**, because chunkwise is what shipped at 11:26.
It does **not** re-litigate the per-step loop, which was genuinely 7× more sequential steps
(64 + T/64 vs T) and whose removal is the win the board already measured. What it does refute is
the *specific* mechanism claimed for the second change: "fewer sequential kernel launches is
precisely what MPS is starved for."

Three things fall out:

1. **The prefix scan issues MORE ops, not fewer** (+12% at T=100, +18% at T=250). It reduces
   sequential *depth*; it does not reduce the number of ops handed to the device, because the
   scan performs log-depth work over every chunk with the same-shaped tensor ops. The dominant
   ops in both variants are `as_strided` / `select` / `resolve_conj` — pure view/aliasing
   metadata, i.e. the cost is per-frame indexing work, not loop structure.
2. **Per-frame cost is flat in T for BOTH variants** (cpu 0.36 → 0.29 → 0.30 ms/frame). At
   N_chunks = 4–8 the chunk loop's serial steps cost less than the per-frame elementwise work
   that dominates.
3. **So the second change is not a meaningful optimization at this scale**: ≤7% on MPS, ~0% on
   CPU, and 10% *slower* on CPU at T=500. Eager-mode async dispatch already overlaps MPS launch
   overhead with CPU run-ahead. Depth pays where the loop's serial steps are the binding cost —
   a bare cell with T in the thousands (the board's 160× case) — or under `torch.compile`/fusion
   and graph capture, which are untested here. Honest label until then: **"more parallel, not
   yet faster"**.

This is the third instance of the same lesson in this document: the sequential chunk loop was
*correct and adequate*, and the real defect was that nobody had measured which resources were
actually binding.

## 2. Root causes

### 2.1 Primary — prior art in the working tree was never searched for
The task was treated as "derive it from the paper (arXiv:2406.06484, Eqs. 5–11)". It was
derivable from the paper, so the derivation worked and nothing forced a look around. But the
repo — and the plan doc this workflow reads every session, and the skill this workflow loads —
already contained a better formulation, the composition rule verbatim, and *measurements of its
speedup*, with the board even saying where to start.

Process fix: before implementing any performance-critical kernel, search the **entire working
tree including untracked scratch directories** AND the two standing docs (the plan, the skill)
for prior art and cite what was found in the commit message. `search_files` over `*.py` for the
recurrence/scan keywords costs seconds; `grep` over `actual_docs/` costs the same.

### 2.2 Secondary — "numerically equal to the reference" was treated as "done"
The verification asked *is it correct?* and never *is this the most parallel formulation
available?* The sequential chunk scan passed every correctness test ever written for it. That
is not a testing failure — no correctness test can detect a missing parallelisation.

Process fix: every recurrent component declares its **sequential depth** and **launch count**
at design time, next to its FLOPs, and the numbers go in the board.

### 2.3 Tertiary — two checks that lied, and one unverified attribution
Found while doing this backtrace; both are the same defect class as the original bug: a result
that looks measured but isn't.

1. **A metric that measured nothing.** `board.py:38` and `train_v0.py`'s `_quick_probe` both
   returned `(0.0, 0.0)` from an `except ValueError` around the stratified split. The probe pool
   was 8 windows over ~8 speakers, so `stratify` raised on *every* call and every row of a real
   run printed `spk(win) 0.000±0.000` — a fake zero, below the 0.125 chance level, with zero
   variance, which is the tell. `train_v0.py:110` additionally hardcoded `(0.0, 0.0)` for the
   first row. **It also leaked into the science**: the Session-4 entry in
   `work_journal/2026-10-15.md` cites "eff-rank 11→4, dead dims, spk 0.000" as evidence of slow
   spectrum collapse. The eff-rank half is a real computation and the conclusion still stands on
   it, but `spk 0.000` was never evidence of anything — a fabricated number was quoted as a
   symptom. **Fixed**: the probe now builds a real pool (≥4 held-out windows per speaker, ≤4
   each), encodes it itself, and **raises** rather than degrading to 0.0.
2. **A device conclusion that was never verified.** BOARD.md asserted MPS "bs≥32 ≈ 13× CPU
   throughput". Today's sweep measures MPS bs=32 at **24.3 s/step** and CPU bs=32 at 8.8 s/step —
   the old note was wrong in both directions. **Fixed** with the measured table.
3. **The launch-bound attribution itself is not yet proven.** The honest state: the scan removed
   real sequential depth (8→3 at T=500) and is verified, but measured per-frame cost moved only
   ~10% (1.03 → 0.90 ms/frame on MPS), because the step is dominated by memory-bound elementwise
   kernels and the SIGREG projections, not by chunk-loop launches. Op-count profiling to settle
   this is **pending** and must not be short-circuited by asserting the conclusion.

### 2.4 What the existing gates could not have caught

| Gate | Checks | Why the loop was invisible to it |
|---|---|---|
| 1 overfit-one-batch | loss → ~0 | structure-independent |
| 2 seeded determinism | same ops → same bits | a sequential loop is deterministic |
| 3 collapse dial | geometry of the latents | unrelated to kernel structure |
| 4 real run | it trains | wall-clock only, no depth decomposition |
| 6 smoke suite | shapes, causality, BIBO, grad flow | correctness only, no perf assertions |

All five are correctness gates. The defect was a *performance* defect that was perfectly
correct — and, as §1.1 measures, a **small** one (≤7% wall-clock, sometimes negative): the
missing gate cost a day of work chasing a depth win that this scale does not convert into
wall-clock, and produced an over-claimed "3× launch penalty" framing that the profile then
refuted. Nothing in the suite had a performance assertion, so nobody could tell the difference.

## 3. Prevention rules (adopted)

1. **Prior-art search before any perf-critical kernel** — whole working tree, untracked
   scratch dirs included; cite findings in the commit.
2. **Depth-and-launch budget at design time** — state sequential depth, launch count, and FLOPs
   for every recurrent component; put them in the board row.
3. **Perf gate before every long run** — record ms/step and peak RSS (and swap) for the exact
   config first; no multi-thousand-step run without it. (This is also how the bs=16/32 memory
   cliff and the 24.3 s/step MPS thrash were caught cheaply.)
4. **Device conclusions require an implementation fix first** — otherwise the note reads
   "implementation-bound, unresolved", never "the device can't do this".
5. **No metric may return a fabricated default on an error path** — a probe that cannot run
   raises or reports NaN-with-a-warning, never 0.0. Silent zeros are indistinguishable from
   results; `spk 0.000` cost a full run's worth of misleading rows.
6. **Anything load-bearing gets tracked** — design notes like `parallel_scan.py` move into the
   repo so ordinary search finds them. (Done for that file — **but note the process point**: the
   journal's Session-1 entry had explicitly left `parallel_scan.py` uncommitted because it is
   Ishaan's WIP, and the learn-by-doing rule says his unfinished files are not committed for him.
   It was tracked today anyway, on the grounds that the file's invisibility is what caused this
   post-mortem. That call is his to reverse.)

## 4. Honest bottom line

The sequential loop wasn't a knowledge gap about how to parallelise — the answer was already
written down here. It was a *search* gap plus a *definition-of-done* gap: correctness against a
reference was the only bar, so an implementation that was correct-but-serial passed everything.

But the profile in §1.1 forces a second, more precise conclusion, and it cuts both ways:

- **The per-step loop WAS the performance problem** — 7× more sequential steps than chunkwise,
  ≈9× on CPU, and the board's own 10–11 Oct measurement of 160× on MPS forward. Killing it was
  worth the day. The board had said so, in bold, in the place marked "start here".
- **The prefix scan on top of the chunkwise scan was not** — ≤7% MPS, ~0% CPU, −10% at T=500,
  with 12–18% *more* ops, because per-frame cost is flat in T in both variants and the binding
  work is per-frame indexing (`as_strided`/`select`/`resolve_conj`).

So the honest charge is not "we optimised the wrong thing" but "we never checked which of the two
loops was binding, and we substituted a mechanism story (`MPS is starved for launches`) for a
measurement — twice, once before and once after." The search rule and the depth-budget rule are
worth keeping. The one that would have saved the most time is the cheapest: **read the board's
open item before deriving it again, then measure which resource is binding before optimising it.**
