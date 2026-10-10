#### This document tracks hwat our current base idea is

# SA‑JEPA v0 Plan (Current Version)
1. Data: Data wise- start with speech, for now, expand to other types of audio later
2. Architecture(v0 basic): Both audio encoder and predictor are just transformers with Mamba layers inserted
  - No need for a tokeniser, log mel spectrograms are already in matrix form.
  - Single encoder, single predictor, train linear probes from time to time
3. Training:
  - Patch embedding based
  - SIGREG + pred loss
4. Future ideas to try(and validate):
  - MoE
  - Constant batch size, but different size clips composed to make that batch size
  - Recurrent attention - to max out params
  - Bigger model + LoRA
  - GQA and other non-vanilla attention types
  - (Add more, later, don't waste time now)
  - The new attention residuals and stuff
  - Dual-timescale memory: leaky fast bank + non-leaky slow bank with learned periodic refresh (cf. Compressive Transformer, LSTM cell state, fast weights)

## SSM Rung Progression — MILESTONE REACHED 10 Oct 2026
Board: blind 0.4835 / wall 0.20245 / ZOH par 0.06198 / full-matrix 0.0299 / real-diag 0.053 / S4D rotation 0.02874 / selective-Delta real-pole 0.019658 (08 Oct) / **selective-Delta COMPLEX 0.01279 @ step 3150 (10 Oct, mid-run, NOT a certified endpoint) — CAPACITY CONTROL CLOSED 10 Oct: ω frozen at init, same seed/steps/112 floats → 0.019277 @ step 1950 (lands on the real-pole arm's 0.019658); ω trained → 0.013567 @ step 1950 ⇒ the win is the LEARNED ROTATION, not the 1.75× capacity. d_state=8 + trained ω → 0.017606 @ step 1950 (provisional).**

**Where the ladder stands:** 0 (toy trainer) DONE · 1 (full-matrix) CERTIFIED · 2 (diagonal) THEORY DONE, real-diag code landed 0.053 · 3 (selective, real-pole) CLOSED 08 Oct · 3b (selective, complex) LANDED 10 Oct — capacity control CLOSED, the rotation is the mechanism (see board). The SSM-rung ladder has *produced its mechanism*: the input-dependent, complex selective recurrence is built and beats everything below it. Rungs 4–5 (Mamba block wrapper, predictor) are now **infrastructure, not research** — the next real questions are the hybrid ratio, recurrent depth, and JEPA dynamics (the thesis tracks).

0. Toy trainer (train_toy.py, train_toy2.py) — DONE, wall 0.20245 certified
1. Full-matrix arm (my_ssm.py): renorm-in-forward 0.9 rowsum + exp(-exp(log_rate)) sliders — BUILT + CERTIFIED Sep 21 (0.0299); Sep 22: batched rewrite DONE + verified (batch loop killed, `h @ eff_a.T`, `_eff_a` hoisted + asserted in-forward, returns (B,T,d_out); seed-0 ~0.029 = certified number reproduced → same function, faster). Loose end: 4–8× speedup band never formally timed.
2. Diagonal arm — theory DONE Sep 23, code NEXT: delete matrix, 16 sliders, elementwise scan; head-to-head vs 0.0299 (answers "was the full matrix worth it?"; diagonal is the ablation that justifies selective). Settled from first principles Sep 23: (a) renorm must go — on a diagonal it deletes A entirely (d(eff)/d(A)≈1e-8, eff = ±0.99·slider); (b) bound is per-entry `|a_j| ≤ 0.99` with learnable sign — and note ‖diag(a)‖₂ = max|a_j| EXACTLY, so no transient-growth gap (the matrix arm only bounds row sums, and singular values can exceed the spectral radius); (c) row ops give singular values, NOT eigenvalues (that was the tangle) — only similarity P⁻¹AP preserves eigenvalues, and a real P cannot diagonalize a matrix with complex eigenvalues (generic: 1000/1000 random renormed 16×16); (d) `exp(-exp(log_rate))` is unusable here (no sign) and Mamba's `-exp(log(1..16))` init is unusable too (no Δ to shrink it → |a| up to 16 → explodes). Known cost, pre-registered: a real `a` hosts only ω=0 or ω=π; 24/25 of the sine board's channels (ω spans 0–1.2063 rad/sample) need other frequencies, so the arm must synthesize oscillation from a real- pole bank. This is the board that maximizes the diagonal's handicap. Parked: column-renorm variant (dim=0, no transpose) as an arm-1b ablation.
3. Selective: input-dependent Delta (Mamba's actual move — the concept the ladder exists to teach). **Rung 3 CLOSED 08 Oct** — real-pole form trained to 0.019658 (wd=0; beats S4D 0.02874 and his own 0.029–0.053 band). **Rung 3b COMPLEX — CODE DONE + TRAINING 10 Oct (`my_sel_s4d_complex.py`).** Complex-native form: `a = -rate+i*omega` (complex64), state `h = (B,d_state)` complex64, kick = `einsum('bc,cs->bs', x.to(complex64), B)` contracted FIRST, then `h = h*A_bar + B_bar*kick`, readout `Re(<h, conj(C)>)` — the `.conj()` is load-bearing (silently wrong without it, finite loss). Hit **0.01279 @ step 3150** (mid-run). Four blockers found by probe and fixed: `complex32`=ComplexHalf (`exp`/`bmm` unimplemented); `-log_rate` missing its `exp` (channel 0 never decayed, rate 0.0 vs 1.0); the kick's `* self.B` batch-axis collision at `d_input>1` (contract over d_input first — the `d_input=1` case hides it because a size-1 contraction degenerates to a promote-legal multiply); missing `.real` (complex loss can't backward). Header rewritten to the real contract. **OPEN: capacity confound** — complex64 = 2 floats/element ⇒ 112 real floats vs the real-pole arm's 64 (1.75×); isolating control is `omega` frozen at init (same seed/steps/floats): 0.0197 ⇒ rotation does the work, 0.0128 ⇒ capacity. ω DID move off its ramp (12/16 >0.1 rad, max 2.36) ⇒ the rotation is used, but per-step angle is ω·δ with δ input-dependent, so a channel is NOT a fixed-frequency resonator the way S4D is. **Timing verdict (measured, do not re-litigate): the T-loop is dispatch-bound — CPU 19.5ms/step, MPS 4.3× SLOWER (80.4ms), torch.compile 0.66×, micro-opts 1.05×; time is FLAT in T (~54µs/time-step) and 40× batch costs only 5× time. The lever is killing the T-loop (rung-2 thread), or simply running fewer steps.**

4. Mamba block wrapper (conv1d + SiLU gate) = toy-trainable Mamba on the sine board — reference: model/Mamba/ (answer-key, consult allowed)
5. Predictor (task 1) -> 6. Encoder->Predictor forward (task 2) -> 7. Real JEPA loop + SIGREG (task 3; trim Epps–Pulley grid) -> 8. Validation/probes (task 4)
Data ladder: toy sines -> LibriSpeech mel frames (B,469,80) via data_pipeline_v0.py -> full audio.

## Next steps (one copy, in order) — re-looked 10 Oct, after rung 3b
The memory mechanism is built. What's left falls into two piles: **(A) close the rung-3 evidence cleanly** so the 0.01279 means something, and **(B) the thesis tracks** — hybrid ratio, recurrent depth, JEPA loop — which are now the real research. D is written as a spec early, built after the dense core exists.

**A — finish the attribution (this week's real scientific work):**
1. ~~**ω-freeze control**~~ — **DONE 10 Oct.** ω frozen at init (same seed/steps/112 floats) → 0.019277 @ step 1950 ≈ the real-pole 0.019658; ω trained → 0.013567 @ step 1950. **Rotation does the work; the extra capacity alone buys nothing.** Side result: `d_state=8` + trained ω → 0.017606 @ 1950 (provisional, mid-run).
2. **Standardize the steps budget** (2–5k for EVERY arm) and re-certify the real-pole 0.019658 at the same budget, so the comparison is like-for-like. Then a clean certified endpoint for the complex arm.

**B — the thesis tracks (the actual research now):**
3. **The transition board** — 25 waves that switch frequency mid-sequence, where selective Δ *should* win decisively and the fixed arm should pay ghost error on segment B. This is the test that justifies selectivity's existence, and it plugs directly into the hybrid question.
4. **Hybrid probe on toy data** — 3 delta-rule blocks : 1 gated-attention block, at fixed params. Pre-register: does the attention layer move the loss at all at this scale? (This is the first measurement of the ratio question, before any real model exists.)
5. **Recurrent-depth probe** — one shared block, looped R times; loss vs R. Watch for the CART failure mode (recurrence vestigial at your width).
6. **JEPA collapse dynamics on toy data** — tiny transformer pair, EMA + stop-grad, SIGREG *and* Gaussian regularizer; instrument for the three silent collapses. Learn it on a model that trains in seconds.

**C — infrastructure, only when A closes (in order, no research here):**
7. Kill the T-loop (associative scan; measured 20× ceiling, makes MPS worth using). **DERIVATION DONE 10 Oct:** unroll `h_t` in `h_0` — every `u_k` is carried by the A's AFTER it (`h₃ = A₃A₂u₁ + A₃u₂ + u₃`); a block is the pair `(M,V)`; merge is `(M_R M_L, M_R V_L + V_R)`; with `h_0=0` the scan's V *is* `h_t`; leaf pair `(A_t, u_t)`. **NEXT SESSION STARTS HERE:** build δ / A_bar / u for all t at once (batched einsums), then log₂T doubling merge rounds.
8. Clean two residue in train_toy2.py (dead `device = torch.mps` line; `x` still single-channel while `__main__` exercises d_input=80).
9. Mamba block wrapper → toy-train on the sine board.
10. d_state sweep {4,8,12,16,25,32,64} on the best arm.
11. LibriSpeech mel frames through the best SSM (audio smoke, B=8).
12. Predictor → encoder-predictor forward → JEPA loop + SIGREG on real data.

**D — the memory hierarchy (spec now, build after the dense core):** the residency map (which layers are hot/cold/stream, what grows with T) written as one page BEFORE the dense core; SSD benchmark the day the drive arrives; MemTensor abstraction; the int4 page file with a speech-locality prefetch predictor. Nothing here is built until the track-C core exists.

Working rules: he writes+commits all code; pose problems, not edits; pre-register guesses before runs; print the board beside every loss; journal 3+1 ≤200w AI-written per session day; triage by AI-assisted-ML transfer or SA-JEPA/alignment value.