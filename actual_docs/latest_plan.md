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

## SSM Rung Progression (Sep 2026) — v0_model_try/
Board: blind 0.4835 / wall 0.20245 / ZOH par 0.06198 / mine 0.0299 (batched-verified 22 Sep)

0. Toy trainer (train_toy.py, train_toy2.py) — DONE, wall 0.20245 certified
1. Full-matrix arm (my_ssm.py): renorm-in-forward 0.9 rowsum + exp(-exp(log_rate)) sliders — BUILT + CERTIFIED Sep 21 (0.0299); Sep 22: batched rewrite DONE + verified (batch loop killed, `h @ eff_a.T`, `_eff_a` hoisted + asserted in-forward, returns (B,T,d_out); seed-0 ~0.029 = certified number reproduced → same function, faster). Loose end: 4–8× speedup band never formally timed.
2. Diagonal arm — NEXT: delete matrix, 16 sliders, elementwise scan; head-to-head vs 0.0299 (answers "was the full matrix worth it?"; diagonal is the ablation that justifies selective). Parked: column-renorm variant (dim=0, no transpose) as an arm-1b ablation.
3. Selective: input-dependent Delta (Mamba's actual move — the concept the ladder exists to teach)
4. Mamba block wrapper (conv1d + SiLU gate) = toy-trainable Mamba on the sine board — reference: model/Mamba/ (answer-key, consult allowed)
5. Predictor (task 1) -> 6. Encoder->Predictor forward (task 2) -> 7. Real JEPA loop + SIGREG (task 3; trim Epps–Pulley grid) -> 8. Validation/probes (task 4)
Data ladder: toy sines -> LibriSpeech mel frames (B,469,80) via data_pipeline_v0.py -> full audio.

## Next steps (one copy, in order)
1. Diagonal arm head-to-head vs 0.0299 (pre-register curve guess; timing vs full-matrix arm comes free)
2. Selective arm (input-dependent Delta)
3. Mamba block wrapper -> toy-train it on the sine board
4. d_state sweep {4,8,12,16,25,32,64} on the best arm — predict curve shape first
5. LibriSpeech mel frames through the best SSM (audio smoke, B=8)
6. Predictor -> encoder-predictor forward -> JEPA loop + SIGREG

Working rules: he writes+commits all code; pose problems, not edits; pre-register guesses before runs; print the board beside every loss; journal 3+1 ≤200w AI-written per session day; triage by AI-assisted-ML transfer or SA-JEPA/alignment value.