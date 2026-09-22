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
0. Toy trainer (train_toy.py, train_toy2.py solo redo) — DONE, wall 0.20245 certified
1. my_ssm.py naive full-matrix arm (renorm-in-forward 0.9 rowsum + exp(-exp(log_rate)) sliders, BIBO in __main__) — BUILT + CERTIFIED Sep 21, 0.0299 on toy (beats wall 0.20245, near ZOH par 0.06198)
   - Sep 22: batched rung-1 edit IN FLIGHT (batch loop killed, _eff_a hoisted) — 3 flags open: line 43 `*`→`@`, return shape, missing A_eff assert; timing + gradient allclose pending (band: 4–8×)
2. Diagonal arm (delete matrix, 16 sliders, elementwise scan) — next; head-to-head vs arm 1
3. Selective: input-dependent Delta (Mamba's move)
4. Mamba block wrapper (conv1d + SiLU gate) — reference: model/Mamba/ (answer-key)
5. Predictor (task 1) -> 6. Encoder->Predictor forward (task 2) -> 7. Real JEPA loop + SIGREG (task 3) -> 8. Validation/probes (task 4)
Training ladder: toy 0.20245 wall -> LibriSpeech mel frames (B,T,80) via data_pipeline_v0.py -> predictor.


Current tasks
Goal: JEPA + Mamba audio hybrid
0. Copy paste or code out a very basic Mamba implementation.
  - You must understand it, is the condition.
1. Define predictor architecture with Mamba integrated
2. Do a basic test, embed some data, run predictor and see that it works(model defined)
3. Setup the training loop, get some epochs running
4. Setup validation loss, linear probes to measure quality
5. Iterate on training loop, make it better
6. Let the loop run for a day or so, get validation that it works
