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
## Ladder to Mamba (agreed 22 Sep night): rung 2 diagonal arm head-to-head vs 0.0299 first (answers "was the full matrix worth it?" — his S4-vs-S4D moment), THEN rung 3 selective (input-dependent Delta — Mamba's actual move, the concept the ladder exists to teach), THEN rung 4 conv1d + SiLU gate block = toy-trainable Mamba on the sine board. Order matters: diagonal is the ablation that justifies selective.
0. Toy trainer (train_toy.py, train_toy2.py solo redo) — DONE, wall 0.20245 certified
1. my_ssm.py naive full-matrix arm (renorm-in-forward 0.9 rowsum + exp(-exp(log_rate)) sliders, BIBO in __main__) — BUILT + CERTIFIED Sep 21, 0.0299 on toy (beats wall 0.20245, near ZOH par 0.06198)
   - Sep 22 (evening, AI-written, 3+1, ≤200w):
     1. Rung-1 batched rewrite DONE + verified: batch loop killed (state (B,d_state), `h @ eff_a.T`), _eff_a hoisted + asserted in-forward (abs row sums ≤ 0.99+1e-6, pre-transpose), double-stack scaffolding removed → returns (B,T,d_out). Seed-0 toy2 run landed ~0.029 = certified 0.0299 reproduced → same function, faster.
     2. Row-norm vs transpose settled from first principles: A@h and h@A.T are ONE equation (substitution A.T[i,j]=A[j,i]) — row sums cover both spellings; h@A (forgotten transpose) is the genuinely different equation (weights = columns). Column-renorm variant (dim=0, no transpose) is valid math but a DIFFERENT model — parked as rung-1b ablation candidate.
     3. Triage rule adopted: tasks ranked by transferable AI-assisted-ML learning or direct SA-JEPA/alignment value; wave-0 f=0 parked for good (keep rows, his call). Timing band 4–8× still unmeasured formally.
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
