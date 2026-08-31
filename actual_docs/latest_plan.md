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
