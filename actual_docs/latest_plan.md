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
1. Define the smallest non-Mamba baseline.
2. Implement SIGReg as a standalone tested module.
3. Build the training loop decomposed, not as one blob.
4. Run a tiny overfit test, then a small real run.
5. Build out actual Mamba module
  - Mamba2 to learn
  - Mamba3 to actually use
6. Define a Mamba based architecture(make variable using config)
7. 