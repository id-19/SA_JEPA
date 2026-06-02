#### This document tracks hwat our current base idea is

# SA‑JEPA v0 Plan (Current Version)
1. Data: Data wise- start with speech, for now, expand to other types of audio later
2. Architecture(v0 basic): Both audio encoder and predictor are just transformers with Mamba layers inserted
  - No need for a tokeniser, log mel spectrograms are already in matrix form.
  - Single encoder, single predictor, train linear probes from time to time
3. Training:
  - SIGREG + pred loss
  - Train o
4. Future ideas to try(and validate):
  - MoE
  - Constant batch size, but different size clips composed to make that batch size
  - Recurrent attention - to max out params
  - Bigger model + LoRA
  - GQA and other non-vanilla attention types
  - (Add more, later, don't waste time now)


Current tasks
1. Define model architecture for encoder and predictor
2. Get basic training loop setup, on even a small model
  - Add visualisations and stuff