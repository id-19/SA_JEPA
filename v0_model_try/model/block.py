"""Residual block: the one layer that the encoder, predictor and decoder are all made of.

After this you should be able to: write a pre-norm residual block from memory, and say what the
norm, the mixer, the MLP and the skip connection are each for.

Why: every layer of the encoder, predictor and decoder is this one block. The attention-only
control arm is the same block with a different mixer.

Steps:
  1. Fill in __init__ and forward below.
  2. Run `uv run pytest eval_test/tests/test_block.py -v` until green.
  3. No help unless you ask.
Do not: write the stack of blocks or the layout-string parser.
"""

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, d_model, mixer, d_ff):
        """Contract (what the tests rely on):

        - The tests read the attributes `self.norm1`, `self.mixer`, `self.norm2`, `self.mlp`.
        - `mixer` is any module with `mixer(x, state=..., return_state=True) -> (y, state)`;
          x and y have shape (B, T, d_model).
        - `mlp` maps (B, T, d_model) -> (B, T, d_model) with hidden width d_ff. The design is yours.
        - The norms are RMSNorm over d_model with a learnable weight initialised to ones
          (torch.nn.RMSNorm or your own).
        """
        super().__init__()
        raise NotImplementedError

    def forward(self, x, state=None, return_state=False):
        """x: (B, T, d_model). Returns y with x's shape, or (y, state) when return_state is True.
        `state` is the mixer's state, passed through."""
        raise NotImplementedError
