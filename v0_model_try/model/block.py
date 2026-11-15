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


class MLP(nn.Module):
    def __init__(self, d_model, d_ff, depth=2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.in_proj = nn.Linear(d_model, d_ff)
        layers = []
        layers.append(nn.SiLU())
        for _ in range(depth):
            layers.append(nn.Linear(d_ff, d_ff))
            layers.append(nn.SiLU())
        self.layers = nn.Sequential(*layers)
        self.out_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.layers(self.in_proj(x))
        return self.out_proj(x) # No norm on outputs(artificially constrains output)


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
        self.norm1 = nn.RMSNorm(d_model)
        self.mixer = mixer
        self.norm2 = nn.RMSNorm(d_model)
        self.mlp = MLP(d_model, d_ff)


    def forward(self, x, state=None, return_state=False):
        """x: (B, T, d_model). Returns y with x's shape, or (y, state) when return_state is True.
        `state` is the mixer's state, passed through."""
        if return_state:
            temp, state = self.mixer(self.norm1(x), state = state, return_state=True)
            x = x + temp
            return (x + self.mlp(self.norm2(x)), state)
        else:
            x = x + self.mixer(self.norm1(x), state=state, return_state=False)
            return x + self.mlp(self.norm2(x))
