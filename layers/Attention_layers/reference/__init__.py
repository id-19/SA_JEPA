"""AI-written reference attention layers (GQA, MLA, RoPE): correctness oracle and speed baseline."""

from .gqa import GQARef
from .mla import MLARef
from .rope import RotaryEmbedding, apply_rope, default_positions
