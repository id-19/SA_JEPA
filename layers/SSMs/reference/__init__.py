"""AI-written reference Gated DeltaNet and Kimi Delta Attention for MPS/CPU: correctness
oracles and speed baselines.

See README.md in this folder.
"""

from .gated_delta_rule import chunk_gated_delta_rule, recurrent_gated_delta_rule
from .gated_deltanet import CausalDepthwiseConv, GatedDeltaNetRef
from .kda import KimiDeltaAttentionRef
from .kda_ops import chunk_kda, kda_step, recurrent_kda
