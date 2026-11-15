"""The reference layers used together: a tiny hybrid stack, 3 delta-rule layers to 1 attention layer.

This is a test of the shared contract, not a model. Every reference mixer has
    forward(x, state=None, return_state=False) -> y  or  (y, state)
and feeding a sequence in pieces with the state passed along must equal one call. That is
what long-context inference rests on: prefill in pieces, decode token by token, save the
state to disk and pick it up later.

Everything is small (batch 1) because the machine has 8 GB of RAM.
"""

import io
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.Attention_layers.reference import GQARef, MLARef, RotaryEmbedding
from layers.SSMs.reference import GatedDeltaNetRef, KimiDeltaAttentionRef

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="no MPS device")
D = 32


class Block(nn.Module):
    """Pre-norm residual block around any reference mixer, plus a small MLP."""

    def __init__(self, mixer):
        super().__init__()
        self.norm1, self.norm2 = nn.RMSNorm(D), nn.RMSNorm(D)
        self.mixer = mixer
        self.mlp = nn.Sequential(nn.Linear(D, 2 * D, bias=False), nn.SiLU(), nn.Linear(2 * D, D, bias=False))

    def forward(self, x, state=None):
        h, state = self.mixer(self.norm1(x), state=state, return_state=True)
        x = x + h
        return x + self.mlp(self.norm2(x)), state


def make_stack():
    torch.manual_seed(0)
    mixers = [
        KimiDeltaAttentionRef(D, 2, chunk_size=16),
        KimiDeltaAttentionRef(D, 2, chunk_size=16, n_v_heads=4),
        GatedDeltaNetRef(D, 2, chunk_size=16),
        MLARef(D, 4, kv_rank=8),                                            # NoPE, as in Kimi Linear
        KimiDeltaAttentionRef(D, 2, chunk_size=16),
        KimiDeltaAttentionRef(D, 2, chunk_size=16),
        GatedDeltaNetRef(D, 2, chunk_size=16),
        GQARef(D, 4, n_kv_heads=2, rope=RotaryEmbedding(D // 4), window=24),
    ]
    return nn.ModuleList(Block(m) for m in mixers).eval()


def run(stack, x, states=None):
    states = states or [None] * len(stack)
    out = []
    for block, st in zip(stack, states):
        x, st = block(x, st)
        out.append(st)
    return x, out


def stream(stack, x, pieces):
    outs, states, t = [], None, 0
    for n in pieces:
        y, states = run(stack, x[:, t:t + n], states)
        outs.append(y)
        t += n
    assert t == x.shape[1]
    return torch.cat(outs, dim=1), states


@pytest.mark.parametrize("pieces", [(60,), (33, 27), (17, 1, 1, 41), (40,) + (1,) * 20])
def test_stack_streaming_matches_single_call(pieces):
    stack = make_stack()
    x = torch.randn(1, 60, D)
    with torch.no_grad():
        y_full, _ = run(stack, x)
        y_stream, _ = stream(stack, x, pieces)
    torch.testing.assert_close(y_stream, y_full, atol=2e-5, rtol=1e-4)


def test_stack_state_survives_save_and_load():
    """Stop after 35 tokens, write the state to bytes, load it, continue: same as never stopping."""
    stack = make_stack()
    x = torch.randn(1, 60, D)
    with torch.no_grad():
        y_full, _ = run(stack, x)
        y1, states = run(stack, x[:, :35])
        blob = io.BytesIO()
        torch.save(states, blob)
        blob.seek(0)
        states = torch.load(blob)
        y2, _ = run(stack, x[:, 35:], states)
    torch.testing.assert_close(torch.cat([y1, y2], dim=1), y_full, atol=2e-5, rtol=1e-4)


def test_delta_rule_state_is_constant_and_attention_cache_grows():
    """The point of the hybrid: delta-rule layers remember in a fixed-size state, attention pays per token."""
    stack = make_stack()
    with torch.no_grad():
        _, s_short = run(stack, torch.randn(1, 40, D))
        _, s_long = run(stack, torch.randn(1, 160, D))
    for block, a, b in zip(stack, s_short, s_long):
        m = block.mixer
        if isinstance(m, (KimiDeltaAttentionRef, GatedDeltaNetRef)):
            size = lambda st: sum(t.numel() for t in st.values())
            assert size(a) == size(b) == m.state_floats()
        elif isinstance(m, MLARef):
            assert (a["len"], b["len"]) == (40, 160)                        # one latent per token, forever
            assert m.cache_floats_per_token() == 8
        else:                                                               # windowed GQA: bounded
            assert a["k"].shape[2] <= 4 * 24 + 40 and b["k"].shape[2] <= 4 * 24 + 160
            assert m.cache_floats_per_token() == 2 * 2 * (D // 4)


def test_stack_backward_through_all_mixers():
    stack = make_stack().train()
    x = torch.randn(1, 48, D, requires_grad=True)
    y, _ = run(stack, x)
    y.pow(2).mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in stack.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@needs_mps
def test_stack_mps_prefill_and_decode_matches_cpu():
    """Prefill 40 tokens, decode 20 one by one on MPS (Metal Gram kernels, Metal step, in-place
    caches). Must equal one CPU call."""
    stack = make_stack()
    x = torch.randn(1, 60, D)
    with torch.no_grad():
        y_cpu, _ = run(stack, x)
        y_mps, _ = stream(stack.to("mps"), x.to("mps"), (40,) + (1,) * 20)
    torch.testing.assert_close(y_mps.cpu(), y_cpu, atol=2e-4, rtol=1e-3)
