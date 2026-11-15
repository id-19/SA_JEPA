"""Tests for the residual block (v0_model_try/model/block.py).

Run:  uv run pytest eval_test/tests/test_block.py -v

Setup: float64, CPU, B=2, T=50, D=16, d_ff=64.

    test_output_shape          output has the same shape as the input (KDA, GDN, attention mixers)
    test_return_state_flag     return_state=False gives a tensor; True gives (tensor, state)
    test_zero_modules_identity with a zero mixer and zero MLP the block returns x exactly, also for 100*x
    test_mixer_input_normed    the mixer sees per-token RMS ~1, and the same thing when x is scaled by 100
    test_mlp_input             the MLP sees the unit-weight RMS-normalised (x + mixer output)
    test_output_sum            with fixed mixer output h and MLP output m, the block returns exactly x + h + m
    test_norms                 norm1 and norm2 are separate modules, each with one weight (D,) of ones
    test_all_grads_nonzero     every parameter gets a non-zero gradient (real KDA mixer)
    test_causal                changing x[:, 20:] does not change out[:, :20] (KDA, GDN)
    test_streaming             feeding pieces with the state passed along equals one call (KDA, GDN)
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v0_model_try.model.block import Block
from layers.SSMs.multi_headed.parallel_kda import MultiHeadParallelKDA
from layers.SSMs.multi_headed.parallel_gdn import MultiHeadParallelGDN
from layers.Attention_layers.reference import GQARef

DT = torch.float64
B, T, D, D_FF = 2, 50, 16, 64
NORM_TOL = dict(rtol=1e-4, atol=1e-4)


def make_mixer(kind):
    torch.manual_seed(0)
    if kind == "kda":
        m = MultiHeadParallelKDA(D, D, 8, 8, 2)
    elif kind == "gdn":
        m = MultiHeadParallelGDN(D, D, 8, 8, 2)
    else:
        m = GQARef(D, 2)
    return m.to(DT)


def make_block(kind):
    torch.manual_seed(1)
    return Block(D, make_mixer(kind), D_FF).to(DT)


def rand_x(seed=2):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, D, generator=g, dtype=DT)


def rms(t):
    return t.pow(2).mean(-1).sqrt()


def unit_rms_norm(t):
    return F.rms_norm(t, (t.shape[-1],))


class SpyMixer(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h, self.seen, self.state_in = h, [], []

    def forward(self, x, state=None, return_state=False, **kw):
        self.seen.append(x.detach().clone())
        self.state_in.append(state)
        y = self.h.expand_as(x)
        return (y, state) if return_state else y


class SpyMLP(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m, self.seen = m, []

    def forward(self, x):
        self.seen.append(x.detach().clone())
        return self.m.expand_as(x)


def spy_block(h, m):
    blk = make_block("kda")
    blk.mixer = SpyMixer(h)
    blk.mlp = SpyMLP(m)
    return blk


@pytest.mark.parametrize("kind", ["kda", "gdn", "gqa"])
def test_output_shape(kind):
    x = rand_x()
    assert make_block(kind)(x).shape == x.shape


@pytest.mark.parametrize("kind", ["kda", "gdn", "gqa"])
def test_return_state_flag(kind):
    blk, x = make_block(kind), rand_x()
    y = blk(x)
    assert isinstance(y, torch.Tensor)
    out = blk(x, return_state=True)
    assert isinstance(out, tuple) and len(out) == 2
    assert isinstance(out[0], torch.Tensor) and out[0].shape == x.shape
    assert out[1] is not None


def test_zero_modules_identity():
    blk = make_block("kda")
    blk.mixer = SpyMixer(torch.zeros(1, 1, D, dtype=DT))
    blk.mlp = SpyMLP(torch.zeros(1, 1, D, dtype=DT))
    x = rand_x()
    assert torch.equal(blk(x), x)
    assert torch.equal(blk(100 * x), 100 * x)


def test_mixer_input_normed():
    blk = spy_block(torch.zeros(1, 1, D, dtype=DT), torch.zeros(1, 1, D, dtype=DT))
    x = rand_x()
    blk(x)
    blk(100 * x)
    a, b = blk.mixer.seen
    assert a.shape == x.shape
    torch.testing.assert_close(rms(a), torch.ones(B, T, dtype=DT), **NORM_TOL)
    torch.testing.assert_close(b, a, **NORM_TOL)


def test_mlp_input():
    h = torch.randn(B, T, D, generator=torch.Generator().manual_seed(3), dtype=DT)
    blk = spy_block(h, torch.zeros(1, 1, D, dtype=DT))
    x = rand_x()
    blk(x)
    seen = blk.mlp.seen[0]
    torch.testing.assert_close(seen, unit_rms_norm(x + h), **NORM_TOL)
    # not a norm of x alone
    assert not torch.allclose(seen, unit_rms_norm(x), rtol=1e-2, atol=1e-2)


def test_output_sum():
    g = torch.Generator().manual_seed(4)
    h = torch.randn(B, T, D, generator=g, dtype=DT)
    m = torch.randn(B, T, D, generator=g, dtype=DT)
    blk = spy_block(h, m)
    x = rand_x()
    torch.testing.assert_close(blk(x), x + h + m, rtol=1e-12, atol=1e-12)


def test_norms():
    blk = make_block("kda")
    assert blk.norm1 is not blk.norm2
    for n in (blk.norm1, blk.norm2):
        ps = list(n.parameters())
        assert len(ps) == 1
        assert ps[0].shape == (D,)
        assert torch.equal(ps[0].detach(), torch.ones(D, dtype=DT))
    assert set(map(id, blk.norm1.parameters())).isdisjoint(map(id, blk.norm2.parameters()))


def test_all_grads_nonzero():
    blk = make_block("kda")
    blk(rand_x()).pow(2).sum().backward()
    for name, p in blk.named_parameters():
        assert p.grad is not None, f"{name}: no grad"
        assert p.grad.abs().max() > 0, f"{name}: zero grad"


@pytest.mark.parametrize("kind", ["kda", "gdn"])
def test_causal(kind):
    blk = make_block(kind)
    x = rand_x()
    x2 = x.clone()
    x2[:, 20:] = torch.randn_like(x2[:, 20:])
    with torch.no_grad():
        a, b = blk(x), blk(x2)
    assert torch.allclose(a[:, :20], b[:, :20], rtol=1e-10, atol=1e-10)
    assert not torch.allclose(a[:, 20:], b[:, 20:])


@pytest.mark.parametrize("kind", ["kda", "gdn"])
@pytest.mark.parametrize("pieces", [(30, 20), (17, 1, 1, 31)])
def test_streaming(kind, pieces):
    blk = make_block(kind)
    x = rand_x()
    with torch.no_grad():
        full = blk(x)
        state, outs, i = None, [], 0
        for n in pieces:
            y, state = blk(x[:, i:i + n], state=state, return_state=True)
            outs.append(y)
            i += n
    assert i == T
    torch.testing.assert_close(torch.cat(outs, 1), full, rtol=1e-10, atol=1e-10)
