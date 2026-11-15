"""Tests for the reference Gated DeltaNet (layers/SSMs/reference/).

The recurrent op is the ground truth: it is the update rule written out token by token.
Everything else must agree with it.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.SSMs.reference import (
    CausalDepthwiseConv,
    GatedDeltaNetRef,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="no MPS device")
ATOL = 1e-5


def _inputs(B=2, H=3, T=150, Dk=16, Dv=24, seed=0, device="cpu"):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)
    q = F.normalize(r(B, H, T, Dk), dim=-1)
    k = F.normalize(r(B, H, T, Dk), dim=-1)
    v = r(B, H, T, Dv)
    g = -F.softplus(r(B, H, T))                       # log alpha <= 0
    beta = torch.rand(B, H, T, generator=gen)
    S0 = r(B, H, Dk, Dv)
    return [x.to(device) for x in (q, k, v, g, beta, S0)]


@pytest.mark.parametrize("inverse", ["solve", "doubling"])
@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("T", [1, 63, 64, 150])
def test_chunk_matches_recurrent(inverse, chunk_size, T):
    q, k, v, g, beta, S0 = _inputs(T=T)
    o_r, S_r = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=S0)
    o_c, S_c = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=S0,
                                      chunk_size=chunk_size, inverse=inverse)
    assert o_c.shape == o_r.shape
    torch.testing.assert_close(o_c, o_r, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(S_c, S_r, atol=ATOL, rtol=1e-4)


def test_many_chunks_match_recurrent():
    """25 chunks: guards the affine chunk scan against error build-up over the sequence."""
    q, k, v, g, beta, S0 = _inputs(B=1, H=2, T=200, Dk=8, Dv=8)
    o_r, S_r = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=S0)
    o_c, S_c = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=S0, chunk_size=8)
    torch.testing.assert_close(o_c, o_r, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(S_c, S_r, atol=ATOL, rtol=1e-4)


def test_state_carries_across_calls():
    """Running two halves with the state handed over equals running the whole sequence."""
    q, k, v, g, beta, S0 = _inputs(T=128)
    o_full, S_full = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=S0)
    h = 70
    o1, S1 = chunk_gated_delta_rule(q[:, :, :h], k[:, :, :h], v[:, :, :h], g[:, :, :h],
                                    beta[:, :, :h], initial_state=S0)
    o2, S2 = chunk_gated_delta_rule(q[:, :, h:], k[:, :, h:], v[:, :, h:], g[:, :, h:],
                                    beta[:, :, h:], initial_state=S1)
    torch.testing.assert_close(torch.cat([o1, o2], dim=2), o_full, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(S2, S_full, atol=ATOL, rtol=1e-4)


def test_delta_rule_recalls_a_stored_pair():
    """beta = 1, no decay: writing (k, v) then reading with q = k returns v exactly."""
    Dk, Dv = 8, 5
    k = F.normalize(torch.randn(1, 1, 1, Dk), dim=-1)
    v = torch.randn(1, 1, 1, Dv)
    q = k.clone()
    g = torch.zeros(1, 1, 1)
    beta = torch.ones(1, 1, 1)
    S0 = torch.randn(1, 1, Dk, Dv)                    # junk already in memory
    o, _ = recurrent_gated_delta_rule(q, k, v, g, beta, scale=1.0, initial_state=S0)
    torch.testing.assert_close(o, v, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
def test_layer_is_causal(mode):
    torch.manual_seed(0)
    layer = GatedDeltaNetRef(d_model=32, n_heads=4, chunk_size=16, mode=mode)
    x = torch.randn(2, 40, 32)
    y = layer(x)
    x2 = x.clone()
    x2[:, 25] += 5.0
    y2 = layer(x2)
    assert y.shape == x.shape
    torch.testing.assert_close(y2[:, :25], y[:, :25])
    assert not torch.allclose(y2[:, 25:], y[:, 25:])


def test_layer_chunk_matches_recurrent():
    torch.manual_seed(0)
    layer = GatedDeltaNetRef(d_model=32, n_heads=4, chunk_size=16, mode="chunk")
    x = torch.randn(2, 50, 32)
    y_c = layer(x)
    layer.mode = "recurrent"
    y_r = layer(x)
    torch.testing.assert_close(y_c, y_r, atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("inverse", ["solve", "doubling"])
def test_layer_backward(inverse):
    torch.manual_seed(0)
    layer = GatedDeltaNetRef(d_model=32, n_heads=4, chunk_size=16, inverse=inverse)
    x = torch.randn(2, 50, 32, requires_grad=True)
    layer(x).pow(2).mean().backward()
    for name, p in [("x", x)] + list(layer.named_parameters()):
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_layer_gradients_match_recurrent():
    torch.manual_seed(0)
    layer = GatedDeltaNetRef(d_model=32, n_heads=4, chunk_size=16, mode="chunk")
    x = torch.randn(2, 40, 32)

    def grads(mode):
        layer.mode = mode
        layer.zero_grad()
        layer(x).pow(2).mean().backward()
        return {n: p.grad.clone() for n, p in layer.named_parameters()}

    g_c, g_r = grads("chunk"), grads("recurrent")
    for n in g_c:
        torch.testing.assert_close(g_c[n], g_r[n], atol=1e-5, rtol=1e-3, msg=n)


def test_conv_tails_match_single_call():
    torch.manual_seed(0)
    conv = CausalDepthwiseConv(6, kernel_size=4)
    x = torch.randn(2, 50, 6)
    y_full, tail_full = conv(x)
    ys, tail, i = [], None, 0
    for n in (17, 1, 2, 1, 29):                      # includes pieces shorter than the kernel
        y, tail = conv(x[:, i:i + n], tail)
        ys.append(y)
        i += n
    torch.testing.assert_close(torch.cat(ys, dim=1), y_full)
    torch.testing.assert_close(tail, tail_full)
    assert tail.shape == (2, 3, 6)


@pytest.mark.parametrize("mode", ["chunk", "recurrent", "auto"])
def test_layer_streaming_equals_single_call(mode):
    torch.manual_seed(0)
    layer = GatedDeltaNetRef(d_model=32, n_heads=4, chunk_size=16, mode=mode)
    x = torch.randn(2, 50, 32)
    y_full = layer(x)
    ys, state, i = [], None, 0
    for n in (17, 1, 1, 31):
        y, state = layer(x[:, i:i + n], state, return_state=True)
        ys.append(y)
        i += n
    torch.testing.assert_close(torch.cat(ys, dim=1), y_full, atol=ATOL, rtol=1e-4)
    assert set(state) == {"S", "conv_q", "conv_k", "conv_v"}


def test_layer_state_floats():
    layer = GatedDeltaNetRef(d_model=32, n_heads=4)
    _, state = layer(torch.randn(2, 5, 32), return_state=True)
    assert layer.state_floats() == sum(t[0].numel() for t in state.values())


@needs_mps
@pytest.mark.parametrize("inverse", ["solve", "doubling"])
def test_mps_matches_cpu(inverse):
    q, k, v, g, beta, S0 = _inputs(T=150)
    o_cpu, S_cpu = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=S0, inverse=inverse)
    mq, mk, mv, mg, mb, mS = (x.to("mps") for x in (q, k, v, g, beta, S0))
    o_mps, S_mps = chunk_gated_delta_rule(mq, mk, mv, mg, mb, initial_state=mS, inverse=inverse)
    torch.testing.assert_close(o_mps.cpu(), o_cpu, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(S_mps.cpu(), S_cpu, atol=1e-4, rtol=1e-3)


@needs_mps
def test_mps_layer_forward_backward():
    torch.manual_seed(0)
    layer = GatedDeltaNetRef(d_model=64, n_heads=4).to("mps")
    x = torch.randn(2, 200, 64, device="mps", requires_grad=True)
    y = layer(x)
    y.pow(2).mean().backward()
    assert y.device.type == "mps"
    assert torch.isfinite(x.grad).all()


def test_conv_cpu_taps_match_conv1d():
    """On CPU the conv is computed as a sum of shifted taps; it must equal nn.Conv1d."""
    from layers.SSMs.reference import CausalDepthwiseConv
    torch.manual_seed(0)
    conv = CausalDepthwiseConv(12, kernel_size=4)
    x = torch.randn(2, 30, 12)
    y, _ = conv(x)
    ref = conv.conv(F.pad(x.transpose(1, 2), (3, 0))).transpose(1, 2)
    torch.testing.assert_close(y, ref, atol=1e-6, rtol=1e-5)
