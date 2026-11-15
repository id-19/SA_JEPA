"""Tests for the reference Kimi Delta Attention layer (layers/SSMs/reference/kda.py).

The recurrent mode is the ground truth; chunk mode, streaming and devices must agree with it.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.SSMs.reference import KimiDeltaAttentionRef

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="no MPS device")
ATOL = 1e-5

CONFIGS = {
    "plain": dict(d_model=32, n_heads=4),
    "grouped": dict(d_model=32, n_heads=2, n_v_heads=4),
    "expand_v": dict(d_model=32, n_heads=4, expand_v=2.0),
}


def _layer(seed=0, **kw):
    torch.manual_seed(seed)
    kw = {"chunk_size": 16, **kw}
    return KimiDeltaAttentionRef(**kw)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_shape_and_chunk_matches_recurrent(cfg):
    layer = _layer(**CONFIGS[cfg], mode="chunk")
    x = torch.randn(2, 50, 32)
    y_c = layer(x)
    layer.mode = "recurrent"
    y_r = layer(x)
    assert y_c.shape == x.shape
    torch.testing.assert_close(y_c, y_r, atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
def test_layer_is_causal(mode):
    layer = _layer(d_model=32, n_heads=4, mode=mode)
    x = torch.randn(2, 40, 32)
    y = layer(x)
    x2 = x.clone()
    x2[:, 25] += 5.0
    y2 = layer(x2)
    torch.testing.assert_close(y2[:, :25], y[:, :25])
    assert not torch.allclose(y2[:, 25:], y[:, 25:])


@pytest.mark.parametrize("mode", ["chunk", "recurrent", "auto"])
def test_streaming_equals_single_call(mode):
    layer = _layer(**CONFIGS["grouped"], mode=mode)
    x = torch.randn(2, 50, 32)
    y_full = layer(x)
    ys, state, i = [], None, 0
    for n in (17, 1, 1, 31):
        y, state = layer(x[:, i:i + n], state, return_state=True)
        ys.append(y)
        i += n
    torch.testing.assert_close(torch.cat(ys, dim=1), y_full, atol=ATOL, rtol=1e-4)


def test_token_by_token_streaming():
    layer = _layer(d_model=32, n_heads=4)
    x = torch.randn(2, 20, 32)
    y_full = layer(x)
    ys, state = [], None
    for t in range(20):
        y, state = layer(x[:, t:t + 1], state, return_state=True)
        ys.append(y)
    torch.testing.assert_close(torch.cat(ys, dim=1), y_full, atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_state_shapes_and_floats(cfg):
    layer = _layer(**CONFIGS[cfg])
    _, state = layer(torch.randn(2, 10, 32), return_state=True)
    key_dim = layer.n_heads * layer.dk
    value_dim = layer.n_v_heads * layer.dv
    assert set(state) == {"S", "conv_q", "conv_k", "conv_v"}
    assert state["S"].shape == (2, layer.n_v_heads, layer.dk, layer.dv)
    assert state["conv_q"].shape == (2, 3, key_dim)
    assert state["conv_k"].shape == (2, 3, key_dim)
    assert state["conv_v"].shape == (2, 3, value_dim)
    assert layer.state_floats() == sum(t[0].numel() for t in state.values())


@pytest.mark.parametrize("gram", ["lean", "autograd"])
def test_backward_finite(gram):
    layer = _layer(d_model=32, n_heads=4, mode="chunk", gram=gram)
    x = torch.randn(2, 50, 32, requires_grad=True)
    layer(x).pow(2).mean().backward()
    for name, p in [("x", x)] + list(layer.named_parameters()):
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name


def test_gradients_match_recurrent():
    layer = _layer(**CONFIGS["grouped"], mode="chunk")
    x = torch.randn(2, 40, 32)

    def grads(mode):
        layer.mode = mode
        layer.zero_grad()
        layer(x).pow(2).mean().backward()
        return {n: p.grad.clone() for n, p in layer.named_parameters()}

    g_c, g_r = grads("chunk"), grads("recurrent")
    for n in g_c:
        torch.testing.assert_close(g_c[n], g_r[n], atol=1e-5, rtol=1e-3, msg=n)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_parameter_count(cfg):
    layer = _layer(**CONFIGS[cfg])
    d, H, Hv, dk, dv, k = 32, layer.n_heads, layer.n_v_heads, layer.dk, layer.dv, 4
    key_dim, value_dim, gate_dim = H * dk, Hv * dv, Hv * dk
    expected = (
        2 * d * key_dim + d * value_dim                 # q_proj, k_proj, v_proj
        + k * (2 * key_dim + value_dim)                 # three depthwise convs, one k-tap filter per channel
        + d * dv + dv * gate_dim                        # f_proj (low rank)
        + Hv + gate_dim                                 # A_log, dt_bias
        + d * Hv                                        # b_proj
        + d * dv + dv * value_dim + value_dim           # g_proj (low rank, bias on the second)
        + dv                                            # o_norm_weight
        + value_dim * d                                 # o_proj
    )
    assert sum(p.numel() for p in layer.parameters()) == expected


@pytest.mark.parametrize("neg", [False, True])
def test_gate_ranges(neg):
    layer = _layer(d_model=32, n_heads=4, allow_neg_eigval=neg)
    x = torch.randn(2, 30, 32) * 3
    g, beta = layer.gates(x)
    assert g.shape == (2, 4, 30, 8) and beta.shape == (2, 4, 30)
    assert (g <= 0).all() and torch.isfinite(g).all()
    assert (beta > 0).all() and (beta < (2 if neg else 1)).all()


def _half_pair(device):
    layer = _layer(d_model=32, n_heads=4, mode="chunk").to(device)
    x = torch.randn(2, 40, 32, device=device)
    with torch.no_grad():
        y32 = layer(x)
    layer16 = _layer(d_model=32, n_heads=4, mode="chunk").to(device).half()
    return layer16, x, y32


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=needs_mps)])
def test_float16(device):
    layer16, x, y32 = _half_pair(device)
    try:
        with torch.no_grad():
            y16 = layer16(x.half())
    except (RuntimeError, NotImplementedError) as e:
        pytest.skip(f"float16 unsupported on {device}: {e}")
    assert y16.dtype == torch.float16
    assert torch.isfinite(y16).all()
    torch.testing.assert_close(y16.float(), y32, atol=2e-2, rtol=2e-2)


@needs_mps
def test_mps_matches_cpu_and_backward():
    layer = _layer(**CONFIGS["grouped"], mode="chunk")
    x = torch.randn(2, 50, 32)
    with torch.no_grad():
        y_cpu = layer(x)
    layer = layer.to("mps")
    xm = x.to("mps").requires_grad_()
    y = layer(xm)
    torch.testing.assert_close(y.detach().cpu(), y_cpu, atol=1e-4, rtol=1e-3)
    y.pow(2).mean().backward()
    assert torch.isfinite(xm.grad).all()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
