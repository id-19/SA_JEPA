"""Tests for carrying state between calls on the multi-head GDN (mirrors the state tests in test_kda_conv_state.py).

The state is the tensor S, (B, H, head_dim_qk, head_dim_v). GDN has no conv, so nothing else is carried.
Run for both output_gate settings: the norm and gate are per token, so they must not change the carry.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.SSMs.multi_headed.parallel_gdn import MultiHeadParallelGDN

DT = torch.float64
B, T, D_IN, D_OUT, H = 2, 50, 16, 8, 3
D_QK, D_V = 12, 18
TIGHT = dict(rtol=1e-9, atol=1e-11)
PIECES = [(50,), (1,) * 50, (7, 13, 30), (32, 18)]
GATES = [False, True]


def make(output_gate, seed=0, dtype=DT):
    torch.manual_seed(seed)
    return MultiHeadParallelGDN(D_IN, D_OUT, D_QK, D_V, H, output_gate=output_gate).to(dtype)


def inputs(seed=1, t=T, dtype=DT):
    torch.manual_seed(seed)
    return torch.randn(B, t, D_IN, dtype=dtype)


def stream(m, x, pieces, chunk=16, state=None):
    ys, t = [], 0
    for n in pieces:
        y, state = m(x[:, t:t + n], state=state, return_state=True, chunk_size=chunk)
        ys.append(y)
        t += n
    assert t == x.shape[1]
    return torch.cat(ys, dim=1), state


def readout_loss(y):
    torch.manual_seed(99)
    return (y * torch.randn_like(y)).sum()


def grads(m, x, loss_fn):
    m.zero_grad()
    x = x.clone().requires_grad_(True)
    loss_fn(m, x).backward()
    return x.grad, {n: p.grad.clone() for n, p in m.named_parameters()}


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("pieces", PIECES)
def test_stream_outputs_and_state_match_one_call(pieces, gate):
    m, x = make(gate), inputs()
    y_one, s_one = m(x, return_state=True, chunk_size=16)
    y, s = stream(m, x, pieces)
    torch.testing.assert_close(y, y_one, **TIGHT)
    torch.testing.assert_close(s, s_one, **TIGHT)


@pytest.mark.parametrize("gate", GATES)
def test_return_state_false_returns_only_y(gate):
    m, x = make(gate), inputs()
    y = m(x)
    assert torch.is_tensor(y) and y.shape == (B, T, D_OUT)


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("pieces", [(25, 25), (7, 13, 30)])
def test_stream_gradients_match_one_call(pieces, gate):
    m, x = make(gate), inputs()
    gx, gp = grads(m, x, lambda m, x: readout_loss(stream(m, x, pieces)[0]))
    ox, op = grads(m, x, lambda m, x: readout_loss(m(x, chunk_size=16)))
    torch.testing.assert_close(gx, ox, **TIGHT)
    assert gp.keys() == op.keys()
    for n in gp:
        torch.testing.assert_close(gp[n], op[n], **TIGHT, msg=n)


@pytest.mark.parametrize("gate", GATES)
def test_detached_state_cuts_the_gradient_exactly(gate):
    m = make(gate)
    x = inputs().requires_grad_(True)
    _, s = m(x[:, :30], return_state=True, chunk_size=16)
    y2, _ = m(x[:, 30:], state=s.detach(), return_state=True, chunk_size=16)
    readout_loss(y2).backward()
    assert x.grad[:, :30].abs().max() == 0
    assert x.grad[:, 30:].abs().max() > 0


@pytest.mark.parametrize("gate", GATES)
@pytest.mark.parametrize("pieces", [(25, 25), (17, 1, 1, 31)])
def test_legacy_out_continues_with_state(pieces, gate):
    m, x = make(gate), inputs()
    y_one, s_one = m(x, return_state=True, chunk_size=16)
    ys, s, t = [], None, 0
    for n in pieces:
        y, s = m.legacy_out(x[:, t:t + n], state=s, return_state=True)
        ys.append(y)
        t += n
    torch.testing.assert_close(torch.cat(ys, dim=1), y_one, **TIGHT)
    torch.testing.assert_close(s, s_one, **TIGHT)


@pytest.mark.parametrize("gate", GATES)
def test_legacy_state_feeds_forward_and_back(gate):
    """Mixed streaming: legacy_out for the first piece, forward for the second."""
    m, x = make(gate), inputs()
    y_one = m(x, chunk_size=16)
    y1, s = m.legacy_out(x[:, :20], return_state=True)
    y2 = m(x[:, 20:], state=s, chunk_size=16)
    torch.testing.assert_close(torch.cat((y1, y2), dim=1), y_one, **TIGHT)


@pytest.mark.parametrize("gate", GATES)
def test_state_size_is_fixed(gate):
    m = make(gate)
    for t in (5, 50, 200):
        _, s = m(inputs(t=t), return_state=True, chunk_size=16)
        assert s.shape == (B, H, D_QK // H, D_V // H), f"T={t}"


@pytest.mark.parametrize("gate", GATES)
def test_bfloat16_runs_and_keeps_the_state_in_float32(gate):
    m, x = make(gate, dtype=torch.bfloat16), inputs(dtype=torch.bfloat16)
    y, s = m(x, return_state=True, chunk_size=16)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()
    assert s.dtype == torch.float32
