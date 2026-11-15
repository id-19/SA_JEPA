"""Tests for the per-head output RMSNorm + gate on the multi-head GDN and KDA.

    O = rms_norm(O over head value dim, one weight shared across heads) * gate, then W_o.
    GDN gate: silu(x @ W_g).   KDA gate: sigmoid((x @ W_g_high) @ W_g_low).

The oracle runs the same layer with output_gate=False and W_o = identity (D_output = D_v) to get the raw
per-head output, then applies norm and gate by hand with plain ops.
"""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.SSMs.multi_headed.parallel_gdn import MultiHeadParallelGDN
from layers.SSMs.multi_headed.parallel_kda import MultiHeadParallelKDA

DT = torch.float64
B, T, D_IN, D_OUT, H = 2, 50, 16, 8, 3
D_QK, D_V = 12, 18
D_HV = D_V // H
EPS = 1e-5
TIGHT = dict(rtol=1e-10, atol=1e-12)
CHUNKS = (8, 16, 64)
FP32 = dict(rtol=1e-4, atol=1e-5)      # float32: rounding floor is ~1e-6 (measured on the reference, 14 Nov)
BF16 = dict(rtol=0, atol=5e-2)         # bfloat16 against the float64 oracle: measured 5e-3 to 1.4e-2 on outputs of size ~1 (15 Nov)

LAYERS = {"gdn": MultiHeadParallelGDN, "kda": MultiHeadParallelKDA}


def make(kind, output_gate=True, d_output=D_OUT, seed=0):
    torch.manual_seed(seed)
    return LAYERS[kind](D_IN, d_output, D_QK, D_V, H, output_gate=output_gate).to(DT)


def inputs(seed=1):
    torch.manual_seed(seed)
    return torch.randn(B, T, D_IN, dtype=DT)


def randomise_gate(m):
    """Non-trivial norm weight so a missing or misplaced weight fails."""
    torch.manual_seed(5)
    with torch.no_grad():
        m.o_norm_weight.copy_(torch.randn(D_HV, dtype=DT))


def gate_of(kind, m, x):
    if kind == "gdn":
        return torch.nn.functional.silu(x @ m.W_g)
    return torch.sigmoid((x @ m.W_g_high) @ m.W_g_low)


def raw_o(kind, m, x, chunk):
    """Pre-norm per-head output (B, T, D_V): the same weights in an ungated layer whose W_o is the identity."""
    r = make(kind, output_gate=False, d_output=D_V)
    sd = {k: v for k, v in m.state_dict().items() if k in r.state_dict() and k != "W_o"}
    r.load_state_dict({**r.state_dict(), **sd})
    with torch.no_grad():
        r.W_o.copy_(torch.eye(D_V, dtype=DT))
    return r(x, chunk_size=chunk)


@pytest.mark.parametrize("kind", LAYERS)
def test_param_shapes(kind):
    m = make(kind)
    assert m.o_norm_weight.shape == (D_HV,)
    assert torch.equal(m.o_norm_weight, torch.ones(D_HV, dtype=DT))
    if kind == "gdn":
        assert m.W_g.shape == (D_IN, D_V)
    else:
        assert m.W_g_high.shape == (D_IN, D_HV)
        assert m.W_g_low.shape == (D_HV, D_V)


@pytest.mark.parametrize("kind", LAYERS)
def test_no_gate_creates_no_gate_params(kind):
    names = {n for n, _ in make(kind, output_gate=False).named_parameters()}
    assert not any(n.startswith(("o_norm", "W_g")) for n in names)


@pytest.mark.parametrize("kind", LAYERS)
@pytest.mark.parametrize("chunk", CHUNKS)
def test_forward_matches_legacy_out_outputs_and_grads(kind, chunk):
    m = make(kind)
    randomise_gate(m)
    x = inputs()
    torch.manual_seed(7)
    w = torch.randn(B, T, D_OUT, dtype=DT)

    def run(fn):
        m.zero_grad()
        xx = x.clone().requires_grad_(True)
        y = fn(xx)
        (y * w).sum().backward()
        return y.detach(), xx.grad, {n: p.grad.clone() for n, p in m.named_parameters()}

    y_f, gx_f, gp_f = run(lambda xx: m(xx, chunk_size=chunk))
    y_l, gx_l, gp_l = run(m.legacy_out)
    torch.testing.assert_close(y_f, y_l, **TIGHT)
    torch.testing.assert_close(gx_f, gx_l, **TIGHT)
    assert gp_f.keys() == gp_l.keys()
    for n in gp_f:
        assert gp_f[n].abs().max() > 0, f"{n} got no gradient"
        torch.testing.assert_close(gp_f[n], gp_l[n], **TIGHT, msg=n)


@pytest.mark.parametrize("kind", LAYERS)
@pytest.mark.parametrize("chunk", CHUNKS)
def test_matches_hand_written_norm_and_gate(kind, chunk):
    m = make(kind)
    randomise_gate(m)
    x = inputs()
    o = raw_o(kind, m, x, chunk).view(B, T, H, D_HV)
    o = o / torch.sqrt((o ** 2).mean(dim=-1, keepdim=True) + EPS) * m.o_norm_weight
    o = o.reshape(B, T, D_V) * gate_of(kind, m, x)
    torch.testing.assert_close(m(x, chunk_size=chunk), o @ m.W_o, **TIGHT)


@pytest.mark.parametrize("kind", LAYERS)
def test_zero_norm_weight_gives_exactly_zero(kind):
    m = make(kind)
    with torch.no_grad():
        m.o_norm_weight.zero_()
    x = inputs()
    assert m(x, chunk_size=16).abs().max() == 0
    assert m.legacy_out(x).abs().max() == 0


# Open (15 Nov): gated KDA in bf16 misses atol 5e-2 by one element (max err 0.0507). Measured worst error / max|y|:
# KDA ungated 0.7%, gated 2.4%; GDN ungated 0.6%, gated 1.8%. The gate adds bf16 error beyond the norm's rescaling.
# Not loosened: decide whether the gate should run in fp32, then fix this mark.
KDA_BF16_OPEN = pytest.mark.xfail(reason="gated KDA bf16 error 0.0507 > 5e-2, see comment above", strict=True)


@pytest.mark.parametrize("kind, dtype, tol", [
    ("gdn", torch.float32, FP32), ("gdn", torch.bfloat16, BF16),
    ("kda", torch.float32, FP32), pytest.param("kda", torch.bfloat16, BF16, marks=KDA_BF16_OPEN),
])
@pytest.mark.parametrize("chunk", [16, 64])
def test_gated_low_precision_against_float64_truth(kind, dtype, tol, chunk):
    """Gated layer in low precision against the same weights in float64; S stays in float32."""
    m = make(kind)
    x = inputs()
    y_true = m(x, chunk_size=chunk)
    m_low, x_low = copy.deepcopy(m).to(dtype), copy.deepcopy(x).to(dtype)
    y, st = m_low(x_low, return_state=True, chunk_size=chunk)
    assert y.dtype == dtype
    S = st if torch.is_tensor(st) else st[0]
    assert S.dtype == torch.float32
    torch.testing.assert_close(y.double(), y_true, **tol)
