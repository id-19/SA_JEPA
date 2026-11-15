"""Tests for the Metal kernels (layers/SSMs/reference/metal_kernels.py).

The PyTorch ops in kda_ops.py are the ground truth; the kernels must agree with them.
Everything is small (batch <= 2) because the machine has 8 GB of RAM.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.SSMs.reference.kda_ops import _decayed_grams_dc, kda_step
from layers.SSMs.reference.metal_kernels import decayed_grams_metal, kda_step_metal

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="no MPS device")


def _gram_inputs(shape, seed=0, device="mps", requires_grad=False):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(*shape, generator=gen)
    k = torch.randn(*shape, generator=gen)
    c = (-F.softplus(torch.randn(*shape, generator=gen))).cumsum(-2)
    out = [x.to(device) for x in (q, k, c)]
    for x in out:
        x.requires_grad_(requires_grad)
    return out


def _close(a, b, atol=1e-5, rtol=1e-4):
    torch.testing.assert_close(a.cpu(), b.cpu(), atol=atol, rtol=rtol)


@needs_mps
@pytest.mark.parametrize("C", [2, 16, 64])
@pytest.mark.parametrize("D", [5, 32])
def test_forward_matches_reference(C, D):
    q, k, c = _gram_inputs((2, 3, 4, C, D))
    M, P = decayed_grams_metal(q, k, c)
    Mr, Pr = _decayed_grams_dc(q, k, c)
    _close(M, Mr)
    _close(P, Pr)
    # and the same maths on CPU
    Mc, Pc = _decayed_grams_dc(q.cpu(), k.cpu(), c.cpu())
    _close(M, Mc)
    _close(P, Pc)


@needs_mps
def test_forward_large_dim():
    q, k, c = _gram_inputs((1, 2, 2, 64, 128), seed=1)
    M, P = decayed_grams_metal(q, k, c)
    Mr, Pr = _decayed_grams_dc(q, k, c)
    _close(M, Mr)
    _close(P, Pr)


@needs_mps
def test_extreme_decay():
    gen = torch.Generator().manual_seed(2)
    C, D = 64, 32
    q = torch.randn(2, C, D, generator=gen)
    k = torch.randn(2, C, D, generator=gen)
    g = torch.zeros(2, C, D)
    g[..., ::3] = -40.0                                           # some channels forget almost at once
    c = g.cumsum(-2)
    q, k, c = q.to("mps"), k.to("mps"), c.to("mps")
    M, P = decayed_grams_metal(q, k, c)
    assert torch.isfinite(M).all() and torch.isfinite(P).all()
    Mr, Pr = _decayed_grams_dc(q, k, c)
    _close(M, Mr)
    _close(P, Pr)


def _weighted_grads(fn, q, k, c, wM, wP):
    for x in (q, k, c):
        x.grad = None
    M, P = fn(q, k, c)
    (M * wM).sum().add((P * wP).sum()).backward()
    return q.grad.clone(), k.grad.clone(), c.grad.clone()


@needs_mps
@pytest.mark.parametrize("C", [2, 16, 64])
def test_backward_matches_autograd(C):
    q, k, c = _gram_inputs((2, 2, C, 16), seed=3, requires_grad=True)
    gen = torch.Generator().manual_seed(4)
    # weights on the upper triangle and M's diagonal too: they must be ignored
    wM = torch.randn(2, 2, C, C, generator=gen).to("mps")
    wP = torch.randn(2, 2, C, C, generator=gen).to("mps")
    got = _weighted_grads(decayed_grams_metal, q, k, c, wM, wP)
    ref = _weighted_grads(_decayed_grams_dc, q, k, c, wM, wP)
    for a, b in zip(got, ref):
        _close(a, b, atol=1e-5, rtol=1e-3)


@needs_mps
def test_slabs_match_default_budget():
    q, k, c = _gram_inputs((7, 16, 32), seed=5, requires_grad=True)
    gen = torch.Generator().manual_seed(6)
    wM = torch.randn(7, 16, 16, generator=gen).to("mps")
    wP = torch.randn(7, 16, 16, generator=gen).to("mps")
    # 0.01 MB = 2621 floats; one slab row needs 2*4*16*32 = 4096 -> slab size 1
    tiny = lambda q, k, c: decayed_grams_metal(q, k, c, table_budget_mb=0.01)
    # 0.05 MB = 13107 floats -> slab size 3 (7 = 3 + 3 + 1, so the last slab is smaller)
    mid = lambda q, k, c: decayed_grams_metal(q, k, c, table_budget_mb=0.05)
    base = _weighted_grads(decayed_grams_metal, q, k, c, wM, wP)
    for fn in (tiny, mid):
        _close(fn(q, k, c)[0], decayed_grams_metal(q, k, c)[0], atol=1e-6)
        _close(fn(q, k, c)[1], decayed_grams_metal(q, k, c)[1], atol=1e-6)
        for a, b in zip(_weighted_grads(fn, q, k, c, wM, wP), base):
            _close(a, b, atol=1e-6)


@needs_mps
def test_non_contiguous_inputs():
    q, k, c = _gram_inputs((2, 3, 16, 8), seed=7)
    # transposed storage: same values, but the views are not contiguous
    qt, kt, ct = (x.transpose(0, 1).contiguous().transpose(0, 1) for x in (q, k, c))
    assert not qt.is_contiguous()
    M, P = decayed_grams_metal(qt, kt, ct)
    Mr, Pr = _decayed_grams_dc(q, k, c)
    _close(M, Mr)
    _close(P, Pr)
    # a slice of the last dim is also non-contiguous
    wide = torch.randn(2, 16, 16, device="mps")
    k2 = wide[..., :8]
    M, P = decayed_grams_metal(q[0, :2], k2, c[0, :2])
    Mr, Pr = _decayed_grams_dc(q[0, :2], k2.contiguous(), c[0, :2])
    _close(M, Mr)
    _close(P, Pr)


def _step_inputs(B, H, Dk, Dv, T, seed=0):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)
    steps = []
    for _ in range(T):
        steps.append((F.normalize(r(B, H, Dk), dim=-1), F.normalize(r(B, H, Dk), dim=-1),
                      r(B, H, Dv), -F.softplus(r(B, H, Dk)), torch.rand(B, H, generator=gen)))
    return steps, r(B, H, Dk, Dv)


@needs_mps
@pytest.mark.parametrize("Dk,Dv", [(16, 24), (24, 16)])
def test_step_chained(Dk, Dv):
    steps, S0 = _step_inputs(2, 3, Dk, Dv, 20)
    S_ref = S0.clone()
    S = S0.to("mps").contiguous()
    S_obj = S
    for q, k, v, g, beta in steps:
        o_ref, S_ref = kda_step(q, k, v, g, beta, S_ref)
        o, S = kda_step_metal(*(x.to("mps") for x in (q, k, v, g, beta)), S)
        assert S is S_obj                                         # updated in place
        _close(o, o_ref, atol=1e-5)
    _close(S, S_ref, atol=1e-5)


@needs_mps
def test_step_scalar_decay():
    steps, S0 = _step_inputs(2, 3, 16, 24, 1, seed=1)
    q, k, v, g, beta = steps[0]
    g_scalar = g[..., 0].contiguous()                             # (B, H)
    o_ref, S_ref = kda_step(q, k, v, g_scalar[..., None].expand(-1, -1, 16), beta, S0.clone())
    S = S0.to("mps").contiguous()
    o, S = kda_step_metal(*(x.to("mps") for x in (q, k, v, g_scalar, beta)), S)
    _close(o, o_ref, atol=1e-5)
    _close(S, S_ref, atol=1e-5)


@needs_mps
def test_step_half_v_gives_half_o():
    steps, S0 = _step_inputs(2, 3, 16, 24, 1, seed=2)
    q, k, v, g, beta = (x.to("mps") for x in steps[0])
    o, S = kda_step_metal(q, k, v.half(), g, beta, S0.to("mps").contiguous())
    assert o.dtype == torch.float16
    assert S.dtype == torch.float32


@needs_mps
def test_step_rejects_bad_state():
    steps, S0 = _step_inputs(2, 3, 16, 24, 1, seed=3)
    q, k, v, g, beta = (x.to("mps") for x in steps[0])
    with pytest.raises(ValueError):
        kda_step_metal(q, k, v, g, beta, S0.to("mps").half())
    with pytest.raises(ValueError):
        kda_step_metal(q, k, v, g, beta, S0.to("mps").transpose(-1, -2))


# ---- the kernels as wired into the ops and layers ----------------------------------------

def _seq(B=1, H=2, T=40, Dk=16, Dv=12, seed=0, scalar_g=False):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)
    q = F.normalize(r(B, H, T, Dk), dim=-1)
    k = F.normalize(r(B, H, T, Dk), dim=-1)
    v = r(B, H, T, Dv)
    g = -F.softplus(r(B, H, T) if scalar_g else r(B, H, T, Dk))
    beta = torch.rand(B, H, T, generator=gen)
    S0 = r(B, H, Dk, Dv)
    return q, k, v, g, beta, S0


@needs_mps
def test_recurrent_kda_uses_metal_step_and_matches_cpu():
    from layers.SSMs.reference.kda_ops import recurrent_kda
    q, k, v, g, beta, S0 = _seq()
    o_cpu, S_cpu = recurrent_kda(q, k, v, g, beta, initial_state=S0)
    m = [x.to("mps") for x in (q, k, v, g, beta, S0)]
    S0_before = m[5].clone()
    o_mps, S_mps = recurrent_kda(*m[:5], initial_state=m[5])        # no grad needed -> Metal step
    torch.testing.assert_close(o_mps.cpu(), o_cpu, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(S_mps.cpu(), S_cpu, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(m[5], S0_before)                     # the caller's state is not written
    assert S_mps.data_ptr() != m[5].data_ptr()


@needs_mps
def test_recurrent_gdn_uses_metal_step_and_matches_cpu():
    from layers.SSMs.reference import recurrent_gated_delta_rule
    q, k, v, g, beta, S0 = _seq(scalar_g=True)
    o_cpu, S_cpu = recurrent_gated_delta_rule(q, k, v, g, beta, initial_state=S0)
    m = [x.to("mps") for x in (q, k, v, g, beta, S0)]
    S0_before = m[5].clone()
    o_mps, S_mps = recurrent_gated_delta_rule(*m[:5], initial_state=m[5])
    torch.testing.assert_close(o_mps.cpu(), o_cpu, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(S_mps.cpu(), S_cpu, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(m[5], S0_before)


@needs_mps
def test_recurrent_on_mps_keeps_gradients_when_needed():
    """With a gradient required the Metal step (no autograd) must not be taken."""
    from layers.SSMs.reference.kda_ops import recurrent_kda
    q, k, v, g, beta, S0 = (x.to("mps") for x in _seq(T=6))
    v.requires_grad_()
    o, _ = recurrent_kda(q, k, v, g, beta, initial_state=S0)
    o.sum().backward()
    assert v.grad is not None and torch.isfinite(v.grad).all()


@needs_mps
@pytest.mark.parametrize("T", [70, 128])
def test_chunk_kda_metal_gram_matches_lean(T):
    from layers.SSMs.reference.kda_ops import chunk_kda
    q, k, v, g, beta, S0 = (x.to("mps") for x in _seq(T=T))

    def run(gram):
        leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
        o, S = chunk_kda(*leaves, initial_state=S0, chunk_size=32, gram=gram)
        (o.pow(2).sum() + S.sum()).backward()
        return o.detach(), S.detach(), [x.grad for x in leaves]

    o_m, S_m, g_m = run("metal")
    o_l, S_l, g_l = run("lean")
    torch.testing.assert_close(o_m, o_l, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(S_m, S_l, atol=1e-5, rtol=1e-4)
    for a, b in zip(g_m, g_l):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-3)


@needs_mps
@pytest.mark.parametrize("cls", ["kda", "gdn"])
def test_layer_metal_decode_matches_cpu_single_call(cls):
    """Prefill 30 tokens (chunked, Metal Grams), then decode 20 one by one (Metal step), on MPS
    under no_grad. Must equal one CPU call on all 50."""
    from layers.SSMs.reference import GatedDeltaNetRef, KimiDeltaAttentionRef
    torch.manual_seed(0)
    layer = (KimiDeltaAttentionRef if cls == "kda" else GatedDeltaNetRef)(32, 2, chunk_size=16)
    x = torch.randn(1, 50, 32)
    with torch.no_grad():
        y_cpu = layer(x)
        layer = layer.to("mps")
        xm = x.to("mps")
        y, state = layer(xm[:, :30], return_state=True)
        outs = [y]
        for t in range(30, 50):
            y, state = layer(xm[:, t:t + 1], state=state, return_state=True)
            outs.append(y)
    torch.testing.assert_close(torch.cat(outs, 1).cpu(), y_cpu, atol=1e-4, rtol=1e-3)
