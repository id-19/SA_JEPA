"""Tests for the reference Kimi Delta Attention ops (layers/SSMs/reference/kda_ops.py).

The recurrent op is the ground truth: it is the update rule written out token by token.
Everything else must agree with it.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.SSMs.reference.gated_delta_rule import recurrent_gated_delta_rule
from layers.SSMs.reference.kda_ops import (
    _decayed_grams_dc,
    _decayed_grams_naive,
    chunk_kda,
    decayed_grams,
    kda_step,
    recurrent_kda,
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
    g = -F.softplus(r(B, H, T, Dk))                   # log alpha <= 0, per channel
    beta = torch.rand(B, H, T, generator=gen)
    S0 = r(B, H, Dk, Dv)
    return [x.to(device) for x in (q, k, v, g, beta, S0)]


def _gram_inputs(shape, seed=0, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(*shape, generator=gen, dtype=dtype)
    k = torch.randn(*shape, generator=gen, dtype=dtype)
    c = (-F.softplus(torch.randn(*shape, generator=gen, dtype=dtype))).cumsum(-2)
    return q, k, c


@pytest.mark.parametrize("C", [2, 16, 64])
def test_grams_dc_and_lean_match_naive(C):
    q, k, c = _gram_inputs((2, 3, 4, C, 8))
    M_n, P_n = _decayed_grams_naive(q, k, c)
    M_d, P_d = _decayed_grams_dc(q, k, c)
    M_l, P_l = decayed_grams(q, k, c)
    torch.testing.assert_close(M_d, M_n, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(P_d, P_n, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(M_l, M_d)
    torch.testing.assert_close(P_l, P_d)


def test_gram_structure():
    q, k, c = _gram_inputs((1, 16, 4))
    M, P = decayed_grams(q, k, c)
    assert torch.equal(M, M.tril(-1))                 # strictly lower
    assert torch.equal(P, P.tril())                   # lower incl. diagonal
    torch.testing.assert_close(P.diagonal(dim1=-2, dim2=-1), (q * k).sum(-1))


def test_gradcheck_decayed_grams():
    q, k, c = _gram_inputs((8, 5), dtype=torch.float64)
    q, k, c = (x.requires_grad_() for x in (q, k, c))
    assert torch.autograd.gradcheck(decayed_grams, (q, k, c))
    qb, kb, cb = _gram_inputs((2, 3, 8, 5), seed=1, dtype=torch.float64)
    qb, kb, cb = (x.requires_grad_() for x in (qb, kb, cb))
    assert torch.autograd.gradcheck(decayed_grams, (qb, kb, cb))


def test_lean_gradients_match_autograd_gradients():
    gen = torch.Generator().manual_seed(2)
    C = 16
    wM = torch.randn(2, 3, C, C, generator=gen)       # weights every entry, upper triangle too
    wP = torch.randn(2, 3, C, C, generator=gen)

    def grads(fn):
        q, k, c = (x.requires_grad_() for x in _gram_inputs((2, 3, C, 8), seed=3))
        M, P = fn(q, k, c)
        ((M * wM).sum() + (P * wP).sum()).backward()
        return q.grad, k.grad, c.grad

    for a, b in zip(grads(decayed_grams), grads(_decayed_grams_dc)):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_lean_gc_is_exact_without_reference_row_term():
    """The lean backward owes nothing to the reference row r (exp(c_i - r) exp(r - c_j) does
    not depend on r). Plain autograd through the forward does account for r implicitly; the
    two gradients of c must agree to rounding error, so dropping the term is exact."""
    gen = torch.Generator().manual_seed(7)
    C = 16
    wM = torch.randn(2, C, C, generator=gen, dtype=torch.float64)
    wP = torch.randn(2, C, C, generator=gen, dtype=torch.float64)

    def grads(fn):
        q, k, c = (x.requires_grad_() for x in _gram_inputs((2, C, 5), seed=8, dtype=torch.float64))
        M, P = fn(q, k, c)
        ((M * wM).sum() + (P * wP).sum()).backward()
        return q.grad, k.grad, c.grad

    for a, b in zip(grads(decayed_grams), grads(_decayed_grams_dc)):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=0)


def test_step_matches_recurrent():
    q, k, v, g, beta, S0 = _inputs(T=1)
    o_r, S_r = recurrent_kda(q, k, v, g, beta, initial_state=S0)
    o_s, S_s = kda_step(q[:, :, 0], k[:, :, 0], v[:, :, 0], g[:, :, 0], beta[:, :, 0], S0)
    torch.testing.assert_close(o_s, o_r[:, :, 0])
    torch.testing.assert_close(S_s, S_r)


@pytest.mark.parametrize("gram", ["lean", "autograd", "naive"])
@pytest.mark.parametrize("inverse", ["solve", "doubling"])
@pytest.mark.parametrize("chunk_size", [16, 64])
@pytest.mark.parametrize("T", [1, 63, 64, 150])
def test_chunk_matches_recurrent(gram, inverse, chunk_size, T):
    q, k, v, g, beta, S0 = _inputs(T=T)
    o_r, S_r = recurrent_kda(q, k, v, g, beta, initial_state=S0)
    o_c, S_c = chunk_kda(q, k, v, g, beta, initial_state=S0,
                         chunk_size=chunk_size, inverse=inverse, gram=gram)
    assert o_c.shape == o_r.shape
    torch.testing.assert_close(o_c, o_r, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(S_c, S_r, atol=ATOL, rtol=1e-4)


def test_many_chunks_match_recurrent():
    """25 chunks: guards the affine chunk scan against error build-up over the sequence."""
    q, k, v, g, beta, S0 = _inputs(B=1, H=2, T=200, Dk=8, Dv=8)
    o_r, S_r = recurrent_kda(q, k, v, g, beta, initial_state=S0)
    o_c, S_c = chunk_kda(q, k, v, g, beta, initial_state=S0, chunk_size=8)
    torch.testing.assert_close(o_c, o_r, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(S_c, S_r, atol=ATOL, rtol=1e-4)


def test_state_carries_across_calls():
    """Running two halves with the state handed over equals running the whole sequence."""
    q, k, v, g, beta, S0 = _inputs(T=128)
    o_full, S_full = chunk_kda(q, k, v, g, beta, initial_state=S0)
    h = 70
    o1, S1 = chunk_kda(q[:, :, :h], k[:, :, :h], v[:, :, :h], g[:, :, :h],
                       beta[:, :, :h], initial_state=S0)
    o2, S2 = chunk_kda(q[:, :, h:], k[:, :, h:], v[:, :, h:], g[:, :, h:],
                       beta[:, :, h:], initial_state=S1)
    torch.testing.assert_close(torch.cat([o1, o2], dim=2), o_full, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(S2, S_full, atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("op", [recurrent_kda, chunk_kda])
def test_delta_rule_recalls_a_stored_pair(op):
    """beta = 1, no decay: writing (k, v) then reading with q = k returns v exactly."""
    Dk, Dv = 8, 5
    k = F.normalize(torch.randn(1, 1, 1, Dk), dim=-1)
    v = torch.randn(1, 1, 1, Dv)
    q = k.clone()
    g = torch.zeros(1, 1, 1, Dk)
    beta = torch.ones(1, 1, 1)
    S0 = torch.randn(1, 1, Dk, Dv)                    # junk already in memory
    o, _ = op(q, k, v, g, beta, scale=1.0, initial_state=S0)
    torch.testing.assert_close(o, v, atol=1e-5, rtol=1e-5)


def test_reduces_to_gated_delta_rule():
    """The same decay on every channel is the scalar-gate GDN."""
    q, k, v, g, beta, S0 = _inputs(T=100)
    g_s = g[..., 0]                                   # (B, H, T)
    g_v = g_s[..., None].expand_as(g).contiguous()
    o_ref, S_ref = recurrent_gated_delta_rule(q, k, v, g_s, beta, initial_state=S0)
    o_r, S_r = recurrent_kda(q, k, v, g_v, beta, initial_state=S0)
    o_c, S_c = chunk_kda(q, k, v, g_v, beta, initial_state=S0, chunk_size=16)
    for o, S in [(o_r, S_r), (o_c, S_c)]:
        torch.testing.assert_close(o, o_ref, atol=ATOL, rtol=1e-4)
        torch.testing.assert_close(S, S_ref, atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("gram", ["lean", "autograd", "naive"])
def test_extreme_decay_is_finite(gram):
    """Some channels forget almost everything each step, others nothing. A factorised
    exp(c_i) exp(-c_j) overflows here; the divide and conquer must not."""
    q, k, v, g, beta, S0 = _inputs(B=1, H=2, T=130, Dk=8, Dv=8)
    fast = torch.rand(8, generator=torch.Generator().manual_seed(5)) < 0.5
    fast[0], fast[1] = True, False                    # make sure both kinds exist
    g = torch.where(fast, torch.full_like(g, -40.0), torch.zeros_like(g))
    leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
    o_c, S_c = chunk_kda(*leaves, initial_state=S0, chunk_size=64, gram=gram)
    o_c.pow(2).sum().backward()
    for name, x in zip("qkvgb", leaves):
        assert torch.isfinite(x.grad).all(), name
    o_r, S_r = recurrent_kda(q, k, v, g, beta, initial_state=S0)
    assert torch.isfinite(o_c).all()
    torch.testing.assert_close(o_c, o_r, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(S_c, S_r, atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("gram", ["lean", "autograd", "naive"])
def test_gradients_match_recurrent(gram):
    q, k, v, g, beta, S0 = _inputs(B=1, H=2, T=40, Dk=8, Dv=8)

    def grads(fn):
        leaves = [x.clone().requires_grad_() for x in (q, k, v, g, beta)]
        o, S = fn(*leaves, initial_state=S0)
        (o.pow(2).sum() + S.pow(2).sum()).backward()
        return [x.grad for x in leaves]

    g_c = grads(lambda *a, **kw: chunk_kda(*a, chunk_size=16, gram=gram, **kw))
    g_r = grads(recurrent_kda)
    for name, a, b in zip("qkvgb", g_c, g_r):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-3, msg=name)


def test_float16_input():
    q, k, v, g, beta, S0 = _inputs(T=70)
    o32, _ = chunk_kda(q, k, v, g, beta, initial_state=S0)
    o16, S16 = chunk_kda(*(x.half() for x in (q, k, v, g, beta)), initial_state=S0)
    assert o16.dtype == torch.float16
    assert S16.dtype == torch.float32
    torch.testing.assert_close(o16.float(), o32, atol=2e-2, rtol=2e-2)


@needs_mps
@pytest.mark.parametrize("gram", ["lean", "autograd", "naive"])
def test_mps_matches_cpu(gram):
    q, k, v, g, beta, S0 = _inputs(T=150)
    o_cpu, S_cpu = chunk_kda(q, k, v, g, beta, initial_state=S0, gram=gram)
    mq, mk, mv, mg, mb, mS = (x.to("mps") for x in (q, k, v, g, beta, S0))
    o_mps, S_mps = chunk_kda(mq, mk, mv, mg, mb, initial_state=mS, gram=gram)
    torch.testing.assert_close(o_mps.cpu(), o_cpu, atol=1e-4, rtol=1e-3)
    torch.testing.assert_close(S_mps.cpu(), S_cpu, atol=1e-4, rtol=1e-3)


@needs_mps
def test_mps_backward_is_finite():
    q, k, v, g, beta, _ = _inputs(T=100)
    leaves = [x.to("mps").requires_grad_() for x in (q, k, v, g, beta)]
    o, _ = chunk_kda(*leaves)
    o.pow(2).mean().backward()
    assert o.device.type == "mps"
    for x in leaves:
        assert torch.isfinite(x.grad).all()
