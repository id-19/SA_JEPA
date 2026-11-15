"""Tests for the reference attention layers (layers/Attention_layers/reference/).

The manual impl and the from-scratch attention written in this file are the ground truth;
sdpa, the blocked path, the streaming path and MLA's absorb path must all agree with them.
Tensors are kept tiny (B <= 2, T <= 200, d_model <= 64): the machine has 8 GB.
"""

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.Attention_layers.reference import GQARef, MLARef, RotaryEmbedding, apply_rope
from layers.Attention_layers.reference.cache import cache_append, cache_view, clone_state

MPS = torch.backends.mps.is_available()
needs_mps = pytest.mark.skipif(not MPS, reason="no MPS device")
ATOL = 1e-5
IMPLS = ["sdpa", "manual"]
PIECES = (17, 1, 1, 31)


def _seed(s=0):
    torch.manual_seed(s)


def _gqa(d_model=32, n_heads=4, n_kv_heads=2, rope=True, seed=0, **kw):
    _seed(seed)
    r = RotaryEmbedding(d_model // n_heads) if rope else None
    return GQARef(d_model, n_heads, n_kv_heads, rope=r, **kw).eval()


def _mla(rope_dim=4, q_rank=None, seed=0, **kw):
    _seed(seed)
    r = RotaryEmbedding(rope_dim) if rope_dim else None
    return MLARef(32, 4, kv_rank=12, rope_dim=rope_dim, q_rank=q_rank, rope=r, **kw).eval()


def _mla_pair(rope_dim=4, q_rank=None, **kw):
    """Same weights, impl sdpa vs manual."""
    return _mla(rope_dim, q_rank, impl="sdpa", **kw), _mla(rope_dim, q_rank, impl="manual", **kw)


def _holes(B, T):
    """Padding mask with scattered holes; key 0 stays visible so no query is ever fully masked."""
    m = torch.ones(B, T, dtype=torch.bool)
    m[:, 3] = False
    m[:, 10:12] = False
    m[:, -2:] = False
    return m


def _stream(layer, x, pieces=PIECES, mask=None, **kw):
    ys, state, i = [], None, 0
    for n in pieces:
        m = None if mask is None else mask[:, i:i + n]
        y, state = layer(x[:, i:i + n], key_padding_mask=m, state=state, return_state=True, **kw)
        ys.append(y)
        i += n
    return torch.cat(ys, 1), state


# --------------------------------------------------------------------------- RoPE

@pytest.mark.parametrize("dim,axis_dims,A", [(16, None, 1), (24, (8, 8, 8), 3), (24, (12, 4, 8), 3)])
@pytest.mark.parametrize("shift", [7, 3.37])
def test_rope_relative_position(dim, axis_dims, A, shift):
    _seed()
    rope = RotaryEmbedding(dim, axis_dims=axis_dims)
    T = 20
    q, k = torch.randn(2, 3, T, dim), torch.randn(2, 3, T, dim)
    pos = torch.randint(0, 30, (T,)) if A == 1 else torch.randint(0, 30, (T, A))
    delta = shift if A == 1 else torch.tensor([shift, -2.5, 11.25])[:A]

    def scores(p):
        cos, sin = rope(p)
        return torch.einsum("bhtd,bhsd->bhts", apply_rope(q, cos, sin), apply_rope(k, cos, sin))

    torch.testing.assert_close(scores(pos), scores(pos + delta), atol=1e-4, rtol=1e-4)
    assert not torch.allclose(scores(pos), scores(pos.flip(0)), atol=1e-2)   # positions do matter


def test_rope_norm_preserved_and_axes_independent():
    _seed()
    rope = RotaryEmbedding(24, axis_dims=(8, 8, 8))
    x = torch.randn(2, 3, 10, 24)
    pos = torch.rand(10, 3) * 50
    cos, sin = rope(pos)
    torch.testing.assert_close(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)
    # the first axis' pairs only see positions[..., 0] and equal a 1-axis RoPE of dim 8
    c1, s1 = RotaryEmbedding(8)(pos[:, 0])
    torch.testing.assert_close(cos[:, :4], c1)
    torch.testing.assert_close(sin[:, :4], s1)


@pytest.mark.parametrize("axis_dims,pos_shape,out_shape", [
    (None, (10,), (10, 8)), (None, (2, 10), (2, 10, 8)),
    ((8, 8), (10, 2), (10, 8)), ((8, 8), (2, 10, 2), (2, 10, 8)),
])
def test_rope_shapes(axis_dims, pos_shape, out_shape):
    rope = RotaryEmbedding(16, axis_dims=axis_dims)
    cos, sin = rope(torch.rand(*pos_shape))
    assert cos.shape == sin.shape == out_shape and cos.dtype == torch.float32
    assert "inv_freq" not in rope.state_dict()
    x = torch.randn(2, 3, 10, 16)
    assert apply_rope(x, cos, sin).shape == x.shape
    assert apply_rope(x[:, 0], cos, sin).shape == x[:, 0].shape


# --------------------------------------------------------------------------- GQA

@pytest.mark.parametrize("causal,window", [(True, None), (False, None), (True, 5)])
@pytest.mark.parametrize("rope", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_gqa_impls_agree(causal, window, rope, qk_norm, padded):
    T = 37
    x = torch.randn(2, T, 32)
    mask = _holes(2, T) if padded else None
    outs = {}
    for impl in IMPLS:
        m = _gqa(rope=rope, qk_norm=qk_norm, causal=causal, window=window, impl=impl)
        outs[impl] = m(x, key_padding_mask=mask)
    torch.testing.assert_close(outs["sdpa"], outs["manual"], atol=ATOL, rtol=1e-4)


def _scratch_mha(m, x):
    """Plain multi-head causal attention from the module's weights; K/V repeated per group."""
    B, T, _ = x.shape
    H, Hk, D = m.n_heads, m.n_kv_heads, m.head_dim
    q = m.q_proj(x).view(B, T, H, D).transpose(1, 2)
    k = m.k_proj(x).view(B, T, Hk, D).transpose(1, 2).repeat_interleave(H // Hk, 1)
    v = m.v_proj(x).view(B, T, Hk, D).transpose(1, 2).repeat_interleave(H // Hk, 1)
    s = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(D)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    o = torch.einsum("bhts,bhsd->bhtd", s.softmax(-1), v)
    return m.o_proj(o.transpose(1, 2).reshape(B, T, H * D))


@pytest.mark.parametrize("n_kv", [4, 2, 1])
@pytest.mark.parametrize("impl", IMPLS)
def test_gqa_matches_scratch_mha(n_kv, impl):
    x = torch.randn(2, 29, 32)
    m = _gqa(n_kv_heads=n_kv, rope=False, impl=impl)
    torch.testing.assert_close(m(x), _scratch_mha(m, x), atol=ATOL, rtol=1e-4)
    assert m.cache_floats_per_token() == 2 * n_kv * 8


@pytest.mark.parametrize("impl", IMPLS)
def test_gqa_causality(impl):
    x = torch.randn(2, 40, 32)
    m = _gqa(impl=impl)
    x2 = x.clone()
    x2[:, 25] += 1.0
    d = (m(x) - m(x2)).abs().amax(dim=(0, 2))
    assert (d[:25] < 1e-6).all() and (d[25:] > 1e-6).all()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("w", [0, 4, 9])
def test_gqa_window(impl, w):
    x = torch.randn(2, 40, 32)
    m = _gqa(impl=impl, window=w)
    x2 = x.clone()
    x2[:, 5] += 1.0
    changed = (m(x) - m(x2)).abs().amax(dim=(0, 2)) > 1e-6
    expect = torch.zeros(40, dtype=torch.bool)
    expect[5:5 + w + 1] = True
    assert torch.equal(changed, expect)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("cfg", [
    dict(), dict(window=8), dict(window=8, qk_norm=True), dict(rope=False), dict(padded=True),
    dict(window=8, padded=True),
])
def test_gqa_streaming(impl, cfg):
    cfg = dict(cfg)
    padded = cfg.pop("padded", False)
    x = torch.randn(2, 50, 32)
    mask = _holes(2, 50) if padded else None
    m = _gqa(impl=impl, **cfg)
    full = m(x, key_padding_mask=mask)
    ys, state, i = [], None, 0
    for n in PIECES:
        y, state = m(x[:, i:i + n], key_padding_mask=None if mask is None else mask[:, i:i + n],
                     state=state, return_state=True)
        ys.append(y)
        i += n
        assert state["n_seen"] == i
        if "window" in cfg:
            w = cfg["window"]    # the readable cache is the last w tokens; capacity stays bounded
            assert cache_view(state["k"], state["len"], 2, w).shape[2] == min(i, w)
            assert state["k"].shape[2] <= 2 * w + max(PIECES)
            assert state["mask"] is None or state["mask"].shape[1] == state["k"].shape[2]
    torch.testing.assert_close(torch.cat(ys, 1), full, atol=ATOL, rtol=1e-4)
    assert state["k"].shape[:2] == (2, 2)


def test_gqa_streaming_explicit_positions_and_state_roundtrip(tmp_path):
    x = torch.randn(1, 20, 32)
    m = _gqa()
    pos = torch.arange(20) * 2.5
    full = m(x, positions=pos)
    y1, st = m(x[:, :12], positions=pos[:12], return_state=True)
    torch.save(st, tmp_path / "s.pt")
    st = torch.load(tmp_path / "s.pt")
    y2 = m(x[:, 12:], positions=pos[12:], state=st)
    torch.testing.assert_close(torch.cat([y1, y2], 1), full, atol=ATOL, rtol=1e-4)


def test_gqa_multi_axis_rope_positions():
    x = torch.randn(2, 12, 32)
    _seed()
    m = GQARef(32, 4, 2, rope=RotaryEmbedding(8, axis_dims=(4, 2, 2))).eval()
    pos = torch.randint(0, 9, (2, 12, 3))
    y = m(x, positions=pos)
    torch.testing.assert_close(y, m(x, positions=pos + torch.tensor([5, 2, 9])), atol=1e-4, rtol=1e-4)
    with pytest.raises(AssertionError):
        m(x)     # multi-axis needs explicit positions


@pytest.mark.parametrize("impl", IMPLS)
def test_gqa_cross_attention(impl):
    B, T, S = 2, 7, 11
    x, ctx = torch.randn(B, T, 32), torch.randn(B, S, 24)
    m = _gqa(rope=False, impl=impl, d_context=24)
    y = m(x, context=ctx)
    assert y.shape == (B, T, 32)
    H, Hk, D = 4, 2, 8
    q = m.q_proj(x).view(B, T, H, D).transpose(1, 2)
    k = m.k_proj(ctx).view(B, S, Hk, D).transpose(1, 2).repeat_interleave(2, 1)
    v = m.v_proj(ctx).view(B, S, Hk, D).transpose(1, 2).repeat_interleave(2, 1)
    o = F.softmax(torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(D), -1) @ v
    torch.testing.assert_close(y, m.o_proj(o.transpose(1, 2).reshape(B, T, -1)), atol=ATOL, rtol=1e-4)

    # a masked context token has zero influence
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[:, 3] = False
    ctx2 = ctx.clone()
    ctx2[:, 3] += 5.0
    torch.testing.assert_close(m(x, context=ctx, key_padding_mask=mask),
                               m(x, context=ctx2, key_padding_mask=mask), atol=1e-6, rtol=0)
    assert not torch.allclose(m(x, context=ctx), m(x, context=ctx2), atol=1e-4)


@pytest.mark.parametrize("masked", [False, True])
def test_gqa_cross_state_reuse_and_rope(masked):
    B, T1, T2, S = 2, 5, 6, 9
    x, ctx = torch.randn(B, T1 + T2, 32), torch.randn(B, S, 24)
    mask = _holes(B, S) if masked else None
    m = _gqa(rope=True, d_context=24)
    cpos = torch.arange(S)

    # rope without context_positions: applied to neither side
    m_nr = copy.deepcopy(m)
    m_nr.rope = None
    torch.testing.assert_close(m(x, context=ctx), m_nr(x, context=ctx), atol=ATOL, rtol=1e-4)

    full = m(x, context=ctx, context_positions=cpos, key_padding_mask=mask)
    y1, st = m(x[:, :T1], context=ctx, context_positions=cpos, key_padding_mask=mask, return_state=True)
    assert st["cross"] and st["n_seen"] == T1 and st["k"].shape == (B, 2, S, 8)
    y2 = m(x[:, T1:], context=None, state=st)          # reuses the cached projected context
    torch.testing.assert_close(torch.cat([y1, y2], 1), full, atol=ATOL, rtol=1e-4)

    # rope is relative: shifting queries and context together changes nothing
    shifted = m(x, context=ctx, positions=torch.arange(T1 + T2) + 13, context_positions=cpos + 13,
                key_padding_mask=mask)
    torch.testing.assert_close(shifted, full, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("cfg", [dict(), dict(window=6, qk_norm=True), dict(n_kv_heads=1, causal=False)])
def test_gqa_backward_finite(impl, cfg):
    x = torch.randn(2, 20, 32, requires_grad=True)
    m = _gqa(impl=impl, **cfg).train()
    m(x, key_padding_mask=_holes(2, 20)).square().sum().backward()
    grads = [x.grad] + [p.grad for p in m.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_gqa_fully_masked_rows_are_zero_and_finite():
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, :4] = False                                 # left padding: queries 0..3 see nothing
    for impl in IMPLS:
        y = _gqa(impl=impl, rope=False)(x, key_padding_mask=mask)
        assert torch.isfinite(y).all() and (y[0, :4] == 0).all()


def _half_ok(fn):
    try:
        return fn()
    except (RuntimeError, NotImplementedError) as e:   # op not implemented for half on this device
        pytest.skip(f"half not supported here: {e}")


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=needs_mps)])
def test_gqa_float16(impl, device):
    x = torch.randn(2, 30, 32)
    m = _gqa(impl=impl, window=12, qk_norm=True)
    ref = m(x, key_padding_mask=_holes(2, 30))
    mh = copy.deepcopy(m).half().to(device)
    y = _half_ok(lambda: mh(x.half().to(device), key_padding_mask=_holes(2, 30).to(device)))
    assert y.dtype == torch.float16 and torch.isfinite(y).all()
    torch.testing.assert_close(y.float().cpu(), ref, atol=3e-2, rtol=3e-2)
    # half streaming with state
    ys, _ = _half_ok(lambda: _stream(mh, x.half().to(device), (20, 10)))
    assert ys.dtype == torch.float16


@needs_mps
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("cfg", [dict(), dict(window=7, qk_norm=True), dict(causal=False)])
def test_gqa_mps_matches_cpu(impl, cfg):
    x, mask = torch.randn(2, 33, 32), _holes(2, 33)
    m = _gqa(impl=impl, **cfg)
    ym = copy.deepcopy(m).to("mps")(x.to("mps"), key_padding_mask=mask.to("mps"))
    torch.testing.assert_close(ym.cpu(), m(x, key_padding_mask=mask), atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------- MLA

@pytest.mark.parametrize("rope_dim", [0, 4])
@pytest.mark.parametrize("q_rank", [None, 10])
@pytest.mark.parametrize("padded", [False, True])
def test_mla_naive_equals_absorb(rope_dim, q_rank, padded):
    x = torch.randn(2, 25, 32)
    m = _mla(rope_dim, q_rank)
    mask = _holes(2, 25) if padded else None
    torch.testing.assert_close(m(x, key_padding_mask=mask, absorb=False),
                               m(x, key_padding_mask=mask, absorb=True), atol=ATOL, rtol=1e-4)
    assert m.cache_floats_per_token() == 12 + rope_dim


def test_mla_absorb_default_and_v_head_dim():
    _seed()
    m = MLARef(32, 4, kv_rank=6, head_dim=10, v_head_dim=7, rope_dim=2, rope=RotaryEmbedding(2), q_rank=8)
    x = torch.randn(1, 9, 32)
    assert m(x).shape == (1, 9, 32)
    m.eval()                                           # absorb by default in eval, same numbers
    torch.testing.assert_close(m(x), m(x, absorb=False), atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("absorb", [False, True])
@pytest.mark.parametrize("rope_dim", [0, 4])
@pytest.mark.parametrize("padded", [False, True])
def test_mla_streaming(absorb, rope_dim, padded):
    x = torch.randn(2, 50, 32)
    mask = _holes(2, 50) if padded else None
    m = _mla(rope_dim, q_rank=10)
    full = m(x, key_padding_mask=mask, absorb=absorb)
    ys, state = _stream(m, x, mask=mask, absorb=absorb)
    torch.testing.assert_close(ys, full, atol=ATOL, rtol=1e-4)
    assert state["len"] == state["n_seen"] == 50
    assert cache_view(state["c"], 50, 1).shape == (2, 50, 12)
    if rope_dim:
        assert cache_view(state["k_r"], 50, 1).shape == (2, 50, 4)
    else:
        assert state["k_r"] is None
    assert (state["mask"] is not None) == padded
    # mixing paths across calls is fine too
    y1, st = m(x[:, :20], return_state=True, absorb=True)
    y2 = m(x[:, 20:], state=st, absorb=False)
    torch.testing.assert_close(torch.cat([y1, y2], 1), m(x), atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("absorb", [False, True])
@pytest.mark.parametrize("rope_dim", [0, 4])
def test_mla_causality_padding_backward(absorb, rope_dim):
    x = torch.randn(2, 40, 32)
    m = _mla(rope_dim, q_rank=10)
    x2 = x.clone()
    x2[:, 25] += 1.0
    d = (m(x, absorb=absorb) - m(x2, absorb=absorb)).abs().amax(dim=(0, 2))
    assert (d[:25] < 1e-6).all() and (d[25:] > 1e-6).all()

    mask = torch.ones(2, 40, dtype=torch.bool)
    mask[:, 7] = False
    x3 = x.clone()
    x3[:, 7] += 3.0
    y, y3 = m(x, key_padding_mask=mask, absorb=absorb), m(x3, key_padding_mask=mask, absorb=absorb)
    other = torch.arange(40) != 7                       # token 7's own output may change, nobody else's
    torch.testing.assert_close(y[:, other], y3[:, other], atol=1e-6, rtol=0)

    m.train()
    xg = x.clone().requires_grad_()
    m(xg, key_padding_mask=_holes(2, 40), absorb=absorb).square().sum().backward()
    assert all(g is not None and torch.isfinite(g).all() for g in [xg.grad] + [p.grad for p in m.parameters()])


def test_mla_fully_masked_rows_are_zero_and_finite():
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, :4] = False
    for absorb in (False, True):
        y = _mla()(x, key_padding_mask=mask, absorb=absorb)
        assert torch.isfinite(y).all() and (y[0, :4] == 0).all()


def _rms_ref(x, w, eps=1e-6):
    return x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


@pytest.mark.parametrize("rope_dim", [0, 4])
@pytest.mark.parametrize("q_rank", [None, 10])
def test_mla_matches_scratch_mha_with_explicit_kv(rope_dim, q_rank):
    """Build the full K (k_c ++ shared k_r broadcast over heads) and V, then plain MHA."""
    B, T, H, D, R, r = 2, 21, 4, 8, rope_dim, 12
    x = torch.randn(B, T, 32)
    m = _mla(rope_dim, q_rank)
    q = m.wq(x) if q_rank is None else m.wq_b(_rms_ref(m.wq_a(x), m.q_norm_weight))
    q = q.view(B, T, H, D + R).transpose(1, 2)
    c, k_r = m.wkv_a(x).split([r, R], -1)
    c = _rms_ref(c, m.kv_norm_weight)
    k_c, v = m.wkv_b(c).view(B, T, H, 2 * D).transpose(1, 2).split([D, D], -1)
    if R:
        cos, sin = m.rope(torch.arange(T))
        q = torch.cat([q[..., :D], apply_rope(q[..., D:], cos, sin)], -1)
        k = torch.cat([k_c, apply_rope(k_r, cos, sin)[:, None].expand(B, H, T, R)], -1)
    else:
        k = k_c
    s = torch.einsum("bhtd,bhsd->bhts", q, k) / math.sqrt(D + R)
    s = s.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
    o = torch.einsum("bhts,bhsd->bhtd", s.softmax(-1), v)
    ref = m.wo(o.transpose(1, 2).reshape(B, T, H * D))
    torch.testing.assert_close(m(x, absorb=False), ref, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(m(x, absorb=True), ref, atol=ATOL, rtol=1e-4)


def test_mla_rope_arguments_validated():
    with pytest.raises(AssertionError):
        MLARef(32, 4, kv_rank=8, rope_dim=4)                           # RoPE branch without a rope
    with pytest.raises(AssertionError):
        MLARef(32, 4, kv_rank=8, rope_dim=0, rope=RotaryEmbedding(4))  # NoPE with a rope


@pytest.mark.parametrize("absorb", [False, True])
@pytest.mark.parametrize("rope_dim", [0, 4])
@needs_mps
def test_mla_mps_matches_cpu(absorb, rope_dim):
    x, mask = torch.randn(2, 30, 32), _holes(2, 30)
    m = _mla(rope_dim, q_rank=10)
    ym = copy.deepcopy(m).to("mps")(x.to("mps"), key_padding_mask=mask.to("mps"), absorb=absorb)
    torch.testing.assert_close(ym.cpu(), m(x, key_padding_mask=mask, absorb=absorb), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("device", ["cpu", pytest.param("mps", marks=needs_mps)])
def test_mla_float16(device):
    x = torch.randn(2, 30, 32)
    m = _mla(4, q_rank=10)
    ref = m(x)
    mh = copy.deepcopy(m).half().to(device)
    y = _half_ok(lambda: mh(x.half().to(device)))
    assert y.dtype == torch.float16 and torch.isfinite(y).all()
    torch.testing.assert_close(y.float().cpu(), ref, atol=3e-2, rtol=3e-2)


# --------------------------------------------------------------------------- cache buffers

def test_cache_append_basics():
    buf, n = None, 0
    ref = torch.randn(1, 2, 70, 3)
    ptrs = set()
    for i in range(70):
        buf, n = cache_append(buf, n, ref[:, :, i:i + 1], 2)
        ptrs.add(buf.data_ptr())
    assert n == 70 and len(ptrs) <= 2
    torch.testing.assert_close(cache_view(buf, n, 2), ref)
    assert cache_view(buf, n, 2, last=5).shape[2] == 5 and cache_view(buf, n, 2, last=500).shape[2] == 70
    # keep: compaction (overlapping move) preserves the last `keep` entries and bounds capacity
    buf, n = None, 0
    for i in range(70):
        buf, n = cache_append(buf, n, ref[:, :, i:i + 1], 2, keep=6)
        torch.testing.assert_close(cache_view(buf, n, 2, 6), ref[:, :, max(i + 1 - 6, 0):i + 1])
    assert buf.shape[2] <= 2 * 6 + 1


def test_clone_state_is_deep():
    st = {"k": torch.zeros(2), "n_seen": 3, "mask": None}
    c = clone_state(st)
    c["k"] += 1
    assert st["k"].sum() == 0 and c["n_seen"] == 3 and c["mask"] is None


def _decode(layer, x, **kw):
    ys, state = [], None
    ptrs = set()
    for t in range(x.shape[1]):
        m = kw.get("mask")
        y, state = layer(x[:, t:t + 1], key_padding_mask=None if m is None else m[:, t:t + 1],
                         state=state, return_state=True, **{k: v for k, v in kw.items() if k != "mask"})
        ys.append(y)
        ptrs.add(state["k"].data_ptr() if "k" in state else state["c"].data_ptr())
    return torch.cat(ys, 1), state, len(ptrs)


@pytest.mark.parametrize("cfg", [dict(), dict(window=8), dict(window=8, rope=False), dict(padded=True)])
@pytest.mark.parametrize("impl", IMPLS)
def test_gqa_decode_300_tokens(cfg, impl):
    cfg = dict(cfg)
    padded = cfg.pop("padded", False)
    x = torch.randn(1, 300, 32)
    mask = torch.rand(1, 300) > 0.2 if padded else None
    if padded:
        mask[:, 0] = True
    m = _gqa(impl=impl, **cfg)
    with torch.no_grad():
        full = m(x, key_padding_mask=mask)
        y, state, n_alloc = _decode(m, x, mask=mask)
    torch.testing.assert_close(y, full, atol=ATOL, rtol=1e-4)
    assert n_alloc <= 8
    if "window" in cfg:
        assert state["k"].shape[2] <= 4 * cfg["window"] + 1     # bounded over the whole stream


@pytest.mark.parametrize("rope_dim", [0, 4])
@pytest.mark.parametrize("absorb", [False, True])
def test_mla_decode_300_tokens(rope_dim, absorb):
    x = torch.randn(1, 300, 32)
    m = _mla(rope_dim, q_rank=10)
    with torch.no_grad():
        full = m(x, absorb=absorb)
        y, state, n_alloc = _decode(m, x, absorb=absorb)
    torch.testing.assert_close(y, full, atol=ATOL, rtol=1e-4)
    assert n_alloc <= 8 and state["len"] == 300


def test_clone_state_branches():
    x = torch.randn(1, 40, 32)
    a, b = torch.randn(1, 10, 32), torch.randn(1, 10, 32)
    for layer, kw in [(_gqa(window=8), {}), (_gqa(), {}), (_mla(4, 10), {})]:
        with torch.no_grad():
            _, st = layer(x, return_state=True)
            ya, _ = layer(a, state=clone_state(st), return_state=True)
            yb, _ = layer(b, state=clone_state(st), return_state=True)     # original prefix untouched
            ref_a = layer(torch.cat([x, a], 1))[:, 40:]
            ref_b = layer(torch.cat([x, b], 1))[:, 40:]
        torch.testing.assert_close(ya, ref_a, atol=ATOL, rtol=1e-4)
        torch.testing.assert_close(yb, ref_b, atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("make", [lambda: _gqa(), lambda: _gqa(window=6), lambda: _mla(4, 10)])
def test_functional_fallback_with_grad_and_state(make):
    layer = make().train()
    x = torch.randn(2, 30, 32, requires_grad=True)
    mask = _holes(2, 30)
    full = layer(x, key_padding_mask=mask)
    ys, state, i = [], None, 0
    for n in (12, 1, 17):
        y, state = layer(x[:, i:i + n], key_padding_mask=mask[:, i:i + n], state=state, return_state=True)
        ys.append(y)
        i += n
    y = torch.cat(ys, 1)
    torch.testing.assert_close(y, full, atol=ATOL, rtol=1e-4)
    g_stream, = torch.autograd.grad(y.square().sum(), x, retain_graph=True)
    g_full, = torch.autograd.grad(full.square().sum(), x)
    torch.testing.assert_close(g_stream, g_full, atol=1e-4, rtol=1e-4)
    assert torch.isfinite(g_stream).all()


# --------------------------------------------------------------------------- MLA impl + blocking

@pytest.mark.parametrize("rope_dim", [0, 4])
@pytest.mark.parametrize("q_rank", [None, 10])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("padded", [False, True])
@pytest.mark.parametrize("absorb", [False, True])
def test_mla_sdpa_equals_manual(rope_dim, q_rank, causal, padded, absorb):
    x = torch.randn(2, 25, 32)
    ms, mm = _mla_pair(rope_dim, q_rank, causal=causal)
    mask = _holes(2, 25) if padded else None
    torch.testing.assert_close(ms(x, key_padding_mask=mask, absorb=absorb),
                               mm(x, key_padding_mask=mask, absorb=absorb), atol=ATOL, rtol=1e-4)
    # with streaming state
    ys, _ = _stream(ms, x[:, :20], (7, 1, 12), mask=None if mask is None else mask[:, :20], absorb=absorb)
    ym, _ = _stream(mm, x[:, :20], (7, 1, 12), mask=None if mask is None else mask[:, :20], absorb=absorb)
    torch.testing.assert_close(ys, ym, atol=ATOL, rtol=1e-4)


@needs_mps
@pytest.mark.parametrize("rope_dim", [0, 4])
@pytest.mark.parametrize("absorb", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_mla_sdpa_mps(rope_dim, absorb, padded):
    x = torch.randn(2, 25, 32)
    ms, mm = _mla_pair(rope_dim, 10)
    mask = _holes(2, 25) if padded else None
    ref = mm(x, key_padding_mask=mask, absorb=absorb)
    dm = copy.deepcopy(ms).to("mps")
    y = dm(x.to("mps"), key_padding_mask=None if mask is None else mask.to("mps"), absorb=absorb)
    torch.testing.assert_close(y.cpu(), ref, atol=1e-4, rtol=1e-4)
    ys, _ = _stream(dm, x.to("mps"), (10, 1, 14), mask=None if mask is None else mask.to("mps"), absorb=absorb)
    torch.testing.assert_close(ys.cpu(), ref, atol=1e-4, rtol=1e-4)


def test_mla_sdpa_fully_masked_rows():
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[0, :4] = False
    for absorb in (False, True):
        y = _mla()(x, key_padding_mask=mask, absorb=absorb)
        assert torch.isfinite(y).all() and (y[0, :4] == 0).all()


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("blk", [1, 8])
@pytest.mark.parametrize("cfg", [dict(), dict(window=6, qk_norm=True), dict(causal=False), dict(padded=True)])
def test_gqa_query_blocking(impl, blk, cfg):
    cfg = dict(cfg)
    padded = cfg.pop("padded", False)
    T = 37                                              # not a multiple of 8
    x = torch.randn(2, T, 32)
    mask = _holes(2, T) if padded else None
    ref = _gqa(impl=impl, **cfg)
    blocked = _gqa(impl=impl, max_scores=4 * T * blk, **cfg)      # blocks of `blk` queries
    with torch.no_grad():
        torch.testing.assert_close(blocked(x, key_padding_mask=mask), ref(x, key_padding_mask=mask),
                                   atol=ATOL, rtol=1e-4)
        # streaming prefill chunk + cross-attention are blocked too
        _, st = ref(x[:, :20], key_padding_mask=None if mask is None else mask[:, :20], return_state=True)
        _, st2 = blocked(x[:, :20], key_padding_mask=None if mask is None else mask[:, :20], return_state=True)
        torch.testing.assert_close(
            blocked(x[:, 20:], key_padding_mask=None if mask is None else mask[:, 20:], state=st2),
            ref(x[:, 20:], key_padding_mask=None if mask is None else mask[:, 20:], state=st),
            atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("impl", IMPLS)
def test_gqa_query_blocking_cross(impl):
    x, ctx = torch.randn(2, 13, 32), torch.randn(2, 9, 24)
    mask = _holes(2, 9)
    ref = _gqa(rope=False, impl=impl, d_context=24)
    blocked = _gqa(rope=False, impl=impl, d_context=24, max_scores=4 * 9 * 5)
    with torch.no_grad():
        torch.testing.assert_close(blocked(x, context=ctx, key_padding_mask=mask),
                                   ref(x, context=ctx, key_padding_mask=mask), atol=ATOL, rtol=1e-4)


def test_blocking_only_without_grad():
    """With grad enabled the full score tensor is used (no blocking); results are equal anyway."""
    x = torch.randn(2, 20, 32)
    m = _gqa(impl="manual", max_scores=1)
    calls = []
    orig = m._manual
    m._manual = lambda *a: (calls.append(a[0].shape[2]), orig(*a))[1]
    m(x)
    assert calls == [20]
    with torch.no_grad():
        m(x)
    assert calls[1:] == [1] * 20


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("rope_dim", [0, 4])
@pytest.mark.parametrize("absorb", [False, True])
@pytest.mark.parametrize("blk", [1, 8])
def test_mla_query_blocking(impl, rope_dim, absorb, blk):
    T = 37
    x, mask = torch.randn(2, T, 32), _holes(2, T)
    ref = _mla(rope_dim, 10, impl=impl)
    blocked = _mla(rope_dim, 10, impl=impl, max_scores=4 * T * blk)
    with torch.no_grad():
        for kw in (dict(), dict(key_padding_mask=mask)):
            torch.testing.assert_close(blocked(x, absorb=absorb, **kw), ref(x, absorb=absorb, **kw),
                                       atol=ATOL, rtol=1e-4)
        _, st = ref(x[:, :20], return_state=True, absorb=absorb)
        _, st2 = blocked(x[:, :20], return_state=True, absorb=absorb)
        torch.testing.assert_close(blocked(x[:, 20:], state=st2, absorb=absorb),
                                   ref(x[:, 20:], state=st, absorb=absorb), atol=ATOL, rtol=1e-4)


# --------------------------------------------------------------------------- decode fast paths

def _prefill(layer, x, mask=None, **kw):
    with torch.no_grad():
        _, st = layer(x, key_padding_mask=mask, return_state=True, **kw)
    return st


def _buffers_finite(state):
    return all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v) and v.is_floating_point())


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("cfg", [dict(), dict(window=8), dict(causal=False)])
def test_gqa_single_token_uses_no_mask(impl, cfg, monkeypatch):
    x = torch.randn(1, 60, 32)
    m = _gqa(impl=impl, rope=False, **cfg)
    with torch.no_grad():
        full = m(x)
        st = _prefill(m, x[:, :30])
        import layers.Attention_layers.reference.gqa as gqa_mod

        def boom(*a, **k):
            raise AssertionError("mask was built for a single decoded token")
        monkeypatch.setattr(gqa_mod, "_mask", boom)
        monkeypatch.setattr(torch, "arange", boom)
        ys = []
        for t in range(30, 60):
            y, st = m(x[:, t:t + 1], state=st, return_state=True)
            ys.append(y)
        monkeypatch.undo()
        if not cfg.get("causal", True):                 # non-causal sees the future: compare to itself
            assert torch.cat(ys, 1).shape == (1, 30, 32)
        else:
            torch.testing.assert_close(torch.cat(ys, 1), full[:, 30:], atol=ATOL, rtol=1e-4)
    assert _buffers_finite(st)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("cfg", [dict(), dict(window=8)])
def test_gqa_no_mask_equals_all_true_mask(impl, cfg):
    """The no-mask decode path and the fixed-shape masked path (all-True padding mask) agree."""
    x = torch.randn(2, 50, 32)
    m = _gqa(impl=impl, **cfg)
    ones = torch.ones(2, 50, dtype=torch.bool)
    with torch.no_grad():
        a, _, _ = _decode(m, x)
        b, _, _ = _decode(m, x, mask=ones)
        torch.testing.assert_close(a, b, atol=ATOL, rtol=1e-4)
        torch.testing.assert_close(a, m(x), atol=ATOL, rtol=1e-4)


@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("cfg", [dict(), dict(window=8)])
def test_gqa_static_shape_padded_decode(impl, cfg, monkeypatch):
    T = 150
    x = torch.randn(2, T, 32)
    mask = torch.rand(2, T) > 0.3
    mask[:, 0] = True
    mask[:, 1:6] = False                                 # hide some early tokens
    m = _gqa(impl=impl, **cfg)
    shapes = []
    orig = F.scaled_dot_product_attention
    monkeypatch.setattr(F, "scaled_dot_product_attention",
                        lambda q, k, *a, **kw: (shapes.append((k.shape[2], kw["attn_mask"] is not None)), orig(q, k, *a, **kw))[1])
    with torch.no_grad():
        full = m(x, key_padding_mask=mask)
        shapes.clear()
        y, state, n_alloc = _decode(m, x, mask=mask)
    torch.testing.assert_close(y, full, atol=ATOL, rtol=1e-4)
    assert torch.isfinite(y).all() and _buffers_finite(state)
    if impl == "sdpa":
        assert all(masked for _, masked in shapes)
        assert len({s for s, _ in shapes}) <= 8        # key length only changes on reallocation


@pytest.mark.parametrize("absorb", [False, True])
@pytest.mark.parametrize("impl", IMPLS)
@pytest.mark.parametrize("rope_dim", [0, 4])
def test_mla_single_token_no_mask_and_static_padded(absorb, impl, rope_dim, monkeypatch):
    T = 120
    x = torch.randn(1, T, 32)
    m = _mla(rope_dim, 10, impl=impl)
    mask = torch.rand(1, T) > 0.3
    mask[:, 0] = True
    mask[:, 1:6] = False
    with torch.no_grad():
        full, full_m = m(x, absorb=absorb), m(x, key_padding_mask=mask, absorb=absorb)
        a, st_a, _ = _decode(m, x, absorb=absorb)                       # no mask anywhere
        b, st_b, _ = _decode(m, x, mask=mask, absorb=absorb)            # fixed-shape masked decode
    torch.testing.assert_close(a, full, atol=ATOL, rtol=1e-4)
    torch.testing.assert_close(b, full_m, atol=ATOL, rtol=1e-4)
    assert _buffers_finite(st_a) and _buffers_finite(st_b) and torch.isfinite(b).all()

    if rope_dim == 0:                                    # no arange at all on the no-mask decode path
        st = _prefill(m, x[:, :30], absorb=absorb)
        monkeypatch.setattr(torch, "arange", lambda *a, **k: (_ for _ in ()).throw(AssertionError("vis built")))
        with torch.no_grad():
            for t in range(30, 40):
                _, st = m(x[:, t:t + 1], state=st, return_state=True, absorb=absorb)


def test_cache_tail_is_finite_after_growth_and_compaction():
    ref = torch.randn(1, 2, 300, 3)
    for keep in (None, 6):
        buf, n = None, 0
        for i in range(300):
            buf, n = cache_append(buf, n, ref[:, :, i:i + 1], 2, keep=keep)
            assert torch.isfinite(buf).all()
        assert (buf[:, :, n:] == 0).all() or keep is not None   # grown, never-written tail is zeros
    # a mask that appears late has a False tail
    st = {"k": torch.randn(1, 2, 10, 3), "v": torch.randn(1, 2, 10, 3), "mask": None, "len": 5}
    from layers.Attention_layers.reference.cache import cache_extend
    out, n = cache_extend(st, {"k": torch.randn(1, 2, 1, 3), "v": torch.randn(1, 2, 1, 3)}, 2,
                          torch.zeros(1, 1, dtype=torch.bool))
    assert out["mask"][:, :5].all() and not out["mask"][:, 5:].any() and n == 6


@needs_mps
@pytest.mark.parametrize("impl", IMPLS)
def test_gqa_padded_and_plain_decode_mps(impl):
    x = torch.randn(1, 60, 32)
    mask = torch.rand(1, 60) > 0.3
    mask[:, 0] = True
    m = _gqa(impl=impl, window=8)
    ref = m(x, key_padding_mask=mask)
    dm = copy.deepcopy(m).to("mps")
    with torch.no_grad():
        y, _, _ = _decode(dm, x.to("mps"), mask=mask.to("mps"))
        y2, _, _ = _decode(dm, x.to("mps"))
    torch.testing.assert_close(y.cpu(), ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(y2.cpu(), m(x), atol=1e-4, rtol=1e-4)
