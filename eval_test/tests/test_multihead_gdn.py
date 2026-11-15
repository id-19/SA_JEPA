"""Tests for the multi-head chunked Gated DeltaNet, [1.2] (ID1-280).

Done when (from the ticket):
    1. With H=1, outputs match the single-head module.
    2. With H heads, outputs match H single-head modules with the weights copied in.
    3. The [1.1] gradient checks still pass: chunked agrees with the recurrent legacy_out.

Weights are copied through the contract at the top of layers/SSMs/multi_headed/parallel_gdn.py.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.SSMs.multi_headed.parallel_gdn import MultiHeadParallelGDN
from layers.SSMs.single_headed.parallel_gdn import ParallelGatedDeltaNet

DT = torch.float64
B, T, D_IN, D_OUT = 2, 50, 16, 8
D_QK, D_V = 12, 18                    # total widths, split across heads; unequal so a swapped k/v axis fails loudly
CHUNKS = (8, 16, 64)                  # T=50: ragged last chunk for 8 and 16; 64 > T is one chunk
TOL = dict(rtol=1e-10, atol=1e-12)


def make(n_heads, seed=0, output_gate=False):
    # output_gate=False by default: the single-head module this is compared against has no norm or gate
    torch.manual_seed(seed)
    return MultiHeadParallelGDN(D_IN, D_OUT, D_QK, D_V, n_heads, output_gate=output_gate).to(DT)


def single_from_head(mh, h):
    """A single-head module holding exactly head h's weights."""
    dqk, dv = D_QK // mh.n_heads, D_V // mh.n_heads
    sh = ParallelGatedDeltaNet(D_IN, D_OUT, dqk, dv).to(DT)
    qk, v = slice(h * dqk, (h + 1) * dqk), slice(h * dv, (h + 1) * dv)
    with torch.no_grad():
        sh.W_q.copy_(mh.W_q[:, qk])
        sh.W_k.copy_(mh.W_k[:, qk])
        sh.W_v.copy_(mh.W_v[:, v])
        sh.W_a.copy_(mh.W_a[:, h])
        sh.W_b.copy_(mh.W_b[:, h:h + 1])
        sh.W_o.copy_(mh.W_o[v])
    return sh


def inputs(seed=1):
    torch.manual_seed(seed)
    return torch.randn(B, T, D_IN, dtype=DT)


@pytest.mark.parametrize("H", [1, 3])
def test_weight_shapes_follow_contract(H):
    m = make(H)
    assert m.W_q.shape == (D_IN, D_QK)
    assert m.W_k.shape == (D_IN, D_QK)
    assert m.W_v.shape == (D_IN, D_V)
    assert m.W_a.shape == (D_IN, H)
    assert m.W_b.shape == (D_IN, H)
    assert m.W_o.shape == (D_V, D_OUT)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_one_head_matches_single_head_module(chunk):
    mh = make(1)
    sh = single_from_head(mh, 0)
    x = inputs()
    torch.testing.assert_close(mh(x, chunk_size=chunk), sh(x, chunk_size=chunk), **TOL)
    torch.testing.assert_close(mh.legacy_out(x), sh.legacy_out(x), **TOL)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_heads_match_independent_single_head_modules(chunk):
    # W_o is linear, so the merged output is the sum of each head's slice projected alone
    H = 3
    mh = make(H)
    singles = [single_from_head(mh, h) for h in range(H)]
    x = inputs()
    torch.testing.assert_close(mh(x, chunk_size=chunk), sum(s(x, chunk_size=chunk) for s in singles), **TOL)
    torch.testing.assert_close(mh.legacy_out(x), sum(s.legacy_out(x) for s in singles), **TOL)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_chunked_gradients_match_recurrent(chunk):
    H = 3
    mh = make(H, output_gate=True)
    x = inputs().requires_grad_()
    w = torch.randn(B, T, D_OUT, dtype=DT)
    params = [x, *mh.parameters()]

    def grads(fn):
        for p in params:
            p.grad = None
        (fn(x) * w).sum().backward()
        return [p.grad.clone() for p in params]

    for g_chunk, g_rec in zip(grads(lambda x: mh(x, chunk_size=chunk)), grads(mh.legacy_out)):
        torch.testing.assert_close(g_chunk, g_rec, **TOL)


def test_gradcheck():
    mh = make(2, output_gate=True)
    x = torch.randn(1, 12, D_IN, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: mh(x, chunk_size=5), (x,))
