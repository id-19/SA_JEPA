"""Tests for the multi-head chunked KDA, [1.2] (ID1-280).

Done when (from the ticket):
    1. With H=1, outputs match the single-head module.                          (retired 15 Nov, see below)
    2. With H heads, outputs match H single-head modules with the weights copied in.  (retired 15 Nov)
    3. The [1.1] gradient checks still pass: chunked agrees with the recurrent legacy_out.

Retired 15 Nov 2026, on purpose ([1.3a], ID1-322): the multi-head layer now has a causal conv and SiLU on
q, k, v, and the single-head module does not, so 1 and 2 can never hold again. Head independence is still
tested: test_kda_conv_state.py compares the layer with an oracle that splits heads through the same contract.
Checked: a planted head-mixing bug (k split into heads with the wrong reshape) fails 57 of its 69 tests.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.SSMs.multi_headed.parallel_kda import MultiHeadParallelKDA

DT = torch.float64
B, T, D_IN, D_OUT = 2, 50, 16, 8
D_QK, D_V = 12, 18                    # total widths, split across heads; unequal so a swapped k/v axis fails loudly
CHUNKS = (8, 16, 64)                  # T=50: ragged last chunk for 8 and 16; 64 > T is one chunk
TOL = dict(rtol=1e-10, atol=1e-12)


def make(n_heads, seed=0):
    torch.manual_seed(seed)
    return MultiHeadParallelKDA(D_IN, D_OUT, D_QK, D_V, n_heads).to(DT)


def inputs(seed=1):
    torch.manual_seed(seed)
    return torch.randn(B, T, D_IN, dtype=DT)


@pytest.mark.parametrize("H", [1, 3])
def test_weight_shapes_follow_contract(H):
    m = make(H)
    assert m.W_q.shape == (D_IN, D_QK)
    assert m.W_k.shape == (D_IN, D_QK)
    assert m.W_v.shape == (D_IN, D_V)
    assert m.W_a.shape == (D_IN, D_QK)
    assert m.W_b.shape == (D_IN, H)
    assert m.W_o.shape == (D_V, D_OUT)


@pytest.mark.parametrize("chunk", CHUNKS)
def test_chunked_gradients_match_recurrent(chunk):
    H = 3
    mh = make(H)
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
    mh = make(2)
    x = torch.randn(1, 12, D_IN, dtype=DT, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: mh(x, chunk_size=5), (x,))
