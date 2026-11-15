"""Tests for [1.3a] (ID1-322), [1.3] (ID1-281) and [1.4] (ID1-282) on your multi-head KDA.

    uv run pytest eval_test/tests/test_kda_conv_state.py -v
    uv run pytest eval_test/tests/test_kda_conv_state.py -v -k conv      # just one group (conv / stream / step)

What your layer has to provide (the contract these tests read):
    Q_conv, K_conv, V_conv   nn.Conv1d, depthwise (groups = channels), kernel_size = layer.kernel_size.
                             A bias is allowed; the oracle uses it if it is there.
    q, k:  x @ W -> causal conv -> SiLU -> split heads -> L2 per head
    v:     x @ W -> causal conv -> SiLU -> split heads
    forward(x, state=None, return_state=True, chunk_size=c) -> (y, state)
        state is whatever you choose (a tensor, tuple or dict of tensors, None for an absent part), as long as passing it back into
        the next call continues the sequence. It must hold everything the future depends on.
    legacy_out(x)  your slow token-by-token version, with the conv.

oracle() below is the slow, obviously correct version the tests trust. It is written from the block
description above and nothing else; test_oracle_conv_matches_reference checks its conv against the
reference layer's.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from layers.SSMs.multi_headed.parallel_kda import MultiHeadParallelKDA
from layers.SSMs.reference.gated_deltanet import CausalDepthwiseConv

DT = torch.float64
B, T, D_IN, D_OUT, H = 2, 50, 16, 8, 3
D_QK, D_V = 12, 18                     # unequal, so a swapped k/v axis fails loudly
TIGHT = dict(rtol=1e-9, atol=1e-11)    # float64: only rounding separates two correct versions
FP32 = dict(rtol=1e-4, atol=1e-5)      # float32: rounding floor is ~1e-6 (measured on the reference, 14 Nov)
BF16 = dict(rtol=0, atol=5e-2)         # bfloat16 against the float64 oracle: measured 5e-3 to 1.4e-2 on outputs of size ~1 (15 Nov)
KERNELS = [1, 2, 4]                    # 1 = conv with no history: the carried tails must be empty


def make(seed=0, dtype=DT, kernel_size=4):
    torch.manual_seed(seed)
    # output_gate=False: the oracle below has no output norm or gate (those are tested in test_output_gate.py)
    return MultiHeadParallelKDA(D_IN, D_OUT, D_QK, D_V, H, kernel_size=kernel_size, output_gate=False).to(dtype)


def inputs(seed=1, t=T, dtype=DT):
    torch.manual_seed(seed)
    return torch.randn(B, t, D_IN, dtype=dtype)


# ---------------------------------------------------------------- the oracle

def causal_conv(z, conv, k):
    """z: (B, T, D). out[t] = sum_j w[:, j] * z[t - (k-1) + j], with zeros before the start."""
    w = conv.weight[:, 0, :]                                   # (D, k)
    zp = F.pad(z, (0, 0, k - 1, 0))                            # left pad on the time axis
    out = sum(zp[:, j:j + z.shape[1]] * w[:, j] for j in range(k))
    return out if conv.bias is None else out + conv.bias


def oracle(m, x):
    """Token by token, from a zero state, everything spelled out."""
    Bx, Tx, _ = x.shape
    k = m.kernel_size
    dqk, dv = D_QK // H, D_V // H
    q = F.normalize(F.silu(causal_conv(x @ m.W_q, m.Q_conv, k)).view(Bx, Tx, H, dqk), dim=-1)
    kk = F.normalize(F.silu(causal_conv(x @ m.W_k, m.K_conv, k)).view(Bx, Tx, H, dqk), dim=-1)
    v = F.silu(causal_conv(x @ m.W_v, m.V_conv, k)).view(Bx, Tx, H, dv)
    alpha = torch.sigmoid(x @ m.W_a).view(Bx, Tx, H, dqk)
    beta = torch.sigmoid(x @ m.W_b)                            # (B, T, H)
    S = x.new_zeros(Bx, H, dqk, dv)
    ys = []
    for t in range(Tx):
        S = alpha[:, t, :, :, None] * S                        # decay each key row
        err = v[:, t] - torch.einsum("bhk,bhkv->bhv", kk[:, t], S)
        S = S + beta[:, t, :, None, None] * kk[:, t, :, :, None] * err[:, :, None, :]
        o = torch.einsum("bhkv,bhk->bhv", S, q[:, t])
        ys.append(o.reshape(Bx, D_V) @ m.W_o)
    return torch.stack(ys, dim=1)


# ---------------------------------------------------------------- state helpers (state is opaque)

def tensors(st):
    if st is None:                         # an absent part of the state, e.g. no conv tail at kernel_size=1
        return []
    if torch.is_tensor(st):
        return [st]
    if isinstance(st, dict):
        return [t for key in sorted(st) for t in tensors(st[key])]
    if isinstance(st, (list, tuple)):
        return [t for s in st for t in tensors(s)]
    raise TypeError(f"state must be tensors, a dict or a tuple of them, got {type(st)}")


def detached(st):
    if st is None:
        return None
    if torch.is_tensor(st):
        return st.detach()
    if isinstance(st, dict):
        return {key: detached(v) for key, v in st.items()}
    return type(st)(detached(s) for s in st)


def stream(m, x, pieces, chunk=16, state=None):
    ys, t = [], 0
    for n in pieces:
        y, state = m(x[:, t:t + n], state=state, return_state=True, chunk_size=chunk)
        ys.append(y)
        t += n
    assert t == x.shape[1]
    return torch.cat(ys, dim=1), state


def slow_forgetting(m, x):
    """Push alpha to ~0.97 (mean). At init alpha averages 0.5, so the last output depends on tokens 30 back
    by only ~1e-7 and the tests above see ~25 tokens of memory. Here 100 tokens back still shows at ~1e-5.
    Feature 0 of x is set to 1 and its row of W_a gets +4, which adds 4 to every alpha logit."""
    x = x.clone()
    x[..., 0] = 1.0
    with torch.no_grad():
        m.W_a[0] += 4.0
    return m, x


def grads(m, x, loss_fn):
    """Gradients of loss_fn(m, x) for x and every parameter."""
    m.zero_grad()
    x = x.clone().requires_grad_(True)
    loss_fn(m, x).backward()
    return x.grad, {n: p.grad.clone() for n, p in m.named_parameters()}


# A fixed random readout, so the loss touches every output channel differently.
def readout_loss(y):
    torch.manual_seed(99)
    return (y * torch.randn_like(y)).sum()


# ================================================================ [1.3a] the conv

def test_conv_is_depthwise_and_causal_shaped():
    m = make()
    k = m.kernel_size
    for conv, D in ((m.Q_conv, D_QK), (m.K_conv, D_QK), (m.V_conv, D_V)):
        assert conv.weight.shape == (D, 1, k), "depthwise: one k-tap filter per channel"
        assert conv.groups == D


def test_oracle_conv_matches_reference():
    """Checks the test's own oracle against the reference layer's conv, so the oracle can be trusted."""
    torch.manual_seed(0)
    ref = CausalDepthwiseConv(6, kernel_size=4).double()
    z = torch.randn(2, 20, 6, dtype=DT)
    y_ref, _ = ref(z)
    torch.testing.assert_close(causal_conv(z, ref.conv, 4), y_ref, **TIGHT)


def test_oracle_recurrence_matches_reference():
    """Checks the test's own oracle against the reference KDA op (scale 1, log decay = logsigmoid)."""
    from layers.SSMs.reference.kda_ops import recurrent_kda
    m, x = make(), inputs()
    Bx, Tx, _ = x.shape
    dqk, dv, k = D_QK // H, D_V // H, m.kernel_size
    heads = lambda z, d: z.view(Bx, Tx, H, d).transpose(1, 2)
    q = F.normalize(heads(F.silu(causal_conv(x @ m.W_q, m.Q_conv, k)), dqk), dim=-1)
    kk = F.normalize(heads(F.silu(causal_conv(x @ m.W_k, m.K_conv, k)), dqk), dim=-1)
    v = heads(F.silu(causal_conv(x @ m.W_v, m.V_conv, k)), dv)
    g = heads(F.logsigmoid(x @ m.W_a), dqk)
    beta = torch.sigmoid(x @ m.W_b).transpose(1, 2)
    o, _ = recurrent_kda(q, kk, v, g, beta, scale=1.0)
    torch.testing.assert_close(oracle(m, x), o.transpose(1, 2).reshape(Bx, Tx, D_V) @ m.W_o, **TIGHT)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("chunk", [8, 16, 64])        # T=50: ragged last chunk for 8 and 16; 64 > T is one chunk
def test_conv_forward_matches_oracle(chunk, kernel):
    m, x = make(kernel_size=kernel), inputs()
    torch.testing.assert_close(m(x, chunk_size=chunk), oracle(m, x), **TIGHT)


def test_conv_legacy_out_matches_oracle():
    m, x = make(), inputs()
    torch.testing.assert_close(m.legacy_out(x), oracle(m, x), **TIGHT)


@pytest.mark.parametrize("pieces", [(25, 25), (17, 1, 1, 31)])
def test_legacy_out_continues_with_state(pieces):
    """The slow version carries the same state as forward, so it can continue a sequence too."""
    m, x = make(), inputs()
    ys, st, t = [], None, 0
    for n in pieces:
        y, st = m.legacy_out(x[:, t:t + n], state=st, return_state=True)
        ys.append(y)
        t += n
    torch.testing.assert_close(torch.cat(ys, dim=1), oracle(m, x), **TIGHT)


@pytest.mark.parametrize("chunk", [8, 64])
def test_conv_layer_is_causal(chunk):
    """Change x at token 25: nothing before 25 moves, and token 25 onward does."""
    m, x = make(), inputs()
    x2 = x.clone()
    x2[:, 25] += 3.0
    y, y2 = m(x, chunk_size=chunk), m(x2, chunk_size=chunk)
    torch.testing.assert_close(y2[:, :25], y[:, :25], **TIGHT)
    assert (y2[:, 25:] - y[:, 25:]).abs().amax() > 1e-3


@pytest.mark.parametrize("chunk", [8, 64])
def test_conv_gradients_match_oracle(chunk):
    m, x = make(), inputs()
    gx, gp = grads(m, x, lambda m, x: readout_loss(m(x, chunk_size=chunk)))
    ox, op = grads(m, x, lambda m, x: readout_loss(oracle(m, x)))
    torch.testing.assert_close(gx, ox, **TIGHT)
    for n in gp:
        torch.testing.assert_close(gp[n], op[n], **TIGHT, msg=n)


# ================================================================ [1.3] carrying state between calls

PIECES = [(50,), (25, 25), (17, 1, 1, 31), (1, 2, 47), (3, 3, 44), (16, 16, 18), (10,) * 5]


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("pieces", PIECES)
def test_stream_outputs_match_one_call(pieces, kernel):
    """Includes pieces shorter than the conv (1, 2, 3 tokens) and splits that are not chunk multiples."""
    m, x = make(kernel_size=kernel), inputs()
    y, _ = stream(m, x, pieces)
    torch.testing.assert_close(y, oracle(m, x), **TIGHT)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("pieces", [(25, 25), (17, 1, 1, 31), (1, 2, 47)])
def test_stream_final_state_matches_one_call(pieces, kernel):
    m, x = make(kernel_size=kernel), inputs()
    _, st_one = m(x, return_state=True, chunk_size=16)
    _, st = stream(m, x, pieces)
    for a, b in zip(tensors(st), tensors(st_one), strict=True):
        torch.testing.assert_close(a, b, **TIGHT)


@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("pieces", [(25, 25), (17, 1, 1, 31)])
def test_stream_gradients_match_one_call(pieces, kernel):
    """Gradients for x and every parameter flow through the carried state exactly as in one call."""
    m, x = make(kernel_size=kernel), inputs()
    gx, gp = grads(m, x, lambda m, x: readout_loss(stream(m, x, pieces)[0]))
    ox, op = grads(m, x, lambda m, x: readout_loss(m(x, chunk_size=16)))
    torch.testing.assert_close(gx, ox, **TIGHT)
    for n in gp:
        torch.testing.assert_close(gp[n], op[n], **TIGHT, msg=n)


def test_detached_state_cuts_the_gradient_exactly():
    """Detach the whole state between two calls; a loss on the second piece gives exactly zero gradient to the first."""
    m = make()
    x = inputs().requires_grad_(True)
    _, st = m(x[:, :30], return_state=True, chunk_size=16)
    y2, _ = m(x[:, 30:], state=detached(st), return_state=True, chunk_size=16)
    readout_loss(y2).backward()
    assert x.grad[:, :30].abs().max() == 0, "something in the state still carries gradient"
    assert x.grad[:, 30:].abs().max() > 0


@pytest.mark.parametrize("kernel", KERNELS)
def test_state_size_is_fixed(kernel):
    """The state is the layer's whole memory: S plus k-1 inputs per conv channel. It must not grow with T."""
    m = make(kernel_size=kernel)
    k = m.kernel_size
    expected = H * (D_QK // H) * (D_V // H) + (k - 1) * (2 * D_QK + D_V)
    for t in (5, 50, 200):
        _, st = m(inputs(t=t), return_state=True, chunk_size=16)
        assert sum(s.numel() for s in tensors(st)) == B * expected, f"T={t}"


def test_stream_float32_within_tolerance():
    """The same check in float32, with the tolerance from the 14 Nov discussion."""
    m, x = make(dtype=torch.float32), inputs(dtype=torch.float32)
    y_one = m(x, chunk_size=16)
    y, _ = stream(m, x, (17, 1, 1, 31))
    torch.testing.assert_close(y, y_one, **FP32)


@pytest.mark.parametrize("chunk", [16, 64])
def test_slow_forgetting_stream_matches_oracle(chunk):
    """Long memory: T=300 with alpha ~0.97, in pieces, against the token-by-token oracle."""
    m, x = slow_forgetting(make(), inputs(t=300))
    y, _ = stream(m, x, (37, 1, 1, 261), chunk=chunk)
    torch.testing.assert_close(y, oracle(m, x), **TIGHT)


@pytest.mark.parametrize("dtype, tol", [(torch.float32, FP32), (torch.bfloat16, BF16)])
@pytest.mark.parametrize("chunk", [16, 64])
def test_low_precision_against_float64_truth(dtype, tol, chunk):
    """Low precision against the float64 oracle, not against itself, with long memory. Measured 15 Nov:
    float32 3e-7, bfloat16 1e-2. (A cumsum in bf16 only shows up at chunk 256, where it doubles the bf16 error.)"""
    m, x = slow_forgetting(make(), inputs(t=300))
    y_true = oracle(m, x)
    m, x = m.to(dtype), x.to(dtype)
    y, _ = stream(m, x, (37, 1, 1, 261), chunk=chunk)
    torch.testing.assert_close(y.double(), y_true, **tol)


def test_bfloat16_runs_and_keeps_the_state_in_float32():
    m, x = make(dtype=torch.bfloat16), inputs(dtype=torch.bfloat16)
    y, st = m(x, return_state=True, chunk_size=16)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y.float()).all()
    assert any(s.dtype == torch.float32 for s in tensors(st)), "S must be carried in float32"


# ================================================================ [1.4] one token at a time

@pytest.mark.parametrize("kernel", KERNELS)
def test_step_one_token_at_a_time_from_zero(kernel):
    m, x = make(kernel_size=kernel), inputs()
    y, _ = stream(m, x, (1,) * T)
    torch.testing.assert_close(y, oracle(m, x), **TIGHT)


def test_step_prefix_then_one_token_at_a_time():
    m, x = make(), inputs()
    y, _ = stream(m, x, (37,) + (1,) * 13)
    torch.testing.assert_close(y, oracle(m, x), **TIGHT)
