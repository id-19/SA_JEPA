"""Kimi Delta Attention ops: a token-by-token reference and a chunked version for MPS/CPU.

Reference implementation written by an AI agent (13 Nov 2026). It is a correctness
oracle and a speed baseline, not the learning implementation: the user's own KDA should
match this one with allclose on the same inputs.

KDA is the gated delta rule with a per-channel decay: alpha is a vector over the key axis.

Shapes (all ops):
    q, k:  (B, H, T, Dk)    keys and queries, already L2-normalised per head
    v:     (B, H, T, Dv)
    g:     (B, H, T, Dk)    log decay per key channel, g <= 0, alpha = exp(g)
    beta:  (B, H, T)        write strength
    state: (B, H, Dk, Dv)   S maps a key to a value: value = S^T key

Recurrence, per head and per token t (decay, delta write, then read):
    S   = Diag(alpha_t) S_{t-1}
    S_t = S + beta_t k_t (v_t - S^T k_t)^T
    o_t = scale * S_t^T q_t

Precision: computed in promote_types(input dtype, float32), so float16 runs in float32 and
float64 stays float64 (gradcheck). o is cast back to v's dtype; the state stays in the
compute dtype.
"""

import torch
from torch.autograd.function import once_differentiable

from .gated_delta_rule import _scan_chunks, _unit_lower_inverse, recurrent_gated_delta_rule  # noqa: F401
from . import metal_kernels as _metal


def kda_step(q, k, v, g, beta, state, scale=None):
    """One token. q, k, g: (B, H, Dk); v: (B, H, Dv); beta: (B, H). Returns (o, new_state)."""
    dtype = torch.promote_types(q.dtype, torch.float32)
    in_dtype = v.dtype
    q, k, v, g, beta, state = (x.to(dtype) for x in (q, k, v, g, beta, state))
    if scale is None:
        scale = q.shape[-1] ** -0.5

    S = state * g.exp()[..., :, None]                             # per-channel decay along Dk
    old = torch.einsum("bhkv,bhk->bhv", S, k)                     # what S currently returns for k
    err = (v - old) * beta[..., None]
    S = S + k[..., :, None] * err[..., None, :]                   # rank-1 write
    o = torch.einsum("bhkv,bhk->bhv", S, q) * scale
    return o.to(in_dtype), S


def _metal_decode_ok(dtype, *tensors):
    """The fused Metal step has no autograd and is float32 only: use it for no-grad decode on MPS."""
    needs_grad = torch.is_grad_enabled() and any(x is not None and x.requires_grad for x in tensors)
    return (dtype == torch.float32 and tensors[0].device.type == "mps"
            and _metal.metal_available() and not needs_grad)


def recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None):
    """Token-by-token KDA. Slow, obviously correct. Returns (o, final_state).

    On MPS with no gradient needed (decode), each token is one fused Metal dispatch
    (metal_kernels.kda_step_metal) instead of about ten PyTorch ops.
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    dtype = torch.promote_types(q.dtype, torch.float32)
    in_dtype = v.dtype
    metal = _metal_decode_ok(dtype, q, k, v, g, beta, initial_state)
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))

    if initial_state is None:
        S = q.new_zeros(B, H, Dk, Dv)
    elif metal:
        S = initial_state.detach().to(dtype).contiguous().clone()   # the kernel writes in place; keep the caller's state
    else:
        S = initial_state.to(dtype)
    step = _metal.kda_step_metal if metal else kda_step
    outs = []
    # unbind, not x[:, :, t]: indexing in a loop makes the backward quadratic in T
    for q_t, k_t, v_t, g_t, b_t in zip(*(x.detach().unbind(2) if metal else x.unbind(2) for x in (q, k, v, g, beta))):
        o_t, S = step(q_t, k_t, v_t, g_t, b_t, S, scale)
        outs.append(o_t)
    return torch.stack(outs, dim=2).to(in_dtype), S


# ----------------------------------------------------------------------------------------
# Decayed Gram matrices.
#
# For one chunk, with cumulative log decay c_i (a vector per token, non-increasing in i):
#     M_ij = sum_d k_id k_jd exp(c_id - c_jd)    for j <  i, else 0
#     P_ij = sum_d q_id k_jd exp(c_id - c_jd)    for j <= i, else 0
# The direct way builds a (C, C, D) tensor. Factorising exp(c_i) exp(-c_j) overflows.
# Divide and conquer fixes both: split a block into a low and a high half and let r be c
# at the last token of the low half. For i in the high half and j in the low half,
# c_i <= r <= c_j, so exp(c_i - c_j) = exp(c_i - r) exp(r - c_j) with both factors <= 1,
# and the (high, low) block is one matmul. Recurse into the halves for the rest.
# ----------------------------------------------------------------------------------------


def _decayed_grams_naive(q, k, c):
    """Explicit (..., C, C, D) version: the definition, written out. Slow and memory-heavy, and
    the one to read first. Everything else must agree with it."""
    C = k.shape[-2]
    diff = c[..., :, None, :] - c[..., None, :, :]                # c_i - c_j, (..., C, C, D)
    idx = torch.arange(C, device=k.device)
    incl = idx[:, None] >= idx[None, :]
    strict = idx[:, None] > idx[None, :]
    # mask before the exp so the upper triangle (positive diffs) can't overflow
    M = (k[..., :, None, :] * k[..., None, :, :] * diff.masked_fill(~strict[..., None], float("-inf")).exp()).sum(-1)
    P = (q[..., :, None, :] * k[..., None, :, :] * diff.masked_fill(~incl[..., None], float("-inf")).exp()).sum(-1)
    return M, P


def _level_factors(k, c, s):
    """Split every size-s block into halves. Returns blocked k, c and the two decay factors."""
    C = k.shape[-2]
    h, nb = s // 2, C // s
    kb, cb = (x.unflatten(-2, (nb, s)) for x in (k, c))           # (..., nb, s, D)
    r = cb[..., h - 1:h, :]                                       # c at the last row of the low half
    e_hi = (cb[..., h:, :] - r).exp()                             # <= 1
    e_lo = (r - cb[..., :h, :]).exp()                             # <= 1
    return kb, e_hi, e_lo


def _decayed_grams_dc(q, k, c):
    """Differentiable divide and conquer, bottom-up. Inputs (..., C, D), returns (M, P) (..., C, C)."""
    C = k.shape[-2]
    assert C & (C - 1) == 0, "chunk size must be a power of two"
    M = k.new_zeros(*k.shape[:-2], C, 1, 1)                       # level 0: C blocks of size 1
    P = (q * k).sum(-1)[..., None, None]
    s = 2
    while s <= C:
        h, nb = s // 2, C // s
        kb, e_hi, e_lo = _level_factors(k, c, s)
        qb = q.unflatten(-2, (nb, s))
        Rt = (kb[..., :h, :] * e_lo).transpose(-1, -2)            # (..., nb, D, h)
        M_blk = (kb[..., h:, :] * e_hi) @ Rt                      # (..., nb, h, h)
        P_blk = (qb[..., h:, :] * e_hi) @ Rt

        def assemble(prev, blk):
            # prev holds 2*nb diagonal blocks of size h; new block = [[even, 0], [blk, odd]]
            top = torch.cat([prev[..., 0::2, :, :], torch.zeros_like(blk)], dim=-1)
            bot = torch.cat([blk, prev[..., 1::2, :, :]], dim=-1)
            return torch.cat([top, bot], dim=-2)

        M, P = assemble(M, M_blk), assemble(P, P_blk)
        s *= 2
    return M.squeeze(-3), P.squeeze(-3)


class _DecayedGrams(torch.autograd.Function):
    """Same values as _decayed_grams_dc, but saves only q, k, c and recomputes the per-level
    factors in backward (autograd would keep several hidden-size tensors per level)."""

    @staticmethod
    def forward(ctx, q, k, c):
        ctx.save_for_backward(q, k, c)
        with torch.no_grad():
            return _decayed_grams_dc(q, k, c)

    @staticmethod
    @once_differentiable
    def backward(ctx, gM, gP):
        q, k, c = ctx.saved_tensors
        C = k.shape[-2]

        # Walk the levels top-down. cur_* hold the nb diagonal s-by-s blocks of the gradient,
        # (..., nb, s, s); the next level's blocks are the two diagonal halves of each.
        cur_M, cur_P = gM[..., None, :, :], gP[..., None, :, :]
        gq, gk, gc = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(c)
        s = C
        while s >= 2:
            h, nb = s // 2, C // s
            kb, e_hi, e_lo = _level_factors(k, c, s)
            qb = q.unflatten(-2, (nb, s))

            gM_blk, gP_blk = cur_M[..., h:, :h], cur_P[..., h:, :h]   # (high rows, low cols)
            Lk, Lq = kb[..., h:, :] * e_hi, qb[..., h:, :] * e_hi
            R = kb[..., :h, :] * e_lo

            dLk, dLq = gM_blk @ R, gP_blk @ R
            dR = gM_blk.transpose(-1, -2) @ Lk + gP_blk.transpose(-1, -2) @ Lq
            t_hi = dLk * Lk + dLq * Lq                            # grad w.r.t. exponent c_hi - r
            t_lo = dR * R                                         # grad w.r.t. exponent r - c_lo
            # Nothing is owed to the reference row r: exp(c_i - r) exp(r - c_j) = exp(c_i - c_j)
            # does not depend on r, so t_lo and t_hi sum to the same thing and cancel.
            gkv, gqv, gcv = (x.unflatten(-2, (nb, s)) for x in (gk, gq, gc))   # views, added in place
            gkv[..., :h, :] += dR * e_lo
            gkv[..., h:, :] += dLk * e_hi
            gqv[..., h:, :] += dLq * e_hi                         # q only appears in the high half
            gcv[..., :h, :] -= t_lo
            gcv[..., h:, :] += t_hi

            cur_M = torch.stack([cur_M[..., :h, :h], cur_M[..., h:, h:]], dim=-3).flatten(-4, -3)
            cur_P = torch.stack([cur_P[..., :h, :h], cur_P[..., h:, h:]], dim=-3).flatten(-4, -3)
            s = h

        # level 0: P_ii = q_i . k_i, cur_P is now the (..., C, 1, 1) diagonal. M's diagonal is
        # a constant zero.
        dP = cur_P[..., 0]
        gq += dP * k
        gk += dP * q
        return gq, gk, gc


def decayed_grams(q, k, c):
    """Decayed Gram matrices (M, P), forward of _decayed_grams_dc with a hand-written backward."""
    return _DecayedGrams.apply(q, k, c)


def chunk_kda(q, k, v, g, beta, scale=None, initial_state=None,
              chunk_size=64, inverse="doubling", gram="auto"):
    """Chunked KDA. Same maths as the recurrent op. Returns (o, final_state).

    gram picks how the decayed Gram matrices M, P are built:
        "metal"     two Metal dispatches forward, two backward (metal_kernels.py); MPS only
        "lean"      divide and conquer in PyTorch with a hand-written backward
        "autograd"  divide and conquer in PyTorch, differentiated by autograd
        "naive"     the definition written out as a (C, C, Dk) tensor; plain, uses the most memory
        "auto"      "metal" on MPS in float32, otherwise "lean"

    Within a chunk of C tokens, with cumulative log-decay c_i = g_1 + ... + g_i (a vector per
    token) and the state S0 entering the chunk, define u_i = beta_i * (v_i - (Diag(alpha_i)
    S_{i-1})^T k_i), the amount actually written at token i. Then

        S_i = Diag(e^{c_i}) S0 + sum_{j<=i} Diag(e^{c_i - c_j}) k_j u_j^T

    and the u's solve a unit lower-triangular system

        (I + diag(beta) M) U = diag(beta) (V - Kt S0),
        Kt_i = e^{c_i} * k_i (elementwise),
        M_ij = sum_d k_id k_jd e^{c_id - c_jd} for j < i, else 0.

    With Tinv = (I + diag(beta) M)^{-1}:  U = Tinv diag(beta) V  -  Tinv diag(beta) Kt S0
                                            = U_pre             -  W S0.
    U_pre, W, M and P depend only on the chunk, so all chunks are done in parallel. What is
    left is carrying S0 from chunk to chunk:

        U = U_pre - W S0
        O = scale * ((e^c * Q) S0 + P U),   P_ij = sum_d q_id k_jd e^{c_id - c_jd} for j <= i
        S_next = Diag(e^{c_C}) S0 + (e^{c_C - c} * K)^T U

    Substituting U makes both lines affine in S0 (q_dec = e^c * Q, k_dec = e^{c_C - c} * K):

        S_next = Abar S0 + Bm,   Abar = Diag(e^{c_C}) - k_dec^T W (Dk x Dk),   Bm = k_dec^T U_pre
        O      = Qe S0 + O0,     Qe = q_dec - P W,                              O0 = P U_pre

    Abar, Bm, Qe, O0 are batched matmuls over all chunks. The sequential loop is then one
    baddbmm per chunk (op-dispatch overhead, not arithmetic, dominates), and the outputs of
    all chunks come from one batched matmul on the stacked entering states. Memory: Abar is
    n x Dk x Dk per head and the stacked states n x Dk x Dv per head (see _scan_chunks).

    M and P need a decay that differs per channel, so they cannot be a masked (C, C) factor
    times Q K^T as in the scalar-decay GDN. See _decayed_grams_dc for how they are built.
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    C = chunk_size
    dtype = torch.promote_types(q.dtype, torch.float32)
    in_dtype = v.dtype
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))
    if scale is None:
        scale = Dk ** -0.5
    if gram == "auto":
        on_mps = q.device.type == "mps" and _metal.metal_available()
        gram = "metal" if (on_mps and dtype == torch.float32) else "lean"
    grams = {"lean": decayed_grams, "autograd": _decayed_grams_dc, "naive": _decayed_grams_naive,
             "metal": _metal.decayed_grams_metal}[gram]

    # Pad T up to a multiple of C. Padded tokens have k = v = q = 0, beta = 0, g = 0
    # (alpha = 1), so they neither write, decay, nor read.
    pad = (-T) % C
    if pad:
        q, k, v, g = (torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (q, k, v, g))
        beta = torch.nn.functional.pad(beta, (0, pad))
    n = (T + pad) // C

    # (B, H, n, C, D)
    q, k, v, g = (x.reshape(B, H, n, C, -1) for x in (q, k, v, g))
    beta = beta.reshape(B, H, n, C)

    c = g.cumsum(-2)                                              # (B, H, n, C, Dk)
    M, P = grams(q, k, c)                                         # (B, H, n, C, C)
    L = torch.eye(C, device=q.device, dtype=dtype) + beta[..., :, None] * M
    Tinv = _unit_lower_inverse(L, inverse)

    e_c = c.exp()
    U_pre = Tinv @ (beta[..., None] * v)                          # (B, H, n, C, Dv)
    W = Tinv @ (beta[..., None] * e_c * k)                        # (B, H, n, C, Dk)
    q_dec = q * e_c                                               # e^{c_i} * q_i
    k_dec = k * (c[..., -1:, :] - c).exp()                        # e^{c_C - c_j} * k_j
    chunk_decay = c[..., -1, :].exp()                             # (B, H, n, Dk)

    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state.to(dtype)
    O, S = _scan_chunks(U_pre, W, q_dec, P, k_dec, chunk_decay, S)   # decay acts on the Dk axis

    o = O.reshape(B, H, n * C, Dv)[:, :, :T] * scale
    return o.to(in_dtype), S
