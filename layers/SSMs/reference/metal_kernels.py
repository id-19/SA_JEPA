"""Metal kernels for KDA on Apple GPUs, called through torch.mps.compile_shader.

Reference implementation written by an AI agent (13 Nov 2026). It is a speed baseline for
MPS, not the learning implementation: every kernel here has a PyTorch twin in kda_ops.py
that it must match with allclose.

Kernels (float32 only):
    kda_gram_tables  q, k, c -> Fk, Fq     per-level decay-scaled rows (N, L, C, D)
    kda_gram_dots    tables  -> M, P       the decayed Gram matrices, forward
    kda_gram_bwd     tables, gM, gP -> gq, gk, gc
    kda_step                               one decode token, state updated in place

The decayed Gram matrices are the divide and conquer of kda_ops._decayed_grams_dc (see the
"Decayed Gram matrices" comment there), but done in two dispatches instead of log2(C)
batched matmuls, which on MPS are dominated by per-op overhead.

compile_shader facts (torch 2.13) the code relies on:
    - tensors bind to `device float*` in argument order, Python floats to `constant float&`
    - Python ints arrive as 64-bit: pass each as its own `constant uint&` (the low 32 bits
      are read), never as a tuple
    - a non-contiguous view is passed as raw storage, so inputs are made contiguous and
      outputs are freshly allocated; a storage offset (contiguous slice) is honoured
    - `threads=(X, Y, Z)` with `uint3 gid [[thread_position_in_grid]]` gives a 3-D grid
"""

import torch
from torch.autograd.function import once_differentiable

_SRC = r"""
#include <metal_stdlib>
using namespace metal;

// Tables for the divide and conquer. Level b splits every block of 2h rows (h = 1 << b)
// into a low and a high half; r is the last row of the low half. For j < i split at level
// b = highest differing bit of (i, j): exp(c_i - c_j) = exp(c_i - c_r) * exp(c_r - c_j),
// both factors <= 1. High rows hold exp(c_i - c_r), low rows exp(c_r - c_i).
// grid: (d, i, n)
kernel void kda_gram_tables(device const float* q, device const float* k, device const float* c,
                            device float* Fk, device float* Fq,
                            constant uint& C, constant uint& D, constant uint& L,
                            uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x, i = gid.y, n = gid.z;
    const uint row = (n * C + i) * D + d;
    const float ci = c[row], ki = k[row], qi = q[row];
    for (uint b = 0; b < L; b++) {
        const uint h = 1u << b;
        const uint r = (i & ~((h << 1) - 1u)) + h - 1u;           // reference row of the block
        const float cr = c[(n * C + r) * D + d];
        const float e = (i & h) ? exp(ci - cr) : exp(cr - ci);    // <= 1 either way
        const uint t = ((n * L + b) * C + i) * D + d;
        Fk[t] = ki * e;
        Fq[t] = qi * e;                                           // only read for high rows
    }
}

// M and P from the tables: one thread per (j, i). grid: (j, i, n)
kernel void kda_gram_dots(device const float* Fk, device const float* Fq,
                          device const float* q, device const float* k,
                          device float* M, device float* P,
                          constant uint& C, constant uint& D, constant uint& L,
                          uint3 gid [[thread_position_in_grid]])
{
    const uint j = gid.x, i = gid.y, n = gid.z;
    const uint out = (n * C + i) * C + j;
    if (j > i) { M[out] = 0.0f; P[out] = 0.0f; return; }
    float m = 0.0f, p = 0.0f;
    if (j == i) {
        // P_ii = q_i . k_i, M_ii = 0
        device const float* qi = q + (n * C + i) * D;
        device const float* ki = k + (n * C + i) * D;
        for (uint d = 0; d < D; d++) p += qi[d] * ki[d];
    } else {
        const uint b = 31u - clz(i ^ j);                          // level where i and j split
        device const float* fki = Fk + ((n * L + b) * C + i) * D;
        device const float* fqi = Fq + ((n * L + b) * C + i) * D;
        device const float* fkj = Fk + ((n * L + b) * C + j) * D;
        // four independent accumulators so the sum is not one long dependency chain
        float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
        float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
        uint d = 0;
        for (; d + 4 <= D; d += 4) {
            m0 += fki[d] * fkj[d];         p0 += fqi[d] * fkj[d];
            m1 += fki[d + 1] * fkj[d + 1]; p1 += fqi[d + 1] * fkj[d + 1];
            m2 += fki[d + 2] * fkj[d + 2]; p2 += fqi[d + 2] * fkj[d + 2];
            m3 += fki[d + 3] * fkj[d + 3]; p3 += fqi[d + 3] * fkj[d + 3];
        }
        for (; d < D; d++) { m0 += fki[d] * fkj[d]; p0 += fqi[d] * fkj[d]; }   // tail
        m = (m0 + m1) + (m2 + m3);
        p = (p0 + p1) + (p2 + p3);
    }
    M[out] = m;
    P[out] = p;
}

// Backward of the Gram matrices. One thread per (d, i); each level contributes the
// partners of row i, which are the other half of its block. The reference row r gets no
// gradient: the product of the two factors does not depend on c_r. grid: (d, i, n)
// gM, gP entries above the diagonal and M's diagonal are never read.
kernel void kda_gram_bwd(device const float* gM, device const float* gP,
                         device const float* Fk, device const float* Fq,
                         device const float* q, device const float* k, device const float* c,
                         device float* gq, device float* gk, device float* gc,
                         constant uint& C, constant uint& D, constant uint& L,
                         uint3 gid [[thread_position_in_grid]])
{
    const uint d = gid.x, i = gid.y, n = gid.z;
    const uint row = (n * C + i) * D + d;
    const float ci = c[row], ki = k[row], qi = q[row];
    device const float* gMi = gM + (n * C + i) * C;               // row i of gM
    device const float* gPi = gP + (n * C + i) * C;
    const float gdiag = gPi[i];                                   // P_ii = q_i . k_i
    float a_gk = gdiag * qi, a_gq = gdiag * ki, a_gc = 0.0f;
    for (uint b = 0; b < L; b++) {
        const uint h = 1u << b;
        const uint start = i & ~((h << 1) - 1u);
        const uint r = start + h - 1u;
        const float cr = c[(n * C + r) * D + d];
        device const float* fk = Fk + (n * L + b) * C * D + d;    // + j * D below
        if (i & h) {
            // high row: partners j are the low half, columns of row i
            const float e = exp(ci - cr);
            float aM = 0.0f, aP = 0.0f;
            for (uint j = start; j < start + h; j++) {
                const float f = fk[j * D];
                aM += gMi[j] * f;
                aP += gPi[j] * f;
            }
            a_gk += aM * e;
            a_gq += aP * e;
            a_gc += (aM * ki + aP * qi) * e;
        } else {
            // low row: partners j are the high half, entries (j, i) of the matrices
            const float e = exp(cr - ci);
            device const float* fq = Fq + (n * L + b) * C * D + d;
            float a = 0.0f;
            for (uint j = start + h; j < start + 2u * h; j++) {
                const uint g = (n * C + j) * C + i;
                a += gM[g] * fk[j * D] + gP[g] * fq[j * D];
            }
            a_gk += a * e;
            a_gc -= a * ki * e;
        }
    }
    gq[row] = a_gq;
    gk[row] = a_gk;
    gc[row] = a_gc;
}

// One decode token. One thread per (head n, value column): each column of S evolves
// independently. grid: (Dv, B * H)
kernel void kda_step(device float* S, device float* out,
                     device const float* q, device const float* k, device const float* v,
                     device const float* g, device const float* beta,
                     constant uint& Dk, constant uint& Dv, constant float& scale,
                     uint2 gid [[thread_position_in_grid]])
{
    // one thread per (head n, value column): each column of S evolves independently
    const uint n = gid.y, col = gid.x;
    const uint kb = n * Dk;
    device float* Sn = S + n * Dk * Dv + col;
    float old = 0.0f;
    for (uint d = 0; d < Dk; d++) old += exp(g[kb + d]) * Sn[d * Dv] * k[kb + d];
    const float err = beta[n] * (v[n * Dv + col] - old);
    float acc = 0.0f;
    for (uint d = 0; d < Dk; d++) {
        const float s = exp(g[kb + d]) * Sn[d * Dv] + k[kb + d] * err;
        Sn[d * Dv] = s;
        acc += q[kb + d] * s;
    }
    out[n * Dv + col] = scale * acc;
}
"""

_lib = None


def metal_available():
    return torch.backends.mps.is_available() and hasattr(torch.mps, "compile_shader")


def _get_lib():
    # compiled lazily and once, so importing this module on a non-MPS machine does nothing
    global _lib
    if _lib is None:
        _lib = torch.mps.compile_shader(_SRC)
    return _lib


def _f32(x):
    return x.to(torch.float32).contiguous()                      # views are passed as raw storage


def kda_step_metal(q, k, v, g, beta, state, scale=None):
    """One decode token on the GPU. q, k: (B, H, Dk); v: (B, H, Dv); g: (B, H, Dk) or (B, H)
    (scalar decay per head, Gated DeltaNet); beta: (B, H); state: (B, H, Dk, Dv) float32,
    contiguous, on MPS. The state is updated IN PLACE and returned. No autograd: inputs must
    not require grad. Returns (o, state), o in v's dtype."""
    for x in (q, k, v, g, beta):
        assert not x.requires_grad, "kda_step_metal has no autograd (decode only)"
    if state.dtype != torch.float32 or not state.is_contiguous() or state.device.type != "mps":
        raise ValueError("state must be a contiguous float32 tensor on MPS "
                         f"(got {state.dtype}, contiguous={state.is_contiguous()}, device={state.device})")
    B, H, Dk = q.shape
    Dv = v.shape[-1]
    if scale is None:
        scale = Dk ** -0.5
    in_dtype = v.dtype
    if g.dim() == 2:
        g = g[..., None].expand(B, H, Dk)                         # scalar decay -> same value per channel
    q, k, v, g, beta = (_f32(x) for x in (q, k, v, g, beta))
    out = torch.empty(B, H, Dv, dtype=torch.float32, device=state.device)
    _get_lib().kda_step(state, out, q, k, v, g, beta, Dk, Dv, float(scale), threads=(Dv, B * H))
    return out.to(in_dtype), state


def _slab_size(N, L, C, D, table_budget_mb):
    budget_floats = int(table_budget_mb * (1 << 20) / 4)
    slab = max(1, budget_floats // (2 * L * C * D))
    slab = min(slab, N)
    # shader index arithmetic is 32-bit
    assert slab * L * C * D < 2 ** 32 and N * C * C < 2 ** 32 and N * C * D < 2 ** 32, "too large for 32-bit indexing"
    return slab


class _DecayedGramsMetal(torch.autograd.Function):
    """q, k, c: (N, C, D) float32 contiguous, C a power of two >= 2. Saves only q, k, c; the
    tables are recomputed in backward (kernel A again)."""

    @staticmethod
    def forward(ctx, q, k, c, table_budget_mb):
        ctx.save_for_backward(q, k, c)
        ctx.budget = table_budget_mb
        N, C, D = q.shape
        L = C.bit_length() - 1
        lib = _get_lib()
        M = torch.empty(N, C, C, dtype=torch.float32, device=q.device)
        P = torch.empty_like(M)
        slab = _slab_size(N, L, C, D, table_budget_mb)
        Fk = torch.empty(slab, L, C, D, dtype=torch.float32, device=q.device)   # reused across slabs
        Fq = torch.empty_like(Fk)
        for n0 in range(0, N, slab):
            n1 = min(n0 + slab, N)
            m = n1 - n0
            qs, ks, cs = q[n0:n1], k[n0:n1], c[n0:n1]             # leading-dim slices stay contiguous
            lib.kda_gram_tables(qs, ks, cs, Fk, Fq, C, D, L, threads=(D, C, m))
            lib.kda_gram_dots(Fk, Fq, qs, ks, M[n0:n1], P[n0:n1], C, D, L, threads=(C, C, m))
        return M, P

    @staticmethod
    @once_differentiable
    def backward(ctx, gM, gP):
        q, k, c = ctx.saved_tensors
        N, C, D = q.shape
        L = C.bit_length() - 1
        lib = _get_lib()
        gM = torch.zeros(N, C, C, device=q.device) if gM is None else _f32(gM)
        gP = torch.zeros(N, C, C, device=q.device) if gP is None else _f32(gP)
        gq, gk, gc = (torch.empty(N, C, D, dtype=torch.float32, device=q.device) for _ in range(3))
        slab = _slab_size(N, L, C, D, ctx.budget)
        Fk = torch.empty(slab, L, C, D, dtype=torch.float32, device=q.device)
        Fq = torch.empty_like(Fk)
        for n0 in range(0, N, slab):
            n1 = min(n0 + slab, N)
            m = n1 - n0
            qs, ks, cs = q[n0:n1], k[n0:n1], c[n0:n1]
            lib.kda_gram_tables(qs, ks, cs, Fk, Fq, C, D, L, threads=(D, C, m))
            lib.kda_gram_bwd(gM[n0:n1], gP[n0:n1], Fk, Fq, qs, ks, cs,
                             gq[n0:n1], gk[n0:n1], gc[n0:n1], C, D, L, threads=(D, C, m))
        return gq, gk, gc, None


def decayed_grams_metal(q, k, c, table_budget_mb=64):
    """Decayed Gram matrices (M, P) on the GPU, differentiable. Inputs (..., C, D) on MPS, C a
    power of two and >= 2 (C == 1 is not supported: there are no levels). Returns float32
    (..., C, C). The per-level tables take 2 * N * L * C * D floats; they are built in slabs
    of N so that they stay under table_budget_mb."""
    C, D = k.shape[-2:]
    assert C >= 2 and C & (C - 1) == 0, "chunk size must be a power of two >= 2"
    lead = k.shape[:-2]
    q, k, c = (_f32(x).reshape(-1, C, D) for x in (q, k, c))
    M, P = _DecayedGramsMetal.apply(q, k, c, table_budget_mb)
    return M.reshape(*lead, C, C), P.reshape(*lead, C, C)
