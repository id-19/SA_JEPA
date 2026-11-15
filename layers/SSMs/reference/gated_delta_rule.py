"""Gated delta rule ops: a token-by-token reference and a chunked version for MPS/CPU.

Reference implementation written by an AI agent (13 Nov 2026). It is a correctness
oracle and a speed baseline, not the learning implementation: the user's own GDN lives
in layers/SSMs/multi_headed/ and should match this one with allclose on the same inputs.

Shapes (all ops):
    q, k:  (B, H, T, Dk)    keys and queries, already L2-normalised per head
    v:     (B, H, T, Dv)
    g:     (B, H, T)        log of the decay alpha, so g <= 0 and alpha = exp(g)
    beta:  (B, H, T)        write strength in (0, 1)
    state: (B, H, Dk, Dv)   S maps a key to a value: value = S^T key

Recurrence, per head and per token t:
    S_t = alpha_t * (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T
    o_t = scale * S_t^T q_t
which is the same as: decay S, read the old value at k_t, write beta_t * (v_t - old).

Everything runs in float32 internally (MPS has no float64).
"""

import torch

from . import metal_kernels as _metal


def recurrent_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None):
    """Token-by-token gated delta rule. Slow, obviously correct. Returns (o, final_state)."""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    in_dtype = v.dtype
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if scale is None:
        scale = Dk ** -0.5

    # Decode on MPS (no gradient needed): one fused Metal dispatch per token instead of about
    # ten PyTorch ops. The kernel is KDA's; a scalar decay is the same value on every channel.
    needs_grad = torch.is_grad_enabled() and any(
        x is not None and x.requires_grad for x in (q, k, v, g, beta, initial_state))
    if q.device.type == "mps" and _metal.metal_available() and not needs_grad:
        S = (q.new_zeros(B, H, Dk, Dv) if initial_state is None
             else initial_state.detach().float().contiguous().clone())    # the kernel writes in place
        outs = []
        for q_t, k_t, v_t, g_t, b_t in zip(*(x.detach().unbind(2) for x in (q, k, v, g, beta))):
            o_t, S = _metal.kda_step_metal(q_t, k_t, v_t, g_t, b_t, S, scale)
            outs.append(o_t)
        return torch.stack(outs, dim=2).to(in_dtype), S

    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state.float().clone()
    o = q.new_empty(B, H, T, Dv)
    for t in range(T):
        S = S * g[:, :, t].exp()[..., None, None]                 # decay
        k_t = k[:, :, t]                                          # (B, H, Dk)
        old = torch.einsum("bhkv,bhk->bhv", S, k_t)               # what S currently returns for k_t
        err = (v[:, :, t] - old) * beta[:, :, t, None]            # (B, H, Dv)
        S = S + k_t[..., :, None] * err[..., None, :]             # rank-1 write
        o[:, :, t] = torch.einsum("bhkv,bhk->bhv", S, q[:, :, t]) * scale
    return o.to(in_dtype), S


def _unit_lower_inverse(L, method):
    """Inverse of a unit lower-triangular matrix L (..., C, C).

    "solve":   torch.linalg.solve_triangular.
    "doubling": L = I + N with N strictly lower, so N^C = 0 and
                L^{-1} = (I - N)(I + N^2)(I + N^4)... — log2(C) rounds of plain matmuls,
                which MPS handles well.
    """
    C = L.shape[-1]
    eye = torch.eye(C, device=L.device, dtype=L.dtype)
    if method == "solve":
        return torch.linalg.solve_triangular(L, eye.expand_as(L), upper=False, unitriangular=True)
    if method == "doubling":
        N = L - eye
        inv = eye - N
        P = N @ N
        n = 2
        while n < C:
            inv = inv + inv @ P          # inv @ (I + P)
            P = P @ P
            n *= 2
        return inv
    raise ValueError(f"unknown method {method!r}")


def _scan_chunks(U_pre, W, q_dec, P, k_dec, decay, S0):
    """Carry the state across chunks with one op per chunk. Shared by GDN and KDA.

    Per chunk, with entering state S: U = U_pre - W S, O = q_dec S + P U,
    S_next = Diag(decay) S + k_dec^T U. Substituting U, everything is affine in S:

        S_next = A S + Bm,   A  = Diag(decay) - k_dec^T W,   Bm = k_dec^T U_pre
        O      = Qe S + O0,  Qe = q_dec - P W,               O0 = P U_pre

    A, Bm, Qe, O0 are batched over all chunks. The loop is a single baddbmm per chunk, and
    the outputs of all chunks come from one batched matmul on the stacked entering states.

    Memory: A is n x Dk x Dk per head and S_stack is n x Dk x Dv per head (autograd kept
    every S in the old loop as well, but A is new).

    U_pre (B,H,n,C,Dv), W, q_dec, k_dec (B,H,n,C,Dk), P (B,H,n,C,C), S0 (B,H,Dk,Dv).
    decay: (B,H,n) scalar per chunk (GDN) or (B,H,n,Dk) per channel (KDA).
    Returns O (B,H,n,C,Dv) (unscaled) and the final state (B,H,Dk,Dv).
    """
    B, H, n, C, Dk = W.shape
    Dv = U_pre.shape[-1]
    if decay.dim() == U_pre.dim() - 2:                            # (B, H, n): same decay on every channel
        decay = decay[..., None].expand(B, H, n, Dk)
    kdT = k_dec.transpose(-1, -2)
    A = torch.diag_embed(decay) - kdT @ W                         # (B, H, n, Dk, Dk)
    Bm = kdT @ U_pre                                              # (B, H, n, Dk, Dv)
    Qe = q_dec - P @ W                                            # (B, H, n, C, Dk)
    O0 = P @ U_pre                                                # (B, H, n, C, Dv)

    S = S0.reshape(B * H, Dk, Dv)
    # unbind, not x[:, i]: the backward of each index op allocates a full-size zero
    # gradient, which made the backward pass O(n^2) in the number of chunks.
    A_i = A.reshape(B * H, n, Dk, Dk).unbind(1)
    Bm_i = Bm.reshape(B * H, n, Dk, Dv).unbind(1)
    entering = []
    for a, b in zip(A_i, Bm_i):
        entering.append(S)
        S = torch.baddbmm(b, a, S)                                # S = A S + Bm

    S_stack = torch.stack(entering, dim=1).reshape(B, H, n, Dk, Dv)
    return Qe @ S_stack + O0, S.reshape(B, H, Dk, Dv)


def chunk_gated_delta_rule(q, k, v, g, beta, scale=None, initial_state=None,
                           chunk_size=64, inverse="doubling"):
    """Chunked gated delta rule. Same maths as the recurrent op. Returns (o, final_state).

    Within a chunk of C tokens, with cumulative log-decay c_i = g_1 + ... + g_i and the
    state S0 entering the chunk, define u_i = beta_i * (v_i - alpha_i S_{i-1}^T k_i), the
    amount actually written at token i. Then

        S_i = e^{c_i} S0 + sum_{j<=i} e^{c_i - c_j} k_j u_j^T

    and the u's solve a unit lower-triangular system

        (I + diag(beta) M) U = diag(beta) (V - diag(e^c) K S0),
        M_ij = e^{c_i - c_j} (k_i . k_j) for j < i, else 0.

    With T = (I + diag(beta) M)^{-1}:  U = T diag(beta) V  -  T diag(beta e^c) K S0
                                         = U_pre           -  W S0.
    U_pre and W depend only on the chunk, so all chunks are done in parallel. What is left
    is carrying S0 from chunk to chunk:

        U = U_pre - W S0
        O = scale * (e^c * Q S0 + (A * Q K^T) U),   A_ij = e^{c_i - c_j} for j <= i
        S_next = e^{c_C} S0 + (e^{c_C - c} * K)^T U

    Substituting U makes both lines affine in S0 (q_dec = e^c * Q, k_dec = e^{c_C - c} * K,
    P = A * Q K^T):

        S_next = Abar S0 + Bm,   Abar = e^{c_C} I - k_dec^T W (Dk x Dk),   Bm = k_dec^T U_pre
        O      = Qe S0 + O0,     Qe = q_dec - P W,                          O0 = P U_pre

    Abar, Bm, Qe, O0 are batched matmuls over all chunks. The sequential loop is then one
    baddbmm per chunk (it is op-dispatch overhead, not arithmetic, that dominates), and the
    outputs of all chunks come from one batched matmul on the stacked entering states.
    See _scan_chunks for the memory cost.
    """
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    C = chunk_size
    in_dtype = v.dtype
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    if scale is None:
        scale = Dk ** -0.5

    # Pad T up to a multiple of C. Padded tokens have k = v = q = 0, beta = 0, g = 0
    # (alpha = 1), so they neither write, decay, nor read.
    pad = (-T) % C
    if pad:
        q, k, v = (torch.nn.functional.pad(x, (0, 0, 0, pad)) for x in (q, k, v))
        g, beta = (torch.nn.functional.pad(x, (0, pad)) for x in (g, beta))
    n = (T + pad) // C

    # (B, H, n, C, D)
    q, k, v = (x.reshape(B, H, n, C, -1) for x in (q, k, v))
    g, beta = (x.reshape(B, H, n, C) for x in (g, beta))

    c = g.cumsum(-1)                                              # (B, H, n, C)
    diff = c[..., :, None] - c[..., None, :]                      # c_i - c_j
    idx = torch.arange(C, device=q.device)
    incl = idx[:, None] >= idx[None, :]                           # j <= i
    strict = idx[:, None] > idx[None, :]                          # j <  i
    # mask before exp so the upper triangle (positive diffs) can't overflow
    A = diff.masked_fill(~incl, float("-inf")).exp()              # (B, H, n, C, C)

    kk = k @ k.transpose(-1, -2)
    M = (A * kk).masked_fill(~strict, 0.0)
    L = torch.eye(C, device=q.device) + beta[..., :, None] * M
    Tinv = _unit_lower_inverse(L, inverse)

    U_pre = Tinv @ (beta[..., None] * v)                          # (B, H, n, C, Dv)
    W = Tinv @ ((beta * c.exp())[..., None] * k)                  # (B, H, n, C, Dk)
    P = A * (q @ k.transpose(-1, -2))                             # (B, H, n, C, C), causal incl. diagonal
    q_dec = q * c.exp()[..., None]                                # e^{c_i} q_i
    k_dec = k * (c[..., -1:] - c).exp()[..., None]                # e^{c_C - c_j} k_j
    chunk_decay = c[..., -1].exp()                                # (B, H, n)

    S = q.new_zeros(B, H, Dk, Dv) if initial_state is None else initial_state.float()
    O, S = _scan_chunks(U_pre, W, q_dec, P, k_dec, chunk_decay, S)

    o = O.reshape(B, H, n * C, Dv)[:, :, :T] * scale
    return o.to(in_dtype), S
