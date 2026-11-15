"""Grouped-query attention, streaming-capable. AI-written reference (13 Nov 2026): a
correctness oracle and speed baseline, not the learning implementation.

    x        (B, T, d_model)               context (B, S, d_context), cross-attention only
    q        (B, H, T, D)                  H = n_heads, D = head_dim
    k, v     (B, Hk, S, D)                 Hk = n_kv_heads; G = H // Hk query heads share one k/v head
    y        (B, T, d_model)

    q, k = L2norm(q), L2norm(k) * exp(logit_scale)   (only if qk_norm; then no 1/sqrt(D))
    q, k = RoPE(q), RoPE(k)                          (only if rope is given)
    y    = o_proj( softmax(q k^T / sqrt(D) + mask) v )

Query head h reads k/v head h // G (consecutive heads share), same as repeat_interleave and
F.scaled_dot_product_attention(enable_gqa=True). Hk == H is plain multi-head, Hk == 1 multi-query.
No residual and no pre-norm: the wrapping block owns them.

Self-attention masks work on the token index in the stream (count of tokens seen), not on
`positions`, which only feed RoPE. causal: t attends s <= t. window=w: also t - s <= w (itself
plus the w previous tokens), the same as layers/Attention_layers/swa.py.
Cross-attention (context given): no causal mask, no window. RoPE is applied to both sides only
if `context_positions` is given, otherwise to neither. Rows with no visible key output zeros.

State (plain dict, save it with torch.save):
    self-attn   {"k": (B, Hk, cap, D) with RoPE applied, "v": same, "mask": (B, cap) bool or None,
                 "len": int valid entries along cap, "n_seen": int tokens fed so far}. k, v, mask
                 are preallocated buffers (see cache.py); read them with cache_view(buf, len, dim).
                 With window=w only the last w entries are ever read and capacity stays bounded.
    cross-attn  {"k", "v": projected context (B, Hk, S, D) exactly sized, "mask", "n_seen" (counts
                 queries, offsets default query positions), "cross": True, "roped": bool}. A later
                 call with such a state reuses k, v and ignores `context` and `key_padding_mask`.
A state is consumed by the call it is passed to (buffers are written in place and shared with the
returned state); use cache.clone_state to branch from one prefix.
key_padding_mask (B, S) bool, True = may be attended; with a state it covers only the new tokens.

Memory: under torch.no_grad(), if n_heads * T * S > max_scores the queries are processed in blocks
of max_scores // (n_heads * S) so the score tensor is never materialised in full. With grad enabled
there is no blocking (autograd would keep every block's scores anyway).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .cache import arange_cached, cache_extend, cache_view
from .rope import apply_rope, default_positions


def _mask(q_idx, S, causal, window, kpm):
    """Bool (1 or B, Tq, S), True = may attend, or None. q_idx: index of each query in the cache."""
    m = None
    if causal:
        dist = q_idx[:, None] - torch.arange(S, device=q_idx.device)[None, :]
        m = (dist >= 0) if window is None else (dist >= 0) & (dist <= window)
        m = m[None]
    if kpm is not None:
        m = kpm[:, None, :] if m is None else m & kpm[:, None, :]
    return m


class GQARef(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads=None, head_dim=None, causal=True, window=None,
                 rope=None, qk_norm=False, d_context=None, impl="sdpa", max_scores=2 ** 26):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = head_dim or d_model // n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert window is None or causal, "window needs causal"
        assert impl in ("sdpa", "manual")
        assert rope is None or rope.dim == self.head_dim
        self.causal, self.window, self.rope = causal, window, rope
        self.qk_norm, self.impl, self.max_scores = qk_norm, impl, max_scores
        D = self.head_dim
        self.q_proj = nn.Linear(d_model, n_heads * D, bias=False)
        self.k_proj = nn.Linear(d_context or d_model, self.n_kv_heads * D, bias=False)
        self.v_proj = nn.Linear(d_context or d_model, self.n_kv_heads * D, bias=False)
        self.o_proj = nn.Linear(n_heads * D, d_model, bias=False)
        if qk_norm:
            # cosine logits are in [-1, 1]; a learned per-head temperature (log-parametrised)
            self.logit_scale = nn.Parameter(torch.full((n_heads,), math.log(D ** 0.5)))
        self.scale = 1.0 if qk_norm else D ** -0.5      # folded into q when qk_norm

    def cache_floats_per_token(self):
        return 2 * self.n_kv_heads * self.head_dim

    def _heads(self, t, n):
        B, L, _ = t.shape
        return t.view(B, L, n, self.head_dim).transpose(1, 2)

    # q (B,H,T,D), k/v (B,Hk,S,D), mask (1|B,T,S) or None  ->  (B,H,T,D)
    def _sdpa(self, q, k, v, mask, is_causal):
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None if mask is None else mask[:, None], is_causal=is_causal,
            scale=self.scale, enable_gqa=True)
        if mask is not None:                            # fully masked rows: zeros, never NaN
            out = out.masked_fill(~mask.any(-1)[:, None, :, None], 0)
        return out

    def _manual(self, q, k, v, mask):
        B, H, T, D = q.shape
        q = q.view(B, self.n_kv_heads, H // self.n_kv_heads, T, D)   # k/v are never expanded
        s = torch.einsum("bkgtd,bksd->bkgts", q, k) * self.scale
        if mask is not None:
            mask = mask[:, None, None]                  # (1|B, 1, 1, T, S)
            # finite fill, not -inf: a fully masked row stays finite (uniform), so backward has no NaN
            s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        w = s.softmax(-1)
        if mask is not None:
            w = w.masked_fill(~mask.any(-1, keepdim=True), 0)
        return torch.einsum("bkgts,bksd->bkgtd", w, v).reshape(B, H, T, D)

    def _attend(self, q, k, v, mask_fn, pure_causal):
        B, H, T, D = q.shape
        S = k.shape[2]
        blk = T
        if not torch.is_grad_enabled() and H * T * S > self.max_scores:
            blk = max(1, self.max_scores // (H * S))
        if blk >= T:
            if self.impl == "sdpa":
                return self._sdpa(q, k, v, None if pure_causal else mask_fn(0, T), pure_causal)
            return self._manual(q, k, v, mask_fn(0, T))
        run = self._sdpa if self.impl == "sdpa" else (lambda q, k, v, m, _: self._manual(q, k, v, m))
        return torch.cat([run(q[:, :, i:i + blk], k, v, mask_fn(i, i + blk), False)
                          for i in range(0, T, blk)], dim=2)

    def forward(self, x, context=None, positions=None, context_positions=None,
                key_padding_mask=None, state=None, return_state=False):
        B, T, _ = x.shape
        cached = state is not None and state.get("cross", False)
        cross = context is not None or cached
        n0 = state["n_seen"] if state is not None else 0

        q = self._heads(self.q_proj(x), self.n_heads)
        if self.qk_norm:
            q = F.normalize(q, dim=-1) * self.logit_scale.exp().view(1, -1, 1, 1).to(q.dtype)

        if cached:
            roped = state["roped"]
        elif cross:
            roped = self.rope is not None and context_positions is not None
        else:
            roped = self.rope is not None
        if roped:
            cos, sin = self.rope(default_positions(positions, T, n0, x.device, self.rope))
            q = apply_rope(q, cos, sin)

        bufs, static = None, False
        if cached:
            k, v, mask = state["k"], state["v"], state["mask"]
        else:
            src = context if cross else x
            k = self._heads(self.k_proj(src), self.n_kv_heads)
            v = self._heads(self.v_proj(src), self.n_kv_heads)
            if self.qk_norm:
                k = F.normalize(k, dim=-1)
            if roped:
                k = apply_rope(k, *self.rope(context_positions)) if cross else apply_rope(k, cos, sin)
            mask = key_padding_mask
            if state is not None and not cross:
                bufs, n_new = cache_extend(state, {"k": k, "v": v}, 2, mask, self.window)
                if T == 1 and bufs["mask"] is not None:
                    # decode with padding: attend over the whole buffer with one fixed-shape validity
                    # mask, so shapes only change when the buffer reallocates (MPS builds a graph per shape)
                    idx = arange_cached(bufs["k"].shape[2], x.device)
                    ok = idx < n_new
                    if self.window is not None:
                        ok = ok & (idx >= n_new - 1 - self.window)
                    k, v, mask, static = bufs["k"], bufs["v"], bufs["mask"] & ok, True
                else:
                    last = None if self.window is None else self.window + T   # all the mask can reach
                    k, v = cache_view(bufs["k"], n_new, 2, last), cache_view(bufs["v"], n_new, 2, last)
                    mask = None if bufs["mask"] is None else cache_view(bufs["mask"], n_new, 1, last)

        S = k.shape[2]
        causal = self.causal and not cross
        window = None if cross else self.window
        if static:
            mask_fn = lambda i, j: mask[:, None, :]
        elif mask is None and (not causal or T == 1):
            # a lone query (or any non-causal one) may see every key it is handed: build no mask
            mask_fn = lambda i, j: None
        else:
            q_idx = (S - T) + torch.arange(T, device=x.device)    # index of each query among the visible keys
            mask_fn = lambda i, j: _mask(q_idx[i:j], S, causal, window, mask)
        pure_causal = causal and window is None and mask is None and S == T
        o = self._attend(q, k, v, mask_fn, pure_causal)
        y = self.o_proj(o.transpose(1, 2).reshape(B, T, -1))
        if not return_state:
            return y

        if cross:
            return y, {"k": k, "v": v, "mask": mask, "n_seen": n0 + T, "cross": True, "roped": roped}
        if bufs is None:        # first call: the cache keeps only what later tokens can reach
            s = 0 if window is None else max(T - window, 0)
            bufs, n_new = cache_extend(None, {"k": k[:, :, s:], "v": v[:, :, s:]}, 2,
                                       None if mask is None else mask[:, s:], window)
        return y, {"k": bufs["k"], "v": bufs["v"], "mask": bufs["mask"], "len": n_new, "n_seen": n0 + T}
