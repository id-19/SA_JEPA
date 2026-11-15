"""Multi-head Latent Attention, streaming-capable. AI-written reference (13 Nov 2026): a
correctness oracle and speed baseline, not the learning implementation.

DeepSeek-V2 (arXiv 2405.04434), structure follows DeepSeek-V3 inference/model.py. K and V of all
heads are generated from one small latent per token, so the cache holds kv_rank + rope_dim numbers
per token instead of 2 * n_heads * head_dim.

    x     (B, T, d_model)         H = n_heads, D = head_dim, R = rope_dim, Dv = v_head_dim, r = kv_rank
    q_c   (B, H, T, D)            q_r (B, H, T, R)       content part / rotary part of the query
    c     (B, S, r)               k_r (B, S, R)          latent (after rms) / rotary key, ONE per token
    y     (B, T, d_model)

    q          = wq(x)  or  wq_b(rms(wq_a(x)))   (the latter if q_rank is set);  q_r = RoPE(q_r)
    [c, k_r]   = wkv_a(x);  c = rms(c);  k_r = RoPE(k_r)
    naive:     [k_c, v] = wkv_b(c) per head
               score_h(t,s) = (q_c.k_c + q_r.k_r) * (D + R) ** -0.5
               y = wo(concat_h  softmax(score) v)
    absorb:    W = wkv_b.weight viewed (H, D + Dv, r); k_c and v are never built
               q_abs = q_c W[:, :D];   score = (q_abs.c + q_r.k_r) * same scale
               z = softmax(score) c;   y = wo(concat_h  z W[:, D:]^T)

rope_dim=0 is NoPE (no k_r branch, rope must be None). Causal mask is on the token index in the
stream. Rows with no visible key output zeros. Both paths are the same function; absorb is the
cheap one for decoding (attention runs in the r-dim latent), naive is the cheap one for training.

impl="sdpa" runs the attention through F.scaled_dot_product_attention (flash kernels on CUDA):
    naive:  q = [q_c, q_r], k = [k_c, k_r broadcast over heads], v   (head dim D + R)
    absorb: it is exactly multi-query attention over the latent (the MQA form Kimi Linear points
            out for NoPE MLA): q = [q_abs, q_r] (B,H,T,r+R), ONE shared k = [c, k_r] (B,1,S,r+R),
            v = c (B,1,S,r), enable_gqa=True; then z W[:, D:]^T.
impl="manual" is the readable einsum version, the ground truth.

Memory: under torch.no_grad(), if n_heads * T * S > max_scores the queries are processed in blocks
of max_scores // (n_heads * S) so the score tensor is never materialised in full. With grad enabled
there is no blocking (autograd would keep every block's scores anyway).

State (plain dict, save it with torch.save): {"c": (B, cap, r), "k_r": (B, cap, R) with RoPE
applied or None, "mask": (B, cap) bool or None, "len": int valid entries, "n_seen": int}. c, k_r,
mask are preallocated buffers (see cache.py); read them with cache_view(buf, len, 1).
A state is consumed by the call it is passed to (buffers are written in place and shared with the
returned state); use cache.clone_state to branch from one prefix.
key_padding_mask (B, T) bool, True = may be attended, covers only the new tokens with a state.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .cache import arange_cached, cache_extend, cache_view
from .rope import apply_rope, default_positions


def _rms(x, weight, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


class MLARef(nn.Module):
    def __init__(self, d_model, n_heads, kv_rank, head_dim=None, rope_dim=0, v_head_dim=None,
                 q_rank=None, causal=True, rope=None, norm_eps=1e-6, impl="sdpa", max_scores=2 ** 26):
        super().__init__()
        assert (rope_dim > 0) == (rope is not None), "rope_dim > 0 needs a rope, rope_dim = 0 (NoPE) forbids one"
        assert rope is None or rope.dim == rope_dim
        assert impl in ("sdpa", "manual")
        self.impl, self.max_scores = impl, max_scores
        self.n_heads, self.kv_rank, self.rope_dim, self.rope = n_heads, kv_rank, rope_dim, rope
        self.head_dim = head_dim or d_model // n_heads
        self.v_head_dim = v_head_dim or self.head_dim
        self.q_rank, self.causal, self.eps = q_rank, causal, norm_eps
        D, R, Dv, H = self.head_dim, rope_dim, self.v_head_dim, n_heads
        self.scale = (D + R) ** -0.5
        if q_rank is None:
            self.wq = nn.Linear(d_model, H * (D + R), bias=False)
        else:
            self.wq_a = nn.Linear(d_model, q_rank, bias=False)
            self.q_norm_weight = nn.Parameter(torch.ones(q_rank))
            self.wq_b = nn.Linear(q_rank, H * (D + R), bias=False)
        self.wkv_a = nn.Linear(d_model, kv_rank + R, bias=False)
        self.kv_norm_weight = nn.Parameter(torch.ones(kv_rank))
        self.wkv_b = nn.Linear(kv_rank, H * (D + Dv), bias=False)
        self.wo = nn.Linear(H * Dv, d_model, bias=False)

    def cache_floats_per_token(self):
        return self.kv_rank + self.rope_dim

    def forward(self, x, positions=None, key_padding_mask=None, state=None, return_state=False,
                absorb=None):
        B, T, _ = x.shape
        H, D, R, Dv = self.n_heads, self.head_dim, self.rope_dim, self.v_head_dim
        if absorb is None:
            absorb = state is not None or return_state or not self.training
        n0 = state["n_seen"] if state is not None else 0

        q = self.wq(x) if self.q_rank is None else self.wq_b(_rms(self.wq_a(x), self.q_norm_weight, self.eps))
        q_c, q_r = q.view(B, T, H, D + R).transpose(1, 2).split([D, R], dim=-1)
        c, k_r = self.wkv_a(x).split([self.kv_rank, R], dim=-1)
        c = _rms(c, self.kv_norm_weight, self.eps)
        if R:
            cos, sin = self.rope(default_positions(positions, T, n0, x.device, self.rope))
            q_r, k_r = apply_rope(q_r, cos, sin), apply_rope(k_r, cos, sin)   # k_r: (B, T, R), no head dim
        new = {"c": c, "k_r": k_r} if R else {"c": c}

        mask, bufs, static = key_padding_mask, None, False
        if state is not None:
            bufs, n_new = cache_extend(state, new, 1, mask)
            if T == 1 and bufs["mask"] is not None:
                # decode with padding: attend over the whole buffer with one fixed-shape validity
                # mask, so shapes only change when the buffer reallocates (MPS builds a graph per shape)
                ok = arange_cached(bufs["c"].shape[1], x.device) < n_new
                c, k_r = bufs["c"], bufs["k_r"] if R else None
                mask, static = bufs["mask"] & ok, True
            else:
                c = cache_view(bufs["c"], n_new, 1)
                k_r = cache_view(bufs["k_r"], n_new, 1) if R else None
                mask = None if bufs["mask"] is None else cache_view(bufs["mask"], n_new, 1)
        S = c.shape[1]

        def vis_fn(i, j):                               # (1|B, 1, Tb, S) bool, True = may attend
            if static:
                return mask[:, None, None, :]
            if mask is None and (not self.causal or T == 1):
                return None                             # a lone query may see every key: build no mask
            q_idx = (S - T) + torch.arange(T, device=x.device)
            vis = None
            if self.causal:
                vis = (torch.arange(S, device=x.device)[None, :] <= q_idx[i:j][:, None])[None, None]
            if mask is not None:
                vis = mask[:, None, None, :] if vis is None else vis & mask[:, None, None, :]
            return vis

        sdpa = self.impl == "sdpa"
        if absorb:
            W = self.wkv_b.weight.view(H, D + Dv, self.kv_rank)
            if sdpa:                                    # MQA: one shared k/v head, the latent
                K = (torch.cat([c, k_r], -1) if R else c)[:, None]
                V = c[:, None]
        else:
            k_c, v = self.wkv_b(c).view(B, S, H, D + Dv).transpose(1, 2).split([D, Dv], dim=-1)
            if sdpa:
                K = torch.cat([k_c, k_r[:, None].expand(-1, H, -1, -1)], -1) if R else k_c
                V = v

        blk = T                                         # grad enabled: never block (autograd keeps all scores)
        if not torch.is_grad_enabled() and H * T * S > self.max_scores:
            blk = max(1, self.max_scores // (H * S))
        fast = sdpa and blk >= T and self.causal and mask is None and S == T

        def block(i, j):
            qc, qr, vis = q_c[:, :, i:j], q_r[:, :, i:j], vis_fn(i, j)
            if absorb:
                qc = torch.einsum("bhtd,hdc->bhtc", qc, W[:, :D])        # q_abs
            if sdpa:
                z = F.scaled_dot_product_attention(
                    torch.cat([qc, qr], -1) if R else qc, K, V if absorb else v,
                    attn_mask=None if fast or vis is None else vis, is_causal=fast,
                    scale=self.scale, enable_gqa=absorb)
                if vis is not None and not fast:        # fully masked rows: zeros, never NaN
                    z = z.masked_fill(~vis.any(-1, keepdim=True), 0)
            else:
                s = torch.einsum("bhtc,bsc->bhts", qc, c) if absorb else torch.einsum("bhtd,bhsd->bhts", qc, k_c)
                if R:
                    s = s + torch.einsum("bhtr,bsr->bhts", qr, k_r)
                s = s * self.scale
                if vis is not None:
                    s = s.masked_fill(~vis, torch.finfo(s.dtype).min)    # finite: no NaN in backward
                w = s.softmax(-1)
                if vis is not None:
                    w = w.masked_fill(~vis.any(-1, keepdim=True), 0)
                z = torch.einsum("bhts,bsc->bhtc", w, c) if absorb else torch.einsum("bhts,bhsv->bhtv", w, v)
            return torch.einsum("bhtc,hvc->bthv", z, W[:, D:]) if absorb else z.transpose(1, 2)

        o = torch.cat([block(i, min(i + blk, T)) for i in range(0, T, blk)], dim=1)
        y = self.wo(o.reshape(B, T, H * Dv))
        if not return_state:
            return y
        if bufs is None:
            bufs, n_new = cache_extend(None, new, 1, key_padding_mask)
        return y, {"c": bufs["c"], "k_r": bufs.get("k_r"), "mask": bufs["mask"], "len": n_new, "n_seen": n0 + T}
