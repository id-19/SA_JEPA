"""Preallocated KV-cache buffers with amortised O(1) append. AI-written reference (13 Nov 2026).

A cache entry is a buffer with spare capacity along `dim` plus a valid length n; entries
[0, n) are real. Appending one token writes in place instead of copying the whole cache
(torch.cat is O(S) per token, O(S^2) over a decode). Readers go through cache_view.

Because appends write in place, a state is CONSUMED by the call it is passed to: the returned
state shares its storage. Use clone_state to branch from one prefix.
"""

import functools

import torch


@functools.lru_cache(maxsize=8)
def arange_cached(n, device):
    """arange(n), built once per (n, device): decode masks reuse it, so shapes stay fixed on MPS."""
    return torch.arange(n, device=device)


def cache_view(buf, n, dim, last=None):
    """Valid part as a view (no copy): [n - last : n] (clamped at 0) if last is given, else [0 : n]."""
    start = 0 if last is None else max(n - last, 0)
    return buf.narrow(dim, start, n - start)


def cache_append(buf, n, new, dim, keep=None, functional=None):
    """Append `new` along `dim` to a buffer whose first n entries are valid. Returns (buf, n_new).

    keep: only the last `keep` valid entries will ever be read (sliding window). When the buffer
          is full they are compacted to the front, so capacity stays bounded (~2*keep + L).
    functional: None = auto. If grad is involved (new or buf requires grad) the in-place write
          would corrupt the graph, so fall back to torch.cat of the readable part and `new`
          (capacity == length). Pass the same bool for every buffer of one state so their
          lengths stay in step.
    Unwritten capacity is always zeros (or old data), never uninitialised: decode attends over the
    whole buffer with a validity mask, and a masked NaN/inf key or value would still poison the output.
    First allocation: capacity L + max(L // 4, 64), or 2*keep + L with keep (not the 2*L that
    would double the memory of a long prefill); later growth is max(2 * capacity, n + L).
    """
    L = new.shape[dim]
    if functional is None:
        functional = new.requires_grad or (buf is not None and buf.requires_grad)
    if functional:
        old = [] if buf is None else [cache_view(buf, n, dim, keep)]
        out = torch.cat(old + [new], dim)
        return out, out.shape[dim]
    if buf is None:
        cap = 2 * keep + L if keep is not None else L + max(L // 4, 64)
        shape = list(new.shape)
        shape[dim] = cap
        buf, n = new.new_zeros(shape), 0
    cap = buf.shape[dim]
    if n + L > cap:
        if keep is not None and n > keep:
            # overlapping source and destination: clone the slice before moving it
            buf.narrow(dim, 0, keep).copy_(buf.narrow(dim, n - keep, keep).clone())
            n = keep
        if n + L > cap:
            shape = list(buf.shape)
            shape[dim] = max(2 * cap, n + L)
            grown = buf.new_zeros(shape)
            grown.narrow(dim, 0, n).copy_(buf.narrow(dim, 0, n))
            buf = grown
    buf.narrow(dim, n, L).copy_(new)
    return buf, n + L


def cache_extend(state, new, dim, mask, keep=None):
    """Append to every cached tensor of a layer state at once; shared by GQARef and MLARef.

    state: previous state dict (keys of `new`, "mask", "len") or None. new: {name: tensor}.
    mask: (B, L) bool for the new tokens or None. Returns ({name: buf, "mask": buf or None}, n_new).
    A mask that appears late is created as True on the n valid entries with the same capacity as the other buffers;
    one that was there but is missing now is filled with True. The unwritten tail of a mask buffer is False. All buffers share capacity and
    length, so compaction and growth decisions are identical.
    """
    first = next(iter(new.values()))
    B, L = first.shape[0], first.shape[dim]
    n = state["len"] if state is not None else 0
    old = (lambda k: state[k]) if state is not None else (lambda k: None)
    functional = any(t.requires_grad for t in new.values()) or \
        any(old(k) is not None and old(k).requires_grad for k in new)
    cap = None if state is None else old(next(iter(new))).shape[dim]
    out, mbuf = {}, old("mask")
    if mbuf is not None or mask is not None:
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=first.device)
        if mbuf is None and cap is not None:
            mbuf = torch.zeros(B, cap, dtype=torch.bool, device=first.device)
            mbuf[:, :n] = True
        mbuf, _ = cache_append(mbuf, n, mask, 1, keep, functional)
    out["mask"] = mbuf
    for name, t in new.items():
        out[name], n_new = cache_append(old(name), n, t, dim, keep, functional)
    return out, n_new


def clone_state(state):
    """Deep copy of a state dict (tensors cloned). Clone before passing a state to branch from it."""
    return {k: v.clone() if torch.is_tensor(v) else v for k, v in state.items()}
