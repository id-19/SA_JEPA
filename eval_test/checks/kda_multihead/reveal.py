"""Runs the predictions in QUIZ.md. Write your answers first, then run this.

    uv run python eval_test/checks/kda_multihead/reveal.py        # all
    uv run python eval_test/checks/kda_multihead/reveal.py 4      # one question

Everything is small and on CPU.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from layers.SSMs.reference import KimiDeltaAttentionRef
from layers.SSMs.reference.gated_delta_rule import recurrent_gated_delta_rule
from layers.SSMs.reference.kda_ops import recurrent_kda

DT = torch.float64


def q1():
    print("Q1. Floats in the delta-rule state S per batch element, d_model = 256")
    for H in (1, 2, 4, 8):
        layer = KimiDeltaAttentionRef(d_model=256, n_heads=H)
        _, state = layer(torch.randn(1, 16, 256), return_state=True)
        S = state["S"]
        print(f"  H={H}: S is {tuple(S.shape)}, {S[0].numel()} floats")


def q2():
    print("Q2a. Op level: k of head 2 changed at token 10 (H=4, T=20). Outputs that change:")
    gen = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=gen, dtype=DT)
    q, k, v = F.normalize(r(1, 4, 20, 8), dim=-1), F.normalize(r(1, 4, 20, 8), dim=-1), r(1, 4, 20, 8)
    g, beta = -F.softplus(r(1, 4, 20, 8)), torch.rand(1, 4, 20, generator=gen, dtype=DT)
    o, _ = recurrent_kda(q, k, v, g, beta)
    k2 = k.clone()
    k2[:, 2, 10] = F.normalize(r(8), dim=-1)
    o2, _ = recurrent_kda(q, k2, v, g, beta)
    changed = (o - o2).abs().amax(-1)[0] > 1e-12                  # (H, T)
    for h in range(4):
        ts = changed[h].nonzero().flatten().tolist()
        print(f"  head {h}: " + (f"tokens {ts[0]}..{ts[-1]} ({len(ts)} of 20)" if ts else "nothing"))

    print("Q2b. Layer level: x changed at token 10 (H=4, T=20). Tokens of y that change:")
    torch.manual_seed(0)
    layer = KimiDeltaAttentionRef(d_model=32, n_heads=4).to(DT)
    x = torch.randn(1, 20, 32, dtype=DT)
    x2 = x.clone()
    x2[:, 10] += 1.0
    ts = ((layer(x) - layer(x2)).abs().amax(-1)[0] > 1e-12).nonzero().flatten().tolist()
    print(f"  tokens {ts[0]}..{ts[-1]} ({len(ts)} of 20)")


def q3():
    print("Q3. KDA with the same decay on every key channel, against GDN with that scalar decay")
    gen = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=gen)
    q, k, v = F.normalize(r(2, 3, 40, 16), dim=-1), F.normalize(r(2, 3, 40, 16), dim=-1), r(2, 3, 40, 24)
    g_scalar, beta = -F.softplus(r(2, 3, 40)), torch.rand(2, 3, 40, generator=gen)
    o_kda, _ = recurrent_kda(q, k, v, g_scalar[..., None].expand(2, 3, 40, 16), beta)
    o_gdn, _ = recurrent_gated_delta_rule(q, k, v, g_scalar, beta)
    print(f"  max |KDA - GDN| = {(o_kda - o_gdn).abs().max().item():.2e}  (outputs are of size ~{o_gdn.abs().mean().item():.2f})")


def _recall_error(N, H, orthogonal, seed, D=64):
    """Write N (key, value) pairs with no decay and full write strength, then read every key back."""
    gen = torch.Generator().manual_seed(seed)
    dk = D // H
    v = torch.randn(N, D, generator=gen, dtype=DT)
    if orthogonal:
        heads = []
        for _ in range(H):
            basis, _ = torch.linalg.qr(torch.randn(max(N, dk), dk, generator=gen, dtype=DT))
            heads.append(F.normalize(basis[:N], dim=-1))          # orthonormal while N <= dk
        k = torch.stack(heads, 0)                                 # (H, N, dk)
    else:
        k = F.normalize(torch.randn(N, D, generator=gen, dtype=DT).view(N, H, dk), dim=-1).transpose(0, 1)
    k, vh = k[None], v.view(N, H, dk).transpose(0, 1)[None]       # head h owns slice h of key and value
    _, S = recurrent_kda(k, k, vh, torch.zeros(1, H, N, dk, dtype=DT), torch.ones(1, H, N, dtype=DT), scale=1.0)
    read = torch.einsum("bhkv,bhnk->bhnv", S, k)[0].transpose(0, 1).reshape(N, D)
    return ((read - v).norm(dim=-1) / v.norm(dim=-1)).mean().item()


def q4():
    print("Q4. Mean relative recall error after N writes. Key and value are 64 wide in total,")
    print("    split across H heads. 0 is perfect recall, 1 is as bad as returning nothing.")
    Ns = (4, 8, 16, 32, 64, 128)
    for orthogonal in (True, False):
        print("  keys as orthogonal as the head allows:" if orthogonal else "  random keys:")
        print("      N " + "".join(f"{n:>7d}" for n in Ns))
        for H in (1, 2, 4, 8):
            row = [sum(_recall_error(N, H, orthogonal, s) for s in range(5)) / 5 for N in Ns]
            print(f"    H={H} " + "".join(f"{e:7.2f}" for e in row))


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3", "4"]
    for name in which:
        {"1": q1, "2": q2, "3": q3, "4": q4}[name]()
        print()
