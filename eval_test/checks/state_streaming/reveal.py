"""Runs the predictions in QUIZ.md. Write your answers first, then run this.

    uv run python eval_test/checks/state_streaming/reveal.py        # all
    uv run python eval_test/checks/state_streaming/reveal.py 4      # one question

Everything is small and on CPU, except Q5c, which uses MPS if there is one.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import layers.SSMs.reference.gated_deltanet as gdn_module
from layers.Attention_layers.reference import GQARef
from layers.SSMs.reference import GatedDeltaNetRef, KimiDeltaAttentionRef
from layers.SSMs.reference.gated_deltanet import CausalDepthwiseConv


def stream(layer, x, pieces, state=None):
    """Feed x in pieces, carrying the state. Returns (y, final state)."""
    ys, t = [], 0
    for n in pieces:
        y, state = layer(x[:, t:t + n], state, return_state=True)
        ys.append(y)
        t += n
    assert t == x.shape[1]
    return torch.cat(ys, dim=1), state


def small_gdn(seed=0, **kw):
    torch.manual_seed(seed)
    return GatedDeltaNetRef(d_model=32, n_heads=2, chunk_size=16, **kw)


def q1():
    print("Q1. State size, GatedDeltaNetRef(d_model=512, n_heads=4), conv_size 4")
    layer = GatedDeltaNetRef(d_model=512, n_heads=4)
    for T in (100, 10_000):
        with torch.no_grad():
            _, st = layer(torch.randn(1, T, 512), return_state=True)
        sizes = {k: v[0].numel() for k, v in st.items()}
        print(f"  after T={T:>6}: " + ", ".join(f"{k} {n}" for k, n in sizes.items())
              + f", total {sum(sizes.values())}")
    total = layer.state_floats()
    S = 4 * 128 * 128
    print(f"  S is {S} floats, the conv tails {total - S}: tails are {100 * (total - S) / total:.1f}% of the state")
    gqa = GQARef(512, 8, n_kv_heads=2)
    per_tok = gqa.cache_floats_per_token()
    print(f"  GQARef(512, 8 heads, 2 kv heads) caches {per_tok} floats per token;"
          f" its cache passes the GDN state after {total / per_tok:.0f} tokens")


def q2():
    print("Q2a. Conv tail after feeding pieces of 5 and then 1 token (input at position t is the number t)")
    conv = CausalDepthwiseConv(1, kernel_size=4)
    x = torch.arange(6.0).view(1, 6, 1)
    _, tail = conv(x[:, :5])
    _, tail = conv(x[:, 5:6], tail)
    print(f"  tail holds the inputs of positions {tail.view(-1).tolist()}")

    print("Q2b. Split at 30 of 60, hand over only S (conv tails dropped). Output tokens that differ:")
    layer = small_gdn()
    x = torch.randn(1, 60, 32)
    with torch.no_grad():
        y_full = layer(x)
        y1, st = layer(x[:, :30], return_state=True)
        y2 = layer(x[:, 30:], {"S": st["S"]})
    diff = (torch.cat([y1, y2], dim=1) - y_full).abs().amax(-1)[0]
    bad = (diff > 1e-5).nonzero().view(-1).tolist()
    print(f"  tokens {bad[0]} to {bad[-1]} ({len(bad)} of 30)" if bad else "  none")
    for t in (30, 31, 32, 33, 34, 40, 59):
        print(f"    token {t}: max |diff| {diff[t]:.2e}")


def q3():
    print("Q3. One call on T=1024 against pieces, float32. max |y - y_one_call| and max |S - S_one_call|")
    for name, cls in (("GDN", GatedDeltaNetRef), ("KDA", KimiDeltaAttentionRef)):
        torch.manual_seed(0)
        kw = dict(gram="naive") if cls is KimiDeltaAttentionRef else {}
        layer = cls(d_model=64, n_heads=2, **kw)                     # chunk_size 64, mode auto
        x = torch.randn(1, 1024, 64)
        with torch.no_grad():
            y_full, st_full = layer(x, return_state=True)
            for label, pieces in (("512 + 512", (512, 512)), ("1000 + 24", (1000, 24)),
                                  ("17 + 1 + 1 + 1005", (17, 1, 1, 1005)), ("1024 x 1 token", (1,) * 1024)):
                y, st = stream(layer, x, pieces)
                dy = (y - y_full).abs().max().item()
                dS = (st["S"] - st_full["S"]).abs().max().item()
                print(f"  {name} {label:>18}: y {dy:.1e}   S {dS:.1e}   exactly equal: {dy == 0 and dS == 0}")


def _grad_case(layer, x, detach):
    """Two calls, split at 30, loss on the second piece only. Returns |dL/dx| per token."""
    x = x.clone().requires_grad_(True)
    _, st = layer(x[:, :30], return_state=True)
    st = {k: (v.detach() if k in detach else v) for k, v in st.items()}
    y2 = layer(x[:, 30:], st)
    y2.pow(2).sum().backward()
    return x.grad.abs().amax(-1)[0]


def q4():
    layer = small_gdn()
    torch.manual_seed(1)
    x = torch.randn(1, 60, 32)

    print("Q4a. Two calls (split at 30) with the state passed as is, loss on both pieces, against one call")
    xa = x.clone().requires_grad_(True)
    layer.zero_grad()
    layer(xa).pow(2).sum().backward()
    g_x, g_p = xa.grad.clone(), {n: p.grad.clone() for n, p in layer.named_parameters()}
    xb = x.clone().requires_grad_(True)
    layer.zero_grad()
    y, _ = stream(layer, xb, (30, 30))
    y.pow(2).sum().backward()
    worst = max((p.grad - g_p[n]).abs().max().item() for n, p in layer.named_parameters())
    print(f"  max |dL/dx difference| {(xb.grad - g_x).abs().max():.1e}, worst parameter gradient difference {worst:.1e}")

    for label, detach in (("b. S detached, conv tails kept", {"S"}),
                          ("c. S and all conv tails detached", {"S", "conv_q", "conv_k", "conv_v"})):
        g = _grad_case(layer, x, detach)
        first = g[:30]
        nz = (first > 0).nonzero().view(-1).tolist()
        print(f"Q4{label}. Loss on piece 2 only. Tokens of piece 1 (0-29) with nonzero gradient: "
              + (f"{nz}" if nz else "none"))


def q5():
    print("Q5a. Which op runs for a piece of T tokens, mode 'auto', on CPU")
    layer = small_gdn()
    calls = []
    real = {n: getattr(gdn_module, n) for n in ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")}
    for n, f in real.items():
        setattr(gdn_module, n, lambda *a, _n=n, _f=f, **k: (calls.append(_n), _f(*a, **k))[1])
    try:
        with torch.no_grad():
            for T in (1, 8, 9, 64):
                calls.clear()
                layer(torch.randn(1, T, 32))
                print(f"  T={T:>2}: {calls[0]}")
    finally:
        for n, f in real.items():
            setattr(gdn_module, n, f)

    print("Q5b. mode='chunk' forced, fed one token at a time for 60 tokens, against one call")
    layer = small_gdn(mode="chunk")
    x = torch.randn(1, 60, 32)
    with torch.no_grad():
        y_full = layer(x)
        y, _ = stream(layer, x, (1,) * 60)
    print(f"  max |diff| {(y - y_full).abs().max():.1e}")

    print("Q5c. On MPS, a 1-token piece under torch.no_grad: which code computes the step?")
    from layers.SSMs.reference import metal_kernels
    if not (torch.backends.mps.is_available() and metal_kernels.metal_available()):
        print("  no MPS / Metal here; answer from reading gated_delta_rule.py")
        return
    real_step = metal_kernels.kda_step_metal
    used = []
    metal_kernels.kda_step_metal = lambda *a, **k: (used.append(1), real_step(*a, **k))[1]
    try:
        layer = small_gdn().to("mps")
        with torch.no_grad():
            layer(torch.randn(1, 1, 32, device="mps"))
        print(f"  metal_kernels.kda_step_metal called: {bool(used)}")
        used.clear()
        layer(torch.randn(1, 1, 32, device="mps", requires_grad=True))
        print(f"  same call with gradients on, kda_step_metal called: {bool(used)}")
    finally:
        metal_kernels.kda_step_metal = real_step


if __name__ == "__main__":
    qs = {"1": q1, "2": q2, "3": q3, "4": q4, "5": q5}
    for k in (sys.argv[1:] or qs):
        qs[k]()
        print()
