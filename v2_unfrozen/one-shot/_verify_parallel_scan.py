# Equivalence check: log-depth parallel scan vs sequential chunk loop vs per-step reference.
# Also reports sequential depth and number of Python-level loop trips for each implementation.
import torch, sys
sys.path.insert(0, '/Users/id19/Programming/ml_ai_and_data_science/SA-JEPA/v2_unfrozen/one-shot')
from model import DeltaCore, D_MODEL, DH, NH_DELTA

torch.manual_seed(0)
core = DeltaCore().eval()


def per_step_ref(q, k, v, beta):
    """Strictly sequential reference, one frame at a time (the definition)."""
    BH, T, DK = k.shape
    S = torch.zeros(BH, DK, DK)
    outs = []
    for t in range(T):
        Sk = torch.einsum('bik,bk->bi', S, k[:, t])
        S = S - beta[:, t].view(-1, 1, 1) * torch.einsum('bi,bj->bij', Sk - v[:, t], k[:, t])
        outs.append(torch.einsum('bik,bk->bi', S, q[:, t]))
    return torch.stack(outs, dim=1)


print(f"{'T':>5} | {'parallel vs loop':>17} | {'parallel vs ref':>16} | {'loop vs ref':>12} | depth loop/par")
for T in [37, 64, 65, 250, 500, 1024]:
    with torch.no_grad():
        x = torch.randn(2, T, D_MODEL)
        q, k, v = core._qkv(x)
        beta = torch.sigmoid(core.wb(x)).view(2, T, NH_DELTA)
        B = 2
        qh = q.permute(0, 2, 1, 3).reshape(B * NH_DELTA, T, DH)
        kh = k.permute(0, 2, 1, 3).reshape(B * NH_DELTA, T, DH)
        vh = v.permute(0, 2, 1, 3).reshape(B * NH_DELTA, T, DH)
        bh = beta.permute(0, 2, 1).reshape(B * NH_DELTA, T)
        y_par = core._chunk_scan_parallel(qh, kh, vh, bh)
        y_loop = core._chunk_scan(qh, kh, vh, bh)
        sub = slice(0, 8)  # 8 head-batches of the 2-sample batch is enough for the ref
        y_ref = per_step_ref(qh[sub], kh[sub], vh[sub], bh[sub])
        d_pl = (y_par[sub] - y_loop[sub]).abs().max().item()
        d_pr = (y_par[sub] - y_ref).abs().max().item()
        d_lr = (y_loop[sub] - y_ref).abs().max().item()
    nc = (T + 63) // 64
    import math
    print(f"{T:>5} | {d_pl:>17.3e} | {d_pr:>16.3e} | {d_lr:>12.3e} | {nc} / {math.ceil(math.log2(nc)) if nc > 1 else 0}")
