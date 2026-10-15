# SA-JEPA v0 — the "works" training run (2026-10-15).
# Trains the same architecture (6 delta + 2 GQA, JEPA horizons, SIGREG) with:
#   * alpha = 1.0  (gate-4 proved 0.01 collapses on real data)
#   * isotropy = 1.0  (holds effective rank; see BUILD_LOG.md §7)
#   * recon_w (small auxiliary mel reconstruction so the latents stay decodable)
# Saves sharded fp16 checkpoints + an utterance-level train/held-out split for probing.
# Usage: uv run python v2_unfrozen/one-shot/train_v0.py --steps 4000 --bs 8 --T 250
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import SAJePAV0  # noqa: E402
from board import board_row  # noqa: E402
from ckpt import save_ckpt  # noqa: E402


def load_windows():
    return list(csv.DictReader(open(HERE / "mel_cache" / "windows.csv")))


def split_utts(seed=0, held_frac=0.2):
    """Utterance-level split so probe utterances are never trained on."""
    rows = load_windows()
    utts = sorted({r["utt"] for r in rows})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(utts))
    n_held = int(len(utts) * held_frac)
    held = {utts[i] for i in perm[:n_held]}
    train = [r for r in rows if r["utt"] not in held]
    probe = [r for r in rows if r["utt"] in held]
    return train, probe


def sample_batch(rows, bs, T, rng, device):
    idx = rng.integers(0, len(rows), size=bs)
    batch = torch.zeros(bs, T, 80)
    meta = []
    for bi, i in enumerate(idx):
        r = rows[i]
        mel = torch.load(HERE / "mel_cache" / f"{r['utt']}.pt").float()
        nf = mel.shape[0]
        t0 = int(rng.integers(0, max(1, nf - T))) if T < nf else 0
        batch[bi] = mel[t0:t0 + T]
        meta.append((r["utt"], t0))
    return batch.to(device), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--T", type=int, default=250)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--iso", type=float, default=1.0)
    ap.add_argument("--recon-w", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(HERE / "runs" / "v0"))
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    rng = np.random.default_rng(a.seed)

    train_rows, probe_rows = split_utts(seed=a.seed)
    print(f"data: {len(train_rows)} train windows / {len(probe_rows)} held-out windows "
          f"({len({r['utt'] for r in probe_rows})} held-out utts)", flush=True)

    model = SAJePAV0().to(a.device)
    model.sigreg_isotropy = a.iso
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    floors = {1: (293.9, 81.7), 10: (292.4, 283.6), 50: (260.3, 274.2), 150: (280.4, 292.9)}

    # fixed held-out eval batch (never trained on): used for every printed row
    xev, mev = sample_batch(probe_rows, a.bs, a.T, np.random.default_rng(1234), a.device)
    yev = [m[0].split("-")[0] for m in mev]

    t0 = time.time()
    for it in range(a.steps + 1):
        if it % a.eval_every == 0 or it == a.steps:
            model.eval()
            with torch.no_grad():
                e_ev, preds_ev = model(xev)
                rec_ev = model.decoder(e_ev)
                rec_mse = torch.nn.functional.mse_loss(rec_ev, xev).item()
                pooled = e_ev.mean(1)
                # board row on POOLED per-window latents for eff-rank, plus per-frame var
                from board import effective_rank, per_dim_var
                zf = e_ev.reshape(-1, e_ev.shape[-1])
                vmean, vmin, vmax = per_dim_var(zf)
                er = effective_rank(zf)
                acc, sd = _quick_probe(pooled, yev) if it >= a.eval_every else (0.0, 0.0)
                print(f"step {it:>5} | held-out recon MSE {rec_mse:7.3f} dB² | eff-rank {er:5.1f} | "
                      f"var[{vmin:.2f},{vmean:.2f},{vmax:.2f}] | spk(win) {acc:.3f}±{sd:.3f} | "
                      f"{time.time()-t0:.0f}s", flush=True)
            model.train()
        if it == a.steps:
            break
        x, _ = sample_batch(train_rows, a.bs, a.T, rng, a.device)
        loss, parts, e = model.loss(x, alpha=a.alpha, recon_w=a.recon_w)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\n{a.steps} steps in {(time.time()-t0)/60:.1f} min")
    save_ckpt(model, a.out, extra={"steps": a.steps, "alpha": a.alpha, "iso": a.iso,
                                   "recon_w": a.recon_w, "T": a.T, "bs": a.bs,
                                   "parallel_scan": True})
    # persist the split so evidence/inference use identical held-out utterances
    torch.save({"held_utts": sorted({r["utt"] for r in probe_rows})},
               Path(str(a.out) + ".split.pt"))
    print(f"saved ckpt prefix {a.out} (+ .split.pt)")


def _quick_probe(Z, y, seeds=2, test_size=0.3):
    """Small logistic-regression speaker probe (few classes at bs=8 -> low support;
    the full probe lives in evidence_v0.py on many more windows)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    X = Z.detach().cpu().numpy()
    yy = np.asarray(y)
    accs = []
    for s in range(seeds):
        if len(set(yy)) < 2:
            return 0.0, 0.0
        try:
            Xtr, Xev, ytr, yev = train_test_split(X, yy, test_size=test_size, random_state=s, stratify=yy)
        except ValueError:
            return 0.0, 0.0
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        clf = LogisticRegression(max_iter=1000)
        clf.fit((Xtr - mu) / sd, ytr)
        accs.append(clf.score((Xev - mu) / sd, yev))
    return float(np.mean(accs)), float(np.std(accs))


if __name__ == "__main__":
    main()
