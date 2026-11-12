"""Stage-1 readouts: functions over plain tensors of latents (ticket [5.1]).

Nothing here imports model code. Every function takes latents as tensors and
returns numbers, so it runs on anything that can produce latents: a trained
checkpoint, a random-init model, or synthetic tensors in a test.

Shapes used throughout:
    (N, D)     N latents of width D, already flattened
    (B, T, D)  B sequences of T chunks; pair it with a (B, T) bool mask that is
               True where the chunk is real and False where it is padding

Usage on saved tensors:
    .venv/bin/python eval_test/readouts/metrics.py latents.pt

where latents.pt holds a dict with "pred" and "target" of shape (B, T, D),
aligned so pred[:, t] is the prediction OF target[:, t], and optionally "mask".
"""

import sys

import torch


def flatten_valid(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """(B, T, D) plus a (B, T) mask -> (N, D), keeping only real chunks."""
    if x.dim() == 2:
        if mask is not None:
            raise ValueError("a mask only makes sense for (B, T, D) input")
        return x
    if x.dim() != 3:
        raise ValueError(f"expected (N, D) or (B, T, D), got {tuple(x.shape)}")
    if mask is None:
        return x.reshape(-1, x.shape[-1])
    if mask.shape != x.shape[:2]:
        raise ValueError(f"mask {tuple(mask.shape)} does not match {tuple(x.shape[:2])}")
    return x[mask.bool()]


def retrieval_rank(pred: torch.Tensor, target: torch.Tensor, metric: str = "l2") -> torch.Tensor:
    """Rank of each prediction's own target among all N targets. 1 is best.

    pred and target are (N, D); pred[i] is meant to land on target[i].

    Ties count against the prediction: the rank is the number of targets at
    least as close as the true one. So if the targets have collapsed to one
    point, every rank is N, which is the honest reading of a collapse.
    """
    if pred.shape != target.shape or pred.dim() != 2:
        raise ValueError(f"expected two (N, D) tensors, got {tuple(pred.shape)} and {tuple(target.shape)}")
    pred, target = pred.float(), target.float()
    if metric == "l2":
        dist = torch.cdist(pred, target)  # (N, N)
    elif metric == "cosine":
        p = torch.nn.functional.normalize(pred, dim=-1)
        t = torch.nn.functional.normalize(target, dim=-1)
        dist = 1.0 - p @ t.T
    else:
        raise ValueError(f"unknown metric {metric!r}")
    true_dist = dist.diagonal().unsqueeze(1)  # (N, 1)
    return (dist <= true_dist).sum(dim=1)  # includes the true target itself, so >= 1


def retrieval_summary(ranks: torch.Tensor, ks: tuple[int, ...] = (1, 10)) -> dict[str, float]:
    """Recall@k, mean reciprocal rank and median rank, with the chance level beside them."""
    n = ranks.numel()
    out: dict[str, float] = {"candidates": float(n)}
    for k in ks:
        out[f"recall@{k}"] = (ranks <= k).float().mean().item()
        out[f"chance_recall@{k}"] = min(k, n) / n
    out["mrr"] = (1.0 / ranks.float()).mean().item()
    out["median_rank"] = ranks.float().median().item()
    return out


def effective_rank(z: torch.Tensor) -> float:
    """How many directions the latents actually use (Roy and Vetterli, 2007).

    exp of the entropy of the normalised singular values of the centred (N, D)
    matrix. Equals k when the data fills k directions equally, and 1 when
    everything lies on a line.
    """
    z = flatten_valid(z).double()
    z = z - z.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(z)
    total = s.sum()
    if total <= 0:
        return 0.0  # every latent is the same point
    p = s / total
    p = p[p > 0]
    return torch.exp(-(p * p.log()).sum()).item()


def per_dim_variance(z: torch.Tensor) -> torch.Tensor:
    """Variance of each latent dimension over the (N, D) batch. Returns (D,)."""
    return flatten_valid(z).float().var(dim=0, unbiased=False)


def variance_explained(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per dimension, the share of the target's variance the prediction accounts for.

    1 - mean((target - pred)^2) / var(target), computed separately for each of
    the D dimensions. 1 is a perfect prediction, 0 is no better than predicting
    the mean, and negative is worse than that. Returns (D,).

    This is the curve that shows whether the latent has sorted itself: high at
    the front, falling toward zero in the tail.
    """
    if pred.shape != target.shape or pred.dim() != 2:
        raise ValueError(f"expected two (N, D) tensors, got {tuple(pred.shape)} and {tuple(target.shape)}")
    pred, target = pred.float(), target.float()
    err = (target - pred).pow(2).mean(dim=0)
    var = target.var(dim=0, unbiased=False)
    return 1.0 - err / var.clamp_min(1e-12)


def consecutive_pairs(
    z: torch.Tensor, mask: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Every (chunk t, chunk t+1) pair of real chunks in (B, T, D), as two (N, D) tensors."""
    if z.dim() != 3:
        raise ValueError(f"expected (B, T, D), got {tuple(z.shape)}")
    prev, nxt = z[:, :-1], z[:, 1:]
    if mask is None:
        return prev.reshape(-1, z.shape[-1]), nxt.reshape(-1, z.shape[-1])
    both = mask[:, :-1].bool() & mask[:, 1:].bool()
    return prev[both], nxt[both]


def copy_previous_baseline(z: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, float]:
    """The floor every predictor must beat: predict that the next chunk equals this one."""
    prev, nxt = consecutive_pairs(z, mask)
    ranks = retrieval_rank(prev, nxt)
    out = {"mse": (nxt.float() - prev.float()).pow(2).mean().item()}
    out.update(retrieval_summary(ranks))
    out["mean_variance_explained"] = variance_explained(prev, nxt).mean().item()
    return out


def subsample(
    pred: torch.Tensor, target: torch.Tensor, max_candidates: int, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep at most max_candidates aligned rows, so the N x N distance table stays small."""
    n = pred.shape[0]
    if n <= max_candidates:
        return pred, target
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)[:max_candidates]
    return pred[idx], target[idx]


def readout_table(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    max_candidates: int = 2000,
) -> dict[str, float]:
    """All stage-1 numbers for one set of predictions.

    pred and target are (B, T, D), aligned so pred[:, t] predicts target[:, t].
    Retrieval is scored among at most max_candidates chunks.
    """
    p, t = flatten_valid(pred, mask), flatten_valid(target, mask)
    ve = variance_explained(p, t)
    ps, ts = subsample(p, t, max_candidates)
    out = {"mse": (t.float() - p.float()).pow(2).mean().item()}
    out.update(retrieval_summary(retrieval_rank(ps, ts)))
    out["mean_variance_explained"] = ve.mean().item()
    out["effective_rank"] = effective_rank(t)
    out["latent_dim"] = float(t.shape[-1])
    out["min_dim_variance"] = per_dim_variance(t).min().item()
    return out


def format_table(rows: dict[str, dict[str, float]]) -> str:
    """rows maps a row name (e.g. "model", "copy previous") to its readouts."""
    keys: list[str] = []
    for r in rows.values():
        keys += [k for k in r if k not in keys]
    width = max(len(k) for k in keys)
    names = list(rows)
    lines = [" " * width + "".join(f"{n:>18}" for n in names)]
    for k in keys:
        cells = "".join(f"{rows[n][k]:>18.4f}" if k in rows[n] else f"{'-':>18}" for n in names)
        lines.append(f"{k:<{width}}{cells}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    saved = torch.load(sys.argv[1], map_location="cpu")
    mask = saved.get("mask")
    print(
        format_table(
            {
                "model": readout_table(saved["pred"], saved["target"], mask),
                "copy previous": copy_previous_baseline(saved["target"], mask),
            }
        )
    )
