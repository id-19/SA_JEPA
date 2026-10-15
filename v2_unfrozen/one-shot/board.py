# SA-JEPA v2 one-shot — board harness: non-collapse metrics + linear speaker probe + row printer.
# Your trainer imports this. Pure evaluation — receives tensors, makes no assumptions about the model.
import numpy as np
import torch


def effective_rank(Z: torch.Tensor) -> float:
    """Participation-ratio effective rank of batch covariance. Z: (N, D).
    Collapsed space -> ~0; healthy N(0,I) space -> ~D."""
    Z = Z - Z.mean(0, keepdim=True)
    if Z.shape[0] < 2:
        return 0.0
    cov = (Z.T @ Z) / (Z.shape[0] - 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1 = ev.sum()
    return float((s1 ** 2 / (ev ** 2).sum()).item()) if s1 > 0 else 0.0


def per_dim_var(Z: torch.Tensor):
    """Per-dim variance stats vs the SIGREG target (var 1.0)."""
    v = Z.var(0, unbiased=False)
    return float(v.mean()), float(v.min()), float(v.max())


def speaker_probe(Z: torch.Tensor, y, n_seeds: int = 3, test_size: float = 0.3):
    """Linear probe: pooled embeddings -> speaker id. Returns (mean_acc, std).
    Z (N, D) numpy/tensor; y: iterable of speaker labels (same length)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    X = Z.detach().cpu().numpy() if torch.is_tensor(Z) else np.asarray(Z)
    y = np.asarray(y)
    accs = []
    for seed in range(n_seeds):
        try:
            Xtr, Xev, ytr, yev = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
        except ValueError as e:
            # Do NOT return 0.0 here. A silent 0.0 is indistinguishable from a measured zero
            # and was printed as "spk 0.000" on every row of a real 2000-step run whose probe
            # pool was 8 windows over ~8 speakers -- a metric that measured nothing while
            # looking like a result. Fail loudly instead; the caller must supply a pool with
            # >=2 windows per speaker.
            raise RuntimeError(
                f"speaker_probe cannot stratify: {len(y)} windows over {len(set(y.tolist()))} "
                f"speakers (needs >=2 per speaker). Build a proper probe pool."
            ) from e
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        clf = LogisticRegression(max_iter=2000)
        clf.fit((Xtr - mu) / sd, ytr)
        accs.append(clf.score((Xev - mu) / sd, yev))
    return float(np.mean(accs)), float(np.std(accs))


def board_row(step: int, losses_per_d: dict, sigreg: float, Z_pool: torch.Tensor, y=None, floors: dict | None = None) -> str:
    """One printable board row. Z_pool: (N, D) pooled latents. floors: {d: (blind, linear)}."""
    er = effective_rank(Z_pool)
    vmean, vmin, vmax = per_dim_var(Z_pool)
    lp = " ".join(f"d{k}:{v:.4f}" for k, v in sorted(losses_per_d.items()))
    line = f"step {step:>6} | {lp} | sigreg {sigreg:.4f} | eff-rank {er:6.1f} | var[{vmin:.2f},{vmean:.2f},{vmax:.2f}]"
    if y is not None:
        acc, sd = speaker_probe(Z_pool, y)
        line += f" | spk {acc:.3f}±{sd:.3f}"
    if floors:
        fb = " ".join(f"d{k}:{v[0]:.3f}/{v[1]:.3f}" for k, v in sorted(floors.items()))
        line += f" | floors(b/l) {fb}"
    return line


def demo_self_test():
    """Harness self-test on synthetic latents: collapsed vs healthy vs N(0,1)."""
    rng = np.random.default_rng(0)
    N, D, K = 400, 256, 8
    # healthy: 8 clusters + within-cluster noise of comparable scale -> full-rank covariance
    y = rng.integers(0, K, N)
    centers = rng.normal(0, 2, (K, D))
    Z_good = torch.tensor(centers[y] + rng.normal(0, 2, (N, D)), dtype=torch.float32)
    # collapsed: rank-2
    base = rng.normal(0, 1, (N, 2))
    Z_bad = torch.tensor(base @ rng.normal(0, 1, (2, D)), dtype=torch.float32)
    for name, Z in [("healthy", Z_good), ("COLLAPSED", Z_bad)]:
        row = board_row(0, {1: 0.5, 10: 0.9}, 0.1, Z, y=y)
        print(f"[{name}] {row}")
    assert effective_rank(Z_good) > 20
    assert effective_rank(Z_bad) < 2.2
    print("self-test OK: eff-rank separates collapsed from healthy")


if __name__ == "__main__":
    demo_self_test()