"""Sine-wave next-sample prediction task for the linear-RNN arm eval.

Adapted from learning_testing/train_toy.py. The factory returns the raw
series; run_eval.py does the next-step shift (x[:, :-1] -> x[:, 1:]).
"""

import torch

NAME = "sine"
SEED = 0
N_WAVES = 256
TRAIN_FRAC = 0.8  # 256 waves -> 205 train / 51 val


def make_data(seed=SEED):
    # time-steps and the frequency grid, straight from train_toy.py
    # t in [0, 4); f = 0..5 cycles per step (25 bins)
    t = torch.arange(100).float() / 25
    f = torch.arange(25).float() / 5

    g = torch.Generator().manual_seed(seed)
    waves = []
    for _ in range(N_WAVES):
        # two DISTINCT random frequency indices and two random phases
        idx = torch.randperm(25, generator=g)[:2]
        phi = 2 * 3.141592653589793 * torch.rand(2, generator=g)
        wave = (
            torch.sin(2 * 3.141592653589793 * f[idx[0]] * t + phi[0])
            + torch.sin(2 * 3.141592653589793 * f[idx[1]] * t + phi[1])
        )
        waves.append(wave)  # (100,) time series

    x = torch.stack(waves).unsqueeze(-1)  # (N, 100, 1) = (B, L=100, d_in=1)

    # split by WHOLE wave: seed-shuffled first, then 80/20
    perm = torch.randperm(N_WAVES, generator=g)
    x = x[perm]
    n_train = int(N_WAVES * TRAIN_FRAC)  # 205

    return {
        "train": x[:n_train],
        "val": x[n_train:],
        "d_in": 1,
        "d_out": 1,
        "loss": "mse",
    }


if __name__ == "__main__":
    d = make_data()
    print(
        f"{NAME}: train {tuple(d['train'].shape)} val {tuple(d['val'].shape)} "
        f"d_in={d['d_in']} d_out={d['d_out']} loss={d['loss']}"
    )