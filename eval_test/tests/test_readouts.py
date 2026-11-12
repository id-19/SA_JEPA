"""Tests for the stage-1 readouts, on synthetic tensors with known answers."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval_test.readouts import metrics as m


def test_retrieval_rank_perfect_prediction():
    target = torch.randn(50, 8, generator=torch.Generator().manual_seed(0))
    assert (m.retrieval_rank(target.clone(), target) == 1).all()


def test_retrieval_rank_known_ranks_on_a_line():
    # targets sit at 0, 1, 2, 3 on a line; ranks worked out by hand
    target = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
    pred = torch.tensor([[0.1], [2.9], [2.1], [0.9]])
    assert m.retrieval_rank(pred, target).tolist() == [1, 3, 1, 4]


def test_retrieval_rank_collapsed_targets_score_worst():
    target = torch.ones(10, 4)
    pred = torch.randn(10, 4, generator=torch.Generator().manual_seed(0))
    assert (m.retrieval_rank(pred, target) == 10).all()


def test_retrieval_rank_cosine_ignores_length():
    target = torch.eye(4)
    assert (m.retrieval_rank(target * 7.0, target, metric="cosine") == 1).all()


def test_retrieval_summary():
    s = m.retrieval_summary(torch.tensor([1, 1, 2, 20]), ks=(1, 10))
    assert s["recall@1"] == pytest.approx(0.5)
    assert s["recall@10"] == pytest.approx(0.75)
    assert s["chance_recall@1"] == pytest.approx(0.25)
    assert s["mrr"] == pytest.approx((1 + 1 + 0.5 + 0.05) / 4)


def test_effective_rank_equal_directions_is_exactly_k():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 16, generator=g, dtype=torch.double)
    x = x - x.mean(dim=0, keepdim=True)
    q, _ = torch.linalg.qr(x)  # 16 orthonormal, zero-mean columns
    for k in (1, 5, 16):
        assert m.effective_rank(q[:, :k]) == pytest.approx(k, abs=1e-6)


def test_effective_rank_collapse_and_line():
    assert m.effective_rank(torch.ones(30, 8)) == 0.0
    line = torch.linspace(-1, 1, 30).unsqueeze(1) * torch.ones(1, 8)
    assert m.effective_rank(line) == pytest.approx(1.0, abs=1e-6)


def test_effective_rank_gaussian_is_near_full():
    z = torch.randn(20000, 32, generator=torch.Generator().manual_seed(0))
    assert m.effective_rank(z) > 0.95 * 32


def test_variance_explained_by_dimension():
    target = torch.randn(5000, 6, generator=torch.Generator().manual_seed(0))
    target = target - target.mean(dim=0, keepdim=True)
    pred = target.clone()
    pred[:, 3:] = 0.0  # perfect on the first three dimensions, mean-only on the rest
    ve = m.variance_explained(pred, target)
    assert torch.allclose(ve[:3], torch.ones(3), atol=1e-6)
    assert torch.allclose(ve[3:], torch.zeros(3), atol=1e-6)


def test_variance_explained_can_go_negative():
    target = torch.randn(1000, 2, generator=torch.Generator().manual_seed(0))
    assert (m.variance_explained(-target, target) < 0).all()


def test_flatten_valid_drops_padding():
    x = torch.arange(12.0).reshape(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, False, False]])
    assert m.flatten_valid(x, mask).tolist() == [[0.0, 1.0], [2.0, 3.0], [6.0, 7.0]]


def test_consecutive_pairs_respect_mask():
    z = torch.arange(8.0).reshape(2, 4, 1)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    prev, nxt = m.consecutive_pairs(z, mask)
    assert prev.squeeze(1).tolist() == [0.0, 1.0, 4.0]
    assert nxt.squeeze(1).tolist() == [1.0, 2.0, 5.0]


def test_copy_previous_on_constant_and_on_noise():
    constant = torch.ones(3, 5, 4) * torch.arange(3.0).reshape(3, 1, 1)
    assert m.copy_previous_baseline(constant)["mse"] == 0.0
    # independent unit-variance chunks: copying the last one costs 2 per dimension
    noise = torch.randn(64, 50, 8, generator=torch.Generator().manual_seed(0))
    base = m.copy_previous_baseline(noise)
    assert base["mse"] == pytest.approx(2.0, rel=0.05)
    assert base["mean_variance_explained"] == pytest.approx(-1.0, abs=0.1)


def test_readout_table_perfect_and_printable():
    target = torch.randn(4, 20, 16, generator=torch.Generator().manual_seed(0))
    mask = torch.ones(4, 20, dtype=torch.bool)
    mask[:, -3:] = False
    table = m.readout_table(target.clone(), target, mask)
    assert table["mse"] == 0.0
    assert table["recall@1"] == 1.0
    assert table["candidates"] == 4 * 17
    text = m.format_table({"model": table, "copy previous": m.copy_previous_baseline(target, mask)})
    assert "recall@10" in text and "effective_rank" in text


def test_subsample_keeps_rows_aligned():
    pred = torch.arange(100.0).unsqueeze(1)
    ps, ts = m.subsample(pred, pred + 0.5, max_candidates=10)
    assert ps.shape == (10, 1)
    assert torch.equal(ts, ps + 0.5)
