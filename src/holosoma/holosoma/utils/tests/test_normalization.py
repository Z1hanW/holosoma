from __future__ import annotations

import numpy as np
import pytest
import torch

from holosoma.utils.normalization import EmpiricalNormalization


def test_empirical_normalization_merges_fractional_batch_weights() -> None:
    normalizer = EmpiricalNormalization(shape=(1,), device="cpu", eps=0.0)

    normalizer.update(torch.tensor([[0.0], [2.0]]), sample_weight=2.0)
    normalizer.update(torch.tensor([[10.0], [14.0]]), sample_weight=1.0)

    expanded = np.asarray([0.0, 2.0, 0.0, 2.0, 10.0, 14.0])
    assert normalizer.count.item() == pytest.approx(6.0)
    assert normalizer.mean.item() == pytest.approx(float(expanded.mean()))
    assert normalizer.std.item() == pytest.approx(float(expanded.std()))


def test_distributed_normalization_reduces_weighted_sufficient_statistics(monkeypatch) -> None:
    normalizer = EmpiricalNormalization(shape=(1,), device="cpu", eps=0.0)

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def add_remote_rank(stats, op):
        assert op == torch.distributed.ReduceOp.SUM
        # Remote rank: samples [10, 14] with rank weight 1.0.
        stats.add_(torch.tensor([24.0, 296.0, 2.0], dtype=stats.dtype))

    monkeypatch.setattr(torch.distributed, "all_reduce", add_remote_rank)
    normalizer.update(torch.tensor([[0.0], [2.0]]), sample_weight=2.0)

    expanded = np.asarray([0.0, 2.0, 0.0, 2.0, 10.0, 14.0])
    assert normalizer.count.item() == pytest.approx(6.0)
    assert normalizer.mean.item() == pytest.approx(float(expanded.mean()))
    assert normalizer.std.item() == pytest.approx(float(expanded.std()))


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_empirical_normalization_rejects_invalid_sample_weight(weight: float) -> None:
    normalizer = EmpiricalNormalization(shape=(1,), device="cpu")

    with pytest.raises(ValueError, match="sample_weight"):
        normalizer.update(torch.ones((2, 1)), sample_weight=weight)
