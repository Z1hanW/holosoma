from __future__ import annotations

import pytest
import torch

from holosoma.managers.randomization.terms.locomotion import (
    _map_unit_samples_to_positive_scale,
)


def test_log_uniform_quartile_boundaries_are_equal_octaves():
    unit_samples = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)

    scales = _map_unit_samples_to_positive_scale(
        unit_samples,
        lower=0.25,
        upper=4.0,
        distribution="log_uniform",
    )

    assert scales.tolist() == pytest.approx([0.25, 0.5, 1.0, 2.0, 4.0])


def test_linear_uniform_mapping_remains_backward_compatible():
    unit_samples = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)

    scales = _map_unit_samples_to_positive_scale(
        unit_samples,
        lower=0.25,
        upper=4.0,
        distribution="uniform",
    )

    assert scales.tolist() == pytest.approx([0.25, 2.125, 4.0])


def test_object_mass_scale_rejects_unknown_distribution():
    with pytest.raises(ValueError, match="uniform.*log_uniform"):
        _map_unit_samples_to_positive_scale(
            torch.tensor([0.5]),
            lower=0.25,
            upper=4.0,
            distribution="normal",
        )
