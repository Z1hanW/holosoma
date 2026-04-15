from __future__ import annotations

import torch

from holosoma.managers.command.terms.wbt import (
    _compute_contact_stage_intervals,
    _probability_mass_on_intervals,
    _select_primary_contact_interval,
)


def test_select_primary_contact_interval_prefers_wrist_union():
    interval = _select_primary_contact_interval(
        {
            "left_palm": [40, 120],
            "right_wrist": [45, 135],
            "arm": [20, 160],
        }
    )

    assert interval == (40, 135)


def test_probability_mass_on_contact_stages_matches_uniform_lengths():
    bin_probabilities = torch.full((10,), 0.1, dtype=torch.float32)
    stage_intervals, after_t2_interval = _compute_contact_stage_intervals(
        t1=20,
        t2=80,
        sample_end_step=100.0,
    )

    stage_masses = _probability_mass_on_intervals(
        bin_probabilities,
        sample_end_step=100.0,
        intervals=stage_intervals,
    )
    after_t2_mass = _probability_mass_on_intervals(
        bin_probabilities,
        sample_end_step=100.0,
        intervals=[after_t2_interval],
    )[0]

    expected = torch.tensor([0.30, 0.06666667, 0.06666667, 0.06666667, 0.30], dtype=torch.float32)
    assert torch.allclose(stage_masses, expected, atol=1.0e-5)
    assert torch.isclose(after_t2_mass, torch.tensor(0.20, dtype=torch.float32), atol=1.0e-5)
    assert torch.isclose(stage_masses.sum() + after_t2_mass, torch.tensor(1.0, dtype=torch.float32), atol=1.0e-5)


def test_short_contact_window_collapses_middle_stages_without_overlap():
    bin_probabilities = torch.full((10,), 0.1, dtype=torch.float32)
    stage_intervals, after_t2_interval = _compute_contact_stage_intervals(
        t1=50,
        t2=70,
        sample_end_step=100.0,
    )

    stage_masses = _probability_mass_on_intervals(
        bin_probabilities,
        sample_end_step=100.0,
        intervals=stage_intervals,
    )
    after_t2_mass = _probability_mass_on_intervals(
        bin_probabilities,
        sample_end_step=100.0,
        intervals=[after_t2_interval],
    )[0]

    expected = torch.tensor([0.60, 0.0, 0.0, 0.0, 0.10], dtype=torch.float32)
    assert torch.allclose(stage_masses, expected, atol=1.0e-5)
    assert torch.isclose(after_t2_mass, torch.tensor(0.30, dtype=torch.float32), atol=1.0e-5)
