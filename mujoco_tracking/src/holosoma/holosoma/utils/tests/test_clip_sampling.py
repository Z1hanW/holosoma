from __future__ import annotations

import torch

from holosoma.utils.clip_sampling import build_prefix_mask, piecewise_constant_schedule_value, project_group_weights


def test_build_prefix_mask_marks_sub_clips_as_clean() -> None:
    clip_ids = ["sub10_largebox_032_mj_w_obj", "box_10", "sub3_largebox_003_mj_w_obj", "box_11"]
    mask = build_prefix_mask(clip_ids, ["sub"])
    assert torch.equal(mask, torch.tensor([True, False, True, False], dtype=torch.bool))


def test_piecewise_constant_schedule_value_uses_requested_mix_breakpoints() -> None:
    starts = [0, 1500, 2000, 2500, 3000, 4000]
    values = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

    assert piecewise_constant_schedule_value(None, starts, values) == 1.0
    assert piecewise_constant_schedule_value(1499, starts, values) == 1.0
    assert piecewise_constant_schedule_value(1500, starts, values) == 0.9
    assert piecewise_constant_schedule_value(2499, starts, values) == 0.8
    assert piecewise_constant_schedule_value(3999, starts, values) == 0.6
    assert piecewise_constant_schedule_value(4000, starts, values) == 0.5
    assert piecewise_constant_schedule_value(10000, starts, values) == 0.5


def test_project_group_weights_preserves_within_group_ratios() -> None:
    base_weights = torch.tensor([0.50, 0.30, 0.15, 0.05], dtype=torch.float32)
    clean_mask = torch.tensor([True, True, False, False], dtype=torch.bool)

    projected = project_group_weights(base_weights, clean_mask=clean_mask, clean_group_probability=0.8)

    assert torch.isclose(projected.sum(), torch.tensor(1.0))
    assert torch.isclose(projected[clean_mask].sum(), torch.tensor(0.8))
    assert torch.isclose(projected[~clean_mask].sum(), torch.tensor(0.2))
    assert torch.allclose(projected[clean_mask], torch.tensor([0.50, 0.30]) / 0.80 * 0.8)
    assert torch.allclose(projected[~clean_mask], torch.tensor([0.15, 0.05]) / 0.20 * 0.2)
