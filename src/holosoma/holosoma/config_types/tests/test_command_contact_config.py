import pytest
from pydantic import ValidationError

from holosoma.config_types.command import MotionConfig


_BASE_MOTION_CONFIG = {
    "motion_file": "motion.npz",
    "body_name_ref": ["torso_link"],
    "body_names_to_track": ["torso_link"],
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("contact_aware_button_window_mode", "unknown"),
        ("contact_aware_carry_window_mode", "unknown"),
        ("contact_aware_peak_height_alpha", True),
        ("contact_aware_peak_height_alpha", "0.91"),
        ("contact_aware_peak_height_alpha", -0.01),
        ("contact_aware_peak_height_alpha", 1.01),
        ("contact_aware_peak_height_alpha", float("nan")),
        ("contact_aware_peak_height_alpha", float("inf")),
        ("contact_aware_peak_height_smoothing_steps", True),
        ("contact_aware_peak_height_smoothing_steps", 0),
        ("contact_aware_peak_height_smoothing_steps", 4097),
        ("contact_aware_sparse_root_command_mode", "segment"),
        ("contact_aware_sparse_root_segment_steps", False),
        ("contact_aware_sparse_root_segment_steps", 0),
        ("contact_aware_sparse_root_segment_steps", 1_000_001),
        ("contact_aware_sparse_root_zero_yaw_threshold_deg", False),
        ("contact_aware_sparse_root_zero_yaw_threshold_deg", -0.1),
        ("contact_aware_sparse_root_zero_yaw_threshold_deg", 180.1),
        ("contact_aware_sparse_root_zero_yaw_threshold_deg", float("nan")),
        ("contact_aware_sparse_root_zero_yaw_threshold_deg", float("inf")),
        ("zero_root_command_when_drop_active", 1),
        ("zero_root_command_when_drop_active", "true"),
    ],
)
def test_motion_config_rejects_invalid_contact_window_contract(field, value):
    with pytest.raises(ValidationError, match=field):
        MotionConfig(**_BASE_MOTION_CONFIG, **{field: value})


def test_motion_config_accepts_canonical_contact_window_boundaries():
    config = MotionConfig(
        **_BASE_MOTION_CONFIG,
        contact_aware_carry_window_mode="peak_height",
        contact_aware_peak_height_alpha=1.0,
        contact_aware_peak_height_smoothing_steps=4096,
        contact_aware_sparse_root_command_mode="t1_aligned_segment",
        contact_aware_sparse_root_segment_steps=1_000_000,
        contact_aware_sparse_root_zero_yaw_threshold_deg=180.0,
    )

    assert config.contact_aware_carry_window_mode == "peak_height"
    assert config.contact_aware_peak_height_alpha == 1.0
    assert config.contact_aware_peak_height_smoothing_steps == 4096
    assert config.contact_aware_sparse_root_command_mode == "t1_aligned_segment"
    assert config.contact_aware_sparse_root_segment_steps == 1_000_000
    assert config.contact_aware_sparse_root_zero_yaw_threshold_deg == 180.0


def test_motion_config_preserves_legacy_button_window_default_and_accepts_kinematic():
    legacy = MotionConfig(**_BASE_MOTION_CONFIG)
    kinematic = MotionConfig(
        **_BASE_MOTION_CONFIG,
        contact_aware_button_window_mode="kinematic_lift",
    )

    assert legacy.contact_aware_button_window_mode == "contact_interval"
    assert kinematic.contact_aware_button_window_mode == "kinematic_lift"


def test_motion_config_accepts_precomputed_turn_then_forward_mode():
    config = MotionConfig(
        **_BASE_MOTION_CONFIG,
        contact_aware_sparse_root_command_mode="precomputed_turn_then_forward",
    )

    assert config.contact_aware_sparse_root_command_mode == "precomputed_turn_then_forward"


def test_motion_config_accepts_rolling_reference_delta_mode():
    config = MotionConfig(
        **_BASE_MOTION_CONFIG,
        contact_aware_sparse_root_command_mode="rolling_reference_delta",
        contact_aware_sparse_root_segment_steps=30,
    )

    assert config.contact_aware_sparse_root_command_mode == "rolling_reference_delta"
    assert config.contact_aware_sparse_root_segment_steps == 30


def test_drop_exclusive_root_command_is_explicit_and_legacy_safe():
    legacy = MotionConfig(**_BASE_MOTION_CONFIG)
    enabled = MotionConfig(
        **_BASE_MOTION_CONFIG,
        zero_root_command_when_drop_active=True,
    )

    assert legacy.zero_root_command_when_drop_active is False
    assert enabled.zero_root_command_when_drop_active is True
