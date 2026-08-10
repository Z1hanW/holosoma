from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "real_viser.py"
SPEC = importlib.util.spec_from_file_location("real_viser", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
real_viser = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(real_viser)


def test_parse_args_can_disable_depth_panel(tmp_path) -> None:
    args = real_viser._parse_args(["--state-path", str(tmp_path / "state.json"), "--no-depth"])

    assert args.no_depth is True


def test_parse_args_can_enable_sim_gt_panel(tmp_path) -> None:
    args = real_viser._parse_args(
        [
            "--state-path",
            str(tmp_path / "state.json"),
            "--sim-gt-depth-shm-name",
            "sim_gt_depth_raw_shm",
        ]
    )

    assert args.sim_gt_depth_shm_name == "sim_gt_depth_raw_shm"
    assert (args.sim_gt_depth_height, args.sim_gt_depth_width) == (60, 106)


def test_normalized_depth_to_meters_maps_policy_range() -> None:
    normalized = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    meters = real_viser.normalized_depth_to_meters(normalized, 0.3, 3.0)
    np.testing.assert_allclose(meters, [0.3, 1.65, 3.0], atol=1.0e-6)


def test_sim_gt_uses_exact_policy_crop_resize_and_normalization() -> None:
    raw_meters = np.full((60, 106), 1.65, dtype=np.float32)
    raw_meters[:2] = 0.3
    raw_meters[:, :4] = 0.3
    raw_meters[:, -4:] = 3.0

    processed = real_viser.prepare_sim_gt_policy_depth(
        raw_meters,
        near=0.3,
        far=3.0,
        output_height=58,
        output_width=87,
        crop_y_start=2,
        crop_x_start=4,
        crop_x_end=-4,
    )

    assert processed.shape == (58, 87)
    np.testing.assert_allclose(processed, 0.0, atol=1.0e-6)


def test_robot_part_comparison_ignores_background_and_color_codes_error() -> None:
    sim = np.array([[-0.5, 0.0, 0.0, 0.5]], dtype=np.float32)
    real = np.array([[-0.5, 0.02, 0.2, 0.5]], dtype=np.float32)
    comparison, stats = real_viser.robot_part_depth_comparison(real, sim, near=0.3, far=3.0)

    np.testing.assert_array_equal(comparison[0, 0], [0, 220, 70])
    np.testing.assert_array_equal(comparison[0, 1], [255, 200, 0])
    np.testing.assert_array_equal(comparison[0, 2], [255, 45, 30])
    np.testing.assert_array_equal(comparison[0, 3], [0, 0, 0])
    assert stats["robot_pixels"] == 3
    assert stats["compared_pixels"] == 3
    assert stats["coverage_percent"] == 100.0


def test_depth_point_cloud_drops_far_sentinel_and_uses_robot_axes() -> None:
    normalized = np.array([[-0.5, 0.5], [0.0, 0.5]], dtype=np.float32)
    points, colors = real_viser.depth_point_cloud(
        normalized,
        near=0.3,
        far=3.0,
        horizontal_fov_deg=90.0,
    )

    assert points.shape == (2, 3)
    assert colors.shape == (2, 3)
    np.testing.assert_allclose(points[:, 0], [0.3, 1.65], atol=1.0e-6)
    assert points[0, 1] > 0.0  # left image pixel -> robot-left y
    assert points[0, 2] > 0.0  # top image pixel -> robot-up z


def test_depth_point_cloud_uses_urdf_render_crop_intrinsics() -> None:
    normalized = np.full((58, 87), -0.5, dtype=np.float32)
    points, _ = real_viser.depth_point_cloud(
        normalized,
        near=1.0,
        far=2.0,
        horizontal_fov_deg=89.5,
        vertical_fov_deg=58.6,
        source_height=60,
        source_width=106,
        crop_y_start=2,
        crop_x_start=4,
        crop_x_end=-4,
    )

    # Cropping the source narrows the horizontal ray spread; cropping only
    # from the top also shifts the processed image slightly downward.
    assert float(points[:, 1].max() - points[:, 1].min()) < 2.0
    assert float(points[:, 2].mean()) < 0.0


def test_joint_values_are_reordered_by_name() -> None:
    actual = real_viser.joint_values_in_viser_order(
        [1.0, 2.0, 3.0],
        ["joint_b", "joint_c", "joint_a"],
        ["joint_a", "joint_b", "missing", "joint_c"],
    )
    np.testing.assert_array_equal(actual, [3.0, 1.0, 0.0, 2.0])


def test_invalid_quaternion_falls_back_to_identity() -> None:
    assert real_viser.normalized_wxyz([0.0, 0.0, 0.0, 0.0]) == (1.0, 0.0, 0.0, 0.0)
