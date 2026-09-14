from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "checkpoint_camera_profile.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_camera_profile", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
camera_profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(camera_profile)


def test_legacy_profile_combines_mount_and_extra_pitch() -> None:
    profile = camera_profile._profile_from_legacy(
        {
            "perception": {
                "camera_source": "far_tracking_warp",
                "camera_height": 60,
                "camera_width": 106,
                "camera_warp_resize": [58, 87],
                "camera_warp_crop_top": 2,
                "camera_warp_crop_bottom": 0,
                "camera_warp_crop_left": 4,
                "camera_warp_crop_right": 4,
                "camera_near": 0.3,
                "camera_far": 3.0,
                "camera_hfov_deg": 89.5,
                "camera_vfov_deg": 58.6,
                "camera_fps": 30.0,
                "camera_mount_quat": [0.00644801, 0.23350163, 0.00644801, 0.97231365],
                "camera_pitch_deg": 10.0,
            }
        }
    )

    rotation = camera_profile._euler_xyz_deg_from_quaternion_xyzw(profile["mount_quaternion_xyzw"])
    np.testing.assert_allclose(rotation, [1.2174, 36.9983, 1.1157], atol=1.0e-3)
    assert profile["crop"] == [2, 0, 4, 4]
    assert profile["position_xyz"] == [0.01, 0.01, 0.44]


def test_v2_profile_uses_exported_effective_schema() -> None:
    profile = camera_profile._profile_from_v2(
        {
            "camera_source": "far_tracking_warp",
            "camera_shape": [60, 106],
            "camera_obs_shape": [58, 87],
            "camera_geometry": {
                "near": 0.3,
                "far": 3.0,
                "hfov_deg": 89.5,
                "vfov_deg": 58.6,
                "fps": 30.0,
                "pitch_deg": 0.0,
                "mount_quaternion": [0.0, 0.31730467, 0.0, 0.94832367],
            },
            "effective_observation_schema": {
                "sensor_offset": [0.0576235, 0.01753, 0.42987],
                "crop": [2, 0, 4, 4],
            },
        }
    )

    rotation = camera_profile._euler_xyz_deg_from_quaternion_xyzw(profile["mount_quaternion_xyzw"])
    np.testing.assert_allclose(rotation, [0.0, 37.0, 0.0], atol=1.0e-3)
    assert profile["position_xyz"] == [0.0576235, 0.01753, 0.42987]
