from __future__ import annotations

from pathlib import Path

import numpy as np
from holosoma.config_values.image_server import DEFAULTS as IMAGE_SERVER_DEFAULTS
from holosoma.config_types.image_server import ImageServerConfig
from holosoma.sensors.image_server import ImageServer
from holosoma_inference.config.config_values import camera
from holosoma_inference.config.config_values.inference import DEFAULTS
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy

REPO_ROOT = Path(__file__).resolve().parents[2]


class _FakeInterface:
    no_action = 1


class _FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)


def test_real_debug_launches_depth_server_and_viser_dashboard() -> None:
    script = (REPO_ROOT / "real_debug.sh").read_text(encoding="utf-8")

    assert "bash real_depth.sh" in script
    assert "HOLOSOMA_REAL_DEBUG_DEPTH" in script
    assert "real_d435i_urdf" in script
    assert 'HOLOSOMA_REAL_IMAGE_SERVER_CONFIG="$depth_server_config"' in script
    assert "scripts/real_viser.py" in script
    assert "bash sim_gt_depth.sh" in script
    assert "HOLOSOMA_REAL_DEBUG_SIM_GT" in script
    assert "--no-depth" not in script
    assert '--depth-profile "Real D435: 0mcqao8k processing"' in script
    assert "--depth-source-height 60" in script
    assert "--depth-source-width 106" in script
    assert "--sim-gt-depth-shm-name" in script
    assert 'HOLOSOMA_POLICY_COMMAND_STATUS_PATH="$command_status_path"' in script


def test_real_drop_launches_pose_synced_flat_ground_sim_gt() -> None:
    script = (REPO_ROOT / "real_drop.sh").read_text(encoding="utf-8")

    assert "bash sim_gt_depth.sh" in script
    assert "HOLOSOMA_REAL_DROP_SIM_GT" in script
    assert 'HOLOSOMA_SIM_GT_STATE_PATH="$command_status_path"' in script
    assert "--sim-gt-depth-shm-name" in script
    assert "--depth-source-height 60" in script
    assert "--depth-source-width 106" in script
    assert "--depth-crop-y-start 2" in script
    assert "--depth-crop-x-start 4" in script
    assert "--depth-crop-x-end -4" in script


def test_sim_gt_renderer_is_isolated_from_real_robot_dds() -> None:
    launcher = (REPO_ROOT / "sim_gt_depth.sh").read_text(encoding="utf-8")
    renderer = (REPO_ROOT / "scripts" / "sim_gt_depth_server.py").read_text(encoding="utf-8")

    assert "bridge=disabled (render-only, no DDS)" in launcher
    assert "run_sim.py" not in launcher
    assert "unitree" not in renderer.lower()
    assert 'default="sim_gt_depth_raw_shm"' in renderer
    assert "[0.0576235, 0.01753, 0.41987]" in renderer
    assert "[0.0, 47.6, 0.0]" in renderer
    assert "scene_option.geomgroup[:] = 0" in renderer
    assert "scene_option.geomgroup[0] = 1" in renderer
    assert "scene_option.geomgroup[1] = 1" in renderer
    assert "robot_option.geomgroup[1] = 1" in renderer
    assert 'floor = spec.geom("floor")' in renderer
    assert "sim_gt_box" not in renderer
    assert 'positions = status.get("q_actual")' in renderer


def test_real_debug_depth_server_matches_0mcqao8k_latency_profile() -> None:
    config = IMAGE_SERVER_DEFAULTS["real_d435i_urdf"]

    assert config.latency_frame == (3, 4)
    assert config.buffer_len == 6
    assert config.resized_width == 87
    assert config.resized_height == 58
    assert config.near_clip == 0.3
    assert config.far_clip == 3.0


def test_real_drop_depth_server_matches_policy_latency() -> None:
    config = IMAGE_SERVER_DEFAULTS["real_d435i"]

    assert config.latency_frame == 1
    assert config.frame_rate == 30


def test_image_server_maps_invalid_depth_to_far_plane_before_normalization() -> None:
    server = ImageServer.__new__(ImageServer)
    server.cfg = ImageServerConfig(
        near_clip=0.3,
        far_clip=3.0,
        resized_height=1,
        resized_width=4,
    )
    frame = np.array([[0.0, np.nan, 0.3, 3.0]], dtype=np.float32)

    result = server._resize_clip_expand_transpose(frame)

    np.testing.assert_allclose(result, [[[0.5, 0.5, -0.5, 0.5]]])


def test_real_debug_uses_unitree_zero_joint_diagnostic_posture() -> None:
    config = DEFAULTS["g1-debug-diagnostic"]
    pose = np.asarray(config.robot.stiff_startup_pos, dtype=np.float32)
    kp = np.asarray(config.robot.stiff_startup_kp, dtype=np.float32)
    kd = np.asarray(config.robot.stiff_startup_kd, dtype=np.float32)

    assert config.camera == camera.single_d435i_urdf_depth
    assert config.camera.props.width == 106
    assert config.camera.props.height == 60
    assert config.camera.props.resized_width == 87
    assert config.camera.props.resized_height == 58
    assert config.camera.props.crop_y_start == 2
    assert config.camera.props.crop_x_start == 4
    assert config.camera.props.crop_x_end == -4
    assert pose.shape == (29,)
    assert config.task.stiff_hold_only is True
    assert config.task.stiff_hold_blend_seconds == 5.0
    np.testing.assert_array_equal(pose, np.zeros(29, dtype=np.float32))
    np.testing.assert_array_equal(
        kp,
        [
            350,
            200,
            200,
            300,
            300,
            150,
            350,
            200,
            200,
            300,
            300,
            150,
            200,
            200,
            200,
            *([40] * 14),
        ],
    )
    np.testing.assert_array_equal(
        kd,
        [
            5,
            5,
            5,
            10,
            5,
            5,
            5,
            5,
            5,
            10,
            5,
            5,
            5,
            5,
            5,
            *([3] * 14),
        ],
    )


def test_stiff_hold_blend_starts_at_measured_pose_and_finishes_at_target() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._stiff_hold_active = True
    policy._stiff_hold_q = np.array([[1.0, -1.0]], dtype=np.float32)
    policy._stiff_hold_kp = np.array([40.0, 40.0], dtype=np.float32)
    policy._stiff_hold_kd = np.array([3.0, 3.0], dtype=np.float32)
    policy._stiff_hold_blend_steps = 2
    policy._stiff_hold_blend_count = 0
    policy._stiff_hold_start_q = None
    policy.num_dofs = 2
    robot_state = np.zeros((1, 9), dtype=np.float32)
    robot_state[0, 7:9] = (0.2, -0.3)

    first = policy._get_manual_command(robot_state)
    policy._get_manual_command(robot_state)
    final = policy._get_manual_command(robot_state)

    np.testing.assert_allclose(first["q"], [[0.2, -0.3]])
    np.testing.assert_allclose(first["kp"], [40.0, 40.0])
    np.testing.assert_allclose(final["q"], [[1.0, -1.0]])


def test_stiff_hold_only_rejects_policy_and_motion_activation() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._stiff_hold_only = True
    policy._stiff_hold_active = True
    policy.use_policy_action = False
    policy.get_ready_state = False
    policy.interface = _FakeInterface()
    policy.logger = _FakeLogger()
    policy.motion_clip_progressing = False

    policy._handle_start_policy()
    policy._handle_start_motion_clip()
    policy._handle_init_state()

    assert policy.use_policy_action is False
    assert policy.get_ready_state is False
    assert policy._stiff_hold_active is True
    assert policy.motion_clip_progressing is False
    assert policy.interface.no_action == 0
