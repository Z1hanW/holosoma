from __future__ import annotations

import numpy as np
from holosoma_inference.config.config_values.inference import DEFAULTS
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


class _FakeInterface:
    no_action = 1


class _FakeLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)

    def warning(self, message: str) -> None:
        self.messages.append(message)


def test_real_debug_pose_uses_g1_forward_elbow_zero() -> None:
    config = DEFAULTS["g1-debug-stiff-90"]
    pose = np.asarray(config.robot.stiff_startup_pos, dtype=np.float32)
    names = tuple(config.robot.dof_names)

    assert pose.shape == (29,)
    assert config.task.stiff_hold_only is True
    assert config.task.stiff_hold_blend_seconds == 5.0
    # G1's neutral elbow link points forward: zero joint angle is the physical
    # forearm-to-upper-arm right angle. Positive angles fold it rearward/down.
    np.testing.assert_allclose(
        pose[[names.index("left_elbow_joint"), names.index("right_elbow_joint")]],
        0.0,
        atol=1.0e-6,
    )
    np.testing.assert_allclose(
        pose[names.index("left_shoulder_pitch_joint")],
        pose[names.index("right_shoulder_pitch_joint")],
    )
    np.testing.assert_allclose(
        pose[names.index("left_shoulder_roll_joint")],
        -pose[names.index("right_shoulder_roll_joint")],
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
