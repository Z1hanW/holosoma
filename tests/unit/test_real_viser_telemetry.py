from __future__ import annotations

import json

import numpy as np
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


def test_sparse_command_status_includes_live_robot_pose(tmp_path) -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._policy_command_status_path = tmp_path / "latest_command.json"
    policy._policy_command_status_next_time = 0.0
    policy._policy_command_status_period = 0.05
    policy._manual_sparse_root_command_offset = np.zeros((1, 3), dtype=np.float32)
    policy._joystick_sparse_root_command_offset = np.zeros((1, 3), dtype=np.float32)
    policy._external_sparse_root_command_mode = False
    policy._force_zero_sparse_root_command = False
    policy._logged_policy_command_status_error = False
    policy.motion_clip_progressing = True
    policy.motion_timestep = 12
    policy.dof_names = ("joint_a", "joint_b")
    policy.num_dofs = 2
    policy.cmd_q = np.array([0.4, -0.2], dtype=np.float32)
    policy.use_policy_action = True
    policy.get_ready_state = False
    policy._stiff_hold_active = True
    policy._stiff_hold_only = True
    policy.obs_dims = {}

    robot_state = np.zeros((1, 9), dtype=np.float32)
    robot_state[0, :3] = (0.1, 0.2, 0.3)
    robot_state[0, 3:7] = (1.0, 0.0, 0.0, 0.0)
    robot_state[0, 7:9] = (0.25, -0.1)
    command = {"sparse_target_root_trajectory_command": np.array([[0.2, -0.3, 0.4]], dtype=np.float32)}

    policy._write_sparse_root_command_status(command, robot_state)

    payload = json.loads(policy._policy_command_status_path.read_text(encoding="utf-8"))
    assert payload["dof_names"] == ["joint_a", "joint_b"]
    np.testing.assert_allclose(payload["q_actual"], [0.25, -0.1])
    np.testing.assert_allclose(payload["q_target"], [0.4, -0.2])
    np.testing.assert_allclose(payload["base_position"], [0.1, 0.2, 0.3])
    assert payload["base_wxyz"] == [1.0, 0.0, 0.0, 0.0]
    assert payload["motion_timestep"] == 12
    assert payload["use_policy_action"] is True
    assert payload["stiff_hold_active"] is True
    assert payload["stiff_hold_only"] is True
