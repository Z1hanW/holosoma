from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from holosoma.config_values.wbt.g1.observation import critic_obs_w_object_terms
from holosoma.managers.observation.terms.wbt import obj_lin_vel_b_v2
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


def test_training_object_velocity_v2_is_translation_invariant() -> None:
    env = SimpleNamespace(num_envs=2, device="cpu")
    motion_command = SimpleNamespace(
        # Translation is deliberately large: a vector-frame transform must
        # not subtract it from a velocity.
        robot_ref_pos_w=torch.tensor([[100.0, -30.0, 4.0], [-9.0, 8.0, 7.0]]),
        robot_ref_quat_w=torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
        simulator_object_lin_vel_w=torch.tensor(
            [[1.0, 2.0, 3.0], [-4.0, 5.0, -6.0]],
            dtype=torch.float32,
        ),
    )

    with patch(
        "holosoma.managers.observation.terms.wbt._get_motion_command_and_assert_type",
        return_value=motion_command,
    ):
        velocity_b = obj_lin_vel_b_v2(env)

    torch.testing.assert_close(velocity_b, motion_command.simulator_object_lin_vel_w)
    assert (
        critic_obs_w_object_terms["obj_lin_vel_b"].func
        == "holosoma.managers.observation.terms.wbt:obj_lin_vel_b_v2"
    )


def _inference_object_velocity(robot_ref_pos_w: np.ndarray) -> np.ndarray:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._motion_data = SimpleNamespace(
        has_object=True,
        object_pos_w=np.zeros((1, 3), dtype=np.float32),
        object_quat_w=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        object_size=np.ones((1, 3), dtype=np.float32),
    )
    policy._maybe_update_motion_alignment = lambda _state: None
    policy._get_motion_index = lambda: 0
    policy._get_observation_reference_pose_in_world = lambda _state: (
        robot_ref_pos_w,
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    policy._motion_align_quat_wxyz = None
    actor_state = np.zeros((1, 13), dtype=np.float32)
    actor_state[:, 3:7] = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    actor_state[:, 7:10] = np.array([[0.5, -1.25, 2.0]], dtype=np.float32)
    policy._get_sim_actor_state = lambda _name: actor_state
    policy.config = SimpleNamespace(task=SimpleNamespace(sim_object_name="object"))
    policy._pose_in_robot_ref_frame = lambda *_args: (
        np.zeros((1, 3), dtype=np.float32),
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
    policy._get_motion_ref_ori_b = lambda _state: np.zeros((1, 6), dtype=np.float32)
    policy._get_base_ang_vel_obs = lambda _state: np.zeros((1, 3), dtype=np.float32)
    policy.motion_command_t = np.zeros((1, 2), dtype=np.float32)
    policy.num_dofs = 1
    policy.default_dof_angles = np.zeros((1,), dtype=np.float32)
    policy.last_policy_action = np.zeros((1, 1), dtype=np.float32)
    robot_state = np.zeros((1, 15), dtype=np.float32)

    return policy._get_object_generalist_obs_buffer_dict(robot_state)["obj_lin_vel_b"]


def test_inference_object_velocity_matches_v2_and_ignores_translation() -> None:
    origin_result = _inference_object_velocity(np.zeros((1, 3), dtype=np.float32))
    translated_result = _inference_object_velocity(
        np.array([[100.0, -30.0, 4.0]], dtype=np.float32)
    )

    expected = np.array([[0.5, -1.25, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(origin_result, expected)
    np.testing.assert_allclose(translated_result, expected)

