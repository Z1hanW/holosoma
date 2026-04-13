from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.sdk.zmq_interface_wrapper import ZmqSimInterfaceWrapper
from holosoma_inference.utils.math.quat import quat_rotate_inverse, xyzw_to_wxyz


class _FakeSimStateSub:
    def __init__(self, state: dict) -> None:
        self._state = state

    def get_state(self) -> dict:
        return self._state


def _make_robot_config(num_joints: int = 2) -> RobotConfig:
    return RobotConfig(
        robot_type="test_robot",
        robot="test",
        default_dof_angles=tuple([0.0] * num_joints),
        default_motor_angles=tuple([0.0] * num_joints),
        joint_pos_min=tuple([-1.0] * num_joints),
        joint_pos_max=tuple([1.0] * num_joints),
        joint_vel_limit=tuple([1.0] * num_joints),
        motor_effort_limit=tuple([1.0] * num_joints),
        motor2joint=tuple(range(num_joints)),
        joint2motor=tuple(range(num_joints)),
        dof_names=tuple(f"j{i}" for i in range(num_joints)),
        dof_names_upper_body=tuple(),
        dof_names_lower_body=tuple(f"j{i}" for i in range(num_joints)),
        num_motors=num_joints,
        num_joints=num_joints,
    )


def test_zmq_wrapper_converts_root_velocities_to_body_frame() -> None:
    yaw = np.pi / 2.0
    quat_xyzw = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float64)
    root_state = np.array(
        [
            1.0,
            2.0,
            3.0,
            *quat_xyzw,
            1.25,
            -0.75,
            0.5,
            0.2,
            -0.4,
            0.6,
        ],
        dtype=np.float64,
    )
    dof_pos = np.array([0.1, -0.2], dtype=np.float64)
    dof_vel = np.array([0.3, -0.4], dtype=np.float64)

    wrapper = object.__new__(ZmqSimInterfaceWrapper)
    wrapper.robot_config = _make_robot_config(num_joints=2)
    wrapper._sim_state_sub = _FakeSimStateSub(
        {
            "robot_root_state": root_state.tolist(),
            "robot_dof_pos": dof_pos.tolist(),
            "robot_dof_vel": dof_vel.tolist(),
        }
    )
    wrapper._last_robot_state_data = None

    robot_state_data = wrapper.get_low_state()
    assert robot_state_data is not None

    quat_wxyz = xyzw_to_wxyz(root_state[3:7].reshape(1, 4))
    expected_lin_vel_b = quat_rotate_inverse(quat_wxyz, root_state[7:10].reshape(1, 3))
    expected_ang_vel_b = quat_rotate_inverse(quat_wxyz, root_state[10:13].reshape(1, 3))

    np.testing.assert_allclose(robot_state_data[:, :7], np.concatenate([root_state[:3], quat_wxyz.reshape(-1)]).reshape(1, -1))
    np.testing.assert_allclose(robot_state_data[:, 7 + 2 : 7 + 2 + 3], expected_lin_vel_b)
    np.testing.assert_allclose(robot_state_data[:, 7 + 2 + 3 : 7 + 2 + 6], expected_ang_vel_b)


def test_wbt_policy_prefers_sim_state_body_frame_velocities() -> None:
    yaw = -np.pi / 3.0
    quat_xyzw = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)], dtype=np.float32)
    root_state = np.array(
        [
            [
                0.0,
                0.0,
                0.8,
                *quat_xyzw,
                -0.6,
                0.9,
                0.1,
                0.25,
                -0.35,
                0.45,
            ]
        ],
        dtype=np.float32,
    )

    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.num_dofs = 2
    policy._get_sim_root_state = lambda: root_state

    robot_state_data = np.full((1, 7 + 2 + 6 + 2), 123.0, dtype=np.float32)
    expected_quat_wxyz = xyzw_to_wxyz(root_state[:, 3:7])
    expected_lin_vel_b = quat_rotate_inverse(expected_quat_wxyz, root_state[:, 7:10])
    expected_ang_vel_b = quat_rotate_inverse(expected_quat_wxyz, root_state[:, 10:13])

    np.testing.assert_allclose(policy._get_base_lin_vel_obs(robot_state_data), expected_lin_vel_b)
    np.testing.assert_allclose(policy._get_base_ang_vel_obs(robot_state_data), expected_ang_vel_b)


def test_wbt_policy_sources_motion_outputs_from_transitioned_motion_data() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._uses_motion_command = True
    policy._motion_data = SimpleNamespace(
        frame_count=3,
        joint_pos=np.array(
            [
                [-0.312, 0.0],
                [-0.2, 0.1],
                [-0.1, 0.2],
            ],
            dtype=np.float32,
        ),
        joint_vel=np.array(
            [
                [0.0, 0.0],
                [0.05, -0.05],
                [0.1, -0.1],
            ],
            dtype=np.float32,
        ),
        ref_quat_w=np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.9238795, 0.0, 0.0, 0.38268343],
                [0.70710677, 0.0, 0.0, 0.70710677],
            ],
            dtype=np.float32,
        ),
        ref_pos_w=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ],
            dtype=np.float32,
        ),
    )
    policy.config = SimpleNamespace(task=SimpleNamespace(apply_training_motion_transitions=True))

    frame0 = policy._query_motion_outputs_at(0)
    assert frame0 is not None
    np.testing.assert_allclose(frame0["joint_pos"], [[-0.312, 0.0]])
    np.testing.assert_allclose(frame0["joint_vel"], [[0.0, 0.0]])
    np.testing.assert_allclose(frame0["ref_quat_xyzw"], [[0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(frame0["ref_pos_xyz"], [[0.1, 0.2, 0.3]])

    clamped = policy._query_motion_outputs_at(99)
    assert clamped is not None
    np.testing.assert_allclose(clamped["joint_pos"], [[-0.1, 0.2]])
    np.testing.assert_allclose(clamped["joint_vel"], [[0.1, -0.1]])
    np.testing.assert_allclose(clamped["ref_quat_xyzw"], [[0.0, 0.0, 0.70710677, 0.70710677]])
    np.testing.assert_allclose(clamped["ref_pos_xyz"], [[0.7, 0.8, 0.9]])


def test_base_policy_forces_active_pd_back_to_training_metadata() -> None:
    robot_cfg = _make_robot_config(num_joints=2)
    robot_cfg = replace(robot_cfg, motor_kp=(350.0, 200.0), motor_kd=(5.0, 5.0))

    policy = object.__new__(BasePolicy)
    policy.onnx_kp = np.array([40.179238, 99.09843], dtype=np.float32)
    policy.onnx_kd = np.array([2.5578897, 6.308802], dtype=np.float32)
    policy.robot_config = robot_cfg
    policy.interface = SimpleNamespace(robot_config=robot_cfg, backend="zmq", kp_level=0.5, kd_level=0.7)
    policy._logged_training_pd_sync = False

    policy._sync_policy_pd_with_training()

    np.testing.assert_allclose(np.asarray(policy.robot_config.motor_kp), policy.onnx_kp)
    np.testing.assert_allclose(np.asarray(policy.robot_config.motor_kd), policy.onnx_kd)
    np.testing.assert_allclose(np.asarray(policy.interface.robot_config.motor_kp), policy.onnx_kp)
    np.testing.assert_allclose(np.asarray(policy.interface.robot_config.motor_kd), policy.onnx_kd)
    assert policy.interface.kp_level == 1.0
    assert policy.interface.kd_level == 1.0


def test_wbt_root_reference_clip_start_is_consumed_after_one_actor_step() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(task=SimpleNamespace(use_root_reference_at_clip_start=True))
    policy._suppress_root_reference_at_clip_start = False
    policy._logged_root_reference_clip_start = False
    policy._remaining_root_reference_clip_start_obs = 1
    policy._motion_data = None
    policy.motion_timestep = 0

    assert policy._should_use_root_reference_at_clip_start() is True

    policy._consume_root_reference_at_clip_start()

    assert policy._remaining_root_reference_clip_start_obs == 0
    assert policy._should_use_root_reference_at_clip_start() is False
