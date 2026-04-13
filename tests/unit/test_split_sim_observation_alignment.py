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


def test_wbt_prime_auto_start_policy_history_consumes_root_reference_and_keeps_last_action() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._dryrun_autostart_policy_history = True
    policy._warm_autostart_obs_history = True
    policy._obs_input_name = "actor_obs"
    policy._action_output_name = "action"
    policy._perception_input_name = None
    policy._time_step_input_name = "time_step"
    policy._autostart_policy_history_prime_steps_override = "2"
    policy.history_length_dict = {"actor_obs": 5}
    policy.config = SimpleNamespace(task=SimpleNamespace(use_root_reference_at_clip_start=True))
    policy.motion_timestep = 7
    policy.motion_start_timestep = 123
    policy._last_clock_reading = 456
    policy._last_motion_output_timestep = 9
    policy.motion_command_0 = np.array([[0.1, 0.2]], dtype=np.float32)
    policy.motion_command_t = policy.motion_command_0.copy()
    policy.ref_quat_xyzw_0 = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    policy.ref_quat_xyzw_t = policy.ref_quat_xyzw_0.copy()
    policy.ref_pos_xyz_t = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    policy.last_policy_action = np.zeros((1, 2), dtype=np.float32)
    policy.scaled_policy_action = np.zeros((1, 2), dtype=np.float32)
    policy.policy_action_scales = np.array([[0.5, 2.0]], dtype=np.float32)
    policy._uses_motion_command = False
    policy._motion_data = None
    policy._logged_root_reference_clip_start = False
    policy._remaining_root_reference_clip_start_obs = 0
    policy._preserve_obs_history_on_next_motion_start = False
    policy._preserve_root_reference_state_on_next_motion_start = False
    policy._auto_start_history_snapshot = {"stale": {}}
    policy._augment_robot_state_with_sim_state = lambda state: state
    policy._refresh_motion_outputs_for_current_timestep = lambda: None
    policy._reset_observation_history_state = lambda: (
        policy.last_policy_action.fill(0.0),
        policy.scaled_policy_action.fill(0.0),
    )
    policy.prepare_obs_for_rl = lambda state: {"actor_obs": np.array([[1.0, 2.0]], dtype=np.float32)}
    policy.policy = lambda feed: {"action": np.array([[0.25, -0.5]], dtype=np.float32)}
    policy.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)

    primed = policy._prime_auto_start_policy_history(np.zeros((1, 16), dtype=np.float32))

    assert primed is True
    np.testing.assert_allclose(policy.last_policy_action, [[0.25, -0.5]])
    np.testing.assert_allclose(policy.scaled_policy_action, [[0.125, -1.0]])
    assert policy.motion_timestep == 0
    assert policy.motion_start_timestep is None
    assert policy._last_clock_reading is None
    assert policy._remaining_root_reference_clip_start_obs == 0
    assert policy._preserve_obs_history_on_next_motion_start is True
    assert policy._preserve_root_reference_state_on_next_motion_start is True
    assert policy._auto_start_history_snapshot is None


def test_wbt_autostart_policy_history_prime_steps_defaults_to_actor_history_length() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._autostart_policy_history_prime_steps_override = ""
    policy.history_length_dict = {"actor_obs": 5}
    policy._training_freeze_zero_extra_holds = 19

    assert policy._get_autostart_policy_history_prime_steps() == 4


def test_wbt_handle_start_motion_clip_preserves_consumed_root_reference_when_requested() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.clock_sub = SimpleNamespace(reset_origin=lambda: None)
    policy._preserve_obs_history_on_next_motion_start = True
    policy._preserve_root_reference_state_on_next_motion_start = True
    policy._reset_observation_history_state = lambda: (_ for _ in ()).throw(AssertionError("history should be preserved"))
    policy._auto_start_history_snapshot = {"stale": {}}
    policy.motion_clip_progressing = False
    policy.motion_start_timestep = 123
    policy.motion_timestep = 4
    policy._last_motion_output_timestep = 9
    policy.motion_command_0 = np.array([[0.1, 0.2]], dtype=np.float32)
    policy.motion_command_t = np.array([[9.0, 9.0]], dtype=np.float32)
    policy.ref_quat_xyzw_0 = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    policy.ref_quat_xyzw_t = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)
    policy._refresh_motion_outputs_for_current_timestep = lambda: None
    policy._last_clock_reading = 88
    policy._training_freeze_zero_extra_holds = 3
    policy._training_freeze_zero_remaining_holds = 0
    policy._logged_training_freeze_zero_alignment = True
    policy._logged_root_reference_clip_start = True
    policy._remaining_root_reference_clip_start_obs = 0
    policy._logged_first_policy_step_debug = True
    policy._motion_alignment_enabled = False
    policy.config = SimpleNamespace(task=SimpleNamespace(use_root_reference_at_clip_start=True))
    policy.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)

    policy._handle_start_motion_clip()

    assert policy._preserve_obs_history_on_next_motion_start is False
    assert policy._preserve_root_reference_state_on_next_motion_start is False
    assert policy._remaining_root_reference_clip_start_obs == 0
    assert policy.motion_timestep == 0
    assert policy.motion_clip_progressing is True
    assert policy._training_freeze_zero_remaining_holds == 3
