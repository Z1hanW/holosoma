from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.sdk import zmq_interface_wrapper as zmq_wrapper_module
from holosoma_inference.sdk.zmq_interface_wrapper import ZmqSimInterfaceWrapper
from holosoma_inference.utils.math.quat import quat_rotate_inverse, xyzw_to_wxyz
from holosoma_inference.utils.perception_obs import PerceptionObsSub


class _FakeSimStateSub:
    def __init__(
        self,
        state: dict | None,
        *,
        message_sequence: int = 0,
        received_monotonic_ns: int | None = None,
    ) -> None:
        self._state = state
        self.message_sequence = message_sequence
        self.last_receive_monotonic_ns = received_monotonic_ns

    def get_state(self) -> dict | None:
        return self._state

    def set_state(
        self,
        state: dict | None,
        *,
        message_sequence: int,
        received_monotonic_ns: int | None,
    ) -> None:
        self._state = state
        self.message_sequence = message_sequence
        self.last_receive_monotonic_ns = received_monotonic_ns


class _FakeSimControlPub:
    def __init__(self) -> None:
        self.messages: list[dict] = []

    def publish(self, payload: dict) -> None:
        self.messages.append(payload)


def _make_sim_state(
    *,
    sim_time_ms: float,
    marker: str,
    root_x: float,
    episode_generation: int = 0,
    dof_names: tuple[str, ...] = ("j0", "j1"),
) -> dict:
    return {
        "sim_time_ms": sim_time_ms,
        "episode_generation": episode_generation,
        "robot_root_state": [root_x, 0.0, 0.8, 0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "robot_dof_pos": [root_x + 0.01, root_x + 0.02],
        "robot_dof_vel": [root_x + 0.03, root_x + 0.04],
        "robot_dof_names": list(dof_names),
        "actors": {"object": [root_x] * 13},
        "metadata": {"marker": marker},
    }


def _make_zmq_wrapper_for_test(
    state_sub: _FakeSimStateSub,
    *,
    max_wall_age_ms: float = 500.0,
) -> ZmqSimInterfaceWrapper:
    wrapper = object.__new__(ZmqSimInterfaceWrapper)
    wrapper.robot_config = _make_robot_config(num_joints=2)
    wrapper._sim_state_sub = state_sub
    wrapper._sim_control_pub = _FakeSimControlPub()
    wrapper._sim_state_max_wall_age_ms = max_wall_age_ms
    wrapper._last_robot_state_data = None
    wrapper._last_sim_time_ms = None
    wrapper._last_sim_state_snapshot = None
    wrapper._pinned_sim_state_snapshot = None
    wrapper._last_processed_receipt_sequence = None
    wrapper._last_processed_state_identity = None
    wrapper._lowcmd_seq = 0
    wrapper.no_action = 1
    wrapper.kp_level = 1.0
    wrapper.kd_level = 1.0
    return wrapper


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
            "sim_time_ms": 20,
            "episode_generation": 0,
            "robot_root_state": root_state.tolist(),
            "robot_dof_pos": dof_pos.tolist(),
            "robot_dof_vel": dof_vel.tolist(),
            "robot_dof_names": ["j0", "j1"],
            "actors": {},
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


def test_zmq_wrapper_malformed_next_packet_does_not_partially_commit_snapshot(monkeypatch) -> None:
    clock = {"now_ns": 1_000_000_000}
    monkeypatch.setattr(
        zmq_wrapper_module,
        "time",
        SimpleNamespace(monotonic_ns=lambda: clock["now_ns"]),
    )
    state_s1 = _make_sim_state(sim_time_ms=10.0, marker="S1", root_x=1.0)
    state_sub = _FakeSimStateSub(
        state_s1,
        message_sequence=1,
        received_monotonic_ns=clock["now_ns"],
    )
    wrapper = _make_zmq_wrapper_for_test(state_sub)

    robot_s1 = wrapper.get_low_state()
    assert robot_s1 is not None
    snapshot_s1 = wrapper.get_latest_sim_state_snapshot()
    assert snapshot_s1 is not None
    robot_s1_copy = robot_s1.copy()

    state_s2 = _make_sim_state(sim_time_ms=20.0, marker="S2", root_x=2.0)
    state_s2["actors"] = {"object": [2.0] * 12}
    state_sub.set_state(
        state_s2,
        message_sequence=2,
        received_monotonic_ns=clock["now_ns"],
    )

    with pytest.raises(ValueError, match=r"actors\['object'\].*at least 13"):
        wrapper.get_low_state()

    assert wrapper._last_sim_state_snapshot is snapshot_s1
    assert wrapper._last_robot_state_data is snapshot_s1.robot_state_data
    assert wrapper._last_sim_time_ms == 10.0
    assert wrapper._last_processed_receipt_sequence == 1
    assert snapshot_s1.payload["metadata"]["marker"] == "S1"
    np.testing.assert_array_equal(wrapper._last_robot_state_data, robot_s1_copy)


def test_zmq_wrapper_rejects_same_size_reordered_dof_names_without_commit(monkeypatch) -> None:
    clock = {"now_ns": 2_000_000_000}
    monkeypatch.setattr(
        zmq_wrapper_module,
        "time",
        SimpleNamespace(monotonic_ns=lambda: clock["now_ns"]),
    )
    state_sub = _FakeSimStateSub(
        _make_sim_state(
            sim_time_ms=10.0,
            marker="S1",
            root_x=1.0,
            episode_generation=3,
        ),
        message_sequence=1,
        received_monotonic_ns=clock["now_ns"],
    )
    wrapper = _make_zmq_wrapper_for_test(state_sub)
    robot_s1 = wrapper.get_low_state()
    assert robot_s1 is not None
    snapshot_s1 = wrapper.get_latest_sim_state_snapshot()
    assert snapshot_s1 is not None

    state_sub.set_state(
        _make_sim_state(
            sim_time_ms=20.0,
            marker="S2",
            root_x=2.0,
            dof_names=("j1", "j0"),
        ),
        message_sequence=2,
        received_monotonic_ns=clock["now_ns"],
    )

    with pytest.raises(ValueError, match="robot_dof_names must exactly match"):
        wrapper.get_low_state()

    assert wrapper._last_sim_state_snapshot is snapshot_s1
    assert wrapper._last_sim_time_ms == 10.0
    np.testing.assert_array_equal(wrapper._last_robot_state_data, robot_s1)


def test_zmq_wrapper_low_command_uses_pinned_snapshot_time_when_latest_advances(monkeypatch) -> None:
    clock = {"now_ns": 3_000_000_000}
    monkeypatch.setattr(
        zmq_wrapper_module,
        "time",
        SimpleNamespace(monotonic_ns=lambda: clock["now_ns"]),
    )
    state_sub = _FakeSimStateSub(
        _make_sim_state(
            sim_time_ms=10.0,
            marker="S1",
            root_x=1.0,
            episode_generation=3,
        ),
        message_sequence=1,
        received_monotonic_ns=clock["now_ns"],
    )
    wrapper = _make_zmq_wrapper_for_test(state_sub)
    assert wrapper.get_low_state() is not None
    pinned_s1 = wrapper.pin_latest_sim_state_for_control_tick()
    assert pinned_s1 is not None

    state_sub.set_state(
        _make_sim_state(
            sim_time_ms=20.0,
            marker="S2",
            root_x=2.0,
            episode_generation=4,
        ),
        message_sequence=2,
        received_monotonic_ns=clock["now_ns"],
    )
    assert wrapper.get_low_state() is not None
    latest_s2 = wrapper.get_latest_sim_state_snapshot()
    assert latest_s2 is not None
    assert latest_s2.sim_time_ms == 20.0
    assert wrapper.get_pinned_sim_state_snapshot() is pinned_s1
    assert wrapper.get_sim_time_ms() == 10.0

    wrapper.send_low_command(
        np.array([0.1, 0.2], dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
    )

    assert len(wrapper._sim_control_pub.messages) == 1
    assert wrapper._sim_control_pub.messages[0]["policy_sim_time_ms"] == 10.0
    assert wrapper._sim_control_pub.messages[0]["episode_generation"] == 3


def test_zmq_wrapper_stale_wall_age_returns_none(monkeypatch) -> None:
    clock = {"now_ns": 4_000_000_000}
    monkeypatch.setattr(
        zmq_wrapper_module,
        "time",
        SimpleNamespace(monotonic_ns=lambda: clock["now_ns"]),
    )
    state_sub = _FakeSimStateSub(
        _make_sim_state(sim_time_ms=10.0, marker="S1", root_x=1.0),
        message_sequence=1,
        received_monotonic_ns=clock["now_ns"],
    )
    wrapper = _make_zmq_wrapper_for_test(state_sub, max_wall_age_ms=5.0)
    assert wrapper.get_low_state() is not None
    assert wrapper.pin_latest_sim_state_for_control_tick() is not None

    clock["now_ns"] += 5_000_001
    state_sub.set_state(
        None,
        message_sequence=1,
        received_monotonic_ns=4_000_000_000,
    )

    assert wrapper.get_low_state() is None
    assert wrapper.get_latest_sim_state_snapshot() is None
    assert wrapper.get_pinned_sim_state_snapshot() is None
    assert wrapper.get_sim_time_ms() is None


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


@pytest.mark.parametrize(
    ("apply_training_motion_transitions", "onnx_outputs"),
    [
        (True, {"action", "joint_pos", "joint_vel", "ref_quat_xyzw"}),
        (False, {"action"}),
    ],
)
def test_wbt_policy_sources_motion_outputs_from_transitioned_or_pure_policy_motion_data(
    apply_training_motion_transitions: bool,
    onnx_outputs: set[str],
) -> None:
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
    policy._motion_output_names = onnx_outputs
    policy.config = SimpleNamespace(
        task=SimpleNamespace(apply_training_motion_transitions=apply_training_motion_transitions)
    )
    policy._effective_motion_transition_settings = {
        "prepend": {
            "applied": apply_training_motion_transitions,
        },
        "append": {"applied": False},
    }

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


def test_wbt_legacy_combined_policy_keeps_embedded_motion_source_without_transition_override() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._uses_motion_command = True
    policy._motion_data = SimpleNamespace(frame_count=1)
    policy._motion_output_names = {"action", "joint_pos", "joint_vel", "ref_quat_xyzw"}
    policy.config = SimpleNamespace(task=SimpleNamespace(apply_training_motion_transitions=False))
    policy._effective_motion_transition_settings = {
        "prepend": {"applied": False},
        "append": {"applied": False},
    }

    assert policy._should_source_motion_outputs_from_motion_data() is False


def test_wbt_inactive_transition_keeps_embedded_motion_and_avoids_external_file_dependency() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(apply_training_motion_transitions=True)
    )
    policy._uses_motion_command = True
    policy._uses_videomimic = False
    policy._uses_object_mocap_distill = False
    policy._uses_object_generalist = False
    policy._uses_legacy_object_obs = False
    policy._uses_sparse_root_command = False
    policy._motion_output_names = {
        "action",
        "joint_pos",
        "joint_vel",
        "ref_quat_xyzw",
    }
    policy._motion_data = SimpleNamespace(frame_count=1)
    policy._effective_motion_transition_settings = {
        "prepend": {"applied": False},
        "append": {"applied": False},
    }

    assert policy._will_apply_authenticated_motion_transition() is False
    assert policy._policy_requires_motion_data_for_setup() is False
    assert policy._should_source_motion_outputs_from_motion_data() is False


def test_wbt_contact_button_observation_requires_external_motion_even_with_embedded_outputs() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(apply_training_motion_transitions=True)
    )
    policy._uses_motion_command = True
    policy._uses_videomimic = False
    policy._uses_object_mocap_distill = False
    policy._uses_object_generalist = False
    policy._uses_legacy_object_obs = False
    policy._uses_sparse_root_command = False
    policy._uses_contact_window_observation = True
    policy._motion_output_names = {
        "action",
        "joint_pos",
        "joint_vel",
        "ref_quat_xyzw",
    }
    policy._effective_motion_transition_settings = {
        "prepend": {"applied": False},
        "append": {"applied": False},
    }

    assert policy._policy_requires_external_motion_data() is False
    assert policy._will_apply_authenticated_motion_transition() is False
    assert policy._policy_requires_motion_data_for_setup() is True


def test_wbt_applied_transition_requires_and_sources_external_motion_data() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(apply_training_motion_transitions=True)
    )
    policy._uses_motion_command = True
    policy._uses_videomimic = False
    policy._uses_object_mocap_distill = False
    policy._uses_object_generalist = False
    policy._uses_legacy_object_obs = False
    policy._uses_sparse_root_command = False
    policy._motion_output_names = {
        "action",
        "joint_pos",
        "joint_vel",
        "ref_quat_xyzw",
    }
    policy._motion_data = SimpleNamespace(frame_count=1)
    policy._effective_motion_transition_settings = {
        "prepend": {"applied": True},
        "append": {"applied": False},
    }

    assert policy._will_apply_authenticated_motion_transition() is True
    assert policy._policy_requires_motion_data_for_setup() is True
    assert policy._should_source_motion_outputs_from_motion_data() is True


@pytest.mark.parametrize(
    ("onnx_outputs", "requires_external_motion_data"),
    [
        ({"action"}, True),
        ({"action", "joint_pos", "joint_vel", "ref_quat_xyzw"}, False),
    ],
)
def test_wbt_plain_motion_policy_loads_external_data_only_for_pure_graph(
    onnx_outputs: set[str],
    requires_external_motion_data: bool,
) -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._uses_motion_command = True
    policy._motion_output_names = set(onnx_outputs)
    policy.onnx_output_names = list(onnx_outputs)

    assert policy._policy_requires_external_motion_data() is requires_external_motion_data


@pytest.mark.parametrize(
    ("first_outputs", "second_outputs", "second_requires_external"),
    [
        (
            {"action", "joint_pos", "joint_vel", "ref_quat_xyzw"},
            {"action"},
            True,
        ),
        (
            {"action"},
            {"action", "joint_pos", "joint_vel", "ref_quat_xyzw"},
            False,
        ),
    ],
)
def test_wbt_policy_setup_motion_state_is_isolated_between_slots(
    first_outputs: set[str],
    second_outputs: set[str],
    second_requires_external: bool,
) -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._uses_motion_command = True
    policy.onnx_output_names = list(first_outputs)
    policy._reset_per_model_motion_state_for_setup()
    policy._motion_data = SimpleNamespace(frame_count=97)
    policy.motion_command_0 = np.ones((1, 4), dtype=np.float32)

    policy.onnx_output_names = list(second_outputs)
    policy._reset_per_model_motion_state_for_setup()

    assert policy._motion_output_names == second_outputs
    assert policy._policy_requires_external_motion_data() is second_requires_external
    assert policy._motion_data is None
    assert policy.motion_command_0 is None


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


def test_base_policy_pairs_perception_with_exact_pinned_episode() -> None:
    contract_sha256 = "ef" * 32
    sub = PerceptionObsSub(port=5558)
    sub.payload_buffer.extend(
        [
            {
                "sim_time_ms": 0.0,
                "episode_generation": 20,
                "perception_contract_sha256": contract_sha256,
                "perception_obs": [1.0, 10.0],
            },
            {
                "sim_time_ms": 0.0,
                "episode_generation": 21,
                "perception_contract_sha256": contract_sha256,
                "perception_obs": [2.0, 20.0],
            },
        ]
    )
    sub.last_payload = sub.payload_buffer[-1]
    policy = object.__new__(BasePolicy)
    policy._perception_contract_sha256 = contract_sha256
    policy._perception_obs_shm_sub = None
    policy._perception_obs_sub = sub

    np.testing.assert_array_equal(
        policy._get_split_perception_obs(
            2,
            target_sim_time_ms=0.0,
            target_episode_generation=20,
        ),
        np.array([[1.0, 10.0]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        policy._get_split_perception_obs(
            2,
            target_sim_time_ms=0.0,
            target_episode_generation=21,
        ),
        np.array([[2.0, 20.0]], dtype=np.float32),
    )


def test_wbt_reads_episode_generation_from_the_pinned_snapshot() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._control_tick_sim_state_snapshot = SimpleNamespace(episode_generation=17)

    assert policy._get_control_tick_episode_generation() == 17


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


def test_wbt_explicit_diagnostic_history_prime_recursively_updates_unpublished_action() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy.num_dofs = 2
    policy._dryrun_autostart_policy_history = True
    policy._warm_autostart_obs_history = True
    policy._obs_input_name = "actor_obs"
    policy._action_output_name = "action"
    policy._perception_input_name = None
    policy._time_step_input_name = "time_step"
    policy._autostart_policy_history_prime_steps_override = ""
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
    policy.policy_action_clip = 0.4
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
    policy_inputs: list[np.ndarray] = []
    policy.prepare_obs_for_rl = lambda state: {"actor_obs": policy.last_policy_action.copy()}

    def run_policy(feed):
        actor_obs = feed["actor_obs"].copy()
        policy_inputs.append(actor_obs)
        return {"action": actor_obs + np.array([[0.25, -0.5]], dtype=np.float32)}

    policy.policy = run_policy
    policy.logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)

    primed = policy._prime_auto_start_policy_history(np.zeros((1, 16), dtype=np.float32))

    assert primed is True
    assert len(policy_inputs) == 4
    np.testing.assert_allclose(policy_inputs[0], [[0.0, 0.0]])
    np.testing.assert_allclose(policy_inputs[1], [[0.25, -0.5]])
    np.testing.assert_allclose(policy_inputs[2], [[0.5, -1.0]])
    np.testing.assert_allclose(policy_inputs[3], [[0.75, -1.5]])
    np.testing.assert_allclose(policy.last_policy_action, [[1.0, -2.0]])
    np.testing.assert_allclose(policy.scaled_policy_action, [[0.2, -0.8]])
    assert policy.motion_timestep == 0
    assert policy.motion_start_timestep is None
    assert policy._last_clock_reading is None
    assert policy._remaining_root_reference_clip_start_obs == 0
    assert policy._preserve_obs_history_on_next_motion_start is True
    assert policy._preserve_root_reference_state_on_next_motion_start is True
    assert policy._auto_start_history_snapshot is None


def test_wbt_scientific_autostart_does_not_prime_unexecuted_action_history() -> None:
    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._dryrun_autostart_policy_history = False
    policy._warm_autostart_obs_history = True
    policy.last_policy_action = np.array([[0.25, -0.5]], dtype=np.float32)
    policy.scaled_policy_action = np.array([[0.1, -0.2]], dtype=np.float32)

    assert policy._prime_auto_start_policy_history(np.zeros((1, 16), dtype=np.float32)) is False
    np.testing.assert_allclose(policy.last_policy_action, [[0.25, -0.5]])
    np.testing.assert_allclose(policy.scaled_policy_action, [[0.1, -0.2]])


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
