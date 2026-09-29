from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.utils.math.quat import quat_rotate_inverse, xyzw_to_wxyz
from holosoma_inference.utils.sim_control import SimControlPush
from holosoma_inference.utils.sim_state import SimStateSub


def _freeze_json_value(value: Any) -> Any:
    """Return a recursively read-only representation of one JSON value."""

    if isinstance(value, dict):
        return MappingProxyType({str(key): _freeze_json_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


@dataclass(frozen=True)
class SimStateSnapshot:
    """One atomically validated simulator payload and its derived robot state."""

    payload: Mapping[str, Any]
    robot_state_data: np.ndarray
    sim_time_ms: float
    episode_generation: int
    receipt_sequence: int
    received_monotonic_ns: int


class ZmqSimInterfaceWrapper:
    """Minimal split sim2sim interface using ZMQ state/control channels instead of Unitree DDS."""

    def __init__(
        self,
        robot_config: RobotConfig,
        *,
        sim_state_port: int = 5557,
        sim_control_port: int = 5559,
        use_joystick: bool = False,
        sim_state_max_wall_age_ms: float = 500.0,
    ) -> None:
        self.robot_config = robot_config
        self.backend = "zmq"
        self.use_joystick = use_joystick
        self.no_action = 0
        self.kp_level = 1.0
        self.kd_level = 1.0
        self._wc_key_map: dict[int, str] = {}
        self._last_robot_state_data: np.ndarray | None = None
        self._last_sim_time_ms: float | None = None
        self._last_sim_state_snapshot: SimStateSnapshot | None = None
        self._pinned_sim_state_snapshot: SimStateSnapshot | None = None
        self._last_processed_receipt_sequence: int | None = None
        self._last_processed_state_identity: int | None = None
        self._lowcmd_seq = 0

        raw_max_age = os.environ.get(
            "HOLOSOMA_SIM_STATE_MAX_WALL_AGE_MS",
            str(sim_state_max_wall_age_ms),
        )
        try:
            self._sim_state_max_wall_age_ms = float(raw_max_age)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "HOLOSOMA_SIM_STATE_MAX_WALL_AGE_MS must be a finite positive number."
            ) from exc
        if not math.isfinite(self._sim_state_max_wall_age_ms) or self._sim_state_max_wall_age_ms <= 0.0:
            raise ValueError(
                "sim_state_max_wall_age_ms must be finite and > 0, "
                f"got {self._sim_state_max_wall_age_ms!r}."
            )

        self._sim_state_sub = SimStateSub(port=sim_state_port)
        self._sim_state_sub.start()
        self._sim_control_pub = SimControlPush(port=sim_control_port)
        self._sim_control_pub.start()

    def _joint_gains_from_robot_config(self) -> tuple[np.ndarray, np.ndarray]:
        joint_kp = np.zeros(self.robot_config.num_joints, dtype=np.float32)
        joint_kd = np.zeros(self.robot_config.num_joints, dtype=np.float32)
        motor_kp = getattr(self.robot_config, "motor_kp", None)
        motor_kd = getattr(self.robot_config, "motor_kd", None)
        if motor_kp is None or motor_kd is None:
            return joint_kp, joint_kd
        for motor_id, joint_id in enumerate(self.robot_config.motor2joint):
            if 0 <= joint_id < self.robot_config.num_joints:
                joint_kp[joint_id] = float(motor_kp[motor_id])
                joint_kd[joint_id] = float(motor_kd[motor_id])
        return joint_kp, joint_kd

    def _snapshot_is_fresh(self, snapshot: SimStateSnapshot | None) -> bool:
        if snapshot is None:
            return False
        age_ms = (time.monotonic_ns() - snapshot.received_monotonic_ns) / 1.0e6
        return 0.0 <= age_ms <= float(
            getattr(self, "_sim_state_max_wall_age_ms", 500.0)
        )

    @staticmethod
    def _finite_vector(value: Any, *, label: str, minimum_size: int | None = None) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if minimum_size is not None and array.size < minimum_size:
            raise ValueError(
                f"sim-state {label} must contain at least {minimum_size} values, got {array.size}."
            )
        if not np.all(np.isfinite(array)):
            raise ValueError(f"sim-state {label} contains NaN or infinity.")
        return array

    def _build_snapshot(
        self,
        state: dict[str, Any],
        *,
        receipt_sequence: int,
        received_monotonic_ns: int,
    ) -> SimStateSnapshot:
        sim_time_raw = state.get("sim_time_ms")
        try:
            sim_time_ms = float(sim_time_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("sim-state sim_time_ms is missing or non-numeric.") from exc
        if not math.isfinite(sim_time_ms) or sim_time_ms < 0.0:
            raise ValueError(
                f"sim-state sim_time_ms must be finite and non-negative, got {sim_time_ms!r}."
            )
        episode_generation = state.get("episode_generation")
        if (
            isinstance(episode_generation, bool)
            or not isinstance(episode_generation, int)
            or episode_generation < 0
        ):
            raise ValueError(
                "sim-state episode_generation must be a non-negative integer, "
                f"got {episode_generation!r}."
            )

        root_state = self._finite_vector(
            state.get("robot_root_state"),
            label="robot_root_state",
            minimum_size=13,
        )
        dof_pos = self._finite_vector(state.get("robot_dof_pos"), label="robot_dof_pos")
        dof_vel = self._finite_vector(state.get("robot_dof_vel"), label="robot_dof_vel")
        expected_dofs = int(self.robot_config.num_joints)
        if dof_pos.size != expected_dofs or dof_vel.size != expected_dofs:
            raise ValueError(
                "sim-state DOF vectors must exactly match robot_config.num_joints: "
                f"positions={dof_pos.size}, velocities={dof_vel.size}, expected={expected_dofs}."
            )
        dof_names = state.get("robot_dof_names")
        expected_names = tuple(str(name) for name in self.robot_config.dof_names)
        if not isinstance(dof_names, (list, tuple)) or tuple(str(name) for name in dof_names) != expected_names:
            raise ValueError(
                "sim-state robot_dof_names must exactly match robot_config.dof_names in order: "
                f"received={dof_names!r}, expected={list(expected_names)!r}."
            )

        ref_state = state.get("robot_ref_state")
        if ref_state is not None:
            self._finite_vector(ref_state, label="robot_ref_state", minimum_size=13)
        actors = state.get("actors", {})
        if not isinstance(actors, Mapping):
            raise ValueError("sim-state actors must be a JSON object.")
        for actor_name, actor_state in actors.items():
            self._finite_vector(
                actor_state,
                label=f"actors[{str(actor_name)!r}]",
                minimum_size=13,
            )

        quat_xyzw = root_state[3:7].reshape(1, 4)
        quat_norm = float(np.linalg.norm(quat_xyzw))
        if not math.isfinite(quat_norm) or quat_norm <= 1.0e-8:
            raise ValueError("sim-state robot_root_state contains a degenerate quaternion.")
        quat_wxyz = xyzw_to_wxyz(quat_xyzw).reshape(-1).astype(np.float64, copy=False)
        base_lin_vel_b = quat_rotate_inverse(
            quat_wxyz.reshape(1, 4), root_state[7:10].reshape(1, 3)
        ).reshape(-1)
        base_ang_vel_b = quat_rotate_inverse(
            quat_wxyz.reshape(1, 4), root_state[10:13].reshape(1, 3)
        ).reshape(-1)
        q = np.concatenate([root_state[:3], quat_wxyz, dof_pos], axis=0)
        dq = np.concatenate([base_lin_vel_b, base_ang_vel_b, dof_vel], axis=0)
        tau_est = np.zeros_like(dq)
        ddq = np.zeros_like(dq)
        robot_state_data = np.concatenate([q, dq, tau_est, ddq], axis=0).reshape(1, -1)
        if not np.all(np.isfinite(robot_state_data)):
            raise ValueError("derived sim-state robot state contains NaN or infinity.")
        robot_state_data.setflags(write=False)
        frozen_payload = _freeze_json_value(state)
        assert isinstance(frozen_payload, Mapping)
        return SimStateSnapshot(
            payload=frozen_payload,
            robot_state_data=robot_state_data,
            sim_time_ms=sim_time_ms,
            episode_generation=int(episode_generation),
            receipt_sequence=int(receipt_sequence),
            received_monotonic_ns=int(received_monotonic_ns),
        )

    def get_low_state(self) -> np.ndarray | None:
        state = self._sim_state_sub.get_state()
        if state is None:
            snapshot = getattr(self, "_last_sim_state_snapshot", None)
            return snapshot.robot_state_data if self._snapshot_is_fresh(snapshot) else None
        if not isinstance(state, dict):
            raise ValueError(f"sim-state subscriber returned {type(state).__name__}, expected dict.")

        receipt_sequence = int(getattr(self._sim_state_sub, "message_sequence", 0))
        received_ns = getattr(self._sim_state_sub, "last_receive_monotonic_ns", None)
        if received_ns is None:
            received_ns = time.monotonic_ns()
        if (
            getattr(self, "_last_sim_state_snapshot", None) is not None
            and getattr(self, "_last_processed_receipt_sequence", None) == receipt_sequence
            and getattr(self, "_last_processed_state_identity", None) == id(state)
        ):
            return (
                self._last_sim_state_snapshot.robot_state_data
                if self._snapshot_is_fresh(self._last_sim_state_snapshot)
                else None
            )

        # Construct and validate every derived field before committing any of
        # robot/time/full-payload state. A malformed next packet cannot pair an
        # old robot vector with a new timestamp.
        snapshot = self._build_snapshot(
            state,
            receipt_sequence=receipt_sequence,
            received_monotonic_ns=int(received_ns),
        )
        self._last_sim_state_snapshot = snapshot
        self._last_processed_receipt_sequence = receipt_sequence
        self._last_processed_state_identity = id(state)
        self._last_robot_state_data = snapshot.robot_state_data
        self._last_sim_time_ms = snapshot.sim_time_ms
        return snapshot.robot_state_data if self._snapshot_is_fresh(snapshot) else None

    def get_latest_sim_state_snapshot(self) -> SimStateSnapshot | None:
        snapshot = getattr(self, "_last_sim_state_snapshot", None)
        return snapshot if self._snapshot_is_fresh(snapshot) else None

    def pin_latest_sim_state_for_control_tick(self) -> SimStateSnapshot | None:
        self._pinned_sim_state_snapshot = self.get_latest_sim_state_snapshot()
        return self._pinned_sim_state_snapshot

    def release_control_tick_sim_state(self) -> None:
        self._pinned_sim_state_snapshot = None

    def get_pinned_sim_state_snapshot(self) -> SimStateSnapshot | None:
        snapshot = getattr(self, "_pinned_sim_state_snapshot", None)
        return snapshot if self._snapshot_is_fresh(snapshot) else None

    def get_sim_time_ms(self) -> float | None:
        snapshot = self.get_pinned_sim_state_snapshot()
        if snapshot is None:
            snapshot = self.get_latest_sim_state_snapshot()
        return None if snapshot is None else snapshot.sim_time_ms

    def send_low_command(
        self,
        cmd_q,
        cmd_dq,
        cmd_tau,
        dof_pos_latest=None,
        kp_override=None,
        kd_override=None,
    ) -> None:
        del dof_pos_latest
        q_target = np.asarray(cmd_q, dtype=np.float32).reshape(-1)
        dq_target = np.asarray(cmd_dq, dtype=np.float32).reshape(-1)
        tau_ff = np.asarray(cmd_tau, dtype=np.float32).reshape(-1)

        if self.no_action:
            kp = np.zeros_like(q_target, dtype=np.float32)
            kd = np.zeros_like(q_target, dtype=np.float32)
            tau_ff = np.zeros_like(q_target, dtype=np.float32)
        else:
            default_joint_kp, default_joint_kd = self._joint_gains_from_robot_config()
            kp = np.asarray(kp_override, dtype=np.float32).reshape(-1) if kp_override is not None else default_joint_kp
            kd = np.asarray(kd_override, dtype=np.float32).reshape(-1) if kd_override is not None else default_joint_kd
            kp *= float(self.kp_level)
            kd *= float(self.kd_level)

        control_snapshot = self.get_pinned_sim_state_snapshot()
        policy_sim_time_ms = (
            None if control_snapshot is None else float(control_snapshot.sim_time_ms)
        )
        episode_generation = (
            None if control_snapshot is None else int(control_snapshot.episode_generation)
        )
        self._sim_control_pub.publish(
            {
                "action": "lowcmd",
                "seq": int(self._lowcmd_seq),
                "policy_sim_time_ms": policy_sim_time_ms,
                "episode_generation": episode_generation,
                "q_target": q_target.tolist(),
                "dq_target": dq_target.tolist(),
                "tau_ff": tau_ff.tolist(),
                "kp": kp.tolist(),
                "kd": kd.tolist(),
            }
        )
        self._lowcmd_seq += 1

    def publish_actor_state(self, name: str, state) -> None:
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        self._sim_control_pub.publish(
            {
                "action": "actor_state",
                "name": str(name),
                "state": state_arr.tolist(),
            }
        )

    def publish_robot_root_state(self, state) -> None:
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        self._sim_control_pub.publish(
            {
                "action": "robot_root_state",
                "state": state_arr.tolist(),
            }
        )

    def publish_robot_dof_state(self, state) -> None:
        state_arr = np.asarray(state, dtype=np.float32)
        self._sim_control_pub.publish(
            {
                "action": "robot_dof_state",
                "state": state_arr.reshape(-1).tolist(),
            }
        )

    def process_joystick_input(self):
        return np.zeros((1, 3), dtype=np.float32)

    def get_joystick_key(self):
        return 0

    def get_joystick_msg(self):
        return None

    def close(self) -> None:
        self._sim_state_sub.close()
        self._sim_control_pub.close()
