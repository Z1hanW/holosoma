"""Simulator-agnostic bridge interface for robot control.

This module provides a unified interface for integrating robot SDK bridges
with different simulators (MuJoCo, IsaacGym, IsaacSim, etc.).
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from holosoma.bridge import BasicSdk2Bridge, create_sdk2py_bridge
from holosoma.config_types.simulator import BridgeConfig
from holosoma.utils.clock import ClockPub
from holosoma.utils.sim_control import SimControlPull
from holosoma.utils.sim_state import SimStatePub
from holosoma.utils.safe_torch_import import torch

if TYPE_CHECKING:
    from holosoma.simulator.base_simulator.base_simulator import BaseSimulator


class SimulatorBridge:
    """Simulator-agnostic bridge interface for robot control.

    This class is intended to provide an interface between robot SDK bridges and simulators,
    allowing robot control to be added to any simulator without breaking existing
    functionality.

    Currently it is tested with MuJoCo-only via the base bridge interface.

    """

    def __init__(self, simulator: BaseSimulator, bridge_config: BridgeConfig):
        """Initialize the simulator bridge.

        Initializes the bridge system for robot SDK integration, including:
        - Robot SDK bridge for state publishing and command receiving
        - Clock publisher for motion synchronization (WBT policies)
        - Optional joystick/gamepad support

        Parameters
        ----------
        simulator : BaseSimulator
            The simulator instance to integrate with
        bridge_config : BridgeConfig
            Configuration for the bridge system
        """
        self.simulator: BaseSimulator = simulator
        self.bridge_config: BridgeConfig = bridge_config
        self.robot_bridge: BasicSdk2Bridge | None = None
        self.sim_state_pub: SimStatePub | None = None
        self.sim_control_sub: SimControlPull | None = None
        self._use_zmq_lowcmd = bool(getattr(self.bridge_config, "use_zmq_lowcmd", False))
        self._latest_lowcmd_payload: dict | None = None
        self._received_external_active_command = False
        self._logged_first_zmq_command = False
        self._logged_active_zmq_command_summaries = 0
        self._last_zmq_torque_preview: list[float] | None = None
        self._debug_lowcmd_tracking = os.environ.get("HOLOSOMA_MJ_DEBUG_LIFT_TELEMETRY", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self._debug_lowcmd_next_time = 0.0
        self._last_pd_stats: dict[str, object] | None = None

        # Initialize clock publisher for WBT motion synchronization
        self.clock_pub: ClockPub = ClockPub()

        if self.bridge_config.interface is None and not self._use_zmq_lowcmd:
            interface = self._auto_detect_interface()
            logger.info(f"Auto-detected bridge interface '{interface}'")
            self.bridge_config = replace(self.bridge_config, interface=interface)

        if bridge_config.enabled:
            logger.info("Robot bridge is enabled, initializing...")
            if self._use_zmq_lowcmd:
                logger.info("Using split sim-control ZMQ lowcmd bridge instead of Unitree DDS")
            else:
                self._init_robot_bridge()
            # Start clock publisher for motion synchronization
            self.clock_pub.start()
            publish_sim_state = bool(getattr(self.bridge_config, "publish_sim_state", False)) or os.environ.get(
                "HOLOSOMA_PUBLISH_SIM_STATE", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            if publish_sim_state:
                port = int(os.environ.get("SIM_STATE_PORT", str(getattr(self.bridge_config, "sim_state_port", 5557))) or "5557")
                self.sim_state_pub = SimStatePub(port=port)
                self.sim_state_pub.start()
            if bool(getattr(self.bridge_config, "listen_control", False)) or self._use_zmq_lowcmd:
                port = int(os.environ.get("SIM_CONTROL_PORT", str(getattr(self.bridge_config, "control_port", 5559))) or "5559")
                self.sim_control_sub = SimControlPull(port=port)
                self.sim_control_sub.start()
            logger.info("Clock publisher initialized for motion synchronization")
        else:
            # We don't support runtime toggling on/off
            logger.info("Robot bridge disabled")

    def _init_robot_bridge(self):
        """Initialize the robot bridge using the copied factory function."""
        try:
            # Create robot bridge using the factory function from holosoma.bridge
            self.robot_bridge = create_sdk2py_bridge(self.simulator, self.simulator.robot_config, self.bridge_config)
            logger.info(
                f"Robot bridge initialized successfully with SDK type: {self.simulator.robot_config.bridge.sdk_type}"
            )

            # Setup joystick if enabled
            if self.bridge_config.use_joystick:
                self._setup_joystick()

        except Exception as e:
            logger.error(f"Failed to initialize robot bridge: {e}")
            raise

    def _setup_joystick(self):
        """Setup joystick/gamepad for robot control."""
        try:
            self.robot_bridge.setup_joystick(
                device_id=self.bridge_config.joystick_device, js_type=self.bridge_config.joystick_type
            )
            logger.info(
                f"Joystick initialized: device={self.bridge_config.joystick_device}, "
                f"type={self.bridge_config.joystick_type}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize joystick: {e}") from e

    def _auto_detect_interface(self):
        # Auto-detect interface based on platform (like holosoma_inference)
        if sys.platform == "linux":
            return "lo"
        if sys.platform == "darwin":
            return "lo0"
        raise NotImplementedError("Only support Linux and MacOS for Unitree SDK.")

    def step(self):
        """Execute bridge step during simulation.

        This method should be called during each simulation step when the bridge
        is enabled. It handles:
        - Publishing robot state to SDK
        - Publishing simulation clock for motion synchronization
        - Processing joystick input (if enabled)
        - Computing and applying torques from SDK commands
        """
        if not self.bridge_config.enabled:
            return

        self._process_control_requests()

        if self._use_zmq_lowcmd:
            self._publish_sim_state()
            torques = self._compute_zmq_lowcmd_torques()
            if torques is not None:
                torques_tensor = torch.from_numpy(torques).to(device=self.simulator.device, dtype=torch.float32)
                self.simulator.apply_torques_at_dof(torques_tensor)
            self.clock_pub.publish(self.simulator.time())
            return

        if not self.robot_bridge:
            return

        # Publish robot state to SDK
        self.robot_bridge.publish_low_state()

        # Handle joystick input if available
        if hasattr(self.robot_bridge, "joystick") and self.robot_bridge.joystick:
            self.robot_bridge.publish_wireless_controller()
            logger.debug("Wireless controller input published")

        # Read incoming commands from DDS
        self.robot_bridge.low_cmd_handler()

        # Compute torques based on received commands
        self.robot_bridge.compute_torques()

        # Apply torques to simulator
        # (for now: convert to/from tensor for unified interface, which is unnecessary for mujoco...)
        torques_tensor = torch.from_numpy(self.robot_bridge.torques).to(
            device=self.simulator.device, dtype=torch.float32
        )
        self.simulator.apply_torques_at_dof(torques_tensor)

        # Publish simulation clock for e.g, WBT policies
        sim_time = self.simulator.time()
        self.clock_pub.publish(sim_time)
        self._publish_sim_state()

    def _process_control_requests(self) -> None:
        if self.sim_control_sub is None:
            return

        for payload in self.sim_control_sub.drain():
            action = str(payload.get("action", "")).lower()
            if action == "reset":
                if hasattr(self.simulator, "_reset_requested"):
                    self.simulator._reset_requested = True
                self.reset_command_state()
                logger.info("Queued simulator reset from sim-control channel ({})", payload.get("reason", "manual"))
                continue
            if action == "lowcmd" and self._use_zmq_lowcmd:
                self._latest_lowcmd_payload = payload
                if self._payload_is_active(payload):
                    if not self._received_external_active_command:
                        logger.info(
                            "Received first ZMQ lowcmd: kp_max={:.3f}, kd_max={:.3f}, q0={:.3f}",
                            float(np.max(np.abs(self._payload_array(payload, "kp")))),
                            float(np.max(np.abs(self._payload_array(payload, "kd")))),
                            float(self._payload_array(payload, "q_target")[0]),
                        )
                        self._logged_first_zmq_command = True
                    self._received_external_active_command = True

    def _payload_array(self, payload: dict | None, key: str) -> np.ndarray:
        if not isinstance(payload, dict):
            return np.zeros(self.simulator.num_dof, dtype=np.float32)
        values = payload.get(key)
        if values is None:
            return np.zeros(self.simulator.num_dof, dtype=np.float32)
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
        if arr.shape[0] < self.simulator.num_dof:
            padded = np.zeros(self.simulator.num_dof, dtype=np.float32)
            padded[: arr.shape[0]] = arr
            return padded
        return arr[: self.simulator.num_dof]

    def _payload_is_active(self, payload: dict | None) -> bool:
        if payload is None:
            return False
        for key in ("kp", "kd", "tau_ff", "dq_target"):
            if np.any(np.abs(self._payload_array(payload, key)) > 1e-6):
                return True
        return False

    def _compute_zmq_lowcmd_torques(self) -> np.ndarray | None:
        payload = self._latest_lowcmd_payload
        cmd_is_active = self._payload_is_active(payload)
        hold_initial = bool(getattr(self.bridge_config, "hold_initial_pose_until_first_command", False))
        hold_default = bool(getattr(self.bridge_config, "hold_default_pose_until_first_command", False))

        if not self._received_external_active_command and hold_initial and not cmd_is_active:
            return self._compute_pd_torques(
                tau_ff=np.zeros(self.simulator.num_dof, dtype=np.float32),
                kp=self._default_hold_kp(),
                kd=self._default_hold_kd(),
                q_target=self.simulator.dof_pos[0].detach().cpu().numpy().astype(np.float32, copy=True),
                dq_target=np.zeros(self.simulator.num_dof, dtype=np.float32),
            )
        if not self._received_external_active_command and hold_default and not cmd_is_active:
            return self._compute_pd_torques(
                tau_ff=np.zeros(self.simulator.num_dof, dtype=np.float32),
                kp=self._default_hold_kp(),
                kd=self._default_hold_kd(),
                q_target=self._default_hold_q(),
                dq_target=np.zeros(self.simulator.num_dof, dtype=np.float32),
            )
        if payload is None:
            return np.zeros(self.simulator.num_dof, dtype=np.float32)

        torques = self._compute_pd_torques(
            tau_ff=self._payload_array(payload, "tau_ff"),
            kp=self._payload_array(payload, "kp"),
            kd=self._payload_array(payload, "kd"),
            q_target=self._payload_array(payload, "q_target"),
            dq_target=self._payload_array(payload, "dq_target"),
        )
        self._last_zmq_torque_preview = torques[:8].astype(float).tolist()
        self._log_lowcmd_debug(payload)
        if (
            bool(getattr(self.bridge_config, "log_first_command_summary", False))
            and self._payload_is_active(payload)
            and self._logged_active_zmq_command_summaries < 5
        ):
            q_target = self._payload_array(payload, "q_target")
            q_actual = self.simulator.dof_pos[0].detach().cpu().numpy()
            logger.info(
                "Active ZMQ lowcmd summary #{:d}: |q_target-q_actual|max={:.4f}, |tau|max={:.4f}",
                self._logged_active_zmq_command_summaries + 1,
                float(np.max(np.abs(q_target - q_actual))),
                float(np.max(np.abs(torques))),
            )
            self._logged_active_zmq_command_summaries += 1
        return torques

    def _compute_pd_torques(
        self,
        *,
        tau_ff: np.ndarray,
        kp: np.ndarray,
        kd: np.ndarray,
        q_target: np.ndarray,
        dq_target: np.ndarray,
    ) -> np.ndarray:
        q_actual = self.simulator.dof_pos[0]
        dq_actual = self.simulator.dof_vel[0]
        device = q_actual.device
        tau = torch.as_tensor(tau_ff, device=device, dtype=q_actual.dtype)
        kp_t = torch.as_tensor(kp, device=device, dtype=q_actual.dtype)
        kd_t = torch.as_tensor(kd, device=device, dtype=q_actual.dtype)
        q_des = torch.as_tensor(q_target, device=device, dtype=q_actual.dtype)
        dq_des = torch.as_tensor(dq_target, device=device, dtype=q_actual.dtype)
        torque_limit = torch.as_tensor(
            self.simulator.robot_config.dof_effort_limit_list,
            device=device,
            dtype=q_actual.dtype,
        )
        torques = tau + kp_t * (q_des - q_actual) + kd_t * (dq_des - dq_actual)
        clipped_torques = torch.clamp(torques, -torque_limit, torque_limit)
        if self._debug_lowcmd_tracking:
            self._last_pd_stats = self._build_pd_stats(
                q_des=q_des,
                q_actual=q_actual,
                dq_des=dq_des,
                dq_actual=dq_actual,
                raw_torques=torques,
                clipped_torques=clipped_torques,
                torque_limit=torque_limit,
            )
        return clipped_torques.detach().cpu().numpy().astype(np.float32, copy=False)

    def _build_pd_stats(
        self,
        *,
        q_des,
        q_actual,
        dq_des,
        dq_actual,
        raw_torques,
        clipped_torques,
        torque_limit,
    ) -> dict[str, object]:
        q_des_np = q_des.detach().cpu().numpy()
        q_actual_np = q_actual.detach().cpu().numpy()
        dq_des_np = dq_des.detach().cpu().numpy()
        dq_actual_np = dq_actual.detach().cpu().numpy()
        raw_np = raw_torques.detach().cpu().numpy()
        clipped_np = clipped_torques.detach().cpu().numpy()
        limit_np = torque_limit.detach().cpu().numpy()

        q_err = q_des_np - q_actual_np
        dq_err = dq_des_np - dq_actual_np
        limit_margin = np.maximum(limit_np - 1e-5, 0.0)
        saturated = np.abs(raw_np) >= limit_margin
        qerr_idx = int(np.argmax(np.abs(q_err))) if q_err.size else -1
        sat_indices = np.flatnonzero(saturated)
        if sat_indices.size:
            top_sat_indices = sat_indices[np.argsort(-np.abs(raw_np[sat_indices]))[:6]]
            sat_joints = [self.simulator.dof_names[int(idx)] for idx in top_sat_indices]
        else:
            sat_joints = []

        return {
            "q_target_absmax": float(np.max(np.abs(q_des_np))) if q_des_np.size else 0.0,
            "q_actual_absmax": float(np.max(np.abs(q_actual_np))) if q_actual_np.size else 0.0,
            "q_error_max": float(np.max(np.abs(q_err))) if q_err.size else 0.0,
            "dq_error_max": float(np.max(np.abs(dq_err))) if dq_err.size else 0.0,
            "raw_torque_absmax": float(np.max(np.abs(raw_np))) if raw_np.size else 0.0,
            "torque_absmax": float(np.max(np.abs(clipped_np))) if clipped_np.size else 0.0,
            "saturated_count": int(np.count_nonzero(saturated)),
            "dof_count": int(raw_np.size),
            "sat_joints": sat_joints,
            "qerr_joint": None if qerr_idx < 0 else self.simulator.dof_names[qerr_idx],
            "qerr_target": 0.0 if qerr_idx < 0 else float(q_des_np[qerr_idx]),
            "qerr_actual": 0.0 if qerr_idx < 0 else float(q_actual_np[qerr_idx]),
            "qerr_value": 0.0 if qerr_idx < 0 else float(q_err[qerr_idx]),
        }

    def _log_lowcmd_debug(self, payload: dict | None) -> None:
        if not self._debug_lowcmd_tracking or not self._payload_is_active(payload):
            return
        sim_time = float(self.simulator.time())
        if sim_time + 1e-9 < self._debug_lowcmd_next_time:
            return
        stats = self._last_pd_stats
        if not isinstance(stats, dict):
            return
        logger.info(
            "LowCmdTelemetry t={:.2f} q_target_absmax={:.4f} q_actual_absmax={:.4f} "
            "qerr_max={:.4f} dqerr_max={:.4f} raw_torque_absmax={:.3f} torque_absmax={:.3f} "
            "sat={}/{} sat_joints={} qerr_joint={} qerr_target={:.4f} qerr_actual={:.4f} qerr_value={:.4f}",
            sim_time,
            float(stats["q_target_absmax"]),
            float(stats["q_actual_absmax"]),
            float(stats["q_error_max"]),
            float(stats["dq_error_max"]),
            float(stats["raw_torque_absmax"]),
            float(stats["torque_absmax"]),
            int(stats["saturated_count"]),
            int(stats["dof_count"]),
            ",".join(stats["sat_joints"]) if stats["sat_joints"] else "none",
            stats["qerr_joint"] or "none",
            float(stats["qerr_target"]),
            float(stats["qerr_actual"]),
            float(stats["qerr_value"]),
        )
        self._debug_lowcmd_next_time = sim_time + 0.5

    def _default_hold_q(self) -> np.ndarray:
        default_joint_angles = getattr(self.simulator.robot_config.init_state, "default_joint_angles", {}) or {}
        q = np.zeros(self.simulator.num_dof, dtype=np.float32)
        for idx, dof_name in enumerate(self.simulator.dof_names):
            if dof_name in default_joint_angles:
                q[idx] = float(default_joint_angles[dof_name])
        return q

    def _default_hold_kp(self) -> np.ndarray:
        stiffness_dict = getattr(self.simulator.robot_config.control, "stiffness", {}) or {}
        kp = np.zeros(self.simulator.num_dof, dtype=np.float32)
        for idx, dof_name in enumerate(self.simulator.dof_names):
            matches = [pattern for pattern in stiffness_dict if pattern in dof_name]
            if len(matches) == 1:
                kp[idx] = float(stiffness_dict[matches[0]])
        return kp

    def _default_hold_kd(self) -> np.ndarray:
        damping_dict = getattr(self.simulator.robot_config.control, "damping", {}) or {}
        kd = np.zeros(self.simulator.num_dof, dtype=np.float32)
        for idx, dof_name in enumerate(self.simulator.dof_names):
            matches = [pattern for pattern in damping_dict if pattern in dof_name]
            if len(matches) == 1:
                kd[idx] = float(damping_dict[matches[0]])
        return kd

    def _publish_sim_state(self) -> None:
        if self.sim_state_pub is None:
            return

        try:
            payload = {
                "sim_time_ms": int(round(self.simulator.time() * 1000.0)),
                "robot_root_state": self.simulator.robot_root_states[0].detach().cpu().tolist(),
                "robot_dof_pos": self.simulator.dof_pos[0].detach().cpu().tolist(),
                "robot_dof_vel": self.simulator.dof_vel[0].detach().cpu().tolist(),
                "robot_dof_names": list(getattr(self.simulator, "dof_names", [])),
            }
            if self._use_zmq_lowcmd:
                latest = self._latest_lowcmd_payload
                payload["lowcmd"] = {
                    "seq": None if latest is None else latest.get("seq"),
                    "policy_sim_time_ms": None if latest is None else latest.get("policy_sim_time_ms"),
                    "q_target_first": self._payload_array(latest, "q_target")[:8].astype(float).tolist(),
                    "kp_first": self._payload_array(latest, "kp")[:8].astype(float).tolist(),
                    "kd_first": self._payload_array(latest, "kd")[:8].astype(float).tolist(),
                    "torque_first": self._last_zmq_torque_preview,
                }
            extra_payload_provider = getattr(self.simulator, "_get_split_sim_state_extra_payload", None)
            if callable(extra_payload_provider):
                extra_payload = extra_payload_provider()
                if isinstance(extra_payload, dict):
                    payload.update(extra_payload)
            self.sim_state_pub.publish(payload)
        except Exception as exc:
            logger.debug("Failed to publish sim state: {}", exc)

    def reset_command_state(self) -> None:
        self._latest_lowcmd_payload = None
        self._received_external_active_command = False
        self._logged_first_zmq_command = False
        self._logged_active_zmq_command_summaries = 0
        self._last_zmq_torque_preview = None
        if self.robot_bridge is not None and hasattr(self.robot_bridge, "reset_command_state"):
            self.robot_bridge.reset_command_state()

    def has_received_external_active_command(self) -> bool:
        if self._use_zmq_lowcmd:
            return bool(self._received_external_active_command)
        if self.robot_bridge is None:
            return False
        return bool(getattr(self.robot_bridge, "received_external_active_command", False))

    def is_enabled(self) -> bool:
        """Check if the bridge is enabled and functional.

        Returns
        -------
        bool
            True if bridge is enabled and robot_bridge is initialized
        """
        return self.bridge_config.enabled and (self._use_zmq_lowcmd or self.robot_bridge is not None)

    def get_bridge_info(self) -> dict:
        """Get information about the current bridge configuration.

        Returns
        -------
        dict
            Dictionary containing bridge status and configuration info
        """
        return {
            "enabled": self.bridge_config.enabled,
            "sdk_type": self.simulator.robot_config.bridge.sdk_type if self.bridge_config.enabled else None,
            "robot_bridge_initialized": self.robot_bridge is not None,
            "has_joystick": self.robot_bridge is not None and self.robot_bridge.joystick is not None,
            "use_zmq_lowcmd": self._use_zmq_lowcmd,
        }
