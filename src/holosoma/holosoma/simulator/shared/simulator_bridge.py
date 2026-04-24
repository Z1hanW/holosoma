"""Simulator-agnostic bridge interface for robot control.

This module provides a unified interface for integrating robot SDK bridges
with different simulators (MuJoCo, IsaacGym, IsaacSim, etc.).
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from holosoma.bridge import BasicSdk2Bridge, create_sdk2py_bridge
from holosoma.config_types.simulator import BridgeConfig
from holosoma.utils.clock import ClockPub
from holosoma.utils.perception_obs import PerceptionObsPub, PerceptionObsShmPub
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
        self.perception_obs_pub: PerceptionObsPub | None = None
        self.perception_obs_shm_pub: PerceptionObsShmPub | None = None
        self.sim_control_sub: SimControlPull | None = None
        self._logged_perception_obs_publish = False
        self._use_zmq_lowcmd = bool(getattr(self.bridge_config, "use_zmq_lowcmd", False))
        self._latest_lowcmd_payload: dict | None = None
        self._received_external_active_command = False
        self._logged_first_command_summary = False
        self._logged_active_command_summaries = 0
        self._logged_default_pose_hold = False
        self._logged_initial_pose_hold = False
        self._default_hold_q: np.ndarray | None = None
        self._default_hold_kp: np.ndarray | None = None
        self._default_hold_kd: np.ndarray | None = None
        self._initial_hold_q: np.ndarray | None = None
        self._zmq_lowcmd_kp_scale = float(os.environ.get("HOLOSOMA_ZMQ_LOWCMD_KP_SCALE", "1.0") or "1.0")
        self._zmq_lowcmd_kd_scale = float(os.environ.get("HOLOSOMA_ZMQ_LOWCMD_KD_SCALE", "1.0") or "1.0")
        self._zmq_lowcmd_torque_limit_scale = float(
            os.environ.get("HOLOSOMA_ZMQ_LOWCMD_TORQUE_LIMIT_SCALE", "1.0") or "1.0"
        )
        self._sim_state_trace_path = os.environ.get("HOLOSOMA_SPLIT_SIM_STATE_TRACE_PATH", "").strip() or None
        self._sim_state_trace_handle = None

        # Initialize clock publisher for WBT motion synchronization
        self.clock_pub: ClockPub = ClockPub(port=self.bridge_config.clock_port)

        if self.bridge_config.interface is None and not self._use_zmq_lowcmd:
            interface = self._auto_detect_interface()
            logger.info(f"Auto-detected bridge interface '{interface}'")
            self.bridge_config = replace(self.bridge_config, interface=interface)

        if bridge_config.enabled:
            logger.info("Robot bridge is enabled, initializing...")
            if self._use_zmq_lowcmd:
                self._init_zmq_lowcmd_state()
                logger.info("Using split sim-control ZMQ lowcmd bridge instead of Unitree DDS")
                if (
                    self._zmq_lowcmd_kp_scale != 1.0
                    or self._zmq_lowcmd_kd_scale != 1.0
                    or self._zmq_lowcmd_torque_limit_scale != 1.0
                ):
                    logger.info(
                        "Applying ZMQ lowcmd MuJoCo scales: kp_scale={} kd_scale={} torque_limit_scale={}",
                        self._zmq_lowcmd_kp_scale,
                        self._zmq_lowcmd_kd_scale,
                        self._zmq_lowcmd_torque_limit_scale,
                    )
            else:
                self._init_robot_bridge()
            if self.bridge_config.publish_sim_state:
                self.sim_state_pub = SimStatePub(port=self.bridge_config.sim_state_port)
                self.sim_state_pub.start()
            if self.bridge_config.listen_control or self._use_zmq_lowcmd:
                self.sim_control_sub = SimControlPull(port=self.bridge_config.control_port)
                self.sim_control_sub.start()
            if self.bridge_config.publish_perception_obs:
                self.perception_obs_pub = PerceptionObsPub(port=self.bridge_config.perception_obs_port)
                self.perception_obs_pub.start()
            if bool(getattr(self.bridge_config, "publish_perception_obs_shm", False)):
                self.perception_obs_shm_pub = PerceptionObsShmPub(
                    name=getattr(self.bridge_config, "perception_obs_shm_name", "depth_img_shm")
                )
                self.perception_obs_shm_pub.start()
            # Start clock publisher for motion synchronization
            self.clock_pub.start()
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

    def _init_zmq_lowcmd_state(self) -> None:
        self._default_hold_q, self._default_hold_kp, self._default_hold_kd = self._build_default_pose_hold_targets()
        self._reset_zmq_lowcmd_runtime_state()

    def _reset_zmq_lowcmd_runtime_state(self) -> None:
        """Clear runtime lowcmd state so simulator resets do not reuse stale commands."""
        self._latest_lowcmd_payload = None
        self._received_external_active_command = False
        self._logged_first_command_summary = False
        self._logged_active_command_summaries = 0
        self._logged_default_pose_hold = False
        self._logged_initial_pose_hold = False
        self._initial_hold_q = None

    def _build_default_pose_hold_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        default_joint_angles = getattr(self.simulator.robot_config.init_state, "default_joint_angles", {}) or {}
        stiffness_dict = getattr(self.simulator.robot_config.control, "stiffness", {}) or {}
        damping_dict = getattr(self.simulator.robot_config.control, "damping", {}) or {}

        q = np.zeros(self.simulator.num_dof, dtype=np.float32)
        kp = np.zeros(self.simulator.num_dof, dtype=np.float32)
        kd = np.zeros(self.simulator.num_dof, dtype=np.float32)
        for sim_idx, dof_name in enumerate(self.simulator.dof_names):
            if dof_name in default_joint_angles:
                q[sim_idx] = float(default_joint_angles[dof_name])

            matches = [pattern for pattern in stiffness_dict if pattern in dof_name]
            if len(matches) == 1:
                pattern = matches[0]
                kp[sim_idx] = float(stiffness_dict[pattern])
                kd[sim_idx] = float(damping_dict[pattern])
        return q, kp, kd

    def _maybe_capture_initial_hold_targets(self) -> np.ndarray:
        if self._initial_hold_q is None:
            self._initial_hold_q = self.simulator.dof_pos[0].detach().cpu().numpy().astype(np.float32, copy=True)
        return self._initial_hold_q

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
            self._publish_perception_obs()
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
        self._publish_sim_state()
        self._publish_perception_obs()

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

    def _process_control_requests(self) -> None:
        if self.sim_control_sub is None:
            return

        for payload in self.sim_control_sub.drain():
            action = str(payload.get("action", "")).lower()
            if action == "reset":
                reason = str(payload.get("reason", "manual"))
                motion_init_mode = str(payload.get("motion_init_mode", "")).strip().lower().replace("-", "_")
                if motion_init_mode in {"raw_motion", "training_default_pose"}:
                    reset_states = getattr(self.simulator, "_motion_init_reset_states_by_mode", None)
                    if isinstance(reset_states, dict):
                        state = reset_states.get(motion_init_mode)
                        if isinstance(state, dict):
                            root_state = state.get("root_state")
                            dof_state = state.get("dof_state")
                            actor_states = state.get("actor_states", {})
                            if root_state is not None and dof_state is not None:
                                self.simulator._motion_init_reset_mode = motion_init_mode
                                self.simulator._motion_init_reset_root_state = root_state.detach().clone()
                                self.simulator._motion_init_reset_dof_state = dof_state.detach().clone()
                                self.simulator._motion_init_reset_actor_states = {
                                    str(name): tensor.detach().clone() for name, tensor in dict(actor_states).items()
                                }
                                logger.info("Selected motion-init reset mode from sim-control: {}", motion_init_mode)
                            else:
                                logger.warning(
                                    "Ignoring sim-control motion_init_mode={} because cached reset tensors are incomplete",
                                    motion_init_mode,
                                )
                        else:
                            logger.warning(
                                "Ignoring sim-control motion_init_mode={} because no cached reset state was found",
                                motion_init_mode,
                            )
                if hasattr(self.simulator, "_pending_reset"):
                    self.simulator._pending_reset = True
                if self._use_zmq_lowcmd:
                    self._reset_zmq_lowcmd_runtime_state()
                logger.info(
                    "Queued simulator reset from sim-control channel ({}, motion_init_mode={})",
                    reason,
                    motion_init_mode or "unchanged",
                )
                continue
            if action == "actor_state":
                actor_name = str(payload.get("name", "object") or "object")
                values = payload.get("state")
                try:
                    state_np = np.asarray(values, dtype=np.float32).reshape(1, -1)
                except Exception as exc:
                    logger.warning("Ignoring invalid actor_state payload for '{}': {}", actor_name, exc)
                    continue
                if state_np.shape[1] < 7:
                    logger.warning("Ignoring actor_state payload for '{}' with shape {}", actor_name, state_np.shape)
                    continue
                if state_np.shape[1] < 13:
                    padded = np.zeros((1, 13), dtype=np.float32)
                    padded[:, : state_np.shape[1]] = state_np
                    state_np = padded
                env_ids = torch.tensor([0], device=self.simulator.device, dtype=torch.long)
                state = torch.as_tensor(state_np[:, :13], device=self.simulator.device, dtype=torch.float32)
                self.simulator.set_actor_states([actor_name], env_ids, state)
                continue
            if action == "robot_root_state":
                values = payload.get("state")
                try:
                    state_np = np.asarray(values, dtype=np.float32).reshape(1, -1)
                except Exception as exc:
                    logger.warning("Ignoring invalid robot_root_state payload: {}", exc)
                    continue
                if state_np.shape[1] < 7:
                    logger.warning("Ignoring robot_root_state payload with shape {}", state_np.shape)
                    continue
                if state_np.shape[1] < 13:
                    padded = np.zeros((1, 13), dtype=np.float32)
                    padded[:, : state_np.shape[1]] = state_np
                    state_np = padded
                env_ids = torch.tensor([0], device=self.simulator.device, dtype=torch.long)
                state = torch.as_tensor(state_np[:, :13], device=self.simulator.device, dtype=torch.float32)
                self.simulator.set_actor_root_state_tensor_robots(env_ids, state)
                continue
            if action == "robot_dof_state":
                values = payload.get("state")
                try:
                    state_np = np.asarray(values, dtype=np.float32)
                except Exception as exc:
                    logger.warning("Ignoring invalid robot_dof_state payload: {}", exc)
                    continue
                try:
                    state_np = state_np.reshape(1, self.simulator.num_dof, 2)
                except ValueError:
                    try:
                        flat = state_np.reshape(-1)
                        if flat.shape[0] == self.simulator.num_dof:
                            padded = np.zeros((1, self.simulator.num_dof, 2), dtype=np.float32)
                            padded[0, :, 0] = flat
                            state_np = padded
                        else:
                            raise
                    except ValueError:
                        logger.warning("Ignoring robot_dof_state payload with shape {}", state_np.shape)
                        continue
                env_ids = torch.tensor([0], device=self.simulator.device, dtype=torch.long)
                state = torch.as_tensor(state_np, device=self.simulator.device, dtype=torch.float32)
                self.simulator.set_dof_state_tensor_robots(env_ids, state)
                continue
            if action == "lowcmd" and self._use_zmq_lowcmd:
                self._latest_lowcmd_payload = payload
                if self._payload_is_active(payload):
                    self._received_external_active_command = True
                    if bool(getattr(self.bridge_config, "log_first_command_summary", False)) and not self._logged_first_command_summary:
                        logger.info(
                            "Received first ZMQ lowcmd: kp_max={:.3f}, kd_max={:.3f}, tau_max={:.3f}",
                            float(np.max(np.abs(self._payload_array(payload, "kp")))),
                            float(np.max(np.abs(self._payload_array(payload, "kd")))),
                            float(np.max(np.abs(self._payload_array(payload, "tau_ff")))),
                        )
                        self._logged_first_command_summary = True

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
        kp_t = kp_t * float(self._zmq_lowcmd_kp_scale)
        kd_t = kd_t * float(self._zmq_lowcmd_kd_scale)
        torque_limit = torque_limit * float(self._zmq_lowcmd_torque_limit_scale)

        torques = tau + kp_t * (q_des - q_actual) + kd_t * (dq_des - dq_actual)
        torques = torch.clamp(torques, -torque_limit, torque_limit)
        return torques.detach().cpu().numpy().astype(np.float32, copy=False)

    def _compute_zmq_lowcmd_torques(self) -> np.ndarray | None:
        payload = self._latest_lowcmd_payload
        cmd_is_active = self._payload_is_active(payload)
        hold_initial = bool(getattr(self.bridge_config, "hold_initial_pose_until_first_command", False))
        hold_default = bool(getattr(self.bridge_config, "hold_default_pose_until_first_command", False))

        if not self._received_external_active_command and hold_initial and not cmd_is_active:
            if not self._logged_initial_pose_hold:
                logger.info("Holding simulator initial pose until first external ZMQ lowcmd arrives")
                self._logged_initial_pose_hold = True
            return self._compute_pd_torques(
                tau_ff=np.zeros(self.simulator.num_dof, dtype=np.float32),
                kp=self._default_hold_kp if self._default_hold_kp is not None else np.zeros(self.simulator.num_dof, dtype=np.float32),
                kd=self._default_hold_kd if self._default_hold_kd is not None else np.zeros(self.simulator.num_dof, dtype=np.float32),
                q_target=self._maybe_capture_initial_hold_targets(),
                dq_target=np.zeros(self.simulator.num_dof, dtype=np.float32),
            )
        if not self._received_external_active_command and hold_default and not cmd_is_active:
            if not self._logged_default_pose_hold:
                logger.info("Holding simulator default pose until first external ZMQ lowcmd arrives")
                self._logged_default_pose_hold = True
            return self._compute_pd_torques(
                tau_ff=np.zeros(self.simulator.num_dof, dtype=np.float32),
                kp=self._default_hold_kp if self._default_hold_kp is not None else np.zeros(self.simulator.num_dof, dtype=np.float32),
                kd=self._default_hold_kd if self._default_hold_kd is not None else np.zeros(self.simulator.num_dof, dtype=np.float32),
                q_target=self._default_hold_q if self._default_hold_q is not None else np.zeros(self.simulator.num_dof, dtype=np.float32),
                dq_target=np.zeros(self.simulator.num_dof, dtype=np.float32),
            )
        if payload is None:
            return np.zeros(self.simulator.num_dof, dtype=np.float32)

        tau_ff = self._payload_array(payload, "tau_ff")
        kp = self._payload_array(payload, "kp")
        kd = self._payload_array(payload, "kd")
        q_target = self._payload_array(payload, "q_target")
        dq_target = self._payload_array(payload, "dq_target")
        torques = self._compute_pd_torques(
            tau_ff=tau_ff,
            kp=kp,
            kd=kd,
            q_target=q_target,
            dq_target=dq_target,
        )
        if (
            bool(getattr(self.bridge_config, "log_first_command_summary", False))
            and self._payload_is_active(payload)
            and self._logged_active_command_summaries < 5
        ):
            q_actual = self.simulator.dof_pos[0].detach().cpu().numpy()
            logger.info(
                "Active ZMQ lowcmd summary #{:d}: |q_target-q_actual|max={:.4f}, |tau|max={:.4f}, "
                "q_target[:6]={}, q_actual[:6]={}, kp[:6]={}, kd[:6]={}, tau_ff[:6]={}",
                self._logged_active_command_summaries + 1,
                float(np.max(np.abs(q_target - q_actual))),
                float(np.max(np.abs(torques))),
                np.array2string(q_target[:6], precision=4),
                np.array2string(q_actual[:6], precision=4),
                np.array2string(kp[:6], precision=4),
                np.array2string(kd[:6], precision=4),
                np.array2string(tau_ff[:6], precision=4),
            )
            self._logged_active_command_summaries += 1
            self._logged_first_command_summary = self._logged_active_command_summaries >= 5
        return torques

    def _publish_sim_state(self) -> None:
        if self.sim_state_pub is None:
            return

        try:
            robot_root_state = self.simulator.robot_root_states[0].detach().cpu().tolist()
            robot_dof_pos = self.simulator.dof_pos[0].detach().cpu().tolist()
            robot_dof_vel = self.simulator.dof_vel[0].detach().cpu().tolist()
            actor_states: dict[str, list[float]] = {}
            env_ids = torch.tensor([0], device=self.simulator.device, dtype=torch.long)

            actor_metadata = getattr(self.simulator, "_actor_root_metadata", {})
            if isinstance(actor_metadata, dict) and actor_metadata:
                actor_names = [name for name in actor_metadata if name != "robot"]
            else:
                actor_names = list(getattr(self.simulator, "_object_urdf_by_name", {}).keys())

            for name in actor_names:
                try:
                    actor_state = self.simulator.get_actor_states([name], env_ids)
                    if actor_state.numel() == 0:
                        continue
                    actor_states[name] = actor_state[0].detach().cpu().tolist()
                except Exception as exc:  # pragma: no cover - best effort side-channel
                    logger.debug("Skipping sim-state actor '{}': {}", name, exc)

            payload = {
                "sim_time_ms": int(self.simulator.time() * 1000.0),
                "robot_root_state": robot_root_state,
                "robot_dof_pos": robot_dof_pos,
                "robot_dof_vel": robot_dof_vel,
                "actors": actor_states,
            }
            extra_payload_provider = getattr(self.simulator, "_get_split_sim_state_extra_payload", None)
            if callable(extra_payload_provider):
                extra_payload = extra_payload_provider()
                if isinstance(extra_payload, dict):
                    payload.update(extra_payload)

            self.sim_state_pub.publish(payload)
            if self._sim_state_trace_path is not None:
                if self._sim_state_trace_handle is None:
                    trace_dir = os.path.dirname(self._sim_state_trace_path)
                    if trace_dir:
                        os.makedirs(trace_dir, exist_ok=True)
                    self._sim_state_trace_handle = open(self._sim_state_trace_path, "a", encoding="utf-8")
                self._sim_state_trace_handle.write(json.dumps(payload) + "\n")
                self._sim_state_trace_handle.flush()
        except Exception as exc:  # pragma: no cover - best effort side-channel
            logger.debug("Failed to publish sim state: {}", exc)

    def _publish_perception_obs(self) -> None:
        if self.perception_obs_pub is None and self.perception_obs_shm_pub is None:
            return

        provider = getattr(self.simulator, "_split_sim_perception_provider", None)
        if not callable(provider):
            return

        try:
            perception_obs = provider()
            if perception_obs is None:
                return
            if not self._logged_perception_obs_publish:
                logger.info("Publishing split sim perception obs with {} values", len(perception_obs))
                self._logged_perception_obs_publish = True
            if self.perception_obs_pub is not None:
                self.perception_obs_pub.publish(
                    {
                        "sim_time_ms": int(self.simulator.time() * 1000.0),
                        "perception_obs": perception_obs,
                    }
                )
            if self.perception_obs_shm_pub is not None:
                self.perception_obs_shm_pub.publish(perception_obs)
        except Exception as exc:  # pragma: no cover - best effort side-channel
            logger.warning("Failed to publish perception obs: {}", exc)

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
