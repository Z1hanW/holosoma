"""Simulator-agnostic bridge interface for robot control.

This module provides a unified interface for integrating robot SDK bridges
with different simulators (MuJoCo, IsaacGym, IsaacSim, etc.).
"""

from __future__ import annotations

import sys
from dataclasses import replace
from typing import TYPE_CHECKING

from loguru import logger

from holosoma.bridge import BasicSdk2Bridge, create_sdk2py_bridge
from holosoma.config_types.simulator import BridgeConfig
from holosoma.utils.clock import ClockPub
from holosoma.utils.perception_obs import PerceptionObsPub
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
        self.sim_control_sub: SimControlPull | None = None
        self._logged_perception_obs_publish = False

        # Initialize clock publisher for WBT motion synchronization
        self.clock_pub: ClockPub = ClockPub(port=self.bridge_config.clock_port)

        if self.bridge_config.interface is None:
            interface = self._auto_detect_interface()
            logger.info(f"Auto-detected bridge interface '{interface}'")
            self.bridge_config = replace(self.bridge_config, interface=interface)

        if bridge_config.enabled:
            logger.info("Robot bridge is enabled, initializing...")
            self._init_robot_bridge()
            if self.bridge_config.publish_sim_state:
                self.sim_state_pub = SimStatePub(port=self.bridge_config.sim_state_port)
                self.sim_state_pub.start()
            if self.bridge_config.listen_control:
                self.sim_control_sub = SimControlPull(port=self.bridge_config.control_port)
                self.sim_control_sub.start()
            if self.bridge_config.publish_perception_obs:
                self.perception_obs_pub = PerceptionObsPub(port=self.bridge_config.perception_obs_port)
                self.perception_obs_pub.start()
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

    def step(self):
        """Execute bridge step during simulation.

        This method should be called during each simulation step when the bridge
        is enabled. It handles:
        - Publishing robot state to SDK
        - Publishing simulation clock for motion synchronization
        - Processing joystick input (if enabled)
        - Computing and applying torques from SDK commands
        """
        if not self.robot_bridge:
            return

        self._process_control_requests()

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
            if str(payload.get("action", "")).lower() != "reset":
                continue

            reason = str(payload.get("reason", "manual"))
            if hasattr(self.simulator, "_pending_reset"):
                self.simulator._pending_reset = True
            logger.info("Queued simulator reset from sim-control channel ({})", reason)

    def _publish_sim_state(self) -> None:
        if self.sim_state_pub is None:
            return

        try:
            robot_root_state = self.simulator.robot_root_states[0].detach().cpu().tolist()
            actor_states: dict[str, list[float]] = {}
            env_ids = torch.tensor([0], device=self.simulator.device, dtype=torch.long)

            actor_metadata = getattr(self.simulator, "_actor_root_metadata", {})
            if isinstance(actor_metadata, dict) and actor_metadata:
                actor_names = [name for name in actor_metadata if name != "robot"]
            else:
                actor_names = list(getattr(self.simulator, "_object_urdf_by_name", {}).keys())

            for name in actor_names:
                try:
                    indices = self.simulator.get_actor_indices(name, env_ids=env_ids)
                    if indices.numel() == 0:
                        continue
                    actor_states[name] = self.simulator.all_root_states[indices[0]].detach().cpu().tolist()
                except Exception as exc:  # pragma: no cover - best effort side-channel
                    logger.debug("Skipping sim-state actor '{}': {}", name, exc)

            payload = {
                "sim_time_ms": int(self.simulator.time() * 1000.0),
                "robot_root_state": robot_root_state,
                "actors": actor_states,
            }
            extra_payload_provider = getattr(self.simulator, "_get_split_sim_state_extra_payload", None)
            if callable(extra_payload_provider):
                extra_payload = extra_payload_provider()
                if isinstance(extra_payload, dict):
                    payload.update(extra_payload)

            self.sim_state_pub.publish(payload)
        except Exception as exc:  # pragma: no cover - best effort side-channel
            logger.debug("Failed to publish sim state: {}", exc)

    def _publish_perception_obs(self) -> None:
        if self.perception_obs_pub is None:
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
            self.perception_obs_pub.publish(
                {
                    "sim_time_ms": int(self.simulator.time() * 1000.0),
                    "perception_obs": perception_obs,
                }
            )
        except Exception as exc:  # pragma: no cover - best effort side-channel
            logger.warning("Failed to publish perception obs: {}", exc)

    def has_received_external_active_command(self) -> bool:
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
        return self.bridge_config.enabled and self.robot_bridge is not None

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
        }
