from __future__ import annotations

import hashlib
import itertools
import json
import os
import stat
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
from loguru import logger
from termcolor import colored

try:
    import netifaces as ni
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    ni = None

try:
    from sshkeyboard import listen_keyboard
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    listen_keyboard = None

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.config.config_types.robot import RobotConfig
from holosoma_inference.sdk.interface_wrapper import InterfaceWrapper
from holosoma_inference.sdk.zmq_interface_wrapper import ZmqSimInterfaceWrapper
from holosoma_inference.utils.latency import LatencyTracker
from holosoma_inference.utils.embedded_motion_timeline import (
    validate_embedded_motion_timeline_model,
)
from holosoma_inference.utils.contact_sidecar_contract import (
    embedded_contact_sidecar_contract_from_metadata,
)
from holosoma_inference.utils.button_window_contract import (
    embedded_button_window_contract_from_metadata,
)
from holosoma_inference.utils.math.quat import quat_rotate_inverse
from holosoma_inference.utils.perception_obs import PerceptionObsShmSub, PerceptionObsSub
from holosoma_inference.utils.policy_contract import (
    perception_observation_contract_sha256_from_metadata,
    recurrent_policy_contract_from_metadata,
    validate_onnx_policy_contract,
)
from holosoma_inference.utils.rate import RateLimiter
from holosoma_inference.utils.sim_control import PolicyControlPull
from holosoma_inference.utils.wandb import load_checkpoint


class BasePolicy:
    """
    Base policy class for Holosoma deployment on humanoid robots.

    Supports both simulation and real robot deployment with keyboard/joystick controls.
    """

    def __init__(self, config: InferenceConfig):
        """Initialize the base policy with configuration and model."""
        self.config = config
        # Initialize robot config
        self._init_robot_config(self.config.robot)
        # Initialize SDK components
        self._init_sdk_components()
        # Initialize observation config
        self._init_obs_config()
        # Initialize communication components
        self._init_communication_components()
        # Initialize policy components
        self._init_policy_components(
            self.config.task.model_path, self.config.task.policy_action_scale, self.config.task.rl_rate
        )
        # Initialize command components
        self._init_command_components()
        # Initialize input handlers
        self._init_input_handlers()
        # Initialize phase components
        self._init_phase_components()
        # Initialize latency tracking
        self._init_latency_tracking()

    # ============================================================================
    # Initialization Methods
    # ============================================================================

    def _init_robot_config(self, robot_config: RobotConfig):
        """Initialize robot configuration and parameters."""
        self.robot_config = robot_config
        self.num_dofs = self.robot_config.num_joints
        self.default_dof_angles = np.array(self.robot_config.default_dof_angles)
        self.num_upper_dofs = robot_config.num_upper_body_joints

        # Initialize motor limits (only position limits are used)
        q_max = self.robot_config.joint_pos_max
        q_min = self.robot_config.joint_pos_min
        self.q_max_arr: np.array | None = np.array(q_max) if q_max is not None else None
        self.q_min_arr: np.array | None = np.array(q_min) if q_min is not None else None
        self._clip_joint_targets = os.environ.get("HOLOSOMA_CLIP_JOINT_TARGETS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        # Setup dof names and indices
        self._setup_dof_mappings()

    def _setup_dof_mappings(self):
        """Setup DOF names and their corresponding indices."""
        self.dof_names = self.robot_config.dof_names
        # TODO: Remove upper body mentions as it's not used anymore.
        self.upper_dof_names = self.robot_config.dof_names_upper_body
        self.lower_dof_names = self.robot_config.dof_names_lower_body

        # These are used by derived classes, so keep them
        if self.upper_dof_names:
            self.upper_dof_indices = [self.dof_names.index(dof) for dof in self.upper_dof_names]
        else:
            self.upper_dof_indices = []

        if self.lower_dof_names:
            self.lower_dof_indices = [self.dof_names.index(dof) for dof in self.lower_dof_names]
        else:
            self.lower_dof_indices = []

    def _init_sdk_components(self):
        """Initialize SDK components based on robot type."""
        self.sdk_type = self.robot_config.sdk_type

        if bool(getattr(self.config.task, "use_zmq_lowcmd", False)):
            return

        if self.sdk_type == "unitree":
            pass  # No channel initialization needed for binding
        elif self.sdk_type == "ros2":
            pass
        elif self.sdk_type == "booster":
            from booster_robotics_sdk import ChannelFactory

            if ni is None:
                raise ModuleNotFoundError(
                    "booster SDK requires the optional 'netifaces' package, which is not installed in this env."
                )
            ip = ni.ifaddresses(self.config.task.interface)[ni.AF_INET][0]["addr"]
            ChannelFactory.Instance().Init(self.config.task.domain_id, ip)
        else:
            raise NotImplementedError(f"SDK type {self.sdk_type} is not supported yet")

    def _init_obs_config(self):
        """Initialize observation metadata and history buffers."""
        self.obs_config = self.config.observation
        self.obs_scales = self.obs_config.obs_scales
        self.obs_dims = self.obs_config.obs_dims
        self.obs_dict = self.obs_config.obs_dict
        self.obs_dim_dict = self._calculate_obs_dim_dict()
        self.history_length_dict = self.obs_config.history_length_dict
        self.obs_term_clips: dict[str, tuple[float, float] | None] = {}
        for term_names in self.obs_dict.values():
            for term in term_names:
                descriptor = self.obs_config.term_descriptors.get(term)
                clip = None if descriptor is None else descriptor.clip
                if clip is None:
                    self.obs_term_clips[term] = None
                    continue
                lower, upper = float(clip[0]), float(clip[1])
                if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
                    raise ValueError(
                        f"Inference observation term {term!r} clip must contain finite ordered bounds, "
                        f"got {clip!r}."
                    )
                self.obs_term_clips[term] = (lower, upper)
        self.observation_clip = float(self.obs_config.clip_observations)
        if not np.isfinite(self.observation_clip) or self.observation_clip <= 0.0:
            raise ValueError(
                f"Inference observation clip must be finite and > 0, got {self.observation_clip!r}."
            )

        # Initialize per-term history buffers using deques
        self._initialize_history_state()

    def _initialize_history_state(self):
        """Create per-term history deques and zero-initialized flattened buffers."""
        self.obs_history_buffers: dict[str, dict[str, deque[np.ndarray]]] = {}
        self.obs_terms_sorted: dict[str, list[str]] = {}
        self.obs_buf_dict: dict[str, np.ndarray] = {}
        self._obs_group_order: list[str] = list(self.obs_dict.keys())

        for group, term_names in self.obs_dict.items():
            self.obs_terms_sorted[group] = sorted(term_names)
            history_len = self.history_length_dict.get(group, 1)
            self.obs_history_buffers[group] = {}
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                term_dim = self.obs_dims[term]
                self.obs_history_buffers[group][term] = deque(maxlen=history_len)
                flattened_terms.append(np.zeros((1, term_dim * history_len), dtype=np.float32))

            self.obs_buf_dict[group] = np.concatenate(flattened_terms, axis=1) if flattened_terms else np.zeros((1, 0))

    def _reset_observation_history_state(self) -> None:
        """Restore observation/action history to the state used after a training reset."""
        self._initialize_history_state()
        if hasattr(self, "last_policy_action"):
            self.last_policy_action.fill(0.0)
        if hasattr(self, "scaled_policy_action"):
            self.scaled_policy_action.fill(0.0)
        self._reset_policy_recurrent_state()

    def _configure_policy_recurrent_state(self, metadata: Mapping[str, Any]) -> None:
        contract = recurrent_policy_contract_from_metadata(metadata)
        self._recurrent_policy_contract = None if contract is None else dict(contract)
        self._policy_recurrent_state: dict[str, np.ndarray] = {}
        if contract is None:
            return
        shape = (
            int(contract["num_layers"]),
            1,
            int(contract["hidden_dim"]),
        )
        for name in contract["state_input_names"]:
            self._policy_recurrent_state[str(name)] = np.zeros(shape, dtype=np.float32)

    def _reset_policy_recurrent_state(self) -> None:
        state = getattr(self, "_policy_recurrent_state", None)
        if isinstance(state, dict):
            for value in state.values():
                value.fill(0.0)

    def _policy_observation_input_names(self) -> list[str]:
        contract = getattr(self, "_recurrent_policy_contract", None)
        state_inputs = set(contract["state_input_names"]) if contract is not None else set()
        return [name for name in self.onnx_input_names if name not in state_inputs]

    def _run_policy_onnx(
        self,
        input_feed: dict[str, np.ndarray],
        requested_outputs: list[str],
    ) -> dict[str, np.ndarray]:
        prepared_feed = self._prepare_policy_input_feed(input_feed)
        contract = getattr(self, "_recurrent_policy_contract", None)
        fetch_names = list(requested_outputs)
        if contract is not None:
            state_inputs = [str(name) for name in contract["state_input_names"]]
            state_outputs = [str(name) for name in contract["state_output_names"]]
            overlap = set(prepared_feed).intersection(state_inputs)
            if overlap:
                raise ValueError(
                    "Recurrent state is runtime-owned and cannot be supplied by an observation caller: "
                    f"{sorted(overlap)}."
                )
            for name in state_inputs:
                prepared_feed[name] = self._policy_recurrent_state[name]
            fetch_names.extend(state_outputs)

        raw_outputs = self.onnx_policy_session.run(fetch_names, prepared_feed)
        validated = {
            name: self._require_finite_array(value, label=f"ONNX output {name!r}")
            for name, value in zip(fetch_names, raw_outputs, strict=True)
        }
        if contract is not None:
            for input_name, output_name in zip(
                contract["state_input_names"],
                contract["state_output_names"],
                strict=True,
            ):
                state_value = validated[str(output_name)]
                expected_shape = self._policy_recurrent_state[str(input_name)].shape
                if state_value.shape != expected_shape or state_value.dtype != np.float32:
                    raise ValueError(
                        f"ONNX recurrent state output {output_name!r} must have shape/dtype "
                        f"{expected_shape}/float32, got {state_value.shape}/{state_value.dtype}."
                    )
            self._policy_recurrent_state = {
                str(input_name): validated[str(output_name)].copy()
                for input_name, output_name in zip(
                    contract["state_input_names"],
                    contract["state_output_names"],
                    strict=True,
                )
            }
        return {name: validated[name] for name in requested_outputs}

    def _init_communication_components(self):
        """Initialize state processor and command sender using the wrapper."""
        if bool(getattr(self.config.task, "use_zmq_lowcmd", False)):
            self.interface = ZmqSimInterfaceWrapper(
                self.robot_config,
                sim_state_port=self.config.task.sim_state_port,
                sim_control_port=self.config.task.sim_control_port,
                use_joystick=self.config.task.use_joystick,
                sim_state_max_wall_age_ms=getattr(
                    self.config.task,
                    "sim_state_max_wall_age_ms",
                    500.0,
                ),
            )
        else:
            self.interface = InterfaceWrapper(
                self.robot_config,
                self.config.task.domain_id,
                self.config.task.interface,
                self.config.task.use_joystick,
            )
        self._perception_obs_sub: PerceptionObsSub | None = None
        self._perception_obs_shm_sub: PerceptionObsShmSub | None = None
        self._perception_contract_sha256: str | None = None
        if bool(getattr(self.config.task, "use_split_perception_obs", False)):
            if bool(getattr(self.config.task, "use_split_perception_obs_shm", False)):
                self._perception_obs_shm_sub = PerceptionObsShmSub(
                    name=getattr(self.config.task, "perception_obs_shm_name", "depth_img_shm")
                )
                self._perception_obs_shm_sub.start()
            else:
                self._perception_obs_sub = PerceptionObsSub(port=self.config.task.perception_obs_port)
                self._perception_obs_sub.start()

    def _init_policy_components(self, model_path, policy_action_scale, rl_rate):
        """Initialize policy-related components."""
        self._configured_policy_action_scale = float(policy_action_scale)
        if not np.isfinite(self._configured_policy_action_scale) or self._configured_policy_action_scale <= 0.0:
            raise ValueError(
                "Inference policy_action_scale must be finite and > 0, "
                f"got {self._configured_policy_action_scale!r}."
            )
        self.policy_action_scale = self._configured_policy_action_scale
        self.policy_action_clip = 100.0
        self.policy_action_scales = np.full((1, self.num_dofs), self.policy_action_scale, dtype=np.float32)
        self.rl_rate = rl_rate
        self.model_paths = self._collect_model_paths(model_path)
        self._policy_states: list[dict] = []
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))
        resolved_paths: list[str] = []

        for path in self.model_paths:
            # A preceding model's metadata must not leak scale/clip values into
            # a legacy or partially serialized model loaded afterwards.
            self.policy_action_scale = self._configured_policy_action_scale
            self.policy_action_clip = 100.0
            local_path = self._resolve_model_path(str(path))
            resolved_paths.append(local_path)
            self.setup_policy(local_path)
            self._policy_states.append(self._capture_policy_state())

        self._validate_policy_state_collection(self._policy_states)
        self.model_paths = resolved_paths
        self.active_policy_index = 0
        self.active_model_path = None
        self._activate_policy(0, announce=False)

        # Determine KP/KD values: config override > ONNX metadata > error
        self._resolve_control_gains()
        self._logged_training_pd_sync = False

    def _collect_model_paths(self, model_path):
        """Normalize model_path into a list of up to nine entries."""
        if isinstance(model_path, (list, tuple)):
            paths = list(model_path)
        elif model_path is not None:
            paths = [model_path]
        else:
            paths = []

        paths = [str(path) for path in paths if path]
        if not paths:
            raise ValueError("At least one model_path must be provided for policy initialization.")
        if len(paths) > 9:
            # Error out instead of warning
            raise ValueError("Received more than nine model paths. Only up to nine model paths are supported.")
        return paths

    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model path, downloading from W&B if required."""
        if model_path.startswith(("wandb://", "https://")):
            download_dir = self.config.task.wandb_download_dir
            logger.info(f"Downloading checkpoint from W&B: {model_path}")
            checkpoint_path = load_checkpoint(None, model_path, download_dir)
            resolved_path = str(checkpoint_path)
            logger.info("Checkpoint downloaded to: %s", resolved_path)
            return resolved_path
        return model_path

    def _capture_policy_state(self) -> dict:
        """Capture the current policy state for later reuse."""
        return {
            "onnx_policy_session": self.onnx_policy_session,
            "onnx_input_names": self.onnx_input_names,
            "onnx_output_names": self.onnx_output_names,
            "policy_callable": self.policy,
            "onnx_kp": self.onnx_kp,
            "onnx_kd": self.onnx_kd,
            "onnx_metadata": getattr(self, "_onnx_metadata", None),
            "onnx_artifact_sha256": getattr(self, "_onnx_artifact_sha256", None),
            "perception_contract_sha256": getattr(self, "_perception_contract_sha256", None),
            "recurrent_policy_contract": getattr(self, "_recurrent_policy_contract", None),
            "policy_recurrent_state": {
                name: value.copy()
                for name, value in getattr(self, "_policy_recurrent_state", {}).items()
            },
            "policy_action_scale": float(self.policy_action_scale),
            "policy_action_clip": float(self.policy_action_clip),
            "policy_action_scales": self.policy_action_scales.copy(),
        }

    def _restore_policy_state(self, state: dict):
        """Restore a previously captured policy state."""
        self.onnx_policy_session = state["onnx_policy_session"]
        self.onnx_input_names = state["onnx_input_names"]
        self.onnx_output_names = state["onnx_output_names"]
        self.policy = state["policy_callable"]
        self.onnx_kp = state["onnx_kp"]
        self.onnx_kd = state["onnx_kd"]
        self._onnx_metadata = state.get("onnx_metadata")
        self._onnx_artifact_sha256 = state.get("onnx_artifact_sha256")
        self._perception_contract_sha256 = state.get("perception_contract_sha256")
        recurrent_contract = state.get("recurrent_policy_contract")
        self._recurrent_policy_contract = (
            None if recurrent_contract is None else dict(recurrent_contract)
        )
        self._policy_recurrent_state = {
            name: value.copy()
            for name, value in state.get("policy_recurrent_state", {}).items()
        }
        self.policy_action_scale = float(state["policy_action_scale"])
        self.policy_action_clip = float(state["policy_action_clip"])
        self.policy_action_scales = state["policy_action_scales"].copy()

    def _validate_policy_state_collection(self, states: list[dict]) -> None:
        """Hook for policy types with extra per-slot contract state."""
        perception_contracts = {state.get("perception_contract_sha256") for state in states}
        if len(perception_contracts) > 1:
            raise ValueError(
                "Preloaded perception policy slots must use one identical producer observation contract; "
                f"found={sorted(str(value) for value in perception_contracts)}."
            )

    def _activate_policy(self, index: int, announce: bool = True):
        """Activate a preloaded policy."""
        if not (0 <= index < len(self.model_paths)):
            return

        self._restore_policy_state(self._policy_states[index])
        self._reset_policy_recurrent_state()
        self.last_policy_action.fill(0.0)
        self.scaled_policy_action.fill(0.0)
        self.active_policy_index = index
        self.active_model_path = self.model_paths[index]
        self._on_policy_switched(self.active_model_path)

        if announce and len(self.model_paths) > 1 and hasattr(self, "logger"):
            name = Path(self.active_model_path).name
            self.logger.info(colored(f"Switched to policy [{index + 1}]: {name}", "blue"))

    def _try_switch_policy_key(self, keycode: str) -> bool:
        """Switch policy slot if a numeric key is pressed."""
        if len(self.model_paths) <= 1:
            return False
        if not keycode.isdigit():
            return False
        slot = int(keycode)
        if slot == 0:
            return False
        index = slot - 1
        if index == self.active_policy_index:
            return True
        if 0 <= index < len(self.model_paths):
            self._activate_policy(index)
            return True
        return False

    def _on_policy_switched(self, model_path: str):
        """Hook for derived classes to reset state after loading a new policy."""
        _ = model_path

    def _init_command_components(self):
        """Initialize control-related components and commands."""
        self.use_policy_action = False
        self._pending_noninteractive_policy_start = False
        self.init_count = 0
        self.get_ready_state = False
        self.desired_base_height = self.config.task.desired_base_height
        self.gait_period = self.config.task.gait_period

        # Initialize command arrays
        self.lin_vel_command = np.array([[0.0, 0.0]])
        self.ang_vel_command = np.array([[0.0]])
        self.stand_command = np.array([[0]])
        self.base_height_command = np.array([[self.desired_base_height]])

        # These are used by derived classes, so keep them
        self.waist_dofs_command = np.zeros((1, 3))
        self.phase_time = np.zeros((1, 1))

        # Upper body controller
        self.upper_body_controller = None

        # Pre-allocate command arrays for postprocessing
        self.cmd_q = np.zeros(self.num_dofs)
        self.cmd_dq = np.zeros(self.num_dofs)
        self.cmd_tau = np.zeros(self.num_dofs)

    def _init_phase_components(self):
        """Initialize phase components."""
        self.use_phase = self.config.task.use_phase
        if self.use_phase:
            self.phase = np.zeros((1, 2))
            self.phase[:, 0] = 0.0  # left foot starts at 0
            self.phase[:, 1] = np.pi  # right foot starts at pi
            self.phase_dt = 2 * np.pi / (self.rl_rate * self.gait_period)

    def _init_latency_tracking(self):
        """Initialize latency tracking components."""
        self.latency_tracker = LatencyTracker(window_size=int(self.rl_rate))

    def _init_input_handlers(self):
        """Initialize input handlers (ROS, joystick, keyboard)."""
        self._init_rate_handler()
        self._init_external_policy_control()
        self._init_input_device()

    def _init_external_policy_control(self):
        """Initialize optional external start/stop/init controls."""
        self._policy_control_sub: PolicyControlPull | None = None
        raw_port = os.environ.get("HOLOSOMA_POLICY_CONTROL_PORT", "").strip()
        if not raw_port:
            return
        try:
            port = int(raw_port)
        except ValueError:
            self.logger.warning("Ignoring invalid HOLOSOMA_POLICY_CONTROL_PORT={}", raw_port)
            return
        if port <= 0:
            return
        try:
            self._policy_control_sub = PolicyControlPull(port=port)
            self._policy_control_sub.start()
        except Exception as exc:
            self.logger.warning("Could not start policy control receiver on port {}: {}", port, exc)
            self._policy_control_sub = None

    def _allow_noninteractive_autostart_with_policy_control(self) -> bool:
        value = os.environ.get("HOLOSOMA_POLICY_CONTROL_ALLOW_NONINTERACTIVE_AUTOSTART", "")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _require_finite_array(value, *, label: str, dtype=np.float32) -> np.ndarray:
        """Convert a runtime tensor to ndarray and reject NaN/Inf fail-closed."""

        try:
            array = np.asarray(value, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a numeric array.") from exc
        if array.size == 0:
            raise ValueError(f"{label} must not be empty.")
        try:
            finite_mask = np.isfinite(array)
        except TypeError as exc:
            raise ValueError(f"{label} must contain only numeric values.") from exc
        if not bool(np.all(finite_mask)):
            bad_indices = np.argwhere(~finite_mask)
            raise FloatingPointError(
                f"{label} contains {int((~finite_mask).sum())} non-finite value(s); "
                f"first_indices={bad_indices[:8].tolist()}."
            )
        return array

    def _has_valid_robot_state(self, robot_state_data: np.ndarray) -> bool:
        if robot_state_data is None:
            return False
        try:
            state = np.asarray(robot_state_data, dtype=np.float32)
        except (TypeError, ValueError):
            return False
        minimum_state_dim = 7 + self.num_dofs + 6 + self.num_dofs
        if state.ndim != 2 or state.shape[0] < 1 or state.shape[1] < minimum_state_dim:
            return False
        if not bool(np.all(np.isfinite(state[:, :minimum_state_dim]))):
            return False
        quat = state[0, 3:7]
        if float(np.linalg.norm(quat)) < 0.5:
            return False
        joint_pos = state[0, 7 : 7 + self.num_dofs]
        return bool(np.any(np.abs(joint_pos) > 1e-6) or np.any(np.abs(quat) > 1e-6))

    def _can_finish_pending_policy_start(self, robot_state_data: np.ndarray) -> bool:  # noqa: ARG002
        return True

    def _after_auto_start_policy(self) -> None:
        """Hook invoked after auto-starting the policy from a valid state."""

    def _maybe_auto_start_rollout(self) -> None:
        """Hook for derived policies to auto-start task-specific rollout state."""

    def _should_auto_start_policy_immediately(self) -> bool:
        """Hook for derived policies to gate base auto-start behavior."""
        return True

    def _init_rate_handler(self):
        """Initialize ROS handler if enabled."""
        self.rl_rate = self.config.task.rl_rate
        if self.config.task.use_ros:
            import rclpy

            rclpy.init(args=None)
            self.node = rclpy.create_node("policy_node")
            self.logger = self.node.get_logger()
            self.rate = self.node.create_rate(self.rl_rate)
            thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
            thread.start()
        else:
            self.logger = logger
            self.rate = RateLimiter(self.rl_rate)

    def _init_input_device(self):
        """Initialize input device (joystick or keyboard)."""
        if self.config.task.use_joystick:
            self._init_joystick_handler()
        else:
            self._init_keyboard_handler()

    def _init_joystick_handler(self):
        """Initialize joystick handler."""
        if sys.platform == "darwin":
            self.logger.warning("Joystick is not supported on Windows or Mac.")
            self.logger.warning("Using keyboard instead")
            self.use_joystick = False
            self._init_keyboard_handler()
        else:
            self.logger.info("Using joystick")
            self.use_joystick = True

    def _init_keyboard_handler(self):
        """Initialize keyboard handler."""
        self.logger.info("Using keyboard")
        self.use_joystick = False
        # Check if running in a TTY environment
        if not sys.stdin.isatty():
            self.logger.warning("Not running in a TTY environment - keyboard input disabled")
            self.logger.warning("This is normal for automated tests or non-interactive environments")
            if (
                self._policy_control_sub is not None
                and not self._allow_noninteractive_autostart_with_policy_control()
            ):
                self.logger.info("Policy control is enabled; waiting for external start/stop/init commands.")
                return
            if self.config.task.defer_policy_start_until_valid_state:
                self.logger.info("Deferring policy auto-start until a valid robot state is received")
                self._pending_noninteractive_policy_start = True
            else:
                self.logger.info("Auto-starting policy in non-interactive mode")
                self.use_policy_action = True
            return
        # Start keyboard listener in a daemon thread
        threading.Thread(target=self.start_key_listener, daemon=True).start()
        self.logger.info("Keyboard Listener Initialized")

    # ============================================================================
    # Policy Methods
    # ============================================================================

    def _load_onnx_session_and_metadata(self, model_path: str):
        """Bind ORT graph and metadata to one stable, hashed byte payload."""

        path = Path(model_path)
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"ONNX policy must be a regular file: {path}")
            payload = stream.read()
            after = os.fstat(stream.fileno())
        if not payload:
            raise ValueError(f"ONNX policy is empty: {path}")
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ctime_ns != after.st_ctime_ns
            or len(payload) != before.st_size
        ):
            raise RuntimeError(f"ONNX policy changed while it was being read: {path}")

        artifact_sha256 = hashlib.sha256(payload).hexdigest()
        onnx_model = onnx.load_model_from_string(payload)
        def reject_nonfinite_json(constant: str):
            raise ValueError(f"non-finite JSON constant {constant!r}")

        metadata = {}
        for prop in onnx_model.metadata_props:
            if not prop.key:
                raise ValueError(f"ONNX policy contains an empty metadata key: {path}")
            if prop.key in metadata:
                raise ValueError(
                    "ONNX policy contains an ambiguous duplicate metadata key "
                    f"{prop.key!r}: {path}"
                )

            try:
                metadata[prop.key] = json.loads(
                    prop.value,
                    parse_constant=reject_nonfinite_json,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ONNX metadata {prop.key!r} is not strict finite JSON: {path}"
                ) from exc
        allow_unsafe_embedded_timeline = os.environ.get(
            "HOLOSOMA_ALLOW_UNSAFE_RAW_EMBEDDED_MOTION_TIMELINE",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        validate_embedded_motion_timeline_model(
            onnx_model,
            metadata,
            allow_unsafe_diagnostic=allow_unsafe_embedded_timeline,
        )
        embedded_contact_sidecar_contract_from_metadata(metadata)
        embedded_button_window_contract_from_metadata(metadata)
        session = onnxruntime.InferenceSession(payload)
        self._onnx_artifact_sha256 = artifact_sha256
        logger.info(
            "Loaded immutable ONNX artifact bytes: name={} sha256={}",
            path.name,
            artifact_sha256,
        )
        return session, metadata

    @staticmethod
    def _effective_perception_contract_sha256(metadata: dict) -> str | None:
        original_digest = perception_observation_contract_sha256_from_metadata(metadata)
        override = os.environ.get(
            "HOLOSOMA_EVAL_PERCEPTION_CONTRACT_SHA256_OVERRIDE",
            "",
        ).strip().lower()
        if not override:
            return original_digest
        allow_override = os.environ.get(
            "HOLOSOMA_EVAL_ALLOW_PERCEPTION_CONTRACT_OVERRIDE",
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not allow_override:
            raise RuntimeError(
                "HOLOSOMA_EVAL_PERCEPTION_CONTRACT_SHA256_OVERRIDE is evaluation-only and "
                "requires HOLOSOMA_EVAL_ALLOW_PERCEPTION_CONTRACT_OVERRIDE=1"
            )
        if original_digest is None:
            raise RuntimeError("Cannot override a missing ONNX perception observation contract")
        try:
            decoded = bytes.fromhex(override)
        except ValueError as exc:
            raise RuntimeError(
                "Evaluation perception contract override must be 64 lowercase hexadecimal characters"
            ) from exc
        if len(decoded) != 32 or override != override.lower():
            raise RuntimeError(
                "Evaluation perception contract override must be 64 lowercase hexadecimal characters"
            )
        logger.warning(
            "Using explicit evaluation-only perception producer contract: original={} effective={}",
            original_digest,
            override,
        )
        return override

    def setup_policy(self, model_path):
        """Setup ONNX policy model and extract metadata."""
        self.onnx_policy_session, metadata = self._load_onnx_session_and_metadata(model_path)
        input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        self.onnx_input_names = input_names
        self.onnx_output_names = output_names

        self._onnx_metadata = metadata
        self._configure_policy_recurrent_state(metadata)

        validate_onnx_policy_contract(
            metadata=metadata,
            input_shapes={inp.name: inp.shape for inp in self.onnx_policy_session.get_inputs()},
            output_shapes={out.name: out.shape for out in self.onnx_policy_session.get_outputs()},
            input_types={inp.name: inp.type for inp in self.onnx_policy_session.get_inputs()},
            output_types={out.name: out.type for out in self.onnx_policy_session.get_outputs()},
            observation=self.config.observation,
            runtime_dof_names=self.dof_names,
            runtime_default_dof_angles=self.default_dof_angles,
            runtime_motor_effort_limits=self.robot_config.motor_effort_limit,
            runtime_joint2motor=self.robot_config.joint2motor,
        )
        self._perception_contract_sha256 = self._effective_perception_contract_sha256(metadata)

        # Extract KP/KD from metadata (will be None if not present)
        self.onnx_kp = self._joint_values_to_motor_order(metadata["kp"], "KP") if "kp" in metadata else None
        self.onnx_kd = self._joint_values_to_motor_order(metadata["kd"], "KD") if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {Path(model_path).name}")

        self._set_policy_action_scales_from_metadata(metadata)

        def policy_act(obs_dict):
            # For example,obs_dict contains:
            # {
            #     'actor_obs_lower_body': np.array([...]),
            #     'actor_obs_upper_body': np.array([...]),
            #     'estimator_obs': np.array([...])
            # }
            input_feed = {
                name: obs_dict[name] for name in self._policy_observation_input_names()
            }
            action_output = "action" if "action" in self.onnx_output_names else "actions"
            return self._run_policy_onnx(input_feed, [action_output])[action_output]

        self.policy = policy_act

    def _resolve_control_gains(self):
        """Resolve KP/KD values with priority: config override > ONNX metadata > error.

        Creates a new config instance with resolved values if needed.
        """
        # Check if config has explicit KP/KD values
        config_has_kp = hasattr(self.robot_config, "motor_kp") and self.robot_config.motor_kp is not None
        config_has_kd = hasattr(self.robot_config, "motor_kd") and self.robot_config.motor_kd is not None

        if config_has_kp and config_has_kd:
            # Config already has values (override) - nothing to do
            logger.info(colored("Using KP/KD from config (override)", "yellow"))
            kp_values = np.array(self.robot_config.motor_kp)
            kd_values = np.array(self.robot_config.motor_kd)
        elif self.onnx_kp is not None and self.onnx_kd is not None:
            # Use ONNX metadata (default) - create new config with values
            logger.info(colored("Using KP/KD from ONNX metadata", "green"))
            kp_values = self.onnx_kp
            kd_values = self.onnx_kd
            # Create new config instance with ONNX values
            self.robot_config = replace(
                self.robot_config, motor_kp=tuple(kp_values.tolist()), motor_kd=tuple(kd_values.tolist())
            )
            # Update InterfaceWrapper's robot_config reference since replace() creates a new object
            self.interface.robot_config = self.robot_config
            # Update sdk2py backend components (booster SDK only)
            if self.interface.backend == "sdk2py":
                self.interface.command_sender.config = self.robot_config
                self.interface.state_processor.config = self.robot_config
        else:
            # No values available - error
            raise ValueError(
                "No KP/KD values found. Either provide them in robot config "
                "or ensure ONNX model has metadata attached during training."
            )

        # Validate dimensions
        if len(kp_values) != self.robot_config.num_motors:
            raise ValueError(
                f"KP array length ({len(kp_values)}) does not match num_motors ({self.robot_config.num_motors})"
            )
        if len(kd_values) != self.robot_config.num_motors:
            raise ValueError(
                f"KD array length ({len(kd_values)}) does not match num_motors ({self.robot_config.num_motors})"
            )

    def _sync_policy_pd_with_training(self) -> None:
        """Use the active ONNX policy's training gains during policy control.

        Robot-config gains are useful for startup/pose holding, but applying
        them to policy actions changes the closed-loop system the actor was
        trained against.  Policy control therefore restores both the gain
        vectors and the interface gain multipliers to the ONNX metadata.
        """
        if self.onnx_kp is None or self.onnx_kd is None:
            return

        expected_kp = np.asarray(self.onnx_kp, dtype=np.float32).reshape(-1)
        expected_kd = np.asarray(self.onnx_kd, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(expected_kp)) or not np.all(np.isfinite(expected_kd)):
            raise ValueError("ONNX PD metadata contains non-finite values.")
        if expected_kp.size != self.robot_config.num_motors or expected_kd.size != self.robot_config.num_motors:
            raise ValueError(
                "ONNX PD metadata dimensions do not match the active robot: "
                f"kp={expected_kp.size}, kd={expected_kd.size}, motors={self.robot_config.num_motors}."
            )

        current_kp = getattr(self.robot_config, "motor_kp", None)
        current_kd = getattr(self.robot_config, "motor_kd", None)
        needs_cfg_sync = (
            current_kp is None
            or current_kd is None
            or np.asarray(current_kp).size != expected_kp.size
            or np.asarray(current_kd).size != expected_kd.size
            or not np.allclose(np.asarray(current_kp, dtype=np.float32), expected_kp)
            or not np.allclose(np.asarray(current_kd, dtype=np.float32), expected_kd)
        )

        if needs_cfg_sync:
            self.robot_config = replace(
                self.robot_config,
                motor_kp=tuple(expected_kp.tolist()),
                motor_kd=tuple(expected_kd.tolist()),
            )
            self.interface.robot_config = self.robot_config
            if getattr(self.interface, "backend", None) == "sdk2py":
                self.interface.command_sender.config = self.robot_config
                self.interface.state_processor.config = self.robot_config

        kp_level = float(getattr(self.interface, "kp_level", 1.0))
        kd_level = float(getattr(self.interface, "kd_level", 1.0))
        levels_reset = abs(kp_level - 1.0) > 1.0e-6 or abs(kd_level - 1.0) > 1.0e-6
        if levels_reset:
            self.interface.kp_level = 1.0
            self.interface.kd_level = 1.0

        if needs_cfg_sync or levels_reset:
            logger.info("Forced active policy PD gains to ONNX/training metadata (gain levels=1.0).")
            self._logged_training_pd_sync = True
        elif not getattr(self, "_logged_training_pd_sync", False):
            logger.info("Active policy PD gains match ONNX/training metadata.")
            self._logged_training_pd_sync = True

    def _calculate_obs_dim_dict(self):
        """Calculate observation dimensions for each observation type."""
        obs_dim_dict = {}
        for key in self.obs_dict:
            obs_dim_dict[key] = 0
            for obs_name in self.obs_dict[key]:
                obs_dim_dict[key] += self.obs_dims[obs_name]
        return obs_dim_dict

    def _joint_values_to_motor_order(self, values, label: str) -> np.ndarray:
        """Map DOF-ordered ONNX metadata to the runtime motor order."""
        joint_values = np.asarray(values, dtype=np.float32).reshape(-1)
        if (
            joint_values.size != self.num_dofs
            or not np.all(np.isfinite(joint_values))
            or np.any(joint_values < 0.0)
        ):
            raise ValueError(
                f"ONNX {label} metadata must contain {self.num_dofs} finite non-negative DOF-ordered values, "
                f"got shape={joint_values.shape}."
            )
        mapping = np.asarray(self.robot_config.joint2motor, dtype=np.int64).reshape(-1)
        if mapping.size != self.num_dofs:
            raise ValueError(
                f"joint2motor length {mapping.size} does not match policy DOF count {self.num_dofs}."
            )
        expected_motors = int(self.robot_config.num_motors)
        if expected_motors != self.num_dofs or sorted(mapping.tolist()) != list(range(expected_motors)):
            raise ValueError(
                "Policy control requires a one-to-one DOF/motor mapping; "
                f"dofs={self.num_dofs}, motors={expected_motors}, joint2motor={mapping.tolist()}."
            )
        motor_values = np.empty((expected_motors,), dtype=np.float32)
        motor_values[mapping] = joint_values
        return motor_values

    def _resolve_motor_kp_from_control_cfg(self, control_cfg: dict) -> np.ndarray | None:
        stiffness_cfg = control_cfg.get("stiffness")
        if not isinstance(stiffness_cfg, dict):
            return None

        joint_kp = np.zeros(self.num_dofs, dtype=np.float32)
        for i, name in enumerate(self.dof_names):
            matched = False
            for dof_name, stiffness in stiffness_cfg.items():
                if dof_name in name:
                    joint_kp[i] = float(stiffness)
                    matched = True
            if not matched:
                return None

        motor_kp = np.zeros(self.robot_config.num_motors, dtype=np.float32)
        joint2motor = tuple(self.robot_config.joint2motor)
        for joint_idx, kp in enumerate(joint_kp):
            motor_kp[joint2motor[joint_idx]] = kp
        return motor_kp

    def _set_policy_action_scales_from_metadata(self, metadata: dict) -> None:
        scale_array = np.full((self.num_dofs,), self.policy_action_scale, dtype=np.float32)

        experiment_cfg = metadata.get("experiment_config", {})
        if not isinstance(experiment_cfg, dict):
            self.policy_action_scales = scale_array.reshape(1, -1)
            return

        saved_robot_cfg = experiment_cfg.get("robot", {})
        if not isinstance(saved_robot_cfg, dict):
            self.policy_action_scales = scale_array.reshape(1, -1)
            return
        control_cfg = saved_robot_cfg.get("control", {})
        if not isinstance(control_cfg, dict):
            self.policy_action_scales = scale_array.reshape(1, -1)
            return

        if control_cfg.get("clip_actions", True):
            action_clip = control_cfg.get("action_clip_value")
            if action_clip is not None:
                self.policy_action_clip = float(action_clip)
                if not np.isfinite(self.policy_action_clip) or self.policy_action_clip <= 0.0:
                    raise ValueError(
                        "ONNX action_clip_value must be finite and > 0, "
                        f"got {self.policy_action_clip!r}."
                    )

        base_scale = control_cfg.get("action_scale")
        if base_scale is not None:
            self.policy_action_scale = float(base_scale)
            if not np.isfinite(self.policy_action_scale) or self.policy_action_scale <= 0.0:
                raise ValueError(
                    "ONNX action_scale must be finite and > 0, "
                    f"got {self.policy_action_scale!r}."
                )
            scale_array.fill(self.policy_action_scale)

        if not control_cfg.get("action_scales_by_effort_limit_over_p_gain", False):
            self._apply_debug_action_scale_multipliers(scale_array)
            self.policy_action_scales = scale_array.reshape(1, -1)
            logger.info(
                "Using ONNX metadata scalar action scale: base={} final_min={:.6f} final_max={:.6f}",
                self.policy_action_scale,
                float(np.min(scale_array)),
                float(np.max(scale_array)),
            )
            return

        motor_kp = self.onnx_kp
        if motor_kp is None:
            motor_kp = self._resolve_motor_kp_from_control_cfg(control_cfg)
        if motor_kp is None:
            raise ValueError("Training metadata requested per-joint action scaling, but KP values were unavailable.")

        motor_kp = np.asarray(motor_kp, dtype=np.float32)
        saved_joint_effort = np.asarray(saved_robot_cfg.get("dof_effort_limit_list", ()), dtype=np.float32)
        if motor_kp.shape != (self.robot_config.num_motors,) or saved_joint_effort.shape != (self.num_dofs,):
            raise ValueError(
                "Per-joint action scaling metadata has incompatible dimensions: "
                f"kp={motor_kp.shape}, saved_joint_effort={saved_joint_effort.shape}, "
                f"motors={self.robot_config.num_motors}, dofs={self.num_dofs}."
            )
        if (
            not np.all(np.isfinite(motor_kp))
            or not np.all(np.isfinite(saved_joint_effort))
            or np.any(motor_kp < 0.0)
            or np.any(saved_joint_effort < 0.0)
        ):
            raise ValueError(
                "Per-joint action scaling metadata contains non-finite or negative KP/effort values."
            )

        joint2motor = np.asarray(self.robot_config.joint2motor, dtype=np.int64)
        for joint_idx in range(self.num_dofs):
            motor_idx = int(joint2motor[joint_idx])
            stiffness = float(motor_kp[motor_idx])
            effort = float(saved_joint_effort[joint_idx])
            scale_array[joint_idx] = 0.0 if stiffness == 0.0 else self.policy_action_scale * effort / stiffness

        self._apply_debug_action_scale_multipliers(scale_array)
        self.policy_action_scales = scale_array.reshape(1, -1)
        logger.info(
            "Using training-aligned per-joint action scales from ONNX metadata: "
            "base={} final_min={:.6f} final_max={:.6f} final_mean={:.6f}",
            self.policy_action_scale,
            float(np.min(scale_array)),
            float(np.max(scale_array)),
            float(np.mean(scale_array)),
        )

    def _apply_debug_action_scale_multipliers(self, scale_array: np.ndarray) -> None:
        """Optional MuJoCo sim2sim diagnostics for separating lower/upper-body scale issues."""

        env_to_markers = (
            ("HOLOSOMA_POLICY_ACTION_SCALE_LOWER_MULT", ("hip", "knee", "ankle")),
            ("HOLOSOMA_POLICY_ACTION_SCALE_WAIST_MULT", ("waist",)),
            ("HOLOSOMA_POLICY_ACTION_SCALE_UPPER_MULT", ("shoulder", "elbow", "wrist")),
            ("HOLOSOMA_POLICY_ACTION_SCALE_WRIST_MULT", ("wrist",)),
        )
        applied: list[str] = []
        for env_name, markers in env_to_markers:
            raw_value = os.environ.get(env_name, "").strip()
            if not raw_value:
                continue
            multiplier = float(raw_value)
            if not np.isfinite(multiplier) or multiplier <= 0.0:
                raise ValueError(f"{env_name} must be finite and > 0, got {multiplier!r}.")
            matched = [idx for idx, name in enumerate(self.dof_names) if any(marker in name for marker in markers)]
            if not matched:
                continue
            scale_array[np.asarray(matched, dtype=np.int64)] *= multiplier
            applied.append(f"{env_name}={multiplier:g}({len(matched)})")
        if applied:
            logger.info("Applied debug policy action-scale multipliers: {}", ", ".join(applied))

    def _update_policy_action_state(self, policy_action, *, label: str) -> np.ndarray:
        """Store raw action history while applying the training control clip.

        Training exposes ``ActionManager.action`` to the next observation.  It
        contains the raw policy action; the joint action term clips a separate
        processed copy before applying control.  Inference must preserve that
        split as well.  The regular observation-group clip is applied later by
        :meth:`_update_obs_history`.
        """
        raw_policy_action = self._require_finite_array(policy_action, label=label)
        expected_shape = (1, int(self.num_dofs))
        if raw_policy_action.ndim != 2 or raw_policy_action.shape != expected_shape:
            raise ValueError(
                f"{label} must have shape {expected_shape}; refusing NumPy broadcasting or partial-action "
                f"padding, got {raw_policy_action.shape}."
            )
        clipped_policy_action = self._require_finite_array(
            np.clip(
                raw_policy_action,
                -self.policy_action_clip,
                self.policy_action_clip,
            ),
            label="clipped policy action",
        )

        action_scales = self._require_finite_array(
            self.policy_action_scales,
            label="policy action scales",
        )
        if action_scales.ndim != 2 or action_scales.shape != expected_shape:
            raise ValueError(
                f"Policy action scales must have shape {expected_shape}; refusing NumPy broadcasting, "
                f"got {action_scales.shape}."
            )
        scaled_policy_action = self._require_finite_array(
            clipped_policy_action * action_scales,
            label="scaled policy action",
        )
        # Commit both pieces together only after every shape/finite check has
        # succeeded.  A rejected output must not partially advance the action
        # history consumed by the next observation.
        self.last_policy_action = raw_policy_action.copy()
        self.scaled_policy_action = scaled_policy_action
        return raw_policy_action

    def rl_inference(self, robot_state_data):
        """Perform RL inference to get policy action."""
        obs = self.prepare_obs_for_rl(robot_state_data)
        self._update_policy_action_state(
            self.policy(obs),
            label="policy action output",
        )

        return self.scaled_policy_action

    def _prepare_policy_input_feed(self, input_feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the serialized training-wide clip to ONNX observation inputs."""
        observation_inputs = set(self.obs_dict)
        observation_inputs.update({"obs", "actor_obs", "perception_obs"})
        observation_inputs.add(getattr(self, "_obs_input_name", None))
        observation_inputs.add(getattr(self, "_perception_obs_input_name", None))

        metadata = getattr(self, "_onnx_metadata", {})
        experiment = metadata.get("experiment_config", {}) if isinstance(metadata, dict) else {}
        algo = experiment.get("algo", {}) if isinstance(experiment, dict) else {}
        algo_config = algo.get("config", {}) if isinstance(algo, dict) else {}
        module_dict = algo_config.get("module_dict", {}) if isinstance(algo_config, dict) else {}
        actor = module_dict.get("actor", {}) if isinstance(module_dict, dict) else {}
        if isinstance(actor, dict):
            layer_config = actor.get("layer_config", {})
            if isinstance(layer_config, dict):
                observation_inputs.add(layer_config.get("perception_input_name"))

        prepared: dict[str, np.ndarray] = {}
        for name, value in input_feed.items():
            if name in observation_inputs:
                value = self._require_finite_array(
                    value,
                    label=f"ONNX observation input {name!r}",
                )
                value = np.clip(
                    value,
                    -self.observation_clip,
                    self.observation_clip,
                ).astype(np.float32, copy=False)
            else:
                value = self._require_finite_array(
                    value,
                    label=f"ONNX input {name!r}",
                    dtype=None,
                )
            prepared[name] = value
        return prepared

    def _get_split_perception_obs(
        self,
        expected_dim: int | None = None,
        *,
        target_sim_time_ms: float | int | None = None,
        target_episode_generation: int | None = None,
    ) -> np.ndarray:
        """Return the latest split-sim perception observation for ONNX perception inputs."""
        if target_episode_generation is not None:
            if (
                isinstance(target_episode_generation, bool)
                or not isinstance(target_episode_generation, (int, np.integer))
                or int(target_episode_generation) < 0
                or int(target_episode_generation) > (1 << 63) - 1
            ):
                raise ValueError(
                    "target_episode_generation must be a non-negative integer within the "
                    f"transport range, got {target_episode_generation!r}."
                )
            target_episode_generation = int(target_episode_generation)
        expected_contract = getattr(self, "_perception_contract_sha256", None)
        if expected_contract is None:
            raise RuntimeError(
                "Perception policy artifact has no authenticated effective producer contract. "
                "Re-export the policy from a live training environment before scientific split-sim use."
            )
        if target_episode_generation is None:
            raise RuntimeError(
                "Live split-sim perception requires a pinned simulator episode_generation. "
                "Refusing an unpaired perception frame; use an authenticated sim-state producer."
            )
        if self._perception_obs_shm_sub is not None:
            if expected_dim is None:
                raise RuntimeError("Shared-memory perception obs requires a known expected dimension.")
            def read_shm_obs() -> np.ndarray | None:
                if target_sim_time_ms is None:
                    return self._perception_obs_shm_sub.get_obs(
                        int(expected_dim),
                        expected_contract,
                        expected_episode_generation=target_episode_generation,
                    )
                return self._perception_obs_shm_sub.get_obs_at_or_before(
                    int(expected_dim),
                    target_sim_time_ms,
                    expected_contract,
                    expected_episode_generation=target_episode_generation,
                )
            obs = read_shm_obs()
            if obs is None:
                deadline = time.perf_counter() + 2.0
                while time.perf_counter() < deadline:
                    obs = read_shm_obs()
                    if obs is not None:
                        break
                    time.sleep(0.01)
            if obs is None:
                raise RuntimeError("Timed out waiting for split-sim shared-memory perception_obs payload.")
            if not hasattr(self, "_logged_split_perception_obs_shm"):
                logger.info("Using split sim shared-memory perception_obs with {} values", obs.shape[1])
                self._logged_split_perception_obs_shm = True
            return obs

        if self._perception_obs_sub is None:
            raise RuntimeError(
                "Policy expects perception_obs, but split perception subscription is disabled. "
                "Pass --task.use-split-perception-obs and match --task.perception-obs-port to run_sim."
            )

        can_select_payload = hasattr(self._perception_obs_sub, "get_payload_at_or_before") and (
            target_sim_time_ms is not None or target_episode_generation is not None
        )
        if can_select_payload:
            payload = self._perception_obs_sub.get_payload_at_or_before(
                target_sim_time_ms,
                expected_episode_generation=target_episode_generation,
            )
        else:
            payload = self._perception_obs_sub.get_payload()
        if payload is None:
            deadline = time.perf_counter() + 2.0
            while time.perf_counter() < deadline:
                if can_select_payload:
                    payload = self._perception_obs_sub.get_payload_at_or_before(
                        target_sim_time_ms,
                        expected_episode_generation=target_episode_generation,
                    )
                else:
                    payload = self._perception_obs_sub.get_payload()
                if payload is not None:
                    break
                time.sleep(0.01)
        if payload is None:
            raise RuntimeError("Timed out waiting for split-sim perception_obs payload.")

        published_contract = payload.get("perception_contract_sha256")
        if published_contract != expected_contract:
            raise RuntimeError(
                "Split-sim perception producer contract does not match the policy artifact: "
                f"published={published_contract!r}, expected={expected_contract!r}."
            )

        if target_episode_generation is not None:
            published_episode_generation = payload.get("episode_generation")
            if (
                isinstance(published_episode_generation, bool)
                or not isinstance(published_episode_generation, int)
                or published_episode_generation != target_episode_generation
            ):
                raise RuntimeError(
                    "Split-sim perception episode does not match the pinned simulator state: "
                    f"published={published_episode_generation!r}, "
                    f"expected={target_episode_generation}."
                )

        values = payload.get("perception_obs")
        if values is None:
            raise RuntimeError(f"Perception payload missing 'perception_obs': keys={sorted(payload.keys())}")
        obs = np.asarray(values, dtype=np.float32).reshape(1, -1)
        if expected_dim is not None and obs.shape[1] != int(expected_dim):
            raise RuntimeError(f"perception_obs dim mismatch: got {obs.shape[1]}, expected {int(expected_dim)}")
        if not hasattr(self, "_logged_split_perception_obs"):
            logger.info("Using split sim perception_obs with {} values", obs.shape[1])
            self._logged_split_perception_obs = True
        return obs

    # ============================================================================
    # Observation Processing Methods
    # ============================================================================

    def get_current_obs_buffer_dict(self, robot_state_data):
        """Extract current observation data from robot state."""
        current_obs_buffer_dict = {}

        # Extract base and joint data
        current_obs_buffer_dict["base_quat"] = robot_state_data[:, 3:7]
        current_obs_buffer_dict["base_ang_vel"] = robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6]
        current_obs_buffer_dict["dof_pos"] = robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles
        current_obs_buffer_dict["dof_vel"] = robot_state_data[
            :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
        ]

        # Calculate projected gravity
        v = np.array([[0, 0, -1]])
        current_obs_buffer_dict["projected_gravity"] = quat_rotate_inverse(current_obs_buffer_dict["base_quat"], v)

        return current_obs_buffer_dict

    def parse_current_obs_dict(self, current_obs_buffer_dict):
        """Parse observation buffer into observation dictionary with per-term scaling."""
        current_obs_dict: dict[str, dict[str, np.ndarray]] = {}
        for group, term_names in self.obs_terms_sorted.items():
            grouped_terms: dict[str, np.ndarray] = {}
            for term in term_names:
                if term not in current_obs_buffer_dict:
                    raise KeyError(f"Observation term '{term}' missing from current observation buffer.")
                term_obs = current_obs_buffer_dict[term]
                if term_obs.ndim == 1:
                    term_obs = term_obs.reshape(1, -1)
                scale = self.obs_scales[term]
                term_obs = (term_obs * scale).astype(np.float32, copy=False)
                term_clip = self.obs_term_clips.get(term)
                if term_clip is not None:
                    term_obs = np.clip(term_obs, term_clip[0], term_clip[1]).astype(
                        np.float32,
                        copy=False,
                    )
                grouped_terms[term] = term_obs
            current_obs_dict[group] = grouped_terms
        return current_obs_dict

    def _prepare_group_observations(self, robot_state_data):
        """Return flattened observations per group with history applied per term."""
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        self._last_current_obs_buffer_dict = {
            key: np.array(value, dtype=np.float32, copy=True) for key, value in current_obs_buffer_dict.items()
        }
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)
        return self._update_obs_history(current_obs_dict)

    def _update_obs_history(self, current_obs_dict: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
        """Update observation history buffers and return flattened observations per group."""
        group_outputs: dict[str, np.ndarray] = {}

        for group, term_dict in current_obs_dict.items():
            history_len = self.history_length_dict.get(group, 1)
            flattened_terms: list[np.ndarray] = []

            for term in self.obs_terms_sorted[group]:
                obs = self._require_finite_array(
                    term_dict[term],
                    label=f"observation term {group}.{term}",
                )
                obs = np.asarray(obs, dtype=np.float32, order="C")
                if obs.ndim == 1:
                    obs = obs.reshape(1, -1)

                buffer = self.obs_history_buffers[group][term]
                buffer.append(obs.copy())

                history = list(buffer)
                if len(history) < history_len:
                    missing = history_len - len(history)
                    history = [np.zeros_like(obs)] * missing + history

                # Match training order: time dimension first, then flatten into [history_len * term_dim].
                stacked = np.stack(history[-history_len:], axis=1)
                flattened_terms.append(stacked.reshape(obs.shape[0], -1))

            group_output = (
                np.concatenate(flattened_terms, axis=1).astype(np.float32, copy=False)
                if flattened_terms
                else np.zeros((1, 0), dtype=np.float32)
            )
            group_outputs[group] = np.clip(
                group_output,
                -self.observation_clip,
                self.observation_clip,
            ).astype(np.float32, copy=False)
            self._require_finite_array(
                group_outputs[group],
                label=f"flattened observation group {group!r}",
            )

        self.obs_buf_dict = {group: value.copy() for group, value in group_outputs.items()}
        return group_outputs

    def _prefill_obs_history(self, robot_state_data, repeats: int | None = None) -> None:
        """Fill observation history with the current frame before a rollout starts."""
        current_obs_buffer_dict = self.get_current_obs_buffer_dict(robot_state_data)
        self._last_current_obs_buffer_dict = {
            key: np.array(value, dtype=np.float32, copy=True) for key, value in current_obs_buffer_dict.items()
        }
        current_obs_dict = self.parse_current_obs_dict(current_obs_buffer_dict)

        for group, term_dict in current_obs_dict.items():
            history_len = int(self.history_length_dict.get(group, 1))
            fill_count = history_len if repeats is None else max(0, min(int(repeats), history_len))
            for term, obs in term_dict.items():
                if group not in self.obs_history_buffers or term not in self.obs_history_buffers[group]:
                    continue
                obs_arr = np.asarray(obs, dtype=np.float32, order="C")
                if obs_arr.ndim == 1:
                    obs_arr = obs_arr.reshape(1, -1)
                buffer = self.obs_history_buffers[group][term]
                buffer.clear()
                for _ in range(fill_count):
                    buffer.append(obs_arr.copy())

    def _assemble_actor_obs(self, group_outputs: dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate actor observation groups to match training input ordering."""
        actor_groups = [
            group
            for group in self._obs_group_order
            if group in group_outputs and (group.startswith("actor_obs") or group == "motion_future_target_poses")
        ]
        actor_groups.extend(
            sorted(
                group
                for group in group_outputs
                if (group.startswith("actor_obs") or group == "motion_future_target_poses")
                and group not in actor_groups
            )
        )

        if not actor_groups:
            raise KeyError("Observation group 'actor_obs' is not configured for this policy.")

        return np.concatenate([group_outputs[group] for group in actor_groups], axis=1).astype(np.float32, copy=False)

    def prepare_obs_for_rl(self, robot_state_data):
        """Prepare observations for RL inference."""
        group_outputs = self._prepare_group_observations(robot_state_data)
        actor_obs = self._assemble_actor_obs(group_outputs)
        return {"actor_obs": actor_obs}

    # ============================================================================
    # Control/Command Methods
    # ============================================================================

    def get_init_target(self, robot_state_data):
        """Get initialization target joint positions."""
        dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
        if self.get_ready_state:
            # Interpolate from current dof_pos to default angles
            q_target = dof_pos + (self.default_dof_angles - dof_pos) * (self.init_count / 500)
            self.init_count += 1
            return q_target
        return dof_pos

    def _pin_control_tick_state(self, robot_state_data) -> None:
        """Pin the full state payload that produced this tick's robot vector."""

        self._control_tick_robot_state_data = robot_state_data
        snapshot = None
        pin_snapshot = getattr(self.interface, "pin_latest_sim_state_for_control_tick", None)
        if callable(pin_snapshot):
            snapshot = pin_snapshot()
        self._control_tick_sim_state_snapshot = snapshot
        self._control_tick_state_pinned = True

    def _release_control_tick_state(self) -> None:
        release_snapshot = getattr(self.interface, "release_control_tick_sim_state", None)
        if callable(release_snapshot):
            release_snapshot()
        self._control_tick_state_pinned = False
        self._control_tick_sim_state_snapshot = None
        self._control_tick_robot_state_data = None

    def _get_control_tick_robot_state(self):
        if bool(getattr(self, "_control_tick_state_pinned", False)):
            return getattr(self, "_control_tick_robot_state_data", None)
        return self.interface.get_low_state()

    def policy_action(self):
        """Execute one control tick within one pinned split-simulator snapshot."""

        self._control_tick_state_pinned = False
        try:
            return self._policy_action_impl()
        finally:
            self._release_control_tick_state()

    def _policy_action_impl(self):
        """Execute policy action and send commands to robot."""

        kp_override = None
        kd_override = None

        # Stage 1: Read State
        with self.latency_tracker.measure("read_state"):
            robot_state_data = self.interface.get_low_state()
            self._pin_control_tick_state(robot_state_data)

        if not self._has_valid_robot_state(robot_state_data):
            if not getattr(self, "_logged_waiting_for_robot_state", False):
                self.logger.info("Waiting for a valid robot state before sending policy commands.")
                self._logged_waiting_for_robot_state = True
            return
        if getattr(self, "_logged_waiting_for_robot_state", False):
            self.logger.info("Valid robot state received; resuming policy command loop.")
            self._logged_waiting_for_robot_state = False

        if (
            self._policy_control_sub is not None
            and not self._allow_noninteractive_autostart_with_policy_control()
            and not self.use_policy_action
            and not self.get_ready_state
        ):
            if not getattr(self, "_logged_waiting_for_external_policy_start", False):
                self.logger.info("Policy control is waiting for external start; not sending lowcmd yet.")
                self._logged_waiting_for_external_policy_start = True
            waiting_overlay_hook = getattr(self, "_publish_waiting_policy_overlay", None)
            if callable(waiting_overlay_hook):
                waiting_overlay_hook(robot_state_data)
            return
        self._logged_waiting_for_external_policy_start = False

        # Stage 2: Pre-processing
        with self.latency_tracker.measure("preprocessing"):
            if (
                self._pending_noninteractive_policy_start
                and not self.use_policy_action
                and self._has_valid_robot_state(robot_state_data)
                and self._can_finish_pending_policy_start(robot_state_data)
            ):
                self.logger.info("Valid robot state received; enabling policy actions.")
                self._pending_noninteractive_policy_start = False
                self._handle_start_policy()
                self._after_auto_start_policy()
            # Determine target joint positions
            if self.get_ready_state:
                q_target = self.get_init_target(robot_state_data)
                self.init_count = min(self.init_count, 500)
            elif not self.use_policy_action:
                manual_cmd = self._get_manual_command(robot_state_data)
                if manual_cmd is not None:
                    q_target = manual_cmd["q"]
                    kp_override = manual_cmd.get("kp")
                    kd_override = manual_cmd.get("kd")
                else:
                    q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
            else:
                # Prepare for inference - any preprocessing before RL inference
                pass

        # Stage 3: Inference
        if self.use_policy_action and not self.get_ready_state:
            with self.latency_tracker.measure("inference"):
                scaled_policy_action = self.rl_inference(robot_state_data)

        # Stage 4: Post-processing
        with self.latency_tracker.measure("postprocessing"):
            if self.use_policy_action and not self.get_ready_state:
                if scaled_policy_action.shape[1] != self.num_dofs:
                    if not self.upper_body_controller:
                        scaled_policy_action = np.concatenate(
                            [np.zeros((1, self.num_dofs - scaled_policy_action.shape[1])), scaled_policy_action], axis=1
                        )
                    else:
                        raise NotImplementedError("Upper body controller not implemented")
                q_target = scaled_policy_action + self.default_dof_angles

            q_target = self._require_finite_array(q_target, label="joint position target")
            if q_target.ndim != 2 or q_target.shape != (1, self.num_dofs):
                raise ValueError(
                    "Joint position target must have shape "
                    f"(1, {self.num_dofs}), got {q_target.shape}."
                )

            # Training/Isaac clips torques, not q targets. Keep q-target clipping opt-in.
            if self._clip_joint_targets and self.q_min_arr is not None and self.q_max_arr is not None:
                np.clip(q_target[0], self.q_min_arr, self.q_max_arr, out=q_target[0])
                self._require_finite_array(q_target, label="clipped joint position target")

            # Prepare command (reuse pre-allocated arrays)
            self.cmd_q[:] = q_target[0]

        # Stage 5: Action Pub
        with self.latency_tracker.measure("action_pub"):
            if bool(getattr(self, "_skip_next_lowcmd_publish", False)):
                self._skip_next_lowcmd_publish = False
                return
            if self.use_policy_action and not self.get_ready_state and kp_override is None and kd_override is None:
                self._sync_policy_pd_with_training()
            self._require_finite_array(self.cmd_q, label="outbound joint position command")
            self._require_finite_array(self.cmd_dq, label="outbound joint velocity command")
            self._require_finite_array(self.cmd_tau, label="outbound joint torque command")
            self.interface.send_low_command(
                self.cmd_q,
                self.cmd_dq,
                self.cmd_tau,
                robot_state_data[0, 7 : 7 + self.num_dofs],
                kp_override=kp_override,
                kd_override=kd_override,
            )

    def _get_manual_command(self, robot_state_data):
        """Optional manual command when policy control is disabled."""
        return

    def _get_obs_phase_time(self):
        """Calculate phase time for gait."""
        cur_time = time.perf_counter() * self.stand_command[0, 0]
        phase_time = cur_time % self.gait_period / self.gait_period
        self.phase_time[:, 0] = phase_time
        return self.phase_time

    def update_phase_time(self):
        """Update phase time."""
        phase_tp1 = self.phase + self.phase_dt
        self.phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi

    # ============================================================================
    # Input Handler Methods
    # ============================================================================

    def start_key_listener(self):
        """Start keyboard listener thread."""

        def on_press(keycode):
            try:
                self.handle_keyboard_button(keycode)
            except AttributeError:
                pass  # Handle special keys if needed

        def on_release(keycode):
            try:
                self.handle_keyboard_release(keycode)
            except AttributeError:
                pass  # Handle special keys if needed

        try:
            if listen_keyboard is None:
                self.logger.warning("sshkeyboard is not installed; keyboard input will not be available")
                return
            listen_keyboard(on_press=on_press, on_release=on_release)
        except OSError as e:
            # Handle termios errors in non-TTY environments
            self.logger.warning("Could not start keyboard listener: %s", e)
            self.logger.warning("Keyboard input will not be available")

    def process_joystick_input(self):
        """Process joystick input and update commands using InterfaceWrapper."""
        # Handle stick input
        self.lin_vel_command, self.ang_vel_command, _ = self.interface.process_joystick_input(
            self.lin_vel_command, self.ang_vel_command, self.stand_command, False
        )
        # Robust key state tracking: update all key states every frame
        self.last_key_states = self.key_states.copy() if hasattr(self, "key_states") else {}
        # Build new key_states: all keys False except the current one
        new_key_states = dict.fromkeys(self.interface._wc_key_map.values(), False)
        cur_key = self.interface.get_joystick_key()
        if cur_key:
            new_key_states[cur_key] = True
        self.key_states = new_key_states
        for key, is_pressed in self.key_states.items():
            if is_pressed and not self.last_key_states.get(key, False):
                self.handle_joystick_button(key)
                self._print_control_status()

    def _process_external_policy_controls(self):
        """Apply start/stop/init/space commands received from the command web."""
        sub = self._policy_control_sub
        if sub is None:
            return
        key_by_action = {
            "start": "]",
            "stop": "o",
            "init": "i",
            "space": "space",
        }
        for action in sub.get_actions():
            self.logger.info("Received external policy control action: {}", action)
            key = key_by_action.get(action)
            if key is not None:
                self.handle_keyboard_button(key)

    # ============================================================================
    # Button Handler Methods
    # ============================================================================

    def handle_keyboard_button(self, keycode):
        """Handle keyboard button presses."""
        if self._try_switch_policy_key(keycode):
            pass
        elif keycode == "]":
            self._handle_start_policy()
        elif keycode == "o":
            self._handle_stop_policy()
        elif keycode == "i":
            self._handle_init_state()
        elif keycode in ["v", "b", "f", "g", "r"]:
            self._handle_kp_control(keycode)

        self._print_control_status()

    def handle_keyboard_release(self, keycode):
        """Handle keyboard button releases."""
        pass

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses."""
        if cur_key == "A":
            self._handle_start_policy()
        elif cur_key == "B":
            self._handle_stop_policy()
        elif cur_key == "Y":
            self._handle_init_state()
        elif cur_key in ["up", "down", "left", "right", "F1"]:
            # TODO: Make this more intuitive
            self._handle_joystick_kp_control(cur_key)
        elif cur_key == "select":
            # Cycle to next policy
            next_index = (self.active_policy_index + 1) % len(self.model_paths)
            self._activate_policy(next_index)
        elif cur_key == "L1+R1":
            # Kill program, works on G1 joystick only.
            self.logger.info(colored("Killing program via joystick command", "red"))
            sys.exit(0)

    # ============================================================================
    # Control Action Methods
    # ============================================================================

    def _handle_start_policy(self):
        """Handle start policy action."""
        self.use_policy_action = True
        self.get_ready_state = False
        self._sync_policy_pd_with_training()
        self.logger.info(colored("Using policy actions", "blue"))
        self.phase = np.array([[0.0, np.pi]])
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _handle_stop_policy(self):
        """Handle stop policy action."""
        self.use_policy_action = False
        self.get_ready_state = False
        self.logger.info("Actions set to zero")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 1

    def _handle_init_state(self):
        """Handle initialization state."""
        self.get_ready_state = True
        self.init_count = 0
        self.logger.info("Setting to init state")
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0

    def _handle_kp_control(self, keycode):
        """Handle keyboard KP control."""
        if keycode == "v":
            self.interface.kp_level -= 0.01
        elif keycode == "b":
            self.interface.kp_level += 0.01
        elif keycode == "f":
            self.interface.kp_level -= 0.1
        elif keycode == "g":
            self.interface.kp_level += 0.1
        elif keycode == "r":
            self.interface.kp_level = 1.0

    def _handle_joystick_kp_control(self, keycode):
        """Handle joystick KP control."""
        print(keycode)
        if keycode == "down":
            self.interface.kp_level -= 0.1
        elif keycode == "up":
            self.interface.kp_level += 0.1
        elif keycode == "left":
            self.interface.kp_level -= 0.01
        elif keycode == "right":
            self.interface.kp_level += 0.01
        elif keycode == "F1":
            self.interface.kp_level = 1.0

    def _print_control_status(self):
        """Print current control status."""
        self.logger.info("------------ Control Status ------------")
        if self.active_model_path:
            total = len(self.model_paths)
            name = Path(self.active_model_path).name
            debug_str = (
                f"Active policy [{self.active_policy_index + 1}/{total}]: {name} Kp level {self.interface.kp_level:.2f}"
            )
            self.logger.info(debug_str)

    # ============================================================================
    # Main Run Method
    # ============================================================================

    def run(self):
        """Main run loop for the policy."""
        try:
            if (
                getattr(self.config.task, "auto_start_policy", False)
                and not self.use_policy_action
                and self._should_auto_start_policy_immediately()
            ):
                self.logger.info("Auto-start enabled: starting policy actions at launch.")
                self._handle_start_policy()

            self._maybe_auto_start_rollout()

            for it in itertools.count():
                self.latency_tracker.start_cycle()

                if self.use_joystick and self.interface.get_joystick_msg() is not None:
                    self.process_joystick_input()
                self._process_external_policy_controls()
                if self.use_phase:
                    self.update_phase_time()

                self.policy_action()

                self.latency_tracker.end_cycle()

                if it % 50 == 0 and self.use_policy_action:
                    debug_str = f"RL FPS: {self.latency_tracker.get_fps():.2f} | {self.latency_tracker.get_stats_str()}"
                    self.logger.info(debug_str, flush=True)

                self.rate.sleep()

        except KeyboardInterrupt:
            pass
