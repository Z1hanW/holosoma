"""Inference policy for exported G1 box sparse-root ONNX policies."""

from __future__ import annotations

import json
import os
from multiprocessing import resource_tracker
from multiprocessing import shared_memory

import numpy as np
import onnx
import onnxruntime
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies.base import BasePolicy


class G1BoxPolicy(BasePolicy):
    """Run single-ONNX G1 box policies with sparse-root actor observations."""

    def __init__(self, config: InferenceConfig):
        self.motion_timestep = 0
        self._obs_input_name: str | None = None
        self._time_step_input_name: str | None = None
        self._perception_obs_input_name: str | None = None
        self._action_output_name: str | None = None
        self._onnx_output_fetch: list[str] = []
        self._perception_shm: shared_memory.SharedMemory | None = None
        self._perception_array: np.ndarray | None = None
        self._perception_dim: int | None = None
        self._logged_zero_perception = False
        self._logged_shm_perception = False
        self.policy_action_scales: np.ndarray | None = None
        super().__init__(config)

    def setup_policy(self, model_path):
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            try:
                metadata[prop.key] = json.loads(prop.value)
            except json.JSONDecodeError:
                metadata[prop.key] = prop.value

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None
        self._set_policy_action_scales_from_metadata(metadata)

        if "obs" in self.onnx_input_names:
            self._obs_input_name = "obs"
        elif "actor_obs" in self.onnx_input_names:
            self._obs_input_name = "actor_obs"
        else:
            raise ValueError(f"Unsupported G1 box ONNX inputs: {self.onnx_input_names}")

        self._time_step_input_name = "time_step" if "time_step" in self.onnx_input_names else None
        self._perception_obs_input_name = "perception_obs" if "perception_obs" in self.onnx_input_names else None

        if "actions" in self.onnx_output_names:
            self._action_output_name = "actions"
        elif "action" in self.onnx_output_names:
            self._action_output_name = "action"
        else:
            self._action_output_name = self.onnx_output_names[0]
        self._onnx_output_fetch = [self._action_output_name]

        def policy_act(input_feed):
            outputs = self.onnx_policy_session.run(self._onnx_output_fetch, input_feed)
            return dict(zip(self._onnx_output_fetch, outputs))

        self.policy = policy_act

    def _capture_policy_state(self) -> dict:
        state = super()._capture_policy_state()
        state.update(
            {
                "obs_input_name": self._obs_input_name,
                "time_step_input_name": self._time_step_input_name,
                "perception_obs_input_name": self._perception_obs_input_name,
                "action_output_name": self._action_output_name,
                "onnx_output_fetch": list(self._onnx_output_fetch),
                "policy_action_scales": None
                if self.policy_action_scales is None
                else self.policy_action_scales.copy(),
            }
        )
        return state

    def _restore_policy_state(self, state: dict):
        super()._restore_policy_state(state)
        self._obs_input_name = state.get("obs_input_name")
        self._time_step_input_name = state.get("time_step_input_name")
        self._perception_obs_input_name = state.get("perception_obs_input_name")
        self._action_output_name = state.get("action_output_name")
        self._onnx_output_fetch = list(state.get("onnx_output_fetch", []))
        policy_action_scales = state.get("policy_action_scales")
        self.policy_action_scales = None if policy_action_scales is None else policy_action_scales.copy()

    @staticmethod
    def _metadata_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    def _set_policy_action_scales_from_metadata(self, metadata: dict) -> None:
        scale_array = np.full((self.num_dofs,), self.policy_action_scale, dtype=np.float32)

        experiment_cfg = metadata.get("experiment_config", {})
        control_cfg = {}
        if isinstance(experiment_cfg, dict):
            robot_cfg = experiment_cfg.get("robot", {})
            if isinstance(robot_cfg, dict):
                control_cfg = robot_cfg.get("control", {}) or {}
        if not isinstance(control_cfg, dict):
            control_cfg = {}

        base_scale = control_cfg.get("action_scale", metadata.get("action_scale"))
        if base_scale is not None:
            self.policy_action_scale = float(base_scale)
            scale_array.fill(self.policy_action_scale)

        by_effort_over_kp = self._metadata_bool(
            control_cfg.get(
                "action_scales_by_effort_limit_over_p_gain",
                metadata.get("action_scales_by_effort_limit_over_p_gain", False),
            )
        )
        if not by_effort_over_kp:
            self.policy_action_scales = scale_array.reshape(1, -1)
            logger.info(
                "Using G1 box scalar action scale: base={} final_min={:.6f} final_max={:.6f}",
                self.policy_action_scale,
                float(np.min(scale_array)),
                float(np.max(scale_array)),
            )
            return

        motor_kp = self.onnx_kp
        motor_effort = getattr(self.robot_config, "motor_effort_limit", None)
        if motor_kp is None or motor_effort is None:
            logger.warning(
                "G1 box metadata requested per-joint action scaling, but kp or effort limits were unavailable."
            )
            self.policy_action_scales = scale_array.reshape(1, -1)
            return

        motor_kp = np.asarray(motor_kp, dtype=np.float32)
        motor_effort = np.asarray(motor_effort, dtype=np.float32)
        if motor_kp.shape[0] != self.robot_config.num_motors or motor_effort.shape[0] != self.robot_config.num_motors:
            logger.warning(
                "Skipping G1 box per-joint action scaling due to shape mismatch: kp={}, effort={}, num_motors={}",
                motor_kp.shape,
                motor_effort.shape,
                self.robot_config.num_motors,
            )
            self.policy_action_scales = scale_array.reshape(1, -1)
            return

        joint2motor = np.asarray(self.robot_config.joint2motor, dtype=np.int64)
        for joint_idx in range(self.num_dofs):
            motor_idx = int(joint2motor[joint_idx])
            stiffness = float(motor_kp[motor_idx])
            effort = float(motor_effort[motor_idx])
            scale_array[joint_idx] = 0.0 if stiffness == 0.0 else self.policy_action_scale * effort / stiffness

        self.policy_action_scales = scale_array.reshape(1, -1)
        logger.info(
            "Using G1 box per-joint action scales from ONNX metadata: "
            "base={} final_min={:.6f} final_max={:.6f} final_mean={:.6f}",
            self.policy_action_scale,
            float(np.min(scale_array)),
            float(np.max(scale_array)),
            float(np.mean(scale_array)),
        )

    def _get_onnx_input_dim(self, input_name: str | None) -> int | None:
        if input_name is None:
            return None
        for inp in self.onnx_policy_session.get_inputs():
            if inp.name == input_name:
                shape = inp.shape
                if len(shape) > 1 and isinstance(shape[1], int):
                    return int(shape[1])
        return None

    def _ensure_perception_shm(self, expected_dim: int) -> bool:
        if (
            self._perception_shm is not None
            and self._perception_array is not None
            and self._perception_dim == int(expected_dim)
        ):
            return True

        self._close_perception_shm()
        shm_name = os.environ.get("HOLOSOMA_BOX_POLICY_PERCEPTION_SHM_NAME", "depth_img_shm")
        try:
            self._perception_shm = shared_memory.SharedMemory(name=shm_name, create=False)
        except FileNotFoundError:
            return False

        try:
            resource_tracker.unregister(self._perception_shm._name, "shared_memory")
        except Exception:
            pass

        expected_bytes = int(expected_dim) * np.dtype(np.float32).itemsize
        if len(self._perception_shm.buf) < expected_bytes:
            self._close_perception_shm()
            return False

        self._perception_dim = int(expected_dim)
        self._perception_array = np.ndarray((self._perception_dim,), dtype=np.float32, buffer=self._perception_shm.buf)
        return True

    def _close_perception_shm(self) -> None:
        shm = self._perception_shm
        self._perception_shm = None
        self._perception_array = None
        self._perception_dim = None
        if shm is not None:
            shm.close()

    def _get_perception_obs(self, expected_dim: int | None) -> np.ndarray | None:
        if expected_dim is None:
            return None
        if self._ensure_perception_shm(int(expected_dim)) and self._perception_array is not None:
            if not self._logged_shm_perception:
                logger.info("Using G1 box perception_obs from shared memory with {} values", expected_dim)
                self._logged_shm_perception = True
            return self._perception_array.copy().reshape(1, int(expected_dim)).astype(np.float32, copy=False)

        if not self._logged_zero_perception:
            logger.warning(
                "G1 box policy expected perception_obs, but shared memory was not available; using zeros."
            )
            self._logged_zero_perception = True
        return np.zeros((1, int(expected_dim)), dtype=np.float32)

    def _get_sparse_root_command(self) -> np.ndarray:
        raw = os.environ.get("HOLOSOMA_BOX_SPARSE_ROOT_COMMAND", "").strip()
        if raw:
            try:
                values = [float(part.strip()) for part in raw.split(",")]
                if len(values) >= 3:
                    return np.asarray([values[:3]], dtype=np.float32)
            except ValueError:
                pass
        return np.zeros((1, 3), dtype=np.float32)

    def get_current_obs_buffer_dict(self, robot_state_data):
        sparse_root_command = self._get_sparse_root_command()
        return {
            "sparse_target_root_trajectory_command": sparse_root_command,
            "sparse_target_root_trajectory_command_contact_aware": sparse_root_command,
            "base_lin_vel": robot_state_data[:, 7 + self.num_dofs : 7 + self.num_dofs + 3],
            "base_ang_vel": robot_state_data[:, 7 + self.num_dofs + 3 : 7 + self.num_dofs + 6],
            "dof_pos": robot_state_data[:, 7 : 7 + self.num_dofs] - self.default_dof_angles,
            "dof_vel": robot_state_data[
                :, 7 + self.num_dofs + 6 : 7 + self.num_dofs + 6 + self.num_dofs
            ],
            "actions": self.last_policy_action,
        }

    def prepare_obs_for_rl(self, robot_state_data):
        group_outputs = self._prepare_group_observations(robot_state_data)
        actor_groups = [
            group for group in self.obs_dict if group in group_outputs and group.startswith("actor_obs")
        ]
        if not actor_groups:
            raise KeyError("G1 box policy requires actor_obs* observation groups.")
        actor_obs = np.concatenate([group_outputs[group] for group in actor_groups], axis=1)
        return {"actor_obs": actor_obs.astype(np.float32, copy=False)}

    def rl_inference(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        input_feed = {self._obs_input_name: obs["actor_obs"]}
        if self._time_step_input_name is not None:
            input_feed[self._time_step_input_name] = np.array([[self.motion_timestep]], dtype=np.float32)
        if self._perception_obs_input_name is not None:
            perception_dim = self._get_onnx_input_dim(self._perception_obs_input_name)
            input_feed[self._perception_obs_input_name] = self._get_perception_obs(perception_dim)

        outputs = self.policy(input_feed)
        policy_action = np.clip(outputs[self._action_output_name], -100, 100)
        self.last_policy_action = policy_action.copy()
        action_scales = self.policy_action_scales
        if action_scales is None:
            action_scales = np.full((1, self.num_dofs), self.policy_action_scale, dtype=np.float32)
        self.scaled_policy_action = policy_action * action_scales
        self.motion_timestep += 1
        return self.scaled_policy_action
