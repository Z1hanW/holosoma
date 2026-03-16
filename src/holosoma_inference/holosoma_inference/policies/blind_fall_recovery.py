"""Blind fall recovery policy for proprioceptive-only reactive standing.

Loads a single ONNX model (student.onnx) trained with the Blind-FallRecovery
curriculum from far_tracking. No depth camera, no velocity commands, no motion
references -- purely reactive balance recovery from proprioceptive observations.

Observations: projected_gravity (3) + base_ang_vel (3) + dof_pos (29) + dof_vel (29) + actions (29) = 93

Push injection: R1 + right joystick applies feedforward torques on hip joints
to simulate external pushes during testing.
"""

from __future__ import annotations

import json
import os

import numpy as np
import onnx
import onnxruntime
from loguru import logger

from holosoma_inference.config.config_types.inference import InferenceConfig
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.utils.math.quat import quat_from_angle_axis, quat_mul


class BlindFallRecoveryPolicy(BasePolicy):
    """Blind (proprioceptive-only) fall recovery policy.

    Single ONNX model architecture -- no depth backbone, no velocity commands.
    Includes R1 + right joystick push injection for testing robustness.
    """

    # Push injection config
    MAX_PUSH_TORQUE = 30.0  # Nm, max torque applied per hip joint

    def __init__(self, config: InferenceConfig):
        self._stiff_hold_active = True
        self._damping_mode_active = False

        super().__init__(config)

        # Load stiff startup parameters from robot config
        if config.robot.stiff_startup_pos is not None:
            self._stiff_hold_q = np.array(config.robot.stiff_startup_pos, dtype=np.float32).reshape(1, -1)
        else:
            self._stiff_hold_q = np.array(config.robot.default_dof_angles, dtype=np.float32).reshape(1, -1)

        if config.robot.stiff_startup_kp is not None:
            self._stiff_hold_kp = np.array(config.robot.stiff_startup_kp, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kp for BlindFallRecoveryPolicy")

        if config.robot.stiff_startup_kd is not None:
            self._stiff_hold_kd = np.array(config.robot.stiff_startup_kd, dtype=np.float32)
        else:
            raise ValueError("Robot config must specify stiff_startup_kd for BlindFallRecoveryPolicy")

        if self._stiff_hold_q.shape[1] != self.num_dofs:
            raise ValueError("Stiff startup pose dimension mismatch with robot DOFs")

        # Hip joint indices in robot order for push injection.
        # These go directly to motors (cmd_tau), not through the policy.
        self._hip_pitch_indices = [
            self.dof_names.index("left_hip_pitch_joint"),
            self.dof_names.index("right_hip_pitch_joint"),
        ]
        self._hip_roll_indices = [
            self.dof_names.index("left_hip_roll_joint"),
            self.dof_names.index("right_hip_roll_joint"),
        ]

    # ------------------------------------------------------------------
    # Policy components override (single model, joint reordering, waist FK)
    # ------------------------------------------------------------------

    def _init_policy_components(self, model_path, policy_action_scale, rl_rate):
        """Load a single student ONNX model and set up joint reordering + waist FK."""
        self.policy_action_scale = policy_action_scale
        self.rl_rate = rl_rate

        self.model_paths = self._collect_model_paths(model_path)
        resolved_paths = []
        for path in self.model_paths:
            local_path = self._resolve_model_path(str(path))
            resolved_paths.append(local_path)
        self.model_paths = resolved_paths

        # Load (first/only) student model
        self._load_student_model(self.model_paths[0])

        # Joint reordering between ONNX model order and robot order
        self._setup_joint_reordering()

        # Waist FK for anchor-body projected gravity
        self._init_waist_joint_indices()

        # Action buffers
        self.last_policy_action = np.zeros((1, self.num_dofs))
        self.scaled_policy_action = np.zeros((1, self.num_dofs))

        # Multi-policy support
        self._policy_states = [self._capture_policy_state()]
        self.active_policy_index = 0
        self.active_model_path = self.model_paths[0]

        self._resolve_control_gains()

    def _load_student_model(self, model_path: str):
        """Load the student ONNX model and extract metadata."""
        self.onnx_policy_session = onnxruntime.InferenceSession(model_path)
        self.onnx_input_names = [inp.name for inp in self.onnx_policy_session.get_inputs()]
        self.onnx_output_names = [out.name for out in self.onnx_policy_session.get_outputs()]

        onnx_model = onnx.load(model_path)
        metadata = {}
        for prop in onnx_model.metadata_props:
            try:
                metadata[prop.key] = json.loads(prop.value)
            except (json.JSONDecodeError, ValueError):
                metadata[prop.key] = prop.value

        self.onnx_kp = np.array(metadata["kp"]) if "kp" in metadata else None
        self.onnx_kd = np.array(metadata["kd"]) if "kd" in metadata else None

        if self.onnx_kp is not None:
            logger.info(f"Loaded KP/KD from ONNX metadata: {os.path.basename(model_path)}")

        # Joint names for reordering
        raw = metadata.get("joint_names", None)
        if raw is not None and isinstance(raw, str):
            self._model_joint_names = raw.split(",")
        elif raw is not None and isinstance(raw, list):
            self._model_joint_names = raw
        else:
            self._model_joint_names = None

        # Action scale from metadata
        if "action_scale" in metadata:
            action_scale = metadata["action_scale"]
            if isinstance(action_scale, (int, float)):
                self.policy_action_scale = float(action_scale)
                logger.info(f"Using action_scale from ONNX metadata: {self.policy_action_scale}")

        def policy_act(obs_dict):
            input_feed = {name: obs_dict[name] for name in self.onnx_input_names}
            outputs = self.onnx_policy_session.run(self.onnx_output_names, input_feed)
            return outputs[0]

        self.policy = policy_act

        logger.info(
            f"[BlindFallRecoveryPolicy] Student model loaded: "
            f"inputs={self.onnx_input_names}, outputs={self.onnx_output_names}"
        )

    # ------------------------------------------------------------------
    # Joint reordering (copied from DepthDistillationPolicy)
    # ------------------------------------------------------------------

    def _setup_joint_reordering(self):
        """Compute joint reordering indices between robot and model joint orders.

        Uses ``joint_names`` from the ONNX model metadata (the training joint order)
        and ``self.robot_config.dof_names`` (the robot's canonical joint order).
        """
        model_joint_names = self._model_joint_names
        real_joint_names = list(self.robot_config.dof_names)

        if model_joint_names is None or list(model_joint_names) == real_joint_names:
            self._real2model_index = None
            self._model2real_index = None
            return

        from holosoma_inference.utils.math.misc import get_index_of_a_in_b

        self._real2model_index = get_index_of_a_in_b(model_joint_names, real_joint_names)

        n = len(self._real2model_index)
        self._model2real_index = [0] * n
        for model_pos, real_pos in enumerate(self._real2model_index):
            self._model2real_index[real_pos] = model_pos

        logger.info(
            f"[BlindFallRecoveryPolicy] Joint reordering enabled: "
            f"model={model_joint_names}, real={real_joint_names}"
        )

    # ------------------------------------------------------------------
    # Anchor-body FK for projected gravity (copied from DepthDistillationPolicy)
    # ------------------------------------------------------------------

    def _init_waist_joint_indices(self):
        """Pre-compute waist joint indices for anchor-body FK.

        Chains waist_yaw -> waist_roll -> waist_pitch to get the torso_link
        quaternion used for ``robot_anchor_projected_gravity``.
        """
        waist_chain = [
            ("waist_yaw_joint", np.array([0.0, 0.0, 1.0])),
            ("waist_roll_joint", np.array([1.0, 0.0, 0.0])),
            ("waist_pitch_joint", np.array([0.0, 1.0, 0.0])),
        ]

        self._waist_joint_info: list[tuple[int, np.ndarray]] = []
        for name, axis in waist_chain:
            if name in self.dof_names:
                self._waist_joint_info.append((self.dof_names.index(name), axis))

        if self._waist_joint_info:
            names = [self.dof_names[idx] for idx, _ in self._waist_joint_info]
            logger.info(f"[BlindFallRecoveryPolicy] Anchor-body FK through waist joints: {names}")

    def _get_gravity_frame_quat(self, robot_state_data, base_quat):
        """Compute the anchor body (torso) quaternion for projected gravity."""
        if not self._waist_joint_info:
            return base_quat

        raw_dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]

        anchor_quat = base_quat.copy()
        for idx, axis in self._waist_joint_info:
            angle = float(raw_dof_pos[0, idx])
            joint_quat = quat_from_angle_axis(angle, axis)
            anchor_quat = quat_mul(anchor_quat, joint_quat)

        return anchor_quat

    # ------------------------------------------------------------------
    # Observation overrides
    # ------------------------------------------------------------------

    def get_current_obs_buffer_dict(self, robot_state_data):
        """Build observation buffer, reordering dof_pos/dof_vel to model order."""
        current_obs_buffer_dict = super().get_current_obs_buffer_dict(robot_state_data)

        # Reorder joint observations to model order if needed
        if self._real2model_index is not None:
            current_obs_buffer_dict["dof_pos"] = current_obs_buffer_dict["dof_pos"][:, self._real2model_index]
            current_obs_buffer_dict["dof_vel"] = current_obs_buffer_dict["dof_vel"][:, self._real2model_index]

        # Add previous actions (already in model order from last inference step)
        current_obs_buffer_dict["actions"] = self.last_policy_action

        return current_obs_buffer_dict

    def prepare_obs_for_rl(self, robot_state_data):
        """Build flat observation: [projected_gravity, base_ang_vel, dof_pos, dof_vel, actions].

        Returns dict with 'actor_obs' key matching the ONNX input name.
        """
        group_outputs = self._prepare_group_observations(robot_state_data)
        if "actor_obs" not in group_outputs:
            raise KeyError("Observation group 'actor_obs' is not configured for this policy.")
        return {"actor_obs": group_outputs["actor_obs"].astype(np.float32, copy=False)}

    # ------------------------------------------------------------------
    # Inference override
    # ------------------------------------------------------------------

    def rl_inference(self, robot_state_data):
        """Run single ONNX model, reorder actions from model order to robot order."""
        obs_dict = self.prepare_obs_for_rl(robot_state_data)

        if self.config.task.print_observations:
            self._print_observations(obs_dict)

        policy_action = self.policy(obs_dict)
        policy_action = np.clip(policy_action, -100, 100)

        # Store in model order for feedback as "actions" observation
        self.last_policy_action = policy_action.copy()

        # Reorder to robot order for motor commands
        if self._model2real_index is not None:
            policy_action = policy_action[:, self._model2real_index]

        self.scaled_policy_action = policy_action * self.policy_action_scale
        return self.scaled_policy_action

    # ------------------------------------------------------------------
    # Push injection via R1 + right joystick
    # ------------------------------------------------------------------

    def process_joystick_input(self):
        """Process joystick, then apply push injection when R1 is held."""
        super().process_joystick_input()

        # Read R1 state from key_states (populated by base class)
        r1_held = self.key_states.get("R1", False)

        if r1_held:
            wc_msg = self.interface.get_joystick_msg()
            if wc_msg is not None:
                # Right stick axes
                rx = getattr(wc_msg, "rx", 0.0)
                ry = getattr(wc_msg, "ry", 0.0)

                # Forward/back push -> hip pitch torques
                pitch_torque = ry * self.MAX_PUSH_TORQUE
                # Lateral push -> hip roll torques
                roll_torque = rx * self.MAX_PUSH_TORQUE

                # Apply same-sign torque to both left and right hips
                # (simulates external force on torso, not a twist)
                for idx in self._hip_pitch_indices:
                    self.cmd_tau[idx] = pitch_torque
                for idx in self._hip_roll_indices:
                    self.cmd_tau[idx] = roll_torque

                if abs(pitch_torque) > 0.1 or abs(roll_torque) > 0.1:
                    logger.debug(
                        f"Push injection: pitch={pitch_torque:.1f} Nm, roll={roll_torque:.1f} Nm"
                    )
        else:
            # R1 released -> zero feedforward torques
            self.cmd_tau[:] = 0.0

    # ------------------------------------------------------------------
    # Stiff hold / damping modes
    # ------------------------------------------------------------------

    def _get_manual_command(self, robot_state_data):
        """Return manual command when policy is not active."""
        if self._stiff_hold_active:
            return {
                "q": self._stiff_hold_q.copy(),
                "kp": self._stiff_hold_kp,
                "kd": self._stiff_hold_kd,
            }
        if self._damping_mode_active:
            dof_pos = robot_state_data[:, 7 : 7 + self.num_dofs]
            return {
                "q": dof_pos,
                "kp": np.zeros_like(self._stiff_hold_kp),
                "kd": self._stiff_hold_kd,
            }
        return None

    # ------------------------------------------------------------------
    # Button handlers
    # ------------------------------------------------------------------

    def handle_joystick_button(self, cur_key):
        """Handle joystick button presses."""
        if cur_key == "start":
            self._handle_enter_stiff_hold()
        else:
            super().handle_joystick_button(cur_key)

    def _handle_enter_stiff_hold(self):
        """Enter stiff hold: hold standing pose with high Kp/Kd."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = True
        self._damping_mode_active = False
        self.cmd_tau[:] = 0.0
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        logger.info("Entering stiff hold mode")

    def _handle_start_policy(self):
        """Start the policy."""
        super()._handle_start_policy()
        self._stiff_hold_active = False
        self._damping_mode_active = False
        logger.info("Blind fall recovery policy started")

    def _handle_stop_policy(self):
        """Enter damping mode: Kp=0, Kd>0."""
        self.use_policy_action = False
        self.get_ready_state = False
        self._stiff_hold_active = False
        self._damping_mode_active = True
        self.cmd_tau[:] = 0.0
        if hasattr(self.interface, "no_action"):
            self.interface.no_action = 0
        logger.info("Entering damping mode (Kp=0, Kd>0)")
