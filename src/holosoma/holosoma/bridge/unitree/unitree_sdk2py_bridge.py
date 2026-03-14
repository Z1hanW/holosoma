import numpy as np
from loguru import logger
from unitree_interface import (
    LowState,
    MessageType,
    MotorCommand,
    RobotType,
    UnitreeInterface,
    WirelessController,
)

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK bridge implementation using unitree_interface C++ bindings."""

    SUPPORTED_ROBOT_TYPES = {"g1_29dof", "h1", "h1-2", "go2_12dof"}

    def _build_default_pose_hold_targets(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        default_joint_angles = getattr(self.robot.init_state, "default_joint_angles", {}) or {}
        stiffness_dict = getattr(self.robot.control, "stiffness", {}) or {}
        damping_dict = getattr(self.robot.control, "damping", {}) or {}

        q = np.zeros(self.num_motor, dtype=np.float32)
        kp = np.zeros(self.num_motor, dtype=np.float32)
        kd = np.zeros(self.num_motor, dtype=np.float32)

        for sim_idx, dof_name in enumerate(self.simulator.dof_names):
            if dof_name in default_joint_angles:
                q[sim_idx] = float(default_joint_angles[dof_name])

            matches = [pattern for pattern in stiffness_dict if pattern in dof_name]
            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly 1 control gain match for '{dof_name}', got {len(matches)}: {matches}"
                )
            pattern = matches[0]
            kp[sim_idx] = float(stiffness_dict[pattern])
            kd[sim_idx] = float(damping_dict[pattern])

        return q, kp, kd

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""

        robot_type = self.robot.asset.robot_type

        # Validate robot type first
        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        # Map robot type to SDK enum
        robot_type_map = {
            "g1_29dof": RobotType.G1,
            "h1": RobotType.H1,
            "h1-2": RobotType.H1_2,
            "go2_12dof": RobotType.GO2,
        }

        # Map to message type (HG for humanoid robots with 35 motors, GO2 for others)
        message_type_map = {
            "g1_29dof": MessageType.HG,
            "h1": MessageType.GO2,
            "h1-2": MessageType.HG,
            "go2_12dof": MessageType.GO2,
        }

        sdk_robot_type = robot_type_map[robot_type]
        sdk_message_type = message_type_map[robot_type]

        # Get network interface from config
        interface_name = self.bridge_config.interface or "eth0"
        # Create interface (handles DDS initialization internally)
        self.interface = UnitreeInterface(interface_name, sdk_robot_type, sdk_message_type)

        # Initialize data structures
        self.low_state = LowState(self.num_motor)
        self.low_cmd = None
        self.wireless_controller = WirelessController()
        self._ignore_default_idle_command = bool(getattr(self.bridge_config, "ignore_default_idle_command", False))
        self._log_first_command_summary = bool(getattr(self.bridge_config, "log_first_command_summary", False))
        self._hold_default_pose_until_first_command = bool(
            getattr(self.bridge_config, "hold_default_pose_until_first_command", False)
        )
        self._hold_initial_pose_until_first_command = bool(
            getattr(self.bridge_config, "hold_initial_pose_until_first_command", False)
        )
        self._logged_first_command_summary = False
        self._logged_active_command_summaries = 0
        self._received_external_active_command = False
        self._logged_default_pose_hold = False
        self._logged_initial_pose_hold = False
        self._idle_low_cmd_signature = self._command_signature(self.interface.create_zero_command())
        self._default_hold_q, self._default_hold_kp, self._default_hold_kd = self._build_default_pose_hold_targets()
        self._default_hold_dq = np.zeros(self.num_motor, dtype=np.float32)
        self._initial_hold_q: np.ndarray | None = None

    def _maybe_capture_initial_hold_targets(self) -> np.ndarray:
        if self._initial_hold_q is None:
            self._initial_hold_q = self.simulator.dof_pos[0].detach().cpu().numpy().astype(np.float32, copy=True)
        return self._initial_hold_q

    @staticmethod
    def _command_signature(cmd: MotorCommand) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.asarray(cmd.q_target, dtype=np.float32),
            np.asarray(cmd.dq_target, dtype=np.float32),
            np.asarray(cmd.kp, dtype=np.float32),
            np.asarray(cmd.kd, dtype=np.float32),
            np.asarray(cmd.tau_ff, dtype=np.float32),
        )

    def _is_default_idle_command(self, cmd: MotorCommand | None) -> bool:
        if cmd is None:
            return True
        cmd_sig = self._command_signature(cmd)
        return all(np.array_equal(lhs, rhs) for lhs, rhs in zip(cmd_sig, self._idle_low_cmd_signature, strict=True))

    @staticmethod
    def _is_active_command(cmd: MotorCommand | None) -> bool:
        if cmd is None:
            return False
        kp = np.asarray(cmd.kp, dtype=np.float32)
        kd = np.asarray(cmd.kd, dtype=np.float32)
        tau_ff = np.asarray(cmd.tau_ff, dtype=np.float32)
        dq_target = np.asarray(cmd.dq_target, dtype=np.float32)
        return bool(
            np.any(np.abs(kp) > 1e-6)
            or np.any(np.abs(kd) > 1e-6)
            or np.any(np.abs(tau_ff) > 1e-6)
            or np.any(np.abs(dq_target) > 1e-6)
        )

    @property
    def received_external_active_command(self) -> bool:
        return bool(getattr(self, "_received_external_active_command", False))

    def low_cmd_handler(self, msg=None):
        """Handle Unitree low-level command messages."""
        # Poll for incoming commands from DDS
        incoming_cmd = self.interface.read_incoming_command()
        if self._ignore_default_idle_command and self._is_default_idle_command(incoming_cmd):
            self.low_cmd = None
            self.torques.fill(0.0)
            return
        if self._is_active_command(incoming_cmd):
            self._received_external_active_command = True
        self.low_cmd = incoming_cmd

    def publish_low_state(self):
        """Publish Unitree low-level state using simulator-agnostic interface."""

        # Get simulator data
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        quaternion, gyro, acceleration = self._get_base_imu_data()

        # Populate motor state
        self.low_state.motor.q = positions.tolist()
        self.low_state.motor.dq = velocities.tolist()
        self.low_state.motor.ddq = accelerations.tolist()
        self.low_state.motor.tau_est = actuator_forces.tolist()

        # Populate IMU state
        # Convert quaternion from torch tensor to list [w, x, y, z]
        quat_array = quaternion.detach().cpu().numpy()
        self.low_state.imu.quat = [
            float(quat_array[0]),  # w
            float(quat_array[1]),  # x
            float(quat_array[2]),  # y
            float(quat_array[3]),  # z
        ]
        self.low_state.imu.omega = gyro.detach().cpu().numpy().tolist()
        self.low_state.imu.accel = acceleration.detach().cpu().numpy().tolist()

        # Set timestamp (milliseconds)
        self.low_state.tick = int(self.sim_time * 1e3)

        # Publish (CRC calculated automatically in C++)
        self.interface.publish_low_state(self.low_state)

    def publish_wireless_controller(self):
        """Publish wireless controller data using unitree_interface."""
        # Call base class to populate wireless_controller from joystick
        super().publish_wireless_controller()

        # Publish using C++ interface
        if self.joystick is not None:
            self.interface.publish_wireless_controller(self.wireless_controller)

    def compute_torques(self):
        """Compute torques using Unitree's unified command structure."""
        cmd_is_active = self._is_active_command(getattr(self, "low_cmd", None))
        if not self._received_external_active_command and self._hold_initial_pose_until_first_command and not cmd_is_active:
            if not self._logged_initial_pose_hold:
                logger.info("Holding simulator initial pose until first external active lowcmd arrives")
                self._logged_initial_pose_hold = True
            return self._compute_pd_torques(
                tau_ff=np.zeros(self.num_motor, dtype=np.float32),
                kp=self._default_hold_kp,
                kd=self._default_hold_kd,
                q_target=self._maybe_capture_initial_hold_targets(),
                dq_target=self._default_hold_dq,
            )
        if not self._received_external_active_command and self._hold_default_pose_until_first_command and not cmd_is_active:
            if not self._logged_default_pose_hold:
                logger.info("Holding default pose until first external active lowcmd arrives")
                self._logged_default_pose_hold = True
            return self._compute_pd_torques(
                tau_ff=np.zeros(self.num_motor, dtype=np.float32),
                kp=self._default_hold_kp,
                kd=self._default_hold_kd,
                q_target=self._default_hold_q,
                dq_target=self._default_hold_dq,
            )

        if not (hasattr(self, "low_cmd") and self.low_cmd):
            if self._hold_initial_pose_until_first_command and not self._received_external_active_command:
                if not self._logged_initial_pose_hold:
                    logger.info("Holding simulator initial pose until first external active lowcmd arrives")
                    self._logged_initial_pose_hold = True
                return self._compute_pd_torques(
                    tau_ff=np.zeros(self.num_motor, dtype=np.float32),
                    kp=self._default_hold_kp,
                    kd=self._default_hold_kd,
                    q_target=self._maybe_capture_initial_hold_targets(),
                    dq_target=self._default_hold_dq,
                )
            if self._hold_default_pose_until_first_command and not self._received_external_active_command:
                if not self._logged_default_pose_hold:
                    logger.info("Holding default pose until first external lowcmd arrives")
                    self._logged_default_pose_hold = True
                return self._compute_pd_torques(
                    tau_ff=np.zeros(self.num_motor, dtype=np.float32),
                    kp=self._default_hold_kp,
                    kd=self._default_hold_kd,
                    q_target=self._default_hold_q,
                    dq_target=self._default_hold_dq,
                )
            self.torques.fill(0.0)
            return self.torques

        try:
            # Extract from Unitree's unified structure
            torques = self._compute_pd_torques(
                tau_ff=self.low_cmd.tau_ff,
                kp=self.low_cmd.kp,
                kd=self.low_cmd.kd,
                q_target=self.low_cmd.q_target,
                dq_target=self.low_cmd.dq_target,
            )
            if self._log_first_command_summary and not self._logged_first_command_summary:
                q_actual = self.simulator.dof_pos[0].detach().cpu().numpy()
                q_target = np.asarray(self.low_cmd.q_target, dtype=np.float32)
                dq_target = np.asarray(self.low_cmd.dq_target, dtype=np.float32)
                kp = np.asarray(self.low_cmd.kp, dtype=np.float32)
                kd = np.asarray(self.low_cmd.kd, dtype=np.float32)
                tau_ff = np.asarray(self.low_cmd.tau_ff, dtype=np.float32)
                is_active_command = self._is_active_command(self.low_cmd)
                if is_active_command and self._logged_active_command_summaries < 5:
                    logger.info(
                        "Active bridge lowcmd summary #{:d}: |q_target-q_actual|max={:.4f}, |tau|max={:.4f}, "
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
        except Exception as e:
            logger.error(f"Error computing torques: {e}")
            raise
