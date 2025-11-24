from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.utils.crc import CRC

from holosoma.bridge.base.basic_sdk2py_bridge import BasicSdk2Bridge


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK2Py bridge implementation."""

    SUPPORTED_ROBOT_TYPES = {"g1_29dof", "h1", "h1-2", "go2_12dof"}

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""

        # Initialize Unitree SDK factory (required before creating channels)
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: PLC0415

        # Use bridge config for domain_id and interface
        domain_id = self.bridge_config.domain_id
        interface = self.bridge_config.interface

        if interface:
            ChannelFactoryInitialize(domain_id, interface)
        else:
            ChannelFactoryInitialize(domain_id)

        robot_type = self.robot.asset.robot_type

        # Validate robot type first
        if robot_type not in self.SUPPORTED_ROBOT_TYPES:
            raise ValueError(f"Invalid robot type '{robot_type}'. Unitree SDK supports: {self.SUPPORTED_ROBOT_TYPES}")

        # Initialize based on robot type
        if robot_type in {"g1_29dof"} or "h1-2" in robot_type:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_  # noqa: PLC0415
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default  # noqa: PLC0415
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_  # noqa: PLC0415

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif robot_type in {"h1", "go2_12dof"}:
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_  # noqa: PLC0415
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default  # noqa: PLC0415
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_  # noqa: PLC0415

            self.low_cmd = unitree_go_msg_dds__LowCmd_()

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.low_cmd_handler, 1)

        # Initialize crc
        self.crc = CRC()

        # Initialize wireless controller for Unitree
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
        self.wireless_controller_puber.Init()

    def low_cmd_handler(self, msg):
        """Handle Unitree low-level command messages."""
        if msg:
            self.low_cmd = msg

    def publish_low_state(self):
        """Publish Unitree low-level state using simulator-agnostic interface."""

        num_motors = self.num_motor
        motor_state = self.low_state.motor_state

        # Use ground truth data (not sensor)
        positions, velocities, accelerations = self._get_dof_states()
        actuator_forces = self._get_actuator_forces()
        for i in range(num_motors):
            m = motor_state[i]
            m.q = positions[i]
            m.dq = velocities[i]
            m.ddq = accelerations[i]
            m.tau_est = actuator_forces[i]

        imu = self.low_state.imu_state
        quaternion, gyro, acceleration = self._get_base_imu_data()
        imu.quaternion[:] = quaternion.detach().cpu().numpy()
        imu.gyroscope[:] = gyro.detach().cpu().numpy()
        imu.accelerometer[:] = acceleration.detach().cpu().numpy()

        self.low_state.tick = int(self.sim_time * 1e3)
        self.low_state.crc = self.crc.Crc(self.low_state)
        self.low_state_puber.Write(self.low_state)
