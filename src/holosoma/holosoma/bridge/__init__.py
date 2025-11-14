from .base import BasicSdk2Bridge
from .booster import BoosterSdk2Bridge
from .unitree import UnitreeSdk2Bridge


def create_sdk2py_bridge(simulator, robot_config, bridge_config, lcm=None):
    """
    Factory function to create the appropriate SDK2Py bridge based on configuration.

    Now uses robot.bridge.sdk_type for SDK selection instead of bridge_config.type.
    This allows robot-specific bridge parameters (motor_type, message_type, etc.)
    to be properly configured.

    Args:
        simulator: BaseSimulator instance (simulator-agnostic)
        robot_config: Robot configuration dataclass (with .bridge containing RobotBridgeConfig)
        bridge_config: Bridge configuration dataclass (simulator-level settings)
        lcm: LCM instance (optional, for LCM-based bridges)

    Returns:
        An instance of the appropriate bridge class
    """
    # Use robot.bridge.sdk_type instead of bridge_config.type for SDK selection
    sdk_type = robot_config.bridge.sdk_type

    if sdk_type == "unitree":
        return UnitreeSdk2Bridge(simulator, robot_config, bridge_config, lcm)
    if sdk_type == "booster":
        return BoosterSdk2Bridge(simulator, robot_config, bridge_config, lcm)
    if sdk_type == "ros2":
        from .ros2 import ROS2Bridge  # noqa: PLC0415

        return ROS2Bridge(simulator, robot_config, bridge_config, lcm)
    raise ValueError(f"Unsupported SDK type: {sdk_type}")


__all__ = [
    "BasicSdk2Bridge",
    "BoosterSdk2Bridge",
    "UnitreeSdk2Bridge",
    "create_sdk2py_bridge",
]
