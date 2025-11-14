from pathlib import Path

import numpy as np
import yaml


class RobotConfig:
    def __init__(self, config):
        self.ROBOT_TYPE = config["ROBOT_TYPE"]
        self.MOTOR2JOINT = config["MOTOR2JOINT"]
        self.JOINT2MOTOR = config["JOINT2MOTOR"]
        self.UNITREE_LEGGED_CONST = config.get("UNITREE_LEGGED_CONST", None)
        self.WeakMotorJointIndex = config.get("WeakMotorJointIndex", None)
        self.NUM_MOTORS = config["NUM_MOTORS"]
        self.NUM_JOINTS = config["NUM_JOINTS"]
        self.USE_SENSOR = config["USE_SENSOR"]
        self.MOTOR_EFFORT_LIMIT_LIST = config["motor_effort_limit_list"]
        self.JOINT_VEL_LIMIT_LIST = config["joint_vel_limit_list"]
        self.JOINT_POS_LOWER_LIMIT_LIST = config["joint_pos_lower_limit_list"]
        self.JOINT_POS_UPPER_LIMIT_LIST = config["joint_pos_upper_limit_list"]

        # TODO: This is ok for now, but in the future I'd like to make MOTOR_KP / MOTOR_KD part of the policy
        # configuration, rather than part of robot. Different policies may require different KP/KD settings.
        self.MOTOR_KP = np.array(config["MOTOR_KP"])
        self.MOTOR_KD = np.array(config["MOTOR_KD"])
        self.DEFAULT_MOTOR_ANGLES = config["DEFAULT_MOTOR_ANGLES"]
        self.DEFAULT_DOF_ANGLES = config["DEFAULT_DOF_ANGLES"]


def get_robot_config(robot_type: str) -> RobotConfig:
    current_dir = Path(__file__).parent
    config_path = current_dir / "config" / f"{robot_type}.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return RobotConfig(config["robot"])


# For backwards compatibility
Robot = RobotConfig
