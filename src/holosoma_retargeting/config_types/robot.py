"""Configuration types for robot retargeting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np


# Default values per robot type
class RobotDefaults(TypedDict):
    robot_dof: int
    robot_height: float
    object_name: str


_ROBOT_DEFAULTS: dict[str, RobotDefaults] = {
    "g1": {"robot_dof": 29, "robot_height": 1.32, "object_name": "ground"},
    "t1": {"robot_dof": 23, "robot_height": 1.2, "object_name": "ground"},
}


@dataclass(frozen=True)
class RobotConfig:
    """Unified configuration for all robot constants (G1, T1) using tyro.
    
    Uses properties instead of __post_init__ - much simpler!
    
    Example usage:
        # From CLI:
        config = tyro.cli(RobotConfig)  # --robot-type g1 --robot-dof 30
        
        # With defaults:
        config = RobotConfig(robot_type="g1")
        
        # Access values:
        robot_dof = config.ROBOT_DOF
        robot_height = config.ROBOT_HEIGHT
    """

    # Robot type selector - determines which defaults to use
    robot_type: Literal["g1", "t1"] = "g1"
    
    # Robot configuration (optional overrides)
    robot_dof: int | None = None
    robot_height: float | None = None
    robot_name: str | None = None
    robot_urdf_file: str | None = None
    
    # Object configuration
    object_name: str | None = None
    
    # Joint definitions (optional overrides)
    smpl_joints: list[str] | None = None
    foot_sticking_links: list[str] | None = None
    
    # Robot-specific optional fields
    q_a_standing: np.ndarray | None = None  # G1 only
    
    # Manual joint limits
    manual_lb: dict[str, float] | None = None
    manual_ub: dict[str, float] | None = None
    manual_cost: dict[str, float] | None = None
    
    # Nominal tracking indices
    nominal_tracking_indices: np.ndarray | None = None

    # Basic robot properties
    @property
    def ROBOT_DOF(self) -> int:
        """Get robot DOF - use override if provided, else use robot_type default."""
        if self.robot_dof is not None:
            return self.robot_dof
        return _ROBOT_DEFAULTS[self.robot_type]["robot_dof"]

    @property
    def ROBOT_HEIGHT(self) -> float:
        """Get robot height - use override if provided, else use robot_type default."""
        if self.robot_height is not None:
            return self.robot_height
        return _ROBOT_DEFAULTS[self.robot_type]["robot_height"]

    @property
    def ROBOT_NAME(self) -> str:
        """Get robot name - use override if provided, else compute from robot_type and DOF."""
        if self.robot_name is not None:
            return self.robot_name
        return f"{self.robot_type}_{self.ROBOT_DOF}dof"

    @property
    def ROBOT_URDF_FILE(self) -> str:
        """Get robot URDF file path."""
        if self.robot_urdf_file is not None:
            return self.robot_urdf_file
        return f"models/{self.robot_type}/{self.robot_type}_{self.ROBOT_DOF}dof.urdf"

    @property
    def OBJECT_NAME(self) -> str:
        """Get object name - use override if provided, else use robot_type default."""
        if self.object_name is not None:
            return self.object_name
        return _ROBOT_DEFAULTS[self.robot_type]["object_name"]

    @property
    def SMPL_JOINTS(self) -> list[str]:
        """Get SMPL joints - common across all robots."""
        if self.smpl_joints is not None:
            return self.smpl_joints
        return [
            "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
            "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
            "Neck", "L_Thorax", "R_Thorax", "Head",
            "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
            "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
        ]

    @property
    def FOOT_STICKING_LINKS(self) -> list[str]:
        """Get foot sticking links - use override if provided, else use robot_type default."""
        if self.foot_sticking_links is not None:
            return self.foot_sticking_links
        
        if self.robot_type == "g1":
            return [
                "left_ankle_roll_sphere_1_link", "right_ankle_roll_sphere_1_link",
                "left_ankle_roll_sphere_2_link", "right_ankle_roll_sphere_2_link",
                "left_ankle_roll_sphere_3_link", "right_ankle_roll_sphere_3_link",
                "left_ankle_roll_sphere_4_link", "right_ankle_roll_sphere_4_link",
            ]
        elif self.robot_type == "t1":
            return [
                "left_foot_sphere_1_link", "right_foot_sphere_1_link",
                "left_foot_sphere_2_link", "right_foot_sphere_2_link",
                "left_foot_sphere_3_link", "right_foot_sphere_3_link",
                "left_foot_sphere_4_link", "right_foot_sphere_4_link",
                "left_foot_sphere_5_link", "right_foot_sphere_5_link",
            ]
        else:
            raise ValueError(f"Invalid robot type: {self.robot_type}")

    @property
    def Q_A_STANDING(self) -> np.ndarray | None:
        """Get standing pose (G1 only)."""
        if self.q_a_standing is not None:
            return self.q_a_standing
        if self.robot_type == "g1":
            return np.array([
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
                0.0, 0.0, 0.0, 0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
                0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
            ])
        return None

    @property
    def MANUAL_LB(self) -> dict[str, float]:
        """Get manual lower bounds."""
        if self.manual_lb is not None:
            return self.manual_lb
        
        # base: dict[str, float] = {"0": -1.0, "1": -1.0, "2": -1.0, "3": -1.0}  # quaternion bounds
        base: dict[str, float] = {"3": -1.0, "4": -1.0, "5": -1.0, "6": -1.0}  # quaternion bounds
        
        if self.robot_type == "g1":
            base.update({
                "20": -0.3,  # waist roll
                "21": -0.1,  # waist pitch
                "26": -0.1,  # right wrist
                "27": -0.1,
                "28": -0.05,
                "33": -0.1,  # left wrist
                "34": -0.1,
                "35": -0.05,
            })
        
        return base

    @property
    def MANUAL_UB(self) -> dict[str, float]:
        """Get manual upper bounds."""
        if self.manual_ub is not None:
            return self.manual_ub
        
        # base: dict[str, float] = {"0": 1.0, "1": 1.0, "2": 1.0, "3": 1.0}  # quaternion bounds
        base: dict[str, float] = {"3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0}  # quaternion bounds
        
        if self.robot_type == "g1":
            base.update({
                "20": 0.3,  # waist roll
                "25": 1.4,  # right elbow
                "26": 0.2,  # right wrist
                "27": 0.3,
                "28": 0.05,
                "32": 1.4,  # elbow
                "33": 0.2,  # left wrist
                "34": 0.3,
                "35": 0.05,
            })
        
        return base

    @property
    def MANUAL_COST(self) -> dict[str, float]:
        """Get manual cost weights."""
        if self.manual_cost is not None:
            return self.manual_cost
        
        if self.robot_type == "g1":
            return {"19": 0.2, "20": 0.2}  # waist yaw, waist roll
        return {}

    @property
    def NOMINAL_TRACKING_INDICES(self) -> np.ndarray:
        """Get nominal tracking indices."""
        if self.nominal_tracking_indices is not None:
            return self.nominal_tracking_indices
        
        if self.robot_type == "g1":
            return np.arange(19)
        elif self.robot_type == "t1":
            return np.concatenate([np.arange(7), np.arange(11, 23)])
        else:
            raise ValueError(f"Invalid robot type: {self.robot_type}")
