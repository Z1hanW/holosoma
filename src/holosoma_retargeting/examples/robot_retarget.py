"""
Unified robot retargeting script for all task types:
- robot_only: Robot-only retargeting with ground interaction
- object_interaction: Object manipulation retargeting (InterMimic)
- climbing: Climbing retargeting with dynamic terrain
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional, Tuple, cast

import numpy as np
import tyro

# Add src to path for direct execution
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import new tyro configs
from constants.motion_data_config_simple import MotionDataConfig  # type: ignore[import-not-found]
from constants.robot.unified_robot_config import RobotConfig  # type: ignore[import-not-found]
from interaction_mesh_retargeter import InteractionMeshRetargeter  # type: ignore[import-not-found]
from utils import (  # type: ignore[import-not-found]
    augment_object_poses,
    calculate_scale_factor,
    create_new_scene_xml_file,
    create_scaled_multi_boxes_urdf,
    create_scaled_multi_boxes_xml,
    estimate_human_orientation,
    extract_foot_sticking_sequence,
    extract_object_first_moving_frame,
    load_intermimic_data,
    load_object_data,
    preprocess_motion_data,
    transform_from_human_to_world,
    transform_y_up_to_z_up,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------- Constants -----------------------------

# Task-specific defaults
DEFAULT_DATA_FORMATS = {
    "robot_only": "smplh",
    "object_interaction": "smplh",
    "climbing": "mocap",
}

DEFAULT_SAVE_DIRS = {
    "robot_only": "demo_results/{robot}/robot_only/omomo",
    "object_interaction": "demo_results/{robot}/object_interaction/omomo",
    "climbing": "demo_results/{robot}/climbing/mocap_climb",
}

# ----------------------------- Tyro config -----------------------------


# Constants for numpy arrays (not in dataclass to avoid tyro parsing issues)
_OBJECT_SCALE_AUGMENTED = np.array([1.0, 1.0, 1.2])
_OBJECT_SCALE_NORMAL = np.array([1.0, 1.0, 1.0])
_AUGMENTATION_TRANSLATION = np.array([0.0, -0.2, 0.0])


@dataclass(frozen=True)
class TaskConfig:
    """Task-specific configuration parameters.
    
    These parameters control task-specific behavior like ground meshgrid generation,
    object sampling, augmentation, and scaling. Can be overridden via CLI.
    """

    # Object name 
    # Auto-determined based on task_type if None: "largebox" for object_interaction, 
    # "multi_boxes" for climbing, "ground" for robot_only
    object_name: Optional[str] = None
    
    # Ground meshgrid (robot_only task)
    ground_size: int = 15
    ground_range: tuple[float, float] = (-1.0, 1.0)
    
    # Climbing ground meshgrid (climbing task)
    climbing_ground_size: int = 8
    climbing_ground_range: tuple[float, float] = (-2.0, 2.0)
    
    # Surface weight parameters for climbing object sampling (climbing task)
    # Used in weighted_surface_sampling: points with z-coordinate > threshold get high weight
    # This biases sampling toward top surfaces (important for climbing contact points)
    surface_weight_threshold: float = 0.9  # z-coordinate threshold for high-weight points
    surface_weight_high: int = 20  # Weight for top surface points (z > threshold)
    surface_weight_low: int = 1  # Weight for other points
    
    # Object directory (for climbing tasks)
    # Auto-determined from data_path / task_name if None
    object_dir: Optional[Path] = None
    
    # Augmentation parameters (object_interaction task)
    augmentation_frame_count: int = 70
    augmentation_trim_frames: int = 400


@dataclass
class Args:
    """Unified Robot Retargeting for all task types"""

    # --- Task type selection ---
    task_type: Literal["robot_only", "object_interaction", "climbing"] = "object_interaction"

    # --- top-level run knobs ---
    robot: Literal["g1", "t1"] = "g1"
    data_format: Optional[Literal["lafan", "smplh", "mocap"]] = None  # Auto-determined by task_type if None
    task_name: str = "sub3_largebox_003"
    data_path: Path = Path("demo_data/OMOMO_new")
    save_dir: Optional[Path] = None
    augmentation: bool = False

    # --- robot config (nested - can override robot_urdf_file, robot_dof, etc. via --robot-config.robot-urdf-file) ---
    robot_config: RobotConfig = RobotConfig(robot_type="g1")

    # --- motion data config (nested - can override demo_joints, joints_mapping, etc. via --motion-data-config.demo-joints) ---
    # Note: data_format default will be set based on task_type in main()
    motion_data_config: MotionDataConfig = MotionDataConfig(data_format="smplh", robot_type="g1")

    # --- task config (nested - can override ground_size, surface_weight_threshold, etc. via --task-config.ground-size) ---
    task_config: TaskConfig = TaskConfig()

    # --- retargeter knobs ---
    q_a_init_idx: int = -7
    activate_joint_limits: bool = True
    activate_obj_non_penetration: bool = True
    activate_foot_sticking: bool = True
    penetration_tolerance: float = 0.001
    step_size: float = 0.2
    visualize: bool = True
    debug: bool = True
    w_nominal_tracking_init: float = 5.0
    nominal_tracking_tau: float = 1e6


# ----------------------------- Helper Functions -----------------------------


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    task_config: TaskConfig,
    task_type: str,
) -> SimpleNamespace:
    """Create combined task constants from robot and motion data configs.
    
    Args:
        robot_config: Robot configuration
        motion_data_config: Motion data format configuration
        task_config: Task-specific configuration
        task_type: Type of task ("robot_only", "object_interaction", "climbing")
    
    Returns:
        SimpleNamespace with all task constants
    """
    task_constants = SimpleNamespace()

    # Copy all attributes from robot_config
    for attr in dir(robot_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(task_constants, attr, getattr(robot_config, attr))

    # Copy all attributes from motion_data_config
    for attr in dir(motion_data_config):
        if attr.isupper() and not attr.startswith("_"):
            setattr(task_constants, attr, getattr(motion_data_config, attr))

    # Task-specific object setup
    if task_type == "robot_only":
        obj_name = task_config.object_name or "ground"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = None
        task_constants.OBJECT_MESH_FILE = None
    elif task_type == "object_interaction":
        obj_name = task_config.object_name or "largebox"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = f"models/{obj_name}/{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = f"models/{obj_name}/{obj_name}.obj"
        task_constants.OBJECT_URDF_TEMPLATE = f"models/templates/{obj_name}.urdf.jinja"
    elif task_type == "climbing":
        obj_name = task_config.object_name or "multi_boxes"
        task_constants.OBJECT_NAME = obj_name
        object_dir = task_config.object_dir
        task_constants.OBJECT_DIR = str(object_dir) if object_dir else ""
        task_constants.OBJECT_URDF_FILE = (
            str(object_dir / f"{obj_name}.urdf") if object_dir else f"{obj_name}.urdf"
        )
        task_constants.OBJECT_MESH_FILE = (
            str(object_dir / f"{obj_name}.obj") if object_dir else f"{obj_name}.obj"
        )
        task_constants.SCENE_XML_FILE = ""  # Will be set later

    return task_constants


def validate_config(cfg: Args) -> None:
    """Validate configuration consistency.
    
    Args:
        cfg: Configuration arguments
    
    Raises:
        ValueError: If configuration is invalid
    """
    if cfg.task_type == "climbing" and cfg.data_format not in (None, "mocap"):
        raise ValueError("Climbing task requires 'mocap' data format")
    if cfg.task_type == "object_interaction" and cfg.data_format not in (None, "smplh"):
        raise ValueError("Object interaction requires 'smplh' data format")
    if cfg.task_type == "robot_only" and cfg.data_format not in (None, "lafan", "smplh"):
        raise ValueError("Robot-only task requires 'lafan' or 'smplh' data format")


def create_ground_points(
    x_range: tuple[float, float], y_range: tuple[float, float], size: int
) -> np.ndarray:
    """Create ground point meshgrid.
    
    Args:
        x_range: (min, max) x-coordinate range
        y_range: (min, max) y-coordinate range
        size: Number of points per dimension
    
    Returns:
        (N, 3) array of ground points
    """
    x = np.linspace(x_range[0], x_range[1], size)
    y = np.linspace(y_range[0], y_range[1], size)
    X, Y = np.meshgrid(x, y)
    return np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)


def load_motion_data(
    task_type: Literal["robot_only", "object_interaction", "climbing"],
    data_format: Literal["lafan", "smplh", "mocap"],
    data_path: Path,
    task_name: str,
    constants: SimpleNamespace,
    motion_data_config: MotionDataConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Load motion data based on task type and format.
    
    Args:
        task_type: Type of task
        data_format: Data format ("lafan", "smplh", "mocap")
        data_path: Path to data directory
        task_name: Name of the task/sequence
        constants: Task constants
        motion_data_config: Motion data configuration
    
    Returns:
        Tuple of (human_joints, object_poses, smpl_scale)
        - human_joints: (T, J, 3) array of joint positions
        - object_poses: (T, 7) array of object poses [qw, qx, qy, qz, x, y, z]
        - smpl_scale: Scaling factor for SMPL compatibility
    
    Raises:
        FileNotFoundError: If required data files are not found
    """
    logger.info(f"Loading motion data for task: {task_name}, format: {data_format}")

    if task_type == "robot_only":
        if data_format == "lafan":
            npy_path = data_path / f"{task_name}.npy"
            if not npy_path.exists():
                raise FileNotFoundError(f"LAFAN data file not found: {npy_path}")
            
            human_joints = np.load(str(npy_path))
            human_joints = transform_y_up_to_z_up(human_joints)
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            # LAFAN-specific spine adjustment
            human_joints[:, spine_joint_idx, -1] -= 0.06
            smpl_scale = motion_data_config.DEFAULT_SCALE_FACTOR or 1.0
        else:  # smplh
            pt_path = data_path / f"{task_name}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")
            
            human_joints, object_poses = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)
        
        # Create dummy object poses for robot_only
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))

    elif task_type == "object_interaction":
        pt_path = data_path / f"{task_name}.pt"
        if not pt_path.exists():
            raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")
        
        human_joints, object_poses = load_intermimic_data(str(pt_path))
        smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)

    elif task_type == "climbing":
        task_dir = data_path / task_name
        npy_files = list(task_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy file found in {task_dir}")
        
        npy_file = npy_files[0]
        # MOCAP-specific downsample factor
        downsample = 4
        human_joints = np.load(str(npy_file))[::downsample]
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))
        default_human_height = motion_data_config.DEFAULT_HUMAN_HEIGHT or 1.78
        smpl_scale = constants.ROBOT_HEIGHT / default_human_height

    logger.debug(f"Loaded {human_joints.shape[0]} frames, scale factor: {smpl_scale:.4f}")
    return human_joints, object_poses, smpl_scale


def setup_object_data(
    task_type: Literal["robot_only", "object_interaction", "climbing"],
    constants: SimpleNamespace,
    object_dir: Optional[Path],
    smpl_scale: float,
    task_config: TaskConfig,
    augmentation: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """Setup object-specific data (ground, object mesh, climbing terrain).
    
    Args:
        task_type: Type of task
        constants: Task constants
        object_dir: Object directory path (for climbing)
        smpl_scale: SMPL scaling factor
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
    
    Returns:
        Tuple of (object_local_pts, object_local_pts_demo, object_urdf_path)
    """
    logger.info(f"Setting up object data for task: {task_type}")

    if task_type == "robot_only":
        # Create ground points meshgrid
        ground_pts = create_ground_points(
            task_config.ground_range, task_config.ground_range, task_config.ground_size
        )
        return ground_pts, ground_pts, None

    elif task_type == "object_interaction":
        # Load object data
        if constants.OBJECT_MESH_FILE is None:
            raise ValueError("OBJECT_MESH_FILE not set for object_interaction task")
        
        object_local_pts, object_local_pts_demo = load_object_data(
            constants.OBJECT_MESH_FILE, smpl_scale=smpl_scale, sample_count=100
        )
        return object_local_pts, object_local_pts_demo, constants.OBJECT_URDF_FILE

    elif task_type == "climbing":
        if object_dir is None:
            raise ValueError("object_dir must be provided for climbing task")
        
        # Setup climbing-specific object
        box_asset_xml = object_dir / "box_assets.xml"
        scene_xml_name = Path(constants.ROBOT_URDF_FILE).name.replace(
            ".urdf", f"_w_{constants.OBJECT_NAME}.xml"
        )
        scene_xml_file = object_dir / scene_xml_name
        # Set SCENE_XML_FILE in constants BEFORE creating retargeter (needed for temp_retargeter)
        constants.SCENE_XML_FILE = str(scene_xml_file)

        np.random.seed(0)
        object_local_pts, object_local_pts_demo_original = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=smpl_scale,
            surface_weights=lambda p: task_config.surface_weight_high if p[2] > task_config.surface_weight_threshold else task_config.surface_weight_low,
            sample_count=100,
        )

        if augmentation:
            ground_pts = create_ground_points(
                task_config.climbing_ground_range, task_config.climbing_ground_range, task_config.climbing_ground_size
            )
            object_local_pts_demo = np.concatenate([object_local_pts_demo_original, ground_pts], axis=0)
            object_scale = _OBJECT_SCALE_AUGMENTED
            object_local_pts = object_scale * object_local_pts_demo
        else:
            object_scale = _OBJECT_SCALE_NORMAL
            object_local_pts_demo = object_local_pts_demo_original
            object_local_pts = object_local_pts_demo

        # Create scaled URDF and XML files
        object_urdf_file = create_scaled_multi_boxes_urdf(
            constants.OBJECT_URDF_FILE, object_scale * smpl_scale
        )
        object_asset_xml_path = create_scaled_multi_boxes_xml(
            str(box_asset_xml), object_scale * smpl_scale
        )
        new_scene_xml_path = create_new_scene_xml_file(
            str(scene_xml_file), object_scale * smpl_scale, object_asset_xml_path
        )
        constants.SCENE_XML_FILE = new_scene_xml_path

        return object_local_pts, object_local_pts_demo, object_urdf_file


def build_retargeter_kwargs(
    cfg: Args, constants: SimpleNamespace, object_urdf_path: Optional[str], task_type: str
) -> dict:
    """Build kwargs for InteractionMeshRetargeter.
    
    Args:
        cfg: Configuration arguments
        constants: Task constants
        object_urdf_path: Path to object URDF file
        task_type: Type of task
    
    Returns:
        Dictionary of kwargs for InteractionMeshRetargeter
    """
    kwargs = {
        "task_constants": constants,
        "object_urdf_path": object_urdf_path,
        "q_a_init_idx": cfg.q_a_init_idx,
        "activate_joint_limits": cfg.activate_joint_limits,
        "activate_obj_non_penetration": cfg.activate_obj_non_penetration,
        "activate_foot_sticking": cfg.activate_foot_sticking,
        "penetration_tolerance": cfg.penetration_tolerance,
        "step_size": cfg.step_size,
        "visualize": cfg.visualize,
        "debug": cfg.debug,
        "w_nominal_tracking_init": cfg.w_nominal_tracking_init,
    }
    if task_type == "climbing":
        kwargs["nominal_tracking_tau"] = cfg.nominal_tracking_tau
    return kwargs


def initialize_robot_pose(
    task_type: Literal["robot_only", "object_interaction", "climbing"],
    data_format: Literal["lafan", "smplh", "mocap"],
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter,
    task_config: TaskConfig,
    augmentation: bool,
    save_dir: Path,
    task_name: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Initialize robot pose (q_init, q_nominal) based on task.
    
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses
        constants: Task constants
        retargeter: Retargeter instance
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        save_dir: Save directory path
        task_name: Task name
    
    Returns:
        Tuple of (q_init, q_nominal, object_poses_augmented, human_joints_modified, object_poses_modified)
        where the last two may be trimmed/modified versions of the input arrays
    """
    logger.info("Initializing robot pose")

    if task_type == "robot_only":
        if data_format == "lafan":
            human_quat_init = estimate_human_orientation(human_joints, constants.DEMO_JOINTS)
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            q_init = np.concatenate(
                [human_quat_init, human_joints[0, spine_joint_idx, :3], np.zeros(constants.ROBOT_DOF)]
            )
        elif data_format == "smplh":  # smplh
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            q_init = np.concatenate(
                [human_quat_init, human_joints[0, 0, :3], np.zeros(constants.ROBOT_DOF)]
            )
        return q_init, None, object_poses, human_joints, object_poses

    elif task_type == "object_interaction":
        if augmentation:
            object_moving_frame_idx = extract_object_first_moving_frame(object_poses)
            trim_end = object_moving_frame_idx + task_config.augmentation_frame_count
            human_joints_trimmed = human_joints[:trim_end]
            object_poses_trimmed = object_poses[:trim_end]
            
            object_poses_augmented = augment_object_poses(
                object_poses_trimmed,
                object_moving_frame_idx,
                human_joints_trimmed[0, 0, :],
                _AUGMENTATION_TRANSLATION,
            )
            
            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(
                    f"Original file not found: {original_path}. "
                    "Run without --augmentation first."
                )
            
            data = np.load(str(original_path))
            q_nominal = data["qpos"][:trim_end]
            return None, q_nominal, object_poses_augmented, human_joints_trimmed, object_poses_trimmed
        else:
            object_poses_augmented = object_poses.copy()
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            q_init = np.concatenate(
                [human_quat_init, human_joints[0, 0, :3], np.zeros(constants.ROBOT_DOF)]
            )
            return q_init, None, object_poses_augmented, human_joints, object_poses

    elif task_type == "climbing":
        if augmentation:
            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(
                    f"Original file not found: {original_path}. "
                    "Run without --augmentation first."
                )
            
            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            return q_nominal[0], q_nominal, object_poses, human_joints, object_poses
        else:
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            spine_joint_idx = retargeter.demo_joints.index("Spine1")
            q_init = np.concatenate(
                [
                    human_quat_init,
                    human_joints[0, spine_joint_idx],
                    np.zeros(constants.ROBOT_DOF),
                ]
            )
            return q_init, None, object_poses, human_joints, object_poses

   
def determine_output_path(
    task_type: Literal["robot_only", "object_interaction", "climbing"],
    save_dir: Path,
    task_name: str,
    augmentation: bool,
) -> str:
    """Determine output file path based on task and augmentation.
    
    Args:
        task_type: Type of task
        save_dir: Save directory path
        task_name: Task name
        augmentation: Whether this is an augmentation run
    
    Returns:
        Output file path
    """
    if task_type == "robot_only":
        return str(save_dir / f"{task_name}.npz")
    elif task_type in ("object_interaction", "climbing"):
        suffix = "_augmented" if augmentation else "_original"
        return str(save_dir / f"{task_name}{suffix}.npz")
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# ----------------------------- Main -----------------------------


def main(cfg: Args) -> None:
    """Main retargeting pipeline.
    
    Args:
        cfg: Configuration arguments
    """
    # Validate configuration
    validate_config(cfg)

    robot = cfg.robot
    task_name = cfg.task_name
    task_type = cfg.task_type

    # Set defaults based on task type
    data_format: Literal["lafan", "smplh", "mocap"] = cfg.data_format or cast(Literal["lafan", "smplh", "mocap"], DEFAULT_DATA_FORMATS[task_type])
    save_dir = cfg.save_dir if cfg.save_dir is not None else Path(
        DEFAULT_SAVE_DIRS[task_type].format(robot=robot)
    )
    data_path = cfg.data_path

    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Task: {task_name}, Type: {task_type}, Format: {data_format}")
    logger.info(f"Data path: {data_path}, Save dir: {save_dir}")

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = RobotConfig(robot_type=robot)

    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=robot)

    # Task-specific object setup: set default object_dir for climbing if not provided
    if task_type == "climbing" and cfg.task_config.object_dir is None:
        from dataclasses import replace
        cfg.task_config = replace(cfg.task_config, object_dir=data_path / task_name)

    constants = create_task_constants(
        robot_config=cfg.robot_config,
        motion_data_config=cfg.motion_data_config,
        task_config=cfg.task_config,
        task_type=task_type,
    )

    # Load motion data
    human_joints, object_poses, smpl_scale = load_motion_data(
        task_type, data_format, data_path, task_name, constants, cfg.motion_data_config
    )
    
    # Get toe names from motion data config (depends only on data_format)
    toe_names = cfg.motion_data_config.TOE_NAMES

    # Setup object data
    object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
        task_type, constants, cfg.task_config.object_dir, smpl_scale, cfg.task_config, cfg.augmentation
    )

    # Create retargeter
    retargeter_kwargs = build_retargeter_kwargs(cfg, constants, object_urdf_path, task_type)
    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)
    logger.info("Retargeter created")

    # Preprocess motion data
    if task_type == "robot_only":
        human_joints = preprocess_motion_data(
            human_joints, retargeter, toe_names, smpl_scale
        )[:cfg.task_config.augmentation_trim_frames]
        object_poses = object_poses[:cfg.task_config.augmentation_trim_frames]
    elif task_type == "object_interaction":
        human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
            human_joints, retargeter, toe_names, scale=smpl_scale, object_poses=object_poses
        )
    elif task_type == "climbing":
        human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            smpl_scale,
            object_poses=object_poses,
            normalize_height=False,
        )

    # Initialize robot pose
    q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
        task_type, data_format, human_joints, object_poses, constants, retargeter, cfg.task_config, cfg.augmentation, save_dir, task_name
    )

    # Extract foot sticking sequences
    foot_sticking_sequences = extract_foot_sticking_sequence(
        human_joints, retargeter.demo_joints, toe_names
    )

    # Task-specific foot sticking adjustments
    if task_type == "object_interaction" and not cfg.augmentation:
        # Disable initial sticking
        foot_sticking_sequences[0][toe_names[0]] = False
        foot_sticking_sequences[0][toe_names[1]] = False

    # Determine output path
    dest_res_path = determine_output_path(task_type, save_dir, task_name, cfg.augmentation)

    # Retarget motion
    logger.info("Starting retargeting...")
    retargeter.retarget_motion(
        human_joint_motions=human_joints,
        object_poses=object_poses,
        object_poses_augmented=object_poses_augmented,
        object_points_local_demo=object_local_pts_demo,
        object_points_local=object_local_pts,
        foot_sticking_sequences=foot_sticking_sequences,
        q_a_init=q_init,
        q_nominal_list=q_nominal,
        original=not cfg.augmentation if task_type != "robot_only" else True,
        dest_res_path=dest_res_path,
    )
    logger.info(f"Retargeting complete. Results saved to: {dest_res_path}")

    if cfg.debug:
        input("Press Enter to exit ...")


if __name__ == "__main__":
    cfg = tyro.cli(Args)
    main(cfg)
