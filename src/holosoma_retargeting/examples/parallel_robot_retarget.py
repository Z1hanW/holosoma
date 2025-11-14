"""
Unified parallel processing script for retargeting all task types:
- robot_only: Robot-only retargeting with ground interaction (LAFAN)
- object_interaction: Object manipulation retargeting (InterMimic)
- climbing: Climbing retargeting with dynamic terrain (MOCAP)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

import numpy as np
import tyro

# Configure for headless operation before any GUI imports
os.environ["MPLBACKEND"] = "Agg"  # Use non-GUI backend for matplotlib
os.environ["DISPLAY"] = ""  # Disable X11 display
os.environ["PYDRAKE_ASSERT_IS_ARMED"] = "0"  # Disable Drake assertions in parallel

# Set matplotlib backend before any imports that might use it
import matplotlib  # type: ignore[import-untyped]

matplotlib.use("Agg", force=True)

# Add src to path for direct execution
import sys

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Import after path modification
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
    extract_foot_sticking_sequence_velocity,
    extract_object_first_moving_frame,
    load_intermimic_data,
    load_object_data,
    preprocess_motion_data,
    transform_from_human_to_world,
    transform_y_up_to_z_up,
)


def create_task_constants(
    robot_config: RobotConfig,
    motion_data_config: MotionDataConfig,
    task_config: "TaskConfig",
    task_type: str,
) -> SimpleNamespace:
    """Create combined task constants from robot and motion data configs."""
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
        task_constants.OBJECT_DIR = object_dir or ""
        task_constants.OBJECT_URDF_FILE = (
            f"{object_dir}/{obj_name}.urdf" if object_dir else f"{obj_name}.urdf"
        )
        task_constants.OBJECT_MESH_FILE = (
            f"{object_dir}/{obj_name}.obj" if object_dir else f"{obj_name}.obj"
        )
        task_constants.SCENE_XML_FILE = ""  # Will be set later

    return task_constants


def find_files(data_dir: Path, data_format: str, object_name: Optional[str] = None):
    """Find files based on data format.
    
    Args:
        data_dir: Directory to search for files
        data_format: Data format ("lafan", "smplh", "mocap")
        object_name: Optional object name to filter files (for smplh format)
    
    Returns:
        Sorted list of file paths
    """
    data_dir = Path(data_dir)
    
    if data_format == "lafan":
        # LAFAN: .npy files in root directory
        pattern = os.path.join(data_dir, "*.npy")
        files = glob.glob(pattern)
        return sorted(files)
    elif data_format == "smplh":
        # SMPLH/OMOMO: .pt files (optionally filtered by object_name)
        if object_name:
            files = [str(p) for p in data_dir.glob(f"*{object_name}*.pt")]
        else:
            files = [str(p) for p in data_dir.glob("*.pt")]
        return sorted(files)
    elif data_format == "mocap":
        # MOCAP: .npy files in subdirectories
        files = [str(p) for p in data_dir.glob("*/*.npy")]
        return sorted(files)
    else:
        raise ValueError(f"Invalid data format: {data_format}")


def extract_task_name(file_path):
    """Extract task name from file path."""
    return Path(file_path).stem


def generate_augmentation_configs(task_type: str, augmentation: bool = True):
    """Generate augmentation configurations based on task type."""
    if task_type == "robot_only":
        # No augmentation for robot_only
        return [{"name": "original"}]
    
    elif task_type == "object_interaction":
        """Generate different augmentation configurations for object interaction."""
        augmentations = []
        augmentations.append({"name": "original", "translation": np.array([0.0, 0.0, 0.0]), "rotation": 0.0})
        
        if augmentation:
            # Translation augmentations
            translations = [
                [0.2, 0.0, 0.0],  # forward
                [0.0, 0.2, 0.0],  # left
                [0.0, -0.2, 0.0],  # right
            ]
            for i, trans in enumerate(translations):
                augmentations.append({"name": f"trans_{i}", "translation": np.array(trans), "rotation": 0.0})
            
            # Rotation augmentations
            rotations = [np.pi / 4, -np.pi / 4]
            for i, rot in enumerate(rotations):
                augmentations.append({
                    "name": f"rot_{i}",
                    "translation": np.array([0.0, 0.2 * (-1) ** i, 0.0]),
                    "rotation": rot,
                })
        
        return augmentations
    
    elif task_type == "climbing":
        """Generate augmentation configurations for climbing (object scaling)."""
        configs = [{"name": "original", "scale": np.array([1, 1, 1])}]
        if augmentation:
            for z_scale in [0.8, 0.9, 1.1, 1.2]:
                configs.append({"name": f"z_scale_{z_scale}", "scale": np.array([1, 1, z_scale])})
        return configs
    
    else:
        raise ValueError(f"Invalid task type: {task_type}")


def process_single_task(args):
    """Process a single task with all augmentations."""
    (
        file_path,
        save_dir,
        task_type,
        robot_config,
        motion_data_config,
        task_config,
        augmentation,
    ) = args
    
    os.makedirs(save_dir, exist_ok=True)
    task_name = extract_task_name(file_path)
    print(f"Processing: {task_name}")

    # Setup object directory for climbing: set default if not provided
    if task_type == "climbing" and task_config.object_dir is None:
        from dataclasses import replace
        task_config = replace(task_config, object_dir=str(Path(file_path).parent))

    # Create task constants
    constants = create_task_constants(
        robot_config, motion_data_config, task_config, task_type
    )

    # Get toe names from motion data config (depends only on data_format)
    toe_names = motion_data_config.TOE_NAMES
    
    # Load data based on task type
    if task_type == "robot_only":
        data_format = motion_data_config.data_format
        if data_format == "lafan":
            human_joints = np.load(file_path)
            human_joints = transform_y_up_to_z_up(human_joints)
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            # LAFAN-specific spine adjustment
            human_joints[:, spine_joint_idx, -1] -= 0.06
            smpl_scale = motion_data_config.DEFAULT_SCALE_FACTOR or 1.0
        elif data_format == "smplh":
            human_joints, object_poses = load_intermimic_data(file_path)
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)
        else:
            raise ValueError(f"Unsupported data format for robot_only: {data_format}")
        
        # Create dummy object poses for robot_only
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))

    elif task_type == "object_interaction":
        human_joints, object_poses = load_intermimic_data(file_path)
        smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)

    elif task_type == "climbing":
        # MOCAP-specific downsample factor
        downsample = 4
        human_joints = np.load(file_path)[::downsample]
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))
        default_human_height = motion_data_config.DEFAULT_HUMAN_HEIGHT or 1.78
        smpl_scale = constants.ROBOT_HEIGHT / default_human_height

    # Setup object data
    object_local_pts = None
    object_local_pts_demo = None
    object_urdf_path = None

    if task_type == "robot_only":
        # Create ground points meshgrid
        x = np.linspace(-5, 5, 15)
        y = np.linspace(-5, 5, 15)
        X, Y = np.meshgrid(x, y)
        ground_pts = np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)
        object_local_pts = ground_pts
        object_local_pts_demo = ground_pts
        object_urdf_path = None

    elif task_type == "object_interaction":
        object_local_pts, object_local_pts_demo = load_object_data(
            constants.OBJECT_MESH_FILE, smpl_scale=smpl_scale, sample_count=100
        )
        object_urdf_path = constants.OBJECT_URDF_FILE

    elif task_type == "climbing":
        box_asset_xml = f"{constants.OBJECT_DIR}/box_assets.xml"
        scene_xml_name = constants.ROBOT_URDF_FILE.split("/")[-1].replace(
            ".urdf", f"_w_{constants.OBJECT_NAME}.xml"
        )
        # Try object directory first, fall back to models directory if not found
        SCENE_XML_FILE = f"{constants.OBJECT_DIR}/{scene_xml_name}"
        if not Path(SCENE_XML_FILE).exists():
            # Use base scene XML from models directory
            base_scene_xml = f"models/{robot_config.robot_type}/{scene_xml_name}"
            if Path(base_scene_xml).exists():
                SCENE_XML_FILE = base_scene_xml
            else:
                raise FileNotFoundError(
                    f"Scene XML file not found at {SCENE_XML_FILE} or {base_scene_xml}"
                )
        # Set SCENE_XML_FILE in constants BEFORE creating retargeter (needed for temp_retargeter)
        constants.SCENE_XML_FILE = SCENE_XML_FILE
        np.random.seed(0)
        object_local_pts, object_local_pts_demo_original = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=smpl_scale,
            surface_weights=lambda p: 20 if p[2] > 0.9 else 1,
            sample_count=100,
        )
        # Ground points for augmentation
        x = np.linspace(-2, 2, 8)
        y = np.linspace(-2, 2, 8)
        X, Y = np.meshgrid(x, y)
        ground_pts = np.stack([X.flatten(), Y.flatten(), np.zeros_like(X.flatten())], axis=1)

    # Create initial retargeter (headless)
    # For climbing, we'll recreate it per augmentation with different URDFs
    if task_type == "climbing":
        # Will be created per augmentation
        retargeter = None
    else:
        # Build retargeter kwargs - only pass nominal_tracking_tau if needed
        retargeter_kwargs = {
            "task_constants": constants,
            "object_urdf_path": object_urdf_path,
            "q_a_init_idx": -7,
            "activate_joint_limits": True,
            "activate_obj_non_penetration": True,
            "activate_foot_sticking": True,
            "penetration_tolerance": 0.001,
            "step_size": 0.2,
            "visualize": False,
            "debug": False,
            "w_nominal_tracking_init": 5.0 if task_type == "object_interaction" else 0.1,
        }
        # Only pass nominal_tracking_tau for climbing (will be set per augmentation)
        # For other tasks, use the default value (10.0)
        retargeter = InteractionMeshRetargeter(**retargeter_kwargs)

    # Preprocess motion data
    # For climbing, create a temporary retargeter for preprocessing
    if task_type == "climbing":
        temp_retargeter = InteractionMeshRetargeter(
            task_constants=constants,
            object_urdf_path=constants.OBJECT_URDF_FILE,
            q_a_init_idx=-7,
            activate_joint_limits=True,
            activate_obj_non_penetration=True,
            activate_foot_sticking=True,
            penetration_tolerance=0.001,
            step_size=0.2,
            visualize=False,
            debug=False,
            w_nominal_tracking_init=0.1,
            nominal_tracking_tau=1e6,
        )
        human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
            human_joints,
            temp_retargeter,
            toe_names,
            smpl_scale,
            object_poses=object_poses,
            normalize_height=False,
        )
    elif task_type == "robot_only":
        human_joints = preprocess_motion_data(human_joints, retargeter, toe_names, smpl_scale)[:400]
        object_poses = object_poses[:400]
    elif task_type == "object_interaction":
        human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
            human_joints, retargeter, toe_names, scale=smpl_scale, object_poses=object_poses
        )

    # Initialize robot pose
    if task_type == "robot_only":
        data_format = motion_data_config.data_format
        if data_format == "lafan":
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            human_quat_init = estimate_human_orientation(human_joints, constants.DEMO_JOINTS)
            q_init_base = np.concatenate(
                [human_quat_init, human_joints[0, spine_joint_idx, :3], np.zeros(constants.ROBOT_DOF)]
            )
        elif data_format == "smplh":
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            q_init_base = np.concatenate(
                [human_quat_init, human_joints[0, 0, :3], np.zeros(constants.ROBOT_DOF)]
            )
    elif task_type == "object_interaction":
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        q_init_base = np.concatenate(
            [human_quat_init, human_joints[0, 0, :3], np.zeros(constants.ROBOT_DOF)]
        )
    elif task_type == "climbing":
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        q_init_base = np.concatenate(
            [
                human_quat_init,
                human_joints[0, temp_retargeter.demo_joints.index("Spine1")],
                np.zeros(constants.ROBOT_DOF),
            ]
        )

    # Extract foot sticking sequences
    # Use temp_retargeter for climbing, regular retargeter for others
    demo_joints_ref = temp_retargeter.demo_joints if task_type == "climbing" else retargeter.demo_joints
    if task_type == "climbing":
        foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
            human_joints, demo_joints_ref, toe_names
        )
    else:
        foot_sticking_sequences = extract_foot_sticking_sequence(
            human_joints, demo_joints_ref, toe_names
        )

    # Process all augmentations
    augmentations = generate_augmentation_configs(task_type, augmentation)
    print("The number of augmentations: ", len(augmentations))

    for k, aug_config in enumerate(augmentations):
        aug_name = aug_config["name"]
        file_name = f"{save_dir}/{task_name}_{aug_name}.npz"

        print(f"  Processing augmentation: {aug_name}")

        # Task-specific augmentation setup
        if task_type == "robot_only":
            # No augmentation, just process original
            object_poses_augmented = object_poses
            q_init = q_init_base
            q_nominal = None
            object_local_pts_final = object_local_pts
            object_local_pts_demo_final = object_local_pts_demo

        elif task_type == "object_interaction":
            # Object pose augmentation
            object_poses_augmented = augment_object_poses(
                object_poses,
                object_moving_frame_idx,
                human_joints[0, 0, :],
                aug_config["translation"],
                aug_config.get("rotation", 0.0),
            )

            # Set initial condition - matches original parallel_robot_object_rt.py logic
            if aug_name == "original":
                # For original: use base initialization
                q_init = np.concatenate([human_quat_init, human_joints[0, 0, :3], np.zeros(constants.ROBOT_DOF)])
                q_nominal = None
                # Disable initial foot sticking
                for foot_name in foot_sticking_sequences[0]:
                    foot_sticking_sequences[0][foot_name] = False
            else:
                # For augmentations: load from original file and use q_nominal[0][retargeter.q_a_indices]
                for foot_name in foot_sticking_sequences[0]:
                    foot_sticking_sequences[0][foot_name] = False
                data = np.load(f"{save_dir}/{task_name}_original.npz")
                q_nominal = data["qpos"]
                q_init = q_nominal[0][retargeter.q_a_indices]

            object_local_pts_final = object_local_pts
            object_local_pts_demo_final = object_local_pts_demo

        elif task_type == "climbing":
            # Object scaling augmentation
            object_scale = aug_config["scale"]
            
            if aug_name == "original":
                object_urdf_file = create_scaled_multi_boxes_urdf(
                    constants.OBJECT_URDF_FILE, object_scale * smpl_scale
                )
                object_local_pts_demo_final = object_local_pts_demo_original
                object_local_pts_final = object_local_pts_demo_final
                q_init = q_init_base
                q_nominal = None
            else:
                object_local_pts_demo_final = np.concatenate([object_local_pts_demo_original, ground_pts], axis=0)
                object_local_pts_final = object_scale * object_local_pts_demo_final
                object_urdf_file = create_scaled_multi_boxes_urdf(
                    constants.OBJECT_URDF_FILE, object_scale * smpl_scale
                )
                data = np.load(f"{save_dir}/{task_name}_original.npz")
                q_nominal = data["qpos"]
                q_init = q_nominal[0]

            # Update XML files
            object_asset_xml_path = create_scaled_multi_boxes_xml(
                box_asset_xml, object_scale * smpl_scale
            )
            # Ensure SCENE_XML_FILE exists and has content before using it
            if not Path(SCENE_XML_FILE).exists() or Path(SCENE_XML_FILE).stat().st_size == 0:
                raise FileNotFoundError(
                    f"Scene XML file is missing or empty: {SCENE_XML_FILE}. "
                    f"Expected file at object directory or models/{robot_config.robot_type}/"
                )
            new_scene_xml_path = create_new_scene_xml_file(
                SCENE_XML_FILE, object_scale * smpl_scale, object_asset_xml_path
            )
            constants.SCENE_XML_FILE = new_scene_xml_path
            
            # Recreate retargeter with new URDF for each augmentation
            retargeter = InteractionMeshRetargeter(
                task_constants=constants,
                object_urdf_path=object_urdf_file,
                q_a_init_idx=-7,
                activate_joint_limits=True,
                activate_obj_non_penetration=True,
                activate_foot_sticking=True,
                penetration_tolerance=0.001,
                step_size=0.2,
                visualize=False,
                debug=False,
                w_nominal_tracking_init=0.1,
                nominal_tracking_tau=1e6,
            )

            object_poses_augmented = object_poses
            object_urdf_path = object_urdf_file

        # Check if file exists and skip retargeting if it does (after setting up conditions)
        if Path.exists(Path(file_name)):
            continue

        # Retarget motion
        retargeted_motions, _, _, _ = retargeter.retarget_motion(
            human_joint_motions=human_joints,
            object_poses=object_poses,
            object_poses_augmented=object_poses_augmented,
            object_points_local_demo=object_local_pts_demo_final,
            object_points_local=object_local_pts_final,
            foot_sticking_sequences=foot_sticking_sequences,
            q_a_init=q_init,
            q_nominal_list=q_nominal,
            original=(k == 0),
            dest_res_path=file_name,
        )

        if k == 0:
            q_a_nominal = np.copy(retargeted_motions)[:, retargeter.q_a_indices]
            q_a_init = q_a_nominal[0]


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
    # Auto-determined from file_path.parent if None (in parallel processing)
    object_dir: Optional[str] = None
    
    # Augmentation parameters (object_interaction task)
    augmentation_frame_count: int = 70
    augmentation_trim_frames: int = 400


@dataclass
class Args:
    """Unified Parallel Robot Retargeting for all task types"""

    # --- Task type selection ---
    task_type: Literal["robot_only", "object_interaction", "climbing"] = "object_interaction"

    # --- top-level run knobs ---
    robot: Literal["g1", "t1", "h1"] = "g1"
    data_format: Optional[Literal["lafan", "smplh", "mocap"]] = None  # Auto-determined by task_type if None
    data_dir: Path = Path("demo_data/OMOMO_new")
    save_dir: Path = Path("demo_results_parallel/g1/object_interaction/omomo")
    augmentation: bool = True
    max_workers: Optional[int] = None  # Auto-determined if None

    # --- robot config (nested - can override robot_urdf_file, robot_dof, etc. via --robot-config.robot-urdf-file) ---
    robot_config: RobotConfig = RobotConfig(robot_type="g1")

    # --- motion data config (nested - can override demo_joints, joints_mapping, etc. via --motion-data-config.demo-joints) ---
    motion_data_config: MotionDataConfig = MotionDataConfig(data_format="lafan", robot_type="g1")

    # --- task config (nested - can override ground_size, surface_weight_threshold, object_name, etc. via --task-config.object-name) ---
    task_config: TaskConfig = TaskConfig()


def main(cfg: Args) -> None:
    # Set defaults based on task type
    default_data_formats = {
        "robot_only": "lafan",
        "object_interaction": "smplh",
        "climbing": "mocap",
    }
    default_save_dirs = {
        "robot_only": "rt_demo_results/lafan/robot_only_parallel",
        "object_interaction": "rt_demo_results/OMOMO_new/largebox_parallel",
        "climbing": "rt_demo_results/climb/climb_parallel",
    }
    data_format = cfg.data_format or default_data_formats[cfg.task_type]
    # Use default save_dir if it matches the default for robot_only (initial default)
    if str(cfg.save_dir) == "rt_demo_results/lafan/robot_only_parallel" and cfg.task_type != "robot_only":
        save_dir = Path(default_save_dirs[cfg.task_type])
    else:
        save_dir = cfg.save_dir
    
    # Set default object_name in task_config if not provided
    if cfg.task_config.object_name is None:
        default_object_names = {
            "robot_only": None,
            "object_interaction": "largebox",
            "climbing": "multi_boxes",
        }
        object_name = default_object_names[cfg.task_type]
        # Create new TaskConfig with default object_name (TaskConfig is frozen, so use replace)
        from dataclasses import replace
        cfg.task_config = replace(cfg.task_config, object_name=object_name)
    else:
        object_name = cfg.task_config.object_name

    os.makedirs(save_dir, exist_ok=True)

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != cfg.robot:
        cfg.robot_config = RobotConfig(robot_type=cfg.robot)

    if cfg.motion_data_config.robot_type != cfg.robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = MotionDataConfig(data_format=data_format, robot_type=cfg.robot)

    # Find files based on data format (use object_name from task_config)
    files = find_files(cfg.data_dir, data_format, cfg.task_config.object_name)
    print(f"Found {len(files)} files for task type: {cfg.task_type}")

    # Pass configs to worker processes
    process_args = [
        (
            file_path,
            save_dir,
            cfg.task_type,
            cfg.robot_config,
            cfg.motion_data_config,
            cfg.task_config,
            cfg.augmentation,
        )
        for file_path in files
    ]

    # Set up parallel processing
    max_workers = cfg.max_workers or mp.cpu_count()
    print(f"Using {max_workers} parallel workers")

    start_time = time.time()
    successful = 0
    failed = 0

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_task, arg): arg[0] for arg in process_args}

        # Process completed tasks
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                print(f"Completed: {file_path}")
                successful += 1
            except Exception as e:
                print(f"Failed {file_path}: {e}")
                import traceback
                traceback.print_exc()
                failed += 1

    end_time = time.time()

    print("\n=== Processing Summary ===")
    print(f"Task type: {cfg.task_type}")
    print(f"Total files: {len(files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    if len(files) > 0:
        print(f"Average time per file: {(end_time - start_time) / len(files):.2f} seconds")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    cfg = tyro.cli(Args)
    main(cfg)

