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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import tyro

src_root = Path(__file__).resolve().parents[2]
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from holosoma_retargeting.config_types.data_type import DEMO_JOINTS_REGISTRY, MotionDataConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeter import RetargeterConfig  # noqa: E402
from holosoma_retargeting.config_types.retargeting import RetargetingConfig  # noqa: E402
from holosoma_retargeting.config_types.robot import RobotConfig  # noqa: E402
from holosoma_retargeting.config_types.task import TaskConfig  # noqa: E402
from holosoma_retargeting.src.interaction_mesh_retargeter import (  # noqa: E402
    InteractionMeshRetargeter,  # type: ignore[import-not-found]
)
from holosoma_retargeting.src.utils import (  # noqa: E402
    augment_object_poses,
    calculate_scale_factor,
    create_new_scene_xml_file,
    create_scaled_multi_boxes_urdf,
    create_scaled_multi_boxes_xml,
    estimate_human_orientation,
    extract_foot_sticking_sequence_velocity,
    extract_foot_sticking_sequence_contact_aware,
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
    "w_obj": "smplh",
    "w-obj-scale": "seedance",
    "climbing": "mocap",
}

DEFAULT_SAVE_DIRS = {
    "robot_only": "demo_results/{robot}/robot_only/omomo",
    "object_interaction": "demo_results/{robot}/object_interaction/omomo",
    "w_obj": "demo_results/{robot}/object_interaction/omomo",
    "w-obj-scale": "demo_results/{robot}/object_interaction/omomo",
    "climbing": "demo_results/{robot}/climbing/mocap_climb",
}


# Constants for numpy arrays (not in dataclass to avoid tyro parsing issues)
_OBJECT_SCALE_AUGMENTED = np.array([1.0, 1.0, 1.2])
_OBJECT_SCALE_NORMAL = np.array([1.0, 1.0, 1.0])
_AUGMENTATION_TRANSLATION = np.array([0.2, 0.0, 0.0])


# Type aliases
TaskType = Literal["robot_only", "object_interaction", "w_obj", "w-obj-scale", "climbing"]
# DataFormat is imported from config_types.data_type


def _normalized_task_type(task_type: str) -> str:
    return task_type.replace("_", "-")


def _is_object_task(task_type: str) -> bool:
    return _normalized_task_type(task_type) in {"object-interaction", "w-obj", "w-obj-scale"}


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

    # Copy legacy motion data constants (upper-case for compatibility)
    for attr, value in motion_data_config.legacy_constants().items():
        setattr(task_constants, attr, value)

    # Task-specific object setup
    if task_type == "robot_only":
        obj_name = task_config.object_name or "ground"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = None
        task_constants.OBJECT_MESH_FILE = None
    elif _is_object_task(task_type):
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
        task_constants.OBJECT_URDF_FILE = str(object_dir / f"{obj_name}.urdf") if object_dir else f"{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = str(object_dir / f"{obj_name}.obj") if object_dir else f"{obj_name}.obj"
        task_constants.SCENE_XML_FILE = ""  # Will be set later

    return task_constants


def validate_config(cfg: RetargetingConfig) -> None:
    """Validate configuration consistency.

    Args:
        cfg: Configuration arguments

    Raises:
        ValueError: If configuration is invalid
    """
    # Validate that data_format exists in registry (if provided)
    if cfg.data_format is not None and cfg.data_format not in DEMO_JOINTS_REGISTRY:
        available = ", ".join(sorted(DEMO_JOINTS_REGISTRY.keys()))
        raise ValueError(
            f"Unknown data_format: '{cfg.data_format}'. "
            f"Available formats: {available}. "
            f"Add your format to DEMO_JOINTS_REGISTRY in config_types/data_type.py"
        )

    # Task-specific format requirements
    if cfg.task_type == "climbing" and cfg.data_format not in (None, "mocap", "smplx"):
        raise ValueError("Climbing task requires 'mocap' or 'smplx' data format")
    if _is_object_task(cfg.task_type) and cfg.data_format not in (None, "smplh", "seedance"):
        raise ValueError("Object interaction requires 'smplh' or 'seedance' data format")
    # robot_only accepts any format in the registry (already validated above)


def _resolve_sequence_npz(data_path: Path, task_name: str) -> Path:
    candidates = [
        data_path / f"{task_name}.npz",
        data_path / task_name / f"{task_name}.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No .npz file found for {task_name} under {data_path}")


def load_seedance_data(npz_file: Path, robot_height: float, default_human_height: float | None = None):
    with np.load(str(npz_file), allow_pickle=True) as data:
        if "global_joint_positions" in data:
            human_joints = np.asarray(data["global_joint_positions"], dtype=float)
        elif "human_joints" in data:
            human_joints = np.asarray(data["human_joints"], dtype=float)
        elif "joints" in data:
            human_joints = np.asarray(data["joints"], dtype=float)
        else:
            raise KeyError(f"{npz_file} does not contain global_joint_positions/human_joints/joints")

        if "object_poses" in data:
            object_poses = np.asarray(data["object_poses"], dtype=float)
        elif "object_pose" in data:
            object_poses = np.asarray(data["object_pose"], dtype=float)
        else:
            object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]], dtype=float), (human_joints.shape[0], 1))

        human_height = None
        for key in ("human_height_m", "height", "human_height"):
            if key in data:
                human_height = float(np.asarray(data[key]).reshape(-1)[0])
                break

    if human_height is None or not np.isfinite(human_height) or human_height <= 0:
        human_height = default_human_height or 1.78
    smpl_scale = float(robot_height) / float(human_height)
    return human_joints, object_poses, smpl_scale


def load_hand_contact_data(
    data_path: Path,
    task_name: str,
    task_type: str,
    smpl_scale: float,
    object_scale_multiplier: float = 1.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    npz_file = _resolve_sequence_npz(data_path, task_name)
    with np.load(str(npz_file), allow_pickle=True) as data:
        points = None
        valid = None
        weight_scale = None
        for key in ("palm_contact_points_local", "object_contact_points_local"):
            if key in data:
                points = np.asarray(data[key], dtype=float)
                break
        for key in ("palm_contact_points_valid", "object_contact_points_valid"):
            if key in data:
                valid = np.asarray(data[key]).astype(bool)
                break
        if "palm_contact_weight_scale" in data:
            weight_scale = np.asarray(data["palm_contact_weight_scale"], dtype=float)

    if points is None or valid is None:
        return None, None, None

    if points.ndim == 2 and points.shape == (2, 3):
        points = np.broadcast_to(points[None, :, :], (valid.shape[0], 2, 3)).copy()
    if points.ndim != 3 or points.shape[1:] != (2, 3):
        raise ValueError(f"Hand contact points must have shape (T, 2, 3), got {points.shape}")
    if valid.ndim == 1:
        valid = np.stack([valid, valid], axis=1)
    if valid.shape != points.shape[:2]:
        raise ValueError(f"Hand contact valid shape {valid.shape} does not match points {points.shape[:2]}")

    num_frames = points.shape[0]
    if weight_scale is None:
        weight_scale = np.ones((num_frames, 2), dtype=float)
    elif weight_scale.ndim == 0:
        weight_scale = np.full((num_frames, 2), float(weight_scale), dtype=float)
    elif weight_scale.shape == (2,):
        weight_scale = np.broadcast_to(weight_scale[None, :], (num_frames, 2)).copy()
    elif weight_scale.shape == (num_frames, 1):
        weight_scale = np.repeat(weight_scale, 2, axis=1)
    elif weight_scale.shape != (num_frames, 2):
        raise ValueError(
            f"palm_contact_weight_scale must be scalar or have shape (2,), (T, 1), or (T, 2); "
            f"got {weight_scale.shape} for T={num_frames}"
        )
    if not np.all(np.isfinite(weight_scale)):
        raise ValueError("palm_contact_weight_scale must contain only finite values")
    if np.any(weight_scale < 0.0):
        raise ValueError("palm_contact_weight_scale must be non-negative")

    if _normalized_task_type(task_type) == "w-obj-scale":
        points = points * smpl_scale * object_scale_multiplier
    return points, valid, weight_scale


def load_required_contact_start_idx(
    data_path: Path,
    task_name: str,
    num_frames: int,
) -> int:
    """Load an exact contact start frame from the staged input without inference."""

    npz_file = _resolve_sequence_npz(data_path, task_name)
    with np.load(str(npz_file), allow_pickle=True) as data:
        if "contact_start_idx" not in data:
            raise KeyError(
                f"before_contact grounding requires contact_start_idx in {npz_file}"
            )
        value = np.asarray(data["contact_start_idx"])
    if value.size != 1:
        raise ValueError(
            f"contact_start_idx in {npz_file} must be scalar, got shape {value.shape}"
        )
    scalar = float(value.reshape(-1)[0])
    if not np.isfinite(scalar) or not scalar.is_integer():
        raise ValueError(f"contact_start_idx in {npz_file} must be a finite integer")
    contact_start_idx = int(scalar)
    if not 0 <= contact_start_idx < num_frames:
        raise ValueError(
            f"contact_start_idx={contact_start_idx} in {npz_file} is outside [0, {num_frames})"
        )
    return contact_start_idx


def create_ground_points(x_range: tuple[float, float], y_range: tuple[float, float], size: int) -> np.ndarray:
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
    task_type: TaskType,
    data_format: str,
    data_path: Path,
    task_name: str,
    constants: SimpleNamespace,
    motion_data_config: MotionDataConfig,
) -> tuple[np.ndarray, np.ndarray, float]:
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
    logger.info("Loading motion data for task: %s, format: %s", task_name, data_format)

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
            smpl_scale = motion_data_config.default_scale_factor or 1.0
        elif data_format == "smplh":  # smplh
            pt_path = data_path / f"{task_name}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")

            human_joints, object_poses = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)
        elif data_format == "mocap":
            downsample = 4
            npy_file = data_path / f"{task_name}.npy"
            if not npy_file.exists():
                raise FileNotFoundError(f"MOCAP data file not found: {npy_file}")

            human_joints = np.load(str(npy_file))[::downsample]

            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height
        elif data_format == "smplx":
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height
        else:
            # For other custom data format, if it uses consistent .npz file like SMPLX,
            # you can use the same logic as SMPLX.
            npz_file = data_path / f"{task_name}.npz"

            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = human_data["height"]
            smpl_scale = constants.ROBOT_HEIGHT / human_height

        # Create dummy object poses for robot_only
        num_frames = human_joints.shape[0]
        object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))

    elif _is_object_task(task_type):
        if data_format == "seedance":
            npz_file = _resolve_sequence_npz(data_path, task_name)
            human_joints, object_poses, smpl_scale = load_seedance_data(
                npz_file, constants.ROBOT_HEIGHT, motion_data_config.default_human_height
            )
        else:
            pt_path = data_path / f"{task_name}.pt"
            if not pt_path.exists():
                raise FileNotFoundError(f"InterMimic data file not found: {pt_path}")

            human_joints, object_poses = load_intermimic_data(str(pt_path))
            smpl_scale = calculate_scale_factor(task_name, constants.ROBOT_HEIGHT)

    elif task_type == "climbing":
        task_dir = data_path / task_name
        if data_format == "smplx":
            npz_file = task_dir / f"{task_name}.npz"
            if not npz_file.exists():
                npz_files = list(task_dir.glob("*.npz"))
                if not npz_files:
                    raise FileNotFoundError(f"No SMPL-X .npz file found in {task_dir}")
                npz_file = npz_files[0]
            human_data = np.load(str(npz_file))
            human_joints = human_data["global_joint_positions"]
            human_height = float(np.asarray(human_data["height"]).reshape(-1)[0])
            smpl_scale = constants.ROBOT_HEIGHT / human_height
            num_frames = human_joints.shape[0]
            object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))
        else:
            npy_files = list(task_dir.glob("*.npy"))
            if not npy_files:
                raise FileNotFoundError(f"No .npy file found in {task_dir}")

            npy_file = npy_files[0]
            # MOCAP-specific downsample factor
            downsample = 4
            human_joints = np.load(str(npy_file))[::downsample]
            num_frames = human_joints.shape[0]
            object_poses = np.tile(np.array([[1, 0, 0, 0, 0, 0, 0]]), (num_frames, 1))
            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height

    logger.debug(
        "Loaded %d frames, scale factor: %.4f",
        human_joints.shape[0],
        smpl_scale,
    )
    return human_joints, object_poses, smpl_scale


def setup_object_data(
    task_type: TaskType,
    constants: SimpleNamespace,
    object_dir: Path | None,
    smpl_scale: float,
    task_config: TaskConfig,
    augmentation: bool,
    object_scale_augmented: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Setup object-specific data (ground, object mesh, climbing terrain).
    Args:
        task_type: Type of task
        constants: Task constants
        object_dir: Object directory path (for climbing)
        smpl_scale: SMPL scaling factor
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        object_scale_augmented: Scale factor for augmented objects (default: [1.0, 1.0, 1.2])
    Returns:
        Tuple of (object_local_pts, object_local_pts_demo, object_urdf_path)
    """
    object_scale_normal = np.array([1.0, 1.0, 1.0])
    if object_scale_augmented is None:
        object_scale_augmented = np.array([1.0, 1.0, 1.2])  # For climbing task augmentation
    logger.info("Setting up object data for task: %s", task_type)

    if task_type == "robot_only":
        # Create ground points meshgrid
        ground_pts = create_ground_points(task_config.ground_range, task_config.ground_range, task_config.ground_size)
        return ground_pts, ground_pts, None

    if _is_object_task(task_type):
        # Load object data
        if constants.OBJECT_MESH_FILE is None:
            raise ValueError("OBJECT_MESH_FILE not set for object_interaction task")

        object_scale_multiplier = float(task_config.object_scale_multiplier)
        if not np.isfinite(object_scale_multiplier) or object_scale_multiplier <= 0.0:
            raise ValueError(
                "task_config.object_scale_multiplier must be positive and finite, "
                f"got {object_scale_multiplier}"
            )
        effective_object_scale = smpl_scale
        if _normalized_task_type(task_type) == "w-obj-scale":
            effective_object_scale *= object_scale_multiplier

        object_local_pts, object_local_pts_demo = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=effective_object_scale,
            sample_count=100,
        )
        if _normalized_task_type(task_type) == "w-obj-scale":
            scale_tag = f"{effective_object_scale:.6f}".replace(".", "p")
            scale_factors = (effective_object_scale,) * 3
            object_urdf = Path(constants.OBJECT_URDF_FILE)
            scaled_urdf = object_urdf.with_name(f"{object_urdf.stem}_w_obj_scale_{scale_tag}{object_urdf.suffix}")
            scene_xml = Path(constants.ROBOT_URDF_FILE.replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml"))
            scaled_scene_xml = scene_xml.with_name(f"{scene_xml.stem}_w_obj_scale_{scale_tag}{scene_xml.suffix}")
            create_scaled_multi_boxes_urdf(str(object_urdf), scale_factors, output_path=str(scaled_urdf))
            create_scaled_multi_boxes_xml(str(scene_xml), scale_factors, output_path=str(scaled_scene_xml))
            if task_config.scene_xml_file is not None:
                scene_override = task_config.scene_xml_file.expanduser().resolve()
                if not scene_override.is_file():
                    raise FileNotFoundError(f"Configured scene XML does not exist: {scene_override}")
                constants.SCENE_XML_FILE = str(scene_override)
            else:
                constants.SCENE_XML_FILE = str(scaled_scene_xml)
            constants.OBJECT_URDF_FILE = str(scaled_urdf)
            return object_local_pts_demo, object_local_pts_demo, str(scaled_urdf)
        return object_local_pts, object_local_pts, constants.OBJECT_URDF_FILE

    if task_type == "climbing":
        if object_dir is None:
            raise ValueError("object_dir must be provided for climbing task")

        # Setup climbing-specific object
        box_asset_xml = object_dir / "box_assets.xml"
        scene_xml_name = Path(constants.ROBOT_URDF_FILE).name.replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml")
        scene_xml_file = object_dir / scene_xml_name
        # Set SCENE_XML_FILE in constants BEFORE creating retargeter (needed for temp_retargeter)
        constants.SCENE_XML_FILE = str(scene_xml_file)

        np.random.seed(0)
        print("object mesh file: ", constants.OBJECT_MESH_FILE)
        object_local_pts, object_local_pts_demo_original = load_object_data(
            constants.OBJECT_MESH_FILE,
            smpl_scale=smpl_scale,
            surface_weights=lambda p: (
                task_config.surface_weight_high
                if p[2] > task_config.surface_weight_threshold
                else task_config.surface_weight_low
            ),
            sample_count=100,
        )

        if augmentation:
            ground_pts = create_ground_points(
                task_config.climbing_ground_range, task_config.climbing_ground_range, task_config.climbing_ground_size
            )
            object_local_pts_demo = np.concatenate([object_local_pts_demo_original, ground_pts], axis=0)
            object_scale = object_scale_augmented
            object_local_pts = object_scale * object_local_pts_demo
        else:
            object_scale = object_scale_normal
            object_local_pts_demo = object_local_pts_demo_original
            object_local_pts = object_local_pts_demo

        # Create scaled URDF and XML files
        scale_factors = tuple(float(value) for value in (object_scale * smpl_scale))
        object_urdf_file = create_scaled_multi_boxes_urdf(constants.OBJECT_URDF_FILE, scale_factors)
        object_asset_xml_path = create_scaled_multi_boxes_xml(str(box_asset_xml), scale_factors)
        new_scene_xml_path = create_new_scene_xml_file(str(scene_xml_file), scale_factors, object_asset_xml_path)
        constants.SCENE_XML_FILE = new_scene_xml_path

        return object_local_pts, object_local_pts_demo, object_urdf_file

    raise ValueError(f"Unknown task type: {task_type}")


def _compute_q_init_base(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter | None = None,
) -> np.ndarray:
    """Compute base robot pose initialization (q_init_base).
    This is a shared helper function used by both single and parallel processing.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses in format [qw, qx, qy, qz, x, y, z]
        constants: Task constants
        retargeter: Optional retargeter instance (needed for climbing)
    Returns:
        q_init_base in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints
    """
    if task_type == "robot_only":
        if data_format == "lafan":
            spine_joint_idx = constants.DEMO_JOINTS.index("Spine1")
            human_quat_init = estimate_human_orientation(human_joints, constants.DEMO_JOINTS)
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate(
                [human_joints[0, spine_joint_idx, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)]
            )
        else:  # smplh
            _, human_quat_init = transform_from_human_to_world(
                human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
            )
            # MuJoCo order: pos first, then quat
            q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif _is_object_task(task_type):
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate([human_joints[0, 0, :3], human_quat_init, np.zeros(constants.ROBOT_DOF)])
    elif task_type == "climbing":
        if retargeter is None:
            raise ValueError("retargeter is required for climbing task")
        _, human_quat_init = transform_from_human_to_world(
            human_joints[0, 0, :], object_poses[0], np.array([0.0, 0.0, 0.0])
        )
        spine_joint_idx = retargeter.demo_joints.index("Spine1")
        # MuJoCo order: pos first, then quat
        q_init_base = np.concatenate(
            [
                human_joints[0, spine_joint_idx],
                human_quat_init,
                np.zeros(constants.ROBOT_DOF),
            ]
        )
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return q_init_base


def convert_object_poses_to_mujoco_order(object_poses: np.ndarray) -> np.ndarray:
    """Convert object poses from [qw, qx, qy, qz, x, y, z] to MuJoCo order [x, y, z, qw, qx, qy, qz].
    Args:
        object_poses: Object poses array of shape (T, 7) in format [qw, qx, qy, qz, x, y, z]
    Returns:
        Object poses array in MuJoCo order [x, y, z, qw, qx, qy, qz]
    """
    return object_poses[:, [4, 5, 6, 0, 1, 2, 3]]


def build_retargeter_kwargs_from_config(
    retargeter_config: RetargeterConfig,
    constants: SimpleNamespace,
    object_urdf_path: str | None,
    task_type: str,
) -> dict:
    """Build kwargs for InteractionMeshRetargeter from a RetargeterConfig.
    This is a convenience function that allows building kwargs directly from
    a RetargeterConfig without needing a full RetargetingConfig.
    Args:
        retargeter_config: Retargeter configuration
        constants: Task constants
        object_urdf_path: Path to object URDF file
        task_type: Type of task
    Returns:
        Dictionary of kwargs for InteractionMeshRetargeter
    """
    kwargs = {
        "task_constants": constants,
        "object_urdf_path": object_urdf_path,
        "q_a_init_idx": retargeter_config.q_a_init_idx,
        "activate_joint_limits": retargeter_config.activate_joint_limits,
        "activate_obj_non_penetration": retargeter_config.activate_obj_non_penetration,
        "terrain_collision_geom_prefix": retargeter_config.terrain_collision_geom_prefix,
        "terrain_collision_foot_only": retargeter_config.terrain_collision_foot_only,
        "terrain_support_mesh_file": retargeter_config.terrain_support_mesh_file,
        "terrain_support_mesh_scale": retargeter_config.terrain_support_mesh_scale,
        "terrain_support_min_normal_z": retargeter_config.terrain_support_min_normal_z,
        "terrain_support_clearance": retargeter_config.terrain_support_clearance,
        "terrain_support_sphere_radius": retargeter_config.terrain_support_sphere_radius,
        "terrain_support_activation_margin": retargeter_config.terrain_support_activation_margin,
        "terrain_support_max_sqp_iterations": retargeter_config.terrain_support_max_sqp_iterations,
        "terrain_support_feasibility_tolerance": retargeter_config.terrain_support_feasibility_tolerance,
        "activate_foot_sticking": retargeter_config.activate_foot_sticking,
        "ground_initial_robot": retargeter_config.ground_initial_robot,
        "initial_ground_clearance": retargeter_config.initial_ground_clearance,
        "foot_sticking_pin_z": retargeter_config.foot_sticking_pin_z,
        "foot_sticking_z_floor": retargeter_config.foot_sticking_z_floor,
        "foot_grounding_weight": retargeter_config.foot_grounding_weight,
        "foot_grounding_mode": retargeter_config.foot_grounding_mode,
        "foot_grounding_schedule": retargeter_config.foot_grounding_schedule,
        "foot_grounding_ramp_frames": retargeter_config.foot_grounding_ramp_frames,
        "foot_lock": retargeter_config.foot_lock,
        "penetration_tolerance": retargeter_config.penetration_tolerance,
        "enforce_exact_nonpenetration": retargeter_config.enforce_exact_nonpenetration,
        "exact_nonpenetration_max_sqp_iterations": (
            retargeter_config.exact_nonpenetration_max_sqp_iterations
        ),
        "exact_nonpenetration_feasibility_tolerance": (
            retargeter_config.exact_nonpenetration_feasibility_tolerance
        ),
        "exact_nonpenetration_interior_margin": (
            retargeter_config.exact_nonpenetration_interior_margin
        ),
        "exact_nonpenetration_qp_safety_margin": (
            retargeter_config.exact_nonpenetration_qp_safety_margin
        ),
        "exact_nonpenetration_restore_infeasible_start": (
            retargeter_config.exact_nonpenetration_restore_infeasible_start
        ),
        "exact_nonpenetration_restoration_max_iterations": (
            retargeter_config.exact_nonpenetration_restoration_max_iterations
        ),
        "exact_nonpenetration_backtracking_steps": (
            retargeter_config.exact_nonpenetration_backtracking_steps
        ),
        "foot_sticking_tolerance": retargeter_config.foot_sticking_tolerance,
        "self_collision": retargeter_config.self_collision,
        "step_size": retargeter_config.step_size,
        "max_frame_root_translation": retargeter_config.max_frame_root_translation,
        "max_frame_root_quaternion_delta": retargeter_config.max_frame_root_quaternion_delta,
        "max_frame_joint_delta": retargeter_config.max_frame_joint_delta,
        "visualize": retargeter_config.visualize,
        "debug": retargeter_config.debug,
        "w_nominal_tracking_init": retargeter_config.w_nominal_tracking_init,
        "w_keypoint_tracking": retargeter_config.w_keypoint_tracking,
        "activate_hand_contact": retargeter_config.activate_hand_contact,
        "hand_contact_mode": retargeter_config.hand_contact_mode,
        "hand_contact_weight": retargeter_config.hand_contact_weight,
        "hand_contact_tolerance": retargeter_config.hand_contact_tolerance,
        "hand_contact_point_offset": retargeter_config.hand_contact_point_offset,
        "hand_contact_point_mode": retargeter_config.hand_contact_point_mode,
        "replace_source_wrist_with_contact": retargeter_config.replace_source_wrist_with_contact,
        "save_partial_on_failure": retargeter_config.save_partial_on_failure,
        "partial_checkpoint_interval_frames": (
            retargeter_config.partial_checkpoint_interval_frames
        ),
        "resume_partial_file": retargeter_config.resume_partial_file,
        "initial_qpos_file": retargeter_config.initial_qpos_file,
    }
    if task_type == "climbing":
        kwargs["nominal_tracking_tau"] = retargeter_config.nominal_tracking_tau
    return kwargs


def initialize_robot_pose(
    task_type: TaskType,
    data_format: str,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    constants: SimpleNamespace,
    retargeter: InteractionMeshRetargeter,
    task_config: TaskConfig,
    augmentation: bool,
    save_dir: Path,
    task_name: str,
    augmentation_translation: np.ndarray | None = None,
    augmentation_rotation: float | None = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """Initialize robot pose (q_init, q_nominal) based on task.
    Returns qpos in MuJoCo order: [0:3] position, [3:7] quaternion, [7:] joints.
    Object poses are returned in MuJoCo order: [0:3] position, [3:7] quaternion.
    Args:
        task_type: Type of task
        data_format: Data format
        human_joints: Human joint positions
        object_poses: Object poses (assumed to be in format: [quat, pos] or [pos, quat])
        constants: Task constants
        retargeter: Retargeter instance
        task_config: Task configuration
        augmentation: Whether augmentation is enabled
        save_dir: Save directory path
        task_name: Task name
        augmentation_translation: Translation vector for augmentation (default: [0.2, 0.0, 0.0])
    Returns:
        Tuple of (q_init, q_nominal, object_poses_augmented, human_joints_modified, object_poses_modified)
        where qpos is in MuJoCo order and object_poses are in MuJoCo order
    """
    # Use default if not provided
    if augmentation_translation is None:
        augmentation_translation = _AUGMENTATION_TRANSLATION
    logger.info("Initializing robot pose")

    if task_type == "robot_only":
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    if _is_object_task(task_type):
        if augmentation:
            object_moving_frame_idx = extract_object_first_moving_frame(object_poses)
            object_poses_augmented = augment_object_poses(
                object_poses,
                object_moving_frame_idx,
                human_joints[0, 0, :],
                augmentation_translation,
                augmentation_rotation,
            )
            # Convert object_poses to MuJoCo order
            object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
            object_poses = convert_object_poses_to_mujoco_order(object_poses)

            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            return q_nominal[0], q_nominal, object_poses_augmented, human_joints, object_poses
        object_poses_augmented = object_poses.copy()
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        object_poses_augmented = convert_object_poses_to_mujoco_order(object_poses_augmented)
        return q_init, None, object_poses_augmented, human_joints, object_poses

    if task_type == "climbing":
        if augmentation:
            original_path = save_dir / f"{task_name}_original.npz"
            if not original_path.exists():
                raise FileNotFoundError(f"Original file not found: {original_path}. Run without --augmentation first.")

            data = np.load(str(original_path))
            q_nominal = data["qpos"]
            # Convert object_poses to MuJoCo order
            object_poses = convert_object_poses_to_mujoco_order(object_poses)
            return q_nominal[0], q_nominal, object_poses, human_joints, object_poses
        q_init = _compute_q_init_base(task_type, data_format, human_joints, object_poses, constants, retargeter)
        # Convert object_poses to MuJoCo order
        object_poses = convert_object_poses_to_mujoco_order(object_poses)
        return q_init, None, object_poses, human_joints, object_poses

    raise ValueError(f"Unknown task type: {task_type}")


def determine_output_path(
    task_type: TaskType,
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
    if _is_object_task(task_type) or task_type == "climbing":
        suffix = "_augmented" if augmentation else "_original"
        return str(save_dir / f"{task_name}{suffix}.npz")
    raise ValueError(f"Unknown task type: {task_type}")


# ----------------------------- Main -----------------------------


def main(cfg: RetargetingConfig) -> None:
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
    data_format: str = cfg.data_format or DEFAULT_DATA_FORMATS[task_type]
    save_dir = cfg.save_dir if cfg.save_dir is not None else Path(DEFAULT_SAVE_DIRS[task_type].format(robot=robot))
    data_path = cfg.data_path

    os.makedirs(save_dir, exist_ok=True)
    logger.info("Task: %s, Type: %s, Format: %s", task_name, task_type, data_format)
    logger.info("Data path: %s, Save dir: %s", data_path, save_dir)

    # Ensure configs match top-level selections
    if cfg.robot_config.robot_type != robot:
        cfg.robot_config = RobotConfig(robot_type=robot)

    if cfg.motion_data_config.robot_type != robot or cfg.motion_data_config.data_format != data_format:
        cfg.motion_data_config = replace(cfg.motion_data_config, data_format=data_format, robot_type=robot)

    # Task-specific object setup: set default object_dir for climbing if not provided
    if task_type == "climbing" and cfg.task_config.object_dir is None:
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
    toe_names = cfg.motion_data_config.toe_names

    # Setup object data
    object_local_pts, object_local_pts_demo, object_urdf_path = setup_object_data(
        task_type,
        constants,
        cfg.task_config.object_dir,
        smpl_scale,
        cfg.task_config,
        cfg.augmentation,
        object_scale_augmented=_OBJECT_SCALE_AUGMENTED,
    )

    # Create retargeter
    retargeter_kwargs = build_retargeter_kwargs_from_config(cfg.retargeter, constants, object_urdf_path, task_type)
    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)
    logger.info("Retargeter created")

    # Preprocess motion data
    if task_type == "robot_only":
        human_joints = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            smpl_scale,
            source_mat_height=cfg.motion_data_config.source_mat_height,
            ground_z_override=cfg.motion_data_config.ground_z_override,
            ground_reference_mode=cfg.motion_data_config.ground_reference_mode,
        )
    elif _is_object_task(task_type) or task_type == "climbing":
        human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            scale=smpl_scale,
            source_mat_height=cfg.motion_data_config.source_mat_height,
            ground_z_override=cfg.motion_data_config.ground_z_override,
            ground_reference_mode=cfg.motion_data_config.ground_reference_mode,
            object_poses=object_poses,
        )

    # Initialize robot pose
    q_init, q_nominal, object_poses_augmented, human_joints, object_poses = initialize_robot_pose(
        task_type,
        data_format,
        human_joints,
        object_poses,
        constants,
        retargeter,
        cfg.task_config,
        cfg.augmentation,
        save_dir,
        task_name,
        augmentation_translation=_AUGMENTATION_TRANSLATION,
    )

    # Persist the detector policy with every complete or partial result so a
    # batch audit can distinguish legacy XY sticking from contact-aware runs.
    retargeter.source_foot_sticking_mode = str(cfg.motion_data_config.foot_sticking_mode)

    # Extract foot sticking sequences
    if cfg.motion_data_config.foot_sticking_mode == "contact_aware":
        foot_sticking_sequences = extract_foot_sticking_sequence_contact_aware(
            human_joints,
            retargeter.demo_joints,
            toe_names,
            velocity_threshold=cfg.motion_data_config.foot_contact_velocity_threshold,
            height_margin=cfg.motion_data_config.foot_contact_height_margin,
        )
    else:
        foot_sticking_sequences = extract_foot_sticking_sequence_velocity(
            human_joints, retargeter.demo_joints, toe_names
        )

    # Task-specific foot sticking adjustments
    if _is_object_task(task_type):
        # A grounded run needs frame 0 support; legacy XY-only sticking keeps
        # the historical frame-0 behavior when Z pinning is not requested.
        initial_sticking = bool(cfg.retargeter.foot_sticking_pin_z)
        foot_sticking_sequences[0][toe_names[0]] = initial_sticking
        foot_sticking_sequences[0][toe_names[1]] = initial_sticking

    # Determine output path
    dest_res_path = determine_output_path(task_type, save_dir, task_name, cfg.augmentation)
    hand_contact_points_local, hand_contact_valid, hand_contact_weight_scale = (None, None, None)
    if cfg.retargeter.activate_hand_contact and _is_object_task(task_type):
        hand_contact_points_local, hand_contact_valid, hand_contact_weight_scale = load_hand_contact_data(
            data_path,
            task_name,
            task_type,
            smpl_scale,
            cfg.task_config.object_scale_multiplier,
        )
        if hand_contact_points_local is not None:
            logger.info("Loaded hand contact targets (%d valid points)", int(np.asarray(hand_contact_valid).sum()))

    contact_start_idx = None
    if cfg.retargeter.foot_grounding_schedule == "before_contact":
        contact_start_idx = load_required_contact_start_idx(
            data_path,
            task_name,
            human_joints.shape[0],
        )
        logger.info(
            "Foot grounding schedule: before_contact, t1=%d, ramp=%d frames",
            contact_start_idx,
            cfg.retargeter.foot_grounding_ramp_frames,
        )

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
        original=not cfg.augmentation,
        dest_res_path=dest_res_path,
        hand_contact_points_local=hand_contact_points_local,
        hand_contact_valid=hand_contact_valid,
        hand_contact_weight_scale=hand_contact_weight_scale,
        contact_start_idx=contact_start_idx,
    )
    logger.info("Retargeting complete. Results saved to: %s", dest_res_path)

    if cfg.retargeter.debug:
        input("Press Enter to exit ...")


if __name__ == "__main__":
    cfg = tyro.cli(RetargetingConfig)
    main(cfg)
