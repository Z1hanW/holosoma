"""
Unified robot retargeting script for all task types:
- robot_only: Robot-only retargeting with ground interaction
- object_interaction: Object manipulation retargeting (InterMimic)
- climbing: Climbing retargeting with dynamic terrain
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Literal
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
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
    extract_object_first_moving_frame,
    load_behave_zup_data,
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


# Constants for numpy arrays (not in dataclass to avoid tyro parsing issues)
_OBJECT_SCALE_AUGMENTED = np.array([1.0, 1.0, 1.2])
_OBJECT_SCALE_NORMAL = np.array([1.0, 1.0, 1.0])
_AUGMENTATION_TRANSLATION = np.array([0.2, 0.0, 0.0])


# Type aliases
TaskType = Literal["robot_only", "object_interaction", "climbing"]
# DataFormat is imported from config_types.data_type


# ----------------------------- Helper Functions -----------------------------


def _ensure_mujoco_mesh(
    obj_name: str, mesh_path: Path, output_dir: Path, *, center_mesh: bool = False
) -> Path:
    """Ensure MuJoCo-compatible mesh (.obj) exists for the object."""
    mesh_path = mesh_path.resolve()
    suffix = mesh_path.suffix.lower()
    if suffix == ".obj":
        if not center_mesh:
            return mesh_path
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"{obj_name}.obj"
        if out_path.exists():
            existing = trimesh.load_mesh(str(out_path), process=False)
            if isinstance(existing, trimesh.Scene):
                existing = existing.dump(concatenate=True)
            if isinstance(existing, trimesh.Trimesh):
                center = existing.vertices.mean(axis=0)
                if float(np.linalg.norm(center)) < 1e-5:
                    return out_path
        mesh = trimesh.load_mesh(str(mesh_path), process=False)
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if isinstance(mesh, trimesh.Trimesh):
            mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
        mesh.export(str(out_path))
        return out_path
    if suffix not in {".ply", ".stl"}:
        raise ValueError(f"Unsupported mesh format for MuJoCo: {mesh_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{obj_name}.obj"
    if out_path.exists() and not center_mesh:
        return out_path

    mesh = trimesh.load_mesh(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if center_mesh:
        mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    if out_path.exists() and center_mesh:
        # Avoid rewriting if already centered.
        existing = trimesh.load_mesh(str(out_path), process=False)
        if isinstance(existing, trimesh.Scene):
            existing = existing.dump(concatenate=True)
        if isinstance(existing, trimesh.Trimesh):
            center = existing.vertices.mean(axis=0)
            if float(np.linalg.norm(center)) < 1e-5:
                return out_path
    mesh.export(str(out_path))
    return out_path


def _normalize_scale_vec(scale: float | tuple[float, float, float] | np.ndarray | None) -> np.ndarray:
    """Normalize scalar/tuple/array scale into a 3-vector."""
    if scale is None:
        return np.ones(3, dtype=float)
    scale_arr = np.asarray(scale, dtype=float).reshape(-1)
    if scale_arr.size == 1:
        return np.repeat(scale_arr, 3)
    if scale_arr.size != 3:
        raise ValueError("Scale must be a scalar or a 3-element sequence.")
    return scale_arr


def _scale_to_str(scale: float | tuple[float, float, float] | np.ndarray | None) -> str:
    scale_vec = _normalize_scale_vec(scale)
    return f"{scale_vec[0]} {scale_vec[1]} {scale_vec[2]}"


def _write_object_urdf(
    obj_name: str,
    mesh_path: Path,
    urdf_path: Path,
    mesh_scale: float | tuple[float, float, float] | np.ndarray | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Create a simple single-link URDF for the object mesh."""
    urdf_path.parent.mkdir(parents=True, exist_ok=True)
    scale_str = _scale_to_str(mesh_scale)

    mesh_ref = mesh_path.name if mesh_path.parent == urdf_path.parent else str(mesh_path)
    if urdf_path.exists() and not overwrite:
        return
    urdf_text = f"""<?xml version="1.0" ?>
<robot name="{obj_name}">
  <dynamics damping="0.5" friction="0.9"/>
  <link name="{obj_name}_link">
    <inertial>
      <mass value="0.1"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_ref}" scale="{scale_str}"/>
      </geometry>
      <material name="mat">
        <color rgba="0.7 0.8 0.9 0.7"/>
      </material>
    </visual>
    <collision name="{obj_name}">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_ref}" scale="{scale_str}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
    tmp_path = urdf_path.with_suffix(f"{urdf_path.suffix}.tmp.{os.getpid()}")
    tmp_path.write_text(urdf_text)
    os.replace(tmp_path, urdf_path)


def _write_robot_object_xml(
    robot_xml_base: Path,
    robot_xml_out: Path,
    obj_name: str,
    mesh_path: Path,
    mesh_scale: float | tuple[float, float, float] | np.ndarray | None = None,
    body_name: str | None = None,
    *,
    overwrite: bool = False,
) -> None:
    """Create a MuJoCo XML that adds a free object body to the robot model."""
    object_body_name = body_name or obj_name
    scale_str = _scale_to_str(mesh_scale)
    mesh_file = str(Path(mesh_path).resolve())
    if robot_xml_out.exists() and not overwrite:
        return

    tree = ET.parse(robot_xml_base)
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    mesh_name = f"{object_body_name}_mesh"
    mesh_elem = next((m for m in asset.findall("mesh") if m.get("name") == mesh_name), None)
    if mesh_elem is None:
        ET.SubElement(
            asset,
            "mesh",
            {
                "name": mesh_name,
                "file": mesh_file,
                "scale": scale_str,
            },
        )
    else:
        mesh_elem.set("file", mesh_file)
        mesh_elem.set("scale", scale_str)

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    body = next((b for b in worldbody.findall("body") if b.get("name") == object_body_name), None)
    if body is None:
        body = ET.SubElement(worldbody, "body", {"name": object_body_name})
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": "0.1", "diaginertia": "0.002 0.002 0.002"})
        ET.SubElement(
            body,
            "geom",
            {
                "type": "mesh",
                "mesh": mesh_name,
                "pos": "0 0 0",
                "quat": "1 0 0 0",
                "rgba": "0.7 0.8 0.9 0.7",
                "friction": "0.9 0.5 0.5",
                "solref": "0.02 1",
                "solimp": "0.9 0.95 0.001",
            },
        )
    else:
        body_mesh_geoms = [g for g in body.findall("geom") if g.get("mesh") is not None]
        if body_mesh_geoms:
            body_mesh_geoms[0].set("mesh", mesh_name)
        else:
            ET.SubElement(
                body,
                "geom",
                {
                    "type": "mesh",
                    "mesh": mesh_name,
                    "pos": "0 0 0",
                    "quat": "1 0 0 0",
                    "rgba": "0.7 0.8 0.9 0.7",
                    "friction": "0.9 0.5 0.5",
                    "solref": "0.02 1",
                    "solimp": "0.9 0.95 0.001",
                },
            )

    robot_xml_out.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = robot_xml_out.with_suffix(f"{robot_xml_out.suffix}.tmp.{os.getpid()}")
    tree.write(tmp_path)
    os.replace(tmp_path, robot_xml_out)


def _resolve_object_interaction_scene_xml(constants: SimpleNamespace) -> Path:
    scene_xml_override = str(getattr(constants, "SCENE_XML_FILE", "") or "").strip()
    if scene_xml_override:
        return Path(scene_xml_override)
    return Path(str(constants.ROBOT_URDF_FILE).replace(".urdf", f"_w_{constants.OBJECT_NAME}.xml"))


def _create_scaled_object_interaction_assets(
    constants: SimpleNamespace,
    mesh_scale: float | tuple[float, float, float] | np.ndarray,
) -> tuple[str, str]:
    """Create scale-specific URDF/XML assets for object_interaction."""
    if constants.OBJECT_MESH_FILE is None or constants.OBJECT_URDF_FILE is None:
        raise ValueError("OBJECT_MESH_FILE/OBJECT_URDF_FILE must be set for object_interaction scaling.")

    mesh_scale_vec = _normalize_scale_vec(mesh_scale)
    scale_tag = f"{mesh_scale_vec[0]:.3f}_{mesh_scale_vec[1]:.3f}_{mesh_scale_vec[2]:.3f}"

    object_name = str(constants.OBJECT_NAME)
    object_contact_name = str(getattr(constants, "OBJECT_CONTACT_NAME", "") or object_name)
    mesh_path = Path(str(constants.OBJECT_MESH_FILE))

    base_urdf_path = Path(str(constants.OBJECT_URDF_FILE))
    scaled_urdf_path = base_urdf_path.with_name(f"{base_urdf_path.stem}_scaled_{scale_tag}{base_urdf_path.suffix}")
    _write_object_urdf(
        object_name,
        mesh_path,
        scaled_urdf_path,
        mesh_scale=mesh_scale_vec,
        overwrite=True,
    )

    scene_xml_base = _resolve_object_interaction_scene_xml(constants)
    robot_xml_base = Path(str(constants.ROBOT_URDF_FILE)).with_suffix(".xml")
    xml_base = scene_xml_base if scene_xml_base.exists() else robot_xml_base
    if not xml_base.exists():
        raise FileNotFoundError(
            f"Cannot find base XML for object_interaction scaling: scene={scene_xml_base}, robot={robot_xml_base}"
        )
    scaled_scene_xml = xml_base.with_name(f"{xml_base.stem}_scaled_{scale_tag}{xml_base.suffix}")
    _write_robot_object_xml(
        xml_base,
        scaled_scene_xml,
        object_name,
        mesh_path,
        mesh_scale=mesh_scale_vec,
        body_name=object_contact_name,
        overwrite=True,
    )

    return str(scaled_urdf_path), str(scaled_scene_xml)


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
        task_constants.OBJECT_CONTACT_NAME = task_config.object_contact_name or obj_name
        task_constants.OBJECT_URDF_FILE = None
        task_constants.OBJECT_MESH_FILE = None
        task_constants.OBJECT_MESH_ROOT = str(task_config.object_mesh_root) if task_config.object_mesh_root else ""
        task_constants.OBJECT_MESH_SUFFIX = (
            str(task_config.object_mesh_suffix) if task_config.object_mesh_suffix else ""
        )
    elif task_type == "object_interaction":
        obj_name = task_config.object_name or "largebox"
        task_constants.OBJECT_NAME = obj_name
        if task_config.object_contact_name is not None:
            task_constants.OBJECT_CONTACT_NAME = task_config.object_contact_name
        elif task_config.scene_xml_file is not None and "_w_obj" in task_config.scene_xml_file.name:
            task_constants.OBJECT_CONTACT_NAME = "obj"
        else:
            task_constants.OBJECT_CONTACT_NAME = obj_name
        task_constants.OBJECT_URDF_FILE = f"models/{obj_name}/{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = f"models/{obj_name}/{obj_name}.obj"
        task_constants.OBJECT_URDF_TEMPLATE = f"models/templates/{obj_name}.urdf.jinja"
        task_constants.SCENE_XML_FILE = str(task_config.scene_xml_file) if task_config.scene_xml_file else ""
        task_constants.OBJECT_MESH_ROOT = str(task_config.object_mesh_root) if task_config.object_mesh_root else ""
        task_constants.OBJECT_MESH_SUFFIX = (
            str(task_config.object_mesh_suffix) if task_config.object_mesh_suffix else ""
        )
    elif task_type == "climbing":
        obj_name = task_config.object_name or "multi_boxes"
        task_constants.OBJECT_NAME = obj_name
        task_constants.OBJECT_CONTACT_NAME = task_config.object_contact_name or obj_name
        object_dir = task_config.object_dir
        task_constants.OBJECT_DIR = str(object_dir) if object_dir else ""
        task_constants.OBJECT_URDF_FILE = str(object_dir / f"{obj_name}.urdf") if object_dir else f"{obj_name}.urdf"
        task_constants.OBJECT_MESH_FILE = str(object_dir / f"{obj_name}.obj") if object_dir else f"{obj_name}.obj"
        task_constants.SCENE_XML_FILE = str(task_config.scene_xml_file) if task_config.scene_xml_file else ""
        task_constants.OBJECT_MESH_ROOT = str(task_config.object_mesh_root) if task_config.object_mesh_root else ""
        task_constants.OBJECT_MESH_SUFFIX = (
            str(task_config.object_mesh_suffix) if task_config.object_mesh_suffix else ""
        )

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
    if cfg.task_type == "climbing" and cfg.data_format not in (None, "mocap"):
        raise ValueError("Climbing task requires 'mocap' data format")
    if cfg.task_type == "object_interaction" and cfg.data_format not in (None, "smplh", "behave_zup"):
        raise ValueError("Object interaction requires 'smplh' or 'behave_zup' data format")
    # robot_only accepts any format in the registry (already validated above)


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

    elif task_type == "object_interaction":
        if data_format == "behave_zup":
            seq_dir = data_path / task_name
            human_joints, object_poses = load_behave_zup_data(seq_dir)
            default_human_height = motion_data_config.default_human_height or 1.78
            smpl_scale = constants.ROBOT_HEIGHT / default_human_height

            parts = task_name.split("_")
            if len(parts) <= 2:
                raise ValueError(
                    f"Cannot parse BEHAVE object name from task_name='{task_name}'. "
                    "Expected format like 'Date03_Sub03_boxlarge'."
                )
            obj_name = parts[2].strip().lower()
            if not obj_name:
                raise ValueError(f"Parsed empty BEHAVE object name from task_name='{task_name}'")

            prev_object_name = str(getattr(constants, "OBJECT_NAME", "") or "")
            prev_contact_name = str(getattr(constants, "OBJECT_CONTACT_NAME", "") or "")
            constants.OBJECT_NAME = obj_name
            # Keep the contact/body token aligned with the parsed BEHAVE object name.
            # If this stays at the default "largebox", scaled scene XML generation adds
            # a second free body instead of updating the existing object body.
            if not prev_contact_name or prev_contact_name == prev_object_name:
                constants.OBJECT_CONTACT_NAME = obj_name
            mesh_root = getattr(constants, "OBJECT_MESH_ROOT", "")
            mesh_suffix = getattr(constants, "OBJECT_MESH_SUFFIX", "") or "_f1000.ply"
            if not mesh_root:
                raise ValueError(
                    "BEHAVE retargeting requires --task-config.object-mesh-root. "
                    "No fallback to built-in models is allowed."
                )

            mesh_path = Path(mesh_root) / obj_name / f"{obj_name}{mesh_suffix}"
            if not mesh_path.exists():
                raise FileNotFoundError(
                    f"Missing BEHAVE mesh for object='{obj_name}': {mesh_path}. "
                    "No fallback mesh is used."
                )

            retarget_root = Path(__file__).resolve().parents[1]
            generated_root = retarget_root / "models" / "behave_objects" / obj_name
            mujoco_mesh_path = _ensure_mujoco_mesh(obj_name, mesh_path, generated_root, center_mesh=True)
            constants.OBJECT_MESH_FILE = str(mujoco_mesh_path)

            urdf_path = generated_root / f"{obj_name}.urdf"
            _write_object_urdf(obj_name, mujoco_mesh_path, urdf_path, mesh_scale=smpl_scale, overwrite=False)
            constants.OBJECT_URDF_FILE = str(urdf_path)
            constants.OBJECT_URDF_TEMPLATE = ""

            robot_urdf_path = Path(constants.ROBOT_URDF_FILE)
            if not robot_urdf_path.is_absolute():
                candidate = retarget_root / robot_urdf_path
                if candidate.exists():
                    robot_urdf_path = candidate
                    constants.ROBOT_URDF_FILE = str(robot_urdf_path)

            robot_xml_base = robot_urdf_path.with_suffix(".xml")
            object_contact_name = str(getattr(constants, "OBJECT_CONTACT_NAME", "") or obj_name)
            scene_xml_override = str(getattr(constants, "SCENE_XML_FILE", "") or "").strip()
            if scene_xml_override:
                scene_xml_path = Path(scene_xml_override)
                if not scene_xml_path.is_absolute():
                    candidate_cwd = Path.cwd() / scene_xml_path
                    candidate_retarget = retarget_root / scene_xml_path
                    scene_xml_path = candidate_cwd if candidate_cwd.exists() else candidate_retarget

                if not scene_xml_path.exists():
                    if not robot_xml_base.exists():
                        raise FileNotFoundError(f"Missing robot xml base for BEHAVE object retargeting: {robot_xml_base}")
                    _write_robot_object_xml(
                        robot_xml_base,
                        scene_xml_path,
                        obj_name,
                        mujoco_mesh_path,
                        mesh_scale=smpl_scale,
                        body_name=object_contact_name,
                        overwrite=True,
                    )

                constants.SCENE_XML_FILE = str(scene_xml_path)
            else:
                safe_task = re.sub(r"[^0-9A-Za-z_\\-]+", "_", task_name).strip("_")
                if not safe_task:
                    safe_task = f"task_{os.getpid()}"
                robot_xml_out = robot_urdf_path.parent / f"{robot_urdf_path.stem}_w_{obj_name}_{safe_task}.xml"
                if not robot_xml_base.exists():
                    raise FileNotFoundError(f"Missing robot xml base for BEHAVE object retargeting: {robot_xml_base}")
                _write_robot_object_xml(
                    robot_xml_base,
                    robot_xml_out,
                    obj_name,
                    mujoco_mesh_path,
                    mesh_scale=smpl_scale,
                    body_name=obj_name,
                    overwrite=True,
                )
                constants.SCENE_XML_FILE = str(robot_xml_out)
        else:
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
) -> tuple[np.ndarray | None, np.ndarray | None, str | None, np.ndarray | None]:
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
        Tuple of (object_local_pts, object_local_pts_demo, object_urdf_path, object_mesh_scale)
    """
    object_scale_normal = np.array([1.0, 1.0, 1.0])
    object_mesh_scale = None
    if object_scale_augmented is None:
        object_scale_augmented = np.array([1.0, 1.0, 1.2])  # For climbing task augmentation
    logger.info("Setting up object data for task: %s", task_type)

    if task_type == "robot_only":
        # Create ground points meshgrid
        ground_pts = create_ground_points(task_config.ground_range, task_config.ground_range, task_config.ground_size)
        return ground_pts, ground_pts, None, None

    if task_type == "object_interaction":
        # Load object data
        if constants.OBJECT_MESH_FILE is None:
            raise ValueError("OBJECT_MESH_FILE not set for object_interaction task")

        _, object_local_pts_scaled = load_object_data(
            constants.OBJECT_MESH_FILE, smpl_scale=smpl_scale, sample_count=100
        )
        # Keep object geometry consistently scaled in optimization and demo space.
        # Using unscaled points here causes a mismatch against scaled human/object poses.
        object_local_pts_demo = np.array(object_local_pts_scaled, copy=True)
        object_scale = np.ones(3, dtype=float)
        if task_config.object_interaction_scale_augmented is not None:
            object_scale = _normalize_scale_vec(task_config.object_interaction_scale_augmented)
        object_local_pts = np.array(object_local_pts_demo * object_scale.reshape(1, 3), copy=True)
        object_mesh_scale = np.array(object_scale * smpl_scale, dtype=float)

        object_urdf_file = constants.OBJECT_URDF_FILE
        if not np.allclose(object_scale, np.ones(3, dtype=float)):
            object_urdf_file, scaled_scene_xml = _create_scaled_object_interaction_assets(constants, object_mesh_scale)
            constants.SCENE_XML_FILE = scaled_scene_xml

        return object_local_pts, object_local_pts_demo, object_urdf_file, object_mesh_scale

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
        object_mesh_scale = np.array(scale_factors, dtype=float)
        object_urdf_file = create_scaled_multi_boxes_urdf(constants.OBJECT_URDF_FILE, scale_factors)
        object_asset_xml_path = create_scaled_multi_boxes_xml(str(box_asset_xml), scale_factors)
        new_scene_xml_path = create_new_scene_xml_file(str(scene_xml_file), scale_factors, object_asset_xml_path)
        constants.SCENE_XML_FILE = new_scene_xml_path

        return object_local_pts, object_local_pts_demo, object_urdf_file, object_mesh_scale

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
    elif task_type == "object_interaction":
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
        "activate_foot_sticking": retargeter_config.activate_foot_sticking,
        "penetration_tolerance": retargeter_config.penetration_tolerance,
        "foot_sticking_tolerance": retargeter_config.foot_sticking_tolerance,
        "step_size": retargeter_config.step_size,
        "visualize": retargeter_config.visualize,
        "debug": retargeter_config.debug,
        "w_nominal_tracking_init": retargeter_config.w_nominal_tracking_init,
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

    if task_type == "object_interaction":
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
    if task_type in ("object_interaction", "climbing"):
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
    toe_names = cfg.motion_data_config.toe_names

    # Setup object data
    object_local_pts, object_local_pts_demo, object_urdf_path, object_mesh_scale = setup_object_data(
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
    retargeter_kwargs["object_mesh_path"] = constants.OBJECT_MESH_FILE
    retargeter_kwargs["object_mesh_scale"] = object_mesh_scale
    retargeter = InteractionMeshRetargeter(**retargeter_kwargs)
    logger.info("Retargeter created")

    # Preprocess motion data
    if task_type == "robot_only":
        human_joints = preprocess_motion_data(human_joints, retargeter, toe_names, smpl_scale)
    elif task_type in {"object_interaction", "climbing"}:
        human_joints, object_poses, object_moving_frame_idx = preprocess_motion_data(
            human_joints,
            retargeter,
            toe_names,
            scale=smpl_scale,
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

    # Extract foot sticking sequences
    foot_sticking_sequences = extract_foot_sticking_sequence_velocity(human_joints, retargeter.demo_joints, toe_names)

    # Task-specific foot sticking adjustments
    if task_type == "object_interaction":
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
        original=not cfg.augmentation,
        dest_res_path=dest_res_path,
    )
    logger.info("Retargeting complete. Results saved to: %s", dest_res_path)

    if cfg.retargeter.debug:
        input("Press Enter to exit ...")


if __name__ == "__main__":
    cfg = tyro.cli(RetargetingConfig)
    main(cfg)
