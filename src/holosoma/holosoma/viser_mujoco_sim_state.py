from __future__ import annotations

import json
import inspect
import os
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from io import StringIO, TextIOWrapper
from pathlib import Path

import numpy as np
import trimesh
import tyro
from loguru import logger

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
INFER_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "holosoma_inference"
DEFAULT_TRACKING_MOTION_FILE = REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "motions" / "g1_29dof" / "whole_body_tracking" / "sub3_largebox_003_mj_w_obj.npz"
DEFAULT_TRACKING_MODEL_PATH = Path(
    "/data/logs_new/boxer/20260316_200048-g1_29dof_wbt_w_object_extend_20260316_200027_s01_scale_1p0-g1_29dof_wbt_w_object_extend_20260316_200027/model_23500.onnx"
)
for path in (SRC_ROOT, INFER_SRC_ROOT):
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port  # noqa: E402

ensure_viser_on_path()

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_values import robot as robot_values  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma_inference.utils.perception_obs import PerceptionObsShmSub, PerceptionObsSub  # noqa: E402
from holosoma_inference.utils.sim_control import ManualRootCommandPub, SimControlPush  # noqa: E402
from holosoma_inference.utils.sim_state import SimStateSub  # noqa: E402


@dataclass(frozen=True)
class MujocoSimStateViewerConfig:
    robot: str = "g1_29dof_w_object"
    state_port: int = 5657
    perception_obs_port: int = 5658
    perception_obs_shm_name: str = "depth_img_shm"
    control_port: int = 5659
    sparse_root_command_port: int = 5661
    object_actor_name: str = "object"
    port: int = 0
    rate_hz: float = 30.0
    recenter_xy: bool = True
    show_object: bool = True
    object_mesh_mode: str = ""
    show_robot_collision: bool = False
    show_object_collision: bool = False
    mujoco_object_geom_snapshot_path: str = str(REPO_ROOT / "logs" / "live_debug" / "viser_mujoco_object_geoms.json")
    show_ref_body: bool = False
    show_motion_overlay: bool = True
    grid_size: float = 8.0
    launch_rollout: bool = False
    launch_env_only: bool = False
    run_script: str = str(REPO_ROOT / "mj_track.sh")
    motion_file: str = str(DEFAULT_TRACKING_MOTION_FILE)
    model_path: str = str(DEFAULT_TRACKING_MODEL_PATH)
    launch_run_seconds: int = 0
    training_headless: bool = True
    rollout_log_path: str = str(REPO_ROOT / "logs" / "live_debug" / "viser_mujoco_sim_state.log")
    rollout_tty_input: bool = False
    auto_reset_after_first_state_sec: float = 0.0
    manual_motion_init_mode: bool = False
    reset_to_default_pose: bool = False
    show_depth: bool = True
    depth_height: int = 58
    depth_width: int = 87
    depth_display_scale: int = 4
    depth_obs_normalized: bool = True
    depth_near: float = 0.1
    depth_far: float = 3.0
    manual_root_enabled: bool = False
    manual_root_mode: str = "manual"
    manual_root_dx: float = 0.0
    manual_root_dy: float = 0.0
    manual_root_dyaw: float = 0.0
    manual_root_publisher_enabled: bool = True
    keyboard_root_command: bool = False
    keyboard_root_command_value: float = 0.5
    keyboard_root_command_mode: str = "manual"


@dataclass(frozen=True)
class MotionOverlay:
    path: Path
    fps: float
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos_viser: np.ndarray
    object_pos_w: np.ndarray | None
    object_quat_wxyz: np.ndarray | None
    object_mesh: trimesh.Trimesh | None


def _resolve_data_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path)).expanduser().resolve()


def _resolve_robot_config(name: str) -> RobotConfig:
    defaults = robot_values.DEFAULTS
    if name not in defaults:
        raise ValueError(f"Unknown robot '{name}'. Available: {sorted(defaults.keys())}")
    return defaults[name]


def _resolve_repo_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _resolve_repo_path_preserve_symlink(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return Path(os.path.abspath(candidate))


def _truthy_env(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_robot_urdf_path(robot_config: RobotConfig) -> Path:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    return _resolve_data_path(str(asset_root / robot_config.asset.urdf_file))


def _xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)


def _normalize_quaternion_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(4)
    quat_norm = float(np.linalg.norm(quat_wxyz))
    if quat_norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return quat_wxyz / quat_norm


def _quat_wxyz_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion_wxyz(quat_wxyz).astype(np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _transform_vertices_local(vertices: np.ndarray, position: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    rotation = _quat_wxyz_to_matrix(quat_wxyz)
    vertices = np.asarray(vertices, dtype=np.float32)
    position = np.asarray(position, dtype=np.float32).reshape(3)
    return (vertices @ rotation.T) + position


def _valid_depth_stats(depth: np.ndarray, near: float, far: float) -> tuple[float | None, float | None, int]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    valid &= depth >= near
    valid &= depth < (far - 1.0e-6)
    if not np.any(valid):
        return None, None, 0
    depth_valid = depth[valid]
    return float(depth_valid.min()), float(depth_valid.max()), int(depth_valid.size)


def _depth_obs_to_meters(depth_obs: np.ndarray, *, normalized: bool, near: float, far: float) -> np.ndarray:
    depth_obs = np.asarray(depth_obs, dtype=np.float32)
    if not normalized:
        return depth_obs
    return (np.clip(depth_obs, -0.5, 0.5) + 0.5) * max(far - near, 1.0e-6) + near


def _depth_to_rgb(depth_m: np.ndarray, near: float, far: float) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth_m)
    valid &= depth_m >= near
    valid &= depth_m <= (far + 1.0e-6)
    rgb = np.zeros(depth_m.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return rgb

    norm = np.clip((depth_m - near) / max(far - near, 1.0e-6), 0.0, 1.0)
    close = 1.0 - norm
    mid = 1.0 - np.abs(norm * 2.0 - 1.0)
    rgb[..., 0] = np.round(close * 255.0).astype(np.uint8)
    rgb[..., 1] = np.round(mid * 255.0).astype(np.uint8)
    rgb[..., 2] = np.round(norm * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


def _scale_image_nearest(image: np.ndarray, scale: int) -> np.ndarray:
    scale = max(int(scale), 1)
    if scale == 1:
        return image
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


OBJECT_MESH_MODE_OPTIONS = ("visual", "collision")


def _resolve_object_mesh_mode(mode_raw: object, *, show_object_collision: bool = False) -> str:
    mode = str(mode_raw).strip().lower()
    if mode in OBJECT_MESH_MODE_OPTIONS:
        return mode
    return "collision" if show_object_collision else "visual"


def _geom_supports_visual_mesh(geom_entry: dict) -> bool:
    rgba = np.asarray(geom_entry.get("rgba", [0.75, 0.75, 0.75, 1.0]), dtype=np.float32).reshape(-1)
    alpha = float(rgba[3]) if rgba.shape[0] >= 4 else 1.0
    return alpha > 1e-4


def _mesh_arrays_from_mujoco_geom(geom_entry: dict, collision_view: bool) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int], float] | None:
    geom_type = str(geom_entry.get("type", ""))
    geom_size = np.asarray(geom_entry.get("size", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1)
    rgba = np.asarray(geom_entry.get("rgba", [0.75, 0.75, 0.75, 1.0]), dtype=np.float32).reshape(-1)

    mesh: trimesh.Trimesh | None = None
    if geom_type == "mesh":
        mesh_payload = geom_entry.get("mesh")
        if not isinstance(mesh_payload, dict):
            return None
        vertices = np.asarray(mesh_payload.get("vertices", []), dtype=np.float32)
        faces = np.asarray(mesh_payload.get("faces", []), dtype=np.int32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
            return None
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    elif geom_type == "box" and geom_size.shape[0] >= 3:
        mesh = trimesh.creation.box(extents=2.0 * geom_size[:3])
    elif geom_type == "sphere" and geom_size.shape[0] >= 1:
        mesh = trimesh.creation.uv_sphere(radius=float(geom_size[0]))
    elif geom_type == "capsule" and geom_size.shape[0] >= 2:
        mesh = trimesh.creation.capsule(height=2.0 * float(geom_size[1]), radius=float(geom_size[0]))
    elif geom_type == "cylinder" and geom_size.shape[0] >= 2:
        mesh = trimesh.creation.cylinder(radius=float(geom_size[0]), height=2.0 * float(geom_size[1]))
    elif geom_type == "ellipsoid" and geom_size.shape[0] >= 3:
        mesh = trimesh.creation.uv_sphere(radius=1.0)
        mesh.vertices *= geom_size[:3]

    if mesh is None:
        return None

    if collision_view:
        color = (255, 72, 72)
        opacity = 0.30
    else:
        rgb = np.clip(np.round(rgba[:3] * 255.0), 0, 255).astype(np.int32)
        color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
        opacity = float(np.clip(rgba[3] if rgba.shape[0] >= 4 else 1.0, 0.0, 1.0))

    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
        color,
        opacity,
    )


def _load_mujoco_object_geom_snapshot(snapshot_path: Path, actor_name: str) -> dict[str, object] | None:
    if not snapshot_path.is_file():
        return None
    try:
        with snapshot_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to load MuJoCo geom snapshot {}: {}", snapshot_path, exc)
        return None

    actors = payload.get("actors")
    if not isinstance(actors, dict) or not actors:
        return None
    actor_payload = actors.get(actor_name)
    if actor_payload is None and len(actors) == 1:
        actor_payload = next(iter(actors.values()))
    return actor_payload if isinstance(actor_payload, dict) else None


def _select_actor_state(state: dict, actor_name: str) -> tuple[str | None, np.ndarray | None]:
    actors = state.get("actors")
    if not isinstance(actors, dict) or not actors:
        return None, None

    actor_state = actors.get(actor_name)
    actor_key = actor_name
    if actor_state is None and len(actors) == 1:
        actor_key, actor_state = next(iter(actors.items()))
    if actor_state is None:
        return None, None

    actor_state_np = np.asarray(actor_state, dtype=np.float32).reshape(-1)
    if actor_state_np.shape[0] < 7:
        return None, None
    return actor_key, actor_state_np


def _build_default_joint_viser(robot_config: RobotConfig, viser_joint_names: list[str]) -> np.ndarray:
    default_joint_angles = getattr(robot_config.init_state, "default_joint_angles", {}) or {}
    name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")
    default_joint_robot = np.zeros(len(robot_config.dof_names), dtype=np.float32)
    for idx, name in enumerate(robot_config.dof_names):
        default_joint_robot[idx] = float(default_joint_angles.get(name, 0.0))
    return np.asarray([default_joint_robot[name_to_robot_idx[name]] for name in viser_joint_names], dtype=np.float32)


def _decode_npz_names(raw_names: np.ndarray) -> list[str]:
    return [str(name.decode("utf-8") if isinstance(name, bytes) else name) for name in np.asarray(raw_names).reshape(-1)]


def _scalar_str(raw: object) -> str:
    array = np.asarray(raw)
    if array.shape == ():
        return str(array.item())
    if array.size == 0:
        return ""
    return str(array.reshape(-1)[0])


def _resolve_motion_root_body_index(body_names: list[str]) -> int:
    for preferred in ("pelvis", "base_link", "base", "torso_link"):
        if preferred in body_names:
            return body_names.index(preferred)
    for idx, name in enumerate(body_names):
        if name and name != "world":
            return idx
    return 0


def _parse_vec3(raw: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if not raw:
        return np.asarray(default, dtype=np.float64)
    parts = [float(part) for part in raw.replace(",", " ").split()]
    if len(parts) != 3:
        return np.asarray(default, dtype=np.float64)
    return np.asarray(parts, dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy.reshape(3)]
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _load_motion_object_mesh(urdf_path: Path, object_size: np.ndarray | None = None) -> trimesh.Trimesh | None:
    meshes: list[trimesh.Trimesh] = []
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        logger.warning("Failed to parse motion object URDF {}: {}", urdf_path, exc)
        root = None

    if root is not None:
        for geom_parent in root.findall(".//visual") + root.findall(".//collision"):
            geometry = geom_parent.find("geometry")
            mesh_tag = geometry.find("mesh") if geometry is not None else None
            if mesh_tag is None:
                continue
            filename = str(mesh_tag.get("filename") or "").strip()
            if not filename:
                continue
            mesh_path = Path(filename).expanduser()
            if not mesh_path.is_absolute():
                mesh_path = urdf_path.parent / mesh_path
            if not mesh_path.is_file():
                continue

            loaded = trimesh.load(mesh_path, process=False)
            if isinstance(loaded, trimesh.Scene):
                loaded = loaded.dump(concatenate=True)
            if not isinstance(loaded, trimesh.Trimesh):
                continue

            mesh = loaded.copy()
            scale = _parse_vec3(mesh_tag.get("scale"), (1.0, 1.0, 1.0))
            mesh.apply_scale(scale)
            origin = geom_parent.find("origin")
            xyz = _parse_vec3(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = _parse_vec3(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = _rpy_to_matrix(rpy)
            transform[:3, 3] = xyz
            mesh.apply_transform(transform)
            meshes.append(mesh)
            break

    if meshes:
        return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]

    if object_size is None:
        return None
    size = np.asarray(object_size, dtype=np.float64).reshape(-1)
    if size.shape[0] < 3 or not np.all(np.isfinite(size[:3])) or np.any(size[:3] <= 0.0):
        return None
    return trimesh.creation.box(extents=size[:3])


def _load_motion_overlay(
    motion_path: Path,
    robot_config: RobotConfig,
    viser_joint_names: list[str],
) -> MotionOverlay | None:
    if not motion_path.is_file() or motion_path.suffix.lower() != ".npz":
        logger.warning("Motion overlay disabled; motion file is not a readable .npz: {}", motion_path)
        return None

    try:
        with np.load(motion_path, allow_pickle=True) as data:
            body_names = _decode_npz_names(np.asarray(data["body_names"]))
            joint_names = _decode_npz_names(np.asarray(data["joint_names"]))
            root_idx = _resolve_motion_root_body_index(body_names)
            root_pos_w = np.asarray(data["body_pos_w"][:, root_idx], dtype=np.float32)
            root_quat_wxyz = np.asarray(data["body_quat_w"][:, root_idx], dtype=np.float32)
            root_quat_wxyz = np.asarray([_normalize_quaternion_wxyz(quat) for quat in root_quat_wxyz], dtype=np.float32)

            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            if joint_pos.ndim != 2:
                raise ValueError(f"joint_pos must be 2-D, got {joint_pos.shape}")
            if joint_pos.shape[1] == len(joint_names) + 7:
                joint_pos = joint_pos[:, 7:]

            joint_name_to_index = {name: idx for idx, name in enumerate(joint_names)}
            fallback_joint = _build_default_joint_viser(robot_config, viser_joint_names)
            joint_pos_viser = np.tile(fallback_joint.reshape(1, -1), (joint_pos.shape[0], 1)).astype(np.float32)
            for viser_idx, joint_name in enumerate(viser_joint_names):
                motion_idx = joint_name_to_index.get(joint_name)
                if motion_idx is not None and motion_idx < joint_pos.shape[1]:
                    joint_pos_viser[:, viser_idx] = joint_pos[:, motion_idx]

            object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float32) if "object_pos_w" in data.files else None
            object_quat_wxyz = np.asarray(data["object_quat_w"], dtype=np.float32) if "object_quat_w" in data.files else None
            if object_quat_wxyz is not None:
                object_quat_wxyz = np.asarray(
                    [_normalize_quaternion_wxyz(quat) for quat in object_quat_wxyz],
                    dtype=np.float32,
                )

            object_size = np.asarray(data["object_size"], dtype=np.float32).reshape(-1) if "object_size" in data.files else None
            object_urdf_path = None
            if "object_urdf_path" in data.files:
                object_urdf_raw = _scalar_str(data["object_urdf_path"])
                if object_urdf_raw:
                    object_urdf_path = Path(object_urdf_raw).expanduser()
                    if not object_urdf_path.is_absolute():
                        object_urdf_path = motion_path.parent / object_urdf_path
                    object_urdf_path = object_urdf_path.resolve()

            fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data.files else 50.0
    except Exception as exc:
        logger.warning("Motion overlay disabled; failed to load {}: {}", motion_path, exc)
        return None

    object_mesh = _load_motion_object_mesh(object_urdf_path, object_size) if object_urdf_path is not None else None
    return MotionOverlay(
        path=motion_path,
        fps=max(float(fps), 1.0),
        root_pos_w=root_pos_w,
        root_quat_wxyz=root_quat_wxyz,
        joint_pos_viser=joint_pos_viser,
        object_pos_w=object_pos_w,
        object_quat_wxyz=object_quat_wxyz,
        object_mesh=object_mesh,
    )


def _terminate_rollout_process(
    proc: subprocess.Popen[bytes] | subprocess.Popen[str] | None,
    *,
    process_group: bool,
    timeout_sec: float = 10.0,
) -> None:
    if proc is None or proc.poll() is not None:
        return
    if process_group:
        os.killpg(proc.pid, signal.SIGTERM)
    else:
        proc.terminate()
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    if process_group:
        os.killpg(proc.pid, signal.SIGKILL)
    else:
        proc.kill()
    proc.wait(timeout=5.0)


def _build_rollout_command(cfg: MujocoSimStateViewerConfig, motion_path: Path | None = None) -> list[str]:
    run_script = _resolve_repo_path(cfg.run_script)
    if not run_script.is_file():
        raise FileNotFoundError(f"run script not found: {run_script}")
    command = [str(run_script)]
    if motion_path is not None:
        command.append(str(_resolve_repo_path_preserve_symlink(motion_path)))
    elif cfg.motion_file:
        command.append(str(_resolve_repo_path_preserve_symlink(cfg.motion_file)))
    if cfg.model_path:
        command.append(str(_resolve_repo_path(cfg.model_path)))
    return command


def _list_motion_choices(initial_motion_path: Path) -> tuple[tuple[str, ...], dict[str, Path], str, Path]:
    initial_motion_path = _resolve_repo_path_preserve_symlink(initial_motion_path)
    motion_dir = initial_motion_path.parent
    motion_paths = sorted(motion_dir.glob("*.npz")) if motion_dir.is_dir() else []
    motion_choice_map = {path.stem: _resolve_repo_path_preserve_symlink(path) for path in motion_paths}
    initial_label = initial_motion_path.stem
    if initial_label not in motion_choice_map:
        motion_choice_map[initial_label] = initial_motion_path
    return tuple(motion_choice_map.keys()), motion_choice_map, initial_label, motion_dir


def _resolve_motion_path_input(raw_value: object, *, base_dir: Path, choices: dict[str, Path]) -> Path | None:
    raw = str(raw_value).strip()
    if not raw:
        return None
    if raw in choices:
        return choices[raw]

    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        if not candidate.suffix:
            base_npz = _resolve_repo_path_preserve_symlink(base_dir / f"{raw}.npz")
            if base_npz.is_file():
                return base_npz
        base_candidate = _resolve_repo_path_preserve_symlink(base_dir / candidate)
        if base_candidate.is_file():
            return base_candidate
        candidate = REPO_ROOT / candidate
    return _resolve_repo_path_preserve_symlink(candidate)


def _motion_path_status(path: Path | None) -> str:
    if path is None:
        return "empty"
    if path.suffix.lower() != ".npz":
        return "not_npz"
    if not path.is_file():
        return "missing"
    return "ok"


def _motion_path_display(path: Path | None) -> str:
    if path is None:
        return "n/a"
    return path.name if path.name else str(path)


def _motion_init_mode_from_default_pose(enabled: bool) -> str:
    return "training_default_pose" if enabled else "raw_motion"


def _motion_init_label(*, manual: bool, default_pose: bool) -> str:
    mode = _motion_init_mode_from_default_pose(default_pose)
    source = "manual" if manual else "auto"
    return f"{source}({mode})"


def _infer_default_pose_start(cfg: MujocoSimStateViewerConfig) -> bool:
    env_mode = os.environ.get("SIM_MOTION_INIT_MODE", "").strip().lower().replace("-", "_")
    if env_mode:
        return env_mode == "training_default_pose"

    try:
        import onnx

        model_path = _resolve_repo_path(cfg.model_path)
        model = onnx.load(str(model_path))
        metadata = {}
        for prop in model.metadata_props:
            try:
                metadata[prop.key] = json.loads(prop.value)
            except Exception:
                metadata[prop.key] = prop.value

        motion_cfg = (
            metadata.get("experiment_config", {})
            .get("command", {})
            .get("setup_terms", {})
            .get("motion_command", {})
            .get("params", {})
            .get("motion_config", {})
        )
        if not isinstance(motion_cfg, dict):
            return False
        return bool(
            (
                motion_cfg.get("enable_default_pose_prepend")
                and float(motion_cfg.get("default_pose_prepend_duration_s", 0.0) or 0.0) > 0.0
            )
            or (
                motion_cfg.get("enable_default_pose_append")
                and float(motion_cfg.get("default_pose_append_duration_s", 0.0) or 0.0) > 0.0
            )
        )
    except Exception as exc:
        logger.debug("Unable to infer default-pose rollout start from model metadata: {}", exc)
        return False


def view_sim_state(cfg: MujocoSimStateViewerConfig) -> None:
    robot_config = _resolve_robot_config(cfg.robot)
    robot_urdf_path = _resolve_robot_urdf_path(robot_config)
    snapshot_path_default = _resolve_repo_path(cfg.mujoco_object_geom_snapshot_path)

    port = resolve_viser_port(cfg.port)
    suppress_direct_url = os.environ.get("HOLOSOMA_VISER_SUPPRESS_DIRECT_URL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if suppress_direct_url:
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            server = viser.ViserServer(port=port, verbose=False)
    else:
        server = viser.ViserServer(port=port, verbose=True)
    rollout_tty_input = bool(cfg.rollout_tty_input or _truthy_env(os.environ.get("HOLOSOMA_ROLLOUT_TTY_INPUT")))
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    ref_root = server.scene.add_frame("/robot_ref", show_axes=bool(cfg.show_ref_body))
    object_root = server.scene.add_frame("/object", show_axes=False)
    object_visual_root = server.scene.add_frame("/object/visual_geoms", show_axes=False)
    object_collision_root = server.scene.add_frame("/object/collision_geoms", show_axes=False)
    server.scene.add_grid("/grid", width=cfg.grid_size, height=cfg.grid_size, position=(0.0, 0.0, 0.0))

    viser_urdf_kwargs = {
        "urdf_or_path": robot_urdf_path,
        "root_node_name": "/robot",
    }
    viser_urdf_signature = inspect.signature(ViserUrdf)
    if "load_collision_meshes" in viser_urdf_signature.parameters:
        viser_urdf_kwargs["load_collision_meshes"] = True
    if "collision_mesh_color_override" in viser_urdf_signature.parameters:
        viser_urdf_kwargs["collision_mesh_color_override"] = (0.15, 0.7, 1.0, 0.28)
    vr = ViserUrdf(server, **viser_urdf_kwargs)
    viser_joint_names = list(vr.get_actuated_joint_names())
    default_joint_viser = _build_default_joint_viser(robot_config, viser_joint_names)
    vr.update_cfg(default_joint_viser)
    if hasattr(vr, "show_collision"):
        vr.show_collision = bool(cfg.show_robot_collision)

    initial_motion_path = _resolve_repo_path_preserve_symlink(cfg.motion_file)
    motion_choices, motion_choice_map, initial_motion_choice, motion_choices_dir = _list_motion_choices(initial_motion_path)
    motion_overlay: MotionOverlay | None = None
    motion_root = server.scene.add_frame("/motion_ref", show_axes=False)
    motion_robot_root = server.scene.add_frame("/motion_ref/robot", show_axes=False)
    motion_object_root = server.scene.add_frame("/motion_ref/object", show_axes=False)
    motion_root.visible = False
    motion_robot_root.visible = False
    motion_object_root.visible = False
    motion_vr: ViserUrdf | None = None
    motion_object_mesh_handle = None

    def _ensure_motion_urdf() -> ViserUrdf:
        nonlocal motion_vr
        if motion_vr is not None:
            return motion_vr
        motion_urdf_kwargs = {
            "urdf_or_path": robot_urdf_path,
            "root_node_name": "/motion_ref/robot",
        }
        if "load_collision_meshes" in viser_urdf_signature.parameters:
            motion_urdf_kwargs["load_collision_meshes"] = True
        if "collision_mesh_color_override" in viser_urdf_signature.parameters:
            motion_urdf_kwargs["collision_mesh_color_override"] = (1.0, 0.62, 0.05, 0.33)
        motion_vr = ViserUrdf(server, **motion_urdf_kwargs)
        if hasattr(motion_vr, "show_visual"):
            motion_vr.show_visual = False
        if hasattr(motion_vr, "show_collision"):
            motion_vr.show_collision = True
        return motion_vr

    def _clear_motion_object_mesh() -> None:
        nonlocal motion_object_mesh_handle
        if motion_object_mesh_handle is not None:
            try:
                motion_object_mesh_handle.remove()
            except Exception:
                pass
            motion_object_mesh_handle = None

    def _set_motion_overlay_path(motion_path: Path) -> bool:
        nonlocal motion_overlay, motion_object_mesh_handle
        motion_overlay = _load_motion_overlay(motion_path, robot_config, viser_joint_names)
        _clear_motion_object_mesh()
        if motion_overlay is None:
            motion_root.visible = False
            motion_robot_root.visible = False
            motion_object_root.visible = False
            return False

        overlay_vr = _ensure_motion_urdf()
        overlay_vr.update_cfg(motion_overlay.joint_pos_viser[0])
        if motion_overlay.object_mesh is not None:
            motion_object_mesh_handle = server.scene.add_mesh_simple(
                "/motion_ref/object/mesh",
                vertices=np.asarray(motion_overlay.object_mesh.vertices, dtype=np.float32),
                faces=np.asarray(motion_overlay.object_mesh.faces, dtype=np.int32),
                color=(255, 168, 25),
                opacity=0.36,
                side="double",
                visible=False,
            )
        return True

    _set_motion_overlay_path(initial_motion_path)

    object_visual_handles: list[object] = []
    object_collision_handles: list[object] = []
    loaded_object_snapshot_path: Path | None = None
    object_visual_uses_collision_fallback = False

    def _clear_object_geom_handles() -> None:
        nonlocal object_visual_handles, object_collision_handles, loaded_object_snapshot_path, object_visual_uses_collision_fallback
        for handle in [*object_visual_handles, *object_collision_handles]:
            try:
                handle.remove()
            except Exception:
                pass
        object_visual_handles = []
        object_collision_handles = []
        loaded_object_snapshot_path = None
        object_visual_uses_collision_fallback = False
        object_visual_root.visible = False
        object_collision_root.visible = False

    def _set_object_mesh_visibility(*, show_object: bool, mesh_mode: str) -> None:
        effective_show_visual = False
        effective_show_collision = False
        normalized_mesh_mode = _resolve_object_mesh_mode(mesh_mode)

        if show_object:
            if normalized_mesh_mode == "collision":
                effective_show_collision = bool(object_collision_handles)
            else:
                effective_show_visual = bool(object_visual_handles)
                if not effective_show_visual and object_visual_uses_collision_fallback:
                    effective_show_collision = bool(object_collision_handles)

        object_visual_root.visible = effective_show_visual
        object_collision_root.visible = effective_show_collision
        for handle in object_visual_handles:
            handle.visible = effective_show_visual
        for handle in object_collision_handles:
            handle.visible = effective_show_collision

    def _load_object_geom_handles(snapshot_path: Path) -> bool:
        nonlocal loaded_object_snapshot_path, object_visual_handles, object_collision_handles, object_visual_uses_collision_fallback
        actor_payload = _load_mujoco_object_geom_snapshot(snapshot_path, cfg.object_actor_name)
        if actor_payload is None:
            return False

        geoms = actor_payload.get("geoms")
        if not isinstance(geoms, list):
            return False

        _clear_object_geom_handles()
        visual_count = 0
        collision_count = 0
        for geom_idx, geom_entry_raw in enumerate(geoms):
            if not isinstance(geom_entry_raw, dict):
                continue
            geom_name = str(geom_entry_raw.get("name", f"geom_{geom_idx}"))
            geom_pos = np.asarray(geom_entry_raw.get("relative_pos", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(3)
            geom_quat_wxyz = _normalize_quaternion_wxyz(geom_entry_raw.get("relative_quat_wxyz", [1.0, 0.0, 0.0, 0.0]))
            is_collision = bool(geom_entry_raw.get("is_collision", False))

            visual_mesh = _mesh_arrays_from_mujoco_geom(geom_entry_raw, collision_view=False)
            if visual_mesh is not None and (not is_collision or _geom_supports_visual_mesh(geom_entry_raw)):
                vertices, faces, color, opacity = visual_mesh
                vertices = _transform_vertices_local(vertices, geom_pos, geom_quat_wxyz)
                object_visual_handles.append(
                    server.scene.add_mesh_simple(
                        f"/object/visual_geoms/{geom_name}_{geom_idx}",
                        vertices=vertices,
                        faces=faces,
                        color=color,
                        opacity=opacity,
                        side="double",
                        visible=False,
                    )
                )
                visual_count += 1

            if is_collision:
                collision_mesh = _mesh_arrays_from_mujoco_geom(geom_entry_raw, collision_view=True)
                if collision_mesh is not None:
                    vertices, faces, color, opacity = collision_mesh
                    vertices = _transform_vertices_local(vertices, geom_pos, geom_quat_wxyz)
                    object_collision_handles.append(
                        server.scene.add_mesh_simple(
                            f"/object/collision_geoms/{geom_name}_{geom_idx}",
                            vertices=vertices,
                            faces=faces,
                            color=color,
                            opacity=opacity,
                            side="double",
                            visible=False,
                        )
                    )
                    collision_count += 1

        object_visual_uses_collision_fallback = not object_visual_handles and bool(object_collision_handles)
        loaded_object_snapshot_path = snapshot_path
        logger.info(
            "Loaded MuJoCo object geoms from {} (visual={}, collision={}, collision_fallback={})",
            snapshot_path,
            visual_count,
            collision_count,
            object_visual_uses_collision_fallback,
        )
        return True

    with server.gui.add_folder("Rollout", order=10.0):
        rollout_md = server.gui.add_markdown("Viewer only", visible=False)
        motion_clip_dropdown = server.gui.add_dropdown(
            "Motion clip",
            options=motion_choices,
            initial_value=initial_motion_choice,
        )
        motion_path_text = server.gui.add_text("Motion path", initial_value=str(initial_motion_path))
        auto_default_pose_start = _infer_default_pose_start(cfg)
        initial_manual_motion_init = bool(cfg.manual_motion_init_mode or cfg.reset_to_default_pose)
        manual_motion_init_mode_cb = server.gui.add_checkbox(
            "Manual init mode",
            initial_value=initial_manual_motion_init,
        )
        reset_to_default_pose_cb = server.gui.add_checkbox(
            "Start from default pose",
            initial_value=bool(cfg.reset_to_default_pose if initial_manual_motion_init else auto_default_pose_start),
        )
        reset_to_default_pose_cb.disabled = not initial_manual_motion_init
        manual_rollout_btn = server.gui.add_button("Manual policy rollout")
        manual_rollout_btn.disabled = bool(cfg.launch_env_only)
        reset_rollout_btn = server.gui.add_button("Reset rollout")

    auto_motion_init_env = os.environ.get("SIM_MOTION_INIT_MODE", "").strip().lower().replace("-", "_")
    if auto_motion_init_env not in {"raw_motion", "training_default_pose"}:
        auto_motion_init_env = ""

    with server.gui.add_folder("Manual Root Command", order=20.0):
        manual_root_enabled_cb = server.gui.add_checkbox("Manual mode", initial_value=bool(cfg.manual_root_enabled))
        manual_root_mode_dropdown = server.gui.add_dropdown(
            "Mode",
            options=("manual", "offset"),
            initial_value=str(cfg.manual_root_mode) if str(cfg.manual_root_mode) in {"manual", "offset"} else "manual",
        )
        manual_root_dx = server.gui.add_number("dX", initial_value=float(cfg.manual_root_dx), min=-3.0, max=3.0, step=0.01)
        manual_root_dy = server.gui.add_number("dY", initial_value=float(cfg.manual_root_dy), min=-3.0, max=3.0, step=0.01)
        manual_root_dyaw = server.gui.add_number("dYaw", initial_value=float(cfg.manual_root_dyaw), min=-3.1416, max=3.1416, step=0.01)
        manual_root_zero_btn = server.gui.add_button("Zero command")
        manual_root_md = server.gui.add_markdown(f"Publishing disabled on port `{cfg.sparse_root_command_port}`")

    depth_image_shape = (
        max(int(cfg.depth_height), 1) * max(int(cfg.depth_display_scale), 1),
        max(int(cfg.depth_width), 1) * max(int(cfg.depth_display_scale), 1),
        3,
    )
    with server.gui.add_folder("Depth", order=30.0):
        show_depth_cb = server.gui.add_checkbox("Show policy depth", initial_value=bool(cfg.show_depth))
        depth_image = server.gui.add_image(
            np.zeros(depth_image_shape, dtype=np.uint8),
            label="perception_obs depth",
            visible=bool(cfg.show_depth),
        )
        depth_md = server.gui.add_markdown("Waiting for perception_obs...")

    with server.gui.add_folder("Viser GUI", order=90.0, expand_by_default=False):
        with server.gui.add_folder("Sim State", order=10.0, expand_by_default=False):
            state_md = server.gui.add_markdown("Waiting for simulator state...")
            actor_md = server.gui.add_markdown("")

        with server.gui.add_folder("Display", order=20.0, expand_by_default=False):
            recenter_cb = server.gui.add_checkbox("Recenter XY", initial_value=bool(cfg.recenter_xy))
            show_object_cb = server.gui.add_checkbox("Show object (MuJoCo)", initial_value=bool(cfg.show_object))
            object_mesh_mode_dropdown = server.gui.add_dropdown(
                "Object mesh",
                options=OBJECT_MESH_MODE_OPTIONS,
                initial_value=_resolve_object_mesh_mode(
                    cfg.object_mesh_mode,
                    show_object_collision=bool(cfg.show_object_collision),
                ),
            )
            show_robot_collision_cb = server.gui.add_checkbox(
                "Show robot collision (URDF)",
                initial_value=bool(cfg.show_robot_collision),
            )
            show_ref_cb = server.gui.add_checkbox("Show policy ref body", initial_value=bool(cfg.show_ref_body))
            reset_offset_btn = server.gui.add_button("Reset offset")

        with server.gui.add_folder("Motion Overlay", order=30.0, expand_by_default=False):
            show_motion_overlay_cb = server.gui.add_checkbox(
                "Show motion overlay",
                initial_value=bool(cfg.show_motion_overlay and motion_overlay is not None),
            )
            show_motion_robot_cb = server.gui.add_checkbox("Motion robot", initial_value=False)
            show_motion_object_cb = server.gui.add_checkbox("Motion object", initial_value=False)
            motion_md = server.gui.add_markdown(
                "Motion overlay unavailable" if motion_overlay is None else "Waiting for sim-state..."
            )

    sub = SimStateSub(port=cfg.state_port)
    sub.start()
    perception_sub = PerceptionObsSub(port=cfg.perception_obs_port)
    perception_sub.start()
    perception_shm_sub = PerceptionObsShmSub(name=cfg.perception_obs_shm_name)
    perception_shm_sub.start()
    control_pub = SimControlPush(port=cfg.control_port)
    control_pub.start()
    manual_root_pub: ManualRootCommandPub | None = None
    if bool(cfg.manual_root_publisher_enabled):
        manual_root_pub = ManualRootCommandPub(port=cfg.sparse_root_command_port)
        manual_root_pub.start()
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(_signum, _frame) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _handle_sigterm)

    offset_xy = np.zeros(2, dtype=np.float32)
    offset_initialized = False
    received_first_state = False
    rollout_proc: subprocess.Popen | None = None
    rollout_log_handle: TextIOWrapper | None = None
    rollout_restart_count = 0
    last_rollout_skip_policy: bool | None = None
    last_rollout_motion_init_mode: str | None = None
    last_rollout_motion_path: Path | None = None
    rollout_restart_lock = threading.Lock()
    pending_restart_reason = "startup" if cfg.launch_rollout else None
    last_rollout_reason = "idle"
    rollout_log_path = _resolve_repo_path(cfg.rollout_log_path)
    auto_reset_scheduled_at: float | None = None
    auto_reset_done = False
    reset_request_time_monotonic: float | None = None
    reset_pending_clock_rewind = False
    pre_reset_sim_time_ms: int | None = None
    last_seen_sim_time_ms: int | None = None

    def _selected_motion_init() -> tuple[str, str | None]:
        manual = bool(manual_motion_init_mode_cb.value)
        if not manual and auto_motion_init_env:
            default_pose = auto_motion_init_env == "training_default_pose"
            return _motion_init_label(manual=False, default_pose=default_pose), auto_motion_init_env
        default_pose = bool(reset_to_default_pose_cb.value) if manual else auto_default_pose_start
        label = _motion_init_label(manual=manual, default_pose=default_pose)
        mode = _motion_init_mode_from_default_pose(default_pose) if manual else None
        return label, mode

    def _selected_motion_path(*, validate: bool = True) -> Path | None:
        dropdown_motion_path = motion_choice_map.get(str(motion_clip_dropdown.value))
        if dropdown_motion_path is not None:
            return dropdown_motion_path if (not validate or _motion_path_status(dropdown_motion_path) == "ok") else None
        motion_path = _resolve_motion_path_input(
            motion_path_text.value,
            base_dir=motion_choices_dir,
            choices=motion_choice_map,
        )
        if not validate:
            return motion_path
        return motion_path if _motion_path_status(motion_path) == "ok" else None

    def _motion_choice_for_path(motion_path: Path | None) -> str | None:
        if motion_path is None:
            return None
        for label, choice_path in motion_choice_map.items():
            if choice_path == motion_path:
                return label
        return None

    def _refresh_motion_init_controls(disabled: bool = False) -> None:
        manual_motion_init_mode_cb.disabled = disabled
        reset_to_default_pose_cb.disabled = bool(disabled or not manual_motion_init_mode_cb.value)

    def _refresh_motion_selection_controls(disabled: bool = False) -> None:
        motion_clip_dropdown.disabled = disabled
        motion_path_text.disabled = disabled

    def _set_motion_overlay_visibility() -> None:
        overlay_visible = bool(show_motion_overlay_cb.value and motion_overlay is not None)
        robot_visible = bool(overlay_visible and show_motion_robot_cb.value)
        object_visible = bool(overlay_visible and show_motion_object_cb.value)
        motion_root.visible = overlay_visible
        motion_robot_root.visible = robot_visible
        motion_object_root.visible = object_visible
        if motion_object_mesh_handle is not None:
            motion_object_mesh_handle.visible = object_visible

    def _motion_frame_index(sim_time_ms: int) -> int:
        if motion_overlay is None:
            return 0
        frame_count = int(motion_overlay.root_pos_w.shape[0])
        if frame_count <= 0:
            return 0
        idx = int(round(max(float(sim_time_ms), 0.0) * motion_overlay.fps / 1000.0))
        return int(np.clip(idx, 0, frame_count - 1))

    def _update_motion_overlay(sim_time_ms: int, root_state: np.ndarray, object_state: np.ndarray | None) -> None:
        if motion_overlay is None:
            _set_motion_overlay_visibility()
            motion_md.content = "Motion overlay unavailable"
            return

        frame_idx = _motion_frame_index(sim_time_ms)
        motion_root_pos = motion_overlay.root_pos_w[frame_idx].copy()
        motion_object_pos = (
            motion_overlay.object_pos_w[frame_idx].copy()
            if motion_overlay.object_pos_w is not None and motion_overlay.object_pos_w.shape[0] > frame_idx
            else None
        )
        if bool(recenter_cb.value):
            motion_root_pos[:2] -= offset_xy
            if motion_object_pos is not None:
                motion_object_pos[:2] -= offset_xy

        with server.atomic():
            motion_robot_root.position = tuple(motion_root_pos.tolist())
            motion_robot_root.wxyz = tuple(motion_overlay.root_quat_wxyz[frame_idx].tolist())
            if motion_vr is not None:
                motion_vr.update_cfg(motion_overlay.joint_pos_viser[frame_idx])

            if motion_object_pos is not None and motion_overlay.object_quat_wxyz is not None:
                motion_object_root.position = tuple(motion_object_pos.tolist())
                motion_object_root.wxyz = tuple(motion_overlay.object_quat_wxyz[frame_idx].tolist())

            _set_motion_overlay_visibility()

        root_pos_err = float(np.linalg.norm(root_state[:3] - motion_overlay.root_pos_w[frame_idx]))
        root_xy_err = float(np.linalg.norm(root_state[:2] - motion_overlay.root_pos_w[frame_idx, :2]))
        object_line = "object_err: `n/a`"
        if object_state is not None and motion_overlay.object_pos_w is not None and motion_overlay.object_pos_w.shape[0] > frame_idx:
            object_pos_err = float(np.linalg.norm(object_state[:3] - motion_overlay.object_pos_w[frame_idx]))
            object_xy_err = float(np.linalg.norm(object_state[:2] - motion_overlay.object_pos_w[frame_idx, :2]))
            object_line = f"object_err: `{object_pos_err:.5f} m` | object_xy_err: `{object_xy_err:.5f} m`"
        motion_md.content = (
            f"motion: `{motion_overlay.path.name}`\n\n"
            f"frame: `{frame_idx}` / `{motion_overlay.root_pos_w.shape[0] - 1}`\n\n"
            f"root_err: `{root_pos_err:.5f} m` | root_xy_err: `{root_xy_err:.5f} m`\n\n"
            f"{object_line}"
        )

    def _refresh_rollout_md() -> None:
        if rollout_proc is None:
            proc_state = "stopped"
            pid = "n/a"
        else:
            poll = rollout_proc.poll()
            proc_state = "running" if poll is None else f"exited({poll})"
            pid = str(rollout_proc.pid)
        if last_rollout_skip_policy is None:
            policy_state = "env"
        elif last_rollout_skip_policy:
            policy_state = "skipped"
        else:
            policy_state = "enabled"
        selected_motion_init_mode, _selected_motion_init_env = _selected_motion_init()
        active_motion_init_mode = last_rollout_motion_init_mode or "n/a"
        selected_motion_path = _selected_motion_path(validate=False)
        selected_motion_status = _motion_path_status(selected_motion_path)
        active_motion = _motion_path_display(last_rollout_motion_path)
        rollout_md.content = (
            f"status: `{proc_state}`\n\n"
            f"pid: `{pid}`\n\n"
            f"restart_count: `{rollout_restart_count}`\n\n"
            f"last_reason: `{last_rollout_reason}`\n\n"
            f"launch_env_only: `{bool(cfg.launch_env_only)}`\n\n"
            f"policy: `{policy_state}`\n\n"
            f"selected_motion: `{_motion_path_display(selected_motion_path)}` ({selected_motion_status})\n\n"
            f"active_motion: `{active_motion}`\n\n"
            f"selected_init: `{selected_motion_init_mode}`\n\n"
            f"active_init: `{active_motion_init_mode}`\n\n"
            f"tty_input: `{rollout_tty_input}`\n\n"
            f"keyboard_root: `{bool(cfg.keyboard_root_command)}` value=`{float(cfg.keyboard_root_command_value):.3f}`\n\n"
            f"reset_mode: `button-only sim-control`\n\n"
            f"log_path: `{rollout_log_path}`"
        )

    def _refresh_depth_view() -> None:
        visible = bool(show_depth_cb.value)
        depth_image.visible = visible
        if not visible:
            depth_md.content = "Hidden"
            return

        expected_dim = max(int(cfg.depth_height), 1) * max(int(cfg.depth_width), 1)
        payload = perception_sub.get_payload()
        values = payload.get("perception_obs") if payload is not None else None
        data_source = f"zmq:{cfg.perception_obs_port}"

        depth_obs_flat: np.ndarray | None = None
        if values is not None:
            try:
                depth_obs_flat = np.asarray(values, dtype=np.float32).reshape(-1)
            except (TypeError, ValueError) as exc:
                depth_md.content = f"Failed to parse perception_obs: `{exc}`"
                return

        if depth_obs_flat is None or depth_obs_flat.size != expected_dim:
            shm_obs = perception_shm_sub.get_obs(expected_dim)
            if shm_obs is not None:
                depth_obs_flat = np.asarray(shm_obs, dtype=np.float32).reshape(-1)
                data_source = f"shm:{cfg.perception_obs_shm_name}"

        if depth_obs_flat is None:
            depth_md.content = (
                f"Waiting for perception_obs on port `{cfg.perception_obs_port}` "
                f"or shared memory `{cfg.perception_obs_shm_name}`..."
            )
            return

        if depth_obs_flat.size != expected_dim:
            depth_md.content = f"perception_obs dim mismatch: got `{depth_obs_flat.size}`, expected `{expected_dim}`"
            return

        depth_obs = depth_obs_flat.reshape(max(int(cfg.depth_height), 1), max(int(cfg.depth_width), 1))
        depth_m = _depth_obs_to_meters(
            depth_obs,
            normalized=bool(cfg.depth_obs_normalized),
            near=float(cfg.depth_near),
            far=float(cfg.depth_far),
        )
        depth_image.image = _scale_image_nearest(
            _depth_to_rgb(depth_m, float(cfg.depth_near), float(cfg.depth_far)),
            int(cfg.depth_display_scale),
        )

        obs_finite = depth_obs[np.isfinite(depth_obs)]
        if obs_finite.size:
            obs_range = f"[{float(obs_finite.min()):.4f}, {float(obs_finite.max()):.4f}]"
        else:
            obs_range = "n/a"
        depth_min_m, depth_max_m, valid_count = _valid_depth_stats(
            depth_m,
            float(cfg.depth_near),
            float(cfg.depth_far),
        )
        if depth_min_m is None or depth_max_m is None:
            depth_range = "n/a"
        else:
            depth_range = f"[{depth_min_m:.3f}, {depth_max_m:.3f}] m"
        sim_time = payload.get("sim_time_ms", "n/a") if payload is not None else "n/a"
        depth_md.content = (
            f"source: `{data_source}`\n\n"
            f"shape: `{cfg.depth_height}x{cfg.depth_width}`\n\n"
            f"sim_time_ms: `{sim_time}`\n\n"
            f"obs_normalized: `{bool(cfg.depth_obs_normalized)}`\n\n"
            f"obs_range: `{obs_range}`\n\n"
            f"depth_valid: `{valid_count}/{expected_dim}`\n\n"
            f"depth_range: `{depth_range}`\n\n"
            "color: close red / mid green / far-or-max blue / nonfinite black"
        )

    def _publish_manual_root_command() -> None:
        command = [
            float(manual_root_dx.value),
            float(manual_root_dy.value),
            float(manual_root_dyaw.value),
        ]
        enabled = bool(manual_root_enabled_cb.value)
        mode = str(manual_root_mode_dropdown.value)
        if manual_root_pub is not None:
            manual_root_pub.publish(enabled=enabled, mode=mode, command=command)
        status = "manual" if enabled else "motion"
        publisher_status = "enabled" if manual_root_pub is not None and manual_root_pub.enabled else "disabled"
        manual_root_md.content = (
            f"status: `{status}`\n\n"
            f"mode: `{mode}`\n\n"
            f"command: `[{command[0]:.3f}, {command[1]:.3f}, {command[2]:.3f}]`\n\n"
            f"publisher: `{publisher_status}`\n\n"
            f"port: `{cfg.sparse_root_command_port}`"
        )

    def _stop_rollout() -> None:
        nonlocal rollout_proc, rollout_log_handle
        if rollout_proc is not None:
            logger.info("Stopping rollout pid={}", rollout_proc.pid)
            _terminate_rollout_process(rollout_proc, process_group=not rollout_tty_input)
            rollout_proc = None
        if rollout_log_handle is not None:
            rollout_log_handle.close()
            rollout_log_handle = None

    def _restart_rollout(reason: str, *, skip_policy: bool | None = None) -> None:
        nonlocal rollout_proc, rollout_log_handle, rollout_restart_count, last_rollout_skip_policy, last_rollout_motion_init_mode, last_rollout_motion_path, offset_initialized, received_first_state, pending_restart_reason, last_rollout_reason, auto_reset_scheduled_at, auto_reset_done, reset_request_time_monotonic, reset_pending_clock_rewind, pre_reset_sim_time_ms, last_seen_sim_time_ms
        if bool(cfg.launch_env_only):
            skip_policy = True
        selected_motion_path = _selected_motion_path(validate=True)
        if selected_motion_path is None:
            raw_motion_value = str(motion_path_text.value).strip() or "empty"
            rollout_md.content = f"Invalid motion path `{raw_motion_value}`; rollout was not restarted."
            logger.warning("Rollout restart requested with invalid motion path: {}", raw_motion_value)
            return

        if not rollout_restart_lock.acquire(blocking=False):
            logger.info("Ignoring rollout restart while another restart is in progress ({})", reason)
            return
        try:
            manual_rollout_btn.disabled = True
            reset_rollout_btn.disabled = True
            _refresh_motion_init_controls(disabled=True)
            _refresh_motion_selection_controls(disabled=True)
            _stop_rollout()
            _clear_object_geom_handles()
            _set_motion_overlay_path(selected_motion_path)
            if motion_overlay is None:
                motion_md.content = f"Motion overlay unavailable: `{selected_motion_path.name}`"
            else:
                motion_md.content = f"motion: `{selected_motion_path.name}`\n\nWaiting for sim-state..."
            _set_motion_overlay_visibility()
            command = _build_rollout_command(cfg, selected_motion_path)
            env = os.environ.copy()
            motion_init_label, motion_init_env = _selected_motion_init()
            env["HOLOSOMA_MJ_TRACK_INTERNAL_CORE"] = "1"
            env["HOLOSOMA_DISABLE_AUTO_RESET"] = "1"
            env["HOLOSOMA_DISABLE_MOTION_END_RESET"] = "1"
            env["HOLOSOMA_DISABLE_CLIP_END_RESET"] = "1"
            env["HOLOSOMA_DISABLE_BAD_TRACKING_RESET"] = "1"
            env.pop("SIM_MOTION_INIT_MODE", None)
            env.pop("HOLOSOMA_RESET_TO_DEFAULT_POSE", None)
            env.pop("HOLOSOMA_DEFAULT_POSE_INIT", None)
            env.pop("HOLOSOMA_MOTION_INIT_MANUAL", None)
            # The wrapper exports the initial clip object. Let mj_track.sh derive it again for the selected motion.
            env.pop("OBJECT_URDF", None)
            env.pop("SIM2SIM_CLIP_OBJECT_URDF_PATH", None)
            if motion_init_env is not None:
                if bool(manual_motion_init_mode_cb.value):
                    env["HOLOSOMA_MOTION_INIT_MANUAL"] = "1"
                env["SIM_MOTION_INIT_MODE"] = motion_init_env
                default_pose_enabled = "1" if motion_init_env == "training_default_pose" else "0"
                env["HOLOSOMA_RESET_TO_DEFAULT_POSE"] = default_pose_enabled
                env["HOLOSOMA_DEFAULT_POSE_INIT"] = default_pose_enabled
            if cfg.launch_run_seconds > 0:
                env["RUN_SECONDS"] = str(cfg.launch_run_seconds)
                env.pop("HOLOSOMA_MJ_TRACK_RUN_FOREVER", None)
            else:
                env.pop("RUN_SECONDS", None)
                env["HOLOSOMA_MJ_TRACK_RUN_FOREVER"] = "1"
            env["TRAINING_HEADLESS"] = "True" if cfg.training_headless else "False"
            env["SIM_STATE_PORT"] = str(cfg.state_port)
            env["PERCEPTION_OBS_PORT"] = str(cfg.perception_obs_port)
            env["SIM_CONTROL_PORT"] = str(cfg.control_port)
            env["SPARSE_ROOT_COMMAND_PORT"] = str(cfg.sparse_root_command_port)
            env["ENABLE_EXTERNAL_SPARSE_ROOT_COMMAND"] = "1"
            if rollout_tty_input:
                env["HOLOSOMA_POLICY_TTY_INPUT"] = "1"
            else:
                env.pop("HOLOSOMA_POLICY_TTY_INPUT", None)
            if cfg.keyboard_root_command:
                env["HOLOSOMA_KEYBOARD_ROOT_COMMAND"] = "1"
                env["HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE"] = str(float(cfg.keyboard_root_command_value))
                env["HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE"] = str(cfg.keyboard_root_command_mode)
            else:
                env.pop("HOLOSOMA_KEYBOARD_ROOT_COMMAND", None)
                env.pop("HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE", None)
                env.pop("HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE", None)
            env["HOLOSOMA_MUJOCO_OBJECT_GEOM_SNAPSHOT_PATH"] = str(snapshot_path_default)
            if skip_policy is not None:
                env["SKIP_POLICY"] = "1" if skip_policy else "0"
            last_rollout_skip_policy = str(env.get("SKIP_POLICY", "0")).strip().lower() in {"1", "true", "yes", "on"}
            try:
                snapshot_path_default.unlink()
            except FileNotFoundError:
                pass
            rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
            rollout_log_handle = rollout_log_path.open("a", encoding="utf-8")
            rollout_proc = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                preexec_fn=None if rollout_tty_input else os.setsid,
                stdout=rollout_log_handle,
                stderr=subprocess.STDOUT,
            )
            rollout_restart_count += 1
            last_rollout_motion_init_mode = motion_init_label
            last_rollout_motion_path = selected_motion_path
            last_rollout_reason = reason
            pending_restart_reason = None
            offset_xy[:] = 0.0
            offset_initialized = False
            received_first_state = False
            sub.last_state = None
            auto_reset_scheduled_at = None
            auto_reset_done = False
            reset_request_time_monotonic = None
            reset_pending_clock_rewind = False
            pre_reset_sim_time_ms = None
            last_seen_sim_time_ms = None
            state_md.content = "Waiting for simulator state after reset..."
            actor_md.content = ""
            logger.info(
                "Started rollout pid={} reason={} motion={} motion_init={}",
                rollout_proc.pid,
                reason,
                selected_motion_path,
                motion_init_label,
            )
            _refresh_rollout_md()
        finally:
            manual_rollout_btn.disabled = bool(cfg.launch_env_only)
            reset_rollout_btn.disabled = False
            _refresh_motion_init_controls()
            _refresh_motion_selection_controls()
            rollout_restart_lock.release()

    def _request_sim_reset(reason: str) -> None:
        nonlocal pending_restart_reason, offset_initialized, received_first_state, auto_reset_scheduled_at, auto_reset_done, reset_request_time_monotonic, reset_pending_clock_rewind, pre_reset_sim_time_ms
        if control_pub.enabled:
            control_pub.request_reset(reason)
            offset_xy[:] = 0.0
            state_md.content = f"Reset requested over sim-control ({reason})..."
            actor_md.content = ""
            sub.last_state = None
            pending_restart_reason = None
            if bool(recenter_cb.value):
                offset_initialized = False
            received_first_state = False
            auto_reset_scheduled_at = None
            auto_reset_done = True
            reset_request_time_monotonic = time.monotonic()
            reset_pending_clock_rewind = True
            pre_reset_sim_time_ms = last_seen_sim_time_ms
            logger.info("Requested simulator reset over sim-control ({})", reason)
        elif cfg.launch_rollout:
            pending_restart_reason = "gui_restart_fallback"
            state_md.content = "Control channel unavailable, falling back to full restart..."
        else:
            logger.warning("Reset rollout requested, but sim-control is unavailable")

    @show_object_cb.on_update
    def _(_evt) -> None:
        _set_object_mesh_visibility(
            show_object=bool(show_object_cb.value and object_root.visible),
            mesh_mode=str(object_mesh_mode_dropdown.value),
        )

    @show_robot_collision_cb.on_update
    def _(_evt) -> None:
        if hasattr(vr, "show_collision"):
            vr.show_collision = bool(show_robot_collision_cb.value)

    @object_mesh_mode_dropdown.on_update
    def _(_evt) -> None:
        _set_object_mesh_visibility(
            show_object=bool(show_object_cb.value and object_root.visible),
            mesh_mode=str(object_mesh_mode_dropdown.value),
        )

    @show_ref_cb.on_update
    def _(_evt) -> None:
        ref_root.visible = bool(show_ref_cb.value)

    @show_motion_overlay_cb.on_update
    def _(_evt) -> None:
        _set_motion_overlay_visibility()

    @show_motion_robot_cb.on_update
    def _(_evt) -> None:
        _set_motion_overlay_visibility()

    @show_motion_object_cb.on_update
    def _(_evt) -> None:
        _set_motion_overlay_visibility()

    @motion_clip_dropdown.on_update
    def _(_evt) -> None:
        motion_path = motion_choice_map.get(str(motion_clip_dropdown.value))
        if motion_path is not None and str(motion_path_text.value).strip() != str(motion_path):
            motion_path_text.value = str(motion_path)
        _refresh_rollout_md()

    @motion_path_text.on_update
    def _(_evt) -> None:
        motion_path = _resolve_motion_path_input(
            motion_path_text.value,
            base_dir=motion_choices_dir,
            choices=motion_choice_map,
        )
        motion_choice = _motion_choice_for_path(motion_path)
        if motion_choice is not None and motion_clip_dropdown.value != motion_choice:
            motion_clip_dropdown.value = motion_choice
        elif motion_choice is None:
            selected_motion_path = motion_choice_map.get(str(motion_clip_dropdown.value))
            if selected_motion_path is not None and str(motion_path_text.value).strip() != str(selected_motion_path):
                motion_path_text.value = str(selected_motion_path)
        _refresh_rollout_md()

    @reset_offset_btn.on_click
    def _(_evt) -> None:
        nonlocal offset_initialized
        offset_xy[:] = 0.0
        offset_initialized = False

    @reset_rollout_btn.on_click
    def _(_evt) -> None:
        selected_motion_init_mode, _selected_motion_init_env = _selected_motion_init()
        selected_motion_path = _selected_motion_path(validate=True)
        if selected_motion_path is None:
            raw_motion_value = str(motion_path_text.value).strip() or "empty"
            rollout_md.content = f"Invalid motion path `{raw_motion_value}`; reset was not sent."
            logger.warning("Reset rollout requested with invalid motion path: {}", raw_motion_value)
            return
        running_rollout = rollout_proc is not None and rollout_proc.poll() is None
        if cfg.launch_rollout and not running_rollout:
            _restart_rollout("gui_reset_start")
            return
        motion_changed = last_rollout_motion_path is not None and selected_motion_path != last_rollout_motion_path
        init_changed = (
            last_rollout_motion_init_mode is not None
            and selected_motion_init_mode != last_rollout_motion_init_mode
        )
        if (
            running_rollout
            and (motion_changed or init_changed)
        ):
            logger.info(
                "Reset rollout requires restart because selected config changed: motion {} -> {}, init {} -> {}",
                _motion_path_display(last_rollout_motion_path),
                _motion_path_display(selected_motion_path),
                last_rollout_motion_init_mode,
                selected_motion_init_mode,
            )
            _restart_rollout(f"gui_reset_{selected_motion_path.stem}_{selected_motion_init_mode}")
            return
        _request_sim_reset("gui_reset")

    @manual_rollout_btn.on_click
    def _(_evt) -> None:
        if bool(cfg.launch_env_only):
            rollout_md.content = "Environment-only launch is active; start policy with `mj_policy.sh`."
            logger.info("Manual policy rollout ignored because launch_env_only is active")
            return
        if rollout_proc is not None and rollout_proc.poll() is None and last_rollout_skip_policy is False:
            logger.info("Manual policy rollout requested, but policy rollout is already running pid={}", rollout_proc.pid)
            return
        _restart_rollout("manual_policy_rollout", skip_policy=False)

    @manual_motion_init_mode_cb.on_update
    def _(_evt) -> None:
        if not bool(manual_motion_init_mode_cb.value):
            reset_to_default_pose_cb.value = bool(auto_default_pose_start)
        _refresh_motion_init_controls()
        _refresh_rollout_md()

    @reset_to_default_pose_cb.on_update
    def _(_evt) -> None:
        _refresh_rollout_md()

    @manual_root_enabled_cb.on_update
    def _(_evt) -> None:
        _publish_manual_root_command()

    @manual_root_mode_dropdown.on_update
    def _(_evt) -> None:
        _publish_manual_root_command()

    @manual_root_dx.on_update
    def _(_evt) -> None:
        _publish_manual_root_command()

    @manual_root_dy.on_update
    def _(_evt) -> None:
        _publish_manual_root_command()

    @manual_root_dyaw.on_update
    def _(_evt) -> None:
        _publish_manual_root_command()

    @manual_root_zero_btn.on_click
    def _(_evt) -> None:
        manual_root_dx.value = 0.0
        manual_root_dy.value = 0.0
        manual_root_dyaw.value = 0.0
        _publish_manual_root_command()

    announce_url = os.environ.get("HOLOSOMA_VISER_ANNOUNCE_URL", "").strip()
    if announce_url:
        logger.info("Open MuJoCo command+scene web at {}", announce_url)
    if not suppress_direct_url:
        logger.info("Open viser at http://localhost:{}", port)
    logger.info("Reading split MuJoCo sim-state from tcp://localhost:{}", cfg.state_port)
    logger.info("Reading split MuJoCo perception_obs from tcp://localhost:{}", cfg.perception_obs_port)
    _refresh_rollout_md()
    _publish_manual_root_command()

    try:
        while True:
            if pending_restart_reason is not None:
                _restart_rollout(pending_restart_reason)
            if auto_reset_scheduled_at is not None and time.monotonic() >= auto_reset_scheduled_at:
                _request_sim_reset("auto_test_reset")
            _refresh_rollout_md()
            _refresh_depth_view()
            _publish_manual_root_command()

            state = sub.get_state()
            if state is None:
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            robot_root_state = state.get("robot_root_state")
            robot_dof_pos = state.get("robot_dof_pos")
            if robot_root_state is None or robot_dof_pos is None:
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            sim_time_ms = int(state.get("sim_time_ms", 0))
            if (
                reset_pending_clock_rewind
                and pre_reset_sim_time_ms is not None
                and pre_reset_sim_time_ms > 0
                and sim_time_ms >= pre_reset_sim_time_ms
            ):
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            root_state = np.asarray(robot_root_state, dtype=np.float32).reshape(-1)
            dof_pos = np.asarray(robot_dof_pos, dtype=np.float32).reshape(-1)
            if root_state.shape[0] < 7 or dof_pos.shape[0] < len(robot_config.dof_names):
                time.sleep(1.0 / max(cfg.rate_hz, 1.0))
                continue

            if not received_first_state:
                if reset_request_time_monotonic is None:
                    logger.info(
                        "Received first sim-state: sim_time_ms={}, ref_body={}",
                        int(state.get("sim_time_ms", 0)),
                        state.get("robot_ref_body_name", "n/a"),
                    )
                else:
                    reset_latency_ms = (time.monotonic() - reset_request_time_monotonic) * 1000.0
                    logger.info(
                        "Received first sim-state after reset: sim_time_ms={}, ref_body={}, latency_ms={:.1f}",
                        sim_time_ms,
                        state.get("robot_ref_body_name", "n/a"),
                        reset_latency_ms,
                    )
                    reset_request_time_monotonic = None
                    reset_pending_clock_rewind = False
                    pre_reset_sim_time_ms = None
                received_first_state = True

            last_seen_sim_time_ms = sim_time_ms

            if bool(recenter_cb.value) and not offset_initialized:
                offset_xy[:] = root_state[:2]
                offset_initialized = True

            joint_viser = dof_pos[: len(robot_config.dof_names)]
            name_to_robot_idx = {name: idx for idx, name in enumerate(robot_config.dof_names)}
            joint_viser = np.asarray([joint_viser[name_to_robot_idx[name]] for name in viser_joint_names], dtype=np.float32)

            root_pos = root_state[:3].copy()
            if bool(recenter_cb.value):
                root_pos[:2] -= offset_xy
            root_quat_wxyz = _xyzw_to_wxyz(root_state[3:7])

            ref_state = state.get("robot_ref_state")
            ref_state_np = None
            if ref_state is not None:
                ref_state_np = np.asarray(ref_state, dtype=np.float32).reshape(-1)

            actor_key, object_state = _select_actor_state(state, cfg.object_actor_name)
            snapshot_path_raw = state.get("mujoco_object_geom_snapshot_path")
            snapshot_path = snapshot_path_default
            if isinstance(snapshot_path_raw, str) and snapshot_path_raw.strip():
                snapshot_path = Path(snapshot_path_raw).expanduser().resolve()
            if loaded_object_snapshot_path is None and snapshot_path.is_file():
                _load_object_geom_handles(snapshot_path)
            _update_motion_overlay(sim_time_ms, root_state, object_state)

            with server.atomic():
                robot_root.position = tuple(root_pos.tolist())
                robot_root.wxyz = tuple(root_quat_wxyz.tolist())
                vr.update_cfg(joint_viser)

                if ref_state_np is not None and ref_state_np.shape[0] >= 7:
                    ref_pos = ref_state_np[:3].copy()
                    if bool(recenter_cb.value):
                        ref_pos[:2] -= offset_xy
                    ref_root.position = tuple(ref_pos.tolist())
                    ref_root.wxyz = tuple(_xyzw_to_wxyz(ref_state_np[3:7]).tolist())
                    ref_root.visible = bool(show_ref_cb.value)
                else:
                    ref_root.visible = False

                if object_state is not None:
                    object_pos = object_state[:3].copy()
                    if bool(recenter_cb.value):
                        object_pos[:2] -= offset_xy
                    object_root.position = tuple(object_pos.tolist())
                    object_root.wxyz = tuple(_xyzw_to_wxyz(object_state[3:7]).tolist())
                    object_root.visible = True
                    _set_object_mesh_visibility(
                        show_object=bool(show_object_cb.value),
                        mesh_mode=str(object_mesh_mode_dropdown.value),
                    )
                else:
                    object_root.visible = False
                    _set_object_mesh_visibility(show_object=False, mesh_mode=str(object_mesh_mode_dropdown.value))

            ref_body_name = state.get("robot_ref_body_name", "n/a")
            object_robot_contacts = int(state.get("object_robot_contact_count", 0))
            object_scene_contacts = int(state.get("object_scene_contact_count", 0))
            state_md.content = (
                f"sim_time_ms: `{sim_time_ms}`\n\n"
                f"ref_body: `{ref_body_name}`\n\n"
                f"robot_root_xyz: `{np.array2string(root_state[:3], precision=4)}`\n\n"
                f"object_robot_contacts: `{object_robot_contacts}`\n\n"
                f"object_scene_contacts: `{object_scene_contacts}`"
            )
            actor_label = actor_key if actor_key is not None else "none"
            snapshot_label = str(loaded_object_snapshot_path) if loaded_object_snapshot_path is not None else "pending"
            requested_mesh_mode = _resolve_object_mesh_mode(str(object_mesh_mode_dropdown.value))
            resolved_mesh_mode = requested_mesh_mode
            if requested_mesh_mode == "visual" and object_visual_uses_collision_fallback:
                resolved_mesh_mode = "visual -> collision fallback"
            available_mesh_modes = []
            if object_visual_handles:
                available_mesh_modes.append("visual")
            if object_collision_handles:
                available_mesh_modes.append("collision")
            object_geom_mode = (
                ", ".join(available_mesh_modes)
                if available_mesh_modes
                else "pending"
            )
            actor_md.content = (
                f"object_actor: `{actor_label}`\n\n"
                f"object_geom_source: `MuJoCo geom snapshot`\n\n"
                f"object_mesh_view: `{resolved_mesh_mode}`\n\n"
                f"object_geom_mode: `{object_geom_mode}`\n\n"
                f"snapshot_path: `{snapshot_label}`"
            )

            time.sleep(1.0 / max(cfg.rate_hz, 1.0))
    except KeyboardInterrupt:
        logger.info("Stopping viser MuJoCo sim-state viewer")
    finally:
        _stop_rollout()
        if manual_root_pub is not None:
            manual_root_pub.close()
        control_pub.close()
        perception_shm_sub.close()
        perception_sub.close()
        sub.close()
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


def main() -> None:
    cfg = tyro.cli(MujocoSimStateViewerConfig)
    view_sim_state(cfg)


if __name__ == "__main__":
    main()
