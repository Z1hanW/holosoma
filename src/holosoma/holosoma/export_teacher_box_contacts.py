from __future__ import annotations

import csv
import json
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from loguru import logger

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.agents.base_algo.base_algo import BaseAlgo  # noqa: E402
from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.managers.command.terms.wbt import MotionCommand  # noqa: E402
from holosoma.observation import apply_observation_overrides  # noqa: E402
from holosoma.perception import apply_perception_overrides  # noqa: E402
from holosoma.utils.eval_utils import (  # noqa: E402
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp  # noqa: E402
from holosoma.utils.helpers import get_class  # noqa: E402
from holosoma.utils.rotations import quat_error_magnitude, quat_inverse, quat_mul  # noqa: E402
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


_LEFT_WRIST_BODY_NAMES = ["left_wrist_yaw_link"]
_RIGHT_WRIST_BODY_NAMES = ["right_wrist_yaw_link"]
_TORSO_BODY_NAMES = ["torso_link"]

_LINK_REGION_BODY_NAMES = {
    "left_wrist": "left_wrist_yaw_link",
    "right_wrist": "right_wrist_yaw_link",
    "left_elbow": "left_elbow_link",
    "right_elbow": "right_elbow_link",
    "left_wrist_roll": "left_wrist_roll_link",
    "right_wrist_roll": "right_wrist_roll_link",
    "left_wrist_pitch": "left_wrist_pitch_link",
    "right_wrist_pitch": "right_wrist_pitch_link",
    "torso": "torso_link",
}
_REGION_EXPORT_LABELS = {
    "left_wrist": "left_wrist",
    "right_wrist": "right_wrist",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
    "left_wrist_roll": "left_wrist_roll",
    "right_wrist_roll": "right_wrist_roll",
    "left_wrist_pitch": "left_wrist_pitch",
    "right_wrist_pitch": "right_wrist_pitch",
    "torso": "torso",
}
_REGION_SPECS: dict[str, dict[str, Any]] = {
    "left_wrist": {
        "label": "left_wrist",
        "body_names": _LEFT_WRIST_BODY_NAMES,
    },
    "right_wrist": {
        "label": "right_wrist",
        "body_names": _RIGHT_WRIST_BODY_NAMES,
    },
    "torso": {
        "label": "torso",
        "body_names": _TORSO_BODY_NAMES,
    },
}
_REGION_SPECS.update(
    {
        region_name: {
            "label": _REGION_EXPORT_LABELS[region_name],
            "body_names": [body_name],
        }
        for region_name, body_name in _LINK_REGION_BODY_NAMES.items()
        if region_name not in _REGION_SPECS
    }
)
_EXPORT_REGION_LABELS = tuple(str(spec["label"]) for spec in _REGION_SPECS.values())
_LEGACY_ARM_AGGREGATE_LABEL = "arm"
_LEGACY_ARM_AGGREGATE_REGION_LABELS = (
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
)

_CONTACT_SENSOR_BODY_NAMES = tuple(
    dict.fromkeys(
        _LEFT_WRIST_BODY_NAMES
        + _RIGHT_WRIST_BODY_NAMES
        + [body_name for region_name, body_name in _LINK_REGION_BODY_NAMES.items() if region_name not in {"left_wrist", "right_wrist"}]
        + _TORSO_BODY_NAMES
    )
)

_BOX_COLOR = np.asarray([180, 180, 190, 72], dtype=np.uint8)
_MESH_COLOR = np.asarray([175, 185, 200, 210], dtype=np.uint8)
_REGION_OVERLAY_STYLE: dict[str, dict[str, Any]] = {
    "left_wrist": {
        "rgba": np.asarray([255, 140, 0, 255], dtype=np.uint8),
        "mpl_color": "#FF8C00",
        "radius_scale": 1.0,
        "scatter_size": 30.0,
        "label": "left wrist",
    },
    "right_wrist": {
        "rgba": np.asarray([255, 220, 0, 255], dtype=np.uint8),
        "mpl_color": "#FFDC00",
        "radius_scale": 1.0,
        "scatter_size": 30.0,
        "label": "right wrist",
    },
    "left_elbow": {
        "rgba": np.asarray([255, 105, 180, 255], dtype=np.uint8),
        "mpl_color": "#FF69B4",
        "radius_scale": 1.15,
        "scatter_size": 36.0,
        "label": "left elbow",
    },
    "right_elbow": {
        "rgba": np.asarray([220, 20, 60, 255], dtype=np.uint8),
        "mpl_color": "#DC143C",
        "radius_scale": 1.15,
        "scatter_size": 36.0,
        "label": "right elbow",
    },
    "left_wrist_roll": {
        "rgba": np.asarray([0, 191, 255, 255], dtype=np.uint8),
        "mpl_color": "#00BFFF",
        "radius_scale": 1.0,
        "scatter_size": 30.0,
        "label": "left wrist roll",
    },
    "right_wrist_roll": {
        "rgba": np.asarray([30, 144, 255, 255], dtype=np.uint8),
        "mpl_color": "#1E90FF",
        "radius_scale": 1.0,
        "scatter_size": 30.0,
        "label": "right wrist roll",
    },
    "left_wrist_pitch": {
        "rgba": np.asarray([50, 205, 50, 255], dtype=np.uint8),
        "mpl_color": "#32CD32",
        "radius_scale": 1.0,
        "scatter_size": 30.0,
        "label": "left wrist pitch",
    },
    "right_wrist_pitch": {
        "rgba": np.asarray([34, 139, 34, 255], dtype=np.uint8),
        "mpl_color": "#228B22",
        "radius_scale": 1.0,
        "scatter_size": 30.0,
        "label": "right wrist pitch",
    },
    "torso": {
        "rgba": np.asarray([128, 64, 192, 255], dtype=np.uint8),
        "mpl_color": "#8040C0",
        "radius_scale": 1.6,
        "scatter_size": 82.0,
        "label": "torso",
    },
}
_REGION_DISPLAY_PRIORITY: dict[str, int] = {
    "torso": 3,
    "left_wrist": 2,
    "right_wrist": 2,
    "left_elbow": 1,
    "right_elbow": 1,
    "left_wrist_roll": 1,
    "right_wrist_roll": 1,
    "left_wrist_pitch": 1,
    "right_wrist_pitch": 1,
}


@dataclass(frozen=True)
class ExportConfig:
    output_dir: str = "outputs/teacher_box_contacts"
    min_contact_frames: int = 10
    contact_force_threshold: float = 1.0
    contact_voxel_size: float = 0.01
    success_position_threshold: float = 0.10
    max_rollout_steps: int | None = None
    project_contact_to_mesh: bool = True
    save_glb: bool = True
    save_preview_png: bool = True
    save_face_heatmap_png: bool = True


@dataclass(frozen=True)
class ClipSummary:
    clip_id: str
    clip_index: int
    object_name: str
    object_urdf_path: str
    batch_index: int
    env_id: int
    num_steps: int
    motion_end_reached: bool
    terminated: bool
    timeout: bool
    success: bool
    stable_contact_success: bool
    final_position_success: bool
    status: str
    final_object_position_error_m: float
    final_object_rotation_error_rad: float
    primitive_extent_x: float
    primitive_extent_y: float
    primitive_extent_z: float
    retained_contact_point_count: int
    left_wrist_contact_frames: int
    right_wrist_contact_frames: int
    left_elbow_contact_frames: int
    right_elbow_contact_frames: int
    left_wrist_roll_contact_frames: int
    right_wrist_roll_contact_frames: int
    left_wrist_pitch_contact_frames: int
    right_wrist_pitch_contact_frames: int
    torso_contact_frames: int


@dataclass
class ClipAccumulator:
    clip_id: str
    clip_index: int
    object_name: str
    object_urdf_path: str
    batch_index: int
    env_id: int
    extents_xyz: np.ndarray
    object_surface_mesh: Any | None
    contact_surface_projection: str
    clip_dir: Path
    region_point_counts: dict[str, dict[tuple[int, int, int], int]]
    region_force_sums: dict[str, float]
    region_force_max: dict[str, float]
    region_contact_frames: dict[str, int]
    region_contact_interval_start: dict[str, int]
    region_contact_interval_end: dict[str, int]
    body_force_sums: dict[str, float]
    body_force_max: dict[str, float]
    body_contact_frames: dict[str, int]
    tracked_body_names: list[str]
    full_body_names: list[str]
    joint_names: list[str]
    ref_body_name: str
    motion_fps: float
    rollout_reference: dict[str, np.ndarray]
    rollout_motion: dict[str, np.ndarray]


def _default_output_dir() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("outputs") / "teacher_box_contacts" / timestamp


def _with_contact_sensor_bodies(config: ExperimentConfig) -> ExperimentConfig:
    sim_cfg = replace(
        config.simulator.config,
        object_filtered_contact_sensor_body_names=list(_CONTACT_SENSOR_BODY_NAMES),
    )
    return replace(config, simulator=replace(config.simulator, config=sim_cfg))


def _normalize_output_dir(raw: str) -> Path:
    path = Path(raw)
    default_raw = ExportConfig.__dataclass_fields__["output_dir"].default  # type: ignore[index]
    if str(raw).strip() == str(default_raw):
        return _default_output_dir().resolve()
    return path.expanduser().resolve()


def _ensure_motion_command(env: Any) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    if motion_command is None or not isinstance(motion_command, MotionCommand):
        raise RuntimeError("motion_command state is unavailable or has unexpected type.")
    return motion_command


def _sanitize_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in name)


def _xyzw_to_wxyz_np(quat_xyzw: np.ndarray) -> np.ndarray:
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float32)
    return np.concatenate([quat_xyzw[..., 3:4], quat_xyzw[..., :3]], axis=-1)


def _normalize_path_key(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).expanduser().resolve())
    except Exception:
        return str(path)


def _object_assignment_key(*, urdf_path: str, object_name: str) -> str:
    normalized = _normalize_path_key(urdf_path)
    if normalized:
        return f"urdf::{normalized}"
    name = str(object_name or "").strip().lower()
    if name:
        return f"name::{name}"
    return "default"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_text_list(path: Path, values: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(values) + ("\n" if values else ""), encoding="utf-8")


def _project_point_to_box_surface(point_xyz: np.ndarray, extents_xyz: np.ndarray) -> np.ndarray:
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    half = 0.5 * np.asarray(extents_xyz, dtype=np.float64).reshape(3)

    outside = np.abs(point) > half
    if np.any(outside):
        return np.clip(point, -half, half).astype(np.float32)

    distances = half - np.abs(point)
    axis = int(np.argmin(distances))
    projected = point.copy()
    sign = 1.0 if projected[axis] >= 0.0 else -1.0
    if abs(projected[axis]) < 1.0e-8:
        sign = 1.0
    projected[axis] = sign * half[axis]
    return projected.astype(np.float32)


def _project_point_to_mesh_surface(point_xyz: np.ndarray, object_mesh: Any) -> np.ndarray:
    import trimesh  # type: ignore[import-not-found]

    point = np.asarray(point_xyz, dtype=np.float64).reshape(1, 3)
    closest, _, _ = trimesh.proximity.closest_point(object_mesh, point)
    return np.asarray(closest[0], dtype=np.float32)


def _project_point_to_object_surface(
    point_xyz: np.ndarray,
    accumulator: ClipAccumulator,
) -> np.ndarray:
    object_mesh = accumulator.object_surface_mesh
    if object_mesh is not None:
        try:
            accumulator.contact_surface_projection = "mesh"
            return _project_point_to_mesh_surface(point_xyz, object_mesh)
        except Exception as exc:
            logger.warning(
                "Falling back to primitive-box contact projection for clip '{}' because mesh projection failed: {}",
                accumulator.clip_id,
                exc,
            )
            accumulator.object_surface_mesh = None
            accumulator.contact_surface_projection = "primitive_box"
    return _project_point_to_box_surface(point_xyz, accumulator.extents_xyz)


def _quantize_point(point_xyz: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    denom = max(float(voxel_size), 1.0e-6)
    return tuple(int(v) for v in np.round(point / denom))


def _dequantize_point(key: tuple[int, int, int], voxel_size: float) -> np.ndarray:
    return np.asarray(key, dtype=np.float32) * np.float32(voxel_size)


def _surface_face_name(point_xyz: np.ndarray, extents_xyz: np.ndarray) -> str:
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    half = 0.5 * np.asarray(extents_xyz, dtype=np.float64).reshape(3)
    deltas = np.abs(np.abs(point) - half)
    axis = int(np.argmin(deltas))
    sign = "+" if point[axis] >= 0.0 else "-"
    return f"{sign}{'xyz'[axis]}"


def _policy_step(algo: BaseAlgo, obs_dict: dict[str, torch.Tensor], policy_fn) -> torch.Tensor:
    actor_obs_raw = torch.cat([obs_dict[key] for key in algo.actor_obs_keys], dim=1)
    policy_state: dict[str, torch.Tensor] = {"actor_obs": actor_obs_raw}
    actor_perception_key = getattr(algo, "actor_perception_key", "") or ""
    if actor_perception_key and actor_perception_key in obs_dict:
        policy_state[actor_perception_key] = obs_dict[actor_perception_key]
    return policy_fn(policy_state)


def _sync_simulator_after_state_write(env: Any) -> None:
    sim = getattr(env.simulator, "sim", None)
    forward = getattr(sim, "forward", None)
    if callable(forward):
        forward()
    env._refresh_sim_tensors()


def _reset_envs_without_advancing(env: Any) -> dict[str, torch.Tensor]:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_envs_idx(env_ids)
    env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)
    env.simulator.set_dof_state_tensor_robots(env_ids, env.simulator.dof_state)
    _sync_simulator_after_state_write(env)
    env._pre_compute_observations_callback()
    env._compute_observations()
    env._post_compute_observations_callback()
    return env.obs_buf_dict


def _write_motion_command_targets_to_sim(
    motion_command: MotionCommand,
    *,
    active_env_ids: list[int],
) -> dict[str, torch.Tensor]:
    env = motion_command._env
    env_ids_tensor = torch.tensor(active_env_ids, device=motion_command.device, dtype=torch.long)
    if env_ids_tensor.numel() == 0:
        return env.obs_buf_dict

    target_root_pos_w = motion_command.root_pos_w.index_select(0, env_ids_tensor)
    target_root_quat_w = motion_command.root_quat_w.index_select(0, env_ids_tensor)
    target_body_lin_vel_w = motion_command.body_lin_vel_w.index_select(0, env_ids_tensor)
    target_body_ang_vel_w = motion_command.body_ang_vel_w.index_select(0, env_ids_tensor)
    root_states = env.simulator.robot_root_states[env_ids_tensor].clone()
    root_states[:, :3] = target_root_pos_w
    root_states[:, 3:7] = target_root_quat_w
    root_states[:, 7:10] = target_body_lin_vel_w[:, 0, :]
    root_states[:, 10:13] = target_body_ang_vel_w[:, 0, :]

    env.simulator.dof_pos[env_ids_tensor] = motion_command.joint_pos.index_select(0, env_ids_tensor)
    env.simulator.dof_vel[env_ids_tensor] = motion_command.joint_vel.index_select(0, env_ids_tensor)
    env.simulator.set_actor_root_state_tensor_robots(env_ids_tensor, root_states)
    env.simulator.set_dof_state_tensor_robots(env_ids_tensor, env.simulator.dof_state)

    if motion_command.motion.has_object:
        object_pos_w = motion_command.object_pos_w.index_select(0, env_ids_tensor)
        object_quat_w = motion_command.object_quat_w.index_select(0, env_ids_tensor)
        object_lin_vel_w = motion_command.object_lin_vel_w.index_select(0, env_ids_tensor)
        object_states = torch.cat(
            [object_pos_w, object_quat_w, object_lin_vel_w, torch.zeros_like(object_lin_vel_w)],
            dim=-1,
        )
        if hasattr(motion_command, "_set_simulator_object_states"):
            motion_command._set_simulator_object_states(env_ids_tensor, object_states)
        else:
            env.simulator.set_actor_states(["object"], env_ids_tensor, object_states)

    for _ in range(4):
        _sync_simulator_after_state_write(env)
        measured_root_states = env.simulator.robot_root_states[env_ids_tensor].clone()
        pos_error = target_root_pos_w - measured_root_states[:, :3]
        quat_error = quat_mul(
            target_root_quat_w,
            quat_inverse(measured_root_states[:, 3:7], w_last=True),
            w_last=True,
        )
        max_pos_error = float(torch.linalg.norm(pos_error, dim=-1).max().detach().cpu().item())
        max_quat_error = float(quat_error_magnitude(target_root_quat_w, measured_root_states[:, 3:7]).max().detach().cpu().item())
        if max_pos_error <= 1.0e-5 and max_quat_error <= 1.0e-5:
            break
        root_states[:, :3] = root_states[:, :3] + pos_error
        root_states[:, 3:7] = quat_mul(quat_error, root_states[:, 3:7], w_last=True)
        root_states[:, 3:7] = root_states[:, 3:7] / torch.linalg.norm(root_states[:, 3:7], dim=-1, keepdim=True).clamp_min(1.0e-8)
        env.simulator.set_actor_root_state_tensor_robots(env_ids_tensor, root_states)
        env.simulator.set_dof_state_tensor_robots(env_ids_tensor, env.simulator.dof_state)

    _sync_simulator_after_state_write(env)
    env._pre_compute_observations_callback()
    env._compute_observations()
    env._post_compute_observations_callback()
    return env.obs_buf_dict


def _retained_points_from_counts(
    point_counts: dict[tuple[int, int, int], int],
    *,
    voxel_size: float,
    min_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    retained = [(key, count) for key, count in point_counts.items() if count >= min_frames]
    if not retained:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    retained.sort(key=lambda item: (-item[1], item[0]))
    points = np.stack([_dequantize_point(key, voxel_size) for key, _ in retained], axis=0).astype(np.float32)
    counts = np.asarray([count for _, count in retained], dtype=np.int32)
    return points, counts


def _contact_interval_from_mask(contact_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(contact_mask, dtype=np.bool_).reshape(-1)
    active_steps = np.flatnonzero(mask)
    if active_steps.size == 0:
        return np.asarray([-1, -1], dtype=np.int32)
    return np.asarray([int(active_steps[0]), int(active_steps[-1]) + 1], dtype=np.int32)


def _build_display_points_from_region_counts(
    overall_point_counts: dict[tuple[int, int, int], int],
    region_point_counts: dict[str, dict[tuple[int, int, int], int]],
    *,
    voxel_size: float,
    min_frames: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    retained = [(key, count) for key, count in overall_point_counts.items() if count >= min_frames]
    if not retained:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0,), dtype=np.int32), []
    retained.sort(key=lambda item: (-item[1], item[0]))

    points: list[np.ndarray] = []
    counts: list[int] = []
    labels: list[str] = []
    for key, total_count in retained:
        region_scores = {
            label: int(point_counts.get(key, 0))
            for label, point_counts in region_point_counts.items()
            if int(point_counts.get(key, 0)) > 0
        }
        if region_scores:
            label = max(
                region_scores,
                key=lambda name: (region_scores[name], _REGION_DISPLAY_PRIORITY.get(name, 0), name),
            )
        else:
            label = _EXPORT_REGION_LABELS[0]
        points.append(_dequantize_point(key, voxel_size))
        counts.append(int(total_count))
        labels.append(label)
    return np.stack(points, axis=0).astype(np.float32), np.asarray(counts, dtype=np.int32), labels


def _quat_to_rotmat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _parse_vec3(raw: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if raw is None:
        return np.asarray(default, dtype=np.float32)
    parts = [part for part in str(raw).replace(",", " ").split() if part]
    if len(parts) != 3:
        return np.asarray(default, dtype=np.float32)
    try:
        return np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32)
    except Exception:
        return np.asarray(default, dtype=np.float32)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = [float(v) for v in rpy]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float32)
    rot_y = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float32)
    rot_z = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rot_z @ rot_y @ rot_x


def _origin_transform(origin_el: ET.Element | None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float32)
    if origin_el is None:
        return transform
    transform[:3, :3] = _rpy_matrix(_parse_vec3(origin_el.get("rpy"), (0.0, 0.0, 0.0)))
    transform[:3, 3] = _parse_vec3(origin_el.get("xyz"), (0.0, 0.0, 0.0))
    return transform


def _load_object_overlay_mesh(
    *,
    clip_id: str,
    object_name: str,
    object_urdf_path: str,
):
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception:
        return None

    resolved_urdf = Path(object_urdf_path).expanduser().resolve()
    if not resolved_urdf.exists():
        return None

    try:
        root = ET.parse(resolved_urdf).getroot()
    except Exception:
        return None

    link = root.find("link")
    if link is None:
        return None

    def _build_from_geom_tag(geom_tag: str):
        meshes: list["trimesh.Trimesh"] = []
        for geom_parent in link.findall(geom_tag):
            geometry_el = geom_parent.find("geometry")
            if geometry_el is None:
                continue
            mesh_obj = None
            box_el = geometry_el.find("box")
            if box_el is not None:
                size = _parse_vec3(box_el.get("size"), (1.0, 1.0, 1.0))
                mesh_obj = trimesh.creation.box(extents=size.astype(np.float64))
            else:
                mesh_el = geometry_el.find("mesh")
                if mesh_el is None:
                    continue
                filename = str(mesh_el.get("filename", "")).strip()
                if not filename:
                    continue
                mesh_path = Path(filename)
                if not mesh_path.is_absolute():
                    mesh_path = (resolved_urdf.parent / mesh_path).resolve()
                if not mesh_path.exists():
                    continue
                try:
                    loaded = trimesh.load(str(mesh_path), process=False)
                except Exception:
                    continue
                if isinstance(loaded, trimesh.Scene):
                    dumped = loaded.dump(concatenate=True)
                    mesh_obj = dumped if isinstance(dumped, trimesh.Trimesh) else None
                elif isinstance(loaded, trimesh.Trimesh):
                    mesh_obj = loaded
                if mesh_obj is None:
                    continue
                scale = _parse_vec3(mesh_el.get("scale"), (1.0, 1.0, 1.0)).astype(np.float64)
                mesh_obj = mesh_obj.copy()
                mesh_obj.apply_scale(scale)

            if mesh_obj is None:
                continue
            mesh_obj = mesh_obj.copy()
            mesh_obj.apply_transform(_origin_transform(geom_parent.find("origin")).astype(np.float64))
            meshes.append(mesh_obj)

        if not meshes:
            return None
        return trimesh.util.concatenate(meshes)

    mesh = _build_from_geom_tag("visual")
    if mesh is None:
        mesh = _build_from_geom_tag("collision")
    if mesh is None:
        return None

    return mesh


def _save_overlay_assets(
    clip_dir: Path,
    *,
    clip_id: str,
    object_name: str,
    object_urdf_path: str,
    extents_xyz: np.ndarray,
    retained_points_xyz: np.ndarray,
    retained_counts: np.ndarray,
    display_points_xyz: np.ndarray,
    display_point_labels: list[str],
    region_points_by_label: dict[str, np.ndarray] | None = None,
    save_glb: bool,
    save_preview_png: bool,
    save_face_heatmap_png: bool,
) -> None:
    if not (save_glb or save_preview_png or save_face_heatmap_png):
        return

    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("Skipping 3D overlay export because trimesh is unavailable: {}", exc)
        trimesh = None

    object_mesh = None
    if trimesh is not None:
        object_mesh = _load_object_overlay_mesh(
            clip_id=clip_id,
            object_name=object_name,
            object_urdf_path=object_urdf_path,
        )

    if save_glb and trimesh is not None:
        scene = trimesh.Scene()
        if object_mesh is not None:
            object_mesh = object_mesh.copy()
            object_mesh.visual.vertex_colors = np.tile(_MESH_COLOR, (len(object_mesh.vertices), 1))
            scene.add_geometry(object_mesh, node_name="object_mesh")

        box_mesh = trimesh.creation.box(extents=np.asarray(extents_xyz, dtype=np.float64))
        box_mesh.visual.vertex_colors = _BOX_COLOR
        scene.add_geometry(box_mesh, node_name="primitive_box")

        visual_region_points: dict[str, np.ndarray] = {}
        if region_points_by_label is not None:
            for region_name in _EXPORT_REGION_LABELS:
                points_xyz = np.asarray(
                    region_points_by_label.get(region_name, np.zeros((0, 3), dtype=np.float32)),
                    dtype=np.float32,
                ).reshape(-1, 3)
                if points_xyz.size > 0:
                    visual_region_points[region_name] = points_xyz
        else:
            for region_name in _EXPORT_REGION_LABELS:
                region_points_xyz = display_points_xyz[
                    [idx for idx, label in enumerate(display_point_labels) if label == region_name]
                ]
                if region_points_xyz.size > 0:
                    visual_region_points[region_name] = np.asarray(region_points_xyz, dtype=np.float32).reshape(-1, 3)

        base_radius = max(float(np.max(extents_xyz)) * 0.02, 0.003)
        for region_name in _EXPORT_REGION_LABELS:
            region_points_xyz = visual_region_points.get(region_name)
            if region_points_xyz is None or region_points_xyz.size == 0:
                continue
            style = _REGION_OVERLAY_STYLE.get(region_name, _REGION_OVERLAY_STYLE["left_wrist"])
            radius = base_radius * float(style["radius_scale"])
            color = np.asarray(style["rgba"], dtype=np.uint8)
            for point_xyz in region_points_xyz:
                sphere = trimesh.creation.icosphere(subdivisions=2, radius=radius)
                sphere.apply_translation(point_xyz.astype(np.float64))
                sphere.visual.vertex_colors = color
                point_name = "_".join(f"{coord:.3f}" for coord in point_xyz.tolist())
                scene.add_geometry(sphere, node_name=f"{region_name}_contact_{point_name}")

        scene.export(clip_dir / "contact_overlay.glb")

    if save_preview_png:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except Exception as exc:
            logger.warning("Skipping preview PNG export because matplotlib is unavailable: {}", exc)
        else:
            fig = plt.figure(figsize=(12, 10))
            ax_3d = fig.add_subplot(221, projection="3d")
            ax_xy = fig.add_subplot(222)
            ax_xz = fig.add_subplot(223)
            ax_yz = fig.add_subplot(224)
            visual_region_points: dict[str, np.ndarray] = {}
            if region_points_by_label is not None:
                for region_name in _EXPORT_REGION_LABELS:
                    points_xyz = np.asarray(
                        region_points_by_label.get(region_name, np.zeros((0, 3), dtype=np.float32)),
                        dtype=np.float32,
                    ).reshape(-1, 3)
                    if points_xyz.size > 0:
                        visual_region_points[region_name] = points_xyz
            else:
                for region_name in _EXPORT_REGION_LABELS:
                    region_points_xyz = display_points_xyz[
                        [idx for idx, label in enumerate(display_point_labels) if label == region_name]
                    ]
                    if region_points_xyz.size > 0:
                        visual_region_points[region_name] = np.asarray(region_points_xyz, dtype=np.float32).reshape(-1, 3)

            box_vertices = np.asarray(
                [
                    [-0.5, -0.5, -0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, 0.5, -0.5],
                    [-0.5, 0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, 0.5, 0.5],
                    [-0.5, 0.5, 0.5],
                ],
                dtype=np.float32,
            )
            box_vertices = box_vertices * extents_xyz.reshape(1, 3)
            box_faces = [
                [box_vertices[idx] for idx in face]
                for face in (
                    [0, 1, 2, 3],
                    [4, 5, 6, 7],
                    [0, 1, 5, 4],
                    [2, 3, 7, 6],
                    [1, 2, 6, 5],
                    [0, 3, 7, 4],
                )
            ]
            collection = Poly3DCollection(
                box_faces,
                alpha=0.12,
                facecolor=np.array([0.6, 0.6, 0.65]),
                edgecolor="gray",
            )
            ax_3d.add_collection3d(collection)

            mesh_vertices = None
            if object_mesh is not None:
                mesh_vertices = np.asarray(object_mesh.vertices, dtype=np.float64)
                mesh_faces = np.asarray(object_mesh.faces, dtype=np.int32)
                tris = mesh_vertices[mesh_faces]
                poly = Poly3DCollection(
                    tris,
                    facecolors=(0.68, 0.72, 0.80, 0.35),
                    edgecolors=(0.35, 0.35, 0.38, 0.15),
                    linewidths=0.12,
                )
                ax_3d.add_collection3d(poly)

            legend_entries: list[tuple[str, Any]] = []
            for region_name in _EXPORT_REGION_LABELS:
                region_points_xyz = visual_region_points.get(region_name)
                if region_points_xyz is None or region_points_xyz.size == 0:
                    continue
                style = _REGION_OVERLAY_STYLE.get(region_name, _REGION_OVERLAY_STYLE["left_wrist"])
                scatter = ax_3d.scatter(
                    region_points_xyz[:, 0],
                    region_points_xyz[:, 1],
                    region_points_xyz[:, 2],
                    c=str(style["mpl_color"]),
                    s=float(style["scatter_size"]),
                    depthshade=False,
                )
                legend_entries.append((str(style["label"]), scatter))

            bound_half = 0.55 * float(np.max(extents_xyz))
            if object_mesh is not None and len(object_mesh.vertices) > 0:
                mesh_bounds = np.asarray(object_mesh.bounds, dtype=np.float64)
                mesh_radius = 0.55 * float(np.max(mesh_bounds[1] - mesh_bounds[0]))
                bound_half = max(bound_half, mesh_radius)
            ax_3d.set_xlim(-bound_half, bound_half)
            ax_3d.set_ylim(-bound_half, bound_half)
            ax_3d.set_zlim(-bound_half, bound_half)
            ax_3d.set_xlabel("X")
            ax_3d.set_ylabel("Y")
            ax_3d.set_zlabel("Z")
            ax_3d.set_box_aspect((1.0, 1.0, 1.0))
            ax_3d.view_init(elev=20, azim=40)
            ax_3d.set_title("3D Perspective")
            if legend_entries:
                ax_3d.legend(
                    [handle for _, handle in legend_entries],
                    [label for label, _ in legend_entries],
                    loc="upper right",
                    frameon=True,
                )

            projection_specs = [
                ("XY Projection", ax_xy, 0, 1, extents_xyz[0], extents_xyz[1], "X", "Y"),
                ("XZ Projection", ax_xz, 0, 2, extents_xyz[0], extents_xyz[2], "X", "Z"),
                ("YZ Projection", ax_yz, 1, 2, extents_xyz[1], extents_xyz[2], "Y", "Z"),
            ]
            for title, axis, dim_a, dim_b, size_a, size_b, label_a, label_b in projection_specs:
                axis.set_title(title)
                axis.plot(
                    [-0.5 * size_a, 0.5 * size_a, 0.5 * size_a, -0.5 * size_a, -0.5 * size_a],
                    [-0.5 * size_b, -0.5 * size_b, 0.5 * size_b, 0.5 * size_b, -0.5 * size_b],
                    color="gray",
                    linewidth=1.2,
                )
                if mesh_vertices is not None and mesh_vertices.size > 0:
                    projected_mesh = mesh_vertices[:, [dim_a, dim_b]]
                    if projected_mesh.shape[0] > 4000:
                        step = max(projected_mesh.shape[0] // 4000, 1)
                        projected_mesh = projected_mesh[::step]
                    axis.scatter(
                        projected_mesh[:, 0],
                        projected_mesh[:, 1],
                        c=[(0.68, 0.72, 0.80, 0.10)],
                        s=2.0,
                        linewidths=0.0,
                    )
                for region_name in _EXPORT_REGION_LABELS:
                    region_points_xyz = visual_region_points.get(region_name)
                    if region_points_xyz is None or region_points_xyz.size == 0:
                        continue
                    style = _REGION_OVERLAY_STYLE.get(region_name, _REGION_OVERLAY_STYLE["left_wrist"])
                    axis.scatter(
                        region_points_xyz[:, dim_a],
                        region_points_xyz[:, dim_b],
                        c=str(style["mpl_color"]),
                        s=float(style["scatter_size"]),
                        linewidths=0.0,
                    )
                axis.set_xlim(-bound_half, bound_half)
                axis.set_ylim(-bound_half, bound_half)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel(label_a)
                axis.set_ylabel(label_b)
                handles = []
                labels = []
                for region_name in _EXPORT_REGION_LABELS:
                    if region_name not in visual_region_points:
                        continue
                    style = _REGION_OVERLAY_STYLE.get(region_name, _REGION_OVERLAY_STYLE["left_wrist"])
                    handles.append(
                        Line2D([], [], marker="o", linestyle="", color=str(style["mpl_color"]), markersize=6)
                    )
                    labels.append(str(style["label"]))
                if handles:
                    axis.legend(handles, labels, loc="upper right", frameon=True)
            plt.tight_layout()
            fig.savefig(clip_dir / "contact_overlay.png", dpi=220)
            plt.close(fig)

    if save_face_heatmap_png:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except Exception as exc:
            logger.warning("Skipping face heatmap PNG export because matplotlib is unavailable: {}", exc)
        else:
            face_axes = {
                "+x": (1, 2, extents_xyz[1], extents_xyz[2]),
                "-x": (1, 2, extents_xyz[1], extents_xyz[2]),
                "+y": (0, 2, extents_xyz[0], extents_xyz[2]),
                "-y": (0, 2, extents_xyz[0], extents_xyz[2]),
                "+z": (0, 1, extents_xyz[0], extents_xyz[1]),
                "-z": (0, 1, extents_xyz[0], extents_xyz[1]),
            }
            face_points: dict[str, list[tuple[np.ndarray, int]]] = defaultdict(list)
            for point_xyz, count in zip(retained_points_xyz, retained_counts, strict=True):
                face_points[_surface_face_name(point_xyz, extents_xyz)].append((point_xyz, int(count)))

            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            for axis, face_name in zip(axes.reshape(-1), ["+x", "-x", "+y", "-y", "+z", "-z"], strict=True):
                dim_a, dim_b, size_a, size_b = face_axes[face_name]
                axis.set_title(face_name)
                axis.add_patch(
                    Rectangle(
                        (-0.5 * size_a, -0.5 * size_b),
                        size_a,
                        size_b,
                        linewidth=1.0,
                        edgecolor="gray",
                        facecolor=(0.92, 0.92, 0.95, 0.3),
                    )
                )
                if face_points[face_name]:
                    pts = np.stack([item[0] for item in face_points[face_name]], axis=0)
                    weights = np.asarray([item[1] for item in face_points[face_name]], dtype=np.float32)
                    scatter = axis.scatter(
                        pts[:, dim_a],
                        pts[:, dim_b],
                        c=weights,
                        cmap="Oranges",
                        s=42,
                        vmin=max(1.0, float(weights.min())),
                        vmax=max(float(weights.max()), float(weights.min())),
                    )
                    fig.colorbar(scatter, ax=axis, fraction=0.046, pad=0.04)
                axis.set_xlim(-0.5 * size_a, 0.5 * size_a)
                axis.set_ylim(-0.5 * size_b, 0.5 * size_b)
                axis.set_aspect("equal", adjustable="box")
                axis.set_xlabel("YZ"[dim_a - 1] if dim_a > 0 else "X")
                axis.set_ylabel("YZ"[dim_b - 1] if dim_b > 0 else "X")
            plt.tight_layout()
            fig.savefig(clip_dir / "primitive_contact_face_heatmaps.png", dpi=220)
            plt.close(fig)


def _build_parallel_clip_batches(env: Any, motion_command: MotionCommand) -> tuple[list[list[int | None]], list[str]]:
    num_envs = int(env.num_envs)
    clip_object_names = list(getattr(motion_command.motion, "clip_object_names", []))
    clip_object_urdf_paths = list(getattr(motion_command.motion, "clip_object_urdf_paths", []))
    clip_queues: dict[str, deque[int]] = defaultdict(deque)

    for clip_idx, clip_id in enumerate(motion_command.motion.clip_ids):
        object_name = clip_object_names[clip_idx] if clip_idx < len(clip_object_names) else ""
        object_urdf = clip_object_urdf_paths[clip_idx] if clip_idx < len(clip_object_urdf_paths) else ""
        key = _object_assignment_key(urdf_path=str(object_urdf), object_name=str(object_name))
        clip_queues[key].append(clip_idx)

    multi_object_enabled = bool(getattr(motion_command, "_multi_object_enabled", False))
    clip_keys = list(clip_queues.keys())
    if multi_object_enabled:
        if not clip_keys:
            raise RuntimeError("No clips available for export.")
        env_keys = [clip_keys[env_id % len(clip_keys)] for env_id in range(num_envs)]
        clip_indices = list(range(motion_command.motion.num_clips))
        batches: list[list[int | None]] = []
        for start in range(0, len(clip_indices), num_envs):
            chunk = clip_indices[start : start + num_envs]
            batches.append(chunk + [None] * (num_envs - len(chunk)))
        return batches, env_keys

    env_object_urdf_paths = getattr(env.simulator, "_env_object_urdf_paths", None)
    if (
        isinstance(env_object_urdf_paths, list)
        and len(env_object_urdf_paths) == num_envs
        and env_object_urdf_paths
    ):
        env_keys = [_object_assignment_key(urdf_path=str(path), object_name="") for path in env_object_urdf_paths]
    else:
        if not clip_keys:
            raise RuntimeError("No clips available for export.")
        if len(clip_keys) == 1:
            env_keys = [clip_keys[0]] * num_envs
        else:
            env_keys = [clip_keys[env_id % len(clip_keys)] for env_id in range(num_envs)]

    missing_keys = [key for key, queue in clip_queues.items() if queue and key not in env_keys]
    if missing_keys and not multi_object_enabled:
        raise RuntimeError(
            "Parallel clip export requires each clip object key to be represented by at least one env slot. "
            f"Missing env slots for {len(missing_keys)} object key(s): {missing_keys[:8]}. "
            "For single-slot multi-URDF export, increase training.num_envs to at least the number of unique "
            "clip object URDFs, or export a smaller sharded motion/object bank."
        )

    batches: list[list[int | None]] = []
    while any(queue for queue in clip_queues.values()):
        batch: list[int | None] = [None] * num_envs
        for env_id, env_key in enumerate(env_keys):
            queue = clip_queues.get(env_key)
            if queue:
                batch[env_id] = queue.popleft()
        if any(item is not None for item in batch):
            batches.append(batch)

    return batches, env_keys


def _make_batch_fixed_clip_ids(
    *,
    motion_command: MotionCommand,
    env_keys: list[str],
    batch_assignments: list[int | None],
) -> torch.Tensor:
    num_envs = len(env_keys)
    device = motion_command.device
    clip_object_names = list(getattr(motion_command.motion, "clip_object_names", []))
    clip_object_urdf_paths = list(getattr(motion_command.motion, "clip_object_urdf_paths", []))
    fallback_by_key: dict[str, int] = {}
    for clip_idx in range(motion_command.motion.num_clips):
        object_name = clip_object_names[clip_idx] if clip_idx < len(clip_object_names) else ""
        object_urdf = clip_object_urdf_paths[clip_idx] if clip_idx < len(clip_object_urdf_paths) else ""
        key = _object_assignment_key(urdf_path=str(object_urdf), object_name=str(object_name))
        fallback_by_key.setdefault(key, clip_idx)

    fixed_clip_ids = torch.zeros(num_envs, dtype=torch.long, device=device)
    for env_id, env_key in enumerate(env_keys):
        assigned = batch_assignments[env_id]
        if assigned is not None:
            fixed_clip_ids[env_id] = int(assigned)
        elif env_key in fallback_by_key:
            fixed_clip_ids[env_id] = int(fallback_by_key[env_key])
        else:
            fixed_clip_ids[env_id] = 0
    return fixed_clip_ids


def _make_clip_accumulator(
    *,
    clip_id: str,
    clip_idx: int,
    object_name: str,
    object_urdf_path: str,
    batch_index: int,
    env_id: int,
    extents_xyz: np.ndarray,
    output_dir: Path,
    tracked_body_names: list[str],
    full_body_names: list[str],
    joint_names: list[str],
    ref_body_name: str,
    motion_fps: float,
    clip_length: int,
    has_object: bool,
    project_contact_to_mesh: bool,
) -> ClipAccumulator:
    clip_dir = output_dir / "clips" / f"{clip_idx:04d}_{_sanitize_name(clip_id)}"
    clip_dir.mkdir(parents=True, exist_ok=True)
    num_bodies = len(tracked_body_names)
    num_full_bodies = len(full_body_names)
    num_joints = len(joint_names)
    rollout_reference: dict[str, np.ndarray] = {
        "valid_steps": np.zeros((clip_length,), dtype=np.bool_),
        "actions": np.zeros((clip_length, num_joints), dtype=np.float32),
        "body_pos_local": np.zeros((clip_length, num_bodies, 3), dtype=np.float32),
        "body_quat_w": np.zeros((clip_length, num_bodies, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((clip_length, num_bodies, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((clip_length, num_bodies, 3), dtype=np.float32),
        "ref_pos_local": np.zeros((clip_length, 3), dtype=np.float32),
        "ref_quat_w": np.zeros((clip_length, 4), dtype=np.float32),
        "ref_lin_vel_w": np.zeros((clip_length, 3), dtype=np.float32),
        "ref_ang_vel_w": np.zeros((clip_length, 3), dtype=np.float32),
        "root_pos_local": np.zeros((clip_length, 3), dtype=np.float32),
        "root_quat_w": np.zeros((clip_length, 4), dtype=np.float32),
        "root_lin_vel_w": np.zeros((clip_length, 3), dtype=np.float32),
        "root_ang_vel_w": np.zeros((clip_length, 3), dtype=np.float32),
    }
    rollout_reference["body_quat_w"][..., 3] = 1.0
    rollout_reference["ref_quat_w"][..., 3] = 1.0
    rollout_reference["root_quat_w"][..., 3] = 1.0
    if has_object:
        rollout_reference["object_pos_local"] = np.zeros((clip_length, 3), dtype=np.float32)
        rollout_reference["object_quat_w"] = np.zeros((clip_length, 4), dtype=np.float32)
        rollout_reference["object_quat_w"][..., 3] = 1.0
        rollout_reference["object_lin_vel_w"] = np.zeros((clip_length, 3), dtype=np.float32)
        rollout_reference["object_ang_vel_w"] = np.zeros((clip_length, 3), dtype=np.float32)
    rollout_reference["target_joint_pos"] = np.zeros((clip_length, 7 + num_joints), dtype=np.float32)
    rollout_reference["target_joint_vel"] = np.zeros((clip_length, 6 + num_joints), dtype=np.float32)
    rollout_reference["target_body_pos_local"] = np.zeros((clip_length, num_bodies, 3), dtype=np.float32)
    rollout_reference["target_body_quat_w"] = np.zeros((clip_length, num_bodies, 4), dtype=np.float32)
    rollout_reference["target_body_quat_w"][..., 3] = 1.0
    rollout_reference["target_body_lin_vel_w"] = np.zeros((clip_length, num_bodies, 3), dtype=np.float32)
    rollout_reference["target_body_ang_vel_w"] = np.zeros((clip_length, num_bodies, 3), dtype=np.float32)
    rollout_reference["target_ref_pos_local"] = np.zeros((clip_length, 3), dtype=np.float32)
    rollout_reference["target_ref_quat_w"] = np.zeros((clip_length, 4), dtype=np.float32)
    rollout_reference["target_ref_quat_w"][..., 3] = 1.0
    rollout_reference["target_root_pos_local"] = np.zeros((clip_length, 3), dtype=np.float32)
    rollout_reference["target_root_quat_w"] = np.zeros((clip_length, 4), dtype=np.float32)
    rollout_reference["target_root_quat_w"][..., 3] = 1.0
    rollout_reference["target_root_lin_vel_w"] = np.zeros((clip_length, 3), dtype=np.float32)
    rollout_reference["target_root_ang_vel_w"] = np.zeros((clip_length, 3), dtype=np.float32)
    if has_object:
        rollout_reference["target_object_pos_local"] = np.zeros((clip_length, 3), dtype=np.float32)
        rollout_reference["target_object_quat_w"] = np.zeros((clip_length, 4), dtype=np.float32)
        rollout_reference["target_object_quat_w"][..., 3] = 1.0
        rollout_reference["target_object_lin_vel_w"] = np.zeros((clip_length, 3), dtype=np.float32)
        rollout_reference["target_object_ang_vel_w"] = np.zeros((clip_length, 3), dtype=np.float32)
    rollout_motion: dict[str, np.ndarray] = {
        "valid_steps": np.zeros((clip_length,), dtype=np.bool_),
        "joint_pos": np.zeros((clip_length, 7 + num_joints), dtype=np.float32),
        "joint_vel": np.zeros((clip_length, 6 + num_joints), dtype=np.float32),
        "body_pos_local": np.zeros((clip_length, num_full_bodies, 3), dtype=np.float32),
        "body_quat_w": np.zeros((clip_length, num_full_bodies, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((clip_length, num_full_bodies, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((clip_length, num_full_bodies, 3), dtype=np.float32),
    }
    rollout_motion["body_quat_w"][..., 3] = 1.0
    if has_object:
        rollout_motion["object_pos_local"] = np.zeros((clip_length, 3), dtype=np.float32)
        rollout_motion["object_quat_w"] = np.zeros((clip_length, 4), dtype=np.float32)
        rollout_motion["object_quat_w"][..., 3] = 1.0
        rollout_motion["object_lin_vel_w"] = np.zeros((clip_length, 3), dtype=np.float32)
        rollout_motion["object_ang_vel_w"] = np.zeros((clip_length, 3), dtype=np.float32)
    object_surface_mesh = None
    if project_contact_to_mesh:
        object_surface_mesh = _load_object_overlay_mesh(
            clip_id=clip_id,
            object_name=object_name,
            object_urdf_path=object_urdf_path,
        )
    return ClipAccumulator(
        clip_id=clip_id,
        clip_index=clip_idx,
        object_name=object_name,
        object_urdf_path=object_urdf_path,
        batch_index=batch_index,
        env_id=env_id,
        extents_xyz=np.asarray(extents_xyz, dtype=np.float32),
        object_surface_mesh=object_surface_mesh,
        contact_surface_projection="mesh" if object_surface_mesh is not None else "primitive_box",
        clip_dir=clip_dir,
        region_point_counts={spec["label"]: defaultdict(int) for spec in _REGION_SPECS.values()},
        region_force_sums={spec["label"]: 0.0 for spec in _REGION_SPECS.values()},
        region_force_max={spec["label"]: 0.0 for spec in _REGION_SPECS.values()},
        region_contact_frames={spec["label"]: 0 for spec in _REGION_SPECS.values()},
        region_contact_interval_start={spec["label"]: -1 for spec in _REGION_SPECS.values()},
        region_contact_interval_end={spec["label"]: -1 for spec in _REGION_SPECS.values()},
        body_force_sums=defaultdict(float),
        body_force_max=defaultdict(float),
        body_contact_frames=defaultdict(int),
        tracked_body_names=list(tracked_body_names),
        full_body_names=list(full_body_names),
        joint_names=list(joint_names),
        ref_body_name=str(ref_body_name),
        motion_fps=float(motion_fps),
        rollout_reference=rollout_reference,
        rollout_motion=rollout_motion,
    )


def _record_rollout_reference_batch(
    motion_command: MotionCommand,
    *,
    accumulators: dict[int, ClipAccumulator],
    active_env_ids: list[int],
) -> None:
    if not active_env_ids:
        return

    env_ids_tensor = torch.tensor(active_env_ids, device=motion_command.device, dtype=torch.long)
    time_steps = motion_command.time_steps.index_select(0, env_ids_tensor).detach().cpu().numpy()
    env_offsets = motion_command._get_env_offsets(env_ids_tensor)

    body_pos_local = (motion_command.robot_body_pos_w.index_select(0, env_ids_tensor) - env_offsets[:, None, :]).detach().cpu().numpy()
    body_quat_w = motion_command.robot_body_quat_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    body_lin_vel_w = motion_command.robot_body_lin_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    body_ang_vel_w = motion_command.robot_body_ang_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    ref_pos_local = (motion_command.robot_ref_pos_w.index_select(0, env_ids_tensor) - env_offsets).detach().cpu().numpy()
    ref_quat_w = motion_command.robot_ref_quat_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    ref_lin_vel_w = motion_command.robot_ref_lin_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    ref_ang_vel_w = motion_command.robot_ref_ang_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    root_pos_local = (motion_command.robot_root_pos_w.index_select(0, env_ids_tensor) - env_offsets).detach().cpu().numpy()
    root_quat_w = motion_command.robot_root_quat_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    root_lin_vel_w = motion_command.robot_root_lin_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    root_ang_vel_w = motion_command.robot_root_ang_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    simulator = motion_command._env.simulator
    full_body_pos_local = (simulator._rigid_body_pos.index_select(0, env_ids_tensor) - env_offsets[:, None, :]).detach().cpu().numpy()
    full_body_quat_w = simulator._rigid_body_rot.index_select(0, env_ids_tensor).detach().cpu().numpy()
    full_body_lin_vel_w = simulator._rigid_body_vel.index_select(0, env_ids_tensor).detach().cpu().numpy()
    full_body_ang_vel_w = simulator._rigid_body_ang_vel.index_select(0, env_ids_tensor).detach().cpu().numpy()
    joint_pos = simulator.dof_pos.index_select(0, env_ids_tensor).detach().cpu().numpy()
    joint_vel = simulator.dof_vel.index_select(0, env_ids_tensor).detach().cpu().numpy()
    joint_pos_full = np.concatenate(
        [root_pos_local, _xyzw_to_wxyz_np(root_quat_w), joint_pos],
        axis=-1,
    )
    joint_vel_full = np.concatenate([root_lin_vel_w, root_ang_vel_w, joint_vel], axis=-1)

    object_pos_local = None
    object_quat_w = None
    object_lin_vel_w = None
    object_ang_vel_w = None
    target_body_pos_local = (motion_command.body_pos_w.index_select(0, env_ids_tensor) - env_offsets[:, None, :]).detach().cpu().numpy()
    target_body_quat_w = motion_command.body_quat_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    target_body_lin_vel_w = motion_command.body_lin_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    target_body_ang_vel_w = motion_command.body_ang_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    target_ref_pos_local = (motion_command.ref_pos_w.index_select(0, env_ids_tensor) - env_offsets).detach().cpu().numpy()
    target_ref_quat_w = motion_command.ref_quat_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    target_root_pos_local = (motion_command.root_pos_w.index_select(0, env_ids_tensor) - env_offsets).detach().cpu().numpy()
    target_root_quat_w = motion_command.root_quat_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
    target_root_lin_vel_w = target_body_lin_vel_w[:, 0, :]
    target_root_ang_vel_w = target_body_ang_vel_w[:, 0, :]
    target_joint_pos = motion_command.joint_pos.index_select(0, env_ids_tensor).detach().cpu().numpy()
    target_joint_vel = motion_command.joint_vel.index_select(0, env_ids_tensor).detach().cpu().numpy()
    target_joint_pos_full = np.concatenate(
        [target_root_pos_local, _xyzw_to_wxyz_np(target_root_quat_w), target_joint_pos],
        axis=-1,
    )
    target_joint_vel_full = np.concatenate([target_root_lin_vel_w, target_root_ang_vel_w, target_joint_vel], axis=-1)
    target_object_pos_local = None
    target_object_quat_w = None
    target_object_lin_vel_w = None
    target_object_ang_vel_w = None
    if motion_command.motion.has_object:
        object_pos_local = (
            motion_command.simulator_object_pos_w.index_select(0, env_ids_tensor) - env_offsets
        ).detach().cpu().numpy()
        object_quat_w = motion_command.simulator_object_quat_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
        object_lin_vel_w = motion_command.simulator_object_lin_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
        object_ang_vel_w = motion_command.simulator_object_ang_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
        target_object_pos_local = (
            motion_command.object_pos_w.index_select(0, env_ids_tensor) - env_offsets
        ).detach().cpu().numpy()
        target_object_quat_w = motion_command.object_quat_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
        target_object_lin_vel_w = motion_command.object_lin_vel_w.index_select(0, env_ids_tensor).detach().cpu().numpy()
        target_object_ang_vel_w = np.zeros_like(target_object_lin_vel_w)

    for batch_slot, env_id in enumerate(active_env_ids):
        accumulator = accumulators[env_id]
        rollout_reference = accumulator.rollout_reference
        rollout_motion = accumulator.rollout_motion
        step_idx = int(time_steps[batch_slot])
        if step_idx < 0 or step_idx >= int(rollout_reference["valid_steps"].shape[0]):
            continue
        if bool(rollout_reference["valid_steps"][step_idx]):
            continue

        rollout_reference["valid_steps"][step_idx] = True
        rollout_reference["body_pos_local"][step_idx] = body_pos_local[batch_slot]
        rollout_reference["body_quat_w"][step_idx] = body_quat_w[batch_slot]
        rollout_reference["body_lin_vel_w"][step_idx] = body_lin_vel_w[batch_slot]
        rollout_reference["body_ang_vel_w"][step_idx] = body_ang_vel_w[batch_slot]
        rollout_reference["ref_pos_local"][step_idx] = ref_pos_local[batch_slot]
        rollout_reference["ref_quat_w"][step_idx] = ref_quat_w[batch_slot]
        rollout_reference["ref_lin_vel_w"][step_idx] = ref_lin_vel_w[batch_slot]
        rollout_reference["ref_ang_vel_w"][step_idx] = ref_ang_vel_w[batch_slot]
        rollout_reference["root_pos_local"][step_idx] = root_pos_local[batch_slot]
        rollout_reference["root_quat_w"][step_idx] = root_quat_w[batch_slot]
        rollout_reference["root_lin_vel_w"][step_idx] = root_lin_vel_w[batch_slot]
        rollout_reference["root_ang_vel_w"][step_idx] = root_ang_vel_w[batch_slot]
        rollout_reference["target_joint_pos"][step_idx] = target_joint_pos_full[batch_slot]
        rollout_reference["target_joint_vel"][step_idx] = target_joint_vel_full[batch_slot]
        rollout_reference["target_body_pos_local"][step_idx] = target_body_pos_local[batch_slot]
        rollout_reference["target_body_quat_w"][step_idx] = target_body_quat_w[batch_slot]
        rollout_reference["target_body_lin_vel_w"][step_idx] = target_body_lin_vel_w[batch_slot]
        rollout_reference["target_body_ang_vel_w"][step_idx] = target_body_ang_vel_w[batch_slot]
        rollout_reference["target_ref_pos_local"][step_idx] = target_ref_pos_local[batch_slot]
        rollout_reference["target_ref_quat_w"][step_idx] = target_ref_quat_w[batch_slot]
        rollout_reference["target_root_pos_local"][step_idx] = target_root_pos_local[batch_slot]
        rollout_reference["target_root_quat_w"][step_idx] = target_root_quat_w[batch_slot]
        rollout_reference["target_root_lin_vel_w"][step_idx] = target_root_lin_vel_w[batch_slot]
        rollout_reference["target_root_ang_vel_w"][step_idx] = target_root_ang_vel_w[batch_slot]
        rollout_motion["valid_steps"][step_idx] = True
        rollout_motion["joint_pos"][step_idx] = joint_pos_full[batch_slot]
        rollout_motion["joint_vel"][step_idx] = joint_vel_full[batch_slot]
        rollout_motion["body_pos_local"][step_idx] = full_body_pos_local[batch_slot]
        rollout_motion["body_quat_w"][step_idx] = full_body_quat_w[batch_slot]
        rollout_motion["body_lin_vel_w"][step_idx] = full_body_lin_vel_w[batch_slot]
        rollout_motion["body_ang_vel_w"][step_idx] = full_body_ang_vel_w[batch_slot]
        if object_pos_local is not None:
            rollout_reference["object_pos_local"][step_idx] = object_pos_local[batch_slot]
            rollout_reference["object_quat_w"][step_idx] = object_quat_w[batch_slot]
            rollout_reference["object_lin_vel_w"][step_idx] = object_lin_vel_w[batch_slot]
            rollout_reference["object_ang_vel_w"][step_idx] = object_ang_vel_w[batch_slot]
            rollout_motion["object_pos_local"][step_idx] = object_pos_local[batch_slot]
            rollout_motion["object_quat_w"][step_idx] = object_quat_w[batch_slot]
            rollout_motion["object_lin_vel_w"][step_idx] = object_lin_vel_w[batch_slot]
            rollout_motion["object_ang_vel_w"][step_idx] = object_ang_vel_w[batch_slot]
            rollout_reference["target_object_pos_local"][step_idx] = target_object_pos_local[batch_slot]
            rollout_reference["target_object_quat_w"][step_idx] = target_object_quat_w[batch_slot]
            rollout_reference["target_object_lin_vel_w"][step_idx] = target_object_lin_vel_w[batch_slot]
            rollout_reference["target_object_ang_vel_w"][step_idx] = target_object_ang_vel_w[batch_slot]


def _record_actions_batch(
    motion_command: MotionCommand,
    *,
    accumulators: dict[int, ClipAccumulator],
    active_env_ids: list[int],
    actions: torch.Tensor,
) -> None:
    if not active_env_ids:
        return

    env_ids_tensor = torch.tensor(active_env_ids, device=motion_command.device, dtype=torch.long)
    time_steps = motion_command.time_steps.index_select(0, env_ids_tensor).detach().cpu().numpy()
    actions_np = actions.index_select(0, env_ids_tensor).detach().cpu().numpy().astype(np.float32, copy=False)

    for batch_slot, env_id in enumerate(active_env_ids):
        accumulator = accumulators[env_id]
        step_idx = int(time_steps[batch_slot])
        actions_out = accumulator.rollout_reference.get("actions")
        if actions_out is None or step_idx < 0 or step_idx >= int(actions_out.shape[0]):
            continue
        actions_out[step_idx] = actions_np[batch_slot]


def _record_policy_io_batch(
    algo: BaseAlgo,
    obs_dict: dict[str, torch.Tensor],
    motion_command: MotionCommand,
    *,
    accumulators: dict[int, ClipAccumulator],
    active_env_ids: list[int],
) -> None:
    if not active_env_ids:
        return

    actor_obs_raw = torch.cat([obs_dict[key] for key in algo.actor_obs_keys], dim=1)
    if hasattr(algo, "_normalize_actor_obs"):
        actor_obs_norm = algo._normalize_actor_obs(actor_obs_raw, update=False)
    else:
        actor_obs_norm = actor_obs_raw

    actor_perception_key = getattr(algo, "actor_perception_key", "") or ""
    perception_obs = obs_dict.get(actor_perception_key) if actor_perception_key else None

    env_ids_tensor = torch.tensor(active_env_ids, device=motion_command.device, dtype=torch.long)
    time_steps = motion_command.time_steps.index_select(0, env_ids_tensor).detach().cpu().numpy()
    actor_obs_raw_np = actor_obs_raw.index_select(0, env_ids_tensor).detach().cpu().numpy().astype(np.float32, copy=False)
    actor_obs_norm_np = actor_obs_norm.index_select(0, env_ids_tensor).detach().cpu().numpy().astype(np.float32, copy=False)
    perception_obs_np = None
    if perception_obs is not None:
        perception_obs_np = (
            perception_obs.index_select(0, env_ids_tensor).detach().cpu().numpy().astype(np.float32, copy=False)
        )

    for batch_slot, env_id in enumerate(active_env_ids):
        accumulator = accumulators[env_id]
        rollout_reference = accumulator.rollout_reference
        step_idx = int(time_steps[batch_slot])
        valid_steps = rollout_reference["valid_steps"]
        if step_idx < 0 or step_idx >= int(valid_steps.shape[0]):
            continue

        if "actor_obs_raw" not in rollout_reference:
            rollout_reference["actor_obs_raw"] = np.zeros(
                (valid_steps.shape[0], actor_obs_raw_np.shape[1]), dtype=np.float32
            )
            rollout_reference["actor_obs_norm"] = np.zeros(
                (valid_steps.shape[0], actor_obs_norm_np.shape[1]), dtype=np.float32
            )
        rollout_reference["actor_obs_raw"][step_idx] = actor_obs_raw_np[batch_slot]
        rollout_reference["actor_obs_norm"][step_idx] = actor_obs_norm_np[batch_slot]

        if perception_obs_np is not None:
            if "perception_obs" not in rollout_reference:
                rollout_reference["perception_obs"] = np.zeros(
                    (valid_steps.shape[0], perception_obs_np.shape[1]), dtype=np.float32
                )
            rollout_reference["perception_obs"][step_idx] = perception_obs_np[batch_slot]


def _collect_region_measurements_batch(
    motion_command: MotionCommand,
    *,
    accumulators: dict[int, ClipAccumulator],
    active_env_ids: list[int],
    step_idx: int,
    force_threshold: float,
    voxel_size: float,
) -> None:
    if not active_env_ids:
        return
    simulator_body_names = list(getattr(motion_command._env.simulator, "body_names", []))
    for region_name, spec in _REGION_SPECS.items():
        body_names = list(spec["body_names"])
        if not body_names:
            continue
        missing = [name for name in body_names if name not in simulator_body_names]
        if missing:
            raise ValueError(f"Requested export contact bodies {missing} are not available in simulator bodies.")

        label = str(spec["label"])
        force_history = motion_command.get_body_object_contact_force_history(body_names)
        per_body_force = torch.max(torch.norm(force_history, dim=-1), dim=1)[0]
        region_force = torch.max(per_body_force, dim=1)[0]

        body_indices = torch.tensor(
            [simulator_body_names.index(name) for name in body_names],
            device=motion_command.device,
            dtype=torch.long,
        )
        body_pos_obj = motion_command._body_positions_in_object_frame(body_indices)

        region_force_np = region_force.detach().cpu().numpy()
        per_body_force_np = per_body_force.detach().cpu().numpy()
        body_pos_obj_np = body_pos_obj.detach().cpu().numpy()
        for env_id in active_env_ids:
            force_value = float(region_force_np[env_id])
            accumulator = accumulators[env_id]
            if force_value <= force_threshold:
                continue
            if accumulator.region_contact_interval_start[label] < 0:
                accumulator.region_contact_interval_start[label] = int(step_idx)
            accumulator.region_contact_interval_end[label] = max(
                int(accumulator.region_contact_interval_end[label]),
                int(step_idx) + 1,
            )
            accumulator.region_force_sums[label] += force_value
            accumulator.region_force_max[label] = max(accumulator.region_force_max[label], force_value)
            accumulator.region_contact_frames[label] += 1
            for body_slot in range(len(body_names)):
                body_force_value = float(per_body_force_np[env_id, body_slot])
                if body_force_value <= force_threshold:
                    continue
                point_obj = body_pos_obj_np[env_id, body_slot]
                point_surface = _project_point_to_object_surface(point_obj, accumulator)
                key = _quantize_point(point_surface, voxel_size)
                accumulator.region_point_counts[label][key] += 1


def _collect_body_force_stats_batch(
    motion_command: MotionCommand,
    *,
    accumulators: dict[int, ClipAccumulator],
    active_env_ids: list[int],
    force_threshold: float,
) -> None:
    if not active_env_ids:
        return
    unique_body_names = sorted(
        {
            body_name
            for region_spec in _REGION_SPECS.values()
            for body_name in region_spec["body_names"]
        }
    )
    if not unique_body_names:
        return
    force_history = motion_command.get_body_object_contact_force_history(unique_body_names)
    current_force = torch.linalg.norm(force_history[:, 0], dim=-1).detach().cpu().numpy()
    for env_id in active_env_ids:
        accumulator = accumulators[env_id]
        for body_name, force_value in zip(unique_body_names, current_force[env_id], strict=True):
            force_scalar = float(force_value)
            accumulator.body_force_sums[body_name] += force_scalar
            accumulator.body_force_max[body_name] = max(accumulator.body_force_max[body_name], force_scalar)
            if force_scalar > force_threshold:
                accumulator.body_contact_frames[body_name] += 1


def _valid_step_indices(valid_steps: np.ndarray) -> np.ndarray:
    valid_steps = np.asarray(valid_steps, dtype=np.bool_).reshape(-1)
    indices = np.flatnonzero(valid_steps)
    if indices.size == 0:
        return np.asarray([0], dtype=np.int64)
    return indices.astype(np.int64, copy=False)


def _finalize_clip_output(
    *,
    accumulator: ClipAccumulator,
    export_cfg: ExportConfig,
    num_steps: int,
    motion_end_reached: bool,
    terminated: bool,
    timeout: bool,
    final_pos_error: float,
    final_rot_error: float,
) -> ClipSummary:
    overall_point_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for per_region_counts in accumulator.region_point_counts.values():
        for key, count in per_region_counts.items():
            overall_point_counts[key] += count

    retained_points_xyz, retained_counts = _retained_points_from_counts(
        overall_point_counts,
        voxel_size=export_cfg.contact_voxel_size,
        min_frames=export_cfg.min_contact_frames,
    )
    np.save(accumulator.clip_dir / "primitive_contact_points.npy", retained_points_xyz)
    np.save(accumulator.clip_dir / "primitive_contact_point_counts.npy", retained_counts)

    stable_contact_success = bool(retained_points_xyz.shape[0] > 0)
    final_position_success = motion_end_reached and final_pos_error <= float(export_cfg.success_position_threshold)
    success = stable_contact_success
    if stable_contact_success and final_position_success:
        status = "success_contact_and_final_position"
    elif stable_contact_success:
        status = "success_stable_contact"
    elif terminated and not motion_end_reached:
        status = "failed_early_termination"
    elif timeout:
        status = "failed_timeout"
    elif motion_end_reached:
        status = "failed_no_stable_contact"
    else:
        status = "failed_max_steps"

    retained_points_by_region: dict[str, np.ndarray] = {}
    contact_intervals_by_region: dict[str, np.ndarray] = {}
    for spec in _REGION_SPECS.values():
        label = str(spec["label"])
        region_points_xyz, region_counts = _retained_points_from_counts(
            accumulator.region_point_counts[label],
            voxel_size=export_cfg.contact_voxel_size,
            min_frames=export_cfg.min_contact_frames,
        )
        retained_points_by_region[label] = region_points_xyz
        np.save(accumulator.clip_dir / f"{label}_contact_points.npy", region_points_xyz)
        np.save(accumulator.clip_dir / f"{label}_contact_point_counts.npy", region_counts)
        start_step = int(accumulator.region_contact_interval_start[label])
        end_step = int(accumulator.region_contact_interval_end[label])
        interval_steps = (
            np.asarray([start_step, end_step], dtype=np.int32)
            if start_step >= 0 and end_step > start_step
            else np.asarray([-1, -1], dtype=np.int32)
        )
        contact_intervals_by_region[label] = interval_steps
        np.save(accumulator.clip_dir / f"{label}_contact_interval_steps.npy", interval_steps)

    legacy_arm_point_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    for label in _LEGACY_ARM_AGGREGATE_REGION_LABELS:
        for key, count in accumulator.region_point_counts[label].items():
            legacy_arm_point_counts[key] += count
    legacy_arm_points_xyz, legacy_arm_counts = _retained_points_from_counts(
        legacy_arm_point_counts,
        voxel_size=export_cfg.contact_voxel_size,
        min_frames=export_cfg.min_contact_frames,
    )
    np.save(accumulator.clip_dir / f"{_LEGACY_ARM_AGGREGATE_LABEL}_contact_points.npy", legacy_arm_points_xyz)
    np.save(accumulator.clip_dir / f"{_LEGACY_ARM_AGGREGATE_LABEL}_contact_point_counts.npy", legacy_arm_counts)

    legacy_arm_intervals: list[tuple[int, int]] = []
    for label in _LEGACY_ARM_AGGREGATE_REGION_LABELS:
        start_step = int(accumulator.region_contact_interval_start[label])
        end_step = int(accumulator.region_contact_interval_end[label])
        if start_step >= 0 and end_step > start_step:
            legacy_arm_intervals.append((start_step, end_step))
    legacy_arm_interval_steps = (
        np.asarray(
            [min(start for start, _ in legacy_arm_intervals), max(end for _, end in legacy_arm_intervals)],
            dtype=np.int32,
        )
        if legacy_arm_intervals
        else np.asarray([-1, -1], dtype=np.int32)
    )
    contact_intervals_by_region[_LEGACY_ARM_AGGREGATE_LABEL] = legacy_arm_interval_steps
    np.save(
        accumulator.clip_dir / f"{_LEGACY_ARM_AGGREGATE_LABEL}_contact_interval_steps.npy",
        legacy_arm_interval_steps,
    )

    np.savez(
        accumulator.clip_dir / "contact_intervals.npz",
        **{label: interval for label, interval in contact_intervals_by_region.items()},
    )
    (accumulator.clip_dir / "contact_intervals.json").write_text(
        json.dumps({label: interval.tolist() for label, interval in contact_intervals_by_region.items()}, indent=2),
        encoding="utf-8",
    )

    display_points_xyz, _, display_point_labels = _build_display_points_from_region_counts(
        overall_point_counts,
        accumulator.region_point_counts,
        voxel_size=export_cfg.contact_voxel_size,
        min_frames=export_cfg.min_contact_frames,
    )

    region_rows: list[dict[str, Any]] = []
    for spec in _REGION_SPECS.values():
        label = str(spec["label"])
        frames = int(accumulator.region_contact_frames[label])
        retained_region_points, _ = _retained_points_from_counts(
            accumulator.region_point_counts[label],
            voxel_size=export_cfg.contact_voxel_size,
            min_frames=export_cfg.min_contact_frames,
        )
        region_rows.append(
            {
                "region": label,
                "body_names": ",".join(spec["body_names"]),
                "contact_frames": frames,
                "avg_force_over_contact_frames": (accumulator.region_force_sums[label] / frames) if frames > 0 else 0.0,
                "max_force": accumulator.region_force_max[label],
                "retained_point_count": int(retained_region_points.shape[0]),
            }
        )
    _write_csv(accumulator.clip_dir / "region_contact_stats.csv", region_rows)

    body_rows: list[dict[str, Any]] = []
    for body_name in sorted(accumulator.body_force_sums.keys()):
        frames = int(accumulator.body_contact_frames[body_name])
        body_rows.append(
            {
                "body_name": body_name,
                "contact_frames": frames,
                "avg_force_over_all_steps": accumulator.body_force_sums[body_name] / max(num_steps, 1),
                "max_force": accumulator.body_force_max[body_name],
            }
        )
    _write_csv(accumulator.clip_dir / "body_contact_stats.csv", body_rows)

    rollout_reference_payload: dict[str, Any] = {
        "clip_id": np.asarray(accumulator.clip_id),
        "clip_index": np.asarray(accumulator.clip_index, dtype=np.int32),
        "tracked_body_names": np.asarray(accumulator.tracked_body_names),
        "ref_body_name": np.asarray(accumulator.ref_body_name),
        "trajectory_length": np.asarray(accumulator.rollout_reference["valid_steps"].shape[0], dtype=np.int32),
    }
    rollout_reference_payload.update(accumulator.rollout_reference)
    np.savez_compressed(accumulator.clip_dir / "teacher_rollout_reference.npz", **rollout_reference_payload)

    motion_bank_dir = accumulator.clip_dir.parent.parent / "motion_bank"
    motion_bank_dir.mkdir(parents=True, exist_ok=True)
    valid_motion_steps = _valid_step_indices(accumulator.rollout_motion["valid_steps"])
    rollout_motion_payload: dict[str, Any] = {
        "fps": np.asarray([accumulator.motion_fps], dtype=np.int32),
        "body_names": np.asarray(accumulator.full_body_names),
        "joint_names": np.asarray(accumulator.joint_names),
        "joint_pos": accumulator.rollout_motion["joint_pos"][valid_motion_steps],
        "joint_vel": accumulator.rollout_motion["joint_vel"][valid_motion_steps],
        "body_pos_w": accumulator.rollout_motion["body_pos_local"][valid_motion_steps],
        "body_quat_w": _xyzw_to_wxyz_np(accumulator.rollout_motion["body_quat_w"][valid_motion_steps]),
        "body_lin_vel_w": accumulator.rollout_motion["body_lin_vel_w"][valid_motion_steps],
        "body_ang_vel_w": accumulator.rollout_motion["body_ang_vel_w"][valid_motion_steps],
        "object_name": np.asarray(accumulator.object_name),
        "object_urdf_path": np.asarray(accumulator.object_urdf_path),
        "object_size": np.asarray(accumulator.extents_xyz, dtype=np.float32),
    }
    if "object_pos_local" in accumulator.rollout_motion:
        rollout_motion_payload["object_pos_w"] = accumulator.rollout_motion["object_pos_local"][valid_motion_steps]
        rollout_motion_payload["object_quat_w"] = _xyzw_to_wxyz_np(
            accumulator.rollout_motion["object_quat_w"][valid_motion_steps]
        )
        rollout_motion_payload["object_lin_vel_w"] = accumulator.rollout_motion["object_lin_vel_w"][valid_motion_steps]
        rollout_motion_payload["object_ang_vel_w"] = accumulator.rollout_motion["object_ang_vel_w"][valid_motion_steps]
    motion_bank_path = motion_bank_dir / f"{_sanitize_name(accumulator.clip_id)}.npz"
    np.savez_compressed(motion_bank_path, **rollout_motion_payload)

    metadata = {
        "clip_id": accumulator.clip_id,
        "clip_index": accumulator.clip_index,
        "object_name": accumulator.object_name,
        "object_urdf_path": accumulator.object_urdf_path,
        "batch_index": accumulator.batch_index,
        "env_id": accumulator.env_id,
        "primitive_extents_xyz": accumulator.extents_xyz.tolist(),
        "contact_surface_projection": accumulator.contact_surface_projection,
        "num_steps": num_steps,
        "motion_end_reached": motion_end_reached,
        "terminated": terminated,
        "timeout": timeout,
        "success": success,
        "stable_contact_success": stable_contact_success,
        "final_position_success": final_position_success,
        "status": status,
        "final_object_position_error_m": final_pos_error,
        "final_object_rotation_error_rad": final_rot_error,
        "min_contact_frames": int(export_cfg.min_contact_frames),
        "contact_force_threshold": float(export_cfg.contact_force_threshold),
        "contact_voxel_size": float(export_cfg.contact_voxel_size),
        "success_position_threshold": float(export_cfg.success_position_threshold),
        "teacher_rollout_reference_path": "teacher_rollout_reference.npz",
        "contact_intervals_path": "contact_intervals.json",
        "teacher_rollout_ref_body_name": accumulator.ref_body_name,
        "teacher_rollout_tracked_body_names": accumulator.tracked_body_names,
        "teacher_rollout_valid_step_count": int(np.count_nonzero(accumulator.rollout_reference["valid_steps"])),
        "teacher_rollout_motion_bank_path": str(Path("..") / ".." / "motion_bank" / motion_bank_path.name),
        "teacher_rollout_motion_valid_step_count": int(valid_motion_steps.size),
    }
    (accumulator.clip_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _save_overlay_assets(
        accumulator.clip_dir,
        clip_id=accumulator.clip_id,
        object_name=accumulator.object_name,
        object_urdf_path=accumulator.object_urdf_path,
        extents_xyz=accumulator.extents_xyz,
        retained_points_xyz=retained_points_xyz,
        retained_counts=retained_counts,
        display_points_xyz=display_points_xyz,
        display_point_labels=display_point_labels,
        save_glb=export_cfg.save_glb,
        save_preview_png=export_cfg.save_preview_png,
        save_face_heatmap_png=export_cfg.save_face_heatmap_png,
    )

    return ClipSummary(
        clip_id=accumulator.clip_id,
        clip_index=accumulator.clip_index,
        object_name=accumulator.object_name,
        object_urdf_path=accumulator.object_urdf_path,
        batch_index=accumulator.batch_index,
        env_id=accumulator.env_id,
        num_steps=num_steps,
        motion_end_reached=motion_end_reached,
        terminated=terminated,
        timeout=timeout,
        success=success,
        stable_contact_success=stable_contact_success,
        final_position_success=final_position_success,
        status=status,
        final_object_position_error_m=final_pos_error,
        final_object_rotation_error_rad=final_rot_error,
        primitive_extent_x=float(accumulator.extents_xyz[0]),
        primitive_extent_y=float(accumulator.extents_xyz[1]),
        primitive_extent_z=float(accumulator.extents_xyz[2]),
        retained_contact_point_count=int(retained_points_xyz.shape[0]),
        left_wrist_contact_frames=int(accumulator.region_contact_frames["left_wrist"]),
        right_wrist_contact_frames=int(accumulator.region_contact_frames["right_wrist"]),
        left_elbow_contact_frames=int(accumulator.region_contact_frames["left_elbow"]),
        right_elbow_contact_frames=int(accumulator.region_contact_frames["right_elbow"]),
        left_wrist_roll_contact_frames=int(accumulator.region_contact_frames["left_wrist_roll"]),
        right_wrist_roll_contact_frames=int(accumulator.region_contact_frames["right_wrist_roll"]),
        left_wrist_pitch_contact_frames=int(accumulator.region_contact_frames["left_wrist_pitch"]),
        right_wrist_pitch_contact_frames=int(accumulator.region_contact_frames["right_wrist_pitch"]),
        torso_contact_frames=int(accumulator.region_contact_frames["torso"]),
    )


def _collect_batch(
    *,
    env: Any,
    algo: BaseAlgo,
    motion_command: MotionCommand,
    batch_index: int,
    batch_assignments: list[int | None],
    env_keys: list[str],
    export_cfg: ExportConfig,
    output_dir: Path,
    policy_fn,
) -> list[ClipSummary]:
    active_env_ids = [env_id for env_id, clip_idx in enumerate(batch_assignments) if clip_idx is not None]
    if not active_env_ids:
        return []

    fixed_clip_ids = _make_batch_fixed_clip_ids(
        motion_command=motion_command,
        env_keys=env_keys,
        batch_assignments=batch_assignments,
    )
    motion_command._fixed_clip_ids = fixed_clip_ids

    obs_dict = _reset_envs_without_advancing(env)
    if hasattr(algo, "actor") and hasattr(algo.actor, "reset"):
        algo.actor.reset(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))

    clip_ids_after_reset = motion_command.clip_ids.detach().cpu().numpy()
    time_steps_after_reset = motion_command.time_steps.detach().cpu().numpy()
    accumulators: dict[int, ClipAccumulator] = {}
    finished: dict[int, bool] = {}
    num_steps_by_env: dict[int, int] = {env_id: 0 for env_id in active_env_ids}
    motion_end_by_env: dict[int, bool] = {env_id: False for env_id in active_env_ids}
    terminated_by_env: dict[int, bool] = {env_id: False for env_id in active_env_ids}
    timeout_by_env: dict[int, bool] = {env_id: False for env_id in active_env_ids}
    pos_error_by_env: dict[int, float] = {env_id: math.inf for env_id in active_env_ids}
    rot_error_by_env: dict[int, float] = {env_id: math.inf for env_id in active_env_ids}

    env_ids_tensor = torch.tensor(active_env_ids, device=env.device, dtype=torch.long)
    if hasattr(motion_command, "_resolved_object_size_for_env_ids"):
        extents_batch = motion_command._resolved_object_size_for_env_ids(env_ids_tensor).detach().cpu().numpy()
    else:
        extents_batch = motion_command.object_size[env_ids_tensor].detach().cpu().numpy()

    clip_object_names = list(getattr(motion_command.motion, "clip_object_names", []))
    clip_object_urdf_paths = list(getattr(motion_command.motion, "clip_object_urdf_paths", []))
    tracked_body_names = list(motion_command.motion_cfg.body_names_to_track)
    full_body_names = list(getattr(env.simulator, "_body_list", []))
    joint_names = list(getattr(env.simulator, "dof_names", []))
    ref_body_name = str(motion_command.motion_cfg.body_name_ref[0])
    motion_fps = float(np.asarray(getattr(motion_command.motion, "fps", 0)).reshape(-1)[0])
    clip_lengths = motion_command.motion.clip_lengths.detach().cpu().numpy()
    for batch_slot, env_id in enumerate(active_env_ids):
        clip_idx = int(batch_assignments[env_id])
        expected_clip_idx = clip_idx
        actual_clip_idx = int(clip_ids_after_reset[env_id])
        if actual_clip_idx != expected_clip_idx:
            raise RuntimeError(
                f"Batch clip assignment mismatch for env {env_id}: expected {expected_clip_idx}, got {actual_clip_idx}."
            )
        if int(time_steps_after_reset[env_id]) != 0:
            raise RuntimeError(
                f"Expected env {env_id} to reset to timestep 0, got {int(time_steps_after_reset[env_id])}."
            )

        clip_id = str(motion_command.motion.clip_ids[clip_idx])
        object_name = clip_object_names[clip_idx] if clip_idx < len(clip_object_names) else ""
        object_urdf_path = clip_object_urdf_paths[clip_idx] if clip_idx < len(clip_object_urdf_paths) else ""
        accumulators[env_id] = _make_clip_accumulator(
            clip_id=clip_id,
            clip_idx=clip_idx,
            object_name=str(object_name),
            object_urdf_path=str(object_urdf_path),
            batch_index=batch_index,
            env_id=env_id,
            extents_xyz=extents_batch[batch_slot],
            output_dir=output_dir,
            tracked_body_names=tracked_body_names,
            full_body_names=full_body_names,
            joint_names=joint_names,
            ref_body_name=ref_body_name,
            motion_fps=motion_fps,
            clip_length=int(clip_lengths[clip_idx]),
            has_object=bool(motion_command.motion.has_object),
            project_contact_to_mesh=export_cfg.project_contact_to_mesh,
        )
        finished[env_id] = False

    obs_dict = _write_motion_command_targets_to_sim(
        motion_command,
        active_env_ids=active_env_ids,
    )

    max_rollout_steps = export_cfg.max_rollout_steps
    if max_rollout_steps is None or max_rollout_steps <= 0:
        max_rollout_steps = max(int(clip_lengths[int(batch_assignments[env_id])]) for env_id in active_env_ids) + 8

    _record_rollout_reference_batch(
        motion_command,
        accumulators=accumulators,
        active_env_ids=active_env_ids,
    )

    for step_idx in range(int(max_rollout_steps)):
        actions = _policy_step(algo, obs_dict, policy_fn)
        _record_policy_io_batch(
            algo,
            obs_dict,
            motion_command,
            accumulators=accumulators,
            active_env_ids=[env_id for env_id in active_env_ids if not finished[env_id]],
        )
        _record_actions_batch(
            motion_command,
            accumulators=accumulators,
            active_env_ids=[env_id for env_id in active_env_ids if not finished[env_id]],
            actions=actions,
        )
        obs_dict, _, _, _ = env.step({"actions": actions})

        still_active = [env_id for env_id in active_env_ids if not finished[env_id]]
        if not still_active:
            break

        _record_rollout_reference_batch(
            motion_command,
            accumulators=accumulators,
            active_env_ids=still_active,
        )

        _collect_region_measurements_batch(
            motion_command,
            accumulators=accumulators,
            active_env_ids=still_active,
            step_idx=step_idx,
            force_threshold=export_cfg.contact_force_threshold,
            voxel_size=export_cfg.contact_voxel_size,
        )
        _collect_body_force_stats_batch(
            motion_command,
            accumulators=accumulators,
            active_env_ids=still_active,
            force_threshold=export_cfg.contact_force_threshold,
        )

        reset_flags = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        timeout_flags = torch.zeros_like(reset_flags)
        if getattr(env, "termination_manager", None) is not None:
            reset_flags, timeout_flags = env.termination_manager.check()
        motion_end_mask = motion_command.motion_end_mask()

        pos_error_tensor = torch.norm(motion_command.object_pos_w - motion_command.simulator_object_pos_w, dim=1)
        rot_error_tensor = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)

        for env_id in still_active:
            num_steps_by_env[env_id] = step_idx + 1
            motion_end_by_env[env_id] = bool(motion_end_mask[env_id].item())
            terminated_by_env[env_id] = bool(reset_flags[env_id].item())
            timeout_by_env[env_id] = bool(timeout_flags[env_id].item())
            pos_error_by_env[env_id] = float(pos_error_tensor[env_id].item())
            rot_error_by_env[env_id] = float(rot_error_tensor[env_id].item())
            if motion_end_by_env[env_id] or terminated_by_env[env_id]:
                finished[env_id] = True

        if all(finished[env_id] for env_id in active_env_ids):
            break

    summaries: list[ClipSummary] = []
    for env_id in active_env_ids:
        summary = _finalize_clip_output(
            accumulator=accumulators[env_id],
            export_cfg=export_cfg,
            num_steps=num_steps_by_env[env_id],
            motion_end_reached=motion_end_by_env[env_id],
            terminated=terminated_by_env[env_id],
            timeout=timeout_by_env[env_id] or not finished[env_id],
            final_pos_error=pos_error_by_env[env_id],
            final_rot_error=rot_error_by_env[env_id],
        )
        summaries.append(summary)

    return summaries


def run_export_with_tyro(
    tyro_config: ExperimentConfig,
    checkpoint_cfg: CheckpointConfig,
    export_cfg: ExportConfig,
    saved_config: ExperimentConfig,
    saved_wandb_path: str | None,
) -> Path:
    tyro_config = apply_observation_overrides(tyro_config)
    tyro_config = apply_perception_overrides(tyro_config)
    tyro_config = _with_contact_sensor_bodies(tyro_config)

    output_dir = _normalize_output_dir(export_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="contact_export")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_cfg.checkpoint is None:
        raise ValueError("A checkpoint must be provided for teacher contact export.")
    checkpoint_path = load_checkpoint(str(checkpoint_cfg.checkpoint), str(eval_log_dir))
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.attach_checkpoint_metadata(saved_config, saved_wandb_path)
    algo.load(str(checkpoint_path))
    logger.info("Checkpoint load returned; switching exporter into eval/inference mode.")

    if hasattr(algo, "_eval_mode"):
        algo._eval_mode()
    logger.info("Algo eval mode enabled.")
    env.set_is_evaluating()
    logger.info("Environment marked as evaluating.")
    policy_fn = algo.get_inference_policy()
    logger.info("Inference policy ready.")

    motion_command = _ensure_motion_command(env)
    logger.info("motion_command resolved; building parallel clip batches.")
    batches, env_keys = _build_parallel_clip_batches(env, motion_command)
    logger.info(
        "Exporting teacher contact rollouts for {} clip(s) in {} batch(es) with {} env(s).",
        motion_command.motion.num_clips,
        len(batches),
        env.num_envs,
    )

    summaries: list[ClipSummary] = []
    original_fixed_clip_ids = motion_command._fixed_clip_ids.clone() if motion_command._fixed_clip_ids is not None else None
    try:
        try:
            for batch_index, batch_assignments in enumerate(batches):
                active_count = sum(clip_idx is not None for clip_idx in batch_assignments)
                logger.info(
                    "Rollout batch {}/{} with {} active env(s).",
                    batch_index + 1,
                    len(batches),
                    active_count,
                )
                try:
                    summaries.extend(
                        _collect_batch(
                            env=env,
                            algo=algo,
                            motion_command=motion_command,
                            batch_index=batch_index,
                            batch_assignments=batch_assignments,
                            env_keys=env_keys,
                            export_cfg=export_cfg,
                            output_dir=output_dir,
                            policy_fn=policy_fn,
                        )
                    )
                except Exception:
                    logger.exception("Teacher rollout export failed in batch {}/{}.", batch_index + 1, len(batches))
                    raise
        finally:
            motion_command._fixed_clip_ids = original_fixed_clip_ids

        summary_rows = [asdict(summary) for summary in summaries]
        _write_csv(output_dir / "summary.csv", summary_rows)

        motion_bank_dir = output_dir / "motion_bank"
        if motion_bank_dir.is_dir():
            clip_object_map = {
                summary.clip_id: {
                    "object_name": summary.object_name,
                    "object_size": [
                        float(summary.primitive_extent_x),
                        float(summary.primitive_extent_y),
                        float(summary.primitive_extent_z),
                    ],
                    "object_urdf_path": summary.object_urdf_path,
                }
                for summary in summaries
            }
            (motion_bank_dir / "_clip_object_urdf_map.json").write_text(
                json.dumps({"clips": clip_object_map}, indent=2),
                encoding="utf-8",
            )

        success_ids = [summary.clip_id for summary in summaries if summary.success]
        failure_ids = [summary.clip_id for summary in summaries if not summary.success]
        _write_text_list(output_dir / "success_clips.txt", success_ids)
        _write_text_list(output_dir / "failure_clips.txt", failure_ids)

        summary_json = {
            "checkpoint": str(checkpoint_cfg.checkpoint),
            "saved_wandb_path": saved_wandb_path,
            "num_clips": len(summaries),
            "num_success": len(success_ids),
            "num_failure": len(failure_ids),
            "num_batches": len(batches),
            "num_envs": int(env.num_envs),
            "export_config": asdict(export_cfg),
        }
        (output_dir / "summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

        logger.info(
            "Teacher contact export finished: {} success / {} failure. Outputs saved to {}",
            len(success_ids),
            len(failure_ids),
            output_dir,
        )
        return output_dir
    finally:
        if simulation_app:
            close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    os.environ.setdefault("HOLOSOMA_DISABLE_AUTO_RESET", "1")
    os.environ.setdefault("HOLOSOMA_DISABLE_CLIP_END_RESET", "1")

    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    export_cfg, remaining_args = tyro.cli(
        ExportConfig,
        args=remaining_args,
        return_unknown_args=True,
        add_help=False,
    )
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config()
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Teacher contact rollout export config overrides.",
        config=TYRO_CONIFG,
    )
    run_export_with_tyro(overwritten_tyro_config, checkpoint_cfg, export_cfg, saved_cfg, saved_wandb_path)


if __name__ == "__main__":
    main()
