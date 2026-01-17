from __future__ import annotations

import os
import sys
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro
import trimesh

# Ensure local packages are importable when running from source.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[3]
VISER_SRC = REPO_ROOT / "viser" / "src"
if VISER_SRC.exists() and str(VISER_SRC) not in sys.path:
    sys.path.insert(0, str(VISER_SRC))

import viser  # type: ignore[import-not-found]  # noqa: E402
from viser.extras import ViserUrdf  # type: ignore[import-not-found]  # noqa: E402

from holosoma.config_types.command import MotionConfig  # noqa: E402
from holosoma.config_types.experiment import ExperimentConfig  # noqa: E402
from holosoma.config_types.robot import RobotConfig  # noqa: E402
from holosoma.config_types.terrain import MeshType  # noqa: E402
from holosoma.config_values.experiment import AnnotatedExperimentConfig  # noqa: E402
from holosoma.managers.command.terms.wbt import FAKE_BODY_NAME_ALIASES, MotionLoader  # noqa: E402
from holosoma.utils.camera_utils import resolve_camera_intrinsics  # noqa: E402
from holosoma.utils.module_utils import get_holosoma_root  # noqa: E402
from holosoma.utils.path import resolve_data_file_path  # noqa: E402
from holosoma.utils.rotations import (  # noqa: E402
    get_euler_xyz,
    matrix_to_quaternion,
    quat_apply,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_batched,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch  # noqa: E402
from holosoma.utils.tyro_utils import TYRO_CONIFG  # noqa: E402


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(cfg: ExperimentConfig) -> str:
    asset_root = _resolve_data_path(cfg.robot.asset.asset_root)
    urdf_path = os.path.join(asset_root, cfg.robot.asset.urdf_file)
    return _resolve_data_path(urdf_path)


_VIRIDIS_LUT = np.array(
    [
        (68, 1, 84),
        (69, 6, 90),
        (70, 12, 95),
        (71, 18, 101),
        (71, 24, 106),
        (72, 29, 111),
        (72, 34, 115),
        (71, 39, 119),
        (71, 44, 123),
        (70, 49, 126),
        (69, 54, 129),
        (67, 59, 131),
        (66, 64, 133),
        (64, 68, 135),
        (62, 73, 137),
        (60, 77, 138),
        (58, 83, 139),
        (56, 87, 140),
        (54, 91, 140),
        (52, 95, 141),
        (50, 99, 141),
        (48, 103, 141),
        (46, 107, 142),
        (45, 111, 142),
        (43, 115, 142),
        (42, 119, 142),
        (40, 122, 142),
        (39, 126, 142),
        (37, 130, 142),
        (36, 134, 141),
        (34, 137, 141),
        (33, 141, 140),
        (31, 146, 140),
        (31, 150, 139),
        (30, 153, 138),
        (30, 157, 136),
        (31, 161, 135),
        (32, 165, 133),
        (35, 168, 131),
        (38, 172, 129),
        (42, 176, 126),
        (47, 179, 123),
        (53, 183, 120),
        (59, 186, 117),
        (66, 190, 113),
        (73, 193, 109),
        (81, 196, 104),
        (89, 199, 100),
        (100, 203, 93),
        (109, 206, 88),
        (119, 208, 82),
        (129, 211, 76),
        (139, 213, 70),
        (149, 215, 63),
        (159, 217, 56),
        (170, 219, 50),
        (181, 221, 43),
        (191, 223, 36),
        (202, 224, 30),
        (212, 225, 26),
        (223, 227, 24),
        (233, 228, 25),
        (243, 229, 30),
        (253, 231, 36),
    ],
    dtype=np.uint8,
)


def _apply_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    lut = _VIRIDIS_LUT
    scaled = values * float(len(lut) - 1)
    idx0 = np.floor(scaled).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, len(lut) - 1)
    t = (scaled - idx0)[..., None]
    c0 = lut[idx0].astype(np.float32)
    c1 = lut[idx1].astype(np.float32)
    return ((1.0 - t) * c0 + t * c1).astype(np.uint8)


def _valid_depth_stats(depth: np.ndarray, near: float, far: float) -> tuple[float | None, float | None, int]:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    valid &= depth >= near
    valid &= depth < (far - 1.0e-6)
    if not np.any(valid):
        return None, None, 0
    depth_valid = depth[valid]
    return float(depth_valid.min()), float(depth_valid.max()), int(depth_valid.size)


def _depth_to_rgb(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth)
    valid &= depth >= near
    valid &= depth < (far - 1.0e-6)
    if not np.any(valid):
        return np.zeros(depth.shape + (3,), dtype=np.uint8)

    depth_clipped = np.clip(depth, near, far)
    depth_valid = depth_clipped[valid]
    min_d = float(depth_valid.min())
    max_d = float(depth_valid.max())
    denom = max(max_d - min_d, 1.0e-6)
    norm = (depth_clipped - min_d) / denom
    norm = np.where(valid, norm, 0.0)
    colored = _apply_colormap(norm)
    colored[~valid] = 0
    return colored


def _parse_vec3(text: str | None) -> tuple[float, float, float]:
    if not text:
        return (0.0, 0.0, 0.0)
    parts = text.split()
    parts += ["0.0"] * max(0, 3 - len(parts))
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _load_terrain_mesh(cfg: ExperimentConfig) -> trimesh.Trimesh | None:
    terrain_cfg = cfg.terrain.terrain_term
    mesh_type = terrain_cfg.mesh_type
    mesh_type_value = mesh_type.value if isinstance(mesh_type, MeshType) else str(mesh_type)
    if mesh_type_value != "load_obj":
        return None
    if not terrain_cfg.obj_file_path:
        return None

    terrain_path = _resolve_data_path(terrain_cfg.obj_file_path)
    base = trimesh.load(terrain_path, process=False)
    if isinstance(base, trimesh.Scene):
        base = base.dump(concatenate=True)
    if not isinstance(base, trimesh.Trimesh):
        raise ValueError(f"Loaded terrain is not a trimesh: {type(base)}")

    num_rows = max(1, int(terrain_cfg.num_rows * terrain_cfg.scale_factor))
    num_cols = max(1, int(terrain_cfg.num_cols * terrain_cfg.scale_factor))
    if num_rows * num_cols <= 1:
        return base

    gap = 1e-4
    stride = (base.bounds[1] - base.bounds[0]) + gap
    tiles = []
    for r in range(num_rows):
        for c in range(num_cols):
            tile = base.copy()
            tile.apply_translation([c * stride[0], r * stride[1], 0.0])
            tiles.append(tile)
    return trimesh.util.concatenate(tiles)


def _resolve_camera_resolution(cfg) -> tuple[int, int]:
    width = cfg.camera_width if cfg.camera_width is not None else cfg.grid_size
    height = cfg.camera_height if cfg.camera_height is not None else cfg.grid_size
    return int(width), int(height)


def _build_default_joint_pos(robot_config: RobotConfig) -> tuple[np.ndarray, dict[str, float]]:
    defaults = robot_config.init_state.default_joint_angles
    joint_pos = np.zeros(len(robot_config.dof_names), dtype=np.float32)
    joint_dict: dict[str, float] = {}
    for idx, name in enumerate(robot_config.dof_names):
        val = float(defaults.get(name, 0.0))
        joint_pos[idx] = val
        joint_dict[name] = val
    return joint_pos, joint_dict


def _get_motion_config(cfg: ExperimentConfig) -> MotionConfig:
    if cfg.command is None:
        raise ValueError("Experiment config has no command manager; motion replay requires motion_command.")
    setup_terms = cfg.command.setup_terms
    if "motion_command" not in setup_terms:
        raise ValueError("Command manager has no motion_command; use a WBT experiment config.")
    motion_params = setup_terms["motion_command"].params
    if "motion_config" not in motion_params:
        raise ValueError("motion_command is missing motion_config.")
    motion_cfg = motion_params["motion_config"]
    if isinstance(motion_cfg, MotionConfig):
        return motion_cfg
    return MotionConfig(**motion_cfg)


def _slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = q0.astype(np.float64, copy=False)
    q1 = q1.astype(np.float64, copy=False)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + u * (q1 - q0)
        return out / max(np.linalg.norm(out), 1e-8)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - u) * theta) * q0 + np.sin(u * theta) * q1) / sin_theta


def _split_qpos(
    qpos: np.ndarray,
    robot_dof: int,
    has_object: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    root_pos = qpos[0:3]
    root_quat = qpos[3:7]
    joints = qpos[7 : 7 + robot_dof]
    if has_object:
        obj_pos = qpos[-7:-4]
        obj_quat = qpos[-4:]
    else:
        obj_pos = None
        obj_quat = None
    return root_pos, root_quat, joints, obj_pos, obj_quat


def _build_transition_segment(
    start_qpos: np.ndarray,
    target_qpos: np.ndarray,
    *,
    robot_dof: int,
    has_object: bool,
    num_steps: int,
    prepend: bool,
) -> np.ndarray:
    if num_steps <= 1:
        return np.zeros((0, start_qpos.shape[0]), dtype=np.float32)

    alphas = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
    alphas = alphas[:-1] if prepend else alphas[1:]
    if alphas.size == 0:
        return np.zeros((0, start_qpos.shape[0]), dtype=np.float32)

    s_root_pos, s_root_quat, s_joints, s_obj_pos, s_obj_quat = _split_qpos(start_qpos, robot_dof, has_object)
    t_root_pos, t_root_quat, t_joints, t_obj_pos, t_obj_quat = _split_qpos(target_qpos, robot_dof, has_object)

    out = np.zeros((alphas.shape[0], start_qpos.shape[0]), dtype=np.float32)
    for i, alpha in enumerate(alphas):
        root_pos = (1.0 - alpha) * s_root_pos + alpha * t_root_pos
        root_quat = _slerp(s_root_quat, t_root_quat, float(alpha))
        joints = (1.0 - alpha) * s_joints + alpha * t_joints

        frame = np.concatenate([root_pos, root_quat, joints], axis=0)
        if has_object and s_obj_pos is not None and t_obj_pos is not None and s_obj_quat is not None and t_obj_quat is not None:
            obj_pos = (1.0 - alpha) * s_obj_pos + alpha * t_obj_pos
            obj_quat = _slerp(s_obj_quat, t_obj_quat, float(alpha))
            frame = np.concatenate([frame, obj_pos, obj_quat], axis=0)

        out[i] = frame.astype(np.float32, copy=False)
    return out


def _build_default_root_pose(
    robot_config: RobotConfig,
    motion_root_pos: np.ndarray,
    motion_root_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    init_rot = robot_config.init_state.rot
    if len(init_rot) == 4:
        init_root_quat = torch.tensor(init_rot, dtype=torch.float32).unsqueeze(0)
    else:
        init_root_quat = quat_from_euler_xyz(
            torch.tensor(init_rot[0]),
            torch.tensor(init_rot[1]),
            torch.tensor(init_rot[2]),
        ).unsqueeze(0)
    init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

    motion_root_quat = torch.tensor(motion_root_quat_xyzw, dtype=torch.float32).unsqueeze(0)
    _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

    default_root_pos = np.array(
        [motion_root_pos[0], motion_root_pos[1], robot_config.init_state.pos[2]],
        dtype=np.float32,
    )
    default_root_quat_xyzw = quat_from_euler_xyz(
        init_roll.squeeze(0),
        init_pitch.squeeze(0),
        motion_yaw.squeeze(0),
    ).cpu().numpy()
    default_root_quat_wxyz = default_root_quat_xyzw[[3, 0, 1, 2]]
    return default_root_pos, default_root_quat_wxyz


def _maybe_add_default_pose_transitions(
    qpos: np.ndarray,
    *,
    motion_cfg: MotionConfig,
    robot_config: RobotConfig,
    robot_dof: int,
    fps: float,
    default_joint_pos: np.ndarray,
) -> np.ndarray:
    if qpos.shape[0] == 0:
        return qpos

    has_object = qpos.shape[1] >= (7 + robot_dof + 7)
    if default_joint_pos.shape[0] != robot_dof:
        raise ValueError("default_joint_pos does not match robot_dof.")
    motion_root_pos_start = qpos[0, 0:3]
    motion_root_quat_start_wxyz = qpos[0, 3:7]
    motion_root_quat_start_xyzw = motion_root_quat_start_wxyz[[1, 2, 3, 0]]

    motion_root_pos_end = qpos[-1, 0:3]
    motion_root_quat_end_wxyz = qpos[-1, 3:7]
    motion_root_quat_end_xyzw = motion_root_quat_end_wxyz[[1, 2, 3, 0]]

    if motion_cfg.enable_default_pose_prepend and motion_cfg.default_pose_prepend_duration_s > 0.0:
        num_steps = int(round(motion_cfg.default_pose_prepend_duration_s * fps))
        default_root_pos, default_root_quat = _build_default_root_pose(
            robot_config,
            motion_root_pos_start,
            motion_root_quat_start_xyzw,
        )
        start = np.concatenate([default_root_pos, default_root_quat, default_joint_pos], axis=0)
        if has_object:
            start = np.concatenate([start, qpos[0, -7:]], axis=0)
        segment = _build_transition_segment(
            start,
            qpos[0],
            robot_dof=robot_dof,
            has_object=has_object,
            num_steps=num_steps,
            prepend=True,
        )
        if segment.shape[0] > 0:
            qpos = np.concatenate([segment, qpos], axis=0)

    if motion_cfg.enable_default_pose_append and motion_cfg.default_pose_append_duration_s > 0.0:
        num_steps = int(round(motion_cfg.default_pose_append_duration_s * fps))
        default_root_pos, default_root_quat = _build_default_root_pose(
            robot_config,
            motion_root_pos_end,
            motion_root_quat_end_xyzw,
        )
        target = np.concatenate([default_root_pos, default_root_quat, default_joint_pos], axis=0)
        if has_object:
            target = np.concatenate([target, qpos[-1, -7:]], axis=0)
        segment = _build_transition_segment(
            qpos[-1],
            target,
            robot_dof=robot_dof,
            has_object=has_object,
            num_steps=num_steps,
            prepend=False,
        )
        if segment.shape[0] > 0:
            qpos = np.concatenate([qpos, segment], axis=0)

    return qpos


def _load_motion_qpos(cfg: ExperimentConfig, robot_config: RobotConfig) -> tuple[np.ndarray, int]:
    motion_cfg = _get_motion_config(cfg)
    robot_body_names = [FAKE_BODY_NAME_ALIASES.get(name, name) for name in robot_config.body_names]
    motion = MotionLoader(
        motion_cfg.motion_file,
        robot_body_names,
        robot_config.dof_names,
        device="cpu",
        motion_clip_id=motion_cfg.motion_clip_id,
        motion_clip_name=motion_cfg.motion_clip_name,
    )

    joint_pos = motion.joint_pos.cpu().numpy()
    root_pos = motion.body_pos_w[:, 0].cpu().numpy()
    root_quat_xyzw = motion.body_quat_w[:, 0].cpu().numpy()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    parts = [root_pos, root_quat_wxyz, joint_pos]
    if motion.has_object:
        object_pos = motion.object_pos_w.cpu().numpy()
        object_quat_xyzw = motion.object_quat_w.cpu().numpy()
        object_quat_wxyz = object_quat_xyzw[:, [3, 0, 1, 2]]
        parts.extend([object_pos, object_quat_wxyz])

    qpos = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    fps = int(motion.fps) if hasattr(motion, "fps") else 30

    default_joint_pos, _ = _build_default_joint_pos(robot_config)
    qpos = _maybe_add_default_pose_transitions(
        qpos,
        motion_cfg=motion_cfg,
        robot_config=robot_config,
        robot_dof=len(robot_config.dof_names),
        fps=float(fps),
        default_joint_pos=default_joint_pos,
    )
    return qpos, fps


@dataclass(frozen=True)
class JointInfo:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


def _parse_urdf_joints(urdf_path: str) -> tuple[str, dict[str, JointInfo]]:
    root = ET.parse(urdf_path).getroot()
    links = {link.get("name") for link in root.findall("link") if link.get("name")}
    child_links: set[str] = set()
    child_to_joint: dict[str, JointInfo] = {}

    for joint in root.findall("joint"):
        joint_type = joint.get("type", "fixed")
        parent_node = joint.find("parent")
        child_node = joint.find("child")
        if parent_node is None or child_node is None:
            continue
        parent_link = parent_node.get("link")
        child_link = child_node.get("link")
        if not parent_link or not child_link:
            continue
        origin = joint.find("origin")
        xyz = _parse_vec3(origin.get("xyz") if origin is not None else None)
        rpy = _parse_vec3(origin.get("rpy") if origin is not None else None)
        axis_node = joint.find("axis")
        axis = _parse_vec3(axis_node.get("xyz") if axis_node is not None else None)

        child_links.add(child_link)
        child_to_joint[child_link] = JointInfo(
            name=joint.get("name", ""),
            joint_type=joint_type,
            parent=parent_link,
            child=child_link,
            origin_xyz=xyz,
            origin_rpy=rpy,
            axis=axis,
        )

    roots = sorted(link for link in links if link not in child_links)
    if not roots:
        raise ValueError(f"No root link found in URDF: {urdf_path}")
    return roots[0], child_to_joint


def _resolve_link_from_joint_name(urdf_path: str, link_or_joint: str) -> str:
    _, child_to_joint = _parse_urdf_joints(urdf_path)
    for child_link, joint_info in child_to_joint.items():
        if joint_info.name == link_or_joint:
            return child_link
    return link_or_joint


def _get_link_pose_base(
    vr: ViserUrdf,
    link_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if link_name not in vr._urdf.link_map:  # type: ignore[attr-defined]
        raise ValueError(f"Link '{link_name}' not found in URDF.")

    base_link = vr._urdf.base_link  # type: ignore[attr-defined]
    tf_base = vr._urdf.get_transform(  # type: ignore[attr-defined]
        link_name,
        base_link,
        collision_geometry=not vr._load_meshes,  # type: ignore[attr-defined]
    )
    pos_base = torch.tensor(tf_base[:3, 3], dtype=torch.float32)
    rot_base = torch.tensor(tf_base[:3, :3], dtype=torch.float32)
    quat_base_wxyz = matrix_to_quaternion(rot_base)
    quat_base_xyzw = quat_base_wxyz[[1, 2, 3, 0]]
    return pos_base, quat_base_xyzw


def _build_camera_rays(
    width: int,
    height: int,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    pitch_deg: float,
) -> torch.Tensor:
    u_coords = torch.arange(width, dtype=torch.float32)
    v_coords = torch.arange(height, dtype=torch.float32)
    v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")

    x = (u_grid - cx) / fx
    y = (v_grid - cy) / fy

    dirs_cam = torch.stack((torch.ones_like(x), -x, y), dim=-1)
    dirs_cam = dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True).clamp(min=1.0e-6)
    dirs_cam = dirs_cam.view(-1, 3)

    pitch_rad = torch.deg2rad(torch.tensor(pitch_deg))
    pitch_quat = quat_from_euler_xyz(torch.tensor(0.0), pitch_rad, torch.tensor(0.0))
    pitch_quat = pitch_quat.unsqueeze(0).expand(dirs_cam.shape[0], -1)
    dirs_base = quat_rotate_inverse(pitch_quat, dirs_cam, w_last=True)
    return dirs_base / torch.norm(dirs_base, dim=-1, keepdim=True).clamp(min=1.0e-6)


def _raycast_depth(
    mesh: trimesh.Trimesh | None,
    ray_origins: np.ndarray,
    ray_dirs: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    depth = np.full(ray_origins.shape[0], max_distance, dtype=np.float32)
    if mesh is None:
        return depth

    try:
        intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    except Exception:
        intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)

    locations, index_ray, _ = intersector.intersects_location(ray_origins, ray_dirs, multiple_hits=False)
    if len(index_ray) == 0:
        return depth

    delta = locations - ray_origins[index_ray]
    distances = np.einsum("ij,ij->i", delta, ray_dirs[index_ray])
    valid = distances > 0.0
    if np.any(valid):
        depth[index_ray[valid]] = distances[valid]
    return np.clip(depth, 0.0, max_distance)


def _normalize_vec(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp(min=1.0e-6)


def _frustum_quat_from_camera(cam_quat_xyzw: torch.Tensor) -> torch.Tensor:
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    x_cam = quat_apply(cam_quat_xyzw.unsqueeze(0), x_axis.unsqueeze(0), w_last=True).squeeze(0)
    y_cam = quat_apply(cam_quat_xyzw.unsqueeze(0), y_axis.unsqueeze(0), w_last=True).squeeze(0)
    z_cam = quat_apply(cam_quat_xyzw.unsqueeze(0), z_axis.unsqueeze(0), w_last=True).squeeze(0)

    z_fwd = _normalize_vec(x_cam)
    y_down = _normalize_vec(-z_cam)
    x_right = _normalize_vec(torch.cross(y_down, z_fwd))
    y_down = _normalize_vec(torch.cross(z_fwd, x_right))

    rot = torch.stack([x_right, y_down, z_fwd], dim=1)
    quat_wxyz = matrix_to_quaternion(rot)
    return quat_wxyz


def replay_perception(cfg: ExperimentConfig) -> None:
    if cfg.perception is None or not cfg.perception.enabled or cfg.perception.output_mode != "camera_depth":
        raise RuntimeError("Perception camera_depth is required for viser_perception.")

    width, height = _resolve_camera_resolution(cfg.perception)
    fx, fy, cx, cy, _, _ = resolve_camera_intrinsics(
        width,
        height,
        vfov_deg=cfg.perception.camera_vfov_deg,
        hfov_deg=cfg.perception.camera_hfov_deg,
        fx=cfg.perception.camera_fx,
        fy=cfg.perception.camera_fy,
        cx=cfg.perception.camera_cx,
        cy=cfg.perception.camera_cy,
    )

    camera_rays_base = _build_camera_rays(
        width,
        height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        pitch_deg=cfg.perception.camera_pitch_deg,
    )

    server = viser.ViserServer(port=int(os.environ.get("HOLOSOMA_VISER_PORT", "6060")))
    robot_root = server.scene.add_frame("/robot", show_axes=False)
    robot_urdf_path = _resolve_robot_urdf_path(cfg)
    vr = ViserUrdf(server, urdf_or_path=Path(robot_urdf_path), root_node_name="/robot")

    camera_frame = server.scene.add_frame(
        "/robot/d435i",
        show_axes=True,
        axes_length=0.12,
        axes_radius=0.006,
        origin_color=(0, 200, 255),
    )
    camera_marker_mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.03)
    camera_marker_mesh.visual.face_colors = np.tile(
        np.array([255, 0, 0, 255], dtype=np.uint8),
        (len(camera_marker_mesh.faces), 1),
    )
    camera_marker = server.scene.add_mesh_trimesh(
        "/robot/d435_marker",
        camera_marker_mesh,
        cast_shadow=False,
        receive_shadow=False,
    )
    camera_frustum = server.scene.add_camera_frustum(
        "/robot/d435_frustum",
        fov=float(np.deg2rad(cfg.perception.camera_vfov_deg)),
        aspect=float(width) / float(height),
        scale=0.2,
        line_width=2.5,
        color=(255, 230, 0),
        variant="wireframe",
    )
    link_marker_specs = [
        ("pelvis", "pelvis", "pelvis", (255, 80, 80, 255)),
        ("waist_yaw_link", "waist_yaw_link", "waist_yaw_link", (80, 255, 80, 255)),
        ("waist_roll_link", "waist_roll_link", "waist_roll_link", (80, 120, 255, 255)),
        ("torso_link", "torso_link", "torso_link", (255, 180, 80, 255)),
        ("mid360_joint", "mid360_link", "mid360_joint", (200, 80, 255, 255)),
    ]
    link_markers: dict[str, tuple[str, viser.GlbHandle, viser.LabelHandle]] = {}
    for name, link, label_text, color in link_marker_specs:
        marker_mesh = trimesh.creation.icosphere(subdivisions=2, radius=0.025)
        marker_mesh.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(marker_mesh.faces), 1))
        marker_handle = server.scene.add_mesh_trimesh(
            f"/robot/marker_{name}",
            marker_mesh,
            cast_shadow=False,
            receive_shadow=False,
        )
        label_handle = server.scene.add_label(
            f"/robot/label_{name}",
            text=label_text,
            font_size_mode="screen",
            font_screen_scale=1.0,
            depth_test=False,
            anchor="bottom-center",
        )
        link_markers[name] = (link, marker_handle, label_handle)

    server.scene.add_grid("/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0))

    terrain_mesh = _load_terrain_mesh(cfg)
    terrain_handle = None
    if terrain_mesh is not None:
        terrain_handle = server.scene.add_mesh_trimesh("/terrain", terrain_mesh)

    viser_joint_names = list(vr.get_actuated_joint_names())
    joint_pos, joint_dict = _build_default_joint_pos(cfg.robot)
    name_to_robot_idx = {name: idx for idx, name in enumerate(cfg.robot.dof_names)}
    missing = [name for name in viser_joint_names if name not in name_to_robot_idx]
    if missing:
        raise ValueError(f"Viser URDF joints missing in robot config: {missing}")

    viser_joint_indices = [name_to_robot_idx[name] for name in viser_joint_names]

    qpos, fps = _load_motion_qpos(cfg, cfg.robot)
    if qpos.shape[0] == 0:
        raise ValueError("Motion file is empty; cannot visualize depth.")

    robot_dof = len(cfg.robot.dof_names)
    has_object = qpos.shape[1] >= (7 + robot_dof + 7)
    camera_body = cfg.perception.camera_body_name or vr._urdf.base_link  # type: ignore[attr-defined]
    camera_body = _resolve_link_from_joint_name(robot_urdf_path, camera_body)

    depth_handle = server.gui.add_image(
        np.zeros((height, width, 3), dtype=np.uint8),
        label="D435i Depth",
    )
    depth_stats = server.gui.add_markdown("Depth range (valid): n/a")

    with server.gui.add_folder("Display"):
        show_meshes_cb = server.gui.add_checkbox("Show meshes", initial_value=True)
        show_terrain_cb = server.gui.add_checkbox("Show terrain", initial_value=terrain_handle is not None)
        show_marker_cb = server.gui.add_checkbox("Show D435 marker", initial_value=True)
        show_frustum_cb = server.gui.add_checkbox("Show D435 frustum", initial_value=True)
        show_link_markers_cb = server.gui.add_checkbox("Show torso chain markers", initial_value=True)

    @show_meshes_cb.on_update
    def _(_evt) -> None:
        vr.show_visual = bool(show_meshes_cb.value)

    @show_terrain_cb.on_update
    def _(_evt) -> None:
        if terrain_handle is not None:
            terrain_handle.visible = bool(show_terrain_cb.value)

    @show_marker_cb.on_update
    def _(_evt) -> None:
        camera_marker.visible = bool(show_marker_cb.value)

    @show_frustum_cb.on_update
    def _(_evt) -> None:
        camera_frustum.visible = bool(show_frustum_cb.value)

    @show_link_markers_cb.on_update
    def _(_evt) -> None:
        visible = bool(show_link_markers_cb.value)
        for _, marker_handle, label_handle in link_markers.values():
            marker_handle.visible = visible
            label_handle.visible = visible

    def _interp_qpos(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
        out = q0.copy()
        out[0:3] = (1.0 - u) * q0[0:3] + u * q1[0:3]
        out[3:7] = _slerp(q0[3:7], q1[3:7], u)
        out[7 : 7 + robot_dof] = (1.0 - u) * q0[7 : 7 + robot_dof] + u * q1[7 : 7 + robot_dof]
        if has_object and q0.shape[0] >= (7 + robot_dof + 7):
            out[-7:-4] = (1.0 - u) * q0[-7:-4] + u * q1[-7:-4]
            out[-4:] = _slerp(q0[-4:], q1[-4:], u)
        return out

    def _apply_qpos(frame: np.ndarray) -> None:
        root_pos = frame[0:3]
        root_quat_wxyz = frame[3:7]
        joints = frame[7 : 7 + robot_dof]

        robot_root.position = root_pos
        robot_root.wxyz = root_quat_wxyz

        viser_joints = joints[viser_joint_indices]
        vr.update_cfg(viser_joints.astype(np.float32, copy=False))

        root_quat_xyzw = torch.tensor(root_quat_wxyz[[1, 2, 3, 0]], dtype=torch.float32)
        root_pos_t = torch.tensor(root_pos, dtype=torch.float32)

        body_pos_base, body_quat_base = _get_link_pose_base(vr, camera_body)
        sensor_offset = torch.tensor(cfg.perception.sensor_offset, dtype=torch.float32)
        offset_base = quat_apply(body_quat_base.unsqueeze(0), sensor_offset.unsqueeze(0), w_last=True).squeeze(0)
        cam_pos_base = body_pos_base + offset_base

        pitch_rad = torch.deg2rad(torch.tensor(cfg.perception.camera_pitch_deg, dtype=torch.float32))
        pitch_quat = quat_from_euler_xyz(torch.tensor(0.0), pitch_rad, torch.tensor(0.0))
        cam_quat_base = quat_mul(body_quat_base.unsqueeze(0), pitch_quat.unsqueeze(0), w_last=True).squeeze(0)

        cam_pos_base_np = cam_pos_base.detach().cpu().numpy()
        cam_quat_base_np = cam_quat_base.detach().cpu().numpy()
        cam_quat_base_wxyz = cam_quat_base_np[[3, 0, 1, 2]]
        camera_frame.position = cam_pos_base_np
        camera_frame.wxyz = cam_quat_base_wxyz
        camera_marker.position = cam_pos_base_np

        frustum_quat_wxyz = _frustum_quat_from_camera(cam_quat_base)
        frustum_quat_np = frustum_quat_wxyz.detach().cpu().numpy()
        camera_frustum.position = cam_pos_base_np
        camera_frustum.wxyz = frustum_quat_np

        label_offset = np.array([0.0, 0.0, 0.06], dtype=np.float32)
        for _, (link_name, marker_handle, label_handle) in link_markers.items():
            link_pos_base, _ = _get_link_pose_base(vr, link_name)
            link_pos_np = link_pos_base.detach().cpu().numpy()
            marker_handle.position = link_pos_np
            label_handle.position = link_pos_np + label_offset

        body_pos_world = root_pos_t + quat_apply(root_quat_xyzw.unsqueeze(0), body_pos_base.unsqueeze(0), w_last=True).squeeze(0)
        body_quat_world = quat_mul(root_quat_xyzw.unsqueeze(0), body_quat_base.unsqueeze(0), w_last=True).squeeze(0)
        offset_world = quat_apply(body_quat_world.unsqueeze(0), sensor_offset.unsqueeze(0), w_last=True).squeeze(0)
        cam_pos_world = body_pos_world + offset_world

        ray_dirs_world = quat_rotate_batched(body_quat_world.unsqueeze(0), camera_rays_base.unsqueeze(0)).view(-1, 3)
        ray_starts = cam_pos_world.unsqueeze(0).expand(ray_dirs_world.shape[0], -1)
        depth = _raycast_depth(
            terrain_mesh,
            ray_starts.detach().cpu().numpy(),
            ray_dirs_world.detach().cpu().numpy(),
            max_distance=cfg.perception.max_distance,
        )
        depth_map = depth.reshape(height, width)
        depth_map = np.flipud(depth_map)
        depth_img = _depth_to_rgb(depth_map, cfg.perception.camera_near, cfg.perception.max_distance)
        depth_handle.image = depth_img
        camera_frustum.image = depth_img
        min_d, max_d, count = _valid_depth_stats(depth_map, cfg.perception.camera_near, cfg.perception.max_distance)
        if count == 0:
            depth_stats.content = "Depth range (valid): n/a (no hits)"
        else:
            total = depth_map.size
            depth_stats.content = (
                f"Depth range (valid): {min_d:.3f} - {max_d:.3f} m | "
                f"valid: {count}/{total}"
            )

    with server.gui.add_folder("Playback"):
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(0, int(qpos.shape[0] - 1)),
            step=1,
            initial_value=0,
        )
        play_btn = server.gui.add_button("Play / Pause")
        fps_in = server.gui.add_number("FPS", initial_value=int(fps), min=1, max=240, step=1)
        interp_mult_in = server.gui.add_number("Visual FPS multiplier", initial_value=1, min=1, max=8, step=1)
        loop_cb = server.gui.add_checkbox("Loop", initial_value=True)

    playing = {"flag": False}
    frame_f = {"value": float(frame_slider.value)}
    updating_programmatically = {"flag": False}

    def _apply_frame_from_float(f_val: float) -> None:
        i0 = int(np.clip(np.floor(f_val), 0, qpos.shape[0] - 1))
        i1 = min(i0 + 1, qpos.shape[0] - 1)
        u = float(f_val - i0)
        if i0 == i1 or u <= 1.0e-6:
            _apply_qpos(qpos[i0])
        else:
            _apply_qpos(_interp_qpos(qpos[i0], qpos[i1], u))

    @play_btn.on_click
    def _(_evt) -> None:
        playing["flag"] = not playing["flag"]

    @frame_slider.on_update
    def _(_evt) -> None:
        if not updating_programmatically["flag"]:
            playing["flag"] = False
            frame_f["value"] = float(frame_slider.value)
            _apply_frame_from_float(frame_f["value"])

    def _player_loop() -> None:
        next_tick = time.perf_counter()
        while True:
            if playing["flag"]:
                now = time.perf_counter()
                fps_val = max(1, int(fps_in.value))
                mult = max(1, int(interp_mult_in.value))
                dt = 1.0 / (fps_val * mult)
                if now >= next_tick:
                    next_tick = now + dt
                    frame_f["value"] += 1.0 / float(mult)
                    if frame_f["value"] >= qpos.shape[0] - 1:
                        if loop_cb.value:
                            frame_f["value"] = 0.0
                        else:
                            frame_f["value"] = float(qpos.shape[0] - 1)
                            playing["flag"] = False
                    updating_programmatically["flag"] = True
                    frame_slider.value = int(frame_f["value"])
                    updating_programmatically["flag"] = False
                    _apply_frame_from_float(frame_f["value"])
            time.sleep(0.001)

    _apply_frame_from_float(frame_f["value"])
    threading.Thread(target=_player_loop, daemon=True).start()

    print("Open the viewer URL printed above. Close the process (Ctrl+C) to exit.")
    while True:
        time.sleep(1.0)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay_perception(tyro_cfg)


if __name__ == "__main__":
    main()
