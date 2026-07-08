"""Far-tracking-style Warp depth camera for dynamic robot self-occlusion."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import torch
import trimesh
import warp as wp
from loguru import logger

from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path

wp.config.quiet = True
wp.init()


FAR_TRACKING_G1_RAYCAST_BODY_MESHES: dict[str, str] = {
    "pelvis": "combined_pelvis.STL",
    "left_hip_pitch_link": "left_hip_pitch_link.STL",
    "left_hip_roll_link": "left_hip_roll_link.STL",
    "left_hip_yaw_link": "left_hip_yaw_link.STL",
    "left_knee_link": "left_knee_link.STL",
    "left_ankle_pitch_link": "left_ankle_pitch_link.STL",
    "left_ankle_roll_link": "left_ankle_roll_link.STL",
    "right_hip_pitch_link": "right_hip_pitch_link.STL",
    "right_hip_roll_link": "right_hip_roll_link.STL",
    "right_hip_yaw_link": "right_hip_yaw_link.STL",
    "right_knee_link": "right_knee_link.STL",
    "right_ankle_pitch_link": "right_ankle_pitch_link.STL",
    "right_ankle_roll_link": "right_ankle_roll_link.STL",
    "waist_yaw_link": "waist_yaw_link_rev_1_0.STL",
    "waist_roll_link": "waist_roll_link_rev_1_0.STL",
    "left_shoulder_pitch_link": "left_shoulder_pitch_link.STL",
    "left_shoulder_roll_link": "left_shoulder_roll_link.STL",
    "left_shoulder_yaw_link": "left_shoulder_yaw_link.STL",
    "left_elbow_link": "left_elbow_link.STL",
    "left_wrist_roll_link": "left_wrist_roll_link.STL",
    "left_wrist_pitch_link": "left_wrist_pitch_link.STL",
    "left_wrist_yaw_link": "combined_left_wrist_spherehand.STL",
    "right_shoulder_pitch_link": "right_shoulder_pitch_link.STL",
    "right_shoulder_roll_link": "right_shoulder_roll_link.STL",
    "right_shoulder_yaw_link": "right_shoulder_yaw_link.STL",
    "right_elbow_link": "right_elbow_link.STL",
    "right_wrist_roll_link": "right_wrist_roll_link.STL",
    "right_wrist_pitch_link": "right_wrist_pitch_link.STL",
    "right_wrist_yaw_link": "combined_right_wrist_spherehand.STL",
}

_MESH_FALLBACKS: dict[str, tuple[str, ...]] = {
    "combined_pelvis.STL": ("pelvis.STL",),
    "combined_left_wrist_spherehand.STL": (
        "left_wrist_yaw_link.STL",
        "left_wrist_roll_rubber_hand.STL",
    ),
    "combined_right_wrist_spherehand.STL": (
        "right_wrist_yaw_link.STL",
        "right_wrist_roll_rubber_hand.STL",
    ),
}

_NO_HIT_RAY_VAL = 1.0e6


@wp.kernel
def _draw_depth_dynamic_kernel(
    terrain_id: wp.uint64,
    robot_ids: wp.array(dtype=wp.uint64),
    object_ids: wp.array(dtype=wp.uint64),
    object_mesh_indices: wp.array(dtype=wp.int32),
    body_poss: wp.array(dtype=wp.vec3, ndim=2),
    body_quats: wp.array(dtype=wp.quat, ndim=2),
    object_poss: wp.array(dtype=wp.vec3),
    object_quats: wp.array(dtype=wp.quat),
    cam_poss: wp.array(dtype=wp.vec3),
    cam_quats: wp.array(dtype=wp.quat),
    k_inv: wp.mat44,
    far_plane: float,
    pixels: wp.array(dtype=wp.float32, ndim=3),
    ray_hits: wp.array(dtype=wp.vec3, ndim=3),
    c_x: int,
    c_y: int,
    num_bodies: int,
    num_objects: int,
):
    env_id, x, y = wp.tid()

    cam_pos = cam_poss[env_id]
    cam_quat = cam_quats[env_id]

    uv = wp.transform_vector(k_inv, wp.vec3(float(x), float(y), 1.0))
    uv_c = wp.transform_vector(k_inv, wp.vec3(float(c_x), float(c_y), 1.0))

    ro = cam_pos
    rd = wp.normalize(wp.quat_rotate(cam_quat, uv))
    rd_c = wp.normalize(wp.quat_rotate(cam_quat, uv_c))

    multiplier = wp.float32(wp.dot(rd, rd_c))
    multiplier = wp.max(multiplier, wp.float32(1.0e-6))
    multiplier = wp.min(multiplier, wp.float32(1.0))

    best = wp.float32(_NO_HIT_RAY_VAL)
    far_bound_world = wp.float32(far_plane)
    best_hit = wp.vec3(_NO_HIT_RAY_VAL, _NO_HIT_RAY_VAL, _NO_HIT_RAY_VAL)

    for body_idx in range(num_bodies):
        body_quat = body_quats[env_id, body_idx]
        body_pos = body_poss[env_id, body_idx]
        ro_l = wp.quat_rotate_inv(body_quat, ro - body_pos)
        rd_l = wp.quat_rotate_inv(body_quat, rd)

        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        face = int(0)
        if wp.mesh_query_ray(robot_ids[body_idx], ro_l, rd_l, far_bound_world / multiplier, t, u, v, sign, n, face):
            depth = multiplier * t
            if (best == _NO_HIT_RAY_VAL) or (depth < best):
                best = depth
                far_bound_world = depth
                best_hit = ro + rd * t

    object_idx = object_mesh_indices[env_id]
    if object_idx >= 0 and object_idx < num_objects:
        object_quat = object_quats[env_id]
        object_pos = object_poss[env_id]
        ro_l = wp.quat_rotate_inv(object_quat, ro - object_pos)
        rd_l = wp.quat_rotate_inv(object_quat, rd)

        t = float(0.0)
        u = float(0.0)
        v = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        face = int(0)
        if wp.mesh_query_ray(object_ids[object_idx], ro_l, rd_l, far_bound_world / multiplier, t, u, v, sign, n, face):
            depth = multiplier * t
            if (best == _NO_HIT_RAY_VAL) or (depth < best):
                best = depth
                far_bound_world = depth
                best_hit = ro + rd * t

    t = float(0.0)
    u = float(0.0)
    v = float(0.0)
    sign = float(0.0)
    n = wp.vec3()
    face = int(0)
    if wp.mesh_query_ray(terrain_id, ro, rd, far_bound_world / multiplier, t, u, v, sign, n, face):
        best = multiplier * t
        best_hit = ro + rd * t

    pixels[env_id, y, x] = best
    ray_hits[env_id, y, x] = best_hit


def _quat_from_rpy_deg_torch(rpy_deg: torch.Tensor) -> torch.Tensor:
    rpy = torch.deg2rad(rpy_deg)
    roll, pitch, yaw = rpy.unbind(dim=-1)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    qw = cy * cr * cp + sy * sr * sp
    return torch.stack((qx, qy, qz, qw), dim=-1)


def _quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = a.unbind(dim=-1)
    x2, y2, z2, w2 = b.unbind(dim=-1)
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _quat_apply_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., :3]
    q_w = q[..., 3:4]
    t = torch.cross(q_xyz, v, dim=-1) * 2.0
    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)


def _resolve_holosoma_path(path: str) -> Path:
    if path.startswith("@holosoma/"):
        return Path(get_holosoma_root()) / path[len("@holosoma/") :]
    return Path(resolve_data_file_path(path))


def _to_trimesh(mesh: trimesh.Trimesh | trimesh.Scene) -> trimesh.Trimesh:
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a trimesh mesh, got {type(mesh)}")
    return mesh


def _parse_float_list(value: str | None, default: tuple[float, ...]) -> list[float]:
    if not value:
        return list(default)
    parts = [part for part in value.replace(",", " ").split() if part]
    if not parts:
        return list(default)
    return [float(part) for part in parts]


def _origin_transform(origin_elem: ET.Element | None) -> np.ndarray:
    xyz = _parse_float_list(origin_elem.get("xyz") if origin_elem is not None else None, (0.0, 0.0, 0.0))
    rpy = _parse_float_list(origin_elem.get("rpy") if origin_elem is not None else None, (0.0, 0.0, 0.0))
    transform = trimesh.transformations.euler_matrix(rpy[0], rpy[1], rpy[2], axes="sxyz")
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _resolve_urdf_mesh_path(filename: str, urdf_path: Path) -> Path:
    filename = filename.strip()
    if filename.startswith("file://"):
        filename = filename[len("file://") :]
    if filename.startswith("package://"):
        filename = filename[len("package://") :]
    path = Path(filename)
    if path.is_absolute():
        return path
    local_path = urdf_path.parent / path
    if local_path.exists():
        return local_path
    return _resolve_holosoma_path(filename)


def _geometry_to_mesh(geometry_elem: ET.Element, urdf_path: Path) -> trimesh.Trimesh | None:
    mesh_elem = geometry_elem.find("mesh")
    if mesh_elem is not None:
        filename = mesh_elem.get("filename")
        if not filename:
            return None
        mesh_path = _resolve_urdf_mesh_path(filename, urdf_path)
        mesh = _to_trimesh(trimesh.load(mesh_path, process=False)).copy()
        scale = _parse_float_list(mesh_elem.get("scale"), (1.0, 1.0, 1.0))
        if len(scale) == 1:
            scale = [scale[0], scale[0], scale[0]]
        mesh.vertices *= np.asarray(scale[:3], dtype=np.float64)
        return mesh

    box_elem = geometry_elem.find("box")
    if box_elem is not None:
        size = _parse_float_list(box_elem.get("size"), (1.0, 1.0, 1.0))
        return trimesh.creation.box(extents=size[:3])

    sphere_elem = geometry_elem.find("sphere")
    if sphere_elem is not None:
        radius = float(sphere_elem.get("radius", "0.5"))
        return trimesh.creation.icosphere(subdivisions=2, radius=radius)

    cylinder_elem = geometry_elem.find("cylinder")
    if cylinder_elem is not None:
        radius = float(cylinder_elem.get("radius", "0.5"))
        length = float(cylinder_elem.get("length", "1.0"))
        return trimesh.creation.cylinder(radius=radius, height=length, sections=32)

    return None


def _load_urdf_raycast_mesh(urdf_path: Path) -> trimesh.Trimesh:
    root = ET.parse(urdf_path).getroot()
    elements = root.findall(".//collision")
    if not elements:
        elements = root.findall(".//visual")

    meshes: list[trimesh.Trimesh] = []
    for elem in elements:
        geometry_elem = elem.find("geometry")
        if geometry_elem is None:
            continue
        mesh = _geometry_to_mesh(geometry_elem, urdf_path)
        if mesh is None:
            continue
        mesh.apply_transform(_origin_transform(elem.find("origin")))
        meshes.append(mesh)

    if not meshes:
        raise RuntimeError(f"No collision or visual mesh geometry found in object URDF: {urdf_path}")
    return _to_trimesh(trimesh.util.concatenate(meshes))


def _convert_to_warp_mesh(mesh: trimesh.Trimesh, device: str) -> wp.Mesh:
    return wp.Mesh(
        points=wp.array(np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device=device),
        indices=wp.array(np.asarray(mesh.faces, dtype=np.int32).flatten(), dtype=wp.int32, device=device),
    )


class FarTrackingWarpDepthCamera:
    """Warp camera matching far-tracking's dynamic ZED depth rendering path."""

    def __init__(self, env: Any, depth_cfg: Any):
        self.env = env
        self.depth_cfg = depth_cfg
        self.device = str(env.device)
        self.num_envs = int(env.num_envs)
        self.width = int(depth_cfg.width)
        self.height = int(depth_cfg.height)
        self.max_range = float(depth_cfg.max_range)
        self.body_name = self._resolve_camera_body_name(env, depth_cfg)

        self.body_names, self.body_indices, mesh_paths = self._resolve_body_meshes(env, depth_cfg)
        if not self.body_names:
            raise RuntimeError("Depth camera self-occlusion is enabled, but no robot body meshes were resolved.")

        self.terrain_mesh = self._build_terrain_mesh(env)
        self.robot_meshes = [_convert_to_warp_mesh(_to_trimesh(trimesh.load(path)), self.device) for path in mesh_paths]
        self.robot_mesh_ids = wp.array([mesh.id for mesh in self.robot_meshes], dtype=wp.uint64, device=self.device)
        self.object_meshes, object_mesh_indices = self._resolve_object_meshes(env)
        self.object_mesh_ids = wp.array([mesh.id for mesh in self.object_meshes], dtype=wp.uint64, device=self.device)
        self.terrain_mesh_wp = _convert_to_warp_mesh(self.terrain_mesh, self.device)
        self.terrain_mesh_id = self.terrain_mesh_wp.id

        self.depth = torch.empty((self.num_envs, self.height, self.width), device=env.device, dtype=torch.float32)
        self.ray_hits_w = torch.empty((self.num_envs, self.height, self.width, 3), device=env.device, dtype=torch.float32)
        self.camera_position = torch.empty((self.num_envs, 3), device=env.device, dtype=torch.float32)
        self.camera_orientation = torch.empty((self.num_envs, 4), device=env.device, dtype=torch.float32)
        self.object_mesh_indices = torch.tensor(object_mesh_indices, device=env.device, dtype=torch.int32)
        self.object_positions = torch.zeros((self.num_envs, 3), device=env.device, dtype=torch.float32)
        self.object_quats = torch.zeros((self.num_envs, 4), device=env.device, dtype=torch.float32)
        self.object_quats[:, 3] = 1.0
        self.body_positions = torch.empty(
            (self.num_envs, len(self.body_indices), 3), device=env.device, dtype=torch.float32
        )
        self.body_quats = torch.empty(
            (self.num_envs, len(self.body_indices), 4), device=env.device, dtype=torch.float32
        )

        self.camera_position_wp = wp.from_torch(self.camera_position, dtype=wp.vec3)
        self.camera_orientation_wp = wp.from_torch(self.camera_orientation, dtype=wp.quat)
        self.object_mesh_indices_wp = wp.from_torch(self.object_mesh_indices, dtype=wp.int32)
        self.object_positions_wp = wp.from_torch(self.object_positions, dtype=wp.vec3)
        self.object_quats_wp = wp.from_torch(self.object_quats, dtype=wp.quat)
        self.body_positions_wp = wp.from_torch(self.body_positions, dtype=wp.vec3)
        self.body_quats_wp = wp.from_torch(self.body_quats, dtype=wp.quat)
        self.depth_wp = wp.from_torch(self.depth, dtype=wp.float32)
        self.ray_hits_wp = wp.from_torch(self.ray_hits_w.view(self.num_envs, self.height, self.width, 3), dtype=wp.vec3)

        self.k_inv = self._make_k_inv()
        self.c_x = int(self.width / 2)
        self.c_y = int(self.height / 2)
        self.local_position, self.local_orientation, self.data_frame_quat = self._sample_local_camera_offsets(depth_cfg)

        logger.info(
            "Initialized far-tracking-style Warp depth camera: envs={} raw={}x{} bodies={} randomize_placement={}",
            self.num_envs,
            self.width,
            self.height,
            len(self.body_indices),
            bool(depth_cfg.randomize_placement),
        )
        if self.object_meshes:
            logger.info(
                "Warp depth camera will raycast {} dynamic object mesh(es) across {} envs.",
                len(self.object_meshes),
                int((self.object_mesh_indices >= 0).sum().item()),
            )

    def capture(self) -> torch.Tensor:
        sim = self.env.simulator
        base_idx = sim.body_names.index(self.body_name)
        base_pos = sim._rigid_body_pos[:, base_idx].to(dtype=torch.float32)
        base_quat = sim._rigid_body_rot[:, base_idx].to(dtype=torch.float32)

        self.camera_position.copy_(_quat_apply_xyzw(base_quat, self.local_position) + base_pos)
        self.camera_orientation.copy_(
            _quat_mul_xyzw(base_quat, _quat_mul_xyzw(self.local_orientation, self.data_frame_quat))
        )
        self.body_positions.copy_(sim._rigid_body_pos[:, self.body_indices].to(dtype=torch.float32))
        self.body_quats.copy_(sim._rigid_body_rot[:, self.body_indices].to(dtype=torch.float32))
        self._update_object_states()

        wp.launch(
            kernel=_draw_depth_dynamic_kernel,
            dim=(self.num_envs, self.width, self.height),
            inputs=[
                self.terrain_mesh_id,
                self.robot_mesh_ids,
                self.object_mesh_ids,
                self.object_mesh_indices_wp,
                self.body_positions_wp,
                self.body_quats_wp,
                self.object_positions_wp,
                self.object_quats_wp,
                self.camera_position_wp,
                self.camera_orientation_wp,
                self.k_inv,
                self.max_range,
                self.depth_wp,
                self.ray_hits_wp,
                self.c_x,
                self.c_y,
                len(self.body_indices),
                len(self.object_meshes),
            ],
            device=self.device,
        )
        return self.depth

    def _update_object_states(self) -> None:
        if not self.object_meshes:
            return
        state = self._current_object_state()
        if state is None:
            self.object_positions.zero_()
            self.object_positions[:, 2] = -100.0
            self.object_quats.zero_()
            self.object_quats[:, 3] = 1.0
            return
        pos, quat = state
        self.object_positions.copy_(pos.to(device=self.env.device, dtype=torch.float32))
        self.object_quats.copy_(quat.to(device=self.env.device, dtype=torch.float32))

    def _current_object_state(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is not None:
            get_state = getattr(command_manager, "get_state", None)
            if callable(get_state):
                motion_command = get_state("motion_command")
                motion = getattr(motion_command, "motion", None)
                if motion_command is not None and getattr(motion, "has_object", False):
                    return motion_command.simulator_object_pos_w, motion_command.simulator_object_quat_w

        sim = self.env.simulator
        get_object_states = getattr(sim, "_get_object_states", None)
        if callable(get_object_states):
            try:
                env_ids = torch.arange(self.num_envs, device=self.env.device, dtype=torch.long)
                states = get_object_states("object", env_ids)
                return states[:, :3], states[:, 3:7]
            except Exception as exc:
                logger.debug("Could not query simulator object states for depth raycast: {}", exc)
        return None

    def _make_k_inv(self) -> wp.mat44:
        u_0 = self.width / 2.0
        v_0 = self.height / 2.0
        h_fov = math.radians(float(self.depth_cfg.horizontal_fov_deg))
        focal_px = self.width / (2.0 * math.tan(h_fov / 2.0))
        vertical_fov = 2.0 * math.atan(self.height / (2.0 * focal_px))
        alpha_u = u_0 / math.tan(h_fov / 2.0)
        alpha_v = v_0 / math.tan(vertical_fov / 2.0)
        k = wp.mat44(
            alpha_u,
            0.0,
            u_0,
            0.0,
            0.0,
            alpha_v,
            v_0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        )
        return wp.inverse(k)

    def _sample_local_camera_offsets(self, depth_cfg: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_pos = torch.tensor(depth_cfg.offset, device=self.env.device, dtype=torch.float32)
        base_rpy = torch.tensor(depth_cfg.offset_rpy_deg, device=self.env.device, dtype=torch.float32)
        if bool(depth_cfg.randomize_placement):
            min_translation = torch.tensor(depth_cfg.min_translation, device=self.env.device, dtype=torch.float32)
            max_translation = torch.tensor(depth_cfg.max_translation, device=self.env.device, dtype=torch.float32)
            min_rpy = torch.tensor(depth_cfg.min_euler_rotation_deg, device=self.env.device, dtype=torch.float32)
            max_rpy = torch.tensor(depth_cfg.max_euler_rotation_deg, device=self.env.device, dtype=torch.float32)
            local_pos = base_pos + min_translation + torch.rand((self.num_envs, 3), device=self.env.device) * (
                max_translation - min_translation
            )
            local_rpy = base_rpy + min_rpy + torch.rand((self.num_envs, 3), device=self.env.device) * (max_rpy - min_rpy)
        else:
            local_pos = base_pos.unsqueeze(0).repeat(self.num_envs, 1)
            local_rpy = base_rpy.unsqueeze(0).repeat(self.num_envs, 1)
        local_quat = _quat_from_rpy_deg_torch(local_rpy)
        data_quat = _quat_from_rpy_deg_torch(
            torch.tensor([-90.0, 0.0, -90.0], device=self.env.device, dtype=torch.float32)
        ).repeat(self.num_envs, 1)
        return local_pos, local_quat, data_quat

    def _build_terrain_mesh(self, env: Any) -> trimesh.Trimesh:
        terrain_state = env.terrain_manager.get_state("locomotion_terrain")
        terrain = getattr(terrain_state, "terrain", None)
        mesh = getattr(terrain, "mesh", None)
        if mesh is None:
            raise RuntimeError("Far-tracking-style depth camera requires a terrain trimesh.")
        return _to_trimesh(mesh)

    def _resolve_camera_body_name(self, env: Any, depth_cfg: Any) -> str:
        if depth_cfg.body_name is not None:
            return depth_cfg.body_name
        for body_name in depth_cfg.fallback_body_names:
            if body_name in env.simulator.body_names:
                return body_name
        raise RuntimeError(f"Cannot resolve depth camera body from available bodies: {env.simulator.body_names}")

    def _resolve_mesh_root(self, env: Any, depth_cfg: Any) -> Path:
        if depth_cfg.self_occlusion_mesh_root:
            return _resolve_holosoma_path(depth_cfg.self_occlusion_mesh_root)
        asset_root = _resolve_holosoma_path(env.robot_config.asset.asset_root)
        return asset_root / Path(env.robot_config.asset.urdf_file).parent / "meshes"

    def _resolve_body_meshes(self, env: Any, depth_cfg: Any) -> tuple[list[str], list[int], list[Path]]:
        mesh_root = self._resolve_mesh_root(env, depth_cfg)
        mesh_map = dict(FAR_TRACKING_G1_RAYCAST_BODY_MESHES)
        mesh_map.update(depth_cfg.self_occlusion_body_meshes or {})
        body_names: list[str] = []
        body_indices: list[int] = []
        mesh_paths: list[Path] = []
        missing: list[str] = []
        for body_name, mesh_file in mesh_map.items():
            if body_name not in env.simulator.body_names:
                continue
            mesh_path = mesh_root / mesh_file
            if not mesh_path.exists():
                for fallback in _MESH_FALLBACKS.get(mesh_file, (f"{body_name}.STL",)):
                    fallback_path = mesh_root / fallback
                    if fallback_path.exists():
                        mesh_path = fallback_path
                        break
            if not mesh_path.exists():
                missing.append(f"{body_name}:{mesh_file}")
                continue
            body_names.append(body_name)
            body_indices.append(env.simulator.body_names.index(body_name))
            mesh_paths.append(mesh_path)
        if missing:
            logger.warning(
                "Skipping {} depth self-occlusion meshes that could not be resolved under {}: {}",
                len(missing),
                mesh_root,
                missing[:8],
            )
        return body_names, body_indices, mesh_paths

    def _resolve_object_meshes(self, env: Any) -> tuple[list[wp.Mesh], list[int]]:
        sim = env.simulator
        env_object_paths = getattr(sim, "_env_object_urdf_paths", None)
        if isinstance(env_object_paths, list) and len(env_object_paths) >= self.num_envs:
            object_paths = [str(path or "") for path in env_object_paths[: self.num_envs]]
        else:
            object_paths = self._fallback_object_paths(sim)

        if not object_paths or not any(object_paths):
            return [], [-1] * self.num_envs

        object_meshes: list[wp.Mesh] = []
        mesh_index_by_path: dict[str, int] = {}
        env_indices: list[int] = []
        for raw_path in object_paths:
            if not raw_path:
                env_indices.append(-1)
                continue
            try:
                resolved = _resolve_holosoma_path(raw_path).resolve()
                key = str(resolved)
                if key not in mesh_index_by_path:
                    object_mesh = _convert_to_warp_mesh(_load_urdf_raycast_mesh(resolved), self.device)
                    mesh_index_by_path[key] = len(object_meshes)
                    object_meshes.append(object_mesh)
                env_indices.append(mesh_index_by_path[key])
            except Exception as exc:
                logger.warning("Skipping object depth raycast mesh from {}: {}", raw_path, exc)
                env_indices.append(-1)

        return object_meshes, env_indices

    def _fallback_object_paths(self, sim: Any) -> list[str]:
        object_urdf_by_name = getattr(sim, "_object_urdf_by_name", None)
        if isinstance(object_urdf_by_name, dict):
            paths = [str(path or "") for name, path in object_urdf_by_name.items() if name != "usd_scene_objects" and path]
            if len(paths) == 1:
                return [paths[0]] * self.num_envs
            if len(paths) > 1:
                logger.warning(
                    "Warp depth camera found multiple simulator object URDFs but no env assignment; "
                    "dynamic object depth raycast is disabled for this run."
                )
        return [""] * self.num_envs
