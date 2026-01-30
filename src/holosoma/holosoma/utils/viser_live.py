from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    matrix_to_quaternion,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    yaw_quat,
)
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.viser_utils import ensure_viser_on_path, resolve_viser_port

LIGHT_BLUE = (130, 180, 235)
HEIGHTMAP_MARKER_COLOR = (255, 165, 0)
CAMERA_MARKER_COLOR = (0, 255, 255)
SENSOR_MARKER_RADIUS = 0.03
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
        (58, 82, 139),
        (56, 86, 139),
        (54, 90, 139),
        (52, 94, 139),
        (50, 98, 139),
        (49, 102, 138),
        (47, 105, 137),
        (46, 109, 136),
        (45, 113, 135),
        (43, 117, 134),
        (42, 120, 133),
        (41, 124, 132),
        (40, 127, 130),
        (40, 131, 129),
        (39, 134, 128),
        (38, 137, 126),
        (38, 140, 125),
        (38, 144, 123),
        (37, 147, 121),
        (37, 150, 120),
        (37, 153, 118),
        (36, 156, 116),
        (36, 159, 114),
        (36, 162, 112),
        (36, 165, 110),
        (36, 168, 108),
        (35, 171, 106),
        (35, 174, 104),
        (35, 177, 102),
        (35, 180, 100),
        (35, 183, 98),
        (35, 186, 96),
        (35, 189, 94),
        (35, 192, 92),
        (35, 194, 90),
        (35, 197, 88),
        (35, 200, 86),
        (35, 203, 84),
        (35, 206, 82),
        (35, 209, 80),
        (35, 211, 78),
        (35, 214, 76),
        (35, 217, 74),
        (35, 220, 72),
        (35, 222, 70),
        (35, 225, 68),
        (35, 228, 66),
        (35, 230, 64),
        (35, 233, 62),
        (35, 235, 60),
        (35, 238, 58),
        (36, 240, 56),
        (36, 243, 54),
        (36, 245, 52),
        (36, 248, 50),
        (36, 250, 48),
        (36, 252, 46),
        (37, 255, 44),
    ],
    dtype=np.uint8,
)


def _apply_colormap(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    scaled = values * float(len(_VIRIDIS_LUT) - 1)
    idx0 = np.floor(scaled).astype(np.int32)
    idx1 = np.clip(idx0 + 1, 0, len(_VIRIDIS_LUT) - 1)
    t = (scaled - idx0)[..., None]
    c0 = _VIRIDIS_LUT[idx0].astype(np.float32)
    c1 = _VIRIDIS_LUT[idx1].astype(np.float32)
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


def _normalize_vec(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec).clamp(min=1.0e-6)


def _frustum_quat_from_camera(cam_quat_xyzw: torch.Tensor) -> torch.Tensor:
    x_axis = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=cam_quat_xyzw.device)
    y_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=cam_quat_xyzw.device)
    z_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=cam_quat_xyzw.device)

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


def _quat_from_forward_up(forward: torch.Tensor, up_hint: torch.Tensor) -> torch.Tensor:
    forward = _normalize_vec(forward)
    up_hint = _normalize_vec(up_hint)
    up_proj = up_hint - torch.dot(up_hint, forward) * forward
    if torch.linalg.norm(up_proj) < 1.0e-4:
        fallback = torch.tensor([0.0, 0.0, 1.0], device=forward.device, dtype=forward.dtype)
        if torch.abs(torch.dot(fallback, forward)) > 0.9:
            fallback = torch.tensor([0.0, 1.0, 0.0], device=forward.device, dtype=forward.dtype)
        up_proj = fallback - torch.dot(fallback, forward) * forward
    z_axis = _normalize_vec(up_proj)
    y_axis = _normalize_vec(torch.cross(z_axis, forward))
    z_axis = _normalize_vec(torch.cross(forward, y_axis))
    rot = torch.stack([forward, y_axis, z_axis], dim=1)
    quat_wxyz = matrix_to_quaternion(rot)
    return quat_wxyz[[1, 2, 3, 0]]


def _make_marker_mesh(color: tuple[int, int, int], radius: float):
    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception:
        return None
    mesh = trimesh.creation.icosphere(subdivisions=2, radius=radius)
    mesh.visual.face_colors = np.tile(np.array(color, dtype=np.uint8), (len(mesh.faces), 1))
    return mesh


def _import_viser() -> tuple[Any | None, Any | None, str | None]:
    ensure_viser_on_path()
    try:
        import viser  # type: ignore[import-not-found]
        from viser.extras import ViserUrdf  # type: ignore[import-not-found]
    except Exception as exc:
        return None, None, str(exc)
    return viser, ViserUrdf, None


def _resolve_data_path(path: str) -> str:
    if path.startswith("@holosoma/"):
        return str(Path(get_holosoma_root()) / path[len("@holosoma/") :])
    return resolve_data_file_path(path)


def _resolve_robot_urdf_path(robot_config: Any) -> str:
    asset_root = _resolve_data_path(robot_config.asset.asset_root)
    urdf_path = os.path.join(asset_root, robot_config.asset.urdf_file)
    return _resolve_data_path(urdf_path)


def _resolve_object_urdf_path(robot_config: Any) -> str | None:
    obj_path = getattr(getattr(robot_config, "object", None), "object_urdf_path", None)
    if not obj_path:
        return None
    return _resolve_data_path(obj_path)


def _is_rank0() -> bool:
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except ValueError:
        return True


def _load_terrain_mesh(
    obj_path: str | None,
    *,
    obj_metadata_path: str | None,
    num_rows: int | None,
    num_cols: int | None,
    clip_name: str | None = None,
):
    if not obj_path:
        return None

    try:
        import trimesh  # type: ignore[import-not-found]
    except Exception as exc:
        logger.warning("Viser terrain disabled (trimesh unavailable): {}", exc)
        return None

    def _load_mesh(path: Path) -> trimesh.Trimesh:
        mesh = trimesh.load(str(path), process=False)
        if isinstance(mesh, trimesh.Scene):
            meshes = mesh.dump(concatenate=False)
            if not meshes:
                raise ValueError("Loaded terrain scene has no geometries.")
            mesh = max(meshes, key=lambda m: len(getattr(m, "faces", [])) or len(m.vertices))
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError(f"Loaded terrain is not a trimesh: {type(mesh)}")
        return mesh

    def _resolve_obj_paths(path_str: str) -> list[Path]:
        path = Path(path_str)
        if path.is_dir():
            matches = list(path.glob("*.obj")) + list(path.glob("*.OBJ"))
            return sorted(matches)
        if any(char in path_str for char in ("*", "?", "[")):
            import glob

            return sorted(Path(p) for p in glob.glob(path_str))
        return [path] if path.exists() else []

    def _select_obj_path(paths: list[Path], name: str | None) -> Path:
        if name:
            for candidate in paths:
                if candidate.stem == name or candidate.stem.lower() == name.lower():
                    return candidate
            logger.warning("No terrain OBJ matching clip '{}'; using {}.", name, paths[0].name)
        return paths[0]

    terrain_path = Path(_resolve_data_path(obj_path))
    obj_paths = _resolve_obj_paths(str(terrain_path))
    if not obj_paths:
        return None

    if obj_metadata_path:
        if len(obj_paths) != 1:
            logger.warning("OBJ metadata requires a single OBJ file; ignoring metadata for directory input.")
        selected_path = obj_paths[0]
    else:
        selected_path = _select_obj_path(obj_paths, clip_name)

    return _load_mesh(selected_path)


class ViserLiveViewer:
    def __init__(self, env: Any) -> None:
        self._env = env
        self._enabled = bool(getattr(env.training_config, "enable_viser", False))
        self._server = None
        self._vr = None
        self._vo = None
        self._robot_root = None
        self._object_root = None
        self._joint_order: np.ndarray | None = None
        self._joint_count = 0
        self._offset: np.ndarray | None = None
        self._last_update = 0.0
        self._scandots_handle = None
        self._scandots_rays_handle = None
        self._scandots_enabled = False
        self._scandots_point_size = 0.02
        self._scandots_color = np.array([255, 0, 0], dtype=np.uint8)
        self._scandots_warned = False
        self._target_keypoints_handle = None
        self._target_keypoints_point_size = 0.03
        self._target_keypoints_color = np.array([128, 0, 128], dtype=np.uint8)
        self._play_control = None
        self._step_button = None
        self._reset_button = None
        self._step_requested = False
        self._reset_requested = False
        self._clip_dropdown = None
        self._clip_apply = None
        self._clip_label = None
        self._clip_names: list[str] = []
        self._pending_clip_idx: int | None = None
        self._control_sleep_s = 0.02
        self._pending_clip_start: int | None = None
        self._grid_handle = None
        self._terrain_handle = None
        self._ground_handle = None
        self._terrain_clip_name = None
        self._terrain_is_local = False
        self._show_robot_cb = None
        self._show_object_cb = None
        self._show_terrain_cb = None
        self._show_grid_cb = None
        self._show_scandots_cb = None
        self._recenter_cb = None
        self._scandots_size_slider = None
        self._contact_force_cb = None
        self._contact_force_scale_slider = None
        self._contact_force_threshold_slider = None
        self._contact_force_handle = None
        self._clip_start_slider = None
        self._clip_lock_cb = None
        self._perception_enabled = False
        self._perception_depth_handle = None
        self._perception_stats = None
        self._perception_show_depth_cb = None
        self._perception_show_frustum_cb = None
        self._perception_show_points_cb = None
        self._perception_show_heightmap_joint_cb = None
        self._perception_show_camera_joint_cb = None
        self._perception_frustum = None
        self._perception_frame = None
        self._heightmap_marker_handle = None
        self._heightmap_marker_is_pc = False
        self._camera_marker_handle = None
        self._camera_marker_is_pc = False
        self._perception_last_shape: tuple[int, int] | None = None
        self._perception_last_mode: str | None = None
        self._perception_last_fov: float | None = None
        self._perception_last_aspect: float | None = None

        if not self._enabled:
            return
        if not _is_rank0():
            self._enabled = False
            return

        cfg = env.training_config
        self._env_id = int(getattr(cfg, "viser_env_id", 0))
        if self._env_id < 0 or self._env_id >= getattr(env, "num_envs", 1):
            logger.warning("Viser env_id {} out of range; defaulting to 0.", self._env_id)
            self._env_id = 0

        update_hz = float(getattr(cfg, "viser_update_hz", 30.0))
        sync_to_sim = bool(getattr(cfg, "viser_sync_to_sim", True))
        force_dt = bool(getattr(cfg, "viser_force_dt", True))
        sim_hz = None
        sim_cfg = getattr(env.simulator, "simulator_config", None)
        if sim_cfg is not None and hasattr(sim_cfg, "sim"):
            sim_fps = getattr(sim_cfg.sim, "fps", None)
            sim_decimation = getattr(sim_cfg.sim, "control_decimation", None)
            if sim_fps:
                sim_decimation = int(sim_decimation or 1)
                sim_hz = float(sim_fps) / max(1, sim_decimation)

        if sync_to_sim and sim_hz:
            update_hz = sim_hz
        elif update_hz <= 0 and sim_hz:
            update_hz = sim_hz

        self._update_period = 0.0 if update_hz <= 0 else 1.0 / update_hz
        self._force_dt = force_dt and self._update_period > 0
        self._next_tick = time.perf_counter()
        self._recenter = bool(getattr(cfg, "viser_recenter", True))
        self._scandots_enabled = bool(getattr(cfg, "viser_show_scandots", False))
        self._scandots_point_size = float(getattr(cfg, "viser_scandots_point_size", 0.02))

        viser_mod, viser_urdf_cls, err = _import_viser()
        if err is not None or viser_mod is None or viser_urdf_cls is None:
            logger.warning("Viser live viewer disabled: {}", err or "missing dependency")
            self._enabled = False
            return

        port_cfg = int(getattr(cfg, "viser_port", 0) or 0)
        port = resolve_viser_port(port_cfg)
        self._server = viser_mod.ViserServer(port=port)

        self._robot_root = self._server.scene.add_frame("/robot", show_axes=False)
        self._object_root = self._server.scene.add_frame("/object", show_axes=False)
        self._grid_handle = self._server.scene.add_grid(
            "/grid", width=8.0, height=8.0, position=(0.0, 0.0, 0.0)
        )
        self._grid_handle.visible = False

        robot_urdf = _resolve_robot_urdf_path(env.robot_config)
        self._vr = viser_urdf_cls(
            self._server,
            urdf_or_path=Path(robot_urdf),
            root_node_name="/robot",
        )

        object_urdf = _resolve_object_urdf_path(env.robot_config)
        if object_urdf:
            self._vo = viser_urdf_cls(
                self._server,
                urdf_or_path=Path(object_urdf),
                root_node_name="/object",
                mesh_color_override=LIGHT_BLUE,
            )

        self._setup_joint_order()
        self._load_terrain()
        self._setup_controls()

        logger.info("Viser live viewer listening on port {}", port)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _set_handle_visible(self, handle, visible: bool) -> None:
        if handle is None:
            return
        try:
            handle.visible = bool(visible)
        except Exception:
            return

    def _set_sim_cfg(self, field: str, value) -> None:
        sim_cfg = getattr(self._env.simulator, "simulator_config", None)
        if sim_cfg is None:
            return
        try:
            object.__setattr__(sim_cfg, field, value)
        except Exception:
            return

    def _clear_terrain_handles(self) -> None:
        for handle in (self._terrain_handle, self._ground_handle):
            if handle is None:
                continue
            try:
                handle.remove()
            except Exception:
                pass
        self._terrain_handle = None
        self._ground_handle = None

    def _update_terrain_transform(self, viewer_offset: np.ndarray | None = None) -> None:
        handle = self._terrain_handle or self._ground_handle
        if handle is None:
            return
        terrain_offset = np.zeros(3, dtype=np.float32)
        if self._terrain_is_local:
            resolved = self._resolve_env_origin()
            if resolved is not None:
                terrain_offset = resolved
        if viewer_offset is None:
            if self._recenter and self._offset is not None:
                viewer_offset = self._offset
            else:
                viewer_offset = np.zeros(3, dtype=np.float32)
        try:
            handle.position = terrain_offset - viewer_offset
        except Exception:
            pass

    def _reload_terrain_for_clip(self, clip_name: str | None) -> None:
        if not self._enabled or self._server is None:
            return
        if clip_name == self._terrain_clip_name:
            return
        self._terrain_clip_name = clip_name
        self._clear_terrain_handles()
        self._load_terrain(clip_name=clip_name)
        self._update_terrain_transform()

    def wait_if_paused(self) -> None:
        if not self._enabled or self._play_control is None:
            return
        while not bool(self._play_control.value):
            if self._step_requested:
                self._step_requested = False
                return
            if self._reset_requested or self._pending_clip_idx is not None or self._pending_clip_start is not None:
                self.apply_pending_controls()
            time.sleep(self._control_sleep_s)

    def apply_pending_controls(self) -> None:
        if not self._enabled:
            return
        if self._reset_requested:
            self._reset_requested = False
            self._reset_env()
        if self._pending_clip_idx is not None or self._pending_clip_start is not None:
            self._apply_clip_selection()

    def on_reset(self, env_ids) -> None:
        if not self._enabled or not self._recenter:
            return
        if self._env_id not in _normalize_env_ids(env_ids):
            return
        self._offset = self._resolve_env_origin()
        self._reload_terrain_for_clip(self._current_clip_name(self._get_motion_command()))

    def record_step(self) -> None:
        if not self._enabled or self._vr is None or self._robot_root is None:
            return

        if self._force_dt:
            now = time.perf_counter()
            if now < self._next_tick:
                time.sleep(self._next_tick - now)
            now = time.perf_counter()
            self._next_tick = now + self._update_period
        else:
            now = time.perf_counter()
        if self._update_period > 0 and (now - self._last_update) < self._update_period:
            return
        self._last_update = now

        root_pos, root_quat_wxyz = self._get_root_state_wxyz()
        if root_pos is None or root_quat_wxyz is None:
            return

        dof_pos = self._get_dof_pos()
        if dof_pos is None:
            return

        if self._offset is None:
            self._offset = self._resolve_env_origin() if self._recenter else np.zeros(3, dtype=np.float32)
            if self._offset is None:
                self._offset = root_pos.copy()
        offset = self._offset if self._recenter else np.zeros(3, dtype=np.float32)
        self._update_terrain_transform(offset)

        with self._server.atomic():
            self._robot_root.position = root_pos - offset
            self._robot_root.wxyz = root_quat_wxyz

            joints = dof_pos
            if self._joint_order is not None:
                joints = joints[self._joint_order]
            if joints.shape[0] != self._joint_count:
                return
            self._vr.update_cfg(joints.astype(np.float32, copy=False))

            if self._scandots_enabled:
                self._update_scandots(offset)
            self._update_target_keypoints(offset)
            if self._perception_enabled:
                self._update_perception_visuals(offset)
            self._update_contact_forces(offset)

            if self._vo is None or self._object_root is None:
                return
            if self._show_object_cb is not None and not bool(self._show_object_cb.value):
                self._vo.show_visual = False
                return
            obj_state = self._get_object_state_wxyz()
            if obj_state is None:
                self._vo.show_visual = False
                return
            obj_pos, obj_quat_wxyz = obj_state
            self._object_root.position = obj_pos - offset
            self._object_root.wxyz = obj_quat_wxyz
            self._vo.show_visual = True

        if self._clip_label is not None:
            motion_cmd = self._get_motion_command()
            if motion_cmd is not None and hasattr(motion_cmd, "clip_ids"):
                try:
                    clip_idx = int(motion_cmd.clip_ids[self._env_id].item())
                    clip_names = getattr(motion_cmd.motion, "clip_ids", [])
                    clip_name = clip_names[clip_idx] if 0 <= clip_idx < len(clip_names) else str(clip_idx)
                    self._clip_label.content = f"Current clip: `{clip_name}`"
                    self._reload_terrain_for_clip(str(clip_name))
                except Exception:
                    pass

    def _setup_joint_order(self) -> None:
        if self._vr is None:
            return
        viser_joint_names = list(self._vr.get_actuated_joint_names())
        self._joint_count = len(viser_joint_names)
        name_to_idx = {name: idx for idx, name in enumerate(self._env.robot_config.dof_names)}
        missing = [name for name in viser_joint_names if name not in name_to_idx]
        if missing:
            if len(viser_joint_names) == len(self._env.robot_config.dof_names):
                logger.warning("Viser joint names mismatch; using robot DOF order.")
                self._joint_order = None
            else:
                logger.warning("Viser joints missing in robot config: {}", missing)
                self._enabled = False
        else:
            self._joint_order = np.array([name_to_idx[name] for name in viser_joint_names], dtype=np.int64)

    def _load_terrain(self, clip_name: str | None = None) -> None:
        if self._server is None:
            return
        terrain_mgr = getattr(self._env, "terrain_manager", None)
        if terrain_mgr is None:
            return

        motion_cmd = self._get_motion_command()
        if clip_name is None:
            clip_name = self._current_clip_name(motion_cmd)
        self._terrain_clip_name = clip_name

        terrain_state = terrain_mgr.get_state("locomotion_terrain")
        terrain_cfg = getattr(terrain_mgr, "cfg", None)
        terrain_term = getattr(terrain_cfg, "terrain_term", None) if terrain_cfg is not None else None

        mesh = None
        mesh_is_local = False
        if terrain_term is not None:
            obj_path = getattr(terrain_term, "obj_file_path", None) or ""
            obj_meta = getattr(terrain_term, "obj_metadata_path", None)
            rows = getattr(terrain_term, "num_rows", None)
            cols = getattr(terrain_term, "num_cols", None)
            if obj_path:
                mesh = _load_terrain_mesh(
                    obj_path,
                    obj_metadata_path=obj_meta,
                    num_rows=rows,
                    num_cols=cols,
                    clip_name=clip_name,
                )
                if mesh is not None:
                    mesh_is_local = True
                    if obj_meta:
                        terrain_obj = getattr(terrain_state, "terrain", None)
                        tile_rows = int(getattr(terrain_obj, "obj_tile_rows", 0) or 0)
                        tile_cols = int(getattr(terrain_obj, "obj_tile_cols", 0) or 0)
                        tile_offsets = getattr(terrain_obj, "obj_tile_offsets", None)
                        tile_count = int(np.asarray(tile_offsets).shape[0]) if tile_offsets is not None else 0
                        if tile_count > 1 or tile_rows > 1 or tile_cols > 1:
                            mesh_is_local = False

        if mesh is None:
            mesh = getattr(terrain_state, "mesh", None)
            if mesh is None:
                terrain = getattr(terrain_state, "terrain", None)
                mesh = getattr(terrain, "mesh", None) if terrain is not None else None
            if mesh is not None:
                try:
                    import trimesh  # type: ignore[import-not-found]
                except Exception:
                    mesh = None
                else:
                    if isinstance(mesh, trimesh.Scene):
                        meshes = mesh.dump(concatenate=False)
                        if meshes:
                            mesh = max(
                                meshes,
                                key=lambda m: len(getattr(m, "faces", [])) or len(m.vertices),
                            )
                    if not isinstance(mesh, trimesh.Trimesh):
                        try:
                            mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.faces))
                        except Exception:
                            mesh = None
                    if mesh is not None:
                        mesh_is_local = False

        if mesh is None:
            try:
                import trimesh  # type: ignore[import-not-found]
            except Exception:
                return
            ground_mesh = trimesh.creation.box(extents=(8.0, 8.0, 0.01))
            ground_mesh.apply_translation([0.0, 0.0, -0.005])
            self._ground_handle = self._server.scene.add_mesh_simple(
                "/ground",
                ground_mesh.vertices,
                ground_mesh.faces,
                color=LIGHT_BLUE,
                side="double",
            )
            if self._show_terrain_cb is not None:
                self._ground_handle.visible = bool(self._show_terrain_cb.value)
            self._terrain_is_local = False
            self._update_terrain_transform()
            return

        self._terrain_handle = self._server.scene.add_mesh_simple(
            "/terrain",
            mesh.vertices,
            mesh.faces,
            color=LIGHT_BLUE,
            side="double",
        )
        if self._show_terrain_cb is not None:
            self._terrain_handle.visible = bool(self._show_terrain_cb.value)
        self._terrain_is_local = bool(mesh_is_local)
        self._update_terrain_transform()

    def _get_perception_manager(self):
        mgr = getattr(self._env, "perception_manager", None)
        if mgr is None or not getattr(mgr, "enabled", False):
            return None
        return mgr

    def _resolve_perception_shape(self, perception_mgr) -> tuple[int, int] | None:
        cfg = getattr(perception_mgr, "cfg", None)
        if cfg is None:
            return None
        output_mode = getattr(cfg, "output_mode", None)
        if output_mode == "camera_depth":
            width = int(getattr(perception_mgr, "_camera_width", 0) or 0)
            height = int(getattr(perception_mgr, "_camera_height", 0) or 0)
            if width > 0 and height > 0:
                return height, width
            width = int(getattr(cfg, "camera_width", 0) or 0)
            height = int(getattr(cfg, "camera_height", 0) or 0)
            if width > 0 and height > 0:
                return height, width
            grid = int(getattr(cfg, "grid_size", 0) or 0)
            if grid > 0:
                return grid, grid
            return None
        if output_mode == "heightmap":
            heightmap = getattr(perception_mgr, "_heightmap", None)
            if isinstance(heightmap, torch.Tensor) and heightmap.ndim >= 3:
                return int(heightmap.shape[-2]), int(heightmap.shape[-1])
            grid_x = int(getattr(perception_mgr, "_heightmap_grid_x", 0) or 0)
            grid_y = int(getattr(perception_mgr, "_heightmap_grid_y", 0) or 0)
            if grid_x > 0 and grid_y > 0:
                return grid_x, grid_y
        return None

    def _resolve_heightmap_fov_aspect(self, perception_mgr) -> tuple[float, float]:
        grid_x = int(getattr(perception_mgr, "_heightmap_grid_x", 0) or 0)
        grid_y = int(getattr(perception_mgr, "_heightmap_grid_y", 0) or 0)
        interval_x = float(getattr(perception_mgr, "_heightmap_interval_x", 0.0) or 0.0)
        interval_y = float(getattr(perception_mgr, "_heightmap_interval_y", 0.0) or 0.0)
        cfg = getattr(perception_mgr, "cfg", None)
        ray_height = float(getattr(cfg, "ray_start_height", 0.6)) if cfg is not None else 0.6
        if grid_x <= 1 or grid_y <= 1 or interval_x <= 0 or interval_y <= 0 or ray_height <= 0:
            return 90.0, 1.0
        half_x = 0.5 * (grid_x - 1) * interval_x
        half_y = 0.5 * (grid_y - 1) * interval_y
        if half_y <= 0:
            return 90.0, 1.0
        fov = float(np.degrees(2.0 * np.arctan(half_y / ray_height)))
        aspect = float(half_x / half_y) if half_y > 0 else 1.0
        fov = float(np.clip(fov, 5.0, 175.0))
        aspect = float(max(aspect, 0.1))
        return fov, aspect

    def _setup_perception_controls(self) -> None:
        if self._server is None:
            return
        perception_mgr = self._get_perception_manager()
        if perception_mgr is None:
            return
        cfg = getattr(perception_mgr, "cfg", None)
        if cfg is None:
            return
        self._perception_enabled = True
        output_mode = getattr(cfg, "output_mode", "")
        shape = self._resolve_perception_shape(perception_mgr) or (64, 64)
        height, width = shape
        self._perception_last_shape = (height, width)
        self._perception_last_mode = str(output_mode)

        with self._server.gui.add_folder("Perception"):
            self._perception_show_depth_cb = self._server.gui.add_checkbox(
                "Show Depth",
                initial_value=True,
                hint="Display perception depth in GUI and frustum",
            )
            self._perception_show_frustum_cb = self._server.gui.add_checkbox(
                "Show Frustum",
                initial_value=True,
                hint="Display the perception camera frustum in 3D",
            )
            self._perception_show_points_cb = self._server.gui.add_checkbox(
                "Show Perception Points",
                initial_value=self._scandots_enabled,
                hint="Toggle perception hit points (heightmap or camera scandots)",
            )
            if getattr(cfg, "heightmap_body_name", None):
                self._perception_show_heightmap_joint_cb = self._server.gui.add_checkbox(
                    "Show Heightmap Joint",
                    initial_value=True,
                    hint="Show the body used for heightmap sampling",
                )
            if getattr(cfg, "camera_body_name", None):
                self._perception_show_camera_joint_cb = self._server.gui.add_checkbox(
                    "Show Camera Joint",
                    initial_value=True,
                    hint="Show the body used for camera sampling",
                )
            self._perception_depth_handle = self._server.gui.add_image(
                np.zeros((height, width, 3), dtype=np.uint8),
                label="Perception Depth",
            )
            self._perception_stats = self._server.gui.add_markdown("Depth range (valid): n/a")

        @self._perception_show_depth_cb.on_update
        def _(_evt) -> None:
            if self._perception_depth_handle is None:
                return
            if not bool(self._perception_show_depth_cb.value):
                self._perception_depth_handle.image = np.zeros((height, width, 3), dtype=np.uint8)

        @self._perception_show_frustum_cb.on_update
        def _(_evt) -> None:
            if self._perception_frustum is not None:
                self._perception_frustum.visible = bool(self._perception_show_frustum_cb.value)

        @self._perception_show_points_cb.on_update
        def _(_evt) -> None:
            self._scandots_enabled = bool(self._perception_show_points_cb.value)
            if self._show_scandots_cb is not None and bool(self._show_scandots_cb.value) != self._scandots_enabled:
                self._show_scandots_cb.value = self._scandots_enabled
            if not self._scandots_enabled and self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if not self._scandots_enabled and self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False

        if self._perception_show_heightmap_joint_cb is not None:
            @self._perception_show_heightmap_joint_cb.on_update
            def _(_evt) -> None:
                if self._heightmap_marker_handle is not None:
                    self._heightmap_marker_handle.visible = bool(self._perception_show_heightmap_joint_cb.value)

        if self._perception_show_camera_joint_cb is not None:
            @self._perception_show_camera_joint_cb.on_update
            def _(_evt) -> None:
                if self._camera_marker_handle is not None:
                    self._camera_marker_handle.visible = bool(self._perception_show_camera_joint_cb.value)

        if output_mode == "heightmap":
            fov, aspect = self._resolve_heightmap_fov_aspect(perception_mgr)
        else:
            fov = float(getattr(cfg, "camera_vfov_deg", 90.0))
            aspect = float(width / max(1, height))
        self._perception_last_fov = fov
        self._perception_last_aspect = aspect

        self._perception_frame = self._server.scene.add_frame("/perception_camera", show_axes=False)
        self._perception_frustum = self._server.scene.add_camera_frustum(
            "/perception_frustum",
            fov=fov,
            aspect=aspect,
            scale=0.3,
            line_width=2.0,
            color=(0, 0, 0),
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(0.0, 0.0, 0.0),
            image=np.zeros((height, width, 3), dtype=np.uint8),
            format="jpeg",
            jpeg_quality=90,
        )
        if self._perception_show_frustum_cb is not None:
            self._perception_frustum.visible = bool(self._perception_show_frustum_cb.value)
    def _setup_controls(self) -> None:
        if self._server is None:
            return

        with self._server.gui.add_folder("Visualization"):
            self._show_scandots_cb = self._server.gui.add_checkbox(
                "Show Scandots",
                initial_value=self._scandots_enabled,
                hint="Toggle mesh-sampled point cloud",
            )
            self._scandots_size_slider = self._server.gui.add_slider(
                "Scandots Size",
                min=0.001,
                max=0.05,
                step=0.001,
                initial_value=float(self._scandots_point_size),
                hint="Point size for scandots visualization",
            )

        @self._show_scandots_cb.on_update
        def _(_evt) -> None:
            self._scandots_enabled = bool(self._show_scandots_cb.value)
            if self._perception_show_points_cb is not None and bool(self._perception_show_points_cb.value) != self._scandots_enabled:
                self._perception_show_points_cb.value = self._scandots_enabled
            if not self._scandots_enabled and self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if not self._scandots_enabled and self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False

        @self._scandots_size_slider.on_update
        def _(_evt) -> None:
            self._scandots_point_size = float(self._scandots_size_slider.value)
            if self._scandots_handle is not None:
                self._scandots_handle.point_size = float(self._scandots_point_size)

        self._setup_perception_controls()

        sim_cfg = getattr(self._env.simulator, "simulator_config", None)
        if sim_cfg is not None and hasattr(sim_cfg, "contact_force_viz"):
            with self._server.gui.add_folder("Contact Forces"):
                self._contact_force_cb = self._server.gui.add_checkbox(
                    "Show Contact Forces",
                    initial_value=bool(getattr(sim_cfg, "contact_force_viz", False)),
                    hint="Toggle contact force debug lines",
                )
                self._contact_force_scale_slider = self._server.gui.add_slider(
                    "Force Scale",
                    min=0.0,
                    max=0.01,
                    step=0.0001,
                    initial_value=float(getattr(sim_cfg, "contact_force_viz_scale", 0.001)),
                    hint="Scale factor for contact force lines",
                )
                self._contact_force_threshold_slider = self._server.gui.add_slider(
                    "Force Threshold",
                    min=0.0,
                    max=50.0,
                    step=0.1,
                    initial_value=float(getattr(sim_cfg, "contact_force_viz_threshold", 1.0)),
                    hint="Minimum force magnitude to display",
                )

            def _sync_contact_cfg() -> None:
                if self._contact_force_cb is None:
                    return
                self._set_sim_cfg("contact_force_viz", bool(self._contact_force_cb.value))
                if self._contact_force_scale_slider is not None:
                    self._set_sim_cfg("contact_force_viz_scale", float(self._contact_force_scale_slider.value))
                if self._contact_force_threshold_slider is not None:
                    self._set_sim_cfg(
                        "contact_force_viz_threshold",
                        float(self._contact_force_threshold_slider.value),
                    )

            @self._contact_force_cb.on_update
            def _(_evt) -> None:
                _sync_contact_cfg()

            @self._contact_force_scale_slider.on_update
            def _(_evt) -> None:
                _sync_contact_cfg()

            @self._contact_force_threshold_slider.on_update
            def _(_evt) -> None:
                _sync_contact_cfg()

        with self._server.gui.add_folder("Advanced", expand_by_default=False):
            with self._server.gui.add_folder("Simulation Control"):
                self._play_control = self._server.gui.add_checkbox(
                    "Play",
                    initial_value=True,
                    hint="Toggle simulation play/pause",
                )
                self._step_button = self._server.gui.add_button(
                    "Step",
                    hint="Step the simulation forward by one frame",
                )
                self._reset_button = self._server.gui.add_button(
                    "Reset",
                    hint="Reset the selected environment",
                )

            @self._step_button.on_click
            def _(_evt) -> None:
                self._step_requested = True

            @self._reset_button.on_click
            def _(_evt) -> None:
                self._reset_requested = True

            with self._server.gui.add_folder("World Viz"):
                self._show_robot_cb = self._server.gui.add_checkbox(
                    "Show Robot",
                    initial_value=bool(getattr(self._vr, "show_visual", True)),
                    hint="Toggle robot mesh visibility",
                )
                if self._vo is not None:
                    self._show_object_cb = self._server.gui.add_checkbox(
                        "Show Object",
                        initial_value=bool(getattr(self._vo, "show_visual", True)),
                        hint="Toggle object mesh visibility",
                    )
                if self._terrain_handle is not None or self._ground_handle is not None:
                    self._show_terrain_cb = self._server.gui.add_checkbox(
                        "Show Terrain",
                        initial_value=bool(
                            getattr(self._terrain_handle or self._ground_handle, "visible", True)
                        ),
                        hint="Toggle terrain mesh visibility",
                    )
                if self._grid_handle is not None:
                    self._show_grid_cb = self._server.gui.add_checkbox(
                        "Show Grid",
                        initial_value=bool(getattr(self._grid_handle, "visible", False)),
                        hint="Toggle the ground grid",
                    )
                self._recenter_cb = self._server.gui.add_checkbox(
                    "Recenter to Env Origin",
                    initial_value=bool(self._recenter),
                    hint="Keep the selected env centered in view",
                )

            def _apply_world_vis() -> None:
                if self._show_robot_cb is not None and self._vr is not None:
                    try:
                        self._vr.show_visual = bool(self._show_robot_cb.value)
                    except Exception:
                        pass
                if self._show_object_cb is not None and self._vo is not None:
                    try:
                        self._vo.show_visual = bool(self._show_object_cb.value)
                    except Exception:
                        pass
                if self._show_terrain_cb is not None:
                    handle = self._terrain_handle or self._ground_handle
                    self._set_handle_visible(handle, bool(self._show_terrain_cb.value))
                if self._show_grid_cb is not None:
                    self._set_handle_visible(self._grid_handle, bool(self._show_grid_cb.value))

            if self._show_robot_cb is not None:
                @self._show_robot_cb.on_update
                def _(_evt) -> None:
                    _apply_world_vis()

            if self._show_object_cb is not None:
                @self._show_object_cb.on_update
                def _(_evt) -> None:
                    _apply_world_vis()

            if self._show_terrain_cb is not None:
                @self._show_terrain_cb.on_update
                def _(_evt) -> None:
                    _apply_world_vis()

            if self._show_grid_cb is not None:
                @self._show_grid_cb.on_update
                def _(_evt) -> None:
                    _apply_world_vis()

            if self._recenter_cb is not None:
                @self._recenter_cb.on_update
                def _(_evt) -> None:
                    self._recenter = bool(self._recenter_cb.value)
                    if self._recenter:
                        offset = self._resolve_env_origin()
                        if offset is None:
                            offset = np.zeros(3, dtype=np.float32)
                        self._offset = offset
                    else:
                        self._offset = np.zeros(3, dtype=np.float32)
                    self._update_terrain_transform(self._offset)

            clip_gui_enabled = os.environ.get("VISER_ENABLE_CLIP_GUI", "1").lower() not in (
                "0",
                "false",
                "no",
            )
            if clip_gui_enabled:
                motion_cmd = self._get_motion_command()
                if motion_cmd is not None and hasattr(motion_cmd, "motion"):
                    clip_names = list(getattr(motion_cmd.motion, "clip_ids", []))
                    if clip_names:
                        self._clip_names = clip_names
                        with self._server.gui.add_folder("Clip Playback"):
                            if len(clip_names) > 1:
                                self._clip_dropdown = self._server.gui.add_dropdown(
                                    "Clip",
                                    options=clip_names,
                                    initial_value=clip_names[0],
                                    hint="Select which motion clip to visualize",
                                )
                            else:
                                self._clip_dropdown = None
                                self._clip_label = self._server.gui.add_markdown(
                                    f"Clip: `{clip_names[0]}`"
                                )
                            self._clip_start_slider = self._server.gui.add_slider(
                                "Clip Start Frame",
                                min=0,
                                max=10000,
                                step=1,
                                initial_value=0,
                                hint="Select starting frame in the clip",
                            )
                            self._clip_lock_cb = self._server.gui.add_checkbox(
                                "Lock Clip",
                                initial_value=True,
                                hint="Keep the selected clip fixed across resets",
                            )
                            self._clip_apply = self._server.gui.add_button("Apply Clip")
                            if self._clip_label is None:
                                self._clip_label = self._server.gui.add_markdown("")

                        def _update_clip_slider(idx: int | None) -> None:
                            if self._clip_start_slider is None or idx is None:
                                return
                            length = self._get_clip_length(motion_cmd, idx)
                            if length is None:
                                return
                            max_frame = max(0, int(length) - 2)
                            self._clip_start_slider.max = max_frame
                            if int(self._clip_start_slider.value) > max_frame:
                                self._clip_start_slider.value = max_frame

                        def _queue_clip_change() -> None:
                            idx = self._current_clip_index(motion_cmd)
                            if idx is None:
                                return
                            self._pending_clip_idx = idx
                            if self._clip_start_slider is not None:
                                self._pending_clip_start = int(self._clip_start_slider.value)
                            _update_clip_slider(idx)

                        if self._clip_dropdown is not None:
                            @self._clip_dropdown.on_update
                            def _(_evt) -> None:
                                _queue_clip_change()

                        @self._clip_start_slider.on_update
                        def _(_evt) -> None:
                            _queue_clip_change()

                        @self._clip_lock_cb.on_update
                        def _(_evt) -> None:
                            if bool(self._clip_lock_cb.value):
                                _queue_clip_change()
                            else:
                                try:
                                    motion_cmd.set_forced_clip(None)
                                except Exception:
                                    motion_cmd._forced_clip_idx = None

                        @self._clip_apply.on_click
                        def _(_evt) -> None:
                            _queue_clip_change()

                        # Force the initial clip so we don't randomize across the bank.
                        self._pending_clip_idx = 0
                        _update_clip_slider(0)

    def _get_motion_command(self):
        cmd_mgr = getattr(self._env, "command_manager", None)
        if cmd_mgr is None:
            return None
        return cmd_mgr.get_state("motion_command")

    def _current_clip_index(self, motion_cmd) -> int | None:
        if self._clip_dropdown is not None:
            try:
                return self._clip_names.index(str(self._clip_dropdown.value))
            except Exception:
                return None
        if motion_cmd is None or not hasattr(motion_cmd, "clip_ids"):
            return None
        try:
            return int(motion_cmd.clip_ids[self._env_id].item())
        except Exception:
            return None

    def _current_clip_name(self, motion_cmd, clip_idx: int | None = None) -> str | None:
        if motion_cmd is None or not hasattr(motion_cmd, "motion"):
            return None
        if clip_idx is None:
            clip_idx = self._current_clip_index(motion_cmd)
        if clip_idx is None:
            return None
        clip_names = list(getattr(motion_cmd.motion, "clip_ids", []))
        if 0 <= int(clip_idx) < len(clip_names):
            return str(clip_names[int(clip_idx)])
        return None

    def _get_clip_length(self, motion_cmd, clip_idx: int) -> int | None:
        if motion_cmd is None or not hasattr(motion_cmd, "motion"):
            return None
        lengths = getattr(motion_cmd.motion, "clip_lengths", None)
        if lengths is None:
            return None
        try:
            if isinstance(lengths, torch.Tensor):
                return int(lengths[clip_idx].item())
            return int(lengths[clip_idx])
        except Exception:
            return None

    def _reset_env(self) -> None:
        if not hasattr(self._env, "reset_envs_idx"):
            return
        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        self._env.reset_envs_idx(env_ids)
        if hasattr(self._env, "reset_buf"):
            self._env.reset_buf[env_ids] = 0
        if hasattr(self._env, "time_out_buf"):
            self._env.time_out_buf[env_ids] = 0

    def _apply_clip_selection(self) -> None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is None:
            self._pending_clip_idx = None
            self._pending_clip_start = None
            return
        clip_idx = self._pending_clip_idx
        clip_start = self._pending_clip_start
        self._pending_clip_idx = None
        self._pending_clip_start = None
        if clip_idx is None:
            return
        lock_enabled = True
        if self._clip_lock_cb is not None:
            lock_enabled = bool(self._clip_lock_cb.value)
        if lock_enabled:
            try:
                motion_cmd.set_forced_clip(int(clip_idx))
            except Exception:
                motion_cmd._forced_clip_idx = int(clip_idx)
        else:
            try:
                motion_cmd.set_forced_clip(None)
            except Exception:
                motion_cmd._forced_clip_idx = None
        self._force_clip_state(motion_cmd, int(clip_idx), clip_start)
        self._reload_terrain_for_clip(self._current_clip_name(motion_cmd, int(clip_idx)))

    def _force_clip_state(self, motion_cmd, clip_idx: int, clip_start: int | None) -> None:
        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        self._env.reset_envs_idx(env_ids)

        clip_length = self._get_clip_length(motion_cmd, clip_idx)
        max_valid = max(0, int(clip_length or 1) - 2)
        start_frame = max(0, min(int(clip_start or 0), max_valid))

        motion_cmd.clip_ids[env_ids] = int(clip_idx)
        motion_cmd.time_steps[env_ids] = start_frame
        if motion_cmd.motion_cfg.align_motion_to_init_yaw:
            motion_cmd._update_motion_alignment(env_ids)

        root_pos = motion_cmd.root_pos_w[env_ids].clone()
        root_rot = motion_cmd.root_quat_w[env_ids].clone()

        motion_idx = motion_cmd._get_motion_indices(motion_cmd.time_steps[env_ids], env_ids)
        root_lin_vel = motion_cmd.motion.body_lin_vel_w[motion_idx, 0].clone()
        root_ang_vel = motion_cmd.motion.body_ang_vel_w[motion_idx, 0].clone()
        if motion_cmd.motion_cfg.align_motion_to_init_yaw:
            root_lin_vel = motion_cmd._apply_motion_alignment_vec(root_lin_vel)
            root_ang_vel = motion_cmd._apply_motion_alignment_vec(root_ang_vel)

        dof_pos = motion_cmd.joint_pos[env_ids].clone()
        dof_vel = motion_cmd.joint_vel[env_ids].clone()

        sim = self._env.simulator
        sim.dof_pos[env_ids] = dof_pos
        sim.dof_vel[env_ids] = dof_vel
        sim.robot_root_states[env_ids, :3] = root_pos
        sim.robot_root_states[env_ids, 3:7] = root_rot
        sim.robot_root_states[env_ids, 7:10] = root_lin_vel
        sim.robot_root_states[env_ids, 10:13] = root_ang_vel

        if hasattr(sim, "set_actor_root_state_tensor_robots"):
            sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
        else:
            sim.set_actor_root_state_tensor(env_ids, sim.all_root_states)

        if hasattr(sim, "set_dof_state_tensor_robots"):
            sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)
        else:
            sim.set_dof_state_tensor(env_ids, sim.dof_state)

        if motion_cmd.motion.has_object:
            obj_pos = motion_cmd.object_pos_w[env_ids]
            obj_ori = motion_cmd.object_quat_w[env_ids]
            obj_lin_vel = motion_cmd.object_lin_vel_w[env_ids]
            obj_states = torch.cat([obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1)
            sim.set_actor_states([motion_cmd.object_name], env_ids, obj_states)

        if hasattr(sim, "scene") and hasattr(sim.scene, "write_data_to_sim"):
            sim.scene.write_data_to_sim()
        sim.refresh_sim_tensors()

        motion_cmd._update_future_target_poses()
        if hasattr(self._env, "_refresh_envs_after_reset"):
            self._env._refresh_envs_after_reset(env_ids)

        if hasattr(self._env, "reset_buf"):
            self._env.reset_buf[env_ids] = 0
        if hasattr(self._env, "time_out_buf"):
            self._env.time_out_buf[env_ids] = 0

        self._env._compute_observations()
        self._env._post_compute_observations_callback()
        self._env._clip_observations()

    def _resolve_env_origin(self) -> np.ndarray | None:
        motion_cmd = self._get_motion_command()
        if motion_cmd is not None and hasattr(motion_cmd, "_get_env_offsets"):
            try:
                device = getattr(motion_cmd, "device", None) or getattr(self._env, "device", "cpu")
                env_ids = torch.tensor([self._env_id], device=device, dtype=torch.long)
                offsets = motion_cmd._get_env_offsets(env_ids)
                if offsets is not None:
                    return offsets[0].detach().cpu().numpy()
            except Exception:
                pass

        origins = None
        terrain_mgr = getattr(self._env, "terrain_manager", None)
        if terrain_mgr is not None and hasattr(terrain_mgr, "get_state"):
            terrain_state = terrain_mgr.get_state("locomotion_terrain")
            if terrain_state is not None and hasattr(terrain_state, "env_origins"):
                origins = terrain_state.env_origins
        if origins is None:
            scene = getattr(self._env.simulator, "scene", None)
            if scene is not None and hasattr(scene, "env_origins"):
                origins = scene.env_origins
        if origins is None:
            return None
        try:
            origin = origins[self._env_id]
        except Exception:
            return None
        if isinstance(origin, torch.Tensor):
            return origin.detach().cpu().numpy()
        return np.asarray(origin, dtype=np.float32)

    def _get_root_state_wxyz(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        root_states = getattr(self._env.simulator, "robot_root_states", None)
        if root_states is None:
            return None, None

        if hasattr(root_states, "tensor_wxyz"):
            root = root_states.tensor_wxyz[self._env_id]
            pos = root[0:3]
            quat_wxyz = root[3:7]
        else:
            root = root_states[self._env_id]
            pos = root[0:3]
            quat_xyzw = root[3:7]
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]

        return pos.detach().cpu().numpy(), quat_wxyz.detach().cpu().numpy()

    def _get_dof_pos(self) -> np.ndarray | None:
        dof_pos = getattr(self._env.simulator, "dof_pos", None)
        if dof_pos is None:
            return None
        return dof_pos[self._env_id].detach().cpu().numpy()

    def _get_object_state_wxyz(self) -> tuple[np.ndarray, np.ndarray] | None:
        if not getattr(self._env.robot_config, "object", None):
            return None
        if not getattr(self._env.robot_config.object, "object_urdf_path", None):
            return None

        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        sim = self._env.simulator
        states = None
        if hasattr(sim, "_get_object_states"):
            try:
                states = sim._get_object_states("object", env_ids)
            except Exception:
                states = None
        if states is None and hasattr(sim, "get_actor_states") and getattr(sim, "has_scene_objects", False):
            try:
                states = sim.get_actor_states(["object"], env_ids)
            except Exception:
                states = None
        if states is None or states.numel() == 0:
            return None

        state = states[0]
        pos = state[0:3]
        quat_xyzw = state[3:7]
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        return pos.detach().cpu().numpy(), quat_wxyz.detach().cpu().numpy()

    def _update_scandots(self, offset: np.ndarray) -> None:
        if not self._server or not self._scandots_enabled:
            return

        perception_mgr = getattr(self._env, "perception_manager", None)
        if perception_mgr is None:
            if not self._scandots_warned:
                logger.warning("Viser scandots requested but perception_manager is unavailable.")
                self._scandots_warned = True
            return

        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        output_mode = getattr(getattr(perception_mgr, "cfg", None), "output_mode", None)
        use_heightmap = output_mode == "heightmap"
        include_misses_env = os.environ.get("VISER_SCANDOTS_INCLUDE_MISSES")
        if include_misses_env is None:
            include_misses = False
        else:
            include_misses = include_misses_env.lower() not in (
                "0",
                "false",
                "no",
            )
        try:
            with torch.no_grad():
                if use_heightmap and hasattr(perception_mgr, "get_heightmap_points"):
                    result = perception_mgr.get_heightmap_points(
                        env_ids,
                        include_misses=include_misses,
                        return_rays=True,
                    )
                else:
                    result = perception_mgr.get_camera_scandots_points(
                        env_ids,
                        include_misses=include_misses,
                        return_rays=True,
                    )
        except Exception as exc:
            if not self._scandots_warned:
                logger.warning("Viser scandots disabled: {}", exc)
                self._scandots_warned = True
            return

        if result is None:
            if not self._scandots_warned:
                if use_heightmap:
                    logger.warning("Viser scandots disabled: heightmap points are unavailable.")
                else:
                    logger.warning("Viser scandots disabled: perception is not using mesh_raycast_scandots.")
                self._scandots_warned = True
            self._scandots_enabled = False
            if self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return

        if not isinstance(result, tuple) or len(result) < 2:
            return
        points = result[0]
        mask = result[1]
        ray_starts = result[2] if len(result) > 2 else None
        ray_hits_world = result[3] if len(result) > 3 else None
        if points.numel() == 0:
            if self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return
        points_env = points[0]
        mask_env = mask[0] if mask is not None else None
        if not include_misses and mask_env is not None and mask_env.numel() > 0:
            points_env = points_env[mask_env]
        if points_env.numel() == 0:
            if self._scandots_handle is not None:
                self._scandots_handle.visible = False
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return

        pts = points_env.detach().cpu().numpy()
        if self._recenter:
            pts = pts - offset

        if self._scandots_handle is None:
            self._scandots_handle = self._server.scene.add_point_cloud(
                "/scandots",
                points=pts.astype(np.float32, copy=False),
                colors=self._scandots_color,
                point_size=float(self._scandots_point_size),
                point_shape="circle",
            )
        else:
            self._scandots_handle.visible = True
            self._scandots_handle.points = pts.astype(np.float32, copy=False)

        if ray_starts is None or ray_hits_world is None:
            return
        starts_env = ray_starts[0]
        hits_env = ray_hits_world[0]
        if mask_env is not None and mask_env.numel() > 0:
            starts_env = starts_env[mask_env]
            hits_env = hits_env[mask_env]
        if starts_env.numel() == 0:
            if self._scandots_rays_handle is not None:
                self._scandots_rays_handle.visible = False
            return

        lines = torch.stack([starts_env, hits_env], dim=1).detach().cpu().numpy()
        if self._recenter:
            lines = lines - offset[None, None, :]
        colors = np.tile(self._scandots_color, (lines.shape[0], 2, 1))
        if self._scandots_rays_handle is None:
            self._scandots_rays_handle = self._server.scene.add_line_segments(
                "/scandots_rays",
                points=lines.astype(np.float32, copy=False),
                colors=colors.astype(np.uint8, copy=False),
                line_width=1.0,
            )
        else:
            self._scandots_rays_handle.visible = True
            self._scandots_rays_handle.points = lines.astype(np.float32, copy=False)
            try:
                self._scandots_rays_handle.colors = colors.astype(np.uint8, copy=False)
            except Exception:
                pass

    def _update_perception_visuals(self, offset: np.ndarray) -> None:
        if not self._server or not self._perception_enabled:
            return
        perception_mgr = self._get_perception_manager()
        if perception_mgr is None:
            return
        cfg = getattr(perception_mgr, "cfg", None)
        if cfg is None:
            return

        output_mode = str(getattr(cfg, "output_mode", ""))
        env_ids = torch.tensor([self._env_id], device=self._env.device, dtype=torch.long)
        depth_map = None
        near = float(getattr(cfg, "camera_near", 0.0))
        far = float(getattr(cfg, "max_distance", 10.0))
        cam_pos = None
        cam_quat_xyzw = None

        if output_mode == "camera_depth":
            try:
                depth = perception_mgr.get_camera_depth_map()
            except Exception:
                depth = None
            if isinstance(depth, torch.Tensor) and depth.numel() > 0:
                depth_map = depth[self._env_id].detach().cpu().numpy()
                depth_map = np.flipud(depth_map)
            try:
                cam_pos_t, cam_quat_t = perception_mgr.get_camera_pose(
                    env_ids,
                    apply_sensor_offset=True,
                    apply_pitch=True,
                )
                cam_pos = cam_pos_t[0].detach().cpu().numpy()
                cam_quat_xyzw = cam_quat_t[0].detach().cpu()
            except Exception:
                cam_pos = None
                cam_quat_xyzw = None
        elif output_mode == "heightmap":
            near = 0.0
            try:
                if hasattr(perception_mgr, "get_heightmap_map"):
                    hm = perception_mgr.get_heightmap_map(env_ids)
                else:
                    hm = getattr(perception_mgr, "_heightmap", None)
            except Exception:
                hm = None
            if isinstance(hm, torch.Tensor) and hm.numel() > 0:
                if hm.ndim == 3:
                    depth_map = hm[0].detach().cpu().numpy()
                elif hm.ndim == 2:
                    depth_map = hm.detach().cpu().numpy()
            try:
                pose = perception_mgr.get_heightmap_pose(
                    env_ids,
                    apply_offsets=False,
                    apply_heading_only=True,
                )
            except Exception:
                pose = None
            if pose is not None:
                cam_pos_t, body_quat = pose
                forward_world = quat_apply(
                    body_quat,
                    torch.tensor([0.0, 0.0, -1.0], device=self._env.device),
                    w_last=True,
                )
                up_hint = quat_apply(
                    body_quat,
                    torch.tensor([1.0, 0.0, 0.0], device=self._env.device),
                    w_last=True,
                )
                cam_quat_t = _quat_from_forward_up(forward_world, up_hint)
                cam_pos = cam_pos_t[0].detach().cpu().numpy()
                cam_quat_xyzw = cam_quat_t.detach().cpu()

        self._update_perception_markers(perception_mgr, env_ids, offset)

        if depth_map is None:
            return

        depth_shape = (int(depth_map.shape[0]), int(depth_map.shape[1]))
        if depth_shape != self._perception_last_shape or output_mode != self._perception_last_mode:
            self._perception_last_shape = depth_shape
            self._perception_last_mode = output_mode
            if self._perception_depth_handle is not None:
                self._perception_depth_handle.image = np.zeros(depth_map.shape + (3,), dtype=np.uint8)

        depth_img = _depth_to_rgb(depth_map, near, far)
        if self._perception_show_depth_cb is None or bool(self._perception_show_depth_cb.value):
            if self._perception_depth_handle is not None:
                self._perception_depth_handle.image = depth_img
        if self._perception_stats is not None:
            min_d, max_d, count = _valid_depth_stats(depth_map, near, far)
            if count == 0:
                self._perception_stats.content = "Depth range (valid): n/a (no hits)"
            else:
                total = depth_map.size
                self._perception_stats.content = (
                    f"Depth range (valid): {min_d:.3f} - {max_d:.3f} m | valid: {count}/{total}"
                )

        if cam_pos is None or cam_quat_xyzw is None:
            return

        cam_pos = cam_pos - offset
        cam_quat_wxyz = cam_quat_xyzw.detach().cpu().numpy()[[3, 0, 1, 2]]
        frustum_quat_wxyz = _frustum_quat_from_camera(cam_quat_xyzw).detach().cpu().numpy()

        if self._perception_frame is not None:
            self._perception_frame.position = cam_pos
            self._perception_frame.wxyz = cam_quat_wxyz

        if self._perception_frustum is None:
            return

        if output_mode == "heightmap":
            fov, aspect = self._resolve_heightmap_fov_aspect(perception_mgr)
        else:
            fov = float(getattr(cfg, "camera_vfov_deg", 90.0))
            aspect = float(depth_shape[1] / max(1, depth_shape[0]))

        if fov != self._perception_last_fov:
            try:
                self._perception_frustum.fov = fov
            except Exception:
                pass
            self._perception_last_fov = fov
        if aspect != self._perception_last_aspect:
            try:
                self._perception_frustum.aspect = aspect
            except Exception:
                pass
            self._perception_last_aspect = aspect

        self._perception_frustum.position = cam_pos
        self._perception_frustum.wxyz = frustum_quat_wxyz
        if self._perception_show_frustum_cb is None or bool(self._perception_show_frustum_cb.value):
            self._perception_frustum.visible = True
        if self._perception_show_depth_cb is None or bool(self._perception_show_depth_cb.value):
            self._perception_frustum.image = depth_img

    def _update_target_keypoints(self, offset: np.ndarray) -> None:
        if not self._server:
            return
        motion_cmd = self._get_motion_command()
        if motion_cmd is None or not hasattr(motion_cmd, "body_pos_w"):
            return
        try:
            points = motion_cmd.body_pos_w
            pts_env = points[self._env_id]
        except Exception:
            return

        pts = pts_env.detach().cpu().numpy()
        if self._recenter:
            pts = pts - offset

        if pts.size == 0:
            return

        if self._target_keypoints_handle is None:
            colors = np.tile(self._target_keypoints_color, (pts.shape[0], 1))
            self._target_keypoints_handle = self._server.scene.add_point_cloud(
                "/target_keypoints",
                points=pts.astype(np.float32, copy=False),
                colors=colors.astype(np.uint8, copy=False),
                point_size=float(self._target_keypoints_point_size),
                point_shape="circle",
                precision="float32",
            )
        else:
            self._target_keypoints_handle.points = pts.astype(np.float32, copy=False)
            if getattr(self._target_keypoints_handle, "colors", None) is not None:
                colors = np.tile(self._target_keypoints_color, (pts.shape[0], 1))
                try:
                    self._target_keypoints_handle.colors = colors.astype(np.uint8, copy=False)
                except Exception:
                    pass

    def _ensure_marker_handle(
        self,
        name: str,
        color: tuple[int, int, int],
        radius: float,
    ) -> tuple[object | None, bool]:
        if self._server is None:
            return None, False
        mesh = _make_marker_mesh(color, radius)
        if mesh is None:
            handle = self._server.scene.add_point_cloud(
                name,
                points=np.zeros((1, 3), dtype=np.float32),
                colors=np.array([color], dtype=np.uint8),
                point_size=max(0.005, radius * 2.0),
                point_shape="circle",
                precision="float32",
            )
            return handle, True
        handle = self._server.scene.add_mesh_trimesh(
            name,
            mesh,
            cast_shadow=False,
            receive_shadow=False,
        )
        return handle, False

    def _update_perception_markers(
        self,
        perception_mgr: Any,
        env_ids: torch.Tensor,
        offset: np.ndarray,
    ) -> None:
        cfg = getattr(perception_mgr, "cfg", None)
        if cfg is None:
            return

        if getattr(cfg, "heightmap_body_name", None):
            show = True if self._perception_show_heightmap_joint_cb is None else bool(
                self._perception_show_heightmap_joint_cb.value
            )
            if show:
                try:
                    pose = perception_mgr.get_heightmap_pose(
                        env_ids,
                        apply_offsets=False,
                        apply_heading_only=False,
                    )
                except Exception:
                    pose = None
                if pose is not None:
                    pos_t, _ = pose
                    pos = pos_t[0].detach().cpu().numpy() - offset
                    if self._heightmap_marker_handle is None:
                        handle, is_pc = self._ensure_marker_handle(
                            "/heightmap_joint_marker",
                            HEIGHTMAP_MARKER_COLOR,
                            SENSOR_MARKER_RADIUS,
                        )
                        self._heightmap_marker_handle = handle
                        self._heightmap_marker_is_pc = is_pc
                    if self._heightmap_marker_handle is not None:
                        self._heightmap_marker_handle.visible = True
                        if self._heightmap_marker_is_pc:
                            self._heightmap_marker_handle.points = pos.reshape(1, 3).astype(np.float32)
                        else:
                            self._heightmap_marker_handle.position = pos
            elif self._heightmap_marker_handle is not None:
                self._heightmap_marker_handle.visible = False

        if getattr(cfg, "camera_body_name", None) and getattr(cfg, "output_mode", "") == "camera_depth":
            show = True if self._perception_show_camera_joint_cb is None else bool(
                self._perception_show_camera_joint_cb.value
            )
            if show:
                try:
                    cam_pos_t, _ = perception_mgr.get_camera_pose(
                        env_ids,
                        apply_sensor_offset=False,
                        apply_pitch=False,
                    )
                except Exception:
                    cam_pos_t = None
                if cam_pos_t is not None:
                    pos = cam_pos_t[0].detach().cpu().numpy() - offset
                    if self._camera_marker_handle is None:
                        handle, is_pc = self._ensure_marker_handle(
                            "/camera_joint_marker",
                            CAMERA_MARKER_COLOR,
                            SENSOR_MARKER_RADIUS,
                        )
                        self._camera_marker_handle = handle
                        self._camera_marker_is_pc = is_pc
                    if self._camera_marker_handle is not None:
                        self._camera_marker_handle.visible = True
                        if self._camera_marker_is_pc:
                            self._camera_marker_handle.points = pos.reshape(1, 3).astype(np.float32)
                        else:
                            self._camera_marker_handle.position = pos
            elif self._camera_marker_handle is not None:
                self._camera_marker_handle.visible = False

    def _update_contact_forces(self, offset: np.ndarray) -> None:
        if not self._server:
            return
        if self._contact_force_cb is None or not bool(self._contact_force_cb.value):
            if self._contact_force_handle is not None:
                self._contact_force_handle.visible = False
            return
        sim = getattr(self._env, "simulator", None)
        if sim is None:
            return
        forces = getattr(sim, "contact_forces", None)
        positions = getattr(sim, "_rigid_body_pos", None)
        if forces is None or positions is None:
            return
        if not isinstance(forces, torch.Tensor) or not isinstance(positions, torch.Tensor):
            return
        if forces.numel() == 0 or positions.numel() == 0:
            return
        if self._env_id < 0 or self._env_id >= forces.shape[0]:
            return

        forces_env = forces[self._env_id]
        positions_env = positions[self._env_id]
        body_count = min(forces_env.shape[0], positions_env.shape[0])
        if body_count == 0:
            return

        forces_env = forces_env[:body_count]
        positions_env = positions_env[:body_count]

        magnitudes = torch.linalg.norm(forces_env, dim=-1)
        threshold = float(getattr(self._env.simulator.simulator_config, "contact_force_viz_threshold", 1.0))
        mask = magnitudes > threshold
        if torch.count_nonzero(mask) == 0:
            if self._contact_force_handle is not None:
                self._contact_force_handle.visible = False
            return

        scale = float(getattr(self._env.simulator.simulator_config, "contact_force_viz_scale", 0.001))
        forces_np = forces_env.detach().cpu().numpy()
        positions_np = positions_env.detach().cpu().numpy()
        mask_np = mask.detach().cpu().numpy()

        positions_np = positions_np[mask_np]
        forces_np = forces_np[mask_np]
        magnitudes_np = np.linalg.norm(forces_np, axis=1, keepdims=True)
        directions = np.zeros_like(forces_np)
        nonzero = magnitudes_np > 1.0e-6
        directions[nonzero[:, 0]] = forces_np[nonzero[:, 0]] / magnitudes_np[nonzero[:, 0]]
        arrow_ends = positions_np + directions * magnitudes_np * scale

        points = np.stack([positions_np, arrow_ends], axis=1)
        if self._recenter:
            points = points - offset[None, None, :]
        colors = np.full((points.shape[0], 2, 3), [255, 0, 0], dtype=np.uint8)

        if self._contact_force_handle is None:
            self._contact_force_handle = self._server.scene.add_line_segments(
                "/contact_force_arrows",
                points=points,
                colors=colors,
                line_width=2.0,
            )
        else:
            self._contact_force_handle.visible = True
            self._contact_force_handle.points = points
            self._contact_force_handle.colors = colors


def _normalize_env_ids(env_ids) -> list[int]:
    if isinstance(env_ids, torch.Tensor):
        return [int(idx.item()) for idx in env_ids.flatten()]
    if isinstance(env_ids, np.ndarray):
        return [int(idx) for idx in env_ids.flatten().tolist()]
    return [int(idx) for idx in env_ids]
