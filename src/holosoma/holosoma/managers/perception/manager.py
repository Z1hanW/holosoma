"""Perception manager for heightmap and camera-style observations."""

from __future__ import annotations

from typing import Any
import importlib.util
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from loguru import logger

from holosoma.config_types.perception import PerceptionConfig
from holosoma.utils.camera_utils import build_camera_parameters, resolve_camera_intrinsics
from holosoma.utils import warp_utils
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.object_geometry import load_urdf_box_primitive_metadata
from holosoma.utils.rotations import (
    matrix_to_quaternion,
    quat_apply,
    quat_apply_yaw,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_batched,
    quat_rotate_inverse,
    quat_rotate_inverse_batched,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type
from holosoma.utils.safe_torch_import import torch
import torch.nn.functional as F
from holosoma.utils.urdf_utils import resolve_fixed_link_offset


class PerceptionManager:
    """Compute terrain-aware perception features (heightmap or camera depth)."""

    @staticmethod
    def _parse_debug_float_list_env(name: str, *, expected_len: int) -> list[float] | None:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return None
        text = raw
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        parts = [part.strip() for part in text.split(",") if part.strip()]
        if len(parts) != expected_len:
            raise ValueError(f"{name} expected {expected_len} comma-separated floats, got: {raw}")
        return [float(part) for part in parts]

    @staticmethod
    def _normalize_object_geometry_mode(raw_value: Any) -> str:
        normalized = str(raw_value or "").strip().lower()
        if normalized in {"", "mesh", "urdf", "off", "false", "0", "no"}:
            return "mesh"
        if normalized in {"primitive", "primitives", "box", "cuboid", "on", "true", "1", "yes"}:
            return "primitive"
        raise ValueError(
            "Unsupported perception object_geometry_mode. Supported values: 'mesh' or 'primitive'. "
            f"Got: {raw_value}"
        )

    def __init__(self, cfg: PerceptionConfig | None, env: Any, device: str):
        if cfg is None:
            cfg = PerceptionConfig(enabled=False)
        self.cfg = cfg
        self.env = env
        self.device = device
        self.enabled = bool(cfg.enabled)
        self.num_envs = env.num_envs
        self.logger = getattr(env, "logger", None)

        self._warp_mesh = None
        self._terrain_mesh = None
        self._grid_points_base: torch.Tensor | None = None
        self._ray_dirs_base: torch.Tensor | None = None
        self._camera_ray_dirs_base: torch.Tensor | None = None
        self._camera_scandots_ray_dirs_base: torch.Tensor | None = None
        self._camera_scandots_width: int | None = None
        self._camera_scandots_height: int | None = None
        sensor_offset = torch.tensor(cfg.sensor_offset, device=self.device)
        sensor_offset_override = self._parse_debug_float_list_env(
            "HOLOSOMA_PERCEPTION_SENSOR_OFFSET_OVERRIDE",
            expected_len=3,
        )
        if sensor_offset_override is not None:
            sensor_offset = torch.tensor(sensor_offset_override, device=self.device, dtype=torch.float32)
        sensor_offset_delta = self._parse_debug_float_list_env(
            "HOLOSOMA_PERCEPTION_SENSOR_OFFSET_DELTA",
            expected_len=3,
        )
        if sensor_offset_delta is not None:
            sensor_offset = sensor_offset + torch.tensor(sensor_offset_delta, device=self.device, dtype=torch.float32)
        self._sensor_offset = sensor_offset.to(dtype=torch.float32)
        self._ray_start_offset = torch.tensor([0.0, 0.0, cfg.ray_start_height], device=self.device)
        self._camera_source = cfg.camera_source
        object_geometry_mode_raw = (
            cfg.object_geometry_mode
            if getattr(cfg, "object_geometry_mode", None) is not None
            else os.environ.get("HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE", "")
        )
        self._object_geometry_mode = self._normalize_object_geometry_mode(object_geometry_mode_raw)
        self._camera_body_name = cfg.camera_body_name
        heightmap_body_name = cfg.heightmap_body_name
        if heightmap_body_name is None:
            robot_cfg = getattr(env, "robot_config", None)
            heightmap_body_name = getattr(robot_cfg, "torso_name", None)
        self._heightmap_body_name = heightmap_body_name
        self._camera_include_robot_mesh = bool(getattr(cfg, "camera_include_robot_mesh", False))
        include_robot_mesh_env = os.environ.get("HOLOSOMA_PERCEPTION_INCLUDE_ROBOT_MESH")
        if include_robot_mesh_env is not None:
            self._camera_include_robot_mesh = include_robot_mesh_env.strip().lower() not in {
                "",
                "0",
                "false",
                "no",
                "off",
            }
        self._camera_robot_mesh_enabled = False
        self._camera_body_index: int | None = None
        self._camera_body_offset_pos = torch.zeros(3, device=self.device)
        self._camera_body_offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._heightmap_body_index: int | None = None
        self._heightmap_body_offset_pos = torch.zeros(3, device=self.device)
        self._heightmap_body_offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._rendered_camera = None
        self._rendered_camera_env_id = int(getattr(cfg, "camera_env_id", 0))
        self._pytorch3d_mesh = None
        self._pytorch3d_mesh_cache: dict[int, Any] = {}
        self._pytorch3d_raster_settings = None
        self._terrain_vertices: np.ndarray | None = None
        self._terrain_faces: np.ndarray | None = None
        self._robot_link_meshes: list[dict[str, torch.Tensor]] = []
        self._camera_warp_mesh = None
        self._warned_robot_mesh = False
        self._far_tracking_camera_sensor: Any = None
        self._far_tracking_tf_apply = None
        self._far_tracking_quat_mul = None
        self._far_tracking_robot_slot_indices: torch.Tensor | None = None
        self._far_tracking_robot_body_indices: torch.Tensor | None = None
        self._far_tracking_object_slot_indices: torch.Tensor | None = None
        self._far_tracking_object_source_indices: torch.Tensor | None = None
        self._far_tracking_primitive_source_indices: torch.Tensor | None = None
        self._far_tracking_object_names: list[str] = []
        self._far_tracking_object_active_env_ids: list[torch.Tensor | None] = []
        self._far_tracking_base_link_indices: torch.Tensor | None = None
        self._shared_camera_sensor_local_position: torch.Tensor | None = None
        self._shared_camera_sensor_local_orientation: torch.Tensor | None = None
        self._shared_camera_sensor_data_frame_quat: torch.Tensor | None = None
        self._registered_object_mesh_cache: dict[str, str] = {}
        self._camera_mount_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._use_camera_mount_quat = False
        self._camera_frame_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._use_camera_frame_quat = False
        cfg_strict_warp = getattr(cfg, "camera_strict_warp", None)
        if cfg_strict_warp is None:
            strict_warp_raw = os.environ.get("HOLOSOMA_CAMERA_STRICT_WARP", "0").strip().lower()
            self._camera_strict_warp = strict_warp_raw not in {"0", "false", "no", "off", ""}
        else:
            self._camera_strict_warp = bool(cfg_strict_warp)
        self._camera_ray_correction_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
        cfg_auto_fix_backward = getattr(cfg, "camera_auto_fix_backward", None)
        if cfg_auto_fix_backward is None:
            auto_fix_raw = os.environ.get("HOLOSOMA_CAMERA_AUTOFIX_BACKWARD", "0").strip().lower()
            auto_fix_enabled = auto_fix_raw not in {"0", "false", "no", "off", ""}
        else:
            auto_fix_enabled = bool(cfg_auto_fix_backward)
        self._camera_auto_fix_backward = auto_fix_enabled and (not self._camera_strict_warp)
        disable_offsets_raw = os.environ.get("HOLOSOMA_CAMERA_DISABLE_OFFSETS", "0").strip().lower()
        self._camera_disable_offsets = disable_offsets_raw not in {"0", "false", "no", "off", ""}
        threshold_raw = os.environ.get("HOLOSOMA_CAMERA_BACKWARD_RATIO_THRESHOLD", "0.6").strip()
        try:
            self._camera_backward_ratio_threshold = float(threshold_raw)
        except Exception:
            self._camera_backward_ratio_threshold = 0.6
        runtime_log_every_raw = os.environ.get("HOLOSOMA_CAMERA_LOG_ROOT_BACK_EVERY", "0").strip()
        try:
            self._camera_runtime_log_every = max(0, int(runtime_log_every_raw))
        except Exception:
            self._camera_runtime_log_every = 0
        runtime_warn_ratio_raw = os.environ.get("HOLOSOMA_CAMERA_WARN_ROOT_BACK_RATIO", "0.6").strip()
        try:
            self._camera_runtime_warn_ratio = float(runtime_warn_ratio_raw)
        except Exception:
            self._camera_runtime_warn_ratio = 0.6
        self._camera_runtime_log_counter = 0
        extra_yaw_raw = os.environ.get("HOLOSOMA_CAMERA_EXTRA_YAW_DEG", "0.0").strip()
        try:
            extra_yaw_deg = float(extra_yaw_raw)
        except Exception:
            extra_yaw_deg = 0.0
        if abs(extra_yaw_deg) > 1.0e-6:
            extra_yaw_rad = torch.deg2rad(torch.tensor(extra_yaw_deg, device=self.device, dtype=torch.float32))
            self._camera_ray_correction_quat = quat_from_euler_xyz(
                torch.tensor(0.0, device=self.device, dtype=torch.float32),
                torch.tensor(0.0, device=self.device, dtype=torch.float32),
                extra_yaw_rad,
            ).to(device=self.device, dtype=torch.float32)
        cfg_frame_quat = getattr(cfg, "camera_frame_quat", None)
        if cfg_frame_quat is not None:
            try:
                quat_vals = [float(v) for v in cfg_frame_quat]
            except Exception:
                quat_vals = None
            if quat_vals and len(quat_vals) == 4:
                self._camera_frame_quat = torch.tensor(quat_vals, device=self.device, dtype=torch.float32)
                self._use_camera_frame_quat = True
        cfg_mount_quat = getattr(cfg, "camera_mount_quat", None)
        if cfg_mount_quat is not None:
            try:
                quat_vals = [float(v) for v in cfg_mount_quat]
            except Exception:
                quat_vals = None
            if quat_vals and len(quat_vals) == 4:
                self._camera_mount_quat = torch.tensor(quat_vals, device=self.device, dtype=torch.float32)
                self._use_camera_mount_quat = True

        if cfg.output_mode not in {"heightmap", "camera_depth"}:
            raise ValueError(f"Unsupported output_mode: {cfg.output_mode}")
        supported_camera_sources = {"far_tracking_warp", "rendered", "rendered_depth_sensor"}
        if cfg.output_mode == "camera_depth" and self._camera_source not in supported_camera_sources:
            raise ValueError(
                "Unsupported camera_source. Supported camera_depth sources: "
                "'far_tracking_warp', 'rendered', 'rendered_depth_sensor'."
            )

        self._camera_width, self._camera_height = self._resolve_camera_resolution()
        fx, fy, cx, cy, vfov, hfov = resolve_camera_intrinsics(
            self._camera_width,
            self._camera_height,
            vfov_deg=cfg.camera_vfov_deg,
            hfov_deg=cfg.camera_hfov_deg,
            fx=cfg.camera_fx,
            fy=cfg.camera_fy,
            cx=cfg.camera_cx,
            cy=cfg.camera_cy,
        )
        self._camera_fx = torch.tensor(fx, device=self.device)
        self._camera_fy = torch.tensor(fy, device=self.device)
        self._camera_cx = torch.tensor(cx, device=self.device)
        self._camera_cy = torch.tensor(cy, device=self.device)
        self._camera_vfov_deg = vfov
        self._camera_hfov_deg = hfov
        self._camera_num_points = self._camera_width * self._camera_height
        self._camera_warp_preprocess = (
            self.cfg.output_mode == "camera_depth" and bool(getattr(self.cfg, "camera_warp_preprocess", False))
        )
        self._camera_warp_freq_ratio = max(1, int(getattr(self.cfg, "camera_warp_freq_ratio", 1) or 1))
        self._camera_warp_latency_frame = max(0, int(getattr(self.cfg, "camera_warp_latency_frame", 0) or 0))
        self._camera_warp_buffer_len = max(1, int(getattr(self.cfg, "camera_warp_buffer_len", 1) or 1))
        if self._camera_warp_latency_frame >= self._camera_warp_buffer_len:
            raise ValueError("camera_warp_latency_frame must be smaller than camera_warp_buffer_len.")

        self._camera_warp_crop_top = max(0, int(getattr(self.cfg, "camera_warp_crop_top", 0) or 0))
        self._camera_warp_crop_bottom = max(0, int(getattr(self.cfg, "camera_warp_crop_bottom", 0) or 0))
        self._camera_warp_crop_left = max(0, int(getattr(self.cfg, "camera_warp_crop_left", 0) or 0))
        self._camera_warp_crop_right = max(0, int(getattr(self.cfg, "camera_warp_crop_right", 0) or 0))

        resize_cfg = getattr(self.cfg, "camera_warp_resize", None)
        self._camera_warp_resize: tuple[int, int] | None = None
        if resize_cfg is not None:
            try:
                resize_h, resize_w = resize_cfg
                self._camera_warp_resize = (max(1, int(resize_h)), max(1, int(resize_w)))
            except Exception as exc:
                raise ValueError("camera_warp_resize must be a (height, width) tuple when provided.") from exc

        self._camera_warp_min_valid_depth = float(getattr(self.cfg, "camera_warp_min_valid_depth", 0.15) or 0.15)
        self._camera_warp_normalize = bool(getattr(self.cfg, "camera_warp_normalize", False))
        self._camera_warp_edge_noise = bool(getattr(self.cfg, "camera_warp_edge_noise", False))
        self._camera_warp_edge_border = max(0, int(getattr(self.cfg, "camera_warp_edge_border", 3) or 0))
        self._camera_warp_edge_shuffle_prob = float(getattr(self.cfg, "camera_warp_edge_shuffle_prob", 0.9) or 0.0)
        self._camera_warp_edge_empty_prob = float(getattr(self.cfg, "camera_warp_edge_empty_prob", 0.7) or 0.0)
        self._camera_warp_edge_thresh_primary = float(
            getattr(self.cfg, "camera_warp_edge_thresh_primary", 1.0) or 0.0
        )
        self._camera_warp_edge_thresh_secondary = float(
            getattr(self.cfg, "camera_warp_edge_thresh_secondary", 0.6) or 0.0
        )
        self._camera_warp_edge_far_depth_thresh = float(
            getattr(self.cfg, "camera_warp_edge_far_depth_thresh", 2.5) or 0.0
        )
        self._camera_warp_enable_holes = bool(getattr(self.cfg, "camera_warp_enable_holes", False))
        self._camera_warp_hole_prob = float(getattr(self.cfg, "camera_warp_hole_prob", 0.0) or 0.0)
        self._camera_apply_sensor_noise = bool(getattr(self.cfg, "camera_apply_sensor_noise", True))

        self._camera_obs_height, self._camera_obs_width = self._resolve_camera_obs_resolution()
        self._camera_obs_fill_value = self._camera_obs_default_fill_value()
        self._camera_obs_step_counter = 0
        self._camera_warp_sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self._camera_warp_sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3)

        (
            self._heightmap_grid_x,
            self._heightmap_grid_y,
            self._heightmap_interval_x,
            self._heightmap_interval_y,
        ) = self._resolve_heightmap_grid()
        self._num_points = self._heightmap_grid_x * self._heightmap_grid_y
        self._update_interval = 0.0 if cfg.update_hz <= 0 else 1.0 / cfg.update_hz
        self._time_since_update = 0.0

        self._heightmap = torch.zeros(
            self.num_envs,
            self._heightmap_grid_x,
            self._heightmap_grid_y,
            device=self.device,
        )
        self._camera_depth = torch.full(
            (self.num_envs, self._camera_height, self._camera_width),
            cfg.max_distance,
            device=self.device,
        )
        self._camera_depth_obs = torch.full(
            (self.num_envs, self._camera_obs_height, self._camera_obs_width),
            self._camera_obs_fill_value,
            device=self.device,
        )
        self._camera_depth_buffer = torch.full(
            (self.num_envs, self._camera_warp_buffer_len, self._camera_obs_height, self._camera_obs_width),
            self._camera_obs_fill_value,
            device=self.device,
        )
        self._camera_depth_buffer_ready = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._warned_invalid_rendered_depth = False
        self._debug_dump_dir = os.environ.get("HOLOSOMA_PERCEPTION_DEBUG_DUMP_DIR", "").strip()
        try:
            self._debug_dump_after_updates = max(
                1, int(os.environ.get("HOLOSOMA_PERCEPTION_DEBUG_DUMP_AFTER_UPDATES", "1"))
            )
        except Exception:
            self._debug_dump_after_updates = 1
        try:
            self._debug_dump_min_sim_time_ms = max(
                0.0, float(os.environ.get("HOLOSOMA_PERCEPTION_DEBUG_DUMP_MIN_SIM_TIME_MS", "0"))
            )
        except Exception:
            self._debug_dump_min_sim_time_ms = 0.0
        self._debug_update_counter = 0
        self._debug_dump_done = False

        self._ray_hits_world = torch.zeros(self.num_envs, self._num_points, 3, device=self.device)
        self._far_tracking_debug_last: dict[str, torch.Tensor] = {}

    def setup(self) -> None:
        if not self.enabled:
            return
        if (
            self._uses_raycast()
            or self._uses_camera_raycast()
            or self._uses_camera_far_tracking()
            or self._uses_camera_scandots()
            or self._uses_pytorch3d()
        ):
            terrain_term = getattr(self.env, "terrain_manager", None)
            if terrain_term is None or not hasattr(terrain_term, "terrain_term"):
                raise RuntimeError("PerceptionManager requires an initialized terrain_manager.")
            terrain_state = terrain_term.terrain_term
            if self._uses_raycast() or self._uses_camera_raycast() or self._uses_camera_far_tracking() or self._uses_camera_scandots():
                if not hasattr(terrain_state, "warp_mesh"):
                    raise RuntimeError("PerceptionManager requires terrain term with warp_mesh support.")
                self._warp_mesh = terrain_state.warp_mesh
            if self._uses_pytorch3d() or self._uses_camera_far_tracking() or (
                self._camera_include_robot_mesh and (self._uses_camera_raycast() or self._uses_camera_scandots())
            ):
                if not hasattr(terrain_state, "mesh"):
                    raise RuntimeError("PerceptionManager requires terrain term with mesh support.")
                self._terrain_mesh = terrain_state.mesh
                if self._camera_include_robot_mesh and (self._uses_camera_raycast() or self._uses_camera_scandots()):
                    self._terrain_vertices = np.asarray(self._terrain_mesh.vertices, dtype=np.float32)
                    self._terrain_faces = np.asarray(self._terrain_mesh.faces, dtype=np.int64)
                    self._load_robot_link_meshes()

        if self._uses_raycast():
            self._resolve_heightmap_body_index()
            self._grid_points_base, self._ray_dirs_base = self._build_grid()

        if self._uses_camera_raycast():
            self._resolve_camera_body_index()
            self._camera_ray_dirs_base = self._build_camera_rays()

        if self._uses_camera_far_tracking():
            self._setup_far_tracking_camera_sensor()

        if self._wants_camera_scandots():
            self._resolve_camera_body_index()
            self._camera_scandots_ray_dirs_base = self._build_camera_scandots_rays()

        if self.cfg.output_mode == "camera_depth":
            self._maybe_fix_camera_backward()
            self._log_camera_ray_alignment()

        if self._uses_pytorch3d():
            self._resolve_camera_body_index()
            self._setup_pytorch3d_renderer()

        if self._uses_rendered_camera():
            self._setup_rendered_camera()

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if not self.enabled:
            return
        if env_ids is None:
            self._heightmap.zero_()
            self._camera_depth.fill_(self.cfg.max_distance)
            self._camera_depth_obs.fill_(self._camera_obs_fill_value)
            self._camera_depth_buffer.fill_(self._camera_obs_fill_value)
            self._camera_depth_buffer_ready.zero_()
            self._camera_obs_step_counter = 0
            self._ray_hits_world.zero_()
            return
        self._heightmap[env_ids] = 0.0
        self._camera_depth[env_ids] = self.cfg.max_distance
        self._camera_depth_obs[env_ids] = self._camera_obs_fill_value
        self._camera_depth_buffer[env_ids] = self._camera_obs_fill_value
        self._camera_depth_buffer_ready[env_ids] = False
        self._ray_hits_world[env_ids] = 0.0

    def update(self, env_ids: torch.Tensor | None = None) -> None:
        if not self.enabled:
            return
        self._debug_update_counter += 1
        if env_ids is None and self._update_interval > 0.0:
            self._time_since_update += float(self.env.dt)
            if self._time_since_update + 1.0e-8 < self._update_interval:
                return
            self._time_since_update -= self._update_interval

        if self._uses_rendered_camera():
            if self._rendered_camera is None:
                raise RuntimeError("Rendered camera is not initialized; call PerceptionManager.setup().")
            if env_ids is not None and self._rendered_camera_env_id not in env_ids.tolist():
                return
            camera_depth = self._rendered_camera.capture_depth()
            if camera_depth.numel() == 0 or camera_depth.shape[-2:] != (
                self._camera_height,
                self._camera_width,
            ):
                if not self._warned_invalid_rendered_depth:
                    (self.logger or logger).warning(
                        "Rendered depth returned invalid shape %s; filling with max_distance.",
                        tuple(camera_depth.shape),
                    )
                    self._warned_invalid_rendered_depth = True
                camera_depth = torch.full(
                    (1, self._camera_height, self._camera_width),
                    self.cfg.max_distance,
                    device=self.device,
                )
            elif camera_depth.ndim == 2:
                camera_depth = camera_depth.unsqueeze(0)
            camera_depth = self._clamp_camera_depth_to_sensor_range(camera_depth)
            env_id = torch.tensor([self._rendered_camera_env_id], device=self.device, dtype=torch.long)
            self._camera_depth[env_id] = camera_depth
            self._update_camera_depth_observation(
                env_id,
                camera_depth,
                refresh=self._consume_camera_obs_refresh_flag(),
            )
            self._maybe_dump_camera_debug(source_label="rendered", env_ids=env_id)
            self._maybe_log_runtime_camera_alignment()
            return

        if self._uses_pytorch3d():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_pytorch3d_depth(env_ids)
            camera_depth = self._clamp_camera_depth_to_sensor_range(camera_depth)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._consume_camera_obs_refresh_flag(),
            )
            self._maybe_dump_camera_debug(source_label="pytorch3d", env_ids=idx)
            self._maybe_log_runtime_camera_alignment()
            return

        if self._uses_camera_far_tracking():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_far_tracking_camera_depth(env_ids)
            camera_depth = self._clamp_camera_depth_to_sensor_range(camera_depth)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._consume_camera_obs_refresh_flag(),
            )
            self._maybe_dump_camera_debug(source_label="far_tracking_warp", env_ids=idx)
            self._maybe_log_runtime_camera_alignment()
            return

        if self._uses_camera_scandots():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_camera_scandots_depth(env_ids)
            camera_depth = self._clamp_camera_depth_to_sensor_range(camera_depth)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._consume_camera_obs_refresh_flag(),
            )
            self._maybe_dump_camera_debug(source_label="scandots", env_ids=idx)
            self._maybe_log_runtime_camera_alignment()
            return

        if self._uses_camera_raycast():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_camera_raycast_depth(env_ids)
            camera_depth = self._clamp_camera_depth_to_sensor_range(camera_depth)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._consume_camera_obs_refresh_flag(),
            )
            self._maybe_dump_camera_debug(source_label="raycast", env_ids=idx)
            self._maybe_log_runtime_camera_alignment()
            return

        ray_starts, ray_dirs, ray_hits_world, root_pos, base_quat, offset_world = self._compute_rays(env_ids)
        distances = self._compute_ray_distances(ray_starts, ray_dirs, ray_hits_world)
        heightmap = distances.view(-1, self._heightmap_grid_x, self._heightmap_grid_y)

        idx = env_ids if env_ids is not None else slice(None)
        self._heightmap[idx] = heightmap
        self._ray_hits_world[idx] = ray_hits_world

        if self.cfg.output_mode == "camera_depth":
            camera_depth = self._project_to_camera(ray_hits_world, root_pos, base_quat, offset_world)
            camera_depth = self._apply_camera_depth_noise(camera_depth)
            camera_depth = self._clamp_camera_depth_to_sensor_range(camera_depth)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._consume_camera_obs_refresh_flag(),
            )
            self._maybe_dump_camera_debug(source_label="projected", env_ids=idx)
            self._maybe_log_runtime_camera_alignment()

    def get_obs(self) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("Perception is disabled but perception observations were requested.")
        if self.cfg.output_mode == "heightmap":
            offset = float(getattr(self.cfg, "heightmap_obs_offset", 0.0) or 0.0)
            if abs(offset) > 1.0e-8:
                return (self._heightmap - offset).view(self.num_envs, -1)
            return self._heightmap.view(self.num_envs, -1)
        if self.cfg.output_mode == "camera_depth":
            return self._camera_depth_obs.view(self.num_envs, -1)
        raise ValueError(f"Unsupported perception output_mode: {self.cfg.output_mode}")

    def get_camera_depth_map(self) -> torch.Tensor:
        if not self.enabled or self.cfg.output_mode != "camera_depth":
            raise RuntimeError("Camera depth map requested but camera_depth output is disabled.")
        return self._camera_depth

    def get_camera_depth_obs_map(self) -> torch.Tensor:
        if not self.enabled or self.cfg.output_mode != "camera_depth":
            raise RuntimeError("Camera depth observation map requested but camera_depth output is disabled.")
        return self._camera_depth_obs

    def _maybe_dump_camera_debug(self, *, source_label: str, env_ids: torch.Tensor | slice | None) -> None:
        if self.cfg.output_mode != "camera_depth":
            return
        if not self._debug_dump_dir or self._debug_dump_done:
            return
        if self._debug_update_counter < self._debug_dump_after_updates:
            return
        sim_time_ms = None
        try:
            sim_time_ms = float(self.env.simulator.time()) * 1000.0
        except Exception:
            sim_time_ms = None
        if (
            self._debug_dump_min_sim_time_ms > 0.0
            and sim_time_ms is not None
            and sim_time_ms < self._debug_dump_min_sim_time_ms
        ):
            return

        if isinstance(env_ids, torch.Tensor) and env_ids.numel() > 0:
            env_index = int(env_ids.view(-1)[0].item())
        else:
            env_index = 0
        if env_index < 0 or env_index >= self.num_envs:
            env_index = 0

        dump_dir = Path(self._debug_dump_dir).expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)

        raw_depth = self._camera_depth[env_index].detach().cpu().to(torch.float32).numpy()
        obs_depth = self._camera_depth_obs[env_index].detach().cpu().to(torch.float32).numpy()
        np.save(dump_dir / "camera_depth_raw.npy", raw_depth)
        np.save(dump_dir / "camera_depth_obs.npy", obs_depth)

        stats = {
            "source_label": source_label,
            "camera_source": str(self._camera_source),
            "sim_time_ms": sim_time_ms,
            "update_counter": int(self._debug_update_counter),
            "env_index": int(env_index),
            "raw_shape": [int(v) for v in raw_depth.shape],
            "obs_shape": [int(v) for v in obs_depth.shape],
            "camera_width": int(self._camera_width),
            "camera_height": int(self._camera_height),
            "camera_obs_width": int(self._camera_obs_width),
            "camera_obs_height": int(self._camera_obs_height),
            "camera_warp_preprocess": bool(self._camera_warp_preprocess),
            "camera_warp_resize": list(self._camera_warp_resize) if self._camera_warp_resize is not None else None,
            "camera_warp_crop_top": int(self._camera_warp_crop_top),
            "camera_warp_crop_bottom": int(self._camera_warp_crop_bottom),
            "camera_warp_crop_left": int(self._camera_warp_crop_left),
            "camera_warp_crop_right": int(self._camera_warp_crop_right),
            "camera_warp_normalize": bool(self._camera_warp_normalize),
            "raw_min": float(np.nanmin(raw_depth)),
            "raw_max": float(np.nanmax(raw_depth)),
            "obs_min": float(np.nanmin(obs_depth)),
            "obs_max": float(np.nanmax(obs_depth)),
        }
        env_id_t = torch.tensor([env_index], device=self.device, dtype=torch.long)
        try:
            cam_pos_t, cam_quat_t = self.get_camera_pose(env_id_t, apply_sensor_offset=True, apply_pitch=True)
            stats["camera_pose_pos"] = [float(v) for v in cam_pos_t[0].detach().cpu().tolist()]
            stats["camera_pose_quat_xyzw"] = [float(v) for v in cam_quat_t[0].detach().cpu().tolist()]
        except Exception:
            stats["camera_pose_pos"] = None
            stats["camera_pose_quat_xyzw"] = None
        if hasattr(self, "get_mujoco_render_camera_pose"):
            try:
                render_pos_t, render_quat_t = self.get_mujoco_render_camera_pose(env_id_t)
                stats["mujoco_render_camera_pose_pos"] = [float(v) for v in render_pos_t[0].detach().cpu().tolist()]
                stats["mujoco_render_camera_pose_quat_xyzw"] = [
                    float(v) for v in render_quat_t[0].detach().cpu().tolist()
                ]
            except Exception:
                stats["mujoco_render_camera_pose_pos"] = None
                stats["mujoco_render_camera_pose_quat_xyzw"] = None
        if self._camera_strict_warp:
            try:
                strict_pos_t, strict_quat_t = self._get_strict_warp_camera_pose(env_id_t)
                stats["strict_warp_camera_pose_pos"] = [float(v) for v in strict_pos_t[0].detach().cpu().tolist()]
                stats["strict_warp_camera_pose_quat_xyzw"] = [
                    float(v) for v in strict_quat_t[0].detach().cpu().tolist()
                ]
            except Exception:
                stats["strict_warp_camera_pose_pos"] = None
                stats["strict_warp_camera_pose_quat_xyzw"] = None
        if self._terrain_mesh is not None:
            try:
                terrain_bounds = np.asarray(self._terrain_mesh.bounds, dtype=np.float32)
                stats["terrain_mesh_bounds"] = terrain_bounds.tolist()
                stats["terrain_mesh_vertices"] = int(len(self._terrain_mesh.vertices))
                stats["terrain_mesh_faces"] = int(len(self._terrain_mesh.faces))
            except Exception:
                stats["terrain_mesh_bounds"] = None
        if self._far_tracking_camera_sensor is not None:
            try:
                sensor = self._far_tracking_camera_sensor
                stats["far_tracking_sensor_camera_pos"] = [
                    float(v) for v in sensor.camera_sensor_position[env_index, 0].detach().cpu().tolist()
                ]
                stats["far_tracking_sensor_camera_quat_xyzw"] = [
                    float(v) for v in sensor.camera_sensor_orientation[env_index, 0].detach().cpu().tolist()
                ]
                stats["far_tracking_sensor_local_pos"] = [
                    float(v) for v in sensor.camera_sensor_local_position[env_index, 0].detach().cpu().tolist()
                ]
                stats["far_tracking_sensor_local_quat_xyzw"] = [
                    float(v) for v in sensor.camera_sensor_local_orientation[env_index, 0].detach().cpu().tolist()
                ]
                stats["far_tracking_sensor_data_frame_quat_xyzw"] = [
                    float(v) for v in sensor.camera_sensor_data_frame_quat[env_index, 0].detach().cpu().tolist()
                ]
                stats["far_tracking_num_robot_mesh_slots"] = int(getattr(sensor, "num_robot_bodies", 0))
                stats["far_tracking_ray_cast_bodies"] = list(getattr(sensor, "ray_cast_bodies", []))
                stats["far_tracking_primitive_bodies"] = list(getattr(sensor, "primitive_bodies", []))
                if getattr(sensor, "ray_cast_body_poses_tensor", None) is not None:
                    stats["far_tracking_ray_cast_body_poses"] = (
                        sensor.ray_cast_body_poses_tensor[env_index].detach().cpu().to(torch.float32).tolist()
                    )
                    stats["far_tracking_ray_cast_body_quats_xyzw"] = (
                        sensor.ray_cast_body_quats_tensor[env_index].detach().cpu().to(torch.float32).tolist()
                    )
                if getattr(sensor, "primitive_body_poses_tensor", None) is not None:
                    stats["far_tracking_primitive_body_poses"] = (
                        sensor.primitive_body_poses_tensor[env_index].detach().cpu().to(torch.float32).tolist()
                    )
                    stats["far_tracking_primitive_body_quats_xyzw"] = (
                        sensor.primitive_body_quats_tensor[env_index].detach().cpu().to(torch.float32).tolist()
                    )
                    stats["far_tracking_primitive_body_half_extents"] = (
                        sensor.primitive_body_half_extents_tensor[env_index].detach().cpu().to(torch.float32).tolist()
                    )
                    stats["far_tracking_primitive_body_active"] = (
                        sensor.primitive_body_active_tensor[env_index].detach().cpu().to(torch.int32).tolist()
                    )
                for debug_key, debug_value in getattr(self, "_far_tracking_debug_last", {}).items():
                    try:
                        value = debug_value[env_index]
                        stats[f"far_tracking_debug_{debug_key}"] = value.detach().cpu().to(torch.float32).tolist()
                    except Exception:
                        stats[f"far_tracking_debug_{debug_key}"] = None
            except Exception:
                stats["far_tracking_sensor_debug_error"] = True
        (dump_dir / "camera_depth_debug.json").write_text(__import__("json").dumps(stats, indent=2))
        self._debug_dump_done = True
        (self.logger or logger).info("Perception camera debug dump written to {}", dump_dir)

    def get_heightmap_map(self, env_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        if not self.enabled or self.cfg.output_mode != "heightmap":
            return None
        idx = env_ids if env_ids is not None else slice(None)
        return self._heightmap[idx]

    def get_camera_depth_ray_samples(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        include_misses: bool = False,
        return_rays: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        if not self.enabled or self.cfg.output_mode != "camera_depth":
            return None

        if self._uses_camera_raycast():
            ray_starts, ray_dirs_world, ray_hits_world, hit_mask, body_quat = self._cast_camera_raycast_rays(env_ids)
        elif self._uses_camera_scandots():
            ray_starts, ray_dirs_world, ray_hits_world, hit_mask, body_quat = self._cast_camera_scandots_rays(env_ids)
        else:
            raise RuntimeError(
                f"Camera depth ray samples are unavailable for camera_source={self._camera_source}"
            )

        ranges = self._compute_camera_ray_distances(ray_starts, ray_dirs_world, ray_hits_world)
        hit_mask = self._filter_camera_hit_mask_by_depth(ranges, ray_dirs_world, body_quat, hit_mask)

        if include_misses:
            miss_ranges = self._compute_camera_miss_ranges(ray_dirs_world, body_quat)
            draw_ranges = torch.where(hit_mask, ranges, miss_ranges)
            points = ray_starts + ray_dirs_world * draw_ranges.unsqueeze(-1)
        else:
            points = ray_hits_world
        if return_rays:
            return points, hit_mask, ray_starts, ray_dirs_world, ray_hits_world
        return points, hit_mask

    def get_camera_scandots_points(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        include_misses: bool = False,
        return_rays: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        if not self.enabled or self.cfg.output_mode != "camera_depth":
            return None
        ray_starts, ray_dirs_world, ray_hits_world, hit_mask, body_quat = self._cast_camera_scandots_rays(env_ids)

        ranges = self._compute_camera_ray_distances(ray_starts, ray_dirs_world, ray_hits_world)
        hit_mask = self._filter_camera_hit_mask_by_depth(ranges, ray_dirs_world, body_quat, hit_mask)

        if include_misses:
            miss_ranges = self._compute_camera_miss_ranges(ray_dirs_world, body_quat)
            draw_ranges = torch.where(hit_mask, ranges, miss_ranges)
            points = ray_starts + ray_dirs_world * draw_ranges.unsqueeze(-1)
        else:
            points = ray_hits_world
        if return_rays:
            return points, hit_mask, ray_starts, ray_dirs_world, ray_hits_world
        return points, hit_mask

    def get_heightmap_points(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        include_misses: bool = False,
        return_rays: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        | None
    ):
        if not self.enabled or self.cfg.output_mode != "heightmap":
            return None
        if self._grid_points_base is None or self._ray_dirs_base is None:
            raise RuntimeError("PerceptionManager grid buffers are not initialized.")

        idx = env_ids if env_ids is not None else slice(None)
        ray_origin_pos, sample_quat = self._get_heightmap_sampling_pose(
            idx,
            apply_offsets=True,
            apply_heading_only=self.cfg.use_heading_only,
        )
        num_envs = sample_quat.shape[0]

        grid_points = self._grid_points_base.unsqueeze(0).expand(num_envs, -1, -1)
        ray_dirs = self._ray_dirs_base.unsqueeze(0).expand(num_envs, -1, -1)

        quat_repeat = sample_quat.repeat(1, self._num_points)
        grid_world = quat_apply(quat_repeat, grid_points, w_last=True)
        ray_dirs_world = quat_apply(quat_repeat, ray_dirs, w_last=True)

        ray_starts = grid_world + ray_origin_pos.unsqueeze(1)
        ray_hits_world = self._ray_hits_world[idx]
        hit_mask = torch.isfinite(ray_hits_world).all(dim=-1)

        if include_misses:
            distances = self._compute_ray_distances(ray_starts, ray_dirs_world, ray_hits_world)
            points = ray_starts + ray_dirs_world * distances.unsqueeze(-1)
        else:
            points = ray_hits_world
        if return_rays:
            return points, hit_mask, ray_starts, ray_dirs_world, ray_hits_world
        return points, hit_mask

    def capture_rendered_rgb(self) -> Any:
        if not self.enabled or self.cfg.output_mode != "camera_depth":
            raise RuntimeError("RGB capture requested but camera_depth output is disabled.")
        if not self._uses_rendered_camera():
            raise RuntimeError("RGB capture requires camera_source=rendered or rendered_depth_sensor.")
        if self._rendered_camera is None:
            raise RuntimeError("Rendered camera is not initialized; call PerceptionManager.setup().")
        if not hasattr(self._rendered_camera, "capture_rgb"):
            raise RuntimeError("Rendered camera does not support RGB capture.")
        return self._rendered_camera.capture_rgb()

    def get_camera_pose(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        apply_sensor_offset: bool = True,
        apply_pitch: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return camera position and orientation in world frame."""
        if not self.enabled or self.cfg.output_mode != "camera_depth":
            raise RuntimeError("Camera pose requested but camera_depth output is disabled.")

        idx = env_ids if env_ids is not None else slice(None)
        body_pos, body_quat = self._get_camera_body_pose(idx)

        if apply_sensor_offset:
            offset_world = quat_apply(body_quat, self._sensor_offset.expand(body_pos.shape[0], -1), w_last=True)
            body_pos = body_pos + offset_world

        if apply_pitch:
            combo = self._camera_ray_rotation_quat(device=body_quat.device, dtype=body_quat.dtype)
            combo = combo.unsqueeze(0).expand(body_quat.shape[0], -1)
            body_quat = quat_mul(body_quat, combo, w_last=True)

        return body_pos, body_quat

    def _ensure_shared_strict_warp_camera_mount(self) -> None:
        if not self._camera_strict_warp:
            return
        if (
            self._shared_camera_sensor_local_position is not None
            and self._shared_camera_sensor_local_orientation is not None
            and self._shared_camera_sensor_data_frame_quat is not None
        ):
            return

        num_envs = self.num_envs
        sensor_offset = self._sensor_offset.to(device=self.device, dtype=torch.float32).view(1, 1, 3)
        sensor_offset = sensor_offset.expand(num_envs, 1, 3).clone()

        mount_rot_deg = torch.tensor([1.0, 27.0, 1.0], device=self.device, dtype=torch.float32).view(1, 1, 3)
        mount_rot_deg = mount_rot_deg.expand(num_envs, 1, 3).clone()
        data_frame_rot_rad = torch.deg2rad(
            torch.tensor([-90.0, 0.0, -90.0], device=self.device, dtype=torch.float32)
        )
        data_frame_quat = quat_from_euler_xyz(
            data_frame_rot_rad[0],
            data_frame_rot_rad[1],
            data_frame_rot_rad[2],
        ).view(1, 1, 4)
        data_frame_quat = data_frame_quat.expand(num_envs, 1, 4).clone()

        translation_jitter_min = torch.tensor([-0.025, -0.025, -0.025], device=self.device, dtype=torch.float32)
        translation_jitter_max = torch.tensor([0.025, 0.025, 0.025], device=self.device, dtype=torch.float32)
        rotation_jitter_min = torch.tensor([-2.5, -3.0, -2.5], device=self.device, dtype=torch.float32)
        rotation_jitter_max = torch.tensor([2.5, 3.0, 2.5], device=self.device, dtype=torch.float32)
        randomize_mount_raw = os.environ.get("HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT", "0").strip().lower()
        randomize_mount = randomize_mount_raw not in {"0", "false", "no", "off", ""}
        if randomize_mount:
            jitter_translation = translation_jitter_min.view(1, 1, 3) + (
                translation_jitter_max - translation_jitter_min
            ).view(1, 1, 3) * torch.rand((num_envs, 1, 3), device=self.device, dtype=torch.float32)
            local_position = sensor_offset + jitter_translation

            jitter_rotation_deg = rotation_jitter_min.view(1, 1, 3) + (
                rotation_jitter_max - rotation_jitter_min
            ).view(1, 1, 3) * torch.rand((num_envs, 1, 3), device=self.device, dtype=torch.float32)
            local_rotation_deg = mount_rot_deg + jitter_rotation_deg
        else:
            local_position = sensor_offset
            local_rotation_deg = mount_rot_deg

        local_rotation_rad = torch.deg2rad(local_rotation_deg)
        local_orientation = quat_from_euler_xyz(
            local_rotation_rad[..., 0],
            local_rotation_rad[..., 1],
            local_rotation_rad[..., 2],
        )
        pitch_deg = float(getattr(self.cfg, "camera_pitch_deg", 0.0) or 0.0)
        if abs(pitch_deg) > 1.0e-6:
            pitch_rad = torch.deg2rad(torch.tensor(pitch_deg, device=self.device, dtype=torch.float32))
            pitch_quat = quat_from_euler_xyz(
                torch.tensor(0.0, device=self.device, dtype=torch.float32),
                pitch_rad,
                torch.tensor(0.0, device=self.device, dtype=torch.float32),
            ).view(1, 1, 4)
            local_orientation = quat_mul(
                pitch_quat.expand_as(local_orientation),
                local_orientation,
                w_last=True,
            )

        self._shared_camera_sensor_local_position = local_position
        self._shared_camera_sensor_local_orientation = local_orientation
        self._shared_camera_sensor_data_frame_quat = data_frame_quat

    def _get_strict_warp_camera_pose(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        idx = env_ids if env_ids is not None else slice(None)
        body_pos, body_quat = self._get_camera_body_pose(idx)
        num_envs = body_quat.shape[0]

        if self._camera_strict_warp:
            self._ensure_shared_strict_warp_camera_mount()
            if (
                self._shared_camera_sensor_local_position is not None
                and self._shared_camera_sensor_local_orientation is not None
                and self._shared_camera_sensor_data_frame_quat is not None
            ):
                local_position_all = self._shared_camera_sensor_local_position
                local_orientation_all = self._shared_camera_sensor_local_orientation
                data_frame_quat_all = self._shared_camera_sensor_data_frame_quat
                if isinstance(idx, slice):
                    local_position = local_position_all[:, 0]
                    local_orientation = local_orientation_all[:, 0]
                    data_frame_quat = data_frame_quat_all[:, 0]
                else:
                    local_position = local_position_all[idx, 0]
                    local_orientation = local_orientation_all[idx, 0]
                    data_frame_quat = data_frame_quat_all[idx, 0]
            else:
                local_position = self._sensor_offset.expand(num_envs, -1)
                local_orientation = (
                    self._camera_mount_quat.to(device=body_quat.device, dtype=body_quat.dtype)
                    .unsqueeze(0)
                    .expand(num_envs, -1)
                )
                data_frame_quat = (
                    self._camera_frame_quat.to(device=body_quat.device, dtype=body_quat.dtype)
                    .unsqueeze(0)
                    .expand(num_envs, -1)
                )
        else:
            local_position = self._sensor_offset.expand(num_envs, -1)
            local_orientation = (
                self._camera_mount_quat.to(device=body_quat.device, dtype=body_quat.dtype)
                .unsqueeze(0)
                .expand(num_envs, -1)
                if self._use_camera_mount_quat
                else torch.tensor([0.0, 0.0, 0.0, 1.0], device=body_quat.device, dtype=body_quat.dtype)
                .unsqueeze(0)
                .expand(num_envs, -1)
            )
            data_frame_quat = (
                self._camera_frame_quat.to(device=body_quat.device, dtype=body_quat.dtype)
                .unsqueeze(0)
                .expand(num_envs, -1)
                if self._use_camera_frame_quat
                else torch.tensor([0.0, 0.0, 0.0, 1.0], device=body_quat.device, dtype=body_quat.dtype)
                .unsqueeze(0)
                .expand(num_envs, -1)
            )

        local_position = local_position.to(device=body_pos.device, dtype=body_pos.dtype)
        local_orientation = local_orientation.to(device=body_quat.device, dtype=body_quat.dtype)
        data_frame_quat = data_frame_quat.to(device=body_quat.device, dtype=body_quat.dtype)

        camera_pos = body_pos + quat_apply(body_quat, local_position, w_last=True)
        camera_quat = quat_mul(body_quat, quat_mul(local_orientation, data_frame_quat, w_last=True), w_last=True)
        return camera_pos, camera_quat

    def _camera_backward_stats(self, ray_dirs_base: torch.Tensor) -> tuple[float, float]:
        if ray_dirs_base is None or ray_dirs_base.numel() == 0:
            return 0.0, 1.0
        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        _body_pos, body_quat = self._get_camera_body_pose(env_ids)
        ray_dirs_world = quat_rotate_batched(body_quat, ray_dirs_base.unsqueeze(0))[0]
        ray_dirs_world = ray_dirs_world / torch.norm(ray_dirs_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

        root_quat = getattr(self.env, "base_quat", None)
        if isinstance(root_quat, torch.Tensor) and root_quat.ndim >= 2 and root_quat.shape[0] > 0:
            root_quat_env = root_quat[0:1].to(device=ray_dirs_world.device, dtype=ray_dirs_world.dtype)
            root_forward = quat_apply(
                root_quat_env,
                torch.tensor([[1.0, 0.0, 0.0]], device=ray_dirs_world.device, dtype=ray_dirs_world.dtype),
                w_last=True,
            )[0]
        else:
            root_forward = torch.tensor([1.0, 0.0, 0.0], device=ray_dirs_world.device, dtype=ray_dirs_world.dtype)
        root_forward = root_forward / torch.norm(root_forward).clamp(min=1.0e-6)

        dots_root = torch.sum(ray_dirs_world * root_forward.unsqueeze(0), dim=-1)
        back_ratio = float((dots_root <= 0.0).to(torch.float32).mean().item())
        min_dot = float(dots_root.min().item()) if dots_root.numel() > 0 else 1.0
        return back_ratio, min_dot

    def _camera_backward_stats_cam_frame(self, ray_dirs_base: torch.Tensor) -> tuple[float, float]:
        if ray_dirs_base is None or ray_dirs_base.numel() == 0:
            return 0.0, 1.0
        env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
        _body_pos, body_quat = self._get_camera_body_pose(env_ids)
        ray_dirs_world = quat_rotate_batched(body_quat, ray_dirs_base.unsqueeze(0))[0]
        ray_dirs_world = ray_dirs_world / torch.norm(ray_dirs_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

        cam_forward = self._get_camera_forward_axis(body_quat)[0]
        cam_forward = cam_forward / torch.norm(cam_forward).clamp(min=1.0e-6)
        dots_cam = torch.sum(ray_dirs_world * cam_forward.unsqueeze(0), dim=-1)
        back_ratio = float((dots_cam <= 0.0).to(torch.float32).mean().item())
        min_dot = float(dots_cam.min().item()) if dots_cam.numel() > 0 else 1.0
        return back_ratio, min_dot

    def _maybe_log_runtime_camera_alignment(self) -> None:
        if self._camera_runtime_log_every <= 0:
            return
        if self._camera_ray_dirs_base is None or self._camera_ray_dirs_base.numel() == 0:
            return
        self._camera_runtime_log_counter += 1
        if (self._camera_runtime_log_counter % self._camera_runtime_log_every) != 0:
            return
        try:
            root_ratio, root_min_dot = self._camera_backward_stats(self._camera_ray_dirs_base)
            cam_ratio, cam_min_dot = self._camera_backward_stats_cam_frame(self._camera_ray_dirs_base)
        except Exception as exc:
            (self.logger or logger).warning("Camera runtime alignment check skipped: {}", exc)
            return
        warn = (root_ratio > self._camera_runtime_warn_ratio) or (cam_ratio > self._camera_runtime_warn_ratio)
        if warn:
            (self.logger or logger).warning(
                "Camera runtime alignment: step={} root_back_ratio={:.3f} min_dot_root={:.3f} cam_back_ratio={:.3f} min_dot_cam={:.3f}",
                self._camera_runtime_log_counter,
                root_ratio,
                root_min_dot,
                cam_ratio,
                cam_min_dot,
            )
        else:
            (self.logger or logger).info(
                "Camera runtime alignment: step={} root_back_ratio={:.3f} min_dot_root={:.3f} cam_back_ratio={:.3f} min_dot_cam={:.3f}",
                self._camera_runtime_log_counter,
                root_ratio,
                root_min_dot,
                cam_ratio,
                cam_min_dot,
            )

    def _maybe_fix_camera_backward(self) -> None:
        if (
            not self._camera_auto_fix_backward
            or self._camera_ray_dirs_base is None
            or self._camera_ray_dirs_base.numel() == 0
        ):
            return
        try:
            before_ratio, before_min_dot = self._camera_backward_stats(self._camera_ray_dirs_base)
        except Exception as exc:
            (self.logger or logger).warning("Camera backward auto-fix skipped: {}", exc)
            return

        threshold = float(np.clip(self._camera_backward_ratio_threshold, 0.0, 1.0))
        if before_ratio <= threshold:
            return

        # Try a 180 deg yaw correction when most rays point behind root forward.
        old_correction = self._camera_ray_correction_quat.clone()
        yaw_pi = quat_from_euler_xyz(
            torch.tensor(0.0, device=self.device, dtype=torch.float32),
            torch.tensor(0.0, device=self.device, dtype=torch.float32),
            torch.tensor(np.pi, device=self.device, dtype=torch.float32),
        ).to(device=self.device, dtype=torch.float32)
        self._camera_ray_correction_quat = quat_mul(yaw_pi, old_correction, w_last=True)
        candidate_rays = self._build_camera_rays()
        after_ratio, after_min_dot = self._camera_backward_stats(candidate_rays)

        if after_ratio + 1.0e-6 < before_ratio:
            self._camera_ray_dirs_base = candidate_rays
            if self._camera_scandots_ray_dirs_base is not None:
                self._camera_scandots_ray_dirs_base = self._build_camera_scandots_rays()
            (self.logger or logger).warning(
                "Applied camera backward auto-fix: ratio {:.3f}->{:.3f}, min_dot {:.3f}->{:.3f}",
                before_ratio,
                after_ratio,
                before_min_dot,
                after_min_dot,
            )
            return

        self._camera_ray_correction_quat = old_correction
        (self.logger or logger).warning(
            "Camera backward auto-fix attempted but not improved: ratio {:.3f}->{:.3f}",
            before_ratio,
            after_ratio,
        )

    def get_heightmap_pose(
        self,
        env_ids: torch.Tensor | None = None,
        *,
        apply_offsets: bool = True,
        apply_heading_only: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return heightmap ray origin pose in world frame."""
        if not self.enabled or self.cfg.output_mode != "heightmap":
            return None

        idx = env_ids if env_ids is not None else slice(None)
        return self._get_heightmap_sampling_pose(
            idx,
            apply_offsets=apply_offsets,
            apply_heading_only=apply_heading_only and self.cfg.use_heading_only,
        )

    def get_camera_parameters(self, extrinsics: torch.Tensor) -> dict[str, torch.Tensor | float | int]:
        """Return camera parameters for supplied extrinsics (batched)."""
        return build_camera_parameters(
            extrinsics,
            width=self._camera_width,
            height=self._camera_height,
            vfov_deg=self._camera_vfov_deg,
            hfov_deg=self._camera_hfov_deg,
            fx=float(self._camera_fx.item()),
            fy=float(self._camera_fy.item()),
            cx=float(self._camera_cx.item()),
            cy=float(self._camera_cy.item()),
            fps=self.cfg.camera_fps,
            near=self.cfg.camera_near,
            far=self.cfg.camera_far,
            distortion=self.cfg.camera_distortion,
        )

    def _setup_far_tracking_camera_sensor(self) -> None:
        if self._far_tracking_camera_sensor is not None:
            return
        if self._terrain_mesh is None:
            raise RuntimeError("far_tracking_warp camera source requires terrain mesh.")

        # Resolve far-tracking package from common locations; allow override via env.
        module_repo_root = Path(__file__).resolve().parents[5]
        holosoma_root = Path(get_holosoma_root()).resolve()
        far_tracking_override = os.environ.get("HOLOSOMA_FAR_TRACKING_PKG_ROOT", "").strip()
        candidate_roots: list[Path] = []
        if far_tracking_override:
            override_path = Path(far_tracking_override).expanduser().resolve()
            if override_path.name == "whole_body_tracking":
                candidate_roots.append(override_path)
            else:
                candidate_roots.append(override_path / "whole_body_tracking")
        candidate_roots.extend(
            [
                module_repo_root / "far-tracking" / "source" / "whole_body_tracking",
                module_repo_root.parent / "far-tracking" / "source" / "whole_body_tracking",
                holosoma_root / "far-tracking" / "source" / "whole_body_tracking",
                holosoma_root.parent / "far-tracking" / "source" / "whole_body_tracking",
            ]
        )

        far_tracking_pkg_root = next((root for root in candidate_roots if root.exists()), None)
        if far_tracking_pkg_root is not None and str(far_tracking_pkg_root) not in sys.path:
            sys.path.insert(0, str(far_tracking_pkg_root))

        # If external package is unavailable, we will use bundled fallback below.
        external_far_tracking_available = far_tracking_pkg_root is not None or (
            importlib.util.find_spec("whole_body_tracking") is not None
        )
        if not external_far_tracking_available:
            searched = ", ".join(str(path) for path in candidate_roots)
            (self.logger or logger).warning(
                "External far-tracking package not found. Searched: {}. Falling back to bundled implementation.",
                searched,
            )

        try:
            from whole_body_tracking.utils.warp_sensors.camera_sensor import (  # noqa: PLC0415
                CameraSensor as FarTrackingCameraSensor,
            )
            from whole_body_tracking.utils.warp_sensors.sensor_utils import (  # noqa: PLC0415
                quat_mul_xyzw as ft_quat_mul_xyzw,
            )
            from whole_body_tracking.utils.warp_sensors.sensor_utils import (  # noqa: PLC0415
                tf_apply_xyzw as ft_tf_apply_xyzw,
            )
        except ModuleNotFoundError:
            # Internal fallback: keep perception runnable even when external far-tracking
            # package is unavailable on the current node.
            from holosoma.third_party.ft_warp_sensors.camera_sensor import (  # noqa: PLC0415
                CameraSensor as FarTrackingCameraSensor,
            )
            from holosoma.third_party.ft_warp_sensors.sensor_utils import (  # noqa: PLC0415
                quat_mul_xyzw as ft_quat_mul_xyzw,
            )
            from holosoma.third_party.ft_warp_sensors.sensor_utils import (  # noqa: PLC0415
                tf_apply_xyzw as ft_tf_apply_xyzw,
            )
            (self.logger or logger).warning(
                "External far-tracking python package not importable. Using bundled holosoma fallback warp_sensors."
            )

        urdf_path, _asset_root = self._resolve_robot_asset_paths()
        mesh_root = os.path.join(os.path.dirname(urdf_path), "meshes")
        ray_cast_bodies_raw = (
            dict(getattr(self.cfg, "camera_mesh_file_map", None) or {})
            if self._camera_include_robot_mesh
            else {}
        )
        if self._camera_include_robot_mesh and not ray_cast_bodies_raw:
            raise RuntimeError("far_tracking_warp requires perception.camera_mesh_file_map to be populated.")
        ray_cast_bodies: dict[str, str] = {}
        for link_name, mesh_name in ray_cast_bodies_raw.items():
            candidates = [
                str(mesh_name),
                f"{link_name}.STL",
                f"{link_name}.stl",
            ]
            resolved = None
            for candidate in candidates:
                if os.path.isfile(os.path.join(mesh_root, candidate)):
                    resolved = candidate
                    break
            if resolved is None:
                (self.logger or logger).warning(
                    "far_tracking_warp mesh missing for link '{}': '{}' not found under '{}'; skipping link.",
                    link_name,
                    mesh_name,
                    mesh_root,
                )
                continue
            if resolved != mesh_name:
                (self.logger or logger).warning(
                    "far_tracking_warp mesh remap: link '{}' '{}' -> '{}'",
                    link_name,
                    mesh_name,
                    resolved,
                )
            ray_cast_bodies[link_name] = resolved
        (self.logger or logger).info(
            "far_tracking_warp object geometry mode: {}",
            self._object_geometry_mode,
        )

        registered_object_primitives = self._collect_registered_sim_object_primitives()
        # Include all registered non-robot simulator objects (e.g., training box)
        # so they participate in depth raycasting alongside terrain/robot meshes.
        registered_object_meshes = self._collect_registered_sim_object_meshes(
            excluded_object_names=set(registered_object_primitives.keys())
        )
        for slot_name, slot_spec in registered_object_meshes.items():
            if slot_name in ray_cast_bodies:
                continue
            ray_cast_bodies[slot_name] = str(slot_spec["mesh_path"])
        if self._camera_include_robot_mesh and not ray_cast_bodies:
            raise RuntimeError(f"No valid far_tracking_warp ray_cast_bodies found under mesh root: {mesh_root}")
        if not self._camera_include_robot_mesh:
            (self.logger or logger).info("far_tracking_warp robot visual mesh raycast disabled by configuration.")
        if registered_object_primitives:
            primitive_source_names = sorted(
                {str(slot_spec["source_name"]) for slot_spec in registered_object_primitives.values()}
            )
            (self.logger or logger).info(
                "far_tracking_warp: added {} analytic primitive body(ies) from {} object(s): {}",
                len(registered_object_primitives),
                len(primitive_source_names),
                ", ".join(primitive_source_names),
            )
        if registered_object_meshes:
            registered_source_names = sorted(
                {str(slot_spec["source_name"]) for slot_spec in registered_object_meshes.values()}
            )
            (self.logger or logger).info(
                "far_tracking_warp: added {} registered simulator object raycast slot(s) from {} object(s): {}",
                len(registered_object_meshes),
                len(registered_source_names),
                ", ".join(registered_source_names),
            )

        camera_body_name = self._camera_body_name or "torso_link"
        offset_pos = tuple(float(v) for v in self._sensor_offset.detach().cpu().tolist())
        # far-tracking G1FlatRsD435iConfig defaults for d435i mount rotation.
        offset_rot_deg = (1.0, 27.0, 1.0)

        self._ensure_shared_strict_warp_camera_mount()

        sensor_cfg = SimpleNamespace(
            num_sensors=1,
            width=int(self._camera_width),
            height=int(self._camera_height),
            horizontal_fov_deg=float(self._camera_hfov_deg),
            max_range=float(self.cfg.max_distance),
            min_range=float(getattr(self.cfg, "camera_near", 0.1) or 0.1),
            calculate_depth=True,
            return_pointcloud=False,
            pointcloud_in_world_frame=False,
            segmentation_camera=False,
            dynamic_meshes=True,
            randomize_placement=False,
            min_translation={"cam_front_depth": [-0.025, -0.025, -0.025]},
            max_translation={"cam_front_depth": [0.025, 0.025, 0.025]},
            min_euler_rotation_deg={"cam_front_depth": [-2.5, -3.0, -2.5]},
            max_euler_rotation_deg={"cam_front_depth": [2.5, 3.0, 2.5]},
            offset_rot_base=[-90.0, 0.0, -90.0],
            offset={"cam_front_depth": {"offset_pos": offset_pos, "offset_rot": offset_rot_deg}},
            base_link_frame={"cam_front_depth": camera_body_name},
            ray_cast_bodies=ray_cast_bodies,
            add_offpath_obstacle=False,
            offpath_obstacle_meshes_root=None,
            offpath_obstacle_bodies={},
            asset_meshes_root=mesh_root,
            primitive_bodies=list(registered_object_primitives.keys()),
        )

        body_names = getattr(self.env, "body_names", None) or getattr(self.env.robot_config, "body_names", None)
        if not body_names:
            raise RuntimeError("Cannot setup far_tracking_warp camera: body_names unavailable.")

        def _resolve_body_index(name: str) -> int | None:
            if name in body_names:
                return int(body_names.index(name))
            resolved = resolve_fixed_link_offset(
                self.env.robot_config,
                name,
                available_links=body_names,
                device=self.device,
            )
            if resolved is None:
                return None
            parent_name, _offset_pos, _offset_quat = resolved
            return int(body_names.index(parent_name))

        base_link_indices = [_resolve_body_index(camera_body_name)]
        if base_link_indices[0] is None:
            raise RuntimeError(f"Body '{camera_body_name}' not found in robot body_names for far_tracking_warp source.")

        robot_slot_indices: list[int] = []
        robot_body_indices: list[int] = []
        object_slot_indices: list[int] = []
        object_names: list[str] = []
        object_source_indices: list[int] = []
        object_active_env_ids: list[torch.Tensor | None] = []
        primitive_source_indices: list[int] = []
        primitive_half_extents: list[torch.Tensor] = []
        primitive_active: list[torch.Tensor] = []
        object_name_to_source_index: dict[str, int] = {}

        def _register_object_source(source_name: str) -> int:
            source_index = object_name_to_source_index.get(source_name)
            if source_index is not None:
                return source_index
            source_index = len(object_names)
            object_name_to_source_index[source_name] = source_index
            object_names.append(source_name)
            return source_index

        for slot_idx, name in enumerate(ray_cast_bodies.keys()):
            body_idx = _resolve_body_index(name)
            if body_idx is not None:
                robot_slot_indices.append(slot_idx)
                robot_body_indices.append(body_idx)
                continue
            slot_spec = registered_object_meshes.get(name)
            if slot_spec is not None:
                object_slot_indices.append(slot_idx)
                source_name = str(slot_spec["source_name"])
                source_index = _register_object_source(source_name)
                object_source_indices.append(source_index)
                active_env_ids = slot_spec.get("env_ids")
                if active_env_ids is None:
                    object_active_env_ids.append(None)
                else:
                    object_active_env_ids.append(
                        torch.tensor(active_env_ids, dtype=torch.long, device=self.device)
                    )
                continue
            raise RuntimeError(
                f"ray_cast body '{name}' is neither a robot body nor a registered object with mesh."
            )

        for primitive_name, primitive_spec in registered_object_primitives.items():
            primitive_source_indices.append(_register_object_source(str(primitive_spec["source_name"])))
            primitive_half_extents.append(primitive_spec["half_extents"])
            primitive_active.append(primitive_spec["active"])

        self._far_tracking_camera_sensor = FarTrackingCameraSensor(
            self.num_envs,
            sensor_cfg,
            self._terrain_mesh,
            device=self.device,
        )
        if (
            self._shared_camera_sensor_local_position is not None
            and self._shared_camera_sensor_local_orientation is not None
            and self._shared_camera_sensor_data_frame_quat is not None
        ):
            self._far_tracking_camera_sensor.camera_sensor_local_position[:] = (
                self._shared_camera_sensor_local_position.to(
                    device=self._far_tracking_camera_sensor.camera_sensor_local_position.device,
                    dtype=self._far_tracking_camera_sensor.camera_sensor_local_position.dtype,
                )
            )
            self._far_tracking_camera_sensor.camera_sensor_local_orientation[:] = (
                self._shared_camera_sensor_local_orientation.to(
                    device=self._far_tracking_camera_sensor.camera_sensor_local_orientation.device,
                    dtype=self._far_tracking_camera_sensor.camera_sensor_local_orientation.dtype,
                )
            )
            self._far_tracking_camera_sensor.camera_sensor_data_frame_quat[:] = (
                self._shared_camera_sensor_data_frame_quat.to(
                    device=self._far_tracking_camera_sensor.camera_sensor_data_frame_quat.device,
                    dtype=self._far_tracking_camera_sensor.camera_sensor_data_frame_quat.dtype,
                )
            )
        self._far_tracking_tf_apply = ft_tf_apply_xyzw
        self._far_tracking_quat_mul = ft_quat_mul_xyzw
        if primitive_half_extents:
            primitive_half_extents_tensor = torch.stack(primitive_half_extents, dim=1).to(
                device=self._far_tracking_camera_sensor.primitive_body_half_extents_tensor.device,
                dtype=self._far_tracking_camera_sensor.primitive_body_half_extents_tensor.dtype,
            )
            primitive_active_tensor = torch.stack(primitive_active, dim=1).to(
                device=self._far_tracking_camera_sensor.primitive_body_active_tensor.device,
                dtype=self._far_tracking_camera_sensor.primitive_body_active_tensor.dtype,
            )
            self._far_tracking_camera_sensor.primitive_body_half_extents_tensor[:] = primitive_half_extents_tensor
            self._far_tracking_camera_sensor.primitive_body_active_tensor[:] = primitive_active_tensor
        self._far_tracking_base_link_indices = torch.tensor(base_link_indices, dtype=torch.long, device=self.device)
        self._far_tracking_robot_slot_indices = torch.tensor(
            robot_slot_indices, dtype=torch.long, device=self.device
        )
        self._far_tracking_robot_body_indices = torch.tensor(
            robot_body_indices, dtype=torch.long, device=self.device
        )
        self._far_tracking_object_slot_indices = torch.tensor(
            object_slot_indices, dtype=torch.long, device=self.device
        )
        self._far_tracking_object_source_indices = torch.tensor(
            object_source_indices, dtype=torch.long, device=self.device
        )
        self._far_tracking_primitive_source_indices = torch.tensor(
            primitive_source_indices, dtype=torch.long, device=self.device
        )
        self._far_tracking_object_names = object_names
        self._far_tracking_object_active_env_ids = object_active_env_ids
        self._initialize_far_tracking_object_slots()

    def _compute_far_tracking_camera_depth(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self._far_tracking_camera_sensor is None:
            raise RuntimeError("far_tracking_warp camera sensor is not initialized.")
        if self._far_tracking_tf_apply is None or self._far_tracking_quat_mul is None:
            raise RuntimeError("far_tracking_warp camera helpers are not initialized.")
        if (
            self._far_tracking_base_link_indices is None
            or self._far_tracking_robot_slot_indices is None
            or self._far_tracking_robot_body_indices is None
            or self._far_tracking_object_slot_indices is None
            or self._far_tracking_object_source_indices is None
            or self._far_tracking_primitive_source_indices is None
        ):
            raise RuntimeError("far_tracking_warp camera indices are not initialized.")

        body_pos = self.env.simulator._rigid_body_pos
        body_quat = self.env.simulator._rigid_body_rot

        # Fill robot-body-backed slots.
        if self._far_tracking_robot_slot_indices.numel() > 0:
            ray_cast_body_poses = body_pos[:, self._far_tracking_robot_body_indices]
            ray_cast_body_quats = body_quat[:, self._far_tracking_robot_body_indices]
            self._far_tracking_camera_sensor.ray_cast_body_poses_tensor[:, self._far_tracking_robot_slot_indices] = (
                ray_cast_body_poses
            )
            self._far_tracking_camera_sensor.ray_cast_body_quats_tensor[:, self._far_tracking_robot_slot_indices] = (
                ray_cast_body_quats
            )

        # Fill registered object slots from simulator actor states.
        if self._far_tracking_object_names:
            env_ids_all = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
            object_states = self.env.simulator.get_actor_states(self._far_tracking_object_names, env_ids_all)
            object_states = object_states.view(len(self._far_tracking_object_names), self.num_envs, -1).permute(1, 0, 2)
            if self._far_tracking_primitive_source_indices.numel() > 0:
                self._far_tracking_camera_sensor.primitive_body_poses_tensor[:] = object_states[
                    :, self._far_tracking_primitive_source_indices, :3
                ]
                self._far_tracking_camera_sensor.primitive_body_quats_tensor[:] = object_states[
                    :, self._far_tracking_primitive_source_indices, 3:7
                ]
            if self._far_tracking_object_slot_indices.numel() == 0:
                object_states = None
        else:
            object_states = None

        if object_states is not None and self._far_tracking_object_slot_indices.numel() > 0:
            for local_slot_idx in range(int(self._far_tracking_object_slot_indices.numel())):
                slot_tensor_idx = int(self._far_tracking_object_slot_indices[local_slot_idx].item())
                source_idx = int(self._far_tracking_object_source_indices[local_slot_idx].item())
                active_env_ids = self._far_tracking_object_active_env_ids[local_slot_idx]
                if active_env_ids is None:
                    self._far_tracking_camera_sensor.ray_cast_body_poses_tensor[:, slot_tensor_idx] = (
                        object_states[:, source_idx, :3]
                    )
                    self._far_tracking_camera_sensor.ray_cast_body_quats_tensor[:, slot_tensor_idx] = (
                        object_states[:, source_idx, 3:7]
                    )
                    continue
                if active_env_ids.numel() == 0:
                    continue
                self._far_tracking_camera_sensor.ray_cast_body_poses_tensor[active_env_ids, slot_tensor_idx] = (
                    object_states[active_env_ids, source_idx, :3]
                )
                self._far_tracking_camera_sensor.ray_cast_body_quats_tensor[active_env_ids, slot_tensor_idx] = (
                    object_states[active_env_ids, source_idx, 3:7]
                )

        camera_base_link_pos = body_pos[:, self._far_tracking_base_link_indices]
        camera_base_link_quat = body_quat[:, self._far_tracking_base_link_indices]
        updated_camera_pos = self._far_tracking_tf_apply(
            camera_base_link_quat,
            camera_base_link_pos,
            self._far_tracking_camera_sensor.camera_sensor_local_position,
        )
        updated_camera_quat = self._far_tracking_quat_mul(
            camera_base_link_quat,
            self._far_tracking_quat_mul(
                self._far_tracking_camera_sensor.camera_sensor_local_orientation,
                self._far_tracking_camera_sensor.camera_sensor_data_frame_quat,
            ),
        )
        self._far_tracking_camera_sensor.camera_sensor_position[:] = updated_camera_pos
        self._far_tracking_camera_sensor.camera_sensor_orientation[:] = updated_camera_quat
        self._far_tracking_debug_last = {
            "base_link_indices": self._far_tracking_base_link_indices.detach().to(torch.float32).view(1, -1).expand(self.num_envs, -1),
            "base_link_pos": camera_base_link_pos.detach().clone().view(self.num_envs, -1),
            "base_link_quat_xyzw": camera_base_link_quat.detach().clone().view(self.num_envs, -1),
            "updated_camera_pos": updated_camera_pos.detach().clone().view(self.num_envs, -1),
            "updated_camera_quat_xyzw": updated_camera_quat.detach().clone().view(self.num_envs, -1),
            "sensor_camera_pos_before_capture": self._far_tracking_camera_sensor.camera_sensor_position.detach()
            .clone()
            .view(self.num_envs, -1),
            "sensor_camera_quat_before_capture": self._far_tracking_camera_sensor.camera_sensor_orientation.detach()
            .clone()
            .view(self.num_envs, -1),
        }

        depth = self._far_tracking_camera_sensor.capture()
        self._far_tracking_debug_last["sensor_camera_pos_after_capture"] = (
            self._far_tracking_camera_sensor.camera_sensor_position.detach().clone().view(self.num_envs, -1)
        )
        self._far_tracking_debug_last["sensor_camera_quat_after_capture"] = (
            self._far_tracking_camera_sensor.camera_sensor_orientation.detach().clone().view(self.num_envs, -1)
        )
        if depth.ndim == 4:
            depth = depth[:, 0]
        if depth.ndim != 3:
            raise RuntimeError(f"Unexpected far_tracking_warp depth shape: {tuple(depth.shape)}")
        depth = self._clamp_camera_depth_to_sensor_range(depth)

        if env_ids is None:
            return depth
        return depth[env_ids]

    def _initialize_far_tracking_object_slots(self) -> None:
        """Park inactive heterogeneous-object slots far away so each env still raycasts a single object mesh."""
        if self._far_tracking_camera_sensor is None or self._far_tracking_object_slot_indices is None:
            return

        inactive_pos = torch.tensor(
            [1.0e6, 1.0e6, 1.0e6],
            device=self.device,
            dtype=self._far_tracking_camera_sensor.ray_cast_body_poses_tensor.dtype,
        )
        inactive_quat = torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            device=self.device,
            dtype=self._far_tracking_camera_sensor.ray_cast_body_quats_tensor.dtype,
        )
        for local_slot_idx in range(int(self._far_tracking_object_slot_indices.numel())):
            active_env_ids = self._far_tracking_object_active_env_ids[local_slot_idx]
            if active_env_ids is None:
                continue
            slot_tensor_idx = int(self._far_tracking_object_slot_indices[local_slot_idx].item())
            self._far_tracking_camera_sensor.ray_cast_body_poses_tensor[:, slot_tensor_idx] = inactive_pos
            self._far_tracking_camera_sensor.ray_cast_body_quats_tensor[:, slot_tensor_idx] = inactive_quat

    def _collect_registered_sim_object_primitives(self) -> dict[str, dict[str, Any]]:
        """Resolve analytic box specs for registered simulator objects when primitive mode is enabled."""
        primitive_map: dict[str, dict[str, Any]] = {}
        if self._object_geometry_mode != "primitive":
            return primitive_map

        simulator = getattr(self.env, "simulator", None)
        object_registry = getattr(simulator, "object_registry", None)
        if object_registry is None:
            return primitive_map

        for name, obj_type, _position_in_type, _indices, _initial_poses in getattr(object_registry, "objects", []):
            if obj_type == "robot":
                continue
            primitive_spec = self._resolve_registered_object_primitive_spec(name)
            if primitive_spec is not None:
                primitive_map[name] = primitive_spec
        return primitive_map

    def _collect_registered_sim_object_meshes(
        self,
        *,
        excluded_object_names: set[str] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Resolve mesh files for all registered non-robot simulator objects."""
        mesh_map: dict[str, dict[str, Any]] = {}
        simulator = getattr(self.env, "simulator", None)
        object_registry = getattr(simulator, "object_registry", None)
        if object_registry is None:
            return mesh_map

        for name, obj_type, _position_in_type, _indices, _initial_poses in getattr(object_registry, "objects", []):
            if obj_type == "robot":
                continue
            if excluded_object_names and name in excluded_object_names:
                continue
            mesh_specs = self._resolve_registered_object_mesh_specs(name)
            if mesh_specs:
                mesh_map.update(mesh_specs)
            else:
                (self.logger or logger).warning(
                    "Skipping registered object '{}' in perception raycast: unable to resolve mesh path.",
                    name,
                )
        return mesh_map

    def _resolve_registered_object_primitive_spec(self, object_name: str) -> dict[str, Any] | None:
        """Resolve per-env analytic box extents for a registered simulator object."""
        asset_candidates = self._resolve_registered_object_asset_candidates(object_name)
        if not asset_candidates:
            return None

        half_extents = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float32)
        active = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        object_scale = self._get_registered_object_scale_xyz()

        for asset_candidate in asset_candidates:
            candidate_path = str(asset_candidate["asset_path"])
            candidate_half_extents = self._resolve_registered_object_box_half_extents_from_asset_path(
                candidate_path,
                object_scale=object_scale,
            )
            if candidate_half_extents is None:
                return None
            half_extents_value = torch.tensor(candidate_half_extents, device=self.device, dtype=torch.float32)
            env_ids = asset_candidate.get("env_ids")
            if env_ids is None:
                half_extents[:] = half_extents_value
                active[:] = 1
                continue
            if len(env_ids) == 0:
                continue
            env_ids_tensor = torch.tensor(env_ids, dtype=torch.long, device=self.device)
            half_extents[env_ids_tensor] = half_extents_value
            active[env_ids_tensor] = 1

        if int(torch.count_nonzero(active).item()) == 0:
            return None
        return {
            "source_name": object_name,
            "half_extents": half_extents,
            "active": active,
        }

    def _resolve_registered_object_box_half_extents_from_asset_path(
        self,
        candidate_path: str,
        *,
        object_scale: tuple[float, float, float] | None,
    ) -> tuple[float, float, float] | None:
        resolved_path = resolve_data_file_path(candidate_path)
        if os.path.splitext(resolved_path)[1].lower() != ".urdf":
            return None

        metadata = load_urdf_box_primitive_metadata(resolved_path)
        if metadata is None:
            return None

        extents = tuple(float(v) for v in metadata.extents)
        if object_scale is not None:
            extents = tuple(float(extents[idx]) * float(object_scale[idx]) for idx in range(3))
        return tuple(max(0.5 * float(extents[idx]), 5.0e-5) for idx in range(3))

    def _resolve_registered_object_mesh_specs(self, object_name: str) -> dict[str, dict[str, Any]]:
        """Resolve one or more raycast slots for a registered simulator object."""
        mesh_specs: dict[str, dict[str, Any]] = {}
        asset_candidates = self._resolve_registered_object_asset_candidates(object_name)
        for slot_variant_idx, asset_candidate in enumerate(asset_candidates):
            candidate_path = str(asset_candidate["asset_path"])
            env_ids = asset_candidate.get("env_ids")
            slot_name = object_name if len(asset_candidates) == 1 and env_ids is None else (
                f"{object_name}__variant_{slot_variant_idx:03d}"
            )
            mesh_path = self._resolve_registered_object_mesh_from_asset_path(candidate_path, object_name)
            if mesh_path is None:
                continue
            mesh_specs[slot_name] = {
                "mesh_path": mesh_path,
                "source_name": object_name,
                "env_ids": env_ids,
            }
        return mesh_specs

    def _resolve_registered_object_asset_candidates(self, object_name: str) -> list[dict[str, Any]]:
        """Resolve the simulator asset path(s) backing a registered object."""
        simulator = getattr(self.env, "simulator", None)
        scene = getattr(simulator, "scene", None)
        rigid_objects = getattr(scene, "rigid_objects", None)
        use_mujoco_object_urdf_fallback = (
            simulator is not None
            and get_simulator_type() == SimulatorType.MUJOCO
            and hasattr(simulator, "_object_urdf_by_name")
        )
        candidate_path = None
        if rigid_objects is None:
            object_urdf_by_name = getattr(simulator, "_object_urdf_by_name", None)
            if use_mujoco_object_urdf_fallback and isinstance(object_urdf_by_name, dict):
                candidate_path = object_urdf_by_name.get(object_name)
            if candidate_path:
                return [{"asset_path": str(candidate_path), "env_ids": None}]
            return self._resolve_simulator_object_asset_candidates_fallback(simulator, object_name)

        rigid_object = rigid_objects.get(object_name, None)
        if rigid_object is None:
            # Best-effort fallback for scene collection objects registered by basename.
            scene_collection = rigid_objects.get("usd_scene_objects", None)
            scene_cfg = getattr(scene_collection, "cfg", None) if scene_collection is not None else None
            scene_rigid_cfgs = getattr(scene_cfg, "rigid_objects", None)
            if isinstance(scene_rigid_cfgs, dict):
                for prim_path, cfg in scene_rigid_cfgs.items():
                    if str(prim_path).split("/")[-1] == object_name:
                        rigid_object = cfg
                        break
        if rigid_object is None:
            object_urdf_by_name = getattr(simulator, "_object_urdf_by_name", None)
            if use_mujoco_object_urdf_fallback and isinstance(object_urdf_by_name, dict):
                candidate_path = object_urdf_by_name.get(object_name)
            if candidate_path:
                return [{"asset_path": str(candidate_path), "env_ids": None}]
            return self._resolve_simulator_object_asset_candidates_fallback(simulator, object_name)

        cfg = getattr(rigid_object, "cfg", None)
        spawn = getattr(cfg, "spawn", None) if cfg is not None else getattr(rigid_object, "spawn", None)
        if spawn is None:
            return self._resolve_simulator_object_asset_candidates_fallback(simulator, object_name)

        for attr_name in ("asset_path", "urdf_path", "usd_path"):
            attr_val = getattr(spawn, attr_name, None)
            if attr_val:
                candidate_path = str(attr_val)
                break
        if candidate_path:
            return [{"asset_path": str(candidate_path), "env_ids": None}]
        return self._resolve_simulator_object_asset_candidates_fallback(simulator, object_name)

    def _resolve_registered_object_mesh_from_asset_path(self, candidate_path: str, object_name: str) -> str | None:
        """Resolve (and cache) a mesh file path for a specific simulator asset path."""
        resolved_path = str(Path(resolve_data_file_path(candidate_path)).resolve())
        cache_key = f"{object_name}::{self._object_geometry_mode}::{resolved_path}"
        if cache_key in self._registered_object_mesh_cache:
            return self._registered_object_mesh_cache[cache_key]

        resolved_path = resolve_data_file_path(candidate_path)
        ext = os.path.splitext(resolved_path)[1].lower()

        mesh_path: str | None = None
        if ext == ".urdf":
            mesh_path = self._export_combined_urdf_visual_mesh(resolved_path, object_name)
        elif ext in {".obj", ".stl", ".ply"}:
            mesh_path = resolved_path if os.path.exists(resolved_path) else None
        elif ext in {".usd", ".usda", ".usdc"}:
            mesh_path = self._export_combined_mesh_from_scene_asset(resolved_path, object_name)

        if mesh_path and os.path.exists(mesh_path):
            self._registered_object_mesh_cache[cache_key] = mesh_path
            return mesh_path
        return None

    def _get_registered_object_scale_xyz(self) -> tuple[float, float, float] | None:
        object_cfg = getattr(getattr(self.env, "robot_config", None), "object", None)
        object_scale_raw = getattr(object_cfg, "scale", None)
        if object_scale_raw is None:
            return None
        if len(object_scale_raw) == 1:
            value = float(object_scale_raw[0])
            return (value, value, value)
        if len(object_scale_raw) == 3:
            return tuple(float(v) for v in object_scale_raw)
        return None

    def _resolve_simulator_object_asset_candidates_fallback(
        self, simulator: Any, object_name: str
    ) -> list[dict[str, Any]]:
        """Best-effort fallback for simulators that keep object assets outside the scene spawn cfg."""
        if simulator is None:
            return []

        object_urdf_by_name = getattr(simulator, "_object_urdf_by_name", None)
        if isinstance(object_urdf_by_name, dict):
            candidate_path = str(object_urdf_by_name.get(object_name, "") or "").strip()
            if candidate_path:
                return [{"asset_path": candidate_path, "env_ids": None}]

        env_object_urdf_paths = getattr(simulator, "_env_object_urdf_paths", None)
        if not isinstance(env_object_urdf_paths, list):
            return []

        env_ids_by_candidate: dict[str, list[int]] = {}
        for env_idx, raw_path in enumerate(env_object_urdf_paths):
            candidate_path = str(raw_path or "").strip()
            if not candidate_path:
                continue
            normalized_path = str(Path(resolve_data_file_path(candidate_path)).resolve())
            env_ids_by_candidate.setdefault(normalized_path, []).append(env_idx)

        if not env_ids_by_candidate:
            return []
        if len(env_ids_by_candidate) == 1 or self.num_envs == 1:
            only_path = next(iter(env_ids_by_candidate.keys()))
            return [{"asset_path": only_path, "env_ids": None}]
        return [
            {"asset_path": candidate_path, "env_ids": env_ids}
            for candidate_path, env_ids in sorted(env_ids_by_candidate.items())
        ]

    def _export_combined_urdf_visual_mesh(self, urdf_path: str, object_name: str) -> str | None:
        """Build a single OBJ mesh from URDF visual geometry for dynamic object raycasting."""
        try:
            import trimesh  # noqa: PLC0415
        except Exception:
            return None

        urdf_file = Path(urdf_path).expanduser()
        if not urdf_file.exists():
            return None

        cache_dir = Path("/tmp/holosoma_perception_mesh_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_mesh_path = cache_dir / f"{object_name}_{urdf_file.stem}_combined.obj"
        if cache_mesh_path.exists():
            return str(cache_mesh_path)

        try:
            root = ET.parse(str(urdf_file)).getroot()
        except Exception:
            return None

        urdf_dir = str(urdf_file.parent)
        meshes: list[Any] = []

        for link in root.findall("link"):
            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is None:
                    continue

                mesh = None
                mesh_tag = geometry.find("mesh")
                if mesh_tag is not None:
                    filename = mesh_tag.get("filename")
                    if not filename:
                        continue
                    mesh_file = self._resolve_urdf_mesh_path(urdf_dir, urdf_dir, filename)
                    if not os.path.exists(mesh_file):
                        continue
                    mesh = trimesh.load(mesh_file, process=False)
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    if not isinstance(mesh, trimesh.Trimesh):
                        continue
                    scale = self._parse_urdf_vec3(mesh_tag.get("scale"), default=None)
                    if scale is not None:
                        mesh.apply_scale(scale.detach().cpu().numpy())
                else:
                    box_tag = geometry.find("box")
                    cylinder_tag = geometry.find("cylinder")
                    sphere_tag = geometry.find("sphere")
                    if box_tag is not None:
                        size = self._parse_urdf_vec3(box_tag.get("size"), default=None)
                        if size is None:
                            continue
                        mesh = trimesh.creation.box(extents=size.detach().cpu().numpy())
                    elif cylinder_tag is not None:
                        radius = float(cylinder_tag.get("radius", "0.0"))
                        length = float(cylinder_tag.get("length", "0.0"))
                        if radius <= 0.0 or length <= 0.0:
                            continue
                        mesh = trimesh.creation.cylinder(radius=radius, height=length)
                    elif sphere_tag is not None:
                        radius = float(sphere_tag.get("radius", "0.0"))
                        if radius <= 0.0:
                            continue
                        mesh = trimesh.creation.icosphere(radius=radius)
                    else:
                        continue

                if not isinstance(mesh, trimesh.Trimesh):
                    continue

                origin = visual.find("origin")
                visual_pos = self._parse_urdf_vec3(
                    origin.get("xyz") if origin is not None else None, default=(0.0, 0.0, 0.0)
                )
                visual_rpy = self._parse_urdf_vec3(
                    origin.get("rpy") if origin is not None else None, default=(0.0, 0.0, 0.0)
                )
                visual_quat = quat_from_euler_xyz(visual_rpy[0], visual_rpy[1], visual_rpy[2])

                quat_batch = visual_quat.unsqueeze(0).expand(mesh.vertices.shape[0], -1)
                verts = torch.as_tensor(mesh.vertices, dtype=torch.float32)
                verts = quat_apply(quat_batch, verts, w_last=True) + visual_pos.unsqueeze(0)
                mesh.vertices = verts.detach().cpu().numpy()
                meshes.append(mesh)

        if not meshes:
            return None

        try:
            combined = trimesh.util.concatenate(meshes)
            combined.export(str(cache_mesh_path))
        except Exception:
            return None
        return str(cache_mesh_path)

    def _export_combined_mesh_from_scene_asset(self, asset_path: str, object_name: str) -> str | None:
        """Best-effort mesh export for USD scene assets."""
        try:
            import trimesh  # noqa: PLC0415
        except Exception:
            return None

        source_path = Path(asset_path).expanduser()
        if not source_path.exists():
            return None

        cache_dir = Path("/tmp/holosoma_perception_mesh_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_mesh_path = cache_dir / f"{object_name}_{source_path.stem}_combined.obj"
        if cache_mesh_path.exists():
            return str(cache_mesh_path)

        try:
            mesh = trimesh.load(str(source_path), process=False)
        except Exception:
            return None

        if isinstance(mesh, trimesh.Scene):
            try:
                mesh = mesh.dump(concatenate=True)
            except Exception:
                return None
        if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
            return None

        try:
            mesh.export(str(cache_mesh_path))
        except Exception:
            return None
        return str(cache_mesh_path)

    def _build_grid(self) -> tuple[torch.Tensor, torch.Tensor]:
        half_extent_x = (self._heightmap_grid_x - 1) * self._heightmap_interval_x / 2.0
        half_extent_y = (self._heightmap_grid_y - 1) * self._heightmap_interval_y / 2.0
        coords_x = torch.linspace(
            -half_extent_x,
            half_extent_x,
            self._heightmap_grid_x,
            device=self.device,
            requires_grad=False,
        )
        coords_y = torch.linspace(
            -half_extent_y,
            half_extent_y,
            self._heightmap_grid_y,
            device=self.device,
            requires_grad=False,
        )
        grid_x, grid_y = torch.meshgrid(coords_x, coords_y, indexing="ij")
        grid_points = torch.zeros(self._num_points, 3, device=self.device)
        grid_points[:, 0] = grid_x.flatten()
        grid_points[:, 1] = grid_y.flatten()

        ray_dirs = torch.zeros(self._num_points, 3, device=self.device)
        ray_dirs[:, 2] = -1.0
        return grid_points, ray_dirs

    def _camera_ray_rotation_quat(self, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        pitch_deg = float(self.cfg.camera_pitch_deg)
        pitch_rad = torch.deg2rad(torch.tensor(pitch_deg, device=device, dtype=dtype))
        pitch_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=device, dtype=dtype),
            pitch_rad,
            torch.tensor(0.0, device=device, dtype=dtype),
        )
        correction_quat = self._camera_ray_correction_quat.to(device=device, dtype=dtype)
        identity_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device, dtype=dtype)

        if self._camera_strict_warp:
            combo = identity_quat
            if self._use_camera_mount_quat:
                combo = self._camera_mount_quat.to(device=device, dtype=dtype)
            if self._use_camera_frame_quat:
                combo = quat_mul(combo, self._camera_frame_quat.to(device=device, dtype=dtype), w_last=True)
            if abs(pitch_deg) > 1.0e-6:
                combo = quat_mul(pitch_quat, combo, w_last=True)
            return quat_mul(correction_quat, combo, w_last=True)

        if self._use_camera_frame_quat:
            combo = quat_mul(pitch_quat, self._camera_frame_quat.to(device=device, dtype=dtype), w_last=True)
            return quat_mul(correction_quat, combo, w_last=True)

        return quat_mul(correction_quat, pitch_quat, w_last=True)

    def _camera_dirs_cam_from_xy(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self._camera_strict_warp:
            dirs_cam = torch.stack((x, y, torch.ones_like(x)), dim=-1)
        elif self._use_camera_frame_quat:
            # USD camera frame: x right, y up, -z forward.
            dirs_cam = torch.stack((x, -y, -torch.ones_like(x)), dim=-1)
        else:
            # VideoMimic pinhole convention: camera (x right, y down, z forward)
            # -> robotics (x forward, y left, z up) => [z, -x, -y] = [1, -x, -y].
            dirs_cam = torch.stack((torch.ones_like(x), -x, -y), dim=-1)
        return dirs_cam / torch.norm(dirs_cam, dim=-1, keepdim=True).clamp(min=1.0e-6)

    def _build_camera_rays_from_coords(self, u_coords: torch.Tensor, v_coords: torch.Tensor) -> torch.Tensor:
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")
        x = (u_grid - self._camera_cx) / self._camera_fx
        y = (v_grid - self._camera_cy) / self._camera_fy
        dirs_cam = self._camera_dirs_cam_from_xy(x, y).view(-1, 3)
        combo = self._camera_ray_rotation_quat(device=self.device, dtype=torch.float32)
        combo = combo.unsqueeze(0).expand(dirs_cam.shape[0], -1)
        dirs_base = quat_apply(combo, dirs_cam, w_last=True)
        return dirs_base / torch.norm(dirs_base, dim=-1, keepdim=True).clamp(min=1.0e-6)

    def _build_camera_rays(self) -> torch.Tensor:
        u_coords = torch.arange(self._camera_width, device=self.device, dtype=torch.float32)
        v_coords = torch.arange(self._camera_height, device=self.device, dtype=torch.float32)
        return self._build_camera_rays_from_coords(u_coords, v_coords)

    def _build_camera_scandots_rays(self) -> torch.Tensor:
        target_w = self.cfg.camera_scandots_width
        target_h = self.cfg.camera_scandots_height
        if target_w is not None or target_h is not None:
            if target_w is None:
                target_w = target_h
            if target_h is None:
                target_h = target_w
            target_w = max(1, int(target_w or 1))
            target_h = max(1, int(target_h or 1))
            u_coords = torch.linspace(
                0.0,
                float(self._camera_width - 1),
                steps=target_w,
                device=self.device,
                dtype=torch.float32,
            )
            v_coords = torch.linspace(
                0.0,
                float(self._camera_height - 1),
                steps=target_h,
                device=self.device,
                dtype=torch.float32,
            )
        else:
            stride = max(1, int(self.cfg.camera_scandots_stride))
            u_coords = torch.arange(0, self._camera_width, stride, device=self.device, dtype=torch.float32)
            v_coords = torch.arange(0, self._camera_height, stride, device=self.device, dtype=torch.float32)
            if u_coords.numel() == 0 or v_coords.numel() == 0:
                raise ValueError("camera_scandots_stride is too large for the camera resolution.")

            if int(u_coords[-1].item()) != self._camera_width - 1:
                u_coords = torch.cat(
                    [u_coords, torch.tensor([self._camera_width - 1], device=self.device, dtype=torch.float32)]
                )
            if int(v_coords[-1].item()) != self._camera_height - 1:
                v_coords = torch.cat(
                    [v_coords, torch.tensor([self._camera_height - 1], device=self.device, dtype=torch.float32)]
                )

        self._camera_scandots_width = int(u_coords.numel())
        self._camera_scandots_height = int(v_coords.numel())

        return self._build_camera_rays_from_coords(u_coords, v_coords)

    def _resolve_heightmap_grid(self) -> tuple[int, int, float, float]:
        size = getattr(self.cfg, "heightmap_size", None)
        resolution = getattr(self.cfg, "heightmap_resolution", None)
        if size is not None and resolution is not None:
            size_x, size_y = size
            resolution = float(resolution)
            if resolution <= 0:
                raise ValueError("heightmap_resolution must be > 0.")
            grid_x = max(1, int(size_x / resolution) + 1)
            grid_y = max(1, int(size_y / resolution) + 1)
            return grid_x, grid_y, resolution, resolution
        grid_size = int(self.cfg.grid_size)
        grid_interval = float(self.cfg.grid_interval)
        return grid_size, grid_size, grid_interval, grid_interval

    def _log_camera_ray_alignment(self) -> None:
        """Log one-time camera ray alignment stats for debugging mount orientation."""
        ray_dirs_base = self._camera_ray_dirs_base
        if ray_dirs_base is None or ray_dirs_base.numel() == 0:
            return

        try:
            env_ids = torch.tensor([0], device=self.device, dtype=torch.long)
            _body_pos, body_quat = self._get_camera_body_pose(env_ids)
            ray_dirs_world = quat_rotate_batched(body_quat, ray_dirs_base.unsqueeze(0))[0]
            ray_dirs_world = ray_dirs_world / torch.norm(ray_dirs_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

            total = int(ray_dirs_world.shape[0])
            if total <= 0:
                return

            root_quat = getattr(self.env, "base_quat", None)
            if isinstance(root_quat, torch.Tensor) and root_quat.ndim >= 2 and root_quat.shape[0] > 0:
                root_quat_env = root_quat[0:1].to(device=ray_dirs_world.device, dtype=ray_dirs_world.dtype)
                root_forward = quat_apply(
                    root_quat_env,
                    torch.tensor([[1.0, 0.0, 0.0]], device=ray_dirs_world.device, dtype=ray_dirs_world.dtype),
                    w_last=True,
                )[0]
                root_forward = root_forward / torch.norm(root_forward).clamp(min=1.0e-6)
                dots_root = torch.sum(ray_dirs_world * root_forward.unsqueeze(0), dim=-1)
                back_root = int((dots_root <= 0.0).sum().item())
                min_dot_root = float(dots_root.min().item())
                (self.logger or logger).info(
                    "Camera ray alignment (root): back={}/{} min_dot={:.3f}",
                    back_root,
                    total,
                    min_dot_root,
                )

            width = int(self._camera_width)
            height = int(self._camera_height)
            if width > 0 and height > 0 and (width * height) == total:
                center_idx = (height // 2) * width + (width // 2)
                center_dir = ray_dirs_world[center_idx]
                dots_center = torch.sum(ray_dirs_world * center_dir.unsqueeze(0), dim=-1)
                back_center = int((dots_center <= 0.0).sum().item())
                min_dot_center = float(dots_center.min().item())
                (self.logger or logger).info(
                    "Camera ray alignment (center): back={}/{} min_dot={:.3f} center=[{:.3f}, {:.3f}, {:.3f}]",
                    back_center,
                    total,
                    min_dot_center,
                    float(center_dir[0].item()),
                    float(center_dir[1].item()),
                    float(center_dir[2].item()),
                )
        except Exception as exc:
            (self.logger or logger).warning("Camera ray alignment debug skipped: {}", exc)

    def _get_camera_forward_axis(self, body_quat: torch.Tensor) -> torch.Tensor:
        # Match warp_sensors depth projection:
        # principal axis is built from integer (c_x, c_y) using the same pinhole model as ray generation.
        device = body_quat.device
        dtype = body_quat.dtype
        cx = float(self._camera_cx.item())
        cy = float(self._camera_cy.item())
        fx = float(self._camera_fx.item())
        fy = float(self._camera_fy.item())
        u0 = float(int(self._camera_width / 2))
        v0 = float(int(self._camera_height / 2))
        x0 = (u0 - cx) / fx
        y0 = (v0 - cy) / fy
        if self._camera_strict_warp:
            principal_cam = torch.tensor([x0, y0, 1.0], device=device, dtype=dtype)
        elif self._use_camera_frame_quat:
            principal_cam = torch.tensor([x0, -y0, -1.0], device=device, dtype=dtype)
        else:
            principal_cam = torch.tensor([1.0, -x0, -y0], device=device, dtype=dtype)
        principal_cam = principal_cam / torch.norm(principal_cam).clamp(min=1.0e-6)
        combo = self._camera_ray_rotation_quat(device=device, dtype=dtype)
        forward_base = quat_apply(combo.unsqueeze(0), principal_cam.unsqueeze(0), w_last=True).squeeze(0)
        forward_base = forward_base.unsqueeze(0).expand(body_quat.shape[0], -1)
        forward_world = quat_apply(body_quat, forward_base, w_last=True)
        return forward_world / torch.norm(forward_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

    def _project_ranges_to_camera_depth(
        self,
        ranges: torch.Tensor,
        ray_dirs_world: torch.Tensor,
        body_quat: torch.Tensor,
        hit_mask: torch.Tensor,
    ) -> torch.Tensor:
        ray_dirs_world = ray_dirs_world / torch.norm(ray_dirs_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
        forward_world = self._get_camera_forward_axis(body_quat)
        dots = torch.sum(ray_dirs_world * forward_world.unsqueeze(1), dim=-1)
        # Match far-tracking warp kernel semantics:
        #   mul = clamp(dot(rd, rd_principal), eps, 1.0), depth = mul * range.
        depth_mul = torch.clamp(dots, min=1.0e-6, max=1.0)
        depth = ranges * depth_mul
        hit_mask = (
            hit_mask
            & torch.isfinite(depth)
            & (depth <= (self.cfg.max_distance + 1.0e-6))
        )
        depth = torch.where(hit_mask, depth, torch.full_like(depth, self.cfg.max_distance))
        return torch.clamp(depth, min=0.0, max=self.cfg.max_distance)

    def _filter_camera_hit_mask_by_depth(
        self,
        ranges: torch.Tensor,
        ray_dirs_world: torch.Tensor,
        body_quat: torch.Tensor,
        hit_mask: torch.Tensor,
    ) -> torch.Tensor:
        ray_dirs_world = ray_dirs_world / torch.norm(ray_dirs_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
        forward_world = self._get_camera_forward_axis(body_quat)
        dots = torch.sum(ray_dirs_world * forward_world.unsqueeze(1), dim=-1)
        depth_mul = torch.clamp(dots, min=1.0e-6, max=1.0)
        depth = ranges * depth_mul
        return (
            hit_mask
            & torch.isfinite(depth)
            & (depth <= (self.cfg.max_distance + 1.0e-6))
        )

    def _compute_camera_miss_ranges(
        self, ray_dirs_world: torch.Tensor, body_quat: torch.Tensor
    ) -> torch.Tensor:
        ray_dirs_world = ray_dirs_world / torch.norm(ray_dirs_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
        forward_world = self._get_camera_forward_axis(body_quat)
        dots = torch.sum(ray_dirs_world * forward_world.unsqueeze(1), dim=-1)
        depth_mul = torch.clamp(dots, min=1.0e-6, max=1.0)
        return self.cfg.max_distance / depth_mul

    def _clamp_camera_depth_to_sensor_range(self, depth: torch.Tensor) -> torch.Tensor:
        min_depth = float(getattr(self.cfg, "camera_near", 0.1) or 0.1)
        max_depth = float(self.cfg.max_distance)
        depth = torch.where(
            torch.isfinite(depth) & (depth <= max_depth),
            depth,
            torch.full_like(depth, max_depth),
        )
        return torch.clamp(depth, min=min_depth, max=max_depth)

    def _apply_camera_depth_noise(self, depth: torch.Tensor) -> torch.Tensor:
        if not self._camera_apply_sensor_noise:
            return self._clamp_camera_depth_to_sensor_range(depth)
        std_mult = getattr(self.env, "_perception_camera_noise_std_mult", None)
        drop_prob = getattr(self.env, "_perception_camera_noise_drop_prob", None)
        if std_mult is None and drop_prob is None:
            return self._clamp_camera_depth_to_sensor_range(depth)

        depth_out = depth
        if std_mult is not None:
            if isinstance(std_mult, torch.Tensor):
                std = std_mult.to(depth.device)
                if std.ndim == 1:
                    std = std.view(-1, 1, 1)
            else:
                std = torch.tensor(float(std_mult), device=depth.device)
            depth_out = depth_out + torch.randn_like(depth_out) * (depth_out * std)

        if drop_prob is not None:
            if isinstance(drop_prob, torch.Tensor):
                prob = drop_prob.to(depth.device)
                if prob.ndim == 1:
                    prob = prob.view(-1, 1, 1)
            else:
                prob = torch.tensor(float(drop_prob), device=depth.device)
            mask = torch.rand_like(depth_out) < prob
            depth_out = torch.where(mask, torch.full_like(depth_out, self.cfg.max_distance), depth_out)

        return self._clamp_camera_depth_to_sensor_range(depth_out)

    def _resolve_camera_obs_resolution(self) -> tuple[int, int]:
        if not self._camera_warp_preprocess:
            return self._camera_height, self._camera_width

        crop_height = max(1, self._camera_height - self._camera_warp_crop_top - self._camera_warp_crop_bottom)
        crop_width = max(1, self._camera_width - self._camera_warp_crop_left - self._camera_warp_crop_right)
        if self._camera_warp_resize is None:
            return crop_height, crop_width
        return self._camera_warp_resize

    def _camera_obs_default_fill_value(self) -> float:
        if self._camera_warp_preprocess and self._camera_warp_normalize:
            return 0.5
        return float(self.cfg.max_distance)

    def _consume_camera_obs_refresh_flag(self) -> bool:
        refresh = True
        if self._camera_warp_preprocess and self._camera_warp_freq_ratio > 1:
            refresh = (self._camera_obs_step_counter % self._camera_warp_freq_ratio) == 0
        self._camera_obs_step_counter += 1
        return refresh

    def _normalize_env_ids(self, idx: torch.Tensor | slice | int | list[int] | tuple[int, ...]) -> torch.Tensor:
        if isinstance(idx, slice):
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)[idx]
        if isinstance(idx, torch.Tensor):
            if idx.ndim == 0:
                idx = idx.view(1)
            return idx.to(device=self.device, dtype=torch.long)
        if isinstance(idx, (list, tuple, np.ndarray)):
            return torch.as_tensor(idx, device=self.device, dtype=torch.long).view(-1)
        return torch.tensor([int(idx)], device=self.device, dtype=torch.long)

    def _update_camera_depth_observation(
        self,
        idx: torch.Tensor | slice | int | list[int] | tuple[int, ...],
        depth: torch.Tensor,
        *,
        refresh: bool,
    ) -> None:
        depth_obs = depth.unsqueeze(0) if depth.ndim == 2 else depth
        env_ids = self._normalize_env_ids(idx)
        if depth_obs.shape[0] != env_ids.numel():
            raise RuntimeError(
                f"Depth batch size ({depth_obs.shape[0]}) does not match env_ids ({env_ids.numel()})."
            )

        if not self._camera_warp_preprocess:
            self._camera_depth_obs[env_ids] = depth_obs
            return

        ready = self._camera_depth_buffer_ready[env_ids]
        should_process = bool(refresh or (~ready).any().item())
        if should_process:
            processed = self._process_camera_depth_for_obs(depth_obs)
            if self._camera_warp_buffer_len > 1:
                self._camera_depth_buffer[env_ids, :-1] = self._camera_depth_buffer[env_ids, 1:].clone()
            self._camera_depth_buffer[env_ids, -1] = processed

            if (~ready).any():
                new_env_ids = env_ids[~ready]
                repeated = processed[~ready].unsqueeze(1).repeat(1, self._camera_warp_buffer_len, 1, 1)
                self._camera_depth_buffer[new_env_ids] = repeated
            self._camera_depth_buffer_ready[env_ids] = True

        self._camera_depth_obs[env_ids] = self._camera_depth_buffer[env_ids, -1 - self._camera_warp_latency_frame]

    def _process_camera_depth_for_obs(self, depth: torch.Tensor) -> torch.Tensor:
        if not self._camera_warp_preprocess:
            return depth

        depth_obs = self._crop_camera_depth(depth)
        if (depth_obs.shape[-2], depth_obs.shape[-1]) != (self._camera_obs_height, self._camera_obs_width):
            depth_obs = F.interpolate(
                depth_obs.unsqueeze(1),
                size=(self._camera_obs_height, self._camera_obs_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        min_depth = float(getattr(self.cfg, "camera_near", 0.1) or 0.1)
        max_depth = float(self.cfg.max_distance)
        depth_obs = torch.clamp(depth_obs, min=min_depth, max=max_depth)
        if self._camera_warp_min_valid_depth > 0.0:
            depth_obs = torch.where(
                depth_obs < self._camera_warp_min_valid_depth,
                torch.full_like(depth_obs, max_depth),
                depth_obs,
            )

        if self._camera_warp_edge_noise:
            depth_obs = self._apply_warp_edge_noise(depth_obs, max_depth=max_depth)

        if self._camera_warp_enable_holes and self._camera_warp_hole_prob > 0.0:
            depth_obs = self._apply_warp_hole_noise(depth_obs, max_depth=max_depth)

        if self._camera_warp_normalize:
            depth_obs = self._normalize_camera_depth_for_obs(depth_obs, max_depth=max_depth)

        return depth_obs

    def _crop_camera_depth(self, depth: torch.Tensor) -> torch.Tensor:
        _, height, width = depth.shape
        top = min(self._camera_warp_crop_top, max(0, height - 1))
        bottom = min(self._camera_warp_crop_bottom, max(0, height - top - 1))
        left = min(self._camera_warp_crop_left, max(0, width - 1))
        right = min(self._camera_warp_crop_right, max(0, width - left - 1))
        height_end = max(top + 1, height - bottom)
        width_end = max(left + 1, width - right)
        return depth[:, top:height_end, left:width_end]

    def _mask_camera_edge_borders(self, mask: torch.Tensor) -> torch.Tensor:
        border = self._camera_warp_edge_border
        if border <= 0:
            return mask
        _, height, width = mask.shape
        border = min(border, height // 2, width // 2)
        if border <= 0:
            return mask
        out = mask.clone()
        out[:, :, :border] = False
        out[:, :, -border:] = False
        out[:, :border, :] = False
        out[:, -border:, :] = False
        return out

    def _apply_warp_edge_noise(self, depth: torch.Tensor, *, max_depth: float) -> torch.Tensor:
        image = depth.unsqueeze(1)
        sobel_x = self._camera_warp_sobel_x.to(device=depth.device, dtype=depth.dtype)
        sobel_y = self._camera_warp_sobel_y.to(device=depth.device, dtype=depth.dtype)
        grad_x = F.conv2d(image, sobel_x, padding=1)
        grad_y = F.conv2d(image, sobel_y, padding=1)
        grad_mag = torch.sqrt(grad_x.square() + grad_y.square()).squeeze(1)

        random_vals = torch.rand_like(grad_mag)
        edge_mask = grad_mag > self._camera_warp_edge_thresh_primary
        edge_mask = self._mask_camera_edge_borders(edge_mask)

        shuffle_mask = edge_mask & (random_vals < self._camera_warp_edge_shuffle_prob)
        if shuffle_mask.any():
            shuffle_mask = F.max_pool2d(
                shuffle_mask.unsqueeze(1).to(depth.dtype), kernel_size=3, stride=1, padding=1
            ).squeeze(1) > 0.5
            num_envs, height, width = depth.shape
            padded = F.pad(depth.unsqueeze(1), (1, 1, 1, 1), mode="circular")
            patches = F.unfold(padded, kernel_size=3, stride=1).view(num_envs, 9, height, width)
            neighbor_idx = torch.randint(0, 9, (num_envs, 1, height, width), device=depth.device)
            shuffled = torch.gather(patches, 1, neighbor_idx).squeeze(1)
            depth = torch.where(shuffle_mask, shuffled, depth)

        edge_mask_empty = (grad_mag > self._camera_warp_edge_thresh_secondary) & (
            F.max_pool2d(depth.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            > self._camera_warp_edge_far_depth_thresh
        )
        edge_mask_empty = self._mask_camera_edge_borders(edge_mask_empty)
        set_empty = edge_mask_empty & (random_vals < self._camera_warp_edge_empty_prob)
        if set_empty.any():
            depth = torch.where(set_empty, torch.full_like(depth, max_depth), depth)

        return depth

    def _apply_warp_hole_noise(self, depth: torch.Tensor, *, max_depth: float) -> torch.Tensor:
        num_envs, height, width = depth.shape
        noise = torch.rand((num_envs, 1, height, width), device=depth.device, dtype=depth.dtype)
        blobs = F.max_pool2d(noise, kernel_size=3, stride=1, padding=1).squeeze(1)
        holes = (blobs < self._camera_warp_hole_prob) & (depth < 2.0) & (depth > 0.2)
        if holes.any():
            depth = torch.where(holes, torch.full_like(depth, max_depth), depth)
        return depth

    def _normalize_camera_depth_for_obs(self, depth: torch.Tensor, *, max_depth: float) -> torch.Tensor:
        near = float(getattr(self.cfg, "camera_near", 0.1) or 0.1)
        denom = max(1.0e-6, max_depth - near)
        depth = (depth - near) / denom - 0.5
        return torch.clamp(depth, min=-0.5, max=0.5)

    def _setup_rendered_camera(self) -> None:
        simulator_type = get_simulator_type()
        if simulator_type == SimulatorType.ISAACSIM:
            from holosoma.simulator.isaacsim.perception_camera import (
                IsaacSimDepthCamera,
                IsaacSimDepthSensorCamera,
            )

            camera_cls = (
                IsaacSimDepthSensorCamera if self._camera_source == "rendered_depth_sensor" else IsaacSimDepthCamera
            )
        elif simulator_type == SimulatorType.MUJOCO:
            if self._camera_source == "rendered_depth_sensor":
                raise RuntimeError("camera_source=rendered_depth_sensor is IsaacSim-only. Use camera_source=rendered.")
            from holosoma.simulator.mujoco.perception_camera import MuJoCoDepthCamera  # noqa: PLC0415

            camera_cls = MuJoCoDepthCamera
        else:
            raise RuntimeError(
                "Rendered camera requires IsaacSim or MuJoCo. Use camera_source=far_tracking_warp for other simulators."
            )
        camera_kwargs = dict(
            env=self.env,
            config=self.cfg,
            width=self._camera_width,
            height=self._camera_height,
            vfov_deg=self._camera_vfov_deg,
            device=getattr(self.env.simulator, "device", self.device),
        )
        if simulator_type == SimulatorType.MUJOCO:
            intrinsics = (
                float(self._camera_fx.item()),
                float(self._camera_fy.item()),
                float(self._camera_cx.item()),
                float(self._camera_cy.item()),
            )
            vfov_deg = float(self._camera_vfov_deg)
            pose_provider = self.get_mujoco_render_camera_pose
            if self._camera_strict_warp:
                width = float(self._camera_width)
                height = float(self._camera_height)
                cx = width / 2.0
                cy = height / 2.0
                fx = width / (2.0 * math.tan(math.radians(float(self._camera_hfov_deg)) / 2.0))
                fy = fx
                vfov_deg = math.degrees(2.0 * math.atan(height / (2.0 * fx)))
                intrinsics = (fx, fy, cx, cy)
                pose_provider = self._get_strict_warp_camera_pose
            camera_kwargs["pose_provider"] = pose_provider
            camera_kwargs["vfov_deg"] = vfov_deg
            camera_kwargs["intrinsics"] = intrinsics
        self._rendered_camera = camera_cls(**camera_kwargs)
        self._rendered_camera.setup()

    @staticmethod
    def _parse_urdf_vec3(
        text: str | None,
        *,
        device: str | torch.device = "cpu",
        default: tuple[float, float, float] | None = (0.0, 0.0, 0.0),
    ) -> torch.Tensor | None:
        if text is None or text.strip() == "":
            if default is None:
                return None
            return torch.tensor(default, device=device, dtype=torch.float32)
        parts = text.split()
        values = [float(val) for val in parts[:3]]
        values += [0.0] * max(0, 3 - len(values))
        return torch.tensor(values[:3], device=device, dtype=torch.float32)

    def _resolve_robot_asset_paths(self) -> tuple[str, str]:
        robot_config = getattr(self.env, "robot_config", None)
        if robot_config is None:
            raise RuntimeError("PerceptionManager requires env.robot_config to load robot meshes.")
        asset_root = robot_config.asset.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())
        asset_root = resolve_data_file_path(asset_root)
        urdf_path = os.path.join(asset_root, robot_config.asset.urdf_file)
        return urdf_path, asset_root

    def _resolve_urdf_mesh_path(self, urdf_dir: str, asset_root: str, filename: str) -> str:
        if filename.startswith("package://"):
            filename = filename[len("package://") :]
            return os.path.join(asset_root, filename)
        if filename.startswith("file://"):
            filename = filename[len("file://") :]
        if os.path.isabs(filename):
            return filename
        return os.path.join(urdf_dir, filename)

    def _load_robot_link_meshes(self) -> None:
        if not self._camera_include_robot_mesh:
            return
        try:
            import trimesh  # noqa: PLC0415
        except Exception as exc:
            if not self._warned_robot_mesh:
                (self.logger or logger).warning("Robot mesh loading skipped; trimesh not available: %s", exc)
                self._warned_robot_mesh = True
            return

        try:
            urdf_path, asset_root = self._resolve_robot_asset_paths()
        except Exception as exc:
            if not self._warned_robot_mesh:
                (self.logger or logger).warning("Robot mesh loading skipped; URDF path error: %s", exc)
                self._warned_robot_mesh = True
            return

        if not os.path.exists(urdf_path):
            if not self._warned_robot_mesh:
                (self.logger or logger).warning("Robot mesh loading skipped; URDF not found: %s", urdf_path)
                self._warned_robot_mesh = True
            return

        try:
            root = ET.parse(urdf_path).getroot()
        except Exception as exc:
            if not self._warned_robot_mesh:
                (self.logger or logger).warning("Robot mesh loading skipped; URDF parse failed: %s", exc)
                self._warned_robot_mesh = True
            return

        body_names = getattr(self.env, "body_names", None) or getattr(self.env.robot_config, "body_names", None)
        if not body_names:
            if not self._warned_robot_mesh:
                (self.logger or logger).warning("Robot mesh loading skipped; body_names unavailable.")
                self._warned_robot_mesh = True
            return

        allowlist = getattr(self.env, "_perception_camera_mesh_allowlist", None)
        if allowlist is None:
            allowlist = getattr(self.cfg, "camera_mesh_allowlist", None)
        if allowlist is not None:
            allowlist = {str(name) for name in allowlist}

        name_to_index = {name: idx for idx, name in enumerate(body_names)}
        link_meshes: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}
        urdf_dir = os.path.dirname(urdf_path)

        # Strict far-tracking parity path: use explicit ray_cast_bodies mesh mapping.
        explicit_mesh_map = getattr(self.cfg, "camera_mesh_file_map", None)
        if explicit_mesh_map:
            for link_name, mesh_file in explicit_mesh_map.items():
                parent_name = str(link_name)
                if parent_name not in name_to_index:
                    resolved = resolve_fixed_link_offset(
                        self.env.robot_config,
                        parent_name,
                        available_links=body_names,
                        device="cpu",
                    )
                    if resolved is None:
                        continue
                    parent_name, _offset_pos, _offset_quat = resolved

                if allowlist is not None and link_name not in allowlist and parent_name not in allowlist:
                    continue

                link_index = name_to_index.get(parent_name)
                if link_index is None:
                    continue

                mesh_path = str(mesh_file)
                if mesh_path.startswith("package://"):
                    mesh_path = mesh_path[len("package://") :]
                    mesh_path = os.path.join(asset_root, mesh_path)
                elif mesh_path.startswith("file://"):
                    mesh_path = mesh_path[len("file://") :]
                elif not os.path.isabs(mesh_path):
                    if "/" in mesh_path or "\\" in mesh_path:
                        mesh_path = os.path.join(asset_root, mesh_path)
                    else:
                        mesh_path = os.path.join(urdf_dir, "meshes", mesh_path)

                if not os.path.exists(mesh_path):
                    continue

                mesh = trimesh.load(mesh_path, process=False)
                if isinstance(mesh, trimesh.Scene):
                    mesh = mesh.dump(concatenate=True)
                if not isinstance(mesh, trimesh.Trimesh):
                    continue

                verts = torch.as_tensor(mesh.vertices, dtype=torch.float32)
                faces = torch.as_tensor(mesh.faces, dtype=torch.int64)
                if verts.numel() == 0 or faces.numel() == 0:
                    continue

                link_meshes.setdefault(link_index, []).append((verts, faces))

            if link_meshes:
                self._robot_link_meshes = []
                for link_index in sorted(link_meshes.keys()):
                    parts = link_meshes[link_index]
                    if not parts:
                        continue
                    verts_list = []
                    faces_list = []
                    vert_offset = 0
                    for verts, faces in parts:
                        verts_list.append(verts)
                        faces_list.append(faces + vert_offset)
                        vert_offset += verts.shape[0]
                    if verts_list:
                        verts = torch.cat(verts_list, dim=0)
                        faces = torch.cat(faces_list, dim=0)
                        self._robot_link_meshes.append(
                            {
                                "link_index": link_index,
                                "vertices": verts,
                                "faces": faces,
                            }
                        )

                if self._robot_link_meshes:
                    self._camera_robot_mesh_enabled = True
                    return

        for link in root.findall("link"):
            link_name = link.get("name")
            if not link_name:
                continue

            parent_name = link_name
            offset_pos = torch.zeros(3, dtype=torch.float32)
            offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)

            if link_name not in name_to_index:
                resolved = resolve_fixed_link_offset(
                    self.env.robot_config,
                    link_name,
                    available_links=body_names,
                    device="cpu",
                )
                if resolved is None:
                    continue
                parent_name, offset_pos, offset_quat = resolved

            if allowlist is not None and link_name not in allowlist and parent_name not in allowlist:
                continue

            link_index = name_to_index.get(parent_name)
            if link_index is None:
                continue

            for visual in link.findall("visual"):
                geometry = visual.find("geometry")
                if geometry is None:
                    continue
                mesh = None
                mesh_tag = geometry.find("mesh")
                if mesh_tag is not None:
                    filename = mesh_tag.get("filename")
                    if not filename:
                        continue
                    mesh_path = self._resolve_urdf_mesh_path(urdf_dir, asset_root, filename)
                    if not os.path.exists(mesh_path):
                        continue
                    mesh = trimesh.load(mesh_path, process=False)
                    if isinstance(mesh, trimesh.Scene):
                        mesh = mesh.dump(concatenate=True)
                    if not isinstance(mesh, trimesh.Trimesh):
                        continue
                    scale = self._parse_urdf_vec3(mesh_tag.get("scale"), default=None)
                    if scale is not None:
                        mesh.apply_scale(scale.numpy())
                else:
                    box_tag = geometry.find("box")
                    cylinder_tag = geometry.find("cylinder")
                    sphere_tag = geometry.find("sphere")
                    if box_tag is not None:
                        size = self._parse_urdf_vec3(box_tag.get("size"), default=None)
                        if size is None:
                            continue
                        mesh = trimesh.creation.box(extents=size.numpy())
                    elif cylinder_tag is not None:
                        radius = float(cylinder_tag.get("radius", "0.0"))
                        length = float(cylinder_tag.get("length", "0.0"))
                        if radius <= 0.0 or length <= 0.0:
                            continue
                        mesh = trimesh.creation.cylinder(radius=radius, height=length)
                    elif sphere_tag is not None:
                        radius = float(sphere_tag.get("radius", "0.0"))
                        if radius <= 0.0:
                            continue
                        mesh = trimesh.creation.icosphere(radius=radius)
                    else:
                        continue

                origin = visual.find("origin")
                visual_pos = self._parse_urdf_vec3(
                    origin.get("xyz") if origin is not None else None, default=(0.0, 0.0, 0.0)
                )
                visual_rpy = self._parse_urdf_vec3(
                    origin.get("rpy") if origin is not None else None, default=(0.0, 0.0, 0.0)
                )
                visual_quat = quat_from_euler_xyz(visual_rpy[0], visual_rpy[1], visual_rpy[2])

                combined_pos = offset_pos + quat_apply(offset_quat.unsqueeze(0), visual_pos.unsqueeze(0), w_last=True)
                combined_pos = combined_pos.squeeze(0)
                combined_quat = quat_mul(offset_quat.unsqueeze(0), visual_quat.unsqueeze(0), w_last=True)
                combined_quat = combined_quat.squeeze(0)

                verts = torch.as_tensor(mesh.vertices, dtype=torch.float32)
                faces = torch.as_tensor(mesh.faces, dtype=torch.int64)
                if verts.numel() == 0 or faces.numel() == 0:
                    continue

                quat_batch = combined_quat.unsqueeze(0).expand(verts.shape[0], -1)
                verts = quat_apply(quat_batch, verts, w_last=True) + combined_pos

                link_meshes.setdefault(link_index, []).append((verts, faces))

        self._robot_link_meshes = []
        for link_index in sorted(link_meshes.keys()):
            parts = link_meshes[link_index]
            if not parts:
                continue
            verts_list = []
            faces_list = []
            vert_offset = 0
            for verts, faces in parts:
                verts_list.append(verts)
                faces_list.append(faces + vert_offset)
                vert_offset += verts.shape[0]
            if verts_list:
                verts = torch.cat(verts_list, dim=0)
                faces = torch.cat(faces_list, dim=0)
                self._robot_link_meshes.append(
                    {
                        "link_index": link_index,
                        "vertices": verts,
                        "faces": faces,
                    }
                )

        if not self._robot_link_meshes:
            if not self._warned_robot_mesh:
                (self.logger or logger).warning("Robot mesh loading found no visual meshes; skipping.")
                self._warned_robot_mesh = True
            return

        self._camera_robot_mesh_enabled = True

    def _apply_link_transform(
        self, vertices: torch.Tensor, link_pos: torch.Tensor, link_quat: torch.Tensor
    ) -> torch.Tensor:
        num_envs = link_pos.shape[0]
        num_verts = vertices.shape[0]
        verts = vertices.unsqueeze(0).expand(num_envs, -1, -1).reshape(-1, 3)
        quats = link_quat.unsqueeze(1).expand(num_envs, num_verts, 4).reshape(-1, 4)
        verts_world = quat_apply(quats, verts, w_last=True).view(num_envs, num_verts, 3)
        return verts_world + link_pos.unsqueeze(1)

    def _build_camera_warp_mesh(self, env_ids: torch.Tensor | None):
        if self._terrain_vertices is None or self._terrain_faces is None or not self._robot_link_meshes:
            return self._warp_mesh

        if env_ids is None or isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device="cpu")
        else:
            env_ids = env_ids.to("cpu")

        if env_ids.numel() == 0:
            return self._warp_mesh

        body_pos = self.env.simulator._rigid_body_pos[env_ids].detach().cpu()
        body_quat = self.env.simulator._rigid_body_rot[env_ids].detach().cpu()
        num_envs = body_pos.shape[0]

        vertices_chunks = [self._terrain_vertices]
        faces_chunks = [self._terrain_faces]
        vertex_offset = self._terrain_vertices.shape[0]

        for link_mesh in self._robot_link_meshes:
            link_index = link_mesh["link_index"]
            verts = link_mesh["vertices"]
            faces = link_mesh["faces"]
            if verts.numel() == 0 or faces.numel() == 0:
                continue

            link_pos = body_pos[:, link_index]
            link_quat = body_quat[:, link_index]
            verts_world = self._apply_link_transform(verts, link_pos, link_quat)
            vertices_chunks.append(verts_world.reshape(-1, 3).numpy())

            faces_np = faces.numpy()
            offsets = np.arange(num_envs, dtype=np.int64) * verts.shape[0] + vertex_offset
            faces_env = faces_np[None, :, :] + offsets[:, None, None]
            faces_chunks.append(faces_env.reshape(-1, 3))

            vertex_offset += num_envs * verts.shape[0]

        vertices = np.concatenate(vertices_chunks, axis=0)
        faces = np.concatenate(faces_chunks, axis=0)
        return warp_utils.convert_to_wp_mesh(vertices, faces, self.device)

    def _get_camera_warp_mesh(self, env_ids: torch.Tensor | None):
        if not self._camera_robot_mesh_enabled:
            return self._warp_mesh
        return self._build_camera_warp_mesh(env_ids)

    def _uses_raycast(self) -> bool:
        return self.cfg.output_mode == "heightmap"

    def _uses_camera_raycast(self) -> bool:
        return False

    def _uses_camera_far_tracking(self) -> bool:
        return self.cfg.output_mode == "camera_depth" and self._camera_source == "far_tracking_warp"

    def _uses_camera_scandots(self) -> bool:
        return False

    def _wants_camera_scandots(self) -> bool:
        return False

    def _uses_pytorch3d(self) -> bool:
        return False

    def _uses_rendered_camera(self) -> bool:
        return self.cfg.output_mode == "camera_depth" and self._camera_source in {"rendered", "rendered_depth_sensor"}

    def get_mujoco_render_camera_pose(self, env_ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Return camera world pose in MuJoCo camera convention.

        MuJoCo cameras use local axes: +x right, +y up, -z forward.
        The returned quaternion uses holosoma xyzw ordering.
        """
        if not self.enabled or self.cfg.output_mode != "camera_depth":
            raise RuntimeError("MuJoCo rendered camera pose requested but camera_depth output is disabled.")

        if self._camera_strict_warp:
            camera_pos, camera_quat = self._get_strict_warp_camera_pose(env_ids)
            # Match the strict warp_sensors camera frame exactly, then convert into
            # MuJoCo's render-camera convention (+x right, +y up, -z forward).
            #
            # MuJoCo fixed cameras feed the OpenGL renderer with:
            #   GL forward = model-camera local +x
            #   GL up      = model-camera local -y
            #
            # The strict warp camera uses local axes:
            #   +x = right, +y = down, +z = forward
            #
            # To make MuJoCo GL forward/up match warp forward/up, the model-camera
            # local frame must satisfy:
            #   +x_model -> +z_warp
            #   +y_model -> +y_warp
            #   +z_model -> -x_warp
            #
            # That is exactly a -90 degree rotation about +y.
            warp_to_mujoco = torch.tensor(
                [0.0, -0.70710677, 0.0, 0.70710677],
                device=camera_quat.device,
                dtype=camera_quat.dtype,
            )
            mujoco_from_warp = warp_to_mujoco.unsqueeze(0).expand(camera_quat.shape[0], -1)
            return camera_pos, quat_mul(camera_quat, mujoco_from_warp, w_last=True)

        idx = env_ids if env_ids is not None else slice(None)
        body_pos, body_quat = self._get_camera_body_pose(idx)
        num_envs = body_quat.shape[0]

        offset_world = quat_apply(body_quat, self._sensor_offset.expand(num_envs, -1), w_last=True)
        camera_pos = body_pos + offset_world

        center_u = float(int(self._camera_width / 2))
        center_v = float(int(self._camera_height / 2))
        right_u = float(min(self._camera_width - 1, int(self._camera_width / 2) + 1))
        down_v = float(min(self._camera_height - 1, int(self._camera_height / 2) + 1))

        u_coords = torch.tensor([center_u, right_u], device=self.device, dtype=torch.float32)
        v_coords = torch.tensor([center_v, down_v], device=self.device, dtype=torch.float32)
        dirs_base = self._build_camera_rays_from_coords(u_coords, v_coords).view(2, 2, 3)

        center_base = dirs_base[0, 0]
        right_base = dirs_base[0, 1]
        down_base = dirs_base[1, 0]

        center_world = quat_apply(body_quat, center_base.unsqueeze(0).expand(num_envs, -1), w_last=True)
        right_world = quat_apply(body_quat, right_base.unsqueeze(0).expand(num_envs, -1), w_last=True)
        down_world = quat_apply(body_quat, down_base.unsqueeze(0).expand(num_envs, -1), w_last=True)

        forward_world = center_world / torch.norm(center_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

        right_world = right_world - torch.sum(right_world * forward_world, dim=-1, keepdim=True) * forward_world
        right_world = right_world / torch.norm(right_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

        down_world = down_world - torch.sum(down_world * forward_world, dim=-1, keepdim=True) * forward_world
        down_world = down_world / torch.norm(down_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

        up_world = -down_world
        up_world = up_world - torch.sum(up_world * forward_world, dim=-1, keepdim=True) * forward_world
        up_world = up_world / torch.norm(up_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

        right_world = torch.cross(forward_world, up_world, dim=-1)
        right_world = right_world / torch.norm(right_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
        up_world = torch.cross(right_world, forward_world, dim=-1)
        up_world = up_world / torch.norm(up_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

        # MuJoCo cameras use local axes (+x right, +y up, -z forward). Keep the same
        # camera pose as the training-side ray builder; only the render-buffer readback
        # needs a vertical flip in MuJoCoDepthCamera.
        rot_world_from_cam = torch.stack((right_world, up_world, -forward_world), dim=-1)
        quat_wxyz = matrix_to_quaternion(rot_world_from_cam)
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
        return camera_pos, quat_xyzw

    def _resolve_heightmap_body_index(self) -> None:
        if self._heightmap_body_name is None:
            return
        body_names = getattr(self.env, "body_names", None)
        if body_names is not None and self._heightmap_body_name in body_names:
            self._heightmap_body_index = int(body_names.index(self._heightmap_body_name))
            return

        resolved = resolve_fixed_link_offset(
            self.env.robot_config,
            self._heightmap_body_name,
            available_links=body_names,
            device=self.device,
        )
        if resolved is None or body_names is None:
            available = body_names if body_names is not None else "unknown"
            raise RuntimeError(
                f"Heightmap body '{self._heightmap_body_name}' not found in body_names: {available}"
            )

        parent_name, offset_pos, offset_quat = resolved
        self._heightmap_body_index = int(body_names.index(parent_name))
        self._heightmap_body_offset_pos = offset_pos
        self._heightmap_body_offset_quat = offset_quat

    def _resolve_camera_body_index(self) -> None:
        if self._camera_body_name is None:
            return
        body_names = getattr(self.env, "body_names", None)
        if body_names is not None and self._camera_body_name in body_names:
            self._camera_body_index = int(body_names.index(self._camera_body_name))
            return

        resolved = resolve_fixed_link_offset(
            self.env.robot_config,
            self._camera_body_name,
            available_links=body_names,
            device=self.device,
        )
        if resolved is None or body_names is None:
            available = body_names if body_names is not None else "unknown"
            raise RuntimeError(f"Camera body '{self._camera_body_name}' not found in body_names: {available}")

        parent_name, offset_pos, offset_quat = resolved
        self._camera_body_index = int(body_names.index(parent_name))
        self._camera_body_offset_pos = offset_pos
        self._camera_body_offset_quat = offset_quat

    def _get_heightmap_body_pose(self, idx: torch.Tensor | slice) -> tuple[torch.Tensor, torch.Tensor]:
        if self._heightmap_body_index is not None:
            body_pos = self.env.simulator._rigid_body_pos[idx, self._heightmap_body_index]
            body_quat = self.env.simulator._rigid_body_rot[idx, self._heightmap_body_index]
        else:
            body_pos = self.env.simulator.robot_root_states[idx, :3]
            body_quat = self.env.base_quat[idx]
        if self._heightmap_body_offset_pos is not None:
            offset_pos = self._heightmap_body_offset_pos.expand(body_pos.shape[0], -1)
            offset_quat = self._heightmap_body_offset_quat.expand(body_pos.shape[0], -1)
            body_pos = body_pos + quat_apply(body_quat, offset_pos, w_last=True)
            body_quat = quat_mul(body_quat, offset_quat, w_last=True)
        return body_pos, body_quat

    def _get_heightmap_sampling_pose(
        self,
        idx: torch.Tensor | slice,
        *,
        apply_offsets: bool = True,
        apply_heading_only: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos, body_quat = self._get_heightmap_body_pose(idx)

        if apply_heading_only:
            body_quat = yaw_quat(body_quat, w_last=True)

        if apply_offsets:
            sensor_offset = self._sensor_offset.expand(body_pos.shape[0], -1)
            ray_start_offset = self._ray_start_offset.expand(body_pos.shape[0], -1)
            body_pos = body_pos + quat_apply(body_quat, sensor_offset, w_last=True)
            body_pos = body_pos + quat_apply(body_quat, ray_start_offset, w_last=True)

        return body_pos, body_quat

    def _get_camera_body_pose(self, idx: torch.Tensor | slice) -> tuple[torch.Tensor, torch.Tensor]:
        if self._camera_body_index is not None:
            body_pos = self.env.simulator._rigid_body_pos[idx, self._camera_body_index]
            body_quat = self.env.simulator._rigid_body_rot[idx, self._camera_body_index]
        else:
            body_pos = self.env.simulator.robot_root_states[idx, :3]
            body_quat = self.env.base_quat[idx]
        if self._camera_body_offset_pos is not None:
            offset_pos = self._camera_body_offset_pos.expand(body_pos.shape[0], -1)
            offset_quat = self._camera_body_offset_quat.expand(body_pos.shape[0], -1)
            body_pos = body_pos + quat_apply(body_quat, offset_pos, w_last=True)
            body_quat = quat_mul(body_quat, offset_quat, w_last=True)
        extra_pos = None if self._camera_disable_offsets else getattr(self.env, "_perception_camera_offset_pos", None)
        extra_quat = None if self._camera_disable_offsets else getattr(self.env, "_perception_camera_offset_quat", None)
        if extra_pos is not None and extra_quat is not None:
            if isinstance(idx, slice):
                offset_pos = extra_pos
                offset_quat = extra_quat
            else:
                offset_pos = extra_pos[idx]
                offset_quat = extra_quat[idx]
            body_pos = body_pos + quat_apply(body_quat, offset_pos, w_last=True)
            body_quat = quat_mul(body_quat, offset_quat, w_last=True)
        return body_pos, body_quat

    def _compute_rays(
        self, env_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._warp_mesh is None:
            raise RuntimeError("PerceptionManager.setup() must be called before update().")
        if self._grid_points_base is None or self._ray_dirs_base is None:
            raise RuntimeError("PerceptionManager grid buffers are not initialized.")

        idx = env_ids if env_ids is not None else slice(None)
        body_pos, _body_quat = self._get_heightmap_body_pose(idx)
        ray_origin_pos, sample_quat = self._get_heightmap_sampling_pose(
            idx,
            apply_offsets=True,
            apply_heading_only=self.cfg.use_heading_only,
        )
        num_envs = sample_quat.shape[0]

        grid_points = self._grid_points_base.unsqueeze(0).expand(num_envs, -1, -1)
        ray_dirs = self._ray_dirs_base.unsqueeze(0).expand(num_envs, -1, -1)

        quat_repeat = sample_quat.repeat(1, self._num_points)
        grid_world = quat_apply(quat_repeat, grid_points, w_last=True)
        ray_dirs_world = quat_apply(quat_repeat, ray_dirs, w_last=True)

        offset_world = ray_origin_pos - body_pos
        ray_starts = grid_world + ray_origin_pos.unsqueeze(1)
        ray_hits_world = warp_utils.ray_cast(ray_starts, ray_dirs_world, self._warp_mesh)

        return ray_starts, ray_dirs_world, ray_hits_world, body_pos, sample_quat, offset_world

    def _cast_camera_raycast_rays(
        self, env_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._warp_mesh is None:
            raise RuntimeError("PerceptionManager.setup() must be called before update().")
        if self._camera_ray_dirs_base is None:
            raise RuntimeError("PerceptionManager camera ray buffers are not initialized.")

        idx = env_ids if env_ids is not None else slice(None)
        body_pos, body_quat = self._get_camera_body_pose(idx)
        num_envs = body_pos.shape[0]

        ray_dirs_base = self._camera_ray_dirs_base.unsqueeze(0).expand(num_envs, -1, -1)
        ray_dirs_world = quat_rotate_batched(body_quat, ray_dirs_base)

        offset_world = quat_apply(body_quat, self._sensor_offset.expand(num_envs, -1), w_last=True)
        ray_starts = body_pos.unsqueeze(1) + offset_world.unsqueeze(1)
        if ray_starts.shape[1] != ray_dirs_world.shape[1]:
            ray_starts = ray_starts.expand(-1, ray_dirs_world.shape[1], -1)

        warp_mesh = self._get_camera_warp_mesh(env_ids)
        ray_hits_world = warp_utils.ray_cast(ray_starts, ray_dirs_world, warp_mesh)
        hit_mask = torch.isfinite(ray_hits_world).all(dim=-1)
        return ray_starts, ray_dirs_world, ray_hits_world, hit_mask, body_quat

    def _compute_camera_raycast_depth(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        ray_starts, ray_dirs_world, ray_hits_world, hit_mask, body_quat = self._cast_camera_raycast_rays(env_ids)
        num_envs = body_quat.shape[0]
        ranges = self._compute_camera_ray_distances(ray_starts, ray_dirs_world, ray_hits_world)
        hit_mask = self._filter_camera_hit_mask_by_depth(ranges, ray_dirs_world, body_quat, hit_mask)
        depth = self._project_ranges_to_camera_depth(ranges, ray_dirs_world, body_quat, hit_mask)
        depth = depth.view(num_envs, self._camera_height, self._camera_width)
        return self._apply_camera_depth_noise(depth)

    def _compute_camera_scandots_depth(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        ray_starts, ray_dirs_world, ray_hits_world, hit_mask, body_quat = self._cast_camera_scandots_rays(env_ids)
        num_envs = body_quat.shape[0]
        ranges = self._compute_camera_ray_distances(ray_starts, ray_dirs_world, ray_hits_world)
        hit_mask = self._filter_camera_hit_mask_by_depth(ranges, ray_dirs_world, body_quat, hit_mask)

        height = self._camera_scandots_height
        width = self._camera_scandots_width
        if height is None or width is None:
            raise RuntimeError("PerceptionManager scandots grid size is not initialized.")

        depth = self._project_ranges_to_camera_depth(ranges, ray_dirs_world, body_quat, hit_mask)
        depth = depth.view(num_envs, height, width).unsqueeze(1)

        upsample_mode = self.cfg.camera_scandots_upsample
        if upsample_mode in {"bilinear", "bicubic"}:
            depth = F.interpolate(
                depth,
                size=(self._camera_height, self._camera_width),
                mode=upsample_mode,
                align_corners=False,
            )
        else:
            depth = F.interpolate(depth, size=(self._camera_height, self._camera_width), mode=upsample_mode)
        depth = depth.squeeze(1)
        return self._apply_camera_depth_noise(depth)

    def _cast_camera_scandots_rays(
        self, env_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Ray-cast camera scandots using the same pinhole + mesh flow as warp_sensors."""
        if self._warp_mesh is None:
            raise RuntimeError("PerceptionManager.setup() must be called before scandots ray queries.")
        if self._camera_scandots_ray_dirs_base is None:
            raise RuntimeError(
                "PerceptionManager scandots ray buffers are not initialized. "
                "Set camera_scandots_width/height or camera_scandots_stride to enable visualization."
            )

        idx = env_ids if env_ids is not None else slice(None)
        body_pos, body_quat = self._get_camera_body_pose(idx)
        num_envs = body_pos.shape[0]

        ray_dirs_base = self._camera_scandots_ray_dirs_base.unsqueeze(0).expand(num_envs, -1, -1)
        ray_dirs_world = quat_rotate_batched(body_quat, ray_dirs_base)

        offset_world = quat_apply(body_quat, self._sensor_offset.expand(num_envs, -1), w_last=True)
        ray_starts = body_pos.unsqueeze(1) + offset_world.unsqueeze(1)
        if ray_starts.shape[1] != ray_dirs_world.shape[1]:
            ray_starts = ray_starts.expand(-1, ray_dirs_world.shape[1], -1)

        warp_mesh = self._get_camera_warp_mesh(env_ids)
        ray_hits_world = warp_utils.ray_cast(ray_starts, ray_dirs_world, warp_mesh)
        hit_mask = torch.isfinite(ray_hits_world).all(dim=-1)
        return ray_starts, ray_dirs_world, ray_hits_world, hit_mask, body_quat

    def _compute_camera_ray_distances(
        self, ray_starts: torch.Tensor, ray_dirs: torch.Tensor, ray_hits_world: torch.Tensor
    ) -> torch.Tensor:
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True).clamp(min=1.0e-6)
        delta = ray_hits_world - ray_starts
        distances = torch.sum(delta * ray_dirs, dim=-1)
        distances = torch.where(torch.isfinite(distances), distances, torch.zeros_like(distances))
        return torch.clamp(distances, min=0.0)

    def _setup_pytorch3d_renderer(self) -> None:
        if self._terrain_mesh is None:
            raise RuntimeError("PerceptionManager requires terrain mesh for pytorch3d rendering.")
        try:
            from pytorch3d.renderer import RasterizationSettings  # noqa: PLC0415
            from pytorch3d.structures import Meshes  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pytorch3d is not available; install it to use camera_source=pytorch3d.") from exc

        verts = torch.as_tensor(self._terrain_mesh.vertices, device=self.device, dtype=torch.float32)
        faces = torch.as_tensor(self._terrain_mesh.faces, device=self.device, dtype=torch.int64)
        self._pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
        self._pytorch3d_mesh_cache = {}
        self._pytorch3d_raster_settings = RasterizationSettings(
            image_size=(self._camera_height, self._camera_width),
            blur_radius=0.0,
            faces_per_pixel=1,
        )

    def _get_pytorch3d_mesh_batch(self, batch_size: int):
        if self._pytorch3d_mesh is None:
            raise RuntimeError("PerceptionManager.setup() must be called before pytorch3d rendering.")
        if batch_size == 1:
            return self._pytorch3d_mesh
        cached = self._pytorch3d_mesh_cache.get(batch_size)
        if cached is None:
            cached = self._pytorch3d_mesh.extend(batch_size)
            self._pytorch3d_mesh_cache[batch_size] = cached
        return cached

    def _build_pytorch3d_cameras(
        self, rotation: torch.Tensor, translation: torch.Tensor, k_matrix: torch.Tensor, image_size: torch.Tensor
    ):
        try:
            from pytorch3d.renderer.cameras import cameras_from_opencv_projection  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pytorch3d is missing cameras_from_opencv_projection; install/upgrade pytorch3d "
                "to use camera_source=pytorch3d."
            ) from exc

        return cameras_from_opencv_projection(rotation, translation, k_matrix, image_size=image_size)

    def _compute_pytorch3d_depth(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if self._pytorch3d_mesh is None or self._pytorch3d_raster_settings is None:
            raise RuntimeError("PerceptionManager.setup() must be called before pytorch3d rendering.")

        try:
            from pytorch3d.renderer import MeshRasterizer  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("pytorch3d is not available; install it to use camera_source=pytorch3d.") from exc

        idx = env_ids if env_ids is not None else slice(None)
        body_pos, body_quat = self._get_camera_body_pose(idx)
        num_envs = body_pos.shape[0]

        offset_world = quat_apply(body_quat, self._sensor_offset.expand(num_envs, -1), w_last=True)
        camera_pos = body_pos + offset_world

        pitch_deg = float(self.cfg.camera_pitch_deg)
        pitch_rad = torch.deg2rad(torch.tensor(pitch_deg, device=self.device))
        pitch_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=self.device),
            pitch_rad,
            torch.tensor(0.0, device=self.device),
        )
        pitch_quat = pitch_quat.unsqueeze(0).expand(num_envs, -1)
        camera_quat = quat_mul(body_quat, pitch_quat, w_last=True)

        forward_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(num_envs, -1)
        right_axis = torch.tensor([0.0, -1.0, 0.0], device=self.device).expand(num_envs, -1)
        up_axis = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(num_envs, -1)

        forward_world = quat_apply(camera_quat, forward_axis, w_last=True)
        right_world = quat_apply(camera_quat, right_axis, w_last=True)
        up_world = quat_apply(camera_quat, up_axis, w_last=True)

        forward_world = forward_world / torch.norm(forward_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
        right_world = right_world / torch.norm(right_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
        up_world = up_world / torch.norm(up_world, dim=-1, keepdim=True).clamp(min=1.0e-6)

        x_axis = right_world
        y_axis = -up_world
        z_axis = forward_world
        rotation = torch.stack((x_axis, y_axis, z_axis), dim=1)
        translation = -(torch.bmm(rotation, camera_pos.unsqueeze(-1))).squeeze(-1)

        k_matrix = torch.zeros(num_envs, 3, 3, device=self.device)
        k_matrix[:, 0, 0] = self._camera_fx
        k_matrix[:, 1, 1] = self._camera_fy
        k_matrix[:, 0, 2] = self._camera_cx
        k_matrix[:, 1, 2] = self._camera_cy
        k_matrix[:, 2, 2] = 1.0

        image_size = torch.tensor([self._camera_height, self._camera_width], device=self.device).repeat(num_envs, 1)
        cameras = self._build_pytorch3d_cameras(rotation, translation, k_matrix, image_size)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=self._pytorch3d_raster_settings)

        meshes = self._get_pytorch3d_mesh_batch(num_envs)
        fragments = rasterizer(meshes)
        depth = fragments.zbuf[..., 0]

        depth = torch.where(
            torch.isfinite(depth) & (depth > 0.0),
            depth,
            torch.full_like(depth, self.cfg.max_distance),
        )
        return torch.clamp(depth, min=0.0, max=self.cfg.max_distance)

    def _compute_ray_distances(
        self, ray_starts: torch.Tensor, ray_dirs: torch.Tensor, ray_hits_world: torch.Tensor
    ) -> torch.Tensor:
        ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True).clamp(min=1.0e-6)
        delta = ray_hits_world - ray_starts
        distances = torch.sum(delta * ray_dirs, dim=-1)
        distances = torch.where(torch.isfinite(distances), distances, torch.full_like(distances, self.cfg.max_distance))
        return torch.clamp(distances, min=0.0, max=self.cfg.max_distance)

    def _project_to_camera(
        self,
        ray_hits_world: torch.Tensor,
        root_pos: torch.Tensor,
        base_quat: torch.Tensor,
        offset_world: torch.Tensor,
    ) -> torch.Tensor:
        num_envs = ray_hits_world.shape[0]
        camera_pos = root_pos + offset_world
        points_relative = ray_hits_world - camera_pos.unsqueeze(1)
        points_base = quat_rotate_inverse_batched(base_quat, points_relative)

        pitch_deg = float(self.cfg.camera_pitch_deg)
        pitch_rad = torch.deg2rad(torch.tensor(pitch_deg, device=self.device))
        pitch_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=self.device),
            pitch_rad,
            torch.tensor(0.0, device=self.device),
        )
        pitch_quat = pitch_quat.unsqueeze(0).expand(num_envs, -1)
        points_cam = quat_rotate_batched(pitch_quat, points_base)

        # Camera frame: forward = +x, right = -y, up = +z (from base frame).
        z = points_cam[..., 0]
        x = -points_cam[..., 1]
        y = points_cam[..., 2]

        z_safe = torch.where(z.abs() < 1.0e-6, torch.full_like(z, 1.0e-6), z)
        u = self._camera_fx * (x / z_safe) + self._camera_cx
        v = self._camera_fy * (y / z_safe) + self._camera_cy

        res_h = self._camera_height
        res_w = self._camera_width

        valid = torch.isfinite(z) & (z > 0.0)
        valid &= (u >= 0.0) & (u < res_w) & (v >= 0.0) & (v < res_h)

        u_idx = u.round().long()
        v_idx = v.round().long()

        depth_map = torch.full((num_envs, res_h, res_w), self.cfg.max_distance, device=self.device)

        if valid.any():
            env_ids = torch.arange(num_envs, device=self.device).unsqueeze(1).expand_as(u_idx)
            flat_indices = (env_ids * res_h * res_w + v_idx * res_w + u_idx).view(-1)
            flat_depths = z.view(-1)
            flat_valid = valid.view(-1)

            flat_indices = flat_indices[flat_valid]
            flat_depths = flat_depths[flat_valid]

            if flat_indices.numel() > 0:
                flat_map = depth_map.view(-1)
                if hasattr(flat_map, "scatter_reduce_"):
                    flat_map.scatter_reduce_(0, flat_indices, flat_depths, reduce="amin", include_self=True)
                else:
                    for idx, depth in zip(flat_indices.tolist(), flat_depths.tolist()):
                        if depth < flat_map[idx]:
                            flat_map[idx] = depth

        return depth_map

    def _resolve_camera_resolution(self) -> tuple[int, int]:
        width = self.cfg.camera_width if self.cfg.camera_width is not None else self.cfg.grid_size
        height = self.cfg.camera_height if self.cfg.camera_height is not None else self.cfg.grid_size
        return int(width), int(height)
