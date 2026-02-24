"""Perception manager for heightmap and camera-style observations."""

from __future__ import annotations

from typing import Any
import os
import xml.etree.ElementTree as ET

import numpy as np

from loguru import logger

from holosoma.config_types.perception import PerceptionConfig
from holosoma.utils.camera_utils import build_camera_parameters, resolve_camera_intrinsics
from holosoma.utils import warp_utils
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
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
        self._sensor_offset = torch.tensor(cfg.sensor_offset, device=self.device)
        self._ray_start_offset = torch.tensor([0.0, 0.0, cfg.ray_start_height], device=self.device)
        self._camera_source = cfg.camera_source
        self._camera_body_name = cfg.camera_body_name
        heightmap_body_name = cfg.heightmap_body_name
        if heightmap_body_name is None:
            robot_cfg = getattr(env, "robot_config", None)
            heightmap_body_name = getattr(robot_cfg, "torso_name", None)
        self._heightmap_body_name = heightmap_body_name
        self._camera_include_robot_mesh = bool(getattr(cfg, "camera_include_robot_mesh", False))
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
        self._camera_mount_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._use_camera_mount_quat = False
        self._camera_frame_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._use_camera_frame_quat = False
        strict_warp_raw = os.environ.get("HOLOSOMA_CAMERA_STRICT_WARP", "0").strip().lower()
        self._camera_strict_warp = strict_warp_raw not in {"0", "false", "no", "off", ""}
        self._camera_ray_correction_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device, dtype=torch.float32)
        auto_fix_raw = os.environ.get("HOLOSOMA_CAMERA_AUTOFIX_BACKWARD", "0").strip().lower()
        self._camera_auto_fix_backward = (auto_fix_raw not in {"0", "false", "no", "off", ""}) and (
            not self._camera_strict_warp
        )
        disable_offsets_raw = os.environ.get("HOLOSOMA_CAMERA_DISABLE_OFFSETS", "0").strip().lower()
        self._camera_disable_offsets = disable_offsets_raw not in {"0", "false", "no", "off", ""}
        threshold_raw = os.environ.get("HOLOSOMA_CAMERA_BACKWARD_RATIO_THRESHOLD", "0.6").strip()
        try:
            self._camera_backward_ratio_threshold = float(threshold_raw)
        except Exception:
            self._camera_backward_ratio_threshold = 0.6
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
        if cfg.output_mode == "camera_depth" and self._camera_source != "mesh_raycast":
            raise ValueError(
                "Unsupported camera_source. Only 'mesh_raycast' is supported for camera_depth "
                "(use perception:camera_depth_d435i)."
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
        self._warned_invalid_rendered_depth = False

        self._ray_hits_world = torch.zeros(self.num_envs, self._num_points, 3, device=self.device)

    def setup(self) -> None:
        if not self.enabled:
            return
        if (
            self._uses_raycast()
            or self._uses_camera_raycast()
            or self._uses_camera_scandots()
            or self._uses_pytorch3d()
        ):
            terrain_term = getattr(self.env, "terrain_manager", None)
            if terrain_term is None or not hasattr(terrain_term, "terrain_term"):
                raise RuntimeError("PerceptionManager requires an initialized terrain_manager.")
            terrain_state = terrain_term.terrain_term
            if self._uses_raycast() or self._uses_camera_raycast() or self._uses_camera_scandots():
                if not hasattr(terrain_state, "warp_mesh"):
                    raise RuntimeError("PerceptionManager requires terrain term with warp_mesh support.")
                self._warp_mesh = terrain_state.warp_mesh
            if self._uses_pytorch3d() or (
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
            self._ray_hits_world.zero_()
            return
        self._heightmap[env_ids] = 0.0
        self._camera_depth[env_ids] = self.cfg.max_distance
        self._ray_hits_world[env_ids] = 0.0

    def update(self, env_ids: torch.Tensor | None = None) -> None:
        if not self.enabled:
            return
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
            self._camera_depth[self._rendered_camera_env_id] = camera_depth.squeeze(0)
            return

        if self._uses_pytorch3d():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_pytorch3d_depth(env_ids)
            self._camera_depth[idx] = camera_depth
            return

        if self._uses_camera_scandots():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_camera_scandots_depth(env_ids)
            self._camera_depth[idx] = camera_depth
            return

        if self._uses_camera_raycast():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_camera_raycast_depth(env_ids)
            self._camera_depth[idx] = camera_depth
            return

        ray_starts, ray_dirs, ray_hits_world, root_pos, base_quat, offset_world = self._compute_rays(env_ids)
        distances = self._compute_ray_distances(ray_starts, ray_dirs, ray_hits_world)
        heightmap = distances.view(-1, self._heightmap_grid_x, self._heightmap_grid_y)

        idx = env_ids if env_ids is not None else slice(None)
        self._heightmap[idx] = heightmap
        self._ray_hits_world[idx] = ray_hits_world

        if self.cfg.output_mode == "camera_depth":
            camera_depth = self._project_to_camera(ray_hits_world, root_pos, base_quat, offset_world)
            self._camera_depth[idx] = camera_depth

    def get_obs(self) -> torch.Tensor:
        if not self.enabled:
            raise RuntimeError("Perception is disabled but perception observations were requested.")
        if self.cfg.output_mode == "heightmap":
            return self._heightmap.view(self.num_envs, -1)
        if self.cfg.output_mode == "camera_depth":
            return self._camera_depth.view(self.num_envs, -1)
        raise ValueError(f"Unsupported perception output_mode: {self.cfg.output_mode}")

    def get_camera_depth_map(self) -> torch.Tensor:
        if not self.enabled or self.cfg.output_mode != "camera_depth":
            raise RuntimeError("Camera depth map requested but camera_depth output is disabled.")
        return self._camera_depth

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
                f"Camera depth ray samples require camera_source=mesh_raycast or mesh_raycast_scandots, got: {self._camera_source}"
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
        body_pos, body_quat = self._get_heightmap_body_pose(idx)
        num_envs = body_quat.shape[0]

        grid_points = self._grid_points_base.unsqueeze(0).expand(num_envs, -1, -1)
        ray_dirs = self._ray_dirs_base.unsqueeze(0).expand(num_envs, -1, -1)

        quat_repeat = body_quat.repeat(1, self._num_points)
        if self.cfg.use_heading_only:
            grid_world = quat_apply_yaw(quat_repeat, grid_points, w_last=True)
            ray_dirs_world = quat_apply_yaw(quat_repeat, ray_dirs, w_last=True)
        else:
            grid_world = quat_apply(quat_repeat, grid_points, w_last=True)
            ray_dirs_world = quat_apply(quat_repeat, ray_dirs, w_last=True)

        offset_world = torch.zeros_like(body_pos)
        height_offset = torch.zeros_like(body_pos)
        ray_starts = grid_world + body_pos.unsqueeze(1)
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
        body_pos, body_quat = self._get_heightmap_body_pose(idx)

        if apply_heading_only and self.cfg.use_heading_only:
            body_quat = yaw_quat(body_quat, w_last=True)

        if apply_offsets:
            offset_world = quat_apply(body_quat, self._sensor_offset.expand(body_pos.shape[0], -1), w_last=True)
            height_offset = quat_apply(body_quat, self._ray_start_offset.expand(body_pos.shape[0], -1), w_last=True)
            body_pos = body_pos + offset_world + height_offset

        return body_pos, body_quat

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
        front = dots > 1.0e-6
        depth = ranges * torch.where(front, dots, torch.ones_like(dots))
        near = float(getattr(self.cfg, "camera_near", 0.0) or 0.0)
        # Match warp_sensors semantics: only rays whose projected depth is inside [near, max_distance] are valid hits.
        hit_mask = (
            hit_mask
            & front
            & torch.isfinite(depth)
            & (depth >= (near - 1.0e-6))
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
        front = dots > 1.0e-6
        depth = ranges * torch.where(front, dots, torch.ones_like(dots))
        near = float(getattr(self.cfg, "camera_near", 0.0) or 0.0)
        return (
            hit_mask
            & front
            & torch.isfinite(depth)
            & (depth >= (near - 1.0e-6))
            & (depth <= (self.cfg.max_distance + 1.0e-6))
        )

    def _compute_camera_miss_ranges(
        self, ray_dirs_world: torch.Tensor, body_quat: torch.Tensor
    ) -> torch.Tensor:
        ray_dirs_world = ray_dirs_world / torch.norm(ray_dirs_world, dim=-1, keepdim=True).clamp(min=1.0e-6)
        forward_world = self._get_camera_forward_axis(body_quat)
        dots = torch.sum(ray_dirs_world * forward_world.unsqueeze(1), dim=-1)
        front = dots > 1.0e-6
        miss_front = self.cfg.max_distance / torch.clamp(dots, min=1.0e-6)
        return torch.where(front, miss_front, torch.zeros_like(miss_front))

    def _apply_camera_depth_noise(self, depth: torch.Tensor) -> torch.Tensor:
        std_mult = getattr(self.env, "_perception_camera_noise_std_mult", None)
        drop_prob = getattr(self.env, "_perception_camera_noise_drop_prob", None)
        if std_mult is None and drop_prob is None:
            return depth

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

        return torch.clamp(depth_out, min=0.0, max=self.cfg.max_distance)

    def _setup_rendered_camera(self) -> None:
        if get_simulator_type() != SimulatorType.ISAACSIM:
            raise RuntimeError(
                "Rendered camera requires IsaacSim. Use camera_source=raycast, mesh_raycast, or pytorch3d "
                "for other simulators."
            )
        from holosoma.simulator.isaacsim.perception_camera import (
            IsaacSimDepthCamera,
            IsaacSimDepthSensorCamera,
        )

        camera_cls = IsaacSimDepthSensorCamera if self._camera_source == "rendered_depth_sensor" else IsaacSimDepthCamera
        self._rendered_camera = camera_cls(
            env=self.env,
            config=self.cfg,
            width=self._camera_width,
            height=self._camera_height,
            vfov_deg=self._camera_vfov_deg,
            device=getattr(self.env.simulator, "device", self.device),
        )
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
        return self.cfg.output_mode == "camera_depth" and self._camera_source == "mesh_raycast"

    def _uses_camera_scandots(self) -> bool:
        return False

    def _wants_camera_scandots(self) -> bool:
        return False

    def _uses_pytorch3d(self) -> bool:
        return False

    def _uses_rendered_camera(self) -> bool:
        return False

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
        body_pos, body_quat = self._get_heightmap_body_pose(idx)
        num_envs = body_quat.shape[0]

        grid_points = self._grid_points_base.unsqueeze(0).expand(num_envs, -1, -1)
        ray_dirs = self._ray_dirs_base.unsqueeze(0).expand(num_envs, -1, -1)

        quat_repeat = body_quat.repeat(1, self._num_points)
        if self.cfg.use_heading_only:
            grid_world = quat_apply_yaw(quat_repeat, grid_points, w_last=True)
            ray_dirs_world = quat_apply_yaw(quat_repeat, ray_dirs, w_last=True)
        else:
            grid_world = quat_apply(quat_repeat, grid_points, w_last=True)
            ray_dirs_world = quat_apply(quat_repeat, ray_dirs, w_last=True)

        offset_world = torch.zeros_like(body_pos)
        height_offset = torch.zeros_like(body_pos)
        ray_starts = grid_world + body_pos.unsqueeze(1)
        ray_hits_world = warp_utils.ray_cast(ray_starts, ray_dirs_world, self._warp_mesh)

        return ray_starts, ray_dirs_world, ray_hits_world, body_pos, body_quat, offset_world

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
        delta = ray_starts - ray_hits_world
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
