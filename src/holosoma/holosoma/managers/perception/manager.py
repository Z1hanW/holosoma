"""Perception manager for heightmap and camera-style observations."""

from __future__ import annotations

from typing import Any

from holosoma.config_types.perception import PerceptionConfig
from holosoma.utils.camera_utils import build_camera_parameters, resolve_camera_intrinsics
from holosoma.utils import warp_utils
from holosoma.utils.rotations import (
    quat_apply,
    quat_apply_yaw,
    quat_from_euler_xyz,
    quat_rotate_batched,
    quat_rotate_inverse_batched,
)
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type
from holosoma.utils.safe_torch_import import torch


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
        self._grid_points_base: torch.Tensor | None = None
        self._ray_dirs_base: torch.Tensor | None = None
        self._sensor_offset = torch.tensor(cfg.sensor_offset, device=self.device)
        self._ray_start_offset = torch.tensor([0.0, 0.0, cfg.ray_start_height], device=self.device)
        self._camera_source = cfg.camera_source
        self._rendered_camera = None
        self._rendered_camera_env_id = int(getattr(cfg, "camera_env_id", 0))

        if cfg.output_mode == "camera_depth" and self._camera_source not in {"raycast", "rendered"}:
            raise ValueError(f"Unsupported camera_source: {self._camera_source}")

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

        self._num_points = cfg.grid_size * cfg.grid_size
        self._update_interval = 0.0 if cfg.update_hz <= 0 else 1.0 / cfg.update_hz
        self._time_since_update = 0.0

        self._heightmap = torch.zeros(self.num_envs, cfg.grid_size, cfg.grid_size, device=self.device)
        self._camera_depth = torch.full(
            (self.num_envs, self._camera_height, self._camera_width),
            cfg.max_distance,
            device=self.device,
        )

        self._ray_hits_world = torch.zeros(self.num_envs, self._num_points, 3, device=self.device)

    def setup(self) -> None:
        if not self.enabled:
            return
        if self._uses_raycast():
            terrain_term = getattr(self.env, "terrain_manager", None)
            if terrain_term is None or not hasattr(terrain_term, "terrain_term"):
                raise RuntimeError("PerceptionManager requires an initialized terrain_manager.")
            terrain_state = terrain_term.terrain_term
            if not hasattr(terrain_state, "warp_mesh"):
                raise RuntimeError("PerceptionManager requires terrain term with warp_mesh support.")
            self._warp_mesh = terrain_state.warp_mesh

            self._grid_points_base, self._ray_dirs_base = self._build_grid()

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
            self._camera_depth[self._rendered_camera_env_id] = camera_depth.squeeze(0)
            return

        ray_starts, ray_dirs, ray_hits_world, root_pos, base_quat, offset_world = self._compute_rays(env_ids)
        distances = self._compute_ray_distances(ray_starts, ray_dirs, ray_hits_world)
        heightmap = distances.view(-1, self.cfg.grid_size, self.cfg.grid_size)

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
        half_extent = (self.cfg.grid_size - 1) * self.cfg.grid_interval / 2.0
        coords = torch.linspace(
            -half_extent,
            half_extent,
            self.cfg.grid_size,
            device=self.device,
            requires_grad=False,
        )
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
        grid_points = torch.zeros(self._num_points, 3, device=self.device)
        grid_points[:, 0] = grid_x.flatten()
        grid_points[:, 1] = grid_y.flatten()

        ray_dirs = torch.zeros(self._num_points, 3, device=self.device)
        ray_dirs[:, 2] = -1.0
        return grid_points, ray_dirs

    def _setup_rendered_camera(self) -> None:
        if get_simulator_type() != SimulatorType.ISAACSIM:
            raise RuntimeError("Rendered camera requires IsaacSim. Use camera_source=raycast for other simulators.")
        from holosoma.simulator.isaacsim.perception_camera import IsaacSimDepthCamera

        self._rendered_camera = IsaacSimDepthCamera(
            env=self.env,
            config=self.cfg,
            width=self._camera_width,
            height=self._camera_height,
            vfov_deg=self._camera_vfov_deg,
            device=getattr(self.env.simulator, "device", self.device),
        )
        self._rendered_camera.setup()

    def _uses_raycast(self) -> bool:
        if self.cfg.output_mode == "heightmap":
            return True
        return self.cfg.output_mode == "camera_depth" and self._camera_source == "raycast"

    def _uses_rendered_camera(self) -> bool:
        return self.cfg.output_mode == "camera_depth" and self._camera_source == "rendered"

    def _compute_rays(
        self, env_ids: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._warp_mesh is None:
            raise RuntimeError("PerceptionManager.setup() must be called before update().")
        if self._grid_points_base is None or self._ray_dirs_base is None:
            raise RuntimeError("PerceptionManager grid buffers are not initialized.")

        idx = env_ids if env_ids is not None else slice(None)
        base_quat = self.env.base_quat[idx]
        root_pos = self.env.simulator.robot_root_states[idx, :3]
        num_envs = base_quat.shape[0]

        grid_points = self._grid_points_base.unsqueeze(0).expand(num_envs, -1, -1)
        ray_dirs = self._ray_dirs_base.unsqueeze(0).expand(num_envs, -1, -1)

        quat_repeat = base_quat.repeat(1, self._num_points)
        if self.cfg.use_heading_only:
            grid_world = quat_apply_yaw(quat_repeat, grid_points, w_last=True)
            ray_dirs_world = quat_apply_yaw(quat_repeat, ray_dirs, w_last=True)
            offset_world = quat_apply_yaw(base_quat, self._sensor_offset.expand(num_envs, -1), w_last=True)
            height_offset = quat_apply_yaw(base_quat, self._ray_start_offset.expand(num_envs, -1), w_last=True)
        else:
            grid_world = quat_apply(quat_repeat, grid_points, w_last=True)
            ray_dirs_world = quat_apply(quat_repeat, ray_dirs, w_last=True)
            offset_world = quat_apply(base_quat, self._sensor_offset.expand(num_envs, -1), w_last=True)
            height_offset = quat_apply(base_quat, self._ray_start_offset.expand(num_envs, -1), w_last=True)

        ray_starts = grid_world + root_pos.unsqueeze(1) + offset_world.unsqueeze(1) + height_offset.unsqueeze(1)
        ray_hits_world = warp_utils.ray_cast(ray_starts, ray_dirs_world, self._warp_mesh)

        return ray_starts, ray_dirs_world, ray_hits_world, root_pos, base_quat, offset_world

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

        pitch_rad = torch.deg2rad(torch.tensor(self.cfg.camera_pitch_deg, device=self.device))
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
