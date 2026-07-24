"""Perception manager for heightmap and camera-style observations."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np

from loguru import logger

from holosoma.config_types.perception import PerceptionConfig
from holosoma.utils.camera_utils import build_camera_parameters, resolve_camera_intrinsics
from holosoma.utils.common import rank_training_seed
from holosoma.utils import warp_utils
from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
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


def _validated_rank_local_perlin_seed(env: Any) -> int:
    """Bind the private hole stream to the declared distributed seed contract."""

    training_config = getattr(env, "training_config", None)
    base_seed = getattr(training_config, "seed", None)
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        global_rank = int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise ValueError(
            "WORLD_SIZE and RANK must be base-10 integers before constructing "
            "a rank-local Perlin producer."
        ) from exc
    expected_seed = rank_training_seed(
        base_seed,
        world_size=world_size,
        global_rank=global_rank,
    )
    live_initial_seed = int(torch.initial_seed())
    if live_initial_seed != expected_seed:
        raise RuntimeError(
            "Rank-local Perlin seed contract mismatch before generator construction: "
            f"torch.initial_seed()={live_initial_seed}, expected training.seed + global_rank="
            f"{expected_seed}."
        )
    return expected_seed


class _InfiniteFractalPerlin3D:
    LEGACY_SEED_SEMANTICS = "legacy_fixed_v1"
    RANK_LOCAL_SEED_SEMANTICS = "rank_local_v2"
    LEGACY_GRADIENT_SEED_MIXER = "python_tuple_hash_mod_2147483647_v1"
    RANK_LOCAL_GRADIENT_SEED_MIXER = "sha256_u63_be_v1"
    LEGACY_SINGLE_OCTAVE_PROFILE = "legacy_single_octave_v1"
    CUSTOM_EXPLICIT_OCTAVE_PROFILE = "custom_explicit_v1"

    def __init__(
        self,
        shape: tuple[int, int],
        resolutions: list[tuple[int, int]],
        periods: list[int],
        factors: list[float],
        *,
        batch_size: int,
        device: torch.device | str,
        seed_semantics: str = LEGACY_SEED_SEMANTICS,
        effective_seed: int | None = None,
        octave_profile: str = CUSTOM_EXPLICIT_OCTAVE_PROFILE,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, numbers.Integral)
            or int(batch_size) < 1
        ):
            raise ValueError(f"Perlin batch_size must be a positive integer, got {batch_size!r}.")
        batch_size = int(batch_size)
        if (
            len(shape) != 2
            or any(isinstance(value, bool) or not isinstance(value, numbers.Integral) or int(value) < 1 for value in shape)
        ):
            raise ValueError(f"Perlin shape must contain two positive integers, got {shape!r}.")
        if not resolutions or len(resolutions) != len(periods):
            raise ValueError(
                "Perlin resolutions and periods must be non-empty and have equal lengths."
            )
        for index, resolution in enumerate(resolutions):
            if (
                len(resolution) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, numbers.Integral)
                    or int(value) < 1
                    for value in resolution
                )
            ):
                raise ValueError(
                    f"Perlin resolutions[{index}] must contain two positive integers."
                )
        if any(
            isinstance(period, bool)
            or not isinstance(period, numbers.Integral)
            or int(period) < 1
            for period in periods
        ):
            raise ValueError("Perlin periods must contain only positive integers.")
        if not factors or any(
            isinstance(factor, bool)
            or not isinstance(factor, numbers.Real)
            or not math.isfinite(float(factor))
            for factor in factors
        ):
            raise ValueError("Perlin factors must contain finite real numbers.")

        octave_profile = str(octave_profile)
        if octave_profile == self.LEGACY_SINGLE_OCTAVE_PROFILE:
            if (
                tuple(tuple(int(value) for value in item) for item in resolutions)
                != ((2, 2), (4, 4), (8, 8), (16, 16), (32, 32))
                or tuple(int(value) for value in periods) != (32, 16, 8, 4, 2)
                or tuple(float(value) for value in factors) != (1.0,)
            ):
                raise ValueError(
                    "legacy_single_octave_v1 requires the authenticated far-tracking "
                    "5-candidate/1-active octave layout."
                )
        elif octave_profile == self.CUSTOM_EXPLICIT_OCTAVE_PROFILE:
            if not (len(resolutions) == len(periods) == len(factors)):
                raise ValueError(
                    "custom_explicit_v1 requires equal resolution, period, and factor lengths; "
                    "use a versioned production profile for intentional inactive candidates."
                )
        else:
            raise ValueError(f"Unsupported Perlin octave profile: {octave_profile!r}.")

        seed_semantics = str(seed_semantics)
        if seed_semantics == self.LEGACY_SEED_SEMANTICS:
            if effective_seed is not None:
                raise ValueError("legacy_fixed_v1 must not declare an effective Perlin seed.")
            gradient_seed_mixer = self.LEGACY_GRADIENT_SEED_MIXER
        elif seed_semantics == self.RANK_LOCAL_SEED_SEMANTICS:
            if (
                isinstance(effective_seed, bool)
                or not isinstance(effective_seed, numbers.Integral)
                or not 0 <= int(effective_seed) <= 2**64 - 1
            ):
                raise ValueError(
                    "rank_local_v2 requires an effective Perlin seed in [0, 2**64 - 1]."
                )
            effective_seed = int(effective_seed)
            gradient_seed_mixer = self.RANK_LOCAL_GRADIENT_SEED_MIXER
        else:
            raise ValueError(f"Unsupported Perlin seed semantics: {seed_semantics!r}.")

        self.shape = shape
        self.batch_size = batch_size
        self.resolutions = resolutions
        self.periods = periods
        self.factors = factors
        self.device = torch.device(device)
        self.seed_semantics = seed_semantics
        self.effective_seed = effective_seed
        self.gradient_seed_mixer = gradient_seed_mixer
        self.octave_profile = octave_profile
        self.grid_shapes = [(shape[0] // res[0], shape[1] // res[1]) for res in resolutions]
        self.linys = [torch.linspace(0, 1, gs[0], device=self.device) for gs in self.grid_shapes]
        self.linxs = [torch.linspace(0, 1, gs[1], device=self.device) for gs in self.grid_shapes]
        self.masks = []
        for lin_y, lin_x in zip(self.linys, self.linxs, strict=True):
            mask_y = self._fade(lin_y)
            mask_x = self._fade(lin_x)
            self.masks.append(
                {
                    (j, k): (mask_y if j == 1 else torch.flip(mask_y, [0]))[:, None]
                    * (mask_x if k == 1 else torch.flip(mask_x, [0]))[None, :]
                    for j in range(2)
                    for k in range(2)
                }
            )
        self.gradient_cache: list[dict[int, torch.Tensor]] = [{} for _ in resolutions]
        self.frame_idx = 0

    @staticmethod
    def _fade(t: torch.Tensor) -> torch.Tensor:
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    def _get_gradients(self, octave: int, z_idx: int) -> torch.Tensor:
        cache = self.gradient_cache[octave]
        if z_idx in cache:
            return cache[z_idx]
        for key in list(cache.keys()):
            if key < z_idx - 1:
                del cache[key]
        generator = torch.Generator(device=self.device)
        if self.seed_semantics == self.LEGACY_SEED_SEMANTICS:
            # Preserve HoloSoma's historical far-tracking-compatible stream
            # exactly for old serialized configs and policies.
            gradient_seed = hash((octave, z_idx)) % (2**31 - 1)
        else:
            # A versioned, process-stable mixer makes the nuisance field a
            # deterministic function of the configured rank-local seed while
            # keeping it isolated from every process-global RNG stream.
            payload = (
                "holosoma-perlin-gradient|rank_local_v2|"
                f"{self.effective_seed}|{int(octave)}|{int(z_idx)}"
            ).encode("ascii")
            gradient_seed = int.from_bytes(
                hashlib.sha256(payload).digest()[:8],
                byteorder="big",
                signed=False,
            ) & (2**63 - 1)
        generator.manual_seed(gradient_seed)
        res_h, res_w = self.resolutions[octave]
        gradients = torch.randn(
            (self.batch_size, res_h + 2, res_w + 2, 3),
            device=self.device,
            generator=generator,
        )
        gradients = gradients / torch.norm(gradients, dim=-1, keepdim=True).clamp(min=1.0e-8)
        cache[z_idx] = gradients
        return gradients

    def generate_frame(
        self,
        *,
        frame_index: int | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if frame_index is None:
            frame_index = self.frame_idx
            self.frame_idx += 1
        elif isinstance(frame_index, bool) or not isinstance(frame_index, numbers.Integral):
            raise ValueError(f"Perlin frame_index must be a non-negative integer, got {frame_index!r}.")
        else:
            frame_index = int(frame_index)
            if frame_index < 0:
                raise ValueError(f"Perlin frame_index must be a non-negative integer, got {frame_index!r}.")

        if env_ids is None:
            selected_env_ids: torch.Tensor | slice = slice(None)
            selected_batch_size = self.batch_size
        else:
            selected_env_ids = env_ids.to(device=self.device, dtype=torch.long).view(-1)
            selected_batch_size = int(selected_env_ids.numel())
        noise = torch.zeros(
            (selected_batch_size, self.shape[0], self.shape[1]),
            device=self.device,
        )
        for octave, factor in enumerate(self.factors):
            period = self.periods[octave]
            z_val = frame_index / period
            z_idx = int(math.floor(z_val))
            z_frac = z_val - z_idx
            grad0 = self._get_gradients(octave, z_idx)[selected_env_ids]
            grad1 = self._get_gradients(octave, z_idx + 1)[selected_env_ids]
            lin_y = self.linys[octave]
            lin_x = self.linxs[octave]
            weight_z1 = self._fade(torch.tensor(z_frac, device=self.device))
            weight_z0 = 1.0 - weight_z1
            octave_noise = 0
            for temporal_corner in range(2):
                current_grad = grad1 if temporal_corner == 1 else grad0
                current_weight_z = weight_z1 if temporal_corner == 1 else weight_z0
                dz = z_frac - temporal_corner
                grad_z = current_grad[..., 0]
                grad_y = current_grad[..., 1]
                grad_x = current_grad[..., 2]
                pos_x = grad_x[..., None] * lin_x
                pos_y = grad_y[..., None] * lin_y
                pos_x = pos_x[..., None, :]
                pos_y = pos_y[..., :, None]
                neg_x = -torch.flip(pos_x, dims=[-1])
                neg_y = -torch.flip(pos_y, dims=[-2])
                offset = (grad_z * dz)[..., None, None]
                for y_corner in range(2):
                    for x_corner in range(2):
                        term_x = pos_x if x_corner == 0 else neg_x
                        term_y = pos_y if y_corner == 0 else neg_y
                        term = offset + term_y + term_x
                        slice_y = slice(None, -1) if y_corner == 0 else slice(1, None)
                        slice_x = slice(None, -1) if x_corner == 0 else slice(1, None)
                        octave_noise += (
                            current_weight_z
                            * self.masks[octave][(y_corner, x_corner)]
                            * term[:, slice_y, slice_x, :, :]
                        )
            res_h, res_w = self.resolutions[octave]
            grid_h, grid_w = self.grid_shapes[octave]
            octave_noise = octave_noise.permute(0, 1, 3, 2, 4).reshape(
                selected_batch_size,
                (res_h + 1) * grid_h,
                (res_w + 1) * grid_w,
            )
            noise += factor * octave_noise[:, : self.shape[0], : self.shape[1]]
        return noise


class PerceptionManager:
    """Compute terrain-aware perception features (heightmap or camera depth)."""

    _PERCEPTION_MESH_CACHE_VERSION = "v2_trimesh_scene_atomic"

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
            raise ValueError(
                "Primitive/box object perception geometry is disabled. Use mesh URDF object geometry."
            )
        raise ValueError(
            "Unsupported perception object_geometry_mode. Supported value: 'mesh'. "
            f"Got: {raw_value}"
        )

    def __init__(self, cfg: PerceptionConfig | None, env: Any, device: str):
        if cfg is None:
            cfg = PerceptionConfig(enabled=False)
        self.cfg = cfg
        self.env = env
        self.device = device
        self.enabled = bool(cfg.enabled)
        self._reset_refresh_semantics = str(
            getattr(cfg, "reset_refresh_semantics", "legacy_full_v1")
        )
        if self._reset_refresh_semantics not in {"legacy_full_v1", "targeted_v2"}:
            raise ValueError(
                "perception.reset_refresh_semantics must be one of "
                "{'legacy_full_v1', 'targeted_v2'}, got "
                f"{self._reset_refresh_semantics!r}."
            )
        self._camera_warp_hole_seed_semantics = str(
            getattr(
                cfg,
                "camera_warp_hole_seed_semantics",
                _InfiniteFractalPerlin3D.LEGACY_SEED_SEMANTICS,
            )
        )
        if self._camera_warp_hole_seed_semantics not in {
            _InfiniteFractalPerlin3D.LEGACY_SEED_SEMANTICS,
            _InfiniteFractalPerlin3D.RANK_LOCAL_SEED_SEMANTICS,
        }:
            raise ValueError(
                "perception.camera_warp_hole_seed_semantics must be one of "
                "{'legacy_fixed_v1', 'rank_local_v2'}, got "
                f"{self._camera_warp_hole_seed_semantics!r}."
            )
        self._camera_warp_hole_octave_profile = str(
            getattr(
                cfg,
                "camera_warp_hole_octave_profile",
                _InfiniteFractalPerlin3D.LEGACY_SINGLE_OCTAVE_PROFILE,
            )
        )
        if (
            self._camera_warp_hole_octave_profile
            != _InfiniteFractalPerlin3D.LEGACY_SINGLE_OCTAVE_PROFILE
        ):
            raise ValueError(
                "perception.camera_warp_hole_octave_profile currently supports only "
                "'legacy_single_octave_v1'; a new distribution requires a separately "
                "versioned and calibrated profile."
            )
        simulator_type = get_simulator_type()
        self._simulator_backend = str(simulator_type)
        self._is_mujoco_perception = simulator_type == SimulatorType.MUJOCO
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
        self._far_tracking_robot_body_names: list[str] = []
        self._far_tracking_robot_body_offset_pos: torch.Tensor | None = None
        self._far_tracking_robot_body_offset_quat: torch.Tensor | None = None
        self._far_tracking_object_slot_indices: torch.Tensor | None = None
        self._far_tracking_object_source_indices: torch.Tensor | None = None
        # Static object topology is mirrored as host integers so normal camera
        # capture never calls ``.item()`` on CUDA scalar tensors.
        self._far_tracking_object_slot_pairs: tuple[tuple[int, int], ...] = ()
        self._far_tracking_primitive_source_indices: torch.Tensor | None = None
        self._far_tracking_object_names: list[str] = []
        self._far_tracking_object_active_env_ids: list[torch.Tensor | None] = []
        self._far_tracking_base_link_indices: torch.Tensor | None = None
        self._far_tracking_geometry_fingerprint: tuple[tuple[str, str, int, str], ...] | None = None
        self._authenticated_observation_contract: dict[str, Any] | None = None
        self._shared_camera_sensor_local_position: torch.Tensor | None = None
        self._shared_camera_sensor_local_orientation: torch.Tensor | None = None
        self._shared_camera_sensor_data_frame_quat: torch.Tensor | None = None
        self._registered_object_mesh_cache: dict[str, str] = {}
        self._camera_mount_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._use_camera_mount_quat = False
        self._camera_frame_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device)
        self._use_camera_frame_quat = False
        self._strict_camera_mount_rotation_deg = torch.tensor(
            [1.0, 27.0, 1.0],
            device=self.device,
            dtype=torch.float32,
        )
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
        self._camera_warp_buffer_len = max(1, int(getattr(self.cfg, "camera_warp_buffer_len", 1) or 1))
        latency_cfg = getattr(self.cfg, "camera_warp_latency_frame", 0) or 0
        self._camera_warp_latency_frame_range: tuple[int, int] | None = None
        if isinstance(latency_cfg, (tuple, list)):
            if len(latency_cfg) != 2:
                raise ValueError("camera_warp_latency_frame range must be a (min, max) pair.")
            latency_min = max(0, int(latency_cfg[0]))
            latency_max = max(0, int(latency_cfg[1]))
            if latency_max < latency_min:
                raise ValueError("camera_warp_latency_frame range max must be >= min.")
            if latency_max >= self._camera_warp_buffer_len:
                raise ValueError("camera_warp_latency_frame range max must be smaller than camera_warp_buffer_len.")
            self._camera_warp_latency_frame = latency_min
            self._camera_warp_latency_frame_range = (latency_min, latency_max)
        else:
            self._camera_warp_latency_frame = max(0, int(latency_cfg))
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
        structured_mujoco_noise = getattr(env, "_allow_mujoco_perception_noise", None)
        if structured_mujoco_noise is None:
            allow_mujoco_perception_noise_raw = os.environ.get(
                "HOLOSOMA_MUJOCO_ALLOW_PERCEPTION_NOISE",
                "0",
            ).strip().lower()
            allow_mujoco_perception_noise = allow_mujoco_perception_noise_raw not in {
                "0",
                "false",
                "no",
                "off",
                "",
            }
        elif not isinstance(structured_mujoco_noise, (bool, np.bool_)):
            raise ValueError(
                "_allow_mujoco_perception_noise must be boolean when supplied by a direct producer."
            )
        else:
            # A structured direct-simulation setting takes precedence over the
            # ambient process environment, making copied launch commands and
            # authenticated producer reconstruction deterministic.
            allow_mujoco_perception_noise = bool(structured_mujoco_noise)
        force_mujoco_noise_off = self._is_mujoco_perception and not allow_mujoco_perception_noise

        requested_camera_warp_edge_noise = bool(getattr(self.cfg, "camera_warp_edge_noise", False))
        self._camera_warp_edge_noise = (
            False if force_mujoco_noise_off else requested_camera_warp_edge_noise
        )
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
        requested_camera_warp_enable_holes = bool(getattr(self.cfg, "camera_warp_enable_holes", False))
        self._camera_warp_enable_holes = (
            False if force_mujoco_noise_off else requested_camera_warp_enable_holes
        )
        self._camera_warp_hole_prob = (
            0.0 if force_mujoco_noise_off else float(getattr(self.cfg, "camera_warp_hole_prob", 0.0) or 0.0)
        )
        requested_camera_warp_additive_noise_std = float(
            getattr(self.cfg, "camera_warp_additive_noise_std", 0.0) or 0.0
        )
        requested_camera_warp_depth_offset_std = float(getattr(self.cfg, "camera_warp_depth_offset_std", 0.0) or 0.0)
        self._camera_warp_additive_noise_std = (
            0.0
            if force_mujoco_noise_off
            else requested_camera_warp_additive_noise_std
        )
        self._camera_warp_depth_offset_std = (
            0.0
            if force_mujoco_noise_off
            else requested_camera_warp_depth_offset_std
        )
        requested_camera_apply_sensor_noise = bool(getattr(self.cfg, "camera_apply_sensor_noise", True))
        self._camera_apply_sensor_noise = (
            False if force_mujoco_noise_off else requested_camera_apply_sensor_noise
        )
        if force_mujoco_noise_off and (
            requested_camera_warp_edge_noise
            or requested_camera_warp_enable_holes
            or requested_camera_apply_sensor_noise
            or requested_camera_warp_additive_noise_std > 0.0
            or requested_camera_warp_depth_offset_std > 0.0
        ):
            (self.logger or logger).warning(
                "MuJoCo perception forcing camera noise off: "
                "edge_noise={} holes={} sensor_noise={} additive_noise={} depth_offset={} -> False/False/False/0/0",
                requested_camera_warp_edge_noise,
                requested_camera_warp_enable_holes,
                requested_camera_apply_sensor_noise,
                requested_camera_warp_additive_noise_std,
                requested_camera_warp_depth_offset_std,
            )

        self._camera_obs_height, self._camera_obs_width = self._resolve_camera_obs_resolution()
        self._camera_obs_fill_value = self._camera_obs_default_fill_value()
        self._camera_obs_step_counter = 0
        self._camera_warp_hole_generator: _InfiniteFractalPerlin3D | None = None
        self._camera_warp_hole_frame_stats: tuple[int, torch.Tensor, torch.Tensor] | None = None
        hole_reference_batch_size = getattr(
            self.cfg,
            "camera_warp_hole_reference_batch_size",
            None,
        )
        if hole_reference_batch_size is None:
            hole_reference_batch_size = self.num_envs
        if (
            isinstance(hole_reference_batch_size, (bool, np.bool_))
            or not isinstance(hole_reference_batch_size, numbers.Integral)
            or int(hole_reference_batch_size) < self.num_envs
        ):
            raise ValueError(
                "camera_warp_hole_reference_batch_size must be an integer no smaller than "
                f"the live environment count ({self.num_envs}), got {hole_reference_batch_size!r}."
            )
        self._camera_warp_hole_reference_batch_size = int(hole_reference_batch_size)
        if self._camera_warp_enable_holes and self._camera_warp_hole_prob > 0.0:
            effective_hole_seed = (
                _validated_rank_local_perlin_seed(self.env)
                if self._camera_warp_hole_seed_semantics
                == _InfiniteFractalPerlin3D.RANK_LOCAL_SEED_SEMANTICS
                else None
            )
            self._camera_warp_hole_generator = _InfiniteFractalPerlin3D(
                (64, 96),
                [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)],
                [32, 16, 8, 4, 2],
                [0.3**i for i in range(1)],
                batch_size=self._camera_warp_hole_reference_batch_size,
                device=self.device,
                seed_semantics=self._camera_warp_hole_seed_semantics,
                effective_seed=effective_hole_seed,
                octave_profile=self._camera_warp_hole_octave_profile,
            )
        if self._camera_warp_depth_offset_std > 0.0:
            self._camera_warp_depth_offset = torch.randn(
                self.num_envs,
                device=self.device,
                dtype=torch.float32,
            ) * self._camera_warp_depth_offset_std
        else:
            self._camera_warp_depth_offset = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
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
        self._camera_randomization_log_done = False

    def setup(self) -> None:
        if not self.enabled:
            return
        # Rendered strict-warp backends otherwise create this sampled mount
        # lazily on the first successful camera update.  If update_hz makes
        # the constructor's warm-up step return early, a fresh resume has no
        # mount while a trained checkpoint does, so exact-load semantics
        # spuriously differ.  Establish it eagerly for every strict camera.
        if self.cfg.output_mode == "camera_depth" and self._camera_strict_warp:
            self._ensure_shared_strict_warp_camera_mount()
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
            self._resolve_camera_body_index()
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
            self._resolve_camera_body_index()
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
            self._camera_warp_hole_frame_stats = None
            self._ray_hits_world.zero_()
            return
        self._heightmap[env_ids] = 0.0
        self._camera_depth[env_ids] = self.cfg.max_distance
        self._camera_depth_obs[env_ids] = self._camera_obs_fill_value
        self._camera_depth_buffer[env_ids] = self._camera_obs_fill_value
        self._camera_depth_buffer_ready[env_ids] = False
        self._ray_hits_world[env_ids] = 0.0

    def reset_canonical_rollout_state(self) -> None:
        """Reset temporal perception state at a checkpoint rollout boundary.

        Physical episodes and camera latency buffers are intentionally
        discarded by the canonical all-environment reset.  Perception clocks
        are checkpointed and continue across the boundary; only the Perlin
        generator's derived cache is discarded so it is rebuilt from the
        authenticated frame index without introducing checkpoint-periodic
        hole patterns.
        """

        if not self.enabled:
            return
        if self._camera_warp_hole_generator is not None:
            self._camera_warp_hole_generator.gradient_cache = [
                {} for _ in self._camera_warp_hole_generator.resolutions
            ]
            self._camera_warp_hole_frame_stats = None

    def persistent_checkpoint_state_required(self) -> bool:
        """Whether a legacy env checkpoint cannot reproduce this manager."""

        # Heightmap managers also retain ``_time_since_update`` across the
        # canonical reset.  Omitting their state can shift the sensor cadence
        # after resume even though they have no camera calibration tensors.
        return bool(self.enabled)

    def validate_exact_resume_supported(self) -> None:
        """Reject camera backends with temporal state outside this manager."""

        if self._uses_rendered_camera() and not self._is_mujoco_perception:
            raise RuntimeError(
                "Exact training resume is unsupported for Isaac rendered camera backends: "
                "the external annotator/render-frame queue and sampling phase are not captured "
                "by the environment checkpoint. Use far_tracking_warp for resumable training "
                "or initialize only policy weights."
            )

    @staticmethod
    def _semantic_tensor_tuple(value: Any) -> tuple[float, ...] | None:
        if not isinstance(value, torch.Tensor):
            return None
        return tuple(float(item) for item in value.detach().to("cpu").reshape(-1).tolist())

    @staticmethod
    def _compose_fixed_body_pose(
        parent_position: torch.Tensor,
        parent_orientation: torch.Tensor,
        local_position: torch.Tensor,
        local_orientation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            parent_position
            + quat_apply(parent_orientation, local_position, w_last=True),
            quat_mul(parent_orientation, local_orientation, w_last=True),
        )

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _fingerprint_far_tracking_geometry(
        cls,
        ray_cast_bodies: dict[str, str],
        *,
        asset_meshes_root: str | os.PathLike[str],
    ) -> tuple[tuple[str, str, int, str], ...]:
        """Content-address the exact meshes loaded into far-tracking slots."""

        records: list[tuple[str, str, int, str]] = []
        root = Path(asset_meshes_root).expanduser()
        digest_cache: dict[Path, tuple[int, str]] = {}
        for slot_name, mesh_reference in ray_cast_bodies.items():
            path = Path(str(mesh_reference)).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"far_tracking_warp geometry for slot {slot_name!r} is not a file: {path}"
                )
            cached = digest_cache.get(path)
            if cached is None:
                stat_before = path.stat()
                digest = cls._file_sha256(path)
                stat_after = path.stat()
                before_identity = (
                    int(stat_before.st_dev),
                    int(stat_before.st_ino),
                    int(stat_before.st_size),
                    int(stat_before.st_mtime_ns),
                )
                after_identity = (
                    int(stat_after.st_dev),
                    int(stat_after.st_ino),
                    int(stat_after.st_size),
                    int(stat_after.st_mtime_ns),
                )
                if before_identity != after_identity:
                    raise RuntimeError(
                        "far_tracking_warp geometry changed while its checkpoint identity was computed: "
                        f"{path}"
                    )
                cached = (int(stat_after.st_size), digest)
                digest_cache[path] = cached
            size, digest = cached
            # The parser is selected by the file suffix, so retain it while
            # deliberately omitting the path/basename.  Identical copied or
            # generated-cache assets then compare equal, while same-name
            # files with different bytes cannot silently resume.
            records.append(
                (
                    str(slot_name),
                    path.suffix.lower(),
                    size,
                    digest,
                )
            )
        return tuple(records)

    def _persistent_checkpoint_semantics(self) -> dict[str, Any]:
        shared_mount_values = (
            self._shared_camera_sensor_local_position,
            self._shared_camera_sensor_local_orientation,
            self._shared_camera_sensor_data_frame_quat,
        )
        shared_mount_presence = [value is not None for value in shared_mount_values]
        if any(shared_mount_presence) and not all(shared_mount_presence):
            raise RuntimeError("Perception strict camera mount state is only partially initialized.")
        shared_mount_present = all(shared_mount_presence)
        hole_generator_schema = None
        if self._camera_warp_hole_generator is not None:
            hole_generator_schema = {
                "shape": tuple(int(value) for value in self._camera_warp_hole_generator.shape),
                "resolutions": tuple(
                    tuple(int(value) for value in resolution)
                    for resolution in self._camera_warp_hole_generator.resolutions
                ),
                "periods": tuple(int(value) for value in self._camera_warp_hole_generator.periods),
                "factors": tuple(float(value) for value in self._camera_warp_hole_generator.factors),
                # The historical far-tracking producer normalizes a raw
                # Perlin field over its full vectorized training batch.  The
                # extrema materially change an individual environment's hole
                # mask, so deployment must reproduce the saved reference
                # batch rather than silently using its live (usually one-env)
                # batch size.
                "normalization_scope": "reference_batch",
                "reference_batch_size": int(
                    getattr(
                        self,
                        "_camera_warp_hole_reference_batch_size",
                        getattr(self._camera_warp_hole_generator, "batch_size", self.num_envs),
                    )
                ),
            }
            hole_seed_semantics = str(
                getattr(
                    self._camera_warp_hole_generator,
                    "seed_semantics",
                    _InfiniteFractalPerlin3D.LEGACY_SEED_SEMANTICS,
                )
            )
            if hole_seed_semantics == _InfiniteFractalPerlin3D.RANK_LOCAL_SEED_SEMANTICS:
                # These fields intentionally exist only for v2.  Their absence
                # in an older contract/state continues to mean the byte-exact
                # legacy fixed stream rather than being guessed as v2.
                hole_generator_schema.update(
                    {
                        "seed_semantics": hole_seed_semantics,
                        "effective_seed": int(
                            self._camera_warp_hole_generator.effective_seed
                        ),
                        "gradient_seed_mixer": str(
                            self._camera_warp_hole_generator.gradient_seed_mixer
                        ),
                        "octave_profile": str(
                            self._camera_warp_hole_generator.octave_profile
                        ),
                    }
                )
            elif hole_seed_semantics != _InfiniteFractalPerlin3D.LEGACY_SEED_SEMANTICS:
                raise RuntimeError(
                    f"Unsupported live Perlin seed semantics: {hole_seed_semantics!r}."
                )
        sensor_offset_tuple = self._semantic_tensor_tuple(getattr(self, "_sensor_offset", None))
        ray_start_offset_tuple = self._semantic_tensor_tuple(
            getattr(self, "_ray_start_offset", None)
        )
        far_tracking_geometry = getattr(self, "_far_tracking_geometry_fingerprint", None)
        rendered_camera = getattr(self, "_rendered_camera", None)
        rendered_backend = None
        if rendered_camera is not None:
            rendered_backend = {
                "implementation": (
                    f"{type(rendered_camera).__module__}.{type(rendered_camera).__qualname__}"
                ),
                "annotator_name": getattr(rendered_camera, "_annotator_name", None),
                "flip_render_array_vertical": getattr(
                    rendered_camera,
                    "_flip_render_array_vertical",
                    None,
                ),
                "depth_prefers_visual_meshes": getattr(
                    rendered_camera,
                    "_depth_prefers_visual_meshes",
                    None,
                ),
                "depth_prefers_robot_visual_meshes": getattr(
                    rendered_camera,
                    "_depth_prefers_robot_visual_meshes",
                    None,
                ),
                "depth_prefers_object_visual_meshes": getattr(
                    rendered_camera,
                    "_depth_prefers_object_visual_meshes",
                    None,
                ),
            }
        return {
            "num_envs": int(self.num_envs),
            "output_mode": str(getattr(self.cfg, "output_mode", "")),
            "camera_source": str(self._camera_source),
            "simulator_backend": str(self._simulator_backend),
            "camera_shape": (int(self._camera_height), int(self._camera_width)),
            "camera_obs_shape": (int(self._camera_obs_height), int(self._camera_obs_width)),
            "camera_geometry": {
                "intrinsics_fx_fy_cx_cy": tuple(
                    float(getattr(self, attr_name).detach().to("cpu").item())
                    for attr_name in ("_camera_fx", "_camera_fy", "_camera_cx", "_camera_cy")
                )
                if all(
                    isinstance(getattr(self, attr_name, None), torch.Tensor)
                    for attr_name in ("_camera_fx", "_camera_fy", "_camera_cx", "_camera_cy")
                )
                else None,
                "vfov_deg": (
                    None
                    if getattr(self, "_camera_vfov_deg", None) is None
                    else float(self._camera_vfov_deg)
                ),
                "hfov_deg": (
                    None
                    if getattr(self, "_camera_hfov_deg", None) is None
                    else float(self._camera_hfov_deg)
                ),
                "fps": float(getattr(self.cfg, "camera_fps", 0.0) or 0.0),
                "near": float(getattr(self.cfg, "camera_near", 0.0) or 0.0),
                "far": float(getattr(self.cfg, "camera_far", 0.0) or 0.0),
                "max_distance": float(getattr(self.cfg, "max_distance", 0.0) or 0.0),
                "pitch_deg": float(getattr(self.cfg, "camera_pitch_deg", 0.0) or 0.0),
                "target_pitch_deg": (
                    None
                    if getattr(self.cfg, "camera_target_pitch_deg", None) is None
                    else float(self.cfg.camera_target_pitch_deg)
                ),
                "distortion": tuple(
                    float(value) for value in (getattr(self.cfg, "camera_distortion", None) or ())
                ),
                "body_name": getattr(self, "_camera_body_name", None),
                "body_index": (
                    None
                    if getattr(self, "_camera_body_index", None) is None
                    else int(self._camera_body_index)
                ),
                "body_offset_position": self._semantic_tensor_tuple(
                    getattr(self, "_camera_body_offset_pos", None)
                ),
                "body_offset_quaternion": self._semantic_tensor_tuple(
                    getattr(self, "_camera_body_offset_quat", None)
                ),
                "rendered_env_id": int(getattr(self, "_rendered_camera_env_id", 0)),
                "mount_quaternion": self._semantic_tensor_tuple(
                    getattr(self, "_camera_mount_quat", None)
                ),
                "use_mount_quaternion": bool(
                    getattr(self, "_use_camera_mount_quat", False)
                ),
                "frame_quaternion": self._semantic_tensor_tuple(
                    getattr(self, "_camera_frame_quat", None)
                ),
                "use_frame_quaternion": bool(
                    getattr(self, "_use_camera_frame_quat", False)
                ),
                "auto_fix_backward": bool(
                    getattr(self, "_camera_auto_fix_backward", False)
                ),
                "backward_ratio_threshold": float(
                    getattr(self, "_camera_backward_ratio_threshold", 0.0)
                ),
                "far_tracking_base_link_indices": (
                    None
                    if not isinstance(
                        getattr(self, "_far_tracking_base_link_indices", None),
                        torch.Tensor,
                    )
                    else tuple(
                        int(value)
                        for value in self._far_tracking_base_link_indices.detach()
                        .to("cpu")
                        .reshape(-1)
                        .tolist()
                    )
                ),
            },
            "heightmap_geometry": {
                "grid_shape": (
                    int(getattr(self, "_heightmap_grid_x", 0)),
                    int(getattr(self, "_heightmap_grid_y", 0)),
                ),
                "grid_interval": (
                    float(getattr(self, "_heightmap_interval_x", 0.0)),
                    float(getattr(self, "_heightmap_interval_y", 0.0)),
                ),
                "body_name": getattr(self, "_heightmap_body_name", None),
                "body_index": (
                    None
                    if getattr(self, "_heightmap_body_index", None) is None
                    else int(self._heightmap_body_index)
                ),
                "body_offset_position": self._semantic_tensor_tuple(
                    getattr(self, "_heightmap_body_offset_pos", None)
                ),
                "body_offset_quaternion": self._semantic_tensor_tuple(
                    getattr(self, "_heightmap_body_offset_quat", None)
                ),
                "ray_start_offset": ray_start_offset_tuple,
                "use_heading_only": bool(getattr(self.cfg, "use_heading_only", False)),
                "observation_offset": float(
                    getattr(self.cfg, "heightmap_obs_offset", 0.0) or 0.0
                ),
            },
            "camera_strict_warp": bool(self._camera_strict_warp),
            "camera_disable_offsets": bool(self._camera_disable_offsets),
            "update_interval": float(self._update_interval),
            "camera_warp_preprocess": bool(self._camera_warp_preprocess),
            "camera_warp_freq_ratio": int(self._camera_warp_freq_ratio),
            "camera_warp_buffer_len": int(self._camera_warp_buffer_len),
            "camera_warp_latency_frame": int(self._camera_warp_latency_frame),
            "camera_warp_latency_frame_range": (
                None
                if self._camera_warp_latency_frame_range is None
                else tuple(int(value) for value in self._camera_warp_latency_frame_range)
            ),
            "camera_reset_randomization": self._camera_reset_randomization_semantics(),
            "camera_setup_randomization": self._camera_setup_randomization_semantics(),
            "reset_refresh_semantics": getattr(
                self,
                "_reset_refresh_semantics",
                "legacy_full_v1",
            ),
            "effective_observation_schema": {
                "sensor_offset": sensor_offset_tuple,
                "camera_include_robot_mesh": bool(
                    getattr(self, "_camera_include_robot_mesh", False)
                ),
                "object_geometry_mode": str(
                    getattr(self, "_object_geometry_mode", "")
                ),
                "crop": (
                    int(getattr(self, "_camera_warp_crop_top", 0)),
                    int(getattr(self, "_camera_warp_crop_bottom", 0)),
                    int(getattr(self, "_camera_warp_crop_left", 0)),
                    int(getattr(self, "_camera_warp_crop_right", 0)),
                ),
                "resize": (
                    None
                    if getattr(self, "_camera_warp_resize", None) is None
                    else tuple(int(value) for value in self._camera_warp_resize)
                ),
                "normalize": bool(getattr(self, "_camera_warp_normalize", False)),
                "min_valid_depth": float(
                    getattr(self, "_camera_warp_min_valid_depth", 0.0)
                ),
                "edge_noise": bool(getattr(self, "_camera_warp_edge_noise", False)),
                "edge_border": int(getattr(self, "_camera_warp_edge_border", 0)),
                "edge_shuffle_prob": float(
                    getattr(self, "_camera_warp_edge_shuffle_prob", 0.0)
                ),
                "edge_empty_prob": float(
                    getattr(self, "_camera_warp_edge_empty_prob", 0.0)
                ),
                "edge_thresh_primary": float(
                    getattr(self, "_camera_warp_edge_thresh_primary", 0.0)
                ),
                "edge_thresh_secondary": float(
                    getattr(self, "_camera_warp_edge_thresh_secondary", 0.0)
                ),
                "edge_far_depth_thresh": float(
                    getattr(self, "_camera_warp_edge_far_depth_thresh", 0.0)
                ),
                "enable_holes": bool(
                    getattr(self, "_camera_warp_enable_holes", False)
                ),
                "hole_prob": float(getattr(self, "_camera_warp_hole_prob", 0.0)),
                "additive_noise_std": float(
                    getattr(self, "_camera_warp_additive_noise_std", 0.0)
                ),
                "depth_offset_std": float(
                    getattr(self, "_camera_warp_depth_offset_std", 0.0)
                ),
                "apply_sensor_noise": bool(
                    getattr(self, "_camera_apply_sensor_noise", False)
                ),
                "obs_fill_value": float(getattr(self, "_camera_obs_fill_value", 0.0)),
            },
            "far_tracking_geometry": far_tracking_geometry,
            "far_tracking_topology": {
                "robot_slot_indices": self._semantic_tensor_tuple(
                    getattr(self, "_far_tracking_robot_slot_indices", None)
                ),
                "robot_body_indices": self._semantic_tensor_tuple(
                    getattr(self, "_far_tracking_robot_body_indices", None)
                ),
                "robot_body_names": tuple(
                    str(value)
                    for value in getattr(self, "_far_tracking_robot_body_names", ())
                ),
                "robot_body_offset_positions": self._semantic_tensor_tuple(
                    getattr(self, "_far_tracking_robot_body_offset_pos", None)
                ),
                "robot_body_offset_quaternions": self._semantic_tensor_tuple(
                    getattr(self, "_far_tracking_robot_body_offset_quat", None)
                ),
                "object_slot_indices": self._semantic_tensor_tuple(
                    getattr(self, "_far_tracking_object_slot_indices", None)
                ),
                "object_source_indices": self._semantic_tensor_tuple(
                    getattr(self, "_far_tracking_object_source_indices", None)
                ),
                "primitive_source_indices": self._semantic_tensor_tuple(
                    getattr(self, "_far_tracking_primitive_source_indices", None)
                ),
                "object_names": tuple(
                    str(value)
                    for value in getattr(self, "_far_tracking_object_names", ())
                ),
                "object_active_env_ids": tuple(
                    None
                    if value is None
                    else tuple(
                        int(item)
                        for item in value.detach().to("cpu").reshape(-1).tolist()
                    )
                    for value in getattr(
                        self,
                        "_far_tracking_object_active_env_ids",
                        (),
                    )
                ),
            },
            "rendered_backend": rendered_backend,
            "shared_mount_present": shared_mount_present,
            "hole_generator_schema": hole_generator_schema,
        }

    @staticmethod
    def _canonical_geometry_sort_key(value: Any) -> str:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )

    @classmethod
    def _normalize_training_geometry_support(
        cls,
        value: Any,
        *,
        path: str = "training_geometry_support",
    ) -> dict[str, Any]:
        """Validate the portable all-rank geometry support authenticated by a policy."""

        if not isinstance(value, Mapping) or set(value) != {
            "version",
            "camera_source",
            "training_rank_count",
            "robot_mesh_bindings",
            "object_mesh_support",
        }:
            raise ValueError(f"{path} has missing or unsupported fields.")
        if type(value.get("version")) is not int or value["version"] != 1:
            raise ValueError(f"{path}.version must equal integer 1.")
        camera_source = value.get("camera_source")
        if not isinstance(camera_source, str) or not camera_source:
            raise ValueError(f"{path}.camera_source must be a non-empty string.")
        training_rank_count = value.get("training_rank_count")
        if type(training_rank_count) is not int or training_rank_count <= 0:
            raise ValueError(f"{path}.training_rank_count must be a positive integer.")

        def mesh_identity(raw: Any, *, mesh_path: str) -> dict[str, Any]:
            if not isinstance(raw, Mapping) or set(raw) != {
                "suffix",
                "size_bytes",
                "sha256",
            }:
                raise ValueError(f"{mesh_path} has missing or unsupported fields.")
            suffix = raw.get("suffix")
            if (
                not isinstance(suffix, str)
                or not suffix.startswith(".")
                or len(suffix) <= 1
                or suffix != suffix.lower()
            ):
                raise ValueError(f"{mesh_path}.suffix must be a lowercase file suffix.")
            size_bytes = raw.get("size_bytes")
            if type(size_bytes) is not int or size_bytes <= 0:
                raise ValueError(f"{mesh_path}.size_bytes must be a positive integer.")
            digest = raw.get("sha256")
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or digest != digest.lower()
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"{mesh_path}.sha256 must be 64 lowercase hexadecimal characters.")
            return {
                "suffix": suffix,
                "size_bytes": size_bytes,
                "sha256": digest,
            }

        def finite_vector(raw: Any, *, vector_path: str, length: int) -> list[float]:
            if not isinstance(raw, (list, tuple)) or len(raw) != length:
                raise ValueError(f"{vector_path} must contain {length} finite numbers.")
            result: list[float] = []
            for index, item in enumerate(raw):
                if isinstance(item, bool) or not isinstance(item, numbers.Real):
                    raise ValueError(f"{vector_path}[{index}] must be a finite number.")
                item_float = float(item)
                if not math.isfinite(item_float):
                    raise ValueError(f"{vector_path}[{index}] must be finite.")
                result.append(item_float)
            return result

        raw_robot = value.get("robot_mesh_bindings")
        if not isinstance(raw_robot, (list, tuple)):
            raise ValueError(f"{path}.robot_mesh_bindings must be a list.")
        robot: list[dict[str, Any]] = []
        robot_slot_names: set[str] = set()
        for index, raw in enumerate(raw_robot):
            item_path = f"{path}.robot_mesh_bindings[{index}]"
            if not isinstance(raw, Mapping) or set(raw) != {
                "slot_name",
                "mesh",
                "tracking_body_name",
                "fixed_position_xyz",
                "fixed_quaternion_xyzw",
            }:
                raise ValueError(f"{item_path} has missing or unsupported fields.")
            slot_name = raw.get("slot_name")
            tracking_body_name = raw.get("tracking_body_name")
            if not isinstance(slot_name, str) or not slot_name:
                raise ValueError(f"{item_path}.slot_name must be a non-empty string.")
            if slot_name in robot_slot_names:
                raise ValueError(f"{path}.robot_mesh_bindings contains duplicate slot {slot_name!r}.")
            robot_slot_names.add(slot_name)
            if not isinstance(tracking_body_name, str) or not tracking_body_name:
                raise ValueError(f"{item_path}.tracking_body_name must be a non-empty string.")
            position = finite_vector(
                raw.get("fixed_position_xyz"),
                vector_path=f"{item_path}.fixed_position_xyz",
                length=3,
            )
            quaternion = finite_vector(
                raw.get("fixed_quaternion_xyzw"),
                vector_path=f"{item_path}.fixed_quaternion_xyzw",
                length=4,
            )
            norm = math.sqrt(sum(component * component for component in quaternion))
            if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
                raise ValueError(f"{item_path}.fixed_quaternion_xyzw must be a unit quaternion.")
            robot.append(
                {
                    "slot_name": slot_name,
                    "mesh": mesh_identity(raw.get("mesh"), mesh_path=f"{item_path}.mesh"),
                    "tracking_body_name": tracking_body_name,
                    "fixed_position_xyz": position,
                    "fixed_quaternion_xyzw": quaternion,
                }
            )

        raw_objects = value.get("object_mesh_support")
        if not isinstance(raw_objects, (list, tuple)):
            raise ValueError(f"{path}.object_mesh_support must be a list.")
        objects: list[dict[str, Any]] = []
        object_keys: set[str] = set()
        for index, raw in enumerate(raw_objects):
            item_path = f"{path}.object_mesh_support[{index}]"
            if not isinstance(raw, Mapping) or set(raw) != {
                "source_name",
                "mesh",
                "training_active_env_count",
            }:
                raise ValueError(f"{item_path} has missing or unsupported fields.")
            source_name = raw.get("source_name")
            if not isinstance(source_name, str) or not source_name:
                raise ValueError(f"{item_path}.source_name must be a non-empty string.")
            active_count = raw.get("training_active_env_count")
            if type(active_count) is not int or active_count <= 0:
                raise ValueError(f"{item_path}.training_active_env_count must be a positive integer.")
            normalized_item = {
                "source_name": source_name,
                "mesh": mesh_identity(raw.get("mesh"), mesh_path=f"{item_path}.mesh"),
                "training_active_env_count": active_count,
            }
            identity_key = cls._canonical_geometry_sort_key(
                {key: normalized_item[key] for key in ("source_name", "mesh")}
            )
            if identity_key in object_keys:
                raise ValueError(f"{path}.object_mesh_support contains duplicate geometry support.")
            object_keys.add(identity_key)
            objects.append(normalized_item)

        expected_robot_order = sorted(robot, key=cls._canonical_geometry_sort_key)
        expected_object_order = sorted(objects, key=cls._canonical_geometry_sort_key)
        if robot != expected_robot_order:
            raise ValueError(f"{path}.robot_mesh_bindings must be in canonical sorted order.")
        if objects != expected_object_order:
            raise ValueError(f"{path}.object_mesh_support must be in canonical sorted order.")
        if camera_source != "far_tracking_warp" and (robot or objects):
            raise ValueError(
                f"{path} may contain mesh bindings only for camera_source='far_tracking_warp'."
            )
        return {
            "version": 1,
            "camera_source": camera_source,
            "training_rank_count": training_rank_count,
            "robot_mesh_bindings": robot,
            "object_mesh_support": objects,
        }

    @classmethod
    def geometry_support_from_checkpoint_semantics(
        cls,
        semantics: Any,
        *,
        path: str = "perception.semantics",
    ) -> dict[str, Any]:
        """Project one rank's backend-local far-tracking state into portable semantics."""

        if not isinstance(semantics, Mapping):
            raise ValueError(f"{path} must be a mapping.")
        camera_source = semantics.get("camera_source")
        if not isinstance(camera_source, str) or not camera_source:
            raise ValueError(f"{path}.camera_source must be a non-empty string.")
        num_envs = semantics.get("num_envs")
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError(f"{path}.num_envs must be a positive integer.")
        geometry = semantics.get("far_tracking_geometry")
        topology = semantics.get("far_tracking_topology")
        if camera_source != "far_tracking_warp":
            if geometry not in (None, (), []):
                raise ValueError(f"{path}.far_tracking_geometry is unexpected for {camera_source!r}.")
            return cls._normalize_training_geometry_support(
                {
                    "version": 1,
                    "camera_source": camera_source,
                    "training_rank_count": 1,
                    "robot_mesh_bindings": [],
                    "object_mesh_support": [],
                },
                path=f"{path}.deployment_geometry",
            )
        if not isinstance(geometry, (list, tuple)):
            raise ValueError(f"{path}.far_tracking_geometry must be a sequence.")
        if not isinstance(topology, Mapping) or set(topology) != {
            "robot_slot_indices",
            "robot_body_indices",
            "robot_body_names",
            "robot_body_offset_positions",
            "robot_body_offset_quaternions",
            "object_slot_indices",
            "object_source_indices",
            "primitive_source_indices",
            "object_names",
            "object_active_env_ids",
        }:
            raise ValueError(f"{path}.far_tracking_topology has missing or unsupported fields.")

        mesh_by_slot: list[dict[str, Any]] = []
        seen_slot_names: set[str] = set()
        for index, raw in enumerate(geometry):
            item_path = f"{path}.far_tracking_geometry[{index}]"
            if not isinstance(raw, (list, tuple)) or len(raw) != 4:
                raise ValueError(f"{item_path} must contain slot, suffix, size, and SHA-256.")
            slot_name, suffix, size_bytes, digest = raw
            candidate = cls._normalize_training_geometry_support(
                {
                    "version": 1,
                    "camera_source": "far_tracking_warp",
                    "training_rank_count": 1,
                    "robot_mesh_bindings": [],
                    "object_mesh_support": [
                        {
                            "source_name": "placeholder",
                            "mesh": {
                                "suffix": suffix,
                                "size_bytes": size_bytes,
                                "sha256": digest,
                            },
                            "training_active_env_count": 1,
                        }
                    ],
                },
                path=item_path,
            )["object_mesh_support"][0]["mesh"]
            if not isinstance(slot_name, str) or not slot_name or slot_name in seen_slot_names:
                raise ValueError(f"{item_path} has an empty or duplicate semantic slot name.")
            seen_slot_names.add(slot_name)
            mesh_by_slot.append({"slot_name": slot_name, "mesh": candidate})

        def sequence(name: str) -> list[Any]:
            raw = topology.get(name)
            if not isinstance(raw, (list, tuple)):
                raise ValueError(f"{path}.far_tracking_topology.{name} must be a sequence.")
            return list(raw)

        def integer_indices(name: str, *, upper: int | None = None) -> list[int]:
            raw_values = sequence(name)
            values: list[int] = []
            for index, item in enumerate(raw_values):
                if (
                    isinstance(item, bool)
                    or not isinstance(item, numbers.Real)
                    or not math.isfinite(float(item))
                    or not float(item).is_integer()
                ):
                    raise ValueError(
                        f"{path}.far_tracking_topology.{name}[{index}] must be an integer index."
                    )
                item_int = int(item)
                if item_int < 0 or (upper is not None and item_int >= upper):
                    raise ValueError(
                        f"{path}.far_tracking_topology.{name}[{index}] is outside its valid range."
                    )
                values.append(item_int)
            if len(values) != len(set(values)) and name.endswith("slot_indices"):
                raise ValueError(f"{path}.far_tracking_topology.{name} contains duplicate slots.")
            return values

        robot_slots = integer_indices("robot_slot_indices", upper=len(mesh_by_slot))
        robot_body_indices = integer_indices("robot_body_indices")
        robot_body_names = sequence("robot_body_names")
        robot_positions = sequence("robot_body_offset_positions")
        robot_quaternions = sequence("robot_body_offset_quaternions")
        robot_count = len(robot_slots)
        if not (
            len(robot_body_indices) == robot_count
            and len(robot_body_names) == robot_count
            and len(robot_positions) == robot_count * 3
            and len(robot_quaternions) == robot_count * 4
        ):
            raise ValueError(f"{path}.far_tracking_topology robot bindings have inconsistent lengths.")

        object_slots = integer_indices("object_slot_indices", upper=len(mesh_by_slot))
        object_source_indices = integer_indices("object_source_indices")
        object_names = sequence("object_names")
        active_env_ids = sequence("object_active_env_ids")
        if not (
            len(object_slots) == len(object_source_indices) == len(active_env_ids)
        ):
            raise ValueError(f"{path}.far_tracking_topology object bindings have inconsistent lengths.")
        primitive_sources = integer_indices("primitive_source_indices")
        if primitive_sources:
            raise ValueError(
                f"{path}.far_tracking_topology contains primitive geometry whose shape is not checkpointed."
            )
        if set(robot_slots) & set(object_slots):
            raise ValueError(f"{path}.far_tracking_topology assigns a geometry slot more than once.")
        if set(robot_slots) | set(object_slots) != set(range(len(mesh_by_slot))):
            raise ValueError(f"{path}.far_tracking_topology does not bind every geometry slot exactly once.")

        robot_bindings: list[dict[str, Any]] = []
        for index, slot_index in enumerate(robot_slots):
            body_name = robot_body_names[index]
            if not isinstance(body_name, str) or not body_name:
                raise ValueError(f"{path}.far_tracking_topology.robot_body_names[{index}] is invalid.")
            binding = {
                "slot_name": mesh_by_slot[slot_index]["slot_name"],
                "mesh": mesh_by_slot[slot_index]["mesh"],
                "tracking_body_name": body_name,
                "fixed_position_xyz": robot_positions[index * 3 : (index + 1) * 3],
                "fixed_quaternion_xyzw": robot_quaternions[index * 4 : (index + 1) * 4],
            }
            robot_bindings.append(binding)

        active_slot_count = [0] * num_envs
        object_counts: dict[str, dict[str, Any]] = {}
        used_source_indices: set[int] = set()
        for index, slot_index in enumerate(object_slots):
            source_index = object_source_indices[index]
            if source_index >= len(object_names):
                raise ValueError(
                    f"{path}.far_tracking_topology.object_source_indices[{index}] is out of range."
                )
            source_name = object_names[source_index]
            if not isinstance(source_name, str) or not source_name:
                raise ValueError(f"{path}.far_tracking_topology.object_names[{source_index}] is invalid.")
            used_source_indices.add(source_index)
            raw_active = active_env_ids[index]
            if raw_active is None:
                active = list(range(num_envs))
            else:
                if not isinstance(raw_active, (list, tuple)):
                    raise ValueError(
                        f"{path}.far_tracking_topology.object_active_env_ids[{index}] must be null or a sequence."
                    )
                active = list(raw_active)
                if any(type(env_id) is not int or env_id < 0 or env_id >= num_envs for env_id in active):
                    raise ValueError(
                        f"{path}.far_tracking_topology.object_active_env_ids[{index}] is out of range."
                    )
                if len(active) != len(set(active)):
                    raise ValueError(
                        f"{path}.far_tracking_topology.object_active_env_ids[{index}] contains duplicates."
                    )
            if not active:
                raise ValueError(f"{path} contains a geometry variant unused by every environment.")
            for env_id in active:
                active_slot_count[env_id] += 1
            support_identity = {
                "source_name": source_name,
                "mesh": mesh_by_slot[slot_index]["mesh"],
            }
            support_key = cls._canonical_geometry_sort_key(support_identity)
            existing = object_counts.get(support_key)
            if existing is None:
                existing = {**support_identity, "training_active_env_count": 0}
                object_counts[support_key] = existing
            existing["training_active_env_count"] += len(active)
        if object_slots and any(count != 1 for count in active_slot_count):
            raise ValueError(
                f"{path} cannot authenticate one-object direct deployment because some training "
                "environments did not have exactly one active object geometry."
            )
        if used_source_indices != set(range(len(object_names))):
            raise ValueError(f"{path}.far_tracking_topology contains unused object source names.")

        return cls._normalize_training_geometry_support(
            {
                "version": 1,
                "camera_source": camera_source,
                "training_rank_count": 1,
                "robot_mesh_bindings": sorted(
                    robot_bindings,
                    key=cls._canonical_geometry_sort_key,
                ),
                "object_mesh_support": sorted(
                    object_counts.values(),
                    key=cls._canonical_geometry_sort_key,
                ),
            },
            path=f"{path}.deployment_geometry",
        )

    @classmethod
    def aggregate_training_geometry_support(
        cls,
        rank_supports: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Union rank-sharded object support while requiring identical robot geometry."""

        if not rank_supports:
            raise ValueError("Perception training geometry aggregation requires at least one rank.")
        normalized = [
            cls._normalize_training_geometry_support(
                support,
                path=f"rank_geometry_support[{index}]",
            )
            for index, support in enumerate(rank_supports)
        ]
        if any(support["training_rank_count"] != 1 for support in normalized):
            raise ValueError("Rank-local perception geometry support must declare training_rank_count=1.")
        camera_source = normalized[0]["camera_source"]
        robot_bindings = normalized[0]["robot_mesh_bindings"]
        object_counts: dict[str, dict[str, Any]] = {}
        for rank, support in enumerate(normalized):
            if support["camera_source"] != camera_source:
                raise ValueError(f"Perception camera_source differs on training rank {rank}.")
            if support["robot_mesh_bindings"] != robot_bindings:
                raise ValueError(f"Perception robot mesh bindings differ on training rank {rank}.")
            for item in support["object_mesh_support"]:
                identity = {key: item[key] for key in ("source_name", "mesh")}
                identity_key = cls._canonical_geometry_sort_key(identity)
                aggregate = object_counts.get(identity_key)
                if aggregate is None:
                    aggregate = {**identity, "training_active_env_count": 0}
                    object_counts[identity_key] = aggregate
                aggregate["training_active_env_count"] += item["training_active_env_count"]
        return cls._normalize_training_geometry_support(
            {
                "version": 1,
                "camera_source": camera_source,
                "training_rank_count": len(normalized),
                "robot_mesh_bindings": robot_bindings,
                "object_mesh_support": sorted(
                    object_counts.values(),
                    key=cls._canonical_geometry_sort_key,
                ),
            }
        )

    def get_local_geometry_support(self) -> dict[str, Any]:
        return self.geometry_support_from_checkpoint_semantics(
            self._persistent_checkpoint_semantics(),
            path="live_perception.semantics",
        )

    def validate_deployment_geometry_support(self, expected: Any) -> dict[str, Any]:
        """Require live static geometry equality and selected-object membership."""

        normalized = self._normalize_training_geometry_support(expected)
        live = self.get_local_geometry_support()
        if live["camera_source"] != normalized["camera_source"]:
            raise ValueError(
                "Live perception camera source does not match training geometry support: "
                f"live={live['camera_source']!r}, training={normalized['camera_source']!r}."
            )
        if live["robot_mesh_bindings"] != normalized["robot_mesh_bindings"]:
            raise ValueError(
                "Live perception robot meshes/fixed-link bindings differ from the training checkpoint."
            )
        expected_objects = {
            self._canonical_geometry_sort_key(
                {key: item[key] for key in ("source_name", "mesh")}
            )
            for item in normalized["object_mesh_support"]
        }
        live_objects = {
            self._canonical_geometry_sort_key(
                {key: item[key] for key in ("source_name", "mesh")}
            )
            for item in live["object_mesh_support"]
        }
        if bool(expected_objects) != bool(live_objects):
            raise ValueError(
                "Live perception object-geometry presence differs from the training checkpoint."
            )
        unknown = sorted(live_objects - expected_objects)
        if unknown:
            raise ValueError(
                "Live perception selected object geometry is not a member of the authenticated "
                f"training support: {unknown}."
            )
        return normalized

    def authenticate_observation_contract(
        self,
        contract: Any,
        *,
        declared_sha256: str,
    ) -> str:
        """Bind a direct producer to an ONNX contract after live-geometry validation."""

        if not isinstance(contract, Mapping) or contract.get("version") != 2:
            raise ValueError("Direct perception requires a version-2 observation contract mapping.")
        lifecycle = contract.get("producer_lifecycle")
        if (
            not isinstance(lifecycle, Mapping)
            or lifecycle.get("reset_refresh_semantics") != "targeted_v2"
            or self.uses_legacy_full_reset_refresh()
        ):
            raise ValueError(
                "Direct one-environment perception requires targeted_v2 reset-refresh semantics; "
                "legacy vectorized reset producers cannot be represented by RunSim."
            )
        expected_support = self.validate_deployment_geometry_support(
            contract.get("training_geometry_support")
        )
        rebuilt = self._build_observation_contract(
            training_geometry_support=expected_support,
        )
        expected_payload = json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        rebuilt_payload = json.dumps(
            rebuilt,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        if rebuilt_payload != expected_payload:
            raise ValueError(
                "Live direct perception transform/noise/cadence differs from the authenticated ONNX contract."
            )
        computed = hashlib.sha256(expected_payload).hexdigest()
        if (
            not isinstance(declared_sha256, str)
            or declared_sha256 != declared_sha256.lower()
            or len(declared_sha256) != 64
            or any(char not in "0123456789abcdef" for char in declared_sha256)
            or computed != declared_sha256
        ):
            raise ValueError("Direct perception observation-contract SHA-256 is invalid or mismatched.")
        self._authenticated_observation_contract = json.loads(expected_payload)
        return computed

    def _build_observation_contract(
        self,
        *,
        training_geometry_support: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the effective, transportable perception-observation contract.

        The training checkpoint state above also records resume-only details
        such as environment count, sampled calibration tensors, simulator
        indices, and active geometry slots.  A deployed student instead needs
        the deterministic transform/distribution that produced its flattened
        perception tensor.  This projection intentionally omits backend
        allocator indices and raw paths while retaining content-addressed
        semantic geometry bindings, resolved image geometry, preprocessing,
        timing, noise/randomization distributions, and renderer conventions.
        """

        semantics = self._persistent_checkpoint_semantics()
        camera_geometry = dict(semantics["camera_geometry"])
        camera_geometry.pop("body_index", None)
        camera_geometry.pop("far_tracking_base_link_indices", None)
        heightmap_geometry = dict(semantics["heightmap_geometry"])
        heightmap_geometry.pop("body_index", None)
        latency_range = semantics["camera_warp_latency_frame_range"]
        reset_semantics = semantics["reset_refresh_semantics"]
        camera_reset_randomization = semantics["camera_reset_randomization"]

        # Targeting only the reset subset prevents a second camera-frequency
        # tick and a second Perlin-hole frame, but it does not make the current
        # implementation's stochastic sample path environment-local.  Camera
        # reset sampling and several pixel-noise stages still use PyTorch's
        # process-global RNG.  Record that limitation explicitly so a
        # one-environment RunSim can authenticate the trained distributions
        # without claiming bitwise replay of a vectorized rollout.
        reset_randomization_consumes_rng = bool(
            camera_reset_randomization is not None
            and any(
                camera_reset_randomization.get(name) is not None
                for name in (
                    "translation_xyz",
                    "rotation_rpy_deg",
                    "noise_std_mult",
                    "noise_drop_prob",
                )
            )
        )
        sensor_noise_on_reset = bool(
            getattr(self, "_camera_apply_sensor_noise", False)
            and camera_reset_randomization is not None
            and (
                camera_reset_randomization.get("noise_std_mult") is not None
                or camera_reset_randomization.get("noise_drop_prob") is not None
            )
        )
        reset_refresh_pixel_rng = bool(
            semantics["output_mode"] == "camera_depth"
            and (
                bool(getattr(self, "_camera_warp_edge_noise", False))
                or float(getattr(self, "_camera_warp_additive_noise_std", 0.0) or 0.0) > 0.0
                or latency_range is not None
                or sensor_noise_on_reset
            )
        )
        reset_refresh_consumes_global_rng = bool(
            reset_randomization_consumes_rng or reset_refresh_pixel_rng
        )

        return {
            "version": 2,
            "output_mode": semantics["output_mode"],
            "camera_source": semantics["camera_source"],
            "camera_shape": semantics["camera_shape"],
            "camera_obs_shape": semantics["camera_obs_shape"],
            "camera_geometry": camera_geometry,
            "heightmap_geometry": heightmap_geometry,
            "camera_strict_warp": semantics["camera_strict_warp"],
            "camera_disable_offsets": semantics["camera_disable_offsets"],
            "update_interval": semantics["update_interval"],
            "camera_warp_preprocess": semantics["camera_warp_preprocess"],
            "camera_warp_freq_ratio": semantics["camera_warp_freq_ratio"],
            "camera_warp_buffer_len": semantics["camera_warp_buffer_len"],
            "camera_warp_latency_frame": (
                None if latency_range is not None else semantics["camera_warp_latency_frame"]
            ),
            "camera_warp_latency_frame_range": latency_range,
            "camera_reset_randomization": camera_reset_randomization,
            "camera_setup_randomization": semantics["camera_setup_randomization"],
            "training_geometry_support": self._normalize_training_geometry_support(
                training_geometry_support
            ),
            "producer_tick_dt": float(getattr(self.env, "dt", 0.0)),
            "producer_lifecycle": {
                "reset_refresh_semantics": reset_semantics,
                "ordinary_manager_update_calls_per_control_tick": 1,
                "initialization_control_ticks_before_first_reset_output": 1,
                "initialization_ordinary_manager_update_calls_before_first_reset_output": 1,
                "reset_output_republished_until_physics_advances": True,
                "reset_output_scope": (
                    "full_vectorized_batch"
                    if reset_semantics == "legacy_full_v1"
                    else "reset_env_subset"
                ),
                "hole_clock_advances_on_reset_refresh": bool(
                    reset_semantics == "legacy_full_v1"
                ),
                "camera_frequency_phase_advances_on_reset_refresh": bool(
                    reset_semantics == "legacy_full_v1"
                ),
                "camera_producer_reset_refresh_consumes_process_global_rng": (
                    reset_refresh_consumes_global_rng
                ),
                "future_noise_sample_path_peer_reset_coupled": bool(
                    reset_semantics == "legacy_full_v1"
                    or reset_refresh_consumes_global_rng
                ),
                "batch_size_invariant_sample_path": False,
                "stochastic_equivalence": (
                    "not_replayable_one_env"
                    if reset_semantics == "legacy_full_v1"
                    else "distribution_only"
                ),
                "seed_replay_scope": "same_execution_trace_only",
            },
            "camera_ray_correction_quaternion_xyzw": self._semantic_tensor_tuple(
                getattr(self, "_camera_ray_correction_quat", None)
            ),
            "effective_observation_schema": semantics["effective_observation_schema"],
            "rendered_backend": semantics["rendered_backend"],
            "hole_generator_schema": semantics["hole_generator_schema"],
        }

    def get_observation_contract(
        self,
        *,
        training_geometry_support: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return the live or explicitly authenticated transport contract."""

        authenticated_contract = getattr(
            self,
            "_authenticated_observation_contract",
            None,
        )
        if training_geometry_support is None and authenticated_contract is not None:
            return copy.deepcopy(authenticated_contract)
        if training_geometry_support is None:
            training_geometry_support = self.get_local_geometry_support()
        return self._build_observation_contract(
            training_geometry_support=training_geometry_support,
        )

    def get_observation_contract_sha256(
        self,
        *,
        training_geometry_support: dict[str, Any] | None = None,
    ) -> str:
        """Hash :meth:`get_observation_contract` using canonical strict JSON."""

        payload = json.dumps(
            self.get_observation_contract(
                training_geometry_support=training_geometry_support,
            ),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def uses_legacy_full_reset_refresh(self) -> bool:
        """Whether any peer reset advances the complete vectorized sensor stream."""

        return getattr(self, "_reset_refresh_semantics", "legacy_full_v1") == "legacy_full_v1"

    @staticmethod
    def _checkpoint_tensor_like(
        value: Any,
        *,
        reference: torch.Tensor,
        path: str,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Perception checkpoint {path} must be a tensor.")
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(
                f"Perception checkpoint {path} shape {tuple(value.shape)} does not match "
                f"runtime {tuple(reference.shape)}."
            )
        if value.dtype != reference.dtype:
            raise ValueError(
                f"Perception checkpoint {path} dtype {value.dtype} does not match runtime {reference.dtype}."
            )
        if not value.is_floating_point() or not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"Perception checkpoint {path} must contain only finite floating values.")
        return value.detach().to(device=reference.device, dtype=reference.dtype).clone()

    @staticmethod
    def _validate_checkpoint_unit_quaternion(value: torch.Tensor, *, path: str) -> None:
        norms = torch.linalg.vector_norm(value, dim=-1)
        if bool((torch.abs(norms - 1.0) > 1.0e-4).any().item()):
            raise ValueError(f"Perception checkpoint {path} must contain unit quaternions.")

    def get_persistent_checkpoint_state(self) -> dict[str, Any]:
        """Serialize sampled calibration that survives ordinary episode resets."""

        shared_mount = None
        semantics = self._persistent_checkpoint_semantics()
        if semantics["shared_mount_present"]:
            shared_mount = {
                "local_position": self._shared_camera_sensor_local_position.detach().to("cpu").clone(),
                "local_orientation": self._shared_camera_sensor_local_orientation.detach().to("cpu").clone(),
                "data_frame_quat": self._shared_camera_sensor_data_frame_quat.detach().to("cpu").clone(),
            }
        return {
            "version": 1,
            "semantics": semantics,
            "camera_warp_depth_offset": self._camera_warp_depth_offset.detach().to("cpu").clone(),
            "camera_ray_correction_quat": self._camera_ray_correction_quat.detach().to("cpu").clone(),
            "camera_obs_step_counter": int(self._camera_obs_step_counter),
            "time_since_update": float(self._time_since_update),
            "hole_frame_idx": (
                None
                if self._camera_warp_hole_generator is None
                else int(self._camera_warp_hole_generator.frame_idx)
            ),
            "shared_mount": shared_mount,
        }

    def _prepare_persistent_checkpoint_state(self, state: Any) -> dict[str, Any]:
        if not isinstance(state, dict):
            raise ValueError("Perception persistent checkpoint state must be a dictionary.")
        if set(state) != {
            "version",
            "semantics",
            "camera_warp_depth_offset",
            "camera_ray_correction_quat",
            "camera_obs_step_counter",
            "time_since_update",
            "hole_frame_idx",
            "shared_mount",
        }:
            raise ValueError(
                "Perception persistent checkpoint state has missing or unsupported fields."
            )
        version = state.get("version")
        if isinstance(version, bool) or not isinstance(version, numbers.Integral) or int(version) != 1:
            raise ValueError(f"Unsupported perception checkpoint version: {version!r}.")
        if state.get("semantics") != self._persistent_checkpoint_semantics():
            raise ValueError("Perception checkpoint semantics differ from the active runtime.")
        prepared = {
            "camera_warp_depth_offset": self._checkpoint_tensor_like(
                state.get("camera_warp_depth_offset"),
                reference=self._camera_warp_depth_offset,
                path="camera_warp_depth_offset",
            ),
            "camera_ray_correction_quat": self._checkpoint_tensor_like(
                state.get("camera_ray_correction_quat"),
                reference=self._camera_ray_correction_quat,
                path="camera_ray_correction_quat",
            ),
            "shared_mount": None,
        }
        self._validate_checkpoint_unit_quaternion(
            prepared["camera_ray_correction_quat"],
            path="camera_ray_correction_quat",
        )
        camera_obs_step_counter = state.get("camera_obs_step_counter")
        if (
            isinstance(camera_obs_step_counter, bool)
            or not isinstance(camera_obs_step_counter, numbers.Integral)
            or int(camera_obs_step_counter) < 0
        ):
            raise ValueError("Perception checkpoint camera_obs_step_counter must be a non-negative integer.")
        time_since_update = state.get("time_since_update")
        if (
            isinstance(time_since_update, bool)
            or not isinstance(time_since_update, numbers.Real)
            or not math.isfinite(float(time_since_update))
            or float(time_since_update) < 0.0
        ):
            raise ValueError("Perception checkpoint time_since_update must be finite and non-negative.")
        hole_frame_idx = state.get("hole_frame_idx")
        if self._camera_warp_hole_generator is None:
            if hole_frame_idx is not None:
                raise ValueError("Perception checkpoint unexpectedly contains a hole frame index.")
        elif (
            isinstance(hole_frame_idx, bool)
            or not isinstance(hole_frame_idx, numbers.Integral)
            or int(hole_frame_idx) < 0
        ):
            raise ValueError("Perception checkpoint hole_frame_idx must be a non-negative integer.")
        prepared["camera_obs_step_counter"] = int(camera_obs_step_counter)
        prepared["time_since_update"] = float(time_since_update)
        prepared["hole_frame_idx"] = None if hole_frame_idx is None else int(hole_frame_idx)
        shared_mount = state.get("shared_mount")
        if self._persistent_checkpoint_semantics()["shared_mount_present"]:
            if not isinstance(shared_mount, dict) or set(shared_mount) != {
                "local_position",
                "local_orientation",
                "data_frame_quat",
            }:
                raise ValueError("Perception checkpoint shared_mount is incomplete.")
            prepared_shared = {
                "local_position": self._checkpoint_tensor_like(
                    shared_mount.get("local_position"),
                    reference=self._shared_camera_sensor_local_position,
                    path="shared_mount.local_position",
                ),
                "local_orientation": self._checkpoint_tensor_like(
                    shared_mount.get("local_orientation"),
                    reference=self._shared_camera_sensor_local_orientation,
                    path="shared_mount.local_orientation",
                ),
                "data_frame_quat": self._checkpoint_tensor_like(
                    shared_mount.get("data_frame_quat"),
                    reference=self._shared_camera_sensor_data_frame_quat,
                    path="shared_mount.data_frame_quat",
                ),
            }
            for name in ("local_orientation", "data_frame_quat"):
                self._validate_checkpoint_unit_quaternion(
                    prepared_shared[name],
                    path=f"shared_mount.{name}",
                )
            sensor = self._far_tracking_camera_sensor
            if sensor is not None:
                for state_name, sensor_name in (
                    ("local_position", "camera_sensor_local_position"),
                    ("local_orientation", "camera_sensor_local_orientation"),
                    ("data_frame_quat", "camera_sensor_data_frame_quat"),
                ):
                    sensor_value = getattr(sensor, sensor_name, None)
                    prepared_value = prepared_shared[state_name]
                    if (
                        not isinstance(sensor_value, torch.Tensor)
                        or tuple(sensor_value.shape) != tuple(prepared_value.shape)
                        or sensor_value.dtype != prepared_value.dtype
                    ):
                        raise ValueError(
                            f"Runtime far-tracking sensor {sensor_name} is incompatible with checkpoint mount state."
                        )
            prepared["shared_mount"] = prepared_shared
        elif shared_mount is not None:
            raise ValueError("Perception checkpoint unexpectedly contains shared_mount state.")
        return prepared

    def validate_persistent_checkpoint_state(self, state: Any) -> None:
        self._prepare_persistent_checkpoint_state(state)

    def load_persistent_checkpoint_state(self, state: Any) -> None:
        """Restore sampled calibration after complete validation."""

        prepared = self._prepare_persistent_checkpoint_state(state)
        # Ray directions are derived from the correction quaternion during
        # setup.  Rebuild them from the authenticated checkpoint value before
        # committing live state; otherwise an auto-fixed camera can expose the
        # restored quaternion while raycasting with stale fresh-process rays.
        restored_correction = prepared["camera_ray_correction_quat"]
        rebuilt_camera_rays = (
            self._build_camera_rays(correction_quat=restored_correction)
            if self._camera_ray_dirs_base is not None
            else None
        )
        rebuilt_scandots_rays = (
            self._build_camera_scandots_rays(correction_quat=restored_correction)
            if self._camera_scandots_ray_dirs_base is not None
            else None
        )
        self._camera_warp_depth_offset.copy_(prepared["camera_warp_depth_offset"])
        self._camera_ray_correction_quat.copy_(restored_correction)
        if rebuilt_camera_rays is not None:
            self._camera_ray_dirs_base = rebuilt_camera_rays
        if rebuilt_scandots_rays is not None:
            self._camera_scandots_ray_dirs_base = rebuilt_scandots_rays
        self._camera_obs_step_counter = prepared["camera_obs_step_counter"]
        self._time_since_update = prepared["time_since_update"]
        if self._camera_warp_hole_generator is not None:
            self._camera_warp_hole_generator.frame_idx = prepared["hole_frame_idx"]
            self._camera_warp_hole_generator.gradient_cache = [
                {} for _ in self._camera_warp_hole_generator.resolutions
            ]
            self._camera_warp_hole_frame_stats = None
        shared_mount = prepared["shared_mount"]
        if shared_mount is None:
            return
        self._shared_camera_sensor_local_position = shared_mount["local_position"]
        self._shared_camera_sensor_local_orientation = shared_mount["local_orientation"]
        self._shared_camera_sensor_data_frame_quat = shared_mount["data_frame_quat"]
        sensor = self._far_tracking_camera_sensor
        if sensor is not None:
            sensor.camera_sensor_local_position.copy_(
                self._shared_camera_sensor_local_position.to(
                    device=sensor.camera_sensor_local_position.device,
                    dtype=sensor.camera_sensor_local_position.dtype,
                )
            )
            sensor.camera_sensor_local_orientation.copy_(
                self._shared_camera_sensor_local_orientation.to(
                    device=sensor.camera_sensor_local_orientation.device,
                    dtype=sensor.camera_sensor_local_orientation.dtype,
                )
            )
            sensor.camera_sensor_data_frame_quat.copy_(
                self._shared_camera_sensor_data_frame_quat.to(
                    device=sensor.camera_sensor_data_frame_quat.device,
                    dtype=sensor.camera_sensor_data_frame_quat.dtype,
                )
            )

    def update(self, env_ids: torch.Tensor | None = None) -> None:
        if not self.enabled:
            return
        self._debug_update_counter += 1
        if env_ids is None and self._update_interval > 0.0:
            self._time_since_update += float(self.env.dt)
            if self._time_since_update + 1.0e-8 < self._update_interval:
                return
            self._time_since_update -= self._update_interval

        self._log_camera_randomization_state_once()

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
            env_id = torch.tensor([self._rendered_camera_env_id], device=self.device, dtype=torch.long)
            camera_depth = self._prepare_camera_depth_for_observation(camera_depth, env_ids=env_id)
            self._camera_depth[env_id] = camera_depth
            self._update_camera_depth_observation(
                env_id,
                camera_depth,
                refresh=self._camera_obs_refresh_flag_for_update(env_ids),
                advance_temporal_noise=env_ids is None,
            )
            self._maybe_dump_camera_debug(source_label="rendered", env_ids=env_id)
            self._maybe_log_runtime_camera_alignment()
            return

        if self._uses_pytorch3d():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_pytorch3d_depth(env_ids)
            camera_depth = self._prepare_camera_depth_for_observation(camera_depth, env_ids=env_ids)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._camera_obs_refresh_flag_for_update(env_ids),
                advance_temporal_noise=env_ids is None,
            )
            self._maybe_dump_camera_debug(source_label="pytorch3d", env_ids=idx)
            self._maybe_log_runtime_camera_alignment()
            return

        if self._uses_camera_far_tracking():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_far_tracking_camera_depth(env_ids)
            camera_depth = self._prepare_camera_depth_for_observation(camera_depth, env_ids=env_ids)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._camera_obs_refresh_flag_for_update(env_ids),
                advance_temporal_noise=env_ids is None,
            )
            self._maybe_dump_camera_debug(source_label="far_tracking_warp", env_ids=idx)
            self._maybe_log_runtime_camera_alignment()
            return

        if self._uses_camera_scandots():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_camera_scandots_depth(env_ids)
            camera_depth = self._prepare_camera_depth_for_observation(camera_depth, env_ids=env_ids)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._camera_obs_refresh_flag_for_update(env_ids),
                advance_temporal_noise=env_ids is None,
            )
            self._maybe_dump_camera_debug(source_label="scandots", env_ids=idx)
            self._maybe_log_runtime_camera_alignment()
            return

        if self._uses_camera_raycast():
            idx = env_ids if env_ids is not None else slice(None)
            camera_depth = self._compute_camera_raycast_depth(env_ids)
            camera_depth = self._prepare_camera_depth_for_observation(camera_depth, env_ids=env_ids)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._camera_obs_refresh_flag_for_update(env_ids),
                advance_temporal_noise=env_ids is None,
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
            camera_depth = self._prepare_camera_depth_for_observation(camera_depth, env_ids=env_ids)
            self._camera_depth[idx] = camera_depth
            self._update_camera_depth_observation(
                idx,
                camera_depth,
                refresh=self._camera_obs_refresh_flag_for_update(env_ids),
                advance_temporal_noise=env_ids is None,
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
            "camera_warp_latency_frame": (
                list(self._camera_warp_latency_frame_range)
                if self._camera_warp_latency_frame_range is not None
                else int(self._camera_warp_latency_frame)
            ),
            "camera_warp_buffer_len": int(self._camera_warp_buffer_len),
            "camera_warp_normalize": bool(self._camera_warp_normalize),
            "camera_warp_enable_holes": bool(self._camera_warp_enable_holes),
            "camera_warp_hole_prob": float(self._camera_warp_hole_prob),
            "camera_warp_additive_noise_std": float(self._camera_warp_additive_noise_std),
            "camera_warp_depth_offset_std": float(self._camera_warp_depth_offset_std),
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

        if apply_sensor_offset and apply_pitch and self._camera_strict_warp:
            return self._get_strict_warp_camera_pose(env_ids)

        idx = env_ids if env_ids is not None else slice(None)
        body_pos, body_quat = self._get_camera_body_pose(idx)

        if apply_sensor_offset and apply_pitch:
            num_envs = body_pos.shape[0]
            local_position = self._sensor_offset.to(
                device=body_pos.device,
                dtype=body_pos.dtype,
            ).unsqueeze(0).expand(num_envs, -1)
            combo = self._camera_ray_rotation_quat(
                device=body_quat.device,
                dtype=body_quat.dtype,
            ).unsqueeze(0).expand(num_envs, -1)
            local_position, combo = self._apply_runtime_camera_mount_offsets(
                local_position,
                combo,
                idx=idx,
            )
            return (
                body_pos + quat_apply(body_quat, local_position, w_last=True),
                quat_mul(body_quat, combo, w_last=True),
            )

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

        mount_rot_deg = self._strict_camera_mount_rotation_deg.to(
            device=self.device,
            dtype=torch.float32,
        ).view(1, 1, 3)
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
        randomize_mount_raw = os.environ.get("HOLOSOMA_CAMERA_RANDOMIZE_PLACEMENT", "1").strip().lower()
        # An enabled reset term supplies fresh camera offsets every episode.
        # Keep the one-shot fallback when the manager does not configure that
        # specific term; otherwise setup jitter and reset jitter would stack.
        has_reset_randomization = self._has_camera_reset_randomization()
        randomize_mount = (
            randomize_mount_raw not in {"0", "false", "no", "off", ""}
            and not has_reset_randomization
        )
        self._camera_setup_randomization_enabled = bool(randomize_mount)
        self._camera_setup_translation_range = tuple(
            (float(low), float(high))
            for low, high in zip(
                translation_jitter_min.detach().to("cpu").tolist(),
                translation_jitter_max.detach().to("cpu").tolist(),
                strict=True,
            )
        )
        self._camera_setup_rotation_range_deg = tuple(
            (float(low), float(high))
            for low, high in zip(
                rotation_jitter_min.detach().to("cpu").tolist(),
                rotation_jitter_max.detach().to("cpu").tolist(),
                strict=True,
            )
        )
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

    def _camera_reset_randomization_params(self) -> dict[str, Any] | None:
        randomization_manager = getattr(getattr(self, "env", None), "randomization_manager", None)
        reset_terms = getattr(getattr(randomization_manager, "cfg", None), "reset_terms", {})
        if not isinstance(reset_terms, dict):
            raise ValueError("Camera randomization reset_terms must be a mapping.")
        canonical_func = "holosoma.managers.randomization.terms.locomotion.randomize_camera_raycast"
        matches: list[dict[str, Any]] = []
        for term_cfg in reset_terms.values():
            func_path = str(getattr(term_cfg, "func", ""))
            normalized_func = func_path.replace(":", ".")
            if normalized_func.rsplit(".", maxsplit=1)[-1] != "randomize_camera_raycast":
                continue
            if normalized_func != canonical_func:
                raise ValueError(
                    "Camera reset randomization must use the canonical randomize_camera_raycast implementation, "
                    f"got {func_path!r}."
                )
            params = getattr(term_cfg, "params", {}) or {}
            if not isinstance(params, dict):
                raise ValueError("Camera reset randomization params must be a mapping.")
            enabled = params.get("enabled", True)
            if not isinstance(enabled, (bool, np.bool_)):
                raise ValueError(f"Camera reset randomization enabled must be boolean, got {enabled!r}.")
            if not bool(enabled):
                continue
            allowed_params = {
                "enabled",
                "translation_range",
                "rotation_range_deg",
                "noise_std_mult_range",
                "noise_drop_prob_range",
            }
            unexpected = sorted(set(params) - allowed_params)
            if unexpected:
                raise ValueError(
                    "Camera reset randomization contains unauthenticated parameters: "
                    f"{unexpected}."
                )
            matches.append(dict(params))
        if len(matches) > 1:
            raise ValueError(
                "At most one enabled randomize_camera_raycast reset term is supported; "
                f"found {len(matches)}."
            )
        return matches[0] if matches else None

    def _camera_setup_randomization_semantics(self) -> dict[str, Any] | None:
        if not self._camera_strict_warp:
            return None
        return {
            "enabled": bool(getattr(self, "_camera_setup_randomization_enabled", False)),
            "translation_xyz": getattr(self, "_camera_setup_translation_range", None),
            "rotation_rpy_deg": getattr(self, "_camera_setup_rotation_range_deg", None),
        }

    def _camera_reset_randomization_semantics(self) -> dict[str, Any] | None:
        """Return the effective, representation-independent reset ranges."""

        params = self._camera_reset_randomization_params()
        if params is None:
            return None

        def checked(value: Any, *, path: str) -> float:
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"Camera randomization {path} must be numeric, got boolean {value!r}.")
            result = float(value)
            if not math.isfinite(result):
                raise ValueError(f"Camera randomization {path} must be finite.")
            return result

        def scalar_range(spec: Any, *, path: str) -> tuple[float, float] | None:
            if spec is None:
                return None
            if isinstance(spec, (list, tuple)):
                if len(spec) != 2:
                    raise ValueError(
                        f"Camera randomization {path} must contain exactly [low, high]."
                    )
                result = (
                    checked(spec[0], path=f"{path}[0]"),
                    checked(spec[1], path=f"{path}[1]"),
                )
            else:
                value = checked(spec, path=path)
                result = (value, value)
            if result[0] > result[1]:
                raise ValueError(
                    f"Camera randomization {path} lower bound {result[0]} exceeds upper bound {result[1]}."
                )
            return result

        def vector_ranges(
            spec: Any,
            *,
            keys: tuple[str, str, str],
            path: str,
        ) -> tuple[tuple[float, float], ...] | None:
            if spec is None:
                return None
            if isinstance(spec, dict):
                if set(spec) != set(keys):
                    raise ValueError(
                        f"Camera randomization {path} must declare exactly {list(keys)}, "
                        f"got {sorted(str(key) for key in spec)}."
                    )
                return tuple(
                    scalar_range(spec[key], path=f"{path}.{key}")
                    for key in keys
                )
            shared = scalar_range(spec, path=path)
            if shared is None:  # pragma: no cover - guarded above.
                return None
            return (shared, shared, shared)

        noise_std_mult = scalar_range(
            params.get("noise_std_mult_range"),
            path="noise_std_mult_range",
        )
        if noise_std_mult is not None and noise_std_mult[0] < 0.0:
            raise ValueError("Camera randomization noise_std_mult_range must be non-negative.")
        noise_drop_prob = scalar_range(
            params.get("noise_drop_prob_range"),
            path="noise_drop_prob_range",
        )
        if noise_drop_prob is not None and (
            noise_drop_prob[0] < 0.0 or noise_drop_prob[1] > 1.0
        ):
            raise ValueError("Camera randomization noise_drop_prob_range must lie within [0, 1].")

        return {
            "enabled": True,
            "translation_xyz": vector_ranges(
                params.get("translation_range"),
                keys=("x", "y", "z"),
                path="translation_range",
            ),
            "rotation_rpy_deg": vector_ranges(
                params.get("rotation_range_deg"),
                keys=("roll", "pitch", "yaw"),
                path="rotation_range_deg",
            ),
            "noise_std_mult": noise_std_mult,
            "noise_drop_prob": noise_drop_prob,
        }

    def _has_camera_reset_randomization(self) -> bool:
        return self._camera_reset_randomization_params() is not None

    def _apply_runtime_camera_mount_offsets(
        self,
        local_position: torch.Tensor,
        local_orientation: torch.Tensor,
        *,
        idx: torch.Tensor | slice,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply reset samples in the camera mount frame.

        This matches the bundled far-tracking placement contract: translation
        perturbs the local sensor position, while RPY noise is added to the
        configured mount Euler angles.  In particular, rotation jitter must
        not rotate the sensor's 0.44 m lever arm around the robot body.
        """

        if self._camera_disable_offsets:
            return local_position, local_orientation

        def select(name: str, reference: torch.Tensor) -> torch.Tensor | None:
            value = getattr(self.env, name, None)
            if not isinstance(value, torch.Tensor):
                return None
            selected = value if isinstance(idx, slice) else value[idx]
            return selected.to(device=reference.device, dtype=reference.dtype)

        translation = select("_perception_camera_offset_pos", local_position)
        if translation is not None:
            local_position = local_position + translation

        rotation_rpy = select("_perception_camera_offset_rpy", local_orientation)
        if self._camera_strict_warp and rotation_rpy is not None:
            base_rotation = torch.deg2rad(
                self._strict_camera_mount_rotation_deg.to(
                    device=local_orientation.device,
                    dtype=local_orientation.dtype,
                )
            ).unsqueeze(0)
            mount_rpy = base_rotation + rotation_rpy
            local_orientation = quat_from_euler_xyz(
                mount_rpy[:, 0],
                mount_rpy[:, 1],
                mount_rpy[:, 2],
            )
            pitch_deg = float(getattr(self.cfg, "camera_pitch_deg", 0.0) or 0.0)
            if abs(pitch_deg) > 1.0e-6:
                pitch_rad = torch.deg2rad(
                    torch.tensor(
                        pitch_deg,
                        device=local_orientation.device,
                        dtype=local_orientation.dtype,
                    )
                )
                pitch_quat = quat_from_euler_xyz(
                    torch.tensor(0.0, device=local_orientation.device, dtype=local_orientation.dtype),
                    pitch_rad,
                    torch.tensor(0.0, device=local_orientation.device, dtype=local_orientation.dtype),
                ).unsqueeze(0).expand_as(local_orientation)
                local_orientation = quat_mul(
                    pitch_quat,
                    local_orientation,
                    w_last=True,
                )
        elif rotation_rpy is not None:
            jitter = quat_from_euler_xyz(
                rotation_rpy[:, 0],
                rotation_rpy[:, 1],
                rotation_rpy[:, 2],
            )
            local_orientation = quat_mul(local_orientation, jitter, w_last=True)
        else:
            # Compatibility with environments that provide only the older
            # quaternion state.  New sampling always records RPY as well.
            jitter = select("_perception_camera_offset_quat", local_orientation)
            if jitter is not None:
                local_orientation = quat_mul(local_orientation, jitter, w_last=True)
        return local_position, local_orientation

    def _log_camera_randomization_state_once(self) -> None:
        """Validate and record the effective first-reset camera randomization state."""

        if getattr(self, "_camera_randomization_log_done", False) or self.cfg.output_mode != "camera_depth":
            return

        params = self._camera_reset_randomization_params()
        std_mult = getattr(self.env, "_perception_camera_noise_std_mult", None)
        drop_prob = getattr(self.env, "_perception_camera_noise_drop_prob", None)
        offset_pos = getattr(self.env, "_perception_camera_offset_pos", None)
        offset_quat = getattr(self.env, "_perception_camera_offset_quat", None)

        if params is not None:
            required = {
                "noise_std_mult_range": std_mult,
                "noise_drop_prob_range": drop_prob,
                "translation_range": offset_pos,
                "rotation_range_deg": offset_quat,
            }
            missing = [name for name, value in required.items() if params.get(name) is not None and value is None]
            if missing:
                raise RuntimeError(
                    "Camera reset randomization is enabled but its sampled runtime state is missing: "
                    + ", ".join(missing)
                )

        if self._camera_apply_sensor_noise and std_mult is None and drop_prob is None:
            if os.environ.get("HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                logger.warning(
                    "perception.camera_apply_sensor_noise=True but runtime sensor-noise state is missing; "
                    "continuing because HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE is enabled."
                )
            else:
                raise RuntimeError(
                    "perception.camera_apply_sensor_noise=True, but the runtime produced neither "
                    "_perception_camera_noise_std_mult nor _perception_camera_noise_drop_prob. "
                    "Refusing to advertise sensor noise while applying none."
                )

        def scalar_stats(value: Any) -> str:
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                return "none"
            sample = value.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
            return f"{float(sample.min()):.6g}/{float(sample.max()):.6g}/{float(sample.mean()):.6g}"

        offset_pos_range = "none"
        if isinstance(offset_pos, torch.Tensor) and offset_pos.numel() > 0:
            sample_pos = offset_pos.detach().to(device="cpu", dtype=torch.float32).reshape(-1, 3)
            axis_min = [float(value) for value in sample_pos.min(dim=0).values.tolist()]
            axis_max = [float(value) for value in sample_pos.max(dim=0).values.tolist()]
            offset_pos_range = f"min={axis_min} max={axis_max}"

        offset_angle_stats = "none"
        if isinstance(offset_quat, torch.Tensor) and offset_quat.numel() > 0:
            sample_quat = offset_quat.detach().to(device="cpu", dtype=torch.float32).reshape(-1, 4)
            quat_norm = torch.linalg.vector_norm(sample_quat, dim=-1).clamp(min=1.0e-12)
            quat_w = (sample_quat[:, 3] / quat_norm).abs().clamp(max=1.0)
            angle_deg = torch.rad2deg(2.0 * torch.acos(quat_w))
            offset_angle_stats = scalar_stats(angle_deg)

        (getattr(self, "logger", None) or logger).info(
            "Perception camera stochastic semantics: source={} sensor_noise={} "
            "reset_pose_randomization={} std_mult_min/max/mean={} drop_prob_min/max/mean={} "
            "offset_pos_range={} offset_angle_deg_min/max/mean={}",
            getattr(self, "_camera_source", "unknown"),
            bool(getattr(self, "_camera_apply_sensor_noise", False)),
            params is not None,
            scalar_stats(std_mult),
            scalar_stats(drop_prob),
            offset_pos_range,
            offset_angle_stats,
        )
        self._camera_randomization_log_done = True

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
        local_position, local_orientation = self._apply_runtime_camera_mount_offsets(
            local_position,
            local_orientation,
            idx=idx,
        )

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

        # Scientific provenance hashes this exact bundled implementation.  Do
        # not search sys.path or adjacent repositories: that made identical
        # provenance execute different camera code on different machines.
        from holosoma.third_party.ft_warp_sensors.camera_sensor import (  # noqa: PLC0415
            CameraSensor as FarTrackingCameraSensor,
        )
        from holosoma.third_party.ft_warp_sensors.sensor_utils import (  # noqa: PLC0415
            quat_mul_xyzw as ft_quat_mul_xyzw,
        )
        from holosoma.third_party.ft_warp_sensors.sensor_utils import (  # noqa: PLC0415
            tf_apply_xyzw as ft_tf_apply_xyzw,
        )

        (self.logger or logger).info(
            "Using provenance-bound bundled holosoma ft_warp_sensors implementation."
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
            disable_combined_meshes = os.environ.get(
                "HOLOSOMA_FAR_TRACKING_DISABLE_COMBINED_DEPTH_MESHES",
                "",
            ).strip().lower() in {"1", "true", "yes", "on"}
            mesh_name_str = str(mesh_name)
            if disable_combined_meshes and mesh_name_str.startswith("combined_"):
                candidates = [
                    f"{link_name}.STL",
                    f"{link_name}.stl",
                    mesh_name_str,
                ]
            else:
                candidates = [
                    mesh_name_str,
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

        # Include all registered non-robot simulator objects (e.g., training box)
        # so they participate in depth raycasting alongside terrain/robot meshes.
        registered_object_primitives: dict[str, dict[str, Any]] = {}
        registered_object_meshes = self._collect_registered_sim_object_meshes()
        for slot_name, slot_spec in registered_object_meshes.items():
            if slot_name in ray_cast_bodies:
                continue
            ray_cast_bodies[slot_name] = str(slot_spec["mesh_path"])
        if self._camera_include_robot_mesh and not ray_cast_bodies:
            raise RuntimeError(f"No valid far_tracking_warp ray_cast_bodies found under mesh root: {mesh_root}")
        if not self._camera_include_robot_mesh:
            (self.logger or logger).info("far_tracking_warp robot visual mesh raycast disabled by configuration.")
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

        def _resolve_robot_body_spec(
            name: str,
        ) -> tuple[str, int, torch.Tensor, torch.Tensor] | None:
            if name in body_names:
                return (
                    str(name),
                    int(body_names.index(name)),
                    torch.zeros(3, device=self.device, dtype=torch.float32),
                    torch.tensor(
                        [0.0, 0.0, 0.0, 1.0],
                        device=self.device,
                        dtype=torch.float32,
                    ),
                )
            resolved = resolve_fixed_link_offset(
                self.env.robot_config,
                name,
                available_links=body_names,
                device=self.device,
            )
            if resolved is None:
                return None
            parent_name, offset_pos, offset_quat = resolved
            return str(parent_name), int(body_names.index(parent_name)), offset_pos, offset_quat

        if self._camera_body_name is not None and self._camera_body_index is None:
            raise RuntimeError(f"Body '{camera_body_name}' not found in robot body_names for far_tracking_warp source.")
        # -1 denotes the simulator root pose.  The effective fixed-link
        # transform is carried by _get_camera_body_pose rather than discarded.
        base_link_indices = [
            -1 if self._camera_body_index is None else int(self._camera_body_index)
        ]

        robot_slot_indices: list[int] = []
        robot_body_indices: list[int] = []
        robot_body_names: list[str] = []
        robot_body_offset_positions: list[torch.Tensor] = []
        robot_body_offset_quaternions: list[torch.Tensor] = []
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
            body_spec = _resolve_robot_body_spec(name)
            if body_spec is not None:
                body_name, body_idx, body_offset_pos, body_offset_quat = body_spec
                robot_slot_indices.append(slot_idx)
                robot_body_indices.append(body_idx)
                robot_body_names.append(body_name)
                robot_body_offset_positions.append(body_offset_pos)
                robot_body_offset_quaternions.append(body_offset_quat)
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

        self._far_tracking_camera_sensor = FarTrackingCameraSensor(
            self.num_envs,
            sensor_cfg,
            self._terrain_mesh,
            device=self.device,
        )
        # Cache the effective loaded-geometry identity once at setup.  State
        # save/load then remains O(1) and path-independent while direct PPO
        # loads that bypass launcher provenance still reject changed meshes.
        self._far_tracking_geometry_fingerprint = self._fingerprint_far_tracking_geometry(
            ray_cast_bodies,
            asset_meshes_root=mesh_root,
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
            self._far_tracking_camera_sensor.camera_sensor_data_frame_quat = (
                self._shared_camera_sensor_data_frame_quat.to(
                    device=self._far_tracking_camera_sensor.camera_sensor_data_frame_quat.device,
                    dtype=self._far_tracking_camera_sensor.camera_sensor_data_frame_quat.dtype,
                ).clone()
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
        self._far_tracking_robot_body_names = robot_body_names
        self._far_tracking_robot_body_offset_pos = (
            torch.stack(robot_body_offset_positions, dim=0).to(
                device=self.device,
                dtype=torch.float32,
            )
            if robot_body_offset_positions
            else torch.empty((0, 3), device=self.device, dtype=torch.float32)
        )
        self._far_tracking_robot_body_offset_quat = (
            torch.stack(robot_body_offset_quaternions, dim=0).to(
                device=self.device,
                dtype=torch.float32,
            )
            if robot_body_offset_quaternions
            else torch.empty((0, 4), device=self.device, dtype=torch.float32)
        )
        self._far_tracking_object_slot_indices = torch.tensor(
            object_slot_indices, dtype=torch.long, device=self.device
        )
        self._far_tracking_object_source_indices = torch.tensor(
            object_source_indices, dtype=torch.long, device=self.device
        )
        self._far_tracking_object_slot_pairs = tuple(
            zip(object_slot_indices, object_source_indices, strict=True)
        )
        self._far_tracking_primitive_source_indices = torch.tensor(
            primitive_source_indices, dtype=torch.long, device=self.device
        )
        self._far_tracking_object_names = object_names
        self._far_tracking_object_active_env_ids = object_active_env_ids
        self._initialize_far_tracking_object_slots()

    def _shared_wbt_active_object_states(self) -> torch.Tensor | None:
        """Reuse WBT's active-object snapshot when far tracking has the exact same source.

        Multi-object banks need every registered (including parked) actor for
        geometry slots, so they intentionally retain the general backend read.
        """

        if len(self._far_tracking_object_names) != 1:
            return None
        command_manager = getattr(self.env, "command_manager", None)
        get_state = getattr(command_manager, "get_state", None)
        motion_command = get_state("motion_command") if callable(get_state) else None
        motion = getattr(motion_command, "motion", None)
        if motion_command is None or not bool(getattr(motion, "has_object", False)):
            return None
        if bool(getattr(motion_command, "_multi_object_enabled", False)):
            return None
        expected_names = list(getattr(motion_command, "_sim_object_names", ()))
        if expected_names != self._far_tracking_object_names:
            return None
        if getattr(motion_command, "object_name", None) != self._far_tracking_object_names[0]:
            return None
        states = getattr(motion_command, "simulator_object_state_snapshot", None)
        if not isinstance(states, torch.Tensor) or states.shape != (self.num_envs, 13):
            return None
        return states

    def _far_tracking_object_slot_pairs_host(self) -> tuple[tuple[int, int], ...]:
        """Return static object slot/source indices without per-frame CUDA reads."""

        slot_indices = self._far_tracking_object_slot_indices
        source_indices = self._far_tracking_object_source_indices
        if slot_indices is None or source_indices is None:
            return ()
        pairs = getattr(self, "_far_tracking_object_slot_pairs", ())
        if len(pairs) == int(slot_indices.numel()):
            return pairs
        # Compatibility for lightweight/custom managers that populate the
        # historical tensor fields directly.  Cache the conversion after this
        # one-time fallback; production setup starts from Python lists above.
        pairs = tuple(
            zip(
                slot_indices.detach().cpu().tolist(),
                source_indices.detach().cpu().tolist(),
                strict=True,
            )
        )
        self._far_tracking_object_slot_pairs = pairs
        return pairs

    def _compute_far_tracking_camera_depth(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        if self._far_tracking_camera_sensor is None:
            raise RuntimeError("far_tracking_warp camera sensor is not initialized.")
        if self._far_tracking_tf_apply is None or self._far_tracking_quat_mul is None:
            raise RuntimeError("far_tracking_warp camera helpers are not initialized.")
        if (
            self._far_tracking_base_link_indices is None
            or self._far_tracking_robot_slot_indices is None
            or self._far_tracking_robot_body_indices is None
            or self._far_tracking_robot_body_offset_pos is None
            or self._far_tracking_robot_body_offset_quat is None
            or self._far_tracking_object_slot_indices is None
            or self._far_tracking_object_source_indices is None
            or self._far_tracking_primitive_source_indices is None
        ):
            raise RuntimeError("far_tracking_warp camera indices are not initialized.")

        selected_env_ids = env_ids
        selected_rows = slice(None) if selected_env_ids is None else selected_env_ids
        num_selected_envs = self.num_envs if selected_env_ids is None else int(selected_env_ids.numel())

        body_pos = self.env.simulator._rigid_body_pos[selected_rows]
        body_quat = self.env.simulator._rigid_body_rot[selected_rows]

        # Fill robot-body-backed slots.
        if self._far_tracking_robot_slot_indices.numel() > 0:
            ray_cast_body_poses = body_pos[:, self._far_tracking_robot_body_indices]
            ray_cast_body_quats = body_quat[:, self._far_tracking_robot_body_indices]
            body_offset_pos = self._far_tracking_robot_body_offset_pos.to(
                device=ray_cast_body_poses.device,
                dtype=ray_cast_body_poses.dtype,
            ).unsqueeze(0).expand(num_selected_envs, -1, -1)
            body_offset_quat = self._far_tracking_robot_body_offset_quat.to(
                device=ray_cast_body_quats.device,
                dtype=ray_cast_body_quats.dtype,
            ).unsqueeze(0).expand(num_selected_envs, -1, -1)
            ray_cast_body_poses, ray_cast_body_quats = self._compose_fixed_body_pose(
                ray_cast_body_poses,
                ray_cast_body_quats,
                body_offset_pos,
                body_offset_quat,
            )
            if selected_env_ids is None:
                self._far_tracking_camera_sensor.ray_cast_body_poses_tensor[
                    :, self._far_tracking_robot_slot_indices
                ] = ray_cast_body_poses
                self._far_tracking_camera_sensor.ray_cast_body_quats_tensor[
                    :, self._far_tracking_robot_slot_indices
                ] = ray_cast_body_quats
            else:
                # Two advanced indices are pairwise by default.  Broadcast them
                # explicitly so every selected environment updates every robot
                # mesh slot while survivor rows remain byte-for-byte untouched.
                selected_grid = selected_env_ids.unsqueeze(1)
                robot_slot_grid = self._far_tracking_robot_slot_indices.unsqueeze(0)
                self._far_tracking_camera_sensor.ray_cast_body_poses_tensor[
                    selected_grid, robot_slot_grid
                ] = ray_cast_body_poses
                self._far_tracking_camera_sensor.ray_cast_body_quats_tensor[
                    selected_grid, robot_slot_grid
                ] = ray_cast_body_quats

        # Fill registered object slots from simulator actor states.
        if self._far_tracking_object_names and num_selected_envs > 0:
            shared_object_states = self._shared_wbt_active_object_states()
            if shared_object_states is not None:
                object_states = shared_object_states[selected_rows].unsqueeze(1)
            else:
                actor_env_ids = (
                    torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                    if selected_env_ids is None
                    else selected_env_ids
                )
                object_states = self.env.simulator.get_actor_states(
                    self._far_tracking_object_names,
                    actor_env_ids,
                )
                object_states = object_states.view(
                    len(self._far_tracking_object_names),
                    num_selected_envs,
                    -1,
                ).permute(1, 0, 2)
            if self._far_tracking_primitive_source_indices.numel() > 0:
                primitive_poses = object_states[:, self._far_tracking_primitive_source_indices, :3]
                primitive_quats = object_states[:, self._far_tracking_primitive_source_indices, 3:7]
                self._far_tracking_camera_sensor.primitive_body_poses_tensor[selected_rows] = primitive_poses
                self._far_tracking_camera_sensor.primitive_body_quats_tensor[selected_rows] = primitive_quats
            if self._far_tracking_object_slot_indices.numel() == 0:
                object_states = None
        else:
            object_states = None

        if object_states is not None and self._far_tracking_object_slot_indices.numel() > 0:
            for local_slot_idx, (slot_tensor_idx, source_idx) in enumerate(
                self._far_tracking_object_slot_pairs_host()
            ):
                active_env_ids = self._far_tracking_object_active_env_ids[local_slot_idx]
                if active_env_ids is None:
                    self._far_tracking_camera_sensor.ray_cast_body_poses_tensor[selected_rows, slot_tensor_idx] = (
                        object_states[:, source_idx, :3]
                    )
                    self._far_tracking_camera_sensor.ray_cast_body_quats_tensor[selected_rows, slot_tensor_idx] = (
                        object_states[:, source_idx, 3:7]
                    )
                    continue
                if active_env_ids.numel() == 0:
                    continue
                if selected_env_ids is None:
                    target_env_ids = active_env_ids
                    source_rows = active_env_ids
                else:
                    # Topology env IDs are static and may be non-contiguous.
                    # Select membership in compact-row space, then write using
                    # the corresponding global simulator rows.  Inactive slots
                    # stay parked exactly as initialized.
                    active_selected_rows = torch.isin(selected_env_ids, active_env_ids)
                    target_env_ids = selected_env_ids[active_selected_rows]
                    source_rows = active_selected_rows
                self._far_tracking_camera_sensor.ray_cast_body_poses_tensor[target_env_ids, slot_tensor_idx] = (
                    object_states[source_rows, source_idx, :3]
                )
                self._far_tracking_camera_sensor.ray_cast_body_quats_tensor[target_env_ids, slot_tensor_idx] = (
                    object_states[source_rows, source_idx, 3:7]
                )

        updated_camera_pos, updated_camera_quat = self._get_strict_warp_camera_pose(selected_env_ids)
        updated_camera_pos = updated_camera_pos.unsqueeze(1)
        updated_camera_quat = updated_camera_quat.unsqueeze(1)
        self._far_tracking_camera_sensor.camera_sensor_position[selected_rows] = updated_camera_pos
        self._far_tracking_camera_sensor.camera_sensor_orientation[selected_rows] = updated_camera_quat
        collect_debug_state = bool(self._debug_dump_dir) and not self._debug_dump_done
        if collect_debug_state:
            # Base-link poses are diagnostic-only.  Keep this full snapshot out
            # of the normal hot path; debug consumers index it by global env ID.
            camera_base_link_pos, camera_base_link_quat = self._get_camera_body_pose(slice(None))
            camera_base_link_pos = camera_base_link_pos.unsqueeze(1)
            camera_base_link_quat = camera_base_link_quat.unsqueeze(1)
            self._far_tracking_debug_last = {
                "base_link_indices": self._far_tracking_base_link_indices.detach()
                .to(torch.float32)
                .view(1, -1)
                .expand(self.num_envs, -1),
                "base_link_pos": camera_base_link_pos.detach().clone().view(self.num_envs, -1),
                "base_link_quat_xyzw": camera_base_link_quat.detach().clone().view(self.num_envs, -1),
                "updated_camera_pos": self._far_tracking_camera_sensor.camera_sensor_position.detach()
                .clone()
                .view(self.num_envs, -1),
                "updated_camera_quat_xyzw": self._far_tracking_camera_sensor.camera_sensor_orientation.detach()
                .clone()
                .view(self.num_envs, -1),
                "sensor_camera_pos_before_capture": self._far_tracking_camera_sensor.camera_sensor_position.detach()
                .clone()
                .view(self.num_envs, -1),
                "sensor_camera_quat_before_capture": self._far_tracking_camera_sensor.camera_sensor_orientation.detach()
                .clone()
                .view(self.num_envs, -1),
            }

        depth = self._far_tracking_camera_sensor.capture(active_env_ids=env_ids)
        if collect_debug_state:
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
        if env_ids is not None:
            depth = depth[env_ids]
        return self._clamp_camera_depth_to_sensor_range(depth)

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
        for local_slot_idx, (slot_tensor_idx, _) in enumerate(
            self._far_tracking_object_slot_pairs_host()
        ):
            active_env_ids = self._far_tracking_object_active_env_ids[local_slot_idx]
            if active_env_ids is None:
                continue
            self._far_tracking_camera_sensor.ray_cast_body_poses_tensor[:, slot_tensor_idx] = inactive_pos
            self._far_tracking_camera_sensor.ray_cast_body_quats_tensor[:, slot_tensor_idx] = inactive_quat

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
                if self._strict_perception_object_meshes():
                    raise RuntimeError(
                        f"Registered simulator object '{name}' has no valid perception raycast mesh."
                    )
                (self.logger or logger).warning(
                    "Skipping registered object '{}' in perception raycast: unable to resolve mesh path.",
                    name,
                )
        return mesh_map

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
                if self._strict_perception_object_meshes():
                    raise RuntimeError(
                        "Failed to resolve a valid perception raycast mesh for registered object "
                        f"'{object_name}' from asset '{candidate_path}'."
                    )
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

    @staticmethod
    def _is_valid_trimesh(mesh: Any, trimesh_module: Any) -> bool:
        return (
            isinstance(mesh, trimesh_module.Trimesh)
            and getattr(mesh, "vertices", np.empty((0,))).size > 0
            and getattr(mesh, "faces", np.empty((0,))).size > 0
        )

    @classmethod
    def _coerce_loaded_trimesh(cls, loaded: Any, trimesh_module: Any, *, source_path: str) -> Any:
        if cls._is_valid_trimesh(loaded, trimesh_module):
            return loaded

        def concatenate(meshes: list[Any]) -> Any:
            valid_meshes = [mesh for mesh in meshes if cls._is_valid_trimesh(mesh, trimesh_module)]
            if not valid_meshes:
                raise ValueError(f"no non-empty mesh geometry found in {source_path}")
            return trimesh_module.util.concatenate(valid_meshes)

        if isinstance(loaded, trimesh_module.Scene):
            dumped = loaded.dump(concatenate=True)
            if cls._is_valid_trimesh(dumped, trimesh_module):
                return dumped
            if isinstance(dumped, (list, tuple)):
                return concatenate(list(dumped))

        if isinstance(loaded, (list, tuple)):
            return concatenate(list(loaded))

        raise ValueError(f"loaded geometry is not a non-empty mesh: {source_path} ({type(loaded).__name__})")

    @classmethod
    def _load_trimesh_file(cls, mesh_path: str | Path, trimesh_module: Any) -> Any:
        mesh_path_str = str(mesh_path)
        loaded = trimesh_module.load(mesh_path_str, process=False)
        return cls._coerce_loaded_trimesh(loaded, trimesh_module, source_path=mesh_path_str)

    @staticmethod
    def _strict_perception_object_meshes() -> bool:
        explicit = os.environ.get("HOLOSOMA_STRICT_PERCEPTION_OBJECT_MESHES", "").strip().lower()
        require_single_slot = os.environ.get("HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS", "").strip().lower()
        return explicit in {"1", "true", "yes", "on"} or require_single_slot in {"1", "true", "yes", "on"}

    @classmethod
    def _perception_mesh_cache_dir(cls) -> Path:
        base_dir = os.environ.get("HOLOSOMA_PERCEPTION_MESH_CACHE_DIR", "/tmp/holosoma_perception_mesh_cache")
        return Path(base_dir).expanduser() / cls._PERCEPTION_MESH_CACHE_VERSION

    def _export_trimesh_atomic(self, mesh: Any, cache_mesh_path: Path, trimesh_module: Any) -> str:
        """Export a mesh cache atomically so other ranks never read a partial OBJ."""
        tmp_path = cache_mesh_path.with_name(
            f".{cache_mesh_path.stem}.{os.getpid()}.tmp{cache_mesh_path.suffix}"
        )
        try:
            mesh.export(str(tmp_path))
            self._load_trimesh_file(tmp_path, trimesh_module)
            os.replace(tmp_path, cache_mesh_path)
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            raise
        return str(cache_mesh_path)

    def _export_combined_urdf_visual_mesh(self, urdf_path: str, object_name: str) -> str | None:
        """Build a single OBJ mesh from URDF visual geometry for dynamic object raycasting."""
        strict = self._strict_perception_object_meshes()
        try:
            import trimesh  # noqa: PLC0415
        except Exception:
            if strict:
                raise
            return None

        urdf_file = Path(urdf_path).expanduser()
        if not urdf_file.exists():
            if strict:
                raise FileNotFoundError(f"Object URDF does not exist for perception mesh export: {urdf_file}")
            return None

        try:
            root = ET.parse(str(urdf_file)).getroot()
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to parse object URDF for perception mesh export: {urdf_file}") from exc
            return None

        urdf_dir = str(urdf_file.parent)
        digest = hashlib.sha1()
        try:
            digest.update(str(urdf_file.resolve()).encode("utf-8"))
            digest.update(urdf_file.read_bytes())
            for mesh_tag in root.findall(".//link/visual/geometry/mesh"):
                filename = mesh_tag.get("filename")
                if not filename:
                    continue
                mesh_file = Path(self._resolve_urdf_mesh_path(urdf_dir, urdf_dir, filename))
                if mesh_file.exists():
                    digest.update(str(mesh_file.resolve()).encode("utf-8"))
                    digest.update(mesh_file.read_bytes())
        except Exception:
            stat = urdf_file.stat()
            digest.update(f"{urdf_file.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))

        cache_dir = self._perception_mesh_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_mesh_path = cache_dir / f"{object_name}_{urdf_file.stem}_{digest.hexdigest()[:12]}_combined.obj"
        if cache_mesh_path.exists():
            try:
                self._load_trimesh_file(cache_mesh_path, trimesh)
                return str(cache_mesh_path)
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"Cached perception mesh is invalid for "
                        f"{self._PERCEPTION_MESH_CACHE_VERSION}: {cache_mesh_path}."
                    ) from exc
                return None

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
                        if strict:
                            raise FileNotFoundError(
                                f"URDF visual mesh file does not exist: {mesh_file} (from {urdf_file})"
                            )
                        continue
                    try:
                        mesh = self._load_trimesh_file(mesh_file, trimesh)
                    except Exception as exc:
                        if strict:
                            raise RuntimeError(
                                f"Failed to load URDF visual mesh for perception: {mesh_file} (from {urdf_file})"
                            ) from exc
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
                    if strict:
                        raise RuntimeError(f"URDF visual geometry did not produce a Trimesh: {urdf_file}")
                    continue
                if not self._is_valid_trimesh(mesh, trimesh):
                    if strict:
                        raise RuntimeError(f"URDF visual geometry produced an empty mesh: {urdf_file}")
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
            if strict:
                raise RuntimeError(f"Object URDF produced no valid visual meshes for perception: {urdf_file}")
            return None

        try:
            combined = trimesh.util.concatenate(meshes)
            if not self._is_valid_trimesh(combined, trimesh):
                if strict:
                    raise RuntimeError(f"Combined perception mesh is empty for object URDF: {urdf_file}")
                return None
            self._export_trimesh_atomic(combined, cache_mesh_path, trimesh)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to export combined perception mesh for object URDF: {urdf_file}") from exc
            return None
        return str(cache_mesh_path)

    def _export_combined_mesh_from_scene_asset(self, asset_path: str, object_name: str) -> str | None:
        """Best-effort mesh export for USD scene assets."""
        strict = self._strict_perception_object_meshes()
        try:
            import trimesh  # noqa: PLC0415
        except Exception:
            if strict:
                raise
            return None

        source_path = Path(asset_path).expanduser()
        if not source_path.exists():
            if strict:
                raise FileNotFoundError(f"Scene asset does not exist for perception mesh export: {source_path}")
            return None

        digest = hashlib.sha1()
        try:
            digest.update(str(source_path.resolve()).encode("utf-8"))
            digest.update(source_path.read_bytes())
        except Exception:
            stat = source_path.stat()
            digest.update(f"{source_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))

        cache_dir = self._perception_mesh_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_mesh_path = cache_dir / f"{object_name}_{source_path.stem}_{digest.hexdigest()[:12]}_combined.obj"
        if cache_mesh_path.exists():
            try:
                self._load_trimesh_file(cache_mesh_path, trimesh)
                return str(cache_mesh_path)
            except Exception as exc:
                if strict:
                    raise RuntimeError(
                        f"Cached perception mesh is invalid for "
                        f"{self._PERCEPTION_MESH_CACHE_VERSION}: {cache_mesh_path}."
                    ) from exc
                return None

        try:
            mesh = self._load_trimesh_file(str(source_path), trimesh)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to load scene asset mesh for perception: {source_path}") from exc
            return None

        if not self._is_valid_trimesh(mesh, trimesh):
            if strict:
                raise RuntimeError(f"Scene asset mesh is empty for perception: {source_path}")
            return None

        try:
            self._export_trimesh_atomic(mesh, cache_mesh_path, trimesh)
        except Exception as exc:
            if strict:
                raise RuntimeError(f"Failed to export scene asset perception mesh: {source_path}") from exc
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

    def _camera_ray_rotation_quat(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        correction_quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pitch_deg = float(self.cfg.camera_pitch_deg)
        pitch_rad = torch.deg2rad(torch.tensor(pitch_deg, device=device, dtype=dtype))
        pitch_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=device, dtype=dtype),
            pitch_rad,
            torch.tensor(0.0, device=device, dtype=dtype),
        )
        if correction_quat is None:
            correction_quat = self._camera_ray_correction_quat
        correction_quat = correction_quat.to(device=device, dtype=dtype)
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

    def _build_camera_rays_from_coords(
        self,
        u_coords: torch.Tensor,
        v_coords: torch.Tensor,
        *,
        correction_quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        v_grid, u_grid = torch.meshgrid(v_coords, u_coords, indexing="ij")
        x = (u_grid - self._camera_cx) / self._camera_fx
        y = (v_grid - self._camera_cy) / self._camera_fy
        dirs_cam = self._camera_dirs_cam_from_xy(x, y).view(-1, 3)
        combo = self._camera_ray_rotation_quat(
            device=self.device,
            dtype=torch.float32,
            correction_quat=correction_quat,
        )
        combo = combo.unsqueeze(0).expand(dirs_cam.shape[0], -1)
        dirs_base = quat_apply(combo, dirs_cam, w_last=True)
        return dirs_base / torch.norm(dirs_base, dim=-1, keepdim=True).clamp(min=1.0e-6)

    def _build_camera_rays(
        self,
        *,
        correction_quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        u_coords = torch.arange(self._camera_width, device=self.device, dtype=torch.float32)
        v_coords = torch.arange(self._camera_height, device=self.device, dtype=torch.float32)
        return self._build_camera_rays_from_coords(
            u_coords,
            v_coords,
            correction_quat=correction_quat,
        )

    def _build_camera_scandots_rays(
        self,
        *,
        correction_quat: torch.Tensor | None = None,
    ) -> torch.Tensor:
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

        return self._build_camera_rays_from_coords(
            u_coords,
            v_coords,
            correction_quat=correction_quat,
        )

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

    def _apply_camera_depth_noise(
        self,
        depth: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
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
                if env_ids is not None and std.ndim > 0:
                    std = std[env_ids.to(device=std.device, dtype=torch.long)]
                if std.ndim == 1:
                    std = std.view(-1, 1, 1)
            else:
                std = torch.tensor(float(std_mult), device=depth.device)
            depth_out = depth_out + torch.randn_like(depth_out) * (depth_out * std)

        if drop_prob is not None:
            if isinstance(drop_prob, torch.Tensor):
                prob = drop_prob.to(depth.device)
                if env_ids is not None and prob.ndim > 0:
                    prob = prob[env_ids.to(device=prob.device, dtype=torch.long)]
                if prob.ndim == 1:
                    prob = prob.view(-1, 1, 1)
            else:
                prob = torch.tensor(float(drop_prob), device=depth.device)
            mask = torch.rand_like(depth_out) < prob
            depth_out = torch.where(mask, torch.full_like(depth_out, self.cfg.max_distance), depth_out)

        return self._clamp_camera_depth_to_sensor_range(depth_out)

    def _prepare_camera_depth_for_observation(
        self,
        depth: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply sensor noise exactly once before observation preprocessing."""

        depth = self._apply_camera_depth_noise(depth, env_ids=env_ids)
        return self._clamp_camera_depth_to_sensor_range(depth)

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

    def _camera_obs_refresh_flag_for_update(
        self,
        env_ids: torch.Tensor | None,
    ) -> bool:
        if env_ids is not None:
            # A targeted refresh initializes only reset environments.  It is not
            # a second control-frame tick and must not shift the global camera
            # frequency phase used by every non-reset environment.
            return True
        return self._consume_camera_obs_refresh_flag()

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
        advance_temporal_noise: bool = True,
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
            processed = self._process_camera_depth_for_obs(
                depth_obs,
                env_ids=env_ids,
                advance_temporal_noise=advance_temporal_noise,
            )
            if self._camera_warp_buffer_len > 1:
                self._camera_depth_buffer[env_ids, :-1] = self._camera_depth_buffer[env_ids, 1:].clone()
            self._camera_depth_buffer[env_ids, -1] = processed

            if (~ready).any():
                new_env_ids = env_ids[~ready]
                repeated = processed[~ready].unsqueeze(1).repeat(1, self._camera_warp_buffer_len, 1, 1)
                self._camera_depth_buffer[new_env_ids] = repeated
            self._camera_depth_buffer_ready[env_ids] = True

        if self._camera_warp_latency_frame_range is not None:
            latency_min, latency_max = self._camera_warp_latency_frame_range
            current_latency = torch.randint(
                latency_min,
                latency_max + 1,
                (env_ids.numel(),),
                device=self.device,
                dtype=torch.long,
            )
            buffer_indices = self._camera_warp_buffer_len - 1 - current_latency
            self._camera_depth_obs[env_ids] = self._camera_depth_buffer[env_ids, buffer_indices]
        else:
            self._camera_depth_obs[env_ids] = self._camera_depth_buffer[env_ids, -1 - self._camera_warp_latency_frame]

    def _process_camera_depth_for_obs(
        self,
        depth: torch.Tensor,
        *,
        env_ids: torch.Tensor | None = None,
        advance_temporal_noise: bool = True,
    ) -> torch.Tensor:
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
            depth_obs = self._apply_warp_hole_noise(
                depth_obs,
                max_depth=max_depth,
                env_ids=env_ids,
                advance_frame=advance_temporal_noise,
            )

        if self._camera_warp_additive_noise_std > 0.0:
            depth_obs = depth_obs + torch.randn_like(depth_obs) * self._camera_warp_additive_noise_std

        if self._camera_warp_depth_offset_std > 0.0:
            offset = self._camera_warp_depth_offset
            if env_ids is not None:
                offset = offset[env_ids]
            else:
                offset = offset[: depth_obs.shape[0]]
            depth_obs = depth_obs + offset.to(
                device=depth_obs.device,
                dtype=depth_obs.dtype,
            ).view(-1, 1, 1)

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

    def _apply_warp_hole_noise(
        self,
        depth: torch.Tensor,
        *,
        max_depth: float,
        env_ids: torch.Tensor | None = None,
        advance_frame: bool = True,
    ) -> torch.Tensor:
        if self._camera_warp_hole_generator is None:
            return depth
        num_envs, height, width = depth.shape

        def preprocess(raw_frame: torch.Tensor) -> torch.Tensor:
            processed = raw_frame.to(device=depth.device, dtype=depth.dtype)
            processed = F.interpolate(
                processed.unsqueeze(1),
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            return F.max_pool2d(processed, kernel_size=3, stride=1, padding=1)

        if advance_frame:
            frame_index = int(self._camera_warp_hole_generator.frame_idx)
            # Generate and preprocess the complete authenticated reference
            # batch before selecting live environments.  Interpolation and
            # pooling are batch-independent, while the subsequent extrema are
            # deliberately not; slicing earlier changes the trained mask.
            full_frame = preprocess(self._camera_warp_hole_generator.generate_frame())
            if env_ids is None:
                frame = full_frame[:num_envs]
            else:
                frame = full_frame[env_ids]
        else:
            # Reset refreshes occur after the normal full sensor refresh in the
            # same control step.  Reuse that temporal hole field for the reset
            # subset; advancing the shared Perlin clock here would accelerate
            # every surviving environment's sensor process whenever any peer
            # terminates.
            latest_full_frame_index = max(
                0,
                int(self._camera_warp_hole_generator.frame_idx) - 1,
            )
            frame_index = latest_full_frame_index
            frame = preprocess(
                self._camera_warp_hole_generator.generate_frame(
                    frame_index=latest_full_frame_index,
                    env_ids=env_ids,
                )
            )
            if env_ids is None:
                frame = frame[:num_envs]
        if advance_frame:
            # Preserve the reference far-tracking preprocessing contract: its
            # Perlin field is normalized over the complete vectorized batch.
            frame_min = full_frame.min()
            frame_max = full_frame.max()
            self._camera_warp_hole_frame_stats = (
                frame_index,
                frame_min.detach().clone(),
                frame_max.detach().clone(),
            )
        else:
            # A targeted post-reset refresh must reinterpret the selected raw
            # frame with the *same* batch extrema used by the preceding full
            # sensor tick.  Using subset extrema changes the mask, while
            # generating a new full tick advances the temporal process.  The
            # normal path hits this tiny derived cache; the fallback rebuilds
            # its deterministic statistics without advancing ``frame_idx``.
            cached_stats = getattr(self, "_camera_warp_hole_frame_stats", None)
            if cached_stats is None or int(cached_stats[0]) != frame_index:
                full_frame = preprocess(
                    self._camera_warp_hole_generator.generate_frame(
                        frame_index=frame_index,
                    )
                )
                cached_stats = (
                    frame_index,
                    full_frame.min().detach().clone(),
                    full_frame.max().detach().clone(),
                )
                self._camera_warp_hole_frame_stats = cached_stats
            frame_min = cached_stats[1].to(device=frame.device, dtype=frame.dtype)
            frame_max = cached_stats[2].to(device=frame.device, dtype=frame.dtype)
        frame = (frame - frame_min) / (frame_max - frame_min).clamp(min=1.0e-6)
        holes = (frame.squeeze(1) < self._camera_warp_hole_prob) & (depth < 2.0) & (depth > 0.2)
        if holes.any():
            depth = torch.where(holes, torch.full_like(depth, max_depth), depth)
        return depth

    def _normalize_camera_depth_for_obs(self, depth: torch.Tensor, *, max_depth: float) -> torch.Tensor:
        near = float(getattr(self.cfg, "camera_near", 0.1) or 0.1)
        denom = max(1.0e-6, max_depth - near)
        return (depth - near) / denom - 0.5

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
        if simulator_type == SimulatorType.ISAACSIM and self._camera_strict_warp:
            if self._camera_source == "rendered_depth_sensor":
                raise RuntimeError(
                    "Strict far-warp camera parity is unsupported for rendered_depth_sensor: "
                    "the selected depth camera can be a transformed child of the sensor asset, "
                    "so applying the manager optical pose to the asset root is not well-defined."
                )
            # Use the manager's exact mount chain so rendered validation sees
            # the same strict D435 pose and reset-sampled offsets as the
            # far_tracking_warp policy input.
            camera_kwargs["pose_provider"] = self._get_strict_warp_camera_pose
        elif (
            simulator_type == SimulatorType.ISAACSIM
            and self._has_camera_reset_randomization()
        ):
            raise RuntimeError(
                "Non-strict Isaac rendered cameras do not consume the manager's reset-sampled "
                "camera pose offsets. Disable that randomization or use strict rendered/far_tracking_warp."
            )
        elif simulator_type == SimulatorType.MUJOCO:
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

                try:
                    mesh = self._load_trimesh_file(mesh_path, trimesh)
                except Exception:
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
                    try:
                        mesh = self._load_trimesh_file(mesh_path, trimesh)
                    except Exception:
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

        local_position = self._sensor_offset.to(
            device=body_pos.device,
            dtype=body_pos.dtype,
        ).unsqueeze(0).expand(num_envs, -1)
        local_orientation = torch.zeros(
            (num_envs, 4),
            device=body_quat.device,
            dtype=body_quat.dtype,
        )
        local_orientation[:, 3] = 1.0
        local_position, local_orientation = self._apply_runtime_camera_mount_offsets(
            local_position,
            local_orientation,
            idx=idx,
        )
        offset_world = quat_apply(body_quat, local_position, w_last=True)
        camera_pos = body_pos + offset_world
        body_quat = quat_mul(body_quat, local_orientation, w_last=True)

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
        return depth

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
        return depth

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
