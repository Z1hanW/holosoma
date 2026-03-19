"""MuJoCo rendered depth camera helper for perception observations."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger

import mujoco
from holosoma.simulator.mujoco.scene_manager import HOLOSOMA_PERCEPTION_CAMERA_NAME
from holosoma.utils.safe_torch_import import torch

_DEPTH_HIDDEN_GEOM_GROUP = 5


class MuJoCoDepthCamera:
    """Rendered depth camera backed by MuJoCo offscreen rendering."""

    def __init__(
        self,
        *,
        env: Any,
        config: Any,
        width: int,
        height: int,
        vfov_deg: float,
        device: str,
        pose_provider: Callable[[torch.Tensor | None], tuple[torch.Tensor, torch.Tensor]] | None = None,
        intrinsics: tuple[float, float, float, float] | None = None,
    ) -> None:
        self._env = env
        self._cfg = config
        self._width = int(width)
        self._height = int(height)
        self._vfov_deg = float(vfov_deg)
        self._device = device
        self._env_id = int(getattr(config, "camera_env_id", 0))
        self._pose_provider = pose_provider
        self._intrinsics = intrinsics

        self._renderer: mujoco.Renderer | None = None
        self._camera_name = HOLOSOMA_PERCEPTION_CAMERA_NAME
        self._camera_id: int | None = None
        self._scene_option: mujoco.MjvOption | None = None
        self._masked_robot_geom_ids: list[int] = []
        self._warned_multi_env = False
        self._warned_invalid_depth = False
        self._warned_invalid_rgb = False
        self._debug_dump_dir = os.environ.get("HOLOSOMA_MUJOCO_DEPTH_DEBUG_DUMP_DIR", "").strip()
        try:
            self._debug_dump_after_captures = max(1, int(os.environ.get("HOLOSOMA_MUJOCO_DEPTH_DEBUG_DUMP_AFTER_CAPTURES", "1")))
        except Exception:
            self._debug_dump_after_captures = 1
        try:
            self._debug_dump_render_width = max(1, int(os.environ.get("HOLOSOMA_MUJOCO_DEPTH_DEBUG_RENDER_WIDTH", "212")))
            self._debug_dump_render_height = max(1, int(os.environ.get("HOLOSOMA_MUJOCO_DEPTH_DEBUG_RENDER_HEIGHT", "120")))
        except Exception:
            self._debug_dump_render_width = 212
            self._debug_dump_render_height = 120
        self._capture_counter = 0
        self._debug_dump_done = False

    def setup(self) -> None:
        if self._pose_provider is None:
            raise RuntimeError("MuJoCoDepthCamera requires a pose_provider.")
        if self._intrinsics is None:
            raise RuntimeError("MuJoCoDepthCamera requires camera intrinsics.")
        if self._env_id < 0 or self._env_id >= self._env.num_envs:
            raise RuntimeError(f"camera_env_id out of range: {self._env_id} (num_envs={self._env.num_envs})")
        if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
            os.environ["MUJOCO_GL"] = "egl"

        model = self._env.simulator.root_model
        if model is None:
            raise RuntimeError("MuJoCo simulator model is not initialized.")
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera_name)
        if camera_id < 0:
            raise RuntimeError(f"Perception camera '{self._camera_name}' not found in MuJoCo model.")
        self._camera_id = int(camera_id)
        self._configure_intrinsics()
        self._configure_scene_mask()
        self._renderer = mujoco.Renderer(model, height=self._height, width=self._width)
        logger.info(
            "MuJoCo rendered perception camera ready: name={} env_id={} size={}x{}",
            self._camera_name,
            self._env_id,
            self._width,
            self._height,
        )

    def capture_depth(self) -> torch.Tensor:
        self._capture_counter += 1
        renderer, render_data = self._prepare_renderer(depth=True)
        depth = renderer.render()
        depth_array = self._sanitize_depth_array(depth)
        # MuJoCo/OpenGL render buffers are bottom-up; align to the top-down pixel order
        # used by the legacy far_tracking_warp pipeline and policy preprocessing.
        depth_array = np.flipud(depth_array).copy()
        self._maybe_dump_debug(render_data, depth=depth_array)
        return torch.as_tensor(depth_array, device=self._device, dtype=torch.float32).unsqueeze(0)

    def capture_rgb(self) -> np.ndarray:
        renderer, _render_data = self._prepare_renderer(depth=False)
        rgb = renderer.render()
        rgb_array = self._sanitize_rgb_array(rgb)
        rgb_array = np.flipud(rgb_array).copy()
        self._maybe_dump_debug(None, rgb=rgb_array)
        return rgb_array

    def _prepare_renderer(self, *, depth: bool) -> tuple[mujoco.Renderer, mujoco.MjData]:
        if self._renderer is None or self._camera_id is None:
            raise RuntimeError("MuJoCoDepthCamera.setup() must be called before capture.")

        if self._env.num_envs > 1 and not self._warned_multi_env:
            self._warned_multi_env = True
            raise RuntimeError(
                f"MuJoCo rendered camera only supports one environment; got num_envs={self._env.num_envs}. "
                "Use num_envs=1 for sim_joystick MuJoCo depth validation."
            )

        model = self._env.simulator.root_model
        if model is None:
            raise RuntimeError("MuJoCo simulator model is not initialized.")
        render_data = self._env.simulator.backend.get_render_data(world_id=self._env_id)
        self._update_camera_pose(render_data)

        if depth:
            self._renderer.enable_depth_rendering()
        else:
            self._renderer.disable_depth_rendering()
        self._renderer.update_scene(
            render_data,
            camera=self._camera_name,
            scene_option=self._scene_option,
        )
        return self._renderer, render_data

    def _configure_intrinsics(self) -> None:
        model = self._env.simulator.root_model
        if model is None or self._camera_id is None:
            raise RuntimeError("MuJoCo simulator model/camera is not initialized.")

        cam_id = self._camera_id
        fx, fy, cx, cy = self._intrinsics

        model.cam_intrinsic[cam_id, :] = np.array([fx, fy, cx, cy], dtype=np.float64)
        model.cam_resolution[cam_id, :] = np.array([self._width, self._height], dtype=np.int32)
        model.cam_sensorsize[cam_id, :] = np.array([float(self._width), float(self._height)], dtype=np.float64)
        model.cam_fovy[cam_id] = self._vfov_deg

    def _update_camera_pose(self, render_data: mujoco.MjData) -> None:
        model = self._env.simulator.root_model
        if model is None or self._camera_id is None:
            raise RuntimeError("MuJoCo simulator model/camera is not initialized.")

        env_ids = torch.tensor([self._env_id], device=self._device, dtype=torch.long)
        camera_pos, camera_quat_xyzw = self._pose_provider(env_ids)
        camera_pos_np = camera_pos[0].detach().cpu().numpy().astype(np.float64)
        camera_quat_wxyz_np = camera_quat_xyzw[0, [3, 0, 1, 2]].detach().cpu().numpy().astype(np.float64)

        model.cam_pos[self._camera_id, :] = camera_pos_np
        model.cam_quat[self._camera_id, :] = camera_quat_wxyz_np
        mujoco.mj_forward(model, render_data)

    def _configure_scene_mask(self) -> None:
        model = self._env.simulator.root_model
        if model is None:
            raise RuntimeError("MuJoCo simulator model is not initialized.")

        self._scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self._scene_option)
        self._scene_option.geomgroup[_DEPTH_HIDDEN_GEOM_GROUP] = 0

        allowlist = set(getattr(self._cfg, "camera_mesh_allowlist", None) or [])
        prefix = getattr(getattr(self._env.simulator, "scene_manager", None), "robot_prefix", "robot_")
        if not allowlist:
            return

        masked_geom_ids: list[int] = []
        for geom_id in range(model.ngeom):
            body_id = int(model.geom_bodyid[geom_id])
            body_name = str(model.body(body_id).name or "")
            if not body_name.startswith(prefix):
                continue
            clean_body_name = body_name[len(prefix) :]
            if clean_body_name in allowlist:
                continue
            model.geom_group[geom_id] = _DEPTH_HIDDEN_GEOM_GROUP
            masked_geom_ids.append(int(geom_id))

        self._masked_robot_geom_ids = masked_geom_ids
        if masked_geom_ids:
            logger.info(
                "MuJoCo rendered depth masked {} robot geom(s) outside camera_mesh_allowlist.",
                len(masked_geom_ids),
            )

    def _sanitize_depth_array(self, depth_array: np.ndarray) -> np.ndarray:
        if depth_array.size == 0:
            if not self._warned_invalid_depth:
                logger.warning("MuJoCo rendered depth returned empty data; filling with max_distance.")
                self._warned_invalid_depth = True
            return np.full((self._height, self._width), float(self._cfg.max_distance), dtype=np.float32)

        if depth_array.ndim == 3:
            depth_array = depth_array[:, :, 0]
        if depth_array.shape != (self._height, self._width):
            if not self._warned_invalid_depth:
                logger.warning(
                    "Unexpected MuJoCo depth shape {}; expected ({}, {}). Filling with max_distance.",
                    depth_array.shape,
                    self._height,
                    self._width,
                )
                self._warned_invalid_depth = True
            return np.full((self._height, self._width), float(self._cfg.max_distance), dtype=np.float32)

        depth_array = depth_array.astype(np.float32, copy=False)
        invalid = ~np.isfinite(depth_array) | (depth_array <= 0.0)
        if np.any(invalid):
            depth_array = depth_array.copy()
            depth_array[invalid] = float(self._cfg.max_distance)
        return np.clip(depth_array, 0.0, float(self._cfg.max_distance))

    def _sanitize_rgb_array(self, rgb_array: np.ndarray) -> np.ndarray:
        if rgb_array.size == 0:
            if not self._warned_invalid_rgb:
                logger.warning("MuJoCo rendered RGB returned empty data; filling with zeros.")
                self._warned_invalid_rgb = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        if rgb_array.ndim == 3 and rgb_array.shape[-1] >= 3:
            rgb_array = rgb_array[:, :, :3]
        else:
            if not self._warned_invalid_rgb:
                logger.warning(
                    "Unexpected MuJoCo RGB shape {}; expected ({}, {}, 3). Filling with zeros.",
                    rgb_array.shape,
                    self._height,
                    self._width,
                )
                self._warned_invalid_rgb = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        if rgb_array.shape[:2] != (self._height, self._width):
            if not self._warned_invalid_rgb:
                logger.warning(
                    "Unexpected MuJoCo RGB shape {}; expected ({}, {}, 3). Filling with zeros.",
                    rgb_array.shape,
                    self._height,
                    self._width,
                )
                self._warned_invalid_rgb = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        return rgb_array.astype(np.uint8, copy=False)

    def _maybe_dump_debug(
        self,
        render_data: mujoco.MjData | None,
        *,
        depth: np.ndarray | None = None,
        rgb: np.ndarray | None = None,
    ) -> None:
        if not self._debug_dump_dir or self._debug_dump_done:
            return
        if self._capture_counter < self._debug_dump_after_captures:
            return
        if depth is None and rgb is None:
            return

        dump_dir = Path(self._debug_dump_dir).expanduser().resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)

        if depth is not None:
            np.save(dump_dir / "mujoco_depth_raw.npy", depth)
            depth_near = float(getattr(self._cfg, "camera_near", 0.0) or 0.0)
            depth_far = float(min(float(getattr(self._cfg, "camera_far", 3.0) or 3.0), float(self._cfg.max_distance)))
            denom = max(depth_far - depth_near, 1.0e-6)
            depth_vis = np.clip((depth - depth_near) / denom, 0.0, 1.0)
            depth_vis = ((1.0 - depth_vis) * 255.0).astype(np.uint8)
            depth_vis_rgb = np.repeat(depth_vis[..., None], 3, axis=-1)
            self._save_image(dump_dir / "mujoco_depth_vis.png", depth_vis_rgb)
            self._save_image(dump_dir / "mujoco_depth_vis_x16.png", np.repeat(np.repeat(depth_vis_rgb, 16, axis=0), 16, axis=1))
            if rgb is None and self._renderer is not None:
                try:
                    self._renderer.disable_depth_rendering()
                    self._renderer.disable_segmentation_rendering()
                    render_data_rgb = render_data if render_data is not None else self._env.simulator.backend.get_render_data(world_id=self._env_id)
                    self._renderer.update_scene(
                        render_data_rgb,
                        camera=self._camera_name,
                        scene_option=self._scene_option,
                    )
                    rgb = self._sanitize_rgb_array(self._renderer.render())
                except Exception:
                    rgb = None

        if rgb is not None:
            self._save_image(dump_dir / "mujoco_rgb.png", rgb)
            self._save_image(dump_dir / "mujoco_rgb_x16.png", np.repeat(np.repeat(rgb, 16, axis=0), 16, axis=1))
        primary_segmentation = None
        if render_data is not None and self._renderer is not None:
            try:
                self._renderer.disable_depth_rendering()
                self._renderer.enable_segmentation_rendering()
                self._renderer.update_scene(
                    render_data,
                    camera=self._camera_name,
                    scene_option=self._scene_option,
                )
                rendered_primary_segmentation = self._renderer.render()
                if isinstance(rendered_primary_segmentation, np.ndarray):
                    primary_segmentation = rendered_primary_segmentation.copy()
                    np.save(dump_dir / "mujoco_segmentation_raw.npy", primary_segmentation)
                    self._save_image(
                        dump_dir / "mujoco_segmentation_vis.png",
                        self._segmentation_to_rgb(primary_segmentation),
                    )
            except Exception:
                primary_segmentation = None
        if render_data is not None:
            debug_rgb, debug_depth, debug_segmentation = self._render_debug_views(render_data)
            if debug_rgb is not None:
                self._save_image(dump_dir / "mujoco_debug_rgb.png", debug_rgb)
            if debug_depth is not None:
                np.save(dump_dir / "mujoco_debug_depth_raw.npy", debug_depth)
                depth_near = float(getattr(self._cfg, "camera_near", 0.0) or 0.0)
                depth_far = float(min(float(getattr(self._cfg, "camera_far", 3.0) or 3.0), float(self._cfg.max_distance)))
                denom = max(depth_far - depth_near, 1.0e-6)
                debug_depth_vis = np.clip((debug_depth - depth_near) / denom, 0.0, 1.0)
                debug_depth_vis = ((1.0 - debug_depth_vis) * 255.0).astype(np.uint8)
                self._save_image(
                    dump_dir / "mujoco_debug_depth_vis.png",
                    np.repeat(debug_depth_vis[..., None], 3, axis=-1),
                )
            if debug_segmentation is not None:
                np.save(dump_dir / "mujoco_debug_segmentation_raw.npy", debug_segmentation)
                seg_vis = self._segmentation_to_rgb(debug_segmentation)
                self._save_image(dump_dir / "mujoco_debug_segmentation_vis.png", seg_vis)

        if render_data is not None:
            object_body_pos = None
            object_forward_dot = None
            object_geom_debug: list[dict[str, object]] = []
            object_segmentation_pixels: dict[str, int] = {}
            object_primary_segmentation_pixels: dict[str, int] = {}
            object_pose_sweep_pixels: dict[str, dict[str, int]] = {}
            if self._camera_id is not None and self._env.simulator.root_model is not None:
                model = self._env.simulator.root_model
                camera_quat_wxyz = model.cam_quat[self._camera_id]
                w, x, y, z = [float(v) for v in camera_quat_wxyz.tolist()]
                rot = np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                    ],
                    dtype=np.float64,
                )
                camera_forward = -rot[:, 2]
                camera_pos = np.array(model.cam_pos[self._camera_id], dtype=np.float64)
                object_body_id = None
                for body_id in range(model.nbody):
                    body_name = str(model.body(body_id).name or "")
                    if body_name.startswith("object_"):
                        object_body_id = body_id
                        break
                if object_body_id is not None:
                    object_body_pos = render_data.xpos[object_body_id].tolist()
                    object_vec = np.array(object_body_pos, dtype=np.float64) - camera_pos
                    object_norm = np.linalg.norm(object_vec)
                    if object_norm > 1.0e-9:
                        object_forward_dot = float(np.dot(camera_forward, object_vec / object_norm))

                for geom_id in range(int(model.ngeom)):
                    body_id = int(model.geom_bodyid[geom_id])
                    body_name = str(model.body(body_id).name or "")
                    geom_name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}")
                    if not body_name.startswith("object_") and "largebox" not in geom_name.lower() and "object" not in geom_name.lower():
                        continue
                    object_geom_debug.append(
                        {
                            "id": int(geom_id),
                            "name": geom_name,
                            "body_name": body_name,
                            "type": int(model.geom_type[geom_id]),
                            "group": int(model.geom_group[geom_id]),
                            "rgba": [float(v) for v in model.geom_rgba[geom_id].tolist()],
                            "contype": int(model.geom_contype[geom_id]),
                            "conaffinity": int(model.geom_conaffinity[geom_id]),
                            "dataid": int(model.geom_dataid[geom_id]),
                        }
                    )

                if debug_segmentation is not None and object_geom_debug:
                    seg_geom_ids = self._extract_segmentation_geom_ids(debug_segmentation)
                    for geom_info in object_geom_debug:
                        geom_id = int(geom_info["id"])
                        object_segmentation_pixels[str(geom_id)] = int(np.count_nonzero(seg_geom_ids == geom_id))
                if primary_segmentation is not None and object_geom_debug:
                    seg_geom_ids = self._extract_segmentation_geom_ids(primary_segmentation)
                    for geom_info in object_geom_debug:
                        geom_id = int(geom_info["id"])
                        object_primary_segmentation_pixels[str(geom_id)] = int(np.count_nonzero(seg_geom_ids == geom_id))
                if object_geom_debug:
                    object_pose_sweep_pixels = self._sweep_camera_quat_object_pixels(
                        render_data,
                        [int(geom_info["id"]) for geom_info in object_geom_debug],
                    )

            stats = {
                "camera_name": self._camera_name,
                "camera_id": int(self._camera_id if self._camera_id is not None else -1),
                "sim_time": float(render_data.time),
                "camera_pos": self._env.simulator.root_model.cam_pos[self._camera_id].tolist()
                if self._camera_id is not None and self._env.simulator.root_model is not None
                else None,
                "camera_quat_wxyz": self._env.simulator.root_model.cam_quat[self._camera_id].tolist()
                if self._camera_id is not None and self._env.simulator.root_model is not None
                else None,
                "masked_robot_geom_count": len(self._masked_robot_geom_ids),
                "capture_counter": int(self._capture_counter),
                "object_body_pos": object_body_pos,
                "object_forward_dot": object_forward_dot,
                "object_geom_debug": object_geom_debug,
                "object_segmentation_pixels": object_segmentation_pixels,
                "object_primary_segmentation_pixels": object_primary_segmentation_pixels,
                "object_pose_sweep_pixels": object_pose_sweep_pixels,
            }
            (dump_dir / "mujoco_depth_debug.json").write_text(__import__("json").dumps(stats, indent=2))

        self._debug_dump_done = True
        logger.info("MuJoCo depth debug dump written to {}", dump_dir)

    def _render_debug_views(self, render_data: mujoco.MjData) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if self._camera_id is None or self._scene_option is None:
            return None, None, None
        try:
            debug_renderer = mujoco.Renderer(
                self._env.simulator.root_model,
                height=self._debug_dump_render_height,
                width=self._debug_dump_render_width,
            )
        except Exception:
            return None, None, None

        rgb = None
        depth = None
        segmentation = None
        try:
            debug_renderer.disable_depth_rendering()
            debug_renderer.disable_segmentation_rendering()
            debug_renderer.update_scene(
                render_data,
                camera=self._camera_name,
                scene_option=self._scene_option,
            )
            rgb = self._sanitize_rgb_array_with_shape(
                debug_renderer.render(),
                self._debug_dump_render_height,
                self._debug_dump_render_width,
            )
            debug_renderer.enable_depth_rendering()
            debug_renderer.update_scene(
                render_data,
                camera=self._camera_name,
                scene_option=self._scene_option,
            )
            depth = debug_renderer.render()
            if isinstance(depth, np.ndarray):
                depth = self._sanitize_depth_array_with_shape(
                    depth,
                    self._debug_dump_render_height,
                    self._debug_dump_render_width,
                )
            debug_renderer.disable_depth_rendering()
            debug_renderer.enable_segmentation_rendering()
            debug_renderer.update_scene(
                render_data,
                camera=self._camera_name,
                scene_option=self._scene_option,
            )
            rendered_segmentation = debug_renderer.render()
            if isinstance(rendered_segmentation, np.ndarray):
                segmentation = rendered_segmentation.copy()
        except Exception:
            rgb = None
            depth = None
            segmentation = None
        finally:
            try:
                debug_renderer.close()
            except Exception:
                pass
        return rgb, depth, segmentation

    def _sanitize_depth_array_with_shape(self, depth_array: np.ndarray, height: int, width: int) -> np.ndarray:
        if depth_array.size == 0:
            return np.full((height, width), float(self._cfg.max_distance), dtype=np.float32)
        if depth_array.ndim == 3:
            depth_array = depth_array[:, :, 0]
        if depth_array.shape != (height, width):
            return np.full((height, width), float(self._cfg.max_distance), dtype=np.float32)
        depth_array = depth_array.astype(np.float32, copy=False)
        invalid = ~np.isfinite(depth_array) | (depth_array <= 0.0)
        if np.any(invalid):
            depth_array = depth_array.copy()
            depth_array[invalid] = float(self._cfg.max_distance)
        return np.clip(depth_array, 0.0, float(self._cfg.max_distance))

    def _sanitize_rgb_array_with_shape(self, rgb_array: np.ndarray, height: int, width: int) -> np.ndarray:
        if rgb_array.size == 0:
            return np.zeros((height, width, 3), dtype=np.uint8)
        if rgb_array.ndim == 3 and rgb_array.shape[-1] >= 3:
            rgb_array = rgb_array[:, :, :3]
        else:
            return np.zeros((height, width, 3), dtype=np.uint8)
        if rgb_array.shape[:2] != (height, width):
            return np.zeros((height, width, 3), dtype=np.uint8)
        return rgb_array.astype(np.uint8, copy=False)

    @staticmethod
    def _extract_segmentation_geom_ids(segmentation: np.ndarray) -> np.ndarray:
        seg = np.asarray(segmentation)
        if seg.ndim == 3 and seg.shape[-1] >= 2:
            return seg[..., 0].astype(np.int32, copy=False)
        if seg.ndim == 2:
            return seg.astype(np.int32, copy=False)
        return np.full(seg.shape[:2], -1, dtype=np.int32)

    def _segmentation_to_rgb(self, segmentation: np.ndarray) -> np.ndarray:
        geom_ids = self._extract_segmentation_geom_ids(segmentation)
        rgb = np.zeros((*geom_ids.shape, 3), dtype=np.uint8)
        valid = geom_ids >= 0
        if not np.any(valid):
            return rgb
        ids = geom_ids[valid].astype(np.uint32, copy=False)
        rgb_valid = np.stack(
            (
                ((ids * 53 + 29) % 255).astype(np.uint8),
                ((ids * 97 + 71) % 255).astype(np.uint8),
                ((ids * 193 + 151) % 255).astype(np.uint8),
            ),
            axis=-1,
        )
        rgb[valid] = rgb_valid
        return rgb

    def _sweep_camera_quat_object_pixels(
        self,
        render_data: mujoco.MjData,
        object_geom_ids: list[int],
    ) -> dict[str, dict[str, int]]:
        if self._camera_id is None or self._renderer is None:
            return {}
        model = self._env.simulator.root_model
        if model is None:
            return {}

        current_quat = np.asarray(model.cam_quat[self._camera_id], dtype=np.float64).copy()
        candidate_corrections = self._generate_cube_rotation_quaternions_wxyz()

        results: dict[str, dict[str, int]] = {}
        try:
            for name, correction in candidate_corrections.items():
                quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mulQuat(quat, current_quat, correction)
                model.cam_quat[self._camera_id, :] = quat
                mujoco.mj_forward(model, render_data)
                self._renderer.disable_depth_rendering()
                self._renderer.enable_segmentation_rendering()
                self._renderer.update_scene(
                    render_data,
                    camera=self._camera_name,
                    scene_option=self._scene_option,
                )
                rendered = self._renderer.render()
                if not isinstance(rendered, np.ndarray):
                    continue
                seg_geom_ids = self._extract_segmentation_geom_ids(rendered)
                results[name] = {
                    str(geom_id): int(np.count_nonzero(seg_geom_ids == int(geom_id))) for geom_id in object_geom_ids
                }
        finally:
            model.cam_quat[self._camera_id, :] = current_quat
            mujoco.mj_forward(model, render_data)
        return results

    @staticmethod
    def _generate_cube_rotation_quaternions_wxyz() -> dict[str, np.ndarray]:
        rotations: dict[str, np.ndarray] = {}
        basis = np.eye(3, dtype=np.float64)
        labels = ("x", "y", "z")
        for ix, x_axis in enumerate((basis[0], -basis[0], basis[1], -basis[1], basis[2], -basis[2])):
            for iy, y_axis in enumerate((basis[0], -basis[0], basis[1], -basis[1], basis[2], -basis[2])):
                if abs(float(np.dot(x_axis, y_axis))) > 1.0e-9:
                    continue
                z_axis = np.cross(x_axis, y_axis)
                if np.linalg.norm(z_axis) < 0.5:
                    continue
                rot = np.stack((x_axis, y_axis, z_axis), axis=-1)
                if np.linalg.det(rot) < 0.5:
                    continue
                quat = np.zeros(4, dtype=np.float64)
                mujoco.mju_mat2Quat(quat, rot.reshape(-1))
                key = tuple(np.round(quat, 6).tolist())
                if any(np.allclose(quat, existing, atol=1.0e-6) or np.allclose(quat, -existing, atol=1.0e-6) for existing in rotations.values()):
                    continue
                x_label = f"{'' if ix % 2 == 0 else '-'}{labels[ix // 2]}"
                y_label = f"{'' if iy % 2 == 0 else '-'}{labels[iy // 2]}"
                rotations[f"x->{x_label},y->{y_label}"] = quat
        return rotations

    @staticmethod
    def _save_image(path: Path, rgb_array: np.ndarray) -> None:
        try:
            from PIL import Image  # noqa: PLC0415

            Image.fromarray(rgb_array).save(path)
            return
        except Exception:
            pass
        np.save(path.with_suffix(".npy"), rgb_array)
