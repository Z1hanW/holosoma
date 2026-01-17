"""IsaacSim depth camera helper for perception rendering."""

from __future__ import annotations

from typing import Any

import numpy as np

from loguru import logger

from holosoma.utils.rotations import quat_apply, quat_from_euler_xyz, quat_mul
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.urdf_utils import resolve_fixed_link_offset


class IsaacSimDepthCamera:
    """Rendered depth camera backed by IsaacSim replicator."""

    def __init__(
        self,
        *,
        env: Any,
        config: Any,
        width: int,
        height: int,
        vfov_deg: float,
        device: str,
    ) -> None:
        self._env = env
        self._cfg = config
        self._width = int(width)
        self._height = int(height)
        self._vfov_deg = float(vfov_deg)
        self._device = device

        self._env_id = int(getattr(config, "camera_env_id", 0))
        self._body_name = getattr(config, "camera_body_name", None)
        self._sensor_offset = torch.tensor(getattr(config, "sensor_offset", [0.0, 0.0, 0.0]), device=self._device)

        self._camera_prim_path: str | None = None
        self._render_product = None
        self._depth_annotator = None
        self._rgb_annotator = None
        self._view = None
        self._rgb_view = None
        self._annotator_name: str | None = None
        self._body_index: int | None = None
        self._body_offset_pos = torch.zeros(3, device=self._device)
        self._body_offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device)
        self._camera_frame_quat = torch.tensor([0.5, -0.5, -0.5, 0.5], device=self._device)
        self._warned_multi_env = False
        self._warned_invalid_depth = False
        self._warned_invalid_rgb = False

    def setup(self) -> None:
        if self._env_id < 0 or self._env_id >= self._env.num_envs:
            raise RuntimeError(
                f"camera_env_id out of range: {self._env_id} (num_envs={self._env.num_envs})"
            )
        self._resolve_body_index()

        import omni.usd
        import omni.replicator.core as rep
        from pxr import UsdGeom
        from isaacsim.core.prims import XFormPrim

        self._camera_prim_path = f"/World/envs/env_{self._env_id}/PerceptionDepthCamera"

        stage = omni.usd.get_context().get_stage()
        camera_prim = UsdGeom.Camera.Define(stage, self._camera_prim_path)

        aspect_ratio = self._width / max(self._height, 1)
        vertical_aperture = 24.0
        horizontal_aperture = vertical_aperture * aspect_ratio
        focal_length = self._calculate_focal_length_from_fov(self._vfov_deg, vertical_aperture)

        camera_prim.GetFocalLengthAttr().Set(focal_length)
        camera_prim.GetClippingRangeAttr().Set((self._cfg.camera_near, self._cfg.camera_far))
        camera_prim.GetHorizontalApertureAttr().Set(horizontal_aperture)
        camera_prim.GetVerticalApertureAttr().Set(vertical_aperture)

        self._view = XFormPrim(self._camera_prim_path, reset_xform_properties=True)
        self._view.initialize()

        self._render_product = rep.create.render_product(self._camera_prim_path, (self._width, self._height))
        self._depth_annotator, self._annotator_name = self._create_depth_annotator(rep)
        self._depth_annotator.attach([self._render_product])
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator(
            "rgb",
            device=self._device,
            do_array_copy=False,
        )
        self._rgb_annotator.attach([self._render_product])

    def capture_depth(self) -> torch.Tensor:
        if self._depth_annotator is None:
            raise RuntimeError("IsaacSimDepthCamera.setup() must be called before capture_depth().")

        if self._env.num_envs > 1 and not self._warned_multi_env:
            self._warned_multi_env = True
            env_count = self._env.num_envs
            raise RuntimeError(
                f"Rendered depth camera only supports one environment; got num_envs={env_count}. "
                "Use num_envs=1 for previs_perception or switch to raycast."
            )

        self._update_pose()
        self._env.simulator.render()

        depth_data = self._depth_annotator.get_data()
        depth_array = self._convert_depth_to_numpy(depth_data)
        depth_array = self._sanitize_depth_array(depth_array)
        depth_tensor = torch.as_tensor(depth_array, device=self._device, dtype=torch.float32)
        return depth_tensor.unsqueeze(0)

    def capture_rgb(self) -> np.ndarray:
        if self._rgb_annotator is None:
            raise RuntimeError("IsaacSimDepthCamera.setup() must be called before capture_rgb().")

        if self._env.num_envs > 1 and not self._warned_multi_env:
            self._warned_multi_env = True
            env_count = self._env.num_envs
            raise RuntimeError(
                f"Rendered RGB camera only supports one environment; got num_envs={env_count}. "
                "Use num_envs=1 for previs_perception or switch to raycast."
            )

        self._update_pose()
        self._env.simulator.render()

        rgb_data = self._rgb_annotator.get_data()
        rgb_array = self._convert_rgb_to_numpy(rgb_data)
        rgb_array = self._sanitize_rgb_array(rgb_array)
        return rgb_array

    def _resolve_body_index(self) -> None:
        if self._body_name is None:
            return
        body_names = getattr(self._env, "body_names", None)
        if body_names is not None and self._body_name in body_names:
            self._body_index = int(body_names.index(self._body_name))
            return

        resolved = resolve_fixed_link_offset(
            self._env.robot_config,
            self._body_name,
            available_links=body_names,
            device=self._device,
        )
        if resolved is None or body_names is None:
            available = body_names if body_names is not None else "unknown"
            raise RuntimeError(f"Camera body '{self._body_name}' not found in body_names: {available}")

        parent_name, offset_pos, offset_quat = resolved
        self._body_index = int(body_names.index(parent_name))
        self._body_offset_pos = offset_pos
        self._body_offset_quat = offset_quat

    def _update_pose(self) -> None:
        if self._view is None:
            return

        env_id = self._env_id
        if self._body_index is not None:
            body_pos = self._env.simulator._rigid_body_pos[env_id, self._body_index]
            body_quat = self._env.simulator._rigid_body_rot[env_id, self._body_index]
        else:
            body_pos = self._env.simulator.robot_root_states[env_id, :3]
            body_quat = self._env.simulator.base_quat[env_id]

        body_pos = body_pos + quat_apply(body_quat.unsqueeze(0), self._body_offset_pos.unsqueeze(0), w_last=True).squeeze(
            0
        )
        body_quat = quat_mul(body_quat.unsqueeze(0), self._body_offset_quat.unsqueeze(0), w_last=True).squeeze(0)

        offset_world = quat_apply(body_quat.unsqueeze(0), self._sensor_offset.unsqueeze(0), w_last=True).squeeze(0)
        camera_pos = body_pos + offset_world

        pitch_rad = torch.deg2rad(torch.tensor(self._cfg.camera_pitch_deg, device=self._device))
        pitch_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=self._device),
            pitch_rad,
            torch.tensor(0.0, device=self._device),
        )
        camera_quat = quat_mul(body_quat.unsqueeze(0), pitch_quat.unsqueeze(0), w_last=True)
        camera_quat = quat_mul(
            camera_quat,
            self._camera_frame_quat.unsqueeze(0),
            w_last=True,
        ).squeeze(0)
        camera_quat_wxyz = camera_quat[[3, 0, 1, 2]].unsqueeze(0)

        self._view.set_world_poses(
            camera_pos.unsqueeze(0),
            camera_quat_wxyz,
            torch.tensor([env_id], device=self._device, dtype=torch.int32),
        )
        if self._rgb_view is not None and self._rgb_view is not self._view:
            self._rgb_view.set_world_poses(
                camera_pos.unsqueeze(0),
                camera_quat_wxyz,
                torch.tensor([env_id], device=self._device, dtype=torch.int32),
            )
        if self._rgb_view is not None and self._rgb_view is not self._view:
            self._rgb_view.set_world_poses(
                camera_pos.unsqueeze(0),
                camera_quat_wxyz,
                torch.tensor([env_id], device=self._device, dtype=torch.int32),
            )

    @staticmethod
    def _calculate_focal_length_from_fov(vertical_fov_degrees: float, aperture_mm: float) -> float:
        import math

        vertical_fov_rad = math.radians(vertical_fov_degrees)
        return aperture_mm / (2 * math.tan(vertical_fov_rad / 2))

    def _create_depth_annotator(self, rep_module):
        candidates = ("distance_to_image_plane", "distance_to_camera", "depth")
        last_error: Exception | None = None
        for name in candidates:
            try:
                annotator = rep_module.AnnotatorRegistry.get_annotator(
                    name,
                    device=self._device,
                    do_array_copy=False,
                )
                return annotator, name
            except Exception as exc:  # pragma: no cover - depends on IsaacSim install
                last_error = exc
                continue
        raise RuntimeError(f"Failed to create depth annotator ({candidates}): {last_error}")

    @staticmethod
    def _convert_depth_to_numpy(depth_data) -> np.ndarray:
        if isinstance(depth_data, np.ndarray):
            return depth_data
        if hasattr(depth_data, "numpy"):
            return depth_data.numpy()
        return np.asarray(depth_data)

    @staticmethod
    def _convert_rgb_to_numpy(rgb_data) -> np.ndarray:
        if isinstance(rgb_data, np.ndarray):
            return rgb_data
        if hasattr(rgb_data, "numpy"):
            return rgb_data.numpy()
        return np.asarray(rgb_data)

    def _sanitize_depth_array(self, depth_array: np.ndarray) -> np.ndarray:
        if depth_array.size == 0:
            if not self._warned_invalid_depth:
                logger.warning("Depth annotator returned empty data; filling with max_distance.")
                self._warned_invalid_depth = True
            return np.full((self._height, self._width), float(self._cfg.max_distance), dtype=np.float32)

        if depth_array.ndim == 3:
            depth_array = depth_array[:, :, 0]
        elif depth_array.ndim == 1 and depth_array.size == self._height * self._width:
            depth_array = depth_array.reshape(self._height, self._width)

        if depth_array.shape != (self._height, self._width):
            if not self._warned_invalid_depth:
                logger.warning(
                    "Unexpected depth shape {}; expected ({}, {}). Filling with max_distance.",
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
                logger.warning("RGB annotator returned empty data; filling with zeros.")
                self._warned_invalid_rgb = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        if rgb_array.ndim == 1:
            if rgb_array.size == self._height * self._width * 4:
                rgb_array = rgb_array.reshape(self._height, self._width, 4)
            elif rgb_array.size == self._height * self._width * 3:
                rgb_array = rgb_array.reshape(self._height, self._width, 3)

        if rgb_array.ndim == 3 and rgb_array.shape[-1] >= 3:
            rgb_array = rgb_array[:, :, :3]
        else:
            if not self._warned_invalid_rgb:
                logger.warning(
                    "Unexpected RGB shape {}; expected ({}, {}, 3). Filling with zeros.",
                    rgb_array.shape,
                    self._height,
                    self._width,
                )
                self._warned_invalid_rgb = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        if rgb_array.shape[:2] != (self._height, self._width):
            if not self._warned_invalid_rgb:
                logger.warning(
                    "Unexpected RGB shape {}; expected ({}, {}, 3). Filling with zeros.",
                    rgb_array.shape,
                    self._height,
                    self._width,
                )
                self._warned_invalid_rgb = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        return rgb_array.astype(np.uint8, copy=False)


class IsaacSimDepthSensorCamera:
    """Rendered depth camera using IsaacSim depth sensor pipeline."""

    def __init__(
        self,
        *,
        env: Any,
        config: Any,
        width: int,
        height: int,
        vfov_deg: float,
        device: str,
    ) -> None:
        self._env = env
        self._cfg = config
        self._width = int(width)
        self._height = int(height)
        self._vfov_deg = float(vfov_deg)
        self._device = device

        self._env_id = int(getattr(config, "camera_env_id", 0))
        self._body_name = getattr(config, "camera_body_name", None)
        self._sensor_offset = torch.tensor(getattr(config, "sensor_offset", [0.0, 0.0, 0.0]), device=self._device)
        self._asset_path = getattr(config, "depth_sensor_asset_path", None)
        self._depth_sensor_prim = getattr(config, "depth_sensor_depth_prim", None)

        self._sensor_prim_path: str | None = None
        self._depth_sensor = None
        self._sensor_asset = None
        self._view = None
        self._rgb_view = None
        self._rgb_camera_prim_path: str | None = None
        self._rgb_render_product = None
        self._rgb_annotator = None
        self._body_index: int | None = None
        self._body_offset_pos = torch.zeros(3, device=self._device)
        self._body_offset_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=self._device)
        self._camera_frame_quat = torch.tensor([0.5, -0.5, -0.5, 0.5], device=self._device)
        self._warned_multi_env = False
        self._warned_invalid_depth = False
        self._warned_invalid_rgb = False

    def setup(self) -> None:
        if self._env_id < 0 or self._env_id >= self._env.num_envs:
            raise RuntimeError(
                f"camera_env_id out of range: {self._env_id} (num_envs={self._env.num_envs})"
            )
        self._resolve_body_index()

        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.prims import XFormPrim
        from pxr import UsdGeom

        try:
            from isaacsim.sensors.camera import SingleViewDepthSensor, SingleViewDepthSensorAsset  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - depends on IsaacSim install
            raise RuntimeError("SingleViewDepthSensor is not available in this IsaacSim install.") from exc

        self._sensor_prim_path = f"/World/envs/env_{self._env_id}/PerceptionDepthSensor"

        if self._asset_path:
            asset_path = self._resolve_asset_path(self._asset_path)
            self._sensor_asset = SingleViewDepthSensorAsset(prim_path=self._sensor_prim_path, asset_path=asset_path)
            self._sensor_asset.initialize()
            depth_prim = self._depth_sensor_prim or self._pick_depth_sensor_prim()
            self._depth_sensor = self._sensor_asset.get_child_depth_sensor(depth_prim)
        else:
            self._depth_sensor = SingleViewDepthSensor(
                prim_path=self._sensor_prim_path,
                name="PerceptionDepthSensor",
                resolution=(self._width, self._height),
                frequency=float(self._cfg.camera_fps),
            )
            self._depth_sensor.initialize()

        if self._depth_sensor is None:
            raise RuntimeError("Failed to initialize depth sensor.")

        if hasattr(self._depth_sensor, "attach_annotator"):
            self._depth_sensor.attach_annotator("DepthSensorDistance")

        stage = omni.usd.get_context().get_stage()
        camera_prim = UsdGeom.Camera.Get(stage, self._sensor_prim_path)
        if camera_prim and camera_prim.GetPrim().IsValid() and camera_prim.GetPrim().IsA(UsdGeom.Camera):
            aspect_ratio = self._width / max(self._height, 1)
            vertical_aperture = 24.0
            horizontal_aperture = vertical_aperture * aspect_ratio
            focal_length = IsaacSimDepthCamera._calculate_focal_length_from_fov(self._vfov_deg, vertical_aperture)
            camera_prim.GetFocalLengthAttr().Set(focal_length)
            camera_prim.GetClippingRangeAttr().Set((self._cfg.camera_near, self._cfg.camera_far))
            camera_prim.GetHorizontalApertureAttr().Set(horizontal_aperture)
            camera_prim.GetVerticalApertureAttr().Set(vertical_aperture)

        self._view = XFormPrim(self._sensor_prim_path, reset_xform_properties=True)
        self._view.initialize()
        self._setup_rgb_camera(stage, rep, UsdGeom)

    def capture_depth(self) -> torch.Tensor:
        if self._depth_sensor is None:
            raise RuntimeError("Depth sensor is not initialized; call setup() first.")

        if self._env.num_envs > 1 and not self._warned_multi_env:
            self._warned_multi_env = True
            env_count = self._env.num_envs
            raise RuntimeError(
                f"Rendered depth sensor only supports one environment; got num_envs={env_count}. "
                "Use num_envs=1 for previs_perception or switch to raycast."
            )

        self._update_pose()
        self._env.simulator.render()

        depth_data = self._read_depth_frame()
        depth_array = IsaacSimDepthCamera._convert_depth_to_numpy(depth_data)
        depth_array = self._sanitize_depth_array(depth_array)
        depth_tensor = torch.as_tensor(depth_array, device=self._device, dtype=torch.float32)
        return depth_tensor.unsqueeze(0)

    def capture_rgb(self) -> np.ndarray:
        if self._rgb_annotator is None:
            raise RuntimeError("RGB camera is not initialized; call setup() first.")

        if self._env.num_envs > 1 and not self._warned_multi_env:
            self._warned_multi_env = True
            env_count = self._env.num_envs
            raise RuntimeError(
                f"Rendered RGB camera only supports one environment; got num_envs={env_count}. "
                "Use num_envs=1 for previs_perception or switch to raycast."
            )

        self._update_pose()
        self._env.simulator.render()

        rgb_data = self._rgb_annotator.get_data()
        rgb_array = IsaacSimDepthCamera._convert_rgb_to_numpy(rgb_data)
        rgb_array = self._sanitize_rgb_array(rgb_array)
        return rgb_array

    def _read_depth_frame(self):
        if hasattr(self._depth_sensor, "get_current_frame"):
            frame = self._depth_sensor.get_current_frame()
            if isinstance(frame, dict):
                for key in ("DepthSensorDistance", "distance_to_image_plane", "depth"):
                    if key in frame:
                        return frame[key]
        if hasattr(self._depth_sensor, "get_depth"):
            return self._depth_sensor.get_depth()
        if hasattr(self._depth_sensor, "get_data"):
            return self._depth_sensor.get_data()
        raise RuntimeError("Depth sensor did not return a depth frame.")

    def _pick_depth_sensor_prim(self) -> str:
        if self._sensor_asset is None:
            raise RuntimeError("Depth sensor asset is not initialized.")
        depth_paths = self._sensor_asset.get_all_depth_sensor_paths()
        if not depth_paths:
            raise RuntimeError("Depth sensor asset has no depth sensor paths.")
        for path in depth_paths:
            if "Pseudo_Depth" in path or "Depth" in path:
                return path
        return depth_paths[0]

    @staticmethod
    def _resolve_asset_path(asset_path: str) -> str:
        if asset_path.startswith("/Isaac/"):
            from isaacsim.storage.native import get_assets_root_path  # noqa: PLC0415

            return get_assets_root_path() + asset_path
        return asset_path

    def _setup_rgb_camera(self, stage, rep_module, usd_geom_module) -> None:
        camera_prim = usd_geom_module.Camera.Get(stage, self._sensor_prim_path)
        use_sensor_camera = (
            camera_prim
            and camera_prim.GetPrim().IsValid()
            and camera_prim.GetPrim().IsA(usd_geom_module.Camera)
        )

        if use_sensor_camera:
            self._rgb_camera_prim_path = self._sensor_prim_path
            self._rgb_view = self._view
        else:
            self._rgb_camera_prim_path = f"/World/envs/env_{self._env_id}/PerceptionRgbCamera"
            rgb_camera = usd_geom_module.Camera.Define(stage, self._rgb_camera_prim_path)
            aspect_ratio = self._width / max(self._height, 1)
            vertical_aperture = 24.0
            horizontal_aperture = vertical_aperture * aspect_ratio
            focal_length = IsaacSimDepthCamera._calculate_focal_length_from_fov(self._vfov_deg, vertical_aperture)
            rgb_camera.GetFocalLengthAttr().Set(focal_length)
            rgb_camera.GetClippingRangeAttr().Set((self._cfg.camera_near, self._cfg.camera_far))
            rgb_camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
            rgb_camera.GetVerticalApertureAttr().Set(vertical_aperture)

            from isaacsim.core.prims import XFormPrim  # noqa: PLC0415

            self._rgb_view = XFormPrim(self._rgb_camera_prim_path, reset_xform_properties=True)
            self._rgb_view.initialize()

        self._rgb_render_product = rep_module.create.render_product(
            self._rgb_camera_prim_path,
            (self._width, self._height),
        )
        self._rgb_annotator = rep_module.AnnotatorRegistry.get_annotator(
            "rgb",
            device=self._device,
            do_array_copy=False,
        )
        self._rgb_annotator.attach([self._rgb_render_product])

    def _sanitize_rgb_array(self, rgb_array: np.ndarray) -> np.ndarray:
        if rgb_array.size == 0:
            if not self._warned_invalid_rgb:
                logger.warning("RGB annotator returned empty data; filling with zeros.")
                self._warned_invalid_rgb = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        if rgb_array.ndim == 1:
            if rgb_array.size == self._height * self._width * 4:
                rgb_array = rgb_array.reshape(self._height, self._width, 4)
            elif rgb_array.size == self._height * self._width * 3:
                rgb_array = rgb_array.reshape(self._height, self._width, 3)

        if rgb_array.ndim == 3 and rgb_array.shape[-1] >= 3:
            rgb_array = rgb_array[:, :, :3]
        else:
            if not self._warned_invalid_rgb:
                logger.warning(
                    "Unexpected RGB shape {}; expected ({}, {}, 3). Filling with zeros.",
                    rgb_array.shape,
                    self._height,
                    self._width,
                )
                self._warned_invalid_rgb = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        if rgb_array.shape[:2] != (self._height, self._width):
            if not self._warned_invalid_rgb:
                logger.warning(
                    "Unexpected RGB shape {}; expected ({}, {}, 3). Filling with zeros.",
                    rgb_array.shape,
                    self._height,
                    self._width,
                )
                self._warned_invalid_rgb = True
            return np.zeros((self._height, self._width, 3), dtype=np.uint8)

        return rgb_array.astype(np.uint8, copy=False)

    def _sanitize_depth_array(self, depth_array: np.ndarray) -> np.ndarray:
        if depth_array.size == 0:
            if not self._warned_invalid_depth:
                logger.warning("Depth sensor returned empty data; filling with max_distance.")
                self._warned_invalid_depth = True
            return np.full((self._height, self._width), float(self._cfg.max_distance), dtype=np.float32)

        if depth_array.ndim == 3:
            depth_array = depth_array[:, :, 0]
        elif depth_array.ndim == 1 and depth_array.size == self._height * self._width:
            depth_array = depth_array.reshape(self._height, self._width)

        if depth_array.shape != (self._height, self._width):
            if not self._warned_invalid_depth:
                logger.warning(
                    "Unexpected depth sensor shape {}; expected ({}, {}). Filling with max_distance.",
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

    def _resolve_body_index(self) -> None:
        if self._body_name is None:
            return
        body_names = getattr(self._env, "body_names", None)
        if body_names is not None and self._body_name in body_names:
            self._body_index = int(body_names.index(self._body_name))
            return

        resolved = resolve_fixed_link_offset(
            self._env.robot_config,
            self._body_name,
            available_links=body_names,
            device=self._device,
        )
        if resolved is None or body_names is None:
            available = body_names if body_names is not None else "unknown"
            raise RuntimeError(f"Camera body '{self._body_name}' not found in body_names: {available}")

        parent_name, offset_pos, offset_quat = resolved
        self._body_index = int(body_names.index(parent_name))
        self._body_offset_pos = offset_pos
        self._body_offset_quat = offset_quat

    def _update_pose(self) -> None:
        if self._view is None:
            return

        env_id = self._env_id
        if self._body_index is not None:
            body_pos = self._env.simulator._rigid_body_pos[env_id, self._body_index]
            body_quat = self._env.simulator._rigid_body_rot[env_id, self._body_index]
        else:
            body_pos = self._env.simulator.robot_root_states[env_id, :3]
            body_quat = self._env.simulator.base_quat[env_id]

        body_pos = body_pos + quat_apply(body_quat.unsqueeze(0), self._body_offset_pos.unsqueeze(0), w_last=True).squeeze(
            0
        )
        body_quat = quat_mul(body_quat.unsqueeze(0), self._body_offset_quat.unsqueeze(0), w_last=True).squeeze(0)

        offset_world = quat_apply(body_quat.unsqueeze(0), self._sensor_offset.unsqueeze(0), w_last=True).squeeze(0)
        camera_pos = body_pos + offset_world

        pitch_rad = torch.deg2rad(torch.tensor(self._cfg.camera_pitch_deg, device=self._device))
        pitch_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=self._device),
            pitch_rad,
            torch.tensor(0.0, device=self._device),
        )
        camera_quat = quat_mul(body_quat.unsqueeze(0), pitch_quat.unsqueeze(0), w_last=True)
        camera_quat = quat_mul(
            camera_quat,
            self._camera_frame_quat.unsqueeze(0),
            w_last=True,
        ).squeeze(0)
        camera_quat_wxyz = camera_quat[[3, 0, 1, 2]].unsqueeze(0)

        self._view.set_world_poses(
            camera_pos.unsqueeze(0),
            camera_quat_wxyz,
            torch.tensor([env_id], device=self._device, dtype=torch.int32),
        )
