"""IsaacSim depth camera helper for perception rendering."""

from __future__ import annotations

from typing import Any

import numpy as np

from holosoma.utils.rotations import quat_apply, quat_from_euler_xyz, quat_mul
from holosoma.utils.safe_torch_import import torch


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
        self._view = None
        self._annotator_name: str | None = None
        self._body_index: int | None = None
        self._warned_multi_env = False

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

        if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
            depth_array = depth_array[:, :, 0]

        depth_array = np.clip(depth_array, self._cfg.camera_near, self._cfg.camera_far)
        depth_tensor = torch.as_tensor(depth_array, device=self._device, dtype=torch.float32)
        return depth_tensor.unsqueeze(0)

    def _resolve_body_index(self) -> None:
        if self._body_name is None:
            return
        body_names = getattr(self._env, "body_names", None)
        if body_names is None or self._body_name not in body_names:
            available = body_names if body_names is not None else "unknown"
            raise RuntimeError(f"Camera body '{self._body_name}' not found in body_names: {available}")
        self._body_index = int(body_names.index(self._body_name))

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

        offset_world = quat_apply(body_quat.unsqueeze(0), self._sensor_offset.unsqueeze(0), w_last=True).squeeze(0)
        camera_pos = body_pos + offset_world

        pitch_rad = torch.deg2rad(torch.tensor(self._cfg.camera_pitch_deg, device=self._device))
        pitch_quat = quat_from_euler_xyz(
            torch.tensor(0.0, device=self._device),
            pitch_rad,
            torch.tensor(0.0, device=self._device),
        )
        camera_quat = quat_mul(body_quat.unsqueeze(0), pitch_quat.unsqueeze(0), w_last=True).squeeze(0)
        camera_quat_wxyz = camera_quat[[3, 0, 1, 2]].unsqueeze(0)

        self._view.set_world_poses(
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
