from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from isaac_utils import torch_utils, rotations

from motion_tracking.utils.device_dtype_mixin import DeviceDtypeModuleMixin


@dataclass
class G1LoadedMotions:
    motion_lengths: Tensor
    motion_weights: Tensor
    motion_timings: Tensor
    motion_fps: Tensor
    motion_dt: Tensor
    motion_num_frames: Tensor
    motion_files: Tuple[str, ...]
    sub_motion_to_motion: Tensor


class G1MotionLib(DeviceDtypeModuleMixin):
    """MotionLib adapter for Holosoma G1 .npz clips.

    This keeps the MotionTracking PPO pipeline intact while swapping SMPL
    motion data for Holosoma's robot-centric motion format.
    """

    def __init__(
        self,
        motion_file: str,
        dof_body_ids: List[int],
        dof_offsets: List[int],
        key_body_ids: List[int],
        device: str = "cpu",
        ref_height_adjust: float = 0.0,
        w_last: bool = True,
        body_names: Optional[List[str]] = None,
        dof_names: Optional[List[str]] = None,
        object_names: Optional[List[str]] = None,
        **_,
    ):
        super().__init__()
        self._device = device
        self.w_last = w_last
        self.ref_height_adjust = ref_height_adjust
        self.dof_body_ids = dof_body_ids
        self.dof_offsets = dof_offsets
        self.num_dof = dof_offsets[-1]
        self.object_names = object_names

        self.register_buffer(
            "key_body_ids",
            torch.tensor(key_body_ids, dtype=torch.long, device=device),
            persistent=False,
        )

        self._body_names_asset = list(body_names) if body_names is not None else None
        self._dof_names_asset = list(dof_names) if dof_names is not None else None

        if not str(motion_file).endswith(".npz"):
            raise ValueError(f"G1MotionLib only supports .npz motion files, got: {motion_file}")

        self._load_npz_motion(motion_file)

    def _load_npz_motion(self, motion_file: str) -> None:
        with np.load(motion_file, allow_pickle=True) as data:
            fps_val = data["fps"]
            fps = float(fps_val[0]) if fps_val.shape else float(fps_val)

            body_names = data["body_names"].tolist()
            joint_names = data["joint_names"].tolist()

            joint_pos_raw = data["joint_pos"][:, 7:]
            joint_vel_raw = data["joint_vel"][:, 6:]

            body_pos = data["body_pos_w"]
            body_quat = data["body_quat_w"]
            body_lin_vel = data["body_lin_vel_w"]
            body_ang_vel = data["body_ang_vel_w"]

        if joint_pos_raw.shape[1] != len(joint_names):
            raise ValueError("Joint names in motion data do not match joint_pos shape")
        if body_pos.shape[1] != len(body_names):
            raise ValueError("Body names in motion data do not match body_pos shape")

        body_indices, body_adjustments = self._map_body_names(self._body_names_asset, body_names)
        joint_indices = self._map_names(self._dof_names_asset, joint_names)

        body_pos = body_pos[:, body_indices]
        body_quat = body_quat[:, body_indices]
        body_lin_vel = body_lin_vel[:, body_indices]
        body_ang_vel = body_ang_vel[:, body_indices]

        joint_pos = joint_pos_raw[:, joint_indices]
        joint_vel = joint_vel_raw[:, joint_indices]

        body_quat = body_quat[..., [1, 2, 3, 0]]  # wxyz -> xyzw

        self._body_pos = torch.tensor(body_pos, dtype=torch.float32, device=self._device)
        self._body_quat = torch.tensor(body_quat, dtype=torch.float32, device=self._device)
        self._body_lin_vel = torch.tensor(body_lin_vel, dtype=torch.float32, device=self._device)
        self._body_ang_vel = torch.tensor(body_ang_vel, dtype=torch.float32, device=self._device)
        self._apply_body_adjustments(body_adjustments)
        self._joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=self._device)
        self._joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=self._device)

        num_frames = self._body_pos.shape[0]
        motion_len = (num_frames - 1) / fps

        motion_lengths = torch.tensor([motion_len], dtype=torch.float32, device=self._device)
        motion_weights = torch.tensor([1.0], dtype=torch.float32, device=self._device)
        motion_timings = torch.tensor([[0.0, motion_len]], dtype=torch.float32, device=self._device)
        motion_fps = torch.tensor([fps], dtype=torch.float32, device=self._device)
        motion_dt = torch.tensor([1.0 / fps], dtype=torch.float32, device=self._device)
        motion_num_frames = torch.tensor([num_frames], dtype=torch.long, device=self._device)
        sub_motion_to_motion = torch.tensor([0], dtype=torch.long, device=self._device)

        self.state = G1LoadedMotions(
            motion_lengths=motion_lengths,
            motion_weights=motion_weights,
            motion_timings=motion_timings,
            motion_fps=motion_fps,
            motion_dt=motion_dt,
            motion_num_frames=motion_num_frames,
            motion_files=(str(motion_file),),
            sub_motion_to_motion=sub_motion_to_motion,
        )

        self.length_starts = torch.tensor([0], dtype=torch.long, device=self._device)

    def _map_names(self, asset_names: Optional[List[str]], motion_names: List[str]) -> List[int]:
        if asset_names is None:
            return list(range(len(motion_names)))
        indices: List[int] = []
        for name in asset_names:
            if name not in motion_names:
                raise ValueError(f"Motion file is missing body/joint '{name}'")
            indices.append(motion_names.index(name))
        return indices

    def _map_body_names(
        self, asset_names: Optional[List[str]], motion_names: List[str]
    ) -> Tuple[List[int], List[Tuple[int, int, np.ndarray]]]:
        if asset_names is None:
            return list(range(len(motion_names))), []

        motion_index = {name: idx for idx, name in enumerate(motion_names)}
        asset_index = {name: idx for idx, name in enumerate(asset_names)}
        adjustments: List[Tuple[int, int, np.ndarray]] = []
        indices: List[int] = []
        fallbacks = self._body_fallbacks()

        for name in asset_names:
            if name in motion_index:
                indices.append(motion_index[name])
                continue
            if name in fallbacks:
                src_name, offset = fallbacks[name]
                if src_name not in motion_index:
                    raise ValueError(
                        f"Motion file is missing body '{src_name}' required for '{name}'"
                    )
                indices.append(motion_index[src_name])
                adjustments.append(
                    (asset_index[name], asset_index[src_name], np.asarray(offset, dtype=np.float32))
                )
                continue
            raise ValueError(f"Motion file is missing body '{name}'")

        return indices, adjustments

    def _apply_body_adjustments(self, adjustments: List[Tuple[int, int, np.ndarray]]) -> None:
        if not adjustments:
            return
        for target_idx, source_idx, offset in adjustments:
            offset_tensor = torch.tensor(offset, device=self._device, dtype=self._body_pos.dtype)
            offset_tensor = offset_tensor.unsqueeze(0).repeat(self._body_pos.shape[0], 1)
            source_rot = self._body_quat[:, source_idx]
            rotated_offset = rotations.quat_rotate(source_rot, offset_tensor, self.w_last)
            self._body_pos[:, target_idx] = self._body_pos[:, source_idx] + rotated_offset
            self._body_quat[:, target_idx] = self._body_quat[:, source_idx]
            self._body_ang_vel[:, target_idx] = self._body_ang_vel[:, source_idx]
            self._body_lin_vel[:, target_idx] = self._body_lin_vel[:, source_idx] + torch.cross(
                self._body_ang_vel[:, source_idx], rotated_offset, dim=-1
            )

    @staticmethod
    def _body_fallbacks() -> dict:
        return {
            "left_foot_contact_point": ("left_ankle_roll_link", (0.0, 0.0, -0.037)),
            "right_foot_contact_point": ("right_ankle_roll_link", (0.0, 0.0, -0.037)),
        }

    def num_motions(self) -> int:
        return int(self.state.motion_lengths.shape[0])

    def num_sub_motions(self) -> int:
        return int(self.state.motion_timings.shape[0])

    def sample_motions(self, n: int) -> Tensor:
        weights = self.state.motion_weights
        return torch.multinomial(weights, num_samples=n, replacement=True)

    def sample_other_motions(self, already_chosen_ids: Tensor) -> Tensor:
        n = already_chosen_ids.shape[0]
        motion_weights = self.state.motion_weights.unsqueeze(0).repeat(n, 1)
        motion_weights = motion_weights.scatter(
            1, already_chosen_ids.unsqueeze(-1), torch.zeros_like(motion_weights)
        )
        return torch.multinomial(motion_weights, num_samples=1).squeeze(-1)

    def sample_time(self, sub_motion_ids: Tensor, truncate_time: Optional[float] = None) -> Tensor:
        phase = torch.rand(sub_motion_ids.shape, device=self._device)
        motion_len = self.state.motion_timings[sub_motion_ids, 1] - self.state.motion_timings[sub_motion_ids, 0]
        if truncate_time is not None:
            motion_len = motion_len - truncate_time
            if torch.any(motion_len < 0.0):
                raise ValueError("truncate_time is larger than motion length")
        return phase * motion_len + self.state.motion_timings[sub_motion_ids, 0]

    def get_sub_motion_length(self, sub_motion_ids: Tensor) -> Tensor:
        return self.state.motion_timings[sub_motion_ids, 1] - self.state.motion_timings[sub_motion_ids, 0]

    def get_motion_length(self, sub_motion_ids: Tensor) -> Tensor:
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]
        return self.state.motion_lengths[motion_ids]

    def get_motion_root_pos(self, sub_motion_ids: Tensor, motion_times: Tensor, *_args, **_kwargs) -> Tensor:
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]
        motion_len = self.state.motion_lengths[motion_ids]
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0 = frame_idx0 + self.length_starts[motion_ids]
        f1 = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self._body_pos[f0, 0]
        root_pos1 = self._body_pos[f1, 0]
        blend = blend.unsqueeze(-1)
        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_pos[:, 2] += self.ref_height_adjust
        return root_pos

    def get_mimic_motion_state(self, sub_motion_ids: Tensor, motion_times: Tensor, *_args, **_kwargs):
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]
        motion_len = self.state.motion_lengths[motion_ids]
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0 = frame_idx0 + self.length_starts[motion_ids]
        f1 = frame_idx1 + self.length_starts[motion_ids]

        body_pos0 = self._body_pos[f0]
        body_pos1 = self._body_pos[f1]
        body_quat0 = self._body_quat[f0]
        body_quat1 = self._body_quat[f1]
        body_vel0 = self._body_lin_vel[f0]
        body_vel1 = self._body_lin_vel[f1]
        body_ang_vel0 = self._body_ang_vel[f0]
        body_ang_vel1 = self._body_ang_vel[f1]
        dof_pos0 = self._joint_pos[f0]
        dof_pos1 = self._joint_pos[f1]
        dof_vel0 = self._joint_vel[f0]
        dof_vel1 = self._joint_vel[f1]

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        body_pos = (1.0 - blend_exp) * body_pos0 + blend_exp * body_pos1
        body_quat = torch_utils.slerp(body_quat0, body_quat1, blend_exp)
        body_vel = (1.0 - blend_exp) * body_vel0 + blend_exp * body_vel1
        body_ang_vel = (1.0 - blend_exp) * body_ang_vel0 + blend_exp * body_ang_vel1
        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1

        body_pos[:, :, 2] += self.ref_height_adjust

        return body_pos, body_quat, dof_pos, body_vel, body_ang_vel, dof_vel

    def get_motion_state(self, sub_motion_ids: Tensor, motion_times: Tensor, *_args, **_kwargs):
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]
        motion_len = self.state.motion_lengths[motion_ids]
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_times, motion_len, num_frames, dt)
        f0 = frame_idx0 + self.length_starts[motion_ids]
        f1 = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self._body_pos[f0, 0]
        root_pos1 = self._body_pos[f1, 0]
        root_rot0 = self._body_quat[f0, 0]
        root_rot1 = self._body_quat[f1, 0]
        root_vel0 = self._body_lin_vel[f0, 0]
        root_vel1 = self._body_lin_vel[f1, 0]
        root_ang_vel0 = self._body_ang_vel[f0, 0]
        root_ang_vel1 = self._body_ang_vel[f1, 0]

        rb_pos0 = self._body_pos[f0]
        rb_pos1 = self._body_pos[f1]
        rb_rot0 = self._body_quat[f0]
        rb_rot1 = self._body_quat[f1]
        rb_vel0 = self._body_lin_vel[f0]
        rb_vel1 = self._body_lin_vel[f1]
        rb_ang_vel0 = self._body_ang_vel[f0]
        rb_ang_vel1 = self._body_ang_vel[f1]

        dof_pos0 = self._joint_pos[f0]
        dof_pos1 = self._joint_pos[f1]
        dof_vel0 = self._joint_vel[f0]
        dof_vel1 = self._joint_vel[f1]

        key_pos0 = rb_pos0[:, self.key_body_ids]
        key_pos1 = rb_pos1[:, self.key_body_ids]

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        root_pos = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_pos[:, 2] += self.ref_height_adjust
        root_rot = torch_utils.slerp(root_rot0, root_rot1, blend)
        root_vel = (1.0 - blend) * root_vel0 + blend * root_vel1
        root_ang_vel = (1.0 - blend) * root_ang_vel0 + blend * root_ang_vel1

        dof_pos = (1.0 - blend) * dof_pos0 + blend * dof_pos1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1

        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        key_pos[:, :, 2] += self.ref_height_adjust

        rb_pos = (1.0 - blend_exp) * rb_pos0 + blend_exp * rb_pos1
        rb_pos[:, :, 2] += self.ref_height_adjust
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        rb_vel = (1.0 - blend_exp) * rb_vel0 + blend_exp * rb_vel1
        rb_ang_vel = (1.0 - blend_exp) * rb_ang_vel0 + blend_exp * rb_ang_vel1

        return (
            root_pos,
            root_rot,
            dof_pos,
            root_vel,
            root_ang_vel,
            dof_vel,
            key_pos,
            rb_pos,
            rb_rot,
            rb_vel,
            rb_ang_vel,
        )

    def _calc_frame_blend(self, time: Tensor, motion_len: Tensor, num_frames: Tensor, dt: Tensor):
        phase = time / motion_len
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt
        return frame_idx0, frame_idx1, blend
