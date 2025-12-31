# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from copy import deepcopy

from isaac_utils import torch_utils, rotations

import numpy as np
import os
import yaml

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from poselib.core.rotation3d import *

from motion_tracking.utils.device_dtype_mixin import DeviceDtypeModuleMixin

import torch
from torch import nn, Tensor
from tqdm import tqdm
from torch import Tensor
from typing import List, Tuple, Any

from dataclasses import dataclass


def quat_w_first(rot):
    rot = torch.cat([rot[..., [-1]], rot[..., :-1]], -1)
    return rot


@dataclass
class MotionState:
    root_pos: Tensor
    root_rot: Tensor
    dof_pos: Tensor
    root_vel: Tensor
    root_ang_vel: Tensor
    dof_vel: Tensor
    key_pos: Tensor


class LoadedMotions(nn.Module):
    """
    Tuples here needed so the class can hash, which is
    needed so the module can be iterated over in a parent
    module's children.
    """

    motions: Tuple[SkeletonMotion]
    motion_lengths: Tensor
    motion_weights: Tensor
    motion_timings: Tensor
    motion_fps: Tensor
    motion_dt: Tensor
    motion_num_frames: Tensor
    motion_files: Tuple[str]
    sub_motion_to_motion: Tensor
    ref_respawn_offsets: Tensor
    text_embeddings: Tensor
    has_text_embeddings: Tensor
    supported_object_names: List[List[str]]

    def __init__(
            self,
            motions: Tuple[SkeletonMotion],
            motion_lengths: Tensor,
            motion_weights: Tensor,
            motion_timings: Tensor,
            motion_fps: Tensor,
            motion_dt: Tensor,
            motion_num_frames: Tensor,
            motion_files: Tuple[str],
            sub_motion_to_motion: Tensor,
            ref_respawn_offsets: Tensor,
            text_embeddings: Tensor,
            has_text_embeddings: Tensor,
            supported_object_names: List[List[str]],
    ):
        super().__init__()
        self.motions = motions
        self.motion_files = motion_files
        self.register_buffer("motion_lengths", motion_lengths, persistent=False)
        self.register_buffer("motion_weights", motion_weights, persistent=False)
        self.register_buffer("motion_timings", motion_timings, persistent=False)
        self.register_buffer("motion_fps", motion_fps, persistent=False)
        self.register_buffer("motion_dt", motion_dt, persistent=False)
        self.register_buffer("motion_num_frames", motion_num_frames, persistent=False)
        self.register_buffer("sub_motion_to_motion", sub_motion_to_motion, persistent=False)
        self.register_buffer("ref_respawn_offsets", ref_respawn_offsets, persistent=False)
        self.register_buffer("text_embeddings", text_embeddings, persistent=False)
        self.register_buffer("has_text_embeddings", has_text_embeddings, persistent=False)
        self.supported_object_names = supported_object_names


class MotionLib(DeviceDtypeModuleMixin):
    gts: Tensor
    grs: Tensor
    lrs: Tensor
    gvs: Tensor
    gavs: Tensor
    grvs: Tensor
    gravs: Tensor
    dvs: Tensor
    length_starts: Tensor
    motion_ids: Tensor
    key_body_ids: Tensor

    def __init__(
            self,
            motion_file,
            dof_body_ids,
            dof_offsets,
            key_body_ids,
            device="cpu",
            ref_height_adjust: float = 0,
            target_frame_rate: int = 30,
            w_last: bool = True,
            create_text_embeddings: bool = False,
            object_names: List[str] = None,
    ):
        super().__init__()
        self.w_last = w_last
        self.create_text_embeddings = create_text_embeddings
        self.dof_body_ids = dof_body_ids
        self.dof_offsets = dof_offsets
        self.num_dof = dof_offsets[-1]
        self.ref_height_adjust = ref_height_adjust
        self.register_buffer(
            "key_body_ids",
            torch.tensor(key_body_ids, dtype=torch.long, device=device),
            persistent=False,
        )

        if str(motion_file).split(".")[-1] in ["yaml", "npy", "npz", "np"]:
            print("Loading motions from yaml/npy file")
            
            self._load_motions(motion_file, target_frame_rate)
        else:
            print("Loading motions from state file")
            with open(motion_file, "rb") as file:
                self.state: LoadedMotions = torch.load(file, map_location="cpu")

        motions = self.state.motions
        
        ###
        # import pdb;pdb.set_trace()
        
        
        self.register_buffer(
            "gts",
            torch.cat([m.global_translation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "grs",
            torch.cat([m.global_rotation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "lrs",
            torch.cat([m.local_rotation for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "grvs",
            torch.cat([m.global_root_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gravs",
            torch.cat([m.global_root_angular_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gavs",
            torch.cat([m.global_angular_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "gvs",
            torch.cat([m.global_velocity for m in motions], dim=0).to(
                dtype=torch.float32
            ),
            persistent=False,
        )
        self.register_buffer(
            "dvs",
            torch.cat([m.dof_vels for m in motions], dim=0).to(
                device=device, dtype=torch.float32
            ),
            persistent=False,
        )

        lengths = self.state.motion_num_frames
        lengths_shifted = lengths.roll(1)
        lengths_shifted[0] = 0
        self.register_buffer(
            "length_starts", lengths_shifted.cumsum(0), persistent=False
        )

        self.register_buffer(
            "motion_ids",
            torch.arange(
                len(self.state.motions), dtype=torch.long, device=self._device
            ),
            persistent=False,
        )

        objects_per_motion, motion_to_object_ids = self.parse_objects(object_names)

        self.register_buffer(
            "objects_per_motion",
            torch.tensor(objects_per_motion, device=self._device, dtype=torch.long),
            persistent=False,
        )

        self.register_buffer(
            "motion_to_object_ids",
            torch.tensor(motion_to_object_ids, device=self._device, dtype=torch.long),
            persistent=False,
        )

        self.to(device)

    def num_motions(self):
        return len(self.state.motions)

    def num_sub_motions(self):
        return self.state.motion_weights.shape[0]

    def get_total_length(self):
        return sum(self.state.motion_lengths)

    def get_total_trainable_length(self):
        return sum(self.state.motion_timings[:, 1] - self.state.motion_timings[:, 0])

    def get_motion(self, motion_id):
        return self.state.motions[motion_id]

    def sample_motions(self, n):
        sub_motion_ids = torch.multinomial(
            self.state.motion_weights, num_samples=n, replacement=True
        )
        return sub_motion_ids

    def sample_other_motions(self, already_chosen_ids: Tensor) -> Tensor:
        """
        Samples motion ids except for the id provided (batched).
        """
        n = already_chosen_ids.shape[0]
        motion_weights = self.state.motion_weights.unsqueeze(0).tile([n, 1])
        motion_weights = motion_weights.scatter(
            1, already_chosen_ids.unsqueeze(-1), torch.zeros_like(motion_weights)
        )
        sub_motion_ids = torch.multinomial(motion_weights, num_samples=1).squeeze(-1)
        return sub_motion_ids

    def sample_text_embeddings(self, sub_motion_ids: Tensor) -> Tensor:
        if hasattr(self.state, "text_embeddings"):
            indices = torch.randint(0, 3, (sub_motion_ids.shape[0],), device=self.device)
            return self.state.text_embeddings[sub_motion_ids, indices]
        return 0

    def sample_time(self, sub_motion_ids, truncate_time=None):
        phase = torch.rand(sub_motion_ids.shape, device=self.device)

        motion_len = self.state.motion_timings[sub_motion_ids, 1] - self.state.motion_timings[sub_motion_ids, 0]
        if truncate_time is not None:
            assert truncate_time >= 0.0
            motion_len -= truncate_time
            assert torch.all(motion_len >= 0)

        motion_time = phase * motion_len
        return motion_time + self.state.motion_timings[sub_motion_ids, 0]

    def get_sub_motion_length(self, sub_motion_ids):
        return self.state.motion_timings[sub_motion_ids, 1] - self.state.motion_timings[sub_motion_ids, 0]

    def get_motion_length(self, sub_motion_ids):
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]
        return self.state.motion_lengths[motion_ids]

    def get_mimic_motion_state(self, sub_motion_ids, motion_times, joint_3d_format="exp_map"):
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]

        motion_len = self.state.motion_lengths[motion_ids]
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        global_translation0 = self.gts[f0l]
        global_translation1 = self.gts[f1l]

        global_rotation0 = self.grs[f0l]
        global_rotation1 = self.grs[f1l]

        local_rotation0 = self.lrs[f0l]
        local_rotation1 = self.lrs[f1l]

        global_vel0 = self.gvs[f0l]
        global_vel1 = self.gvs[f1l]

        global_ang_vel0 = self.gavs[f0l]
        global_ang_vel1 = self.gavs[f1l]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        blend = blend.unsqueeze(-1)
        blend_exp = blend.unsqueeze(-1)

        global_translation: Tensor = (1.0 - blend_exp) * global_translation0 + blend_exp * global_translation1
        global_rotation: Tensor = torch_utils.slerp(global_rotation0, global_rotation1, blend_exp)

        local_rotation: Tensor = torch_utils.slerp(local_rotation0, local_rotation1, blend_exp)

        dof_pos: Tensor = self._local_rotation_to_dof(local_rotation, joint_3d_format)
        global_vel = (1.0 - blend_exp) * global_vel0 + blend_exp * global_vel1
        global_ang_vel = (1.0 - blend_exp) * global_ang_vel0 + blend_exp * global_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1

        global_translation[:, :, 2] += self.ref_height_adjust

        if not self.w_last:
            global_rotation = quat_w_first(global_rotation)

        return (
            global_translation,
            global_rotation,
            dof_pos,
            global_vel,
            global_ang_vel,
            dof_vel
        )

    def get_global_motion_state(self, sub_motion_ids, motion_times):
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]

        motion_len = self.state.motion_lengths[motion_ids]
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        global_translation0 = self.gts[f0l]
        global_translation1 = self.gts[f1l]

        global_rotation0 = self.grs[f0l]
        global_rotation1 = self.grs[f1l]

        global_vel = self.gvs[f0l]
        global_ang_vel = self.gavs[f0l]

        blend = blend.unsqueeze(-1).unsqueeze(-1)

        global_translation: Tensor = (1.0 - blend) * global_translation0 + blend * global_translation1
        global_rotation: Tensor = torch_utils.slerp(
            global_rotation0, global_rotation1, blend
        )

        global_translation[:, :, 2] += self.ref_height_adjust

        if not self.w_last:
            global_rotation = quat_w_first(global_rotation)

        return (
            global_translation,
            global_rotation,
            global_vel,
            global_ang_vel
        )

    def get_motion_root_pos(self, sub_motion_ids, motion_times, joint_3d_format="exp_map"):

        

        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]

        motion_len = self.state.motion_lengths[motion_ids]
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]


        vals = [
            root_pos0,
            root_pos1,

        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        root_pos: Tensor = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_pos[:, 2] += self.ref_height_adjust


        return root_pos
    
    def get_motion_state(self, sub_motion_ids, motion_times, joint_3d_format="exp_map"):

        

        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]

        motion_len = self.state.motion_lengths[motion_ids]
        num_frames = self.state.motion_num_frames[motion_ids]
        dt = self.state.motion_dt[motion_ids]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 + self.length_starts[motion_ids]
        f1l = frame_idx1 + self.length_starts[motion_ids]

        root_pos0 = self.gts[f0l, 0]
        root_pos1 = self.gts[f1l, 0]

        root_rot0 = self.grs[f0l, 0]
        root_rot1 = self.grs[f1l, 0]

        local_rot0 = self.lrs[f0l]
        local_rot1 = self.lrs[f1l]

        root_vel0 = self.grvs[f0l]
        root_vel1 = self.grvs[f1l]

        root_ang_vel0 = self.gravs[f0l]
        root_ang_vel1 = self.gravs[f1l]

        global_vel0 = self.gvs[f0l]
        global_vel1 = self.gvs[f1l]

        global_ang_vel0 = self.gavs[f0l]
        global_ang_vel1 = self.gavs[f1l]

        key_pos0 = self.gts[f0l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]
        key_pos1 = self.gts[f1l.unsqueeze(-1), self.key_body_ids.unsqueeze(0)]

        dof_vel0 = self.dvs[f0l]
        dof_vel1 = self.dvs[f1l]

        rb_pos0 = self.gts[f0l]
        rb_pos1 = self.gts[f1l]

        rb_rot0 = self.grs[f0l]
        rb_rot1 = self.grs[f1l]

        vals = [
            root_pos0,
            root_pos1,
            local_rot0,
            local_rot1,
            root_vel0,
            root_vel1,
            root_ang_vel0,
            root_ang_vel1,
            global_vel0,
            global_vel1,
            global_ang_vel0,
            global_ang_vel1,
            dof_vel0,
            dof_vel1,
            key_pos0,
            key_pos1,
            rb_pos0,
            rb_pos1,
            rb_rot0,
            rb_rot1,
        ]
        for v in vals:
            assert v.dtype != torch.float64

        blend = blend.unsqueeze(-1)

        root_pos: Tensor = (1.0 - blend) * root_pos0 + blend * root_pos1
        root_pos[:, 2] += self.ref_height_adjust

        root_rot: Tensor = torch_utils.slerp(root_rot0, root_rot1, blend)

        blend_exp = blend.unsqueeze(-1)
        key_pos = (1.0 - blend_exp) * key_pos0 + blend_exp * key_pos1
        key_pos[:, :, 2] += self.ref_height_adjust

        local_rot = torch_utils.slerp(local_rot0, local_rot1, torch.unsqueeze(blend, axis=-1))
        dof_pos: Tensor = self._local_rotation_to_dof(local_rot, joint_3d_format)
        root_vel = (1.0 - blend) * root_vel0 + blend * root_vel1
        root_ang_vel = (1.0 - blend) * root_ang_vel0 + blend * root_ang_vel1
        dof_vel = (1.0 - blend) * dof_vel0 + blend * dof_vel1
        rb_pos = (1.0 - blend_exp) * rb_pos0 + blend_exp * rb_pos1
        rb_rot = torch_utils.slerp(rb_rot0, rb_rot1, blend_exp)
        global_vel = (1.0 - blend_exp) * global_vel0 + blend_exp * global_vel1
        global_ang_vel = (1.0 - blend_exp) * global_ang_vel0 + blend_exp * global_ang_vel1

        if not self.w_last:
            root_rot = quat_w_first(root_rot)
            rb_rot = quat_w_first(rb_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, global_vel, global_ang_vel

    def _load_motions(self, motion_file, target_frame_rate):
        if self.create_text_embeddings:
            from transformers import AutoTokenizer, XCLIPTextModel
            model = XCLIPTextModel.from_pretrained("microsoft/xclip-base-patch32")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/xclip-base-patch32")

        motions = []
        motion_lengths = []
        motion_fpses = []
        motion_dt = []
        motion_num_frames = []
        text_embeddings = []
        has_text_embeddings = []

        (
            motion_files, motion_weights, motion_timings,
            motion_durations, sub_motion_to_motion, ref_respawn_offsets,
            motion_labels, supported_object_names
        ) = self._fetch_motion_files(motion_file)

        

        num_motion_files = len(motion_files)

        for f in range(num_motion_files):
            curr_file = motion_files[f]

            print(
                "Loading {:d}/{:d} motion files: {:s}".format(
                    f + 1, num_motion_files, curr_file
                )
            )

            curr_motion = SkeletonMotion.from_file(curr_file)

            


            if motion_durations[f] is not None:
                curr_motion = fix_motion_fps(curr_motion, motion_durations[f], target_frame_rate)
            
            # curr_motion = fix_heights(curr_motion)

            

            motion_fps = float(curr_motion.fps)
            curr_dt = 1.0 / motion_fps

            num_frames = curr_motion.tensor.shape[0]
            curr_len = 1.0 / motion_fps * (num_frames - 1)

            motion_fpses.append(motion_fps)
            motion_dt.append(curr_dt)
            motion_num_frames.append(num_frames)

            curr_dof_vels = self._compute_motion_dof_vels(curr_motion)
            curr_motion.dof_vels = curr_dof_vels

            motions.append(curr_motion)
            motion_lengths.append(curr_len)
        
        

        num_sub_motions = len(sub_motion_to_motion)

        for f in range(num_sub_motions):
            # Incase start/end weren't provided, set to (0, motion_length)
            motion_f = sub_motion_to_motion[f]

            

            if motion_timings[f][1] == -1:
                motion_timings[f][1] = motion_lengths[motion_f]

            motion_timings[f][1] = min(motion_timings[f][1],
                                       motion_lengths[motion_f])  # CT hack: fix small timing differences

            assert (motion_timings[f][0] < motion_timings[f][
                1]), f"Motion start {motion_timings[f][0]} >= motion end {motion_timings[f][1]} in motion {motion_f}"

            if self.create_text_embeddings:
                with torch.inference_mode():
                    inputs = tokenizer(motion_labels[f], padding=True, truncation=True, return_tensors="pt")
                    outputs = model(**inputs)
                    pooled_output = outputs.pooler_output  # pooled (EOS token) states
                    text_embeddings.append(pooled_output)  # should be [3, 512]
                    has_text_embeddings.append(True)
            else:
                text_embeddings.append(torch.zeros((3, 512), dtype=torch.float32))  # just hold something temporary
                has_text_embeddings.append(False)

        motion_lengths = torch.tensor(
            motion_lengths, device=self._device, dtype=torch.float32
        )

        motion_weights = torch.tensor(
            motion_weights, dtype=torch.float32, device=self._device
        )
        motion_weights /= motion_weights.sum()

        motion_timings = torch.tensor(
            motion_timings, dtype=torch.float32, device=self._device
        )

        sub_motion_to_motion = torch.tensor(
            sub_motion_to_motion, dtype=torch.long, device=self._device
        )

        ref_respawn_offsets = torch.tensor(
            ref_respawn_offsets, dtype=torch.float32, device=self._device
        )

        motion_fpses = torch.tensor(
            motion_fpses, device=self._device, dtype=torch.float32
        )
        motion_dt = torch.tensor(motion_dt, device=self._device, dtype=torch.float32)
        motion_num_frames = torch.tensor(motion_num_frames, device=self._device)

        text_embeddings = torch.stack(text_embeddings).detach().to(device=self._device)
        has_text_embeddings = torch.tensor(has_text_embeddings, dtype=torch.bool, device=self._device)

        

        self.state = LoadedMotions(
            motions=tuple(motions),
            motion_lengths=motion_lengths,
            motion_weights=motion_weights,
            motion_timings=motion_timings,
            motion_fps=motion_fpses,
            motion_dt=motion_dt,
            motion_num_frames=motion_num_frames,
            motion_files=tuple(motion_files),
            sub_motion_to_motion=sub_motion_to_motion,
            ref_respawn_offsets=ref_respawn_offsets,
            text_embeddings=text_embeddings,
            has_text_embeddings=has_text_embeddings,
            supported_object_names=supported_object_names,
        )
        
        # temp = motion_files[0].split("/")[2].split("_")
        # print(f"motion_{temp[1]}_{temp[2]}.pt")
        # import pdb;pdb.set_trace()
        # torch.save(self.state, f"motion_{temp[1]}_{temp[2]}.pt")

        num_motions = self.num_motions()
        total_len = self.get_total_length()

        print(
            "Loaded {:d} motions with a total length of {:.3f}s.".format(
                num_motions, total_len
            )
        )

        num_sub_motions = self.num_sub_motions()
        total_trainable_len = self.get_total_trainable_length()

        print(
            "Loaded {:d} sub motions with a total trainable length of {:.3f}s.".format(
                num_sub_motions, total_trainable_len
            )
        )

    def _fetch_motion_files(self, motion_file):
        ext = os.path.splitext(motion_file)[1]
        if ext == ".yaml":
            dir_name = os.path.dirname(motion_file)
            motion_files = []
            sub_motion_to_motion = []
            ref_respawn_offsets = []
            motion_weights = []
            motion_timings = []
            motion_durations = []
            motion_labels = []
            supported_object_names = []

            with open(os.path.join(os.getcwd(), motion_file), "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)

            motion_list = sorted(motion_config["motions"], key=lambda x: 1e6 if not "idx" in x else int(x["idx"]))

            motion_index = 0

            for motion_id, motion_entry in enumerate(motion_list):
                curr_file = motion_entry["file"]
                curr_file = os.path.join(dir_name, curr_file)
                motion_files.append(curr_file)

                if "dur" in motion_entry:
                    motion_durations.append(motion_entry["dur"])
                else:
                    motion_durations.append(None)

                if "sub_motions" not in motion_entry:
                    motion_entry["sub_motions"] = [deepcopy(motion_entry)]
                    motion_entry["sub_motions"][0]["idx"] = motion_index

                for sub_motion in sorted(motion_entry["sub_motions"], key=lambda x: int(x["idx"])):
                    curr_weight = sub_motion["weight"]
                    assert curr_weight >= 0

                    assert motion_index == sub_motion["idx"]

                    motion_weights.append(curr_weight)

                    sub_motion_to_motion.append(motion_id)

                    ref_respawn_offset = 0
                    if "ref_respawn_offset" in sub_motion:
                        ref_respawn_offset = sub_motion["ref_respawn_offset"]

                    ref_respawn_offsets.append(ref_respawn_offset)

                    if "timings" in sub_motion:
                        curr_timing = sub_motion["timings"]
                        start = curr_timing["start"]
                        end = curr_timing["end"]
                    else:
                        start = 0
                        end = -1

                    motion_timings.append([start, end])

                    sub_motion_labels = []
                    if "labels" in sub_motion:
                        for label in sub_motion["labels"]:
                            sub_motion_labels.append(label)
                            if len(sub_motion_labels) == 3:
                                break
                        if len(sub_motion_labels) == 0:
                            sub_motion_labels.append("")
                        while len(sub_motion_labels) < 3:
                            sub_motion_labels.append(sub_motion_labels[-1])
                    else:
                        sub_motion_labels.append("")
                        sub_motion_labels.append("")
                        sub_motion_labels.append("")

                    motion_labels.append(sub_motion_labels)

                    if "supported_objects" in sub_motion:
                        supported_object_names.append([
                            sub_motion["object_category"] + "_" + supported_object for supported_object in sub_motion["supported_objects"]
                        ])
                    else:
                        supported_object_names.append(None)

                    motion_index += 1
        else:
            motion_files = [motion_file]
            motion_weights = [1.0]
            motion_timings = [[0, -1]]
            motion_durations = [None]
            sub_motion_to_motion = [0]
            ref_respawn_offsets = [0]
            motion_labels = [[]]
            supported_object_names = [None]

        return (motion_files, motion_weights, motion_timings, motion_durations, sub_motion_to_motion,
                ref_respawn_offsets, motion_labels, supported_object_names)

    def _calc_frame_blend(self, time, len, num_frames, dt):

        phase = time / len
        # assert torch.all(phase >= 0)
        # assert torch.all(phase <= 1)
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0 = (phase * (num_frames - 1)).long()
        frame_idx1 = torch.min(frame_idx0 + 1, num_frames - 1)
        blend = (time - frame_idx0 * dt) / dt

        return frame_idx0, frame_idx1, blend

    def _get_num_bodies(self):
        motion = self.get_motion(0)
        num_bodies = motion.num_joints
        return num_bodies

    def _compute_motion_dof_vels(self, motion: SkeletonMotion):
        num_frames = motion.tensor.shape[0]
        dt = 1.0 / motion.fps
        dof_vels = []

        for f in range(num_frames - 1):
            local_rot0 = motion.local_rotation[f]
            local_rot1 = motion.local_rotation[f + 1]
            frame_dof_vel = self._local_rotation_to_dof_vel(local_rot0, local_rot1, dt)
            frame_dof_vel = frame_dof_vel
            dof_vels.append(frame_dof_vel)

        dof_vels.append(dof_vels[-1])
        dof_vels = torch.stack(dof_vels, dim=0)

        return dof_vels

    # jp hack
    # get rid of this ASAP, need a proper way of projecting from max coords to reduced coords
    def _local_rotation_to_dof(self, local_rot, joint_3d_format):
        body_ids = self.dof_body_ids
        dof_offsets = self.dof_offsets

        n = local_rot.shape[0]
        dof_pos = torch.zeros((n, self.num_dof), dtype=torch.float, device=self._device)

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_q = local_rot[:, body_id]
                if joint_3d_format == "exp_map":
                    formatted_joint = torch_utils.quat_to_exp_map(joint_q, w_last=True)
                elif joint_3d_format == "xyz":
                    x, y, z = rotations.get_euler_xyz(joint_q, w_last=True)
                    formatted_joint = torch.stack([x, y, z], dim=-1)
                else:
                    raise ValueError(f"Unknown 3d format '{joint_3d_format}'")

                dof_pos[:, joint_offset: (joint_offset + joint_size)] = formatted_joint
            elif joint_size == 1:
                joint_q = local_rot[:, body_id]
                joint_theta, joint_axis = torch_utils.quat_to_angle_axis(joint_q, w_last=True)
                joint_theta = (
                        joint_theta * joint_axis[..., 1]
                )  # assume joint is always along y axis

                joint_theta = rotations.normalize_angle(joint_theta)
                dof_pos[:, joint_offset] = joint_theta

            else:
                print("Unsupported joint type")
                assert False

        return dof_pos

    def _local_rotation_to_dof_vel(self, local_rot0, local_rot1, dt):
        body_ids = self.dof_body_ids
        dof_offsets = self.dof_offsets

        dof_vel = torch.zeros([self.num_dof], device=self._device)

        diff_quat_data = quat_mul_norm(quat_inverse(local_rot0), local_rot1)
        diff_angle, diff_axis = quat_angle_axis(diff_quat_data)
        local_vel = diff_axis * diff_angle.unsqueeze(-1) / dt
        local_vel = local_vel

        for j in range(len(body_ids)):
            body_id = body_ids[j]
            joint_offset = dof_offsets[j]
            joint_size = dof_offsets[j + 1] - joint_offset

            if joint_size == 3:
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset: (joint_offset + joint_size)] = joint_vel

            elif joint_size == 1:
                assert joint_size == 1
                joint_vel = local_vel[body_id]
                dof_vel[joint_offset] = joint_vel[
                    1
                ]  # assume joint is always along y axis

            else:
                print("Unsupported joint type")
                assert False

        return dof_vel

    def parse_objects(self, object_names):
        # If motions may have an object, create the mapping to allow sampling objects for motions.
        motion_to_object_ids = []
        objects_per_motion = []
        if hasattr(self.state, "supported_object_names") and object_names is not None:
            def indices(lst, element):
                result = []
                offset = -1
                while True:
                    try:
                        offset = lst.index(element, offset + 1)
                    except ValueError:
                        return result
                    result.append(offset)

            max_num_objects = max(max([len(object_names) if object_names is not None else 0 for object_names in self.state.supported_object_names]), len(object_names))

            for i in range(len(self.state.supported_object_names)):
                if self.state.supported_object_names[i] is None:
                    motion_to_object_ids.append([-1] * max_num_objects)
                    objects_per_motion.append(-1)
                else:
                    all_object_ids = []
                    for object_name in self.state.supported_object_names[i]:
                        if object_name in object_names:
                            # store all indices that match, multiple options may exist
                            object_ids = indices(object_names, object_name)
                            for object_id in object_ids:
                                all_object_ids.append(object_id)

                    objects_per_motion.append(len(all_object_ids))

                    if len(all_object_ids) == 0:
                        all_object_ids = [-1]
                    while len(all_object_ids) < max_num_objects:
                        all_object_ids.append(-1)
                    motion_to_object_ids.append(all_object_ids)

        return objects_per_motion, motion_to_object_ids


def fix_motion_fps(motion, dur, target_frame_rate):
    true_fps = round(motion.local_rotation.shape[0] / dur)

    skip = int(true_fps / target_frame_rate)

    lr = motion.local_rotation[::skip]
    rt = motion.root_translation[::skip]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        motion.skeleton_tree,
        lr,
        rt,
        is_local=True,
    )
    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=target_frame_rate)

    return new_motion

def fix_heights(motion):
    body_heights = motion.global_translation[..., 2]
    min_height = body_heights.min()
    # 
    new_root_translation = motion.root_translation.clone()
    new_root_translation[...,2] -= min_height
    new_sk_state = SkeletonState.from_rotation_and_root_translation(
            motion.skeleton_tree,
            motion.local_rotation,
            new_root_translation,
            is_local=True,
        )

    new_motion = SkeletonMotion.from_skeleton_state(new_sk_state, fps=motion.fps)

    return new_motion
