# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import copy

from motion_tracking.utils.motion_lib import MotionLib, quat_w_first
import torch
import numpy as np

from copy import deepcopy

from isaac_utils import torch_utils, rotations

import os
import yaml

from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
from poselib.core.rotation3d import *

from torch import Tensor


curve_parameters = {
    "walk": {  # works great with only pelvis full
        "speed": 1.3,
        "acceleration": 1.0,
        "head_height": 1.48,
        "pelvis_height": 0.89,
        "pelvis_delay_frames": 0,
        # TODO:
        "hands_height": 0,
        "hands_distance": 0,
        "hands_gap": 0
    },
    "run": {  # best without head constraint
        "speed": 3.5,
        "acceleration": 1,
        "deceleration": 1,  # todo add support for this
        "head_height": 1.48,
        "pelvis_height": 0.89,
        "pelvis_delay_frames": 1,
        # TODO:
        "hands_height": 0,
        "hands_distance": 0,
        "hands_gap": 0
    },
    "grumpy": {
        "speed": 1.5,
        "acceleration": 1,
        "head_height": 1.3,
        "pelvis_height": 0.89,
        "pelvis_delay_frames": 4,
        # TODO:
        "hands_height": 0,
        "hands_distance": 0,
        "hands_gap": 0
    },
    "crawl": {
        # use with pelvis full, head position
        # rotate pelvis
        "speed": 0.45,
        "acceleration": 0.5,
        "head_height": 0.4,
        "pelvis_height": 0.4,
        "pelvis_delay_frames": 40,
        "hands_height": 0.2,
        "hands_distance": 0.65,
        "hands_gap": 0.5
    },
    "crawl_4s": {
        # works well on the i
        # use with occulus_pelvis
        "speed": 1.3,
        "acceleration": 0.8,
        "head_height": 0.7,
        "pelvis_height": 0.6,
        "pelvis_delay_frames": 15,
        "hands_height": 0.2,
        "hands_distance": 0.6,
        "hands_gap": 0.5
    },
    "zombie": {
        # use with vr
        "speed": 1.5,
        "acceleration": 1.0,
        "head_height": 1.48,
        "pelvis_height": 0.89,
        "pelvis_delay_frames": 5,
        "hands_height": 1.3,
        "hands_distance": 0.6,
        "hands_gap": 0.8
    },
    "hands_up": {
        # use with pelvis full, head position
        "speed": 1.4,
        "acceleration": 1.0,
        "head_height": 1.4,
        "pelvis_height": 0.89,
        "pelvis_delay_frames": 5,
        "hands_height": 1.7,
        "hands_distance": 0.3,
        "hands_gap": 0.4
    },
    "hands_head": {
        # use with pelvis full, head position
        "speed": 1.4,
        "acceleration": 1.0,
        "head_height": 1.45,
        "pelvis_height": 0.89,
        "pelvis_delay_frames": 4,
        "hands_height": 1.6,
        "hands_distance": 0.3,
        "hands_gap": 0.2
    },
    "crouch": {  # use with "pelvis full, head position"
        "speed": 1.2,
        "acceleration": 1.0,
        "head_height": 1.,
        "pelvis_height": 0.7,
        "pelvis_delay_frames": 5,
        # TODO:
        "hands_height": 0,
        "hands_distance": 0,
        "hands_gap": 0
    },
    "hands_side": {
        # use with pelvis full, head position
        "speed": 1.4,
        "acceleration": 1.0,
        "head_height": 1.4,
        "pelvis_height": 0.89,
        "pelvis_delay_frames": 0,
        "hands_height": 1.2,
        "hands_distance": 0,
        "hands_gap": 1.2
    },
}

letter_standing_frames = [
    [0],  # S
    [0],  # I
    [0],  # G
    [0],  # G
    [0],  # R
    [0],  # A
    [0],  # P
    [0],  # H
]

letter_offsets = [
    [0, 0],  # S
    [-7, 0.5],  # I
    [-4, 0],  # G
    [-3.5, 0],  # G
    [-9, -9],  # R
    [-8.5, -9],  # A
    [-7, -9],  # P
    [-7, 0],  # H
]

# TODO: add hands up for walking G
LETTER_CONTROL = [
    ["hands_head", {"pelvis": 1, "head": 0}],  # S
    ["walk", {"pelvis": 1, "head": 0}],  # I ["zombie", {"pelvis": 1, "head": 0}],  # I
    ["run", {"pelvis": 1}],  # G
    ["zombie", {"pelvis": 1}],  # G
    ["crouch", {"pelvis": 1, "head": 0}],  # R
    ["hands_side", {"pelvis": 1, "head": 0}],  # A
    ["walk", {"head": 1}],  # P
    ["hands_up", {"pelvis": 1, "head": 0}],  # H
]

STANDSTILL_FRAMES = 20

class MimicMotionLib(MotionLib):
    def __init__(
            self,
            motion_file,
            dof_body_ids,
            dof_offsets,
            key_body_ids,
            num_envs,
            device="cpu",
            ref_height_adjust: float = 0,
            target_frame_rate: int = 30,
            w_last: bool = True,
            create_text_embeddings: bool = False,
            object_names: List[str] = None,
    ):
        super().__init__(
            motion_file,
            dof_body_ids,
            dof_offsets,
            key_body_ids,
            device,
            ref_height_adjust,
            target_frame_rate,
            w_last,
            create_text_embeddings,
            object_names
        )

        all_letters = torch.from_numpy(np.load("motion_tracking/data/motions/siggraph_letters.npy")).to(self.device) * 0.1

        self.num_envs = num_envs

        letter_mappings = [
            [None, 12],  # S 0
            [12, 14],  # I 1
            [14, 30],  # G 2
            [30, 46],  # G 3
            [46, 60],  # R 4
            [60, 70],  # A 5
            [70, 83],  # P 6
            [83, None],  # H 7
        ]

        self.all_letters = []
        self.all_letters_hands = []
        self.all_masks = []

        global LETTER_CONTROL, letter_standing_frames, letter_offsets
        fixed_letter_idx = 4
        letter_mappings = [letter_mappings[fixed_letter_idx]]
        LETTER_CONTROL = [LETTER_CONTROL[fixed_letter_idx]]
        letter_standing_frames = [letter_standing_frames[fixed_letter_idx]]
        letter_offsets = [letter_offsets[fixed_letter_idx]]

        original_num_letters = len(LETTER_CONTROL)
        for idx in range(max(self.num_envs - original_num_letters, 0)):
            letter_idx = idx % original_num_letters
            LETTER_CONTROL.append(copy.deepcopy(LETTER_CONTROL[letter_idx]))
            letter_mappings.append(copy.deepcopy(letter_mappings[letter_idx]))
            letter_standing_frames.append(copy.deepcopy(letter_standing_frames[letter_idx]))
            letter_offsets.append(copy.deepcopy(letter_offsets[letter_idx]))

        max_letter_length = 0

        all_letters_neutral_hands = []

        for env_id in range(self.num_envs):
            letter_idx = env_id % len(letter_mappings)
            (start, end) = letter_mappings[letter_idx]

            motion = LETTER_CONTROL[letter_idx][0]

            if start is None:
                letter = all_letters[:end].clone()
            elif end is None:
                letter = all_letters[start:].clone()
            else:
                letter = all_letters[start:end].clone()

            letter[:] -= letter[0].clone()

            neutral_hands = torch.zeros((2, 3), device=self.device, dtype=torch.float64)

            neutral_hands[0, 0] = curve_parameters[motion]["hands_distance"]
            neutral_hands[0, 1] = curve_parameters[motion]["hands_gap"] * 1. / 2
            neutral_hands[0, 2] = curve_parameters[motion]["hands_height"]

            neutral_hands[1, 0] = curve_parameters[motion]["hands_distance"]
            neutral_hands[1, 1] = -curve_parameters[motion]["hands_gap"] * 1. / 2
            neutral_hands[1, 2] = curve_parameters[motion]["hands_height"]

            all_letters_neutral_hands.append(neutral_hands)

            initial_position = letter_offsets[letter_idx]
            letter[:, 0] += initial_position[0]
            letter[:, 1] += initial_position[1]

            # initial_position = self.gts[letter_idx][0].clone()
            # letter[:] += initial_position[:2]

            interpolated_letter = []

            speed = 0
            current_point = letter[0]
            interpolated_letter.append(current_point.clone())
            for point_idx, point in enumerate(letter):
                if point_idx in letter_standing_frames[letter_idx] or point_idx == letter.shape[0] - 1:
                    print(f"Stand still {point_idx}")
                    for _ in range(STANDSTILL_FRAMES):
                        current_point = point
                        interpolated_letter.append(current_point.clone())

                    if point_idx == letter.shape[0] - 1:
                        continue

                if point_idx + 1 not in letter_standing_frames[letter_idx] and point_idx + 1 != letter.shape[0] - 1:
                    print(f"Accelerate \ move {point_idx}")
                    distance = torch.norm(point - letter[point_idx + 1])
                    while distance > 0:  # keep running until we reach the next point
                        direction = (letter[point_idx + 1] - current_point) / distance

                        if speed < curve_parameters[motion]["speed"]:
                            # Accelerate
                            speed = min(
                                speed + curve_parameters[motion]["acceleration"] * self.state.motion_dt[0],
                                curve_parameters[motion]["speed"]
                            )

                        current_point += direction * speed * self.state.motion_dt[0]
                        interpolated_letter.append(current_point.clone())

                        distance -= speed * self.state.motion_dt[0]

                    continue

                if point_idx + 1 in letter_standing_frames[letter_idx] or point_idx + 1 == letter.shape[0] - 1:
                    print(f"Decelerate {point_idx}")
                    # Next point is standing still, let's decelerate in time
                    distance = torch.norm(point - letter[point_idx + 1])
                    # x = v0 * t + 0.5 * a * t^2
                    deceleration_frames = int(speed / curve_parameters[motion]["acceleration"])
                    deceleration_distance = speed * deceleration_frames + 0.5 * curve_parameters[motion]["acceleration"] * deceleration_frames ** 2
                    deceleration_distance = min(deceleration_distance, distance)

                    while distance > deceleration_distance:
                        direction = (letter[point_idx + 1] - current_point) / distance

                        if speed < curve_parameters[motion]["speed"]:
                            # Accelerate
                            speed = min(
                                speed + curve_parameters[motion]["acceleration"] * self.state.motion_dt[0],
                                curve_parameters[motion]["speed"]
                            )

                        current_point += direction * speed * self.state.motion_dt[0]
                        interpolated_letter.append(current_point.clone())

                        distance -= speed * self.state.motion_dt[0]

                    while distance > 0 and speed > 0:  # keep running until we reach the next point
                        direction = (letter[point_idx + 1] - current_point) / distance

                        # Decelerate
                        speed = max(
                            speed - curve_parameters[motion]["acceleration"] * self.state.motion_dt[0],
                            0
                        )

                        current_point += direction * speed * self.state.motion_dt[0]
                        interpolated_letter.append(current_point.clone())

                        distance -= speed * self.state.motion_dt[0]

                    continue

            self.all_letters.append(torch.stack(interpolated_letter, dim=0))
            max_letter_length = max(max_letter_length, self.all_letters[-1].shape[0])

            single_dir_window_size = 20
            for i in range(self.all_letters[-1].shape[0]):
                min_i = max(0, i - single_dir_window_size)
                max_i = min(self.all_letters[-1].shape[0], i + single_dir_window_size)
                average_x = self.all_letters[-1][min_i:max_i, 0].mean()
                average_y = self.all_letters[-1][min_i:max_i, 1].mean()
                self.all_letters[-1][i, 0] = average_x
                self.all_letters[-1][i, 1] = average_y

        self.all_letters_length = max_letter_length
        self.state.motion_timings[:, 0] = 0
        self.state.motion_timings[:, 1] = (self.all_letters_length - 1) * self.state.motion_dt[0]
        self.state.motion_num_frames[:] = self.all_letters_length
        self.state.motion_lengths[:] = (self.all_letters_length - 1) * self.state.motion_dt[0]

        for idx, letter in enumerate(self.all_letters):
            if letter.shape[0] < self.all_letters_length:
                letter = torch.cat([letter, letter[-1].unsqueeze(0).repeat(self.all_letters_length - letter.shape[0], 1)], dim=0)
                self.all_letters[idx] = letter
        self.all_letters = torch.stack(self.all_letters, dim=0)

        angles = []
        # initial_angle = torch_utils.calc_heading(self.grs[:self.num_envs][0].clone().view(1, -1), w_last=True)
        # neg = initial_angle < 0
        # initial_angle[neg] += 2 * torch.pi
        # angles.append(initial_angle.view(-1).repeat(self.all_letters.shape[0]))

        # for idx in range(1, self.all_letters_length):
        for idx in range(self.all_letters_length):
            if idx == 0:
                vector = self.all_letters[:, 1] - self.all_letters[:, 0]
            else:
                vector = self.all_letters[:, idx] - self.all_letters[:, idx - 1]
            angle = rotations.vec_to_heading(vector).view(-1)
            # angle = rotations.vec_to_heading(-vector).view(-1)  # TO GO IN REVERSE
            neg = angle < 0
            angle[neg] += 2 * torch.pi
            angles.append(angle)

        all_angles = torch.stack(angles, dim=1)

        # single_dir_window_size = 20
        # swirl = 0
        # swirl_rate = torch.pi / 20
        # for i in range(all_angles.shape[1]):
        #     min_i = max(0, i - single_dir_window_size)
        #     max_i = min(all_angles.shape[1], i + single_dir_window_size)
        #
        #     current_angles = all_angles[:, min_i:max_i]
        #     sin_angles = torch.sin(current_angles).sum(dim=-1)
        #     cos_angles = torch.cos(current_angles).sum(dim=-1)
        #
        #     average_val = torch.atan2(sin_angles, cos_angles)
        #
        #     # average_val = all_angles[:, min_i:max_i].mean(dim=-1)
        #
        #     all_angles[:, i] = average_val + swirl
        #     swirl = (swirl + swirl_rate) % (2 * torch.pi)

        self.all_letters_direction = rotations.heading_to_quat(all_angles, w_last=True)

        all_letters_neutral_hands = torch.stack(all_letters_neutral_hands, dim=0)
        all_letters_direction_expanded = self.all_letters_direction.unsqueeze(-2).repeat(1, 1, all_letters_neutral_hands.shape[-2], 1).view(-1, 4)
        neutral_hands_full = all_letters_neutral_hands.view(-1, 1, 2, 3).repeat(1, self.all_letters.shape[1], 1, 1)
        shape = neutral_hands_full.shape
        # rotate neutral_hands_full to face all letters direction
        self.hands = rotations.quat_rotate(all_letters_direction_expanded, neutral_hands_full.view(-1, 3), w_last=True).view(*shape)
        self.hands[..., :2] += self.all_letters.unsqueeze(-2).repeat(1, 1, 2, 1)

        # downwards_tilt = rotations.quat_from_euler_xyz(torch.tensor(0., device=self.device), torch.tensor(torch.pi / 6, device=self.device), torch.tensor(0., device=self.device), w_last=True).view(1, 1, -1).repeat(self.all_letters.shape[0], self.all_letters.shape[1], 1)
        # self.all_letters_direction = rotations.quat_mul(self.all_letters_direction, downwards_tilt, w_last=True)

        try:
            all_letters = ["S", "I", "G", "G", "R", "A", "P", "H"]
            name = f"letter_{all_letters[fixed_letter_idx]}"
        except:
            name = "all_letters"
        torch.save(self.all_letters, f"{name}.pt")

    def get_sub_motion_length(self, sub_motion_ids):
        tmp_len = self.state.motion_timings[sub_motion_ids, 1] - self.state.motion_timings[sub_motion_ids, 0]

        return tmp_len * 0 + self.all_letters_length * self.state.motion_dt[0]

    def get_motion_length(self, sub_motion_ids):
        motion_ids = self.state.sub_motion_to_motion[sub_motion_ids]
        tmp_len = self.state.motion_lengths[motion_ids]

        return tmp_len * 0 + self.all_letters_length * self.state.motion_dt[0]

    def get_mimic_motion_state(self, sub_motion_ids, motion_times, joint_3d_format="exp_map"):
        fmod_idx = torch.fmod(sub_motion_ids, self.all_letters.shape[0]).long()

        motion_len = self.state.motion_lengths[fmod_idx]
        num_frames = self.state.motion_num_frames[fmod_idx]
        dt = self.state.motion_dt[fmod_idx]

        frame_idx0, frame_idx1, blend = self._calc_frame_blend(
            motion_times, motion_len, num_frames, dt
        )

        f0l = frame_idx0 #+ self.length_starts[motion_ids]
        f1l = frame_idx1 #+ self.length_starts[motion_ids]

        global_translation0 = self.gts[f0l * 0]
        global_translation1 = self.gts[f1l * 0]

        global_vel = self.gvs[f0l * 0]
        global_ang_vel = self.gavs[f0l * 0]

        global_rotation0 = self.grs[f0l * 0]
        global_rotation1 = self.grs[f1l * 0]

        for idx in range(sub_motion_ids.shape[0]):
            # Head position
            global_translation0[idx, 13, :2] = self.all_letters[fmod_idx[idx], f0l[idx]]
            global_translation0[idx, 13, 2] = curve_parameters[LETTER_CONTROL[fmod_idx[idx]][0]]["head_height"]
            global_translation1[idx, 13, :2] = self.all_letters[fmod_idx[idx], f1l[idx]]
            global_translation1[idx, 13, 2] = curve_parameters[LETTER_CONTROL[fmod_idx[idx]][0]]["head_height"]

            # Pelvis delay and position
            pelvis_f0l = torch.max(f0l[idx] - curve_parameters[LETTER_CONTROL[fmod_idx[idx]][0]]["pelvis_delay_frames"], torch.zeros_like(f0l[idx]))
            pelvis_f0l = torch.min(pelvis_f0l, torch.tensor(self.all_letters_length - 1, device=self.device))
            pelvis_f1l = torch.max(f1l[idx] - curve_parameters[LETTER_CONTROL[fmod_idx[idx]][0]]["pelvis_delay_frames"], torch.zeros_like(f1l[idx]))
            pelvis_f1l = torch.min(pelvis_f1l, torch.tensor(self.all_letters_length - 1, device=self.device))

            global_translation0[idx, 0, :2] = self.all_letters[fmod_idx[idx], pelvis_f0l]
            global_translation0[idx, 0, 2] = curve_parameters[LETTER_CONTROL[fmod_idx[idx]][0]]["pelvis_height"]
            global_translation1[idx, 0, :2] = self.all_letters[fmod_idx[idx], pelvis_f1l]
            global_translation1[idx, 0, 2] = curve_parameters[LETTER_CONTROL[fmod_idx[idx]][0]]["pelvis_height"]

            # Hands position, L_Hand 18, R_Hand 23
            global_translation0[idx, 18] = self.hands[fmod_idx[idx], pelvis_f0l, 0]
            global_translation1[idx, 18] = self.hands[fmod_idx[idx], pelvis_f1l, 0]

            global_translation0[idx, 23] = self.hands[fmod_idx[idx], pelvis_f0l, 1]
            global_translation1[idx, 23] = self.hands[fmod_idx[idx], pelvis_f1l, 1]

            # Pelvis rotation
            global_rotation0[idx, 0] = self.all_letters_direction[fmod_idx[idx], f0l[idx]]
            global_rotation1[idx, 0] = self.all_letters_direction[fmod_idx[idx], f1l[idx]]

            # Head rotation
            global_rotation0[idx, 13] = self.all_letters_direction[fmod_idx[idx], f0l[idx]]
            global_rotation1[idx, 13] = self.all_letters_direction[fmod_idx[idx], f1l[idx]]

        local_rotation0 = self.lrs[f0l * 0]
        local_rotation1 = self.lrs[f1l * 0]

        dof_vel = self.dvs[f0l * 0]

        blend = blend.unsqueeze(-1).unsqueeze(-1)

        global_translation: Tensor = (1.0 - blend) * global_translation0 + blend * global_translation1
        global_rotation: Tensor = torch_utils.slerp(
            global_rotation0, global_rotation1, blend
        )
        local_rotation: Tensor = torch_utils.slerp(
            local_rotation0, local_rotation1, blend
        )
        dof_pos: Tensor = self._local_rotation_to_dof(local_rotation, joint_3d_format)

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

    def get_motion_state(self, sub_motion_ids, motion_times, joint_3d_format="exp_map"):
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, rb_pos, rb_rot, global_vel, global_ang_vel = super().get_motion_state(sub_motion_ids, motion_times, joint_3d_format)

        inv_heading = torch_utils.calc_heading_quat_inv(root_rot, w_last=True)
        self_rotated_rot = rotations.quat_mul(inv_heading, root_rot, w_last=True)

        target_heading = torch_utils.calc_heading_quat(self.all_letters_direction[sub_motion_ids, 0], w_last=True).type(inv_heading.dtype)

        root_rot = rotations.quat_mul(target_heading, self_rotated_rot, w_last=True)

        self_rotated_root_vel = torch_utils.quat_rotate(inv_heading, root_vel, w_last=True)
        root_vel = torch_utils.quat_rotate(target_heading, self_rotated_root_vel, w_last=True)

        self_rotated_root_ang_vel = torch_utils.quat_rotate(inv_heading, root_ang_vel, w_last=True)
        root_ang_vel = torch_utils.quat_rotate(target_heading, self_rotated_root_ang_vel, w_last=True)

        inv_heading_rot_expand = inv_heading.unsqueeze(-2)
        inv_heading_rot_expand = inv_heading_rot_expand.repeat((1, rb_pos.shape[1], 1))

        flat_heading_rot = inv_heading_rot_expand.reshape(
            inv_heading_rot_expand.shape[0] * inv_heading_rot_expand.shape[1],
            inv_heading_rot_expand.shape[2]
        )

        target_heading_rot_expand = target_heading.unsqueeze(-2)
        target_heading_rot_expand = target_heading_rot_expand.repeat((1, rb_pos.shape[1], 1))

        flat_target_heading_rot = target_heading_rot_expand.reshape(
            target_heading_rot_expand.shape[0] * target_heading_rot_expand.shape[1],
            target_heading_rot_expand.shape[2]
        )

        root_pos_expand = root_pos.unsqueeze(-2)

        local_body_pos = rb_pos - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(
            local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
        )
        flat_local_body_pos = rotations.quat_rotate(
            flat_heading_rot, flat_local_body_pos, w_last=True
        )
        body_pos = torch_utils.quat_rotate(flat_target_heading_rot, flat_local_body_pos, w_last=True)
        global_body_pos = body_pos.reshape(rb_pos.shape) + root_pos_expand  # rb_pos

        flat_body_rot = rb_rot.reshape(
            rb_rot.shape[0] * rb_rot.shape[1], rb_rot.shape[2]
        )
        flat_local_body_rot = rotations.quat_mul(flat_heading_rot, flat_body_rot, w_last=True)
        flat_body_rot = rotations.quat_mul(flat_target_heading_rot, flat_local_body_rot, w_last=True)
        body_rot = flat_body_rot.reshape(rb_rot.shape) # rb_rot

        if not self.w_last:
            root_rot = quat_w_first(root_rot)
            body_rot = quat_w_first(body_rot)

        return root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos, global_body_pos, body_rot, global_vel, global_ang_vel