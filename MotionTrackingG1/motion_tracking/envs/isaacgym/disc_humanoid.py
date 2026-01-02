# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import numpy as np
import torch
import struct
from motion_tracking.envs.isaacgym.humanoid import Humanoid

from motion_tracking.envs.common.common_disc import BaseDisc
from isaac_utils import torch_utils

from motion_tracking.utils.motion_lib import MotionLib
from hydra.utils import instantiate

from typing import List, Optional, Tuple, Union, Dict
from poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState


class DiscHumanoid(BaseDisc, Humanoid):
    def __init__(
        self, config, device: torch.device, motion_lib: Optional[MotionLib] = None
    ):
        super().__init__(config, device, motion_lib)


    ###############################################################
    # Handle Resets
    ###############################################################
    def set_env_state(
        self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, rb_pos, rb_rot, rb_vel, rb_ang_vel
    ):
        self.humanoid_root_states[env_ids, 0:3] = root_pos
        self.humanoid_root_states[env_ids, 3:7] = root_rot
        self.humanoid_root_states[env_ids, 7:10] = root_vel
        self.humanoid_root_states[env_ids, 10:13] = root_ang_vel

        self.rigid_body_pos[env_ids] = rb_pos
        self.rigid_body_rot[env_ids] = rb_rot
        self.rigid_body_vel[env_ids] = rb_vel
        self.rigid_body_ang_vel[env_ids] = rb_ang_vel

        self.reset_states = {
            "root_pos": root_pos.clone(),
            "root_rot": root_rot.clone(),
            "root_vel": root_vel.clone(),
            "root_ang_vel": root_ang_vel.clone(),
            "dof_pos": dof_pos.clone(),
            "dof_vel": dof_vel.clone(),
            "rb_pos": rb_pos.clone(),
            "rb_rot": rb_rot.clone(),
            "rb_vel": rb_vel.clone(),
            "rb_ang_vel": rb_ang_vel.clone(),
        }

        self.dof_pos[env_ids] = dof_pos
        self.dof_vel[env_ids] = dof_vel

    ###############################################################
    # Helpers
    ###############################################################
    def instantiate_motion_lib(self, motion_lib: MotionLib):
        if motion_lib is None:
            motion_lib: MotionLib = instantiate(
                self.config.motion_lib,
                dof_body_ids=self.dof_body_ids,
                dof_offsets=self.dof_offsets,
                key_body_ids=self.key_body_ids,
                device=self.device,
                object_names=self.spawned_object_names,
                body_names=self.body_names,
                dof_names=self.dof_names,
            )
        return motion_lib
