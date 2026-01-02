# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from isaacgym import gymapi, gymtorch
from isaac_utils import torch_utils, rotations

from envs.isaacgym.task_humanoid import TaskHumanoid
from envs.isaacgym.disc_humanoid import DiscHumanoid
from envs.isaacgym.humanoid import compute_humanoid_reset
from motion_tracking.utils.motion_lib import MotionLib

import torch
from torch import Tensor
from typing import Optional, Tuple
import numpy as np


class HumanoidBlock(TaskHumanoid):
    def __init__(self, config, device, motion_lib: Optional[MotionLib] = None):
        self.read_task_config(config)
        super().__init__(config, device, motion_lib=motion_lib)

    def setup_task(self):
        self.prev_root_pos = torch.zeros(
            [self.num_envs, 3], device=self.device, dtype=torch.float
        )
        strike_body_names = self.config["strikeBodyNames"]
        self.strike_body_ids = self.build_strike_body_ids_tensor(
            self.envs[0], self.humanoid_handles[0], strike_body_names
        )
        self.build_target_tensors()

    def read_task_config(self, cfg):
        self.tar_dist_min = cfg["tar_dist_min"]
        self.tar_dist_max = cfg["tar_dist_max"]
        self.near_dist = cfg["near_dist"]
        self.near_prob = cfg["near_prob"]
        self.hack_output_motion_file = None

    def get_task_obs_size(self):
        obs_size = 0
        if self.enable_task_obs:
            obs_size = 15
        return obs_size

    def pre_create_envs_task(self, num_envs, spacing, num_per_row):
        self.target_handles = []
        self.load_target_asset()

    def build_env_task(self, env_id, env_ptr, humanoid_asset):
        self.build_target(env_id, env_ptr)
        return

    def load_target_asset(self):
        asset_root = "motion_tracking/data/assets/mjcf/"
        asset_file = "strike_target.urdf"

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 30.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self.target_asset = self.gym.load_asset(
            self.sim, asset_root, asset_file, asset_options
        )
        self.num_target_bodies = self.gym.get_asset_rigid_body_count(self.target_asset)

        return

    def set_marker_color(self, env_ids, col):
        for env_id in env_ids:
            env_ptr = self.envs[env_id]
            handle = self.target_handles[env_id]

            for j in range(self.num_target_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr,
                    handle,
                    j,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(col[0], col[1], col[2]),
                )

        return

    def build_target(self, env_id, env_ptr):
        default_pose = gymapi.Transform()
        default_pose.p.x = 1.0

        target_handle = self.gym.create_actor(
            env_ptr, self.target_asset, default_pose, "target", env_id, 2
        )
        self.target_handles.append(target_handle)

        return

    def build_strike_body_ids_tensor(self, env_ptr, actor_handle, body_names):
        env_ptr = self.envs[0]
        actor_handle = self.humanoid_handles[0]
        body_ids = []

        for body_name in body_names:
            body_id = self.gym.find_actor_rigid_body_handle(
                env_ptr, actor_handle, body_name
            )
            assert body_id != -1
            body_ids.append(body_id)

        body_ids = torch_utils.to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self.target_states = self.root_states.view(
            self.num_envs, num_actors, self.root_states.shape[-1]
        )[..., self.target_handles[0], :]

        self.tar_actor_ids = (
            torch_utils.to_torch(
                num_actors * np.arange(self.num_envs),
                device=self.device,
                dtype=torch.int32,
            )
            + self.target_handles[0]
        )

        bodies_per_env = self.rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self.tar_contact_forces = contact_force_tensor.view(
            self.num_envs, bodies_per_env, 3
        )[..., self.num_bodies, :]

        return

    # def reset_actors(self, env_ids):
    #     super()._reset_actors(env_ids)
    #     self.reset_target(env_ids)
    #     return

    def get_task_actor_ids_for_reset(self, env_ids):
        return self.tar_actor_ids[env_ids]

    def reset_task_post_actors(self, env_ids):
        self.reset_target(env_ids)
        # env_ids_int32 = self.tar_actor_ids[env_ids]
        # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
        #                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def reset_target(self, env_ids):
        n = len(env_ids)

        init_near = (
            torch.rand(
                [n], dtype=self.target_states.dtype, device=self.target_states.device
            )
            < self.near_prob
        )
        dist_max = self.tar_dist_max * torch.ones(
            [n], dtype=self.target_states.dtype, device=self.target_states.device
        )
        dist_max[init_near] = self.near_dist
        rand_dist = (dist_max - self.tar_dist_min) * torch.rand(
            [n], dtype=self.target_states.dtype, device=self.target_states.device
        ) + self.tar_dist_min

        rand_theta = (
            2
            * np.pi
            * torch.rand(
                [n], dtype=self.target_states.dtype, device=self.target_states.device
            )
        )

        self.set_target(env_ids, rand_dist, rand_theta)
        return

    def make_rand_rot_height(self, n):
        rand_rot_theta = (
            2
            * np.pi
            * torch.rand(
                [n], dtype=self.target_states.dtype, device=self.target_states.device
            )
        )
        axis = torch.tensor(
            [0.0, 0.0, 1.0],
            dtype=self.target_states.dtype,
            device=self.target_states.device,
        )
        rand_rot = rotations.quat_from_angle_axis(rand_rot_theta, axis)

        height = (
            torch.ones(
                n, dtype=self.target_states.dtype, device=self.target_states.device
            )
            * 0.9
        )

        return rand_rot, height

    def set_target(self, env_ids, rand_dist, rand_theta):
        n = len(env_ids)

        self.target_states[env_ids, 0] = (
            rand_dist * torch.cos(rand_theta) + self.humanoid_root_states[env_ids, 0]
        )
        self.target_states[env_ids, 1] = (
            rand_dist * torch.sin(rand_theta) + self.humanoid_root_states[env_ids, 1]
        )

        rand_rot, height = self.make_rand_rot_height(n)

        self.target_states[env_ids, 2] = height

        self.target_states[env_ids, 3:7] = rand_rot
        self.target_states[env_ids, 7:10] = 0.0
        self.target_states[env_ids, 10:13] = 0.0

    def update_task(self):
        self.prev_root_pos[:] = self.humanoid_root_states[..., 0:3]
        if self.config.output_motion:
            self.output_motion_target()

    # def reset_env_tensors(self, env_ids):
    #     super()._reset_env_tensors(env_ids)

    #     env_ids_int32 = self.tar_actor_ids[env_ids]
    #     self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
    #                                                  gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    #     return

    # def pre_physics_step(self, actions):
    #     super().pre_physics_step(actions)
    #     self.prev_root_pos[:] = self.humanoid_root_states[..., 0:3]
    #     return

    def compute_task_obs(self, env_ids=None):
        if env_ids is None:
            root_states = self.humanoid_root_states
            tar_states = self.target_states
        else:
            root_states = self.humanoid_root_states[env_ids]
            tar_states = self.target_states[env_ids]

        obs = compute_strike_observations(root_states, tar_states)
        return obs

    # def compute_reward(self, actions):
    #     tar_pos = self.target_states[..., 0:3]
    #     tar_rot = self.target_states[..., 3:7]
    #     char_root_state = self.humanoid_root_states
    #     strike_body_vel = self.rigid_body_vel[..., self.strike_body_ids[0], :]

    #     self.rew_buf[:] = compute_strike_reward(
    #         tar_pos,
    #         tar_rot,
    #         char_root_state,
    #         self.prev_root_pos,
    #         strike_body_vel,
    #         self.dt,
    #         self.near_dist,
    #     )
    #     return

    # def compute_reset(self):
    #     self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(
    #         self.reset_buf,
    #         self.progress_buf,
    #         self.contact_forces,
    #         self.contact_body_ids,
    #         self.rigid_body_pos,
    #         self.tar_contact_forces,
    #         self.strike_body_ids,
    #         self.max_episode_length,
    #         self.enable_early_termination,
    #         self.termination_heights,
    #     )
    #     return

    def draw_task(self):
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        starts = self.humanoid_root_states[..., 0:3]
        ends = self.target_states[..., 0:3]
        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(
                self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols
            )

        return

    def output_motion_target(self):
        if not hasattr(self, "output_motion_target_pos"):
            self.output_motion_target_pos = []
            self.output_motion_target_rot = []

        tar_pos = self.target_states[0, 0:3].cpu().numpy()
        self.output_motion_target_pos.append(tar_pos)

        tar_rot = self.target_states[0, 3:7].cpu().numpy()
        self.output_motion_target_rot.append(tar_rot)

        reset = self.reset_buf[0].cpu().numpy() == 1

        output_tar_pos = np.array(self.output_motion_target_pos)
        output_tar_rot = np.array(self.output_motion_target_rot)
        output_data = np.concatenate([output_tar_pos, output_tar_rot], axis=-1)

        if self.hack_output_motion_file is None:
            file = "output/record_tar_motion.npy"
        else:
            file = self.hack_output_motion_file

        np.save(file, output_data)

        if reset and len(self.output_motion_target_pos) > 1:

            self.output_motion_target_pos = []
            self.output_motion_target_rot = []

        return

    def compute_reset(self):
        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            self.contact_forces,
            self.contact_body_ids,
            self.rigid_body_pos,
            self.max_episode_length,
            self.enable_height_termination,
            self.termination_heights,
        )

        tar_pos = self.target_states[..., 0:3]
        tar_rot = self.target_states[..., 3:7]
        strike_reward = compute_strike_reward(tar_pos, tar_rot)

        if self.config.strike_thresh_term is not None:
            block_too_low = strike_reward >= self.config.strike_thresh_term
            ones = torch.ones_like(self.reset_buf)
            self.reset_buf[:] = torch.where(block_too_low, ones, self.reset_buf)
            self.terminate_buf[:] = torch.where(block_too_low, ones, self.terminate_buf)

    def compute_reward(self, actions):
        tar_pos = self.target_states[..., 0:3]
        tar_rot = self.target_states[..., 3:7]
        strike_reward = compute_strike_reward(tar_pos, tar_rot)

        root_pos = self.humanoid_root_states[..., 0:3]
        root_rot = self.humanoid_root_states[..., 3:7]
        tar_dir = tar_pos[..., 0:2] - root_pos[..., 0:2]
        tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
        pos_reward, pos_err = compute_location_reward(
            root_pos, tar_pos[..., 0:2], self.pos_err_scale
        )
        pos_dist = pos_err.sqrt()

        heading_reward = compute_heading_reward(
            root_pos,
            self.prev_root_pos,
            tar_dir,
            self.config.tar_speed,
            self.dt,
            self.config.vel_err_scale,
            self.config.tangent_err_w,
            negatives_allowed=self.negatives_allowed,
        )

        facing_reward = compute_facing_reward(root_pos, root_rot, tar_dir)
        if self.facing_reward_thresh is not None:
            facing_reward = torch.clamp_max(facing_reward, self.facing_reward_thresh)

        if self.strike_thresh is not None:
            one = torch.tensor(
                1.0, dtype=heading_reward.dtype, device=heading_reward.device
            )
            pos_reward = torch.where(
                strike_reward >= self.strike_thresh, one, pos_reward
            )
            heading_reward = torch.where(
                strike_reward >= self.strike_thresh, one, heading_reward
            )
            facing_reward = torch.where(
                strike_reward >= self.strike_thresh, one, facing_reward
            )

        if self.loc_dist_thresh is not None:
            one = torch.tensor(
                1.0, dtype=heading_reward.dtype, device=heading_reward.device
            )
            pos_reward = torch.where(pos_dist < self.loc_dist_thresh, one, pos_reward)
            heading_reward = torch.where(
                pos_dist < self.loc_dist_thresh, one, heading_reward
            )
            facing_reward = torch.where(
                pos_dist < self.loc_dist_thresh, one, facing_reward
            )
            strike_reward = torch.where(
                pos_dist < self.loc_dist_thresh, one, strike_reward
            )

        reward = (
            self.pos_reward_w * pos_reward
            + self.strike_reward_w * strike_reward
            + self.heading_reward_w * heading_reward
            + self.facing_reward_w * facing_reward
        )

        self.rew_buf[:] = reward
        return reward


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_location_reward(root_pos, tar_pos, pos_err_scale):
    # type: (Tensor, Tensor, float) -> Tuple[Tensor, Tensor]

    pos_diff = tar_pos - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    return pos_reward, pos_err


@torch.jit.script
def compute_strike_reward(tar_pos, tar_rot):
    # type: (Tensor, Tensor) -> Tensor

    up = torch.zeros_like(tar_pos)
    up[..., -1] = 1
    tar_up = rotations.quat_rotate(tar_rot, up)
    tar_rot_err = torch.sum(up * tar_up, dim=-1)
    tar_rot_r = torch.clamp_min(1.0 - tar_rot_err, 0.0)

    reward = tar_rot_r

    succ = tar_rot_err < 0.2
    reward = torch.where(succ, torch.ones_like(reward), reward)

    return reward


@torch.jit.script
def compute_strike_observations(root_states: Tensor, tar_states: Tensor, w_last: bool) -> Tensor:
    root_pos = root_states[:, 0:3]
    root_rot = root_states[:, 3:7]

    tar_pos = tar_states[:, 0:3]
    tar_rot = tar_states[:, 3:7]
    tar_vel = tar_states[:, 7:10]
    tar_ang_vel = tar_states[:, 10:13]

    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    local_tar_pos = tar_pos - root_pos
    local_tar_pos[..., -1] = tar_pos[..., -1]
    local_tar_pos = rotations.quat_rotate(heading_rot, local_tar_pos)
    local_tar_vel = rotations.quat_rotate(heading_rot, tar_vel)
    local_tar_ang_vel = rotations.quat_rotate(heading_rot, tar_ang_vel)

    local_tar_rot = rotations.quat_mul(heading_rot, tar_rot)
    local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

    obs = torch.cat(
        [local_tar_pos, local_tar_rot_obs, local_tar_vel, local_tar_ang_vel], dim=-1
    )
    return obs


@torch.jit.script
def compute_facing_reward(root_pos, root_rot, tar_face_dir):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = rotations.quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)

    return facing_reward


@torch.jit.script
def compute_thresholded_facing_reward(
    root_pos, root_rot, tar_face_dir, max_rew, exp_scale
):
    # type: (Tensor, Tensor, Tensor, float, float) -> Tensor
    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = rotations.quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_face_dir * facing_dir[..., 0:2], dim=-1)
    to_exp = torch.clamp_max(facing_err, 0.0)

    facing_reward = torch.exp(exp_scale * to_exp)

    return facing_reward


@torch.jit.script
def compute_heading_reward(
    root_pos,
    prev_root_pos,
    tar_dir,
    tar_speed,
    dt,
    vel_err_scale,
    tangent_err_w,
    negatives_allowed=False,
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, float, bool) -> Tensor

    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)

    tar_dir_vel = tar_dir_speed.unsqueeze(-1) * tar_dir
    tangent_vel = root_vel[..., :2] - tar_dir_vel

    tangent_speed = torch.sum(tangent_vel, dim=-1)

    tar_vel_diff = tar_speed - tar_dir_speed
    tar_vel_err, _ = torch.max(tar_vel_diff, 0)

    tangent_vel_err = tangent_speed

    dir_reward = torch.exp(
        -vel_err_scale
        * (
            tar_vel_err * tar_vel_err
            + tangent_err_w * tangent_vel_err * tangent_vel_err
        )
    )

    speed_mask = tar_dir_speed <= 0

    if negatives_allowed:
        dir_reward[speed_mask] = tar_dir_speed[speed_mask]
    else:
        dir_reward[speed_mask] = 0

    return dir_reward


class DiscHumanoidBlock(DiscHumanoid, HumanoidBlock):
    def __init__(self, config, device: torch.device, motion_lib: Optional[MotionLib] = None):
        super().__init__(config, device, motion_lib=motion_lib)
