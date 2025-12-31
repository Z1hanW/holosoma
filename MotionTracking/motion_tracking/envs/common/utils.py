from typing import List, Union, Tuple, Dict

import torch
from torch import Tensor
import numpy as np

from isaac_utils import torch_utils, rotations


@torch.jit.script
def transfer_to_local_coordinates(root_pos: Tensor, env_ids: Tensor, env_pos: Tensor) -> Tensor:
    root_pos = root_pos - env_pos[env_ids]
    return root_pos


@torch.jit.script
def dof_to_obs(
        pose: Tensor,
        dof_obs_size: int,
        dof_offsets: List[int],
        w_last: bool
) -> Tensor:
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    assert pose.shape[-1] == dof_offsets[-1]
    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset : (dof_offset + dof_size)]

        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose, w_last)
        elif dof_size == 1:
            axis = torch.tensor(
                [0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device
            )
            joint_pose_q = rotations.quat_from_angle_axis(joint_pose[..., 0], axis, w_last)
        else:
            joint_pose_q = None
            assert False, "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q, w_last)
        dof_obs[:, (j * joint_obs_size): ((j + 1) * joint_obs_size)] = joint_dof_obs

    assert (num_joints * joint_obs_size) == dof_obs_size

    return dof_obs


@torch.jit.script
def compute_humanoid_reward(obs_buf: Tensor) -> Tensor:
    reward = torch.ones_like(obs_buf[:, 0])
    return reward


def build_pd_action_offset_scale(dof_offsets, dof_limits_lower, dof_limits_upper, device, dof_names=None, is_smpl=False, specific_pd_fixes=False):
    num_joints = len(dof_offsets) - 1

    lim_low = dof_limits_lower.cpu().numpy()
    lim_high = dof_limits_upper.cpu().numpy()

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]

        if dof_size == 3:
            curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
            curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
            curr_low = np.max(np.abs(curr_low))
            curr_high = np.max(np.abs(curr_high))
            curr_scale = max([curr_low, curr_high])
            curr_scale = 1.2 * curr_scale
            curr_scale = min([curr_scale, np.pi])

            lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
            lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale

            # lim_low[dof_offset:(dof_offset + dof_size)] = -np.pi
            # lim_high[dof_offset:(dof_offset + dof_size)] = np.pi

        elif dof_size == 1:
            curr_low = lim_low[dof_offset]
            curr_high = lim_high[dof_offset]
            curr_mid = 0.5 * (curr_high + curr_low)

            # extend the action range to be a bit beyond the joint limits so that the motors
            # don't lose their strength as they approach the joint limits
            curr_scale = 0.7 * (curr_high - curr_low)
            curr_low = curr_mid - curr_scale
            curr_high = curr_mid + curr_scale

            lim_low[dof_offset] = curr_low
            lim_high[dof_offset] = curr_high

    pd_action_offset = 0.5 * (lim_high + lim_low)
    pd_action_scale = 0.5 * (lim_high - lim_low)
    pd_action_offset = torch.tensor(pd_action_offset, device=device)
    pd_action_scale = torch.tensor(pd_action_scale, device=device)

    if is_smpl:
        for dof_idx, dof_name in enumerate(dof_names):
            # ZL: Modified SMPL
            if specific_pd_fixes:
                if "L_Knee_y" in dof_name or "R_Knee_y" in dof_name:
                    pd_action_scale[dof_idx] = 5
                if "L_Shoulder_x" in dof_name:
                    pd_action_offset[dof_idx] = -np.pi / 2
                if "R_Shoulder_x" in dof_name:
                    pd_action_offset[dof_idx] = np.pi / 2
            else:
                if "L_Knee" in dof_name or "R_Knee" in dof_name:
                    pd_action_scale[dof_idx] = 5
    return pd_action_offset, pd_action_scale


@torch.jit.script
def compute_humanoid_observations(
        root_pos: Tensor,
        root_rot: Tensor,
        root_vel: Tensor,
        root_ang_vel: Tensor,
        dof_pos: Tensor,
        dof_vel: Tensor,
        key_body_pos: Tensor,
        ground_height: Tensor,
        local_root_obs: bool,
        dof_obs_size: int,
        dof_offsets: List[int],
        w_last: bool
) -> Tensor:
    root_h = root_pos[:, 2:3] - ground_height
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    if local_root_obs:
        root_rot_obs = rotations.quat_mul(heading_rot, root_rot, w_last)
    else:
        root_rot_obs = root_rot

    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs, w_last)

    local_root_vel = rotations.quat_rotate(heading_rot, root_vel, w_last)
    local_root_ang_vel = rotations.quat_rotate(heading_rot, root_ang_vel, w_last)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = rotations.quat_rotate(flat_heading_rot, flat_end_pos, w_last)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets, w_last)

    obs = torch.cat(
        (
            root_h,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_humanoid_observations_max(
        body_pos: Tensor,
        body_rot: Tensor,
        body_vel: Tensor,
        body_ang_vel: Tensor,
        ground_height: Tensor,
        local_root_obs: bool,
        root_height_obs: bool,
        w_last: bool
) -> Tensor:
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h - ground_height

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2]
    )

    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(
        local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2]
    )
    flat_local_body_pos = rotations.quat_rotate(
        flat_heading_rot, flat_local_body_pos, w_last
    )
    local_body_pos = flat_local_body_pos.reshape(
        local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2]
    )
    local_body_pos = local_body_pos[..., 3:]  # remove root pos

    flat_body_rot = body_rot.reshape(
        body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2]
    )
    flat_local_body_rot = rotations.quat_mul(flat_heading_rot, flat_body_rot, w_last)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot, w_last)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(
        body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1]
    )

    if not local_root_obs:
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot, w_last)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(
        body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2]
    )
    flat_local_body_vel = rotations.quat_rotate(flat_heading_rot, flat_body_vel, w_last)
    local_body_vel = flat_local_body_vel.reshape(
        body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2]
    )

    flat_body_ang_vel = body_ang_vel.reshape(
        body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2]
    )
    flat_local_body_ang_vel = rotations.quat_rotate(
        flat_heading_rot, flat_body_ang_vel, w_last
    )
    local_body_ang_vel = flat_local_body_ang_vel.reshape(
        body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2]
    )

    obs = torch.cat(
        (
            root_h_obs,
            local_body_pos,
            local_body_rot_obs,
            local_body_vel,
            local_body_ang_vel
        ),
        dim=-1
    )
    return obs


@torch.jit.script
def compute_humanoid_reset(
    reset_buf: Tensor,
    progress_buf: Tensor,
    contact_buf: Tensor,
    contact_body_ids: Tensor,
    rigid_body_pos: Tensor,
    max_episode_length: float,
    enable_early_termination: bool,
    termination_heights: Tensor,
) -> Tuple[Tensor, Tensor]:
    terminated = torch.zeros_like(reset_buf)

    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_heights
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= progress_buf > 1
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)

    reset = torch.where(
        progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated
    )

    return reset, terminated


# @torch.jit.script
def build_disc_observations(
        root_pos: Tensor,
        root_rot: Tensor,
        root_vel: Tensor,
        root_ang_vel: Tensor,
        dof_pos: Tensor,
        dof_vel: Tensor,
        key_body_pos: Tensor,
        ground_height: Tensor,
        local_root_obs: bool,
        root_height_obs: bool,
        dof_obs_size: int,
        dof_offsets: List[int],
        return_dict: bool,
        w_last: bool
) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot, w_last)

    if local_root_obs:
        root_rot_obs = rotations.quat_mul(heading_rot, root_rot, w_last)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs, w_last)

    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h - ground_height

    local_root_vel = torch_utils.quat_rotate(heading_rot, root_vel, w_last)
    local_root_ang_vel = torch_utils.quat_rotate(heading_rot, root_ang_vel, w_last)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand

    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(
        local_key_body_pos.shape[0] * local_key_body_pos.shape[1],
        local_key_body_pos.shape[2],
    )
    flat_heading_rot = heading_rot_expand.view(
        heading_rot_expand.shape[0] * heading_rot_expand.shape[1],
        heading_rot_expand.shape[2],
    )
    local_end_pos = torch_utils.quat_rotate(flat_heading_rot, flat_end_pos, w_last)
    flat_local_key_pos = local_end_pos.view(
        local_key_body_pos.shape[0],
        local_key_body_pos.shape[1] * local_key_body_pos.shape[2],
    )

    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets, w_last)

    obs = torch.cat(
        (
            root_h_obs,
            root_rot_obs,
            local_root_vel,
            local_root_ang_vel,
            dof_obs,
            dof_vel,
            flat_local_key_pos,
        ),
        dim=-1,
    )

    if not return_dict:
        return obs

    else:
        obs_dict = {
            "root_h": root_h,
            "root_rot": root_rot_obs,
            "root_vel": local_root_vel,
            "root_ang_vel": local_root_ang_vel,
            "dof_obs": dof_obs,
            "dof_vel": dof_vel,
            "key_pos": flat_local_key_pos,
        }

        return obs, obs_dict
