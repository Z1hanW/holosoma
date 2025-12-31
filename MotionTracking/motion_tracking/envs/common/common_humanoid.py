import numpy as np
import torch

from isaac_utils import torch_utils, rotations
from hydra.utils import instantiate

from motion_tracking.envs.common.utils import (
    compute_humanoid_observations,
    compute_humanoid_observations_max,
    compute_humanoid_reward,
    compute_humanoid_reset,
    transfer_to_local_coordinates
)
from motion_tracking.envs.terrains.terrain import Terrain

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from motion_tracking.envs.isaacgym.humanoid import Humanoid
else:
    Humanoid = object
import time

from motion_tracking.utils.calculate_voxel import read_urdf,compute_voxel_grid_batch

class BaseHumanoid(Humanoid):
    def __init__(
            self,
            config,
            device: torch.device
    ):
        self.config = config
        self.device = device
        self.num_envs = self.config.num_envs
        self.init_done = False
        self.use_mimic_goal_obs = getattr(self.config, "use_mimic_goal_obs", False)
        self.force_point_cloud_obs = getattr(self.config, "force_point_cloud_obs", False)
        self.point_sample_radius = getattr(self.config, "point_sample_radius", 0.0)
        self.point_sample_random = getattr(self.config, "point_sample_random", False)
        self.voxel = False
        self._latest_sampled_scene_points = None
        # Object storage
        self.total_num_objects = 0
        self.object_types = [  # Maintain a fixed list of object types, this is for indexing purposes. Need to find a better way.
            "Armchairs",
            "StraightChairs",
            "HighStools",
            "LowStools",
            "Sofas",
            "Tables",
            "LargeSofas"
        ]
        self.object_action_types = self.config.object_action_types if self.config.object_action_type is None else [self.config.object_action_type]
        self.object_action_type_to_object_category = self.config.object_action_type_to_object_category if self.config.object_category is None else {self.config.object_action_type: [self.config.object_category]}
        self.object_action_type_keys = []
        self.object_action_type_to_object = {}
        self.object_type_motion_offset = {}
        self.spawned_object_names = []
        self.object_id_to_object_position = []
        self.object_id_to_object_bounding_box = []
        self.object_root_states = []
        self.object_id_to_target_position = []
        self.object_offsets = self.config.object_offsets

        # General configurations
        self.pd_control = self.config.pd_control
        self.power_scale = self.config.power_scale
        self.local_root_obs = self.config.local_root_obs
        self.root_height_obs = self.config.root_height_obs
        self.enable_height_termination = self.config.enable_height_termination
        self.max_episode_length = self.config.max_episode_length
        self.control_freq_inv = self.config.control_freq_inv

        self.setup_character_props()

        self.not_mimic = self.config.not_mimic

        # Buffers
        self.obs_buf = torch.zeros((self.num_envs, self.get_obs_size()), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}
        self.log_dict = {}

        self.terrain = None
        self.force_respawn_on_flat = False
        if self.config.terrain is not None:# or self.config.dummy_terrain:
            self.create_terrain()
            self.terrain_obs = torch.zeros(self.num_envs, self.num_height_points, device=self.device, dtype=torch.float)

        self.reset_happened = False
        self.reset_ref_env_ids = []
        self.reset_states = None
        

        super().__init__(config, device)

        # After objects have been populated, finalize structure
        for action_type in self.object_action_type_to_object.keys():
            if len(self.object_action_type_to_object[action_type]) > 0:
                self.object_action_type_to_object[action_type] = torch.stack(self.object_action_type_to_object[action_type])
        if len(self.object_id_to_object_position) > 0:
            self.object_id_to_object_position = torch.stack(self.object_id_to_object_position)
            self.object_id_to_object_bounding_box = torch.stack(self.object_id_to_object_bounding_box).reshape(self.total_num_objects, -1, 3)
            self.object_root_states = torch.stack(self.object_root_states)
            self.object_id_to_target_position = torch.stack(self.object_id_to_target_position)
        
        if self.config.target_next_root:
            self.goal_weight = torch.rand(self.num_envs).to(self.device)
            # motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
            # next_motion_times_ = self.goal_weight * self.motion_times + (1 - self.goal_weight) * motion_lengths
            self.next_motion_times = torch.zeros_like(self.goal_weight).to(self.device) #torch.clamp(next_motion_times_, max = motion_lengths)
            

    ###############################################################
    # Getters
    ###############################################################
    def get_obs_size(self):
        #need double check
        if self.not_mimic:
            return self.num_obs #+ self.config.num_task_obs
        else:
            return self.num_obs + self.config.num_task_obs


    def get_action_size(self):
        return self.num_act

    def get_body_id(self, body_name):
        raise NotImplementedError

    def get_envs_respawn_position(self, env_ids, offset=0, rb_pos: torch.tensor=None, object_ids: torch.tensor=None):
        xy_position = torch.zeros((len(env_ids), 2), device=self.device)
        xy_position[:, :2] += offset
        # 
        # 
        if self.terrain is not None:
            if self.force_respawn_on_flat:
                xy_position = self.terrain.sample_flat_locations(len(env_ids))
            else:
                # xy_position = self.terrain.sample_valid_locations(len(env_ids))
                xy_position = xy_position

        if object_ids is not None:
            if -2 in object_ids:
                raise NotImplementedError("Attempting to use a motion that requires an object that is not spawned.")

            object_interaction_envs_mask = object_ids != -1  # Object id -1 corresponds to no object
            if torch.any(object_interaction_envs_mask):
                xy_position[object_interaction_envs_mask] = self.object_id_to_object_position[object_ids[object_interaction_envs_mask], :2].clone()
                xy_position[object_interaction_envs_mask, :2] += offset[object_interaction_envs_mask]

        if rb_pos is not None:
            normalized_dof_pos = rb_pos.clone()
            normalized_dof_pos[:, :, :2] -= rb_pos[:, :1, :2]  # remove root position
            normalized_dof_pos[:, :, :2] += xy_position.unsqueeze(1)  # add respawn offset
            flat_normalized_dof_pos = normalized_dof_pos.view(-1, 3)
            z_all_joints = self.get_heights_below_base(flat_normalized_dof_pos)
            z_all_joints = z_all_joints.view(normalized_dof_pos.shape[:-1])

            z_diff = z_all_joints - normalized_dof_pos[:, :, 2]
            z_indices = torch.max(z_diff, dim=1).indices.view(-1, 1)

            # Extra offset is added to ensure the character is above the terrain.
            # This is the minimal required distance of any joint to avoid collisions.
            # If the character is above this height (e.g., jumping), do not add any offset.
            min_joint_height = rb_pos[:, :, 2].min(dim=1).values.view(-1, 1)
            extra_offset = (self.config.ref_respawn_offset - min_joint_height).clamp(min=0)
            # We want to add the offset based on the ground terrain-height below the joint.
            # Unlike the diff. The reason is that while jumping, we want to ensure the character retains
            # the relative height above the terrain.
            z_offset = z_all_joints.gather(1, z_indices).view(-1, 1) + extra_offset
        else:
            z_root = self.get_heights_below_base(xy_position)
            z_offset = z_root.view(-1, 1) + self.config.ref_respawn_offset

        respawn_position = torch.cat([xy_position, z_offset], dim=-1)
        respawn_position = transfer_to_local_coordinates(respawn_position, env_ids, self.env_offsets)

        return respawn_position

    def sample_object_ids(self, motion_ids, get_first_matching_object=False):
        object_ids = torch.zeros_like(motion_ids, dtype=torch.long, device=self.device) - 1  # index -1 corresponds to no object
        if self.motion_lib.motion_to_object_ids.shape[0] > 0 and self.total_num_objects > 0:
            non_object_motions = self.motion_lib.objects_per_motion[motion_ids] == -1
            object_ids[non_object_motions] = -1
            motions_lacking_an_object = self.motion_lib.objects_per_motion[motion_ids] == 0
            object_ids[motions_lacking_an_object] = -2

            motions_with_objects = self.motion_lib.objects_per_motion[motion_ids] > 0
            objects_per_motion = self.motion_lib.objects_per_motion[motion_ids[motions_with_objects]]
            # sample random [0,1], then re-map to "objects_per_motion" and then round.
            if get_first_matching_object:
                sampled_object_per_motion_internal_mapping = 0
            else:
                sampled_object_per_motion_internal_mapping = torch.round(torch.rand(len(objects_per_motion), device=self.device) * (objects_per_motion - 1).float()).long()
            # then select the actual motion ids
            sampled_object_per_motion = self.motion_lib.motion_to_object_ids[motion_ids[motions_with_objects], sampled_object_per_motion_internal_mapping]

            object_ids[motions_with_objects] = sampled_object_per_motion

        return object_ids

    ###############################################################
    # Environment step logic
    ###############################################################
    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device)

        clamp_actions = self.config.clamp_actions
        if clamp_actions is not None:
            self.actions = torch.clamp(self.actions, -clamp_actions, clamp_actions)
            self.log_dict["action_clamp_frac"] = (
                self.actions.abs() == clamp_actions
            ).sum() / self.actions.numel()

        if self.pd_control:
            self.apply_pd_control()
        else:
            self.apply_motor_forces()

    def post_physics_step(self):
        self.progress_buf += 1

        if self.world_running():
            self.compute_observations()
            self.compute_reward(self.actions)
            if not self.disable_reset:
                self.compute_reset()

        self.log_dict["terminate_frac"] = self.terminate_buf.float().mean()

        self.extras["terminate"] = self.terminate_buf
        self.extras["to_log"] = self.log_dict

    def world_running(self):
        # Override in IsaacSim.
        return True

    def compute_observations(self, env_ids=None):
        obs = self.compute_humanoid_obs(env_ids)
        
        # 
        if not self.not_mimic:
            if self.config.enable_task_obs and self.config.num_task_obs == 90:
                point_obs = self.compute_point_obs(env_ids)

                if env_ids is None:
                    
                    self.obs_buf[:] = torch.cat([obs, point_obs], dim=-1)
                else:

                    self.obs_buf[env_ids] = torch.cat([obs, point_obs], dim=-1)

            elif self.config.enable_task_obs and self.config.num_task_obs == 93:
                
                motion_lenghths = self.motion_lib.get_motion_length(self.motion_ids)

                root_pos_goal = self.motion_lib.get_motion_root_pos(self.motion_ids, motion_lenghths)
                
                point_obs = self.compute_point_obs(env_ids)
                goal_obs = self.compute_goal_obs(root_pos_goal, env_ids)
                
                # 
                # print(self.reset_ref_motion_ids.clone())#, goal_obs.shape)
                
                if env_ids is None:
                    
                    self.obs_buf[:] = torch.cat([obs, point_obs, goal_obs], dim=-1)
                    # self.mimic_scene[:] = torch.cat([point_obs, goal_obs], dim=-1)
                else:

                    self.obs_buf[env_ids] = torch.cat([obs, point_obs, goal_obs], dim=-1)
                    # self.mimic_scene[env_ids] = torch.cat([point_obs, goal_obs], dim=-1)
            
            elif self.config.enable_task_obs and self.config.num_task_obs == 0:
                
                if not self.voxel:
                
                    motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
                    
                    if self.config.target_next_root:
                        # weights = torch.rand_like(self.motion_times)
                        goal_motion_times = (self.motion_times >= self.next_motion_times) * motion_lengths + (self.motion_times < self.next_motion_times) * self.next_motion_times
                        root_pos_goal = self.motion_lib.get_motion_root_pos(self.motion_ids, goal_motion_times) 
                        # root_pos_goal += torch.randn_like(root_pos_goal) * 0.1
                        # self.root_pos_goal = root_pos_goal.clone()
                    else:
                        root_pos_goal = self.motion_lib.get_motion_root_pos(self.motion_ids, motion_lengths)
                    # 
                    
                    # print(root_pos_goal)
                    
                    point_obs = self.compute_point_obs(env_ids)

                    need_goal_obs = (not self.seperate_point_goal) or self.use_mimic_goal_obs
                    goal_obs = None
                    if need_goal_obs:
                        goal_obs = self.compute_goal_obs(root_pos_goal, env_ids)

                    
                    if not self.seperate_point_goal:
                        if env_ids is None:
                            
                            self.obs_buf[:] = obs #torch.cat([obs, point_obs, goal_obs], dim=-1)
                            self.mimic_scene[:] = torch.cat([point_obs, goal_obs], dim=-1)
                        else:

                            self.obs_buf[env_ids] = obs #torch.cat([obs, point_obs, goal_obs], dim=-1)
                            self.mimic_scene[env_ids] = torch.cat([point_obs, goal_obs], dim=-1)
                    else:
                        if env_ids is None:
                            self.obs_buf[:] = obs #torch.cat([obs, point_obs, goal_obs], dim=-1)
                            self.mimic_scene[:] = point_obs
                            if self.use_mimic_goal_obs and goal_obs is not None and self.mimic_goal is not None:
                                self.mimic_goal[:] = goal_obs
                        else:

                            self.obs_buf[env_ids] = obs #torch.cat([obs, point_obs, goal_obs], dim=-1)
                            self.mimic_scene[env_ids] = point_obs
                            if self.use_mimic_goal_obs and goal_obs is not None and self.mimic_goal is not None:
                                self.mimic_goal[env_ids] = goal_obs
            
                else:
                    motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
                    
                    if self.config.target_next_root:
                        # weights = torch.rand_like(self.motion_times)
                        goal_motion_times = (self.motion_times >= self.next_motion_times) * motion_lengths + (self.motion_times < self.next_motion_times) * self.next_motion_times
                        root_pos_goal = self.motion_lib.get_motion_root_pos(self.motion_ids, goal_motion_times) 
                        # root_pos_goal += torch.randn_like(root_pos_goal) * 0.1
                        # self.root_pos_goal = root_pos_goal.clone()
                    else:
                        root_pos_goal = self.motion_lib.get_motion_root_pos(self.motion_ids, motion_lengths)
                    
                    point_obs = self.compute_point_voxel_obs(env_ids)

                    need_goal_obs = (not self.seperate_point_goal) or self.use_mimic_goal_obs
                    goal_obs = None
                    if need_goal_obs:
                        goal_obs = self.compute_goal_obs(root_pos_goal, env_ids)                   
                   
                    if env_ids is None:
                        self.obs_buf[:] = obs #torch.cat([obs, point_obs, goal_obs], dim=-1)
                        self.mimic_scene[:] = point_obs
                        if self.use_mimic_goal_obs and goal_obs is not None and self.mimic_goal is not None:
                            self.mimic_goal[:] = goal_obs
                    else:

                        self.obs_buf[env_ids] = obs #torch.cat([obs, point_obs, goal_obs], dim=-1)
                        self.mimic_scene[env_ids] = point_obs
                        if self.use_mimic_goal_obs and goal_obs is not None and self.mimic_goal is not None:
                            self.mimic_goal[env_ids] = goal_obs
            else:
                if env_ids is None:
                    self.obs_buf[:] = obs
                else:
                    self.obs_buf[env_ids] = obs
            
        elif self.terrain is not None:
            height_obs = self.get_heights(env_ids)
            if env_ids is None:
                self.terrain_obs[:] = height_obs
            else:
                self.terrain_obs[env_ids] = height_obs

        else:
            if env_ids is None:
                
                self.obs_buf[:] = obs
            else:

                self.obs_buf[env_ids] = obs
            if self.seperate_point_goal and self.config.use_transformer:
                point_obs = self.compute_point_obs(env_ids)
                if env_ids is None:
                    self.mimic_scene[:] = point_obs
                else:
                    self.mimic_scene[env_ids] = point_obs

        if self.force_point_cloud_obs and not self.config.enable_task_obs:
            point_obs = self.compute_point_obs(env_ids)
            if env_ids is None:
                self.mimic_scene[:] = point_obs
            else:
                self.mimic_scene[env_ids] = point_obs

    def compute_point_voxel_obs(self, env_ids=None):
        
        root_states = self.get_humanoid_root_states()

        # Example Usage
        
        root_positions = root_states[...,:3]  # 1000 random root positions in [-1,1]
        start_time = time.time()

        if env_ids is None:
            batch_size = root_states.shape[0]
            occupancy_grids = compute_voxel_grid_batch(root_positions, self.boxes_tensor).float()
        
        else:
            batch_size = len(env_ids)
            occupancy_grids = compute_voxel_grid_batch(root_positions[env_ids], self.boxes_tensor[env_ids]).float()

        # print(time.time() - start_time)
        return occupancy_grids.view(batch_size,-1)

    
    def compute_point_obs(self, env_ids=None):
        
        root_states = self.get_humanoid_root_states()

        if env_ids is None:
            
            root_pos = root_states[:, :3]
            root_rot = root_states[:, 3:7]
            points = self.pointcloud
        
        else:
            root_pos = root_states[env_ids, :3]
            root_rot = root_states[env_ids, 3:7]
            points = self.pointcloud[env_ids]

        points_delta = points - root_pos.unsqueeze(-2)
        dist_sq = torch.sum(points_delta * points_delta, dim=-1)

        num_points = points_delta.shape[1]
        k = min(self.num_closest_point, num_points)
        if k <= 0:
            return torch.zeros(points_delta.shape[0], 0, device=points_delta.device)

        use_random = self.point_sample_random and self.point_sample_radius > 0.0
        radius_sq = self.point_sample_radius * self.point_sample_radius if use_random else None

        selected = torch.zeros(points_delta.shape[0], k, 3, device=points_delta.device, dtype=points_delta.dtype)
        for b in range(points_delta.shape[0]):
            if use_random:
                valid_idx = torch.nonzero(dist_sq[b] <= radius_sq, as_tuple=False).squeeze(-1)
                if valid_idx.numel() >= k:
                    perm = torch.randperm(valid_idx.numel(), device=points_delta.device)
                    chosen = valid_idx[perm[:k]]
                elif valid_idx.numel() > 0:
                    need = k - valid_idx.numel()
                    extra = torch.randint(0, valid_idx.numel(), (need,), device=points_delta.device)
                    chosen = torch.cat([valid_idx, valid_idx[extra]], dim=0)
                else:
                    _, chosen = torch.topk(dist_sq[b], k, largest=False)
            else:
                _, chosen = torch.topk(dist_sq[b], k, largest=False)

            selected[b] = points_delta[b, chosen]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot, self.w_last)
        heading_rot_exp = heading_rot.unsqueeze(1).expand(-1, k, -1).reshape(-1, heading_rot.shape[-1])
        selected_flat = selected.reshape(-1, 3)
        local_points_pos_flat = rotations.quat_rotate(heading_rot_exp, selected_flat, self.w_last)
        local_points_pos = local_points_pos_flat.view(points_delta.shape[0], k, 3)

        world_samples = selected + root_pos.unsqueeze(1)

        if self._latest_sampled_scene_points is None or self._latest_sampled_scene_points.shape[1] != k:
            self._latest_sampled_scene_points = torch.zeros(self.num_envs, k, 3, device=points_delta.device)
        if env_ids is None:
            self._latest_sampled_scene_points[:world_samples.shape[0]] = world_samples
        else:
            env_idx = torch.as_tensor(env_ids, device=points_delta.device, dtype=torch.long)
            self._latest_sampled_scene_points[env_idx] = world_samples

        return local_points_pos.reshape(local_points_pos.shape[0], -1)

    def compute_goal_obs(self, root_pos_goal, env_ids=None):
        
        root_states = self.get_humanoid_root_states()      

        if env_ids is None:
            
            root_pos = root_states[:, :3]
            root_rot = root_states[:, 3:7]
            root_pos_target = root_pos_goal
            
        else:
            root_pos = root_states[env_ids, :3]
            root_rot = root_states[env_ids, 3:7]
            root_pos_target = root_pos_goal[env_ids]
        
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot, self.w_last)
        local_tar_pos = rotations.quat_rotate(heading_rot, root_pos_target - root_pos, self.w_last)
        return local_tar_pos

    
    def compute_humanoid_obs(self, env_ids=None):
        # Retrieve body transforms & velocities
        (
            body_pos,
            body_rot,
            body_vel,
            body_ang_vel
        ) = self.get_bodies_state()

        env_global_positions = self.convert_to_global_coords(self.get_humanoid_root_states()[..., :2], self.env_offsets[..., :2])
        if self.config.obs_relative_to_surface:
            ground_heights = self.get_heights_below_base(env_global_positions)
        else:
            ground_heights = self.get_ground_heights_below_base(env_global_positions)

        if self.config.use_max_coords_obs:
            if env_ids is not None:
                body_pos = body_pos[env_ids]
                body_rot = body_rot[env_ids]
                body_vel = body_vel[env_ids]
                body_ang_vel = body_ang_vel[env_ids]
                ground_heights = ground_heights[env_ids]

            obs = compute_humanoid_observations_max(
                body_pos,
                body_rot,
                body_vel,
                body_ang_vel,
                ground_heights,
                self.local_root_obs,
                self.root_height_obs,
                self.w_last
            )

        else:
            dof_pos, dof_vel = self.get_dof_state()
            if env_ids is None:
                root_pos = body_pos[:, 0, :]
                root_rot = body_rot[:, 0, :]
                root_vel = body_vel[:, 0, :]
                root_ang_vel = body_ang_vel[:, 0, :]
                dof_pos = dof_pos
                dof_vel = dof_vel
                key_body_pos = body_pos[:, self.key_body_ids, :]
            else:
                root_pos = body_pos[env_ids][:, 0, :]
                root_rot = body_rot[env_ids][:, 0, :]
                root_vel = body_vel[env_ids][:, 0, :]
                root_ang_vel = body_ang_vel[env_ids][:, 0, :]
                dof_pos = dof_pos[env_ids]
                dof_vel = dof_vel[env_ids]
                key_body_pos = body_pos[env_ids][:, self.key_body_ids, :]
                ground_heights = ground_heights[env_ids]

            obs = compute_humanoid_observations(
                root_pos,
                root_rot,
                root_vel,
                root_ang_vel,
                dof_pos,
                dof_vel,
                key_body_pos,
                ground_heights,
                self.local_root_obs,
                self.dof_obs_size,
                self.get_dof_offsets(),
                self.w_last
            )
        return obs

    def compute_reset(self):
        bodies_positions = self.get_body_positions()
        bodies_contact_buf = self.get_bodies_contact_buf()
        env_global_positions = self.convert_to_global_coords(self.get_humanoid_root_states()[..., :2], self.env_offsets[..., :2])

        self.reset_buf[:], self.terminate_buf[:] = compute_humanoid_reset(
            self.reset_buf,
            self.progress_buf,
            bodies_contact_buf,
            self.contact_body_ids,
            bodies_positions,
            self.max_episode_length,
            self.enable_height_termination,
            self.termination_heights + self.get_ground_heights_below_base(env_global_positions),
        )

    def compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)

    ###############################################################
    # Handle Resets
    ###############################################################
    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if type(env_ids) == list:
            env_ids = torch.tensor(env_ids, device=self.device, dtype=torch.long)
        env_ids = env_ids.to(self.device)

        if len(env_ids) > 0:
            self.reset_happened = True

            

        self.reset_envs(env_ids)

        if self.config.target_next_root and len(env_ids) > 0:
            self.goal_weight[env_ids] = torch.rand_like(self.motion_times[env_ids])

            motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
            next_motion_times_ = self.goal_weight * self.motion_times + (1 - self.goal_weight) * motion_lengths
            self.next_motion_times = torch.clamp(next_motion_times_, max = motion_lengths)
            root_pos_goal = self.motion_lib.get_motion_root_pos(self.motion_ids, self.next_motion_times) 
            # self.root_pos_goal = root_pos_goal.clone()
            
        return self.obs_buf

    def reset_env_tensors(self, env_ids):
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.terminate_buf[env_ids] = 0

    def reset_actors(self, env_ids):
        self.reset_default(env_ids)

    ###############################################################
    # Terrain Helpers
    ###############################################################
    def create_terrain(self):
        if self.config.dummy_terrain:
            self.config.terrain._target_ = "motion_tracking.envs.terrains.dummy_terrain.DummyTerrain"
        self.terrain: Terrain = instantiate(
            self.config.terrain,
            device=self.device
        )
        self.only_terrain_height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device) * self.terrain.vertical_scale
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device) * self.terrain.vertical_scale
        self.height_points = self.init_height_points()

    def get_ground_heights_below_base(self, root_states):
        # We use this function for basic motion offsets.
        num_envs = root_states.shape[0]
        heights = torch.zeros(num_envs, device=self.device)

        if self.terrain is not None:
            points = root_states[..., :2].clone().reshape(num_envs, 1, 2)
            points -= self.terrain_offset
            points = (points / self.terrain.horizontal_scale).long()
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.only_terrain_height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.only_terrain_height_samples.shape[1] - 2)

            heights1 = self.only_terrain_height_samples[px, py]
            heights2 = self.only_terrain_height_samples[px + 1, py + 1]
            heights = torch.max(heights1, heights2)

        return heights.view(num_envs, -1)

    def get_heights_below_base(self, root_states):
        num_envs = root_states.shape[0]
        heights = torch.zeros(num_envs, device=self.device)

        if self.terrain is not None:
            points = root_states[..., :2].clone().reshape(num_envs, 1, 2)
            points -= self.terrain_offset
            points = (points / self.terrain.horizontal_scale).long()
            px = points[:, :, 0].view(-1)
            py = points[:, :, 1].view(-1)
            px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
            py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

            heights1 = self.height_samples[px, py]
            heights2 = self.height_samples[px + 1, py + 1]
            heights = torch.max(heights1, heights2)

        return heights.view(num_envs, -1)

    def init_height_points(self):
        y = torch.tensor(np.linspace(-self.config.terrain.terrain_config.sample_width, self.config.terrain.terrain_config.sample_width, self.config.terrain.terrain_config.num_samples), device=self.device, requires_grad=False)
        x = torch.tensor(np.linspace(-self.config.terrain.terrain_config.sample_width, self.config.terrain.terrain_config.sample_width, self.config.terrain.terrain_config.num_samples), device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def get_heights(self, env_ids=None, return_all_dims=False):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device).long()
        num_envs = len(env_ids)

        if self.terrain is None:
            return torch.zeros(num_envs, self.num_height_points, 1, device=self.device, requires_grad=False).view(num_envs, -1)
        root_states = self.get_humanoid_root_states()[env_ids].clone().view(num_envs, -1)
        base_pos = root_states[:, :3]
        base_pos[:, :2] = self.convert_to_global_coords(base_pos[:, :2], self.env_offsets[env_ids, :2].clone())
        base_quat = root_states[:, 3:7]

        points = rotations.quat_apply_yaw(
            base_quat.repeat(1, self.num_height_points),
            self.height_points[env_ids],
            self.w_last
        ) + (base_pos[:, :3]).unsqueeze(1)

        points -= self.terrain_offset
        points = points / self.terrain.horizontal_scale
        floored_points = points.long()
        # this encompases 4 possible points.
        # points are the top left corner of the 4 points
        # we will interpolate between the 4 points to get the height
        px = floored_points[:, :, 0].view(-1)
        py = floored_points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0] - 2)
        py = torch.clip(py, 0, self.height_samples.shape[1] - 2)

        # Calculate the fractional part of the points' positions
        fx = points[:, :, 0].view(-1) - px.float()
        fy = points[:, :, 1].view(-1) - py.float()

        # Get the heights of the four surrounding points
        h_tl = self.height_samples[px, py]  # Top-left
        h_tr = self.height_samples[px + 1, py]  # Top-right
        h_bl = self.height_samples[px, py + 1]  # Bottom-left
        h_br = self.height_samples[px + 1, py + 1]  # Bottom-right

        # Perform bilinear interpolation
        h_t = h_tl + (h_tr - h_tl) * fx  # Top interpolation
        h_b = h_bl + (h_br - h_bl) * fx  # Bottom interpolation
        interpolated_heights = h_t + (h_b - h_t) * fy  # Final interpolation

        # heights = torch.min(heights1, heights2).view(num_envs, -1)
        heights = base_pos[:, 2:3] - interpolated_heights.view(num_envs, -1)

        if False: ## TODO support this --> self.config.terrain.velocity_map:
            velocity_map = torch.zeros((num_envs, self.num_height_points, 2)).to(points)

            velocities = self.get_humanoid_root_velocities()[env_ids]

            heading_rot = torch_utils.calc_heading_quat_inv(base_quat, self.w_last)

            linear_vel_ego = torch_utils.quat_rotate(heading_rot, velocities, self.w_last)
            velocity_map[:] = velocity_map[:] - linear_vel_ego[:, None, :2]  # Flip velocity to be in agent's point of view

        if return_all_dims:
            # This is only for visualization purposes, plotting the height map the humanoid sees
            points = rotations.quat_apply_yaw(
                base_quat.repeat(1, self.num_height_points),
                self.height_points[env_ids],
                self.w_last
            ) + (base_pos[:, :3]).unsqueeze(1)
            heights = interpolated_heights.view(num_envs, -1, 1)
            return torch.cat([points[..., :2].view(num_envs, -1, 2), heights], dim=-1).clone()


        return heights.view(num_envs, -1).clone()

    ###############################################################
    # Helpers
    ###############################################################
    def action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def build_termination_heights(self):
        head_term_height = self.config.head_termination_height
        shield_term_height = self.config.shield_termination_height
        termination_height = self.config.termination_height

        termination_heights = np.array([termination_height] * self.num_bodies)

        if "smpl" in self.config.asset.asset_file_name:
            head_id = self.get_body_id("Head")
        else:
            head_id = self.get_body_id("head")

        termination_heights[head_id] = max(
            head_term_height, termination_heights[head_id]
        )

        asset_file = self.config.asset.asset_file_name
        if "amp_humanoid_sword_shield" in asset_file:
            left_arm_id = self.get_body_id("left_lower_arm")

            termination_heights[left_arm_id] = max(
                shield_term_height, termination_heights[left_arm_id]
            )

        self.termination_heights = torch_utils.to_torch(
            termination_heights, device=self.device
        )

    def transfer_to_env_coordinates(self, root_pos, env_ids):
        return root_pos

    def randomize_color(self, env_ids):
        base_col = np.array([0.4, 0.4, 0.4])
        range_col = np.array([0.0706, 0.149, 0.2863])
        range_sum = np.linalg.norm(range_col)

        rand_col = np.random.uniform(0.0, 1.0, size=3)
        rand_col = range_sum * rand_col / np.linalg.norm(rand_col)
        rand_col += base_col
        self.set_char_color(rand_col, env_ids)

    def set_char_color(self, rand_col, env_ids):
        raise NotImplementedError

    def convert_to_global_coords(self, humanoid_root_states, env_offsets):
        return humanoid_root_states

    def convert_to_local_coords(self, humanoid_root_states, env_offsets):
        return humanoid_root_states
    
