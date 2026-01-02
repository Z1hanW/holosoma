# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import torch

from motion_tracking.utils.motion_lib import MotionLib

from typing import Optional, TYPE_CHECKING
if TYPE_CHECKING:
    from motion_tracking.envs.isaacgym.task_humanoid import TaskHumanoid
else:
    TaskHumanoid = object


class BaseTask(TaskHumanoid):
    def __init__(self, config, device, motion_lib: Optional[MotionLib] = None):
        self.enable_task_obs = config.enable_task_obs
        self.use_transformer = config.use_transformer
        
        
        super().__init__(config, device, motion_lib=motion_lib)
        self.mimic_goal = None
        if self.use_transformer:

            if self.seperate_point_goal:

                if self.config.num_task_obs == 0:
                    self.mimic_scene = torch.zeros(
                            self.num_envs, self.config.num_obs_mimic_scene * self.config.num_obs_num_point,
                            dtype=torch.float, device=self.device
                        )
                    if self.use_mimic_goal_obs and self.config.num_obs_mimic_goal > 0:
                        self.mimic_goal = torch.zeros(
                                self.num_envs, self.config.num_obs_mimic_goal,
                                dtype=torch.float, device=self.device
                            )
                elif self.config.num_task_obs == 1000:
                    self.mimic_scene = torch.zeros(
                            self.num_envs, 1000,
                            dtype=torch.float, device=self.device
                        )
                    if self.use_mimic_goal_obs and self.config.num_obs_mimic_goal > 0:
                        self.mimic_goal = torch.zeros(
                            self.num_envs, self.config.num_obs_mimic_goal,
                            dtype=torch.float, device=self.device
                        )


            else:
                self.mimic_scene = torch.zeros(
                        self.num_envs, self.config.num_obs_mimic_scene,
                        dtype=torch.float, device=self.device
                    )
                self.mimic_goal = None

            
        self.setup_task()

    ###############################################################
    # Set up environment
    ###############################################################
    def setup_task(self):
        pass

    ###############################################################
    # Getters
    ###############################################################
    def get_obs_size(self):
        obs_size = super().get_obs_size()
        if self.enable_task_obs:
            task_obs_size = self.get_task_obs_size()
            if not self.use_transformer:
                obs_size += task_obs_size
            else:
                obs_size = obs_size
        return obs_size

    def get_task_obs_size(self):
        return 0

    ###############################################################
    # Handle reset
    ###############################################################
    def reset_envs(self, env_ids):
        super().reset_envs(env_ids)
        self.reset_task(env_ids)

    def reset_task(self, env_ids):
        pass

    ###############################################################
    # Environment step logic
    ###############################################################
    def compute_humanoid_obs(self, env_ids=None):
        humanoid_obs = super().compute_humanoid_obs(env_ids)
        
        if self.enable_task_obs:
            task_obs = self.compute_task_obs(env_ids)
            if not self.use_transformer:
                obs = torch.cat([humanoid_obs, task_obs], dim=-1)
            else:
                obs = humanoid_obs
                if self.seperate_point_goal:
                    goal_dim = self.config.num_obs_mimic_goal if self.use_mimic_goal_obs else 0
                    if env_ids is None:
                        if goal_dim > 0:
                            self.mimic_scene[:] = task_obs[:,:-goal_dim]
                        else:
                            self.mimic_scene[:] = task_obs
                        if self.use_mimic_goal_obs and self.mimic_goal is not None and goal_dim > 0:
                            self.mimic_goal[:] = task_obs[:,-goal_dim:]
                    else:
                        
                        if goal_dim > 0:
                            self.mimic_scene[env_ids] = task_obs[:,:-goal_dim]
                        else:
                            self.mimic_scene[env_ids] = task_obs
                        if self.use_mimic_goal_obs and self.mimic_goal is not None and goal_dim > 0:
                            self.mimic_goal[env_ids] = task_obs[:,-goal_dim:]
                else:
                    if env_ids is None:
                        self.mimic_scene[:] = task_obs
                    else:
                        self.mimic_scene[env_ids] = task_obs
        else:
            obs = humanoid_obs

        return obs

    def compute_task_obs(self, env_ids=None):
        return 0

    def compute_reward(self, actions):
        return 0

    def pre_physics_step(self, actions):
        self.update_task(actions)
        super().pre_physics_step(actions)

    def update_task(self, actions):
        pass

    ###############################################################
    # Helpers
    ###############################################################
    def draw_task(self):
        return
