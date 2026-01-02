# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from isaacgym import gymtorch

from motion_tracking.envs.common.common_task import BaseTask
from motion_tracking.envs.isaacgym.disc_humanoid import DiscHumanoid


class TaskHumanoid(BaseTask, DiscHumanoid):
    ###############################################################
    # Set up IsaacGym environment
    ###############################################################
    def build_env(self, env_id, env_ptr, humanoid_asset):
        super().build_env(env_id, env_ptr, humanoid_asset)
        self.build_env_task(env_id, env_ptr, humanoid_asset)

    def build_env_task(self, env_id, env_ptr, humanoid_asset):
        pass

    ###############################################################
    # Handle reset
    ###############################################################
    def reset_env_tensors(self, env_ids):
        super().reset_env_tensors(env_ids)

        env_ids_int32 = self.get_task_actor_ids_for_reset(env_ids)
        if env_ids_int32 is not None:
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states),
                                                         gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    ###############################################################
    # Getters
    ###############################################################
    def get_task_actor_ids_for_reset(self, env_ids):
        return None

    ###############################################################
    # Helpers
    ###############################################################
    def render(self):
        super().render()

        if self.viewer:
            self.draw_task()
