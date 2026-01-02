# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
from isaacgym import gymapi
import torch
from motion_tracking.envs.common.common_interface import BaseInterface
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    # Import IsaacGym Humanoid for autocompletion. Not imported during runtime.
    from motion_tracking.envs.isaacgym.humanoid import Humanoid as Tmp
else:
    Tmp = object


# Base class for RL tasks
class GymBaseInterface(BaseInterface, Tmp):
    def __init__(
        self,
        config,
        device: torch.device,
    ):
        super().__init__(config, device)
        self.gym = gymapi.acquire_gym()

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.original_props = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if not self.headless:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT"
            )
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync"
            )

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        self.init_done = True
        

    def get_obs_size(self):
        raise NotImplementedError

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self):
        sim = self.gym.create_sim(
            self.device.index, self.graphics_device_id, self.physics_engine, self.sim_params
        )
        if sim is None:
            print("*** Failed to create sim")
            quit()

        self.sim = sim

    def step(self, actions):
        # apply actions
        self.pre_physics_step(actions)

        # step physics and render each frame
        self.physics_step()

        # to fix!
        if self.device.type == "cpu":
            self.gym.fetch_results(self.sim, True)

        # compute observations, rewards, resets, ...
        self.post_physics_step()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def render(self):
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device.type != "cpu":
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)

    def simulate(self):
        self.render()
        self.gym.simulate(self.sim)

    def close(self):
        if self.viewer:
            self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
