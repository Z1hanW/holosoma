from __future__ import annotations

import math
import numbers
from dataclasses import replace
from typing import Any

from loguru import logger

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.torch_utils import torch_rand_float


class LeggedRobotLocomotionManager(BaseTask):
    # _reset_buffers_callback marks exactly the rows passed to reset_envs_idx;
    # BaseTask may therefore reuse its already-materialized reset selection.
    supports_direct_post_reset_refresh_ids = True

    BASE_NUM_ENVS = 4096

    def __init__(
        self,
        tyro_config,
        *,
        device,
    ):
        self.init_done = False
        super().__init__(
            tyro_config,
            device=device,
        )
        self.init_done = True

    def _get_task_name(self) -> str:
        return "locomotion"

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        self.base_quat = self.simulator.base_quat

        # initialize some data used later on
        self._init_counters()
        # joint positions offsets
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)  # (1, num_dof)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()  # (num_envs, num_dof)

        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)

        self._init_domain_rand_buffers()

        self.lidar_height_offset = getattr(self.robot_config, "lidar_height_offset", 0.5)

    def _init_counters(self):
        self.common_step_counter = 0

    def _update_counters_each_step(self):
        self.common_step_counter += 1

    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_push_robots = False
        self._max_push_vel = torch.zeros(2, dtype=torch.float32, device=self.device)

    def _setup_robot_body_indices(self):
        foot_body_names = [s for s in self.body_names if self.robot_config.foot_body_name in s]
        foot_height_names = [s for s in self.body_names if self.robot_config.foot_height_name in s]

        termination_contact_names = []
        for name in self.robot_config.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in self.body_names if name in s])

        self.feet_indices = torch.zeros(len(foot_body_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i, name in enumerate(foot_body_names):
            self.feet_indices[i] = self.simulator.find_rigid_body_indice(name)

        self.feet_height_indices = torch.zeros(
            len(foot_height_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i, name in enumerate(foot_height_names):
            self.feet_height_indices[i] = self.simulator.find_rigid_body_indice(name)

        self.termination_contact_indices = torch.zeros(
            len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False
        )
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.simulator.find_rigid_body_indice(termination_contact_names[i])

        if self.robot_config.has_torso:
            self.torso_name = self.robot_config.torso_name
            self.torso_index = self.simulator.find_rigid_body_indice(self.torso_name)

    def set_is_evaluating(self, command=None):
        logger.info("Setting Env is evaluating")
        super().set_is_evaluating()
        commands = self.command_manager.commands
        commands.zero_()
        if command is not None:
            command_tensor = torch.as_tensor(command, device=self.device, dtype=commands.dtype)
            commands[:] = command_tensor.view(1, -1).expand_as(commands)
        gait_state = self.command_manager.get_state("locomotion_gait")
        gait_state.set_eval_mode(True)

    def _setup_simulator_next_task(self):
        pass

    def _setup_simulator_control(self):
        self.simulator.commands = self.command_manager.commands

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback(env_ids)

    def _pre_compute_observations_callback(self, env_ids: torch.Tensor | None = None):
        # prepare quantities
        if env_ids is None:
            self.base_quat[:] = self.simulator.base_quat[:]
        else:
            self.base_quat[env_ids] = self.simulator.base_quat[env_ids]
        super()._pre_compute_observations_callback(env_ids)
        self.terrain_manager.update_heights(env_ids)

    def _update_tasks_callback(self):
        super()._update_tasks_callback()

        # Assign commands to simulator for headless recording
        if hasattr(self.simulator, "headless_recording") and self.simulator.headless_recording:
            if hasattr(self.command_manager, "commands"):
                self.simulator.commands = self.command_manager.commands

    def _post_compute_observations_callback(self):
        return

    def reset_all(self):
        self._init_buffers()
        return super().reset_all()

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # if target_states is not None, reset to target states
        if target_states is not None:
            self._reset_dofs(env_ids, target_states["dof_states"])
            self._reset_root_states(env_ids, target_states["root_states"])
        else:
            self._reset_dofs(env_ids)
            self._reset_root_states(env_ids)

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        # Observation manager reset is now handled in base_task.py
        self.need_to_refresh_envs[env_ids] = True

        if target_buf is not None:
            self.simulator.dof_pos[env_ids] = target_buf["dof_pos"].to(self.simulator.dof_pos.dtype)
            self.simulator.dof_vel[env_ids] = target_buf["dof_vel"].to(self.simulator.dof_vel.dtype)
            self.base_quat[env_ids] = target_buf["base_quat"].to(self.base_quat.dtype)
            self.episode_length_buf[env_ids] = target_buf["episode_length_buf"].to(self.episode_length_buf.dtype)
            self.reset_buf[env_ids] = target_buf["reset_buf"].to(self.reset_buf.dtype)
            self.time_out_buf[env_ids] = target_buf["time_out_buf"].to(self.time_out_buf.dtype)
            self._pending_episode_update_mask[env_ids] = False
            self._pending_episode_lengths[env_ids] = 0
        else:
            self.episode_length_buf[env_ids] = 0
            self.reset_buf[env_ids] = 1
            self._pending_episode_update_mask[env_ids] = True

    def _update_log_dict(self):
        avg = self._get_average_episode_tracker().get_average()
        # LoggingHelper performs a single batched device-to-host transfer at
        # the iteration boundary; avoid synchronizing the rollout here.
        self.log_dict["average_episode_length"] = avg.detach()

    ################ Curriculum #################

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return tracker

    def _get_penalty_curriculum(self):
        return self.curriculum_manager.get_term("penalty_curriculum")

    @property
    def average_episode_length(self) -> float:
        avg = self._get_average_episode_tracker().get_average()
        return float(avg.detach().cpu().item())

    # ------------------------------------------------------------------
    # Checkpoint helpers

    def get_checkpoint_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "version": 3,
            "average_episode_tracker": self._get_average_episode_tracker().state_dict(),
            "perception_managers": self._get_perception_checkpoint_state(),
        }
        penalty = self._get_penalty_curriculum()
        if penalty is not None and bool(getattr(penalty, "enabled", False)):
            state["penalty_curriculum"] = penalty.state_dict()
        return state

    def validate_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        if not isinstance(state, dict):
            raise ValueError("Locomotion environment checkpoint state must be a dictionary.")
        version = state.get("version", 1)
        if isinstance(version, bool) or not isinstance(version, int) or version not in (1, 2, 3):
            raise ValueError(f"Unsupported locomotion environment checkpoint version: {version!r}.")

        tracker_state = state.get("average_episode_tracker")
        if not isinstance(tracker_state, dict):
            raise ValueError("Locomotion checkpoint is missing average_episode_tracker state.")
        self._get_average_episode_tracker().validate_state_dict(tracker_state)

        penalty = self._get_penalty_curriculum()
        penalty_enabled = penalty is not None and bool(
            getattr(penalty, "enabled", False)
        )
        if version in (2, 3):
            penalty_state = state.get("penalty_curriculum")
            if not penalty_enabled:
                if penalty_state is not None:
                    raise ValueError(
                        "Locomotion checkpoint contains penalty curriculum state, but it is disabled."
                    )
            elif not isinstance(penalty_state, dict):
                raise ValueError("Locomotion checkpoint is missing enabled penalty curriculum state.")
            else:
                penalty.validate_state_dict(penalty_state)
            supported_keys = {
                "version",
                "average_episode_tracker",
                "penalty_curriculum",
            }
            if version == 3:
                supported_keys.add("perception_managers")
            unexpected = set(state) - supported_keys
            if unexpected:
                raise ValueError(
                    f"Locomotion checkpoint contains unsupported state keys: {sorted(unexpected)}."
                )
        else:
            legacy_scale = state.get("reward_penalty_scale")
            if penalty_enabled:
                if torch.is_tensor(legacy_scale):
                    if legacy_scale.numel() != 1:
                        raise ValueError("Legacy reward_penalty_scale must be scalar.")
                    legacy_scale = legacy_scale.detach().cpu().item()
                if isinstance(legacy_scale, bool) or not isinstance(legacy_scale, numbers.Real):
                    raise ValueError(
                        "Legacy locomotion checkpoint is missing a numeric reward_penalty_scale."
                    )
                scale = float(legacy_scale)
                if not math.isfinite(scale) or not penalty.min_scale <= scale <= penalty.max_scale:
                    raise ValueError("Legacy reward_penalty_scale is outside configured bounds.")
        if version == 3:
            self._validate_perception_checkpoint_state(state.get("perception_managers"))
        else:
            self._validate_perception_checkpoint_state(None)

    def _load_checkpoint_state(
        self,
        state: dict[str, Any] | None,
        *,
        suppress_tracker_update: bool,
        restore_perception: bool,
    ) -> None:
        if not state:
            return
        self.validate_checkpoint_state(state)
        version = state.get("version", 1)
        tracker = self._get_average_episode_tracker()
        tracker.load_state_dict(state["average_episode_tracker"])
        # The caller owns the one-shot reset guard.  In particular, a
        # canonical checkpoint-boundary reset consumes the guard armed by a
        # normal checkpoint load; restoring the adaptive snapshot afterward
        # must not resurrect it and suppress the first genuine episode.
        tracker.set_next_update_suppressed(suppress_tracker_update)

        penalty = self._get_penalty_curriculum()
        if penalty is not None and bool(getattr(penalty, "enabled", False)):
            if version in (2, 3):
                penalty.load_state_dict(state["penalty_curriculum"])
            else:
                legacy_scale = state["reward_penalty_scale"]
                if torch.is_tensor(legacy_scale):
                    legacy_scale = legacy_scale.detach().cpu().item()
                penalty_state = penalty.state_dict()
                penalty_state["current_scale"] = float(legacy_scale)
                penalty.load_state_dict(penalty_state)
        if restore_perception and version == 3:
            self._load_perception_checkpoint_state(state["perception_managers"])

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        self._load_checkpoint_state(
            state,
            suppress_tracker_update=True,
            restore_perception=True,
        )

    def _restore_checkpoint_state_after_canonical_reset(self, state: dict[str, Any] | None) -> None:
        self._load_checkpoint_state(
            state,
            suppress_tracker_update=False,
            restore_perception=False,
        )

    def synchronize_curriculum_state(self, *, device: str, world_size: int, process_group=None) -> None:
        if world_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed locomotion curriculum synchronization requires an initialized process group."
            )
        tracker = self._get_average_episode_tracker()
        avg_tensor = tracker.get_average().clone().detach().to(device)
        torch.distributed.broadcast(avg_tensor, src=0, group=process_group)
        tracker.set_average(avg_tensor.to(self.device), suppress_update=False)

        penalty = self._get_penalty_curriculum()
        legacy_cfg = getattr(self, "_curriculum_penalty_cfg", None)
        if penalty is not None and bool(getattr(penalty, "enabled", False)):
            penalty_tensor = torch.tensor(
                float(penalty.current_scale),
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.broadcast(penalty_tensor, src=0, group=process_group)
            penalty_state = penalty.state_dict()
            penalty_state["current_scale"] = float(penalty_tensor.item())
            # The term owns both the authoritative scalar and the live reward
            # weights.  Updating only ``env.reward_penalty_scale`` leaves the
            # actual distributed objectives divergent.
            penalty.load_state_dict(penalty_state)
        elif isinstance(legacy_cfg, dict) and bool(
            getattr(self, "use_reward_penalty_curriculum", False)
        ):
            penalty_tensor = torch.tensor(
                float(legacy_cfg["current_scale"]),
                device=device,
                dtype=torch.float64,
            )
            torch.distributed.broadcast(penalty_tensor, src=0, group=process_group)
            synchronized_scale = float(penalty_tensor.item())
            if not math.isfinite(synchronized_scale) or not (
                float(legacy_cfg["min_scale"])
                <= synchronized_scale
                <= float(legacy_cfg["max_scale"])
            ):
                raise ValueError("Synchronized legacy penalty scale is invalid.")
            legacy_cfg["current_scale"] = synchronized_scale
            original_weights = getattr(self, "_curriculum_penalty_original_weights", {})
            penalty_names = getattr(self, "_curriculum_penalty_reward_names", [])
            for name in penalty_names:
                if name not in original_weights or name not in self.reward_manager.active_terms:
                    continue
                term_cfg = self.reward_manager.get_term_cfg(name)
                self.reward_manager.set_term_cfg(
                    name,
                    replace(
                        term_cfg,
                        weight=float(original_weights[name]) * synchronized_scale,
                    ),
                )
            self.reward_penalty_scale = synchronized_scale
            if hasattr(self, "log_dict"):
                self.log_dict["penalty_scale"] = torch.tensor(
                    synchronized_scale,
                    dtype=torch.float,
                )
        elif hasattr(self, "reward_penalty_scale"):
            raise RuntimeError(
                "Penalty curriculum exposes a mirrored scale but no authoritative term/config to synchronize."
            )

    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 2:
            raise ValueError("Locomotion push velocity vector must have exactly 2 components.")

        rand = torch.rand(len(env_ids), 2, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        self.simulator.robot_root_states[env_ids, 7:9] = self.push_robot_vel_buf[env_ids]
        # This writes through immediately.  Do not queue reset-only refresh work:
        # that path also clears contact history and re-runs perception for the rows.
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    ################ ENV CALLBACKS #################

    def _reset_dofs(self, env_ids, target_state=None):
        """Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.
        If target_state is not None, reset to target_state

        Args:
            env_ids (List[int]): Environemnt ids
            target_state (Tensor): Target state
        """
        if target_state is not None:
            self.simulator.dof_pos[env_ids] = target_state[..., 0]
            self.simulator.dof_vel[env_ids] = target_state[..., 1]
        else:
            self.simulator.dof_pos[env_ids] = self.default_dof_pos[env_ids] * torch_rand_float(
                0.5, 1.5, (len(env_ids), self.num_dof), device=str(self.device)
            )
            self.simulator.dof_vel[env_ids] = 0.0

    def _reset_root_states(self, env_ids, target_root_states=None):
        """Resets ROOT states position and velocities of selected environmments
            if target_root_states is not None, reset to target_root_states
        Args:
            env_ids (List[int]): Environemnt ids
            target_root_states (Tensor): Target root states
        """
        if target_root_states is not None:
            self.simulator.robot_root_states[env_ids] = target_root_states
            self.simulator.robot_root_states[env_ids, :3] += self.terrain_manager.get_state(
                "locomotion_terrain"
            ).env_origins[env_ids]

        else:
            # base position
            self.simulator.robot_root_states[env_ids] = self.base_init_state
            self.simulator.robot_root_states[env_ids, :3] += self.terrain_manager.get_state(
                "locomotion_terrain"
            ).env_origins[env_ids]

            # Apply randomized XY offset if custom_origins
            if self.terrain_manager.get_state("locomotion_terrain").custom_origins:
                # Get spawn config from terrain term
                spawn_cfg = self.terrain_manager.cfg.terrain_term.spawn

                # Generate random XY offsets
                xy_offsets = torch_rand_float(
                    -spawn_cfg.xy_offset_range, spawn_cfg.xy_offset_range, (len(env_ids), 2), device=str(self.device)
                )

                if spawn_cfg.query_terrain_height:
                    # ACCURATE: Query terrain height at new XY position (slower, for rough terrain)
                    current_xy = self.simulator.robot_root_states[env_ids, :2]
                    new_xy = current_xy + xy_offsets

                    terrain_state = self.terrain_manager.get_state("locomotion_terrain")
                    terrain_heights = terrain_state.query_terrain_heights(
                        new_xy,
                        use_grid_sampling=spawn_cfg.use_grid_sampling,
                        grid_size=spawn_cfg.grid_size,
                        grid_spacing=spawn_cfg.grid_spacing,
                    )
                    robot_base_height = self.robot_config.init_state.pos[2]  # Robot base height above ground
                    new_z = terrain_heights + robot_base_height

                    # Write new XYZ position all at once
                    new_xyz = torch.cat([new_xy, new_z.unsqueeze(1)], dim=1)
                    self.simulator.robot_root_states[env_ids, :3] = new_xyz
                else:
                    # FAST: Original simple spawning - just apply XY offset, keep original Z (faster, for flat terrain)
                    self.simulator.robot_root_states[env_ids, :2] += xy_offsets

            # base velocities
            self.simulator.robot_root_states[env_ids, 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids), 6), device=str(self.device)
            )  # [7:10]: lin vel, [10:13]: ang vel
