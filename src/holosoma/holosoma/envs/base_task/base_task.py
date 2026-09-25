from __future__ import annotations

import inspect
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from holosoma.config_types.env import EnvConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.managers.action import ActionManager
from holosoma.managers.command import CommandManager
from holosoma.managers.curriculum import CurriculumManager
from holosoma.managers.observation import ObservationManager
from holosoma.managers.perception import PerceptionManager
from holosoma.managers.randomization import RandomizationManager
from holosoma.managers.reset_events.manager import ResetEventManager
from holosoma.managers.reward import RewardManager
from holosoma.managers.termination import TerminationManager
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.utils.helpers import get_class
from holosoma.utils.rollout_recorder import RolloutRecorder
from holosoma.utils.simulator_config import SimulatorType
from holosoma.utils.viser_live import ViserLiveViewer
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.step_timing import StepTiming
from holosoma.utils.torch_utils import to_torch


# Base class for RL tasks built around the manager-based architecture.
class BaseTask:
    def __init__(
        self,
        tyro_config: EnvConfig,
        *,
        device: str,
    ):
        """Initialize task with manager-based observation, action, and reward systems.

        Parameters
        ----------
        tyro_config: EnvConfig
            Environment configuration
        device: str
            Device to run on
        """
        self._manager_domain_rand_cfg = None
        self.is_evaluating = False
        # Direct/FastSAC consumers historically receive dense per-environment
        # reward episode statistics on every step.  PPO opts into sparse
        # reset-only statistics through ``set_collection_extras_contract`` so
        # its common no-reset path can skip the entire reset stack.
        self._dense_episode_stats_each_step = True

        observation_config = tyro_config.observation
        simulator_config = tyro_config.simulator
        terrain_config = tyro_config.terrain
        perception_config = tyro_config.perception
        teacher_perception_config = tyro_config.teacher_perception
        critic_perception_config = tyro_config.critic_perception
        robot_config = tyro_config.robot
        action_config = tyro_config.action
        reward_config = tyro_config.reward
        termination_config = tyro_config.termination
        randomization_config = tyro_config.randomization
        command_config = tyro_config.command
        curriculum_config = tyro_config.curriculum
        training_config = tyro_config.training

        self.training_config = training_config
        self.robot_config = robot_config

        # Validate configs: manager workflow requires all manager configs to be provided
        if observation_config is None:
            raise ValueError("observation_config must be provided for manager-based environments.")
        if action_config is None:
            raise ValueError("action_config must be provided for manager-based environments.")
        if reward_config is None:
            raise ValueError("reward_config must be provided for manager-based environments.")
        if termination_config is None:
            raise ValueError("termination_config must be provided for manager-based environments.")
        if randomization_config is None:
            raise ValueError("randomization_config must be provided for manager-based environments.")
        if command_config is None:
            raise ValueError("command_config must be provided for manager-based environments.")
        if curriculum_config is None:
            raise ValueError("curriculum_config must be provided for manager-based environments.")
        if terrain_config is None:
            raise ValueError("terrain_config must be provided for manager-based environments.")

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # Training publishes one process-wide identity before constructing the
        # environment.  Reuse it so simulator/video paths cannot drift away
        # from checkpoints and rank logs due to a later local timestamp.
        from holosoma.utils.experiment_paths import get_process_experiment_dir

        experiment_dir = get_process_experiment_dir(
            tyro_config.logger,
            tyro_config.training,
            task_name=self._get_task_name(),
            use_override_task_name=True,
        )

        SimulatorClass = get_class(simulator_config._target_)
        full_sim_config = FullSimConfig(
            simulator=simulator_config.config,
            robot=robot_config,
            training=training_config,
            logger=tyro_config.logger,
            command=command_config,
            experiment_dir=str(experiment_dir),
        )

        self.num_envs = training_config.num_envs
        self.dim_obs = robot_config.policy_obs_dim
        self.dim_critic_obs = robot_config.critic_obs_dim
        self.dim_actions = robot_config.actions_dim
        self.device = device
        self.step_timing = StepTiming.from_env(device=self.device)

        self.terrain_manager = TerrainManager(terrain_config, self, device)
        self.simulator: BaseSimulator = SimulatorClass(
            tyro_config=full_sim_config, terrain_manager=self.terrain_manager, device=device
        )
        setattr(self.simulator, "step_timing", self.step_timing)

        self.headless = self.training_config.headless
        self.simulator.set_headless(self.headless)
        self.simulator.setup()
        self.sim_dt = self.simulator.sim_dt

        self.dt = simulator_config.config.sim.control_decimation * self.sim_dt
        self.max_episode_length_s = simulator_config.config.sim.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.simulator.setup_terrain()
        # create envs, sim and viewer
        self._load_assets()

        # For IsaacGym manager-based environments: Initialize randomization manager BEFORE creating envs
        # so it can be applied during env creation (before prepare_sim).
        # For IsaacSim: The manager will be initialized later (scene is already created in __init__)
        is_isaacgym_manager = hasattr(self.simulator, "gym")
        if is_isaacgym_manager:
            self.randomization_manager = RandomizationManager(randomization_config, self, self.device)
            if self.randomization_manager is not None:
                self.simulator.set_startup_randomization_callback(self.randomization_manager.setup)

        self._create_envs()
        self.dof_pos_limits, self.dof_vel_limits, self.torque_limits = self.simulator.get_dof_limits_properties()
        self._setup_robot_body_indices()
        self.simulator.prepare_sim()

        self.perception_manager = None
        if perception_config is not None:
            self.perception_manager = PerceptionManager(perception_config, self, self.device)
        self.teacher_perception_manager = None
        if teacher_perception_config is not None:
            self.teacher_perception_manager = PerceptionManager(teacher_perception_config, self, self.device)
        self.critic_perception_manager = None
        if critic_perception_config is not None:
            if perception_config is not None and critic_perception_config == perception_config:
                self.critic_perception_manager = self.perception_manager
            elif teacher_perception_config is not None and critic_perception_config == teacher_perception_config:
                self.critic_perception_manager = self.teacher_perception_manager
            else:
                self.critic_perception_manager = PerceptionManager(critic_perception_config, self, self.device)

        # if running with a viewer, set up keyboard shortcuts and camera
        self.viewer = None
        if not self.headless:
            self.debug_viz = False
            self.simulator.setup_viewer()
            self.viewer = self.simulator.viewer
        self._isaac_scandots_last_update = 0.0
        self._isaac_scandots_warned = False
        self._isaac_scandots_payload: dict[str, object] | None = None

        # Initialize remaining managers
        self.observation_manager = ObservationManager(observation_config, self, self.device)
        self.action_manager = ActionManager(action_config, self, self.device)
        self.reward_manager = RewardManager(reward_config, self, self.device)
        self.termination_manager = TerminationManager(termination_config, self, self.device)
        # For IsaacSim, initialize randomization_manager now
        if not is_isaacgym_manager:
            self.randomization_manager = RandomizationManager(randomization_config, self, self.device)
        self.command_manager = CommandManager(command_config, self, self.device)
        self.curriculum_manager = CurriculumManager(curriculum_config, self, self.device)

        self._init_buffers()

        # Prepare fields required by managers (BEFORE setup calls)
        # This scans decorator metadata and expands model fields for per-environment randomization
        self.simulator.prepare_manager_fields(
            randomization_manager=self.randomization_manager,
            observation_manager=self.observation_manager,
            reward_manager=self.reward_manager,
        )

        # Call setup for managers that need it
        if self.randomization_manager is not None and not is_isaacgym_manager:
            self.randomization_manager.setup()
        if self.action_manager is not None:
            self.action_manager.setup()
        if self.command_manager is not None:
            self.command_manager.setup()
        if self.curriculum_manager is not None:
            self.curriculum_manager.setup()
        if self.terrain_manager is not None:
            self.terrain_manager.setup()
        self._validate_rendered_perception_topology()
        if self.perception_manager is not None:
            self.perception_manager.setup()
        if self.teacher_perception_manager is not None:
            self.teacher_perception_manager.setup()
        if (
            self.critic_perception_manager is not None
            and self.critic_perception_manager is not self.perception_manager
            and self.critic_perception_manager is not self.teacher_perception_manager
        ):
            self.critic_perception_manager.setup()
        self._init_depth_logging_state()
        self._rollout_recorder = RolloutRecorder(self)
        self._viser_live = ViserLiveViewer(self)

        # Initialize reset manager from simulator config
        self.reset_manager = ResetEventManager(
            self.simulator.simulator_config.reset_manager, self.simulator, self.device
        )

        if not self.headless:
            self.viewer = self.simulator.viewer

    def _init_buffers(self):
        # Record history length from observation manager config
        self.history_length = {}
        for group_name, group_cfg in self.observation_manager.cfg.groups.items():
            self.history_length[group_name] = group_cfg.history_length

        self.obs_buf_dict = {}

        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.log_dict = {}
        self._pending_episode_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._pending_episode_update_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._pending_torque_rfi: tuple[bool, float] = (False, 0.0)

    def _refresh_sim_tensors(self):
        self.simulator.refresh_sim_tensors()

    def _perception_checkpoint_topology(
        self,
    ) -> tuple[dict[str, str | None], dict[str, PerceptionManager]]:
        """Return role aliases and each unique enabled perception manager."""

        role_owners: dict[str, str | None] = {}
        managers_by_owner: dict[str, PerceptionManager] = {}
        owner_by_identity: dict[int, str] = {}
        for role, attribute in (
            ("actor", "perception_manager"),
            ("teacher", "teacher_perception_manager"),
            ("critic", "critic_perception_manager"),
        ):
            manager = getattr(self, attribute, None)
            if manager is None or not bool(getattr(manager, "enabled", False)):
                role_owners[role] = None
                continue
            identity = id(manager)
            owner = owner_by_identity.get(identity)
            if owner is None:
                owner = role
                owner_by_identity[identity] = owner
                managers_by_owner[owner] = manager
            role_owners[role] = owner
        return role_owners, managers_by_owner

    def _validate_rendered_perception_topology(self) -> None:
        """Reject unique rendered managers that target one shared camera."""

        _, managers_by_owner = self._perception_checkpoint_topology()
        target_owners: dict[tuple[str, int], str] = {}
        for owner, manager in managers_by_owner.items():
            if not manager._uses_rendered_camera():
                continue
            target = (
                str(getattr(manager, "_simulator_backend", "unknown")),
                int(getattr(manager, "_rendered_camera_env_id", 0)),
            )
            existing = target_owners.get(target)
            if existing is not None:
                raise RuntimeError(
                    "Multiple unique rendered perception managers target the same simulator camera "
                    f"{target}: owners={existing!r},{owner!r}. Their prim/model camera state would "
                    "overwrite each other; share one manager or use distinct camera targets."
                )
            target_owners[target] = owner

    def _get_perception_checkpoint_state(self) -> dict[str, Any]:
        # Every PPO checkpoint advertises an exact canonical continuation.
        # Refuse to publish one for a backend whose hidden temporal state is
        # known to be outside this envelope.
        self._validate_perception_exact_resume_supported()
        role_owners, managers_by_owner = self._perception_checkpoint_topology()
        return {
            "version": 1,
            "role_owners": role_owners,
            "states": {
                owner: manager.get_persistent_checkpoint_state()
                for owner, manager in managers_by_owner.items()
            },
        }

    def _perception_checkpoint_state_required(self) -> bool:
        _, managers_by_owner = self._perception_checkpoint_topology()
        return any(
            manager.persistent_checkpoint_state_required()
            for manager in managers_by_owner.values()
        )

    @property
    def environment_state_checkpoint_required(self) -> bool:
        """Whether a full resume must include this rank's environment state."""

        return self._perception_checkpoint_state_required()

    def _validate_perception_checkpoint_state(self, state: Any) -> None:
        expected_roles, managers_by_owner = self._perception_checkpoint_topology()
        if state is None:
            if self._perception_checkpoint_state_required():
                raise RuntimeError(
                    "Legacy environment checkpoint has no perception calibration/stream state, but an "
                    "enabled perception manager has persistent policy-input phase; exact resume is impossible."
                )
            return
        if not isinstance(state, dict) or set(state) != {
            "version",
            "role_owners",
            "states",
        }:
            raise ValueError("Perception-manager checkpoint envelope is malformed.")
        version = state.get("version")
        if isinstance(version, bool) or not isinstance(version, int) or version != 1:
            raise ValueError(f"Unsupported perception-manager checkpoint version: {version!r}.")
        if state.get("role_owners") != expected_roles:
            raise ValueError(
                "Perception-manager role/alias topology differs from the active runtime."
            )
        states = state.get("states")
        if not isinstance(states, dict) or set(states) != set(managers_by_owner):
            raise ValueError(
                "Perception-manager checkpoint state owners differ from the active runtime."
            )
        for owner, manager in managers_by_owner.items():
            manager.validate_persistent_checkpoint_state(states[owner])

    def _load_perception_checkpoint_state(self, state: Any) -> None:
        self._validate_perception_checkpoint_state(state)
        if state is None:
            return
        _, managers_by_owner = self._perception_checkpoint_topology()
        # Check every owner before mutating any of them so a mixed backend
        # topology fails atomically.
        for manager in managers_by_owner.values():
            manager.validate_exact_resume_supported()
        for owner, manager in managers_by_owner.items():
            manager.load_persistent_checkpoint_state(state["states"][owner])

    def _validate_perception_exact_resume_supported(self) -> None:
        _, managers_by_owner = self._perception_checkpoint_topology()
        for manager in managers_by_owner.values():
            manager.validate_exact_resume_supported()

    def _reset_perception_canonical_rollout_state(self) -> None:
        """Canonicalize each unique manager exactly once before reset warm-up."""

        _, managers_by_owner = self._perception_checkpoint_topology()
        for manager in managers_by_owner.values():
            manager.reset_canonical_rollout_state()

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Return environment-specific state to persist in checkpoints."""
        return {}

    def validate_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Validate environment checkpoint state without mutating live state."""
        if not state:
            return

    def validate_full_resume_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Validate state plus backend capabilities for an exact full resume."""

        self.validate_checkpoint_state(state)
        self._validate_perception_exact_resume_supported()

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore environment-specific state from a checkpoint."""
        if not state:
            return

    def _restore_checkpoint_state_after_canonical_reset(self, state: dict[str, Any] | None) -> None:
        """Restore adaptive state without arming a future reset suppression."""

        self.load_checkpoint_state(state)

    def reset_all_at_checkpoint_boundary(self):
        """Reset simulator/manager episode state while preserving curricula.

        Checkpoint restart intentionally discards live physical episodes, but
        the forced reset itself is not training evidence.  Preserve adaptive
        state around it so uninterrupted and resumed processes start from the
        same curriculum rather than counting the operational reset as a fall
        or successful episode.
        """

        checkpoint_state = self.get_checkpoint_state()
        self._reset_perception_canonical_rollout_state()
        observations = self.reset_all()
        self._restore_checkpoint_state_after_canonical_reset(checkpoint_state)
        return observations

    def synchronize_curriculum_state(self, *, device: str, world_size: int, process_group=None) -> None:
        """Synchronize curriculum-related state across distributed processes."""
        return

    def reset_all(self):
        """Reset all robots"""
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.reset_envs_idx(env_ids)

        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        # ``None`` is the standardized "write current state" contract.  On
        # IsaacSim this avoids assembling a full [env, dof, 2] tensor merely to
        # write the selected reset rows.
        self.simulator.set_dof_state_tensor_robots(env_ids)

        actions = torch.zeros(self.num_envs, self.dim_actions, device=self.device, requires_grad=False)
        actor_state = {}
        actor_state["actions"] = actions
        obs_dict, _, _, _ = self.step(actor_state)
        self._log_startup_depth_if_needed()
        return obs_dict

    def _simulator_episode_callback_ids(self, env_ids) -> list[int] | None:
        """Materialize reset IDs once when simulator lifecycle hooks are active.

        BaseSimulator exposes an explicit capability so the common training
        configuration avoids both CUDA iteration and synchronization.  Legacy
        duck-typed simulators without that capability remain compatible: the
        presence of either batched or scalar lifecycle hooks enables callbacks.
        """

        capability = getattr(self.simulator, "requires_episode_callbacks", None)
        if capability is None:
            callbacks_required = any(
                callable(getattr(self.simulator, hook_name, None))
                for hook_name in (
                    "on_episodes_start",
                    "on_episodes_end",
                    "on_episode_start",
                    "on_episode_end",
                )
            )
        else:
            callbacks_required = capability() if callable(capability) else bool(capability)

        if not callbacks_required:
            return None
        return env_ids.tolist()

    def _notify_simulator_episode_callbacks(
        self,
        phase: str,
        env_ids: list[int] | None,
    ) -> None:
        """Dispatch one batched hook, falling back to the legacy scalar hook."""

        if env_ids is None:
            return

        batch_callback = getattr(self.simulator, f"on_episodes_{phase}", None)
        if callable(batch_callback):
            batch_callback(env_ids)
            return

        scalar_callback = getattr(self.simulator, f"on_episode_{phase}", None)
        if callable(scalar_callback):
            for env_id in env_ids:
                scalar_callback(env_id)

    def reset_envs_idx(self, env_ids, target_states=None, target_buf=None):
        """Reset some environments and handle video recording callbacks."""

        # Callback IDs are copied to the host at most once, and only when a
        # simulator lifecycle consumer is actually active.
        episode_callback_ids = self._simulator_episode_callback_ids(env_ids)
        self._notify_simulator_episode_callbacks("end", episode_callback_ids)
        self._finalize_depth_logging_if_needed()
        self._finalize_startup_depth_video_if_needed(env_ids)
        if hasattr(self, "_rollout_recorder"):
            self._rollout_recorder.on_reset(env_ids)
        if hasattr(self, "_viser_live"):
            self._viser_live.on_reset(env_ids)

        # Reset observation history BEFORE state changes (must happen first to clear history buffers)
        self.observation_manager.reset(env_ids)
        reset_perception_manager_ids: set[int] = set()
        for perception_manager in (
            self.perception_manager,
            self.teacher_perception_manager,
            self.critic_perception_manager,
        ):
            if perception_manager is None or id(perception_manager) in reset_perception_manager_ids:
                continue
            reset_perception_manager_ids.add(id(perception_manager))
            perception_manager.reset(env_ids)

        self._pending_episode_lengths[env_ids] = self.episode_length_buf[env_ids]
        self._pending_episode_update_mask[env_ids] = False

        # Call the actual reset implementation (to be overridden by subclasses)
        self._reset_envs_idx_impl(env_ids, target_states, target_buf)

        # Reset all managers AFTER state changes
        if self.randomization_manager is not None:
            self.randomization_manager.reset(env_ids)

        if self.action_manager is not None:
            self.action_manager.reset(env_ids)

        if self.command_manager is not None:
            self.command_manager.reset(env_ids)

        if self.curriculum_manager is not None:
            self.curriculum_manager.reset(env_ids)

        if self.termination_manager is not None:
            self.termination_manager.reset(env_ids)

        # Call manager-based reset events
        self.reset_manager.reset_scene(env_ids)

        # Call episode start for environments that have been reset.
        self._notify_simulator_episode_callbacks("start", episode_callback_ids)
        self._start_depth_logging_if_needed()
        # Simulator state written by a reset still needs the task-specific
        # refresh pass.  Keep this host-side marker in addition to the per-env
        # device mask so the common no-reset step can avoid materializing that
        # mask with ``nonzero()``.  This also covers reset_all() and explicit
        # resets issued outside the post-physics loop.
        if len(env_ids) > 0:
            self._reset_refresh_pending = True

    def _reset_envs_idx_impl(self, env_ids, target_states=None, target_buf=None):
        """Template implementation of environment reset.

        Subclasses can override the helper hooks below to customize the reset behaviour.

        Args
        ----
        env_ids:
            Environments to reset.
        target_states:
            Optional dictionary containing desired DOF/root states.
        target_buf:
            Optional dictionary of buffered tensors to restore (e.g., for replay).
        """
        self._reset_buffers_callback(env_ids, target_buf)
        self._reset_tasks_callback(env_ids)
        self._reset_robot_states_callback(env_ids, target_states)
        self._fill_extras(env_ids)

    def render(self, sync_frame_time=True):
        if self.viewer:
            self.simulator.render(sync_frame_time)

    ###########################################################################
    #### Helper functions

    @property
    def domain_rand_cfg(self):
        """Return the active domain randomization configuration."""
        return self._manager_domain_rand_cfg

    ###########################################################################
    def _load_assets(self):
        self.simulator.load_assets()
        self.num_dof, self.num_bodies, self.dof_names, self.body_names = (
            self.simulator.num_dof,
            self.simulator.num_bodies,
            self.simulator.dof_names,
            self.simulator.body_names,
        )

        # check dimensions
        assert self.num_dof == self.dim_actions, (
            f"Number of DOFs ({self.num_dof}) must be equal to number of actions ({self.dim_actions})"
        )

        # other properties
        self.num_bodies = len(self.body_names)
        self.num_dofs = len(self.dof_names)
        base_init_state_list = (
            self.robot_config.init_state.pos
            + self.robot_config.init_state.rot
            + self.robot_config.init_state.lin_vel
            + self.robot_config.init_state.ang_vel
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def _create_envs(self):
        """Creates environments:
        1. loads the robot URDF/MJCF asset,
        2. For each environment
           2.1 creates the environment,
           2.2 calls DOF and Rigid shape properties callbacks,
           2.3 create actor with these properties and add them to the env
        3. Store indices of different bodies of the robot
        """
        self.simulator.create_envs(self.num_envs, self._get_env_origins(), self.base_init_state)

    def _setup_robot_body_indices(self):
        """Hook for subclasses to prepare body index caches (default no-op)."""

    def set_is_evaluating(self) -> None:
        """
        Called by agent during pre_evaluate_policy
        """
        self.is_evaluating = True

    # ------------------------------------------------------------------
    # Hooks for subclasses

    def _get_task_name(self) -> str:
        """Return a task identifier for logging/experiment directory naming."""
        training_task_name = getattr(self.training_config, "task_name", None)
        if isinstance(training_task_name, str) and training_task_name:
            return training_task_name
        return self.__class__.__name__.lower()

    def _get_env_origins(self):
        """Return environment origins used when creating simulator environments."""
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        if terrain_state is None or not hasattr(terrain_state, "env_origins"):
            raise RuntimeError("Terrain manager state 'locomotion_terrain' must provide env_origins.")
        return terrain_state.env_origins

    # ------------------------------------------------------------------
    # Reset hooks

    def set_collection_extras_contract(self, *, dense_episode_stats: bool) -> None:
        """Select dense-every-step or reset-only episode reward statistics."""

        if not isinstance(dense_episode_stats, bool):
            raise TypeError("dense_episode_stats must be a boolean.")
        self._dense_episode_stats_each_step = dense_episode_stats

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        """Reset environment-specific buffers prior to manager resets.

        Default implementation is a no-op. Override in subclasses to zero custom tensors
        or restore from a buffered state.
        """

    def _reset_tasks_callback(self, env_ids):
        """Hook for subclasses to extend reset-time logic."""

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        """Reset simulator DOF/root states for the specified environments.

        Subclasses must implement this to place robots back into their initial configuration.
        """
        raise NotImplementedError("Subclasses must implement `_reset_robot_states_callback` to reset simulator states.")

    def _fill_extras(self, env_ids):
        """Populate per-episode extras after a reset."""
        if self.reward_manager is None:
            return

        include_all = bool(getattr(self, "_dense_episode_stats_each_step", True))
        capability = getattr(
            type(self.reward_manager),
            "supports_include_all_episode_extras",
            None,
        )
        if capability is None:
            cached_capability = getattr(
                self,
                "_reward_reset_include_all_capability_cache",
                None,
            )
            if cached_capability is None or cached_capability[0] is not self.reward_manager:
                try:
                    reset_parameters = inspect.signature(self.reward_manager.reset).parameters.values()
                except (TypeError, ValueError):
                    supports_include_all = False
                else:
                    supports_include_all = any(
                        parameter.name == "include_all"
                        or parameter.kind is inspect.Parameter.VAR_KEYWORD
                        for parameter in reset_parameters
                    )
                self._reward_reset_include_all_capability_cache = (
                    self.reward_manager,
                    supports_include_all,
                )
            else:
                supports_include_all = bool(cached_capability[1])
        else:
            supports_include_all = bool(capability)

        if supports_include_all:
            reward_extras = self.reward_manager.reset(env_ids, include_all=include_all)
        else:
            # Preserve duck-typed fake and third-party managers that implement
            # the historical reset(env_ids) signature.
            reward_extras = self.reward_manager.reset(env_ids)

        # Normalise extras dictionary to contain (possibly empty) sub-sections.
        self.extras["episode"] = reward_extras.get("episode", {})
        self.extras["episode_all"] = reward_extras.get("episode_all", {})
        self.extras["raw_episode"] = reward_extras.get("raw_episode", {})
        self.extras["raw_episode_all"] = reward_extras.get("raw_episode_all", {})
        self.extras["episode_rate"] = reward_extras.get("episode_rate", {})
        self.extras["raw_episode_mean"] = reward_extras.get("raw_episode_mean", {})

        self.extras["time_outs"] = self.time_out_buf

    ###########################################################################
    # Simulation loop helpers

    def step(self, actor_state):
        """Apply actions, advance the simulation, and return rollout buffers."""
        timing = self.step_timing if self.step_timing.enabled else None
        if timing is not None:
            with timing.record("env_step_total"):
                return self._step_impl(actor_state)
        return self._step_impl(actor_state)

    def _step_impl(self, actor_state):
        if hasattr(self, "_viser_live") and getattr(self._viser_live, "enabled", False):
            self._viser_live.apply_pending_controls()
            self._viser_live.wait_if_paused()
        # Per-env-step boundaries are intentionally verbose diagnostics.  The
        # launcher documents HOLOSOMA_DEBUG_HEARTBEAT as iteration-level
        # liveness and HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE as the opt-in for every
        # rollout step.  Using the coarse flag here emitted four log records
        # per step on every rank and forced reset_buf.sum().item() even when
        # verbose heartbeat was explicitly disabled.
        debug_heartbeat = os.environ.get(
            "HOLOSOMA_DEBUG_HEARTBEAT_VERBOSE",
            "",
        ).lower() not in ("", "0", "false", "no")
        actions = actor_state["actions"]
        if debug_heartbeat:
            logger.info(
                "Heartbeat: BaseTask.step begin (num_envs={}, action_shape={})",
                self.num_envs,
                tuple(actions.shape),
            )
        timing = self.step_timing if self.step_timing.enabled else None
        if timing is not None:
            with timing.record("pre_physics"):
                self._pre_physics_step(actions)
        else:
            self._pre_physics_step(actions)
        if debug_heartbeat:
            logger.info("Heartbeat: BaseTask.step after _pre_physics_step")
        if timing is not None:
            with timing.record("physics"):
                self._physics_step()
        else:
            self._physics_step()
        if debug_heartbeat:
            logger.info("Heartbeat: BaseTask.step after _physics_step")
        if timing is not None:
            with timing.record("post_physics"):
                self._post_physics_step()
        else:
            self._post_physics_step()
        if debug_heartbeat:
            reset_count = int(self.reset_buf.sum().item()) if self.reset_buf is not None else 0
            logger.info("Heartbeat: BaseTask.step after _post_physics_step (reset_envs={})", reset_count)
        return self.obs_buf_dict, self.rew_buf, self.reset_buf, self.extras

    def _pre_physics_step(self, actions):
        if self.action_manager is not None:
            self.action_manager.process_actions(actions)

    def _physics_step(self):
        timing = self.step_timing if self.step_timing.enabled else None
        if timing is not None:
            with timing.record("physics/render"):
                self.render()
        else:
            self.render()
        for _ in range(self.simulator.simulator_config.sim.control_decimation):
            if timing is not None:
                with timing.record("physics/apply_force"):
                    self._apply_force_in_physics_step()
                with timing.record("physics/simulate_step"):
                    self.simulator.simulate_at_each_physics_step()
            else:
                self._apply_force_in_physics_step()
                self.simulator.simulate_at_each_physics_step()

    def _apply_force_in_physics_step(self):
        if self.action_manager is not None:
            self.action_manager.apply_actions()

    def _post_physics_step(self):
        # ``extras`` is reused across environment steps.  Episode summaries and
        # final observations are transition-local reset data, so clear them
        # explicitly instead of relying on an empty ``reset_envs_idx`` call to
        # overwrite them.  ``time_outs`` remains a live view of the buffer that
        # ``_check_termination`` refreshes in-place on every step.
        self.extras.pop("final_observations", None)
        for key in (
            "episode",
            "episode_all",
            "raw_episode",
            "raw_episode_all",
            "episode_rate",
            "raw_episode_mean",
        ):
            self.extras[key] = {}
        self.extras["time_outs"] = self.time_out_buf
        timing = self.step_timing if self.step_timing.enabled else None
        dense_episode_stats = bool(
            getattr(self, "_dense_episode_stats_each_step", True)
        )
        if timing is None:
            self._refresh_sim_tensors()
            self.episode_length_buf += 1
            self._update_counters_each_step()

            self._pre_compute_observations_callback()
            self._check_termination()
            self._compute_reward()
            self._update_log_dict()
            if hasattr(self, "_rollout_recorder"):
                self._rollout_recorder.record_step()
            self._draw_scandots_in_viewer()
            if hasattr(self, "_viser_live"):
                self._viser_live.record_step()

            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            # Publish the already-materialized reset selection for rollout
            # consumers.  This is assigned on every transition (including an
            # empty one), so a reused extras dictionary cannot expose stale IDs.
            self.extras["reset_env_ids"] = env_ids
            refresh_was_pending = bool(
                getattr(self, "_reset_refresh_pending", False)
            )
            final_obs_dict = {}
            if env_ids.numel() > 0 and torch.any(self.time_out_buf[env_ids]):
                final_obs_dict = self._compute_final_observations()

            if env_ids.numel() > 0 or dense_episode_stats:
                self.reset_envs_idx(env_ids)

            refresh_required = dense_episode_stats or env_ids.numel() > 0 or bool(
                getattr(self, "_reset_refresh_pending", False)
            )
            if refresh_required:
                refresh_env_ids = self._select_post_reset_refresh_env_ids(
                    env_ids,
                    refresh_was_pending=refresh_was_pending,
                    dense_episode_stats=dense_episode_stats,
                )
                if refresh_env_ids.numel() > 0:
                    self._refresh_envs_after_reset(refresh_env_ids)
                self._reset_refresh_pending = False

            # Advance task-specific state after termination/reset handling so managers
            # see the post-reset timestep on short clips when computing the next obs.
            self._update_tasks_callback()
            self._compute_observations()

            if final_obs_dict:
                # WBT may reject an unsafe timeout preview while computing the
                # final observation.  Store only rows that remain eligible for
                # PPO timeout bootstrapping after that validation.
                timeout_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
                if timeout_env_ids.numel() > 0:
                    timeout_env_ids = self._ensure_long_tensor(timeout_env_ids)
                    self._store_final_observations(timeout_env_ids, final_obs_dict)

            self._post_compute_observations_callback()
            self._clip_observations()

            self.extras["to_log"] = self.log_dict
            if self.viewer:
                self._setup_simulator_control()
                self._setup_simulator_next_task()
            return

        with timing.record("post/refresh"):
            self._refresh_sim_tensors()
        with timing.record("post/counters"):
            self.episode_length_buf += 1
            self._update_counters_each_step()

        with timing.record("post/perception"):
            self._pre_compute_observations_callback()
        with timing.record("post/termination"):
            self._check_termination()
        with timing.record("post/reward"):
            self._compute_reward()
        with timing.record("post/log_update"):
            with timing.record("post/log_update/update_log_dict"):
                self._update_log_dict()
            if hasattr(self, "_rollout_recorder"):
                with timing.record("post/log_update/rollout_recorder"):
                    self._rollout_recorder.record_step()
            with timing.record("post/log_update/draw_scandots"):
                self._draw_scandots_in_viewer()
            if hasattr(self, "_viser_live"):
                with timing.record("post/log_update/viser_live"):
                    self._viser_live.record_step()

        with timing.record("post/reset_select"):
            env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            self.extras["reset_env_ids"] = env_ids
            refresh_was_pending = bool(
                getattr(self, "_reset_refresh_pending", False)
            )
        final_obs_dict = {}
        if env_ids.numel() > 0 and torch.any(self.time_out_buf[env_ids]):
            with timing.record("post/final_observations"):
                final_obs_dict = self._compute_final_observations()

        with timing.record("post/reset_envs"):
            if env_ids.numel() > 0 or dense_episode_stats:
                self.reset_envs_idx(env_ids)

        with timing.record("post/reset_refresh"):
            refresh_required = dense_episode_stats or env_ids.numel() > 0 or bool(
                getattr(self, "_reset_refresh_pending", False)
            )
            if refresh_required:
                refresh_env_ids = self._select_post_reset_refresh_env_ids(
                    env_ids,
                    refresh_was_pending=refresh_was_pending,
                    dense_episode_stats=dense_episode_stats,
                )
                if refresh_env_ids.numel() > 0:
                    self._refresh_envs_after_reset(refresh_env_ids)
                self._reset_refresh_pending = False

        # Advance task-specific state after termination/reset handling so managers
        # see the post-reset timestep on short clips when computing the next obs.
        with timing.record("post/tasks"):
            self._update_tasks_callback()
        with timing.record("post/observations"):
            self._compute_observations()

        if final_obs_dict:
            # WBT may reject an unsafe timeout preview while computing the
            # final observation.  Store only rows that remain eligible for
            # PPO timeout bootstrapping after that validation.
            timeout_env_ids = self.time_out_buf.nonzero(as_tuple=False).flatten()
            if timeout_env_ids.numel() > 0:
                with timing.record("post/store_final_observations"):
                    timeout_env_ids = self._ensure_long_tensor(timeout_env_ids)
                    self._store_final_observations(timeout_env_ids, final_obs_dict)

        with timing.record("post/post_observations"):
            self._post_compute_observations_callback()
            self._clip_observations()

        with timing.record("post/extras_viewer"):
            self.extras["to_log"] = self.log_dict
            if self.viewer:
                self._setup_simulator_control()
                self._setup_simulator_next_task()

    def _draw_scandots_in_viewer(self) -> None:
        if not getattr(self.training_config, "isaac_show_scandots", True):
            self._isaac_scandots_payload = None
            return
        if self.headless:
            self._isaac_scandots_payload = None
            return
        if self.simulator.get_simulator_type() != SimulatorType.ISAACSIM:
            self._isaac_scandots_payload = None
            return
        if self.perception_manager is None:
            self._isaac_scandots_payload = None
            return
        if not getattr(self.simulator, "draw", None):
            self._isaac_scandots_payload = None
            return

        update_hz = float(getattr(self.training_config, "viser_update_hz", 30.0))
        if update_hz > 0.0:
            period = 1.0 / update_hz
            now = time.perf_counter()
            if now - self._isaac_scandots_last_update < period:
                return
            self._isaac_scandots_last_update = now

        env_id = int(getattr(self.training_config, "viser_env_id", 0))
        if env_id < 0 or env_id >= self.num_envs:
            env_id = 0
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)

        output_mode = getattr(getattr(self.perception_manager, "cfg", None), "output_mode", None)
        use_heightmap = output_mode == "heightmap"
        use_camera_depth = output_mode == "camera_depth"
        camera_source = getattr(getattr(self.perception_manager, "cfg", None), "camera_source", None)
        if not use_heightmap and not use_camera_depth:
            self._isaac_scandots_payload = None
            return
        if use_camera_depth and camera_source not in ("mesh_raycast_scandots", "mesh_raycast"):
            self._isaac_scandots_payload = None
            return
        include_misses_env = os.environ.get("ISAAC_SCANDOTS_INCLUDE_MISSES")
        if include_misses_env is None:
            include_misses = False
        else:
            include_misses = include_misses_env.lower() not in (
                "0",
                "false",
                "no",
            )
        use_depth_mask_env = os.environ.get("ISAAC_SCANDOTS_USE_DEPTH_MASK")
        if use_depth_mask_env is None:
            use_depth_mask = False
        else:
            use_depth_mask = use_depth_mask_env.lower() not in (
                "0",
                "false",
                "no",
            )
        try:
            with torch.no_grad():
                if use_heightmap and hasattr(self.perception_manager, "get_heightmap_points"):
                    result = self.perception_manager.get_heightmap_points(
                        env_ids,
                        include_misses=include_misses,
                        return_rays=True,
                    )
                elif use_camera_depth and camera_source == "mesh_raycast" and hasattr(
                    self.perception_manager, "get_camera_depth_ray_samples"
                ):
                    result = self.perception_manager.get_camera_depth_ray_samples(
                        env_ids,
                        include_misses=include_misses,
                        return_rays=True,
                    )
                elif hasattr(self.perception_manager, "get_camera_scandots_points"):
                    result = self.perception_manager.get_camera_scandots_points(
                        env_ids,
                        include_misses=include_misses,
                        return_rays=True,
                    )
                else:
                    result = None
        except Exception as exc:
            if not self._isaac_scandots_warned:
                self._isaac_scandots_warned = True
                logger.warning("IsaacSim scandots draw disabled: {}", exc)
            self._isaac_scandots_payload = None
            return

        if result is None:
            if use_heightmap and not self._isaac_scandots_warned:
                self._isaac_scandots_warned = True
                logger.warning("IsaacSim scandots draw disabled: heightmap points are unavailable.")
            self._isaac_scandots_payload = None
            return

        if not isinstance(result, tuple) or len(result) < 2:
            self._isaac_scandots_payload = None
            return

        points = result[0]
        mask = result[1]
        ray_starts = result[2] if len(result) > 2 else None
        ray_hits_world = result[4] if len(result) > 4 else None

        points_env = points[0]
        mask_env = mask[0] if mask is not None else None
        mask_env_bool = None
        if mask_env is not None and mask_env.numel() > 0:
            mask_env_bool = mask_env.to(torch.bool)

        draw_hit_mask = None
        if ray_hits_world is not None and ray_hits_world.numel() > 0:
            hits_env = ray_hits_world[0]
            if hits_env.shape == points_env.shape:
                hit_mask = torch.isfinite(hits_env).all(dim=-1)
                if use_depth_mask and mask_env_bool is not None and mask_env_bool.shape == hit_mask.shape:
                    hit_mask = hit_mask & mask_env_bool
                points_env = hits_env[hit_mask]
                draw_hit_mask = hit_mask
        elif mask_env_bool is not None and use_depth_mask:
            points_env = points_env[mask_env_bool]
        line_starts_np = np.zeros((0, 3), dtype=np.float32)
        line_ends_np = np.zeros((0, 3), dtype=np.float32)
        if (
            draw_hit_mask is not None
            and ray_starts is not None
            and ray_starts.numel() > 0
            and ray_hits_world is not None
            and ray_hits_world.numel() > 0
        ):
            starts_env = ray_starts[0]
            hits_env = ray_hits_world[0]
            if starts_env.shape == hits_env.shape and draw_hit_mask.shape[0] == starts_env.shape[0]:
                starts_draw = starts_env[draw_hit_mask]
                ends_draw = hits_env[draw_hit_mask]
                line_starts_np = starts_draw.detach().cpu().numpy().astype(np.float32, copy=False)
                line_ends_np = ends_draw.detach().cpu().numpy().astype(np.float32, copy=False)
        if points_env.numel() == 0:
            self.simulator.draw.clear_points()
            self._isaac_scandots_payload = {
                "env_id": int(env_id),
                "points": np.zeros((0, 3), dtype=np.float32),
                "line_starts": line_starts_np,
                "line_ends": line_ends_np,
            }
            return

        pts = points_env.detach().cpu().numpy()
        point_color = [1.0, 0.0, 0.0]
        point_size = float(getattr(self.training_config, "isaac_scandots_point_size", 2.0))

        colors = [point_color for _ in range(pts.shape[0])]
        sizes = [point_size for _ in range(pts.shape[0])]

        self.simulator.draw.clear_points()
        from holosoma.utils import draw as draw_utils

        draw_utils.draw_points(self.simulator, pts, colors, sizes, env_id)
        self._isaac_scandots_payload = {
            "env_id": int(env_id),
            "points": pts.astype(np.float32, copy=False),
            "line_starts": line_starts_np,
            "line_ends": line_ends_np,
        }

    def _ensure_long_tensor(self, tensor_like):
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(tensor_like, device=self.device, dtype=torch.long)

    def _get_envs_to_refresh(self):
        return torch.empty(0, device=self.device, dtype=torch.long)

    def _select_post_reset_refresh_env_ids(
        self,
        reset_env_ids: torch.Tensor,
        *,
        refresh_was_pending: bool,
        dense_episode_stats: bool,
    ) -> torch.Tensor:
        """Reuse the transition's reset IDs when they are the exact dirty set.

        ``reset_buf.nonzero()`` already materializes the reset rows.  WBT and
        locomotion reset hooks mark exactly those same rows dirty, so scanning
        their device mask with a second dynamic-size ``nonzero()`` forces an
        unnecessary CUDA-to-host synchronization.  Explicit/out-of-band
        resets leave ``_reset_refresh_pending`` armed before this transition;
        dense/legacy tasks may also have broader refresh semantics.  Both use
        the mask-based fallback.

        Every concrete subclass must opt in itself, and only when an in-loop
        ``reset_envs_idx(ids)`` cannot dirty rows outside ``ids``.  Looking in
        the concrete class dictionary deliberately prevents a third-party
        subclass from inheriting this capability after overriding reset
        semantics.
        """

        direct_ids_capability = bool(
            type(self).__dict__.get("supports_direct_post_reset_refresh_ids", False)
        )
        if (
            direct_ids_capability
            and not refresh_was_pending
            and not dense_episode_stats
            and reset_env_ids.numel() > 0
        ):
            return self._ensure_long_tensor(reset_env_ids)
        return self._ensure_long_tensor(self._get_envs_to_refresh())

    def _refresh_envs_after_reset(self, env_ids):
        """Hook for subclasses to synchronise simulator state after resets."""
        return

    def _store_final_observations(self, env_ids, final_obs_dict):
        if not final_obs_dict:
            return
        final_store = self.extras.setdefault("final_observations", {})
        for obs_key, values in final_obs_dict.items():
            if obs_key not in final_store:
                final_store[obs_key] = torch.zeros_like(self.obs_buf_dict[obs_key])
            final_store[obs_key][env_ids] = values[env_ids]

    def _clip_observations(self):
        clip_limit = self.observation_manager.cfg.clip_observations
        for obs_key, obs_val in self.obs_buf_dict.items():
            self.obs_buf_dict[obs_key] = torch.clip(obs_val, -clip_limit, clip_limit)

    def _compute_reward(self):
        self.rew_buf[:] = self.reward_manager.compute(self.dt)
        self.episode_sums = getattr(self.reward_manager, "episode_sums", {})
        self.episode_sums_raw = getattr(self.reward_manager, "episode_sums_raw", {})

    def _compute_observations(self):
        self.obs_buf_dict = self.observation_manager.compute()

    def _compute_final_observations(self):
        final_observations = self.observation_manager.compute(modify_history=False)
        # The regular next observation is clipped later in `_clip_observations`.
        # Timeout values must be evaluated on the same critic input domain,
        # rather than an unclipped terminal-only observation.
        clip_limit = self.observation_manager.cfg.clip_observations
        return {
            obs_key: torch.clip(obs_value, -clip_limit, clip_limit)
            for obs_key, obs_value in final_observations.items()
        }

    def _update_tasks_callback(self):
        timing = self.step_timing if self.step_timing.enabled else None
        if timing is None:
            self.command_manager.step()
            self.curriculum_manager.step()
            self.randomization_manager.step()
            return

        with timing.record("post/tasks/command_manager"):
            self.command_manager.step()
        with timing.record("post/tasks/curriculum_manager"):
            self.curriculum_manager.step()
        with timing.record("post/tasks/randomization_manager"):
            self.randomization_manager.step()

    def _init_counters(self):
        return

    def _update_counters_each_step(self):
        return

    def _check_termination(self):
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0
        if os.environ.get("HOLOSOMA_DISABLE_AUTO_RESET", "0").lower() in ("1", "true", "yes", "on"):
            return
        if self.termination_manager is None:
            return

        reset_flags, timeout_flags = self.termination_manager.check()
        self.reset_buf |= reset_flags.to(dtype=self.reset_buf.dtype)
        # A transition can hit the time limit on the same control step as a
        # genuine terminal condition (fall, bad tracking, motion end, ...).
        # Such a transition is terminal and must not receive a value bootstrap;
        # ``time_outs`` denotes truncation-only transitions for PPO.
        self.time_out_buf |= timeout_flags & ~reset_flags
        self.reset_buf |= self.time_out_buf
        self._log_termination_masks(reset_flags, timeout_flags)

    def _log_termination_masks(self, reset_flags, timeout_flags) -> None:
        if self.termination_manager is None:
            return
        timeout_only_flags = timeout_flags & ~reset_flags
        done_flags = reset_flags | timeout_flags
        # Keep these scalar reductions on their producing device.  The
        # logging helper batches every environment metric into one device-to-
        # host transfer at the iteration boundary; copying each term here
        # serialized the CUDA stream once per termination statistic on every
        # rollout step.
        self.log_dict["termination/reset_frac"] = reset_flags.float().mean().detach()
        # Keep the raw timeout term for diagnostics, and expose its mutually
        # exclusive rollout meaning separately.  The latter is exactly the
        # mask eligible for PPO value bootstrapping.
        self.log_dict["termination/timeout_frac"] = timeout_flags.float().mean().detach()
        self.log_dict["termination/timeout_only_frac"] = timeout_only_flags.float().mean().detach()
        self.log_dict["termination/done_frac"] = done_flags.float().mean().detach()
        for term_name in getattr(self.termination_manager, "_term_names", []):
            term_result = self.termination_manager.get_last_term_result(term_name)
            if term_result is None:
                continue
            self.log_dict[f"termination/{term_name}_frac"] = term_result.float().mean().detach()
            component_getter = getattr(
                self.termination_manager,
                "get_last_term_components",
                None,
            )
            if not callable(component_getter):
                continue
            components = component_getter(term_name)
            if not components:
                continue
            component_names = tuple(components)
            component_masks = []
            for component_name in component_names:
                component = components[component_name]
                if component.dtype != torch.bool or component.shape != term_result.shape:
                    raise TypeError(
                        f"Termination component '{term_name}/{component_name}' must be a bool tensor "
                        f"with shape {tuple(term_result.shape)}, got {component.dtype} {tuple(component.shape)}."
                    )
                component_masks.append(component)
            component_fractions = torch.stack(component_masks, dim=0).float().mean(dim=1).detach()
            for component_name, component_fraction in zip(
                component_names,
                component_fractions,
                strict=True,
            ):
                self.log_dict[
                    f"termination/{term_name}/condition_{component_name}_frac"
                ] = component_fraction

    def _pre_compute_observations_callback(
        self,
        env_ids: torch.Tensor | None = None,
    ):
        """Refresh unique perception streams before computing observations.

        ``env_ids=None`` denotes the one normal all-environment refresh for a
        control step.  Reset synchronization passes only the environments whose
        simulator state changed, so unrelated latency/history streams are not
        advanced a second time merely because another environment terminated.
        """

        updated_perception_manager_ids: set[int] = set()
        for perception_manager in (
            self.perception_manager,
            self.teacher_perception_manager,
            self.critic_perception_manager,
        ):
            if perception_manager is None or id(perception_manager) in updated_perception_manager_ids:
                continue
            updated_perception_manager_ids.add(id(perception_manager))
            effective_env_ids = env_ids
            if env_ids is not None:
                uses_legacy_full_refresh = getattr(
                    perception_manager,
                    "uses_legacy_full_reset_refresh",
                    None,
                )
                if callable(uses_legacy_full_refresh) and uses_legacy_full_refresh():
                    effective_env_ids = None
            perception_manager.update(effective_env_ids)

    def _post_compute_observations_callback(self):
        """Hook invoked after observation buffers are produced (no-op by default)."""
        self._capture_depth_frame()

    def _setup_simulator_control(self):
        """Hook for pushing controller state back to the simulator/viewer (no-op by default)."""
        return

    def _setup_simulator_next_task(self):
        """Hook for interactive viewer task selection (no-op by default)."""
        return

    def _init_depth_logging_state(self) -> None:
        self._depth_log_is_main_process = int(os.environ.get("RANK", "0")) == 0
        self._depth_log_active = False
        self._depth_log_pending_frame0 = False
        self._depth_log_frames: list[np.ndarray] = []
        self._depth_log_episode_id: int | None = None
        self._depth_log_record_env_id = int(getattr(self.simulator.video_config, "record_env_id", 0))
        self._depth_log_obs_group: str | None = None
        self._depth_log_obs_term_name: str | None = None
        self._depth_log_obs_slice: slice | None = None
        self._depth_log_obs_history = 1
        self._depth_log_obs_frame_dim: int | None = None
        self._depth_log_obs_scale: float | tuple | None = None
        self._depth_log_obs_unavailable = False
        self._depth_log_group_concatenate = True
        self._depth_log_startup_done = False
        self._depth_log_startup_video_active = False
        self._depth_log_startup_video_done = False
        self._depth_log_startup_video_frames: list[np.ndarray] = []
        self._depth_log_startup_video_episode_id: int | None = None

    def _depth_logging_enabled(self) -> bool:
        if not self._depth_log_is_main_process:
            return False
        if self.perception_manager is None or not self.perception_manager.enabled:
            return False
        if self.perception_manager.cfg.output_mode != "camera_depth":
            return False
        if self.simulator.video_recorder is None or not self.simulator.video_config.enabled:
            return False
        if getattr(self.simulator.logger_cfg, "type", "disabled") != "wandb":
            return False
        if not self.simulator.video_config.upload_to_wandb:
            return False
        self._resolve_depth_obs_source()
        if self._depth_log_obs_group is None:
            return False
        return True

    def _start_depth_logging_if_needed(self) -> None:
        self._start_startup_depth_video_if_needed()
        if not self._depth_logging_enabled():
            return
        if self.simulator.video_recorder is None or not self.simulator.video_recorder.is_recording:
            return
        if self._depth_log_active:
            return
        self._depth_log_active = True
        self._depth_log_pending_frame0 = True
        self._depth_log_frames = []
        self._depth_log_episode_id = getattr(self.simulator.video_recorder, "current_episode", None)

    def _finalize_depth_logging_if_needed(self) -> None:
        if not self._depth_log_active:
            return
        if self.simulator.video_recorder is None or self.simulator.video_recorder.is_recording:
            return
        self._log_depth_video()
        self._depth_log_active = False
        self._depth_log_pending_frame0 = False
        self._depth_log_frames = []
        self._depth_log_episode_id = None

    def _startup_depth_video_enabled(self) -> bool:
        if not self._depth_log_is_main_process:
            return False
        if self.perception_manager is None or not self.perception_manager.enabled:
            return False
        if self.perception_manager.cfg.output_mode != "camera_depth":
            return False
        if getattr(self.simulator.logger_cfg, "type", "disabled") != "wandb":
            return False
        if not self.simulator.video_config.upload_to_wandb:
            return False
        self._resolve_depth_obs_source()
        if self._depth_log_obs_group is None:
            return False
        return True

    def _start_startup_depth_video_if_needed(self) -> None:
        if self._depth_log_startup_video_done or self._depth_log_startup_video_active:
            return
        if not self._startup_depth_video_enabled():
            return
        self._depth_log_startup_video_active = True
        self._depth_log_startup_video_frames = []
        self._depth_log_startup_video_episode_id = getattr(self.simulator.video_recorder, "current_episode", None)

    def _finalize_startup_depth_video_if_needed(self, env_ids) -> None:
        if not self._depth_log_startup_video_active:
            return
        if not self._startup_depth_video_enabled():
            return
        record_env = self._depth_log_record_env_id
        has_record_env = False
        if isinstance(env_ids, torch.Tensor):
            if env_ids.numel() > 0:
                has_record_env = bool((env_ids == record_env).any().item())
        else:
            try:
                has_record_env = record_env in env_ids
            except TypeError:
                has_record_env = False
        if not has_record_env:
            return
        self._log_startup_depth_video()
        self._depth_log_startup_video_active = False
        self._depth_log_startup_video_done = True
        self._depth_log_startup_video_frames = []
        self._depth_log_startup_video_episode_id = None

    def _capture_depth_frame(self) -> None:
        capture_rollout = self._depth_log_active
        capture_startup = self._depth_log_startup_video_active
        if not (capture_rollout or capture_startup):
            return
        if capture_rollout and not self._depth_logging_enabled():
            capture_rollout = False
        if capture_rollout and (
            self.simulator.video_recorder is None or not self.simulator.video_recorder.is_recording
        ):
            capture_rollout = False
        if capture_startup and not self._startup_depth_video_enabled():
            capture_startup = False
        if not (capture_rollout or capture_startup):
            return
        depth_map = self._extract_policy_depth_frame()
        if depth_map is None:
            return
        depth_frame = self._depth_to_rgb(depth_map)
        if capture_rollout:
            if self._depth_log_pending_frame0:
                self._log_depth_frame0(depth_frame)
                self._depth_log_pending_frame0 = False
            self._depth_log_frames.append(depth_frame)
        if capture_startup:
            self._depth_log_startup_video_frames.append(depth_frame)

    def _depth_to_rgb(self, depth_map: np.ndarray) -> np.ndarray:
        max_distance = float(self.perception_manager.cfg.max_distance)
        scale = self._depth_log_obs_scale
        if isinstance(scale, (int, float)):
            max_distance *= float(scale)
        elif isinstance(scale, tuple) and len(scale) > 0:
            max_distance *= float(max(abs(val) for val in scale))
        if max_distance <= 1.0e-6:
            max_distance = 1.0
        depth = np.nan_to_num(depth_map, nan=max_distance, posinf=max_distance, neginf=0.0)
        depth = np.clip(depth, 0.0, max_distance)
        normalized = depth / max_distance
        # Brighter = closer for quick visualization.
        gray = (255.0 * (1.0 - normalized)).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=-1)

    def _log_depth_frame0(self, frame: np.ndarray) -> None:
        try:
            import wandb  # noqa: PLC0415
        except Exception:
            return
        if wandb.run is None:
            return
        caption = f"episode {self._depth_log_episode_id}" if self._depth_log_episode_id is not None else None
        # Media captured during rollout belongs to the next PPO metrics row.
        # Buffer it without advancing W&B's implicit history cursor; the
        # iteration-indexed scalar log commits the complete row.
        wandb.log(
            {"Depth/frame0": wandb.Image(frame, caption=caption)},
            commit=False,
        )

    def _log_startup_depth_if_needed(self) -> None:
        if self._depth_log_startup_done:
            return
        if not self._depth_log_is_main_process:
            return
        if self.perception_manager is None or not self.perception_manager.enabled:
            return
        if self.perception_manager.cfg.output_mode != "camera_depth":
            return
        if getattr(self.simulator.logger_cfg, "type", "disabled") != "wandb":
            return
        try:
            import wandb  # noqa: PLC0415
        except Exception:
            return
        if wandb.run is None:
            return

        self._resolve_depth_obs_source()
        depth_map = self._extract_policy_depth_frame()
        if depth_map is None:
            depth_map = (
                self.perception_manager.get_camera_depth_map()[self._depth_log_record_env_id]
                .detach()
                .cpu()
                .numpy()
            )
        depth_frame = self._depth_to_rgb(depth_map)
        # The first PPO update is iteration 0.  Committing this startup image
        # would move W&B's implicit cursor to 1 and make those scientific
        # metrics stale, so leave it buffered for the iteration-0 scalar row.
        wandb.log(
            {"Depth/startup": wandb.Image(depth_frame, caption="startup")},
            commit=False,
        )
        self._depth_log_startup_done = True

    def _log_startup_depth_video(self) -> None:
        if not self._depth_log_startup_video_frames:
            return
        try:
            from holosoma.utils.video_utils import create_video  # noqa: PLC0415
        except Exception:
            return
        sim_config = self.simulator.simulator_config.sim
        control_frequency = sim_config.fps / sim_config.control_decimation
        display_fps = control_frequency * self.simulator.video_config.playback_rate
        save_dir = (
            Path(self.simulator.video_config.save_dir)
            if self.simulator.video_config.save_dir is not None
            else Path("/data/logs_new/videos")
        )
        video_frames = np.stack(self._depth_log_startup_video_frames, axis=0).astype(np.uint8)
        create_video(
            video_frames=video_frames,
            fps=display_fps,
            save_dir=save_dir,
            output_format=self.simulator.video_config.output_format,
            wandb_logging=True,
            wandb_commit=False,
            episode_id=self._depth_log_startup_video_episode_id,
            wandb_key="Depth rollout (startup)",
        )

    def _log_depth_video(self) -> None:
        if not self._depth_log_frames:
            return
        try:
            from holosoma.utils.video_utils import create_video  # noqa: PLC0415
        except Exception:
            return
        sim_config = self.simulator.simulator_config.sim
        control_frequency = sim_config.fps / sim_config.control_decimation
        display_fps = control_frequency * self.simulator.video_config.playback_rate
        save_dir = (
            Path(self.simulator.video_config.save_dir)
            if self.simulator.video_config.save_dir is not None
            else Path("/data/logs_new/videos")
        )
        video_frames = np.stack(self._depth_log_frames, axis=0).astype(np.uint8)
        create_video(
            video_frames=video_frames,
            fps=display_fps,
            save_dir=save_dir,
            output_format=self.simulator.video_config.output_format,
            wandb_logging=True,
            wandb_commit=False,
            episode_id=self._depth_log_episode_id,
            wandb_key="Depth rollout",
        )

    def _resolve_depth_obs_source(self) -> None:
        if self._depth_log_obs_group is not None or self._depth_log_obs_unavailable:
            return
        if self.observation_manager is None:
            self._depth_log_obs_unavailable = True
            return

        groups = self.observation_manager.cfg.groups
        group_name = None
        term_name = None
        if "actor_obs" in groups:
            if "perception_obs" in groups["actor_obs"].terms:
                group_name = "actor_obs"
                term_name = "perception_obs"
            elif "perception" in groups["actor_obs"].terms:
                group_name = "actor_obs"
                term_name = "perception"
        else:
            for name in sorted(groups.keys()):
                if "perception_obs" in groups[name].terms:
                    group_name = name
                    term_name = "perception_obs"
                    break
                if "perception" in groups[name].terms:
                    group_name = name
                    term_name = "perception"
                    break

        if group_name is None:
            self._depth_log_obs_unavailable = True
            return
        if term_name is None:
            self._depth_log_obs_unavailable = True
            return

        group_cfg = groups[group_name]
        term_cfg = group_cfg.terms[term_name]
        self._depth_log_obs_group = group_name
        self._depth_log_obs_term_name = term_name
        self._depth_log_obs_scale = term_cfg.scale
        self._depth_log_obs_history = group_cfg.history_length
        self._depth_log_group_concatenate = group_cfg.concatenate

        if group_cfg.concatenate:
            term_slices = self.observation_manager.get_term_slices(group_name)
            self._depth_log_obs_slice = term_slices.get(term_name)
            if self._depth_log_obs_slice is not None:
                slice_len = self._depth_log_obs_slice.stop - self._depth_log_obs_slice.start
                if self._depth_log_obs_history > 1:
                    self._depth_log_obs_frame_dim = slice_len // self._depth_log_obs_history
                else:
                    self._depth_log_obs_frame_dim = slice_len

    def _extract_policy_depth_frame(self) -> np.ndarray | None:
        if self._depth_log_obs_group is None:
            return None
        group_obs = self.obs_buf_dict.get(self._depth_log_obs_group)
        if group_obs is None:
            return None
        if self._depth_log_obs_term_name is None:
            return None

        if isinstance(group_obs, dict):
            term_obs = group_obs.get(self._depth_log_obs_term_name)
        else:
            if self._depth_log_obs_slice is None:
                return None
            term_obs = group_obs[:, self._depth_log_obs_slice]

        if term_obs is None:
            return None

        frame_dim = self._depth_log_obs_frame_dim or term_obs.shape[1]
        if self._depth_log_obs_history > 1:
            if term_obs.shape[1] % self._depth_log_obs_history != 0:
                return None
            frame_dim = term_obs.shape[1] // self._depth_log_obs_history
            term_obs = term_obs[:, -frame_dim:]

        depth_vec = term_obs[self._depth_log_record_env_id].detach().cpu().numpy()
        height, width = self.perception_manager.get_camera_depth_map().shape[-2:]
        if depth_vec.size != height * width:
            return None
        return depth_vec.reshape(height, width)

    def _update_log_dict(self):
        """Hook for appending task-specific metrics to `self.log_dict` (no-op by default)."""
        return
