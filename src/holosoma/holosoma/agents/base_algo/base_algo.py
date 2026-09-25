from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Callable

from holosoma.config_types.algo import AlgoInitConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.rng_checkpoint import capture_rng_checkpoint_state
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.training_provenance import training_provenance_from_env

if TYPE_CHECKING:
    from holosoma.config_types.experiment import ExperimentConfig


class BaseAlgo:
    def __init__(self, env: BaseTask, config: AlgoInitConfig, device, multi_gpu_cfg=None):
        self.env = env
        self.config = config
        self.device = device

        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_local_rank = multi_gpu_cfg["local_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_local_rank = 0
            self.gpu_world_size = 1
        self.is_main_process = self.gpu_global_rank == 0
        self._experiment_config: ExperimentConfig | None = None
        self._source_experiment_config_dict: dict[str, Any] | None = None
        self._policy_load_runtime_config: ExperimentConfig | None = None
        self._wandb_run_path: str | None = None
        self._training_provenance: dict[str, Any] | None = None
        self._source_checkpoint_sha256: str | None = None
        self._actor_perception_training_geometry_support: dict[str, Any] | None = None
        self._motion_transition_contract: dict[str, Any] | None = None
        self._motion_transition_contract_sha256: str | None = None
        self._evaluation_completed_iteration: int | None = None
        self._evaluation_rng_boundary_state: dict[str, Any] | None = None
        self._evaluation_only = False
        self._evaluation_allow_ood_object_geometry = False
        self._evaluation_ood_object_geometry_audit: dict[str, Any] | None = None

    def setup(self):
        return NotImplementedError

    def learn(self):
        return NotImplementedError

    def load(self, path):
        return NotImplementedError

    def load_policy_init(self, path):
        return NotImplementedError

    def load_stage4_init(self, path):
        """Warm-start actor and critic while leaving continuation state fresh."""

        return NotImplementedError

    def load_evaluation(self, path):
        """Load the state needed for policy evaluation.

        Algorithms should override this when evaluation does not require a
        full optimizer/environment continuation.  The fallback preserves
        compatibility for algorithms that have not yet separated the paths.
        """

        return self.load(path)

    @property
    def inference_model(self):
        return NotImplementedError

    @property
    def actor_onnx_wrapper(self):
        return NotImplementedError

    def env_step(self, actions, extra_info=None):
        obs_dict, rewards, dones, extras = self.env.step(actions, extra_info)
        return obs_dict, rewards, dones, extras

    def attach_checkpoint_metadata(
        self,
        experiment_config: ExperimentConfig,
        wandb_run_path: str | None = None,
    ) -> None:
        """Attach metadata that should be saved with checkpoints."""

        self._experiment_config = experiment_config
        self._source_experiment_config_dict = None
        self._policy_load_runtime_config = experiment_config
        self._wandb_run_path = wandb_run_path
        self._training_provenance = training_provenance_from_env()
        self._source_checkpoint_sha256 = None
        self._actor_perception_training_geometry_support = None
        self._motion_transition_contract = None
        self._motion_transition_contract_sha256 = None
        self._evaluation_completed_iteration = None
        self._evaluation_rng_boundary_state = None
        self._evaluation_only = False
        self._evaluation_allow_ood_object_geometry = False
        self._evaluation_ood_object_geometry_audit = None

    def attach_evaluation_metadata(
        self,
        source_experiment_config: ExperimentConfig,
        runtime_experiment_config: ExperimentConfig,
        wandb_run_path: str | None = None,
    ) -> None:
        """Attach distinct checkpoint-source and live evaluation contracts.

        The source config is retained for exported policy metadata, while the
        runtime config is used to prove that equal-shaped actor tensors still
        have the same observation-to-action semantics.  Source provenance and
        checkpoint SHA are populated only after ``load_evaluation`` validates
        the checkpoint payload.
        """

        self._experiment_config = source_experiment_config
        self._source_experiment_config_dict = None
        self._policy_load_runtime_config = runtime_experiment_config
        self._wandb_run_path = wandb_run_path
        self._training_provenance = None
        self._source_checkpoint_sha256 = None
        self._actor_perception_training_geometry_support = None
        self._motion_transition_contract = None
        self._motion_transition_contract_sha256 = None
        self._evaluation_completed_iteration = None
        # Policy placeholder construction during setup must not shift the
        # environment/evaluation stochastic stream. load_evaluation restores
        # this boundary after the validated actor payload is committed.
        self._evaluation_rng_boundary_state = capture_rng_checkpoint_state()
        self._evaluation_only = True
        self._evaluation_allow_ood_object_geometry = False
        self._evaluation_ood_object_geometry_audit = None

    def enable_evaluation_only_ood_object_geometry(self) -> None:
        """Permit an explicitly labelled OOD object only for policy evaluation.

        This does not relax camera-source, robot-geometry, observation, actor,
        normalizer, checkpoint, or provenance validation.  It only allows the
        selected live object mesh to fall outside the authenticated training
        support, and it is unavailable to training, resume, or policy-init.
        """

        if not self._evaluation_only:
            raise RuntimeError(
                "OOD object geometry can only be enabled after attaching "
                "evaluation metadata."
            )
        self._evaluation_allow_ood_object_geometry = True
        self._evaluation_ood_object_geometry_audit = None

    def evaluation_ood_object_geometry_audit(self) -> dict[str, Any] | None:
        """Return a detached copy of the explicit OOD evaluation evidence."""

        audit = self._evaluation_ood_object_geometry_audit
        return copy.deepcopy(audit) if audit is not None else None

    def _checkpoint_metadata(self, iteration: int | None = None) -> dict[str, Any]:
        if self._experiment_config is None:
            raise RuntimeError(
                "Experiment config metadata missing. Call attach_checkpoint_metadata() or "
                "attach_evaluation_metadata() before publication."
            )

        serialized_config = (
            copy.deepcopy(self._source_experiment_config_dict)
            if self._source_experiment_config_dict is not None
            else self._experiment_config.to_serializable_dict()
        )
        metadata: dict[str, Any] = {"experiment_config": serialized_config}
        if self._wandb_run_path:
            metadata["wandb_run_path"] = self._wandb_run_path
        if self._training_provenance is not None:
            metadata["training_provenance"] = dict(self._training_provenance)
        if self._source_checkpoint_sha256 is not None:
            metadata["source_checkpoint_sha256"] = self._source_checkpoint_sha256
        transition_contract = getattr(self, "_motion_transition_contract", None)
        transition_digest = getattr(self, "_motion_transition_contract_sha256", None)
        if (transition_contract is None) != (transition_digest is None):
            raise RuntimeError(
                "Motion-transition metadata must provide both motion_transition_contract and "
                "motion_transition_contract_sha256, or neither."
            )
        if transition_contract is not None:
            metadata["motion_transition_contract"] = copy.deepcopy(transition_contract)
            metadata["motion_transition_contract_sha256"] = transition_digest
        if iteration is not None:
            metadata["iteration"] = int(iteration)
        return metadata

    def has_curricula_enabled(self) -> bool:
        """Check if any curricula are enabled in the environment.

        This helper method checks for the presence of various curriculum flags
        to determine if any curriculum learning is active. This is commonly used
        for multi-GPU synchronization and logging purposes.

        Returns
        -------
        bool
            True if any curriculum is enabled, False otherwise.
        """
        env = self._unwrap_env()
        return (
            getattr(env, "use_reward_penalty_curriculum", False)
            or getattr(env, "use_domain_rand_scale_curriculum", False)
            or getattr(env, "curriculum_state_sync_enabled", False)
        )

    def _validate_distributed_penalty_curriculum_contract(self) -> None:
        """Reject rank-local adaptive reward objectives in distributed runs.

        Locomotion penalty scaling is currently updated from rank-local episode
        resets.  Synchronizing only at rollout boundaries cannot prevent ranks
        from using different reward weights inside the rollout, so averaging
        those gradients would not optimize one well-defined objective.
        """

        if int(getattr(self, "gpu_world_size", 1) or 1) <= 1:
            return
        env = self._unwrap_env()
        if not bool(getattr(env, "use_reward_penalty_curriculum", False)):
            return

        raw_degree: Any = None
        curriculum_manager = getattr(env, "curriculum_manager", None)
        if curriculum_manager is not None and hasattr(curriculum_manager, "get_term"):
            penalty_term = curriculum_manager.get_term("penalty_curriculum")
            if penalty_term is not None:
                if not bool(getattr(penalty_term, "enabled", True)):
                    return
                raw_degree = getattr(penalty_term, "degree", None)
        if raw_degree is None:
            legacy_cfg = getattr(env, "_curriculum_penalty_cfg", None)
            if isinstance(legacy_cfg, dict):
                raw_degree = legacy_cfg.get("degree")
        if isinstance(raw_degree, bool):
            raise ValueError("Penalty curriculum degree must be a finite real number, not boolean.")
        try:
            degree = float(raw_degree)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Enabled penalty curriculum does not expose a valid finite degree."
            ) from exc
        if not math.isfinite(degree):
            raise ValueError("Penalty curriculum degree must be finite.")
        if degree == 0.0:
            return
        raise RuntimeError(
            "Distributed training with adaptive reward penalty curriculum is scientifically invalid: "
            "rank-local episode resets can change reward weights inside a rollout, so ranks optimize "
            "different objectives. Use one training rank or disable the adaptive penalty curriculum "
            "until a global sufficient-statistics update is implemented."
        )

    def _synchronize_curriculum_metrics(self):
        """Synchronize curriculum-related metrics across all GPUs."""
        # Check if any curricula are enabled before synchronizing
        if not self.has_curricula_enabled():
            return

        env = self._unwrap_env()
        env.synchronize_curriculum_state(device=self.device, world_size=self.gpu_world_size)

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        """Get a callable policy function for inference.

        This method returns a function that takes observations as input and returns
        actions. The policy function is configured to run on the specified device.

        Parameters
        ----------
        device : str | None, optional
            The device to run the policy on (e.g., 'cuda', 'cpu').
            If None, uses the default device.

        Returns
        -------
        Callable[[torch.Tensor], torch.Tensor]
            A function that takes observations as input and returns actions.
            The function expects input observations as a torch.Tensor and
            returns actions as a torch.Tensor. Both input and output tensors
            should be on the specified device.

        Notes
        -----
        This is an abstract method that should be implemented by subclasses.
        The returned policy function should:
        - Run the policy network on the specified device
        - Return actions in the expected format
        """
        raise NotImplementedError

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        raise NotImplementedError

    def save(self, path=None, name="last.ckpt"):
        raise NotImplementedError

    def _unwrap_env(self) -> BaseTask | Any:
        """Return the underlying environment.

        Algorithms that wrap the task (e.g. ``FastSACEnv``) keep a reference to the
        original environment as ``unwrapped_env`` during construction, so we simply
        return that when it is present.
        """
        return getattr(self, "unwrapped_env", self.env)

    def _collect_env_state(self) -> dict[str, torch.Tensor | float]:
        """Collect environment state for checkpointing via the environment interface."""
        env = self._unwrap_env()
        state = env.get_checkpoint_state()
        return state or {}

    def _restore_env_state(self, env_state: dict[str, torch.Tensor | float] | None) -> None:
        """Restore environment state from checkpoint via the environment interface."""
        if not env_state:
            return
        env = self._unwrap_env()
        env.load_checkpoint_state(env_state)

    def _validate_env_state(self, env_state: dict[str, torch.Tensor | float] | None) -> None:
        """Validate environment state before any algorithm state is mutated."""
        if not env_state:
            return
        env = self._unwrap_env()
        validator = getattr(env, "validate_full_resume_checkpoint_state", None)
        if not callable(validator):
            validator = getattr(env, "validate_checkpoint_state", None)
        if callable(validator):
            validator(env_state)
