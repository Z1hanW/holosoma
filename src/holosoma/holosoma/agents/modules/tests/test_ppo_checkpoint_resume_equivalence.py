from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from holosoma.agents.ppo.ppo import PPO
from holosoma.utils.resume_preflight import ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV
from holosoma.utils.rng_checkpoint import (
    capture_rng_checkpoint_state,
    restore_rng_checkpoint_state,
)


class _ToyMotionCommand:
    """Small iteration-dependent curriculum used by this core-resume smoke test."""

    def __init__(self) -> None:
        self.current_iteration = -1
        self.total_iterations = -1
        self.reset_at_zero_probability = -1.0

    def set_training_iteration(
        self,
        current_iteration: int,
        *,
        total_iterations: int,
    ) -> None:
        self.current_iteration = int(current_iteration)
        self.total_iterations = int(total_iterations)
        self.reset_at_zero_probability = float(current_iteration) / float(total_iterations)

    def get_motion_transition_contract(self) -> dict[str, Any]:
        return {
            "version": 1,
            "control_dt_s": 0.02,
            "source_semantics": "global_multi_clip_runtime",
            "prepend": {
                "implementation": "none",
                "applied": False,
                "steps": 0,
            },
            "append": {
                "implementation": "none",
                "applied": False,
                "steps": 0,
            },
        }


class _ToyCommandManager:
    def __init__(self, motion_command: _ToyMotionCommand) -> None:
        self.motion_command = motion_command

    def get_state(self, name: str) -> _ToyMotionCommand | None:
        return self.motion_command if name == "motion_command" else None


class _ToyCurriculumEnv:
    """Toy checkpoint state plus a stochastic canonical-reset surface.

    This deliberately does not model WBT, adaptive-timestep sampling, or a
    perception manager.  Their production checkpoint contracts have focused
    tests elsewhere; this fixture isolates PPO's single-rank resume wiring.
    """

    num_envs = 3
    max_episode_length = 97
    curriculum_state_checkpoint_required = True
    curriculum_state_sync_enabled = False

    def __init__(self, *, adaptive_weight: float, exposure_count: int) -> None:
        self.motion_command = _ToyMotionCommand()
        self.command_manager = _ToyCommandManager(self.motion_command)
        self.adaptive_weight = float(adaptive_weight)
        self.exposure_count = int(exposure_count)
        self.episode_length_buf = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
        )
        self.last_reset_draws: tuple[float, float, float] | None = None

    def get_checkpoint_state(self) -> dict[str, Any]:
        return {
            "version": 1,
            "adaptive_weight": torch.tensor(self.adaptive_weight, dtype=torch.float64),
            "exposure_count": self.exposure_count,
        }

    def validate_full_resume_checkpoint_state(self, state: dict[str, Any]) -> None:
        assert set(state) == {"version", "adaptive_weight", "exposure_count"}
        assert state["version"] == 1
        assert isinstance(state["adaptive_weight"], torch.Tensor)
        assert state["adaptive_weight"].shape == ()
        assert torch.isfinite(state["adaptive_weight"])
        assert type(state["exposure_count"]) is int
        assert state["exposure_count"] >= 0

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self.validate_full_resume_checkpoint_state(state)
        self.adaptive_weight = float(state["adaptive_weight"].item())
        self.exposure_count = int(state["exposure_count"])

    def reset_all_at_checkpoint_boundary(self) -> dict[str, torch.Tensor]:
        python_draw = random.random()
        numpy_draw = float(np.random.random())
        torch_draw = float(torch.rand((), dtype=torch.float64).item())
        self.last_reset_draws = (python_draw, numpy_draw, torch_draw)

        # Model toy adaptive/exposure mutations caused by selecting a new
        # episode. Both branches must start from the checkpointed values.
        self.exposure_count += self.num_envs
        self.adaptive_weight += 0.125 * python_draw + 0.25 * numpy_draw

        rows = []
        for env_id in range(self.num_envs):
            rows.append(
                [
                    self.adaptive_weight,
                    float(self.exposure_count),
                    self.motion_command.reset_at_zero_probability,
                    python_draw,
                    numpy_draw,
                    torch_draw + float(env_id),
                ]
            )
        return {"actor_obs": torch.tensor(rows, dtype=torch.float64)}

    # PPO requires reset_all to exist while this test intentionally exercises
    # the explicit checkpoint-boundary entry point above.
    reset_all = reset_all_at_checkpoint_boundary


@dataclass(frozen=True)
class _TrainingStateSnapshot:
    actor_state: dict[str, Any]
    critic_state: dict[str, Any]
    actor_output: torch.Tensor
    critic_output: torch.Tensor
    actor_optimizer_state: dict[str, Any]
    critic_optimizer_state: dict[str, Any]
    actor_learning_rate: float
    critic_learning_rate: float


@dataclass(frozen=True)
class _ContinuationSnapshot:
    observations: torch.Tensor
    reset_draws: tuple[float, float, float]
    subsequent_draws: tuple[float, float, torch.Tensor]
    episode_lengths: torch.Tensor
    dagger_episode_ages: torch.Tensor
    adaptive_weight: float
    exposure_count: int
    curriculum_iteration: tuple[int, int, float]
    ppo_coeff: float
    effective_bc_weight: float
    teacher_action_mix_ratio: float
    recurrent_hidden_cleared: tuple[bool, bool]
    before_followup_update: _TrainingStateSnapshot
    after_followup_update: _TrainingStateSnapshot


def _checkpoint_metadata(self: PPO, *, iteration: int) -> dict[str, Any]:
    metadata: dict[str, Any] = {"iteration": int(iteration)}
    if self._motion_transition_contract is not None:
        metadata.update(
            {
                "motion_transition_contract": self._motion_transition_contract,
                "motion_transition_contract_sha256": self._motion_transition_contract_sha256,
            }
        )
    return metadata


def _initialize_optimizer_state(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.arange(18, dtype=torch.float64).reshape(3, 6) / 10.0
    model(inputs).square().sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def _make_ppo(
    env: _ToyCurriculumEnv,
    *,
    model_offset: float,
    checkpoint_path,
) -> PPO:
    ppo = object.__new__(PPO)
    ppo.env = env
    ppo.device = "cpu"
    ppo.gpu_global_rank = 0
    ppo.gpu_world_size = 1
    ppo.is_multi_gpu = False
    ppo.is_main_process = True
    ppo.current_learning_iteration = 3

    ppo.actor = nn.Linear(6, 2, dtype=torch.float64)
    ppo.critic = nn.Linear(6, 1, dtype=torch.float64)
    with torch.no_grad():
        actor_values = torch.arange(12, dtype=torch.float64).reshape(2, 6) / 20.0
        critic_values = torch.arange(6, dtype=torch.float64).reshape(1, 6) / 30.0
        ppo.actor.weight.copy_(actor_values + model_offset)
        ppo.actor.bias.copy_(torch.tensor([0.25, -0.5], dtype=torch.float64) + model_offset)
        ppo.critic.weight.copy_(critic_values + model_offset)
        ppo.critic.bias.fill_(0.75 + model_offset)

    # Recurrent state is deliberately process-local and not checkpointed. The
    # canonical boundary must clear it identically on both branches.
    ppo.actor.perception_time_gru = SimpleNamespace(hidden=torch.ones(1, 3, 2))
    ppo.critic.perception_time_gru = SimpleNamespace(hidden=torch.ones(1, 3, 1))

    # Give the restart branch deliberately different optimizer hyperparameters
    # as well as different moments. A correct load must replace the complete
    # state/param-group configuration and the public learning-rate mirrors.
    restarted_fixture = model_offset != 0.0
    actor_lr = 0.15 if restarted_fixture else 0.05
    critic_lr = 0.125 if restarted_fixture else 0.025
    # Non-LR optimizer options are part of the scientific runtime contract and
    # must already match; only LR is intentionally different because load()
    # is responsible for restoring it from the checkpoint.
    momentum = 0.9
    ppo.actor_optimizer = torch.optim.SGD(
        ppo.actor.parameters(),
        lr=actor_lr,
        momentum=momentum,
    )
    ppo.critic_optimizer = torch.optim.SGD(
        ppo.critic.parameters(),
        lr=critic_lr,
        momentum=momentum,
    )
    _initialize_optimizer_state(ppo.actor, ppo.actor_optimizer)
    _initialize_optimizer_state(ppo.critic, ppo.critic_optimizer)

    ppo.actor_learning_rate = actor_lr
    ppo.critic_learning_rate = critic_lr
    ppo.min_actor_learning_rate = 1.0e-6
    ppo.max_actor_learning_rate = 1.0
    ppo.min_critic_learning_rate = 1.0e-6
    ppo.max_critic_learning_rate = 1.0
    ppo.actor_obs_normalizers = {"actor_obs": nn.Identity()}
    ppo.critic_obs_normalizers = {"critic_obs": nn.Identity()}
    ppo.actor_perception_key = ""
    # Exercise the production DAgger/PPO schedule refresh without pretending
    # that this toy fixture covers teacher inference, student perception, or
    # real AS. Non-zero teacher rollout mixing is scientifically incompatible
    # with a PPO-contributing schedule, so the valid value here is zero.
    ppo.distill_enabled = True
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_start_epoch = 2
    ppo.dagger_end_epoch = 6
    ppo.ppo_start_coeff = 0.0
    ppo.ppo_target_coeff = 1.0
    ppo.ppo_schedule_step_epochs = 0
    ppo.ppo_coeff = -10.0 if restarted_fixture else -1.0
    ppo.dagger_loss_coef = 2.0
    ppo.bc_loss_coef = 1.0
    ppo._configured_bc_loss_coef = 1.0
    ppo.switch_to_rl_after = -1
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.0
    ppo._configured_teacher_action_mix_ratio = 0.0
    ppo.teacher_action_mix_ratio_start = None
    ppo.teacher_action_mix_ratio_end = None
    ppo.teacher_action_mix_ratio_end_iteration = -1
    ppo.use_teacher_action_mix_schedule = False
    ppo.teacher_use_stochastic_actions = False
    ppo.use_multi_teacher = False
    ppo.multi_teacher_select_obs_var = "teacher_checkpoint_index"
    ppo.dagger_match_std = False
    ppo.ppo_start_noise_std_until_coeff = 0.1
    ppo.fixed_bc_eval_num_samples = 0
    ppo.fixed_bc_guard_enabled = False
    ppo.dagger_ignore_episode_initial_steps = 2
    ppo.ppo_start_noise_std = None
    ppo.config = SimpleNamespace(
        load_optimizer=True,
        normalize_actor_obs=False,
        normalize_critic_obs=False,
        init_noise_std=0.2,
        init_at_random_ep_len=True,
        save_interval=4,
        reset_rollout_at_checkpoint=True,
    )
    ppo._experiment_config = None
    ppo._training_provenance = None
    ppo._motion_transition_contract = None
    ppo._motion_transition_contract_sha256 = None
    ppo._checkpoint_metadata = MethodType(_checkpoint_metadata, ppo)
    ppo.logging_helper = SimpleNamespace(
        save_checkpoint_artifact=lambda checkpoint, _path: torch.save(
            checkpoint,
            checkpoint_path,
        )
    )
    # Guard against accidentally constructing a schedule that production setup
    # would reject even though this object bypasses PPO.__init__ for speed.
    ppo._validate_teacher_rollout_action_config()
    return ppo


def _training_state_snapshot(ppo: PPO, observations: torch.Tensor) -> _TrainingStateSnapshot:
    return _TrainingStateSnapshot(
        actor_state=copy.deepcopy(ppo.actor.state_dict()),
        critic_state=copy.deepcopy(ppo.critic.state_dict()),
        actor_output=ppo.actor(observations).detach().clone(),
        critic_output=ppo.critic(observations).detach().clone(),
        actor_optimizer_state=copy.deepcopy(ppo.actor_optimizer.state_dict()),
        critic_optimizer_state=copy.deepcopy(ppo.critic_optimizer.state_dict()),
        actor_learning_rate=float(ppo.actor_learning_rate),
        critic_learning_rate=float(ppo.critic_learning_rate),
    )


def _apply_identical_followup_update(ppo: PPO, observations: torch.Tensor) -> None:
    actor_target = torch.arange(
        observations.shape[0] * 2,
        dtype=observations.dtype,
    ).reshape(observations.shape[0], 2) / 7.0
    critic_target = torch.arange(
        observations.shape[0],
        dtype=observations.dtype,
    ).reshape(observations.shape[0], 1) / 11.0

    ppo.actor_optimizer.zero_grad(set_to_none=True)
    (ppo.actor(observations) - actor_target).square().mean().backward()
    ppo.actor_optimizer.step()
    ppo.actor_optimizer.zero_grad(set_to_none=True)

    ppo.critic_optimizer.zero_grad(set_to_none=True)
    (ppo.critic(observations) - critic_target).square().mean().backward()
    ppo.critic_optimizer.step()
    ppo.critic_optimizer.zero_grad(set_to_none=True)


def _continue_from_checkpoint_boundary(
    ppo: PPO,
    *,
    next_iteration: int,
    total_iterations: int,
) -> _ContinuationSnapshot:
    ppo.current_learning_iteration = next_iteration
    # This is the same objective-before-reset ordering used by PPO.learn() for
    # both a resumed initial rollout and an uninterrupted post-save rollout.
    ppo._prepare_rollout_objective_for_iteration(next_iteration)
    observations = ppo._reset_rollout_stream_at_canonical_boundary(
        current_iteration=next_iteration,
        total_iterations=total_iterations,
    )["actor_obs"]
    env = ppo.env
    assert env.last_reset_draws is not None
    before_followup_update = _training_state_snapshot(ppo, observations)
    subsequent_draws = (
        random.random(),
        float(np.random.random()),
        torch.rand(4, dtype=torch.float64),
    )
    _apply_identical_followup_update(ppo, observations)
    after_followup_update = _training_state_snapshot(ppo, observations)
    assert not torch.equal(
        after_followup_update.actor_output,
        before_followup_update.actor_output,
    )
    assert not torch.equal(
        after_followup_update.critic_output,
        before_followup_update.critic_output,
    )
    return _ContinuationSnapshot(
        observations=observations.detach().clone(),
        reset_draws=env.last_reset_draws,
        subsequent_draws=subsequent_draws,
        episode_lengths=env.episode_length_buf.detach().clone(),
        dagger_episode_ages=ppo._dagger_episode_step_buf.detach().clone(),
        adaptive_weight=env.adaptive_weight,
        exposure_count=env.exposure_count,
        curriculum_iteration=(
            env.motion_command.current_iteration,
            env.motion_command.total_iterations,
            env.motion_command.reset_at_zero_probability,
        ),
        ppo_coeff=float(ppo.ppo_coeff),
        effective_bc_weight=float(ppo._effective_dagger_loss_weight()),
        teacher_action_mix_ratio=float(ppo.teacher_action_mix_ratio),
        recurrent_hidden_cleared=(
            ppo.actor.perception_time_gru.hidden is None,
            ppo.critic.perception_time_gru.hidden is None,
        ),
        before_followup_update=before_followup_update,
        after_followup_update=after_followup_update,
    )


def _assert_tree_equal(actual: Any, expected: Any, *, path: str) -> None:
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), path
        assert actual.dtype == expected.dtype, path
        assert actual.device == expected.device, path
        assert torch.equal(actual, expected), path
        return
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_tree_equal(actual[key], expected[key], path=f"{path}[{key!r}]")
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected)), path
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            _assert_tree_equal(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    assert actual == expected, path


def _assert_training_states_equal(
    actual: _TrainingStateSnapshot,
    expected: _TrainingStateSnapshot,
    *,
    phase: str,
) -> None:
    _assert_tree_equal(actual.actor_state, expected.actor_state, path=f"{phase}.actor_state")
    _assert_tree_equal(actual.critic_state, expected.critic_state, path=f"{phase}.critic_state")
    assert torch.equal(actual.actor_output, expected.actor_output)
    assert torch.equal(actual.critic_output, expected.critic_output)
    _assert_tree_equal(
        actual.actor_optimizer_state,
        expected.actor_optimizer_state,
        path=f"{phase}.actor_optimizer_state",
    )
    _assert_tree_equal(
        actual.critic_optimizer_state,
        expected.critic_optimizer_state,
        path=f"{phase}.critic_optimizer_state",
    )
    assert actual.actor_learning_rate == expected.actor_learning_rate
    assert actual.critic_learning_rate == expected.critic_learning_rate


def _assert_snapshots_equal(
    uninterrupted: _ContinuationSnapshot,
    resumed: _ContinuationSnapshot,
) -> None:
    assert torch.equal(resumed.observations, uninterrupted.observations)
    assert resumed.reset_draws == uninterrupted.reset_draws
    assert resumed.subsequent_draws[:2] == uninterrupted.subsequent_draws[:2]
    assert torch.equal(resumed.subsequent_draws[2], uninterrupted.subsequent_draws[2])
    assert torch.equal(resumed.episode_lengths, uninterrupted.episode_lengths)
    assert torch.equal(resumed.dagger_episode_ages, uninterrupted.dagger_episode_ages)
    assert resumed.adaptive_weight == uninterrupted.adaptive_weight
    assert resumed.exposure_count == uninterrupted.exposure_count
    assert resumed.curriculum_iteration == uninterrupted.curriculum_iteration
    assert resumed.ppo_coeff == uninterrupted.ppo_coeff == 0.5
    assert resumed.effective_bc_weight == uninterrupted.effective_bc_weight == 1.0
    assert resumed.teacher_action_mix_ratio == uninterrupted.teacher_action_mix_ratio == 0.0
    assert resumed.recurrent_hidden_cleared == (True, True)
    assert uninterrupted.recurrent_hidden_cleared == (True, True)
    _assert_training_states_equal(
        resumed.before_followup_update,
        uninterrupted.before_followup_update,
        phase="before_followup_update",
    )
    _assert_training_states_equal(
        resumed.after_followup_update,
        uninterrupted.after_followup_update,
        phase="after_followup_update",
    )


def test_single_rank_core_resume_matches_post_save_canonical_boundary(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Single-rank PPO core state matches after save/load and one next update.

    This is intentionally not a full scientific-resume, real-AS, simulator, or
    perception test. It uses the explicit legacy-unprovenanced hatch solely to
    isolate PPO's model/optimizer/RNG/toy-environment boundary mechanics.
    """

    monkeypatch.setenv(ALLOW_LEGACY_UNPROVENANCED_RESUME_ENV, "1")
    checkpoint_path = tmp_path / "model_00004.pt"
    original_rng = capture_rng_checkpoint_state()
    try:
        uninterrupted_ppo = _make_ppo(
            _ToyCurriculumEnv(adaptive_weight=2.5, exposure_count=11),
            model_offset=0.0,
            checkpoint_path=checkpoint_path,
        )
        random.seed(4101)
        np.random.seed(4102)
        torch.manual_seed(4103)

        # PPO.save() captures RNG before serialization, records the toy
        # rank-local state, and restores the pre-publication RNG on return.
        uninterrupted_ppo.save(
            checkpoint_path,
            next_iteration=4,
        )
        uninterrupted = _continue_from_checkpoint_boundary(
            uninterrupted_ppo,
            next_iteration=4,
            total_iterations=8,
        )

        # Start from deliberately unrelated model, optimizer, curriculum and
        # process RNG state. Production load() must replace every continuation
        # input before the same canonical reset is applied.
        resumed_ppo = _make_ppo(
            _ToyCurriculumEnv(adaptive_weight=999.0, exposure_count=700),
            model_offset=10.0,
            checkpoint_path=checkpoint_path,
        )
        random.seed(9901)
        np.random.seed(9902)
        torch.manual_seed(9903)
        resumed_ppo.load(str(checkpoint_path))
        assert resumed_ppo.current_learning_iteration == 4

        resumed = _continue_from_checkpoint_boundary(
            resumed_ppo,
            next_iteration=4,
            total_iterations=8,
        )
        _assert_snapshots_equal(uninterrupted, resumed)
    finally:
        restore_rng_checkpoint_state(original_rng)
