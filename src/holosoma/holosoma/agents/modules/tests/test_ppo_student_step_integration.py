from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal

from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.ppo.ppo import PPO


class _PerceptionActor(nn.Module):
    def __init__(self, perception_key: str = "student_depth"):
        super().__init__()
        self.obs_weight = nn.Parameter(torch.tensor(0.4))
        self.perception_weight = nn.Parameter(torch.tensor(0.3))
        self.perception_input_name = perception_key
        self.distribution: Normal | None = None

    def _mean(self, policy_state: dict[str, torch.Tensor]) -> torch.Tensor:
        return (
            policy_state["actor_obs"][:, :1] * self.obs_weight
            + policy_state[self.perception_input_name][:, :1] * self.perception_weight
        )

    def update_distribution_from_policy_state(self, policy_state: dict[str, torch.Tensor]) -> None:
        mean = self._mean(policy_state)
        self.distribution = Normal(mean, torch.ones_like(mean))

    def act(self, policy_state: dict[str, torch.Tensor]) -> torch.Tensor:
        self.update_distribution_from_policy_state(policy_state)
        return self.action_mean

    def act_inference(self, policy_state: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._mean(policy_state)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        assert self.distribution is not None
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor) -> None:
        del dones


class _PerceptionActorWithTrainableStd(_PerceptionActor):
    """Tiny Gaussian actor exposing the same trainable std as PPOActor."""

    def __init__(self, perception_key: str = "student_depth"):
        super().__init__(perception_key)
        self.std = nn.Parameter(torch.tensor(1.0))

    def update_distribution_from_policy_state(self, policy_state: dict[str, torch.Tensor]) -> None:
        mean = self._mean(policy_state)
        self.distribution = Normal(mean, self.std.expand_as(mean))


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def evaluate(self, policy_state: dict[str, torch.Tensor]) -> torch.Tensor:
        return policy_state["critic_obs"][:, :1] * self.weight

    def reset(self, dones: torch.Tensor) -> None:
        del dones


def _student_step_stub() -> PPO:
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.current_learning_iteration = 0
    ppo.gpu_global_rank = 0
    ppo.gpu_world_size = 1
    ppo.is_multi_gpu = False
    ppo.use_time_gru = False
    ppo.use_symmetry = False
    ppo.actor_perception_key = "student_depth"
    ppo.critic_perception_key = ""
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.critic_obs_keys = ["critic_obs"]
    ppo.actor = _PerceptionActor(ppo.actor_perception_key)
    ppo.critic = _TinyCritic()
    ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=0.05)
    ppo.critic_optimizer = torch.optim.SGD(ppo.critic.parameters(), lr=0.05)
    ppo.dagger_enabled = True
    ppo.distill_enabled = True
    ppo.distill_mode = "dagger"
    ppo.distill_loss_fn = F.mse_loss
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_start_epoch = 0
    ppo.dagger_end_epoch = 6300
    ppo.ppo_start_coeff = 0.0
    ppo.ppo_target_coeff = 0.9
    ppo.ppo_schedule_step_epochs = 700
    ppo.ppo_coeff = 0.0
    ppo.bc_loss_coef = 1.0
    ppo._configured_bc_loss_coef = 1.0
    ppo.switch_to_rl_after = -1
    ppo.dagger_loss_coef = 1.0
    ppo.clip_teacher_actions = False
    ppo.dagger_ignore_zero_teacher_actions = False
    ppo.dagger_match_std = False
    ppo.ppo_start_noise_std = None
    ppo.config = SimpleNamespace(
        desired_kl=None,
        schedule="fixed",
        clip_param=0.2,
        entropy_coef=0.0,
        symmetry_actor_coef=0.0,
        symmetry_critic_coef=0.0,
        value_loss_coef=1.0,
        max_grad_norm=100.0,
        init_noise_std=1.0,
    )
    ppo.max_grad_norm = ppo.config.max_grad_norm
    return ppo


def _student_minibatch(
    ppo: PPO,
    *,
    teacher_matches_student: bool,
) -> dict[str, torch.Tensor]:
    actor_obs = torch.tensor([[1.0], [2.0]])
    actor_perception = torch.tensor([[3.0], [4.0]])
    critic_obs = torch.tensor([[1.5], [2.5]])
    with torch.no_grad():
        policy_state = {
            "actor_obs": actor_obs,
            ppo.actor_perception_key: actor_perception,
        }
        old_mean = ppo.actor.act_inference(policy_state)
        old_values = ppo.critic.evaluate({"critic_obs": critic_obs})
    actions = old_mean + 0.5
    old_log_prob = Normal(old_mean, torch.ones_like(old_mean)).log_prob(actions).sum(-1, keepdim=True)
    teacher_actions = old_mean.clone() if teacher_matches_student else torch.zeros_like(old_mean)
    return {
        "actor_obs": actor_obs,
        "critic_obs": critic_obs,
        ppo.actor_perception_key: actor_perception,
        "actions": actions,
        "values": old_values,
        "advantages": torch.ones_like(old_values),
        "returns": torch.zeros_like(old_values),
        "actions_log_prob": old_log_prob,
        "action_mean": old_mean,
        "action_sigma": torch.ones_like(old_mean),
        "teacher_actions": teacher_actions,
        "teacher_bc_mask": torch.ones_like(teacher_actions, dtype=torch.bool),
        "teacher_indices": torch.zeros_like(teacher_actions, dtype=torch.long),
    }


def _empty_loss_dict() -> dict[str, float]:
    return {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0}


def _disable_supervised_shortcuts(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "HOLOSOMA_DAGGER_SUPERVISED_ONLY",
        "HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP",
        "HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_pure_bc_update_moves_actor_perception_and_critic_with_expected_gradients(
    monkeypatch: pytest.MonkeyPatch,
):
    _disable_supervised_shortcuts(monkeypatch)
    ppo = _student_step_stub()
    ppo._refresh_distillation_iteration_state(0)
    minibatch = _student_minibatch(ppo, teacher_matches_student=False)
    actor_obs_weight_before = ppo.actor.obs_weight.detach().clone()
    actor_perception_weight_before = ppo.actor.perception_weight.detach().clone()
    critic_weight_before = ppo.critic.weight.detach().clone()

    with torch.no_grad():
        initial_mean = ppo.actor.act_inference(
            {
                "actor_obs": minibatch["actor_obs"],
                ppo.actor_perception_key: minibatch[ppo.actor_perception_key],
            }
        )
        expected_obs_grad = (2.0 * initial_mean * minibatch["actor_obs"]).mean()
        expected_perception_grad = (
            2.0 * initial_mean * minibatch[ppo.actor_perception_key]
        ).mean()
        initial_values = ppo.critic.evaluate({"critic_obs": minibatch["critic_obs"]})
        expected_critic_grad = (2.0 * initial_values * minibatch["critic_obs"]).mean()

    losses = ppo._update_algo_step(minibatch, _empty_loss_dict())

    assert ppo.ppo_coeff == 0.0
    assert ppo._use_deterministic_student_actions() is True
    assert ppo.actor.obs_weight.grad.item() == pytest.approx(expected_obs_grad.item())
    assert ppo.actor.perception_weight.grad.item() == pytest.approx(expected_perception_grad.item())
    assert ppo.critic.weight.grad.item() == pytest.approx(expected_critic_grad.item())
    assert not torch.equal(ppo.actor.obs_weight, actor_obs_weight_before)
    assert not torch.equal(ppo.actor.perception_weight, actor_perception_weight_before)
    assert not torch.equal(ppo.critic.weight, critic_weight_before)
    assert losses["bc_loss"] > 0.0
    assert losses["Value"] > 0.0


@pytest.mark.parametrize("scheduled", [True, False])
def test_pure_bc_never_constructs_overflowing_stale_ppo_ratio_and_critic_steps(
    monkeypatch: pytest.MonkeyPatch,
    scheduled: bool,
) -> None:
    _disable_supervised_shortcuts(monkeypatch)
    ppo = _student_step_stub()
    ppo.use_ppo_dagger_schedule = scheduled
    ppo.ppo_coeff = 0.0
    ppo.bc_loss_coef = 1.0
    minibatch = _student_minibatch(ppo, teacher_matches_student=False)
    # The current log probability is O(1), so this stale value would make
    # exp(new_logp - old_logp) overflow if the irrelevant PPO path ran.
    minibatch["actions_log_prob"].fill_(-1.0e30)
    monkeypatch.setattr(
        ppo.actor,
        "get_actions_log_prob",
        lambda _actions: pytest.fail("pure BC must not evaluate action log probabilities"),
    )
    critic_before = ppo.critic.weight.detach().clone()

    losses = ppo._update_algo_step(minibatch, _empty_loss_dict())

    assert losses["Surrogate"] == pytest.approx(0.0)
    assert losses["bc_loss"] > 0.0
    assert losses["Value"] > 0.0
    assert all(torch.isfinite(torch.as_tensor(value)) for value in losses.values())
    assert not torch.equal(ppo.critic.weight, critic_before)


@pytest.mark.parametrize("scheduled", [True, False])
def test_hybrid_objective_still_constructs_ppo_ratio(
    monkeypatch: pytest.MonkeyPatch,
    scheduled: bool,
) -> None:
    _disable_supervised_shortcuts(monkeypatch)
    ppo = _student_step_stub()
    ppo.use_ppo_dagger_schedule = scheduled
    if scheduled:
        ppo.ppo_coeff = 0.1
    else:
        ppo.bc_loss_coef = 0.9
    minibatch = _student_minibatch(ppo, teacher_matches_student=False)
    original_get_log_prob = ppo.actor.get_actions_log_prob
    calls = []

    def record_log_prob(actions):
        calls.append(actions)
        return original_get_log_prob(actions)

    monkeypatch.setattr(ppo.actor, "get_actions_log_prob", record_log_prob)

    losses = ppo._compute_ppo_loss(minibatch)

    assert len(calls) == 1
    assert torch.isfinite(losses["surrogate_loss"])
    assert torch.isfinite(losses["actor_loss"])


def test_schedule_at_full_ppo_does_not_require_or_compute_bc_labels() -> None:
    ppo = _student_step_stub()
    ppo.ppo_coeff = 1.0
    minibatch = _student_minibatch(ppo, teacher_matches_student=False)
    minibatch.pop("teacher_actions")
    minibatch.pop("teacher_bc_mask")
    minibatch.pop("teacher_indices")

    losses = ppo._compute_ppo_loss(minibatch)

    assert losses["dagger_weight"].item() == pytest.approx(0.0)
    assert losses["bc_loss"].item() == pytest.approx(0.0)
    assert losses["actor_optimizer_step_skipped_no_signal"] == pytest.approx(0.0)


def test_rollout_wide_bc_denominator_weights_each_valid_sample_equally() -> None:
    ppo = _student_step_stub()
    ppo.ppo_coeff = 0.0
    base = _student_minibatch(ppo, teacher_matches_student=False)
    first = {key: value.clone() for key, value in base.items()}
    second = {key: value.clone() for key, value in base.items()}
    first["teacher_bc_mask"] = torch.tensor([[True], [False]])
    second["teacher_bc_mask"] = torch.tensor([[True], [True]])

    full = {
        key: torch.cat([first[key], second[key]], dim=0)
        for key in first
    }
    full_loss = ppo._compute_ppo_loss(full)["actor_loss"]
    parameters = tuple(ppo.actor.parameters())
    full_gradients = torch.autograd.grad(full_loss, parameters)

    # Three valid samples across two minibatches -> fixed per-step
    # denominator 3 / 2.  Averaging the two same-parameter minibatch
    # gradients must reconstruct the full valid-sample mean.
    fixed_denominator = torch.tensor(1.5)
    minibatch_gradients = []
    for minibatch in (first, second):
        minibatch["_dagger_bc_denominator"] = fixed_denominator
        loss = ppo._compute_ppo_loss(minibatch)["actor_loss"]
        minibatch_gradients.append(torch.autograd.grad(loss, parameters))

    averaged_gradients = tuple(
        (left + right) / 2.0
        for left, right in zip(*minibatch_gradients)
    )
    for actual, expected in zip(averaged_gradients, full_gradients):
        assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)


def test_rollout_bc_denominator_uses_total_valid_count_per_minibatch() -> None:
    ppo = _student_step_stub()
    ppo.ppo_coeff = 0.0
    ppo.config.num_mini_batches = 2
    ppo.dagger_ignore_zero_teacher_actions = False
    buffers = {
        "teacher_actions": torch.ones(4, 1, 1),
        "teacher_bc_mask": torch.tensor([True, False, True, True]).view(4, 1, 1),
    }

    class _Storage:
        step = 4

        def __getitem__(self, key):
            return buffers[key]

    ppo.storage = _Storage()

    denominator = ppo._rollout_bc_denominator_per_minibatch()

    assert denominator is not None
    assert denominator.item() == pytest.approx(1.5)


def test_contiguous_bc_presence_is_reduced_once_for_the_complete_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ppo = _student_step_stub()
    ppo.config.num_mini_batches = 2
    ppo.config.num_learning_epochs = 2
    ppo.dagger_ignore_zero_teacher_actions = False
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    buffers = {
        "teacher_actions": torch.ones(4, 1, 1),
        "teacher_bc_mask": torch.tensor([True, False, False, False]).view(4, 1, 1),
    }

    class _Storage:
        step = 4

        def __getitem__(self, key):
            return buffers[key]

    ppo.storage = _Storage()
    ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)
    reduced_payloads: list[torch.Tensor] = []

    def reduce_presence(self, payload, *, op):
        assert op == torch.distributed.ReduceOp.SUM
        reduced_payloads.append(payload.detach().clone())
        # Rank zero has one valid sample in minibatch zero; emulate a remote
        # rank with one valid sample only in minibatch one.
        return payload + payload.new_tensor([0.0, 1.0])

    ppo._all_reduce_small_tensor = MethodType(reduce_presence, ppo)
    monkeypatch.setenv("HOLOSOMA_CONTIGUOUS_MINIBATCHES", "1")

    presence = ppo._rollout_bc_minibatch_presence()

    assert len(reduced_payloads) == 1
    assert torch.equal(reduced_payloads[0], torch.tensor([1.0, 0.0]))
    assert presence == (True, True, True, True)


def test_cached_bc_presence_avoids_per_minibatch_collective() -> None:
    ppo = _student_step_stub()
    ppo._global_bc_denominator_and_presence = MethodType(
        lambda self, count: pytest.fail("cached presence must avoid a per-minibatch collective"),
        ppo,
    )

    denominator, has_valid_samples = ppo._bc_denominator_and_presence_for_minibatch(
        {
            "_dagger_bc_denominator": torch.tensor(1.5),
            "_dagger_bc_has_valid_samples": False,
        },
        torch.tensor(1.0),
    )

    assert denominator.item() == pytest.approx(1.5)
    assert has_valid_samples is False


def test_cached_bc_presence_rejects_non_boolean_marker() -> None:
    ppo = _student_step_stub()

    with pytest.raises(ValueError, match="must be a Python bool"):
        ppo._bc_denominator_and_presence_for_minibatch(
            {
                "_dagger_bc_denominator": torch.tensor(1.5),
                "_dagger_bc_has_valid_samples": torch.tensor(True),
            },
            torch.tensor(1.0),
        )


def test_first_nonzero_ppo_tier_switches_actor_from_bc_only_to_hybrid(
    monkeypatch: pytest.MonkeyPatch,
):
    _disable_supervised_shortcuts(monkeypatch)
    ppo = _student_step_stub()
    minibatch = _student_minibatch(ppo, teacher_matches_student=True)
    actor_before = tuple(parameter.detach().clone() for parameter in ppo.actor.parameters())

    ppo.current_learning_iteration = 699
    ppo._refresh_distillation_iteration_state(699)
    ppo._update_algo_step(minibatch, _empty_loss_dict())

    assert ppo.ppo_coeff == 0.0
    assert ppo._use_deterministic_student_actions() is True
    assert all(torch.equal(parameter, before) for parameter, before in zip(ppo.actor.parameters(), actor_before))

    ppo.current_learning_iteration = 700
    ppo._refresh_distillation_iteration_state(700)
    tier_losses = ppo._update_algo_step(minibatch, _empty_loss_dict())

    assert ppo.ppo_coeff == pytest.approx(0.1)
    assert ppo._use_deterministic_student_actions() is False
    assert any(not torch.equal(parameter, before) for parameter, before in zip(ppo.actor.parameters(), actor_before))
    assert tier_losses["ppo_coeff"] == pytest.approx(0.1)
    assert tier_losses["dagger_weight"] == pytest.approx(0.9)


def test_pure_bc_all_invalid_batch_does_not_apply_optimizer_momentum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_supervised_shortcuts(monkeypatch)
    ppo = _student_step_stub()
    ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=0.05, momentum=0.9)
    ppo._refresh_distillation_iteration_state(0)

    valid_minibatch = _student_minibatch(ppo, teacher_matches_student=False)
    ppo._update_algo_step(valid_minibatch, _empty_loss_dict())
    actor_after_valid_step = tuple(parameter.detach().clone() for parameter in ppo.actor.parameters())

    invalid_minibatch = _student_minibatch(ppo, teacher_matches_student=False)
    invalid_minibatch["teacher_bc_mask"].zero_()
    invalid_minibatch["_dagger_bc_denominator"] = torch.tensor(1.0)
    invalid_minibatch["_dagger_bc_has_valid_samples"] = False
    losses = ppo._update_algo_step(invalid_minibatch, _empty_loss_dict())

    assert all(
        torch.equal(parameter, expected)
        for parameter, expected in zip(ppo.actor.parameters(), actor_after_valid_step)
    )
    assert losses["actor_optimizer_step_skipped_no_signal"] == pytest.approx(1.0)


def test_pure_bc_entropy_diagnostic_does_not_decay_or_step_policy_std(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_supervised_shortcuts(monkeypatch)
    ppo = _student_step_stub()
    ppo.actor = _PerceptionActorWithTrainableStd(ppo.actor_perception_key)
    ppo.actor_optimizer = torch.optim.AdamW(
        ppo.actor.parameters(),
        lr=0.1,
        weight_decay=0.1,
    )
    ppo.config.entropy_coef = 0.25
    ppo._refresh_distillation_iteration_state(0)
    minibatch = _student_minibatch(ppo, teacher_matches_student=False)
    std_before = ppo.actor.std.detach().clone()

    losses = ppo._update_algo_step(minibatch, _empty_loss_dict())

    assert losses["Entropy"] > 0.0
    assert torch.equal(ppo.actor.std, std_before)
    assert ppo.actor.std.grad is None
    assert ppo.actor.std not in ppo.actor_optimizer.state


def test_permanent_pure_bc_with_zero_value_weight_never_evaluates_critic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_supervised_shortcuts(monkeypatch)
    ppo = _student_step_stub()
    ppo._supervised_dagger_only = False
    ppo.use_ppo_dagger_schedule = False
    ppo.bc_loss_coef = 1.0
    ppo._configured_bc_loss_coef = 1.0
    ppo.config.value_loss_coef = 0.0
    minibatch = _student_minibatch(ppo, teacher_matches_student=False)
    ppo.critic.evaluate = lambda _state: pytest.fail("zero-weight critic was evaluated")

    losses = ppo._compute_ppo_loss(minibatch)

    assert losses["value_loss"].item() == pytest.approx(0.0)
    assert losses["critic_loss"].item() == pytest.approx(0.0)
    assert losses["bc_loss"].item() > 0.0


def test_streamed_microbatch_backward_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ppo = _student_step_stub()
    ppo._refresh_distillation_iteration_state(0)
    minibatch = _student_minibatch(ppo, teacher_matches_student=False)
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ONLY", "1")
    monkeypatch.setenv("HOLOSOMA_DAGGER_SUPERVISED_ACTOR_ONLY_STEP", "1")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_STREAM_BACKWARD", "1")
    monkeypatch.setenv("HOLOSOMA_SUPERVISED_ACTOR_MICROBATCH", "1")
    actor_before = tuple(parameter.detach().clone() for parameter in ppo.actor.parameters())

    def fail_backward(_gradient):
        raise RuntimeError("injected streamed backward failure")

    hook = ppo.actor.obs_weight.register_hook(fail_backward)
    try:
        with pytest.raises(RuntimeError, match="injected streamed backward failure"):
            ppo._update_algo_step(minibatch, _empty_loss_dict())
    finally:
        hook.remove()

    assert all(
        torch.equal(parameter, before)
        for parameter, before in zip(ppo.actor.parameters(), actor_before)
    )


class _TeacherActor(nn.Module):
    perception_input_name = "teacher_depth"

    def act_inference(self, policy_state: dict[str, torch.Tensor]) -> torch.Tensor:
        return policy_state["actor_obs"] + policy_state[self.perception_input_name]

    def reset(self, dones: torch.Tensor) -> None:
        del dones


class _NonFiniteTeacherActor(_TeacherActor):
    def act_inference(self, policy_state: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.full_like(policy_state["actor_obs"], float("inf"))


class _CaptureStorage:
    def __init__(self):
        self.transition: dict[str, torch.Tensor] = {}
        self.computed: dict[str, torch.Tensor] = {}
        self.required_on_add_keys = frozenset(
            {
                "actor_obs",
                "critic_obs",
                "actor_obs_raw",
                "critic_obs_raw",
                "actions",
                "values",
                "actions_log_prob",
                "action_mean",
                "action_sigma",
                "rewards",
                "dones",
                "teacher_actions",
                "teacher_indices",
                "student_depth",
            }
        )

    def add(self, **data: torch.Tensor | None) -> None:
        self.transition = {
            key: value.detach().clone()
            for key, value in data.items()
            if value is not None
        }

    def __getitem__(self, key: str) -> torch.Tensor:
        if key in self.computed:
            return self.computed[key]
        return self.transition[key].unsqueeze(0)

    def __setitem__(self, key: str, value: torch.Tensor) -> None:
        self.computed[key] = value.detach().clone()


class _StepEnv:
    def __init__(self, initial_obs: dict[str, torch.Tensor]):
        self.initial_obs = initial_obs
        self.command_manager = None
        self.actions_seen: torch.Tensor | None = None

    def step(self, action_dict: dict[str, torch.Tensor]):
        self.actions_seen = action_dict["actions"].detach().clone()
        next_obs = {key: value + 100.0 for key, value in self.initial_obs.items()}
        batch = next(iter(next_obs.values())).shape[0]
        return (
            next_obs,
            torch.zeros(batch),
            # BaseTask.reset_buf uses integer storage; PPO must canonicalize the
            # transition field to the bool RolloutStorage schema.
            torch.zeros(batch, dtype=torch.long),
            {
                "time_outs": torch.zeros(batch, dtype=torch.bool),
                "final_observations": {},
            },
        )


def test_rollout_teacher_labels_and_student_perception_share_pre_step_state():
    ppo = _student_step_stub()
    ppo.config.num_steps_per_env = 1
    ppo.config.gamma = 0.99
    ppo.config.lam = 0.95
    ppo.algo_timing = SimpleNamespace(enabled=False)
    ppo.log_dir = None
    ppo.teacher_obs_keys = ["teacher_obs"]
    ppo.teacher_actor = _TeacherActor()
    ppo.use_multi_teacher = False
    ppo.teacher_use_stochastic_actions = False
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.0
    ppo.dagger_ignore_episode_initial_steps = 0
    ppo._normalize_actor_obs = MethodType(lambda self, obs, update: obs, ppo)
    ppo._normalize_critic_obs = MethodType(lambda self, obs, update: obs, ppo)
    ppo._normalize_teacher_actor_obs = MethodType(lambda self, obs, normalizers=None: obs, ppo)
    ppo._maybe_capture_fixed_bc_eval_samples = MethodType(lambda self, **kwargs: None, ppo)
    initial_obs = {
        "actor_obs": torch.tensor([[1.0], [2.0]]),
        "critic_obs": torch.tensor([[1.5], [2.5]]),
        "student_depth": torch.tensor([[3.0], [4.0]]),
        "teacher_obs": torch.tensor([[5.0], [6.0]]),
        "teacher_depth": torch.tensor([[10.0], [20.0]]),
    }
    ppo.env = _StepEnv(initial_obs)
    ppo.storage = _CaptureStorage()

    next_obs = ppo._rollout_step(initial_obs)

    assert torch.equal(
        ppo.storage.transition["teacher_actions"],
        initial_obs["teacher_obs"] + initial_obs["teacher_depth"],
    )
    assert torch.equal(
        ppo.storage.transition[ppo.actor_perception_key],
        initial_obs[ppo.actor_perception_key],
    )
    assert torch.equal(next_obs["teacher_obs"], initial_obs["teacher_obs"] + 100.0)


def test_supervised_only_rollout_is_independent_of_critic_observations_and_forward():
    ppo = _student_step_stub()
    ppo._supervised_dagger_only = True
    ppo.config.value_loss_coef = 0.0
    ppo.config.num_steps_per_env = 1
    ppo.config.gamma = 0.99
    ppo.config.lam = 0.95
    ppo.algo_timing = SimpleNamespace(enabled=False)
    ppo.log_dir = None
    ppo.algo_obs_dim_dict = {"actor_obs": 1, "critic_obs": 1, "student_depth": 1}
    ppo.teacher_obs_keys = ["teacher_obs"]
    ppo.teacher_actor = _TeacherActor()
    ppo.use_multi_teacher = False
    ppo.teacher_use_stochastic_actions = False
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.0
    ppo.dagger_ignore_episode_initial_steps = 0
    ppo._normalize_actor_obs = MethodType(lambda self, obs, update: obs, ppo)
    ppo._normalize_critic_obs = MethodType(
        lambda self, obs, update: pytest.fail("frozen critic observations were normalized"),
        ppo,
    )
    ppo._normalize_teacher_actor_obs = MethodType(lambda self, obs, normalizers=None: obs, ppo)
    ppo._maybe_capture_fixed_bc_eval_samples = MethodType(lambda self, **kwargs: None, ppo)
    ppo.critic.evaluate = lambda _state: pytest.fail("frozen critic was evaluated")
    ppo.critic.reset = lambda _dones: pytest.fail("frozen critic was reset")
    initial_obs = {
        "actor_obs": torch.tensor([[1.0], [2.0]]),
        "student_depth": torch.tensor([[3.0], [4.0]]),
        "teacher_obs": torch.tensor([[5.0], [6.0]]),
        "teacher_depth": torch.tensor([[10.0], [20.0]]),
    }
    ppo.env = _StepEnv(initial_obs)
    ppo.storage = _CaptureStorage()

    next_obs = ppo._rollout_step(initial_obs)

    assert "critic_obs" not in initial_obs
    assert "critic_obs" not in next_obs
    assert torch.count_nonzero(ppo.storage.transition["critic_obs"]) == 0
    assert torch.count_nonzero(ppo.storage.transition["critic_obs_raw"]) == 0
    assert torch.count_nonzero(ppo.storage.transition["values"]) == 0
    assert torch.count_nonzero(ppo.storage.computed["returns"]) == 0
    assert torch.count_nonzero(ppo.storage.computed["advantages"]) == 0
    assert torch.equal(
        ppo.storage.transition["teacher_actions"],
        initial_obs["teacher_obs"] + initial_obs["teacher_depth"],
    )


def test_full_rollout_writes_strict_storage_schema_and_derived_fields():
    ppo = _student_step_stub()
    ppo.config.num_steps_per_env = 1
    ppo.config.gamma = 0.99
    ppo.config.lam = 0.95
    ppo.algo_timing = SimpleNamespace(enabled=False)
    ppo.log_dir = None
    ppo.teacher_obs_keys = ["teacher_obs"]
    ppo.teacher_actor = _TeacherActor()
    ppo.use_multi_teacher = False
    ppo.teacher_use_stochastic_actions = False
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.0
    ppo.dagger_ignore_episode_initial_steps = 0
    ppo._normalize_actor_obs = MethodType(lambda self, obs, update: obs, ppo)
    ppo._normalize_critic_obs = MethodType(lambda self, obs, update: obs, ppo)
    ppo._normalize_teacher_actor_obs = MethodType(lambda self, obs, normalizers=None: obs, ppo)
    ppo._maybe_capture_fixed_bc_eval_samples = MethodType(lambda self, **kwargs: None, ppo)
    initial_obs = {
        "actor_obs": torch.tensor([[1.0], [2.0]]),
        "critic_obs": torch.tensor([[1.5], [2.5]]),
        "student_depth": torch.tensor([[3.0], [4.0]]),
        "teacher_obs": torch.tensor([[5.0], [6.0]]),
        "teacher_depth": torch.tensor([[10.0], [20.0]]),
    }
    ppo.env = _StepEnv(initial_obs)
    ppo.storage = RolloutStorage(num_envs=2, num_transitions_per_env=1)
    for key, shape, dtype, required_on_add in (
        ("actor_obs", (1,), torch.float32, True),
        ("critic_obs", (1,), torch.float32, True),
        ("actions", (1,), torch.float32, True),
        ("values", (1,), torch.float32, True),
        ("actions_log_prob", (1,), torch.float32, True),
        ("action_mean", (1,), torch.float32, True),
        ("action_sigma", (1,), torch.float32, True),
        ("rewards", (1,), torch.float32, True),
        ("dones", (1,), torch.bool, True),
        ("teacher_actions", (1,), torch.float32, True),
        ("student_depth", (1,), torch.float32, True),
        ("returns", (1,), torch.float32, False),
        ("advantages", (1,), torch.float32, False),
    ):
        ppo.storage.register(
            key,
            shape=shape,
            dtype=dtype,
            required_on_add=required_on_add,
        )

    ppo._rollout_step(initial_obs)

    assert ppo.storage.step == 1
    assert ppo.storage["dones"].dtype == torch.bool
    assert ppo.storage["returns"].shape == (1, 2, 1)
    assert ppo.storage["advantages"].shape == (1, 2, 1)
    # Generation proves both derived fields were marked complete.
    assert len(list(ppo.storage.mini_batch_generator(1, 1))) == 1


def test_full_ppo_rollout_never_calls_or_resets_teacher() -> None:
    ppo = _student_step_stub()
    ppo.ppo_coeff = 1.0
    ppo.config.num_steps_per_env = 1
    ppo.config.gamma = 0.99
    ppo.config.lam = 0.95
    ppo.algo_timing = SimpleNamespace(enabled=False)
    ppo.log_dir = None
    ppo.teacher_obs_keys = ["teacher_obs"]
    teacher_actor = SimpleNamespace(
        perception_input_name="",
        act=pytest.fail,
        act_inference=pytest.fail,
        reset=pytest.fail,
    )
    ppo.teacher_actor = teacher_actor
    ppo.use_multi_teacher = False
    ppo.teacher_use_stochastic_actions = False
    ppo.take_teacher_actions = False
    ppo.teacher_action_mix_ratio = 0.0
    ppo.dagger_ignore_episode_initial_steps = 0
    ppo.fixed_bc_eval_num_samples = 0
    ppo._fixed_bc_eval_ready = True
    ppo._normalize_actor_obs = MethodType(lambda self, obs, update: obs, ppo)
    ppo._normalize_critic_obs = MethodType(lambda self, obs, update: obs, ppo)
    initial_obs = {
        "actor_obs": torch.tensor([[1.0], [2.0]]),
        "critic_obs": torch.tensor([[1.5], [2.5]]),
        "student_depth": torch.tensor([[3.0], [4.0]]),
        "teacher_obs": torch.tensor([[5.0], [6.0]]),
    }
    ppo.env = _StepEnv(initial_obs)
    ppo.storage = _CaptureStorage()

    ppo._rollout_step(initial_obs)

    assert "teacher_actions" in ppo.storage.transition
    assert torch.count_nonzero(ppo.storage.transition["teacher_actions"]) == 0


def test_rollout_rejects_non_finite_teacher_before_env_step():
    ppo = _student_step_stub()
    ppo.config.num_steps_per_env = 1
    ppo.config.gamma = 0.99
    ppo.config.lam = 0.95
    ppo.algo_timing = SimpleNamespace(enabled=False)
    ppo.log_dir = None
    ppo.teacher_obs_keys = ["teacher_obs"]
    ppo.teacher_actor = _NonFiniteTeacherActor()
    ppo.use_multi_teacher = False
    ppo.teacher_use_stochastic_actions = False
    ppo.take_teacher_actions = True
    ppo.teacher_action_mix_ratio = 0.0
    ppo.dagger_ignore_episode_initial_steps = 0
    ppo._normalize_actor_obs = MethodType(lambda self, obs, update: obs, ppo)
    ppo._normalize_critic_obs = MethodType(lambda self, obs, update: obs, ppo)
    ppo._normalize_teacher_actor_obs = MethodType(lambda self, obs, normalizers=None: obs, ppo)
    ppo._maybe_capture_fixed_bc_eval_samples = MethodType(lambda self, **kwargs: None, ppo)
    initial_obs = {
        "actor_obs": torch.tensor([[1.0], [2.0]]),
        "critic_obs": torch.tensor([[1.5], [2.5]]),
        "student_depth": torch.tensor([[3.0], [4.0]]),
        "teacher_obs": torch.tensor([[5.0], [6.0]]),
        "teacher_depth": torch.tensor([[10.0], [20.0]]),
    }
    ppo.env = _StepEnv(initial_obs)
    ppo.storage = _CaptureStorage()

    with pytest.raises(FloatingPointError, match=r"teacher_actions.*actions_to_step"):
        ppo._rollout_step(initial_obs)

    assert ppo.env.actions_seen is None
    assert ppo.storage.transition == {}


def test_dagger_initial_step_mask_tracks_real_steps_not_randomized_episode_length() -> None:
    ppo = object.__new__(PPO)
    ppo.dagger_ignore_episode_initial_steps = 2
    # The environment's timeout-staggering counter can start arbitrarily high;
    # it must not make a freshly reset sample eligible for BC.
    ppo.env = SimpleNamespace(episode_length_buf=torch.tensor([91, 37]))

    initial_mask = ppo._dagger_episode_age_mask(2, torch.device("cpu"))
    ppo._advance_dagger_episode_age(torch.tensor([False, False]))
    second_mask = ppo._dagger_episode_age_mask(2, torch.device("cpu"))
    ppo._advance_dagger_episode_age(torch.tensor([False, True]))
    third_mask = ppo._dagger_episode_age_mask(2, torch.device("cpu"))

    assert initial_mask.view(-1).tolist() == [False, False]
    assert second_mask.view(-1).tolist() == [False, False]
    assert third_mask.view(-1).tolist() == [True, False]
