from __future__ import annotations

import copy
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from pydantic import ValidationError
from torch import nn
from torch.distributions import Normal

from holosoma.agents.ppo.ppo import PPO, _DaggerReplayBuffer
from holosoma.config_types.algo import DistillationConfig


_FIXED_DIGEST = "a" * 64


def _buffer(*, rank: int = 0, capacity: int = 4) -> _DaggerReplayBuffer:
    return _DaggerReplayBuffer(
        capacity=capacity,
        actor_obs_dim=2,
        actor_perception_dim=1,
        action_dim=1,
        base_seed=17,
        rank=rank,
    )


def _rows(start: int, count: int) -> dict[str, torch.Tensor]:
    values = torch.arange(start, start + count, dtype=torch.float32)
    return {
        "actor_obs_raw": torch.stack((values, values + 0.25), dim=1),
        "actor_perception": values[:, None] + 0.5,
        "teacher_actions": values[:, None] + 1.0,
        "mask": torch.ones((count, 1), dtype=torch.bool),
    }


def _bind_and_insert(
    buffer: _DaggerReplayBuffer,
    *,
    start: int,
    count: int,
    iteration: int = 3,
) -> None:
    buffer.bind_fixed_dataset(iteration=iteration, global_digest=_FIXED_DIGEST)
    buffer.insert(**_rows(start, count))


def test_typed_replay_config_defaults_off_and_rejects_coercion_or_bad_bounds():
    config = DistillationConfig()
    assert config.dagger_replay_enabled is False
    assert config.dagger_replay_capacity == 512
    assert config.dagger_replay_batch_size == 512
    assert config.dagger_replay_fraction == 0.5

    for field, value in (
        ("dagger_replay_enabled", "true"),
        ("dagger_replay_capacity", True),
        ("dagger_replay_capacity", 0),
        ("dagger_replay_batch_size", "8"),
        ("dagger_replay_batch_size", 0),
        ("dagger_replay_fraction", 0.0),
        ("dagger_replay_fraction", 1.0),
        ("dagger_replay_fraction", float("nan")),
        ("dagger_replay_seed", -1),
    ):
        with pytest.raises(ValidationError, match=field):
            DistillationConfig(**{field: value})


def test_replay_setup_rejects_any_future_operational_ppo_before_teacher_load():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(
        distill=DistillationConfig(
            enabled=True,
            mode="dagger",
            policy_to_clone="teacher.pt",
            dagger_replay_enabled=True,
            ppo_start_epoch=0,
            dagger_end_epoch=10,
            ppo_start_coeff=0.0,
            ppo_target_coeff=0.1,
        )
    )

    with pytest.raises(ValueError, match="operational PPO.*exactly zero"):
        ppo._setup_distillation()


def test_buffer_refuses_collection_before_fixed_gate_boundary():
    buffer = _buffer()

    with pytest.raises(RuntimeError, match="fixed-BC boundary"):
        buffer.insert(**_rows(0, 1))


def test_reservoir_and_sampling_are_rank_local_deterministic_and_rng_isolated():
    first = _buffer()
    second = _buffer()
    torch_rng_before = torch.get_rng_state().clone()

    _bind_and_insert(first, start=0, count=12)
    _bind_and_insert(second, start=0, count=12)
    first_sample = first.sample(9)
    second_sample = second.sample(9)

    assert torch.equal(torch.get_rng_state(), torch_rng_before)
    assert first.state_dict()["sha256"] == second.state_dict()["sha256"]
    for key in first_sample:
        assert torch.equal(first_sample[key], second_sample[key])

    other_rank = _buffer(rank=1)
    _bind_and_insert(other_rank, start=0, count=12)
    assert other_rank.state_dict()["effective_seed"] != first.state_dict()["effective_seed"]
    assert other_rank.state_dict()["sha256"] != first.state_dict()["sha256"]


def test_buffer_full_resume_restores_exact_future_insert_and_sample_sequence():
    uninterrupted = _buffer(capacity=5)
    _bind_and_insert(uninterrupted, start=0, count=9)
    uninterrupted.sample(7)
    checkpoint = uninterrupted.state_dict()

    resumed = _buffer(capacity=5)
    resumed.load_state_dict(checkpoint)
    uninterrupted.insert(**_rows(20, 8))
    resumed.insert(**_rows(20, 8))
    expected = uninterrupted.sample(13)
    actual = resumed.sample(13)

    for key in expected:
        assert torch.equal(actual[key], expected[key])
    assert resumed.state_dict()["sha256"] == uninterrupted.state_dict()["sha256"]


@pytest.mark.parametrize("corruption", ["digest", "finite", "schema", "mask"])
def test_buffer_checkpoint_validation_fails_closed_and_is_atomic(corruption: str):
    source = _buffer()
    _bind_and_insert(source, start=0, count=3)
    state = source.state_dict()
    target = _buffer()
    before = target.state_dict()

    corrupted = copy.deepcopy(state)
    if corruption == "digest":
        corrupted["sha256"] = "0" * 64
    elif corruption == "finite":
        corrupted["actor_obs_raw"][0, 0] = float("nan")
        corrupted["sha256"] = _DaggerReplayBuffer._state_sha256(
            {key: value for key, value in corrupted.items() if key != "sha256"}
        )
    elif corruption == "schema":
        corrupted["unexpected"] = 1
    else:
        corrupted["mask"][0] = False
        corrupted["sha256"] = _DaggerReplayBuffer._state_sha256(
            {key: value for key, value in corrupted.items() if key != "sha256"}
        )

    with pytest.raises(ValueError):
        target.load_state_dict(corrupted)

    assert target.state_dict()["sha256"] == before["sha256"]


class _LinearActor(nn.Module):
    supports_flow_matching = False

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
        self.distribution = None

    def update_distribution_from_policy_state(self, policy_state):
        mean = self.linear(policy_state["actor_obs"])
        self.distribution = Normal(mean, torch.ones_like(mean))

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def act_inference(self, policy_state):
        return self.linear(policy_state["actor_obs"])


def _replay_loss_ppo() -> PPO:
    ppo = object.__new__(PPO)
    ppo.device = "cpu"
    ppo.dagger_replay_enabled = True
    ppo.dagger_replay_fraction = 0.5
    ppo.dagger_ignore_zero_teacher_actions = False
    ppo.clip_teacher_actions = False
    ppo.actor_perception_key = ""
    ppo.actor = _LinearActor()
    ppo.distill_loss_fn = F.mse_loss
    ppo._dagger_replay_buffer = _DaggerReplayBuffer(
        capacity=4,
        actor_obs_dim=2,
        actor_perception_dim=0,
        action_dim=1,
        base_seed=1,
        rank=0,
    )
    ppo._normalize_actor_obs = MethodType(
        lambda self, obs, *, update: obs,
        ppo,
    )
    return ppo


def _replay_envelope(
    actor_obs: torch.Tensor,
    teacher_actions: torch.Tensor,
    mask: torch.Tensor,
    *,
    denominator: float,
    presence: bool = True,
):
    return {
        "_dagger_replay": {
            "batch": {
                "actor_obs_raw": actor_obs.to(torch.float32),
                "actor_perception": torch.empty((actor_obs.shape[0], 0)),
                "teacher_actions": teacher_actions.to(torch.float32),
                "mask": mask.to(torch.bool),
            },
            "denominator": torch.tensor(denominator, dtype=torch.float32),
            "has_valid_samples": presence,
        }
    }


def test_replay_loss_is_independent_action_bc_and_handles_empty_local_rank():
    ppo = _replay_loss_ppo()
    minibatch = _replay_envelope(
        torch.tensor([[1.0, 0.0], [3.0, 0.0]]),
        torch.zeros((2, 1)),
        torch.ones((2, 1), dtype=torch.bool),
        denominator=2.0,
    )

    loss, presence, active = ppo._compute_dagger_replay_bc_loss(
        minibatch,
        reference=torch.tensor(0.0),
    )

    assert loss.item() == pytest.approx((1.0**2 + 3.0**2) / 2.0)
    assert presence is True
    assert active is True
    # The replay schema contains no actions_log_prob, advantages, or old policy
    # distribution; the helper therefore cannot construct a PPO ratio.
    assert set(minibatch["_dagger_replay"]["batch"]) == {
        "actor_obs_raw",
        "actor_perception",
        "teacher_actions",
        "mask",
    }

    empty = _replay_envelope(
        torch.empty((0, 2)),
        torch.empty((0, 1)),
        torch.empty((0, 1), dtype=torch.bool),
        denominator=0.5,
        presence=True,
    )
    empty_loss, empty_presence, empty_active = ppo._compute_dagger_replay_bc_loss(
        empty,
        reference=torch.tensor(0.0),
    )
    assert empty_loss.item() == 0.0
    assert empty_presence is True
    assert empty_active is True


def test_pure_bc_actor_loss_mixes_current_and_replay_without_ratio():
    ppo = _replay_loss_ppo()
    ppo.current_learning_iteration = 1
    ppo.gpu_global_rank = 0
    ppo.gpu_world_size = 1
    ppo.is_multi_gpu = False
    ppo.use_time_gru = False
    ppo.use_symmetry = False
    ppo.critic_perception_key = ""
    ppo.distill_mode = "dagger"
    ppo.distill_enabled = True
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_coeff = 0.0
    ppo.dagger_loss_coef = 1.0
    ppo.bc_loss_coef = 1.0
    ppo.dagger_match_std = False
    ppo.use_multi_teacher = False
    ppo._supervised_dagger_only = False
    ppo.critic = nn.Linear(2, 1, bias=False)
    ppo.config = SimpleNamespace(
        value_loss_coef=0.0,
        symmetry_critic_coef=0.0,
        symmetry_actor_coef=0.0,
        entropy_coef=0.0,
        clip_param=0.2,
    )
    actor_obs = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    old_mean = actor_obs[:, :1]
    actions = old_mean.clone()
    old_log_prob = Normal(old_mean, torch.ones_like(old_mean)).log_prob(actions).sum(
        dim=-1, keepdim=True
    )
    minibatch = {
        "actor_obs": actor_obs,
        "critic_obs": actor_obs,
        "actions": actions,
        "values": torch.zeros((2, 1)),
        "advantages": torch.ones((2, 1)),
        "returns": torch.zeros((2, 1)),
        "actions_log_prob": old_log_prob,
        "action_mean": old_mean,
        "action_sigma": torch.ones_like(old_mean),
        "teacher_actions": torch.zeros((2, 1)),
        "_dagger_bc_denominator": torch.tensor(2.0),
        "_dagger_bc_has_valid_samples": True,
        **_replay_envelope(
            torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
            torch.zeros((2, 1)),
            torch.ones((2, 1), dtype=torch.bool),
            denominator=2.0,
        ),
    }

    losses = ppo._compute_ppo_loss(minibatch)

    assert losses["current_bc_loss"].item() == pytest.approx(5.0)
    assert losses["replay_bc_loss"].item() == pytest.approx(10.0)
    assert losses["bc_loss"].item() == pytest.approx(7.5)
    assert losses["actor_loss"].item() == pytest.approx(7.5)
    assert losses["surrogate_loss"].item() == 0.0
    assert losses["ppo_coeff"] == 0.0


def test_supervised_actor_only_loss_mixes_replay_and_backpropagates_no_ppo_signal():
    ppo = _replay_loss_ppo()
    ppo.current_learning_iteration = 1
    ppo.gpu_global_rank = 0
    ppo.gpu_world_size = 1
    ppo.is_multi_gpu = False
    ppo.use_time_gru = False
    ppo.use_symmetry = False
    ppo.critic_perception_key = ""
    ppo.distill_mode = "dagger"
    ppo.distill_enabled = True
    ppo.dagger_enabled = True
    ppo.use_ppo_dagger_schedule = True
    ppo.ppo_coeff = 0.0
    ppo.dagger_loss_coef = 1.0
    ppo.bc_loss_coef = 1.0
    ppo.dagger_match_std = False
    ppo.use_multi_teacher = False
    ppo._supervised_dagger_only = True
    ppo._supervised_actor_only_step = True
    ppo._supervised_actor_microbatch_size = 0
    ppo._stream_supervised_actor_backward = False
    ppo.critic = nn.Linear(2, 1, bias=False)
    ppo.config = SimpleNamespace(
        value_loss_coef=1.0,
        symmetry_critic_coef=0.0,
    )
    actor_obs = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    minibatch = {
        "actor_obs": actor_obs,
        "critic_obs": actor_obs,
        "actions": torch.zeros((2, 1)),
        "values": torch.zeros((2, 1)),
        "advantages": torch.ones((2, 1)),
        "returns": torch.zeros((2, 1)),
        # A ratio path would turn this into a non-finite loss.  The supervised
        # actor-only branch must not read rollout likelihoods at all.
        "actions_log_prob": torch.full((2, 1), float("nan")),
        "action_mean": torch.zeros((2, 1)),
        "action_sigma": torch.ones((2, 1)),
        "teacher_actions": torch.zeros((2, 1)),
        "_dagger_bc_denominator": torch.tensor(2.0),
        "_dagger_bc_has_valid_samples": True,
        **_replay_envelope(
            torch.tensor([[2.0, 0.0], [4.0, 0.0]]),
            torch.zeros((2, 1)),
            torch.ones((2, 1), dtype=torch.bool),
            denominator=2.0,
        ),
    }

    losses = ppo._compute_ppo_loss(minibatch)
    losses["actor_loss"].backward()

    assert losses["current_bc_loss"].item() == pytest.approx(5.0)
    assert losses["replay_bc_loss"].item() == pytest.approx(10.0)
    assert losses["actor_loss"].item() == pytest.approx(7.5)
    assert torch.equal(
        ppo.actor.linear.weight.grad,
        torch.tensor([[15.0, 0.0]]),
    )
    assert ppo.actor.distribution is None
    assert ppo.critic.weight.grad is None
    assert losses["surrogate_loss"] == 0.0
    assert losses["critic_loss"] == 0.0


def test_distributed_weighting_formula_matches_global_weighted_replay_mean():
    # rank0: two squared errors [1, 9], weight .5
    # rank1: one squared error [4], weight 1.5
    world_size = 2.0
    global_weighted_count = 0.5 * 2.0 + 1.5 * 1.0
    denominator = global_weighted_count / world_size
    rank0_local_loss = (1.0 + 9.0) / denominator
    rank1_local_loss = 4.0 / denominator
    reduced_loss = (0.5 * rank0_local_loss + 1.5 * rank1_local_loss) / world_size
    expected = (0.5 * (1.0 + 9.0) + 1.5 * 4.0) / global_weighted_count

    assert reduced_loss == pytest.approx(expected)


def test_single_rank_replay_denominator_ignores_distributed_loss_weight():
    ppo = _replay_loss_ppo()
    ppo.is_multi_gpu = False
    ppo.gpu_world_size = 1
    ppo.dagger_replay_batch_size = 2
    ppo._get_distributed_loss_weight = MethodType(lambda self: 2.0, ppo)
    ppo._all_ranks_fixed_bc_ready_before_rollout = MethodType(
        lambda self: True,
        ppo,
    )
    ppo._insert_current_rollout_into_dagger_replay = MethodType(
        lambda self: None,
        ppo,
    )
    ppo._synchronize_training_phase_error = MethodType(
        lambda self, error, **kwargs: (_ for _ in ()).throw(error)
        if error is not None
        else None,
        ppo,
    )
    buffer = ppo._dagger_replay_buffer
    assert buffer is not None
    buffer.bind_fixed_dataset(iteration=0, global_digest=_FIXED_DIGEST)
    buffer.insert(
        actor_obs_raw=torch.tensor([[1.0, 0.0], [2.0, 0.0]]),
        actor_perception=None,
        teacher_actions=torch.ones((2, 1)),
        mask=torch.ones((2, 1), dtype=torch.bool),
    )

    plan = ppo._prepare_dagger_replay_update_plan(num_updates=1)

    assert plan is not None
    assert plan[0]["denominator"].item() == pytest.approx(2.0)


def test_full_resume_plan_restores_replay_but_policy_init_reset_does_not_inherit():
    ppo = _replay_loss_ppo()
    ppo.gpu_world_size = 1
    ppo.gpu_global_rank = 0
    ppo.dagger_replay_capacity = 4
    ppo.dagger_replay_seed = 1
    ppo.algo_obs_dim_dict = {"actor_obs": 2}
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.num_act = 1
    source = ppo._dagger_replay_buffer
    assert source is not None
    source.bind_fixed_dataset(iteration=2, global_digest=_FIXED_DIGEST)
    source_rows = _rows(0, 3)
    source.insert(
        actor_obs_raw=source_rows["actor_obs_raw"],
        actor_perception=None,
        teacher_actions=source_rows["teacher_actions"],
        mask=source_rows["mask"],
    )
    source_state = source.state_dict()
    ppo._fixed_bc_guard_checkpoint_dataset_digests = MethodType(
        lambda self, states: ({"0": "b" * 64}, _FIXED_DIGEST),
        ppo,
    )
    checkpoint = {
        "dagger_replay_by_rank": {"0": source_state},
        "fixed_bc_eval_by_rank": {"0": {}},
    }

    plan = ppo._prepare_dagger_replay_checkpoint_state(
        checkpoint,
        next_iteration=3,
    )
    ppo._commit_dagger_replay_checkpoint_plan(plan)
    assert ppo._dagger_replay_buffer is not None
    assert ppo._dagger_replay_buffer.state_dict()["sha256"] == source_state["sha256"]

    ppo._reset_dagger_replay_state()
    assert ppo._dagger_replay_buffer is not None
    assert ppo._dagger_replay_buffer.size == 0
    assert ppo._dagger_replay_buffer.seen_valid_count == 0
    assert ppo._dagger_replay_buffer.capture_start_iteration is None


def test_unbound_empty_replay_resume_stays_unverified_then_can_bind_and_insert():
    ppo = _replay_loss_ppo()
    ppo.gpu_world_size = 1
    ppo.gpu_global_rank = 0
    ppo.dagger_replay_capacity = 4
    ppo.dagger_replay_seed = 1
    ppo.algo_obs_dim_dict = {"actor_obs": 2}
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.num_act = 1
    source = ppo._dagger_replay_buffer
    assert source is not None
    source_state = source.state_dict()
    assert source_state["size"] == 0
    assert source_state["capture_start_iteration"] is None
    assert source_state["fixed_bc_global_dataset_digest"] is None

    plan = ppo._prepare_dagger_replay_checkpoint_state(
        {"dagger_replay_by_rank": {"0": source_state}},
        next_iteration=3,
    )
    ppo._commit_dagger_replay_checkpoint_plan(plan)

    restored = ppo._dagger_replay_buffer
    assert restored is not None
    assert restored.state_dict()["sha256"] == source_state["sha256"]
    assert ppo._dagger_replay_fixed_boundary_verified_runtime is False

    restored.bind_fixed_dataset(iteration=3, global_digest=_FIXED_DIGEST)
    rows = _rows(0, 2)
    inserted = restored.insert(
        actor_obs_raw=rows["actor_obs_raw"],
        actor_perception=None,
        teacher_actions=rows["teacher_actions"],
        mask=rows["mask"],
    )
    assert inserted == 2
    assert restored.size == 2


def test_actor_policy_init_hook_discards_live_and_checkpoint_replay_state():
    ppo = _replay_loss_ppo()
    ppo.gpu_world_size = 1
    ppo.gpu_global_rank = 0
    ppo.is_multi_gpu = False
    ppo.current_learning_iteration = 0
    ppo.dagger_replay_capacity = 4
    ppo.dagger_replay_seed = 1
    ppo.algo_obs_dim_dict = {"actor_obs": 2}
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.num_act = 1
    ppo.config = SimpleNamespace(normalize_actor_obs=False)
    old_buffer = ppo._dagger_replay_buffer
    assert old_buffer is not None
    old_buffer.bind_fixed_dataset(iteration=0, global_digest=_FIXED_DIGEST)
    rows = _rows(0, 2)
    old_buffer.insert(
        actor_obs_raw=rows["actor_obs_raw"],
        actor_perception=None,
        teacher_actions=rows["teacher_actions"],
        mask=rows["mask"],
    )
    actor_state = {
        key: value.detach().clone() for key, value in ppo.actor.state_dict().items()
    }
    loaded = {
        "actor_model_state_dict": actor_state,
        "dagger_replay_by_rank": {"0": old_buffer.state_dict()},
        "iter": 7,
        "infos": None,
    }
    ppo._prepare_policy_init_checkpoint = MethodType(
        lambda self, path, **kwargs: (
            loaded,
            actor_state,
            False,
            "c" * 64,
            None,
            None,
        ),
        ppo,
    )
    ppo._synchronize_distributed_operation_error = MethodType(
        lambda self, error, **kwargs: (_ for _ in ()).throw(error)
        if error is not None
        else None,
        ppo,
    )
    ppo._sanitize_actor_std = MethodType(lambda self: None, ppo)
    ppo._assert_model_parameters_finite = MethodType(lambda self, **kwargs: None, ppo)

    ppo._load_policy_init_impl("actor-init.pt")

    assert ppo._dagger_replay_buffer is not None
    assert ppo._dagger_replay_buffer is not old_buffer
    assert ppo._dagger_replay_buffer.size == 0
    assert ppo._dagger_replay_buffer.capture_start_iteration is None
