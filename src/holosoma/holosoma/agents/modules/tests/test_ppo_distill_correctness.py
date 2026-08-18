from __future__ import annotations

import multiprocessing
import os
import queue
import random
import time
import traceback
from datetime import timedelta
from types import MethodType, SimpleNamespace
from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest
import torch
from torch import nn

from holosoma.agents.ppo.ppo import PPO
from holosoma.utils.normalization import EmpiricalNormalization
from holosoma.utils.rng_checkpoint import (
    capture_rng_checkpoint_state,
    restore_rng_checkpoint_state,
)
from holosoma.utils.training_provenance import (
    ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV,
)


@pytest.fixture(autouse=True)
def _explicit_legacy_teacher_hatch_for_unprovenanced_unit_fixtures(monkeypatch):
    monkeypatch.setenv(ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV, "1")


def test_std_matching_loss_is_invariant_to_repeated_action_dimensions():
    student = torch.tensor([[0.1, 0.5]])
    teacher = torch.tensor([[0.3, 0.9]])

    small = PPO._std_matching_loss_per_sample(student, teacher)
    repeated = PPO._std_matching_loss_per_sample(student.repeat(1, 10), teacher.repeat(1, 10))

    assert repeated.item() == pytest.approx(small.item())


def test_teacher_normalization_uses_checkpoint_setting_when_available():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(normalize_actor_obs=True)
    checkpoint = {
        "experiment_config": {
            "algo": {"config": {"normalize_actor_obs": False}},
        }
    }

    assert ppo._teacher_normalization_enabled(checkpoint) is False
    assert ppo._teacher_normalization_enabled({}) is True


def test_teacher_normalizer_state_maps_checkpoint_key_to_runtime_alias():
    ppo = object.__new__(PPO)
    source = EmpiricalNormalization(shape=(2,), device="cpu")
    target = EmpiricalNormalization(shape=(2,), device="cpu")
    with torch.no_grad():
        source._mean.copy_(torch.tensor([[1.0, 2.0]]))
        source._var.copy_(torch.tensor([[3.0, 4.0]]))
        source._std.copy_(torch.sqrt(source._var + source.eps))
        source.count.fill_(17)

    ppo._load_teacher_normalizer_states(
        {"actor_obs_teacher_compat": target},
        {"actor_obs": source.state_dict()},
        ["actor_obs"],
    )

    assert torch.equal(target._mean, source._mean)
    assert torch.equal(target._var, source._var)
    assert torch.equal(target._std, source._std)
    assert target.count.item() == 17


@pytest.mark.parametrize(
    ("runtime_order", "expected_means"),
    [
        (["a", "b"], {"a": 1.0, "b": 2.0}),
        (["b", "a"], {"b": 1.0, "a": 2.0}),
    ],
)
def test_teacher_normalizers_follow_declared_positional_input_mapping(
    runtime_order,
    expected_means,
):
    ppo = object.__new__(PPO)
    source_a = EmpiricalNormalization(shape=(1,), device="cpu")
    source_b = EmpiricalNormalization(shape=(1,), device="cpu")
    source_a._mean.fill_(1.0)
    source_b._mean.fill_(2.0)
    targets = {
        key: EmpiricalNormalization(shape=(1,), device="cpu")
        for key in runtime_order
    }

    ppo._load_teacher_normalizer_states(
        targets,
        {"a": source_a.state_dict(), "b": source_b.state_dict()},
        ["a", "b"],
        require_state=True,
    )

    for runtime_key, expected_mean in expected_means.items():
        assert targets[runtime_key]._mean.item() == pytest.approx(expected_mean)


def test_teacher_normalizer_alias_mapping_rejects_group_count_mismatch():
    ppo = object.__new__(PPO)
    target = EmpiricalNormalization(shape=(2,), device="cpu")

    with pytest.raises(ValueError, match="Cannot map teacher checkpoint normalizers"):
        ppo._load_teacher_normalizer_states(
            {"alias_a": target, "alias_b": target},
            {"actor_obs": target.state_dict()},
            ["actor_obs"],
        )


def test_teacher_normalizer_validation_is_atomic_across_aliases():
    ppo = object.__new__(PPO)
    first = EmpiricalNormalization(shape=(1,), device="cpu")
    second = EmpiricalNormalization(shape=(1,), device="cpu")
    source_first = EmpiricalNormalization(shape=(1,), device="cpu")
    source_second = EmpiricalNormalization(shape=(1,), device="cpu")
    source_first._mean.fill_(7.0)
    invalid_second = {
        key: value.detach().clone()
        for key, value in source_second.state_dict().items()
    }
    invalid_second["_mean"].fill_(float("nan"))
    first_before = {
        key: value.detach().clone() for key, value in first.state_dict().items()
    }

    with pytest.raises(ValueError, match="non-finite"):
        ppo._load_teacher_normalizer_states(
            {"runtime_a": first, "runtime_b": second},
            {
                "checkpoint_a": source_first.state_dict(),
                "checkpoint_b": invalid_second,
            },
            ["checkpoint_a", "checkpoint_b"],
            require_state=True,
        )

    for key, expected in first_before.items():
        assert torch.equal(first.state_dict()[key], expected)


def test_teacher_actor_rejects_invalid_std_before_parameter_mutation():
    ppo = object.__new__(PPO)
    ppo.gpu_global_rank = 0
    ppo.actor_obs_keys = ["actor_obs"]
    ppo.algo_obs_dim_dict = {"actor_obs": 1}
    ppo.algo_history_length_dict = {"actor_obs": 1}
    ppo.num_act = 1
    ppo.device = "cpu"
    ppo.strict_teacher_load = True
    ppo.config = SimpleNamespace(init_noise_std=0.1)
    ppo._extract_teacher_actor_config = MethodType(
        lambda self, state: SimpleNamespace(input_dim=["actor_obs"]),
        ppo,
    )
    ppo._build_teacher_actor_config = MethodType(
        lambda self, obs_keys, base_actor_cfg=None: SimpleNamespace(),
        ppo,
    )
    ppo._validate_teacher_checkpoint_runtime_config = MethodType(
        lambda self, state, **kwargs: None,
        ppo,
    )
    teacher_actor = nn.Linear(1, 1)
    teacher_actor.std = nn.Parameter(torch.tensor([0.1]))
    teacher_actor.min_noise_std = 0.01
    teacher_actor.min_mean_noise_std = None
    teacher_actor.max_noise_std = 0.5
    original_weight = teacher_actor.weight.detach().clone()
    actor_state = {
        key: value.detach().clone()
        for key, value in teacher_actor.state_dict().items()
    }
    actor_state["weight"].fill_(7.0)
    actor_state["std"].fill_(-3.0)
    checkpoint = {"actor_model_state_dict": actor_state}

    with (
        patch(
            "holosoma.agents.ppo.ppo.load_verified_torch_checkpoint",
            return_value=(checkpoint, "0" * 64),
        ),
        patch(
            "holosoma.agents.ppo.ppo.setup_ppo_actor_module",
            return_value=teacher_actor,
        ),
        pytest.raises(ValueError, match=r"teacher\.actor_model_state_dict\.std"),
    ):
        ppo._load_teacher_actor("teacher.pt")

    assert torch.equal(teacher_actor.weight, original_weight)


def test_teacher_load_requires_authenticated_current_provenance_by_default(monkeypatch):
    monkeypatch.delenv(ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV, raising=False)
    ppo = object.__new__(PPO)
    ppo.gpu_global_rank = 0

    with (
        patch("holosoma.agents.ppo.ppo.training_provenance_from_env", return_value=None),
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="requires finalized current training provenance"),
    ):
        ppo._load_teacher_actor("teacher.pt")

    load_mock.assert_not_called()


def test_teacher_legacy_identity_hatch_must_be_exact(monkeypatch):
    monkeypatch.setenv(ALLOW_LEGACY_UNVERIFIED_TEACHER_LOAD_ENV, "true")
    ppo = object.__new__(PPO)
    ppo.gpu_global_rank = 0

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="must be exactly 0 or 1"),
    ):
        ppo._load_teacher_actor("teacher.pt")

    load_mock.assert_not_called()


def test_malformed_attached_teacher_provenance_never_falls_back_or_uses_hatch():
    ppo = object.__new__(PPO)
    ppo.gpu_global_rank = 0
    ppo._training_provenance = "malformed"

    with (
        patch("holosoma.agents.ppo.ppo.training_provenance_from_env") as env_provenance,
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="must be a mapping when present"),
    ):
        ppo._load_teacher_actor("teacher.pt")

    env_provenance.assert_not_called()
    load_mock.assert_not_called()


def test_incomplete_teacher_provenance_cannot_claim_authentication_through_hatch():
    ppo = object.__new__(PPO)
    ppo.gpu_global_rank = 0
    ppo._training_provenance = {
        "teacher_enabled": True,
        "teacher_sha256": "a" * 64,
    }

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="unsupported training provenance version"),
    ):
        ppo._load_teacher_actor("teacher.pt")

    load_mock.assert_not_called()


def test_teacher_enabled_provenance_cannot_omit_authenticated_digest():
    ppo = object.__new__(PPO)
    ppo.gpu_global_rank = 0
    ppo._training_provenance = {"teacher_enabled": True}

    with (
        patch("holosoma.agents.ppo.ppo.load_verified_torch_checkpoint") as load_mock,
        pytest.raises(ValueError, match="no authenticated teacher_sha256"),
    ):
        ppo._load_teacher_actor("teacher.pt")

    load_mock.assert_not_called()


def test_teacher_resolution_construction_and_outcome_do_not_advance_training_rng():
    original_rng = capture_rng_checkpoint_state()
    try:
        ppo = object.__new__(PPO)

        def noisy_impl(self, ckpt_path, obs_keys=None):
            assert ckpt_path == "wandb://authenticated/teacher"
            assert obs_keys == ["teacher_obs"]
            random.random()
            np.random.random()
            torch.rand(1)
            return nn.Linear(1, 1), {}

        def noisy_outcome(self, local_error, *, operation):
            assert local_error is None
            assert operation == "Teacher checkpoint resolution/construction"
            random.random()
            np.random.random()
            torch.rand(1)

        ppo._load_teacher_actor_impl = MethodType(noisy_impl, ppo)
        ppo._synchronize_distributed_operation_error = MethodType(noisy_outcome, ppo)
        random.seed(2101)
        np.random.seed(2102)
        torch.manual_seed(2103)
        boundary = capture_rng_checkpoint_state()
        expected = (random.random(), float(np.random.random()), torch.rand(3))
        restore_rng_checkpoint_state(boundary)

        teacher, normalizers = ppo._load_teacher_actor(
            "wandb://authenticated/teacher",
            obs_keys=["teacher_obs"],
        )

        assert isinstance(teacher, nn.Linear)
        assert normalizers == {}
        assert random.random() == expected[0]
        assert float(np.random.random()) == expected[1]
        assert torch.equal(torch.rand(3), expected[2])
    finally:
        restore_rng_checkpoint_state(original_rng)


def test_teacher_std_projection_matches_mean_floor_and_hard_cap():
    ppo = object.__new__(PPO)
    ppo.config = SimpleNamespace(init_noise_std=0.1)
    actor = SimpleNamespace(
        std=torch.nn.Parameter(torch.tensor([1e-6, 0.8])),
        min_noise_std=None,
        min_mean_noise_std=0.8,
        max_noise_std=0.8,
    )

    projected = ppo._get_actor_std_for_loss(actor)

    assert projected.mean().item() >= 0.8 - 1e-6
    assert projected.max().item() <= 0.8 + 1e-6


class _PerceptionTeacher(nn.Module):
    perception_input_name = "teacher_depth"

    def __init__(self):
        super().__init__()
        self.last_policy_state = None

    def act_inference(self, policy_state):
        self.last_policy_state = policy_state
        return policy_state["actor_obs"] + policy_state[self.perception_input_name]


def test_legacy_teacher_labels_use_teacher_obs_groups_and_teacher_perception():
    ppo = object.__new__(PPO)
    ppo.actor_obs_keys = ["student_obs"]
    ppo.teacher_obs_keys = ["teacher_a", "teacher_b"]
    ppo.teacher_obs_dim = 2
    ppo.use_multi_teacher = False
    ppo.teacher_use_stochastic_actions = False
    ppo.teacher_actor = _PerceptionTeacher()
    ppo._normalize_teacher_actor_obs = MethodType(lambda self, obs, normalizers=None: obs, ppo)
    obs_dict = {
        "student_obs": torch.tensor([[99.0, 99.0]]),
        "teacher_a": torch.tensor([[1.0]]),
        "teacher_b": torch.tensor([[2.0]]),
        "teacher_depth": torch.tensor([[10.0, 20.0]]),
    }
    actor_obs_raw = obs_dict["student_obs"]

    teacher_obs_raw = ppo._build_teacher_obs_raw(obs_dict, actor_obs_raw)
    teacher_actions, _ = ppo._select_teacher_actions(teacher_obs_raw, obs_dict)

    assert torch.equal(teacher_obs_raw, torch.tensor([[1.0, 2.0]]))
    assert torch.equal(teacher_actions, torch.tensor([[11.0, 22.0]]))
    assert torch.equal(ppo.teacher_actor.last_policy_state["teacher_depth"], obs_dict["teacher_depth"])


def test_checkpoint_env_state_selects_current_global_rank_and_supports_legacy():
    ppo = object.__new__(PPO)
    ppo.gpu_global_rank = 1
    ppo.gpu_world_size = 2

    selected = ppo._select_checkpoint_env_state(
        {"env_state_by_rank": {"0": {"value": 10}, "1": {"value": 20}}}
    )

    assert selected == {"value": 20}
    assert ppo._select_checkpoint_env_state({"env_state": {"value": 7}}) == {"value": 7}


def test_collect_distributed_env_states_keeps_rank_mapping():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 1
    ppo._collect_env_state = MethodType(lambda self: {"value": torch.tensor(20)}, ppo)
    gloo_group = object()
    ppo._setup_gloo_barrier_group = MethodType(lambda self: gloo_group, ppo)

    def fake_all_gather(output, local_result, *, group):
        assert group is gloo_group
        output[0] = {
            "rank": 0,
            "error": None,
            "state": {"value": torch.tensor(10)},
        }
        output[1] = local_result

    with (
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=1),
        patch("holosoma.agents.ppo.ppo.torch.distributed.all_gather_object", side_effect=fake_all_gather),
    ):
        states = ppo._collect_distributed_env_states()

    assert states["0"]["value"].item() == 10
    assert states["1"]["value"].item() == 20
    assert states["1"]["value"].device.type == "cpu"


def test_collect_distributed_env_states_gathers_local_error_before_raise():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.gpu_global_rank = 0
    ppo._collect_env_state = MethodType(
        lambda self: (_ for _ in ()).throw(ValueError("corrupt local AS state")),
        ppo,
    )
    group = object()
    ppo._setup_gloo_barrier_group = MethodType(lambda self: group, ppo)
    gather_calls = []

    def fake_all_gather(output, local_result, *, group):
        gather_calls.append(local_result)
        output[0] = local_result
        output[1] = {"rank": 1, "error": None, "state": {}}

    with (
        patch("holosoma.agents.ppo.ppo.torch.distributed.is_initialized", return_value=True),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_world_size", return_value=2),
        patch("holosoma.agents.ppo.ppo.torch.distributed.get_rank", return_value=0),
        patch("holosoma.agents.ppo.ppo.torch.distributed.all_gather_object", side_effect=fake_all_gather),
        pytest.raises(RuntimeError, match=r"rank=0: ValueError: corrupt local AS state"),
    ):
        ppo._collect_distributed_env_states()

    assert len(gather_calls) == 1


def test_save_persists_ranked_env_state_and_rank_zero_legacy_alias():
    ppo = object.__new__(PPO)
    ppo.is_main_process = True
    ppo.current_learning_iteration = 12
    ppo.actor = nn.Linear(1, 1)
    ppo.critic = nn.Linear(1, 1)
    ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=0.1)
    ppo.critic_optimizer = torch.optim.SGD(ppo.critic.parameters(), lr=0.1)
    ppo.actor_obs_normalizers = {}
    ppo.critic_obs_normalizers = {}
    ppo.fixed_bc_eval_num_samples = 2
    ppo.dagger_enabled = True
    ppo._fixed_bc_eval_ready = True
    ppo._fixed_bc_eval_size = 2
    ppo._fixed_bc_eval_dataset = {
        "actor_obs_raw": torch.tensor([[1.0], [2.0]]),
        "teacher_actions": torch.tensor([[3.0], [4.0]]),
    }
    ranked_state = {"0": {"ema": torch.tensor([1.0])}, "1": {"ema": torch.tensor([2.0])}}
    ppo._collect_distributed_env_states = MethodType(lambda self: ranked_state, ppo)
    ppo._checkpoint_metadata = MethodType(lambda self, iteration=None: {}, ppo)
    captured = {}
    ppo.logging_helper = SimpleNamespace(
        save_checkpoint_artifact=lambda checkpoint, path: captured.update(
            checkpoint=checkpoint,
            path=path,
        )
    )

    ppo.save("model.pt")

    assert captured["checkpoint"]["env_state_by_rank"] is ranked_state
    assert torch.equal(captured["checkpoint"]["env_state"]["ema"], torch.tensor([1.0]))
    assert torch.equal(
        captured["checkpoint"]["fixed_bc_eval_by_rank"]["0"]["teacher_actions"],
        torch.tensor([[3.0], [4.0]]),
    )
    assert captured["checkpoint"]["fixed_bc_eval_by_rank"]["0"]["allocation_version"] == 1
    assert set(captured["checkpoint"]["rng_state_by_rank"]) == {"0"}
    assert captured["checkpoint"]["rng_state_by_rank"]["0"]["version"] == 1
    assert captured["path"] == "model.pt"


def test_rank_loss_weight_scales_gradient_payload_before_reduction():
    ppo = object.__new__(PPO)
    ppo.actor = nn.Linear(1, 1, bias=False)
    ppo.critic = nn.Linear(1, 1, bias=False)
    ppo.actor.weight.grad = torch.ones_like(ppo.actor.weight)
    ppo.env = SimpleNamespace(distributed_loss_weight=2.5)
    ppo.gpu_world_size = 1
    ppo.gpu_global_rank = 0
    ppo.current_learning_iteration = 0
    ppo._all_reduce_grad_payload = MethodType(lambda self, payload: "test", ppo)

    ppo._reduce_parameters(include_critic=False)

    assert ppo.actor.weight.grad.item() == pytest.approx(2.5)


def test_weighted_gradient_sum_is_divided_once_by_global_world_size():
    ppo = object.__new__(PPO)
    ppo.actor = nn.Linear(1, 1, bias=False)
    ppo.critic = nn.Linear(1, 1, bias=False)
    ppo.actor.weight.grad = torch.tensor([[2.0]])
    ppo.env = SimpleNamespace(distributed_loss_weight=1.5)
    ppo.gpu_world_size = 4
    ppo.gpu_global_rank = 0
    ppo.current_learning_iteration = 0

    def fake_hierarchical_sum(self, payload):
        # Local contribution is 2 * 1.5 = 3.  The other three ranks contribute
        # a total of 9; hierarchical reduction must produce the global SUM=12,
        # leaving _reduce_parameters to apply the sole world-size division.
        payload[0].add_(9.0)
        payload[-1].add_(3.0)
        return "hierarchical_cpu_leader"

    ppo._all_reduce_grad_payload = MethodType(fake_hierarchical_sum, ppo)

    ppo._reduce_parameters(include_critic=False)

    assert ppo.actor.weight.grad.item() == pytest.approx(3.0)


@pytest.mark.parametrize("device", [torch.device("cuda:2"), torch.device("cpu")])
def test_flat_grad_payload_synchronizes_cuda_packing_and_collective_completion(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
):
    """CUDA flat reduction is fenced without imposing CUDA calls on CPU payloads."""
    ppo = object.__new__(PPO)
    monkeypatch.delenv("HOLOSOMA_GLOO_GRAD_REDUCE", raising=False)
    monkeypatch.delenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", raising=False)
    payload = SimpleNamespace(device=device)
    events: list[tuple[str, object]] = []

    with (
        patch(
            "torch.cuda.synchronize",
            side_effect=lambda device: events.append(("synchronize", device)),
        ),
        patch(
            "torch.distributed.all_reduce",
            side_effect=lambda tensor, op: events.append(("all_reduce", tensor, op)),
        ),
    ):
        reduce_path = ppo._all_reduce_grad_payload(payload)

    assert reduce_path == "flat"
    expected = [("all_reduce", payload, torch.distributed.ReduceOp.SUM)]
    if device.type == "cuda":
        expected = [("synchronize", device), *expected, ("synchronize", device)]
    assert events == expected


def _hierarchical_ppo(*, rank: int = 0, local_rank: int = 0) -> PPO:
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    ppo.is_main_process = rank == 0
    ppo.gpu_world_size = 104
    ppo.gpu_global_rank = rank
    ppo.gpu_local_world_size = 8
    ppo.gpu_topology_local_rank = local_rank
    ppo._hierarchical_grad_reduce_ready = False
    ppo._hierarchical_grad_reduce_available = False
    ppo._hierarchical_grad_reduce_cpu_leader = False
    ppo._hierarchical_local_group = None
    ppo._hierarchical_local_barrier_group = None
    ppo._hierarchical_leader_group = None
    ppo._hierarchical_leader_gloo_group = None
    ppo._hierarchical_local_leader_rank = 0
    ppo._hierarchical_is_leader_rank = False
    return ppo


def test_rank_visible_topology_uses_preserved_local_rank_and_world_size(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_RANK", "6")
    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE", "8")

    assert PPO._topology_local_rank() == 6
    assert PPO._topology_local_world_size() == 8
    assert PPO._is_node_local_main_process() is False

    monkeypatch.setenv("HOLOSOMA_ORIGINAL_LOCAL_RANK", "0")
    assert PPO._is_node_local_main_process() is True


def test_hierarchical_subgroup_timeout_is_explicit_and_overridable(
    monkeypatch: pytest.MonkeyPatch,
):
    ppo = object.__new__(PPO)
    monkeypatch.delenv("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", raising=False)
    assert ppo._hierarchical_pg_timeout() == timedelta(seconds=300)

    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", "47")
    assert ppo._hierarchical_pg_timeout() == timedelta(seconds=47)


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "five"])
def test_hierarchical_subgroup_timeout_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
):
    ppo = object.__new__(PPO)
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC", value)

    with pytest.raises(ValueError, match="HOLOSOMA_HIERARCHICAL_PG_TIMEOUT_SEC"):
        ppo._hierarchical_pg_timeout()


@pytest.mark.parametrize(
    ("rank", "local_rank", "expected_node_leader", "is_leader"),
    [(0, 0, 0, True), (42, 2, 40, False)],
)
def test_hierarchical_cpu_leader_groups_never_create_cross_node_nccl(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    local_rank: int,
    expected_node_leader: int,
    is_leader: bool,
):
    ppo = _hierarchical_ppo(rank=rank, local_rank=local_rank)
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "1")
    group_calls: list[tuple[tuple[int, ...], str | None, timedelta | None]] = []

    def fake_new_group(*, ranks, backend=None, timeout=None):
        group = (tuple(ranks), backend, len(group_calls))
        group_calls.append((tuple(ranks), backend, timeout))
        return group

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.new_group", side_effect=fake_new_group),
    ):
        assert ppo._setup_hierarchical_grad_reduce_groups() is True

    expected_calls: list[tuple[tuple[int, ...], str | None, timedelta]] = []
    for node_idx in range(13):
        local_ranks = tuple(range(node_idx * 8, node_idx * 8 + 8))
        expected_calls.extend(
            (
                (local_ranks, "nccl", timedelta(seconds=300)),
                (local_ranks, "gloo", timedelta(seconds=300)),
            )
        )
    leader_ranks = tuple(range(0, 104, 8))
    expected_calls.append((leader_ranks, "gloo", timedelta(seconds=300)))

    assert group_calls == expected_calls
    assert ppo._hierarchical_leader_group is None
    assert ppo._hierarchical_leader_gloo_group is not None
    assert ppo._hierarchical_local_leader_rank == expected_node_leader
    assert ppo._hierarchical_is_leader_rank is is_leader


def test_hierarchical_gpu_gradient_leaders_also_create_gloo_control_group_when_requested(
    monkeypatch: pytest.MonkeyPatch,
):
    ppo = _hierarchical_ppo()
    ppo._hierarchical_small_collectives = True
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "0")
    group_calls: list[tuple[tuple[int, ...], str | None, timedelta | None]] = []

    def fake_new_group(*, ranks, backend=None, timeout=None):
        group = (tuple(ranks), backend, len(group_calls))
        group_calls.append((tuple(ranks), backend, timeout))
        return group

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.new_group", side_effect=fake_new_group),
    ):
        assert ppo._setup_hierarchical_grad_reduce_groups() is True

    leader_ranks = tuple(range(0, 104, 8))
    assert group_calls[-2:] == [
        (leader_ranks, "nccl", timedelta(seconds=300)),
        (leader_ranks, "gloo", timedelta(seconds=300)),
    ]
    assert ppo._hierarchical_leader_group is not None
    assert ppo._hierarchical_leader_gloo_group is not None


def _ready_hierarchical_small_ppo(*, leader: bool) -> PPO:
    ppo = _hierarchical_ppo(rank=0 if leader else 1, local_rank=0 if leader else 1)
    ppo._hierarchical_small_collectives = True
    ppo._hierarchical_grad_reduce_ready = True
    ppo._hierarchical_grad_reduce_available = True
    ppo._hierarchical_local_barrier_group = "local_gloo"
    ppo._hierarchical_leader_gloo_group = "leader_gloo"
    ppo._hierarchical_local_leader_rank = 0
    ppo._hierarchical_is_leader_rank = leader
    return ppo


@pytest.mark.parametrize(
    "op",
    [
        torch.distributed.ReduceOp.SUM,
        torch.distributed.ReduceOp.MIN,
        torch.distributed.ReduceOp.MAX,
    ],
)
def test_hierarchical_small_control_leader_collective_order_without_barrier(op):
    ppo = _ready_hierarchical_small_ppo(leader=True)
    payload = torch.tensor([1, 2], dtype=torch.int32)
    events: list[tuple] = []

    with (
        patch(
            "torch.distributed.reduce",
            side_effect=lambda tensor, dst, op, group: events.append(
                ("reduce", tensor, dst, op, group)
            ),
        ),
        patch(
            "torch.distributed.all_reduce",
            side_effect=lambda tensor, op, group: events.append(
                ("all_reduce", tensor, op, group)
            ),
        ),
        patch(
            "torch.distributed.broadcast",
            side_effect=lambda tensor, src, group: events.append(
                ("broadcast", tensor, src, group)
            ),
        ),
        patch("torch.distributed.barrier") as barrier,
    ):
        result = ppo._hierarchical_all_reduce_small_cpu_tensor(payload, op=op)

    assert result is payload
    assert [(event[0], *event[2:]) for event in events] == [
        ("reduce", 0, op, "local_gloo"),
        ("all_reduce", op, "leader_gloo"),
        ("broadcast", 0, "local_gloo"),
    ]
    barrier.assert_not_called()


def test_hierarchical_small_control_nonleader_skips_leader_collective():
    ppo = _ready_hierarchical_small_ppo(leader=False)
    payload = torch.tensor([1], dtype=torch.int64)
    events: list[str] = []

    with (
        patch(
            "torch.distributed.reduce",
            side_effect=lambda *args, **kwargs: events.append("reduce"),
        ),
        patch("torch.distributed.all_reduce") as leader_all_reduce,
        patch(
            "torch.distributed.broadcast",
            side_effect=lambda *args, **kwargs: events.append("broadcast"),
        ),
    ):
        ppo._hierarchical_all_reduce_small_cpu_tensor(
            payload,
            op=torch.distributed.ReduceOp.MAX,
        )

    assert events == ["reduce", "broadcast"]
    leader_all_reduce.assert_not_called()


def _hierarchical_small_gloo_payload(
    rank: int,
    iteration: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    base = (rank + 1) * ((iteration % 11) + 1) - (iteration % 7) * 3
    if (iteration // 6) % 2 == 0:
        return torch.tensor(base, dtype=dtype)
    return torch.tensor(
        [base, -base + rank, base * 2 - rank],
        dtype=dtype,
    )


def _hierarchical_small_gloo_expected(
    iteration: int,
    dtype: torch.dtype,
    op: torch.distributed.ReduceOp,
) -> torch.Tensor:
    expected = _hierarchical_small_gloo_payload(0, iteration, dtype)
    for rank in range(1, 4):
        candidate = _hierarchical_small_gloo_payload(rank, iteration, dtype)
        if op == torch.distributed.ReduceOp.SUM:
            expected.add_(candidate)
        elif op == torch.distributed.ReduceOp.MIN:
            expected = torch.minimum(expected, candidate)
        else:
            expected = torch.maximum(expected, candidate)
    return expected


def _run_hierarchical_small_gloo_worker(
    rank: int,
    init_file: str,
    result_queue,
) -> None:
    process_group_timeout = timedelta(seconds=20)
    try:
        # The integration test must remain entirely on the host loopback
        # interface and must not discover or contact any remote training host.
        os.environ["GLOO_SOCKET_IFNAME"] = "lo"
        torch.set_num_threads(1)
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=4,
            timeout=process_group_timeout,
        )

        # Every rank creates every group in exactly the same global order. A
        # rank then retains its own local group plus the shared leader group.
        local_groups = [
            torch.distributed.new_group(
                ranks=[0, 1],
                backend="gloo",
                timeout=process_group_timeout,
            ),
            torch.distributed.new_group(
                ranks=[2, 3],
                backend="gloo",
                timeout=process_group_timeout,
            ),
        ]
        leader_group = torch.distributed.new_group(
            ranks=[0, 2],
            backend="gloo",
            timeout=process_group_timeout,
        )

        node_index = rank // 2
        ppo = object.__new__(PPO)
        ppo._hierarchical_grad_reduce_ready = True
        ppo._hierarchical_grad_reduce_available = True
        ppo._hierarchical_local_barrier_group = local_groups[node_index]
        ppo._hierarchical_leader_gloo_group = leader_group
        ppo._hierarchical_local_leader_rank = node_index * 2
        ppo._hierarchical_is_leader_rank = rank in (0, 2)

        ops = (
            torch.distributed.ReduceOp.SUM,
            torch.distributed.ReduceOp.MIN,
            torch.distributed.ReduceOp.MAX,
        )
        dtypes = (torch.int32, torch.int64)
        # 96 iterations cover every op x dtype x scalar/vector combination
        # eight times while preserving one identical collective order per rank.
        for iteration in range(96):
            op = ops[iteration % len(ops)]
            dtype = dtypes[(iteration // 3) % len(dtypes)]
            payload = _hierarchical_small_gloo_payload(rank, iteration, dtype)
            expected = _hierarchical_small_gloo_expected(iteration, dtype, op)

            result = ppo._hierarchical_all_reduce_small_cpu_tensor(payload, op=op)
            if not torch.equal(result, expected):
                raise AssertionError(
                    f"rank={rank} iteration={iteration} op={op} dtype={dtype} "
                    f"expected={expected.tolist()} actual={result.tolist()}"
                )

        torch.distributed.barrier()
        result_queue.put((rank, None))
    except BaseException:
        result_queue.put((rank, traceback.format_exc()))
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_hierarchical_small_control_real_four_process_gloo(tmp_path):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "hierarchical-small-gloo-init")
    processes = [
        context.Process(
            target=_run_hierarchical_small_gloo_worker,
            args=(rank, init_file, result_queue),
        )
        for rank in range(4)
    ]
    deadline = time.monotonic() + 60

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))

        alive = [process.pid for process in processes if process.is_alive()]
        assert not alive, f"Gloo integration workers exceeded 60s timeout: {alive}"

        results = []
        for _ in processes:
            try:
                results.append(result_queue.get(timeout=2))
            except queue.Empty:
                break
        errors = {rank: error for rank, error in results if error is not None}
        exit_codes = [process.exitcode for process in processes]
        assert len(results) == len(processes), (
            f"Only received {len(results)}/4 worker results; exit_codes={exit_codes}"
        )
        assert not errors, "Gloo integration worker failures:\n" + "\n".join(
            f"rank {rank}:\n{error}" for rank, error in sorted(errors.items())
        )
        assert exit_codes == [0, 0, 0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()


def _run_compute_failure_protocol_worker(
    rank: int,
    init_file: str,
    result_queue,
) -> None:
    """Exercise the real two-rank loss/backward failure protocol on Gloo."""

    try:
        torch.distributed.init_process_group(
            backend="gloo",
            init_method=f"file://{init_file}",
            rank=rank,
            world_size=2,
            timeout=timedelta(seconds=20),
        )
        ppo = object.__new__(PPO)
        ppo.device = "cpu"
        ppo.current_learning_iteration = 3
        ppo.gpu_global_rank = rank
        ppo.gpu_world_size = 2
        ppo.is_multi_gpu = True
        ppo.distill_mode = "mse"
        ppo.dagger_enabled = False
        ppo._supervised_dagger_only = False
        ppo._supervised_actor_only_step = False
        ppo.use_symmetry = False
        ppo.actor = nn.Linear(1, 1, bias=False)
        ppo.critic = nn.Linear(1, 1, bias=False)
        ppo.actor_optimizer = torch.optim.SGD(ppo.actor.parameters(), lr=0.1)
        ppo.critic_optimizer = torch.optim.SGD(ppo.critic.parameters(), lr=0.1)
        ppo.max_grad_norm = 1.0
        ppo.ppo_start_noise_std = None
        ppo.config = SimpleNamespace(
            value_loss_coef=1.0,
            symmetry_critic_coef=0.0,
        )
        initial_actor = ppo.actor.weight.detach().clone()
        initial_critic = ppo.critic.weight.detach().clone()

        def all_reduce_small(self, tensor, *, op):
            torch.distributed.all_reduce(tensor, op=op)
            return tensor

        def compute_loss(self, minibatch):  # noqa: ARG001
            if rank == 0:
                raise RuntimeError("injected rank-local actor forward failure")
            return {
                "actor_loss": self.actor.weight.square().sum(),
                "critic_loss": self.critic.weight.square().sum(),
                "value_loss": torch.tensor(0.5),
                "surrogate_loss": torch.tensor(0.25),
                "entropy_loss": torch.tensor(0.1),
                "kl_mean": torch.tensor(0.01),
            }

        ppo._all_reduce_small_tensor = MethodType(all_reduce_small, ppo)
        ppo._get_distributed_loss_weight = MethodType(lambda self: 1.0, ppo)
        ppo._setup_gloo_barrier_group = MethodType(
            lambda self: torch.distributed.group.WORLD,
            ppo,
        )
        ppo._compute_ppo_loss = MethodType(compute_loss, ppo)

        error = None
        try:
            ppo._update_algo_step(
                {},
                {"Value": 0.0, "Surrogate": 0.0, "Entropy": 0.0, "KL": 0.0},
            )
        except Exception as exc:  # expected on both ranks
            error = f"{type(exc).__name__}: {exc}"

        result_queue.put(
            {
                "rank": rank,
                "error": error,
                "actor_unchanged": torch.equal(ppo.actor.weight, initial_actor),
                "critic_unchanged": torch.equal(ppo.critic.weight, initial_critic),
                "actor_grad_cleared": ppo.actor.weight.grad is None,
                "critic_grad_cleared": ppo.critic.weight.grad is None,
            }
        )
    except Exception:
        result_queue.put({"rank": rank, "worker_error": traceback.format_exc()})
    finally:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def test_compute_failure_is_atomic_across_two_real_gloo_ranks(tmp_path):
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    init_file = str(tmp_path / "compute-failure-protocol-init")
    processes = [
        context.Process(
            target=_run_compute_failure_protocol_worker,
            args=(rank, init_file, result_queue),
        )
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        results = [result_queue.get(timeout=30) for _ in processes]
        for process in processes:
            process.join(timeout=30)

        assert all(not process.is_alive() for process in processes)
        assert all(process.exitcode == 0 for process in processes)
        assert not [result for result in results if "worker_error" in result]
        results_by_rank = {result["rank"]: result for result in results}
        assert set(results_by_rank) == {0, 1}
        for result in results_by_rank.values():
            assert "failed on at least one rank" in result["error"]
            assert "rank=0: RuntimeError: injected rank-local actor forward failure" in result[
                "error"
            ]
            assert result["actor_unchanged"] is True
            assert result["critic_unchanged"] is True
            assert result["actor_grad_cleared"] is True
            assert result["critic_grad_cleared"] is True
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
        result_queue.close()
        result_queue.join_thread()


def test_hierarchical_small_control_fails_before_collective_when_groups_missing():
    ppo = _ready_hierarchical_small_ppo(leader=True)
    ppo._hierarchical_leader_gloo_group = None

    with (
        patch("torch.distributed.reduce") as local_reduce,
        patch("torch.distributed.all_reduce") as leader_all_reduce,
        patch("torch.distributed.broadcast") as local_broadcast,
        pytest.raises(RuntimeError, match="node-local and leader Gloo groups"),
    ):
        ppo._hierarchical_all_reduce_small_cpu_tensor(
            torch.tensor([1], dtype=torch.int32),
            op=torch.distributed.ReduceOp.SUM,
        )

    local_reduce.assert_not_called()
    leader_all_reduce.assert_not_called()
    local_broadcast.assert_not_called()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.complex64,
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
    ],
)
def test_hierarchical_small_control_rejects_unsupported_dtype_before_collective(dtype):
    ppo = _ready_hierarchical_small_ppo(leader=True)
    payload = torch.ones(1, dtype=dtype)

    with (
        patch("torch.distributed.reduce") as local_reduce,
        pytest.raises(TypeError, match="integral"),
    ):
        ppo._hierarchical_all_reduce_small_cpu_tensor(
            payload,
            op=torch.distributed.ReduceOp.SUM,
        )

    local_reduce.assert_not_called()


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.complex64,
        torch.bool,
        torch.uint8,
        torch.int8,
        torch.int16,
    ],
)
def test_small_unsupported_tensor_stays_on_flat_gloo_when_hierarchy_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
    dtype,
):
    ppo = _ready_hierarchical_small_ppo(leader=True)
    monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "1")
    ppo._setup_gloo_barrier_group = MethodType(lambda self: "flat_gloo", ppo)
    payload = torch.ones(2, dtype=dtype)

    with (
        patch.object(ppo, "_hierarchical_all_reduce_small_cpu_tensor") as hierarchical,
        patch("torch.distributed.all_reduce") as flat_all_reduce,
    ):
        result = ppo._all_reduce_small_tensor(
            payload,
            op=torch.distributed.ReduceOp.SUM,
        )

    assert torch.equal(result, payload)
    hierarchical.assert_not_called()
    flat_all_reduce.assert_called_once()
    assert flat_all_reduce.call_args.kwargs["group"] == "flat_gloo"


def test_ppo_setup_initializes_hierarchical_small_groups_before_models():
    ppo = object.__new__(PPO)
    ppo._hierarchical_small_collectives = True
    ppo._evaluation_only = True
    ppo.is_multi_gpu = True
    events: list[str] = []
    ppo._gloo_small_collectives_enabled = MethodType(lambda self: True, ppo)
    ppo._hierarchical_grad_reduce_enabled = MethodType(lambda self: True, ppo)

    def setup_groups(self):
        events.append("groups")
        self._hierarchical_local_barrier_group = "local_gloo"
        self._hierarchical_leader_gloo_group = "leader_gloo"
        return True

    ppo._setup_hierarchical_grad_reduce_groups = MethodType(setup_groups, ppo)
    ppo._setup_models_and_optimizer = MethodType(
        lambda self: events.append("models"),
        ppo,
    )
    ppo._configure_active_observation_groups = MethodType(lambda self: None, ppo)

    ppo.setup()

    assert events == ["groups", "models"]


def _hierarchical_setup_agreement_ppo() -> PPO:
    ppo = _hierarchical_ppo(rank=0, local_rank=0)
    ppo.gpu_world_size = 4
    ppo.gpu_local_world_size = 2
    ppo.gpu_topology_local_rank = 0
    ppo._hierarchical_small_collectives = True
    ppo.device = "cpu"
    return ppo


def _copy_hierarchical_setup_records(records):
    def copy_records(gathered, _local_record):
        assert len(gathered) == len(records)
        for target, values in zip(gathered, records, strict=True):
            target.copy_(torch.tensor(values, dtype=target.dtype, device=target.device))

    return copy_records


def _valid_hierarchical_setup_records():
    return [
        (1, 1, 1, 0, 300, 4, rank, 2, rank % 2)
        for rank in range(4)
    ]


def test_single_rank_evaluation_ignores_inherited_hierarchical_training_flag():
    ppo = object.__new__(PPO)
    ppo._hierarchical_small_collectives = True
    ppo._evaluation_only = True
    ppo.is_multi_gpu = False

    assert ppo._validate_hierarchical_collective_setup_agreement() is False


def test_hierarchical_setup_agreement_accepts_consistent_all_rank_topology(
    monkeypatch: pytest.MonkeyPatch,
):
    ppo = _hierarchical_setup_agreement_ppo()
    monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "0")

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_backend", return_value="gloo"),
        patch("torch.distributed.get_world_size", return_value=4),
        patch(
            "torch.distributed.all_gather",
            side_effect=_copy_hierarchical_setup_records(
                _valid_hierarchical_setup_records()
            ),
        ),
    ):
        assert ppo._validate_hierarchical_collective_setup_agreement() is True


@pytest.mark.parametrize("mismatch", ["flag", "timeout", "topology"])
def test_hierarchical_setup_agreement_fails_before_any_subgroup_creation(
    monkeypatch: pytest.MonkeyPatch,
    mismatch: str,
):
    ppo = _hierarchical_setup_agreement_ppo()
    monkeypatch.setenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "0")
    records = _valid_hierarchical_setup_records()
    if mismatch == "flag":
        records[3] = (0, *records[3][1:])
    elif mismatch == "timeout":
        records[3] = (*records[3][:4], 301, *records[3][5:])
    else:
        records[3] = (*records[3][:-1], 0)
    ppo._setup_models_and_optimizer = MethodType(lambda self: None, ppo)

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_backend", return_value="gloo"),
        patch("torch.distributed.get_world_size", return_value=4),
        patch(
            "torch.distributed.all_gather",
            side_effect=_copy_hierarchical_setup_records(records),
        ),
        patch("torch.distributed.new_group") as new_group,
        patch.object(ppo, "_setup_hierarchical_grad_reduce_groups") as setup_groups,
        pytest.raises(RuntimeError, match="before subgroup creation"),
    ):
        ppo.setup()

    setup_groups.assert_not_called()
    new_group.assert_not_called()


class _FakeHierarchicalPayload:
    def __init__(self, events: list[tuple], label: str, *, device: str):
        self.events = events
        self.label = label
        self.device = torch.device(device)
        self.dtype = torch.float32

    def detach(self):
        self.events.append(("detach", self.label))
        return self

    def cpu(self):
        self.events.append(("cpu", self.label))
        return _FakeHierarchicalPayload(self.events, "cpu_payload", device="cpu")

    def to(self, *, device, dtype):
        self.events.append(("to", self.label, device, dtype))
        return _FakeHierarchicalPayload(self.events, "gpu_result", device=str(device))

    def copy_(self, source):
        self.events.append(("copy", self.label, source.label))
        return self


def test_hierarchical_cpu_leader_collective_order_and_cuda_fences(
    monkeypatch: pytest.MonkeyPatch,
):
    ppo = _hierarchical_ppo()
    ppo._hierarchical_grad_reduce_ready = True
    ppo._hierarchical_grad_reduce_available = True
    ppo._hierarchical_grad_reduce_cpu_leader = True
    ppo._hierarchical_local_group = "local_nccl"
    ppo._hierarchical_local_barrier_group = "local_gloo"
    ppo._hierarchical_leader_gloo_group = "leader_gloo"
    ppo._hierarchical_local_leader_rank = 0
    ppo._hierarchical_is_leader_rank = True
    monkeypatch.delenv("HOLOSOMA_GLOO_GRAD_REDUCE", raising=False)
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "1")
    events: list[tuple] = []
    payload = _FakeHierarchicalPayload(events, "payload", device="cuda:0")

    with (
        patch("torch.cuda.synchronize", side_effect=lambda device: events.append(("sync", device))),
        patch(
            "torch.distributed.reduce",
            side_effect=lambda tensor, dst, op, group: events.append(("reduce", tensor.label, dst, op, group)),
        ),
        patch(
            "torch.distributed.all_reduce",
            side_effect=lambda tensor, op, group: events.append(("all_reduce", tensor.label, op, group)),
        ),
        patch("torch.distributed.barrier", side_effect=lambda *, group: events.append(("barrier", group))),
        patch(
            "torch.distributed.broadcast",
            side_effect=lambda tensor, src, group: events.append(("broadcast", tensor.label, src, group)),
        ),
    ):
        path = ppo._all_reduce_grad_payload(payload)

    assert path == "hierarchical_cpu_leader"
    assert events == [
        ("sync", payload.device),
        ("reduce", "payload", 0, torch.distributed.ReduceOp.SUM, "local_nccl"),
        ("sync", payload.device),
        ("detach", "payload"),
        ("cpu", "payload"),
        ("all_reduce", "cpu_payload", torch.distributed.ReduceOp.SUM, "leader_gloo"),
        ("to", "cpu_payload", payload.device, payload.dtype),
        ("copy", "payload", "gpu_result"),
        ("sync", payload.device),
        ("barrier", "local_gloo"),
        ("sync", payload.device),
        ("broadcast", "payload", 0, "local_nccl"),
        ("sync", payload.device),
    ]


def test_hierarchical_nonleader_skips_inter_node_collective_but_joins_broadcast(
    monkeypatch: pytest.MonkeyPatch,
):
    ppo = _hierarchical_ppo(rank=42, local_rank=2)
    ppo._hierarchical_grad_reduce_ready = True
    ppo._hierarchical_grad_reduce_available = True
    ppo._hierarchical_grad_reduce_cpu_leader = True
    ppo._hierarchical_local_group = "local_nccl"
    ppo._hierarchical_local_barrier_group = "local_gloo"
    ppo._hierarchical_leader_gloo_group = "not_a_member"
    ppo._hierarchical_local_leader_rank = 40
    ppo._hierarchical_is_leader_rank = False
    monkeypatch.delenv("HOLOSOMA_GLOO_GRAD_REDUCE", raising=False)
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "1")
    events: list[tuple] = []
    payload = _FakeHierarchicalPayload(events, "payload", device="cuda:0")

    with (
        patch("torch.cuda.synchronize", side_effect=lambda device: events.append(("sync", device))),
        patch(
            "torch.distributed.reduce",
            side_effect=lambda tensor, dst, op, group: events.append(("reduce", tensor.label, dst, op, group)),
        ),
        patch("torch.distributed.all_reduce") as leader_all_reduce,
        patch("torch.distributed.barrier", side_effect=lambda *, group: events.append(("barrier", group))),
        patch(
            "torch.distributed.broadcast",
            side_effect=lambda tensor, src, group: events.append(("broadcast", tensor.label, src, group)),
        ),
    ):
        path = ppo._all_reduce_grad_payload(payload)

    assert path == "hierarchical_cpu_leader"
    leader_all_reduce.assert_not_called()
    assert events == [
        ("sync", payload.device),
        ("reduce", "payload", 40, torch.distributed.ReduceOp.SUM, "local_nccl"),
        ("sync", payload.device),
        ("barrier", "local_gloo"),
        ("sync", payload.device),
        ("broadcast", "payload", 40, "local_nccl"),
        ("sync", payload.device),
    ]


def _ready_hierarchical_gpu_leader_ppo(*, leader: bool) -> PPO:
    ppo = _hierarchical_ppo(rank=0 if leader else 1, local_rank=0 if leader else 1)
    ppo._hierarchical_grad_reduce_ready = True
    ppo._hierarchical_grad_reduce_available = True
    ppo._hierarchical_grad_reduce_cpu_leader = False
    ppo._hierarchical_local_group = "local_nccl"
    ppo._hierarchical_local_barrier_group = "local_gloo"
    ppo._hierarchical_leader_group = "leader_nccl"
    ppo._hierarchical_local_leader_rank = 0
    ppo._hierarchical_is_leader_rank = leader
    return ppo


def test_hierarchical_gpu_leader_stays_on_cuda_and_preserves_collective_order(
    monkeypatch: pytest.MonkeyPatch,
):
    ppo = _ready_hierarchical_gpu_leader_ppo(leader=True)
    monkeypatch.delenv("HOLOSOMA_GLOO_GRAD_REDUCE", raising=False)
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "0")
    events: list[tuple] = []
    payload = _FakeHierarchicalPayload(events, "payload", device="cuda:0")

    with (
        patch("torch.cuda.synchronize", side_effect=lambda device: events.append(("sync", device))),
        patch(
            "torch.distributed.reduce",
            side_effect=lambda tensor, dst, op, group: events.append(
                ("reduce", tensor.label, dst, op, group)
            ),
        ),
        patch(
            "torch.distributed.all_reduce",
            side_effect=lambda tensor, op, group: events.append(
                ("all_reduce", tensor.label, op, group)
            ),
        ),
        patch(
            "torch.distributed.barrier",
            side_effect=lambda *, group: events.append(("barrier", group)),
        ),
        patch(
            "torch.distributed.broadcast",
            side_effect=lambda tensor, src, group: events.append(
                ("broadcast", tensor.label, src, group)
            ),
        ),
    ):
        path = ppo._all_reduce_grad_payload(payload)

    assert path == "hierarchical"
    assert events == [
        ("sync", payload.device),
        ("reduce", "payload", 0, torch.distributed.ReduceOp.SUM, "local_nccl"),
        ("sync", payload.device),
        ("all_reduce", "payload", torch.distributed.ReduceOp.SUM, "leader_nccl"),
        ("sync", payload.device),
        ("barrier", "local_gloo"),
        ("sync", payload.device),
        ("broadcast", "payload", 0, "local_nccl"),
        ("sync", payload.device),
    ]
    assert not any(event[0] in {"detach", "cpu", "to", "copy"} for event in events)


def test_hierarchical_gpu_nonleader_skips_leader_nccl_but_joins_local_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
):
    ppo = _ready_hierarchical_gpu_leader_ppo(leader=False)
    monkeypatch.delenv("HOLOSOMA_GLOO_GRAD_REDUCE", raising=False)
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "0")
    events: list[tuple] = []
    payload = _FakeHierarchicalPayload(events, "payload", device="cuda:1")

    with (
        patch("torch.cuda.synchronize", side_effect=lambda device: events.append(("sync", device))),
        patch(
            "torch.distributed.reduce",
            side_effect=lambda tensor, dst, op, group: events.append(
                ("reduce", tensor.label, dst, op, group)
            ),
        ),
        patch("torch.distributed.all_reduce") as leader_all_reduce,
        patch(
            "torch.distributed.barrier",
            side_effect=lambda *, group: events.append(("barrier", group)),
        ),
        patch(
            "torch.distributed.broadcast",
            side_effect=lambda tensor, src, group: events.append(
                ("broadcast", tensor.label, src, group)
            ),
        ),
    ):
        path = ppo._all_reduce_grad_payload(payload)

    assert path == "hierarchical"
    leader_all_reduce.assert_not_called()
    assert events == [
        ("sync", payload.device),
        ("reduce", "payload", 0, torch.distributed.ReduceOp.SUM, "local_nccl"),
        ("sync", payload.device),
        ("barrier", "local_gloo"),
        ("sync", payload.device),
        ("broadcast", "payload", 0, "local_nccl"),
        ("sync", payload.device),
    ]
    assert not any(event[0] in {"detach", "cpu", "to", "copy"} for event in events)


def test_hierarchical_gpu_leader_collective_error_never_falls_back_to_flat_world(
    monkeypatch: pytest.MonkeyPatch,
):
    ppo = _ready_hierarchical_gpu_leader_ppo(leader=True)
    monkeypatch.delenv("HOLOSOMA_GLOO_GRAD_REDUCE", raising=False)
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "0")
    payload = _FakeHierarchicalPayload([], "payload", device="cuda:0")
    all_reduce_groups: list[object] = []

    def fail_leader_all_reduce(tensor, op, group=None):
        all_reduce_groups.append(group)
        raise RuntimeError("injected leader NCCL failure")

    with (
        patch("torch.cuda.synchronize"),
        patch("torch.distributed.reduce"),
        patch("torch.distributed.all_reduce", side_effect=fail_leader_all_reduce),
        patch("torch.distributed.barrier") as local_barrier,
        patch("torch.distributed.broadcast") as local_broadcast,
        pytest.raises(RuntimeError, match="injected leader NCCL failure"),
    ):
        ppo._all_reduce_grad_payload(payload)

    assert all_reduce_groups == ["leader_nccl"]
    local_barrier.assert_not_called()
    local_broadcast.assert_not_called()
    assert not any(
        event[0] in {"detach", "cpu", "to", "copy"} for event in payload.events
    )


def test_hierarchical_reduction_fails_closed_instead_of_using_world_nccl(
    monkeypatch: pytest.MonkeyPatch,
):
    ppo = _hierarchical_ppo()
    monkeypatch.delenv("HOLOSOMA_GLOO_GRAD_REDUCE", raising=False)
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE_CPU_LEADER", "1")
    ppo._setup_hierarchical_grad_reduce_groups = MethodType(lambda self: False, ppo)

    with (
        patch("torch.distributed.all_reduce") as world_all_reduce,
        pytest.raises(RuntimeError, match="requested but the distributed topology is unsupported"),
    ):
        ppo._all_reduce_grad_payload(SimpleNamespace(device=torch.device("cpu")))

    world_all_reduce.assert_not_called()


def test_gradient_reduce_modes_are_mutually_exclusive(monkeypatch: pytest.MonkeyPatch):
    ppo = _hierarchical_ppo()
    monkeypatch.setenv("HOLOSOMA_GLOO_GRAD_REDUCE", "1")
    monkeypatch.setenv("HOLOSOMA_HIERARCHICAL_GRAD_REDUCE", "1")

    with pytest.raises(RuntimeError, match="mutually exclusive"):
        ppo._all_reduce_grad_payload(SimpleNamespace(device=torch.device("cpu")))


@pytest.mark.parametrize("device", [torch.device("cuda:1"), torch.device("cpu")])
def test_small_default_group_collective_fences_cuda_only(
    monkeypatch: pytest.MonkeyPatch,
    device: torch.device,
):
    ppo = object.__new__(PPO)
    monkeypatch.delenv("HOLOSOMA_GLOO_SMALL_COLLECTIVES", raising=False)
    payload = SimpleNamespace(device=device)
    events: list[tuple[str, object]] = []

    with (
        patch(
            "torch.cuda.synchronize",
            side_effect=lambda selected: events.append(("synchronize", selected)),
        ),
        patch(
            "torch.distributed.all_reduce",
            side_effect=lambda tensor, op: events.append(("all_reduce", tensor, op)),
        ),
    ):
        result = ppo._all_reduce_small_tensor(payload, op=torch.distributed.ReduceOp.MAX)

    expected = [("all_reduce", payload, torch.distributed.ReduceOp.MAX)]
    if device.type == "cuda":
        expected = [("synchronize", device), *expected, ("synchronize", device)]
    assert result is payload
    assert events == expected


def test_global_bc_denominator_preserves_sparse_fraction_below_one():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.env = SimpleNamespace(distributed_loss_weight=1.0)
    # Rank 0 has one valid label and rank 1 has none, so the global weighted
    # count divided by DDP world size is exactly 0.5.
    ppo._all_reduce_small_tensor = MethodType(lambda self, tensor, op: tensor, ppo)

    denominator = ppo._global_bc_denominator(torch.tensor(1.0))

    assert denominator.item() == pytest.approx(0.5)


def test_global_bc_presence_is_false_only_when_all_weighted_ranks_are_empty():
    ppo = object.__new__(PPO)
    ppo.is_multi_gpu = True
    ppo.gpu_world_size = 2
    ppo.env = SimpleNamespace(distributed_loss_weight=1.0)
    ppo._all_reduce_small_tensor = MethodType(
        lambda self, tensor, op: torch.zeros_like(tensor),
        ppo,
    )

    denominator, has_valid_samples = ppo._global_bc_denominator_and_presence(
        torch.tensor(0.0)
    )

    assert denominator.item() == pytest.approx(1.0)
    assert has_valid_samples is False


def test_onnx_export_failure_restores_training_mode():
    ppo = object.__new__(PPO)
    ppo.actor = nn.Linear(1, 1)
    ppo.actor.train()
    ppo.actor_perception_key = ""
    ppo._eval_mode = MethodType(lambda self: self.actor.eval(), ppo)
    ppo._train_mode = MethodType(lambda self: self.actor.train(), ppo)
    ppo._get_zero_input = MethodType(lambda self: torch.zeros(1, 1), ppo)
    ppo._get_zero_perception_input = MethodType(lambda self: None, ppo)

    with (
        patch.object(PPO, "actor_onnx_wrapper", new_callable=PropertyMock, return_value=object()),
        patch("holosoma.agents.ppo.ppo.export_policy_as_onnx", side_effect=RuntimeError("export failed")),
        pytest.raises(RuntimeError, match="export failed"),
    ):
        ppo.export("policy.onnx")

    assert ppo.actor.training is True


def test_evaluation_policy_mode_is_explicit_and_fail_closed(monkeypatch):
    monkeypatch.delenv("HOLOSOMA_EVAL_POLICY", raising=False)
    assert PPO._requested_evaluation_policy_mode() == "checkpoint_actor"

    monkeypatch.setenv("HOLOSOMA_EVAL_POLICY", "student")
    assert PPO._requested_evaluation_policy_mode() == "checkpoint_actor"

    monkeypatch.setenv("HOLOSOMA_EVAL_POLICY", "distill_label_teacher")
    assert PPO._requested_evaluation_policy_mode() == "distill_label_teacher"

    monkeypatch.setenv("HOLOSOMA_EVAL_POLICY", "distill_label_teacher_bc_target")
    assert PPO._requested_evaluation_policy_mode() == "distill_label_teacher_bc_target"

    monkeypatch.setenv("HOLOSOMA_EVAL_POLICY", "teacher")
    with pytest.raises(ValueError, match="ambiguous.*checkpoint_actor.*distill_label_teacher"):
        PPO._requested_evaluation_policy_mode()

    monkeypatch.setenv("HOLOSOMA_EVAL_POLICY", "reference")
    with pytest.raises(ValueError, match="checkpoint_actor.*distill_label_teacher"):
        PPO._requested_evaluation_policy_mode()


@pytest.mark.parametrize(
    ("evaluation_mode", "expected_clip_enabled"),
    [
        ("distill_label_teacher", False),
        ("distill_label_teacher_bc_target", True),
    ],
)
def test_teacher_evaluation_loads_source_authenticated_teacher(
    monkeypatch,
    evaluation_mode,
    expected_clip_enabled,
):
    monkeypatch.setenv("HOLOSOMA_EVAL_POLICY", evaluation_mode)
    teacher = nn.Linear(1, 1)
    teacher.train()
    normalizer = nn.Identity()
    normalizer.train()
    captured = {}

    distill_cfg = SimpleNamespace(
        enabled=True,
        use_multi_teacher=False,
        policy_to_clone="/tmp/authenticated-teacher.pt",
        teacher_checkpoint=None,
        teacher_obs_keys="actor_obs",
        strict_teacher_load=True,
        teacher_perception_obs_key=None,
        clip_teacher_actions=True,
        clip_actions_threshold=8.0,
    )
    ppo = object.__new__(PPO)
    ppo._evaluation_only = True
    ppo._experiment_config = SimpleNamespace(
        algo=SimpleNamespace(config=SimpleNamespace(distill=distill_cfg))
    )
    ppo._training_provenance = {
        "teacher_enabled": True,
        "teacher_sha256": "a" * 64,
    }
    ppo.actor_obs_keys = ["student_obs"]
    ppo.algo_obs_dim_dict = {"actor_obs": 3, "student_obs": 2}
    ppo._build_obs_slices = lambda keys: {"actor_obs": slice(0, 3)}
    ppo._get_obs_dim = lambda keys: 3
    ppo._load_teacher_actor = lambda path, obs_keys: (
        captured.update(path=path, obs_keys=obs_keys) or (teacher, {"actor_obs": normalizer})
    )
    ppo._validate_loaded_teacher_inference_contract = lambda: captured.update(validated=True)
    ppo._configure_active_observation_groups = lambda: captured.update(active_groups=True)

    ppo._prepare_selected_evaluation_policy()

    assert ppo._evaluation_policy_mode == evaluation_mode
    assert ppo.distill_enabled is True
    assert ppo.dagger_enabled is False
    assert ppo.teacher_use_stochastic_actions is False
    assert ppo._evaluation_teacher_action_clip_enabled is expected_clip_enabled
    assert ppo._evaluation_teacher_action_clip_threshold == (
        8.0 if expected_clip_enabled else None
    )
    assert captured == {
        "path": "/tmp/authenticated-teacher.pt",
        "obs_keys": ["actor_obs"],
        "validated": True,
        "active_groups": True,
    }
    assert teacher.training is False
    assert normalizer.training is False


def test_teacher_evaluation_keeps_teacher_groups_active_after_late_load():
    ppo = object.__new__(PPO)
    ppo._evaluation_only = True
    ppo._evaluation_policy_mode = "distill_label_teacher"
    ppo.actor_obs_keys = ["student_obs"]
    ppo.critic_obs_keys = ["critic_obs"]
    ppo.teacher_obs_keys = ["actor_obs"]
    ppo.actor_perception_key = "student_depth"
    ppo.critic_perception_key = ""
    ppo.teacher_perception_obs_key = ""
    ppo.distill_enabled = True
    ppo.distill_mode = "dagger"
    ppo.dagger_enabled = False
    ppo.use_multi_teacher = False
    ppo.teacher_actor = SimpleNamespace(perception_input_name="")
    ppo.teacher_actors = []
    ppo.is_main_process = False
    set_active_groups = Mock()
    ppo.env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            active_group_names=("student_obs", "critic_obs", "student_depth"),
            set_active_groups=set_active_groups,
            cfg=SimpleNamespace(groups={}),
        )
    )

    ppo._configure_active_observation_groups()

    assert set_active_groups.call_args.args[0] == [
        "student_obs",
        "critic_obs",
        "student_depth",
        "actor_obs",
    ]


def test_teacher_evaluation_step_uses_teacher_observation_and_action_path():
    ppo = object.__new__(PPO)
    ppo._evaluation_policy_mode = "distill_label_teacher"
    ppo.actor_obs_keys = ["student_obs"]
    ppo.teacher_obs_keys = ["teacher_obs"]
    ppo.eval_callbacks = []
    captured = {}
    ppo._normalize_teacher_actor_obs = lambda value: value + 10.0

    def select_teacher(teacher_obs_raw, obs_dict, *, stochastic):
        captured["teacher_obs_raw"] = teacher_obs_raw.clone()
        captured["stochastic"] = stochastic
        return torch.full((teacher_obs_raw.shape[0], 1), 42.0), None

    ppo._select_teacher_actions = select_teacher
    ppo._maybe_debug_eval_policy_io = lambda **kwargs: captured.update(debug=kwargs)
    actor_state = {
        "step": 7,
        "obs": {
            "student_obs": torch.tensor([[1.0, 2.0]]),
            "teacher_obs": torch.tensor([[3.0, 4.0, 5.0]]),
        },
    }

    result = ppo._pre_eval_env_step(actor_state)

    assert torch.equal(captured["teacher_obs_raw"], torch.tensor([[3.0, 4.0, 5.0]]))
    assert captured["stochastic"] is False
    assert torch.equal(result["actions"], torch.tensor([[42.0]]))
    assert torch.equal(captured["debug"]["actor_obs"], torch.tensor([[13.0, 14.0, 15.0]]))


def test_teacher_bc_target_evaluation_clips_actions_before_env_step():
    ppo = object.__new__(PPO)
    ppo._evaluation_policy_mode = "distill_label_teacher_bc_target"
    ppo._evaluation_teacher_action_clip_enabled = True
    ppo._evaluation_teacher_action_clip_threshold = 8.0
    ppo.actor_obs_keys = ["student_obs"]
    ppo.teacher_obs_keys = ["teacher_obs"]
    ppo.eval_callbacks = []
    captured = {}
    ppo._normalize_teacher_actor_obs = lambda value: value
    ppo._select_teacher_actions = lambda *_args, **_kwargs: (
        torch.tensor([[9.0, -10.0, 3.0]]),
        None,
    )
    ppo._maybe_debug_eval_policy_io = lambda **kwargs: captured.update(debug=kwargs)
    actor_state = {
        "step": 0,
        "obs": {
            "student_obs": torch.tensor([[1.0, 2.0]]),
            "teacher_obs": torch.tensor([[3.0, 4.0, 5.0]]),
        },
    }

    result = ppo._pre_eval_env_step(actor_state)

    expected = torch.tensor([[8.0, -8.0, 3.0]])
    assert torch.equal(result["actions"], expected)
    assert torch.equal(captured["debug"]["actions"], expected)
