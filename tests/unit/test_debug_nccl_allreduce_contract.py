from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "debug_nccl_allreduce.py"
SPEC = importlib.util.spec_from_file_location("debug_nccl_allreduce_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
canary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = canary
SPEC.loader.exec_module(canary)


def _env(**overrides: str) -> dict[str, str]:
    values = {
        "RANK": "0",
        "WORLD_SIZE": "1",
        "LOCAL_RANK": "0",
        "LOCAL_WORLD_SIZE": "1",
        "HOLOSOMA_NCCL_TEST_BACKEND": "gloo",
    }
    values.update(overrides)
    return values


def test_legacy_environment_parser_and_rank_visible_topology() -> None:
    config = canary._parse_config(
        _env(
            RANK="5",
            WORLD_SIZE="8",
            LOCAL_RANK="0",
            LOCAL_WORLD_SIZE="1",
            HOLOSOMA_ORIGINAL_LOCAL_RANK="1",
            HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE="4",
            HOLOSOMA_NCCL_TEST_NUMEL="3076719",
            HOLOSOMA_NCCL_TEST_ITERS="25",
            HOLOSOMA_NCCL_TEST_OPS_PER_ROUND="2",
            HOLOSOMA_NCCL_TEST_HIERARCHICAL="yes",
            HOLOSOMA_NCCL_TEST_CONTROL_MODE="hierarchical",
        )
    )

    assert config["rank"] == 5
    assert config["device_rank"] == 0
    assert config["local_rank"] == 1
    assert config["local_world"] == 4
    assert config["numel"] == 3_076_719
    assert config["rounds"] == 25
    assert config["warmup"] == 10
    assert config["ops_per_round"] == 2
    assert config["control_ops"] == ("sum", "min", "max")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"HOLOSOMA_NCCL_TEST_HIERARCHICAL": "treu"}, "must be a boolean"),
        ({"HOLOSOMA_NCCL_TEST_CPU_LEADER": "1"}, "requires hierarchical"),
        (
            {
                "WORLD_SIZE": "8",
                "LOCAL_WORLD_SIZE": "4",
                "HOLOSOMA_NCCL_TEST_HIERARCHICAL": "1",
                "HOLOSOMA_NCCL_TEST_GLOO_PAYLOAD": "1",
            },
            "mutually exclusive",
        ),
        ({"HOLOSOMA_NCCL_TEST_CONTROL_OPS": "sum,mean"}, "unique subset"),
        ({"HOLOSOMA_NCCL_TEST_EXPECTED_WORLD_SIZE": "104"}, "Expected world"),
        ({"LOCAL_RANK": "-1"}, "must be non-negative"),
    ],
)
def test_invalid_protocol_configuration_fails_closed(
    overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        canary._parse_config(_env(**overrides))


def test_exact_13_by_8_gpu_leader_group_creation_order() -> None:
    config = canary._parse_config(
        _env(
            WORLD_SIZE="104",
            LOCAL_WORLD_SIZE="8",
            HOLOSOMA_NCCL_TEST_HIERARCHICAL="1",
            HOLOSOMA_NCCL_TEST_CONTROL_MODE="hierarchical",
        )
    )
    plan = canary._group_plan(config)

    assert len(plan) == 28
    for node in range(13):
        ranks = list(range(node * 8, (node + 1) * 8))
        assert plan[node * 2] == (f"local_payload_{node}", ranks, "nccl")
        assert plan[node * 2 + 1] == (f"local_barrier_{node}", ranks, "gloo")
    leaders = list(range(0, 104, 8))
    assert plan[-2] == ("leader_payload", leaders, "nccl")
    assert plan[-1] == ("leader_control", leaders, "gloo")


def test_cpu_leader_reuses_gloo_group_for_hierarchical_controls() -> None:
    config = canary._parse_config(
        _env(
            WORLD_SIZE="8",
            LOCAL_WORLD_SIZE="4",
            HOLOSOMA_NCCL_TEST_HIERARCHICAL="1",
            HOLOSOMA_NCCL_TEST_CPU_LEADER="1",
            HOLOSOMA_NCCL_TEST_CONTROL_MODE="hierarchical",
        )
    )
    plan = canary._group_plan(config)

    assert plan[-1] == ("leader_payload", [0, 4], "gloo")
    assert all(name != "leader_control" for name, _, _ in plan)


def test_hierarchical_control_sequence_has_no_local_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(
        canary.dist,
        "reduce",
        lambda tensor, *, dst, op, group: events.append(("reduce", group)),
    )
    monkeypatch.setattr(
        canary.dist,
        "all_reduce",
        lambda tensor, *, op, group: events.append(("all_reduce", group)),
    )
    monkeypatch.setattr(
        canary.dist,
        "broadcast",
        lambda tensor, *, src, group: events.append(("broadcast", group)),
    )
    monkeypatch.setattr(
        canary.dist,
        "barrier",
        lambda *, group: events.append(("barrier", group)),
    )
    groups = {
        "local_leader": 0,
        "local_barrier": "local_gloo",
        "leader_control": "leader_gloo",
        "is_leader": True,
    }

    canary._control_reduce(torch.ones(1, dtype=torch.int64), "sum", {"control_mode": "hierarchical"}, groups)

    assert events == [
        ("reduce", "local_gloo"),
        ("all_reduce", "leader_gloo"),
        ("broadcast", "local_gloo"),
    ]


def test_percentiles_ignore_nonleader_nan_samples() -> None:
    stats = canary._percentiles(
        torch.tensor([1.0, 2.0, float("nan"), 4.0], dtype=torch.float64)
    )

    assert stats["samples"] == 3
    assert stats["p50"] == pytest.approx(2.0)
    assert stats["max"] == pytest.approx(4.0)
