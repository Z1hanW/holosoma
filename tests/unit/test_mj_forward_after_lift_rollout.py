from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/mj_forward_after_lift_rollout.py"
SPEC = importlib.util.spec_from_file_location("mj_forward_after_lift_rollout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
trigger_source = MODULE._trigger_source


def test_height_gate_triggers_immediately_before_deadline() -> None:
    assert trigger_source(
        rel_z=0.271,
        sim_time_ms=1800,
        lift_rel_z_delta_m=0.27,
        latest_forward_actor_sim_time_ms=2500,
        deadline_publish_lead_ms=40,
    ) == "height"


def test_height_gate_wins_when_height_and_deadline_coincide() -> None:
    assert trigger_source(
        rel_z=0.27,
        sim_time_ms=2460,
        lift_rel_z_delta_m=0.27,
        latest_forward_actor_sim_time_ms=2500,
        deadline_publish_lead_ms=40,
    ) == "height"


def test_time_fallback_publishes_with_lead_for_actor_deadline() -> None:
    common = {
        "rel_z": 0.1,
        "lift_rel_z_delta_m": 0.27,
        "latest_forward_actor_sim_time_ms": 2500,
        "deadline_publish_lead_ms": 40,
    }
    assert trigger_source(sim_time_ms=2455, **common) is None
    assert trigger_source(sim_time_ms=2460, **common) == "time_fallback"


def test_disabled_time_fallback_never_bypasses_height_gate() -> None:
    assert (
        trigger_source(
            rel_z=0.1,
            sim_time_ms=9000,
            lift_rel_z_delta_m=0.27,
            latest_forward_actor_sim_time_ms=None,
            deadline_publish_lead_ms=40,
        )
        is None
    )
