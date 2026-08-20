from __future__ import annotations

import importlib.util
from pathlib import Path


def load_runner():
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "run_prism_trajectory_batch.py"
    )
    spec = importlib.util.spec_from_file_location("prism_batch_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_classify_reports_accepts_monotonic_gated_result() -> None:
    runner = load_runner()
    geometric = {
        "collision_feasible": True,
        "tracking_improved": True,
    }
    dynamics = [
        {
            "initial_linearized_defect_mean": 10.0,
            "final_nonlinear_defect_mean": 9.0,
            "final_collision_violation_m": 1e-6,
            "collision_violation_limit_m": 2e-6,
            "final_qvel_consistency_max": 4.0,
            "accepted_step_size": 0.25,
        }
    ]
    assert runner.classify_reports(
        geometric,
        dynamics,
        maximum_qvel_consistency=5.0,
    ) == ("accepted", [])


def test_classify_reports_preserves_geometric_only_result() -> None:
    runner = load_runner()
    geometric = {
        "collision_feasible": True,
        "tracking_improved": True,
    }
    dynamics = [
        {
            "initial_linearized_defect_mean": 10.0,
            "final_nonlinear_defect_mean": 10.0,
            "final_collision_violation_m": 1e-6,
            "collision_violation_limit_m": 2e-6,
            "final_qvel_consistency_max": 4.0,
            "accepted_step_size": 0.0,
        }
    ]
    state, reasons = runner.classify_reports(
        geometric,
        dynamics,
        maximum_qvel_consistency=5.0,
    )
    assert state == "geometric_only"
    assert reasons == ["no_dynamics_step_accepted"]


def test_geometric_gate_reasons_names_failed_gates() -> None:
    runner = load_runner()
    assert runner.geometric_gate_reasons(
        {
            "collision_feasible": False,
            "tracking_improved": True,
        }
    ) == ["geometric_collision_gate_failed"]
