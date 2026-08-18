#!/usr/bin/env python3
"""Fail-closed audit for a MuJoCo pure-forward-after-live-lift rollout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def _first(stats: dict[str, Any], size: int) -> np.ndarray:
    values = np.asarray(stats.get("first"), dtype=np.float64).reshape(-1)
    if values.size < size or not np.isfinite(values[:size]).all():
        raise ValueError(f"Invalid debug stats first-values: {stats}")
    return values[:size]


def _assert_close(actual: np.ndarray, expected: np.ndarray, *, label: str) -> None:
    if actual.shape != expected.shape or not np.allclose(actual, expected, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"{label}: expected {expected.tolist()}, got {actual.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-not-triggered",
        action="store_true",
        help="Accept a complete zero-command rollout that never reached the live lift gate.",
    )
    parser.add_argument(
        "--allow-terminal-loss",
        action="store_true",
        help="Accept a command-correct triggered rollout that is not carrying at the terminal step.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    summary = json.loads((run_dir / "audit/gate_summary.json").read_text(encoding="utf-8"))
    policy_rows = _load_jsonl(run_dir / "policy_io.jsonl")
    state_rows = _load_jsonl(run_dir / "audit/sim_state.jsonl")

    triggered = bool(summary.get("triggered"))
    trigger_source = summary.get("trigger_source")
    if triggered and trigger_source is None:
        trigger_source = "height"
    if trigger_source not in ({"height", "time_fallback"} if triggered else {None}):
        raise ValueError(f"Invalid trigger source: {trigger_source!r}")
    latest_forward_actor_sim_time_ms = summary.get("latest_forward_actor_sim_time_ms")
    if latest_forward_actor_sim_time_ms is not None:
        latest_forward_actor_sim_time_ms = float(latest_forward_actor_sim_time_ms)
        if not math.isfinite(latest_forward_actor_sim_time_ms) or latest_forward_actor_sim_time_ms <= 0:
            raise ValueError("Invalid latest actor-forward deadline")
    if len(policy_rows) != 501:
        raise ValueError(f"Expected 501 policy-I/O rows, got {len(policy_rows)}")
    counts = [row.get("count") for row in policy_rows]
    if counts != list(range(501)):
        raise ValueError("Policy-I/O counts are not contiguous 0..500")

    zero = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    forward = np.asarray([0.15, 0.0, 0.0], dtype=np.float64)
    policy_commands: list[np.ndarray] = []
    for index, row in enumerate(policy_rows):
        for finite_key in ("actor_obs", "perception_obs", "policy_action_raw", "policy_action_scaled"):
            stats = row[finite_key]
            if int(stats["finite"]) != int(stats["count"]):
                raise ValueError(f"Non-finite {finite_key} at policy row {index}")
        root_command = _first(row["sparse_target_root_trajectory_command"], 3)
        contact_aware_command = _first(
            row["sparse_target_root_trajectory_command_contact_aware"], 3
        )
        _assert_close(contact_aware_command, root_command, label=f"contact-aware root command row {index}")
        drop = _first(row["drop_button"], 1)
        _assert_close(drop, np.asarray([0.0]), label=f"drop button row {index}")
        actor_values = np.asarray(row["actor_obs_values"], dtype=np.float64).reshape(-1)
        if actor_values.size < 4 or not np.isfinite(actor_values).all():
            raise ValueError(f"Invalid actor_obs_values at policy row {index}")
        _assert_close(actor_values[:3], root_command, label=f"actor root command row {index}")
        _assert_close(actor_values[3:4], np.asarray([0.0]), label=f"actor drop input row {index}")
        policy_commands.append(root_command)

    forward_rows = [
        index for index, command in enumerate(policy_commands) if np.allclose(command, forward, rtol=0.0, atol=1.0e-6)
    ]
    first_forward_row: int | None = forward_rows[0] if forward_rows else None
    first_forward_actor_sim_time_ms = (
        None if first_forward_row is None else float(policy_rows[first_forward_row]["sim_time_ms"])
    )
    if triggered:
        if first_forward_row is None:
            raise ValueError("Triggered live lift gate but no forward command reached the actor")
        for index, command in enumerate(policy_commands[:first_forward_row]):
            _assert_close(command, zero, label=f"pre-gate actor command row {index}")
        for index, command in enumerate(policy_commands[first_forward_row:], start=first_forward_row):
            _assert_close(command, forward, label=f"post-gate actor command row {index}")
        if (
            latest_forward_actor_sim_time_ms is not None
            and first_forward_actor_sim_time_ms > latest_forward_actor_sim_time_ms
        ):
            raise ValueError(
                "Forward reached the actor after the hard deadline: "
                f"{first_forward_actor_sim_time_ms} > {latest_forward_actor_sim_time_ms} ms"
            )
    else:
        if first_forward_row is not None:
            raise ValueError("Forward command reached the actor without a triggered live lift gate")
        for index, command in enumerate(policy_commands):
            _assert_close(command, zero, label=f"not-triggered actor command row {index}")

    generations = {row.get("episode_generation") for row in state_rows}
    if generations != {summary["episode_generation"]}:
        raise ValueError(f"Episode generation changed: {generations}")
    threshold = float(summary["lift_rel_z_delta_m"])
    first_trigger_state: dict[str, Any] | None = None
    for index, row in enumerate(state_rows):
        rel_z = float(row["observer_object_rel_z"])
        command = np.asarray(row["observer_published_root_command"], dtype=np.float64)
        if float(row["observer_published_drop_button"]) != 0.0:
            raise ValueError(f"Observer published non-zero drop at state row {index}")
        observer_triggered = bool(
            row.get("observer_forward_triggered", row.get("observer_lift_triggered", False))
        )
        if observer_triggered:
            if first_trigger_state is None:
                first_trigger_state = row
            row_source = row.get("observer_forward_trigger_source", trigger_source)
            if row_source != trigger_source:
                raise ValueError(f"Trigger source changed at state row {index}: {row_source!r}")
            _assert_close(command, forward, label=f"post-gate observer command row {index}")
        else:
            if rel_z >= threshold:
                raise ValueError(f"State row {index} reached the gate without latching")
            _assert_close(command, zero, label=f"pre-gate observer command row {index}")
    if triggered:
        if first_trigger_state is None:
            raise ValueError("State observer never recorded a triggered gate")
        trigger_rel_z = float(first_trigger_state["observer_object_rel_z"])
        if trigger_source == "height" and trigger_rel_z < threshold:
            raise ValueError("Height-trigger state is below the configured relative-z threshold")
        if trigger_source == "time_fallback":
            if trigger_rel_z >= threshold:
                raise ValueError("Time fallback was used even though the height gate had been reached")
            publish_time_ms = float(summary["time_fallback_publish_sim_time_ms"])
            if float(first_trigger_state["sim_time_ms"]) < publish_time_ms:
                raise ValueError("Time fallback triggered before its configured publish time")
    elif first_trigger_state is not None:
        raise ValueError("State observer triggered despite a not-triggered gate summary")

    last_policy_time = float(policy_rows[-1]["sim_time_ms"])
    final_state = min(state_rows, key=lambda row: abs(float(row["sim_time_ms"]) - last_policy_time))
    trigger_state = (
        min(
            state_rows,
            key=lambda row: abs(float(row["sim_time_ms"]) - float(summary["trigger_sim_time_ms"])),
        )
        if triggered
        else None
    )
    initial_state = state_rows[0]
    final_rel_z = float(final_state["observer_object_rel_z"])
    terminal_contact_count = int(final_state["object_robot_contact_count"])
    terminal_carry = triggered and final_rel_z >= threshold and terminal_contact_count >= 1

    root_initial = np.asarray(initial_state["robot_root_state"][:3], dtype=np.float64)
    root_trigger = (
        np.asarray(trigger_state["robot_root_state"][:3], dtype=np.float64)
        if trigger_state is not None
        else None
    )
    root_final = np.asarray(final_state["robot_root_state"][:3], dtype=np.float64)
    object_initial = np.asarray(initial_state["actors"]["object"][:3], dtype=np.float64)
    object_trigger = (
        np.asarray(trigger_state["actors"]["object"][:3], dtype=np.float64)
        if trigger_state is not None
        else None
    )
    object_final = np.asarray(final_state["actors"]["object"][:3], dtype=np.float64)

    report = {
        "passed": True,
        "command_contract_passed": True,
        "triggered": triggered,
        "terminal_carry": terminal_carry,
        "rollout_status": (
            "not_triggered"
            if not triggered
            else "triggered_terminal_carry"
            if terminal_carry
            else "triggered_terminal_loss"
        ),
        "policy_io_rows": len(policy_rows),
        "state_rows": len(state_rows),
        "first_forward_actor_row": first_forward_row,
        "first_forward_actor_sim_time_ms": first_forward_actor_sim_time_ms,
        "latest_forward_actor_sim_time_ms": latest_forward_actor_sim_time_ms,
        "last_policy_sim_time_ms": last_policy_time,
        "trigger_source": trigger_source,
        "trigger_sim_time_ms": None if not triggered else int(summary["trigger_sim_time_ms"]),
        "trigger_object_rel_z": (
            None if first_trigger_state is None else float(first_trigger_state["observer_object_rel_z"])
        ),
        "lift_rel_z_delta_m": threshold,
        "max_object_rel_z": float(summary["max_object_rel_z"]),
        "terminal_object_rel_z": final_rel_z,
        "terminal_object_robot_contact_count": terminal_contact_count,
        "terminal_object_robot_contact_bodies": final_state["object_robot_contact_bodies"],
        "robot_xy_displacement_before_gate_m": (
            None if root_trigger is None else float(np.linalg.norm(root_trigger[:2] - root_initial[:2]))
        ),
        "robot_xy_displacement_after_gate_m": (
            None if root_trigger is None else float(np.linalg.norm(root_final[:2] - root_trigger[:2]))
        ),
        "object_xy_displacement_after_gate_m": (
            None if object_trigger is None else float(np.linalg.norm(object_final[:2] - object_trigger[:2]))
        ),
        "terminal_object_minus_robot_xyz": (object_final - root_final).tolist(),
        "pre_gate_actor_command": zero.tolist(),
        "post_gate_actor_command": forward.tolist(),
        "drop_input_all_zero": True,
        "episode_generation_constant": True,
    }
    output_path = run_dir / "audit/command_audit.json"
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not triggered and not args.allow_not_triggered:
        raise ValueError("Neither the live lift gate nor the actor deadline triggered")
    if triggered and not terminal_carry and not args.allow_terminal_loss:
        raise ValueError(
            "Object was not still carried at the terminal policy step: "
            f"rel_z={final_rel_z:.6f}, contacts={terminal_contact_count}"
        )


if __name__ == "__main__":
    main()
