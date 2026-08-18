#!/usr/bin/env python3
"""Run and audit the pure-forward-after-lift command gate in split MuJoCo."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from holosoma_inference.utils.policy_overlay import PolicyOverlaySub
from holosoma_inference.utils.sim_control import (
    ManualRootCommandPub,
    PolicyControlPush,
)
from holosoma_inference.utils.sim_state import SimStateSub


def _object_z(state: dict[str, Any] | None, object_name: str) -> float | None:
    if not isinstance(state, dict):
        return None
    actors = state.get("actors")
    if not isinstance(actors, dict) or not actors:
        return None
    actor = actors.get(object_name)
    if actor is None and len(actors) == 1:
        actor = next(iter(actors.values()))
    try:
        values = np.asarray(actor, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if values.size < 3 or not math.isfinite(float(values[2])):
        return None
    return float(values[2])


def _episode_generation(state: dict[str, Any] | None) -> int | None:
    if not isinstance(state, dict):
        return None
    value = state.get("episode_generation")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _trigger_source(
    *,
    rel_z: float | None,
    sim_time_ms: int | None,
    lift_rel_z_delta_m: float,
    latest_forward_actor_sim_time_ms: int | None,
    deadline_publish_lead_ms: int,
) -> str | None:
    """Choose the first enabled trigger, preferring a real lift at the boundary."""
    if rel_z is not None and rel_z >= lift_rel_z_delta_m:
        return "height"
    if latest_forward_actor_sim_time_ms is None or sim_time_ms is None:
        return None
    publish_deadline_ms = latest_forward_actor_sim_time_ms - deadline_publish_lead_ms
    if sim_time_ms >= publish_deadline_ms:
        return "time_fallback"
    return None


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-port", type=int, required=True)
    parser.add_argument("--sparse-root-command-port", type=int, required=True)
    parser.add_argument("--policy-control-port", type=int, required=True)
    parser.add_argument("--policy-overlay-port", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--object-name", default="object")
    parser.add_argument("--forward-command-m", type=float, default=0.15)
    parser.add_argument("--lift-rel-z-delta-m", type=float, default=0.30)
    parser.add_argument(
        "--latest-forward-actor-sim-time-s",
        type=float,
        default=None,
        help="Optional hard deadline for the first forward command visible to the actor.",
    )
    parser.add_argument(
        "--deadline-publish-lead-ms",
        type=int,
        default=40,
        help="Publish this many milliseconds before the actor-input deadline.",
    )
    parser.add_argument("--actor-steps", type=int, default=501)
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--rollout-timeout-s", type=float, default=90.0)
    parser.add_argument("--publish-rate-hz", type=float, default=200.0)
    parser.add_argument("--trigger-gap-s", type=float, default=0.6)
    args = parser.parse_args()

    if not math.isfinite(args.forward_command_m) or args.forward_command_m <= 0.0:
        raise ValueError("--forward-command-m must be finite and positive")
    if not math.isfinite(args.lift_rel_z_delta_m) or args.lift_rel_z_delta_m <= 0.0:
        raise ValueError("--lift-rel-z-delta-m must be finite and positive")
    if args.latest_forward_actor_sim_time_s is not None and (
        not math.isfinite(args.latest_forward_actor_sim_time_s)
        or args.latest_forward_actor_sim_time_s <= 0.0
    ):
        raise ValueError("--latest-forward-actor-sim-time-s must be finite and positive")
    if args.deadline_publish_lead_ms < 0:
        raise ValueError("--deadline-publish-lead-ms must be non-negative")
    if args.actor_steps < 1:
        raise ValueError("--actor-steps must be positive")

    latest_forward_actor_sim_time_ms = (
        None
        if args.latest_forward_actor_sim_time_s is None
        else int(round(float(args.latest_forward_actor_sim_time_s) * 1000.0))
    )
    if (
        latest_forward_actor_sim_time_ms is not None
        and args.deadline_publish_lead_ms >= latest_forward_actor_sim_time_ms
    ):
        raise ValueError("--deadline-publish-lead-ms must be smaller than the actor deadline")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "sim_state.jsonl"
    overlay_path = output_dir / "policy_overlay.jsonl"
    summary_path = output_dir / "gate_summary.json"
    state_path.write_text("", encoding="utf-8")
    overlay_path.write_text("", encoding="utf-8")

    command_pub = ManualRootCommandPub(port=args.sparse_root_command_port)
    policy_control = PolicyControlPush(port=args.policy_control_port)
    state_sub = SimStateSub(port=args.state_port)
    overlay_sub = PolicyOverlaySub(port=args.policy_overlay_port)
    command_pub.start()
    policy_control.start()
    state_sub.start()
    overlay_sub.start()

    pre_command = (0.0, 0.0, 0.0)
    post_command = (float(args.forward_command_m), 0.0, 0.0)
    current_command = pre_command
    started_at = time.monotonic()
    ready_deadline = started_at + float(args.startup_timeout_s)
    state: dict[str, Any] | None = None
    overlay: dict[str, Any] | None = None
    try:
        while time.monotonic() < ready_deadline:
            command_pub.publish(
                enabled=True,
                mode="manual",
                command=current_command,
                pickup_button=None,
                drop_button=0.0,
            )
            state = state_sub.get_state()
            overlay = overlay_sub.get_payload()
            if _object_z(state, args.object_name) is not None and isinstance(overlay, dict):
                break
            time.sleep(0.01)
        else:
            raise TimeoutError(
                "Timed out waiting for live MuJoCo object state and policy overlay "
                f"(state_ready={_object_z(state, args.object_name) is not None}, "
                f"overlay_ready={isinstance(overlay, dict)})"
            )

        initial_z = _object_z(state, args.object_name)
        assert initial_z is not None
        generation = _episode_generation(state)
        baseline_generation = generation

        # Keep the zero/drop=0 payload alive across the PUB/SUB slow-join window,
        # then start the checkpoint-native pickup rollout.
        settle_deadline = time.monotonic() + 1.0
        while time.monotonic() < settle_deadline:
            command_pub.publish(
                enabled=True,
                mode="manual",
                command=pre_command,
                pickup_button=None,
                drop_button=0.0,
            )
            time.sleep(0.01)
        if not policy_control.publish("space", source="mj_forward_after_lift_rollout"):
            raise RuntimeError("Failed to publish policy motion-start command")
        time.sleep(max(float(args.trigger_gap_s), 0.0))
        if not policy_control.publish("start", source="mj_forward_after_lift_rollout"):
            raise RuntimeError("Failed to publish policy actor-start command")

        deadline = time.monotonic() + float(args.rollout_timeout_s)
        latched = False
        trigger_source: str | None = None
        trigger_sim_time_ms: int | None = None
        trigger_object_z: float | None = None
        max_rel_z = 0.0
        state_rows = 0
        overlay_rows = 0
        last_state_key: tuple[int | None, int | None] | None = None
        last_overlay_key: tuple[Any, ...] | None = None
        max_motion_timestep = -1
        max_policy_step_count = 0

        while time.monotonic() < deadline:
            loop_started = time.monotonic()
            state = state_sub.get_state()
            overlay = overlay_sub.get_payload()

            current_generation = _episode_generation(state)
            object_z = _object_z(state, args.object_name)
            if current_generation != baseline_generation and object_z is not None:
                raise RuntimeError(
                    "MuJoCo episode generation changed during the fixed rollout; "
                    f"baseline={baseline_generation}, current={current_generation}"
                )

            rel_z = None if object_z is None else object_z - initial_z
            if rel_z is not None:
                max_rel_z = max(max_rel_z, rel_z)
            state_sim_time_ms = (
                int(state["sim_time_ms"])
                if isinstance(state, dict)
                and isinstance(state.get("sim_time_ms"), int)
                and not isinstance(state.get("sim_time_ms"), bool)
                else None
            )
            if not latched:
                selected_source = _trigger_source(
                    rel_z=rel_z,
                    sim_time_ms=state_sim_time_ms,
                    lift_rel_z_delta_m=float(args.lift_rel_z_delta_m),
                    latest_forward_actor_sim_time_ms=latest_forward_actor_sim_time_ms,
                    deadline_publish_lead_ms=int(args.deadline_publish_lead_ms),
                )
                if selected_source is not None:
                    latched = True
                    trigger_source = selected_source
                    current_command = post_command
                    trigger_object_z = object_z
                    trigger_sim_time_ms = state_sim_time_ms

            command_pub.publish(
                enabled=True,
                mode="manual",
                command=current_command,
                pickup_button=None,
                drop_button=0.0,
            )

            if isinstance(state, dict):
                state_key = (current_generation, state.get("sim_time_ms"))
                if state_key != last_state_key:
                    state_row = dict(state)
                    state_row.update(
                        {
                            "observer_initial_object_z": initial_z,
                            "observer_object_rel_z": rel_z,
                            "observer_lift_triggered": latched,
                            "observer_forward_triggered": latched,
                            "observer_forward_trigger_source": trigger_source,
                            "observer_published_root_command": list(current_command),
                            "observer_published_pickup_button": None,
                            "observer_published_drop_button": 0.0,
                        }
                    )
                    _append_jsonl(
                        state_path,
                        state_row,
                    )
                    state_rows += 1
                    last_state_key = state_key

            if isinstance(overlay, dict):
                overlay_key = (
                    overlay.get("clip_active"),
                    overlay.get("policy_step_count"),
                    overlay.get("motion_timestep"),
                    tuple(overlay.get("sparse_effective_command") or ()),
                    overlay.get("sparse_command_source"),
                )
                if overlay_key != last_overlay_key:
                    row = dict(overlay)
                    row["gate_latched_at_observer"] = latched
                    row["forward_trigger_source_at_observer"] = trigger_source
                    row["observer_published_root_command"] = list(current_command)
                    row["observer_published_drop_button"] = 0.0
                    _append_jsonl(overlay_path, row)
                    overlay_rows += 1
                    last_overlay_key = overlay_key
                motion_timestep = overlay.get("motion_timestep")
                if isinstance(motion_timestep, int):
                    max_motion_timestep = max(max_motion_timestep, motion_timestep)
                policy_step_count = overlay.get("policy_step_count")
                if isinstance(policy_step_count, int):
                    max_policy_step_count = max(max_policy_step_count, policy_step_count)
                    if policy_step_count >= args.actor_steps:
                        break

            sleep_s = 1.0 / float(args.publish_rate_hz) - (time.monotonic() - loop_started)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
        else:
            raise TimeoutError(
                f"Rollout did not reach actor step {args.actor_steps - 1}; "
                f"max_policy_step_count={max_policy_step_count}, "
                f"max_motion_timestep={max_motion_timestep}"
            )

        summary = {
            "semantics": (
                "legacy_constant_robot_heading_frame_pure_forward_after_live_object_lift"
                if latest_forward_actor_sim_time_ms is None
                else "legacy_constant_robot_heading_frame_pure_forward_after_lift_or_actor_deadline"
            ),
            "object_name": args.object_name,
            "initial_object_z": initial_z,
            "lift_rel_z_delta_m": float(args.lift_rel_z_delta_m),
            "forward_command": list(post_command),
            "pre_lift_command": list(pre_command),
            "pickup_button_override": None,
            "drop_button_override": 0.0,
            "consecutive_steps": 0,
            "heading_lock": False,
            "triggered": latched,
            "trigger_source": trigger_source,
            "trigger_sim_time_ms": trigger_sim_time_ms,
            "trigger_object_z": trigger_object_z,
            "latest_forward_actor_sim_time_ms": latest_forward_actor_sim_time_ms,
            "deadline_publish_lead_ms": int(args.deadline_publish_lead_ms),
            "time_fallback_publish_sim_time_ms": (
                None
                if latest_forward_actor_sim_time_ms is None
                else latest_forward_actor_sim_time_ms - int(args.deadline_publish_lead_ms)
            ),
            "max_object_rel_z": max_rel_z,
            "max_motion_timestep": max_motion_timestep,
            "max_policy_step_count": max_policy_step_count,
            "state_rows": state_rows,
            "overlay_rows": overlay_rows,
            "episode_generation": baseline_generation,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
        if not latched:
            raise RuntimeError(
                "Neither the +{:.3f} m lift gate nor the configured actor deadline triggered; "
                "forward command was not sent".format(args.lift_rel_z_delta_m)
            )
    finally:
        command_pub.publish(
            enabled=True,
            mode="manual",
            command=pre_command,
            pickup_button=None,
            drop_button=0.0,
        )
        command_pub.close()
        policy_control.close()
        state_sub.close()
        overlay_sub.close()


if __name__ == "__main__":
    main()
