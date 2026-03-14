#!/usr/bin/env python3
"""Measure basic root stability during split sim2sim replay."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from holosoma_inference.utils.math.quat import quat_to_rpy, xyzw_to_wxyz
from holosoma_inference.utils.sim_state import SimStateSub


def collect_samples(
    *,
    port: int,
    duration: float,
    startup_timeout: float,
    idle_timeout: float,
    poll_hz: float,
) -> tuple[list[int], list[np.ndarray]]:
    sub = SimStateSub(port=port)
    sub.start()
    timestamps_ms: list[int] = []
    root_states: list[np.ndarray] = []
    first_sample_wall_time: float | None = None
    last_message_wall_time = time.monotonic()
    last_sim_time_ms: int | None = None
    poll_sleep = 1.0 / max(poll_hz, 1.0)

    try:
        while True:
            state = sub.get_state()
            now = time.monotonic()
            if state is not None:
                sim_time_ms = int(state.get("sim_time_ms", -1))
                robot_state = state.get("robot_root_state")
                if (
                    sim_time_ms >= 0
                    and robot_state is not None
                    and (last_sim_time_ms is None or sim_time_ms != last_sim_time_ms)
                ):
                    timestamps_ms.append(sim_time_ms)
                    root_states.append(np.asarray(robot_state, dtype=np.float64))
                    first_sample_wall_time = first_sample_wall_time or now
                    last_message_wall_time = now
                    last_sim_time_ms = sim_time_ms

            if first_sample_wall_time is None:
                if now - last_message_wall_time > startup_timeout:
                    raise TimeoutError(
                        f"No valid sim-state sample received within {startup_timeout:.1f}s on port {port}"
                    )
            else:
                if now - first_sample_wall_time >= duration:
                    break
                if now - last_message_wall_time > idle_timeout:
                    break

            time.sleep(poll_sleep)
    finally:
        sub.close()

    return timestamps_ms, root_states


def summarize_samples(
    timestamps_ms: list[int],
    root_states: list[np.ndarray],
    *,
    max_z_std: float,
    max_roll_deg: float,
    max_pitch_deg: float,
) -> dict[str, object]:
    if len(timestamps_ms) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(timestamps_ms)}")

    roots = np.stack(root_states, axis=0)
    pos = roots[:, :3]
    quat_xyzw = roots[:, 3:7]
    quat_wxyz = xyzw_to_wxyz(quat_xyzw)
    rpy = np.asarray([quat_to_rpy(q) for q in quat_wxyz], dtype=np.float64)
    rpy_deg = np.rad2deg(rpy)

    xy_disp_vec = pos[-1, :2] - pos[0, :2]
    z_values = pos[:, 2]
    roll_abs_max = float(np.max(np.abs(rpy_deg[:, 0])))
    pitch_abs_max = float(np.max(np.abs(rpy_deg[:, 1])))
    yaw_change = float(rpy_deg[-1, 2] - rpy_deg[0, 2])
    z_std = float(np.std(z_values))

    stable = z_std <= max_z_std and roll_abs_max <= max_roll_deg and pitch_abs_max <= max_pitch_deg

    return {
        "sample_count": len(timestamps_ms),
        "sim_duration_s": float((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0),
        "root_start_xyz": pos[0].round(6).tolist(),
        "root_end_xyz": pos[-1].round(6).tolist(),
        "xy_displacement": float(np.linalg.norm(xy_disp_vec)),
        "xy_displacement_vec": xy_disp_vec.round(6).tolist(),
        "z_mean": float(np.mean(z_values)),
        "z_std": z_std,
        "z_min": float(np.min(z_values)),
        "z_max": float(np.max(z_values)),
        "roll_abs_max_deg": roll_abs_max,
        "pitch_abs_max_deg": pitch_abs_max,
        "yaw_change_deg": yaw_change,
        "thresholds": {
            "max_z_std": max_z_std,
            "max_roll_deg": max_roll_deg,
            "max_pitch_deg": max_pitch_deg,
        },
        "stable_root_motion": stable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5557, help="ZMQ sim-state subscription port")
    parser.add_argument("--duration", type=float, default=12.0, help="Collection duration after first valid sample")
    parser.add_argument("--startup-timeout", type=float, default=45.0, help="Wait this long for first sample")
    parser.add_argument("--idle-timeout", type=float, default=1.5, help="Stop if no fresh sample arrives for this long")
    parser.add_argument("--poll-hz", type=float, default=200.0, help="Polling rate for the subscriber")
    parser.add_argument("--max-z-std", type=float, default=0.08, help="Maximum acceptable root height std-dev")
    parser.add_argument("--max-roll-deg", type=float, default=20.0, help="Maximum acceptable absolute roll angle")
    parser.add_argument("--max-pitch-deg", type=float, default=20.0, help="Maximum acceptable absolute pitch angle")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    timestamps_ms, root_states = collect_samples(
        port=args.port,
        duration=args.duration,
        startup_timeout=args.startup_timeout,
        idle_timeout=args.idle_timeout,
        poll_hz=args.poll_hz,
    )
    summary = summarize_samples(
        timestamps_ms,
        root_states,
        max_z_std=args.max_z_std,
        max_roll_deg=args.max_roll_deg,
        max_pitch_deg=args.max_pitch_deg,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
