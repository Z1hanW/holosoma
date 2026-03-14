#!/usr/bin/env python3
"""Measure whether an object roughly co-moves with the robot in split sim2sim."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from holosoma_inference.utils.sim_state import SimStateSub


def _safe_cosine(vec_a: np.ndarray, vec_b: np.ndarray) -> float | None:
    denom = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    if denom <= 1e-8:
        return None
    return float(np.dot(vec_a, vec_b) / denom)


def _to_xy(state: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
    arr = np.asarray(state, dtype=np.float64)
    if arr.shape[0] < 2:
        raise ValueError(f"Expected at least 2 values, got shape {arr.shape}")
    return arr[:2]


def collect_samples(
    *,
    port: int,
    object_name: str,
    duration: float,
    startup_timeout: float,
    idle_timeout: float,
    poll_hz: float,
) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    sub = SimStateSub(port=port)
    sub.start()
    timestamps_ms: list[int] = []
    robot_xy: list[np.ndarray] = []
    object_xy: list[np.ndarray] = []
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
                actors = state.get("actors", {})
                actor_state = actors.get(object_name)
                robot_state = state.get("robot_root_state")
                if (
                    sim_time_ms >= 0
                    and robot_state is not None
                    and actor_state is not None
                    and (last_sim_time_ms is None or sim_time_ms != last_sim_time_ms)
                ):
                    timestamps_ms.append(sim_time_ms)
                    robot_xy.append(_to_xy(robot_state))
                    object_xy.append(_to_xy(actor_state))
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

    return timestamps_ms, robot_xy, object_xy


def summarize_samples(
    timestamps_ms: list[int],
    robot_xy: list[np.ndarray],
    object_xy: list[np.ndarray],
    *,
    min_robot_disp: float,
    min_object_disp: float,
    max_relative_drift: float,
    min_cosine: float,
) -> dict[str, object]:
    if len(timestamps_ms) < 2:
        raise ValueError(f"Need at least 2 samples, got {len(timestamps_ms)}")

    robot_arr = np.stack(robot_xy, axis=0)
    object_arr = np.stack(object_xy, axis=0)
    relative_xy = object_arr - robot_arr

    robot_disp_vec = robot_arr[-1] - robot_arr[0]
    object_disp_vec = object_arr[-1] - object_arr[0]
    vector_gap = object_disp_vec - robot_disp_vec
    relative_drift = relative_xy - relative_xy[0]
    relative_drift_norms = np.linalg.norm(relative_drift, axis=1)

    robot_disp = float(np.linalg.norm(robot_disp_vec))
    object_disp = float(np.linalg.norm(object_disp_vec))
    cosine = _safe_cosine(robot_disp_vec, object_disp_vec)
    vector_gap_norm = float(np.linalg.norm(vector_gap))
    max_rel_drift = float(np.max(relative_drift_norms))
    mean_rel_drift = float(np.mean(relative_drift_norms))
    offset_start = relative_xy[0]
    offset_end = relative_xy[-1]

    moves_together = (
        robot_disp >= min_robot_disp
        and object_disp >= min_object_disp
        and (cosine is not None and cosine >= min_cosine)
        and max_rel_drift <= max_relative_drift
    )

    return {
        "sample_count": len(timestamps_ms),
        "sim_duration_s": float((timestamps_ms[-1] - timestamps_ms[0]) / 1000.0),
        "robot_start_xy": robot_arr[0].round(6).tolist(),
        "robot_end_xy": robot_arr[-1].round(6).tolist(),
        "object_start_xy": object_arr[0].round(6).tolist(),
        "object_end_xy": object_arr[-1].round(6).tolist(),
        "robot_disp_xy": robot_disp_vec.round(6).tolist(),
        "object_disp_xy": object_disp_vec.round(6).tolist(),
        "robot_disp_norm": robot_disp,
        "object_disp_norm": object_disp,
        "disp_cosine": cosine,
        "disp_vector_gap_xy": vector_gap.round(6).tolist(),
        "disp_vector_gap_norm": vector_gap_norm,
        "relative_offset_start_xy": offset_start.round(6).tolist(),
        "relative_offset_end_xy": offset_end.round(6).tolist(),
        "relative_drift_max": max_rel_drift,
        "relative_drift_mean": mean_rel_drift,
        "thresholds": {
            "min_robot_disp": min_robot_disp,
            "min_object_disp": min_object_disp,
            "max_relative_drift": max_relative_drift,
            "min_cosine": min_cosine,
        },
        "moves_together": moves_together,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5557, help="ZMQ sim-state subscription port")
    parser.add_argument("--object-name", default="object", help="Actor name for the carried object")
    parser.add_argument("--duration", type=float, default=12.0, help="Collection duration after first valid sample")
    parser.add_argument("--startup-timeout", type=float, default=45.0, help="Wait this long for first sample")
    parser.add_argument("--idle-timeout", type=float, default=1.5, help="Stop if no fresh sample arrives for this long")
    parser.add_argument("--poll-hz", type=float, default=200.0, help="Polling rate for the subscriber")
    parser.add_argument("--min-robot-disp", type=float, default=0.10, help="Minimum robot XY travel for a valid check")
    parser.add_argument("--min-object-disp", type=float, default=0.10, help="Minimum object XY travel for a valid check")
    parser.add_argument(
        "--max-relative-drift",
        type=float,
        default=0.35,
        help="Maximum allowed change in object-vs-robot XY offset",
    )
    parser.add_argument(
        "--min-cosine",
        type=float,
        default=0.70,
        help="Minimum cosine similarity between robot/object displacement vectors",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    timestamps_ms, robot_xy, object_xy = collect_samples(
        port=args.port,
        object_name=args.object_name,
        duration=args.duration,
        startup_timeout=args.startup_timeout,
        idle_timeout=args.idle_timeout,
        poll_hz=args.poll_hz,
    )
    summary = summarize_samples(
        timestamps_ms,
        robot_xy,
        object_xy,
        min_robot_disp=args.min_robot_disp,
        min_object_disp=args.min_object_disp,
        max_relative_drift=args.max_relative_drift,
        min_cosine=args.min_cosine,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
