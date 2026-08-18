#!/usr/bin/env python3
"""Render one audited MuJoCo rollout in the unified _check_vis trajectory layout."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np

from holosoma.record_mujoco_rollout_video import MujocoStateRenderer


WIDTH = 640
HEIGHT = 560
SCENE_TOP = 56
SCENE_BOTTOM = 360
PANEL_TOP = 360
CYAN = (235, 235, 20)
ORANGE = (0, 165, 255)
MAGENTA = (225, 70, 220)
WHITE = (235, 235, 235)
MUTED = (190, 190, 190)


def _put(
    image: np.ndarray,
    text: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.43,
    color: tuple[int, int, int] = WHITE,
    thickness: int = 1,
) -> None:
    cv2.putText(image, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def _load_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"No valid state rows in {path}")
    return rows


def _clip_label(clip_slug: str) -> tuple[str, str]:
    parts = clip_slug.split("_")
    if len(parts) < 4:
        return clip_slug, ""
    return f"{parts[-2]} {parts[-1]}", parts[0]


def _path_length(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.asarray([], dtype=np.float64)
    increments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(increments)])


def _draw_trajectory_panel(
    frame: np.ndarray,
    *,
    robot_xy: np.ndarray,
    object_xy: np.ndarray,
    robot_path: np.ndarray,
    object_path: np.ndarray,
    state_index: int,
    sim_time_s: float,
    rel_z: float,
    contacts: int,
    command: list[float],
    bounds: tuple[float, float, float, float],
) -> None:
    frame[PANEL_TOP:, :] = (22, 22, 22)
    graph = (14, 380, 169, 169)
    gx, gy, gw, gh = graph
    xmin, xmax, ymin, ymax = bounds
    cv2.rectangle(frame, (gx, gy), (gx + gw, gy + gh), (100, 100, 100), 1)
    for step in range(1, 5):
        px = gx + int(round(step * gw / 5.0))
        py = gy + int(round(step * gh / 5.0))
        cv2.line(frame, (px, gy), (px, gy + gh), (55, 55, 55), 1)
        cv2.line(frame, (gx, py), (gx + gw, py), (55, 55, 55), 1)

    def project(point: np.ndarray) -> tuple[int, int]:
        x = gx + int(np.clip((float(point[0]) - xmin) / (xmax - xmin), 0.0, 1.0) * gw)
        y = gy + gh - int(np.clip((float(point[1]) - ymin) / (ymax - ymin), 0.0, 1.0) * gh)
        return x, y

    stride = max(1, state_index // 500)
    robot_points = np.asarray([project(point) for point in robot_xy[: state_index + 1 : stride]], dtype=np.int32)
    object_points = np.asarray([project(point) for point in object_xy[: state_index + 1 : stride]], dtype=np.int32)
    if len(robot_points) >= 2:
        cv2.polylines(frame, [robot_points], False, CYAN, 3, cv2.LINE_AA)
    if len(object_points) >= 2:
        cv2.polylines(frame, [object_points], False, ORANGE, 3, cv2.LINE_AA)
    origin = project(np.zeros(2, dtype=np.float64))
    cv2.drawMarker(frame, origin, WHITE, cv2.MARKER_TILTED_CROSS, 9, 2, cv2.LINE_AA)
    cv2.circle(frame, project(robot_xy[state_index]), 4, CYAN, -1, cv2.LINE_AA)
    cv2.circle(frame, project(object_xy[state_index]), 4, ORANGE, -1, cv2.LINE_AA)
    _put(frame, "+X", (gx + gw - 25, gy + gh - 5), scale=0.34, color=MUTED)

    text_x = 201
    _put(frame, "TOP-DOWN TRAJECTORY  |  world XY, origin = G1 root at t0", (text_x, 390), scale=0.39)
    _put(frame, "Shared map bounds across this delivery batch", (text_x, 414), scale=0.37, color=MUTED)
    cv2.line(frame, (text_x, 435), (text_x + 36, 435), CYAN, 5, cv2.LINE_AA)
    _put(frame, "G1 root", (text_x + 45, 440), scale=0.40, color=CYAN)
    cv2.line(frame, (text_x + 150, 435), (text_x + 186, 435), ORANGE, 5, cv2.LINE_AA)
    _put(frame, "Object", (text_x + 195, 440), scale=0.40, color=ORANGE)
    _put(frame, f"time  {sim_time_s:5.2f} s", (text_x, 470), scale=0.43)
    rxy = robot_xy[state_index]
    oxy = object_xy[state_index]
    _put(
        frame,
        f"G1      dXY=({rxy[0]:+.2f}, {rxy[1]:+.2f}) m   path={robot_path[state_index]:.2f} m",
        (text_x, 495),
        scale=0.39,
        color=CYAN,
    )
    _put(
        frame,
        f"Object  dXY=({oxy[0]:+.2f}, {oxy[1]:+.2f}) m   path={object_path[state_index]:.2f} m",
        (text_x, 520),
        scale=0.39,
        color=ORANGE,
    )
    separation = float(np.linalg.norm(robot_xy[state_index] - object_xy[state_index]))
    cmd_x = float(command[0]) if command else 0.0
    _put(
        frame,
        f"G1-object XY={separation:.3f} m | rel_z={rel_z:.3f} m | contacts={contacts} | dx={cmd_x:.2f}",
        (text_x, 545),
        scale=0.36,
        color=MUTED,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--clip-slug", required=True)
    parser.add_argument("--source-label", choices=("corl79", "debug30"), default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--shared-bounds",
        type=float,
        nargs=4,
        default=(-0.48280534744262704, 6.1525414943695065, -6.123810565471649, 0.5115362763404847),
    )
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    states = _load_jsonl(run_dir / "audit/sim_state.jsonl")
    gate = _load_json(run_dir / "audit/gate_summary.json")
    audit = _load_json(run_dir / "audit/command_audit.json")
    if not bool(audit.get("passed", False)):
        raise ValueError(f"Refusing to render unaudited rollout: {run_dir}")

    origin = np.asarray(states[0]["robot_root_state"][:2], dtype=np.float64)
    robot_xy = np.asarray([state["robot_root_state"][:2] for state in states], dtype=np.float64) - origin
    object_xy = np.asarray([state["actors"]["object"][:2] for state in states], dtype=np.float64) - origin
    robot_path = _path_length(robot_xy)
    object_path = _path_length(object_xy)
    frame_count = int(audit["policy_io_rows"])
    if frame_count != 501:
        raise ValueError(f"Unified video contract requires 501 policy rows, got {frame_count}")

    raw_output = output.with_name(f".{output.stem}.raw.mp4")
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), float(args.fps), (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {raw_output}")

    renderer = MujocoStateRenderer(run_dir / "mujoco_scene.xml", width=WIDTH, height=360)
    object_label, scale_label = _clip_label(args.clip_slug)
    triggered = bool(audit.get("triggered", gate.get("triggered", False)))
    terminal_carry = bool(audit.get("terminal_carry", False))
    lift_threshold_m = float(gate["lift_rel_z_delta_m"])
    latest_forward_s = (
        None
        if gate.get("latest_forward_actor_sim_time_ms") is None
        else float(gate["latest_forward_actor_sim_time_ms"]) / 1000.0
    )
    first_forward_actor_s = (
        None
        if audit.get("first_forward_actor_sim_time_ms") is None
        else float(audit["first_forward_actor_sim_time_ms"]) / 1000.0
    )
    trigger_source = audit.get("trigger_source", gate.get("trigger_source"))
    status = (
        "NOT TRIGGERED"
        if not triggered
        else "TRIGGERED / TERMINAL CARRY"
        if terminal_carry
        else "TRIGGERED / TERMINAL LOSS"
    )
    try:
        for frame_no in range(frame_count):
            state_index = min(len(states) - 1, int(round(frame_no * (len(states) - 1) / (frame_count - 1))))
            state = states[state_index]
            scene = renderer.render(state)
            scene = cv2.resize(scene, (WIDTH, SCENE_BOTTOM - SCENE_TOP), interpolation=cv2.INTER_AREA)
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            frame[SCENE_TOP:SCENE_BOTTOM] = scene
            frame[:SCENE_TOP] = (48, 48, 48)
            _put(
                frame,
                (
                    f"0mc40K | {args.source_label or 'bank'} | {object_label} / {scale_label} | "
                    f"MuJoCo | +{lift_threshold_m:.2f}m"
                    + ("" if latest_forward_s is None else f" OR <= {latest_forward_s:.2f}s")
                    + " -> dx=.15"
                ),
                (12, 23),
                scale=0.48,
                color=MAGENTA,
                thickness=2,
            )
            _put(
                frame,
                (
                    f"actor forward {first_forward_actor_s:.2f}s via {trigger_source} | "
                    f"drop=0 all 501 actor steps | {status}"
                    if first_forward_actor_s is not None
                    else f"gate never reached | command stayed zero | drop=0 all 501 actor steps | {status}"
                ),
                (12, 47),
                scale=0.43,
                color=WHITE,
            )
            _draw_trajectory_panel(
                frame,
                robot_xy=robot_xy,
                object_xy=object_xy,
                robot_path=robot_path,
                object_path=object_path,
                state_index=state_index,
                sim_time_s=frame_no / float(args.fps),
                rel_z=float(state["observer_object_rel_z"]),
                contacts=int(state["object_robot_contact_count"]),
                command=list(state["observer_published_root_command"]),
                bounds=tuple(float(value) for value in args.shared_bounds),
            )
            writer.write(frame)
    finally:
        writer.release()
        renderer.close()

    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(raw_output),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ],
        check=True,
    )
    raw_output.unlink()

    metadata = {
        "run_dir": str(run_dir),
        "clip_slug": args.clip_slug,
        "video": str(output),
        "fps": float(args.fps),
        "frames": frame_count,
        "width": WIDTH,
        "height": HEIGHT,
        "shared_bounds": [float(value) for value in args.shared_bounds],
        "source_label": args.source_label,
        "triggered": triggered,
        "terminal_carry": terminal_carry,
        "rollout_status": audit.get("rollout_status", status.lower().replace(" / ", "_")),
        "trigger_sim_time_ms": (
            None if gate.get("trigger_sim_time_ms") is None else int(gate["trigger_sim_time_ms"])
        ),
        "terminal_object_rel_z": float(audit["terminal_object_rel_z"]),
        "terminal_object_robot_contact_count": int(audit["terminal_object_robot_contact_count"]),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
