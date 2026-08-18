#!/usr/bin/env python3
"""Render, validate, and package an exact audited MuJoCo policy-rollout batch."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"No valid rows: {path}")
    return rows


def write_json_atomic(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def video_probe(path: Path, *, width: int, height: int, decode: bool = True) -> dict[str, Any]:
    payload = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    streams = payload.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or audios:
        raise ValueError(f"Expected one video and no audio streams: {path}")
    stream = videos[0]
    result = {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": stream.get("avg_frame_rate"),
        "frames": int(stream.get("nb_read_frames", 0)),
        "duration_s": float(payload.get("format", {}).get("duration", 0.0)),
        "bytes": path.stat().st_size,
    }
    expected = {"codec": "h264", "width": width, "height": height, "fps": "50/1", "frames": 501}
    for key, expected_value in expected.items():
        if result[key] != expected_value:
            raise ValueError(f"{key}: expected {expected_value}, got {result[key]}: {path}")
    if abs(result["duration_s"] - 10.02) > 1.0e-6:
        raise ValueError(f"duration: expected 10.02, got {result['duration_s']}: {path}")
    if decode:
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)
    return result


def compute_shared_bounds(audit_root: Path, input_rows: list[dict[str, Any]]) -> list[float]:
    points: list[np.ndarray] = []
    for row in input_rows:
        run_dir = audit_root / "runs" / f"{int(row['index']):03d}_{row['clip_slug']}"
        states = load_jsonl(run_dir / "audit/sim_state.jsonl")
        origin = np.asarray(states[0]["robot_root_state"][:2], dtype=np.float64)
        robot = np.asarray([state["robot_root_state"][:2] for state in states], dtype=np.float64) - origin
        obj = np.asarray([state["actors"]["object"][:2] for state in states], dtype=np.float64) - origin
        if not np.isfinite(robot).all() or not np.isfinite(obj).all():
            raise ValueError(f"Non-finite XY trajectory: {run_dir}")
        points.extend((robot, obj))
    all_points = np.concatenate(points, axis=0)
    minimum = np.min(all_points, axis=0)
    maximum = np.max(all_points, axis=0)
    center = 0.5 * (minimum + maximum)
    side = max(float(np.max(maximum - minimum)), 1.0) * 1.10
    half = 0.5 * side
    return [float(center[0] - half), float(center[0] + half), float(center[1] - half), float(center[1] + half)]


def build_grid(paths: list[Path], output: Path, *, columns: int) -> dict[str, Any]:
    tile_width, tile_height = 320, 280
    rows = math.ceil(len(paths) / columns)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for path in paths:
        command.extend(["-i", str(path)])
    filters = [f"[{index}:v]scale={tile_width}:{tile_height}:flags=lanczos[v{index}]" for index in range(len(paths))]
    layout = "|".join(
        f"{(index % columns) * tile_width}_{(index // columns) * tile_height}"
        for index in range(len(paths))
    )
    inputs = "".join(f"[v{index}]" for index in range(len(paths)))
    filters.append(f"{inputs}xstack=inputs={len(paths)}:layout={layout}:fill=black[v]")
    command.extend(
        [
            "-filter_complex",
            ";".join(filters),
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ]
    )
    subprocess.run(command, check=True)
    return video_probe(output, width=columns * tile_width, height=rows * tile_height)


def read_frame(path: Path, frame_number: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ok, frame = capture.read()
        if not ok or frame is None:
            raise ValueError(f"Could not read frame {frame_number}: {path}")
        return frame
    finally:
        capture.release()


def make_review(masters: list[Path], output: Path) -> None:
    target_width, target_height = 1600, 1120
    panels: list[np.ndarray] = []
    for frame_number in (0, 167, 334, 500):
        for master in masters:
            frame = read_frame(master, frame_number)
            scale = min(target_width / frame.shape[1], target_height / frame.shape[0])
            resized = cv2.resize(
                frame,
                (max(1, int(round(frame.shape[1] * scale))), max(1, int(round(frame.shape[0] * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            x = (target_width - resized.shape[1]) // 2
            y = (target_height - resized.shape[0]) // 2
            canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
            cv2.putText(
                canvas,
                f"frame {frame_number} / t={frame_number / 50.0:.2f}s",
                (18, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            panels.append(canvas)
    review = np.vstack([np.hstack(panels[index : index + 2]) for index in range(0, len(panels), 2)])
    if not cv2.imwrite(str(output), review):
        raise RuntimeError(f"Could not write review image: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument(
        "--subset-label",
        default=None,
        help="Package a selected input-manifest subset into one labeled master.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    audit_root = args.audit_root.expanduser().resolve()
    stage_root = args.stage_root.expanduser().resolve()
    input_manifest = load_json(audit_root / "input_manifest.json")
    batch_manifest = load_json(audit_root / "batch_rollout_manifest.json")
    input_rows = input_manifest.get("rows")
    if not isinstance(input_rows, list) or not input_rows:
        raise ValueError("Input manifest has no selected clip rows")
    expected_count = len(input_rows)
    if args.subset_label is None:
        if expected_count != 109:
            raise ValueError("Full-bank packaging requires the exact 109-clip bank")
        if not batch_manifest.get("all_109_command_audits_passed"):
            raise ValueError("All 109 command audits must pass before rendering")
    elif not batch_manifest.get("all_selected_command_audits_passed"):
        raise ValueError("All selected command audits must pass before rendering")
    gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    if args.workers < 1 or len(gpu_ids) < args.workers:
        raise ValueError("Invalid workers/GPU assignment")

    stage_root.mkdir(parents=True, exist_ok=True)
    individual_root = stage_root / "individual"
    master_root = stage_root / "masters"
    review_root = stage_root / "review"
    individual_root.mkdir(exist_ok=True)
    master_root.mkdir(exist_ok=True)
    review_root.mkdir(exist_ok=True)

    bounds_path = stage_root / "shared_xy_bounds.json"
    if bounds_path.exists():
        shared_bounds = load_json(bounds_path)["bounds"]
    else:
        shared_bounds = compute_shared_bounds(audit_root, input_rows)
        write_json_atomic(
            bounds_path,
            {"bounds": shared_bounds, "semantics": f"square_global_bounds_all_{expected_count}"},
        )

    progress_lock = threading.Lock()
    rendered_count = 0

    def render_one(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal rendered_count
        index = int(row["index"])
        run_dir = audit_root / "runs" / f"{index:03d}_{row['clip_slug']}"
        audit = load_json(run_dir / "audit/command_audit.json")
        if not audit.get("passed"):
            raise ValueError(f"Command audit did not pass: {run_dir}")
        output = individual_root / (
            f"{index:03d}_{row['source_label']}__{row['clip_slug']}__"
            "0mcqao8k_model40000__mujoco_forward015__zoomout_xy_trajectory.mp4"
        )
        try:
            probe = video_probe(output, width=640, height=560, decode=False)
            reused = True
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError):
            env = os.environ.copy()
            env.update(
                {
                    "MUJOCO_GL": "egl",
                    "CUDA_VISIBLE_DEVICES": gpu_ids[(index - 1) % args.workers],
                    "HOLOSOMA_RECORD_HIDE_WRIST_YAW_CYLINDERS": "0",
                    "PYTHONPATH": f"{repo_root / 'src/holosoma'}{os.pathsep}{env.get('PYTHONPATH', '')}",
                }
            )
            subprocess.run(
                [
                    "/data/ubuntu/conda-envs/dexjoco/bin/python",
                    str(repo_root / "scripts/render_mujoco_xy_trajectory_video.py"),
                    "--run-dir",
                    str(run_dir),
                    "--clip-slug",
                    str(row["clip_slug"]),
                    "--source-label",
                    str(row["source_label"]),
                    "--shared-bounds",
                    *(str(value) for value in shared_bounds),
                    "--output",
                    str(output),
                ],
                cwd=repo_root,
                env=env,
                check=True,
            )
            probe = video_probe(output, width=640, height=560)
            reused = False
        with progress_lock:
            rendered_count += 1
            print(
                f"[render {rendered_count:03d}/{expected_count:03d}] {row['clip_id']}",
                flush=True,
            )
        return {
            "index": index,
            "clip_id": row["clip_id"],
            "clip_slug": row["clip_slug"],
            "source_label": row["source_label"],
            "category": row["category"],
            "rollout_status": audit["rollout_status"],
            "triggered": audit["triggered"],
            "trigger_source": audit.get("trigger_source"),
            "first_forward_actor_sim_time_ms": audit.get("first_forward_actor_sim_time_ms"),
            "terminal_carry": audit["terminal_carry"],
            "path": str(output),
            "sha256": sha256(output),
            "video": probe,
            "reused": reused,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        videos = list(executor.map(render_one, input_rows))
    videos.sort(key=lambda item: int(item["index"]))

    masters: list[dict[str, Any]] = []
    if args.subset_label is None:
        master_specs = [
            ("corl79", 10, [row for row in videos if row["source_label"] == "corl79"]),
            ("debug30", 6, [row for row in videos if row["source_label"] == "debug30"]),
        ]
    else:
        master_specs = [(args.subset_label, 5, videos)]
    for source_label, columns, source_rows in master_specs:
        source_videos = [Path(row["path"]) for row in source_rows]
        if not source_videos:
            raise ValueError(f"Master group is empty: {source_label}")
        output = master_root / (
            f"00_master__{source_label}__0mcqao8k_model40000__mujoco_post_lift_"
            "persistent_forward015__all_clips.mp4"
        )
        rows = math.ceil(len(source_videos) / columns)
        try:
            probe = video_probe(output, width=columns * 320, height=rows * 280)
        except (FileNotFoundError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError):
            probe = build_grid(source_videos, output, columns=columns)
        masters.append(
            {
                "source_label": source_label,
                "clip_count": len(source_videos),
                "path": str(output),
                "sha256": sha256(output),
                "video": probe,
            }
        )

    review_stem = "full109" if args.subset_label is None else args.subset_label
    review_path = review_root / f"{review_stem}_master_review_4_times.png"
    make_review([Path(row["path"]) for row in masters], review_path)
    status_counts = Counter(str(row["rollout_status"]) for row in videos)
    trigger_source_counts = Counter(str(row["trigger_source"]) for row in videos)
    source_status_counts: dict[str, dict[str, int]] = {}
    for source in sorted({str(row["source_label"]) for row in videos}):
        source_status_counts[source] = dict(
            Counter(str(row["rollout_status"]) for row in videos if row["source_label"] == source)
        )
    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "semantics": input_manifest["semantics"],
        "backend": "mujoco",
        "mujoco_version": "3.4.0",
        "is_policy_rollout": True,
        "is_reference_motion_replay": False,
        "audit_root": str(audit_root),
        "input_manifest_sha256": sha256(audit_root / "input_manifest.json"),
        "batch_rollout_manifest_sha256": sha256(audit_root / "batch_rollout_manifest.json"),
        "bank_manifest_sha256": input_manifest["bank_manifest_sha256"],
        "object_map_sha256": input_manifest["object_map_sha256"],
        "model_onnx_sha256": input_manifest["model_onnx_sha256"],
        "clip_count": len(videos),
        "source_counts": input_manifest["source_counts"],
        "category_counts": input_manifest["category_counts"],
        "status_counts": dict(status_counts),
        "trigger_source_counts": dict(trigger_source_counts),
        "source_status_counts": source_status_counts,
        "command_contract": {
            "pre_lift_root_command": [0.0, 0.0, 0.0],
            "lift_relative_world_z_threshold_m": float(input_manifest["lift_rel_z_delta_m"]),
            "latest_forward_actor_sim_time_s": input_manifest.get(
                "latest_forward_actor_sim_time_s"
            ),
            "deadline_publish_lead_ms": input_manifest.get("deadline_publish_lead_ms"),
            "consecutive_steps": 0,
            "post_lift_actor_input": [0.15, 0.0, 0.0, 0.0],
            "drop_input_all_steps": 0.0,
            "preserve_native_pickup": True,
            "heading_lock": False,
            "semantics": "legacy_constant_robot_heading_frame",
        },
        "video_contract": {
            "individual_count": expected_count,
            "individual_width": 640,
            "individual_height": 560,
            "fps": 50,
            "frames": 501,
            "duration_s": 10.02,
            "shared_xy_bounds": shared_bounds,
            "ffprobe_all_passed": True,
            "full_decode_all_passed": True,
        },
        "individual_videos": videos,
        "masters": masters,
        "review_image": str(review_path),
        "manual_visual_review": "pending",
    }
    manifest_path = stage_root / "delivery_manifest.json"
    write_json_atomic(manifest_path, manifest)

    threshold_m = float(input_manifest["lift_rel_z_delta_m"])
    latest_forward_s = input_manifest.get("latest_forward_actor_sim_time_s")
    command_line = (
        f"Command: zero before the live +{threshold_m:.2f} m lift gate; persistent "
        "`[0.15, 0, 0, 0]` after it"
        if latest_forward_s is None
        else f"Command: zero until the live +{threshold_m:.2f} m lift gate or the time fallback; "
        f"the actor sees persistent `[0.15, 0, 0, 0]` no later than {float(latest_forward_s):.2f} s"
    )
    title = (
        "MuJoCo CORL79 + debug30 full-bank rollout"
        if args.subset_label is None
        else f"MuJoCo supplemental rollout: {args.subset_label}"
    )
    bank_line = (
        "Exact bank: CORL79 79 + debug30 30 = 109 clips"
        if args.subset_label is None
        else f"Selected subset: {expected_count} clips from the immutable CORL79 + debug30 bank"
    )
    readme = f"""# {title}

- Policy: `0mcqao8k/model_40000.onnx`
- Backend: MuJoCo 3.4.0
- {bank_line}
- Categories: {json.dumps(input_manifest['category_counts'], sort_keys=True)}
- {command_line}
- Pickup: checkpoint-native cue preserved
- Drop: zero for every actor step
- Rollout length: 501 actor steps / 501 video frames / 10.02 s at 50 FPS
- Individual videos: `individual/`
- Source masters: `masters/`
- Status counts: `{json.dumps(dict(status_counts), sort_keys=True)}`
- Trigger-source counts: `{json.dumps(dict(trigger_source_counts), sort_keys=True)}`

`not_triggered` means neither the live-height gate nor any configured time fallback sent a forward command.
`triggered_terminal_loss` means the command gate triggered correctly but the object was not still carried at the terminal step.
`triggered_terminal_carry` means the gate triggered and terminal lift/contact both passed.
"""
    (stage_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"stage_root": str(stage_root), "status_counts": dict(status_counts)}, indent=2))


if __name__ == "__main__":
    main()
