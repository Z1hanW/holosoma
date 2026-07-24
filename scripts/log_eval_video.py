#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATA_DIR = "data/ds_as_data/prism_debug30_convexhull_allmesh_solid_box_bin_barrel_ball"
DEFAULT_SPEEDS = "0.05,0.11,0.2"
DEFAULT_VIDEO_SECONDS = 10.0
DEFAULT_LIFT_WAIT_SECONDS = 20.0
DEFAULT_CONTROL_HZ = 50.0


@dataclass(frozen=True)
class ClipSelection:
    clip_id: str
    category: str
    scale_group: str


@dataclass(frozen=True)
class PairRecord:
    clip_id: str
    pair_dir: Path
    pair_map: Path
    active_urdf: Path
    source_npz: Path
    template_video_dir: Path


@dataclass(frozen=True)
class Task:
    task_id: int
    checkpoint: str
    checkpoint_slug: str
    clip: ClipSelection
    pair: PairRecord
    speed: float
    gpu: str
    video_dir: Path
    mp4_path: Path
    log_path: Path
    auto_forward_log_path: Path


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return default if raw is None else int(raw)


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return default if raw is None else float(raw)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def safe_name(value: str, *, max_len: int = 120) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    if not cleaned:
        cleaned = "item"
    if len(cleaned) > max_len:
        cleaned = cleaned[: max_len - 9].rstrip("_.-") + "_" + str(abs(hash(value)) % 10_000_000)
    return cleaned


def checkpoint_slug(ref: str) -> str:
    wandb_match = re.fullmatch(r"wandb://([^/]+)/([^/]+)/([^/]+)/(.+)", ref.strip())
    if wandb_match:
        _entity, _project, run_id, file_name = wandb_match.groups()
        return safe_name(f"{run_id}_{Path(file_name).stem}")

    url_match = re.search(r"https://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)(?:/files/([^?#]+))?", ref.strip())
    if url_match:
        _entity, _project, run_id, file_name = url_match.groups()
        if file_name:
            return safe_name(f"{run_id}_{Path(file_name).stem}")
        return safe_name(run_id)

    path = Path(ref)
    if ref.endswith(".pt") or path.suffix == ".pt":
        parent = safe_name(path.parent.name, max_len=48)
        return safe_name(f"{parent}_{path.stem}")
    return safe_name(ref)


def parse_speeds(raw: str) -> list[float]:
    speeds = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not speeds:
        raise ValueError("At least one speed is required.")
    return speeds


def speed_key(value: str | float) -> str:
    return f"{float(value):g}"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def default_motion_dir(data_dir: Path) -> Path:
    single_slot = data_dir / "_single_slot_motion_bank"
    if single_slot.is_dir():
        return single_slot
    return data_dir


def default_object_map(data_dir: Path, motion_dir: Path) -> Path:
    motion_map = motion_dir / "_clip_object_urdf_map.json"
    if motion_map.is_file():
        return motion_map
    return data_dir / "_clip_object_urdf_map.json"


def load_clip_map(object_map: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = read_json(object_map)
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        return {key: value for key, value in payload.items() if key != "clips"}, payload["clips"]
    if isinstance(payload, dict):
        return {}, payload
    raise ValueError(f"Invalid object map: {object_map}")


def infer_category(clip_id: str, entry: Any) -> str | None:
    values = [clip_id]
    if isinstance(entry, dict):
        for key in ("object_name", "object_category", "category", "object_type", "object_urdf_path", "object_mesh_path"):
            raw = entry.get(key)
            if raw:
                values.append(str(raw))
    blob = " ".join(values).lower()
    for category in ("ball", "barrel", "bin"):
        if category in blob:
            return category
    return None


def infer_scale_group(clip_id: str, entry: Any) -> str | None:
    values = [clip_id]
    if isinstance(entry, dict):
        raw_name = entry.get("object_name")
        if raw_name:
            values.append(str(raw_name))

    for value in values:
        normalized = value.strip().lower()
        if normalized.startswith(("noscale__", "noscale_", "no_scale__", "unscale__", "unscale_", "unscaled__")):
            return "unscale"
        if normalized.startswith(("scale__", "scale_", "scaled__", "scaled_", "scaledown__", "scaledown_", "scaleup__", "scaleup_")):
            return "scale"

    blob = " ".join(values).lower()
    if "noscale" in blob or "unscale" in blob or "unscaled" in blob:
        return "unscale"
    if "scaledown" in blob or "scaleup" in blob or "scale__" in blob:
        return "scale"
    return None


def select_default_clips(motion_dir: Path, object_map: Path, *, allow_missing: bool = False) -> list[ClipSelection]:
    _metadata, clips = load_clip_map(object_map)
    if not clips:
        raise ValueError(f"Object map has no clips: {object_map}")

    buckets: dict[tuple[str, str], list[str]] = {(category, scale): [] for category in ("ball", "barrel", "bin") for scale in ("unscale", "scale")}
    counts: dict[tuple[str | None, str | None], int] = {}
    for clip_id, entry in clips.items():
        if not (motion_dir / f"{clip_id}.npz").is_file():
            continue
        category = infer_category(clip_id, entry)
        scale = infer_scale_group(clip_id, entry)
        counts[(category, scale)] = counts.get((category, scale), 0) + 1
        key = (category, scale)
        if key in buckets:
            buckets[key].append(clip_id)

    selected: list[ClipSelection] = []
    missing: list[str] = []
    for category in ("ball", "barrel", "bin"):
        for scale in ("unscale", "scale"):
            candidates = sorted(buckets[(category, scale)])
            if not candidates:
                missing.append(f"{scale}/{category}")
                continue
            selected.append(ClipSelection(clip_id=candidates[0], category=category, scale_group=scale))

    if missing and not allow_missing:
        available = ", ".join(
            f"{scale or 'unknown'}/{category or 'unknown'}={count}"
            for (category, scale), count in sorted(counts.items(), key=lambda item: (str(item[0][1]), str(item[0][0])))
            if category in {"ball", "barrel", "bin"} or scale in {"unscale", "scale"}
        )
        raise ValueError(
            "Could not select the requested 6 clips: missing "
            + ", ".join(missing)
            + f". Available counts: {available or 'none'}. "
            "Pass --clip-list, or set DATA_DIR/OBJECT_MAP to a bank with one scaled and one unscaled ball/barrel/bin."
        )

    if missing:
        print(
            "[WARN] Default clip selection is missing "
            + ", ".join(missing)
            + "; continuing because --allow-missing-default-clips is set.",
            flush=True,
        )

    return selected


def read_clip_list(path: Path, object_map: Path) -> list[ClipSelection]:
    _metadata, clips = load_clip_map(object_map)
    selected: list[ClipSelection] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for part in line.split(","):
            clip_id = part.strip()
            if not clip_id:
                continue
            entry = clips.get(clip_id, {})
            selected.append(
                ClipSelection(
                    clip_id=clip_id,
                    category=infer_category(clip_id, entry) or "unknown",
                    scale_group=infer_scale_group(clip_id, entry) or "unknown",
                )
            )
    if not selected:
        raise ValueError(f"Clip list is empty: {path}")
    return selected


def write_selected_clip_files(selected: list[ClipSelection], out_root: Path) -> tuple[Path, Path]:
    clip_list = out_root / "selected_clips.txt"
    clip_tsv = out_root / "selected_clips.tsv"
    clip_list.write_text("\n".join(clip.clip_id for clip in selected) + "\n", encoding="utf-8")
    with clip_tsv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["clip_id", "category", "scale_group"])
        for clip in selected:
            writer.writerow([clip.clip_id, clip.category, clip.scale_group])
    return clip_list, clip_tsv


def run_checked(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    print("[INFO] " + " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def prepare_pairs(args: argparse.Namespace, selected_clip_list: Path) -> Path:
    manifest = args.out_root / "pairs_manifest.tsv"
    cmd = [
        str(args.python_bin),
        "scripts/prepare_as_replay_pairs.py",
        "--motion-dir",
        str(args.motion_dir),
        "--object-map",
        str(args.object_map),
        "--work-root",
        str(args.pair_work_root / "pairs"),
        "--video-root",
        str(args.out_root / "videos" / "by_clip_template"),
        "--manifest",
        str(manifest),
        "--single-slot",
        "--force",
        "--clip-list",
        str(selected_clip_list),
    ]
    if args.expected_total is not None:
        cmd.extend(["--expected-total", str(args.expected_total)])
    run_checked(cmd, cwd=args.repo_root)
    return manifest


def read_pair_manifest(manifest: Path) -> dict[str, PairRecord]:
    pairs: dict[str, PairRecord] = {}
    with manifest.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            clip_id, pair_dir, pair_map, active_urdf, source_npz, template_video_dir = line.split("\t")
            pairs[clip_id] = PairRecord(
                clip_id=clip_id,
                pair_dir=Path(pair_dir),
                pair_map=Path(pair_map),
                active_urdf=Path(active_urdf),
                source_npz=Path(source_npz),
                template_video_dir=Path(template_video_dir),
            )
    return pairs


def select_video(video_dir: Path, min_frames: int) -> Path | None:
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        return None

    try:
        import cv2  # type: ignore
    except Exception:
        cv2 = None

    rows: list[tuple[int, int, float, Path]] = []
    for path in videos:
        frames = -1
        if cv2 is not None:
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        rows.append((frames, path.stat().st_size, path.stat().st_mtime, path))

    eligible = [row for row in rows if row[0] >= min_frames]
    if not eligible and cv2 is None:
        eligible = [row for row in rows if row[1] > 0]
    if not eligible:
        return None
    return max(eligible, key=lambda row: (row[0], row[1], row[2]))[3]


def auto_forward_summary(path: Path, *, lift_threshold: float, lift_metric: str) -> dict[str, Any]:
    states: dict[str, int] = {}
    events: dict[str, int] = {}
    active_or_done = 0
    lifted_active_or_done = 0
    max_rel_z_delta: float | None = None
    max_object_z: float | None = None
    max_contact_force = 0.0

    if not path.is_file():
        return {
            "exists": False,
            "triggered": False,
            "duration_complete": False,
            "active_lift_fraction": 0.0,
            "max_rel_z_delta": None,
            "max_object_z": None,
            "max_contact_force": 0.0,
            "states": states,
            "events": events,
        }

    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not raw_line.strip():
            continue
        try:
            row = json.loads(raw_line)
        except Exception:
            continue
        state = str(row.get("state", ""))
        event = str(row.get("event", ""))
        if state:
            states[state] = states.get(state, 0) + 1
        if event:
            events[event] = events.get(event, 0) + 1

        metric_raw = row.get("lift_metric_value")
        if not isinstance(metric_raw, (int, float)):
            metric_raw = row.get("object_z_delta") if lift_metric == "object_z_delta" else row.get("object_rel_z_delta")
        delta_raw = metric_raw
        delta = float(delta_raw) if isinstance(delta_raw, (int, float)) else None
        if delta is not None:
            max_rel_z_delta = delta if max_rel_z_delta is None else max(max_rel_z_delta, delta)

        object_pos = row.get("object_pos_w")
        if isinstance(object_pos, list) and len(object_pos) >= 3 and isinstance(object_pos[2], (int, float)):
            object_z = float(object_pos[2])
            max_object_z = object_z if max_object_z is None else max(max_object_z, object_z)

        for key, value in row.items():
            if not str(key).endswith("_object_force"):
                continue
            if isinstance(value, (int, float)):
                max_contact_force = max(max_contact_force, float(value))

        if state in {"active", "done"}:
            active_or_done += 1
            if delta is not None and delta >= lift_threshold:
                lifted_active_or_done += 1

    triggered = bool(events.get("trigger", 0) or states.get("active", 0) or states.get("done", 0))
    duration_complete = bool(events.get("duration_complete", 0) or states.get("done", 0))
    return {
        "exists": True,
        "triggered": triggered,
        "duration_complete": duration_complete,
        "active_lift_fraction": (lifted_active_or_done / active_or_done) if active_or_done else 0.0,
        "max_rel_z_delta": max_rel_z_delta,
        "max_object_z": max_object_z,
        "max_contact_force": max_contact_force,
        "states": states,
        "events": events,
    }


def auto_forward_gate_ok(args: argparse.Namespace, task: Task) -> tuple[bool, str]:
    summary = auto_forward_summary(
        task.auto_forward_log_path,
        lift_threshold=args.lift_rel_z_delta,
        lift_metric=args.lift_trigger_metric,
    )
    if args.require_lift_trigger and not summary["triggered"]:
        return False, "missing_lift_trigger"
    if args.require_lift_duration_complete and not summary["duration_complete"]:
        return False, "missing_lift_duration_complete"
    if args.min_hand_contact_force > 0.0 and float(summary.get("max_contact_force") or 0.0) < args.min_hand_contact_force:
        return False, f"max_contact_force={float(summary.get('max_contact_force') or 0.0):.3f}"
    if args.min_active_lift_fraction > 0.0 and summary["active_lift_fraction"] < args.min_active_lift_fraction:
        return False, f"active_lift_fraction={summary['active_lift_fraction']:.3f}"
    return True, (
        f"triggered={summary['triggered']};duration_complete={summary['duration_complete']};"
        f"active_lift_fraction={summary['active_lift_fraction']:.3f};"
        f"max_rel_z_delta={summary['max_rel_z_delta']};max_contact_force={summary['max_contact_force']}"
    )


def task_command(args: argparse.Namespace, task: Task) -> list[str]:
    cmd = [
        "bash",
        "./infer_as_joystick.sh",
        task.checkpoint,
        "--training.max-eval-steps",
        str(args.eval_steps),
        "--command.setup-terms.motion-command.params.motion-config.motion-clip-name",
        task.clip.clip_id,
        "--command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.adaptive-sampling-contact-interval-root",
        "",
        "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob",
        "1.0",
        "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob",
        "0.0",
        "--command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale",
        "0.0",
        "logger:disabled",
        "--logger.video.enabled",
        "True",
        "--logger.headless-recording",
        "True",
        "--logger.video.upload-to-wandb",
        "False",
        "--logger.video.interval",
        "1",
        "--logger.video.save-dir",
        str(task.video_dir),
        "--logger.video.width",
        str(args.video_width),
        "--logger.video.height",
        str(args.video_height),
        "--logger.video.output-format",
        args.video_format,
        "--logger.video.playback-rate",
        "1.0",
        "--logger.video.camera-smoothing",
        str(args.video_camera_smoothing),
        "--logger.video.show-command-overlay",
        "False",
        "--logger.video.record-env-id",
        "0",
    ]
    if args.infer_extra_args:
        cmd.extend(shlex.split(args.infer_extra_args))
    return cmd


def task_env(args: argparse.Namespace, task: Task) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHON_BIN": str(args.python_bin),
            "OMOMO_DATA_DIR": str(task.pair.pair_dir),
            "OMOMO_OBJECT_MAP": str(task.pair.pair_map),
            "OMOMO_EXPECTED_TOTAL": "1",
            # Isaac's URDF converter writes robot USDs into converted_rank{rank}.
            # The eval launcher constrains CUDA_VISIBLE_DEVICES per task, so all
            # children otherwise look like local rank 0 and can race on the same
            # temporary USD files. Use a stable per-task conversion rank instead.
            "HOLOSOMA_ORIGINAL_LOCAL_RANK": str(task.task_id),
            "AS_AUTO_FORWARD_AFTER_LIFT": "1",
            "AS_AUTO_FORWARD_AFTER_LIFT_COMMAND": f"{task.speed:g},0,0",
            "AS_AUTO_FORWARD_AFTER_LIFT_DURATION_S": f"{args.video_seconds:g}",
            "AS_AUTO_FORWARD_AFTER_LIFT_REL_Z_DELTA": f"{args.lift_rel_z_delta:g}",
            "AS_AUTO_FORWARD_AFTER_LIFT_CONSECUTIVE_STEPS": str(args.lift_consecutive_steps),
            "VISER_AUTO_FORWARD_AFTER_LIFT_DURATION_HZ": f"{args.control_hz:g}",
            "VISER_AUTO_FORWARD_AFTER_LIFT_DURATION_STEPS": str(
                max(1, int(round(args.video_seconds * args.control_hz)))
            ),
            "VISER_AUTO_FORWARD_AFTER_LIFT_MIN_HAND_CONTACT_FORCE": f"{args.min_hand_contact_force:g}",
            "VISER_AUTO_FORWARD_AFTER_LIFT_LOG_PATH": str(task.auto_forward_log_path),
            "VISER_AUTO_FORWARD_AFTER_LIFT_METRIC": args.lift_trigger_metric,
            "VISER_AUTO_FORWARD_AFTER_LIFT_MIN_ROOT_Z": f"{args.min_root_z:g}",
            "VISER_AUTO_FORWARD_AFTER_LIFT_RECORD_ON_TRIGGER": "1",
            "VISER_AUTO_FORWARD_AFTER_LIFT_STOP_RECORDING_ON_DONE": "1",
            "VISER_ENABLE_CLIP_GUI": "0",
            "VISER_ENABLE_MANUAL_GUI": "0",
            # Start with zero root command so the policy attempts the stationary
            # lift first. Auto-forward overwrites this with the requested speed
            # only after the lift trigger fires.
            "VISER_FORCE_MANUAL_CONTROL": "1",
            "VISER_MANUAL_CONTROL_DEFAULT": "1",
            "VISER_MANUAL_COMMAND_DEFAULT": "0,0,0",
            "DEPTH_PERCEPTION_PRESET": "checkpoint",
            "HOLOSOMA_RESET_TO_DEFAULT_POSE": "0",
            "HEADLESS": "True",
            "NUM_ENVS": "1",
            "OBJECT_SPAWN_MODE": "urdf",
            "OBJECT_GEOMETRY_MODE": "mesh",
            "HOLOSOMA_OBJECT_COLLIDER_TYPE": "convex_decomposition",
            "HOLOSOMA_DISABLE_AUTO_RESET": "1",
            "HOLOSOMA_DISABLE_MOTION_END_RESET": "1",
            "HOLOSOMA_DISABLE_CLIP_END_RESET": "1",
            "HOLOSOMA_EVAL_DISABLE_ROLLOUT_REFERENCE_REWARDS": "1",
            "HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE": "1",
            "HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME": "1",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
        }
    )
    if task.gpu != "auto":
        # infer_as_joystick.sh delegates to the standard eval stack, which does
        # not consistently consume a custom HOLOSOMA_DEVICE env var. Restrict
        # visibility instead so every parallel Isaac Sim process sees exactly
        # one GPU and uses it as cuda:0.
        env["CUDA_VISIBLE_DEVICES"] = str(task.gpu)
        env["HOLOSOMA_DEVICE"] = "cuda:0"
    return env


def terminate_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def run_task(args: argparse.Namespace, task: Task) -> dict[str, Any]:
    task.video_dir.mkdir(parents=True, exist_ok=True)
    task.mp4_path.parent.mkdir(parents=True, exist_ok=True)
    task.log_path.parent.mkdir(parents=True, exist_ok=True)
    task.auto_forward_log_path.parent.mkdir(parents=True, exist_ok=True)
    for old_video in task.video_dir.glob("*.mp4"):
        old_video.unlink()

    if args.skip_existing and task.mp4_path.is_file() and task.mp4_path.stat().st_size > 0:
        return task_status(task, "exists", "already_present")

    cmd = task_command(args, task)
    env = task_env(args, task)
    if args.dry_run:
        task.log_path.write_text("[DRY_RUN] " + " ".join(shlex.quote(part) for part in cmd) + "\n", encoding="utf-8")
        return task_status(task, "dry_run", "dry_run", command=cmd)

    started = time.time()
    with task.log_path.open("w", encoding="utf-8") as log:
        log.write("[INFO] " + " ".join(shlex.quote(part) for part in cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(args.repo_root),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        timed_out = False
        stopped_after_video = False
        deadline = started + args.task_timeout_s
        while True:
            polled = proc.poll()
            if polled is not None:
                return_code = polled
                break

            now = time.time()
            if (
                args.stop_after_valid_video
                and now - started >= args.min_task_runtime_s
                and (produced_now := select_video(task.video_dir, args.min_video_frames)) is not None
                and now - produced_now.stat().st_mtime >= args.video_stable_s
            ):
                gate_ok, _gate_note = auto_forward_gate_ok(args, task)
                if gate_ok:
                    stopped_after_video = True
                    terminate_process_group(proc)
                    return_code = 124
                    break

            if now >= deadline:
                timed_out = True
                terminate_process_group(proc)
                return_code = 124
                break

            time.sleep(min(args.video_check_interval_s, max(0.5, deadline - now)))

    produced = select_video(task.video_dir, args.min_video_frames)
    elapsed_s = time.time() - started
    gate_ok, gate_note = auto_forward_gate_ok(args, task)
    if produced is not None and not gate_ok:
        return task_status(task, "failed", gate_note, return_code=return_code, elapsed_s=elapsed_s)

    if produced is not None:
        shutil.copy2(produced, task.mp4_path)
        status = "ok" if return_code == 0 else f"ok_after_exit_{return_code}"
        note = f"source={produced};{gate_note}"
        if timed_out:
            status = "ok_after_timeout"
        elif stopped_after_video:
            status = "ok_after_video_complete"
        return task_status(task, status, note, return_code=return_code, elapsed_s=elapsed_s)

    note = f"exit={return_code}"
    if timed_out:
        note = "timeout"
    elif return_code == 0:
        note = f"missing_video_min_frames_{args.min_video_frames}"
    return task_status(task, "failed", note, return_code=return_code, elapsed_s=elapsed_s)


def task_status(
    task: Task,
    status: str,
    note: str,
    *,
    return_code: int | None = None,
    elapsed_s: float | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "checkpoint_slug": task.checkpoint_slug,
        "checkpoint": task.checkpoint,
        "clip_id": task.clip.clip_id,
        "category": task.clip.category,
        "scale_group": task.clip.scale_group,
        "speed": task.speed,
        "gpu": task.gpu,
        "mp4_path": str(task.mp4_path),
        "log_path": str(task.log_path),
        "auto_forward_log_path": str(task.auto_forward_log_path),
        "note": note,
        "return_code": return_code,
        "elapsed_s": elapsed_s,
        "command": " ".join(shlex.quote(part) for part in command) if command is not None else "",
    }


def write_tasks_tsv(tasks: list[Task], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(
            [
                "task_id",
                "checkpoint_slug",
                "checkpoint",
                "clip_id",
                "category",
                "scale_group",
                "speed",
                "gpu",
                "pair_dir",
                "pair_map",
                "source_npz",
                "video_dir",
                "mp4_path",
                "log_path",
                "auto_forward_log_path",
            ]
        )
        for task in tasks:
            writer.writerow(
                [
                    task.task_id,
                    task.checkpoint_slug,
                    task.checkpoint,
                    task.clip.clip_id,
                    task.clip.category,
                    task.clip.scale_group,
                    f"{task.speed:g}",
                    task.gpu,
                    task.pair.pair_dir,
                    task.pair.pair_map,
                    task.pair.source_npz,
                    task.video_dir,
                    task.mp4_path,
                    task.log_path,
                    task.auto_forward_log_path,
                ]
            )


def write_status_tsv(statuses: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "status",
        "checkpoint_slug",
        "checkpoint",
        "clip_id",
        "category",
        "scale_group",
        "speed",
        "gpu",
        "mp4_path",
        "log_path",
        "auto_forward_log_path",
        "note",
        "return_code",
        "elapsed_s",
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in statuses:
            writer.writerow(row)


def build_tasks(args: argparse.Namespace, selected: list[ClipSelection], pairs: dict[str, PairRecord]) -> list[Task]:
    speeds = parse_speeds(args.speeds)
    gpu_ids = [part.strip() for part in args.gpu_ids.split(",") if part.strip()] or ["auto"]
    tasks: list[Task] = []
    task_id = 0
    for checkpoint in args.checkpoints:
        slug = checkpoint_slug(checkpoint)
        for clip in selected:
            pair = pairs[clip.clip_id]
            for speed in speeds:
                speed_safe = str(speed).replace(".", "p").replace("-", "m")
                clip_safe = safe_name(clip.clip_id)
                gpu = gpu_ids[task_id % len(gpu_ids)]
                stem = f"{slug}__{clip.category}_{clip.scale_group}__{clip_safe}__vx{speed_safe}"
                tasks.append(
                    Task(
                        task_id=task_id,
                        checkpoint=checkpoint,
                        checkpoint_slug=slug,
                        clip=clip,
                        pair=pair,
                        speed=speed,
                        gpu=gpu,
                        video_dir=args.out_root / "videos" / slug / clip_safe / f"vx{speed_safe}",
                        mp4_path=args.out_root / "mp4" / f"{safe_name(stem)}.mp4",
                        log_path=args.out_root / "logs" / f"{safe_name(stem)}.log",
                        auto_forward_log_path=args.out_root / "auto_forward" / f"{safe_name(stem)}.jsonl",
                    )
                )
                task_id += 1
    return tasks


def filter_retry_tasks(args: argparse.Namespace, tasks: list[Task]) -> list[Task]:
    if args.retry_status_tsv is None:
        return tasks

    wanted_statuses = {part.strip() for part in args.retry_statuses.split(",") if part.strip()}
    if not wanted_statuses:
        raise SystemExit("[ERROR] --retry-statuses must include at least one status.")

    retry_keys: set[tuple[str, str, str]] = set()
    with args.retry_status_tsv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        required = {"status", "checkpoint_slug", "clip_id", "speed"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"[ERROR] Retry status TSV missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row["status"] in wanted_statuses:
                retry_keys.add((row["checkpoint_slug"], row["clip_id"], speed_key(row["speed"])))

    selected = [
        task
        for task in tasks
        if (task.checkpoint_slug, task.clip.clip_id, speed_key(task.speed)) in retry_keys
    ]
    if not selected:
        raise SystemExit(
            f"[ERROR] --retry-status-tsv matched no tasks for statuses: {', '.join(sorted(wanted_statuses))}"
        )

    matched_keys = {(task.checkpoint_slug, task.clip.clip_id, speed_key(task.speed)) for task in selected}
    missing_keys = retry_keys.difference(matched_keys)
    if missing_keys:
        print(
            f"[WARN] Retry TSV requested {len(missing_keys)} task(s) that are not in the current checkpoint/clip/speed set.",
            flush=True,
        )
    print(
        f"[INFO] Retry filter: selected {len(selected)} / {len(tasks)} task(s) from {args.retry_status_tsv}",
        flush=True,
    )
    return selected


def upload_to_wandb(args: argparse.Namespace, statuses: list[dict[str, Any]], selected_clips_tsv: Path, tasks_tsv: Path, status_tsv: Path) -> None:
    if args.no_wandb or args.dry_run:
        print("[INFO] W&B upload skipped.", flush=True)
        return

    successful = [row for row in statuses if str(row.get("status", "")).startswith(("ok", "exists")) and Path(str(row["mp4_path"])).is_file()]
    if not successful:
        print("[WARN] No successful videos to upload to W&B; uploading status artifact only.", flush=True)

    try:
        import wandb
    except Exception as exc:
        raise RuntimeError(
            "wandb is not importable in this Python environment. Set PYTHON_BIN to the hssim env or pass --no-wandb."
        ) from exc

    init_kwargs: dict[str, Any] = {
        "project": args.wandb_project,
        "name": args.wandb_run_name,
        "job_type": "eval_video",
        "config": {
            "checkpoints": args.checkpoints,
            "data_dir": str(args.data_dir),
            "motion_dir": str(args.motion_dir),
            "object_map": str(args.object_map),
            "speeds": parse_speeds(args.speeds),
            "video_seconds": args.video_seconds,
            "lift_wait_seconds": args.lift_wait_seconds,
            "eval_steps": args.eval_steps,
            "require_lift_trigger": args.require_lift_trigger,
            "require_lift_duration_complete": args.require_lift_duration_complete,
            "min_active_lift_fraction": args.min_active_lift_fraction,
            "lift_trigger_metric": args.lift_trigger_metric,
            "min_root_z": args.min_root_z,
            "min_hand_contact_force": args.min_hand_contact_force,
            "selected_clips": [row.clip_id for row in read_selected_clips_tsv(selected_clips_tsv)],
        },
    }
    if args.wandb_entity:
        init_kwargs["entity"] = args.wandb_entity

    run = wandb.init(**init_kwargs)
    try:
        if successful:
            for idx, row in enumerate(successful):
                speed = float(row["speed"])
                key = "/".join(
                    [
                        "eval_video",
                        safe_name(str(row["checkpoint_slug"]), max_len=64),
                        f"{row['category']}_{row['scale_group']}",
                        safe_name(str(row["clip_id"]), max_len=64),
                        f"vx_{str(speed).replace('.', 'p').replace('-', 'm')}",
                    ]
                )
                run.log(
                    {
                        key: wandb.Video(str(row["mp4_path"]), format="mp4"),
                        "eval/speed": speed,
                        "eval/video_index": idx,
                    },
                    step=idx,
                )
        else:
            run.log({"eval/successful_videos": 0}, step=0)

        artifact = wandb.Artifact(f"{safe_name(args.wandb_run_name)}_bundle", type="eval-video-bundle")
        for path in (selected_clips_tsv, tasks_tsv, status_tsv):
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
        mp4_dir = args.out_root / "mp4"
        if mp4_dir.is_dir():
            artifact.add_dir(str(mp4_dir), name="mp4")
        run.log_artifact(artifact)
        print(f"[INFO] W&B project={args.wandb_project} run={run.name} url={run.url}", flush=True)
    finally:
        run.finish()


def read_selected_clips_tsv(path: Path) -> list[ClipSelection]:
    rows: list[ClipSelection] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(ClipSelection(row["clip_id"], row["category"], row["scale_group"]))
    return rows


def run_tasks(args: argparse.Namespace, tasks: list[Task]) -> list[dict[str, Any]]:
    statuses: list[dict[str, Any]] = []
    lock = threading.Lock()

    print(
        f"[INFO] Running {len(tasks)} task(s); parallel_jobs={args.parallel_jobs}; "
        f"gpu_ids={args.gpu_ids}; eval_steps={args.eval_steps}",
        flush=True,
    )
    if args.parallel_jobs <= 1:
        for task in tasks:
            print(
                f"[INFO] task={task.task_id} ckpt={task.checkpoint_slug} clip={task.clip.clip_id} "
                f"speed={task.speed:g} gpu={task.gpu}",
                flush=True,
            )
            statuses.append(run_task(args, task))
        return statuses

    with ThreadPoolExecutor(max_workers=args.parallel_jobs) as pool:
        future_to_task = {pool.submit(run_task, args, task): task for task in tasks}
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                row = future.result()
            except Exception as exc:
                row = task_status(task, "failed", f"exception={exc}")
            with lock:
                statuses.append(row)
                print(
                    f"[INFO] finished task={task.task_id} status={row['status']} "
                    f"ckpt={task.checkpoint_slug} clip={task.clip.clip_id} speed={task.speed:g}",
                    flush=True,
                )
    return sorted(statuses, key=lambda row: (str(row["checkpoint_slug"]), str(row["clip_id"]), float(row["speed"])))


def parse_args(argv: list[str]) -> argparse.Namespace:
    stamp = _env("RUN_STAMP", time.strftime("%Y%m%d_%H%M%S", time.gmtime()))
    root = repo_root()
    data_dir_default = root / (_env("DATA_DIR", DEFAULT_DATA_DIR) or DEFAULT_DATA_DIR)
    out_root_default = root / "outputs" / f"log_eval_video_{stamp}"
    video_seconds_default = _env_float("VIDEO_SECONDS", DEFAULT_VIDEO_SECONDS)
    lift_wait_seconds_default = _env_float("LIFT_WAIT_SECONDS", DEFAULT_LIFT_WAIT_SECONDS)
    control_hz_default = _env_float("CONTROL_HZ", DEFAULT_CONTROL_HZ)
    eval_steps_default = _env_int(
        "EVAL_STEPS",
        int(round((video_seconds_default + lift_wait_seconds_default) * control_hz_default)),
    )

    parser = argparse.ArgumentParser(
        description=(
            "Record Isaac Sim lift-and-carry eval videos for one or more checkpoints, "
            "then upload them to a new W&B project/run."
        )
    )
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint refs: wandb://..., W&B run URL, or local .pt path.")
    parser.add_argument("--data-dir", type=Path, default=Path(_env("DATA_DIR", str(data_dir_default))))
    parser.add_argument("--motion-dir", type=Path, default=Path(_env("MOTION_DIR")) if _env("MOTION_DIR") else None)
    parser.add_argument("--object-map", type=Path, default=None)
    parser.add_argument("--clip-list", type=Path, default=Path(_env("CLIP_LIST")) if _env("CLIP_LIST") else None)
    parser.add_argument("--out-root", type=Path, default=Path(_env("OUT_ROOT", str(out_root_default))))
    parser.add_argument(
        "--pair-work-root",
        type=Path,
        default=Path(_env("PAIR_WORK_ROOT", str(root / "data" / "ds_as_data" / "_log_eval_video_pairs" / stamp))),
        help="Temporary one-clip motion banks. Must remain under repo data/ for infer_as_joystick.sh.",
    )
    parser.add_argument("--expected-total", type=int, default=int(_env("EXPECTED_TOTAL")) if _env("EXPECTED_TOTAL") else None)
    parser.add_argument("--speeds", default=_env("SPEEDS", DEFAULT_SPEEDS))
    parser.add_argument("--video-seconds", type=float, default=video_seconds_default)
    parser.add_argument("--lift-wait-seconds", type=float, default=lift_wait_seconds_default)
    parser.add_argument("--control-hz", type=float, default=control_hz_default)
    parser.add_argument("--eval-steps", type=int, default=eval_steps_default)
    parser.add_argument("--task-timeout-s", type=int, default=_env_int("TASK_TIMEOUT_S", 1200))
    parser.add_argument("--parallel-jobs", type=int, default=_env_int("PARALLEL_JOBS", 1))
    parser.add_argument("--gpu-ids", default=_env("GPU_IDS", "auto"))
    parser.add_argument("--video-width", type=int, default=_env_int("VIDEO_WIDTH", 640))
    parser.add_argument("--video-height", type=int, default=_env_int("VIDEO_HEIGHT", 360))
    parser.add_argument("--video-format", default=_env("VIDEO_FORMAT", "mp4"))
    parser.add_argument("--video-camera-smoothing", type=float, default=_env_float("VIDEO_CAMERA_SMOOTHING", 0.90))
    parser.add_argument(
        "--min-video-frames",
        type=int,
        default=_env_int(
            "MIN_VIDEO_FRAMES",
            max(30, int(round(video_seconds_default * control_hz_default * 0.8))),
        ),
    )
    parser.add_argument("--stop-after-valid-video", action=argparse.BooleanOptionalAction, default=_env_bool("STOP_AFTER_VALID_VIDEO", True))
    parser.add_argument("--min-task-runtime-s", type=float, default=_env_float("MIN_TASK_RUNTIME_S", 30.0))
    parser.add_argument("--video-stable-s", type=float, default=_env_float("VIDEO_STABLE_S", 5.0))
    parser.add_argument("--video-check-interval-s", type=float, default=_env_float("VIDEO_CHECK_INTERVAL_S", 5.0))
    parser.add_argument("--lift-rel-z-delta", type=float, default=_env_float("LIFT_REL_Z_DELTA", 0.10))
    parser.add_argument("--lift-consecutive-steps", type=int, default=_env_int("LIFT_CONSECUTIVE_STEPS", 5))
    parser.add_argument("--lift-trigger-metric", choices=["object_z_delta", "rel_z_delta"], default=_env("LIFT_TRIGGER_METRIC", "object_z_delta"))
    parser.add_argument("--min-root-z", type=float, default=_env_float("MIN_ROOT_Z", 0.45))
    parser.add_argument("--min-hand-contact-force", type=float, default=_env_float("MIN_HAND_CONTACT_FORCE", 1.0))
    parser.add_argument("--require-lift-trigger", action=argparse.BooleanOptionalAction, default=_env_bool("REQUIRE_LIFT_TRIGGER", True))
    parser.add_argument(
        "--require-lift-duration-complete",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("REQUIRE_LIFT_DURATION_COMPLETE", True),
    )
    parser.add_argument("--min-active-lift-fraction", type=float, default=_env_float("MIN_ACTIVE_LIFT_FRACTION", 0.80))
    parser.add_argument("--infer-extra-args", default=_env("INFER_EXTRA_ARGS", ""))
    parser.add_argument("--wandb-project", default=_env("WANDB_PROJECT", f"log-eval-video-{stamp}"))
    parser.add_argument("--wandb-run-name", default=_env("WANDB_RUN_NAME", f"log_eval_video_{stamp}"))
    parser.add_argument("--wandb-entity", default=_env("WANDB_ENTITY", ""))
    parser.add_argument(
        "--retry-status-tsv",
        type=Path,
        default=Path(_env("RETRY_STATUS_TSV")) if _env("RETRY_STATUS_TSV") else None,
        help="Optional prior status.tsv. When set, only tasks whose prior status is in --retry-statuses are run.",
    )
    parser.add_argument("--retry-statuses", default=_env("RETRY_STATUSES", "failed"))
    parser.add_argument(
        "--allow-missing-default-clips",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("ALLOW_MISSING_DEFAULT_CLIPS", False),
        help="Use available default scale/category buckets instead of failing when a bucket is absent.",
    )
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=_env_bool("SKIP_EXISTING", True))
    parser.add_argument("--dry-run", action="store_true", default=_env_bool("DRY_RUN", False))
    parser.add_argument("--no-wandb", action="store_true", default=_env_bool("NO_WANDB", False))
    args = parser.parse_args(argv)

    args.repo_root = root
    args.python_bin = Path(_env("PYTHON_BIN", sys.executable)).expanduser().resolve()
    args.data_dir = args.data_dir.expanduser()
    if not args.data_dir.is_absolute():
        args.data_dir = (root / args.data_dir).resolve()
    else:
        args.data_dir = args.data_dir.resolve()

    if args.motion_dir is None:
        args.motion_dir = default_motion_dir(args.data_dir)
    else:
        args.motion_dir = args.motion_dir.expanduser()
        if not args.motion_dir.is_absolute():
            args.motion_dir = (root / args.motion_dir).resolve()
        else:
            args.motion_dir = args.motion_dir.resolve()

    object_map_env = _env("OBJECT_MAP")
    if args.object_map is None:
        args.object_map = Path(object_map_env).expanduser() if object_map_env else default_object_map(args.data_dir, args.motion_dir)
    if not args.object_map.is_absolute():
        args.object_map = (root / args.object_map).resolve()
    else:
        args.object_map = args.object_map.resolve()

    args.out_root = args.out_root.expanduser()
    if not args.out_root.is_absolute():
        args.out_root = (root / args.out_root).resolve()
    else:
        args.out_root = args.out_root.resolve()

    args.pair_work_root = args.pair_work_root.expanduser()
    if not args.pair_work_root.is_absolute():
        args.pair_work_root = (root / args.pair_work_root).resolve()
    else:
        args.pair_work_root = args.pair_work_root.resolve()

    if args.clip_list is not None:
        args.clip_list = args.clip_list.expanduser()
        if not args.clip_list.is_absolute():
            args.clip_list = (root / args.clip_list).resolve()
        else:
            args.clip_list = args.clip_list.resolve()

    if args.retry_status_tsv is not None:
        args.retry_status_tsv = args.retry_status_tsv.expanduser()
        if not args.retry_status_tsv.is_absolute():
            args.retry_status_tsv = (root / args.retry_status_tsv).resolve()
        else:
            args.retry_status_tsv = args.retry_status_tsv.resolve()

    if not args.data_dir.is_dir():
        raise SystemExit(f"[ERROR] DATA_DIR not found: {args.data_dir}")
    if not args.motion_dir.is_dir():
        raise SystemExit(f"[ERROR] MOTION_DIR not found: {args.motion_dir}")
    if not args.object_map.is_file():
        raise SystemExit(f"[ERROR] OBJECT_MAP not found: {args.object_map}")
    if args.retry_status_tsv is not None and not args.retry_status_tsv.is_file():
        raise SystemExit(f"[ERROR] --retry-status-tsv not found: {args.retry_status_tsv}")
    if args.eval_steps <= 0:
        raise SystemExit("[ERROR] --eval-steps must be positive.")
    if args.parallel_jobs <= 0:
        raise SystemExit("[ERROR] --parallel-jobs must be positive.")
    if args.video_check_interval_s <= 0:
        raise SystemExit("[ERROR] --video-check-interval-s must be positive.")
    if args.lift_wait_seconds < 0:
        raise SystemExit("[ERROR] --lift-wait-seconds must be non-negative.")
    if not 0.0 <= args.min_active_lift_fraction <= 1.0:
        raise SystemExit("[ERROR] --min-active-lift-fraction must be in [0, 1].")
    parse_speeds(args.speeds)
    return args


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    (args.out_root / "logs").mkdir(parents=True, exist_ok=True)
    (args.out_root / "mp4").mkdir(parents=True, exist_ok=True)

    if args.clip_list is not None:
        selected = read_clip_list(args.clip_list, args.object_map)
    else:
        selected = select_default_clips(
            args.motion_dir,
            args.object_map,
            allow_missing=args.allow_missing_default_clips,
        )

    selected_clip_list, selected_clips_tsv = write_selected_clip_files(selected, args.out_root)
    print("[INFO] Selected clips:", flush=True)
    for clip in selected:
        print(f"[INFO]   {clip.category}/{clip.scale_group}: {clip.clip_id}", flush=True)

    if args.dry_run:
        print(f"[INFO] Dry run output root: {args.out_root}", flush=True)
        pairs = {
            clip.clip_id: PairRecord(
                clip_id=clip.clip_id,
                pair_dir=args.pair_work_root / "pairs" / safe_name(clip.clip_id),
                pair_map=args.pair_work_root / "pairs" / safe_name(clip.clip_id) / "_clip_object_urdf_map.json",
                active_urdf=Path("dry_run.urdf"),
                source_npz=args.motion_dir / f"{clip.clip_id}.npz",
                template_video_dir=args.out_root / "videos" / "by_clip_template" / safe_name(clip.clip_id),
            )
            for clip in selected
        }
    else:
        manifest = prepare_pairs(args, selected_clip_list)
        pairs = read_pair_manifest(manifest)

    tasks = build_tasks(args, selected, pairs)
    tasks = filter_retry_tasks(args, tasks)
    tasks_tsv = args.out_root / "tasks.tsv"
    write_tasks_tsv(tasks, tasks_tsv)

    statuses = run_tasks(args, tasks)
    status_tsv = args.out_root / "status.tsv"
    write_status_tsv(statuses, status_tsv)

    ok_count = sum(1 for row in statuses if str(row.get("status", "")).startswith(("ok", "exists")))
    failed_count = sum(1 for row in statuses if row.get("status") == "failed")
    print(f"[INFO] Status: ok_or_exists={ok_count} failed={failed_count} total={len(statuses)}", flush=True)
    print(f"[INFO] Output root: {args.out_root}", flush=True)
    print(f"[INFO] Status TSV: {status_tsv}", flush=True)
    print(f"[INFO] MP4 dir: {args.out_root / 'mp4'}", flush=True)

    upload_to_wandb(args, statuses, selected_clips_tsv, tasks_tsv, status_tsv)
    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
