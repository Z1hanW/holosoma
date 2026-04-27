#!/usr/bin/env python3
"""Launch and score the MuJoCo generalist success rollout from sim-state."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATHS = [
    REPO_ROOT / "src" / "holosoma",
    REPO_ROOT / "src" / "holosoma_inference",
]
for src_path in reversed(SRC_PATHS):
    sys.path.insert(0, str(src_path))

from holosoma_inference.utils.policy_overlay import PolicyOverlaySub  # noqa: E402
from holosoma_inference.utils.sim_control import PolicyControlPush  # noqa: E402
from holosoma_inference.utils.sim_state import SimStateSub  # noqa: E402


@dataclass
class Sample:
    frame_idx: int
    sim_time_ms: int
    root_err: float | None = None
    object_err: float | None = None
    object_xy_err: float | None = None
    object_z_err: float | None = None
    key_mean_err: float | None = None
    object_robot_contacts: int | None = None
    object_scene_contacts: int | None = None
    object_z: float | None = None
    target_object_z: float | None = None
    contact_bodies: tuple[str, ...] = field(default_factory=tuple)
    root_pos: tuple[float, float, float] | None = None
    root_quat_xyzw: tuple[float, float, float, float] | None = None
    root_yaw: float | None = None
    target_root_pos: tuple[float, float, float] | None = None
    target_root_quat_xyzw: tuple[float, float, float, float] | None = None
    target_root_yaw: float | None = None
    object_pos: tuple[float, float, float] | None = None
    target_object_pos: tuple[float, float, float] | None = None
    key_errors: dict[str, float] = field(default_factory=dict)
    key_body_pos: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    target_key_body_pos: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    key_object_dist: dict[str, float] = field(default_factory=dict)


def _yaw_from_xyzw(quat: np.ndarray) -> float | None:
    quat = np.asarray(quat, dtype=np.float64).reshape(-1)
    if quat.size < 4:
        return None
    x, y, z, w = quat[:4]
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1.0e-9:
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _parse_buckets(raw: str) -> list[tuple[int, int]]:
    buckets: list[tuple[int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        start_raw, end_raw = item.split(":", 1)
        start, end = int(start_raw), int(end_raw)
        if end < start:
            raise ValueError(f"invalid bucket {item!r}")
        buckets.append((start, end))
    return buckets


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def _actor_state(state: dict[str, Any]) -> np.ndarray | None:
    actors = state.get("actors")
    if not isinstance(actors, dict) or not actors:
        return None
    actor = actors.get("object")
    if actor is None and len(actors) == 1:
        actor = next(iter(actors.values()))
    if actor is None:
        return None
    arr = np.asarray(actor, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr


def _key_body_errors(state: dict[str, Any], overlay: dict[str, Any]) -> dict[str, float]:
    sim_key_states = state.get("key_body_states")
    body_names = overlay.get("body_names")
    body_pos_w = overlay.get("body_pos_w")
    if not isinstance(sim_key_states, dict) or not isinstance(body_names, list) or body_pos_w is None:
        return {}

    target_pos = np.asarray(body_pos_w, dtype=np.float64).reshape(-1, 3)
    target_by_name = {
        str(name): target_pos[idx]
        for idx, name in enumerate(body_names)
        if idx < target_pos.shape[0]
    }
    errors: dict[str, float] = {}
    for name, sim_payload in sim_key_states.items():
        if name not in target_by_name:
            continue
        sim_pos = np.asarray(sim_payload, dtype=np.float64).reshape(-1)
        if sim_pos.size < 3:
            continue
        errors[str(name)] = float(np.linalg.norm(sim_pos[:3] - target_by_name[name]))
    return errors


def _key_body_positions(
    state: dict[str, Any],
    overlay: dict[str, Any],
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float]]]:
    sim_key_states = state.get("key_body_states")
    if not isinstance(sim_key_states, dict):
        return {}, {}

    sim_positions: dict[str, tuple[float, float, float]] = {}
    for name, sim_payload in sim_key_states.items():
        sim_pos = np.asarray(sim_payload, dtype=np.float64).reshape(-1)
        if sim_pos.size >= 3:
            sim_positions[str(name)] = tuple(float(value) for value in sim_pos[:3])

    body_names = overlay.get("body_names")
    body_pos_w = overlay.get("body_pos_w")
    if not isinstance(body_names, list) or body_pos_w is None:
        return sim_positions, {}
    target_pos = np.asarray(body_pos_w, dtype=np.float64).reshape(-1, 3)
    target_positions = {
        str(name): tuple(float(value) for value in target_pos[idx])
        for idx, name in enumerate(body_names)
        if idx < target_pos.shape[0]
    }
    return sim_positions, target_positions


def _make_sample(state: dict[str, Any], overlay: dict[str, Any]) -> Sample | None:
    if not overlay.get("clip_active"):
        return None
    frame_idx = int(overlay.get("frame_idx", -1))
    if frame_idx < 0:
        return None
    sample = Sample(frame_idx=frame_idx, sim_time_ms=int(state.get("sim_time_ms", -1)))

    root_state = state.get("robot_root_state")
    target_root = overlay.get("root_pos_w")
    if root_state is not None and target_root is not None:
        root = np.asarray(root_state, dtype=np.float64).reshape(-1)
        target = np.asarray(target_root, dtype=np.float64).reshape(-1)
        if root.size >= 3 and target.size >= 3:
            sample.root_err = float(np.linalg.norm(root[:3] - target[:3]))
            sample.root_pos = tuple(float(value) for value in root[:3])
            sample.target_root_pos = tuple(float(value) for value in target[:3])
        if root.size >= 7:
            sample.root_quat_xyzw = tuple(float(value) for value in root[3:7])
            sample.root_yaw = _yaw_from_xyzw(root[3:7])
        target_quat = overlay.get("root_quat_xyzw")
        if target_quat is None:
            target_quat = overlay.get("root_quat_wxyz")
            target_quat_is_wxyz = target_quat is not None
        else:
            target_quat_is_wxyz = False
        if target_quat is not None:
            target_quat_arr = np.asarray(target_quat, dtype=np.float64).reshape(-1)
            if target_quat_arr.size >= 4:
                if target_quat_is_wxyz:
                    target_quat_arr = target_quat_arr[[1, 2, 3, 0]]
                sample.target_root_quat_xyzw = tuple(float(value) for value in target_quat_arr[:4])
                sample.target_root_yaw = _yaw_from_xyzw(target_quat_arr[:4])

    actor = _actor_state(state)
    target_object = overlay.get("object_pos_w")
    if actor is not None and target_object is not None:
        target = np.asarray(target_object, dtype=np.float64).reshape(-1)
        if target.size >= 3:
            diff = actor[:3] - target[:3]
            sample.object_err = float(np.linalg.norm(diff))
            sample.object_xy_err = float(np.linalg.norm(diff[:2]))
            sample.object_z_err = float(abs(diff[2]))
            sample.object_z = float(actor[2])
            sample.target_object_z = float(target[2])
            sample.object_pos = tuple(float(value) for value in actor[:3])
            sample.target_object_pos = tuple(float(value) for value in target[:3])

    sample.key_errors = _key_body_errors(state, overlay)
    sample.key_body_pos, sample.target_key_body_pos = _key_body_positions(state, overlay)
    if actor is not None and sample.key_body_pos:
        object_pos = np.asarray(actor[:3], dtype=np.float64)
        sample.key_object_dist = {
            name: float(np.linalg.norm(np.asarray(pos, dtype=np.float64) - object_pos))
            for name, pos in sample.key_body_pos.items()
        }
    if sample.key_errors:
        sample.key_mean_err = float(np.mean(list(sample.key_errors.values())))
    sample.object_robot_contacts = int(state.get("object_robot_contact_count", 0))
    sample.object_scene_contacts = int(state.get("object_scene_contact_count", 0))
    bodies = state.get("object_robot_contact_bodies")
    if isinstance(bodies, list):
        sample.contact_bodies = tuple(sorted(str(item) for item in bodies))
    return sample


def _mean(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not valid:
        return None
    return float(np.mean(valid))


def _max(values: list[float | None]) -> float | None:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not valid:
        return None
    return float(np.max(valid))


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def _summarize(samples: list[Sample], buckets: list[tuple[int, int]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "sample_count": len(samples),
        "frame_min": min((sample.frame_idx for sample in samples), default=None),
        "frame_max": max((sample.frame_idx for sample in samples), default=None),
        "buckets": [],
    }
    for start, end in buckets:
        bucket_samples = [sample for sample in samples if start <= sample.frame_idx <= end]
        contact_bodies = sorted({body for sample in bucket_samples for body in sample.contact_bodies})
        summary["buckets"].append(
            {
                "range": [start, end],
                "count": len(bucket_samples),
                "root_err_mean": _mean([sample.root_err for sample in bucket_samples]),
                "root_err_max": _max([sample.root_err for sample in bucket_samples]),
                "object_err_mean": _mean([sample.object_err for sample in bucket_samples]),
                "object_err_max": _max([sample.object_err for sample in bucket_samples]),
                "object_xy_err_mean": _mean([sample.object_xy_err for sample in bucket_samples]),
                "object_z_err_mean": _mean([sample.object_z_err for sample in bucket_samples]),
                "key_mean_err": _mean([sample.key_mean_err for sample in bucket_samples]),
                "object_robot_contacts_mean": _mean([float(sample.object_robot_contacts or 0) for sample in bucket_samples]),
                "object_scene_contacts_mean": _mean([float(sample.object_scene_contacts or 0) for sample in bucket_samples]),
                "object_z_mean": _mean([sample.object_z for sample in bucket_samples]),
                "target_object_z_mean": _mean([sample.target_object_z for sample in bucket_samples]),
                "contact_bodies": contact_bodies,
            }
        )
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        f"samples={summary['sample_count']} frame_min={summary['frame_min']} "
        f"frame_max={summary['frame_max']}"
    )
    for bucket in summary["buckets"]:
        start, end = bucket["range"]
        print(
            f"{start:03d}-{end:03d} n={bucket['count']:3d} "
            f"root={_fmt(bucket['root_err_mean'])}/{_fmt(bucket['root_err_max'])}m "
            f"obj={_fmt(bucket['object_err_mean'])}/{_fmt(bucket['object_err_max'])}m "
            f"obj_xy={_fmt(bucket['object_xy_err_mean'])}m "
            f"obj_z={_fmt(bucket['object_z_err_mean'])}m "
            f"key={_fmt(bucket['key_mean_err'])}m "
            f"contact={_fmt(bucket['object_robot_contacts_mean'], 1)} "
            f"z={_fmt(bucket['object_z_mean'])}->{_fmt(bucket['target_object_z_mean'])}"
        )
        if bucket["contact_bodies"]:
            print(f"  contact_bodies={','.join(bucket['contact_bodies'])}")


def _terminate_process_group(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    port_base = int(args.port_base)
    sim_clock_port = port_base
    sim_state_port = port_base + 2
    perception_obs_port = port_base + 3
    sim_control_port = port_base + 4
    sparse_root_port = port_base + 6
    policy_control_port = port_base + 7
    policy_overlay_port = port_base + 8

    env = os.environ.copy()
    run_dir = REPO_ROOT / "logs" / "sim2sim_runs" / f"{Path(args.clip).stem}__benchmark_{port_base}"
    env.update(
        {
            "PYTHONPATH": ":".join([str(path) for path in SRC_PATHS] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])),
            "PYTHONSAFEPATH": env.get("PYTHONSAFEPATH", "1"),
            "HOLOSOMA_MJ_TRACK_INTERNAL_CORE": "1",
            "RUN_DIR": str(run_dir),
            "RUN_SECONDS": "0",
            "SIM_CLOCK_PORT": str(sim_clock_port),
            "SIM_STATE_PORT": str(sim_state_port),
            "PERCEPTION_OBS_PORT": str(perception_obs_port),
            "SIM_CONTROL_PORT": str(sim_control_port),
            "SPARSE_ROOT_COMMAND_PORT": str(sparse_root_port),
            "POLICY_CONTROL_PORT": str(policy_control_port),
            "HOLOSOMA_POLICY_CONTROL_PORT": str(policy_control_port),
            "POLICY_OVERLAY_PORT": str(policy_overlay_port),
            "HOLOSOMA_POLICY_OVERLAY_PORT": str(policy_overlay_port),
            "HOLOSOMA_SKIP_STIFF_PROMPT": "1",
            "HOLOSOMA_SIM_STATE_INCLUDE_KEY_BODY_STATES": "1",
            "HOLOSOMA_SIM_STATE_INCLUDE_OBJECT_CONTACT_DETAILS": "1",
            "POLICY_STDIO": "log",
            "SIM_READY_TIMEOUT": str(max(1, int(round(float(args.startup_timeout_s))))),
        }
    )
    for override in args.env:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"--env expects KEY=VALUE, got {override!r}")
        env[key] = value

    cmd = ["bash", str(REPO_ROOT / args.launcher), args.clip]
    if args.extra_launcher_arg:
        cmd.extend(args.extra_launcher_arg)

    stdout_path = REPO_ROOT / "logs" / "sim2sim_runs" / f"{args.clip}__benchmark_launcher.log"
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    policy_control = PolicyControlPush(port=policy_control_port)
    policy_control.start()
    sim_sub = SimStateSub(port=sim_state_port)
    overlay_sub = PolicyOverlaySub(port=policy_overlay_port)
    sim_sub.start()
    overlay_sub.start()

    proc: subprocess.Popen[str] | None = None
    samples: list[Sample] = []
    try:
        with stdout_path.open("w", encoding="utf-8") as stdout_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=stdout_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )

            policy_log = run_dir / "policy.log"
            sim_log = run_dir / "mujoco.log"

            launched_at = time.monotonic()
            triggered = False
            last_frame = -1
            startup_deadline = launched_at + float(args.startup_timeout_s)
            collect_deadline = launched_at + float(args.timeout_s)
            while time.monotonic() < collect_deadline:
                if proc.poll() is not None:
                    raise RuntimeError(
                        f"rollout exited with status {proc.returncode}; "
                        f"launcher_log={stdout_path} sim_log={sim_log} policy_log={policy_log}"
                    )

                state = sim_sub.get_state()
                overlay = overlay_sub.get_payload()
                policy_log_text = _read_text(policy_log)
                state_ready = isinstance(state, dict) and state.get("robot_root_state") is not None
                policy_ready = "Policy control receiver started" in policy_log_text

                if not triggered and state_ready and policy_ready:
                    time.sleep(float(args.pre_trigger_wait_s))
                    trigger_actions = [item.strip() for item in str(args.trigger_sequence).split(",") if item.strip()]
                    for idx, action in enumerate(trigger_actions):
                        policy_control.publish(action, source="mj_success_rollout_benchmark")
                        if idx + 1 < len(trigger_actions):
                            time.sleep(float(args.trigger_gap_s))
                    triggered = True
                    launched_at = time.monotonic()
                    collect_deadline = launched_at + float(args.collect_timeout_s)

                if not triggered and time.monotonic() > startup_deadline:
                    raise TimeoutError(
                        f"timed out waiting for rollout readiness; "
                        f"state_ready={state_ready} policy_ready={policy_ready} "
                        f"launcher_log={stdout_path} sim_log={sim_log} policy_log={policy_log}"
                    )

                if triggered and isinstance(state, dict) and isinstance(overlay, dict):
                    sample = _make_sample(state, overlay)
                    if sample is not None:
                        samples.append(sample)
                        last_frame = max(last_frame, sample.frame_idx)
                        if last_frame >= int(args.max_frame):
                            break

                time.sleep(float(args.poll_s))

        if not samples:
            raise RuntimeError(f"no samples collected; launcher_log={stdout_path}")
        summary = _summarize(samples, _parse_buckets(args.buckets))
        summary["ports"] = {
            "sim_clock": sim_clock_port,
            "sim_state": sim_state_port,
            "sim_control": sim_control_port,
            "policy_control": policy_control_port,
            "policy_overlay": policy_overlay_port,
        }
        summary["launcher_log"] = str(stdout_path)
        if args.samples_path:
            samples_path = Path(args.samples_path).expanduser()
            if not samples_path.is_absolute():
                samples_path = REPO_ROOT / samples_path
            samples_path.parent.mkdir(parents=True, exist_ok=True)
            samples_path.write_text(
                json.dumps([asdict(sample) for sample in samples], indent=2, sort_keys=True),
                encoding="utf-8",
            )
            summary["samples_path"] = str(samples_path)
        return summary
    finally:
        if proc is not None:
            _terminate_process_group(proc)
        policy_control.close()
        sim_sub.close()
        overlay_sub.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", default="box_74")
    parser.add_argument("--launcher", default="mj_track_generalist_success.sh")
    parser.add_argument("--port-base", type=int, default=6655)
    parser.add_argument("--max-frame", type=int, default=210)
    parser.add_argument("--buckets", default="0:10,40:60,90:110,140:160,180:200")
    parser.add_argument("--startup-timeout-s", type=float, default=180.0)
    parser.add_argument("--timeout-s", type=float, default=260.0)
    parser.add_argument("--collect-timeout-s", type=float, default=80.0)
    parser.add_argument("--pre-trigger-wait-s", type=float, default=2.0)
    parser.add_argument("--trigger-gap-s", type=float, default=0.6)
    parser.add_argument("--trigger-sequence", default="space,start")
    parser.add_argument("--poll-s", type=float, default=0.01)
    parser.add_argument("--env", action="append", default=[], help="Extra launcher env override as KEY=VALUE")
    parser.add_argument("extra_launcher_arg", nargs="*")
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary")
    parser.add_argument("--samples-path", default="", help="Optional path for raw per-sample JSON")
    args = parser.parse_args()

    summary = run_benchmark(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
        print(f"launcher_log={summary['launcher_log']}")
        print(f"ports={summary['ports']}")


if __name__ == "__main__":
    main()
