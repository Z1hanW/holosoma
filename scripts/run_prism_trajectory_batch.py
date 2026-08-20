#!/usr/bin/env python3
"""Run resumable geometric and dynamics TrajOpt over a PRISM staging set."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

import numpy as np


REQUIRED_INPUT_KEYS = {
    "human_joints",
    "object_poses",
    "joint_names",
    "object_pose_quat_order",
    "mesh_file",
    "contact_start_idx",
    "contact_end_idx",
    "sequence",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def input_metadata(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        missing = sorted(REQUIRED_INPUT_KEYS.difference(data.files))
        if missing:
            raise ValueError(f"{path} is missing keys: {', '.join(missing)}")
        sequence = str(np.asarray(data["sequence"]).item())
        frames = int(len(data["human_joints"]))
        object_frames = int(len(data["object_poses"]))
    if frames != object_frames:
        raise ValueError(f"{path} has inconsistent frame counts")
    return {
        "sequence": sequence,
        "input": str(path.resolve()),
        "frames": frames,
    }


def discover_inputs(staging_root: Path) -> list[dict]:
    by_sequence: dict[str, dict] = {}
    duplicate_paths: dict[str, list[str]] = {}
    for path in sorted(staging_root.rglob("input_for_retarget.npz")):
        metadata = input_metadata(path)
        sequence = metadata["sequence"]
        duplicate_paths.setdefault(sequence, []).append(str(path.resolve()))
        previous = by_sequence.get(sequence)
        if previous is None or str(path) < previous["input"]:
            by_sequence[sequence] = metadata
    result = []
    for sequence in sorted(by_sequence):
        item = by_sequence[sequence]
        item["duplicate_inputs"] = duplicate_paths[sequence]
        sequence_root = Path(item["input"]).parent
        preferred_mesh = sequence_root / "object_mesh_yup_coacd" / "object.obj"
        item["mesh"] = (
            str(preferred_mesh.resolve())
            if preferred_mesh.is_file()
            else None
        )
        result.append(item)
    return result


def dynamics_report_paths(dynamics_root: Path) -> list[Path]:
    reports = sorted(dynamics_root.glob("iteration_*/report.json"))
    reports.extend(
        sorted(dynamics_root.glob("resume_after_*/iteration_*/report.json"))
    )
    return reports


def latest_dynamics_report(dynamics_root: Path) -> tuple[Path | None, dict | None]:
    reports = dynamics_report_paths(dynamics_root)
    if not reports:
        return None, None
    return reports[-1], read_json(reports[-1])


def classify_reports(
    geometric: dict,
    dynamics_reports: list[dict],
    *,
    maximum_qvel_consistency: float,
) -> tuple[str, list[str]]:
    reasons = []
    if not geometric.get("collision_feasible", False):
        reasons.append("geometric_collision_gate_failed")
    if not geometric.get("tracking_improved", False):
        reasons.append("geometric_tracking_not_improved")
    if not dynamics_reports:
        reasons.append("missing_dynamics_report")
        return "failed", reasons

    first = dynamics_reports[0]
    final = dynamics_reports[-1]
    initial_defect = float(first["initial_linearized_defect_mean"])
    final_defect = float(final["final_nonlinear_defect_mean"])
    if final_defect > initial_defect + 1e-9:
        reasons.append("dynamics_defect_regressed")
    if (
        float(final["final_collision_violation_m"])
        > float(final["collision_violation_limit_m"]) + 1e-12
    ):
        reasons.append("dynamics_collision_gate_failed")
    if (
        float(final["final_qvel_consistency_max"])
        > maximum_qvel_consistency + 1e-9
    ):
        reasons.append("qvel_consistency_gate_failed")
    accepted_steps = sum(
        float(report["accepted_step_size"]) > 0.0
        for report in dynamics_reports
    )
    if accepted_steps == 0:
        reasons.append("no_dynamics_step_accepted")

    hard_failures = [
        reason
        for reason in reasons
        if reason
        not in {
            "no_dynamics_step_accepted",
        }
    ]
    if hard_failures:
        return "failed", reasons
    if reasons:
        return "geometric_only", reasons
    return "accepted", reasons


def geometric_gate_reasons(geometric: dict) -> list[str]:
    reasons = []
    if not geometric.get("collision_feasible", False):
        reasons.append("geometric_collision_gate_failed")
    if not geometric.get("tracking_improved", False):
        reasons.append("geometric_tracking_not_improved")
    return reasons


def metric_row(sequence_root: Path, status: dict) -> dict:
    geometric_path = sequence_root / "geometric" / "report.json"
    geometric = read_json(geometric_path) if geometric_path.is_file() else {}
    dynamics_paths = dynamics_report_paths(sequence_root / "dynamics")
    dynamics = [read_json(path) for path in dynamics_paths]
    first_dynamics = dynamics[0] if dynamics else {}
    final_dynamics = dynamics[-1] if dynamics else {}
    initial = geometric.get("initial", {})
    final = geometric.get("final", {})
    return {
        "sequence": status["sequence"],
        "state": status["state"],
        "gpu": status.get("gpu"),
        "seed_index": geometric.get("seed_index"),
        "seed_sequence": geometric.get("seed_sequence"),
        "geometric_initial_keypoint_mm": (
            1000.0 * initial["mean_keypoint_error_m"] if initial else None
        ),
        "geometric_final_keypoint_mm": (
            1000.0 * final["mean_keypoint_error_m"] if final else None
        ),
        "geometric_initial_wrist_mm": (
            1000.0 * initial["contact_wrist_error_m"] if initial else None
        ),
        "geometric_final_wrist_mm": (
            1000.0 * final["contact_wrist_error_m"] if final else None
        ),
        "geometric_collision_violation_mm": (
            1000.0 * final["maximum_collision_violation_m"]
            if final
            else None
        ),
        "dynamics_iterations": len(dynamics),
        "dynamics_initial_defect": first_dynamics.get(
            "initial_linearized_defect_mean"
        ),
        "dynamics_final_defect": final_dynamics.get(
            "final_nonlinear_defect_mean"
        ),
        "dynamics_final_wrist_mm": (
            1000.0 * final_dynamics["final_contact_wrist_error_m"]
            if final_dynamics
            else None
        ),
        "dynamics_collision_violation_mm": (
            1000.0 * final_dynamics["final_collision_violation_m"]
            if final_dynamics
            else None
        ),
        "qvel_consistency_max": final_dynamics.get(
            "final_qvel_consistency_max"
        ),
        "accepted_dynamics_steps": sum(
            float(report["accepted_step_size"]) > 0.0 for report in dynamics
        ),
        "reasons": ";".join(status.get("reasons", [])),
        "elapsed_s": status.get("elapsed_s"),
    }


def write_summary(output_root: Path, manifest: list[dict]) -> dict:
    rows = []
    counts: dict[str, int] = {}
    for item in manifest:
        sequence_root = output_root / "sequences" / item["sequence"]
        status_path = sequence_root / "status.json"
        if status_path.is_file():
            status = read_json(status_path)
        else:
            status = {
                "sequence": item["sequence"],
                "state": "pending",
            }
        counts[status["state"]] = counts.get(status["state"], 0) + 1
        rows.append(metric_row(sequence_root, status))
    summary = {
        "updated_at": utc_now(),
        "sequence_count": len(manifest),
        "counts": counts,
        "rows": rows,
    }
    atomic_json(output_root / "summary.json", summary)
    fieldnames = list(rows[0]) if rows else ["sequence", "state"]
    temporary = output_root / "summary.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_root / "summary.csv")
    return summary


def run_command(command: list[str], *, env: dict[str, str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"[{utc_now()}] command={json.dumps(command)}\n")
        stream.flush()
        result = subprocess.run(
            command,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
        stream.write(f"[{utc_now()}] returncode={result.returncode}\n")
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def run_sequence(
    item: dict,
    *,
    args: argparse.Namespace,
    repo_root: Path,
    gpu: int,
    gpu_lock: threading.Lock,
) -> dict:
    sequence = item["sequence"]
    sequence_root = args.output_root / "sequences" / sequence
    if args.force and sequence_root.exists():
        shutil.rmtree(sequence_root)
    sequence_root.mkdir(parents=True, exist_ok=True)
    status_path = sequence_root / "status.json"
    if status_path.is_file() and not args.force:
        existing = read_json(status_path)
        if existing.get("state") in {"accepted", "geometric_only"}:
            return existing

    started = time.monotonic()
    status = {
        "sequence": sequence,
        "state": "running",
        "gpu": gpu,
        "started_at": utc_now(),
        "input": item["input"],
        "mesh": item["mesh"],
    }
    atomic_json(status_path, status)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    package_root = str(repo_root / "src" / "holosoma_retargeting")
    env["PYTHONPATH"] = (
        package_root
        if not env.get("PYTHONPATH")
        else package_root + os.pathsep + env["PYTHONPATH"]
    )
    try:
        geometric_root = sequence_root / "geometric"
        geometric_result = geometric_root / "result.npz"
        geometric_report = geometric_root / "report.json"
        if not geometric_result.is_file() or not geometric_report.is_file():
            command = [
                sys.executable,
                str(repo_root / "scripts" / "run_real_trajectory_canary.py"),
                "--input",
                item["input"],
                "--seed-bank",
                str(args.seed_bank),
                "--model",
                str(args.model),
                "--seed-index",
                "auto",
                "--seed-collision-candidates",
                str(args.seed_collision_candidates),
                "--max-iterations",
                str(args.geometric_iterations),
                "--velocity-weight",
                str(args.geometric_velocity_weight),
                "--minimum-trust-radius",
                str(args.geometric_minimum_trust_radius),
                "--line-search-steps",
                str(args.geometric_line_search_steps),
                "--collision-linearization-margin",
                str(args.geometric_collision_linearization_margin),
                "--backend",
                "osqp",
                "--output",
                str(geometric_result),
                "--report",
                str(geometric_report),
            ]
            if item["mesh"]:
                command.extend(["--mesh", item["mesh"]])
            run_command(
                command,
                env=env,
                log=geometric_root / "run.log",
            )

        geometric = read_json(geometric_report)
        geometric_reasons = geometric_gate_reasons(geometric)
        if geometric_reasons:
            status.update(
                {
                    "state": "failed",
                    "reasons": geometric_reasons,
                    "completed_at": utc_now(),
                    "elapsed_s": time.monotonic() - started,
                    "geometric_report": str(geometric_report),
                    "dynamics_reports": [],
                }
            )
            atomic_json(status_path, status)
            return status

        dynamics_root = sequence_root / "dynamics"
        env["ITERATIONS"] = str(args.dynamics_iterations)
        env["PYTHON_BIN"] = sys.executable
        env["CUDA_DEVICE"] = str(gpu)
        command = [
            "bash",
            str(repo_root / "scripts" / "run_real_dynamics_iterations.sh"),
            item["input"],
            str(geometric_result),
            str(args.model),
            str(dynamics_root),
            "--backend",
            "cupy-direct-cuda",
            "--solver-time-limit",
            str(args.solver_time_limit),
            "--max-transition-coefficient",
            str(args.max_transition_coefficient),
            "--allow-inaccurate-qp",
            "--project-inaccurate-qp",
            "--maximum-inaccurate-qp-violation",
            str(args.maximum_inaccurate_qp_violation),
            "--maximum-qvel-consistency",
            str(args.maximum_qvel_consistency),
            "--line-search-steps",
            str(args.line_search_steps),
        ]
        if item["mesh"]:
            command.extend(["--mesh", item["mesh"]])
        report_path, _ = latest_dynamics_report(dynamics_root)
        completed_iterations = len(dynamics_report_paths(dynamics_root))
        if completed_iterations < args.dynamics_iterations:
            if completed_iterations:
                env["ITERATIONS"] = str(
                    args.dynamics_iterations - completed_iterations
                )
                command[3] = str(report_path.parent / "result.npz")
                resume_root = dynamics_root / f"resume_after_{completed_iterations:02d}"
                command[5] = str(resume_root)
            with gpu_lock:
                run_command(
                    command,
                    env=env,
                    log=dynamics_root / "driver.log",
                )

        dynamics_paths = dynamics_report_paths(dynamics_root)
        dynamics = [read_json(path) for path in dynamics_paths]
        state, reasons = classify_reports(
            geometric,
            dynamics,
            maximum_qvel_consistency=args.maximum_qvel_consistency,
        )
        status.update(
            {
                "state": state,
                "reasons": reasons,
                "completed_at": utc_now(),
                "elapsed_s": time.monotonic() - started,
                "geometric_report": str(geometric_report),
                "dynamics_reports": [str(path) for path in dynamics_paths],
            }
        )
    except Exception as exc:
        status.update(
            {
                "state": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "completed_at": utc_now(),
                "elapsed_s": time.monotonic() - started,
            }
        )
    atomic_json(status_path, status)
    return status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed-collision-candidates", type=int, default=5)
    parser.add_argument("--geometric-iterations", type=int, default=96)
    parser.add_argument("--geometric-velocity-weight", type=float, default=10.0)
    parser.add_argument("--geometric-line-search-steps", type=int, default=16)
    parser.add_argument(
        "--geometric-collision-linearization-margin",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--geometric-minimum-trust-radius",
        type=float,
        default=1e-3,
    )
    parser.add_argument("--dynamics-iterations", type=int, default=3)
    parser.add_argument("--solver-time-limit", type=float, default=30.0)
    parser.add_argument("--max-transition-coefficient", type=float, default=1000.0)
    parser.add_argument("--maximum-inaccurate-qp-violation", type=float, default=1.0)
    parser.add_argument("--maximum-qvel-consistency", type=float, default=5.0)
    parser.add_argument("--line-search-steps", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.staging_root = args.staging_root.expanduser().resolve()
    args.seed_bank = args.seed_bank.expanduser().resolve()
    args.model = args.model.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]
    gpus = [int(value) for value in args.gpus.split(",") if value.strip()]
    if not gpus:
        raise ValueError("--gpus must name at least one device")
    if args.workers_per_gpu <= 0:
        raise ValueError("--workers-per-gpu must be positive")
    manifest = discover_inputs(args.staging_root)
    if args.sequence:
        selected = set(args.sequence)
        manifest = [item for item in manifest if item["sequence"] in selected]
        missing = sorted(selected.difference(item["sequence"] for item in manifest))
        if missing:
            raise ValueError(f"sequences not found: {', '.join(missing)}")
    if args.limit is not None:
        manifest = manifest[: args.limit]
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_root / "manifest.json",
        {
            "created_at": utc_now(),
            "staging_root": str(args.staging_root),
            "seed_bank": str(args.seed_bank),
            "model": str(args.model),
            "sequence_count": len(manifest),
            "sequences": manifest,
        },
    )
    write_summary(args.output_root, manifest)
    if args.manifest_only:
        return 0

    summary_lock = threading.Lock()
    executors = [
        ThreadPoolExecutor(max_workers=args.workers_per_gpu)
        for _ in gpus
    ]
    gpu_locks = [threading.Lock() for _ in gpus]
    try:
        futures = {
            executors[index % len(gpus)].submit(
                run_sequence,
                item,
                args=args,
                repo_root=repo_root,
                gpu=gpus[index % len(gpus)],
                gpu_lock=gpu_locks[index % len(gpus)],
            ): item
            for index, item in enumerate(manifest)
        }
        for future in as_completed(futures):
            status = future.result()
            print(
                json.dumps(
                    {
                        "sequence": status["sequence"],
                        "state": status["state"],
                        "gpu": status.get("gpu"),
                        "elapsed_s": status.get("elapsed_s"),
                    }
                ),
                flush=True,
            )
            with summary_lock:
                write_summary(args.output_root, manifest)
    finally:
        for executor in executors:
            executor.shutdown(wait=True)
    summary = write_summary(args.output_root, manifest)
    return 1 if summary["counts"].get("error", 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
