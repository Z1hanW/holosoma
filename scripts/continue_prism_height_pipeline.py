#!/usr/bin/env python3
"""Continue the PRISM height update after the strict retarget batch finishes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def run_logged(
    command: list[str],
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"running: {json.dumps(command)}")
    with log_path.open("a", encoding="utf-8") as stream:
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
    return result.returncode


def wait_for_retarget_summary(
    run_root: Path,
    *,
    timeout_hours: float,
    poll_seconds: float,
) -> dict:
    summary_path = run_root / "analysis" / "retarget_batch_summary.json"
    deadline = time.monotonic() + timeout_hours * 3600.0
    while not summary_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"retarget summary did not appear: {summary_path}")
        status_count = len(list((run_root / "status").glob("*.tsv")))
        result_count = len(
            list((run_root / "retarget_strict").glob("*/*_original.npz"))
        )
        log(f"waiting for retarget: status={status_count}/68 results={result_count}/68")
        time.sleep(poll_seconds)
    summary = read_json(summary_path)
    log(
        "retarget launcher finished: "
        f"complete={summary.get('complete', 0)}/{summary.get('total', 0)}"
    )
    return summary


def recover_retarget(args: argparse.Namespace, summary: dict) -> dict:
    if (
        int(summary.get("complete", 0)) == args.expected_count
        and summary.get("status") == "pass"
    ):
        return summary

    env = os.environ.copy()
    env.update(
        {
            "RUN_ROOT": str(args.retarget_run_root),
            "STAGED_ROOT": str(args.staging_root),
            "PATCH_ROOT": str(args.patch_root),
            "OBJ_REPO": str(args.object_repo),
            "EXPECTED_COUNT": str(args.expected_count),
            "WORKERS": "8",
            "MAX_PASSES": str(args.max_recovery_passes),
        }
    )
    returncode = run_logged(
        ["bash", str(args.recovery_script)],
        log_path=args.output_root / "logs" / "retarget_recovery.log",
        env=env,
    )
    recovery_summary = (
        args.retarget_run_root / "analysis" / "recovery_summary.json"
    )
    if not recovery_summary.is_file():
        raise RuntimeError(
            f"retarget recovery returned {returncode} without a summary"
        )
    summary = read_json(recovery_summary)
    if (
        returncode
        or int(summary.get("complete", 0)) != args.expected_count
        or summary.get("status") != "pass"
    ):
        raise RuntimeError(
            "retarget recovery did not reach "
            f"{args.expected_count}/{args.expected_count}: {summary}"
        )
    log(f"retarget recovery reached {args.expected_count}/{args.expected_count}")
    return summary


def scalar(payload: dict[str, np.ndarray], key: str, default: object = None) -> object:
    if key not in payload:
        return default
    return np.asarray(payload[key]).reshape(()).item()


def build_seed_bank(args: argparse.Namespace) -> tuple[Path, dict]:
    sequences = (
        args.retarget_run_root / "sequences.txt"
    ).read_text(encoding="utf-8").splitlines()
    if len(sequences) != args.expected_count or len(set(sequences)) != len(sequences):
        raise ValueError(
            f"expected {args.expected_count} unique sequences, got {len(sequences)}"
        )

    candidates = []
    heights = []
    scales = []
    maximum_violations = []
    rows = []
    expected_shape = None
    for sequence in sequences:
        result_path = (
            args.retarget_run_root
            / "retarget_strict"
            / sequence
            / f"{sequence}_original.npz"
        )
        staged_path = args.staging_root / sequence / "input_for_retarget.npz"
        with np.load(result_path, allow_pickle=True) as data:
            payload = {key: np.asarray(data[key]) for key in data.files}
        with np.load(staged_path, allow_pickle=True) as data:
            height = float(np.asarray(data["human_height_m"]).item())
            expected_frames = len(np.asarray(data["object_poses"]))

        qpos = np.asarray(payload["qpos"], dtype=np.float64)
        partial = bool(scalar(payload, "retarget_partial", False))
        failed = "retarget_failed_frame" in payload
        violation = np.asarray(
            payload.get("retarget_exact_nonpenetration_max_violation_m", [np.inf]),
            dtype=np.float64,
        )
        maximum_violation = float(np.max(violation))
        scale = float(scalar(payload, "retgt_smpl_scale", np.nan))
        if (
            len(qpos) != expected_frames
            or not np.isfinite(qpos).all()
            or partial
            or failed
            or not np.isfinite(maximum_violation)
            or maximum_violation > 1e-6 + 1e-12
            or not np.isfinite(scale)
        ):
            raise ValueError(f"{sequence}: strict retarget result failed seed audit")
        if expected_shape is None:
            expected_shape = qpos.shape
        if qpos.shape != expected_shape:
            raise ValueError(
                f"{sequence}: qpos shape {qpos.shape} != {expected_shape}"
            )
        candidates.append(qpos)
        heights.append(height)
        scales.append(scale)
        maximum_violations.append(maximum_violation)
        rows.append(
            {
                "sequence": sequence,
                "result": str(result_path),
                "frames": len(qpos),
                "qpos_dim": qpos.shape[1],
                "human_height_m": height,
                "retgt_smpl_scale": scale,
                "maximum_exact_violation_m": maximum_violation,
            }
        )

    seed_bank = args.output_root / "seed_bank" / "heightmesh_strict68_qpos.npz"
    seed_bank.parent.mkdir(parents=True, exist_ok=True)
    temporary = seed_bank.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            qpos_candidates=np.stack(candidates),
            source_sequences=np.asarray(sequences),
            strict_source_run=np.asarray(str(args.retarget_run_root)),
            source_max_exact_violation_m=np.asarray(maximum_violations),
            source_human_height_m=np.asarray(heights),
            source_retgt_smpl_scale=np.asarray(scales),
        )
    temporary.replace(seed_bank)
    report = {
        "status": "pass",
        "created_at": utc_now(),
        "seed_bank": str(seed_bank),
        "strict_source_run": str(args.retarget_run_root),
        "sequence_count": len(sequences),
        "qpos_shape": list(np.stack(candidates).shape),
        "human_height_min_m": min(heights),
        "human_height_median_m": float(np.median(heights)),
        "human_height_max_m": max(heights),
        "retgt_smpl_scale_min": min(scales),
        "retgt_smpl_scale_median": float(np.median(scales)),
        "retgt_smpl_scale_max": max(scales),
        "maximum_exact_violation_m": max(maximum_violations),
        "rows": rows,
    }
    atomic_json(args.output_root / "seed_bank" / "report.json", report)
    log(f"built new strict seed bank: {seed_bank}")
    return seed_bank, report


def trajopt_command(
    args: argparse.Namespace,
    seed_bank: Path,
    *,
    sequences: list[str] | None = None,
    force: bool = False,
    iteration: int = 0,
) -> list[str]:
    command = [
        sys.executable,
        str(args.trajopt_repo / "scripts" / "run_prism_trajectory_batch.py"),
        "--staging-root",
        str(args.staging_root),
        "--seed-bank",
        str(seed_bank),
        "--model",
        str(args.model),
        "--output-root",
        str(args.output_root / "trajopt"),
        "--gpus",
        args.gpus,
        "--workers-per-gpu",
        str(args.workers_per_gpu),
        "--seed-collision-candidates",
        str(5 + 2 * iteration),
        "--geometric-iterations",
        str(96 + 48 * iteration),
        "--geometric-velocity-weight",
        "10.0",
        "--geometric-line-search-steps",
        str(16 + 4 * iteration),
        "--geometric-collision-linearization-margin",
        str(5e-4 + iteration * 2.5e-4),
        "--geometric-minimum-trust-radius",
        str(max(1e-3 / (2**iteration), 2.5e-4)),
        "--dynamics-iterations",
        "3",
        "--solver-time-limit",
        "30.0",
        "--max-transition-coefficient",
        "1000.0",
        "--maximum-inaccurate-qp-violation",
        "1.0",
        "--maximum-qvel-consistency",
        "5.0",
        "--line-search-steps",
        "16",
    ]
    if force:
        command.append("--force")
    for sequence in sequences or []:
        command.extend(["--sequence", sequence])
    return command


def run_trajopt(args: argparse.Namespace, seed_bank: Path) -> dict:
    trajopt_root = args.output_root / "trajopt"
    run_logged(
        trajopt_command(args, seed_bank),
        log_path=args.output_root / "logs" / "trajopt_pass_00.log",
    )
    summary_path = trajopt_root / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError("TrajOpt did not produce summary.json")

    for retry in range(1, args.trajopt_retry_passes + 1):
        summary = read_json(summary_path)
        errors = [
            row for row in summary["rows"] if row.get("state") == "error"
        ]
        failed = [
            row for row in summary["rows"] if row.get("state") == "failed"
        ]
        if not errors and not failed:
            break
        targets = [row["sequence"] for row in errors]
        targets.extend(row["sequence"] for row in failed)
        log(
            f"TrajOpt retry {retry}: errors={len(errors)} failed={len(failed)}"
        )
        run_logged(
            trajopt_command(
                args,
                seed_bank,
                sequences=targets,
                force=bool(failed),
                iteration=retry,
            ),
            log_path=args.output_root / "logs" / f"trajopt_pass_{retry:02d}.log",
        )

    summary = read_json(summary_path)
    counts = summary.get("counts", {})
    terminal = sum(
        int(counts.get(key, 0))
        for key in ("accepted", "geometric_only", "failed", "error")
    )
    if terminal != args.expected_count:
        raise RuntimeError(f"TrajOpt summary is incomplete: {counts}")
    log(f"TrajOpt finished: {counts}")
    return summary


def numeric_summary(rows: list[dict], key: str) -> dict | None:
    values = [
        float(row[key])
        for row in rows
        if row.get(key) is not None and np.isfinite(float(row[key]))
    ]
    if not values:
        return None
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": min(values),
        "max": max(values),
    }


def compare_with_baseline(args: argparse.Namespace, summary: dict) -> dict:
    baseline = read_json(args.baseline_root / "summary.json")
    metric_keys = [
        "geometric_initial_keypoint_mm",
        "geometric_final_keypoint_mm",
        "geometric_initial_wrist_mm",
        "geometric_final_wrist_mm",
        "geometric_collision_violation_mm",
        "dynamics_initial_defect",
        "dynamics_final_defect",
        "dynamics_final_wrist_mm",
        "dynamics_collision_violation_mm",
        "qvel_consistency_max",
    ]
    metrics = {}
    for key in metric_keys:
        before = numeric_summary(baseline["rows"], key)
        after = numeric_summary(summary["rows"], key)
        metrics[key] = {
            "baseline": before,
            "height_updated": after,
            "median_delta": (
                after["median"] - before["median"]
                if before is not None and after is not None
                else None
            ),
        }
    comparison = {
        "created_at": utc_now(),
        "status": (
            "pass"
            if not summary.get("counts", {}).get("error", 0)
            and not summary.get("counts", {}).get("failed", 0)
            else "partial"
        ),
        "baseline_root": str(args.baseline_root),
        "height_updated_root": str(args.output_root / "trajopt"),
        "baseline_counts": baseline.get("counts", {}),
        "height_updated_counts": summary.get("counts", {}),
        "metrics": metrics,
    }
    atomic_json(args.output_root / "comparison.json", comparison)
    return comparison


def latest_dynamics_result(sequence_root: Path) -> Path | None:
    reports = sorted((sequence_root / "dynamics").glob("iteration_*/report.json"))
    reports.extend(
        sorted(
            (sequence_root / "dynamics").glob(
                "resume_after_*/iteration_*/report.json"
            )
        )
    )
    return reports[-1].with_name("result.npz") if reports else None


def viewer_object_qpos(object_poses: np.ndarray, quaternion_order: str) -> np.ndarray:
    if quaternion_order == "xyzw":
        quaternion_wxyz = object_poses[:, [3, 0, 1, 2]]
    elif quaternion_order == "wxyz":
        quaternion_wxyz = object_poses[:, :4]
    else:
        raise ValueError(f"unsupported object quaternion order: {quaternion_order}")
    return np.concatenate((object_poses[:, 4:7], quaternion_wxyz), axis=1)


def build_viewer(args: argparse.Namespace, summary: dict) -> dict:
    viewer_root = args.output_root / "viewer"
    retarget_root = viewer_root / "retarget"
    metadata_root = viewer_root / "metadata"
    rows = []
    for row in summary["rows"]:
        sequence = row["sequence"]
        state = row["state"]
        if state not in {"accepted", "geometric_only"}:
            continue
        sequence_root = args.output_root / "trajopt" / "sequences" / sequence
        if state == "accepted":
            result_path = latest_dynamics_result(sequence_root)
            source_stage = "dynamics"
        else:
            result_path = sequence_root / "geometric" / "result.npz"
            source_stage = "geometric"
        if result_path is None or not result_path.is_file():
            raise FileNotFoundError(f"{sequence}: optimization result is missing")

        input_path = args.staging_root / sequence / "input_for_retarget.npz"
        with np.load(input_path, allow_pickle=True) as data:
            staged = {key: np.asarray(data[key]) for key in data.files}
        with np.load(result_path, allow_pickle=True) as data:
            result = {key: np.asarray(data[key]) for key in data.files}
        robot_qpos = np.asarray(
            result.get("qpos", result.get("qpos_optimized")),
            dtype=np.float64,
        )
        object_poses = np.asarray(staged["object_poses"], dtype=np.float64)
        quaternion_order = str(np.asarray(staged["object_pose_quat_order"]).item())
        qpos = np.concatenate(
            (robot_qpos, viewer_object_qpos(object_poses, quaternion_order)),
            axis=1,
        )
        if len(qpos) != len(object_poses) or not np.isfinite(qpos).all():
            raise ValueError(f"{sequence}: invalid viewer qpos")

        destination = retarget_root / sequence / f"{sequence}_original.npz"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                qpos=qpos.astype(np.float32),
                fps=np.asarray(30, dtype=np.int32),
                human_joints=np.asarray(staged["human_joints"], dtype=np.float32),
                object_mesh_scale=np.asarray(1.0, dtype=np.float32),
                object_scale_baked_into_geometry=np.asarray(True),
                retgt_smpl_scale=np.asarray(1.0, dtype=np.float32),
                sequence=np.asarray(sequence),
                optimization_state=np.asarray(state),
                optimization_source_stage=np.asarray(source_stage),
                optimization_result=np.asarray(str(result_path)),
                human_height_m=np.asarray(staged["human_height_m"], dtype=np.float32),
                contact_start_idx=np.asarray(staged["contact_start_idx"]),
                contact_end_idx=np.asarray(staged["contact_end_idx"]),
                object_contact_points_local=np.asarray(
                    staged["object_contact_points_local"], dtype=np.float32
                ),
                object_contact_points_world=np.asarray(
                    staged["object_contact_points_world"], dtype=np.float32
                ),
                object_contact_points_valid=np.asarray(
                    staged["object_contact_points_valid"]
                ),
                object_contact_points_surface_mode=np.asarray(
                    staged["object_contact_points_surface_mode"]
                ),
                retgt_coacd_source_dir=np.asarray(
                    str(input_path.parent / "object_mesh_yup_coacd")
                ),
            )
        temporary.replace(destination)

        mesh_source = Path(str(np.asarray(staged["mesh_file"]).item()))
        mesh_destination = (
            metadata_root / "box_mesh" / sequence / f"{sequence}.obj"
        )
        mesh_destination.parent.mkdir(parents=True, exist_ok=True)
        if mesh_destination.exists() or mesh_destination.is_symlink():
            mesh_destination.unlink()
        mesh_destination.symlink_to(mesh_source)
        rows.append(
            {
                "sequence": sequence,
                "state": state,
                "source_stage": source_stage,
                "frames": len(qpos),
                "qpos_dim": qpos.shape[1],
                "result": str(result_path),
                "viewer_npz": str(destination),
            }
        )
    atomic_json(viewer_root / "adapter_manifest.json", rows)
    if len(rows) != args.expected_count:
        raise RuntimeError(
            f"viewer only has {len(rows)}/{args.expected_count} sequences"
        )
    return {"sequence_count": len(rows), "viewer_root": str(viewer_root)}


def stop_old_viewer(args: argparse.Namespace) -> None:
    pid_path = args.baseline_root / "viewer" / f"viser_{args.viewer_port}.pid"
    if not pid_path.is_file():
        return
    try:
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
        log(f"stopped previous viewer pid={pid}")
        time.sleep(2.0)
    except (ProcessLookupError, ValueError):
        pass


def start_viewer(args: argparse.Namespace) -> dict:
    viewer_root = args.output_root / "viewer"
    stop_old_viewer(args)
    package_root = args.trajopt_repo / "src" / "holosoma_retargeting"
    viewer_script = (
        args.object_repo
        / "vis_scripts"
        / "viser_m_obj"
        / "sequence_retarget_viewer.py"
    )
    bootstrap = (
        "import runpy,sys; "
        f"sys.path.insert(0,{str(package_root)!r}); "
        "import holosoma_retargeting.src.viser_utils; "
        f"runpy.run_path({str(viewer_script)!r}, run_name='__main__')"
    )
    command = [
        sys.executable,
        "-c",
        bootstrap,
        "--retgt_root",
        str(viewer_root / "retarget"),
        "--output_root",
        str(viewer_root / "metadata"),
        "--before_retgt_root",
        str(viewer_root / "metadata" / "before"),
        "--post_scene_root",
        str(viewer_root / "metadata"),
        "--robot_urdf",
        str(args.model.with_suffix(".urdf")),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.viewer_port),
        "--sequence",
        args.viewer_sequence,
        "--object_visual_max_faces",
        "60000",
        "--collision_visual_max_faces",
        "20000",
        "--initial_camera_position",
        "4.4",
        "-0.4",
        "2.9",
        "--initial_camera_look_at",
        "0.9",
        "2.4",
        "0.8",
    ]
    viewer_log = viewer_root / f"viser_{args.viewer_port}.log"
    stream = viewer_log.open("a", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (viewer_root / f"viser_{args.viewer_port}.pid").write_text(
        f"{process.pid}\n", encoding="utf-8"
    )
    time.sleep(5.0)
    if process.poll() is not None:
        stream.close()
        raise RuntimeError(
            f"viewer exited with status {process.returncode}; see {viewer_log}"
        )
    stream.close()
    log(f"viewer started pid={process.pid} port={args.viewer_port}")
    return {
        "pid": process.pid,
        "port": args.viewer_port,
        "sequence": args.viewer_sequence,
        "log": str(viewer_log),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retarget-run-root", type=Path, required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--recovery-script", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--object-repo", type=Path, required=True)
    parser.add_argument("--trajopt-repo", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=68)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--retarget-timeout-hours", type=float, default=24.0)
    parser.add_argument("--max-recovery-passes", type=int, default=12)
    parser.add_argument("--trajopt-retry-passes", type=int, default=3)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--viewer-port", type=int, default=9304)
    parser.add_argument("--viewer-sequence", default="prism_cf_bin_m0_v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.expanduser().resolve())
    args.output_root.mkdir(parents=True, exist_ok=False)
    atomic_json(
        args.output_root / "pipeline_config.json",
        {
            "created_at": utc_now(),
            **{
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
    )
    state = {"status": "running", "started_at": utc_now()}
    atomic_json(args.output_root / "pipeline_state.json", state)
    try:
        summary = wait_for_retarget_summary(
            args.retarget_run_root,
            timeout_hours=args.retarget_timeout_hours,
            poll_seconds=args.poll_seconds,
        )
        summary = recover_retarget(args, summary)
        seed_bank, seed_report = build_seed_bank(args)
        trajopt_summary = run_trajopt(args, seed_bank)
        comparison = compare_with_baseline(args, trajopt_summary)
        viewer = build_viewer(args, trajopt_summary)
        viewer.update(start_viewer(args))
        state.update(
            {
                "status": "complete",
                "completed_at": utc_now(),
                "retarget_complete": summary["complete"],
                "seed_bank": str(seed_bank),
                "seed_sequence_count": seed_report["sequence_count"],
                "trajopt_counts": trajopt_summary["counts"],
                "comparison": comparison,
                "viewer": viewer,
            }
        )
        atomic_json(args.output_root / "pipeline_state.json", state)
        return 0
    except Exception as exc:
        state.update(
            {
                "status": "failed",
                "failed_at": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_json(args.output_root / "pipeline_state.json", state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
