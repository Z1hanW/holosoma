#!/usr/bin/env python3
"""Record the exact CORL79+debug30 immutable bank with MuJoCo policy rollout."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import math
import os
import queue
import shutil
import subprocess
import threading
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BANK = Path(
    "/data/holosoma_inputs/corl79_plus_debug30_decoupled_turn_forward_v1/by-source/"
    "307e9662d498bd507b9d17ca9abf74a3654f7bf66ac6ab989c6f19c3889bddef"
)
DEFAULT_MODEL = Path(
    "/data/holosoma_eval_audits/sx_0mc_checkpoint_progression_native_20260808_185459/"
    "runs/0mcqao8k/wandb_files/model_40000/model_40000.onnx"
)
EXPECTED_MODEL_SHA256 = "b2eb5206e255efb7a8974def2aa533f3ad493378affd351829521d11c38a4483"
EXPECTED_BANK_MANIFEST_SHA256 = "2de9ee5ca188b70e877c32dd9f0d2975eea99d11aa077bb077cf06ea9ab897bb"
EXPECTED_OBJECT_MAP_SHA256 = "70b466aad04837a79f6dd0f4491cb345a73c687209981acd3eb7f4a0365d8f5c"
EXPECTED_CATEGORY_COUNTS = {"box": 25, "ball": 9, "barrel": 36, "bin": 39}
EXPECTED_SOURCE_COUNTS = {"corl79": 79, "debug30": 30}
REQUIRED_RUNTIME_LINES = (
    "iterations=100 noslip_iterations=10 impratio=10.0",
    "Limiting MuJoCo object contacts to 20 carry body(ies)",
    "friction=[1.6, 0.02, 0.005]",
    "camera_reset_randomization=False",
)


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


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def clip_slug(clip_id: str) -> str:
    return clip_id.replace("__", "_")


def build_inputs(bank: Path, model: Path) -> dict[str, Any]:
    manifest_path = bank / "manifest.json"
    object_map_path = bank / "_clip_object_urdf_map.json"
    if sha256(model) != EXPECTED_MODEL_SHA256:
        raise ValueError("Exact 0mcqao8k/model_40000 ONNX digest changed")
    if sha256(manifest_path) != EXPECTED_BANK_MANIFEST_SHA256:
        raise ValueError("Exact 109-bank manifest digest changed")
    if sha256(object_map_path) != EXPECTED_OBJECT_MAP_SHA256:
        raise ValueError("Exact 109-bank object map digest changed")

    manifest = load_json(manifest_path)
    object_map = load_json(object_map_path)
    manifest_clips = manifest.get("clips")
    mapped_clips = object_map.get("clips")
    if not isinstance(manifest_clips, list) or not isinstance(mapped_clips, dict):
        raise ValueError("Malformed immutable bank manifest/object map")
    if len(manifest_clips) != 109 or len(mapped_clips) != 109:
        raise ValueError("Immutable bank must contain exactly 109 clips")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, clip in enumerate(manifest_clips, start=1):
        if not isinstance(clip, dict) or not isinstance(clip.get("clip_id"), str):
            raise ValueError(f"Malformed clip row at canonical index {index}")
        clip_id = clip["clip_id"]
        if clip_id in seen:
            raise ValueError(f"Duplicate canonical clip ID: {clip_id}")
        seen.add(clip_id)
        mapped = mapped_clips.get(clip_id)
        if not isinstance(mapped, dict):
            raise ValueError(f"Missing object mapping: {clip_id}")
        source_label = mapped.get("merged_bank_source_label")
        category = clip.get("category")
        if source_label not in EXPECTED_SOURCE_COUNTS or category not in EXPECTED_CATEGORY_COUNTS:
            raise ValueError(f"Unexpected source/category for {clip_id}: {source_label}/{category}")
        motion = bank / f"{clip_id}.npz"
        urdf_rel = mapped.get("object_urdf_path")
        if not isinstance(urdf_rel, str):
            raise ValueError(f"Missing object URDF path for {clip_id}")
        urdf = bank / urdf_rel
        if not motion.is_file() or not urdf.is_file():
            raise FileNotFoundError(f"Incomplete motion/object pair: {clip_id}")
        expected_motion_sha = clip.get("derived_npz_sha256")
        actual_motion_sha = sha256(motion)
        if actual_motion_sha != expected_motion_sha:
            raise ValueError(f"Motion SHA mismatch for {clip_id}")
        rows.append(
            {
                "index": index,
                "clip_id": clip_id,
                "clip_slug": clip_slug(clip_id),
                "source_label": source_label,
                "category": category,
                "motion_path": str(motion.resolve()),
                "motion_sha256": actual_motion_sha,
                "object_urdf_path": str(urdf.resolve()),
                "object_urdf_sha256": sha256(urdf),
            }
        )

    category_counts = Counter(str(row["category"]) for row in rows)
    source_counts = Counter(str(row["source_label"]) for row in rows)
    if dict(category_counts) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"Category counts changed: {dict(category_counts)}")
    if dict(source_counts) != EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"Source counts changed: {dict(source_counts)}")
    return {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "semantics": "0mcqao8k_model40000_mujoco_full109_post_lift_persistent_forward015",
        "bank_root": str(bank),
        "bank_manifest_sha256": EXPECTED_BANK_MANIFEST_SHA256,
        "object_map_sha256": EXPECTED_OBJECT_MAP_SHA256,
        "model_onnx": str(model),
        "model_onnx_sha256": EXPECTED_MODEL_SHA256,
        "clip_count": len(rows),
        "source_counts": dict(source_counts),
        "category_counts": dict(category_counts),
        "rows": rows,
    }


def validate_completed_run(
    run_dir: Path,
    *,
    expected_lift_threshold_m: float,
    expected_latest_forward_actor_sim_time_s: float | None,
) -> dict[str, Any]:
    audit = load_json(run_dir / "audit/command_audit.json")
    gate = load_json(run_dir / "audit/gate_summary.json")
    if not audit.get("passed") or not audit.get("command_contract_passed"):
        raise ValueError("command audit did not pass")
    if abs(float(gate.get("lift_rel_z_delta_m", -1.0)) - expected_lift_threshold_m) > 1.0e-9:
        raise ValueError("lift threshold does not match this batch contract")
    if "lift_rel_z_delta_m" in audit and abs(
        float(audit["lift_rel_z_delta_m"]) - expected_lift_threshold_m
    ) > 1.0e-9:
        raise ValueError("audited lift threshold does not match this batch contract")
    if int(audit.get("policy_io_rows", -1)) != 501:
        raise ValueError("policy-I/O row count is not 501")
    if not audit.get("drop_input_all_zero"):
        raise ValueError("drop input was not zero for the full rollout")
    expected_deadline_ms = (
        None
        if expected_latest_forward_actor_sim_time_s is None
        else expected_latest_forward_actor_sim_time_s * 1000.0
    )
    gate_deadline_ms = gate.get("latest_forward_actor_sim_time_ms")
    if gate_deadline_ms != expected_deadline_ms:
        raise ValueError("actor-forward deadline does not match this batch contract")
    if expected_deadline_ms is not None:
        if not audit.get("triggered"):
            raise ValueError("hard actor-forward deadline batch contains an untriggered rollout")
        if audit.get("trigger_source") not in {"height", "time_fallback"}:
            raise ValueError("hard actor-forward deadline batch has an invalid trigger source")
        first_forward_ms = float(audit.get("first_forward_actor_sim_time_ms", math.inf))
        if first_forward_ms > expected_deadline_ms:
            raise ValueError("forward reached the actor after the hard deadline")
    log = (run_dir / "runtime/mujoco.log").read_text(encoding="utf-8", errors="replace")
    if not all(line in log for line in REQUIRED_RUNTIME_LINES):
        raise ValueError("frozen MuJoCo runtime contract is incomplete")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bank-root", type=Path, default=DEFAULT_BANK)
    parser.add_argument("--model-onnx", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--port-base", type=int, default=8200)
    parser.add_argument("--lift-rel-z-delta-m", type=float, default=0.30)
    parser.add_argument("--latest-forward-actor-sim-time-s", type=float, default=None)
    parser.add_argument("--deadline-publish-lead-ms", type=int, default=40)
    parser.add_argument(
        "--select-prior-not-triggered-from",
        type=Path,
        default=None,
        help="Run only rows reported as not_triggered by a prior exact-bank audit root.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_root = args.output_root.expanduser().resolve()
    bank = args.bank_root.expanduser().resolve()
    model = args.model_onnx.expanduser().resolve()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if not 0.0 < args.lift_rel_z_delta_m < 2.0:
        raise ValueError("--lift-rel-z-delta-m must be in (0, 2)")
    if args.latest_forward_actor_sim_time_s is not None and not (
        0.0 < args.latest_forward_actor_sim_time_s < 10.02
    ):
        raise ValueError("--latest-forward-actor-sim-time-s must be in (0, 10.02)")
    if args.deadline_publish_lead_ms < 0:
        raise ValueError("--deadline-publish-lead-ms must be non-negative")
    if (
        args.latest_forward_actor_sim_time_s is not None
        and args.deadline_publish_lead_ms >= args.latest_forward_actor_sim_time_s * 1000.0
    ):
        raise ValueError("--deadline-publish-lead-ms must be smaller than the actor deadline")
    gpu_ids = [value.strip() for value in args.gpu_ids.split(",") if value.strip()]
    if len(gpu_ids) < args.workers:
        raise ValueError("Provide at least one distinct --gpu-ids entry per worker")
    if args.port_base < 1024 or args.port_base + (args.workers - 1) * 32 + 8 > 65535:
        raise ValueError("Port range is invalid")

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "runs").mkdir(exist_ok=True)
    (output_root / "failed_attempts").mkdir(exist_ok=True)
    (output_root / "logs").mkdir(exist_ok=True)
    inputs = build_inputs(bank, model)
    if args.select_prior_not_triggered_from is not None:
        prior_root = args.select_prior_not_triggered_from.expanduser().resolve()
        prior_inputs = load_json(prior_root / "input_manifest.json")
        prior_batch = load_json(prior_root / "batch_rollout_manifest.json")
        for key in ("bank_manifest_sha256", "object_map_sha256", "model_onnx_sha256"):
            if prior_inputs.get(key) != inputs.get(key):
                raise ValueError(f"Prior audit root disagrees on {key}")
        prior_results = prior_batch.get("results")
        if not isinstance(prior_results, list):
            raise ValueError("Prior batch manifest has no result rows")
        selected_indices = {
            int(item["index"])
            for item in prior_results
            if isinstance(item, dict)
            and isinstance(item.get("audit"), dict)
            and item["audit"].get("rollout_status") == "not_triggered"
        }
        if not selected_indices:
            raise ValueError("Prior batch has no not_triggered rows")
        selected_rows = [row for row in inputs["rows"] if int(row["index"]) in selected_indices]
        if len(selected_rows) != len(selected_indices):
            raise ValueError("Prior not-triggered selection is not a subset of the exact bank")
        inputs["rows"] = selected_rows
        inputs["clip_count"] = len(selected_rows)
        inputs["source_counts"] = dict(Counter(str(row["source_label"]) for row in selected_rows))
        inputs["category_counts"] = dict(Counter(str(row["category"]) for row in selected_rows))
        inputs["selection"] = {
            "semantics": "prior_rollout_status_equals_not_triggered",
            "prior_audit_root": str(prior_root),
            "prior_batch_rollout_manifest_sha256": sha256(prior_root / "batch_rollout_manifest.json"),
            "prior_threshold_m": float(prior_inputs.get("lift_rel_z_delta_m", 0.30)),
            "selected_original_indices": sorted(selected_indices),
        }
        gate_slug = f"{args.lift_rel_z_delta_m:.2f}".replace(".", "p")
        inputs["semantics"] = f"0mcqao8k_model40000_mujoco_prior_not_triggered_retrigger_gate{gate_slug}"
    inputs["lift_rel_z_delta_m"] = float(args.lift_rel_z_delta_m)
    inputs["latest_forward_actor_sim_time_s"] = args.latest_forward_actor_sim_time_s
    inputs["deadline_publish_lead_ms"] = int(args.deadline_publish_lead_ms)
    if args.latest_forward_actor_sim_time_s is not None:
        deadline_slug = f"{args.latest_forward_actor_sim_time_s:.2f}".replace(".", "p")
        inputs["semantics"] += f"_or_actor_deadline{deadline_slug}s"
    input_manifest_path = output_root / "input_manifest.json"
    if input_manifest_path.exists():
        old = load_json(input_manifest_path)
        compare_keys = [
            "bank_root",
            "bank_manifest_sha256",
            "object_map_sha256",
            "model_onnx_sha256",
            "rows",
            "lift_rel_z_delta_m",
            "latest_forward_actor_sim_time_s",
            "deadline_publish_lead_ms",
        ]
        if args.select_prior_not_triggered_from is not None:
            compare_keys.append("selection")
        for key in compare_keys:
            if old.get(key) != inputs.get(key):
                raise ValueError(f"Existing input manifest disagrees on {key}")
        inputs = old
    else:
        write_json_atomic(input_manifest_path, inputs)

    slot_queue: queue.Queue[int] = queue.Queue()
    for slot in range(args.workers):
        slot_queue.put(slot)
    progress_lock = threading.Lock()
    completed_counter = 0
    expected_count = int(inputs["clip_count"])

    def record_one(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed_counter
        number = f"{int(row['index']):03d}"
        run_name = f"{number}_{row['clip_slug']}"
        run_dir = output_root / "runs" / run_name
        input_record = {
            "clip": row,
            "command_contract": {
                "pre_lift_root_command": [0.0, 0.0, 0.0],
                "lift_relative_world_z_threshold_m": float(args.lift_rel_z_delta_m),
                "latest_forward_actor_sim_time_s": args.latest_forward_actor_sim_time_s,
                "deadline_publish_lead_ms": int(args.deadline_publish_lead_ms),
                "consecutive_steps": 0,
                "post_lift_actor_input": [0.15, 0.0, 0.0, 0.0],
                "drop_input_all_steps": 0.0,
                "preserve_native_pickup": True,
                "heading_lock": False,
                "semantics": "legacy_constant_robot_heading_frame",
            },
        }
        try:
            audit = validate_completed_run(
                run_dir,
                expected_lift_threshold_m=float(args.lift_rel_z_delta_m),
                expected_latest_forward_actor_sim_time_s=args.latest_forward_actor_sim_time_s,
            )
            result = {"index": row["index"], "clip_id": row["clip_id"], "reused": True, "audit": audit}
            with progress_lock:
                completed_counter += 1
                print(
                    f"[batch {completed_counter:03d}/{expected_count:03d}] REUSE {run_name}: "
                    f"{audit['rollout_status']}",
                    flush=True,
                )
            return result
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

        if run_dir.exists() and any(run_dir.iterdir()):
            retry_stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            preserved = output_root / "failed_attempts" / f"{run_name}__retry_{retry_stamp}"
            shutil.move(str(run_dir), str(preserved))
        run_dir.mkdir(parents=True, exist_ok=False)
        write_json_atomic(run_dir / "input.json", input_record)

        slot = slot_queue.get()
        try:
            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": gpu_ids[slot],
                    "SIM_COPY_COLLISION_GEOMS_FROM_ROBOT_XML": "0",
                    "SIM_COPY_CONTACT_PAIRS_FROM_ROBOT_XML": "0",
                    "MUJOCO_LIMIT_OBJECT_CONTACTS_TO_CARRY_BODIES": "1",
                    "HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_LATERAL_FRICTION": "1.6",
                    "HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_SPIN_FRICTION": "0.02",
                    "HOLOSOMA_MUJOCO_TRAINING_OBJECT_CONTACT_ROLLING_FRICTION": "0.005",
                    "HOLOSOMA_MUJOCO_NOSLIP_ITERATIONS": "10",
                    "HOLOSOMA_MUJOCO_IMPRATIO": "10",
                }
            )
            runner_log = run_dir / "batch_runner.log"
            with runner_log.open("w", encoding="utf-8") as stream:
                runner_command = [
                    str(repo_root / "scripts/run_mujoco_forward_after_lift_rollout.sh"),
                    "--motion-file",
                    str(row["motion_path"]),
                    "--object-urdf",
                    str(row["object_urdf_path"]),
                    "--model-onnx",
                    str(model),
                    "--output-dir",
                    str(run_dir),
                    "--port-base",
                    str(args.port_base + slot * 32),
                    "--forward-command-m",
                    "0.15",
                    "--lift-rel-z-delta-m",
                    f"{args.lift_rel_z_delta_m:.8g}",
                    "--deadline-publish-lead-ms",
                    str(args.deadline_publish_lead_ms),
                    "--actor-steps",
                    "501",
                    "--startup-timeout-s",
                    "600",
                    "--rollout-timeout-s",
                    "600",
                ]
                if args.latest_forward_actor_sim_time_s is not None:
                    runner_command.extend(
                        [
                            "--latest-forward-actor-sim-time-s",
                            f"{args.latest_forward_actor_sim_time_s:.8g}",
                        ]
                    )
                runner = subprocess.run(
                    runner_command,
                    cwd=repo_root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            audit_stdout = run_dir / "audit/audit_stdout.json"
            audit_stderr = run_dir / "audit/audit_stderr.log"
            with audit_stdout.open("w", encoding="utf-8") as stdout, audit_stderr.open(
                "w", encoding="utf-8"
            ) as stderr:
                auditor = subprocess.run(
                    [
                        "/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python",
                        str(repo_root / "scripts/audit_mujoco_forward_after_lift_rollout.py"),
                        "--run-dir",
                        str(run_dir),
                        "--allow-not-triggered",
                        "--allow-terminal-loss",
                    ],
                    cwd=repo_root,
                    stdout=stdout,
                    stderr=stderr,
                    check=False,
                )
            if auditor.returncode != 0:
                raise RuntimeError(
                    f"audit failed (runner={runner.returncode}, auditor={auditor.returncode}); "
                    f"see {audit_stderr}"
                )
            audit = validate_completed_run(
                run_dir,
                expected_lift_threshold_m=float(args.lift_rel_z_delta_m),
                expected_latest_forward_actor_sim_time_s=args.latest_forward_actor_sim_time_s,
            )
            result = {
                "index": row["index"],
                "clip_id": row["clip_id"],
                "reused": False,
                "runner_returncode": runner.returncode,
                "audit": audit,
            }
            with progress_lock:
                completed_counter += 1
                print(
                    f"[batch {completed_counter:03d}/{expected_count:03d}] DONE  {run_name}: "
                    f"{audit['rollout_status']}",
                    flush=True,
                )
            return result
        finally:
            slot_queue.put(slot)

    rows = list(inputs["rows"])
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(record_one, row): row for row in rows}
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - preserve all per-clip failures in manifest.
                failure = {"index": row["index"], "clip_id": row["clip_id"], "error": repr(exc)}
                failures.append(failure)
                with progress_lock:
                    print(f"[full109 FAIL] {row['clip_id']}: {exc}", flush=True)

    results.sort(key=lambda item: int(item["index"]))
    failures.sort(key=lambda item: int(item["index"]))
    status_counts = Counter(str(item["audit"]["rollout_status"]) for item in results)
    trigger_source_counts = Counter(str(item["audit"]["trigger_source"]) for item in results)
    summary = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "all_selected_command_audits_passed": len(results) == expected_count and not failures,
        "all_109_command_audits_passed": expected_count == 109 and len(results) == 109 and not failures,
        "expected_count": expected_count,
        "lift_rel_z_delta_m": float(args.lift_rel_z_delta_m),
        "latest_forward_actor_sim_time_s": args.latest_forward_actor_sim_time_s,
        "deadline_publish_lead_ms": int(args.deadline_publish_lead_ms),
        "completed_count": len(results),
        "failure_count": len(failures),
        "status_counts": dict(status_counts),
        "trigger_source_counts": dict(trigger_source_counts),
        "results": results,
        "failures": failures,
    }
    write_json_atomic(output_root / "batch_rollout_manifest.json", summary)
    print(json.dumps({key: summary[key] for key in summary if key not in {"results", "failures"}}, indent=2))
    if failures or len(results) != expected_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
