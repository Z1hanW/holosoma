#!/usr/bin/env python3
"""Run stable-carry mass-bin rollouts for rebuttal datasets with fixed policies."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import xml.etree.ElementTree as ET

import numpy as np


REPO_ROOT = Path("/home/ubuntu/FAR/holosoma")
DATA_ROOT = Path("/data/holosoma_eval_audits/rebuttal_eval")
CHECKPOINT_SRC = REPO_ROOT / "data/checkpoints/swl41n4x/model_39999.pt"
CHECKPOINT_SHA256 = "7f62166a326423704841f87446fd953d6ea7d466a97872cb020f835f36db3081"
PYTHON_BIN = Path("/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python")
BASE_AUDIT = Path("/data/holosoma_eval_audits/swl41n4x_model39999_training80_massbins_stablecarry_20260806_004307")
EXPORTER_SRC = BASE_AUDIT / "runtime_source/export_policy_rollouts.py"
CONTACT_ROOT = BASE_AUDIT / "inputs/contact_windows/full80_motion_time_valid"
RUNTIME_PYTHON = BASE_AUDIT / "runtime_python"

MASS_BINS = (
    ("mass_0p1_1p0", 0.1, 1.0, 3),
    ("mass_1p0_2p0", 1.0, 2.0, 3),
    ("mass_2p0_3p0", 2.0, 3.0, 3),
    ("mass_3p0_4p0", 3.0, 4.0, 3),
    ("mass_4p0_5p0", 4.0, 5.0, 3),
)
MAX_ROLLOUT_STEPS = 1000
SEED17_MAX_NUM_ENVS = 5
ASSIGNMENT_SEED = "rebuttal-seed-mass-sweep-stable-carry-v1-20260807"
SUCCESS_RANDOM_SEED = 42

PICKUP_LIFT_M = 0.10
PICKUP_CONSECUTIVE_STEPS = 10
CARRY_LIFT_M = 0.15
CARRY_ROOT_MIN_Z_M = 0.5
CARRY_MAX_ROOT_OBJECT_XY_M = 0.7
CARRY_ALPHA = 0.91
CARRY_SMOOTHING_STEPS = 5
CARRY_CONSECUTIVE_STEPS = 5
STRICT_CARRY_FRACTION = 0.90


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure(cond: bool, message: str) -> None:
    if not cond:
        raise RuntimeError(message)


def assigned_mass(bin_id: str, low: float, high: float, clip_id: str) -> tuple[float, str, float]:
    key = f"{ASSIGNMENT_SEED}\0{bin_id}\0{clip_id}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    unit = (int(digest[:16], 16) + 0.5) / float(1 << 64)
    mass = low + (high - low) * unit
    ensure(low < mass < high, f"mass assignment out of interval: {bin_id}/{clip_id} => {mass}")
    return mass, digest, unit


def resolve_mesh_path(source_urdf: Path, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = source_urdf.parent / candidate
    return candidate.resolve(strict=True)


def derive_urdf(
    clip_id: str,
    source_urdf: Path,
    target_mass: float,
    destination: Path,
) -> dict[str, Any]:
    tree = ET.parse(source_urdf)
    root = tree.getroot()

    masses = root.findall(".//inertial/mass")
    inertias = root.findall(".//inertial/inertia")
    meshes = root.findall(".//mesh")
    ensure(len(masses) == 1, f"{clip_id}: unexpected inertial mass node count {len(masses)}")
    ensure(len(inertias) == 1, f"{clip_id}: unexpected inertial inertia node count {len(inertias)}")
    ensure(len(meshes) >= 1, f"{clip_id}: missing mesh nodes")

    source_mass = float(masses[0].attrib["value"])
    ensure(math.isclose(source_mass, 0.1, rel_tol=0.0, abs_tol=1e-12), f"{clip_id}: source mass drift {source_mass}")
    ratio = target_mass / source_mass
    masses[0].set("value", format(target_mass, ".17g"))

    source_inertia: dict[str, float] = {}
    target_inertia: dict[str, float] = {}
    for name in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
        value = float(inertias[0].attrib[name])
        scaled = value * ratio
        source_inertia[name] = value
        target_inertia[name] = scaled
        inertias[0].set(name, format(scaled, ".17g"))

    mesh_records: list[dict[str, Any]] = []
    for mesh in meshes:
        source_filename = str(mesh.attrib["filename"])
        resolved = resolve_mesh_path(source_urdf, source_filename)
        mesh.set("filename", str(resolved))
        mesh_records.append(
            {
                "source_filename": source_filename,
                "resolved_path": str(resolved),
                "sha256": sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(destination, encoding="utf-8", xml_declaration=True)

    validation = ET.parse(destination)
    observed_mass = float(validation.find(".//inertial/mass").attrib["value"])
    ensure(math.isclose(observed_mass, target_mass, rel_tol=0.0, abs_tol=1e-14), f"{clip_id}: derived mass mismatch")

    return {
        "source_urdf": str(source_urdf),
        "source_mass_kg": source_mass,
        "target_mass_kg": target_mass,
        "source_inertia_kg_m2": source_inertia,
        "target_inertia_kg_m2": target_inertia,
        "meshes": mesh_records,
        "target_urdf_sha256": sha256_file(destination),
    }


def write_bank(bank_dir: Path, clip_ids: list[str], motion_root: Path, object_entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    bank_dir.mkdir(parents=True, exist_ok=True)
    for clip_id in clip_ids:
        src_motion = motion_root / f"{clip_id}.npz"
        dst_motion = bank_dir / f"{clip_id}.npz"
        if not dst_motion.exists():
            dst_motion.symlink_to(src_motion.resolve())

    map_path = bank_dir / "_clip_object_urdf_map.json"
    write_json(map_path, {"clips": {clip_id: object_entries[clip_id] for clip_id in clip_ids}})
    return {
        "path": str(bank_dir),
        "count": len(clip_ids),
        "clip_ids": clip_ids,
        "object_map": str(map_path),
    }


def ensure_contact_windows(
    contact_root: Path,
    motion_root: Path,
    clip_ids: list[str],
) -> dict[str, Any]:
    """Create synthetic adaptive contact windows for missing active clips.

    The stable-carry evaluation uses adaptive contact windows only as required
    contract inputs for contact-aware observation terms. Some rebuttal datasets
    do not have native per-clip window exports, so we synthesize conservative
    full-clip windows to avoid launch failure while preserving rollout semantics
    (rollout starts are fixed at t=0 in this evaluation setup).
    """

    contact_root = contact_root.resolve(strict=True)
    created = 0
    existing = 0
    details: list[dict[str, Any]] = []

    for clip_id in clip_ids:
        clip_dir = contact_root / clip_id
        if clip_dir.exists():
            existing += 1
            continue
        clip_path = motion_root / f"{clip_id}.npz"
        ensure(clip_path.is_file(), f"missing source motion for window synthesis: {clip_path}")
        try:
            clip = np.load(clip_path, allow_pickle=False)
            if "object_pos_w" in clip:
                step_count = int(np.asarray(clip["object_pos_w"]).shape[0])
            elif "root_pos_w" in clip:
                step_count = int(np.asarray(clip["root_pos_w"]).shape[0])
            else:
                first_key = next(iter(clip.files))
                step_count = int(np.asarray(clip[first_key]).shape[0])
        finally:
            clip.close()

        ensure(step_count > 1, f"{clip_id}: motion clip too short for contact window synthesis ({step_count})")
        clip_dir.mkdir(parents=True, exist_ok=True)
        window = [0, step_count]
        payload = {
            "clip_id": clip_id,
            "contact_intervals": {
                "left_wrist": [int(window[0]), int(window[1])],
                "right_wrist": [int(window[0]), int(window[1])],
                "left_elbow": [int(window[0]), int(window[1])],
                "right_elbow": [int(window[0]), int(window[1])],
            },
        }
        (clip_dir / "contact_intervals.json").write_text(
            json.dumps(payload["contact_intervals"], indent=2),
            encoding="utf-8",
        )
        created += 1
        details.append({"clip_id": clip_id, "contact_clip_dir": str(clip_dir), "step_count": step_count})

    # Keep this lightweight contract for downstream audits/debugging.
    report = {
        "requested_clip_count": len(clip_ids),
        "existing_clip_windows": existing,
        "synthesized_clip_windows": created,
        "missing_clip_windows": len(clip_ids) - existing - created,
        "synthesized": details,
    }
    return report


def copy_with_checks(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    shutil.copy2(source, destination)
    ensure(source.is_file(), f"missing source: {source}")
    ensure(sha256_file(source) == sha256_file(destination), f"copy mismatch: {source} -> {destination}")
    return {"source": str(source), "destination": str(destination), "sha256": sha256_file(destination)}


def prepare_dataset(name: str, source_root: Path, audit_root: Path) -> dict[str, Any]:
    ensure(source_root.exists(), f"missing source root: {source_root}")
    ensure(source_root.joinpath("_clip_object_urdf_map.json").is_file(), f"missing object map: {source_root}")
    ensure(CHECKPOINT_SRC.is_file(), f"missing checkpoint source: {CHECKPOINT_SRC}")
    ensure(EXPORTER_SRC.is_file(), f"missing exporter: {EXPORTER_SRC}")
    ensure(CONTACT_ROOT.is_dir(), f"missing contact windows: {CONTACT_ROOT}")
    ensure(sha256_file(CHECKPOINT_SRC) == CHECKPOINT_SHA256, "checkpoint sha mismatch")

    if audit_root.exists():
        # Keep reruns simple and fail closed.
        raise FileExistsError(f"Audit root already exists: {audit_root}")

    source_map = json.loads((source_root / "_clip_object_urdf_map.json").read_text())
    clip_ids = sorted(source_map["clips"].keys())
    ensure(len(clip_ids) > 0, f"{name}: empty source dataset")

    # Prepare inputs.
    input_root = audit_root / "inputs"
    input_motions = input_root / "motions"
    input_motions.mkdir(parents=True, exist_ok=True)
    (audit_root / "runtime_source").mkdir(parents=True, exist_ok=True)
    (audit_root / "outputs").mkdir(parents=True, exist_ok=True)
    (audit_root / "compact").mkdir(parents=True, exist_ok=True)
    (audit_root / "runtime_cache").mkdir(parents=True, exist_ok=True)
    (audit_root / "logs").mkdir(parents=True, exist_ok=True)

    copy_with_checks(CHECKPOINT_SRC, input_root / "checkpoint/model_39999.pt")
    copy_with_checks(EXPORTER_SRC, audit_root / "runtime_source/export_policy_rollouts.py")
    shutil.copytree(CONTACT_ROOT, input_root / "contact_windows/full80_motion_time_valid", symlinks=False)

    for clip_id in clip_ids:
        shutil.copy2(source_root / f"{clip_id}.npz", input_motions / f"{clip_id}.npz")

    contact_window_report = ensure_contact_windows(
        input_root / "contact_windows/full80_motion_time_valid",
        input_motions,
        clip_ids,
    )

    bank_root = input_root / "banks"
    all_assignments: list[dict[str, Any]] = []
    bin_records: list[dict[str, Any]] = []

    for bin_id, low, high, shard_count in MASS_BINS:
        assignments: list[dict[str, Any]] = []
        object_entries: dict[str, dict[str, Any]] = {}
        urdf_records: list[dict[str, Any]] = []
        effective_shard_count = max(
            shard_count,
            math.ceil(len(clip_ids) / SEED17_MAX_NUM_ENVS),
        )

        for clip_id in clip_ids:
            source_entry = source_map["clips"][clip_id]
            source_urdf_path = source_entry["object_urdf_path"]
            source_urdf = Path(source_urdf_path)
            if not source_urdf.is_absolute():
                source_urdf = source_root / source_urdf_path
            source_urdf = source_urdf.resolve(strict=True)

            target_mass, assignment_hash, assignment_unit = assigned_mass(bin_id, low, high, clip_id)
            derived_urdf = input_root / "urdfs" / bin_id / f"{clip_id}.urdf"
            urdf_record = derive_urdf(clip_id, source_urdf, target_mass, derived_urdf)

            entry = dict(source_entry)
            entry["object_urdf_path"] = str(derived_urdf)
            object_entries[clip_id] = entry

            assignment = {
                "bin_id": bin_id,
                "interval_kg": {"low": low, "high": high},
                "clip_id": clip_id,
                "category": "anything",
                "target_mass_kg": target_mass,
                "assignment_sha256": assignment_hash,
                "assignment_unit_interval": assignment_unit,
                "derived_urdf": str(derived_urdf),
                "derived_urdf_sha256": urdf_record["target_urdf_sha256"],
            }
            assignments.append(assignment)
            all_assignments.append(assignment)
            urdf_records.append(
                {
                    "clip_id": clip_id,
                    **urdf_record,
                }
            )

        bin_dir = bank_root / bin_id
        full_bank = write_bank(bin_dir / "full", clip_ids, input_motions, object_entries)

        shards: list[dict[str, Any]] = []
        # distribute clips across shards as round-robin in a deterministic way.
        shard_clip_ids = [clip_ids[i::effective_shard_count] for i in range(effective_shard_count)]
        for shard_index, shard_ids in enumerate(shard_clip_ids):
            shard_name = f"shard_{shard_index:02d}"
            shard_record = write_bank(
                bin_dir / shard_name,
                shard_ids,
                input_motions,
                object_entries,
            )
            shard_record["shard_index"] = shard_index
            shard_record["path"] = str(shard_record["path"])
            shard_record["count"] = int(shard_record["count"])
            shard_record["clip_ids"] = shard_record["clip_ids"]
            shards.append(shard_record)

        masses = [entry["target_mass_kg"] for entry in assignments]
        bin_records.append(
            {
                "bin_id": bin_id,
                "interval_kg": {"low": low, "high": high, "convention": "[low, high)"},
                "clip_count": len(assignments),
                "mass_min_assigned_kg": min(masses),
                "mass_max_assigned_kg": max(masses),
                "mass_mean_assigned_kg": sum(masses) / len(masses),
                "assignments_digest": canonical_digest(assignments),
                "assignments": assignments,
                "shards": shards,
                "urdf_records": urdf_records,
                "full_bank": full_bank,
            }
        )

    data_manifest = {
        "schema_version": 1,
        "dataset_name": name,
        "source_root": str(source_root),
        "audit_root": str(audit_root),
        "checkpoint": {
            "path": str(input_root / "checkpoint/model_39999.pt"),
            "sha256": sha256_file(input_root / "checkpoint/model_39999.pt"),
        },
        "clip_count": len(clip_ids),
        "motion_count": len(clip_ids),
        "bin_count": len(MASS_BINS),
        "rollout_count": len(MASS_BINS) * len(clip_ids),
        "all_assignments_digest": canonical_digest(all_assignments),
        "all_assignments": all_assignments,
        "bins": bin_records,
        "object_collision_mesh_mode": source_map.get("object_collision_mesh_mode", "single_visual_mesh"),
        "run_config": {
            "seed": ASSIGNMENT_SEED,
            "success_random_seed": SUCCESS_RANDOM_SEED,
        },
    }
    write_json(audit_root / "manifests/data_manifest.json", data_manifest)
    write_json(audit_root / "manifests/contact_windows.json", contact_window_report)
    write_json(audit_root / "manifests/creation.json", {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_root": str(source_root),
        "clip_count": len(clip_ids),
        "bin_count": len(MASS_BINS),
    })
    return data_manifest


@dataclass(frozen=True)
class RolloutTask:
    dataset_name: str
    audit_root: Path
    bin_id: str
    shard_name: str
    bank_dir: Path
    output_dir: Path
    num_envs: int
    task_id: int
    gpu_id: int


def run_rollout(task: RolloutTask) -> dict[str, Any]:
    repo_root = REPO_ROOT
    py = PYTHON_BIN
    exporter = task.audit_root / "runtime_source/export_policy_rollouts.py"
    checkpoint = task.audit_root / "inputs/checkpoint/model_39999.pt"
    object_map = task.bank_dir / "_clip_object_urdf_map.json"
    log_path = task.audit_root / "logs" / f"{task.bin_id}_{task.shard_name}.log"
    cmd = [
        str(py),
        str(exporter),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(task.output_dir),
        "--min-contact-frames",
        "10",
        "--contact-force-threshold",
        "1.0",
        "--contact-voxel-size",
        "0.01",
        "--success-position-threshold",
        "0.10",
        "--no-save-glb",
        "--no-save-preview-png",
        "--no-save-face-heatmap-png",
        "--training.num-envs",
        str(task.num_envs),
        "--training.headless",
        "True",
        "--training.seed",
        str(SUCCESS_RANDOM_SEED),
        "--max-rollout-steps",
        str(MAX_ROLLOUT_STEPS),
        "--simulator.config.sim.max-episode-length-s",
        "1000000",
        "--simulator.config.sim.physx.gpu-collision-stack-size",
        "268435456",
        "--robot.object.enabled",
        "True",
        "--robot.object.object-urdf-path",
        str(object_map),
        "--command.setup-terms.motion-command.params.motion-config.motion-file",
        str(task.bank_dir),
        "--command.setup-terms.motion-command.params.motion-config.adaptive-sampling-contact-interval-root",
        str(task.audit_root / "inputs" / "contact_windows/full80_motion_time_valid"),
        "--command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.uniform-t1-window-sampling-enabled",
        "False",
        "--command.setup-terms.motion-command.params.motion-config.uniform-t1-window-density-boost",
        "1.0",
        "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob",
        "1.0",
        "--command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob-end",
        "1.0",
        "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob",
        "0.0",
        "--command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob-end",
        "0.0",
        "--command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale",
        "0.0",
        "--perception.object-geometry-mode",
        "mesh",
        "reward:g1-29dof-wbt-w-object-generalist-tracking-no-contact",
        "logger:disabled",
        "randomization:disabled",
    ]

    collider_type = os.environ.get("HOLOSOMA_OBJECT_COLLIDER_TYPE", "convex_decomposition")

    cache_root = task.audit_root / "runtime_cache" / f"{task.bin_id}_{task.shard_name}"
    perception_mesh_cache = cache_root / "perception_mesh"
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(task.gpu_id),
            "HOLOSOMA_DEVICE": "cuda:0",
            "HOLOSOMA_ORIGINAL_LOCAL_RANK": str(task.task_id),
            "HOLOSOMA_ORIGINAL_LOCAL_WORLD_SIZE": "1",
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ACCEPT_EULA": "Y",
            "TMPDIR": str(cache_root / "tmp"),
            "XDG_CACHE_HOME": str(cache_root / "xdg_cache"),
            "XDG_CONFIG_HOME": str(cache_root / "xdg_config"),
            "XDG_DATA_HOME": str(cache_root / "xdg_data"),
            "HOLOSOMA_ROBOT_USD_CACHE_DIR": str(cache_root / "robot_usd"),
            "HOLOSOMA_OBJECT_USD_CACHE_DIR": str(cache_root / "object_usd"),
            "HOLOSOMA_OBJECT_COLLIDER_TYPE": collider_type,
            "HOLOSOMA_OBJECT_SPAWN_MODE": "mesh",
            "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS": os.environ.get(
                "HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS",
                "0",
            ),
            "HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK": os.environ.get(
                "HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK",
                "0",
            ),
            "HOLOSOMA_DISABLE_AUTO_RESET": "1",
            "HOLOSOMA_DISABLE_MOTION_END_RESET": "1",
            "HOLOSOMA_DISABLE_CLIP_END_RESET": "1",
            "HOLOSOMA_DISABLE_BAD_TRACKING_RESET": "1",
            "HOLOSOMA_EVAL_DISABLE_ROLLOUT_REFERENCE_REWARDS": "1",
            "HOLOSOMA_EVAL_ALLOW_MISSING_CAMERA_SENSOR_NOISE_STATE": "1",
            "HOLOSOMA_ALLOW_FIXED_BC_EVAL_RESET_ON_RESUME": "1",
            "HOLOSOMA_EVAL_POLICY": "checkpoint_actor",
            "HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE": "mesh",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(RUNTIME_PYTHON),
            "DEPTH_PERCEPTION_PRESET": "checkpoint",
            "HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT": os.environ.get(
                "HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT",
                "1",
            ),
            "HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT": os.environ.get(
                "HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT",
                "1",
            ),
            "HOLOSOMA_FORCE_HETEROGENEOUS_OBJECT_SINGLE_SLOT": os.environ.get(
                "HOLOSOMA_FORCE_HETEROGENEOUS_OBJECT_SINGLE_SLOT",
                "0",
            ),
            "HOLOSOMA_PERCEPTION_MESH_CACHE_DIR": str(perception_mesh_cache),
        }
    )
    for key in (
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "HOLOSOMA_ROBOT_USD_CACHE_DIR",
        "HOLOSOMA_OBJECT_USD_CACHE_DIR",
        "HOLOSOMA_PERCEPTION_MESH_CACHE_DIR",
    ):
        Path(env[key]).mkdir(parents=True, exist_ok=True)

    task.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    start_msg = {
        "cmd": " ".join(map(repr, cmd)),
        "dataset": task.dataset_name,
        "collider_type": collider_type,
    }
    log_path.write_text(json.dumps(start_msg, ensure_ascii=False), encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - start
    summary_path = task.output_dir / "summary.json"
    if proc.returncode != 0:
        raise RuntimeError(
            f"rollout failed for {task.bin_id}/{task.shard_name}: rc={proc.returncode}"
        )
    if not summary_path.is_file():
        raise RuntimeError(f"summary missing for {task.bin_id}/{task.shard_name}")
    return {
        "bin_id": task.bin_id,
        "shard_name": task.shard_name,
        "task_id": task.task_id,
        "gpu_id": task.gpu_id,
        "num_envs": task.num_envs,
        "elapsed_sec": elapsed,
        "summary": json.loads(summary_path.read_text()),
    }


def first_sustained(mask: np.ndarray, count: int, start: int = 0) -> int | None:
    values = np.asarray(mask, dtype=bool)
    run = 0
    for index in range(max(start, 0), values.size):
        run = run + 1 if values[index] else 0
        if run >= count:
            return index - count + 1
    return None


def peak_height_carry_window(
    object_height: np.ndarray,
    alpha: float = CARRY_ALPHA,
    smoothing_steps: int = CARRY_SMOOTHING_STEPS,
    consecutive_steps: int = CARRY_CONSECUTIVE_STEPS,
) -> tuple[int, int]:
    height = np.asarray(object_height, dtype=np.float64).reshape(-1)
    ensure(height.size > 0 and np.all(np.isfinite(height)), "invalid object height trajectory")
    window = max(int(smoothing_steps), 1)
    left = window // 2
    right = window - 1 - left
    padded = np.pad(height, (left, right), mode="edge")
    smooth = np.convolve(padded, np.ones(window, dtype=np.float64) / float(window), mode="valid")
    threshold = float(smooth.min() + np.clip(alpha, 0.0, 1.0) * (smooth.max() - smooth.min()))
    high = smooth >= threshold
    carry_start = first_sustained(high, consecutive_steps)
    if carry_start is None:
        indices = np.flatnonzero(high)
        carry_start = int(indices[0]) if indices.size else int(np.argmax(smooth))
    peak_step = int(np.argmax(smooth))
    carry_end = first_sustained(~high, consecutive_steps, min(peak_step + 1, height.size))
    if carry_end is None:
        carry_end = int(height.size)
    carry_start = min(max(int(carry_start), 0), int(height.size))
    carry_end = min(max(int(carry_end), carry_start), int(height.size))
    return carry_start, carry_end


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def boolean_metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(bool(row[key]) for row in rows)
    total = len(rows)
    return {
        "count": successes,
        "total": total,
        "rate": successes / total if total else None,
        "wilson_95": wilson_interval(successes, total),
    }


def analyze_dataset(audit_root: Path) -> dict[str, Any]:
    data = json.loads((audit_root / "manifests/data_manifest.json").read_text())
    assignments_by_bin_clip: dict[tuple[str, str], dict[str, Any]] = {}
    for bin_record in data["bins"]:
        for assignment in bin_record["assignments"]:
            assignments_by_bin_clip[(bin_record["bin_id"], assignment["clip_id"])] = assignment

    rows: list[dict[str, Any]] = []
    for bin_record in data["bins"]:
        interval = bin_record["interval_kg"]
        for shard_record in bin_record["shards"]:
            output_dir = audit_root / "outputs" / bin_record["bin_id"] / Path(shard_record["path"]).name
            if not output_dir.is_dir():
                raise RuntimeError(f"missing rollout output dir: {output_dir}")
            for clip_id in shard_record["clip_ids"]:
                clip_dir = output_dir / "clips" / clip_id
                if not clip_dir.is_dir():
                    raise RuntimeError(f"missing clip dir: {clip_id} in {output_dir}")
                metadata = json.loads((clip_dir / "metadata.json").read_text())
                assignment = assignments_by_bin_clip[(bin_record["bin_id"], clip_id)]
                expected_urdf = assignment["derived_urdf"]
                if metadata.get("object_urdf_path") != expected_urdf:
                    raise RuntimeError(
                        f"URDF mismatch: {bin_record['bin_id']}/{clip_id}: "
                        f"{metadata.get('object_urdf_path')} != {expected_urdf}"
                    )

                rollout = np.load(clip_dir / "teacher_rollout_reference.npz", allow_pickle=False)
                valid = np.asarray(rollout["valid_steps"], dtype=bool)
                valid_indices = np.flatnonzero(valid)
                if not np.array_equal(valid_indices, np.arange(valid_indices.size)):
                    raise RuntimeError(f"non-prefix valid_steps: {bin_record['bin_id']}/{clip_id}")
                n = int(valid_indices.size)
                object_pos = np.asarray(rollout["object_pos_local"])[valid].astype(np.float64)
                root_pos = np.asarray(rollout["root_pos_local"])[valid].astype(np.float64)
                actor_command = np.asarray(rollout["actor_obs_raw"])[valid].astype(np.float64)
                if object_pos.shape != (n, 3) or root_pos.shape != (n, 3) or actor_command.shape != (n, 4):
                    raise RuntimeError(f"unexpected array shapes: {bin_record['bin_id']}/{clip_id}")

                motion_path = audit_root / "inputs" / "motions" / f"{clip_id}.npz"
                motion = np.load(motion_path, allow_pickle=False)
                reference_height = np.asarray(motion["object_pos_w"][:, 2], dtype=np.float64)
                source_steps = int(reference_height.size)
                policy_steps = int(metadata["num_steps"])
                if n not in {source_steps - 1, source_steps}:
                    raise RuntimeError(f"rollout/source length mismatch: {bin_record['bin_id']}/{clip_id}: {n}/{source_steps}")
                if policy_steps != n - 1:
                    raise RuntimeError(f"policy/steps mismatch: {bin_record['bin_id']}/{clip_id}")

                finite = bool(
                    np.all(np.isfinite(object_pos))
                    and np.all(np.isfinite(root_pos))
                    and np.all(np.isfinite(actor_command))
                )
                lift = object_pos[:, 2] - float(object_pos[0, 2])
                pickup_step = first_sustained(lift >= PICKUP_LIFT_M, PICKUP_CONSECUTIVE_STEPS)
                pickup = pickup_step is not None
                carry_start, carry_end = peak_height_carry_window(reference_height)
                carry_end_reachable = min(carry_end, n)
                ensure(carry_start < carry_end_reachable, f"empty carry window for {bin_record['bin_id']}/{clip_id}")

                root_object_xy = np.linalg.norm(root_pos[:, :2] - object_pos[:, :2], axis=1)
                physical_carry = (
                    (lift >= CARRY_LIFT_M)
                    & (root_pos[:, 2] > CARRY_ROOT_MIN_Z_M)
                    & (root_object_xy < CARRY_MAX_ROOT_OBJECT_XY_M)
                )
                carry_fraction = float(np.mean(physical_carry[carry_start:carry_end_reachable]))

                motion_end = bool(metadata["motion_end_reached"])
                terminated = bool(metadata["terminated"])
                timeout = bool(metadata["timeout"])
                finite_nonterminated = bool(finite and not terminated and not timeout)
                stable = bool(
                    motion_end
                    and finite_nonterminated
                    and pickup
                    and carry_fraction >= STRICT_CARRY_FRACTION
                )

                failure_reasons = []
                if not motion_end:
                    failure_reasons.append("motion_end_not_reached")
                if terminated:
                    failure_reasons.append("terminated")
                if timeout:
                    failure_reasons.append("timeout")
                if not finite:
                    failure_reasons.append("nonfinite")
                if not pickup:
                    failure_reasons.append("no_sustained_pickup")
                if carry_fraction < STRICT_CARRY_FRACTION:
                    failure_reasons.append("carry_fraction_below_0p90")

                rows.append(
                    {
                        "dataset": data["dataset_name"],
                        "bin_id": bin_record["bin_id"],
                        "interval_low_kg": interval["low"],
                        "interval_high_kg": interval["high"],
                        "shard_name": shard_record["path"].split("/")[-1],
                        "clip_id": clip_id,
                        "category": "anything",
                        "target_mass_kg": assignment["target_mass_kg"],
                        "motion_end_reached": motion_end,
                        "terminated": terminated,
                        "timeout": timeout,
                        "finite_rollout": finite,
                        "finite_nonterminated": finite_nonterminated,
                        "pickup_success": pickup,
                        "pickup_first_step": pickup_step,
                        "carry_fraction": carry_fraction,
                        "max_object_lift_m": float(np.max(lift)),
                        "stable_carry_success": stable,
                        "stable_carry_failure_reasons": failure_reasons,
                        "policy_action_steps": int(policy_steps),
                    }
                )

    if len(rows) != data["rollout_count"]:
        raise RuntimeError(f"row count mismatch: {len(rows)} != {data['rollout_count']}")

    summary_rows: dict[str, Any] = []
    per_bin: list[dict[str, Any]] = []
    for bin_record in data["bins"]:
        bin_id = bin_record["bin_id"]
        bin_rows = [row for row in rows if row["bin_id"] == bin_id]
        stable = boolean_metric(bin_rows, "stable_carry_success")
        pickup = boolean_metric(bin_rows, "pickup_success")
        masses = [row["target_mass_kg"] for row in bin_rows]
        carry_vals = [row["carry_fraction"] for row in bin_rows]
        per_bin.append(
            {
                "bin_id": bin_id,
                "label": f"{bin_record['interval_kg']['low']:.1f}–{bin_record['interval_kg']['high']:.1f}",
                "stable_carry_success": stable,
                "pickup_success": pickup,
                "carry_fraction_mean": float(np.mean(carry_vals)),
                "carry_fraction_median": float(np.median(carry_vals)),
                "mass_min": float(min(masses)),
                "mass_max": float(max(masses)),
                "mass_mean": float(sum(masses) / len(masses)),
                "count": len(bin_rows),
                "failures": sorted(
                    [
                        {
                            "clip_id": row["clip_id"],
                            "target_mass_kg": row["target_mass_kg"],
                            "carry_fraction": row["carry_fraction"],
                            "max_object_lift_m": row["max_object_lift_m"],
                            "reasons": row["stable_carry_failure_reasons"],
                        }
                        for row in bin_rows
                        if not row["stable_carry_success"]
                    ],
                    key=lambda item: item["clip_id"],
                ),
            }
        )

    summary = {
        "schema_version": 1,
        "status": "complete",
        "dataset": data["dataset_name"],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "rollout_count": len(rows),
        "bin_count": len(data["bins"]),
        "clip_count": data["clip_count"],
        "per_bin": per_bin,
        "summary_rows": rows,
        "summary_digest": canonical_digest(rows),
    }
    analysis_dir = audit_root / "analysis"
    write_json(analysis_dir / "summary.json", summary)
    with (analysis_dir / "summary.csv").open("w", encoding="utf-8") as handle:
        handle.write(",".join([
            "dataset","bin_id","label","success_count","success_total","success_rate","pickup_count","pickup_total",
            "mass_min","mass_mean","mass_max","carry_mean","carry_median"
        ]) + "\n")
        for row in per_bin:
            handle.write(
                f"{data['dataset_name']},{row['bin_id']},{row['label']},"
                f"{row['stable_carry_success']['count']},{row['stable_carry_success']['total']},"
                f"{row['stable_carry_success']['rate']},"
                f"{row['pickup_success']['count']},{row['pickup_success']['total']},"
                f"{row['mass_min']},{row['mass_mean']},{row['mass_max']},"
                f"{row['carry_fraction_mean']},{row['carry_fraction_median']}\n"
            )
    return summary


def run_dataset(name: str, source_root: Path, out_root: Path, force: bool = False) -> dict[str, Any]:
    audit_root = out_root
    if audit_root.exists():
        if not force:
            raise FileExistsError(f"Output root exists: {audit_root}")
        shutil.rmtree(audit_root)
    data = prepare_dataset(name, source_root, audit_root)

    shard_tasks: list[RolloutTask] = []
    task_id = 0
    for bin_record in data["bins"]:
        bin_id = bin_record["bin_id"]
        for shard_index, shard in enumerate(bin_record["shards"]):
            shard_envs = min(len(shard["clip_ids"]), SEED17_MAX_NUM_ENVS)
            shard_tasks.append(
                RolloutTask(
                    dataset_name=name,
                    audit_root=audit_root,
                    bin_id=bin_id,
                    shard_name=Path(shard["path"]).name,
                    bank_dir=Path(shard["path"]),
                    output_dir=audit_root / "outputs" / bin_id / Path(shard["path"]).name,
                    num_envs=shard_envs,
                    task_id=task_id,
                    gpu_id=task_id % 8,
                )
            )
            task_id += 1

    # Use single-task execution to avoid per-process GPU OOM when IsaacSim
    # is running with 44GiB devices and a small free-memory headroom.
    max_parallel = 1
    print(f"[{name}] scheduling {len(shard_tasks)} rollouts with {max_parallel} parallel tasks")
    futures = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        for task in shard_tasks:
            futures[executor.submit(run_rollout, task)] = task
            while len([f for f in futures if not f.done()]) >= max_parallel:
                time.sleep(5)
        for future in concurrent.futures.as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
                print(
                    f"[{name}] finished {result['bin_id']}/{result['shard_name']} "
                    f"on gpu:{result['gpu_id']} envs={result['num_envs']} time={result['elapsed_sec']:.1f}s"
                )
            except Exception as exc:  # pragma: no cover - surfaced for user
                print(f"[{name}] task failed: {task.bin_id}/{task.shard_name} -> {exc}", file=sys.stderr)
                raise
    # Wait for all futures before analyzing

    summary = analyze_dataset(audit_root)
    return summary


def format_rate(metric: dict[str, Any]) -> str:
    if metric["total"] == 0:
        return "0/0 (NaN)"
    low, high = metric["wilson_95"]
    return f"{metric['count']}/{metric['total']} ({metric['rate']*100:.2f}%, CI {low*100:.1f}–{high*100:.1f}%)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="remove existing output dirs before rerun")
    args = parser.parse_args()

    seed17_source_candidates = [
        DATA_ROOT / "anything_ref_w50_rubber_grid16_12",
        DATA_ROOT / "anything_ref_w50_rubber_grid16_128",
    ]
    for candidate in seed17_source_candidates:
        if not candidate.is_dir() or not (candidate / "_clip_object_urdf_map.json").is_file():
            continue
        if candidate.is_dir():
            seed17_source = candidate
            break
    else:
        raise FileNotFoundError(f"Could not find seed17 source under expected paths: {seed17_source_candidates}")

    datasets = [
        (
            "seed1x",
            DATA_ROOT / "anything_ref_w50_rubber_main8",
            DATA_ROOT / "seed1x_massbins_stablecarry_20260807_1",
        ),
        (
            "seed17x",
            seed17_source,
            DATA_ROOT / "seed17x_massbins_stablecarry_20260807_1",
        ),
    ]

    all_summaries: list[dict[str, Any]] = []
    for name, source, out_root in datasets:
        summary = run_dataset(name, source, out_root, force=args.force)
        all_summaries.append(summary)
        print(f"\n{name} per-bin stable-carry rates:")
        for row in summary["per_bin"]:
            rate = row["stable_carry_success"]
            print(f"  {row['label']}: {format_rate(rate)}")

    print("\nAll datasets complete:")
    for summary in all_summaries:
        print(f"- {summary['dataset']}: total={summary['rollout_count']} rows")


if __name__ == "__main__":
    main()
