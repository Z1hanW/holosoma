#!/usr/bin/env python3
"""Publish the exact ch2ckwzw/model_40000 native batch rollouts as a sealed bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np


EXPECTED_CHECKPOINT = {
    "wandb_path": "zihanw22/carry-any/ch2ckwzw/model_40000.pt",
    "completed_iteration": 39999,
    "next_iteration": 40000,
    "pt_sha256": "14e644323b8e6a7b769dbf641d9f625bee895742b7064368b55afd0710a0d665",
    "onnx_sha256": "90c33ea012226b88d206fe2ed0224f505b6c3ba0d9d1306ab2e2eb49d3ef5cc6",
    "pair_manifest_sha256": "c4139bc04eef3db9a907c4de00e1071a9c911f2220259cd3c5450679ec84cb20",
}
EXPECTED_COUNTS = {"ball": 34, "barrel": 34, "bin": 34, "box": 35}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def category(clip_id: str) -> str:
    matches = [name for name in EXPECTED_COUNTS if f"_{name}_" in clip_id]
    if len(matches) != 1:
        raise ValueError(f"Cannot classify clip: {clip_id}")
    return matches[0]


def require_regular(path: Path) -> None:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Missing regular file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--original-view", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    EVAL_ROOT = args.evaluation_root.resolve()
    ORIGINAL_VIEW = args.original_view.resolve()
    PUBLICATION_ROOT = args.output_root.resolve()
    if not (EVAL_ROOT / "all_shards_completed.marker").is_file():
        raise RuntimeError("Batch rollout is not complete")
    evaluation_contract_path = EVAL_ROOT / "evaluation_contract.json"
    object_map_path = ORIGINAL_VIEW / "_clip_object_urdf_map.json"
    original_manifest_path = ORIGINAL_VIEW / "manifest.json"
    pair_path = EVAL_ROOT / "checkpoint/model_40000.pair.json"
    for path in (evaluation_contract_path, object_map_path, original_manifest_path, pair_path):
        require_regular(path)

    evaluation_contract = json.loads(evaluation_contract_path.read_text())
    object_map = json.loads(object_map_path.read_text())
    pair = json.loads(pair_path.read_text())
    clips = object_map["clips"]
    if len(clips) != 137:
        raise RuntimeError("Original object map is not exact137")
    if evaluation_contract["checkpoint"]["pt_sha256"] != EXPECTED_CHECKPOINT["pt_sha256"]:
        raise RuntimeError("Unexpected evaluation checkpoint")
    if pair["pt"]["sha256"] != EXPECTED_CHECKPOINT["pt_sha256"]:
        raise RuntimeError("Checkpoint pair PT mismatch")
    if pair["onnx"]["sha256"] != EXPECTED_CHECKPOINT["onnx_sha256"]:
        raise RuntimeError("Checkpoint pair ONNX mismatch")
    summaries = sorted(EVAL_ROOT.glob("output/shard_*/summary.json"))
    if len(summaries) != 8:
        raise RuntimeError("Expected all eight completed rollout shard summaries")
    for summary_path in summaries:
        summary = json.loads(summary_path.read_text())
        if summary.get("source_checkpoint_sha256") != EXPECTED_CHECKPOINT["pt_sha256"]:
            raise RuntimeError(f"Rollout checkpoint identity mismatch: {summary_path}")

    metadata_paths = sorted(EVAL_ROOT.glob("output/shard_*/clips/*/metadata.json"))
    if len(metadata_paths) != 137:
        raise RuntimeError(f"Expected 137 metadata files, found {len(metadata_paths)}")

    source_records: list[dict[str, object]] = []
    source_paths: dict[str, Path] = {}
    seen: set[str] = set()
    for metadata_path in metadata_paths:
        metadata = json.loads(metadata_path.read_text())
        clip_id = str(metadata["clip_id"])
        if clip_id in seen or clip_id not in clips:
            raise RuntimeError(f"Duplicate or unknown clip: {clip_id}")
        seen.add(clip_id)
        rollout_reference_path = metadata_path.parent / str(
            metadata["teacher_rollout_reference_path"]
        )
        motion_path = metadata_path.parents[2] / "motion_bank" / f"{clip_id}.npz"
        require_regular(rollout_reference_path)
        require_regular(motion_path)
        with np.load(motion_path, allow_pickle=False) as data:
            if int(np.asarray(data["fps"]).reshape(-1)[0]) != 50:
                raise RuntimeError(f"Unexpected FPS: {clip_id}")
            if data["joint_pos"].shape != (359, 36):
                raise RuntimeError(f"Unexpected joint_pos shape: {clip_id}")
            if data["joint_vel"].shape != (359, 35):
                raise RuntimeError(f"Unexpected joint_vel shape: {clip_id}")
            if data["body_pos_w"].shape != (359, 32, 3):
                raise RuntimeError(f"Unexpected body_pos_w shape: {clip_id}")
            if str(np.asarray(data["object_name"]).item()) != clip_id:
                raise RuntimeError(f"Unexpected object identity: {clip_id}")
            for key in data.files:
                array = data[key]
                if array.dtype.kind in "fc" and not np.isfinite(array).all():
                    raise RuntimeError(f"Non-finite rollout data: {clip_id}/{key}")

        core_success = bool(
            metadata["motion_end_reached"]
            and metadata["stable_contact_success"]
            and metadata["final_position_success"]
            and int(metadata["teacher_rollout_valid_step_count"]) == 359
        )
        strict_success = bool(metadata["success"])
        if strict_success and not core_success:
            raise RuntimeError(f"Strict success without core success: {clip_id}")
        source_paths[clip_id] = motion_path
        source_records.append(
            {
                "clip_id": clip_id,
                "category": category(clip_id),
                "source_motion": {
                    "path": str(motion_path),
                    "size": motion_path.stat().st_size,
                    "sha256": sha256(motion_path),
                },
                "source_rollout_reference": {
                    "path": str(rollout_reference_path),
                    "size": rollout_reference_path.stat().st_size,
                    "sha256": sha256(rollout_reference_path),
                },
                "core_success": core_success,
                "strict_success": strict_success,
                "status": str(metadata["status"]),
                "final_object_position_error_m": float(
                    metadata["final_object_position_error_m"]
                ),
            }
        )

    if seen != set(clips):
        raise RuntimeError("Rollout outputs do not cover the original exact137 bank")
    category_counts = {
        name: sum(row["category"] == name for row in source_records)
        for name in EXPECTED_COUNTS
    }
    if category_counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Unexpected category counts: {category_counts}")
    core_failures = sorted(
        str(row["clip_id"]) for row in source_records if not row["core_success"]
    )
    strict_failures = sorted(
        str(row["clip_id"]) for row in source_records if not row["strict_success"]
    )
    source_identity = {
        "version": 1,
        "semantics": "checkpoint_actor_simulator_rollout_motion_bank",
        "policy_role": "checkpoint_native_privileged_teacher_actor",
        "checkpoint": EXPECTED_CHECKPOINT,
        "source_git": evaluation_contract["source"],
        "source_reference_view_digest": evaluation_contract["motion_bank"]["view_digest"],
        "source_object_map_sha256": sha256(object_map_path),
        "source_reference_manifest_sha256": sha256(original_manifest_path),
        "source_evaluation_contract_sha256": sha256(evaluation_contract_path),
        "clip_count": 137,
        "category_counts": category_counts,
        "core_success_count": 137 - len(core_failures),
        "strict_success_count": 137 - len(strict_failures),
        "core_failures_retained": core_failures,
        "strict_failures_retained": strict_failures,
        "timeline": {"fps": 50, "frames_per_clip": 359, "duration_seconds": 7.18},
        "records": sorted(source_records, key=lambda row: str(row["clip_id"])),
    }
    digest = canonical_sha(source_identity)
    target = PUBLICATION_ROOT / digest
    manifest_path = target / "manifest.json"
    if target.exists():
        require_regular(manifest_path)
        existing = json.loads(manifest_path.read_text())
        if existing.get("source_digest") != digest or existing.get("source_identity") != source_identity:
            raise RuntimeError(f"Existing publication differs: {target}")
        print(json.dumps({"reused": True, "target": str(target), "source_digest": digest}, sort_keys=True))
        return

    PUBLICATION_ROOT.mkdir(parents=True, exist_ok=True)
    staging = PUBLICATION_ROOT / f".{digest}.staging-{os.getpid()}"
    if staging.exists():
        raise RuntimeError(f"Staging path already exists: {staging}")
    staging.mkdir()
    published_records: list[dict[str, object]] = []
    try:
        shutil.copyfile(object_map_path, staging / "_clip_object_urdf_map.json")
        shutil.copytree(ORIGINAL_VIEW / "_single_slot_urdfs", staging / "_single_slot_urdfs")
        for clip_id in sorted(source_paths):
            source_path = source_paths[clip_id]
            output_path = staging / f"{clip_id}.npz"
            with np.load(source_path, allow_pickle=False) as data:
                payload = {key: np.array(data[key], copy=True) for key in data.files}
            raw_urdf = Path(str(clips[clip_id]["object_urdf_path"]))
            source_urdf = raw_urdf if raw_urdf.is_absolute() else ORIGINAL_VIEW / raw_urdf
            published_urdf = raw_urdf if raw_urdf.is_absolute() else target / raw_urdf
            require_regular(source_urdf)
            payload["object_urdf_path"] = np.asarray(str(published_urdf))
            np.savez_compressed(output_path, **payload)
            published_records.append(
                {
                    "clip_id": clip_id,
                    "path": output_path.name,
                    "size": output_path.stat().st_size,
                    "sha256": sha256(output_path),
                    "object_urdf_path": str(published_urdf),
                    "object_urdf_sha256": sha256(source_urdf),
                }
            )
        manifest = {
            "version": 1,
            "source_digest": digest,
            "source_identity": source_identity,
            "publication": {
                "clip_count": len(published_records),
                "object_map": {
                    "path": "_clip_object_urdf_map.json",
                    "size": (staging / "_clip_object_urdf_map.json").stat().st_size,
                    "sha256": sha256(staging / "_clip_object_urdf_map.json"),
                },
                "records": published_records,
                "normalization": "only embedded object_urdf_path rewritten to stable publication path",
            },
        }
        manifest_path_staging = staging / "manifest.json"
        manifest_path_staging.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (staging / "COMPLETED").write_text(
            json.dumps(
                {
                    "source_digest": digest,
                    "manifest_sha256": sha256(manifest_path_staging),
                    "clip_count": 137,
                },
                sort_keys=True,
            )
            + "\n"
        )
        for file_path in staging.rglob("*"):
            if file_path.is_file():
                with file_path.open("rb") as stream:
                    os.fsync(stream.fileno())
                file_path.chmod(0o444)
        directories = sorted([p for p in staging.rglob("*") if p.is_dir()], reverse=True)
        for directory in directories + [staging]:
            descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            directory.chmod(0o555)
        staging.rename(target)
        descriptor = os.open(PUBLICATION_ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        json.dumps(
            {
                "reused": False,
                "target": str(target),
                "source_digest": digest,
                "manifest_sha256": sha256(target / "manifest.json"),
                "object_map_sha256": sha256(target / "_clip_object_urdf_map.json"),
                "clip_count": 137,
                "core_success_count": 137 - len(core_failures),
                "strict_success_count": 137 - len(strict_failures),
                "core_failures_retained": core_failures,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
