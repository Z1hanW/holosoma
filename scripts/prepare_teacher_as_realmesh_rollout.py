#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.export_teacher_box_contacts import _EXPORT_REGION_LABELS, _save_overlay_assets  # noqa: E402


MARKER_NAME = ".generated_by_realmesh_rollout"
DEFAULT_CONTACT_EXPORT_NAME = "contact_export_from_teacher_realmesh_rollout"


def _load_clip_map(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        return {key: value for key, value in payload.items() if key != "clips"}, payload["clips"]
    if isinstance(payload, dict):
        return {}, payload
    raise ValueError(f"Invalid object map: {path}")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_remove_generated(path: Path, *, force: bool) -> None:
    if not path.exists():
        return
    marker = path / MARKER_NAME
    if not force and not marker.exists():
        raise SystemExit(f"[ERROR] Refusing to overwrite non-generated path: {path}")
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _copy_or_symlink(src: Path, dst: Path, *, symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if symlink:
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    elif src.is_dir():
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy2(src, dst)


def _category_for(clip_id: str, entry: object) -> str:
    parts = [clip_id]
    if isinstance(entry, dict):
        for key in ("object_name", "object_urdf_path", "object_mesh_path", "object_category", "category", "object_type"):
            value = str(entry.get(key, "")).strip()
            if value:
                path = Path(value)
                parts.extend([path.name, path.stem] if key.endswith("_path") else [value])
    raw = " ".join(parts).lower().replace("-", "_")
    if "barrel" in raw:
        return "barrel"
    if "bin" in raw or "trash" in raw or "basket" in raw:
        return "bin"
    if "ball" in raw or "sphere" in raw:
        return "ball"
    if "box" in raw or "cube" in raw or "largebox" in raw:
        return "box"
    return "other"


def _parse_csv_set(raw: str) -> set[str]:
    return {item.strip() for item in raw.split(",") if item.strip()}


def _summary_row_success(row: dict[str, str]) -> bool:
    return str(row.get("success", "")).strip().lower() == "true"


def _infer_clip_id_from_dir_name(dir_name: str) -> str:
    if "_" not in dir_name:
        return dir_name.strip()
    return dir_name.split("_", 1)[1].strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_clip_dir_with_bank_metadata(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, symlinks=False)
    metadata_path = dst / "metadata.json"
    if not metadata_path.is_file():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    clip_id = str(metadata.get("clip_id") or _infer_clip_id_from_dir_name(dst.name))
    metadata["teacher_rollout_motion_bank_path"] = str(Path("..") / ".." / ".." / "_single_slot_motion_bank" / f"{clip_id}.npz")
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_object_frame_visualization(
    source_clip_dir: Path,
    vis_clip_dir: Path,
    *,
    save_preview_png: bool,
    save_face_heatmap_png: bool,
) -> None:
    metadata_path = source_clip_dir / "metadata.json"
    if not metadata_path.is_file():
        return
    vis_clip_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    primitive_points = np.load(source_clip_dir / "primitive_contact_points.npy")
    primitive_counts = np.load(source_clip_dir / "primitive_contact_point_counts.npy")
    region_points_by_label: dict[str, np.ndarray] = {}
    for label in _EXPORT_REGION_LABELS:
        points_path = source_clip_dir / f"{label}_contact_points.npy"
        region_points_by_label[label] = (
            np.load(points_path) if points_path.is_file() else np.zeros((0, 3), dtype=np.float32)
        )
    _save_overlay_assets(
        vis_clip_dir,
        clip_id=str(metadata["clip_id"]),
        object_name=str(metadata.get("object_name", "")),
        object_urdf_path=str(metadata.get("object_urdf_path", "")),
        extents_xyz=np.asarray(metadata["primitive_extents_xyz"], dtype=np.float32),
        retained_points_xyz=primitive_points,
        retained_counts=primitive_counts,
        display_points_xyz=primitive_points,
        display_point_labels=[_EXPORT_REGION_LABELS[0]] * int(primitive_points.shape[0]),
        region_points_by_label=region_points_by_label,
        save_glb=True,
        save_preview_png=save_preview_png,
        save_face_heatmap_png=save_face_heatmap_png,
    )


def prepare_shards(args: argparse.Namespace) -> None:
    source_bank = args.source_bank.expanduser().resolve()
    source_map = args.source_map.expanduser().resolve()
    shard_root = args.shard_root.expanduser().resolve()
    allowed = _parse_csv_set(args.allowed_categories)
    excluded = _parse_csv_set(args.exclude_clips)

    if not source_bank.is_dir():
        raise SystemExit(f"[ERROR] Missing source bank: {source_bank}")
    if not source_map.is_file():
        raise SystemExit(f"[ERROR] Missing source object map: {source_map}")
    npz_files = sorted(source_bank.glob("*.npz"))
    if args.expected_total and len(npz_files) != args.expected_total:
        raise SystemExit(f"[ERROR] Expected {args.expected_total} source npz files, found {len(npz_files)}")

    metadata, clips = _load_clip_map(source_map)
    if args.expected_total and len(clips) != args.expected_total:
        raise SystemExit(f"[ERROR] Expected {args.expected_total} object-map entries, found {len(clips)}")

    selected_ids: list[str] = []
    skipped_by_category: Counter[str] = Counter()
    for npz_path in npz_files:
        clip_id = npz_path.stem
        entry = clips.get(clip_id)
        if entry is None:
            raise SystemExit(f"[ERROR] Missing object-map entry for {clip_id}")
        category = _category_for(clip_id, entry)
        if allowed and category not in allowed:
            skipped_by_category[category] += 1
            continue
        if clip_id in excluded:
            continue
        selected_ids.append(clip_id)

    if not selected_ids:
        raise SystemExit("[ERROR] No clips selected for realmesh rollout.")

    _safe_remove_generated(shard_root, force=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    (shard_root / MARKER_NAME).write_text("generated by prepare_teacher_as_realmesh_rollout.py prepare-shards\n", encoding="utf-8")

    base, rem = divmod(len(selected_ids), args.num_shards)
    shard_counts = [base + (1 if idx < rem else 0) for idx in range(args.num_shards)]
    per_gpu_envs = int(args.per_gpu_envs)
    if per_gpu_envs <= 0:
        per_gpu_envs = max(shard_counts)
    if max(shard_counts) > per_gpu_envs:
        raise SystemExit(
            f"[ERROR] PER_GPU_ENVS={per_gpu_envs} is below largest shard count {max(shard_counts)}. "
            "Increase PER_GPU_ENVS or NUM_SHARDS."
        )

    offset = 0
    manifest: dict[str, Any] = {
        "source_bank": str(source_bank),
        "source_map": str(source_map),
        "allowed_categories": sorted(allowed),
        "excluded_clips": sorted(excluded),
        "source_npz_count": len(npz_files),
        "selected_clip_count": len(selected_ids),
        "selected_category_counts": dict(sorted(Counter(_category_for(cid, clips[cid]) for cid in selected_ids).items())),
        "skipped_by_category": dict(sorted(skipped_by_category.items())),
        "num_shards": args.num_shards,
        "per_gpu_envs": per_gpu_envs,
        "shards": [],
    }
    for shard_idx, count in enumerate(shard_counts):
        shard_ids = selected_ids[offset : offset + count]
        offset += count
        shard_dir = shard_root / f"shard_{shard_idx:02d}"
        shard_dir.mkdir(parents=True)
        objects_dir = source_bank / "objects"
        if objects_dir.exists():
            _copy_or_symlink(objects_dir, shard_dir / "objects", symlink=True)
        for clip_id in shard_ids:
            _copy_or_symlink(source_bank / f"{clip_id}.npz", shard_dir / f"{clip_id}.npz", symlink=True)
        shard_payload = dict(metadata)
        shard_payload["clips"] = {clip_id: clips[clip_id] for clip_id in shard_ids}
        _write_json(shard_dir / "_clip_object_urdf_map.json", shard_payload)
        (shard_dir / "clip_ids.txt").write_text("\n".join(shard_ids) + ("\n" if shard_ids else ""), encoding="utf-8")
        manifest["shards"].append(
            {
                "shard_index": shard_idx,
                "count": count,
                "motion_dir": str(shard_dir),
                "object_map": str(shard_dir / "_clip_object_urdf_map.json"),
                "first_clip": shard_ids[0] if shard_ids else None,
                "last_clip": shard_ids[-1] if shard_ids else None,
            }
        )

    _write_json(shard_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def merge_outputs(args: argparse.Namespace) -> None:
    output_root = args.output_root.expanduser().resolve()
    target_bank = args.target_bank.expanduser().resolve()
    source_bank = args.source_bank.expanduser().resolve()
    contact_export_name = args.contact_export_name.strip() or DEFAULT_CONTACT_EXPORT_NAME
    shards_root = output_root / "shards"
    merged_clips = output_root / "clips"
    merged_motion = output_root / "motion_bank"

    if not shards_root.is_dir():
        raise SystemExit(f"[ERROR] Missing shard outputs root: {shards_root}")
    _safe_remove_generated(merged_clips, force=True)
    _safe_remove_generated(merged_motion, force=True)
    merged_clips.mkdir(parents=True, exist_ok=True)
    merged_motion.mkdir(parents=True, exist_ok=True)
    (merged_clips / MARKER_NAME).write_text("generated success clip merge\n", encoding="utf-8")
    (merged_motion / MARKER_NAME).write_text("generated success motion merge\n", encoding="utf-8")

    all_rows: list[dict[str, str]] = []
    success_rows: list[dict[str, str]] = []
    failure_rows: list[dict[str, str]] = []
    clip_object_map: dict[str, Any] = {}
    success_clip_dirs: dict[str, Path] = {}

    for shard_output in sorted(path for path in shards_root.glob("shard_*") if path.is_dir()):
        summary_csv = shard_output / "summary.csv"
        clips_src = shard_output / "clips"
        motion_src = shard_output / "motion_bank"
        if not summary_csv.is_file() or not clips_src.is_dir() or not motion_src.is_dir():
            raise SystemExit(f"[ERROR] Shard output is incomplete: {shard_output}")

        shard_rows = _read_csv(summary_csv)
        all_rows.extend(shard_rows)
        source_map = motion_src / "_clip_object_urdf_map.json"
        _, shard_map = _load_clip_map(source_map)
        for row in shard_rows:
            clip_id = str(row.get("clip_id", "")).strip()
            if not clip_id:
                continue
            if _summary_row_success(row):
                success_rows.append(row)
                matches = [p for p in clips_src.iterdir() if p.is_dir() and _infer_clip_id_from_dir_name(p.name) == clip_id]
                if len(matches) != 1:
                    raise SystemExit(f"[ERROR] Expected one clip dir for {clip_id}, found {len(matches)} in {clips_src}")
                success_clip_dirs[clip_id] = matches[0]
                clip_object_map[clip_id] = shard_map[clip_id]
                shutil.copy2(motion_src / f"{clip_id}.npz", merged_motion / f"{clip_id}.npz")
                shutil.copytree(matches[0], merged_clips / matches[0].name)
            else:
                failure_rows.append(row)

    _write_csv(output_root / "summary_all.csv", all_rows)
    _write_csv(output_root / "summary.csv", success_rows)
    _write_json(merged_motion / "_clip_object_urdf_map.json", {"clips": dict(sorted(clip_object_map.items()))})
    (output_root / "success_clips.txt").write_text(
        "\n".join(row["clip_id"] for row in success_rows) + ("\n" if success_rows else ""),
        encoding="utf-8",
    )
    (output_root / "failure_clips.txt").write_text(
        "\n".join(row["clip_id"] for row in failure_rows) + ("\n" if failure_rows else ""),
        encoding="utf-8",
    )

    _safe_remove_generated(target_bank, force=args.force)
    target_bank.mkdir(parents=True, exist_ok=True)
    (target_bank / MARKER_NAME).write_text("generated by prepare_teacher_as_realmesh_rollout.py merge\n", encoding="utf-8")
    for path in sorted(merged_motion.glob("*.npz")):
        shutil.copy2(path, target_bank / path.name)
    _write_json(target_bank / "_clip_object_urdf_map.json", {"clips": dict(sorted(clip_object_map.items()))})

    slot_bank = target_bank / "_single_slot_motion_bank"
    slot_bank.mkdir(parents=True, exist_ok=True)
    for path in sorted(merged_motion.glob("*.npz")):
        shutil.copy2(path, slot_bank / path.name)
    _write_json(slot_bank / "_clip_object_urdf_map.json", {"clips": dict(sorted(clip_object_map.items()))})

    source_objects = source_bank / "objects"
    if source_objects.exists():
        _copy_or_symlink(source_objects, target_bank / "objects", symlink=True)

    contact_root = target_bank / contact_export_name / "clips"
    contact_root.mkdir(parents=True, exist_ok=True)
    for clip_id, source_clip_dir in sorted(success_clip_dirs.items()):
        _copy_clip_dir_with_bank_metadata(source_clip_dir, contact_root / source_clip_dir.name)

    if args.save_visualization:
        vis_root = target_bank / "object_frame_contact_vis" / "clips"
        for source_clip_dir in sorted(success_clip_dirs.values()):
            _save_object_frame_visualization(
                source_clip_dir,
                vis_root / source_clip_dir.name,
                save_preview_png=bool(args.save_visualization_preview_png),
                save_face_heatmap_png=bool(args.save_visualization_face_heatmap_png),
            )

    category_counts = Counter(_category_for(clip_id, entry) for clip_id, entry in clip_object_map.items())
    manifest = {
        "source_bank": str(source_bank),
        "output_root": str(output_root),
        "target_bank": str(target_bank),
        "contact_export_name": contact_export_name,
        "total_rollout_rows": len(all_rows),
        "success_count": len(success_rows),
        "failure_count": len(failure_rows),
        "category_counts": dict(sorted(category_counts.items())),
        "save_visualization": bool(args.save_visualization),
        "save_visualization_preview_png": bool(args.save_visualization_preview_png),
        "save_visualization_face_heatmap_png": bool(args.save_visualization_face_heatmap_png),
        "success_status_counts": dict(sorted(Counter(row.get("status", "") for row in success_rows).items())),
        "failure_status_counts": dict(sorted(Counter(row.get("status", "") for row in failure_rows).items())),
    }
    _write_json(output_root / "merge_manifest.json", manifest)
    _write_json(target_bank / "realmesh_rollout_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and merge parallel AS real-mesh teacher rollout exports.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-shards")
    prepare.add_argument("--source-bank", required=True, type=Path)
    prepare.add_argument("--source-map", required=True, type=Path)
    prepare.add_argument("--shard-root", required=True, type=Path)
    prepare.add_argument("--num-shards", required=True, type=int)
    prepare.add_argument("--per-gpu-envs", required=True, type=int)
    prepare.add_argument("--expected-total", type=int, default=0)
    prepare.add_argument("--allowed-categories", default="box,ball,bin,barrel")
    prepare.add_argument("--exclude-clips", default="")
    prepare.set_defaults(func=prepare_shards)

    merge = subparsers.add_parser("merge")
    merge.add_argument("--output-root", required=True, type=Path)
    merge.add_argument("--target-bank", required=True, type=Path)
    merge.add_argument("--source-bank", required=True, type=Path)
    merge.add_argument("--contact-export-name", default=DEFAULT_CONTACT_EXPORT_NAME)
    merge.add_argument("--save-visualization", action="store_true")
    merge.add_argument("--save-visualization-preview-png", action="store_true")
    merge.add_argument("--save-visualization-face-heatmap-png", action="store_true")
    merge.add_argument("--force", action="store_true")
    merge.set_defaults(func=merge_outputs)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
