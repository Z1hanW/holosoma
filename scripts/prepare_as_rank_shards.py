#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


_MARKER_NAME = ".generated_by_prepare_as_rank_shards"


@dataclass(frozen=True)
class ClipSpec:
    clip_id: str
    npz_path: Path
    entry: dict[str, Any]
    urdf_paths: tuple[Path, ...]
    motion_weight: float
    urdf_weight: float


@dataclass(frozen=True)
class AssignmentUnit:
    unit_id: str
    clip_ids: tuple[str, ...]
    urdf_paths: tuple[str, ...]
    weight: float


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _relpath(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve())


def _load_clip_map(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        return {key: value for key, value in payload.items() if key != "clips"}, payload["clips"]
    if isinstance(payload, dict):
        return {}, payload
    raise ValueError(f"Invalid object map: {path}")


def _motion_weight(path: Path) -> float:
    try:
        size_mb = path.stat().st_size / (1024.0 * 1024.0)
    except OSError:
        size_mb = 0.0
    return 1.0 + size_mb / 128.0


def _mesh_file_size_mb(raw: str, base_dir: Path) -> float:
    if not raw or raw.startswith(("package://", "http://", "https://", "file://")):
        return 0.0
    try:
        path = _resolve_path(raw, base_dir)
        return path.stat().st_size / (1024.0 * 1024.0) if path.is_file() else 0.0
    except OSError:
        return 0.0


def _urdf_weight(path: Path) -> float:
    try:
        root = ET.parse(path).getroot()
    except Exception:
        return 1.0

    collision_count = len(root.findall(".//collision"))
    visual_count = len(root.findall(".//visual"))
    mesh_size_mb = 0.0
    seen_meshes: set[str] = set()
    for mesh in root.findall(".//mesh"):
        filename = str(mesh.get("filename", "")).strip()
        if filename in seen_meshes:
            continue
        seen_meshes.add(filename)
        mesh_size_mb += _mesh_file_size_mb(filename, path.parent)

    return 1.0 + 2.0 * collision_count + 0.5 * visual_count + mesh_size_mb / 8.0


def _normalize_entry(raw_entry: Any) -> dict[str, Any]:
    if isinstance(raw_entry, dict):
        return dict(raw_entry)
    return {"object_urdf_path": str(raw_entry)}


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return [str(value)]


def _entry_urdf_paths(entry: dict[str, Any], *, base_dir: Path) -> tuple[Path, ...]:
    raw_paths: list[str] = []
    raw_paths.extend(_as_string_list(entry.get("object_urdf_path")))
    raw_paths.extend(_as_string_list(entry.get("object_urdf_paths")))
    raw_paths.extend(_as_string_list(entry.get("object_urdfs")))

    resolved: list[Path] = []
    seen: set[Path] = set()
    for raw in raw_paths:
        raw = str(raw).strip()
        if not raw:
            continue
        urdf_path = _resolve_path(raw, base_dir)
        if urdf_path in seen:
            continue
        seen.add(urdf_path)
        resolved.append(urdf_path)
    return tuple(resolved)


def _active_clip_specs(*, motion_dir: Path, object_map: Path) -> tuple[dict[str, Any], list[ClipSpec]]:
    metadata, clips = _load_clip_map(object_map)
    if not clips:
        raise ValueError(f"Object map has no clips: {object_map}")

    npz_by_clip = {path.stem: path for path in sorted(motion_dir.glob("*.npz"))}
    if not npz_by_clip:
        raise FileNotFoundError(f"No .npz clips found under motion dir: {motion_dir}")

    missing_map_entries = sorted(set(npz_by_clip).difference(clips))
    if missing_map_entries:
        preview = ", ".join(missing_map_entries[:10])
        raise ValueError(
            f"Object map is missing {len(missing_map_entries)} active clip(s) from {motion_dir}: {preview}"
        )

    urdf_weights: dict[Path, float] = {}
    specs: list[ClipSpec] = []
    for clip_id in sorted(npz_by_clip):
        entry = _normalize_entry(clips[clip_id])
        urdf_paths = _entry_urdf_paths(entry, base_dir=object_map.parent)
        if not urdf_paths:
            raise ValueError(f"Clip '{clip_id}' has no object URDF dependency")
        for urdf_path in urdf_paths:
            if not urdf_path.is_file():
                raise FileNotFoundError(f"Clip '{clip_id}' object URDF is missing: {urdf_path}")
            if urdf_path not in urdf_weights:
                urdf_weights[urdf_path] = _urdf_weight(urdf_path)
        specs.append(
            ClipSpec(
                clip_id=clip_id,
                npz_path=npz_by_clip[clip_id],
                entry=entry,
                urdf_paths=urdf_paths,
                motion_weight=_motion_weight(npz_by_clip[clip_id]),
                urdf_weight=sum(urdf_weights[urdf_path] for urdf_path in urdf_paths),
            )
        )
    return metadata, specs


def _assignment_units(specs: list[ClipSpec], world_size: int) -> tuple[str, list[AssignmentUnit]]:
    specs_by_urdf: dict[tuple[str, ...], list[ClipSpec]] = {}
    for spec in specs:
        closure_key = tuple(str(path) for path in spec.urdf_paths)
        specs_by_urdf.setdefault(closure_key, []).append(spec)

    if len(specs_by_urdf) >= world_size:
        units: list[AssignmentUnit] = []
        for urdf_paths, group_specs in sorted(specs_by_urdf.items()):
            first = group_specs[0]
            units.append(
                AssignmentUnit(
                    unit_id="|".join(urdf_paths),
                    clip_ids=tuple(spec.clip_id for spec in sorted(group_specs, key=lambda item: item.clip_id)),
                    urdf_paths=urdf_paths,
                    weight=first.urdf_weight + sum(spec.motion_weight for spec in group_specs),
                )
            )
        return "object_closure", units

    units = [
        AssignmentUnit(
            unit_id=spec.clip_id,
            clip_ids=(spec.clip_id,),
            urdf_paths=tuple(str(path) for path in spec.urdf_paths),
            weight=spec.urdf_weight + spec.motion_weight,
        )
        for spec in specs
    ]
    return "clip", units


def _assign_units(units: list[AssignmentUnit], world_size: int) -> tuple[list[list[AssignmentUnit]], bool]:
    assignments: list[list[AssignmentUnit]] = [[] for _ in range(world_size)]
    rank_weights = [0.0 for _ in range(world_size)]

    for unit in sorted(units, key=lambda item: (-item.weight, item.unit_id)):
        rank = min(range(world_size), key=lambda idx: (rank_weights[idx], len(assignments[idx]), idx))
        assignments[rank].append(unit)
        rank_weights[rank] += unit.weight

    duplicated = False
    empty_ranks = [rank for rank, rank_units in enumerate(assignments) if not rank_units]
    if empty_ranks:
        duplicated = True
        source_units = sorted(units, key=lambda item: (-item.weight, item.unit_id))
        for offset, rank in enumerate(empty_ranks):
            unit = source_units[offset % len(source_units)]
            assignments[rank].append(unit)
            rank_weights[rank] += unit.weight

    return assignments, duplicated


def _clean_output_root(output_root: Path) -> None:
    marker = output_root / _MARKER_NAME
    if output_root.exists():
        if not marker.exists():
            raise ValueError(
                f"Refusing to clean non-generated AS rank shard root: {output_root}. "
                f"Remove it manually or choose a different --output-root."
            )
        for child in output_root.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        output_root.mkdir(parents=True)
    marker.write_text("generated by prepare_as_rank_shards.py\n", encoding="utf-8")


def _rewrite_entry_paths(entry: dict[str, Any], *, source_base: Path, target_base: Path) -> dict[str, Any]:
    rewritten = dict(entry)
    for key in ("object_urdf_path", "object_mesh_path"):
        raw = str(rewritten.get(key, "")).strip()
        if not raw:
            continue
        if raw.startswith(("package://", "http://", "https://", "file://")):
            continue
        rewritten[key] = _relpath(_resolve_path(raw, source_base), target_base)
    for key in ("object_urdf_paths", "object_urdfs", "object_mesh_paths"):
        raw_values = _as_string_list(rewritten.get(key))
        if not raw_values:
            continue
        rewritten[key] = [
            raw
            if raw.startswith(("package://", "http://", "https://", "file://"))
            else _relpath(_resolve_path(raw, source_base), target_base)
            for raw in (str(item).strip() for item in raw_values)
            if raw
        ]
    return rewritten


def _write_rank_shard(
    *,
    rank: int,
    rank_dir: Path,
    rank_units: list[AssignmentUnit],
    specs_by_clip: dict[str, ClipSpec],
    metadata: dict[str, Any],
    object_map: Path,
    world_size: int,
    strategy: str,
) -> dict[str, Any]:
    rank_dir.mkdir(parents=True)

    clip_ids: list[str] = []
    for unit in rank_units:
        clip_ids.extend(unit.clip_ids)
    clip_ids = sorted(dict.fromkeys(clip_ids))

    for clip_id in clip_ids:
        src = specs_by_clip[clip_id].npz_path.resolve()
        dst = rank_dir / f"{clip_id}.npz"
        dst.symlink_to(_relpath(src, rank_dir))

    clips_payload = {
        clip_id: _rewrite_entry_paths(specs_by_clip[clip_id].entry, source_base=object_map.parent, target_base=rank_dir)
        for clip_id in clip_ids
    }
    payload = dict(metadata)
    payload["clips"] = clips_payload
    payload["rank_local_shard"] = {
        "rank": rank,
        "world_size": world_size,
        "strategy": strategy,
        "source_object_map": _relpath(object_map, rank_dir),
        "source_motion_dir": _relpath(specs_by_clip[clip_ids[0]].npz_path.parent, rank_dir) if clip_ids else "",
    }
    (rank_dir / "_clip_object_urdf_map.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (rank_dir / "clip_ids.txt").write_text("\n".join(clip_ids) + ("\n" if clip_ids else ""), encoding="utf-8")

    unique_urdfs = sorted({str(path) for clip_id in clip_ids for path in specs_by_clip[clip_id].urdf_paths})
    weight = sum(unit.weight for unit in rank_units)
    return {
        "rank": rank,
        "dir": str(rank_dir),
        "clip_count": len(clip_ids),
        "unique_urdf_count": len(unique_urdfs),
        "weight": weight,
    }


def prepare_rank_shards(*, motion_dir: Path, object_map: Path, output_root: Path, world_size: int) -> dict[str, Any]:
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")

    metadata, specs = _active_clip_specs(motion_dir=motion_dir, object_map=object_map)
    strategy, units = _assignment_units(specs, world_size)
    assignments, duplicated = _assign_units(units, world_size)

    _clean_output_root(output_root)

    specs_by_clip = {spec.clip_id: spec for spec in specs}
    clip_cover_counts = {spec.clip_id: 0 for spec in specs}
    shards = []
    for rank, rank_units in enumerate(assignments):
        rank_dir = output_root / f"rank_{rank}"
        shard = _write_rank_shard(
            rank=rank,
            rank_dir=rank_dir,
            rank_units=rank_units,
            specs_by_clip=specs_by_clip,
            metadata=metadata,
            object_map=object_map,
            world_size=world_size,
            strategy=strategy,
        )
        shards.append(shard)
        for unit in rank_units:
            for clip_id in unit.clip_ids:
                clip_cover_counts[clip_id] += 1

    unique_urdfs = {str(path) for spec in specs for path in spec.urdf_paths}
    exact_partition = all(count == 1 for count in clip_cover_counts.values())
    manifest = {
        "motion_dir": str(motion_dir),
        "object_map": str(object_map),
        "output_root": str(output_root),
        "world_size": world_size,
        "strategy": strategy,
        "clip_count": len(specs),
        "unique_urdf_count": len(unique_urdfs),
        "exact_clip_partition": exact_partition,
        "duplicated_to_fill_empty_ranks": duplicated,
        "shards": shards,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-dir", required=True, type=Path)
    parser.add_argument("--object-map", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--world-size", required=True, type=int)
    args = parser.parse_args()

    motion_dir = args.motion_dir.expanduser().resolve()
    object_map = args.object_map.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    try:
        manifest = prepare_rank_shards(
            motion_dir=motion_dir,
            object_map=object_map,
            output_root=output_root,
            world_size=args.world_size,
        )
    except Exception as exc:
        print(f"[ERROR] Failed to prepare AS rank-local shards: {exc}", file=sys.stderr)
        return 2

    print(f"[INFO] Prepared AS rank-local motion shards: {output_root}", file=sys.stderr)
    print(
        "[INFO]   "
        f"world_size={manifest['world_size']} strategy={manifest['strategy']} "
        f"clips={manifest['clip_count']} unique_urdfs={manifest['unique_urdf_count']} "
        f"exact_clip_partition={manifest['exact_clip_partition']}",
        file=sys.stderr,
    )
    if manifest["duplicated_to_fill_empty_ranks"]:
        print(
            "[WARN]   world_size exceeds available assignment units; some clips were duplicated to avoid empty ranks.",
            file=sys.stderr,
        )
    print(str(output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
