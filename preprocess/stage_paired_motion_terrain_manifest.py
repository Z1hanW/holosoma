#!/usr/bin/env python3
"""Stage paired motion-terrain assets into ds_crisp_data-style directories."""

from __future__ import annotations

import argparse
import fnmatch
import glob
import json
import os
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_MOTION_SUFFIXES = {".npz"}
SUPPORTED_TERRAIN_SUFFIXES = {".obj"}
DEFAULT_MOTION_SUBDIR = "___crisp_clean_motion_gmr_g1"
LEGACY_MOTION_SUBDIR = "___crisp_clean_motion"
DEFAULT_GEOMETRY_SUBDIR = "___crisp_clean_geometry"


def _load_manifest(path: Path) -> tuple[Path | None, list[dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() in {".yaml", ".yml"}:
            payload = yaml.safe_load(handle)
        else:
            payload = json.load(handle)

    if isinstance(payload, list):
        return None, payload
    if isinstance(payload, dict):
        base_dir_raw = payload.get("base_dir")
        entries = payload.get("clips")
        if entries is None:
            entries = payload.get("pairs")
        if entries is None and payload.get("ds_crisp_data_root"):
            entries = [payload]
        if entries is None:
            raise ValueError("Manifest dict must contain a 'clips'/'pairs' list or 'ds_crisp_data_root'.")
        if not isinstance(entries, list):
            raise ValueError("Manifest 'clips'/'pairs' must be a list.")
        base_dir = Path(str(base_dir_raw)) if base_dir_raw else None
        return base_dir, entries
    raise ValueError("Manifest must be either a list or a dict with a 'clips'/'pairs' list.")


def _resolve_path(raw_path: str, *, base_dir: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _resolve_one_path(raw_path: str, *, base_dir: Path, description: str) -> Path:
    resolved = _resolve_path(raw_path, base_dir=base_dir)
    if any(char in raw_path for char in "*?[]"):
        if Path(raw_path).is_absolute():
            matches = sorted(Path(path).resolve() for path in glob.glob(raw_path))
        else:
            matches = sorted(base_dir.glob(raw_path))
        if len(matches) != 1:
            raise ValueError(f"{description} pattern '{raw_path}' must resolve to exactly one file; got {len(matches)}.")
        return matches[0].resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def _iter_ds_crisp_data_entries(entry: dict[str, Any], *, manifest_base_dir: Path) -> list[tuple[str, Path, Path, Path | None]]:
    entry_base_dir = manifest_base_dir
    if entry.get("base_dir"):
        entry_base_dir = _resolve_path(str(entry["base_dir"]), base_dir=manifest_base_dir)

    ds_root = _resolve_path(str(entry["ds_crisp_data_root"]), base_dir=entry_base_dir)
    motion_dir_name_raw = entry.get("motion_dir_name")
    if motion_dir_name_raw is None or str(motion_dir_name_raw).strip() == "":
        motion_dir = ds_root / DEFAULT_MOTION_SUBDIR
        if not motion_dir.is_dir():
            legacy_motion_dir = ds_root / LEGACY_MOTION_SUBDIR
            if legacy_motion_dir.is_dir():
                motion_dir = legacy_motion_dir
    else:
        motion_dir = ds_root / str(motion_dir_name_raw)
    terrain_dir = ds_root / str(entry.get("geometry_dir_name", DEFAULT_GEOMETRY_SUBDIR))
    if not motion_dir.is_dir():
        raise FileNotFoundError(
            f"ds_crisp_data motion dir not found: {motion_dir} "
            f"(also checked legacy default '{LEGACY_MOTION_SUBDIR}')"
        )
    if not terrain_dir.is_dir():
        raise FileNotFoundError(f"ds_crisp_data geometry dir not found: {terrain_dir}")

    # Preserve clip_id from the visible filename in the dataset directory.
    # Some datasets (e.g. gmr_g1) stage symlinks like stair_88.npz -> *_unitree_g1_qpos.npz;
    # resolving before taking .stem would break motion/terrain name matching.
    motion_paths = sorted(motion_dir.glob("*.npz"))
    terrain_paths = sorted(terrain_dir.glob("*.obj"))
    motion_map = {path.stem: path.resolve() for path in motion_paths}
    terrain_map = {path.stem: path.resolve() for path in terrain_paths}

    include_patterns_raw = entry.get("include")
    if include_patterns_raw is None:
        include_patterns_raw = entry.get("clip_glob")
    exclude_patterns_raw = entry.get("exclude", [])
    clip_ids_raw = entry.get("clip_ids")

    include_patterns: list[str]
    if include_patterns_raw is None:
        include_patterns = []
    elif isinstance(include_patterns_raw, str):
        include_patterns = [include_patterns_raw]
    else:
        include_patterns = [str(pattern) for pattern in include_patterns_raw]

    if isinstance(exclude_patterns_raw, str):
        exclude_patterns = [exclude_patterns_raw]
    else:
        exclude_patterns = [str(pattern) for pattern in exclude_patterns_raw]

    clip_filter_active = clip_ids_raw is not None or bool(include_patterns) or bool(exclude_patterns)

    if clip_ids_raw is not None:
        if isinstance(clip_ids_raw, str):
            candidate_clip_ids = [clip_ids_raw]
        else:
            candidate_clip_ids = [str(clip_id) for clip_id in clip_ids_raw]
    else:
        candidate_clip_ids = sorted(set(motion_map) | set(terrain_map))

    if include_patterns:
        candidate_clip_ids = [
            clip_id
            for clip_id in candidate_clip_ids
            if any(fnmatch.fnmatch(clip_id, pattern) for pattern in include_patterns)
        ]
    if exclude_patterns:
        candidate_clip_ids = [
            clip_id
            for clip_id in candidate_clip_ids
            if not any(fnmatch.fnmatch(clip_id, pattern) for pattern in exclude_patterns)
        ]

    missing_motion = sorted(clip_id for clip_id in candidate_clip_ids if clip_id not in motion_map)
    missing_terrain = sorted(clip_id for clip_id in candidate_clip_ids if clip_id not in terrain_map)
    if missing_motion:
        raise ValueError(f"Missing motions for clip ids: {missing_motion[:10]}")
    if missing_terrain:
        raise ValueError(f"Missing terrains for clip ids: {missing_terrain[:10]}")
    if not clip_filter_active:
        extra_motion = sorted(set(motion_map) - set(candidate_clip_ids))
        extra_terrain = sorted(set(terrain_map) - set(candidate_clip_ids))
        if extra_motion or extra_terrain:
            raise ValueError(
                "ds_crisp_data_root contains unmatched clip ids after filtering. "
                f"Extra motions: {extra_motion[:10]} Extra terrains: {extra_terrain[:10]}"
            )

    if not candidate_clip_ids:
        raise ValueError(f"No clip ids selected from ds_crisp_data_root: {ds_root}")

    resolved_entries: list[tuple[str, Path, Path, Path | None]] = []
    for clip_id in candidate_clip_ids:
        terrain_path = terrain_map[clip_id]
        support_path = terrain_path.with_name(f"{terrain_path.stem}.support.npz")
        resolved_entries.append(
            (
                clip_id,
                motion_map[clip_id],
                terrain_path,
                support_path if support_path.is_file() else None,
            )
        )
    return resolved_entries


def _resolve_entry(
    entry: dict[str, Any],
    *,
    manifest_base_dir: Path,
) -> list[tuple[str, Path, Path, Path | None]]:
    if not isinstance(entry, dict):
        raise TypeError(f"Manifest entries must be objects; got {type(entry)!r}.")
    if entry.get("ds_crisp_data_root"):
        return _iter_ds_crisp_data_entries(entry, manifest_base_dir=manifest_base_dir)

    entry_base_dir = manifest_base_dir
    if entry.get("base_dir"):
        entry_base_dir = _resolve_path(str(entry["base_dir"]), base_dir=manifest_base_dir)

    folder_path_raw = entry.get("folder_path")
    folder_path = None
    if folder_path_raw:
        folder_path = _resolve_path(str(folder_path_raw), base_dir=entry_base_dir)
        entry_base_dir = folder_path

    motion_path_raw = entry.get("motion_path")
    terrain_path_raw = entry.get("terrain_obj_path", entry.get("terrain_path"))
    terrain_support_path_raw = entry.get("terrain_support_path")

    if motion_path_raw is None and folder_path is not None:
        motion_pattern = entry.get("motion_pattern", entry.get("human_video_data_pattern", "*.npz"))
        motion_path = _resolve_one_path(str(motion_pattern), base_dir=folder_path, description="motion pattern")
    elif motion_path_raw is not None:
        motion_path = _resolve_one_path(str(motion_path_raw), base_dir=entry_base_dir, description="motion path")
    else:
        raise ValueError("Each manifest entry must provide 'motion_path' or 'folder_path' + 'motion_pattern'.")

    if terrain_path_raw is None and folder_path is not None:
        terrain_pattern = entry.get("terrain_pattern", "*.obj")
        terrain_pattern = entry.get("human_video_terrain_pattern", terrain_pattern)
        terrain_path = _resolve_one_path(str(terrain_pattern), base_dir=folder_path, description="terrain pattern")
    elif terrain_path_raw is not None:
        terrain_path = _resolve_one_path(str(terrain_path_raw), base_dir=entry_base_dir, description="terrain path")
    else:
        raise ValueError(
            "Each manifest entry must provide 'terrain_obj_path'/'terrain_path' or 'folder_path' + 'terrain_pattern'."
        )

    clip_id_raw = entry.get("clip_id")
    if clip_id_raw is not None:
        clip_id = str(clip_id_raw).strip()
    else:
        clip_id = motion_path.stem
    if not clip_id:
        raise ValueError(f"Failed to derive clip_id for entry: {entry!r}")

    if motion_path.suffix.lower() not in SUPPORTED_MOTION_SUFFIXES:
        raise ValueError(
            f"Motion path '{motion_path}' has unsupported suffix '{motion_path.suffix}'. "
            f"Manifest staging currently supports per-clip {sorted(SUPPORTED_MOTION_SUFFIXES)} motions."
        )
    if terrain_path.suffix.lower() not in SUPPORTED_TERRAIN_SUFFIXES:
        raise ValueError(
            f"Terrain path '{terrain_path}' has unsupported suffix '{terrain_path.suffix}'. "
            f"Expected one of {sorted(SUPPORTED_TERRAIN_SUFFIXES)}."
        )
    terrain_support_path: Path | None = None
    if terrain_support_path_raw is not None:
        terrain_support_path = _resolve_one_path(
            str(terrain_support_path_raw),
            base_dir=entry_base_dir,
            description="terrain support path",
        )
    else:
        candidate_support_path = terrain_path.with_name(f"{terrain_path.stem}.support.npz")
        if candidate_support_path.is_file():
            terrain_support_path = candidate_support_path

    return [(clip_id, motion_path, terrain_path, terrain_support_path)]


def _decode_strings(values: Any) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _validate_motion_terrain_alignment(clip_id: str, motion_path: Path, terrain_path: Path) -> None:
    """Catch catastrophic motion/terrain mismatches before launching training.

    Some raw motion exports contain world-space offsets that are not aligned to the
    paired terrain mesh origin. Those cases can silently pass name-based preflight
    and only show up later as a robot rendered far away from its terrain tile.
    """
    if motion_path.suffix.lower() != ".npz" or terrain_path.suffix.lower() != ".obj":
        return

    try:
        import numpy as np
        import trimesh
    except ImportError:
        return

    with np.load(motion_path, allow_pickle=True) as payload:
        body_pos_w = payload.get("body_pos_w")
        body_names = payload.get("body_names")
        if body_pos_w is None or body_names is None:
            return

        body_pos_w = np.asarray(body_pos_w, dtype=np.float32)
        if body_pos_w.ndim != 3 or body_pos_w.shape[0] == 0 or body_pos_w.shape[2] < 2:
            return

        names = _decode_strings(np.asarray(body_names).reshape(-1))
        ref_name_candidates = ("pelvis", "pelvis_link", "base_link", "torso_link")
        ref_idx = 0
        for candidate in ref_name_candidates:
            if candidate in names:
                ref_idx = names.index(candidate)
                break
        ref_xy = body_pos_w[:, ref_idx, :2]

    mesh = trimesh.load(terrain_path, force="mesh")
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    if bounds.shape != (2, 3):
        return

    lower_xy = bounds[0, :2]
    upper_xy = bounds[1, :2]
    inside_xy = np.logical_and(ref_xy >= lower_xy, ref_xy <= upper_xy).all(axis=1)
    inside_fraction = float(np.mean(inside_xy))

    start_xy = ref_xy[0]
    clipped_start_xy = np.minimum(np.maximum(start_xy, lower_xy), upper_xy)
    start_distance = float(np.linalg.norm(start_xy - clipped_start_xy))

    if inside_fraction < 0.25 and start_distance > 2.0:
        raise ValueError(
            "Motion/terrain alignment check failed for clip "
            f"'{clip_id}': reference path is far outside the paired OBJ bounds. "
            f"start_xy={start_xy.tolist()} bounds_xy={[lower_xy.tolist(), upper_xy.tolist()]} "
            f"inside_fraction={inside_fraction:.3f} start_distance={start_distance:.3f}. "
            "This usually means the manifest is pointing at an unnormalized/raw motion export."
        )


def _stage_symlink(source: Path, dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(source, dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--manifest", help="Path to YAML/JSON manifest describing motion-terrain pairs.")
    source_group.add_argument(
        "--ds-crisp-data-root",
        help=(
            "Path to a ds_crisp_data root containing ___crisp_clean_motion_gmr_g1/ "
            "(or legacy ___crisp_clean_motion/) and ___crisp_clean_geometry/."
        ),
    )
    parser.add_argument(
        "--out-root",
        required=True,
        help="Output directory containing staged ___crisp_clean_motion_gmr_g1/ and ___crisp_clean_geometry/.",
    )
    args = parser.parse_args()

    manifest_path: Path | None = None
    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    out_root = Path(args.out_root).expanduser().resolve()
    motion_out_dir = out_root / DEFAULT_MOTION_SUBDIR
    terrain_out_dir = out_root / DEFAULT_GEOMETRY_SUBDIR
    motion_out_dir.mkdir(parents=True, exist_ok=True)
    terrain_out_dir.mkdir(parents=True, exist_ok=True)

    if manifest_path is not None:
        manifest_base_dir_raw, entries = _load_manifest(manifest_path)
        manifest_base_dir = manifest_path.parent.resolve()
        if manifest_base_dir_raw is not None:
            manifest_base_dir = _resolve_path(str(manifest_base_dir_raw), base_dir=manifest_base_dir)
        source_description = str(manifest_path)
    else:
        ds_root = Path(args.ds_crisp_data_root).expanduser().resolve()
        if not ds_root.exists():
            raise FileNotFoundError(f"ds_crisp_data_root not found: {ds_root}")
        entries = [{"ds_crisp_data_root": str(ds_root)}]
        manifest_base_dir = ds_root.parent.resolve()
        source_description = str(ds_root)

    staged_records: list[dict[str, str]] = []
    seen_clip_ids: set[str] = set()

    for idx, entry in enumerate(entries):
        if isinstance(entry, dict) and entry.get("enabled", True) is False:
            continue
        resolved_entries = _resolve_entry(
            entry,
            manifest_base_dir=manifest_base_dir,
        )
        for clip_id, motion_path, terrain_path, terrain_support_path in resolved_entries:
            if clip_id in seen_clip_ids:
                raise ValueError(f"Duplicate clip_id '{clip_id}' at manifest entry {idx}.")
            seen_clip_ids.add(clip_id)

            _validate_motion_terrain_alignment(clip_id, motion_path, terrain_path)

            staged_motion_path = motion_out_dir / f"{clip_id}{motion_path.suffix.lower()}"
            staged_terrain_path = terrain_out_dir / f"{clip_id}.obj"

            _stage_symlink(motion_path, staged_motion_path)
            _stage_symlink(terrain_path, staged_terrain_path)

            staged_record = {
                "clip_id": clip_id,
                "motion_source": str(motion_path),
                "terrain_source": str(terrain_path),
                "staged_motion": str(staged_motion_path),
                "staged_terrain": str(staged_terrain_path),
            }
            if terrain_support_path is not None:
                staged_support_path = terrain_out_dir / f"{clip_id}.support.npz"
                _stage_symlink(terrain_support_path, staged_support_path)
                staged_record["terrain_support_source"] = str(terrain_support_path)
                staged_record["staged_terrain_support"] = str(staged_support_path)

            staged_records.append(staged_record)

    summary_path = out_root / "manifest.summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_path": source_description,
                "manifest_path": str(manifest_path) if manifest_path is not None else None,
                "clip_count": len(staged_records),
                "motion_dir": str(motion_out_dir),
                "geometry_dir": str(terrain_out_dir),
                "clips": staged_records,
            },
            handle,
            indent=2,
        )

    print(f"[INFO] Staged {len(staged_records)} motion-terrain pairs from {source_description}")
    print(f"[INFO] Staged motions: {motion_out_dir}")
    print(f"[INFO] Staged terrains: {terrain_out_dir}")
    print(f"[INFO] Stage summary: {summary_path}")


if __name__ == "__main__":
    main()
