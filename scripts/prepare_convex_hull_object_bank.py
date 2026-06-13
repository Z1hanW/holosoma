#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


MARKER_NAME = ".generated_by_prepare_convex_hull_object_bank"


def _load_clip_map(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        return {key: value for key, value in payload.items() if key != "clips"}, payload["clips"]
    if isinstance(payload, dict):
        return {}, payload
    raise ValueError(f"Invalid object map: {path}")


def _resolve_path(raw: str, base_dir: Path) -> Path:
    value = str(raw).strip()
    if not value:
        raise ValueError("empty path")
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned or "object"


def _relpath(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve())


def _scalar_str(value: object) -> str:
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
    if hasattr(item, "item"):
        item = item.item()
    return str(item).strip()


def _write_npz_with_urdf(src: Path, dst: Path, object_urdf_path: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with np.load(src, allow_pickle=True) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["object_urdf_path"] = np.asarray(object_urdf_path)
    fd, tmp_raw = tempfile.mkstemp(prefix=f"{dst.stem}.", suffix=".npz")
    os.close(fd)
    tmp = Path(tmp_raw)
    try:
        np.savez_compressed(tmp, **payload)
        shutil.copy2(tmp, dst)
    finally:
        if tmp.exists():
            tmp.unlink()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=True)
    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"scene has no mesh geometry: {path}")
        loaded = trimesh.util.concatenate(meshes)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f"unsupported mesh type {type(loaded).__name__}: {path}")
    if len(loaded.vertices) < 4 or len(loaded.faces) < 4:
        raise ValueError(f"mesh is too degenerate for convex hull: {path}")
    return loaded


def _export_convex_hull(src_mesh: Path, dst_mesh: Path) -> tuple[int, int, int, int]:
    mesh = _load_mesh(src_mesh)
    hull = mesh.convex_hull
    if len(hull.vertices) < 4 or len(hull.faces) < 4:
        raise ValueError(f"convex hull is degenerate for {src_mesh}")
    dst_mesh.parent.mkdir(parents=True, exist_ok=True)
    hull.export(dst_mesh)
    return len(mesh.vertices), len(mesh.faces), len(hull.vertices), len(hull.faces)


def _rewrite_urdf_to_hulls(
    *,
    src_urdf: Path,
    dst_urdf: Path,
    target_bank: Path,
    hull_by_mesh: dict[Path, Path],
    stats_by_mesh: dict[str, dict[str, int | str]],
) -> list[Path]:
    tree = ET.parse(src_urdf)
    root = tree.getroot()
    used_hulls: list[Path] = []
    for mesh_tag in root.findall(".//mesh"):
        mesh_ref = str(mesh_tag.get("filename", "")).strip()
        if not mesh_ref:
            raise ValueError(f"empty mesh filename in {src_urdf}")
        if mesh_ref.startswith(("package://", "http://", "https://", "file://")):
            raise ValueError(f"unsupported mesh URI in {src_urdf}: {mesh_ref}")
        src_mesh = _resolve_path(mesh_ref, src_urdf.parent)
        if not src_mesh.is_file():
            raise FileNotFoundError(f"missing source mesh referenced by {src_urdf}: {src_mesh}")

        dst_mesh = hull_by_mesh.get(src_mesh)
        if dst_mesh is None:
            safe_stem = _safe_name(src_mesh.stem)
            digest = hashlib.sha1(str(src_mesh).encode("utf-8")).hexdigest()[:10]
            dst_mesh = target_bank / "objects_convex_hull" / f"{safe_stem}_{digest}" / f"{safe_stem}_convex_hull.obj"
            v0, f0, v1, f1 = _export_convex_hull(src_mesh, dst_mesh)
            hull_by_mesh[src_mesh] = dst_mesh
            stats_by_mesh[str(src_mesh)] = {
                "source_mesh": str(src_mesh),
                "convex_hull_mesh": str(dst_mesh),
                "source_vertices": v0,
                "source_faces": f0,
                "hull_vertices": v1,
                "hull_faces": f1,
            }
        mesh_tag.set("filename", _relpath(dst_mesh, dst_urdf.parent))
        used_hulls.append(dst_mesh)

    dst_urdf.parent.mkdir(parents=True, exist_ok=True)
    tree.write(dst_urdf, encoding="utf-8", xml_declaration=True)
    return used_hulls


def _copy_or_link(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(os.path.relpath(src.resolve(), start=dst.parent.resolve()))
    elif src.is_dir():
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy2(src, dst)


def _category_for(clip_id: str) -> str:
    if clip_id.startswith("box_"):
        return "box"
    lowered = clip_id.lower()
    for key in ("ball", "barrel", "bin", "chair", "lamp", "monitor", "table"):
        if key in lowered:
            return key
    return "other"


def prepare_bank(
    *,
    source_bank: Path,
    target_bank: Path,
    expected_count: int | None,
    payload_mode: str,
    sidecar_mode: str,
    force: bool,
) -> dict[str, Any]:
    source_bank = source_bank.expanduser().resolve()
    target_bank = target_bank.expanduser().resolve()
    source_map = source_bank / "_clip_object_urdf_map.json"
    if not source_bank.is_dir():
        raise FileNotFoundError(f"source bank does not exist: {source_bank}")
    if not source_map.is_file():
        raise FileNotFoundError(f"source object map does not exist: {source_map}")

    metadata, clips = _load_clip_map(source_map)
    if expected_count is not None and len(clips) != expected_count:
        raise ValueError(f"expected {expected_count} clips, found {len(clips)} in {source_map}")

    marker = target_bank / MARKER_NAME
    if target_bank.exists():
        if not force and not marker.exists():
            raise FileExistsError(f"refusing to overwrite non-generated target bank: {target_bank}")
        for child in target_bank.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        target_bank.mkdir(parents=True)

    hull_by_mesh: dict[Path, Path] = {}
    stats_by_mesh: dict[str, dict[str, int | str]] = {}
    urdf_by_src: dict[Path, Path] = {}
    hulls_by_urdf: dict[Path, list[Path]] = {}
    updated_clips: dict[str, Any] = {}

    urdf_dir = target_bank / "_single_slot_motion_bank" / "_single_slot_urdfs"
    for clip_id, raw_entry in sorted(clips.items()):
        entry = dict(raw_entry) if isinstance(raw_entry, dict) else {"object_urdf_path": str(raw_entry)}
        raw_urdf = str(entry.get("object_urdf_path", "")).strip()
        if not raw_urdf:
            raise ValueError(f"clip {clip_id} has empty object_urdf_path")
        src_urdf = _resolve_path(raw_urdf, source_map.parent)
        if not src_urdf.is_file():
            raise FileNotFoundError(f"clip {clip_id} missing source URDF: {src_urdf}")

        dst_urdf = urdf_by_src.get(src_urdf)
        if dst_urdf is None:
            dst_urdf = urdf_dir / src_urdf.name
            hulls_by_urdf[src_urdf] = _rewrite_urdf_to_hulls(
                src_urdf=src_urdf,
                dst_urdf=dst_urdf,
                target_bank=target_bank,
                hull_by_mesh=hull_by_mesh,
                stats_by_mesh=stats_by_mesh,
            )
            urdf_by_src[src_urdf] = dst_urdf

        installed_urdf = str(dst_urdf)
        source_npz = source_bank / f"{clip_id}.npz"
        if not source_npz.is_file():
            raise FileNotFoundError(f"clip {clip_id} missing source npz: {source_npz}")
        _write_npz_with_urdf(source_npz, target_bank / f"{clip_id}.npz", installed_urdf)

        source_slot_npz = source_bank / "_single_slot_motion_bank" / f"{clip_id}.npz"
        slot_npz_source = source_slot_npz if source_slot_npz.is_file() else source_npz
        _write_npz_with_urdf(
            slot_npz_source,
            target_bank / "_single_slot_motion_bank" / f"{clip_id}.npz",
            installed_urdf,
        )

        entry["object_urdf_path"] = installed_urdf
        mesh_paths = sorted({str(path) for path in hulls_by_urdf.get(src_urdf, [])})
        if len(mesh_paths) == 1:
            entry["object_mesh_path"] = mesh_paths[0]
        updated_clips[clip_id] = entry

    for child in source_bank.iterdir():
        if child.name in {
            "_clip_object_urdf_map.json",
            "objects",
            "objects_convex_hull",
            MARKER_NAME,
            "convex_hull_manifest.json",
            "nfs_package_manifest.json",
            "cp_corl_local_manifest.json",
        }:
            continue
        if child.name.startswith("_single_slot_motion_bank"):
            continue
        if child.suffix == ".npz":
            continue
        if child.name.startswith("."):
            continue
        target_child = target_bank / child.name
        if child.is_dir():
            _copy_or_link(child, target_child, mode=sidecar_mode)
        elif child.is_file() or child.is_symlink():
            _copy_or_link(child, target_child, mode=payload_mode)

    output_payload = dict(metadata)
    output_payload["clips"] = updated_clips
    output_payload["convex_hull_object_bank"] = {
        "source_bank": str(source_bank),
        "unique_source_urdf_count": len(urdf_by_src),
        "unique_source_mesh_count": len(hull_by_mesh),
        "visual_and_collision_meshes": "convex_hull",
    }
    map_path = target_bank / "_clip_object_urdf_map.json"
    map_path.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    slot_map = target_bank / "_single_slot_motion_bank" / "_clip_object_urdf_map.json"
    slot_map.parent.mkdir(parents=True, exist_ok=True)
    slot_map.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    category_counts = dict(sorted(Counter(_category_for(clip_id) for clip_id in updated_clips).items()))
    contact_root = target_bank / "contact_export_from_teacher_success133_final0p5" / "clips"
    contact_dirs = [p for p in contact_root.iterdir() if p.is_dir()] if contact_root.exists() else []
    sidecar_files = sum(1 for d in contact_dirs for f in d.iterdir() if f.is_file())
    manifest = {
        "package_kind": "convex_hull_object_bank",
        "source_bank": str(source_bank),
        "target_bank": str(target_bank),
        "clip_count": len(updated_clips),
        "category_counts": category_counts,
        "top_level_npz": len(list(target_bank.glob("*.npz"))),
        "single_slot_motion_npz": len(list((target_bank / "_single_slot_motion_bank").glob("*.npz"))),
        "urdf_files": len(list(urdf_dir.glob("*.urdf"))),
        "convex_hull_meshes": len(list((target_bank / "objects_convex_hull").rglob("*.obj"))),
        "unique_source_meshes": len(hull_by_mesh),
        "contact_clip_dirs": len(contact_dirs),
        "sidecar_files": sidecar_files,
        "payload_mode": payload_mode,
        "sidecar_mode": sidecar_mode,
        "mesh_stats": list(stats_by_mesh.values()),
    }
    (target_bank / "convex_hull_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_bank / "nfs_package_manifest.json").write_text(
        json.dumps({k: v for k, v in manifest.items() if k != "mesh_stats"}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    marker.write_text(
        "generated by scripts/prepare_convex_hull_object_bank.py\n"
        f"source_bank={source_bank}\n"
        f"clips={len(updated_clips)}\n",
        encoding="utf-8",
    )
    return manifest


def validate_bank(target_bank: Path, expected_count: int | None) -> list[str]:
    errors: list[str] = []
    _, clips = _load_clip_map(target_bank / "_clip_object_urdf_map.json")
    if expected_count is not None and len(clips) != expected_count:
        errors.append(f"map clips={len(clips)} expected={expected_count}")
    for clip_id, raw_entry in sorted(clips.items()):
        entry = dict(raw_entry) if isinstance(raw_entry, dict) else {"object_urdf_path": str(raw_entry)}
        urdf = Path(str(entry.get("object_urdf_path", "")).strip())
        if not urdf.is_file():
            errors.append(f"{clip_id}: missing urdf {urdf}")
            continue
        root = ET.parse(urdf).getroot()
        mesh_filenames: list[str] = []
        for mesh_tag in root.findall(".//mesh"):
            filename = str(mesh_tag.get("filename", "")).strip()
            mesh = _resolve_path(filename, urdf.parent)
            mesh_filenames.append(str(mesh))
            if not mesh.is_file():
                errors.append(f"{clip_id}: missing mesh {mesh}")
            if "objects_convex_hull" not in str(mesh):
                errors.append(f"{clip_id}: mesh is not in objects_convex_hull: {mesh}")
        if not mesh_filenames:
            errors.append(f"{clip_id}: urdf has no mesh tags")
        visuals = [str(m.get("filename", "")).strip() for m in root.findall(".//visual//mesh")]
        collisions = [str(m.get("filename", "")).strip() for m in root.findall(".//collision//mesh")]
        if visuals and collisions and set(visuals) != set(collisions):
            errors.append(f"{clip_id}: visual/collision mesh refs differ")
        for npz_path in [target_bank / f"{clip_id}.npz", target_bank / "_single_slot_motion_bank" / f"{clip_id}.npz"]:
            if npz_path.exists():
                with np.load(npz_path, allow_pickle=True) as data:
                    npz_urdf = _scalar_str(data.get("object_urdf_path"))
                if npz_urdf != str(urdf):
                    errors.append(f"{clip_id}: {npz_path.name} object_urdf_path mismatch")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a bank whose URDF visual and collision meshes use convex hull OBJ assets.")
    parser.add_argument("--source-bank", required=True, type=Path)
    parser.add_argument("--target-bank", required=True, type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--payload-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--sidecar-mode", choices=("copy", "symlink"), default="copy")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        manifest = prepare_bank(
            source_bank=args.source_bank,
            target_bank=args.target_bank,
            expected_count=args.expected_count,
            payload_mode=args.payload_mode,
            sidecar_mode=args.sidecar_mode,
            force=args.force,
        )
        errors = validate_bank(args.target_bank.expanduser().resolve(), args.expected_count)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    if errors:
        print("[ERROR] Convex hull bank validation failed:", file=sys.stderr)
        for error in errors[:30]:
            print(f"  {error}", file=sys.stderr)
        return 3

    print(json.dumps({k: v for k, v in manifest.items() if k != "mesh_stats"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
