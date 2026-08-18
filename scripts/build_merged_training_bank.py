#!/usr/bin/env python3
"""Build and verify a portable, immutable merged AS training bank.

The builder is intentionally fail-closed.  It accepts multiple already-sealed
single-slot motion views, proves that their clip namespaces and motion schemas
are compatible, closes every URDF visual/collision mesh dependency, and copies
the per-clip contact sidecars.  Published object-map paths and URDF mesh paths
are relative to the bank, so no source-machine or NFS fallback is required.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tarfile
import tempfile
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np


MANIFEST_VERSION = 1
CONSTRUCTION_CONTRACT_VERSION = 1
MANIFEST_NAME = "manifest.json"
MARKER_NAME = ".generated_by_build_merged_training_bank"
OBJECT_MAP_NAME = "_clip_object_urdf_map.json"
URDF_DIR_NAME = "_single_slot_urdfs"
MESH_DIR_NAME = "_mesh_assets"
ALLOWED_CATEGORIES = ("box", "ball", "barrel", "bin")
SINGLE_MESH_KEYS = (
    "object_mesh_path",
    "object_visual_mesh_path",
    "object_collision_mesh_path",
)
MULTI_MESH_KEYS = (
    "object_mesh_paths",
    "object_visual_mesh_paths",
    "object_collision_mesh_paths",
)
REQUIRED_MOTION_KEYS = frozenset(
    {
        "body_ang_vel_w",
        "body_lin_vel_w",
        "body_names",
        "body_pos_w",
        "body_quat_w",
        "fps",
        "joint_names",
        "joint_pos",
        "joint_vel",
        "object_ang_vel_w",
        "object_lin_vel_w",
        "object_name",
        "object_pos_w",
        "object_quat_w",
        "object_size",
        "object_urdf_path",
    }
)


@dataclass(frozen=True)
class SourceSpec:
    label: str
    motion_dir: Path
    contact_root: Path


@dataclass(frozen=True)
class FileRecord:
    path: str
    size: int
    sha256: str

    def payload(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass
class MeshRecord:
    source_path: Path
    source_record: FileRecord
    clips_and_roles: set[tuple[str, str]]
    has_mtllib: bool


@dataclass
class ClipRecord:
    clip_id: str
    source_label: str
    category: str
    motion_path: Path
    motion_record: FileRecord
    source_map_entry: dict[str, Any]
    source_map_dir: Path
    urdf_path: Path
    urdf_record: FileRecord
    urdf_mesh_paths: list[tuple[str, Path]]
    urdf_semantic_signature: Any
    contact_dir: Path
    contact_records: list[FileRecord]


@dataclass
class AuditResult:
    sources: list[dict[str, Any]]
    clips: list[ClipRecord]
    meshes_by_path: dict[Path, MeshRecord]
    payload_digest: str
    identity: dict[str, Any]
    schema_contract: dict[str, Any]
    category_counts: dict[str, int]


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned or "object"


def require_regular_file(path: Path, *, role: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be one regular non-symlink file: {path}")
    return path


def stable_file_record(path: Path, *, record_path: str | None = None) -> FileRecord:
    path = require_regular_file(path, role="Hashed input")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise RuntimeError(f"File changed while hashing: {path}")
    return FileRecord(
        path=record_path if record_path is not None else str(path.resolve()),
        size=int(after.st_size),
        sha256=digest.hexdigest(),
    )


def copy_verified(source: Path, destination: Path, expected: FileRecord) -> None:
    require_regular_file(source, role="Copy source")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as src, destination.open("xb") as dst:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b""):
            dst.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    if size != expected.size or digest.hexdigest() != expected.sha256:
        raise RuntimeError(
            f"Source changed while copying: {source}; "
            f"expected={expected.sha256}/{expected.size} "
            f"actual={digest.hexdigest()}/{size}"
        )
    os.chmod(destination, 0o444)


def load_object_map(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    require_regular_file(path, role="Object map")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Object map must be a JSON object: {path}")
    raw_clips = payload.get("clips")
    if not isinstance(raw_clips, dict) or not raw_clips:
        raise ValueError(f"Object map has no clips mapping: {path}")
    clips: dict[str, dict[str, Any]] = {}
    for clip_id, raw_entry in raw_clips.items():
        if not isinstance(clip_id, str) or not clip_id:
            raise ValueError(f"Invalid object-map clip ID in {path}: {clip_id!r}")
        if isinstance(raw_entry, str):
            clips[clip_id] = {"object_urdf_path": raw_entry}
        elif isinstance(raw_entry, dict):
            clips[clip_id] = dict(raw_entry)
        else:
            raise ValueError(f"Invalid object-map entry for {clip_id}: {type(raw_entry).__name__}")
    return {key: value for key, value in payload.items() if key != "clips"}, clips


def resolve_path(raw: str, *, base_dir: Path, role: str) -> Path:
    value = str(raw).strip()
    if not value:
        raise ValueError(f"Empty path for {role}")
    if value.startswith(("package://", "http://", "https://", "file://")):
        raise ValueError(f"Unsupported non-file-backed path for {role}: {value}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return require_regular_file(path.resolve(), role=role)


def infer_contact_clip_id(directory_name: str) -> str:
    match = re.fullmatch(r"\d+_(.+)", directory_name)
    return match.group(1) if match else directory_name


def discover_contact_dirs(contact_root: Path) -> dict[str, Path]:
    clips_root = contact_root / "clips" if (contact_root / "clips").is_dir() else contact_root
    if clips_root.is_symlink() or not clips_root.is_dir():
        raise ValueError(f"Contact clips root is missing or symlinked: {clips_root}")
    result: dict[str, Path] = {}
    for directory in sorted(clips_root.iterdir()):
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Unexpected non-directory contact entry: {directory}")
        clip_id = infer_contact_clip_id(directory.name)
        if clip_id in result:
            raise ValueError(f"Duplicate contact directories for clip {clip_id}: {result[clip_id]}, {directory}")
        result[clip_id] = directory
    return result


def category_for_clip(clip_id: str, entry: dict[str, Any]) -> str:
    explicit = str(
        entry.get("mesh_physics_category")
        or entry.get("object_category")
        or entry.get("category")
        or ""
    ).strip().lower()
    if explicit in ALLOWED_CATEGORIES:
        return explicit
    for category in ALLOWED_CATEGORIES:
        if re.search(rf"(?:^|_){re.escape(category)}(?:_|$)", clip_id.lower()):
            return category
    raise ValueError(f"Could not infer object category for clip {clip_id}")


def contains_obj_mtllib(path: Path) -> bool:
    if path.suffix.lower() != ".obj":
        return False
    pattern = re.compile(br"(?im)^[ \t]*mtllib[ \t]+")
    carry = b""
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            data = carry + chunk
            if pattern.search(data):
                return True
            carry = data[-256:]
    return False


def xml_signature(root: ET.Element, mesh_tokens: dict[int, str]) -> Any:
    def visit(node: ET.Element) -> Any:
        attrs = dict(node.attrib)
        if node.tag == "mesh" and id(node) in mesh_tokens:
            attrs["filename"] = mesh_tokens[id(node)]
        return (
            node.tag,
            tuple(sorted((str(key), str(value)) for key, value in attrs.items())),
            (node.text or "").strip(),
            tuple(visit(child) for child in list(node)),
        )

    return visit(root)


def audit_urdf(
    *,
    clip_id: str,
    urdf_path: Path,
) -> tuple[list[tuple[str, Path]], Any]:
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF for {clip_id}: {urdf_path}: {exc}") from exc
    links = root.findall("link")
    if len(links) != 1:
        raise ValueError(f"{clip_id} URDF must have exactly one link, found {len(links)}: {urdf_path}")
    mesh_paths: list[tuple[str, Path]] = []
    mesh_tokens: dict[int, str] = {}
    for role in ("visual", "collision"):
        geometries = root.findall(f".//{role}/geometry")
        if not geometries:
            raise ValueError(f"{clip_id} URDF has no {role} geometry: {urdf_path}")
        for geometry in geometries:
            children = list(geometry)
            if len(children) != 1 or children[0].tag != "mesh":
                tags = [child.tag for child in children]
                raise ValueError(
                    f"{clip_id} {role} geometry is not exactly one real mesh; "
                    f"tags={tags}: {urdf_path}"
                )
            mesh = children[0]
            mesh_path = resolve_path(
                str(mesh.get("filename", "")),
                base_dir=urdf_path.parent,
                role=f"{clip_id} {role} mesh",
            )
            mesh_paths.append((role, mesh_path))
            mesh_tokens[id(mesh)] = str(mesh_path)
    return mesh_paths, xml_signature(root, mesh_tokens)


def audit_motion_schema(
    path: Path,
    *,
    baseline_static: dict[str, np.ndarray] | None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as data:
        keys = frozenset(data.files)
        if keys != REQUIRED_MOTION_KEYS:
            missing = sorted(REQUIRED_MOTION_KEYS - keys)
            extra = sorted(keys - REQUIRED_MOTION_KEYS)
            raise ValueError(f"Motion schema mismatch in {path}: missing={missing} extra={extra}")
        body_names = np.asarray(data["body_names"])
        joint_names = np.asarray(data["joint_names"])
        fps = np.asarray(data["fps"])
        if body_names.shape != (32,) or joint_names.shape != (29,) or fps.size != 1:
            raise ValueError(
                f"Static motion schema mismatch in {path}: "
                f"body_names={body_names.shape} joint_names={joint_names.shape} fps={fps.shape}"
            )
        if int(fps.reshape(-1)[0]) != 50:
            raise ValueError(f"Expected 50 FPS in {path}, found {fps!r}")
        trajectory_length = int(np.asarray(data["body_pos_w"]).shape[0])
        expected_shapes = {
            "body_pos_w": (trajectory_length, 32, 3),
            "body_quat_w": (trajectory_length, 32, 4),
            "body_lin_vel_w": (trajectory_length, 32, 3),
            "body_ang_vel_w": (trajectory_length, 32, 3),
            "joint_pos": (trajectory_length, 36),
            "joint_vel": (trajectory_length, 35),
            "object_pos_w": (trajectory_length, 3),
            "object_quat_w": (trajectory_length, 4),
            "object_lin_vel_w": (trajectory_length, 3),
            "object_ang_vel_w": (trajectory_length, 3),
            "object_size": (3,),
        }
        for key, expected_shape in expected_shapes.items():
            value = np.asarray(data[key])
            if value.shape != expected_shape:
                raise ValueError(f"{path} {key} shape={value.shape}, expected={expected_shape}")
            if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                raise ValueError(f"{path} {key} contains non-finite values")
        current_static = {"body_names": body_names.copy(), "joint_names": joint_names.copy()}
        if baseline_static is not None:
            for key, value in current_static.items():
                if not np.array_equal(value, baseline_static[key]):
                    raise ValueError(f"{path} {key} differs from the merged-bank baseline")
        contract = {
            "keys": sorted(keys),
            "fps": 50,
            "body_count": 32,
            "joint_name_count": 29,
            "joint_pos_width": 36,
            "joint_vel_width": 35,
        }
        return contract, current_static


def source_relative_record(path: Path, *, root: Path) -> FileRecord:
    relative = os.path.relpath(path, root)
    return stable_file_record(path, record_path=relative)


def audit_sources(source_specs: list[SourceSpec], *, expected_total: int | None) -> AuditResult:
    if len(source_specs) < 2:
        raise ValueError("At least two --source specifications are required")
    labels = [source.label for source in source_specs]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Source labels must be unique: {labels}")

    all_clips: list[ClipRecord] = []
    source_payloads: list[dict[str, Any]] = []
    meshes_by_path: dict[Path, MeshRecord] = {}
    owner_by_clip: dict[str, str] = {}
    baseline_static: dict[str, np.ndarray] | None = None
    schema_contract: dict[str, Any] | None = None
    transition_semantics: tuple[int, str] | None = None

    for source in source_specs:
        motion_dir = source.motion_dir.expanduser().resolve()
        contact_root = source.contact_root.expanduser().resolve()
        if motion_dir.is_symlink() or not motion_dir.is_dir():
            raise ValueError(f"Source motion dir is missing or symlinked: {motion_dir}")
        map_path = motion_dir / OBJECT_MAP_NAME
        metadata, clip_map = load_object_map(map_path)
        motion_paths = {path.stem: path for path in sorted(motion_dir.glob("*.npz"))}
        if not motion_paths:
            raise ValueError(f"No top-level motion NPZ files under {motion_dir}")
        if set(motion_paths) != set(clip_map):
            raise ValueError(
                f"Motion/map clip mismatch for {source.label}: "
                f"motion_only={sorted(set(motion_paths) - set(clip_map))[:10]} "
                f"map_only={sorted(set(clip_map) - set(motion_paths))[:10]}"
            )
        transition = metadata.get("motion_transition_source")
        if not isinstance(transition, dict):
            raise ValueError(f"Missing motion_transition_source in {map_path}")
        current_transition = (
            int(transition.get("version", -1)),
            str(transition.get("source_semantics", "")),
        )
        if current_transition != (1, "global_multi_clip_runtime"):
            raise ValueError(f"Unsupported motion transition source in {map_path}: {transition}")
        if transition_semantics is None:
            transition_semantics = current_transition
        elif transition_semantics != current_transition:
            raise ValueError("Source banks use different motion-transition semantics")

        contacts = discover_contact_dirs(contact_root)
        if set(contacts) != set(motion_paths):
            raise ValueError(
                f"Motion/contact clip mismatch for {source.label}: "
                f"motion_only={sorted(set(motion_paths) - set(contacts))[:10]} "
                f"contact_only={sorted(set(contacts) - set(motion_paths))[:10]}"
            )

        source_manifest = motion_dir / MANIFEST_NAME
        source_manifest_record = (
            stable_file_record(source_manifest, record_path=MANIFEST_NAME).payload()
            if source_manifest.is_file() and not source_manifest.is_symlink()
            else None
        )
        source_clip_identity: list[dict[str, Any]] = []
        for clip_id in sorted(motion_paths):
            previous_owner = owner_by_clip.get(clip_id)
            if previous_owner is not None:
                raise ValueError(
                    f"Clip ID collision across sources: {clip_id} is in {previous_owner} and {source.label}"
                )
            owner_by_clip[clip_id] = source.label
            motion_path = require_regular_file(motion_paths[clip_id], role=f"{clip_id} motion")
            current_contract, current_static = audit_motion_schema(
                motion_path,
                baseline_static=baseline_static,
            )
            if baseline_static is None:
                baseline_static = current_static
                schema_contract = current_contract
            elif current_contract != schema_contract:
                raise ValueError(f"Motion schema contract differs for {motion_path}")
            motion_record = source_relative_record(motion_path, root=motion_dir)

            entry = clip_map[clip_id]
            raw_urdf = str(entry.get("object_urdf_path", "")).strip()
            urdf_path = resolve_path(raw_urdf, base_dir=map_path.parent, role=f"{clip_id} URDF")
            urdf_record = stable_file_record(urdf_path)
            urdf_mesh_paths, urdf_signature = audit_urdf(clip_id=clip_id, urdf_path=urdf_path)
            for role, mesh_path in urdf_mesh_paths:
                mesh = meshes_by_path.get(mesh_path)
                if mesh is None:
                    mesh_record = stable_file_record(mesh_path)
                    mesh = MeshRecord(
                        source_path=mesh_path,
                        source_record=mesh_record,
                        clips_and_roles=set(),
                        has_mtllib=contains_obj_mtllib(mesh_path),
                    )
                    meshes_by_path[mesh_path] = mesh
                mesh.clips_and_roles.add((clip_id, role))
                if mesh.has_mtllib:
                    raise ValueError(
                        f"Mesh has an external OBJ material dependency; package it explicitly before merging: "
                        f"{mesh_path}"
                    )

            contact_dir = contacts[clip_id]
            contact_files = sorted(path for path in contact_dir.rglob("*") if path.is_file())
            if not contact_files:
                raise ValueError(f"No contact sidecar files for {clip_id}: {contact_dir}")
            if any(path.is_symlink() for path in contact_dir.rglob("*")):
                raise ValueError(f"Contact sidecar closure contains a symlink: {contact_dir}")
            required_sidecars = {
                "left_wrist_contact_points.npy",
                "left_wrist_contact_point_counts.npy",
                "left_wrist_contact_interval_steps.npy",
                "right_wrist_contact_points.npy",
                "right_wrist_contact_point_counts.npy",
                "right_wrist_contact_interval_steps.npy",
                "teacher_rollout_reference.npz",
            }
            direct_names = {path.name for path in contact_dir.iterdir() if path.is_file()}
            missing_sidecars = sorted(required_sidecars - direct_names)
            if missing_sidecars:
                raise ValueError(f"{clip_id} is missing required contact sidecars: {missing_sidecars}")
            contact_records = [
                source_relative_record(path, root=contact_dir)
                for path in contact_files
            ]
            category = category_for_clip(clip_id, entry)
            clip_record = ClipRecord(
                clip_id=clip_id,
                source_label=source.label,
                category=category,
                motion_path=motion_path,
                motion_record=motion_record,
                source_map_entry=entry,
                source_map_dir=map_path.parent,
                urdf_path=urdf_path,
                urdf_record=urdf_record,
                urdf_mesh_paths=urdf_mesh_paths,
                urdf_semantic_signature=urdf_signature,
                contact_dir=contact_dir,
                contact_records=contact_records,
            )
            all_clips.append(clip_record)
            source_clip_identity.append(
                {
                    "clip_id": clip_id,
                    "category": category,
                    "motion": motion_record.payload(),
                    "urdf": {
                        "size": urdf_record.size,
                        "sha256": urdf_record.sha256,
                    },
                    "meshes": [
                        {
                            "role": role,
                            "size": meshes_by_path[path].source_record.size,
                            "sha256": meshes_by_path[path].source_record.sha256,
                        }
                        for role, path in urdf_mesh_paths
                    ],
                    "contacts": [record.payload() for record in contact_records],
                }
            )
        source_payloads.append(
            {
                "label": source.label,
                "clip_count": len(motion_paths),
                "object_map": stable_file_record(
                    map_path, record_path=OBJECT_MAP_NAME
                ).payload(),
                "source_manifest": source_manifest_record,
                "clips": source_clip_identity,
            }
        )

    all_clips.sort(key=lambda clip: clip.clip_id)
    if expected_total is not None and len(all_clips) != expected_total:
        raise ValueError(f"Expected {expected_total} merged clips, found {len(all_clips)}")
    assert schema_contract is not None
    category_counts = dict(sorted(Counter(clip.category for clip in all_clips).items()))
    identity = {
        "manifest_version": MANIFEST_VERSION,
        "construction_contract_version": CONSTRUCTION_CONTRACT_VERSION,
        "kind": "portable_immutable_merged_as_training_bank",
        "motion_schema": schema_contract,
        "motion_transition_source": {
            "version": 1,
            "source_semantics": "global_multi_clip_runtime",
            "source_clip_count": len(all_clips),
        },
        "geometry_contract": {
            "visual": "source_real_mesh",
            "collision": "source_real_mesh",
            "primitive_geometry_count": 0,
            "fallback_allowed": False,
            "mesh_assets_content_addressed": True,
        },
        "clip_count": len(all_clips),
        "category_counts": category_counts,
        "sources": source_payloads,
    }
    return AuditResult(
        sources=source_payloads,
        clips=all_clips,
        meshes_by_path=meshes_by_path,
        payload_digest=sha256_json(identity),
        identity=identity,
        schema_contract=schema_contract,
        category_counts=category_counts,
    )


def relpath(path: Path, root: Path) -> str:
    return os.path.relpath(path.resolve(), root.resolve())


def map_mesh_path(raw: str, *, clip: ClipRecord, mesh_outputs: dict[Path, Path]) -> str:
    source_path = resolve_path(raw, base_dir=clip.source_map_dir, role=f"{clip.clip_id} map mesh")
    output = mesh_outputs.get(source_path)
    if output is None:
        raise ValueError(
            f"{clip.clip_id} map mesh is not part of its URDF visual/collision closure: {source_path}"
        )
    return output.as_posix()


def build_published_map(
    audit: AuditResult,
    *,
    urdf_outputs: dict[str, Path],
    mesh_outputs: dict[Path, Path],
) -> dict[str, Any]:
    clips: dict[str, Any] = {}
    for clip in audit.clips:
        entry = dict(clip.source_map_entry)
        entry["object_urdf_path"] = urdf_outputs[clip.clip_id].as_posix()
        for key in SINGLE_MESH_KEYS:
            raw = str(entry.get(key, "")).strip()
            if raw:
                entry[key] = map_mesh_path(raw, clip=clip, mesh_outputs=mesh_outputs)
        for key in MULTI_MESH_KEYS:
            raw_values = entry.get(key)
            if raw_values is None:
                continue
            if isinstance(raw_values, str):
                raw_values = [raw_values]
            if not isinstance(raw_values, (list, tuple)) or not raw_values:
                raise ValueError(f"Invalid {key} for {clip.clip_id}: {raw_values!r}")
            entry[key] = [
                map_mesh_path(str(raw), clip=clip, mesh_outputs=mesh_outputs)
                for raw in raw_values
            ]
        entry["merged_bank_source_label"] = clip.source_label
        clips[clip.clip_id] = entry
    return {
        "motion_transition_source": {
            "version": 1,
            "source_semantics": "global_multi_clip_runtime",
            "source_clip_count": len(clips),
        },
        "merged_bank_contract": {
            "version": CONSTRUCTION_CONTRACT_VERSION,
            "source_labels": [source["label"] for source in audit.sources],
            "geometry": "real_visual_and_real_collision_mesh_no_fallback",
        },
        "clips": clips,
    }


def publish_urdf(
    clip: ClipRecord,
    *,
    destination: Path,
    mesh_outputs: dict[Path, Path],
    temp_root: Path,
) -> None:
    tree = ET.parse(clip.urdf_path)
    root = tree.getroot()
    source_tokens: dict[int, str] = {}
    output_tokens: dict[int, str] = {}
    for mesh in root.findall(".//mesh"):
        source_path = resolve_path(
            str(mesh.get("filename", "")),
            base_dir=clip.urdf_path.parent,
            role=f"{clip.clip_id} URDF mesh",
        )
        output_rel = mesh_outputs[source_path]
        mesh.set("filename", os.path.relpath(temp_root / output_rel, destination.parent))
        token = f"sha256:{clip_mesh_digest(clip, source_path)}"
        source_tokens[id(mesh)] = token
        output_tokens[id(mesh)] = token
    # The parsed source tree has only changed mesh filename strings.  Replacing
    # those strings with content tokens must reproduce the audited semantics.
    rewritten_signature = xml_signature(root, output_tokens)
    expected_signature = replace_mesh_paths_in_signature(
        clip.urdf_semantic_signature,
        clip=clip,
    )
    if rewritten_signature != expected_signature:
        raise RuntimeError(f"URDF semantic equivalence failed before publication: {clip.clip_id}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    os.chmod(destination, 0o444)


def clip_mesh_digest(clip: ClipRecord, path: Path) -> str:
    for _role, candidate in clip.urdf_mesh_paths:
        if candidate == path:
            # The same source path has one stable digest independent of role.
            return stable_digest_lookup[clip.clip_id][path]
    raise KeyError(path)


# Filled only during publication.  Keeping this lookup module-local avoids
# embedding filesystem paths in the immutable identity or URDF XML.
stable_digest_lookup: dict[str, dict[Path, str]] = {}


def replace_mesh_paths_in_signature(signature: Any, *, clip: ClipRecord) -> Any:
    path_to_digest = stable_digest_lookup[clip.clip_id]

    def visit(node: Any) -> Any:
        tag, attrs, text, children = node
        updated = []
        for key, value in attrs:
            if tag == "mesh" and key == "filename":
                path = Path(value)
                digest = path_to_digest.get(path)
                if digest is None:
                    raise RuntimeError(f"Missing audited mesh digest for {clip.clip_id}: {path}")
                value = f"sha256:{digest}"
            updated.append((key, value))
        return (tag, tuple(updated), text, tuple(visit(child) for child in children))

    return visit(signature)


def freeze_tree(root: Path) -> None:
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Published bank contains a symlink: {path}")
        if path.is_file():
            os.chmod(path, 0o444)
        elif path.is_dir():
            directories.append(path)
    for directory in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        os.chmod(directory, 0o555)


def thaw_and_remove(root: Path) -> None:
    if not root.exists():
        return
    for current, directories, _files in os.walk(root):
        os.chmod(current, 0o700)
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                os.chmod(path, 0o700)
    shutil.rmtree(root)


def published_file_records(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in {MANIFEST_NAME, MARKER_NAME}:
            continue
        records.append(stable_file_record(path, record_path=relative).payload())
    return records


def build_bank(
    source_specs: list[SourceSpec],
    *,
    output_base: Path,
    contact_export_name: str,
    expected_total: int | None,
) -> tuple[Path, str]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", contact_export_name):
        raise ValueError(f"Unsafe contact export name: {contact_export_name!r}")
    audit = audit_sources(source_specs, expected_total=expected_total)
    output_base = output_base.expanduser().resolve()
    output_root = output_base / "by-source" / audit.payload_digest
    if output_root.exists() or output_root.is_symlink():
        manifest_sha = verify_bank(output_root, expected_digest=audit.payload_digest)
        return output_root, manifest_sha

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{audit.payload_digest}.incoming-", dir=output_root.parent)
    )
    try:
        global stable_digest_lookup
        stable_digest_lookup = {
            clip.clip_id: {
                path: audit.meshes_by_path[path].source_record.sha256
                for _role, path in clip.urdf_mesh_paths
            }
            for clip in audit.clips
        }

        for clip in audit.clips:
            copy_verified(clip.motion_path, temp_root / f"{clip.clip_id}.npz", clip.motion_record)

        mesh_outputs: dict[Path, Path] = {}
        output_by_content: dict[tuple[str, str], Path] = {}
        for source_path, mesh in sorted(audit.meshes_by_path.items(), key=lambda item: str(item[0])):
            suffix = source_path.suffix.lower() or ".mesh"
            key = (mesh.source_record.sha256, suffix)
            output_rel = output_by_content.get(key)
            if output_rel is None:
                output_rel = Path(MESH_DIR_NAME) / f"{mesh.source_record.sha256}{suffix}"
                copy_verified(source_path, temp_root / output_rel, mesh.source_record)
                output_by_content[key] = output_rel
            mesh_outputs[source_path] = output_rel

        urdf_outputs: dict[str, Path] = {}
        used_urdf_names: set[str] = set()
        for clip in audit.clips:
            stem = safe_stem(clip.clip_id)
            if stem in used_urdf_names:
                raise ValueError(f"Sanitized URDF name collision for clip {clip.clip_id}: {stem}")
            used_urdf_names.add(stem)
            output_rel = Path(URDF_DIR_NAME) / f"{stem}.urdf"
            publish_urdf(
                clip,
                destination=temp_root / output_rel,
                mesh_outputs=mesh_outputs,
                temp_root=temp_root,
            )
            urdf_outputs[clip.clip_id] = output_rel

        object_map = build_published_map(
            audit,
            urdf_outputs=urdf_outputs,
            mesh_outputs=mesh_outputs,
        )
        map_path = temp_root / OBJECT_MAP_NAME
        map_path.write_text(json.dumps(object_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(map_path, 0o444)

        contact_root = temp_root / contact_export_name / "clips"
        for index, clip in enumerate(audit.clips):
            destination_dir = contact_root / f"{index:04d}_{clip.clip_id}"
            destination_dir.mkdir(parents=True, exist_ok=False)
            source_by_relative = {
                record.path: record for record in clip.contact_records
            }
            for relative, record in sorted(source_by_relative.items()):
                copy_verified(clip.contact_dir / relative, destination_dir / relative, record)

        marker_path = temp_root / MARKER_NAME
        marker_path.write_text("generated by build_merged_training_bank.py\n", encoding="utf-8")
        os.chmod(marker_path, 0o444)
        published_records = published_file_records(temp_root)
        mesh_publication: dict[str, dict[str, Any]] = {}
        for source_path, mesh in audit.meshes_by_path.items():
            relative = mesh_outputs[source_path].as_posix()
            item = mesh_publication.setdefault(
                relative,
                {
                    "path": relative,
                    "size": mesh.source_record.size,
                    "sha256": mesh.source_record.sha256,
                    "source_paths": [],
                    "clips_and_roles": [],
                },
            )
            item["source_paths"].append(str(source_path))
            item["clips_and_roles"].extend(
                {"clip_id": clip_id, "role": role}
                for clip_id, role in sorted(mesh.clips_and_roles)
            )
        for item in mesh_publication.values():
            item["source_paths"] = sorted(set(item["source_paths"]))
            unique_roles = {(value["clip_id"], value["role"]) for value in item["clips_and_roles"]}
            item["clips_and_roles"] = [
                {"clip_id": clip_id, "role": role}
                for clip_id, role in sorted(unique_roles)
            ]
        manifest = {
            "version": MANIFEST_VERSION,
            "construction_contract_version": CONSTRUCTION_CONTRACT_VERSION,
            "kind": "portable_immutable_merged_as_training_bank",
            "payload_digest": audit.payload_digest,
            "clip_count": len(audit.clips),
            "category_counts": audit.category_counts,
            "contact_export_name": contact_export_name,
            "source_labels": [source["label"] for source in audit.sources],
            # Deliberately not named ``source_identity``: that key is reserved
            # for a single teacher-rollout solid-bank lineage.  This bank is a
            # heterogeneous union with two independently recorded sources.
            "merge_source_identity": audit.identity,
            "motion_schema": audit.schema_contract,
            "motion_transition_source": {
                "version": 1,
                "source_semantics": "global_multi_clip_runtime",
                "source_clip_count": len(audit.clips),
            },
            "geometry_contract": audit.identity["geometry_contract"],
            "clip_sources": {clip.clip_id: clip.source_label for clip in audit.clips},
            "published_object_map": stable_file_record(
                map_path, record_path=OBJECT_MAP_NAME
            ).payload(),
            "published_meshes": sorted(mesh_publication.values(), key=lambda item: item["path"]),
            "published_files": published_records,
        }
        manifest_path = temp_root / MANIFEST_NAME
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.chmod(manifest_path, 0o444)
        freeze_tree(temp_root)
        os.replace(temp_root, output_root)
    finally:
        if temp_root.exists():
            thaw_and_remove(temp_root)
        stable_digest_lookup = {}

    manifest_sha = verify_bank(output_root, expected_digest=audit.payload_digest)
    return output_root, manifest_sha


def verify_bank(
    root: Path,
    *,
    expected_digest: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> str:
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"Bank root is missing or symlinked: {root}")
    manifest_path = require_regular_file(root / MANIFEST_NAME, role="Merged-bank manifest")
    manifest_record = stable_file_record(manifest_path, record_path=MANIFEST_NAME)
    if expected_manifest_sha256 and manifest_record.sha256 != expected_manifest_sha256:
        raise ValueError(
            f"Manifest SHA mismatch: expected={expected_manifest_sha256} actual={manifest_record.sha256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = str(manifest.get("payload_digest", ""))
    if expected_digest and digest != expected_digest:
        raise ValueError(f"Payload digest mismatch: expected={expected_digest} actual={digest}")
    if root.name != digest:
        raise ValueError(f"Bank path is not bound to payload digest: path={root.name} digest={digest}")
    if manifest.get("kind") != "portable_immutable_merged_as_training_bank":
        raise ValueError(f"Unexpected merged-bank kind: {manifest.get('kind')!r}")
    clip_count = int(manifest.get("clip_count", -1))
    if clip_count <= 0:
        raise ValueError(f"Invalid clip count in manifest: {clip_count}")

    expected_records_raw = manifest.get("published_files")
    if not isinstance(expected_records_raw, list) or not expected_records_raw:
        raise ValueError("Manifest has no published_files records")
    expected_records = {str(record["path"]): record for record in expected_records_raw}
    actual_paths = set()
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative not in {MANIFEST_NAME, MARKER_NAME}:
            actual_paths.add(relative)
    if actual_paths != set(expected_records):
        raise ValueError(
            f"Published file set mismatch: missing={sorted(set(expected_records) - actual_paths)[:10]} "
            f"extra={sorted(actual_paths - set(expected_records))[:10]}"
        )
    for relative, expected in sorted(expected_records.items()):
        actual = stable_file_record(root / relative, record_path=relative)
        if actual.payload() != expected:
            raise ValueError(f"Published file digest mismatch: {relative}")

    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            raise ValueError(f"Merged bank contains a symlink: {path}")
        if path.stat().st_mode & stat.S_IWUSR:
            raise ValueError(f"Merged bank contains a user-writable path: {path}")

    motion_paths = sorted(root.glob("*.npz"))
    metadata, clips = load_object_map(root / OBJECT_MAP_NAME)
    if len(motion_paths) != clip_count or {path.stem for path in motion_paths} != set(clips):
        raise ValueError(
            f"Motion/map coverage mismatch: motion={len(motion_paths)} map={len(clips)} expected={clip_count}"
        )
    transition = metadata.get("motion_transition_source")
    expected_transition = {
        "version": 1,
        "source_semantics": "global_multi_clip_runtime",
        "source_clip_count": clip_count,
    }
    if transition != expected_transition:
        raise ValueError(f"Invalid merged motion-transition source: {transition}")

    category_counts: Counter[str] = Counter()
    for clip_id, entry in sorted(clips.items()):
        category_counts[category_for_clip(clip_id, entry)] += 1
        urdf = resolve_path(
            str(entry.get("object_urdf_path", "")),
            base_dir=root,
            role=f"{clip_id} published URDF",
        )
        try:
            urdf.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Published URDF escapes bank root: {urdf}") from exc
        mesh_paths, _signature = audit_urdf(clip_id=clip_id, urdf_path=urdf)
        for _role, mesh in mesh_paths:
            try:
                mesh.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"Published mesh escapes bank root: {mesh}") from exc
    if dict(sorted(category_counts.items())) != manifest.get("category_counts"):
        raise ValueError(
            f"Category counts differ from manifest: actual={dict(category_counts)} "
            f"manifest={manifest.get('category_counts')}"
        )

    contact_name = str(manifest.get("contact_export_name", ""))
    contact_dirs = discover_contact_dirs(root / contact_name)
    if set(contact_dirs) != set(clips):
        raise ValueError(
            f"Contact coverage mismatch: contacts={len(contact_dirs)} clips={len(clips)}"
        )
    return manifest_record.sha256


def verify_archive(
    archive: Path,
    *,
    expected_digest: str,
    expected_manifest_sha256: str,
    expected_archive_sha256: str | None = None,
) -> tuple[str, int]:
    archive = require_regular_file(archive.expanduser().resolve(), role="Merged-bank archive")
    archive_record = stable_file_record(archive, record_path=archive.name)
    if expected_archive_sha256 and archive_record.sha256 != expected_archive_sha256:
        raise ValueError(
            f"Archive SHA mismatch: expected={expected_archive_sha256} actual={archive_record.sha256}"
        )

    prefix = f"{expected_digest}/"
    file_records: dict[str, dict[str, Any]] = {}
    directory_names: set[str] = set()
    manifest_payload: dict[str, Any] | None = None
    manifest_sha = ""
    marker_bytes: bytes | None = None
    member_count = 0
    with tarfile.open(archive, mode="r:*") as stream:
        for member in stream:
            member_count += 1
            name = member.name.rstrip("/")
            if name == expected_digest and member.isdir():
                relative = ""
            elif name.startswith(prefix):
                relative = name[len(prefix) :]
            else:
                raise ValueError(f"Archive member escapes the digest root: {member.name}")
            if not relative and not member.isdir():
                raise ValueError(f"Digest-root archive member is not a directory: {member.name}")
            if member.issym() or member.islnk() or member.isdev() or member.isfifo():
                raise ValueError(f"Archive contains a forbidden special member: {member.name}")
            if member.mode & 0o222:
                raise ValueError(f"Archive contains a writable member: {member.name} mode={oct(member.mode)}")
            if member.isdir():
                directory_names.add(relative)
                continue
            if not member.isfile():
                raise ValueError(f"Archive contains an unsupported member type: {member.name}")
            extracted = stream.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not stream archive member: {member.name}")
            digest = hashlib.sha256()
            size = 0
            content = bytearray() if relative in {MANIFEST_NAME, MARKER_NAME} else None
            for chunk in iter(lambda: extracted.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                if content is not None:
                    content.extend(chunk)
            record = {"path": relative, "size": size, "sha256": digest.hexdigest()}
            if relative in file_records:
                raise ValueError(f"Duplicate archive member: {relative}")
            file_records[relative] = record
            if relative == MANIFEST_NAME:
                manifest_sha = record["sha256"]
                try:
                    manifest_payload = json.loads(bytes(content or b"").decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(f"Invalid manifest inside archive: {exc}") from exc
            elif relative == MARKER_NAME:
                marker_bytes = bytes(content or b"")

    if manifest_payload is None:
        raise ValueError("Archive has no merged-bank manifest")
    if manifest_sha != expected_manifest_sha256:
        raise ValueError(
            f"Archived manifest SHA mismatch: expected={expected_manifest_sha256} actual={manifest_sha}"
        )
    if manifest_payload.get("payload_digest") != expected_digest:
        raise ValueError(
            f"Archived payload digest mismatch: {manifest_payload.get('payload_digest')!r}"
        )
    if marker_bytes != b"generated by build_merged_training_bank.py\n":
        raise ValueError("Archive marker content is invalid")
    published_raw = manifest_payload.get("published_files")
    if not isinstance(published_raw, list) or not published_raw:
        raise ValueError("Archived manifest has no published_files")
    expected_files = {str(record["path"]): record for record in published_raw}
    actual_payload_files = {
        path: record
        for path, record in file_records.items()
        if path not in {MANIFEST_NAME, MARKER_NAME}
    }
    if actual_payload_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_payload_files))[:10]
        extra = sorted(set(actual_payload_files) - set(expected_files))[:10]
        changed = sorted(
            path
            for path in set(actual_payload_files).intersection(expected_files)
            if actual_payload_files[path] != expected_files[path]
        )[:10]
        raise ValueError(
            f"Archive payload differs from manifest: missing={missing} extra={extra} changed={changed}"
        )
    if "" not in directory_names:
        raise ValueError("Archive is missing its digest-root directory member")
    return archive_record.sha256, member_count


def parse_source(values: list[str]) -> SourceSpec:
    label, motion_dir, contact_root = values
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(f"Unsafe source label: {label!r}")
    return SourceSpec(label=label, motion_dir=Path(motion_dir), contact_root=Path(contact_root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Audit sources and atomically publish a merged bank")
    build.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("LABEL", "MOTION_DIR", "CONTACT_ROOT"),
        required=True,
    )
    build.add_argument("--output-base", required=True, type=Path)
    build.add_argument("--contact-export-name", required=True)
    build.add_argument("--expected-total", type=int)

    verify = subparsers.add_parser("verify", help="Rehash and validate one published bank")
    verify.add_argument("--bank", required=True, type=Path)
    verify.add_argument("--expected-digest")
    verify.add_argument("--expected-manifest-sha256")

    verify_archive_parser = subparsers.add_parser(
        "verify-archive",
        help="Stream and validate a merged-bank tar/tar.gz without extracting it",
    )
    verify_archive_parser.add_argument("--archive", required=True, type=Path)
    verify_archive_parser.add_argument("--expected-digest", required=True)
    verify_archive_parser.add_argument("--expected-manifest-sha256", required=True)
    verify_archive_parser.add_argument("--expected-archive-sha256")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "build":
            sources = [parse_source(value) for value in args.source]
            root, manifest_sha = build_bank(
                sources,
                output_base=args.output_base,
                contact_export_name=args.contact_export_name,
                expected_total=args.expected_total,
            )
            print(
                json.dumps(
                    {
                        "bank": str(root),
                        "manifest_sha256": manifest_sha,
                    },
                    sort_keys=True,
                )
            )
        elif args.command == "verify":
            manifest_sha = verify_bank(
                args.bank,
                expected_digest=args.expected_digest,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            manifest = json.loads((args.bank / MANIFEST_NAME).read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {
                        "bank": str(args.bank.expanduser().resolve()),
                        "clip_count": manifest["clip_count"],
                        "manifest_sha256": manifest_sha,
                        "payload_digest": manifest["payload_digest"],
                    },
                    sort_keys=True,
                )
            )
        else:
            archive_sha, member_count = verify_archive(
                args.archive,
                expected_digest=args.expected_digest,
                expected_manifest_sha256=args.expected_manifest_sha256,
                expected_archive_sha256=args.expected_archive_sha256,
            )
            print(
                json.dumps(
                    {
                        "archive": str(args.archive.expanduser().resolve()),
                        "archive_sha256": archive_sha,
                        "member_count": member_count,
                        "payload_digest": args.expected_digest,
                    },
                    sort_keys=True,
                )
            )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
