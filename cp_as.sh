#!/usr/bin/env bash
set -euo pipefail

# Copy and normalize an AS real-mesh training bank from NFS.
#
# This script builds a local bank under data/ds_as_data for
# train_as_general.sh. Same-named clips from later sources are kept with a
# source-name prefix so all listed sources train together.
#
# Usage:
#   bash cp_as.sh
#
# Optional env:
#   NFS_AS_ROOT=/nfs/zzzihanw/ds_as_data
#   NFS_AS_SOURCES="carryany_filter_scale_noscale_keep169_20260513 /abs/path/to/box_teacher_rollout_motion_bank"
#     (space-separated bank names under NFS_AS_ROOT, or absolute paths)
#   LOCAL_AS_ROOT=data/ds_as_data
#   OUTPUT_BANK_NAME=carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout
#   DEDUPE_IDENTICAL=0
#   DRY_RUN=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

DEFAULT_AS_BANK=carryany_filter_scale_noscale_keep169_20260513
DEFAULT_BOX_TEACHER_ROLLOUT_MOTION_BANK="${SCRIPT_DIR}/outputs/teacher_box_contacts_rollout_ref_motionbank_20260415b_utc/motion_bank"
DEFAULT_OUTPUT_BANK="${DEFAULT_AS_BANK}_plus_box_teacher_rollout"
NFS_AS_ROOT=${NFS_AS_ROOT:-/nfs/zzzihanw/ds_as_data}
NFS_AS_SOURCES=${NFS_AS_SOURCES:-"${DEFAULT_AS_BANK} ${DEFAULT_BOX_TEACHER_ROLLOUT_MOTION_BANK}"}
LOCAL_AS_ROOT=${LOCAL_AS_ROOT:-"data/ds_as_data"}
OUTPUT_BANK_NAME=${OUTPUT_BANK_NAME:-"${DEFAULT_OUTPUT_BANK}"}
DRY_RUN=${DRY_RUN:-0}
DEDUPE_IDENTICAL=${DEDUPE_IDENTICAL:-0}

EXPECTED_LOCAL_ROOT=$(python3 - "${SCRIPT_DIR}/data/ds_as_data" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)
LOCAL_AS_ROOT_ABS=$(python3 - "${LOCAL_AS_ROOT}" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)

if [[ "${LOCAL_AS_ROOT_ABS}" != "${EXPECTED_LOCAL_ROOT}" ]]; then
  echo "[ERROR] Refusing unexpected LOCAL_AS_ROOT: ${LOCAL_AS_ROOT_ABS}" >&2
  echo "[ERROR] Expected exactly: ${EXPECTED_LOCAL_ROOT}" >&2
  exit 2
fi
LOCAL_AS_ROOT="data/ds_as_data"
if [[ "${OUTPUT_BANK_NAME}" == "" || "${OUTPUT_BANK_NAME}" == "." || "${OUTPUT_BANK_NAME}" == ".." || "${OUTPUT_BANK_NAME}" == */* ]]; then
  echo "[ERROR] Unsafe OUTPUT_BANK_NAME: ${OUTPUT_BANK_NAME}" >&2
  exit 2
fi

python3 - "${NFS_AS_ROOT}" "${NFS_AS_SOURCES}" "${LOCAL_AS_ROOT}" "${OUTPUT_BANK_NAME}" "${DRY_RUN}" "${DEDUPE_IDENTICAL}" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def is_truthy(raw: str) -> bool:
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_str(value) -> str:
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
    if hasattr(item, "item"):
        item = item.item()
    return str(item).strip()


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_.")
    return cleaned or "source"


def load_clip_map(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        meta = {key: value for key, value in payload.items() if key != "clips"}
        clips = payload["clips"]
    elif isinstance(payload, dict):
        meta = {}
        clips = payload
    else:
        raise SystemExit(f"[ERROR] Invalid object map: {path}")
    return meta, clips


def rewrite_npz(src: Path, dst: Path, clip_id: str, object_name: str, entry: dict) -> None:
    with np.load(src, allow_pickle=True) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["object_name"] = np.asarray(object_name or clip_id)
    payload["object_urdf_path"] = np.asarray(str(entry.get("object_urdf_path", "")).strip())
    if entry.get("object_mesh_path"):
        payload["object_mesh_path"] = np.asarray(str(entry["object_mesh_path"]).strip())
    if entry.get("object_size") is not None:
        payload["object_size"] = np.asarray(entry["object_size"], dtype=np.float32)
    tmp_path = dst.with_name(f".{dst.stem}.tmp.npz")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def resolve_path(raw: str, base: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def resolve_source_candidates(name: str, nfs_root: Path) -> list[Path]:
    raw = Path(name).expanduser()
    if raw.is_absolute():
        return [raw.resolve()]

    nfs_home = nfs_root.parent if nfs_root.name == "ds_as_data" else nfs_root
    bases = [
        nfs_root,
        nfs_home,
        nfs_home / "ds_as_data",
        nfs_home / "as_raw",
        nfs_home / "debug_data",
    ]
    candidates: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        candidate = (base / raw).resolve()
        key = str(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
    return candidates


def validate_bank(out_dir: Path, clips: dict[str, dict]) -> None:
    npz_paths = sorted(out_dir.glob("*.npz"))
    if not npz_paths:
        raise SystemExit(f"[ERROR] No .npz clips were written to {out_dir}")
    npz_ids = {path.stem for path in npz_paths}
    map_ids = set(clips)
    if npz_ids != map_ids:
        raise SystemExit(
            f"[ERROR] Clip set mismatch: missing_map={sorted(npz_ids - map_ids)[:10]} "
            f"missing_npz={sorted(map_ids - npz_ids)[:10]}"
        )

    bad: list[str] = []
    for clip_id, entry in sorted(clips.items()):
        urdf = resolve_path(entry.get("object_urdf_path", ""), out_dir)
        mesh_raw = str(entry.get("object_mesh_path", "")).strip()
        if not urdf.is_file():
            bad.append(f"{clip_id}: missing URDF {urdf}")
            continue
        if mesh_raw and not resolve_path(mesh_raw, out_dir).is_file():
            bad.append(f"{clip_id}: missing mesh {resolve_path(mesh_raw, out_dir)}")
        try:
            root = ET.parse(urdf).getroot()
        except Exception as exc:
            bad.append(f"{clip_id}: invalid URDF {urdf}: {exc}")
            continue
        mesh_tags = root.findall(".//mesh")
        if not mesh_tags:
            bad.append(f"{clip_id}: URDF has no mesh tags {urdf}")
        for tag in mesh_tags:
            filename = str(tag.get("filename", "")).strip()
            if not filename:
                bad.append(f"{clip_id}: empty mesh filename in {urdf}")
                continue
            mesh_path = resolve_path(filename, urdf.parent)
            if not mesh_path.is_file():
                bad.append(f"{clip_id}: URDF mesh missing {mesh_path}")
        with np.load(out_dir / f"{clip_id}.npz", allow_pickle=True) as data:
            npz_urdf = scalar_str(data["object_urdf_path"]) if "object_urdf_path" in data else ""
            if npz_urdf != str(entry["object_urdf_path"]):
                bad.append(f"{clip_id}: npz/map object_urdf_path mismatch")
    if bad:
        raise SystemExit("[ERROR] AS bank validation failed:\n  " + "\n  ".join(bad[:30]))


def copy_external_urdf_bundle(
    object_urdf: str,
    object_mesh: str,
    object_size,
    *,
    source: Path,
    tmp_dir: Path,
    source_prefix: str,
    clip_id: str,
) -> tuple[str, str]:
    urdf_src = resolve_path(object_urdf, source)
    bundle_name = sanitize_name(f"{source_prefix}__{clip_id}")
    bundle_rel = Path("objects") / bundle_name
    bundle_dir = tmp_dir / bundle_rel
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    if not urdf_src.is_file():
        size = np.asarray(object_size if object_size is not None else [], dtype=np.float32).reshape(-1)
        if size.size != 3:
            raise SystemExit(f"[ERROR] Missing external URDF for {clip_id}: {urdf_src}")
        x, y, z = [max(float(v), 1.0e-4) for v in size.tolist()]
        hx, hy, hz = x / 2.0, y / 2.0, z / 2.0
        mesh_name = f"{bundle_name}.obj"
        urdf_name = f"{bundle_name}.urdf"
        (bundle_dir / mesh_name).write_text(
            "\n".join(
                [
                    f"v {-hx:.8g} {-hy:.8g} {-hz:.8g}",
                    f"v {hx:.8g} {-hy:.8g} {-hz:.8g}",
                    f"v {hx:.8g} {hy:.8g} {-hz:.8g}",
                    f"v {-hx:.8g} {hy:.8g} {-hz:.8g}",
                    f"v {-hx:.8g} {-hy:.8g} {hz:.8g}",
                    f"v {hx:.8g} {-hy:.8g} {hz:.8g}",
                    f"v {hx:.8g} {hy:.8g} {hz:.8g}",
                    f"v {-hx:.8g} {hy:.8g} {hz:.8g}",
                    "f 1 2 3 4",
                    "f 5 8 7 6",
                    "f 1 5 6 2",
                    "f 2 6 7 3",
                    "f 3 7 8 4",
                    "f 4 8 5 1",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        mass = 0.1
        ixx = mass * (y * y + z * z) / 12.0
        iyy = mass * (x * x + z * z) / 12.0
        izz = mass * (x * x + y * y) / 12.0
        (bundle_dir / urdf_name).write_text(
            f"""<?xml version="1.0" ?>
<robot name="{bundle_name}">
  <link name="{bundle_name}_link">
    <inertial>
      <mass value="{mass:.8g}"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="{ixx:.8g}" ixy="0" ixz="0" iyy="{iyy:.8g}" iyz="0" izz="{izz:.8g}"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="{mesh_name}" scale="1 1 1"/></geometry>
      <material name="mat"><color rgba="0.7 0.8 0.9 0.7"/></material>
    </visual>
    <collision name="{bundle_name}">
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <geometry><mesh filename="{mesh_name}" scale="1 1 1"/></geometry>
    </collision>
  </link>
</robot>
""",
            encoding="utf-8",
        )
        return str(bundle_rel / urdf_name), str(bundle_rel / mesh_name)

    try:
        tree = ET.parse(urdf_src)
        root = tree.getroot()
    except Exception as exc:
        raise SystemExit(f"[ERROR] Invalid external URDF for {clip_id}: {urdf_src}: {exc}") from exc

    copied_meshes: dict[Path, str] = {}
    first_mesh_rel = ""

    def copy_mesh(raw_filename: str) -> str:
        nonlocal first_mesh_rel
        mesh_src = resolve_path(raw_filename, urdf_src.parent)
        if not mesh_src.is_file():
            raise SystemExit(f"[ERROR] Missing mesh referenced by {urdf_src}: {mesh_src}")
        mesh_name = sanitize_name(mesh_src.name)
        if not mesh_name:
            mesh_name = "mesh.obj"
        base_name = mesh_name
        suffix = 1
        while (bundle_dir / mesh_name).exists() and copied_meshes.get(mesh_src) != mesh_name:
            stem = Path(base_name).stem
            ext = Path(base_name).suffix
            mesh_name = f"{stem}_{suffix}{ext}"
            suffix += 1
        if mesh_src not in copied_meshes:
            shutil.copy2(mesh_src, bundle_dir / mesh_name)
            copied_meshes[mesh_src] = mesh_name
        if not first_mesh_rel:
            first_mesh_rel = str(bundle_rel / copied_meshes[mesh_src])
        return copied_meshes[mesh_src]

    for tag in root.findall(".//mesh"):
        filename = str(tag.get("filename", "")).strip()
        if not filename:
            continue
        tag.set("filename", copy_mesh(filename))

    if object_mesh:
        mesh_src = resolve_path(object_mesh, source)
        if mesh_src.is_file():
            copied_name = copy_mesh(str(mesh_src))
            if not first_mesh_rel:
                first_mesh_rel = str(bundle_rel / copied_name)

    urdf_name = sanitize_name(urdf_src.name) or f"{bundle_name}.urdf"
    urdf_dst = bundle_dir / urdf_name
    tree.write(urdf_dst, encoding="utf-8", xml_declaration=True)
    return str(bundle_rel / urdf_name), first_mesh_rel


nfs_root = Path(sys.argv[1]).expanduser().resolve()
source_names = [item for item in sys.argv[2].split() if item]
local_root = Path(sys.argv[3]).expanduser()
output_bank_name = sys.argv[4]
dry_run = is_truthy(sys.argv[5])
dedupe_identical = is_truthy(sys.argv[6])

sources = []
source_searches: dict[Path, list[Path]] = {}
for name in source_names:
    candidates = resolve_source_candidates(name, nfs_root)
    source = next((candidate for candidate in candidates if candidate.is_dir()), candidates[0])
    sources.append(source)
    source_searches[source] = candidates
for source in sources:
    if not source.is_dir():
        searched = "\n  ".join(str(path) for path in source_searches.get(source, [source]))
        raise SystemExit(f"[ERROR] Missing source directory: {source}\n[ERROR] Searched candidates:\n  {searched}")
    if not (source / "_clip_object_urdf_map.json").is_file():
        raise SystemExit(f"[ERROR] Missing source object map: {source / '_clip_object_urdf_map.json'}")
    if not list(source.glob("*.npz")):
        raise SystemExit(f"[ERROR] No .npz clips in source: {source}")

out_dir = local_root / output_bank_name
tmp_parent = local_root
tmp_parent.mkdir(parents=True, exist_ok=True)
tmp_dir = Path(tempfile.mkdtemp(prefix=f".{output_bank_name}.tmp.", dir=tmp_parent))

clips_out: dict[str, dict] = {}
source_records: list[dict] = []
seen_by_name: dict[str, tuple[str, dict, str]] = {}
copied_object_dirs: set[str] = set()
deduped = 0
prefixed = 0

try:
    objects_out = tmp_dir / "objects"
    objects_out.mkdir(parents=True, exist_ok=True)

    for source in sources:
        source_name = source.name
        source_prefix = sanitize_name(source_name)
        source_meta, source_clips = load_clip_map(source / "_clip_object_urdf_map.json")
        npz_paths = sorted(source.glob("*.npz"))
        source_records.append(
            {
                "source": str(source),
                "clip_count": len(npz_paths),
                "map_count": len(source_clips),
                "metadata": source_meta,
            }
        )

        for npz_path in npz_paths:
            base_clip_id = npz_path.stem
            if base_clip_id not in source_clips:
                raise SystemExit(f"[ERROR] {source}: missing map entry for {base_clip_id}")
            npz_hash = sha256_file(npz_path)
            raw_entry = source_clips[base_clip_id]
            entry = dict(raw_entry) if isinstance(raw_entry, dict) else {"object_urdf_path": str(raw_entry)}
            dedupe_entry = dict(entry)
            previous = seen_by_name.get(base_clip_id)
            if dedupe_identical and previous is not None and previous[0] == npz_hash and previous[1] == dedupe_entry:
                deduped += 1
                continue

            clip_id = base_clip_id
            is_conflict = False
            if previous is not None:
                clip_id = f"{source_prefix}__{base_clip_id}"
                is_conflict = True
                prefixed += 1
                while clip_id in clips_out:
                    clip_id = f"{source_prefix}_{prefixed}__{base_clip_id}"

            object_urdf = str(entry.get("object_urdf_path", "")).strip()
            object_mesh = str(entry.get("object_mesh_path", "")).strip()
            if not object_urdf:
                raise SystemExit(f"[ERROR] {source}: empty object_urdf_path for {base_clip_id}")

            object_urdf_path_raw = Path(object_urdf).expanduser()
            object_mesh_path_raw = Path(object_mesh).expanduser() if object_mesh else None
            object_dir_rel = Path(object_urdf).parts[0:2]
            if (
                not object_urdf_path_raw.is_absolute()
                and not (object_mesh_path_raw is not None and object_mesh_path_raw.is_absolute())
                and len(object_dir_rel) >= 2
                and object_dir_rel[0] == "objects"
                and (source / object_dir_rel[0] / object_dir_rel[1]).is_dir()
            ):
                object_src_dir = source / object_dir_rel[0] / object_dir_rel[1]
                object_dst_name = f"{source_prefix}__{object_dir_rel[1]}" if is_conflict else object_dir_rel[1]
                object_dst_dir = tmp_dir / object_dir_rel[0] / object_dst_name
                if str(object_dst_dir) not in copied_object_dirs:
                    if object_dst_dir.exists():
                        shutil.rmtree(object_dst_dir)
                    shutil.copytree(object_src_dir, object_dst_dir, symlinks=False)
                    copied_object_dirs.add(str(object_dst_dir))
                if is_conflict:
                    object_urdf_parts = list(Path(object_urdf).parts)
                    object_urdf_parts[1] = object_dst_name
                    object_urdf = str(Path(*object_urdf_parts))
                    if object_mesh:
                        object_mesh_parts = list(Path(object_mesh).parts)
                        if len(object_mesh_parts) >= 2 and object_mesh_parts[0] == "objects":
                            object_mesh_parts[1] = object_dst_name
                            object_mesh = str(Path(*object_mesh_parts))
            elif object_urdf_path_raw.is_absolute() or (
                object_mesh_path_raw is not None and object_mesh_path_raw.is_absolute()
            ):
                object_urdf, copied_mesh = copy_external_urdf_bundle(
                    object_urdf,
                    object_mesh,
                    entry.get("object_size"),
                    source=source,
                    tmp_dir=tmp_dir,
                    source_prefix=source_prefix,
                    clip_id=clip_id,
                )
                object_mesh = copied_mesh or object_mesh
            else:
                for raw in (object_urdf, object_mesh):
                    if not raw:
                        continue
                    src_path = resolve_path(raw, source)
                    dst_path = tmp_dir / raw
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)

            object_name = str(entry.get("object_name", "")).strip() or clip_id
            entry["object_name"] = object_name if clip_id == base_clip_id else clip_id
            entry["object_urdf_path"] = object_urdf
            if object_mesh:
                entry["object_mesh_path"] = object_mesh
            entry["source_bank"] = source_name
            entry["source_clip_id"] = base_clip_id
            clips_out[clip_id] = entry
            source_object_name = str(raw_entry.get("object_name", "")).strip() if isinstance(raw_entry, dict) else ""
            source_object_urdf = str(raw_entry.get("object_urdf_path", "")).strip() if isinstance(raw_entry, dict) else ""
            source_object_mesh = str(raw_entry.get("object_mesh_path", "")).strip() if isinstance(raw_entry, dict) else ""
            needs_npz_rewrite = (
                clip_id != base_clip_id
                or source_object_name != entry["object_name"]
                or source_object_urdf != entry["object_urdf_path"]
                or source_object_mesh != str(entry.get("object_mesh_path", "")).strip()
            )
            if needs_npz_rewrite:
                rewrite_npz(npz_path, tmp_dir / f"{clip_id}.npz", clip_id, entry["object_name"], entry)
            else:
                shutil.copy2(npz_path, tmp_dir / f"{clip_id}.npz")
            seen_by_name[base_clip_id] = (npz_hash, dedupe_entry, source_name)

    if not clips_out:
        raise SystemExit("[ERROR] No clips selected for output bank")

    payload = {
        "clips": dict(sorted(clips_out.items())),
        "source_banks": source_records,
        "merge": {
            "source_policy": "dedupe-identical-prefix-conflicts",
            "deduped_duplicate_clips": deduped,
            "prefixed_conflicting_clips": prefixed,
            "output_clip_count": len(clips_out),
        },
        "notes": "Generated by cp_as.sh for train_as_general.sh.",
    }
    (tmp_dir / "_clip_object_urdf_map.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    (tmp_dir / "_pack_summary.json").write_text(
        json.dumps(payload["merge"], indent=2, sort_keys=True), encoding="utf-8"
    )
    validate_bank(tmp_dir, clips_out)

    print(f"[INFO] Built AS local union bank: {tmp_dir}")
    print(f"[INFO]   sources={len(sources)} output_clips={len(clips_out)} deduped={deduped} prefixed_conflicts={prefixed}")
    if dry_run:
        print(f"[DRY_RUN] Would replace: {out_dir}")
    else:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.parent.mkdir(parents=True, exist_ok=True)
        os.replace(tmp_dir, out_dir)
        tmp_dir = None
        print(f"[INFO] Installed AS bank: {out_dir}")
        print(f"[INFO] Use: AS_DATA_DIR={out_dir} bash train_as_general.sh")
finally:
    if tmp_dir is not None and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
PY
