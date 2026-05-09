#!/usr/bin/env bash
set -euo pipefail

# Copy and normalize the AS/OMOMO real-mesh training bank from NFS.
#
# This script builds a local concatenated bank at data/ds_as_data/omomo for
# train_as_general.sh. Same-named clips from later sources are kept with a
# source-name prefix so all listed sources train together.
#
# Usage:
#   bash cp_as.sh
#
# Optional env:
#   NFS_AS_ROOT=/nfs/zzzihanw/ds_as_data
#   NFS_AS_SOURCES="omomo_45 retarget_vanilla_w_obj_scale_coacd500_curated18_20260509"
#   LOCAL_AS_ROOT=data/ds_as_data
#   OUTPUT_BANK_NAME=omomo
#   DEDUPE_IDENTICAL=0
#   DRY_RUN=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

NFS_AS_ROOT=${NFS_AS_ROOT:-/nfs/zzzihanw/ds_as_data}
NFS_AS_SOURCES=${NFS_AS_SOURCES:-"omomo_45 retarget_vanilla_w_obj_scale_coacd500_curated18_20260509"}
LOCAL_AS_ROOT=${LOCAL_AS_ROOT:-"data/ds_as_data"}
OUTPUT_BANK_NAME=${OUTPUT_BANK_NAME:-omomo}
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


nfs_root = Path(sys.argv[1]).expanduser().resolve()
source_names = [item for item in sys.argv[2].split() if item]
local_root = Path(sys.argv[3]).expanduser()
output_bank_name = sys.argv[4]
dry_run = is_truthy(sys.argv[5])
dedupe_identical = is_truthy(sys.argv[6])

sources = []
for name in source_names:
    path = Path(name).expanduser()
    sources.append(path.resolve() if path.is_absolute() else (nfs_root / path).resolve())
for source in sources:
    if not source.is_dir():
        raise SystemExit(f"[ERROR] Missing source directory: {source}")
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

            object_dir_rel = Path(object_urdf).parts[0:2]
            if (
                len(object_dir_rel) >= 2
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
            rewrite_npz(npz_path, tmp_dir / f"{clip_id}.npz", clip_id, entry["object_name"], entry)
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
        print(f"[INFO] Use: OMOMO_DATA_DIR={out_dir} bash train_as_general.sh")
finally:
    if tmp_dir is not None and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
PY
