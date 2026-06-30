#!/usr/bin/env bash
set -euo pipefail

# Copy the ds_as_data debug bank from NFS into this repo's local data tree.
#
# The installed layout is:
#   data/ds_as_data/debug/
#
# Usage:
#   bash cp_debug.sh
#
# Optional env:
#   NFS_DEBUG_BANK=/nfs/zzzihanw/ds_as_data/debug
#   EXPECTED_CLIP_COUNT=39
#   DRY_RUN=1
#   KEEP_BACKUP=1
#   RSYNC_INFO=stats2,progress2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

NFS_DEBUG_BANK=${NFS_DEBUG_BANK:-/nfs/zzzihanw/ds_as_data/debug}
EXPECTED_CLIP_COUNT=${EXPECTED_CLIP_COUNT:-39}
DRY_RUN=${DRY_RUN:-0}
KEEP_BACKUP=${KEEP_BACKUP:-1}
RSYNC_INFO=${RSYNC_INFO:-stats2}

LOCAL_DATA_ROOT="${SCRIPT_DIR}/data/ds_as_data"
LOCAL_BANK="${LOCAL_DATA_ROOT}/debug"
STAMP=$(date +%Y%m%d_%H%M%S)
TMP_BANK="${LOCAL_DATA_ROOT}/.debug.tmp.${STAMP}.$$"
BACKUP_BANK="${LOCAL_BANK}.bak.${STAMP}"

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ERROR] rsync not found in PATH." >&2
  exit 1
fi

NFS_DEBUG_BANK_ABS=$(
  python3 - "${NFS_DEBUG_BANK}" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)

EXPECTED_LOCAL_DATA_ROOT=$(
  python3 - "${SCRIPT_DIR}/data/ds_as_data" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)
LOCAL_DATA_ROOT_ABS=$(
  python3 - "${LOCAL_DATA_ROOT}" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)

if [[ "${LOCAL_DATA_ROOT_ABS}" != "${EXPECTED_LOCAL_DATA_ROOT}" ]]; then
  echo "[ERROR] Refusing unexpected LOCAL_DATA_ROOT: ${LOCAL_DATA_ROOT_ABS}" >&2
  exit 2
fi
if [[ "${LOCAL_BANK}" == "/" || "${LOCAL_BANK}" == "${SCRIPT_DIR}" || "${LOCAL_BANK}" == "${SCRIPT_DIR}/data" ]]; then
  echo "[ERROR] Unsafe local target: ${LOCAL_BANK}" >&2
  exit 2
fi
if [[ ! -d "${NFS_DEBUG_BANK_ABS}" ]]; then
  echo "[ERROR] Missing NFS debug bank: ${NFS_DEBUG_BANK_ABS}" >&2
  exit 1
fi
if [[ ! -f "${NFS_DEBUG_BANK_ABS}/_clip_object_urdf_map.json" ]]; then
  echo "[ERROR] Missing source object map: ${NFS_DEBUG_BANK_ABS}/_clip_object_urdf_map.json" >&2
  exit 1
fi
if [[ ! -f "${NFS_DEBUG_BANK_ABS}/_pack_summary.json" ]]; then
  echo "[ERROR] Missing source pack summary: ${NFS_DEBUG_BANK_ABS}/_pack_summary.json" >&2
  exit 1
fi

echo "[INFO] source=${NFS_DEBUG_BANK_ABS}"
echo "[INFO] target=${LOCAL_BANK}"
echo "[INFO] expected_clips=${EXPECTED_CLIP_COUNT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY-RUN] Would rsync source into staging dir:"
  echo "          ${NFS_DEBUG_BANK_ABS}/ -> ${TMP_BANK}/"
  echo "[DRY-RUN] Would validate and install as ${LOCAL_BANK}"
  exit 0
fi

cleanup_tmp() {
  if [[ -d "${TMP_BANK}" ]]; then
    rm -rf "${TMP_BANK}"
  fi
}
trap cleanup_tmp EXIT

mkdir -p "${LOCAL_DATA_ROOT}"
rm -rf "${TMP_BANK}"
mkdir -p "${TMP_BANK}"

echo "[INFO] Copying NFS debug bank into staging dir..."
rsync -aL --delete --human-readable --info="${RSYNC_INFO}" \
  "${NFS_DEBUG_BANK_ABS}/" "${TMP_BANK}/"

echo "[INFO] Validating staged debug bank..."
python3 - "${NFS_DEBUG_BANK_ABS}" "${TMP_BANK}" "${LOCAL_BANK}" "${EXPECTED_CLIP_COUNT}" <<'PY'
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


source_bank = Path(sys.argv[1]).expanduser().resolve()
staging_bank = Path(sys.argv[2]).expanduser().resolve()
installed_bank = Path(sys.argv[3]).expanduser().resolve()
expected_clip_count = int(sys.argv[4])

map_path = staging_bank / "_clip_object_urdf_map.json"
summary_path = staging_bank / "_pack_summary.json"


def scalar_str(value: object) -> str:
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
    if hasattr(item, "item"):
        item = item.item()
    return str(item).strip()


def load_clips(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        return payload, payload["clips"]
    if isinstance(payload, dict):
        return {}, payload
    raise SystemExit(f"[ERROR] Invalid clip-object map: {path}")


def resolve_bank_path(raw_path: str, bank: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (bank / path).resolve()


def require_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] {label} escapes bank root: {path}") from exc


def count_files(root: Path) -> int:
    return sum(1 for path in root.rglob("*") if path.is_file())


summary = json.loads(summary_path.read_text(encoding="utf-8"))
packed = summary.get("packed", [])
packed_count = int(summary.get("packed_count", len(packed)))
failed_count = int(summary.get("failed_count", 0))
if packed_count != expected_clip_count:
    raise SystemExit(f"[ERROR] packed_count={packed_count}, expected {expected_clip_count}")
if failed_count != 0:
    raise SystemExit(f"[ERROR] failed_count={failed_count}, expected 0")

map_payload, clips = load_clips(map_path)
if len(clips) != expected_clip_count:
    raise SystemExit(f"[ERROR] object-map entries={len(clips)}, expected {expected_clip_count}")

npz_paths = sorted(staging_bank.glob("*.npz"))
if len(npz_paths) != expected_clip_count:
    raise SystemExit(f"[ERROR] .npz clips={len(npz_paths)}, expected {expected_clip_count}")

npz_ids = {path.stem for path in npz_paths}
map_ids = set(clips)
if npz_ids != map_ids:
    missing_map = sorted(npz_ids - map_ids)[:10]
    missing_npz = sorted(map_ids - npz_ids)[:10]
    raise SystemExit(
        "[ERROR] Clip ids differ between .npz files and object map; "
        f"missing_map={missing_map} missing_npz={missing_npz}"
    )

required_arrays = [
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
]

updated_clips: dict[str, dict] = {}
rewritten_npz = 0
prefix_counts = {"scaledown": 0, "unscale": 0}

for npz_path in npz_paths:
    clip_id = npz_path.stem
    if clip_id.startswith("scaledown__"):
        prefix_counts["scaledown"] += 1
    elif clip_id.startswith("unscale__"):
        prefix_counts["unscale"] += 1

    raw_entry = clips[clip_id]
    if isinstance(raw_entry, str):
        entry = {"object_urdf_path": raw_entry}
    elif isinstance(raw_entry, dict):
        entry = dict(raw_entry)
    else:
        raise SystemExit(f"[ERROR] Invalid object-map entry for {clip_id}: {type(raw_entry).__name__}")

    urdf_raw = str(entry.get("object_urdf_path", "")).strip()
    mesh_raw = str(entry.get("object_mesh_path", "")).strip()
    if not urdf_raw:
        raise SystemExit(f"[ERROR] Missing object_urdf_path for {clip_id}")
    if not mesh_raw:
        raise SystemExit(f"[ERROR] Missing object_mesh_path for {clip_id}")
    if Path(urdf_raw).is_absolute() or Path(mesh_raw).is_absolute():
        raise SystemExit(f"[ERROR] Expected bank-relative object paths for {clip_id}")

    urdf_path = resolve_bank_path(urdf_raw, staging_bank)
    mesh_path = resolve_bank_path(mesh_raw, staging_bank)
    require_inside(urdf_path, staging_bank, f"{clip_id} object_urdf_path")
    require_inside(mesh_path, staging_bank, f"{clip_id} object_mesh_path")
    if not urdf_path.is_file():
        raise SystemExit(f"[ERROR] Missing URDF for {clip_id}: {urdf_path}")
    if not mesh_path.is_file():
        raise SystemExit(f"[ERROR] Missing mesh for {clip_id}: {mesh_path}")

    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        raise SystemExit(f"[ERROR] Invalid URDF for {clip_id}: {urdf_path}: {exc}") from exc
    for mesh_tag in root.findall(".//mesh"):
        filename = str(mesh_tag.get("filename", "")).strip()
        if not filename:
            raise SystemExit(f"[ERROR] Empty mesh filename in URDF for {clip_id}: {urdf_path}")
        ref = Path(filename).expanduser()
        ref_path = ref.resolve() if ref.is_absolute() else (urdf_path.parent / ref).resolve()
        require_inside(ref_path, staging_bank, f"{clip_id} URDF mesh")
        if not ref_path.is_file():
            raise SystemExit(f"[ERROR] Missing URDF mesh for {clip_id}: {ref_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}

    missing_fields = [key for key in required_arrays if key not in payload]
    if missing_fields:
        raise SystemExit(f"[ERROR] {clip_id}: missing required arrays {missing_fields}")

    frame_count = int(payload["joint_pos"].shape[0])
    if frame_count <= 0:
        raise SystemExit(f"[ERROR] {clip_id}: empty joint_pos")
    for key in required_arrays:
        arr = payload[key]
        if arr.shape[0] != frame_count:
            raise SystemExit(f"[ERROR] {clip_id}: {key}.shape[0]={arr.shape[0]}, expected {frame_count}")
        if not np.issubdtype(arr.dtype, np.number):
            raise SystemExit(f"[ERROR] {clip_id}: {key} is non-numeric dtype={arr.dtype}")
        if not np.isfinite(arr).all():
            raise SystemExit(f"[ERROR] {clip_id}: {key} contains non-finite values")

    npz_urdf = scalar_str(payload.get("object_urdf_path"))
    npz_mesh = scalar_str(payload.get("object_mesh_path"))
    changed = False
    if npz_urdf != urdf_raw:
        payload["object_urdf_path"] = np.asarray(urdf_raw)
        changed = True
    if npz_mesh != mesh_raw:
        payload["object_mesh_path"] = np.asarray(mesh_raw)
        changed = True
    if "object_name" not in payload or not scalar_str(payload["object_name"]):
        payload["object_name"] = np.asarray(str(entry.get("object_name") or clip_id))
        changed = True
    if changed:
        tmp_npz = npz_path.with_name(f".{npz_path.name}.rewriting.npz")
        if tmp_npz.exists():
            tmp_npz.unlink()
        try:
            np.savez_compressed(tmp_npz, **payload)
            os.replace(tmp_npz, npz_path)
        finally:
            if tmp_npz.exists():
                tmp_npz.unlink()
        rewritten_npz += 1

    entry["object_name"] = str(entry.get("object_name") or scalar_str(payload.get("object_name")) or clip_id)
    entry["object_urdf_path"] = urdf_raw
    entry["object_mesh_path"] = mesh_raw
    updated_clips[clip_id] = entry

out_payload = dict(map_payload)
out_payload["clips"] = updated_clips
map_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

source_files = count_files(source_bank)
staging_files = count_files(staging_bank)
if source_files != staging_files:
    raise SystemExit(f"[ERROR] File count mismatch: source={source_files} staging={staging_files}")

total_size = sum(path.stat().st_size for path in staging_bank.rglob("*") if path.is_file())
manifest = {
    "source_bank": str(source_bank),
    "installed_bank": str(installed_bank),
    "clips": len(updated_clips),
    "motion_npz": len(npz_paths),
    "packed_count": packed_count,
    "failed_count": failed_count,
    "files": staging_files,
    "bytes": total_size,
    "scaledown": prefix_counts["scaledown"],
    "unscale": prefix_counts["unscale"],
    "rewritten_npz": rewritten_npz,
}
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

if [[ -e "${LOCAL_BANK}" ]]; then
  if [[ "${KEEP_BACKUP}" == "1" ]]; then
    echo "[INFO] Moving existing target to backup: ${BACKUP_BANK}"
    mv "${LOCAL_BANK}" "${BACKUP_BANK}"
  else
    echo "[INFO] Removing existing target: ${LOCAL_BANK}"
    rm -rf "${LOCAL_BANK}"
  fi
fi

echo "[INFO] Installing staged debug bank."
mv "${TMP_BANK}" "${LOCAL_BANK}"
trap - EXIT

echo "[INFO] Done."
echo "[INFO] local_bank=${LOCAL_BANK}"
if [[ -d "${BACKUP_BANK}" ]]; then
  echo "[INFO] backup=${BACKUP_BANK}"
fi
