#!/usr/bin/env bash
set -euo pipefail

# Copy a CoRL AS distillation bank from NFS into this repo's relative
# data/ds_as_data tree.
#
# The NFS package can be either a directory or a .tar archive. It is
# self-contained, but its object map may point at NFS or source-machine URDF
# paths. This script rewrites the object map and per-clip npz object_urdf_path
# values to the repo-local copy so training can run without reading object
# assets from NFS.
#
# Usage:
#   bash cp_corl.sh
#
# Optional env:
#   CORL_BANK_NAME=corl_128
#   NFS_CORL_BANK=/nfs/zzzihanw/ds_as_data/_distill/<bank>[.tar]
#   LOCAL_DATA_ROOT=data/ds_as_data
#   LOCAL_BANK_NAME=<bank>
#   EXPECTED_CLIP_COUNT=128
#   DRY_RUN=1
#   KEEP_BACKUP=0
#   SEED_LOCAL_EXISTING=0
#     Default 1. If the same local bank already exists, seed staging from it
#     with symlinks dereferenced before overlaying the NFS package. This avoids
#     recopying large motion/object files from NFS when the local symlink bank
#     already points at the same payload.
#   SEED_LOCAL_OBJECT_ASSETS=0
#     Default 1. If the existing local bank's map points to local source URDFs,
#     pre-copy those URDFs and meshes into staging before the NFS overlay.
#   RSYNC_INFO=stats2,progress2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

DEFAULT_BANK_NAME=${CORL_BANK_NAME:-"corl_128"}
DEFAULT_NFS_BANK="/nfs/zzzihanw/ds_as_data/_distill/${DEFAULT_BANK_NAME}"
DEFAULT_NFS_TAR="${DEFAULT_NFS_BANK}.tar"

if [[ -z "${NFS_CORL_BANK:-}" ]]; then
  if [[ -f "${DEFAULT_NFS_TAR}" ]]; then
    NFS_CORL_BANK="${DEFAULT_NFS_TAR}"
  else
    NFS_CORL_BANK="${DEFAULT_NFS_BANK}"
  fi
fi
LOCAL_DATA_ROOT=${LOCAL_DATA_ROOT:-"data/ds_as_data"}
if [[ -z "${LOCAL_BANK_NAME:-}" ]]; then
  LOCAL_BANK_NAME="$(basename "${NFS_CORL_BANK}")"
  LOCAL_BANK_NAME="${LOCAL_BANK_NAME%.tar}"
fi
EXPECTED_CLIP_COUNT=${EXPECTED_CLIP_COUNT:-}
DRY_RUN=${DRY_RUN:-0}
KEEP_BACKUP=${KEEP_BACKUP:-1}
SEED_LOCAL_EXISTING=${SEED_LOCAL_EXISTING:-1}
SEED_LOCAL_OBJECT_ASSETS=${SEED_LOCAL_OBJECT_ASSETS:-1}
RSYNC_INFO=${RSYNC_INFO:-stats2}

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ERROR] rsync not found in PATH." >&2
  exit 1
fi

if [[ "${LOCAL_BANK_NAME}" == "" || "${LOCAL_BANK_NAME}" == "." || "${LOCAL_BANK_NAME}" == ".." || "${LOCAL_BANK_NAME}" == */* ]]; then
  echo "[ERROR] Unsafe LOCAL_BANK_NAME: ${LOCAL_BANK_NAME}" >&2
  exit 2
fi

NFS_CORL_BANK_ABS=$(
  python3 - "${NFS_CORL_BANK}" <<'PY'
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
EXPECTED_LOCAL_ROOT=$(
  python3 - "${SCRIPT_DIR}/data/ds_as_data" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)

if [[ "${LOCAL_DATA_ROOT_ABS}" != "${EXPECTED_LOCAL_ROOT}" ]]; then
  echo "[ERROR] Refusing unexpected LOCAL_DATA_ROOT: ${LOCAL_DATA_ROOT_ABS}" >&2
  echo "[ERROR] Expected repo-relative data root: ${EXPECTED_LOCAL_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${NFS_CORL_BANK_ABS}" && ! -f "${NFS_CORL_BANK_ABS}" ]]; then
  echo "[ERROR] NFS_CORL_BANK does not exist: ${NFS_CORL_BANK_ABS}" >&2
  exit 1
fi
if [[ -d "${NFS_CORL_BANK_ABS}" && ! -f "${NFS_CORL_BANK_ABS}/_clip_object_urdf_map.json" ]]; then
  echo "[ERROR] Missing object map in NFS bank: ${NFS_CORL_BANK_ABS}/_clip_object_urdf_map.json" >&2
  exit 1
fi
if [[ -f "${NFS_CORL_BANK_ABS}" && "${NFS_CORL_BANK_ABS}" != *.tar ]]; then
  echo "[ERROR] File NFS_CORL_BANK must be a .tar archive: ${NFS_CORL_BANK_ABS}" >&2
  exit 1
fi

LOCAL_BANK_ABS="${LOCAL_DATA_ROOT_ABS}/${LOCAL_BANK_NAME}"
STAMP=$(date +%Y%m%d_%H%M%S)
TMP_BANK_ABS="${LOCAL_DATA_ROOT_ABS}/.${LOCAL_BANK_NAME}.tmp.${STAMP}.$$"
BACKUP_BANK_ABS="${LOCAL_BANK_ABS}.bak.${STAMP}"

echo "[INFO] source=${NFS_CORL_BANK_ABS}"
echo "[INFO] target=${LOCAL_BANK_ABS}"

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ -f "${NFS_CORL_BANK_ABS}" ]]; then
    echo "[DRY-RUN] Would extract NFS tar package into staging dir:"
    echo "          ${NFS_CORL_BANK_ABS} -> ${TMP_BANK_ABS}/"
  else
    echo "[DRY-RUN] Would rsync NFS bank into staging dir:"
    echo "          ${NFS_CORL_BANK_ABS}/ -> ${TMP_BANK_ABS}/"
  fi
  echo "[DRY-RUN] Would rewrite local object paths and replace target."
  exit 0
fi

cleanup_tmp() {
  if [[ -d "${TMP_BANK_ABS}" ]]; then
    rm -rf "${TMP_BANK_ABS}"
  fi
}
trap cleanup_tmp EXIT

mkdir -p "${LOCAL_DATA_ROOT_ABS}"
rm -rf "${TMP_BANK_ABS}"
mkdir -p "${TMP_BANK_ABS}"

echo "[INFO] Copying NFS package into staging dir..."
if [[ -f "${NFS_CORL_BANK_ABS}" ]]; then
  echo "[INFO] Extracting NFS tar package..."
  tar -xf "${NFS_CORL_BANK_ABS}" -C "${TMP_BANK_ABS}" --strip-components=1
elif [[ "${SEED_LOCAL_EXISTING}" == "1" && -d "${LOCAL_BANK_ABS}" ]]; then
  echo "[INFO] Seeding staging dir from existing local bank with symlinks dereferenced..."
  rsync -aL --delete --human-readable --info="${RSYNC_INFO}" \
    --exclude="/contact_export_from_teacher_success133_final0p5/" \
    "${LOCAL_BANK_ABS}/" "${TMP_BANK_ABS}/"
  if [[ "${SEED_LOCAL_OBJECT_ASSETS}" == "1" && -f "${LOCAL_BANK_ABS}/_clip_object_urdf_map.json" ]]; then
    echo "[INFO] Seeding object URDF/mesh assets from existing local map..."
    python3 - "${NFS_CORL_BANK_ABS}" "${LOCAL_BANK_ABS}" "${TMP_BANK_ABS}" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def load_clips(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        clips = payload["clips"]
    elif isinstance(payload, dict):
        clips = payload
    else:
        raise SystemExit(f"[ERROR] Invalid object map: {path}")
    if not isinstance(clips, dict):
        raise SystemExit(f"[ERROR] Invalid clips payload: {path}")
    return clips


nfs_bank = Path(sys.argv[1]).expanduser().resolve()
local_bank = Path(sys.argv[2]).expanduser().resolve()
tmp_bank = Path(sys.argv[3]).expanduser().resolve()
nfs_clips = load_clips(nfs_bank / "_clip_object_urdf_map.json")
local_clips = load_clips(local_bank / "_clip_object_urdf_map.json")
urdf_dir = tmp_bank / "_single_slot_motion_bank" / "_single_slot_urdfs"
target_root = tmp_bank.resolve()
urdf_dir.mkdir(parents=True, exist_ok=True)

copied_urdfs = 0
copied_meshes = 0
missing: list[str] = []
for clip_id, nfs_entry in sorted(nfs_clips.items()):
    if not isinstance(nfs_entry, dict):
        missing.append(f"{clip_id}: invalid NFS map entry")
        continue
    local_entry = local_clips.get(clip_id)
    if not isinstance(local_entry, dict):
        missing.append(f"{clip_id}: missing local map entry")
        continue

    nfs_urdf_name = Path(str(nfs_entry.get("object_urdf_path", "")).strip()).name
    local_urdf = Path(str(local_entry.get("object_urdf_path", "")).strip()).expanduser()
    if not local_urdf.is_absolute():
        local_urdf = (local_bank / local_urdf).resolve()
    if not nfs_urdf_name:
        nfs_urdf_name = local_urdf.name
    if not local_urdf.is_file():
        missing.append(f"{clip_id}: missing local URDF {local_urdf}")
        continue

    target_urdf = urdf_dir / nfs_urdf_name
    if not target_urdf.exists() or target_urdf.stat().st_size != local_urdf.stat().st_size:
        shutil.copy2(local_urdf, target_urdf)
        copied_urdfs += 1

    try:
        root = ET.parse(local_urdf).getroot()
    except Exception as exc:
        missing.append(f"{clip_id}: invalid local URDF {local_urdf}: {exc}")
        continue

    for mesh_tag in root.findall(".//mesh"):
        mesh_ref = str(mesh_tag.get("filename", "")).strip()
        if not mesh_ref:
            missing.append(f"{clip_id}: empty mesh filename in {local_urdf}")
            continue
        src_mesh = Path(mesh_ref).expanduser()
        if not src_mesh.is_absolute():
            src_mesh = (local_urdf.parent / src_mesh).resolve()
        if not src_mesh.is_file():
            missing.append(f"{clip_id}: missing mesh {src_mesh}")
            continue
        dst_mesh = (target_urdf.parent / mesh_ref).resolve()
        try:
            dst_mesh.relative_to(target_root)
        except ValueError:
            missing.append(f"{clip_id}: mesh destination escapes staging {dst_mesh}")
            continue
        dst_mesh.parent.mkdir(parents=True, exist_ok=True)
        if not dst_mesh.exists() or dst_mesh.stat().st_size != src_mesh.stat().st_size:
            shutil.copy2(src_mesh, dst_mesh)
            copied_meshes += 1

if missing:
    preview = "\n  ".join(missing[:20])
    print(f"[WARN] Some local object assets could not be seeded:\n  {preview}", flush=True)

print(f"[INFO] Seeded local object assets: urdfs_copied={copied_urdfs} meshes_copied={copied_meshes}", flush=True)
PY
  fi
  echo "[INFO] Overlaying NFS package; same-sized seeded files are reused."
  rsync -aL --delete --size-only --human-readable --info="${RSYNC_INFO}" \
    "${NFS_CORL_BANK_ABS}/" "${TMP_BANK_ABS}/"
else
  rsync -aL --delete --human-readable --info="${RSYNC_INFO}" \
    "${NFS_CORL_BANK_ABS}/" "${TMP_BANK_ABS}/"
fi

echo "[INFO] Rewriting object paths to repo-local data dir..."
python3 - "${TMP_BANK_ABS}" "${LOCAL_BANK_ABS}" "${EXPECTED_CLIP_COUNT}" <<'PY'
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


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


def write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(f".{path.name}.rewriting.npz")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


staging_bank = Path(sys.argv[1]).expanduser().resolve()
installed_bank = Path(sys.argv[2]).expanduser().resolve()
expected_count_raw = sys.argv[3].strip()
map_path = staging_bank / "_clip_object_urdf_map.json"
contact_dir = staging_bank / "contact_export_from_teacher_success133_final0p5" / "clips"
slot_bank = staging_bank / "_single_slot_motion_bank"
slot_map_path = slot_bank / "_clip_object_urdf_map.json"

if expected_count_raw:
    try:
        expected_count = int(expected_count_raw)
    except ValueError as exc:
        raise SystemExit(f"[ERROR] EXPECTED_CLIP_COUNT must be an integer. Got: {expected_count_raw}") from exc
else:
    expected_count = 0


def find_staging_urdf(old_urdf: Path) -> Path:
    urdf_name = old_urdf.name
    preferred = [
        staging_bank / "_single_slot_motion_bank_teacher_export_20260520_105947" / "_single_slot_urdfs" / urdf_name,
        staging_bank / "_single_slot_motion_bank" / "_single_slot_urdfs" / urdf_name,
    ]
    for candidate in preferred:
        if candidate.is_file():
            return candidate.resolve()
    matches = sorted(
        path.resolve()
        for path in staging_bank.rglob(urdf_name)
        if path.is_file() and path.suffix == ".urdf"
    )
    if not matches:
        raise SystemExit(f"[ERROR] Missing staging URDF named {urdf_name} under {staging_bank}")
    if len(matches) > 1:
        preview = ", ".join(str(path) for path in matches[:5])
        raise SystemExit(f"[ERROR] Ambiguous staging URDF for {urdf_name}: {preview}")
    return matches[0]

payload = json.loads(map_path.read_text(encoding="utf-8"))
if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
    clips = payload["clips"]
    out_payload = {key: value for key, value in payload.items() if key != "clips"}
elif isinstance(payload, dict):
    clips = payload
    out_payload = {}
else:
    raise SystemExit(f"[ERROR] Invalid object map: {map_path}")

if expected_count and len(clips) != expected_count:
    raise SystemExit(f"[ERROR] Expected {expected_count} clips, got {len(clips)}")
expected_count = len(clips)

updated_clips: dict[str, dict] = {}
rewritten_npz = 0
for clip_id, raw_entry in sorted(clips.items()):
    if isinstance(raw_entry, str):
        entry = {"object_urdf_path": raw_entry}
    elif isinstance(raw_entry, dict):
        entry = dict(raw_entry)
    else:
        raise SystemExit(f"[ERROR] Invalid map entry for {clip_id}: {type(raw_entry).__name__}")

    old_urdf = Path(str(entry.get("object_urdf_path", "")).strip())
    if not old_urdf.name:
        old_urdf = Path(f"{clip_id}.urdf")
    staging_urdf = find_staging_urdf(old_urdf)
    installed_urdf = (installed_bank / staging_urdf.relative_to(staging_bank)).resolve()
    npz_path = staging_bank / f"{clip_id}.npz"

    if not npz_path.is_file():
        raise SystemExit(f"[ERROR] Missing motion npz for {clip_id}: {npz_path}")

    entry["object_urdf_path"] = str(installed_urdf)
    if not str(entry.get("object_name", "")).strip():
        entry["object_name"] = clip_id
    updated_clips[clip_id] = entry

    npz_paths = [npz_path]
    slot_npz_path = slot_bank / f"{clip_id}.npz"
    if slot_npz_path.is_file():
        npz_paths.append(slot_npz_path)
    for candidate_npz in npz_paths:
        with np.load(candidate_npz, allow_pickle=True) as data:
            npz_payload = {key: np.asarray(data[key]) for key in data.files}
        old_npz_urdf = scalar_str(npz_payload.get("object_urdf_path"))
        if old_npz_urdf != str(installed_urdf):
            npz_payload["object_urdf_path"] = np.asarray(str(installed_urdf))
            if "object_name" not in npz_payload or not scalar_str(npz_payload.get("object_name")):
                npz_payload["object_name"] = np.asarray(str(entry["object_name"]))
            write_npz_atomic(candidate_npz, npz_payload)
            rewritten_npz += 1

out_payload["clips"] = updated_clips
map_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

npz_count = len(list(staging_bank.glob("*.npz")))
slot_npz_count = len(list(slot_bank.glob("*.npz"))) if slot_bank.is_dir() else 0
ref_count = len(list(contact_dir.glob("*/teacher_rollout_reference.npz")))
left_count = len(list(contact_dir.glob("*/left_wrist_contact_points.npy")))
right_count = len(list(contact_dir.glob("*/right_wrist_contact_points.npy")))
urdf_count = len(list(staging_bank.rglob("*.urdf")))
if slot_map_path.is_file():
    slot_map_path.write_text(json.dumps(out_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
symlink_count = 0
for root, dirs, files in os.walk(staging_bank):
    for name in dirs + files:
        if os.path.islink(os.path.join(root, name)):
            symlink_count += 1

errors: list[str] = []
for clip_id, entry in updated_clips.items():
    installed_urdf = Path(str(entry["object_urdf_path"]))
    urdf = staging_bank / installed_urdf.relative_to(installed_bank)
    if not urdf.is_file():
        errors.append(f"{clip_id}: missing staging URDF {urdf}")
        continue
    try:
        root = ET.parse(urdf).getroot()
    except Exception as exc:
        errors.append(f"{clip_id}: invalid URDF {urdf}: {exc}")
        continue
    for mesh in root.findall(".//mesh"):
        filename = str(mesh.get("filename", "")).strip()
        if not filename:
            errors.append(f"{clip_id}: empty mesh filename in {urdf}")
            continue
        mesh_path = Path(filename)
        if not mesh_path.is_absolute():
            mesh_path = (urdf.parent / mesh_path).resolve()
        if not mesh_path.is_file():
            errors.append(f"{clip_id}: missing URDF mesh {mesh_path}")

summary = {
    "bank": str(installed_bank),
    "clips": len(updated_clips),
    "motion_npz": npz_count,
    "single_slot_motion_npz": slot_npz_count,
    "urdfs": urdf_count,
    "teacher_rollout_reference": ref_count,
    "left_wrist_contact_points": left_count,
    "right_wrist_contact_points": right_count,
    "rewritten_npz": rewritten_npz,
    "symlinks": symlink_count,
}
(staging_bank / "cp_corl_local_manifest.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, indent=2, sort_keys=True))

expected = {
    "clips": len(updated_clips),
    "motion_npz": npz_count,
    "urdfs": urdf_count,
    "teacher_rollout_reference": ref_count,
    "left_wrist_contact_points": left_count,
    "right_wrist_contact_points": right_count,
}
for key, value in expected.items():
    if key == "urdfs":
        if value < expected_count:
            errors.append(f"{key}={value}")
    elif value != expected_count:
        errors.append(f"{key}={value}")
if slot_bank.is_dir() and slot_npz_count != expected_count:
    errors.append(f"single_slot_motion_npz={slot_npz_count}")
if symlink_count != 0:
    errors.append(f"symlinks={symlink_count}")
if errors:
    preview = "\n  ".join(errors[:20])
    raise SystemExit(f"[ERROR] Local CoRL bank validation failed:\n  {preview}")
PY

if [[ -e "${LOCAL_BANK_ABS}" ]]; then
  if [[ "${KEEP_BACKUP}" == "1" ]]; then
    echo "[INFO] Moving existing target to backup: ${BACKUP_BANK_ABS}"
    mv "${LOCAL_BANK_ABS}" "${BACKUP_BANK_ABS}"
  else
    echo "[INFO] Removing existing target: ${LOCAL_BANK_ABS}"
    rm -rf "${LOCAL_BANK_ABS}"
  fi
fi

echo "[INFO] Installing staged bank."
mv "${TMP_BANK_ABS}" "${LOCAL_BANK_ABS}"
trap - EXIT

echo "[INFO] Done."
echo "[INFO] local_bank=${LOCAL_BANK_ABS}"
if [[ -d "${BACKUP_BANK_ABS}" ]]; then
  echo "[INFO] backup=${BACKUP_BANK_ABS}"
fi
