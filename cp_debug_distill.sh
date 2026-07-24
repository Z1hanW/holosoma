#!/usr/bin/env bash
set -euo pipefail

# Copy the 29-clip debug teacher-rollout distillation bank from NFS into this
# repo's local data tree.
#
# Installed layout:
#   data/ds_as_data/debug_teacher_rollout_u8udzw0u_model05000_20260626_191753/
#
# Usage:
#   bash cp_debug_distill.sh
#
# Optional env:
#   NFS_DEBUG_DISTILL_ARCHIVE=/nfs/zzzihanw/ds_as_data/debug_teacher_rollout_u8udzw0u_model05000_20260626_191753.tar
#   NFS_DEBUG_DISTILL_BANK=/nfs/zzzihanw/ds_as_data/debug_teacher_rollout_u8udzw0u_model05000_20260626_191753
#   OUTPUT_BANK_NAME=debug_teacher_rollout_u8udzw0u_model05000_20260626_191753
#   CONTACT_EXPORT_NAME=contact_export_from_teacher_model05000
#   EXPECTED_CLIP_COUNT=29
#   DEBUG_OBJECTS_DIR=data/ds_as_data/debug/objects
#   DRY_RUN=1
#   KEEP_BACKUP=1
#   RSYNC_INFO=stats2,progress2

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
cd "${SCRIPT_DIR}"

DEFAULT_BANK_NAME=debug_teacher_rollout_u8udzw0u_model05000_20260626_191753
NFS_DEBUG_DISTILL_ARCHIVE=${NFS_DEBUG_DISTILL_ARCHIVE:-/nfs/zzzihanw/ds_as_data/${DEFAULT_BANK_NAME}.tar}
NFS_DEBUG_DISTILL_BANK=${NFS_DEBUG_DISTILL_BANK:-/nfs/zzzihanw/ds_as_data/${DEFAULT_BANK_NAME}}
OUTPUT_BANK_NAME=${OUTPUT_BANK_NAME:-${DEFAULT_BANK_NAME}}
CONTACT_EXPORT_NAME=${CONTACT_EXPORT_NAME:-contact_export_from_teacher_model05000}
EXPECTED_CLIP_COUNT=${EXPECTED_CLIP_COUNT:-29}
DEBUG_OBJECTS_DIR=${DEBUG_OBJECTS_DIR:-${SCRIPT_DIR}/data/ds_as_data/debug/objects}
DRY_RUN=${DRY_RUN:-0}
KEEP_BACKUP=${KEEP_BACKUP:-1}
RSYNC_INFO=${RSYNC_INFO:-stats2}

if [[ "${OUTPUT_BANK_NAME}" == "" || "${OUTPUT_BANK_NAME}" == "." || "${OUTPUT_BANK_NAME}" == ".." || "${OUTPUT_BANK_NAME}" == */* ]]; then
  echo "[ERROR] Unsafe OUTPUT_BANK_NAME: ${OUTPUT_BANK_NAME}" >&2
  exit 2
fi
if [[ "${CONTACT_EXPORT_NAME}" == "" || "${CONTACT_EXPORT_NAME}" == "." || "${CONTACT_EXPORT_NAME}" == ".." || "${CONTACT_EXPORT_NAME}" == */* ]]; then
  echo "[ERROR] Unsafe CONTACT_EXPORT_NAME: ${CONTACT_EXPORT_NAME}" >&2
  exit 2
fi
if ! [[ "${EXPECTED_CLIP_COUNT}" =~ ^[0-9]+$ ]] || (( EXPECTED_CLIP_COUNT < 1 )); then
  echo "[ERROR] EXPECTED_CLIP_COUNT must be a positive integer. Got: ${EXPECTED_CLIP_COUNT}" >&2
  exit 2
fi
if ! command -v tar >/dev/null 2>&1; then
  echo "[ERROR] tar not found in PATH." >&2
  exit 1
fi

LOCAL_DATA_ROOT="${SCRIPT_DIR}/data/ds_as_data"
LOCAL_BANK="${LOCAL_DATA_ROOT}/${OUTPUT_BANK_NAME}"
STAMP=$(date +%Y%m%d_%H%M%S)
TMP_BANK="${LOCAL_DATA_ROOT}/.${OUTPUT_BANK_NAME}.tmp.${STAMP}.$$"
TMP_EXTRACT_PARENT="${LOCAL_DATA_ROOT}/.${OUTPUT_BANK_NAME}.extract.${STAMP}.$$"
BACKUP_BANK="${LOCAL_BANK}.bak.${STAMP}"

NFS_DEBUG_DISTILL_ARCHIVE_ABS=$(
  python3 - "${NFS_DEBUG_DISTILL_ARCHIVE}" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
)
NFS_DEBUG_DISTILL_BANK_ABS=$(
  python3 - "${NFS_DEBUG_DISTILL_BANK}" <<'PY'
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
SOURCE_KIND=""
if [[ -f "${NFS_DEBUG_DISTILL_ARCHIVE_ABS}" ]]; then
  SOURCE_KIND=archive
elif [[ -d "${NFS_DEBUG_DISTILL_BANK_ABS}" ]]; then
  SOURCE_KIND=directory
else
  echo "[ERROR] Missing NFS debug distill archive or bank:" >&2
  echo "        archive=${NFS_DEBUG_DISTILL_ARCHIVE_ABS}" >&2
  echo "        bank=${NFS_DEBUG_DISTILL_BANK_ABS}" >&2
  exit 1
fi
if [[ "${SOURCE_KIND}" == "directory" ]]; then
  if ! command -v rsync >/dev/null 2>&1; then
    echo "[ERROR] rsync not found in PATH." >&2
    exit 1
  fi
  if [[ ! -f "${NFS_DEBUG_DISTILL_BANK_ABS}/_clip_object_urdf_map.json" ]]; then
    echo "[ERROR] Missing source object map: ${NFS_DEBUG_DISTILL_BANK_ABS}/_clip_object_urdf_map.json" >&2
    exit 1
  fi
  if [[ ! -d "${NFS_DEBUG_DISTILL_BANK_ABS}/${CONTACT_EXPORT_NAME}/clips" ]]; then
    echo "[ERROR] Missing source contact export clips: ${NFS_DEBUG_DISTILL_BANK_ABS}/${CONTACT_EXPORT_NAME}/clips" >&2
    exit 1
  fi
fi

if [[ "${SOURCE_KIND}" == "archive" ]]; then
  echo "[INFO] source_archive=${NFS_DEBUG_DISTILL_ARCHIVE_ABS}"
else
  echo "[INFO] source_bank=${NFS_DEBUG_DISTILL_BANK_ABS}"
fi
echo "[INFO] target=${LOCAL_BANK}"
echo "[INFO] contact_export=${CONTACT_EXPORT_NAME}"
echo "[INFO] expected_clips=${EXPECTED_CLIP_COUNT}"

if [[ "${DRY_RUN}" == "1" ]]; then
  if [[ "${SOURCE_KIND}" == "archive" ]]; then
    echo "[DRY-RUN] Would extract archive into staging dir:"
    echo "          ${NFS_DEBUG_DISTILL_ARCHIVE_ABS} -> ${TMP_BANK}/"
  else
    echo "[DRY-RUN] Would rsync source into staging dir:"
    echo "          ${NFS_DEBUG_DISTILL_BANK_ABS}/ -> ${TMP_BANK}/"
  fi
  echo "[DRY-RUN] Would validate and install as ${LOCAL_BANK}"
  exit 0
fi

cleanup_tmp() {
  if [[ -d "${TMP_BANK}" ]]; then
    rm -rf "${TMP_BANK}"
  fi
  if [[ -n "${TMP_EXTRACT_PARENT:-}" && -d "${TMP_EXTRACT_PARENT}" ]]; then
    rm -rf "${TMP_EXTRACT_PARENT}"
  fi
}
trap cleanup_tmp EXIT

mkdir -p "${LOCAL_DATA_ROOT}"
rm -rf "${TMP_BANK}"
rm -rf "${TMP_EXTRACT_PARENT}"

if [[ "${SOURCE_KIND}" == "archive" ]]; then
  mkdir -p "${TMP_EXTRACT_PARENT}"
  echo "[INFO] Extracting NFS debug distill archive into staging dir..."
  tar -xf "${NFS_DEBUG_DISTILL_ARCHIVE_ABS}" -C "${TMP_EXTRACT_PARENT}"

  EXTRACTED_BANK=""
  for candidate in \
    "${TMP_EXTRACT_PARENT}/${OUTPUT_BANK_NAME}" \
    "${TMP_EXTRACT_PARENT}/${DEFAULT_BANK_NAME}"
  do
    if [[ -d "${candidate}" ]]; then
      EXTRACTED_BANK="${candidate}"
      break
    fi
  done
  if [[ -z "${EXTRACTED_BANK}" ]]; then
    mapfile -t EXTRACTED_DIRS < <(find "${TMP_EXTRACT_PARENT}" -mindepth 1 -maxdepth 1 -type d | sort)
    if [[ "${#EXTRACTED_DIRS[@]}" -eq 1 && -f "${EXTRACTED_DIRS[0]}/_clip_object_urdf_map.json" ]]; then
      EXTRACTED_BANK="${EXTRACTED_DIRS[0]}"
    elif [[ -f "${TMP_EXTRACT_PARENT}/_clip_object_urdf_map.json" ]]; then
      EXTRACTED_BANK="${TMP_EXTRACT_PARENT}"
    fi
  fi
  if [[ -z "${EXTRACTED_BANK}" ]]; then
    echo "[ERROR] Archive did not contain a recognizable debug distill bank." >&2
    exit 1
  fi
  mv "${EXTRACTED_BANK}" "${TMP_BANK}"
  if [[ "${EXTRACTED_BANK}" == "${TMP_EXTRACT_PARENT}" ]]; then
    TMP_EXTRACT_PARENT=""
  fi
else
  mkdir -p "${TMP_BANK}"
  echo "[INFO] Copying NFS debug distill bank into staging dir..."
  rsync -a --delete --human-readable --info="${RSYNC_INFO}" \
    "${NFS_DEBUG_DISTILL_BANK_ABS}/" "${TMP_BANK}/"
fi

echo "[INFO] Validating staged debug distill bank..."
if [[ ! -e "${TMP_BANK}/objects" ]]; then
  DEBUG_OBJECTS_DIR_ABS=$(
    python3 - "${DEBUG_OBJECTS_DIR}" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
  )
  if [[ ! -d "${DEBUG_OBJECTS_DIR_ABS}" ]]; then
    echo "[ERROR] Source did not include objects/ and local debug objects are missing: ${DEBUG_OBJECTS_DIR_ABS}" >&2
    echo "[ERROR] Run ./cp_debug.sh first or set DEBUG_OBJECTS_DIR." >&2
    exit 1
  fi
  ln -s "${DEBUG_OBJECTS_DIR_ABS}" "${TMP_BANK}/objects"
  echo "[INFO] Linked staged objects -> ${DEBUG_OBJECTS_DIR_ABS}"
fi
python3 - "${TMP_BANK}" "${EXPECTED_CLIP_COUNT}" "${CONTACT_EXPORT_NAME}" <<'PY'
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

bank = Path(sys.argv[1]).expanduser().resolve()
expected_clip_count = int(sys.argv[2])
contact_export_name = sys.argv[3]


def load_clips(path: Path) -> tuple[dict, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        return payload, payload["clips"]
    if isinstance(payload, dict):
        return {}, payload
    raise SystemExit(f"[ERROR] Invalid clip-object map: {path}")


def resolve_bank_path(raw_path: str, base: Path) -> Path:
    path = Path(str(raw_path).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def resolve_urdf_mesh_path(urdf_path: Path, raw_mesh: str) -> Path:
    path = Path(str(raw_mesh).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (urdf_path.parent / path).resolve()


map_path = bank / "_clip_object_urdf_map.json"
_, clips = load_clips(map_path)
if len(clips) != expected_clip_count:
    raise SystemExit(f"[ERROR] object-map entries={len(clips)}, expected {expected_clip_count}")

npz_paths = sorted(bank.glob("*.npz"))
if len(npz_paths) != expected_clip_count:
    raise SystemExit(f"[ERROR] .npz clips={len(npz_paths)}, expected {expected_clip_count}")
npz_ids = {path.stem for path in npz_paths}
map_ids = set(clips)
if npz_ids != map_ids:
    raise SystemExit(
        "[ERROR] Clip ids differ between .npz files and object map; "
        f"missing_map={sorted(npz_ids - map_ids)[:10]} missing_npz={sorted(map_ids - npz_ids)[:10]}"
    )

bad: list[str] = []
for clip_id, raw_entry in sorted(clips.items()):
    if not isinstance(raw_entry, dict):
        bad.append(f"{clip_id}: map entry is not a dict")
        continue
    urdf_path = resolve_bank_path(str(raw_entry.get("object_urdf_path", "")), map_path.parent)
    mesh_raw = str(raw_entry.get("object_mesh_path", "")).strip()
    mesh_path = resolve_bank_path(mesh_raw, map_path.parent) if mesh_raw else None
    if not urdf_path.is_file():
        bad.append(f"{clip_id}: missing URDF {urdf_path}")
        continue
    if mesh_path is not None and not mesh_path.is_file():
        bad.append(f"{clip_id}: missing mesh {mesh_path}")
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        bad.append(f"{clip_id}: failed to parse URDF {urdf_path}: {exc}")
        continue
    if not root.findall(".//visual//mesh"):
        bad.append(f"{clip_id}: URDF has no visual mesh: {urdf_path}")
    if not root.findall(".//collision//mesh"):
        bad.append(f"{clip_id}: URDF has no collision mesh: {urdf_path}")
    for tag_name in ("box", "sphere", "cylinder", "capsule"):
        if root.findall(f".//{tag_name}"):
            bad.append(f"{clip_id}: URDF contains primitive <{tag_name}> geometry: {urdf_path}")
            break
    for mesh_tag in root.findall(".//mesh"):
        mesh_raw = str(mesh_tag.get("filename", "")).strip()
        if not mesh_raw:
            bad.append(f"{clip_id}: URDF has empty mesh filename: {urdf_path}")
            continue
        urdf_mesh_path = resolve_urdf_mesh_path(urdf_path, mesh_raw)
        if not urdf_mesh_path.is_file():
            bad.append(f"{clip_id}: URDF mesh missing: {urdf_mesh_path}")

if bad:
    raise SystemExit("[ERROR] Real-mesh bank validation failed:\n  " + "\n  ".join(bad[:20]))

contact_clips = bank / contact_export_name / "clips"
clip_dirs = sorted(path for path in contact_clips.iterdir() if path.is_dir())
if len(clip_dirs) != expected_clip_count:
    raise SystemExit(f"[ERROR] contact clip dirs={len(clip_dirs)}, expected {expected_clip_count}")

required_files = (
    "teacher_rollout_reference.npz",
    "left_wrist_contact_points.npy",
    "left_wrist_contact_point_counts.npy",
    "left_wrist_contact_interval_steps.npy",
    "right_wrist_contact_points.npy",
    "right_wrist_contact_point_counts.npy",
    "right_wrist_contact_interval_steps.npy",
)
missing_files: list[str] = []
contact_ids: set[str] = set()
for clip_dir in clip_dirs:
    metadata_path = clip_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            clip_id = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("clip_id", "")).strip()
        except Exception:
            clip_id = ""
    else:
        clip_id = ""
    if not clip_id:
        normalized = clip_dir.name.strip()
        if normalized in npz_ids:
            clip_id = normalized
        else:
            prefix, separator, suffix = normalized.partition("_")
            clip_id = suffix.strip() if separator and prefix.isdecimal() and suffix.strip() else normalized
    if clip_id in contact_ids:
        raise SystemExit(f"[ERROR] Duplicate contact directories resolve to clip {clip_id!r}")
    contact_ids.add(clip_id)
    for file_name in required_files:
        if not (clip_dir / file_name).is_file():
            missing_files.append(f"{clip_id}:{file_name}")

missing_contacts = sorted(npz_ids.difference(contact_ids))
if missing_contacts:
    raise SystemExit(f"[ERROR] Contact export missing active clips: {missing_contacts[:20]}")
if missing_files:
    raise SystemExit("[ERROR] Contact export has incomplete sidecars: " + ", ".join(missing_files[:20]))

print(
    f"[INFO] Validated debug distill bank: {bank} "
    f"({len(npz_paths)} clips, {len(clip_dirs)} contact clip dirs)"
)
PY

if [[ -d "${LOCAL_BANK}" ]]; then
  if [[ "${KEEP_BACKUP}" == "1" ]]; then
    echo "[INFO] Existing target found; moving to backup: ${BACKUP_BANK}"
    rm -rf "${BACKUP_BANK}"
    mv "${LOCAL_BANK}" "${BACKUP_BANK}"
  else
    echo "[INFO] Removing existing target: ${LOCAL_BANK}"
    rm -rf "${LOCAL_BANK}"
  fi
fi

mv "${TMP_BANK}" "${LOCAL_BANK}"
trap - EXIT

echo "[INFO] Installed debug distill bank:"
du -sh "${LOCAL_BANK}"
find "${LOCAL_BANK}" -maxdepth 1 -name '*.npz' | wc -l | awk '{print "[INFO] npz_count=" $1}'
find "${LOCAL_BANK}/${CONTACT_EXPORT_NAME}/clips" -mindepth 1 -maxdepth 1 -type d | wc -l | awk '{print "[INFO] contact_clip_count=" $1}'
