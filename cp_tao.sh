#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

NFS_TAO_ROOT=${NFS_TAO_ROOT:-/nfs/zzzihanw/tao}
ROLLOUT_ASSET_ROOT=${ROLLOUT_ASSET_ROOT:-${NFS_TAO_ROOT}/teacher_rollout_assets_20260415}
ROLLOUT_ARCHIVE=${ROLLOUT_ARCHIVE:-${NFS_TAO_ROOT}/teacher_box_contacts_rollout_ref_motionbank_20260415b_utc.tar.gz}
RAW_EXPORT_DEST=${RAW_EXPORT_DEST:-${SCRIPT_DIR}/outputs/teacher_box_contacts_rollout_ref_motionbank_20260415b_utc}
FILTERED_MOTION_BANK_NAME=${FILTERED_MOTION_BANK_NAME:-motion_bank_success_box_0_92_0p3}

# Optional standalone AS teacher-rollout distillation bank filtered by:
#   stable_contact_success=True and final_object_position_error_m<=0.5
# The source can be a prepared NFS directory with symlinked npz/contact assets;
# this script installs a repo-local copy with only relative symlinks.
COPY_AS_SUCCESS133=${COPY_AS_SUCCESS133:-0}
ONLY_AS_SUCCESS133=${ONLY_AS_SUCCESS133:-0}
AS_SUCCESS133_BANK_NAME=${AS_SUCCESS133_BANK_NAME:-carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5}
AS_SUCCESS133_EXPECTED_TOTAL=${AS_SUCCESS133_EXPECTED_TOTAL:-133}
AS_SUCCESS133_CONTACT_EXPORT_NAME=${AS_SUCCESS133_CONTACT_EXPORT_NAME:-contact_export_from_teacher_success133_final0p5}
AS_SUCCESS133_SOURCE=${AS_SUCCESS133_SOURCE:-}
LOCAL_AS_ROOT=${LOCAL_AS_ROOT:-data/ds_as_data}
NFS_TAO_PARENT=$(dirname "${NFS_TAO_ROOT}")
AS_SUCCESS133_SOURCE_CANDIDATES=${AS_SUCCESS133_SOURCE_CANDIDATES:-"${NFS_TAO_ROOT}/${AS_SUCCESS133_BANK_NAME} ${NFS_TAO_ROOT}/ds_as_data/${AS_SUCCESS133_BANK_NAME} ${NFS_TAO_PARENT}/ds_as_data/${AS_SUCCESS133_BANK_NAME}"}

case "${1:-}" in
  success133|as-success133|as_success133|success133-final0p5|success133_final0p5)
    ONLY_AS_SUCCESS133=1
    COPY_AS_SUCCESS133=1
    shift
    ;;
esac
if [[ $# -gt 0 ]]; then
  echo "[ERROR] Unsupported cp_tao.sh argument(s): $*" >&2
  echo "[ERROR] Use 'bash cp_tao.sh success133' to install only the 133-clip AS distill bank." >&2
  exit 2
fi

case "$(echo "${ONLY_AS_SUCCESS133}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) ;;
  *) mkdir -p outputs ;;
esac

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing directory: ${path}" >&2
    exit 1
  fi
}

is_truthy() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

if ! is_truthy "${ONLY_AS_SUCCESS133}"; then
  if [[ -f "${ROLLOUT_ARCHIVE}" ]]; then
    echo "[INFO] Restoring raw rollout export -> ${RAW_EXPORT_DEST}"
    rm -rf "${RAW_EXPORT_DEST}"
    mkdir -p "${RAW_EXPORT_DEST}"
    tar -xzf "${ROLLOUT_ARCHIVE}" --strip-components=1 -C "${RAW_EXPORT_DEST}"
  else
    echo "[WARN] Rollout archive not found at ${ROLLOUT_ARCHIVE}; skipping raw export restore"
  fi
fi

SOURCE_MOTION_BANK="${ROLLOUT_ASSET_ROOT}/outputs/motion_bank"
SOURCE_FILTERED_MOTION_BANK="${ROLLOUT_ASSET_ROOT}/outputs/${FILTERED_MOTION_BANK_NAME}"
SOURCE_DROP_FINAL_MOTION_BANK="${ROLLOUT_ASSET_ROOT}/outputs/motion_bank_drop_final_1aaf51f7c2"
SOURCE_CLIPS="${ROLLOUT_ASSET_ROOT}/outputs/clips"
SOURCE_OUTPUTS_VIS="${ROLLOUT_ASSET_ROOT}/outputs_vis"
SOURCE_OUTPUTS_STS="${ROLLOUT_ASSET_ROOT}/outputs_sts"
SOURCE_SUCCESS_CLIPS="${SOURCE_OUTPUTS_STS}/success_clips.txt"

if ! is_truthy "${ONLY_AS_SUCCESS133}"; then
  require_dir "${SOURCE_MOTION_BANK}"
  require_dir "${SOURCE_CLIPS}"
fi

validate_rollout_assets() {
  local motion_bank_dir="$1"
  local clips_dir="$2"
  local success_clips_file="$3"
  python - "${motion_bank_dir}" "${clips_dir}" "${success_clips_file}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

motion_bank_dir = Path(sys.argv[1]).expanduser().resolve()
clips_dir = Path(sys.argv[2]).expanduser().resolve()
success_clips_file = Path(sys.argv[3]).expanduser().resolve()

if not success_clips_file.is_file():
    motion_count = len(list(motion_bank_dir.glob("*.npz")))
    clip_count = sum(1 for path in clips_dir.iterdir() if path.is_dir())
    if motion_count == 0 or clip_count == 0:
        raise SystemExit(f"[ERROR] rollout assets are empty: motion_bank={motion_count}, clips={clip_count}")
    if motion_count != clip_count:
        raise SystemExit(
            f"[ERROR] rollout assets are incomplete or mismatched and no success filter exists: "
            f"motion_bank={motion_count}, clips={clip_count}"
        )
    print(f"[INFO] Validated unfiltered rollout assets: {motion_count} motion clips, {clip_count} contact dirs")
    raise SystemExit(0)

clip_ids: list[str] = []
seen: set[str] = set()
for raw_line in success_clips_file.read_text(encoding="utf-8").splitlines():
    clip_id = raw_line.strip()
    if not clip_id or clip_id.startswith("#"):
        continue
    if clip_id not in seen:
        clip_ids.append(clip_id)
        seen.add(clip_id)
if not clip_ids:
    raise SystemExit(f"[ERROR] No clips listed in {success_clips_file}")

def infer_clip_id(dir_name: str) -> str:
    return dir_name.split("_", 1)[1].strip() if "_" in dir_name else dir_name.strip()

contact_dirs: dict[str, Path] = {}
for candidate in clips_dir.iterdir():
    if candidate.is_dir():
        contact_dirs.setdefault(infer_clip_id(candidate.name), candidate)

missing: list[str] = []
for clip_id in clip_ids:
    if not (motion_bank_dir / f"{clip_id}.npz").is_file():
        missing.append(f"{clip_id}:motion")
    clip_dir = contact_dirs.get(clip_id)
    if clip_dir is None:
        missing.append(f"{clip_id}:contact_dir")
        continue
    for file_name in ("teacher_rollout_reference.npz", "left_wrist_contact_interval_steps.npy"):
        if not (clip_dir / file_name).is_file():
            missing.append(f"{clip_id}:{file_name}")

if missing:
    preview = ", ".join(missing[:20])
    raise SystemExit(f"[ERROR] success-filtered rollout assets are incomplete: {preview}")

print(
    f"[INFO] Validated success-filtered rollout assets: "
    f"{len(clip_ids)} clips from {success_clips_file}"
)
PY
}

if ! is_truthy "${ONLY_AS_SUCCESS133}"; then
  validate_rollout_assets "${SOURCE_MOTION_BANK}" "${SOURCE_CLIPS}" "${SOURCE_SUCCESS_CLIPS}"

  mkdir -p outputs/motion_bank
  mkdir -p "outputs/${FILTERED_MOTION_BANK_NAME}"
  mkdir -p outputs/clips
  mkdir -p outputs_vis
  mkdir -p outputs_sts
fi

count_npz_in_dir() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    find "${path}" -maxdepth 1 -type f -name '*.npz' ! -name '_clip_object_urdf_map.json' | wc -l | tr -d ' '
  else
    echo 0
  fi
}

count_npz_or_symlinks_in_dir() {
  local path="$1"
  if [[ -d "${path}" ]]; then
    find "${path}" -maxdepth 1 \( -type f -o -type l \) -name '*.npz' | wc -l | tr -d ' '
  else
    echo 0
  fi
}

prepare_success_motion_subset() {
  local source_dir="$1"
  local success_clips_file="$2"
  local subset_dir="$3"

  if [[ ! -f "${success_clips_file}" ]]; then
    echo "[WARN] Success clip filter not found at ${success_clips_file}; skipping ${subset_dir} generation"
    return 0
  fi

  mkdir -p "${subset_dir}"
  python - "${source_dir}" "${success_clips_file}" "${subset_dir}" <<'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

source_dir = Path(sys.argv[1]).expanduser().resolve()
success_clips_file = Path(sys.argv[2]).expanduser().resolve()
subset_dir = Path(sys.argv[3]).expanduser().resolve()

clip_ids: list[str] = []
seen: set[str] = set()
for raw_line in success_clips_file.read_text(encoding="utf-8").splitlines():
    clip_id = raw_line.strip()
    if not clip_id or clip_id.startswith("#"):
        continue
    if "/" in clip_id or "\\" in clip_id:
        raise SystemExit(f"[ERROR] Invalid clip id in {success_clips_file}: {clip_id}")
    if clip_id not in seen:
        clip_ids.append(clip_id)
        seen.add(clip_id)

if not clip_ids:
    raise SystemExit(f"[ERROR] No clips listed in {success_clips_file}")

missing = [clip_id for clip_id in clip_ids if not (source_dir / f"{clip_id}.npz").is_file()]
if missing:
    preview = ", ".join(missing[:20])
    raise SystemExit(f"[ERROR] Success-filtered motion clips missing from {source_dir}: {preview}")

subset_dir.mkdir(parents=True, exist_ok=True)
for existing in subset_dir.glob("*.npz"):
    existing.unlink()

for clip_id in clip_ids:
    source_path = source_dir / f"{clip_id}.npz"
    target = subset_dir / source_path.name
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(os.path.relpath(source_path, start=subset_dir))

clip_metadata = {}
metadata_uses_clips_key = True
for candidate in (source_dir / "_clip_object_urdf_map.json", source_dir / "clip_object_urdf_map.json"):
    if candidate.is_file():
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
            clip_metadata = payload["clips"]
            metadata_uses_clips_key = True
        elif isinstance(payload, dict):
            clip_metadata = payload
            metadata_uses_clips_key = False
        break

filtered = {}
missing_metadata: list[str] = []
for clip_id in clip_ids:
    entry = clip_metadata.get(clip_id) if isinstance(clip_metadata, dict) else None
    if entry is not None:
        filtered[clip_id] = entry
        continue
    urdf = ""
    try:
        data = np.load(source_dir / f"{clip_id}.npz", allow_pickle=True)
        if "object_urdf_path" in data:
            arr = np.asarray(data["object_urdf_path"])
            if arr.size:
                item = arr.item() if arr.shape == () else arr.reshape(-1)[0]
                urdf = str(item).strip()
    except Exception:
        urdf = ""
    if urdf:
        filtered[clip_id] = {"object_urdf_path": urdf}
    else:
        missing_metadata.append(clip_id)

if missing_metadata:
    preview = ", ".join(missing_metadata[:20])
    raise SystemExit(f"[ERROR] Success-filtered motion clips missing object metadata: {preview}")

map_path = subset_dir / "_clip_object_urdf_map.json"
if map_path.exists() or map_path.is_symlink():
    map_path.unlink()
payload = {"clips": filtered} if metadata_uses_clips_key else filtered
map_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

print(f"[INFO] Generated {subset_dir} with {len(clip_ids)} clips")
PY
}

resolve_as_success133_source() {
  if [[ -n "${AS_SUCCESS133_SOURCE}" ]]; then
    if [[ ! -d "${AS_SUCCESS133_SOURCE}" ]]; then
      echo "[ERROR] AS_SUCCESS133_SOURCE is not a directory: ${AS_SUCCESS133_SOURCE}" >&2
      return 2
    fi
    printf '%s\n' "${AS_SUCCESS133_SOURCE}"
    return 0
  fi

  local candidate
  for candidate in ${AS_SUCCESS133_SOURCE_CANDIDATES}; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

install_as_success133_bank() {
  if ! is_truthy "${COPY_AS_SUCCESS133}"; then
    echo "[INFO] COPY_AS_SUCCESS133=0; skipping AS success133 distill bank install"
    return 0
  fi
  if [[ "${AS_SUCCESS133_BANK_NAME}" == "" || "${AS_SUCCESS133_BANK_NAME}" == "." || "${AS_SUCCESS133_BANK_NAME}" == ".." || "${AS_SUCCESS133_BANK_NAME}" == */* ]]; then
    echo "[ERROR] Unsafe AS_SUCCESS133_BANK_NAME: ${AS_SUCCESS133_BANK_NAME}" >&2
    exit 2
  fi

  local source_dir=""
  local candidate
  if ! source_dir="$(resolve_as_success133_source)"; then
    if [[ -n "${AS_SUCCESS133_SOURCE}" || "$(echo "${ONLY_AS_SUCCESS133}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
      echo "[ERROR] AS success133 source not found. Searched:" >&2
      for candidate in ${AS_SUCCESS133_SOURCE_CANDIDATES}; do
        echo "  - ${candidate}" >&2
      done
      echo "[ERROR] Set AS_SUCCESS133_SOURCE=/path/to/${AS_SUCCESS133_BANK_NAME}" >&2
      exit 2
    fi
    echo "[WARN] AS success133 source not found; skipping. Searched:"
    for candidate in ${AS_SUCCESS133_SOURCE_CANDIDATES}; do
      echo "  - ${candidate}"
    done
    echo "[WARN] Set AS_SUCCESS133_SOURCE=/path/to/${AS_SUCCESS133_BANK_NAME} to install it explicitly."
    return 0
  fi

  local dest_dir="${LOCAL_AS_ROOT}/${AS_SUCCESS133_BANK_NAME}"
  local tmp_dir="${LOCAL_AS_ROOT}/.${AS_SUCCESS133_BANK_NAME}.tmp.$$"
  local source_abs
  local dest_abs
  source_abs=$(python3 - "${source_dir}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)
  dest_abs=$(python3 - "${dest_dir}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)

  mkdir -p "${LOCAL_AS_ROOT}"
  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"
  echo "[INFO] Installing AS success133 distill bank"
  echo "[INFO]   source=${source_abs}"
  echo "[INFO]   dest=${dest_abs}"

  rsync -aL --delete --exclude '/_single_slot_motion_bank/' "${source_abs}/" "${tmp_dir}/"

  python3 - "${tmp_dir}" <<'PY'
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

bank_dir = Path(sys.argv[1]).expanduser().resolve()
view_dir = bank_dir / "_single_slot_motion_bank"
marker = view_dir / ".generated_by_train_as_general"
if view_dir.exists():
    for child in view_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
else:
    view_dir.mkdir(parents=True)

for npz_path in sorted(bank_dir.glob("*.npz")):
    target = view_dir / npz_path.name
    target.symlink_to(os.path.relpath(npz_path, start=view_dir))
marker.write_text("generated by cp_tao.sh for AS success133 final0p5\n", encoding="utf-8")
PY

  python3 "${SCRIPT_DIR}/scripts/prepare_single_slot_object_map.py" \
    --motion-dir "${tmp_dir}/_single_slot_motion_bank" \
    --object-map "${tmp_dir}/_clip_object_urdf_map.json" \
    --output-map "${tmp_dir}/_single_slot_motion_bank/_clip_object_urdf_map.json" >/dev/null

  python3 - "${tmp_dir}" "${AS_SUCCESS133_EXPECTED_TOTAL}" "${AS_SUCCESS133_CONTACT_EXPORT_NAME}" <<'PY'
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

bank_dir = Path(sys.argv[1]).expanduser().resolve()
expected = int(sys.argv[2]) if sys.argv[2].strip() else 0
contact_export_name = sys.argv[3].strip()

def fail(message: str) -> None:
    raise SystemExit(f"[ERROR] {message}")

npz_paths = sorted(bank_dir.glob("*.npz"))
if expected and len(npz_paths) != expected:
    fail(f"Expected {expected} top-level .npz clips in {bank_dir}, found {len(npz_paths)}")
if not npz_paths:
    fail(f"No top-level .npz clips found in {bank_dir}")

map_path = bank_dir / "_clip_object_urdf_map.json"
if not map_path.is_file():
    fail(f"Missing object map: {map_path}")
payload = json.loads(map_path.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if expected and len(clips) != expected:
    fail(f"Expected {expected} object-map entries in {map_path}, found {len(clips)}")
npz_ids = {path.stem for path in npz_paths}
map_ids = set(clips)
if npz_ids != map_ids:
    fail(f"Clip set mismatch: missing_map={sorted(npz_ids - map_ids)[:10]} missing_npz={sorted(map_ids - npz_ids)[:10]}")

def resolve_path(raw: str, base: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()

bad: list[str] = []
for clip_id, entry_raw in sorted(clips.items()):
    entry = entry_raw if isinstance(entry_raw, dict) else {"object_urdf_path": str(entry_raw)}
    urdf_raw = str(entry.get("object_urdf_path", "")).strip()
    if Path(urdf_raw).is_absolute():
        bad.append(f"{clip_id}: object_urdf_path is absolute: {urdf_raw}")
        continue
    urdf = resolve_path(urdf_raw, bank_dir)
    if not urdf.is_file():
        bad.append(f"{clip_id}: missing URDF {urdf}")
        continue
    mesh_raw = str(entry.get("object_mesh_path", "")).strip()
    if mesh_raw:
        if Path(mesh_raw).is_absolute():
            bad.append(f"{clip_id}: object_mesh_path is absolute: {mesh_raw}")
        elif not resolve_path(mesh_raw, bank_dir).is_file():
            bad.append(f"{clip_id}: missing mesh {resolve_path(mesh_raw, bank_dir)}")
    try:
        root = ET.parse(urdf).getroot()
    except Exception as exc:
        bad.append(f"{clip_id}: invalid URDF {urdf}: {exc}")
        continue
    for tag in root.findall(".//mesh"):
        filename = str(tag.get("filename", "")).strip()
        if not filename:
            bad.append(f"{clip_id}: empty URDF mesh filename in {urdf}")
            continue
        if Path(filename).is_absolute():
            bad.append(f"{clip_id}: absolute URDF mesh filename in {urdf}: {filename}")
        elif not resolve_path(filename, urdf.parent).is_file():
            bad.append(f"{clip_id}: missing URDF mesh file {resolve_path(filename, urdf.parent)}")

single_slot = bank_dir / "_single_slot_motion_bank"
single_npz = sorted(single_slot.glob("*.npz"))
if expected and len(single_npz) != expected:
    bad.append(f"single-slot motion view has {len(single_npz)} clips, expected {expected}")
single_map_path = single_slot / "_clip_object_urdf_map.json"
if not single_map_path.is_file():
    bad.append(f"missing single-slot object map: {single_map_path}")
else:
    single_payload = json.loads(single_map_path.read_text(encoding="utf-8"))
    single_clips = (
        single_payload["clips"]
        if isinstance(single_payload, dict) and isinstance(single_payload.get("clips"), dict)
        else single_payload
    )
    if expected and len(single_clips) != expected:
        bad.append(f"single-slot object map has {len(single_clips)} entries, expected {expected}")

contact_root = bank_dir / contact_export_name
clips_root = contact_root / "clips" if (contact_root / "clips").is_dir() else contact_root
if not clips_root.is_dir():
    bad.append(f"missing contact sidecars: {clips_root}")
else:
    def infer_clip_id(dir_name: str) -> str:
        return dir_name.split("_", 1)[1].strip() if "_" in dir_name else dir_name.strip()

    required_files = (
        "metadata.json",
        "left_wrist_contact_points.npy",
        "left_wrist_contact_point_counts.npy",
        "left_wrist_contact_interval_steps.npy",
        "right_wrist_contact_points.npy",
        "right_wrist_contact_point_counts.npy",
        "right_wrist_contact_interval_steps.npy",
    )
    contact_ids: set[str] = set()
    for clip_dir in sorted(path for path in clips_root.iterdir() if path.is_dir()):
        clip_id = infer_clip_id(clip_dir.name)
        metadata_path = clip_dir / "metadata.json"
        if metadata_path.is_file():
            try:
                clip_id = str(json.loads(metadata_path.read_text(encoding="utf-8")).get("clip_id") or clip_id)
            except Exception as exc:
                bad.append(f"{clip_dir.name}: invalid metadata.json: {exc}")
        contact_ids.add(clip_id)
        for file_name in required_files:
            if not (clip_dir / file_name).is_file():
                bad.append(f"{clip_id}: missing contact sidecar {file_name}")
    missing_contacts = sorted(npz_ids.difference(contact_ids))
    if missing_contacts:
        bad.append(f"missing contact dirs for active clips: {', '.join(missing_contacts[:10])}")

absolute_symlinks = []
for path in bank_dir.rglob("*"):
    if path.is_symlink() and Path(os.readlink(path)).is_absolute():
        absolute_symlinks.append(str(path.relative_to(bank_dir)))
if absolute_symlinks:
    bad.append("absolute symlinks remain: " + ", ".join(absolute_symlinks[:20]))

if bad:
    fail("AS success133 bank validation failed:\n  " + "\n  ".join(bad[:40]))

print(
    f"[INFO] Validated AS success133 distill bank: {bank_dir} "
    f"({len(npz_paths)} clips, contact_sidecars={len(npz_ids)})"
)
PY

  rm -rf "${dest_dir}"
  mkdir -p "$(dirname "${dest_dir}")"
  mv "${tmp_dir}" "${dest_dir}"
  echo "[INFO] Installed AS success133 distill bank: ${dest_dir}"
  echo "[INFO] Use: bash distill_as_perception.sh success133"
}

if is_truthy "${ONLY_AS_SUCCESS133}"; then
  install_as_success133_bank
  echo "[INFO] Installed only AS success133 final0p5 bank:"
  echo "  - $(count_npz_or_symlinks_in_dir "${LOCAL_AS_ROOT}/${AS_SUCCESS133_BANK_NAME}") clips in ${LOCAL_AS_ROOT}/${AS_SUCCESS133_BANK_NAME}"
  exit 0
fi

echo "[INFO] Syncing teacher rollout motion bank -> outputs/motion_bank"
rsync -avh --delete "${SOURCE_MOTION_BANK}/" outputs/motion_bank/

if [[ -d "${SOURCE_FILTERED_MOTION_BANK}" ]]; then
  echo "[INFO] Syncing filtered rollout motion bank -> outputs/${FILTERED_MOTION_BANK_NAME}"
  rsync -avh --delete "${SOURCE_FILTERED_MOTION_BANK}/" "outputs/${FILTERED_MOTION_BANK_NAME}/"
fi

if [[ -n "${SOURCE_DROP_FINAL_MOTION_BANK}" && -d "${SOURCE_DROP_FINAL_MOTION_BANK}" ]]; then
  mkdir -p outputs/motion_bank_drop_final_1aaf51f7c2
  echo "[INFO] Syncing drop-final motion bank -> outputs/motion_bank_drop_final_1aaf51f7c2"
  rsync -avh --delete "${SOURCE_DROP_FINAL_MOTION_BANK}/" outputs/motion_bank_drop_final_1aaf51f7c2/
else
  echo "[WARN] Drop-final motion bank not found in NFS/archive sources; leaving outputs/motion_bank_drop_final_1aaf51f7c2 unchanged"
fi

echo "[INFO] Syncing rollout clip references -> outputs/clips"
rsync -avh --delete "${SOURCE_CLIPS}/" outputs/clips/

if [[ -d "${SOURCE_OUTPUTS_VIS}" ]]; then
  echo "[INFO] Syncing rollout visualizations -> outputs_vis"
  rsync -avh --delete "${SOURCE_OUTPUTS_VIS}/" outputs_vis/
else
  echo "[WARN] Visualization directory not found at ${SOURCE_OUTPUTS_VIS}; skipping outputs_vis sync"
fi

if [[ -d "${SOURCE_OUTPUTS_STS}" ]]; then
  echo "[INFO] Syncing rollout statistics -> outputs_sts"
  rsync -avh --delete "${SOURCE_OUTPUTS_STS}/" outputs_sts/
else
  echo "[WARN] Statistics directory not found at ${SOURCE_OUTPUTS_STS}; skipping outputs_sts sync"
fi

prepare_success_motion_subset \
  outputs/motion_bank \
  outputs_sts/success_clips.txt \
  "outputs/${FILTERED_MOTION_BANK_NAME}"

if is_truthy "${COPY_AS_SUCCESS133}"; then
  install_as_success133_bank
fi

echo "[INFO] Restored rollout assets:"
echo "  - $(count_npz_in_dir outputs/motion_bank) clips in outputs/motion_bank"
echo "  - $(count_npz_or_symlinks_in_dir "outputs/${FILTERED_MOTION_BANK_NAME}") clips in outputs/${FILTERED_MOTION_BANK_NAME}"
echo "  - $(count_npz_in_dir outputs/motion_bank_drop_final_1aaf51f7c2) clips in outputs/motion_bank_drop_final_1aaf51f7c2"
echo "  - $(find outputs/clips -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ') clip dirs in outputs/clips"
echo "  - $(find outputs_vis/clips -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ') clip dirs in outputs_vis/clips"
echo "  - $(find outputs_sts/clips -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ') clip dirs in outputs_sts/clips"
if [[ -d "${LOCAL_AS_ROOT}/${AS_SUCCESS133_BANK_NAME}" ]]; then
  echo "  - $(count_npz_or_symlinks_in_dir "${LOCAL_AS_ROOT}/${AS_SUCCESS133_BANK_NAME}") clips in ${LOCAL_AS_ROOT}/${AS_SUCCESS133_BANK_NAME}"
fi
