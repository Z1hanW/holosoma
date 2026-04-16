#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

NFS_TAO_ROOT=${NFS_TAO_ROOT:-/nfs/zzzihanw/tao}
ROLLOUT_ASSET_ROOT=${ROLLOUT_ASSET_ROOT:-${NFS_TAO_ROOT}/teacher_rollout_assets_20260415}
ROLLOUT_ARCHIVE=${ROLLOUT_ARCHIVE:-${NFS_TAO_ROOT}/teacher_box_contacts_rollout_ref_motionbank_20260415b_utc.tar.gz}
RAW_EXPORT_DEST=${RAW_EXPORT_DEST:-${SCRIPT_DIR}/outputs/teacher_box_contacts_rollout_ref_motionbank_20260415b_utc}
FILTERED_MOTION_BANK_NAME=${FILTERED_MOTION_BANK_NAME:-motion_bank_success_box_0_92_0p3}

mkdir -p outputs

require_dir() {
  local path="$1"
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing directory: ${path}" >&2
    exit 1
  fi
}

if [[ -f "${ROLLOUT_ARCHIVE}" ]]; then
  echo "[INFO] Restoring raw rollout export -> ${RAW_EXPORT_DEST}"
  rm -rf "${RAW_EXPORT_DEST}"
  mkdir -p "${RAW_EXPORT_DEST}"
  tar -xzf "${ROLLOUT_ARCHIVE}" --strip-components=1 -C "${RAW_EXPORT_DEST}"
else
  echo "[WARN] Rollout archive not found at ${ROLLOUT_ARCHIVE}; skipping raw export restore"
fi

SOURCE_MOTION_BANK="${ROLLOUT_ASSET_ROOT}/outputs/motion_bank"
SOURCE_FILTERED_MOTION_BANK="${ROLLOUT_ASSET_ROOT}/outputs/${FILTERED_MOTION_BANK_NAME}"
SOURCE_DROP_FINAL_MOTION_BANK="${ROLLOUT_ASSET_ROOT}/outputs/motion_bank_drop_final_1aaf51f7c2"
SOURCE_CLIPS="${ROLLOUT_ASSET_ROOT}/outputs/clips"
SOURCE_OUTPUTS_VIS="${ROLLOUT_ASSET_ROOT}/outputs_vis"
SOURCE_OUTPUTS_STS="${ROLLOUT_ASSET_ROOT}/outputs_sts"
SOURCE_SUCCESS_CLIPS="${SOURCE_OUTPUTS_STS}/success_clips.txt"

require_dir "${SOURCE_MOTION_BANK}"
require_dir "${SOURCE_CLIPS}"

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

validate_rollout_assets "${SOURCE_MOTION_BANK}" "${SOURCE_CLIPS}" "${SOURCE_SUCCESS_CLIPS}"

mkdir -p outputs/motion_bank
mkdir -p "outputs/${FILTERED_MOTION_BANK_NAME}"
mkdir -p outputs/clips
mkdir -p outputs_vis
mkdir -p outputs_sts

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

echo "[INFO] Restored rollout assets:"
echo "  - $(count_npz_in_dir outputs/motion_bank) clips in outputs/motion_bank"
echo "  - $(count_npz_or_symlinks_in_dir "outputs/${FILTERED_MOTION_BANK_NAME}") clips in outputs/${FILTERED_MOTION_BANK_NAME}"
echo "  - $(count_npz_in_dir outputs/motion_bank_drop_final_1aaf51f7c2) clips in outputs/motion_bank_drop_final_1aaf51f7c2"
echo "  - $(find outputs/clips -maxdepth 1 -mindepth 1 -type d | wc -l | tr -d ' ') clip dirs in outputs/clips"
echo "  - $(find outputs_vis/clips -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ') clip dirs in outputs_vis/clips"
echo "  - $(find outputs_sts/clips -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l | tr -d ' ') clip dirs in outputs_sts/clips"
