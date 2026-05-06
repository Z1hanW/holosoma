#!/usr/bin/env bash
set -euo pipefail

# Refresh the local DS box data from NFS.
#
# This intentionally replaces the whole local data/ds_box_data tree. The source
# /nfs/zzzihanw/ds_box_data is treated as the canonical new DS data layout.
#
# Usage:
#   bash cp_box.sh
#
# Optional env:
#   NFS_DS_ROOT=/nfs/zzzihanw/ds_box_data
#   NFS_OMOMO_PREPARED_ROOT=/nfs/zzzihanw/ds_box_data_v2_apr_15/train_g1_w_obj_prepared_plus_omomo_orig
#   BUILD_OMOMO_MIXED_BANK=1
#   LOCAL_DS_ROOT=/home/ubuntu/FAR/holosoma/data/ds_box_data
#   CANONICALIZE_ONLY=1
#   DRY_RUN=1

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)

NFS_DS_ROOT=${NFS_DS_ROOT:-/nfs/zzzihanw/ds_box_data}
NFS_OMOMO_PREPARED_ROOT=${NFS_OMOMO_PREPARED_ROOT:-/nfs/zzzihanw/ds_box_data_v2_apr_15/train_g1_w_obj_prepared_plus_omomo_orig}
BUILD_OMOMO_MIXED_BANK=${BUILD_OMOMO_MIXED_BANK:-1}
REPO_DS_RELATIVE_PATH="data/ds_box_data"
EXPECTED_LOCAL_ROOT="${SCRIPT_DIR}/${REPO_DS_RELATIVE_PATH}"
if [[ -n "${LOCAL_DS_ROOT+x}" ]]; then
  LOCAL_DS_ROOT="$(
    python - "${LOCAL_DS_ROOT}" <<'PY'
import os
import sys

print(os.path.abspath(sys.argv[1]))
PY
  )"
else
  LOCAL_DS_ROOT="${EXPECTED_LOCAL_ROOT}"
fi
CANONICALIZE_ONLY=${CANONICALIZE_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ "${LOCAL_DS_ROOT}" != "${EXPECTED_LOCAL_ROOT}" ]]; then
  echo "[ERROR] Refusing to delete unexpected LOCAL_DS_ROOT: ${LOCAL_DS_ROOT}" >&2
  echo "[ERROR] Expected repo-relative target exactly: ${EXPECTED_LOCAL_ROOT}" >&2
  exit 2
fi
if [[ "${LOCAL_DS_ROOT}" == "/" || "${LOCAL_DS_ROOT}" == "${SCRIPT_DIR}" || "${LOCAL_DS_ROOT}" == "${SCRIPT_DIR}/data" || "${LOCAL_DS_ROOT}" != "${SCRIPT_DIR}/"* ]]; then
  echo "[ERROR] Unsafe LOCAL_DS_ROOT: ${LOCAL_DS_ROOT}" >&2
  exit 2
fi

canonicalize_and_validate_prepared_bank() {
  local local_root="$1"

  python - "${local_root}" <<'PY'
import json
import os
import sys
from pathlib import Path

import numpy as np


def scalar_str(value) -> str:
    if value is None:
        return ""
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    if arr.shape == ():
        item = arr.item()
    else:
        item = arr.reshape(-1)[0]
        if hasattr(item, "item"):
            item = item.item()
    return str(item).strip()


def object_size_list(value) -> list[float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1 and arr.shape[0] == 3:
        return [float(v) for v in arr.tolist()]
    if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 1:
        return [float(v) for v in arr[0].tolist()]
    return None


def write_npz_atomic(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp_path = path.with_name(f".{path.name}.canonicalizing.npz")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        np.savez_compressed(tmp_path, **payload)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


local_root = Path(sys.argv[1]).expanduser().resolve()
prepared_dir = local_root / "train_g1_w_obj_prepared"
generated_urdf_dir = prepared_dir / "_generated_urdfs"
map_path = prepared_dir / "_clip_object_urdf_map.json"

for required_path in (prepared_dir, generated_urdf_dir, map_path):
    if not required_path.exists():
        raise SystemExit(f"[ERROR] Missing required prepared-bank path: {required_path}")

npz_paths = sorted(prepared_dir.glob("*.npz"))
if not npz_paths:
    raise SystemExit(f"[ERROR] No .npz clips found in prepared bank: {prepared_dir}")

expected_urdf_by_clip = {
    npz_path.stem: (generated_urdf_dir / f"{npz_path.stem}.urdf").resolve()
    for npz_path in npz_paths
}
missing_urdfs = [str(path) for path in expected_urdf_by_clip.values() if not path.is_file()]
if missing_urdfs:
    preview = ", ".join(missing_urdfs[:10])
    raise SystemExit(f"[ERROR] Missing generated URDF(s) for prepared clips: {preview}")

raw_payload = json.loads(map_path.read_text(encoding="utf-8"))
if isinstance(raw_payload, dict) and isinstance(raw_payload.get("clips"), dict):
    raw_clips = raw_payload["clips"]
    new_payload = {key: value for key, value in raw_payload.items() if key != "clips"}
elif isinstance(raw_payload, dict):
    raw_clips = raw_payload
    new_payload = {}
else:
    raise SystemExit(f"[ERROR] Invalid clip-object map payload: {map_path}")

updated_clips: dict[str, dict] = {}
rewritten_npz = 0
rewritten_map_entries = 0

for npz_path in npz_paths:
    clip_id = npz_path.stem
    expected_urdf = str(expected_urdf_by_clip[clip_id])

    with np.load(npz_path, allow_pickle=True) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}

    old_npz_urdf = scalar_str(payload.get("object_urdf_path"))
    old_object_name = scalar_str(payload.get("object_name"))
    npz_changed = old_npz_urdf != expected_urdf
    payload["object_urdf_path"] = np.asarray(expected_urdf)
    if not old_object_name:
        payload["object_name"] = np.asarray(clip_id)
        npz_changed = True

    if npz_changed:
        write_npz_atomic(npz_path, payload)
        rewritten_npz += 1

    raw_entry = raw_clips.get(clip_id, {})
    if isinstance(raw_entry, str):
        entry = {"object_name": "", "object_urdf_path": raw_entry.strip()}
    elif isinstance(raw_entry, dict):
        entry = dict(raw_entry)
    else:
        entry = {}

    old_map_urdf = str(entry.get("object_urdf_path", "")).strip()
    if old_map_urdf != expected_urdf:
        rewritten_map_entries += 1
    if not str(entry.get("object_name", "")).strip():
        entry["object_name"] = scalar_str(payload.get("object_name")) or clip_id
    if "object_size" not in entry:
        size = object_size_list(payload.get("object_size"))
        if size is not None:
            entry["object_size"] = size
    entry["object_urdf_path"] = expected_urdf
    updated_clips[clip_id] = entry

new_payload["clips"] = updated_clips
map_path.write_text(json.dumps(new_payload, indent=2, sort_keys=True), encoding="utf-8")

validated_payload = json.loads(map_path.read_text(encoding="utf-8"))
validated_clips = validated_payload.get("clips", validated_payload)
if not isinstance(validated_clips, dict):
    raise SystemExit(f"[ERROR] Invalid rewritten clip-object map payload: {map_path}")

clip_ids = [path.stem for path in npz_paths]
if set(validated_clips) != set(clip_ids):
    missing = sorted(set(clip_ids) - set(validated_clips))
    extra = sorted(set(validated_clips) - set(clip_ids))
    raise SystemExit(
        f"[ERROR] Rewritten object map clip set mismatch. missing={missing[:10]} extra={extra[:10]}"
    )

validation_errors: list[str] = []
for npz_path in npz_paths:
    clip_id = npz_path.stem
    expected_urdf = str(expected_urdf_by_clip[clip_id])
    expected_name = f"{clip_id}.urdf"

    map_entry = validated_clips[clip_id]
    map_urdf = map_entry.strip() if isinstance(map_entry, str) else str(map_entry.get("object_urdf_path", "")).strip()
    if map_urdf != expected_urdf:
        validation_errors.append(f"{clip_id}: map object_urdf_path is not canonical: {map_urdf}")
    elif Path(map_urdf).name != expected_name or not Path(map_urdf).is_file():
        validation_errors.append(f"{clip_id}: map object_urdf_path does not resolve to generated URDF: {map_urdf}")

    with np.load(npz_path, allow_pickle=True) as data:
        npz_urdf = scalar_str(data["object_urdf_path"]) if "object_urdf_path" in data else ""
    if npz_urdf != expected_urdf:
        validation_errors.append(f"{clip_id}: npz object_urdf_path is not canonical: {npz_urdf}")
    elif Path(npz_urdf).name != expected_name or not Path(npz_urdf).is_file():
        validation_errors.append(f"{clip_id}: npz object_urdf_path does not resolve to generated URDF: {npz_urdf}")

if validation_errors:
    preview = "\n  ".join(validation_errors[:10])
    raise SystemExit(f"[ERROR] Prepared-bank object path validation failed:\n  {preview}")

print("[INFO] Canonicalized prepared DS bank object paths.")
print(f"[INFO]   prepared_dir={prepared_dir}")
print(f"[INFO]   clips={len(npz_paths)}")
print(f"[INFO]   rewritten_npz={rewritten_npz}")
print(f"[INFO]   rewritten_map_entries={rewritten_map_entries}")
PY
}

build_omomo_box_mixed_bank() {
  local local_root="$1"
  local omomo_root="$2"

  python - "${local_root}" "${omomo_root}" <<'PY'
import json
import os
import re
import shutil
import sys
from pathlib import Path


def load_clip_map(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips", payload) if isinstance(payload, dict) else {}
    if not isinstance(clips, dict):
        raise SystemExit(f"[ERROR] Invalid clip-object map: {path}")
    return clips


def normalize_entry(entry, source_dir: Path) -> dict:
    if isinstance(entry, str):
        normalized = {"object_urdf_path": entry}
    elif isinstance(entry, dict):
        normalized = dict(entry)
    else:
        normalized = {}

    for key in ("object_urdf_path", "object_mesh_path"):
        raw = str(normalized.get(key, "")).strip()
        if not raw:
            continue
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (source_dir / path).resolve()
        normalized[key] = str(path)
    return normalized


def symlink_or_copy(src: Path, dst: Path) -> None:
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


local_root = Path(sys.argv[1]).expanduser().resolve()
omomo_root = Path(sys.argv[2]).expanduser().resolve()
prepared_dir = local_root / "train_g1_w_obj_prepared"
out_dir = local_root / "train_g1_w_obj_prepared_plus_omomo_orig"

if not prepared_dir.is_dir():
    raise SystemExit(f"[ERROR] Missing local prepared DS bank: {prepared_dir}")
if not omomo_root.is_dir():
    raise SystemExit(f"[ERROR] Missing OMOMO prepared bank: {omomo_root}")

ds_map_path = prepared_dir / "_clip_object_urdf_map.json"
omomo_map_path = omomo_root / "_clip_object_urdf_map.json"
if not ds_map_path.is_file():
    raise SystemExit(f"[ERROR] Missing DS object map: {ds_map_path}")
if not omomo_map_path.is_file():
    raise SystemExit(f"[ERROR] Missing OMOMO object map: {omomo_map_path}")

if out_dir.exists() or out_dir.is_symlink():
    if out_dir.is_dir() and not out_dir.is_symlink():
        shutil.rmtree(out_dir)
    else:
        out_dir.unlink()
out_dir.mkdir(parents=True, exist_ok=True)

ds_clips = load_clip_map(ds_map_path)
omomo_clips = load_clip_map(omomo_map_path)

box_files = sorted(prepared_dir.glob("box_*.npz"))
omomo_files = [
    path
    for path in sorted(omomo_root.glob("sub*_mj_w_obj.npz"))
    if not re.search(r"_(rot|trans)_\d+_mj_w_obj$", path.stem)
]
if not box_files:
    raise SystemExit(f"[ERROR] No box_*.npz clips found in DS prepared bank: {prepared_dir}")
if not omomo_files:
    raise SystemExit(f"[ERROR] No base sub*_mj_w_obj.npz OMOMO clips found in: {omomo_root}")

selected_map: dict[str, dict] = {}
missing_entries: list[str] = []

for src_npz in box_files:
    clip_id = src_npz.stem
    if clip_id not in ds_clips:
        missing_entries.append(clip_id)
        continue
    symlink_or_copy(src_npz, out_dir / src_npz.name)
    selected_map[clip_id] = normalize_entry(ds_clips[clip_id], prepared_dir)

for src_npz in omomo_files:
    clip_id = src_npz.stem
    if clip_id not in omomo_clips:
        missing_entries.append(clip_id)
        continue
    symlink_or_copy(src_npz, out_dir / src_npz.name)
    selected_map[clip_id] = normalize_entry(omomo_clips[clip_id], omomo_root)

if missing_entries:
    preview = ", ".join(missing_entries[:10])
    raise SystemExit(f"[ERROR] Missing object-map entries for mixed bank clips: {preview}")

payload = {
    "clips": selected_map,
    "mixed_bank": {
        "source_ds_prepared_bank": str(prepared_dir),
        "source_omomo_prepared_bank": str(omomo_root),
        "include": "base OMOMO sub* clips plus current DS box_* clips",
        "exclude": "behave_* and lc_* clips",
        "counts": {
            "box": len(box_files),
            "omomo": len(omomo_files),
            "total": len(selected_map),
        },
    },
}
(out_dir / "_clip_object_urdf_map.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True),
    encoding="utf-8",
)

print("[INFO] Rebuilt OMOMO+box mixed source bank.")
print(f"[INFO]   out_dir={out_dir}")
print(f"[INFO]   omomo={len(omomo_files)} box={len(box_files)} total={len(selected_map)}")
PY
}

if [[ "${CANONICALIZE_ONLY}" == "1" ]]; then
  echo "[INFO] Canonicalize-only mode; leaving existing local DS tree in place."
  canonicalize_and_validate_prepared_bank "${LOCAL_DS_ROOT}"
  if [[ "${BUILD_OMOMO_MIXED_BANK}" == "1" ]]; then
    if [[ ! -d "${NFS_OMOMO_PREPARED_ROOT}" ]]; then
      echo "[ERROR] NFS_OMOMO_PREPARED_ROOT does not exist: ${NFS_OMOMO_PREPARED_ROOT}" >&2
      exit 1
    fi
    build_omomo_box_mixed_bank "${LOCAL_DS_ROOT}" "${NFS_OMOMO_PREPARED_ROOT}"
  fi
  exit 0
fi

if [[ ! -d "${NFS_DS_ROOT}" ]]; then
  echo "[ERROR] NFS_DS_ROOT does not exist: ${NFS_DS_ROOT}" >&2
  exit 1
fi
if [[ "${BUILD_OMOMO_MIXED_BANK}" == "1" && ! -d "${NFS_OMOMO_PREPARED_ROOT}" ]]; then
  echo "[ERROR] NFS_OMOMO_PREPARED_ROOT does not exist: ${NFS_OMOMO_PREPARED_ROOT}" >&2
  exit 1
fi

for required in train_g1_w_obj train_g1_w_obj_geometry train_g1_w_obj_prepared; do
  if [[ ! -e "${NFS_DS_ROOT%/}/${required}" ]]; then
    echo "[ERROR] NFS_DS_ROOT is missing expected entry: ${NFS_DS_ROOT%/}/${required}" >&2
    exit 1
  fi
done

echo "[INFO] Refreshing local DS data"
echo "[INFO]   source: ${NFS_DS_ROOT}"
echo "[INFO]   target: ${LOCAL_DS_ROOT} (${REPO_DS_RELATIVE_PATH} under this repo)"
echo "[INFO]   mode  : delete target, rsync source, canonicalize object paths"
if [[ "${BUILD_OMOMO_MIXED_BANK}" == "1" ]]; then
  echo "[INFO]   mixed : rebuild train_g1_w_obj_prepared_plus_omomo_orig from fresh box_* + OMOMO"
  echo "[INFO]   omomo : ${NFS_OMOMO_PREPARED_ROOT}"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[DRY_RUN] rm -rf ${LOCAL_DS_ROOT}"
  echo "[DRY_RUN] mkdir -p ${LOCAL_DS_ROOT}"
  echo "[DRY_RUN] rsync -aH --info=progress2 ${NFS_DS_ROOT%/}/ ${LOCAL_DS_ROOT%/}/"
  echo "[DRY_RUN] canonicalize and strictly validate train_g1_w_obj_prepared object paths"
  if [[ "${BUILD_OMOMO_MIXED_BANK}" == "1" ]]; then
    echo "[DRY_RUN] rebuild ${LOCAL_DS_ROOT%/}/train_g1_w_obj_prepared_plus_omomo_orig from current box_* clips and ${NFS_OMOMO_PREPARED_ROOT}"
  fi
  exit 0
fi

rm -rf -- "${LOCAL_DS_ROOT}"
mkdir -p -- "${LOCAL_DS_ROOT}"
rsync -aH --info=progress2 "${NFS_DS_ROOT%/}/" "${LOCAL_DS_ROOT%/}/"
canonicalize_and_validate_prepared_bank "${LOCAL_DS_ROOT}"
if [[ "${BUILD_OMOMO_MIXED_BANK}" == "1" ]]; then
  build_omomo_box_mixed_bank "${LOCAL_DS_ROOT}" "${NFS_OMOMO_PREPARED_ROOT}"
fi

echo "[INFO] Local DS data refreshed."
find "${LOCAL_DS_ROOT}" -maxdepth 2 -type d | sort | sed -n '1,80p'
