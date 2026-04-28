#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NFS_DS_SCALE_ROOT=${NFS_DS_SCALE_ROOT:-/nfs/zzzihanw/ds_box_data/scale_mix_all}
NFS_OMOMO_PREPARED_ROOT=${NFS_OMOMO_PREPARED_ROOT:-/nfs/zzzihanw/ds_box_data_v2_apr_15/train_g1_w_obj_prepared_plus_omomo_orig}
LOCAL_DS_ROOT=${LOCAL_DS_ROOT:-"${SCRIPT_DIR}/data/ds_box_data"}
LOCAL_SCALE_ROOT="${LOCAL_DS_ROOT%/}/scale_mix_all"
LOCAL_MIXED_ROOT="${LOCAL_SCALE_ROOT}/train_g1_w_obj_prepared_plus_omomo_orig"
LARGEBOX_URDF=${LARGEBOX_URDF:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"}
CLEAN_OUT=${CLEAN_OUT:-1}

if [[ ! -d "${NFS_DS_SCALE_ROOT}" ]]; then
  echo "[ERROR] NFS_DS_SCALE_ROOT not found: ${NFS_DS_SCALE_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${NFS_OMOMO_PREPARED_ROOT}" ]]; then
  echo "[ERROR] NFS_OMOMO_PREPARED_ROOT not found: ${NFS_OMOMO_PREPARED_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${LARGEBOX_URDF}" ]]; then
  echo "[ERROR] LARGEBOX_URDF not found: ${LARGEBOX_URDF}" >&2
  exit 1
fi

SIM_PYTHON=/home/ubuntu/miniconda3/envs/sim/bin/python
if [[ -z "${PYTHON_BIN:-}" && -x "${SIM_PYTHON}" ]]; then
  PYTHON_BIN="${SIM_PYTHON}"
else
  PYTHON_BIN=${PYTHON_BIN:-python}
fi

mkdir -p "${LOCAL_DS_ROOT}"
if [[ "${CLEAN_OUT}" == "1" ]]; then
  rm -rf "${LOCAL_SCALE_ROOT}"
fi
mkdir -p "${LOCAL_SCALE_ROOT}"

echo "[INFO] Syncing DS scale_mix_all:"
echo "[INFO]   from: ${NFS_DS_SCALE_ROOT}"
echo "[INFO]   to  : ${LOCAL_SCALE_ROOT}"
rsync -avh --delete "${NFS_DS_SCALE_ROOT}/" "${LOCAL_SCALE_ROOT}/"

echo "[INFO] Adding 62 original base OMOMO clips into local mixed bank:"
echo "[INFO]   from: ${NFS_OMOMO_PREPARED_ROOT}"
echo "[INFO]   to  : ${LOCAL_MIXED_ROOT}"

"${PYTHON_BIN}" - "${SCRIPT_DIR}" "${LOCAL_MIXED_ROOT}" "${NFS_OMOMO_PREPARED_ROOT}" "${LARGEBOX_URDF}" <<'PY'
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

repo = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
omomo_prepared = Path(sys.argv[3]).resolve()
largebox_urdf = Path(sys.argv[4]).resolve()

expected_omomo_count = 62
expected_ds_count = 712
expected_total = expected_ds_count + expected_omomo_count
largebox_size = [0.47115421295166016, 0.45873013138771057, 0.4078954756259918]

target_map_path = target / "_clip_object_urdf_map.json"
source_map_path = omomo_prepared / "_clip_object_urdf_map.json"

if not target.is_dir():
    raise SystemExit(f"[ERROR] Local mixed bank missing after rsync: {target}")
if not target_map_path.is_file():
    raise SystemExit(f"[ERROR] Local mixed bank map missing: {target_map_path}")
if not source_map_path.is_file():
    raise SystemExit(f"[ERROR] OMOMO source map missing: {source_map_path}")
if not largebox_urdf.is_file():
    raise SystemExit(f"[ERROR] largebox URDF missing: {largebox_urdf}")


def load_clips(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    clips = payload.get("clips", payload) if isinstance(payload, dict) else {}
    if not isinstance(clips, dict):
        raise ValueError(f"Invalid clip-object map: {path}")
    return clips


def save_npz_exact(path: Path, payload: dict[str, np.ndarray]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp_npz = Path(str(tmp) + ".npz")
    for candidate in (tmp, tmp_npz):
        if candidate.exists():
            candidate.unlink()
    with tmp.open("wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)


target_clips = load_clips(target_map_path)
source_clips = load_clips(source_map_path)

for stale in target.glob("sub*_mj_w_obj.npz"):
    stale.unlink()

omomo_files = sorted(omomo_prepared.glob("sub*_mj_w_obj.npz"))
omomo_files = [
    path
    for path in omomo_files
    if not re.search(r"_(rot|trans)_\d+_mj_w_obj$", path.stem)
]
if len(omomo_files) != expected_omomo_count:
    raise SystemExit(
        f"[ERROR] Expected {expected_omomo_count} base OMOMO clips, found {len(omomo_files)} in {omomo_prepared}"
    )

rewritten_ds_npz = 0
for npz_path in sorted(target.glob("*.npz")):
    if npz_path.stem.startswith("sub"):
        continue

    urdf_path = target / "_generated_urdfs" / f"{npz_path.stem}.urdf"
    if not urdf_path.is_file():
        raise SystemExit(f"[ERROR] Missing local DS URDF for {npz_path.stem}: {urdf_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}

    desired_urdf = str(urdf_path.resolve())
    current_urdf = ""
    if "object_urdf_path" in payload:
        current_urdf = str(np.asarray(payload["object_urdf_path"]).item())
    if current_urdf != desired_urdf:
        payload["object_urdf_path"] = np.array(desired_urdf)
        save_npz_exact(npz_path, payload)
        rewritten_ds_npz += 1

    entry = target_clips.get(npz_path.stem, {})
    if not isinstance(entry, dict):
        entry = {"object_name": npz_path.stem}
    entry = dict(entry)
    entry["object_urdf_path"] = desired_urdf
    if not str(entry.get("object_name", "")).strip():
        entry["object_name"] = npz_path.stem
    if "object_size" not in entry and "object_size" in payload:
        entry["object_size"] = [float(v) for v in np.asarray(payload["object_size"]).reshape(-1).tolist()]
    target_clips[npz_path.stem] = entry

copied_omomo = 0
for src in omomo_files:
    dst = target / src.name
    shutil.copy2(src, dst)

    stem = src.stem
    entry = source_clips.get(stem, {})
    if not isinstance(entry, dict):
        entry = {}
    entry = dict(entry)
    entry["object_name"] = str(entry.get("object_name") or "largebox")
    entry["object_urdf_path"] = str(largebox_urdf)
    entry["object_size"] = entry.get("object_size", largebox_size)
    target_clips[stem] = entry
    copied_omomo += 1

output_payload = {
    "clips": dict(sorted(target_clips.items())),
    "notes": "Generated by cp_box.sh: DS scale_mix_all plus 62 original base OMOMO carry clips.",
    "ds_source": "/nfs/zzzihanw/ds_box_data/scale_mix_all",
    "omomo_source": str(omomo_prepared),
}
tmp_map = target_map_path.with_name(target_map_path.name + ".tmp")
tmp_map.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp_map, target_map_path)

npz_files = sorted(target.glob("*.npz"))
sub_count = sum(1 for path in npz_files if path.stem.startswith("sub"))
ds_count = len(npz_files) - sub_count
clips = load_clips(target_map_path)

missing_map = [path.stem for path in npz_files if path.stem not in clips]
if missing_map:
    raise SystemExit(f"[ERROR] Map missing {len(missing_map)} active clip entries, sample={missing_map[:10]}")

missing_fields: list[tuple[str, str]] = []
missing_urdf: list[tuple[str, str]] = []
for npz_path in npz_files:
    with np.load(npz_path, allow_pickle=True) as data:
        for key in ("object_pos_w", "object_quat_w", "object_size", "object_name", "object_urdf_path"):
            if key not in data.files:
                missing_fields.append((npz_path.name, key))
                break

    entry = clips.get(npz_path.stem, {})
    urdf = Path(str(entry.get("object_urdf_path", ""))).expanduser()
    if not urdf.is_file():
        missing_urdf.append((npz_path.stem, str(urdf)))

if missing_fields:
    raise SystemExit(f"[ERROR] Missing object fields in npz files, sample={missing_fields[:10]}")
if missing_urdf:
    raise SystemExit(f"[ERROR] Missing URDFs in map, sample={missing_urdf[:10]}")
if ds_count != expected_ds_count or sub_count != expected_omomo_count or len(clips) != expected_total:
    raise SystemExit(
        f"[ERROR] Unexpected final counts: ds={ds_count}, omomo={sub_count}, map={len(clips)}; "
        f"expected ds={expected_ds_count}, omomo={expected_omomo_count}, total={expected_total}"
    )

print(f"[INFO] Rewritten DS npz URDF paths: {rewritten_ds_npz}")
print(f"[INFO] Copied original base OMOMO clips: {copied_omomo}")
print(f"[INFO] Final local mixed bank: {target}")
print(f"[INFO] Final counts: npz={len(npz_files)}, ds={ds_count}, omomo={sub_count}, map={len(clips)}")
PY

echo "[INFO] cp_box.sh complete. Training should read from local data/ds_box_data only."
