#!/usr/bin/env bash
set -euo pipefail

# Copy CoRL baseline real-data banks into repo-local data/.
# Training scripts should read only from data/, never directly from /nfs.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

NFS_CORL_ROOT=${NFS_CORL_ROOT:-/nfs/zzzihanw/corl}
LOCAL_CORL_ROOT=${LOCAL_CORL_ROOT:-data/corl_numbers}
OMOMO_BANK=omomo_z0p4_nofoot_bimanual161_training_ready
BEHAVE_BANK=behave_z0p4_first_lift_run_bimanual56_w_obj_training_ready

LOCAL_CORL_ROOT_ABS=$(python3 - "${LOCAL_CORL_ROOT}" "${SCRIPT_DIR}/data" <<'PY'
from pathlib import Path
import sys

dst = Path(sys.argv[1]).expanduser().resolve()
data = Path(sys.argv[2]).expanduser().resolve()
if dst != data and data not in dst.parents:
    raise SystemExit(f"[ERROR] LOCAL_CORL_ROOT must be under repo-local data/: {dst}")
print(dst)
PY
)

copy_bank() {
  local bank_name=$1
  local src="${NFS_CORL_ROOT%/}/${bank_name}"
  local dst="${LOCAL_CORL_ROOT_ABS}/${bank_name}"

  if [[ ! -d "${src}" ]]; then
    echo "[ERROR] Missing source bank: ${src}" >&2
    exit 2
  fi
  if [[ ! -f "${src}/_clip_object_urdf_map.json" ]]; then
    echo "[ERROR] Missing source object map: ${src}/_clip_object_urdf_map.json" >&2
    exit 2
  fi

  mkdir -p "${dst}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --info=progress2 "${src}/" "${dst}/"
  else
    rm -rf "${dst}"
    mkdir -p "${dst}"
    cp -a "${src}/." "${dst}/"
  fi
}

copy_bank "${OMOMO_BANK}"
copy_bank "${BEHAVE_BANK}"

python3 - "${LOCAL_CORL_ROOT_ABS}" "${OMOMO_BANK}" "${BEHAVE_BANK}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
expected = {
    sys.argv[2]: 161,
    sys.argv[3]: 56,
}

total = 0
for bank_name, expected_count in expected.items():
    bank_dir = root / bank_name
    map_path = bank_dir / "_clip_object_urdf_map.json"
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    clips = payload.get("clips", payload) if isinstance(payload, dict) else {}
    npz_files = sorted(bank_dir.glob("*.npz"))
    if len(npz_files) != expected_count:
        raise SystemExit(f"[ERROR] {bank_name}: expected {expected_count} .npz files, found {len(npz_files)}")
    if len(clips) != expected_count:
        raise SystemExit(f"[ERROR] {bank_name}: expected {expected_count} map entries, found {len(clips)}")
    missing = [path.stem for path in npz_files if path.stem not in clips]
    if missing:
        raise SystemExit(f"[ERROR] {bank_name}: missing object-map entries: {missing[:10]}")
    total += len(npz_files)
    print(f"[INFO] Validated {bank_name}: {len(npz_files)} clips")

print(f"[INFO] Baseline CoRL data ready under {root} ({total} clips total)")
PY
