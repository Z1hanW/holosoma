#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=${SRC:-"/nfs/zzzihanw/omomo_45"}
DST=${DST:-"${SCRIPT_DIR}/data/ds_as_data/omomo"}
EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-45}

validate_bank() {
  local bank_dir=$1
  python3 - "${bank_dir}" "${EXPECTED_TOTAL}" <<'PY'
import json
import sys
import numpy as np
from pathlib import Path

bank_dir = Path(sys.argv[1]).expanduser().resolve()
expected = int(sys.argv[2])
map_path = bank_dir / "_clip_object_urdf_map.json"

if not bank_dir.is_dir():
    raise SystemExit(f"[ERROR] OMOMO bank directory does not exist: {bank_dir}")
if not map_path.is_file():
    raise SystemExit(f"[ERROR] Missing object map: {map_path}")

npz_files = sorted(bank_dir.glob("*.npz"))
if len(npz_files) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} .npz files in {bank_dir}, found {len(npz_files)}")

payload = json.loads(map_path.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clips, dict):
    raise SystemExit(f"[ERROR] Invalid object map format: {map_path}")
if len(clips) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} map entries in {map_path}, found {len(clips)}")

def resolve_relative(raw: str, label: str, source: str) -> Path:
    raw = str(raw or "").strip()
    if not raw:
        raise ValueError(f"{source}: empty {label}")
    path = Path(raw)
    if path.is_absolute():
        raise ValueError(f"{source}: {label} must be relative for a self-contained bank, got {raw}")
    resolved = (bank_dir / path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{source}: missing {label} {resolved}")
    return resolved

errors = []
for npz_path in npz_files:
    clip_id = npz_path.stem
    entry = clips.get(clip_id)
    if not isinstance(entry, dict):
        errors.append(f"{clip_id}: missing or invalid object-map entry")
        continue
    try:
        resolve_relative(entry.get("object_urdf_path", ""), "map object_urdf_path", clip_id)
        resolve_relative(entry.get("object_mesh_path", ""), "map object_mesh_path", clip_id)
    except Exception as exc:
        errors.append(str(exc))

    try:
        with np.load(npz_path, allow_pickle=True) as data:
            for key in ("object_urdf_path", "object_mesh_path"):
                if key not in data.files:
                    errors.append(f"{clip_id}: missing npz {key}")
                    continue
                value = np.asarray(data[key]).item()
                resolve_relative(value, f"npz {key}", clip_id)
    except Exception as exc:
        errors.append(f"{clip_id}: failed to validate npz metadata: {exc}")

if errors:
    raise SystemExit("[ERROR] OMOMO bank validation failed:\n  " + "\n  ".join(errors[:30]))

unique_objects = sorted({str(entry.get("object_name", "")).strip() for entry in clips.values()})
print(f"[INFO] Validated self-contained OMOMO bank: {bank_dir} ({len(npz_files)} clips, objects={unique_objects})")
PY
}

if [[ ! -d "$SRC" ]]; then
  echo "[ERROR] Source directory not found: $SRC" >&2
  exit 1
fi

validate_bank "$SRC"

mkdir -p "$DST"

if command -v rsync >/dev/null 2>&1; then
  rsync -a --info=progress2 "$SRC"/ "$DST"/
else
  cp -a "$SRC"/. "$DST"/
fi

validate_bank "$DST"

echo "Copied $SRC -> $DST"
