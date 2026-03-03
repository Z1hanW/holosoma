#!/usr/bin/env bash
set -euo pipefail

# Build a mixed motion bank for object-tracking training from:
# - OMOMO converted clips
# - BEHAVE converted clips (optionally filtered by object keywords)
#
# Default output:
#   src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_mix_ml
#
# Usage:
#   bash prepare_mixed_object_bank.sh
#
# Optional env vars:
#   OMOMO_DIR=/abs/path/to/omomo_carry_dir
#   BEHAVE_DIR=/abs/path/to/behave_dir
#   OUT_DIR=/abs/path/to/output_dir
#   BEHAVE_FILTER=boxmedium,boxlarge
#   LINK_MODE=symlink|copy
#   CLEAN_OUT=1|0
#   PREFIX_DATASET=1|0
#   OMOMO_OBJECT_NAME=largebox
#   OMOMO_URDF=/abs/path/to/objects_largebox.urdf
#   BEHAVE_URDF_ROOT=/abs/path/to/src/holosoma_retargeting/models/behave_objects
#   BEHAVE_MAP_FILE=/abs/path/to/_clip_object_urdf_map.json

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

OMOMO_DIR=${OMOMO_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"}
BEHAVE_DIR=${BEHAVE_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry"}
OUT_DIR=${OUT_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml"}

BEHAVE_FILTER=${BEHAVE_FILTER:-"boxmedium,boxlarge"}
LINK_MODE=${LINK_MODE:-"symlink"}     # symlink|copy
CLEAN_OUT=${CLEAN_OUT:-1}             # 1: remove output dir first
PREFIX_DATASET=${PREFIX_DATASET:-1}   # 1: prefix filenames with omomo__/behave__
OMOMO_OBJECT_NAME=${OMOMO_OBJECT_NAME:-"largebox"}
OMOMO_URDF=${OMOMO_URDF:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"}
BEHAVE_URDF_ROOT=${BEHAVE_URDF_ROOT:-"${SCRIPT_DIR}/src/holosoma_retargeting/models/behave_objects"}
BEHAVE_MAP_FILE=${BEHAVE_MAP_FILE:-"${BEHAVE_DIR}/_clip_object_urdf_map.json"}
MAP_FILE="${OUT_DIR}/_clip_object_urdf_map.json"

if [[ ! -d "${OMOMO_DIR}" ]]; then
  echo "[ERROR] OMOMO_DIR not found: ${OMOMO_DIR}" >&2
  exit 1
fi
if [[ ! -d "${BEHAVE_DIR}" ]]; then
  echo "[ERROR] BEHAVE_DIR not found: ${BEHAVE_DIR}" >&2
  exit 1
fi
if [[ "${LINK_MODE}" != "symlink" && "${LINK_MODE}" != "copy" ]]; then
  echo "[ERROR] LINK_MODE must be symlink|copy, got: ${LINK_MODE}" >&2
  exit 1
fi
if [[ "${PREFIX_DATASET}" != "1" ]]; then
  echo "[ERROR] PREFIX_DATASET must be 1 for mixed OMOMO+BEHAVE (needed for clip->URDF mapping)." >&2
  exit 1
fi

if [[ "${CLEAN_OUT}" == "1" ]]; then
  rm -rf "${OUT_DIR}"
fi
mkdir -p "${OUT_DIR}"

shopt -s nullglob
omomo_files=("${OMOMO_DIR}"/*_mj_w_obj.npz)
if [[ ${#omomo_files[@]} -eq 0 ]]; then
  echo "[ERROR] No OMOMO files found in ${OMOMO_DIR}" >&2
  exit 1
fi

IFS=',' read -r -a behave_patterns <<< "${BEHAVE_FILTER}"
behave_files=()
for f in "${BEHAVE_DIR}"/*_mj_w_obj.npz; do
  base=$(basename "${f}")
  lower=$(echo "${base}" | tr '[:upper:]' '[:lower:]')
  keep=0
  for p in "${behave_patterns[@]}"; do
    p_trim=$(echo "${p}" | tr -d ' ' | tr '[:upper:]' '[:lower:]')
    if [[ -n "${p_trim}" && "${lower}" == *"${p_trim}"* ]]; then
      keep=1
      break
    fi
  done
  if [[ "${keep}" == "1" ]]; then
    behave_files+=("${f}")
  fi
done

if [[ ${#behave_files[@]} -eq 0 ]]; then
  echo "[ERROR] No BEHAVE files matched BEHAVE_FILTER='${BEHAVE_FILTER}' in ${BEHAVE_DIR}" >&2
  exit 1
fi

link_or_copy() {
  local src="$1"
  local dst="$2"
  if [[ -e "${dst}" ]]; then
    rm -f "${dst}"
  fi
  if [[ "${LINK_MODE}" == "symlink" ]]; then
    ln -s "${src}" "${dst}"
  else
    cp -f "${src}" "${dst}"
  fi
}

for src in "${omomo_files[@]}"; do
  base=$(basename "${src}")
  if [[ "${PREFIX_DATASET}" == "1" ]]; then
    dst="${OUT_DIR}/omomo__${base}"
  else
    dst="${OUT_DIR}/${base}"
  fi
  link_or_copy "${src}" "${dst}"
done

for src in "${behave_files[@]}"; do
  base=$(basename "${src}")
  if [[ "${PREFIX_DATASET}" == "1" ]]; then
    dst="${OUT_DIR}/behave__${base}"
  else
    dst="${OUT_DIR}/${base}"
  fi
  link_or_copy "${src}" "${dst}"
done

python - <<'PY' "${OUT_DIR}" "${OMOMO_OBJECT_NAME}" "${OMOMO_URDF}" "${BEHAVE_URDF_ROOT}" "${BEHAVE_MAP_FILE}" "${MAP_FILE}"
import json
import sys
from pathlib import Path
import numpy as np

out_dir = Path(sys.argv[1])
omomo_object_name = sys.argv[2].strip()
omomo_urdf = str(Path(sys.argv[3]).resolve())
behave_urdf_root = Path(sys.argv[4]).resolve()
behave_map_file = Path(sys.argv[5]).resolve()
map_file = Path(sys.argv[6]).resolve()
files = sorted(out_dir.glob("*.npz"))
if not files:
    raise SystemExit("[ERROR] No files in mixed output dir")

ref_joint_names = None
ref_fps = None
clip_map = {}

def infer_behave_object(stem: str) -> str | None:
    key = stem.lower()
    for obj in ("boxmedium", "boxlarge", "boxsmall", "boxtiny", "boxlong"):
        if obj in key:
            return obj
    return None

behave_clip_map: dict[str, dict[str, str]] = {}
if behave_map_file.is_file():
    try:
        payload = json.loads(behave_map_file.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
            payload = payload["clips"]
        if isinstance(payload, dict):
            for clip, entry in payload.items():
                if not isinstance(clip, str):
                    continue
                if isinstance(entry, str):
                    behave_clip_map[clip] = {"object_name": "", "object_urdf_path": entry.strip()}
                elif isinstance(entry, dict):
                    behave_clip_map[clip] = {
                        "object_name": str(entry.get("object_name", "")).strip(),
                        "object_urdf_path": str(entry.get("object_urdf_path", "")).strip(),
                    }
    except Exception as exc:
        raise SystemExit(f"[ERROR] Failed to parse BEHAVE_MAP_FILE '{behave_map_file}': {exc}")

for p in files:
    with np.load(p, allow_pickle=True) as d:
        if "object_pos_w" not in d:
            raise SystemExit(f"[ERROR] Missing object_pos_w in {p.name}")
        fps = float(np.asarray(d["fps"]).reshape(-1)[0])
        jn = tuple(str(x.decode() if isinstance(x, (bytes, np.bytes_)) else x) for x in np.asarray(d["joint_names"]))
        if ref_joint_names is None:
            ref_joint_names = jn
        elif jn != ref_joint_names:
            raise SystemExit(f"[ERROR] joint_names mismatch: {p.name}")
        if ref_fps is None:
            ref_fps = fps
        elif abs(fps - ref_fps) > 1e-6:
            raise SystemExit(f"[ERROR] fps mismatch: {p.name} ({fps} vs {ref_fps})")

    stem = p.stem
    if stem.startswith("omomo__"):
        clip_map[stem] = {
            "object_name": omomo_object_name,
            "object_urdf_path": omomo_urdf,
        }
    elif stem.startswith("behave__"):
        clip_key = stem[len("behave__") :]
        mapped = behave_clip_map.get(clip_key, {})
        obj = str(mapped.get("object_name", "")).strip()
        urdf_path = str(mapped.get("object_urdf_path", "")).strip()

        if urdf_path:
            urdf = Path(urdf_path)
            if not urdf.is_absolute():
                urdf = (behave_map_file.parent / urdf).resolve()
            if not urdf.is_file():
                raise SystemExit(f"[ERROR] Missing mapped BEHAVE URDF for clip '{stem}': {urdf}")
        else:
            if not obj:
                obj = infer_behave_object(stem) or ""
            if not obj:
                raise SystemExit(f"[ERROR] Failed to infer BEHAVE object from clip name: {stem}")
            urdf = behave_urdf_root / obj / f"{obj}.urdf"
            if not urdf.is_file():
                raise SystemExit(f"[ERROR] Missing BEHAVE URDF for clip '{stem}': {urdf}")

        if not obj:
            obj = urdf.stem
        clip_map[stem] = {
            "object_name": obj,
            "object_urdf_path": str(urdf.resolve()),
        }
    else:
        raise SystemExit(f"[ERROR] Mixed-bank clip must be prefixed by dataset (omomo__/behave__): {stem}")

payload = {
    "clips": clip_map,
    "notes": "Generated by prepare_mixed_object_bank.sh",
}
map_file.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

print(f"[INFO] Mixed bank validated: {len(files)} clips, fps={ref_fps}")
print(f"[INFO] Wrote clip->URDF map: {map_file}")
PY

echo "[INFO] OMOMO clips : ${#omomo_files[@]}"
echo "[INFO] BEHAVE clips: ${#behave_files[@]} (filter=${BEHAVE_FILTER})"
echo "[INFO] Output dir  : ${OUT_DIR}"
echo "[INFO] LINK_MODE   : ${LINK_MODE}"
echo "[INFO] Map file    : ${MAP_FILE}"
