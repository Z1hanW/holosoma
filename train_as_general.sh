#!/usr/bin/env bash
set -euo pipefail

# OMOMO/AS real-mesh object generalist training.
#
# This is a thin launcher over train_object_generalist_ds.sh for the real OMOMO
# bank copied by cp_real.sh. It intentionally disables primitive/box object
# spawning: both Isaac Sim object spawning and optional perception geometry must
# use the object URDF mesh assets.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/omomo"}
OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${OMOMO_DATA_DIR}/_clip_object_urdf_map.json"}
OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-45}

LOCAL_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data")
OMOMO_DATA_DIR=$(realpath -m "${OMOMO_DATA_DIR}")
OMOMO_OBJECT_MAP=$(realpath -m "${OMOMO_OBJECT_MAP}")

case "${OMOMO_DATA_DIR}" in
  /nfs|/nfs/*)
    echo "[ERROR] OMOMO_DATA_DIR must be local, not NFS: ${OMOMO_DATA_DIR}" >&2
    echo "[ERROR] Run ./cp_real.sh first and train from ${SCRIPT_DIR}/data/ds_as_data/omomo." >&2
    exit 2
    ;;
esac
case "${OMOMO_DATA_DIR}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] OMOMO_DATA_DIR must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${OMOMO_DATA_DIR}" >&2
    exit 2
    ;;
esac
case "${OMOMO_OBJECT_MAP}" in
  /nfs|/nfs/*)
    echo "[ERROR] OMOMO_OBJECT_MAP must be local, not NFS: ${OMOMO_OBJECT_MAP}" >&2
    echo "[ERROR] Run ./cp_real.sh first and use the copied map under ${SCRIPT_DIR}/data." >&2
    exit 2
    ;;
esac
case "${OMOMO_OBJECT_MAP}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] OMOMO_OBJECT_MAP must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${OMOMO_OBJECT_MAP}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${OMOMO_DATA_DIR}" ]]; then
  echo "[ERROR] OMOMO_DATA_DIR does not exist: ${OMOMO_DATA_DIR}" >&2
  echo "[ERROR] Run ./cp_real.sh first, or set OMOMO_DATA_DIR to a prepared motion bank." >&2
  exit 2
fi

if ! compgen -G "${OMOMO_DATA_DIR}/*.npz" >/dev/null; then
  echo "[ERROR] No .npz files found in OMOMO_DATA_DIR: ${OMOMO_DATA_DIR}" >&2
  echo "[ERROR] Run ./cp_real.sh first, or set OMOMO_DATA_DIR to a prepared motion bank." >&2
  exit 2
fi

if [[ ! -f "${OMOMO_OBJECT_MAP}" ]]; then
  echo "[ERROR] Missing clip-object URDF map: ${OMOMO_OBJECT_MAP}" >&2
  exit 2
fi

OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE:-urdf}
OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-mesh}
case "$(echo "${OBJECT_SPAWN_MODE}" | tr '[:upper:]' '[:lower:]')" in
  urdf|mesh)
    OBJECT_SPAWN_MODE=urdf
    ;;
  *)
    echo "[ERROR] train_as_general.sh requires real URDF mesh spawning." >&2
    echo "[ERROR] Do not use primitive/box mode here. Got OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE}" >&2
    exit 2
    ;;
esac
case "$(echo "${OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
  mesh|urdf|off|disable|disabled|0|false|no)
    OBJECT_GEOMETRY_MODE=mesh
    ;;
  *)
    echo "[ERROR] train_as_general.sh requires mesh object geometry." >&2
    echo "[ERROR] Do not use primitive/box geometry here. Got OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE}" >&2
    exit 2
    ;;
esac

python3 - "${OMOMO_DATA_DIR}" "${OMOMO_OBJECT_MAP}" "${OMOMO_EXPECTED_TOTAL}" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

motion_dir = Path(sys.argv[1]).expanduser().resolve()
map_path = Path(sys.argv[2]).expanduser().resolve()
expected_raw = sys.argv[3].strip()
expected = int(expected_raw) if expected_raw else None

npz_files = sorted(motion_dir.glob("*.npz"))
if expected is not None and len(npz_files) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} .npz clips under {motion_dir}, found {len(npz_files)}")
if not npz_files:
    raise SystemExit(f"[ERROR] No .npz clips found under {motion_dir}")

payload = json.loads(map_path.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clips, dict) or not clips:
    raise SystemExit(f"[ERROR] Invalid or empty object map: {map_path}")
if expected is not None and len(clips) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} object-map entries in {map_path}, found {len(clips)}")

missing_entries = [p.stem for p in npz_files if p.stem not in clips]
if missing_entries:
    preview = ", ".join(missing_entries[:10])
    raise SystemExit(f"[ERROR] Missing object-map entries for {len(missing_entries)} clip(s): {preview}")

def resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()

bad = []
unique_urdfs = {}
for clip_id, entry in clips.items():
    if not isinstance(entry, dict):
        bad.append(f"{clip_id}: map entry is not a dict")
        continue
    urdf_path = resolve_path(entry.get("object_urdf_path", ""), map_path.parent)
    mesh_path_raw = str(entry.get("object_mesh_path", "")).strip()
    mesh_path = resolve_path(mesh_path_raw, map_path.parent) if mesh_path_raw else None
    if not urdf_path.is_file():
        bad.append(f"{clip_id}: missing URDF {urdf_path}")
        continue
    if mesh_path is not None and not mesh_path.is_file():
        bad.append(f"{clip_id}: missing mesh {mesh_path}")
    unique_urdfs[str(urdf_path)] = clip_id

for urdf_raw, clip_id in sorted(unique_urdfs.items()):
    urdf_path = Path(urdf_raw)
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        bad.append(f"{clip_id}: failed to parse URDF {urdf_path}: {exc}")
        continue
    mesh_tags = root.findall(".//mesh")
    if not mesh_tags:
        bad.append(f"{clip_id}: URDF has no <mesh> geometry: {urdf_path}")
        continue
    for tag in mesh_tags:
        filename = str(tag.get("filename", "")).strip()
        if not filename:
            bad.append(f"{clip_id}: URDF mesh tag has empty filename: {urdf_path}")
            continue
        mesh_path = resolve_path(filename, urdf_path.parent)
        if not mesh_path.is_file():
            bad.append(f"{clip_id}: URDF mesh file missing: {mesh_path}")

if bad:
    raise SystemExit("[ERROR] Real-mesh OMOMO validation failed:\n  " + "\n  ".join(bad[:20]))

print(
    f"[INFO] Validated real-mesh OMOMO bank: {motion_dir} "
    f"({len(npz_files)} clips, {len(unique_urdfs)} unique URDF mesh asset(s))"
)
PY

export DATA_MODE=pure-real
export DS_DATA_ROOT="${SCRIPT_DIR}/data/ds_as_data"
export MOTION_DIR="${OMOMO_DATA_DIR}"
export OBJECT_SPEC_PATH="${OMOMO_OBJECT_MAP}"
export AUTO_PREP_DS_BANK=0
export STRICT_DEFAULT_DS_BANK_VALIDATION=0

export OBJECT_SPAWN_MODE
export OBJECT_GEOMETRY_MODE
export HOLOSOMA_OBJECT_SPAWN_MODE="${OBJECT_SPAWN_MODE}"
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${OBJECT_GEOMETRY_MODE}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}

export EXP=${EXP:-g1-29dof-wbt-w-object-generalist}
export SEQUENCE_NAME=${SEQUENCE_NAME:-omomo-real-mesh-cotrack}
export WANDB_PROJECT=${WANDB_PROJECT:-boxer}
export CLIP_WEIGHTING_STRATEGY=${CLIP_WEIGHTING_STRATEGY:-uniform_clip}

# Loose co-tracking defaults: keep the robot feasible while emphasizing object tracking.
export ROOT_POS_W=${ROOT_POS_W:-0.5}
export ROOT_ORI_W=${ROOT_ORI_W:-0.5}
export FULL_BODY_POS_W=${FULL_BODY_POS_W:-1.0}
export FULL_BODY_ORI_W=${FULL_BODY_ORI_W:-1.0}
export FULL_BODY_LIN_VEL_W=${FULL_BODY_LIN_VEL_W:-0.75}
export FULL_BODY_ANG_VEL_W=${FULL_BODY_ANG_VEL_W:-0.75}
export OBJECT_POS_W=${OBJECT_POS_W:-1.5}
export OBJECT_ORI_W=${OBJECT_ORI_W:-1.25}
export ROOT_POS_SIGMA=${ROOT_POS_SIGMA:-0.45}
export ROOT_ORI_SIGMA=${ROOT_ORI_SIGMA:-0.6}
export FULL_BODY_POS_SIGMA=${FULL_BODY_POS_SIGMA:-0.45}
export FULL_BODY_ORI_SIGMA=${FULL_BODY_ORI_SIGMA:-0.6}
export FULL_BODY_LIN_VEL_SIGMA=${FULL_BODY_LIN_VEL_SIGMA:-1.5}
export FULL_BODY_ANG_VEL_SIGMA=${FULL_BODY_ANG_VEL_SIGMA:-4.5}
export OBJECT_POS_SIGMA=${OBJECT_POS_SIGMA:-0.45}
export OBJECT_ORI_SIGMA=${OBJECT_ORI_SIGMA:-0.6}

echo "[INFO] Launching AS/OMOMO real-mesh co-tracking generalist training"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH}"
echo "[INFO] HOLOSOMA_OBJECT_SPAWN_MODE=${HOLOSOMA_OBJECT_SPAWN_MODE}"
echo "[INFO] HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE}"
echo "[INFO] HOLOSOMA_OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE}"

exec bash "${SCRIPT_DIR}/train_object_generalist_ds.sh" pure-real "$@"
