#!/usr/bin/env bash
set -euo pipefail

# AS real-mesh object generalist training.
#
# This is a thin launcher over train_object_generalist_ds.sh for the AS union
# bank copied by cp_as.sh. It intentionally disables primitive/box object
# spawning: both Isaac Sim object spawning and optional perception geometry must
# use the object URDF mesh assets.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

AS_DATA_DIR=${AS_DATA_DIR:-${OMOMO_DATA_DIR:-"data/ds_as_data/omomo"}}
AS_OBJECT_MAP=${AS_OBJECT_MAP:-${OMOMO_OBJECT_MAP:-"${AS_DATA_DIR}/_clip_object_urdf_map.json"}}
AS_EXPECTED_TOTAL=${AS_EXPECTED_TOTAL:-${OMOMO_EXPECTED_TOTAL:-63}}
# Optional override knobs forwarded to train_object_generalist_ds.sh:
#   NUM_ENVS / NPROC / PER_GPU_ENVS / MASTER_PORT
#   TRAINING_SEED or SEED
#   RANDOMIZATION_PRESET or RANDOMIZATION
#   INIT_AT_RANDOM_EP_LEN
#   SAVE_INTERVAL
TRAINING_SEED=${TRAINING_SEED:-${SEED:-}}
RANDOMIZATION_PRESET=${RANDOMIZATION_PRESET:-${RANDOMIZATION:-}}
INIT_AT_RANDOM_EP_LEN=${INIT_AT_RANDOM_EP_LEN:-}

LOCAL_DATA_ROOT=$(realpath -m "data")
AS_DATA_DIR_ABS=$(realpath -m "${AS_DATA_DIR}")
AS_OBJECT_MAP_ABS=$(realpath -m "${AS_OBJECT_MAP}")

case "${AS_DATA_DIR_ABS}" in
  /nfs|/nfs/*)
    echo "[ERROR] AS_DATA_DIR must be local, not NFS: ${AS_DATA_DIR_ABS}" >&2
    echo "[ERROR] Run ./cp_as.sh first and train from data/ds_as_data/omomo." >&2
    exit 2
    ;;
esac
case "${AS_DATA_DIR_ABS}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] AS_DATA_DIR must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${AS_DATA_DIR_ABS}" >&2
    exit 2
    ;;
esac
case "${AS_OBJECT_MAP_ABS}" in
  /nfs|/nfs/*)
    echo "[ERROR] AS_OBJECT_MAP must be local, not NFS: ${AS_OBJECT_MAP_ABS}" >&2
    echo "[ERROR] Run ./cp_as.sh first and use the copied map under data/." >&2
    exit 2
    ;;
esac
case "${AS_OBJECT_MAP_ABS}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] AS_OBJECT_MAP must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${AS_OBJECT_MAP_ABS}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${AS_DATA_DIR_ABS}" ]]; then
  echo "[ERROR] AS_DATA_DIR does not exist: ${AS_DATA_DIR}" >&2
  echo "[ERROR] Run ./cp_as.sh first, or set AS_DATA_DIR to a prepared motion bank under data/." >&2
  exit 2
fi

if ! compgen -G "${AS_DATA_DIR_ABS}/*.npz" >/dev/null; then
  echo "[ERROR] No .npz files found in AS_DATA_DIR: ${AS_DATA_DIR}" >&2
  echo "[ERROR] Run ./cp_as.sh first, or set AS_DATA_DIR to a prepared motion bank under data/." >&2
  exit 2
fi

if [[ ! -f "${AS_OBJECT_MAP_ABS}" ]]; then
  echo "[ERROR] Missing clip-object URDF map: ${AS_OBJECT_MAP}" >&2
  exit 2
fi

OBJECT_SPAWN_MODE_FROM_ENV=0
if [[ -n "${OBJECT_SPAWN_MODE+x}" ]]; then
  OBJECT_SPAWN_MODE_FROM_ENV=1
fi
OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE:-single_slot_multi_urdf}
AS_OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-mesh}
case "$(echo "${AS_OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
  mesh|urdf|off|disable|disabled|0|false|no)
    AS_OBJECT_GEOMETRY_MODE=mesh
    if [[ "${OBJECT_SPAWN_MODE_FROM_ENV}" != "1" ]]; then
      OBJECT_SPAWN_MODE=single_slot_multi_urdf
    fi
    ;;
  *)
    echo "[ERROR] train_as_general.sh requires mesh object geometry." >&2
    echo "[ERROR] Do not use primitive/box geometry here. Got OBJECT_GEOMETRY_MODE=${AS_OBJECT_GEOMETRY_MODE}" >&2
    exit 2
    ;;
esac
case "$(echo "${OBJECT_SPAWN_MODE}" | tr '[:upper:]' '[:lower:]')" in
  urdf|mesh)
    OBJECT_SPAWN_MODE=urdf
    ;;
  single_slot_multi_urdf|single-slot-multi-urdf|single_slot|single-slot|heterogeneous_single_slot|heterogeneous-single-slot)
    OBJECT_SPAWN_MODE=single_slot_multi_urdf
    ;;
  *)
    echo "[ERROR] train_as_general.sh requires real URDF mesh spawning." >&2
    echo "[ERROR] Do not use primitive/box mode here. Got OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE}" >&2
    exit 2
    ;;
esac

python3 - "${AS_DATA_DIR_ABS}" "${AS_OBJECT_MAP_ABS}" "${AS_EXPECTED_TOTAL}" <<'PY'
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
    raise SystemExit("[ERROR] Real-mesh AS validation failed:\n  " + "\n  ".join(bad[:20]))

print(
    f"[INFO] Validated real-mesh AS bank: {motion_dir} "
    f"({len(npz_files)} clips, {len(unique_urdfs)} unique URDF mesh asset(s))"
)
PY

if [[ "${OBJECT_SPAWN_MODE}" == "single_slot_multi_urdf" ]]; then
  AS_SINGLE_SLOT_MOTION_DIR=${AS_SINGLE_SLOT_MOTION_DIR:-"${AS_DATA_DIR}/_single_slot_motion_bank"}
  AS_SINGLE_SLOT_MOTION_DIR_ABS=$(realpath -m "${AS_SINGLE_SLOT_MOTION_DIR}")
  case "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" in
    "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
      ;;
    *)
      echo "[ERROR] Generated AS single-slot motion bank must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
      echo "[ERROR] Got: ${AS_SINGLE_SLOT_MOTION_DIR_ABS}" >&2
      exit 2
      ;;
  esac

  python3 - "${AS_DATA_DIR_ABS}" "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" <<'PY'
import shutil
import sys
from pathlib import Path

source_dir = Path(sys.argv[1]).resolve()
view_dir = Path(sys.argv[2]).resolve()
if view_dir == source_dir or source_dir not in view_dir.parents:
    raise SystemExit(f"[ERROR] Refusing unexpected generated motion view path: {view_dir}")

marker = view_dir / ".generated_by_train_as_general"
if view_dir.exists():
    if not marker.exists():
        raise SystemExit(
            f"[ERROR] Refusing to clean non-generated AS motion view: {view_dir}. "
            "Choose an empty AS_SINGLE_SLOT_MOTION_DIR or remove it manually."
        )
    for child in view_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
else:
    view_dir.mkdir(parents=True)

for npz_path in sorted(source_dir.glob("*.npz")):
    target = view_dir / npz_path.name
    target.symlink_to(npz_path.resolve())
marker.write_text("generated by train_as_general.sh\n", encoding="utf-8")
PY

  AS_SINGLE_SLOT_OBJECT_MAP="${AS_SINGLE_SLOT_MOTION_DIR_ABS}/_clip_object_urdf_map.json"
  AS_OBJECT_MAP=$(python3 "${SCRIPT_DIR}/scripts/prepare_single_slot_object_map.py" \
    --motion-dir "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" \
    --object-map "${AS_OBJECT_MAP_ABS}" \
    --output-map "${AS_SINGLE_SLOT_OBJECT_MAP}")
  AS_OBJECT_MAP_ABS=$(realpath -m "${AS_OBJECT_MAP}")
  case "${AS_OBJECT_MAP_ABS}" in
    "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
      ;;
    *)
      echo "[ERROR] Generated AS single-slot object map must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
      echo "[ERROR] Got: ${AS_OBJECT_MAP_ABS}" >&2
      exit 2
      ;;
  esac
  AS_DATA_DIR="${AS_SINGLE_SLOT_MOTION_DIR}"
  AS_DATA_DIR_ABS="${AS_SINGLE_SLOT_MOTION_DIR_ABS}"
fi

export DATA_MODE=mix-naive
export DS_DATA_ROOT="data/ds_as_data"
export MOTION_DIR="${AS_DATA_DIR}"
export OBJECT_SPEC_PATH="${AS_OBJECT_MAP}"
export ASSERT_NEW_DS_DATA=${ASSERT_NEW_DS_DATA:-0}
export AUTO_PREP_DS_BANK=0
export STRICT_DEFAULT_DS_BANK_VALIDATION=0

export OBJECT_SPAWN_MODE
export HOLOSOMA_OBJECT_SPAWN_MODE="${OBJECT_SPAWN_MODE}"
if [[ "${OBJECT_SPAWN_MODE}" == "urdf" ]]; then
  export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK="${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-0}"
elif [[ "${OBJECT_SPAWN_MODE}" == "single_slot_multi_urdf" ]]; then
  export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK="${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-0}"
else
  export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK="${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-1}"
fi
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${AS_OBJECT_GEOMETRY_MODE}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}
unset OBJECT_GEOMETRY_MODE
export TRAINING_SEED
export RANDOMIZATION_PRESET
export INIT_AT_RANDOM_EP_LEN
export WANDB_PROJECT=${WANDB_PROJECT:-carry-any}

export SEQUENCE_NAME=${SEQUENCE_NAME:-as-general-real-mesh-cotrack}

if [[ -n "${TRAINING_SEED}" ]]; then
  if [[ ! "${TRAINING_SEED}" =~ ^-?[0-9]+$ ]]; then
    echo "[ERROR] TRAINING_SEED/SEED must be an integer. Got: ${TRAINING_SEED}" >&2
    exit 2
  fi
fi

if [[ -n "${INIT_AT_RANDOM_EP_LEN}" ]]; then
  case "$(echo "${INIT_AT_RANDOM_EP_LEN}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      INIT_AT_RANDOM_EP_LEN=True
      ;;
    0|false|no|off)
      INIT_AT_RANDOM_EP_LEN=False
      ;;
    *)
      echo "[ERROR] INIT_AT_RANDOM_EP_LEN must be a boolean. Got: ${INIT_AT_RANDOM_EP_LEN}" >&2
      exit 2
      ;;
  esac
fi

if [[ -n "${RANDOMIZATION_PRESET}" ]]; then
  case "${RANDOMIZATION_PRESET}" in
    none|disabled|t1_29dof|g1_29dof|g1_29dof_wbt|g1_29dof_wbt_with_action_delay|g1_29dof_wbt_w_object|g1_29dof_wbt_w_object_with_action_delay)
      ;;
    *)
      echo "[ERROR] RANDOMIZATION/RANDOMIZATION_PRESET must be one of:" >&2
      echo "[ERROR]   none, disabled, t1_29dof, g1_29dof, g1_29dof_wbt, g1_29dof_wbt_with_action_delay, g1_29dof_wbt_w_object, g1_29dof_wbt_w_object_with_action_delay" >&2
      echo "[ERROR] Got: ${RANDOMIZATION_PRESET}" >&2
      exit 2
      ;;
  esac
fi

echo "[INFO] Launching AS real-mesh co-tracking generalist training"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH}"
echo "[INFO] HOLOSOMA_OBJECT_SPAWN_MODE=${HOLOSOMA_OBJECT_SPAWN_MODE}"
echo "[INFO] HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK}"
echo "[INFO] HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE}"
echo "[INFO] HOLOSOMA_OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE}"
echo "[INFO] NPROC=${NPROC:-<auto>} PER_GPU_ENVS=${PER_GPU_ENVS:-4096} NUM_ENVS=${NUM_ENVS:-<NPROC*PER_GPU_ENVS>} MASTER_PORT=${MASTER_PORT:-<random>}"
echo "[INFO] TRAINING_SEED=${TRAINING_SEED:-<config-default>} RANDOMIZATION=${RANDOMIZATION_PRESET:-<exp-default>} INIT_AT_RANDOM_EP_LEN=${INIT_AT_RANDOM_EP_LEN:-<algo-default>}"

exec bash "${SCRIPT_DIR}/train_object_generalist_ds.sh" mix-naive "$@"
