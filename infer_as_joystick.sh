#!/usr/bin/env bash
set -euo pipefail

# Interactive inference for policies trained by distill_as_perception.sh.
#
# This wrapper keeps AS/OMOMO real-mesh defaults aligned with
# distill_as_perception.sh, then delegates the Viser/manual/joystick runtime to
# infer_box_joystick.sh depth.

usage() {
  cat <<'EOF'
Usage:
  bash infer_as_joystick.sh [policy_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  POLICY_CHECKPOINT=<checkpoint> bash infer_as_joystick.sh [extra args...]

Examples:
  bash infer_as_joystick.sh /data/logs_new/carry-any/<run>/model_01000.pt
  bash infer_as_joystick.sh wandb://zihanw22/carry-any/kul723jb/model_00000.pt
  bash infer_as_joystick.sh https://wandb.ai/zihanw22/carry-any/runs/kul723jb
  HEADLESS=False bash infer_as_joystick.sh /abs/path/to/model.pt
  VISER_MANUAL_USE_HW_JOYSTICK=1 bash infer_as_joystick.sh /abs/path/to/model.pt
  DRY_RUN=1 bash infer_as_joystick.sh wandb://zihanw22/carry-any/kul723jb/model_00000.pt

Defaults:
  OMOMO_DATA_DIR=./data/ds_as_data/omomo
  OMOMO_OBJECT_MAP=./data/ds_as_data/omomo/_clip_object_urdf_map.json
  OMOMO_EXPECTED_TOTAL=<auto>

Forwarded controls include VISER_PORT, HEADLESS, MOTION_CLIP_NAME, DRY_RUN,
VISER_MANUAL_USE_HW_JOYSTICK, VISER_MANUAL_HW_BACKEND, VISER_MANUAL_HW_DEVICE,
VISER_MANUAL_HW_TYPE, and any extra holosoma.visualize args.
EOF
}

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/*/runs/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

find_latest_as_distill_checkpoint() {
  local log_root="$1"
  local training_name="$2"
  local latest_run=""
  local latest_ckpt=""

  latest_run=$(ls -dt "${log_root}"/*-"${training_name}"* 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest_run}" ]]; then
    echo ""
    return 0
  fi

  latest_ckpt=$(ls -1 "${latest_run}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  echo "${latest_ckpt}"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    as|as-joystick|as_joystick|as-perception|as_perception|depth|perception|omomo|omomo-real|omomo_real|pure-real|pure_real|pure-omomo|pure_omomo|real)
      shift
      ;;
    *)
      break
      ;;
  esac
done

PYTHON_BIN=${PYTHON_BIN:-python}
WANDB_PROJECT=${WANDB_PROJECT:-carry-any}
LOG_ROOT=${LOG_ROOT:-"/data/logs_new/${WANDB_PROJECT}"}
AS_DISTILL_TRAINING_NAME=${AS_DISTILL_TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_real_mesh_perception}

POLICY_CHECKPOINT=${POLICY_CHECKPOINT:-${AS_CHECKPOINT:-${CKPT:-${CHECKPOINT:-}}}}
if [[ $# -gt 0 ]] && is_checkpoint_ref "$1"; then
  POLICY_CHECKPOINT="$1"
  shift
fi

if [[ -z "${POLICY_CHECKPOINT}" ]]; then
  POLICY_CHECKPOINT="$(find_latest_as_distill_checkpoint "${LOG_ROOT}" "${AS_DISTILL_TRAINING_NAME}")"
  if [[ -n "${POLICY_CHECKPOINT}" ]]; then
    echo "[INFO] Auto-selected latest local AS distill checkpoint: ${POLICY_CHECKPOINT}"
  fi
fi

if [[ -z "${POLICY_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing policy checkpoint trained by distill_as_perception.sh." >&2
  echo "[ERROR] Pass a checkpoint path/W&B run, or set POLICY_CHECKPOINT." >&2
  echo "[ERROR] Searched LOG_ROOT=${LOG_ROOT} for ${AS_DISTILL_TRAINING_NAME}." >&2
  exit 1
fi

OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/omomo"}
OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${OMOMO_DATA_DIR}/_clip_object_urdf_map.json"}
OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-}

LOCAL_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data")
OMOMO_DATA_DIR=$(realpath -m "${OMOMO_DATA_DIR}")
OMOMO_OBJECT_MAP=$(realpath -m "${OMOMO_OBJECT_MAP}")

case "${OMOMO_DATA_DIR}" in
  /nfs|/nfs/*)
    echo "[ERROR] OMOMO_DATA_DIR must be local, not NFS: ${OMOMO_DATA_DIR}" >&2
    echo "[ERROR] Run ./cp_as.sh first and infer from ${SCRIPT_DIR}/data/ds_as_data/omomo." >&2
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
    echo "[ERROR] Run ./cp_as.sh first and use the copied map under ${SCRIPT_DIR}/data." >&2
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
  echo "[ERROR] Run ./cp_as.sh first, or set OMOMO_DATA_DIR to a prepared motion bank." >&2
  exit 2
fi
if ! compgen -G "${OMOMO_DATA_DIR}/*.npz" >/dev/null; then
  echo "[ERROR] No .npz files found in OMOMO_DATA_DIR: ${OMOMO_DATA_DIR}" >&2
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
    echo "[ERROR] infer_as_joystick.sh requires real URDF mesh spawning. Got OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE}" >&2
    exit 2
    ;;
esac
case "$(echo "${OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
  mesh|urdf|off|disable|disabled|0|false|no)
    OBJECT_GEOMETRY_MODE=mesh
    ;;
  *)
    echo "[ERROR] infer_as_joystick.sh requires mesh object geometry. Got OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE}" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" - "${OMOMO_DATA_DIR}" "${OMOMO_OBJECT_MAP}" "${OMOMO_EXPECTED_TOTAL}" <<'PY'
import json
import sys
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
missing_npz = [clip_id for clip_id in sorted(clips) if not (motion_dir / f"{clip_id}.npz").is_file()]
if missing_npz:
    preview = ", ".join(missing_npz[:10])
    raise SystemExit(f"[ERROR] Missing .npz files for {len(missing_npz)} object-map entries: {preview}")

def resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()

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
    if mesh_path is not None and not mesh_path.is_file():
        bad.append(f"{clip_id}: missing mesh {mesh_path}")
    unique_urdfs[str(urdf_path)] = clip_id

if bad:
    raise SystemExit("[ERROR] Real-mesh AS/OMOMO validation failed:\n  " + "\n  ".join(bad[:20]))

print(
    f"[INFO] Validated real-mesh AS/OMOMO bank: {motion_dir} "
    f"({len(npz_files)} clips, {len(unique_urdfs)} unique URDF mesh asset(s))"
)
PY

export WANDB_PROJECT
export LOG_ROOT
export INFER_DATASET=omomo
export MOTION_DIR="${OMOMO_DATA_DIR}"
export OBJECT_URDF="${OMOMO_OBJECT_MAP}"
export OBJECT_SPEC_PATH="${OMOMO_OBJECT_MAP}"
export OBJECT_GEOMETRY_MODE
export HOLOSOMA_OBJECT_SPAWN_MODE="${OBJECT_SPAWN_MODE}"
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${OBJECT_GEOMETRY_MODE}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE="${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}"
export VISER_LOAD_URDF="${VISER_LOAD_URDF:-1}"

# distill_as_perception.sh defaults to a single-frame sparse-root/proprio/action
# student. Keep that as the fallback, but do not mark it as an explicit override:
# infer_box_joystick.sh should prefer the checkpoint-saved observation override
# whenever it can read one.
export DEFAULT_DISTILL_PROPRIO_HISTORY_ONLY="${DEFAULT_DISTILL_PROPRIO_HISTORY_ONLY:-0}"
export DEFAULT_DISTILL_PROPRIO_HISTORY_LENGTH="${DEFAULT_DISTILL_PROPRIO_HISTORY_LENGTH:-1}"
export DEPTH_PERCEPTION_PRESET="${DEPTH_PERCEPTION_PRESET:-checkpoint}"
export HOLOSOMA_RESET_TO_DEFAULT_POSE="${HOLOSOMA_RESET_TO_DEFAULT_POSE:-0}"

echo "[INFO] Launching AS/OMOMO real-mesh joystick inference"
echo "[INFO] checkpoint=${POLICY_CHECKPOINT}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJECT_URDF=${OBJECT_URDF}"
echo "[INFO] INFER_DATASET=${INFER_DATASET}"
echo "[INFO] DEPTH_PERCEPTION_PRESET=${DEPTH_PERCEPTION_PRESET}"
echo "[INFO] DEFAULT_DISTILL_PROPRIO_HISTORY_ONLY=${DEFAULT_DISTILL_PROPRIO_HISTORY_ONLY}"
echo "[INFO] HOLOSOMA_OBJECT_SPAWN_MODE=${HOLOSOMA_OBJECT_SPAWN_MODE}"
echo "[INFO] HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE}"
echo "[INFO] HOLOSOMA_RESET_TO_DEFAULT_POSE=${HOLOSOMA_RESET_TO_DEFAULT_POSE}"

exec bash "${SCRIPT_DIR}/infer_box_joystick.sh" depth "${POLICY_CHECKPOINT}" "$@"
