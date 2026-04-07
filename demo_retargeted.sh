#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
PORT="${PORT:-18086}"
MOTION_ROOT="${MOTION_ROOT:-${ROOT_DIR}/data/ds_box_data/train_g1_w_obj_prepared}"
ROBOT_URDF="${ROBOT_URDF:-${ROOT_DIR}/src/holosoma_retargeting/models/g1/g1_29dof.urdf}"
DEFAULT_OBJECT_URDF="${DEFAULT_OBJECT_URDF:-}"
N="${1:-${N:-0}}"
CLIP_LIST="${CLIP_LIST:-}"
DIVERSE_TOPK="${DIVERSE_TOPK:-0}"
AUTOPLAY="${AUTOPLAY:-True}"
LOOP="${LOOP:-True}"
SHOW_ROBOT_MESHES="${SHOW_ROBOT_MESHES:-True}"
SHOW_OBJECT_MESHES="${SHOW_OBJECT_MESHES:-True}"
SHOW_GRID="${SHOW_GRID:-True}"
ALIGN_ANCHOR="${ALIGN_ANCHOR:-robot}"
ALIGN_XY_ONLY="${ALIGN_XY_ONLY:-True}"
PLAYBACK_FPS="${PLAYBACK_FPS:-0}"
VISUAL_FPS_MULTIPLIER="${VISUAL_FPS_MULTIPLIER:-2}"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1; then
import numpy
import tyro
import viser
import yourdfpy
PY
  echo "[ERROR] Missing dependencies in ${PYTHON_BIN} environment (need: numpy, tyro, viser, yourdfpy)." >&2
  exit 1
fi

if [[ ! -d "${MOTION_ROOT}" ]]; then
  echo "[ERROR] MOTION_ROOT not found: ${MOTION_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${ROBOT_URDF}" ]]; then
  echo "[ERROR] ROBOT_URDF not found: ${ROBOT_URDF}" >&2
  exit 1
fi

if [[ -z "${CLIP_LIST}" && "${DIVERSE_TOPK}" != "0" ]]; then
  CLIP_LIST="$(
    "${PYTHON_BIN}" "${ROOT_DIR}/src/holosoma_retargeting/select_diverse_motions.py" \
      --motion-root "${MOTION_ROOT}" \
      --topk "${DIVERSE_TOPK}" \
      --output csv
  )"
  if [[ -z "${CLIP_LIST}" ]]; then
    echo "[ERROR] Failed to compute diverse CLIP_LIST from ${MOTION_ROOT}" >&2
    exit 1
  fi
  if [[ "${N}" == "0" ]]; then
    N="${DIVERSE_TOPK}"
  fi
  echo "[INFO] Selected diverse clips: ${CLIP_LIST}"
fi

cmd=(
  "${PYTHON_BIN}" "${ROOT_DIR}/src/holosoma_retargeting/viser_multi_retargeted_player.py"
  --motion-root "${MOTION_ROOT}"
  --robot-urdf "${ROBOT_URDF}"
  --port "${PORT}"
  --limit "${N}"
  --align-anchor "${ALIGN_ANCHOR}"
  --playback-fps "${PLAYBACK_FPS}"
  --visual-fps-multiplier "${VISUAL_FPS_MULTIPLIER}"
)

if [[ -n "${DEFAULT_OBJECT_URDF}" ]]; then
  cmd+=(--default-object-urdf "${DEFAULT_OBJECT_URDF}")
fi

if [[ -n "${CLIP_LIST}" ]]; then
  cmd+=(--clip-names-csv "${CLIP_LIST}")
fi

bool_arg() {
  local name="$1"
  local value
  value="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
  local positive="$3"
  local negative="$4"
  case "${value}" in
    1|true|yes|on)
      cmd+=("${positive}")
      ;;
    0|false|no|off)
      cmd+=("${negative}")
      ;;
    *)
      echo "[ERROR] ${name} must be true/false/1/0, got: $2" >&2
      exit 1
      ;;
  esac
}

bool_arg "AUTOPLAY" "${AUTOPLAY}" "--autoplay" "--no-autoplay"
bool_arg "LOOP" "${LOOP}" "--loop" "--no-loop"
bool_arg "SHOW_ROBOT_MESHES" "${SHOW_ROBOT_MESHES}" "--show-robot-meshes" "--no-show-robot-meshes"
bool_arg "SHOW_OBJECT_MESHES" "${SHOW_OBJECT_MESHES}" "--show-object-meshes" "--no-show-object-meshes"
bool_arg "SHOW_GRID" "${SHOW_GRID}" "--show-grid" "--no-show-grid"
bool_arg "ALIGN_XY_ONLY" "${ALIGN_XY_ONLY}" "--align-xy-only" "--no-align-xy-only"

echo "[INFO] Running pure-ds retargeted viser demo"
echo "[INFO] MOTION_ROOT=${MOTION_ROOT}"
echo "[INFO] N=${N} (0 means load all clips)"
echo "[INFO] ALIGN_ANCHOR=${ALIGN_ANCHOR}, ALIGN_XY_ONLY=${ALIGN_XY_ONLY}"
echo "[INFO] Open: http://localhost:${PORT}"
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
