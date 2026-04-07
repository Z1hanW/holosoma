#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MOTION_ROOT="${MOTION_ROOT:-${ROOT_DIR}/data/ds_box_data/train_g1_w_obj_prepared}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/logs/plots/retargeted_root_heatmap.png}"
CLIP_LIST="${CLIP_LIST:-}"
DIVERSE_TOPK="${DIVERSE_TOPK:-10}"
BOUNDS_CSV="${BOUNDS_CSV:-}"
BINS="${BINS:-180}"
BLUR_SIGMA_BINS="${BLUR_SIGMA_BINS:-1.25}"
ALIGN_START="${ALIGN_START:-True}"
ALIGN_HEADING="${ALIGN_HEADING:-False}"
SHOW_TRAJECTORIES="${SHOW_TRAJECTORIES:-True}"
SHOW_START_MARKERS="${SHOW_START_MARKERS:-True}"
TITLE="${TITLE:-Top-Down Root Trajectory Heatmap}"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1; then
import matplotlib
import numpy
import tyro
PY
  echo "[ERROR] Missing dependencies in ${PYTHON_BIN} environment (need: matplotlib, numpy, tyro)." >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" "${ROOT_DIR}/src/holosoma_retargeting/plot_root_trajectory_heatmap.py"
  --motion-root "${MOTION_ROOT}"
  --output-path "${OUTPUT_PATH}"
  --bins "${BINS}"
  --blur-sigma-bins "${BLUR_SIGMA_BINS}"
  --title "${TITLE}"
)

if [[ -n "${BOUNDS_CSV}" ]]; then
  cmd+=(--bounds-csv "${BOUNDS_CSV}")
fi

if [[ -n "${CLIP_LIST}" ]]; then
  cmd+=(--clip-names-csv "${CLIP_LIST}")
elif [[ "${DIVERSE_TOPK}" != "0" ]]; then
  cmd+=(--diverse-topk "${DIVERSE_TOPK}")
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

bool_arg "ALIGN_START" "${ALIGN_START}" "--align-start" "--no-align-start"
bool_arg "ALIGN_HEADING" "${ALIGN_HEADING}" "--align-heading" "--no-align-heading"
bool_arg "SHOW_TRAJECTORIES" "${SHOW_TRAJECTORIES}" "--show-trajectories" "--no-show-trajectories"
bool_arg "SHOW_START_MARKERS" "${SHOW_START_MARKERS}" "--show-start-markers" "--no-show-start-markers"

echo "[INFO] Plotting top-down root heatmap"
echo "[INFO] MOTION_ROOT=${MOTION_ROOT}"
echo "[INFO] OUTPUT_PATH=${OUTPUT_PATH}"
if [[ -n "${BOUNDS_CSV}" ]]; then
  echo "[INFO] BOUNDS_CSV=${BOUNDS_CSV}"
fi
if [[ -n "${CLIP_LIST}" ]]; then
  echo "[INFO] CLIP_LIST=${CLIP_LIST}"
else
  echo "[INFO] DIVERSE_TOPK=${DIVERSE_TOPK}"
fi
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
