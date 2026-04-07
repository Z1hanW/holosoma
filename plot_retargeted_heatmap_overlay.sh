#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MOTION_ROOT_A="${MOTION_ROOT_A:-${ROOT_DIR}/data/ds_box_data/train_g1_w_obj_prepared}"
MOTION_ROOT_B="${MOTION_ROOT_B:-${ROOT_DIR}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry}"
LABEL_A="${LABEL_A:-pure-ds (43)}"
LABEL_B="${LABEL_B:-OMOMO-carry (62)}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/logs/plots/root_heatmap_overlay_pureds43_vs_omomo_carry.png}"
BOUNDS_CSV="${BOUNDS_CSV:-}"
BINS="${BINS:-180}"
BLUR_SIGMA_BINS="${BLUR_SIGMA_BINS:-1.25}"
ALIGN_START="${ALIGN_START:-True}"
ALIGN_HEADING="${ALIGN_HEADING:-False}"
SHOW_START_MARKERS="${SHOW_START_MARKERS:-True}"
SHOW_TRAJECTORIES="${SHOW_TRAJECTORIES:-False}"
TITLE="${TITLE:-Overlayed Top-Down Root Trajectory Heatmaps}"

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1; then
import matplotlib
import numpy
import tyro
PY
  echo "[ERROR] Missing dependencies in ${PYTHON_BIN} environment (need: matplotlib, numpy, tyro)." >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" "${ROOT_DIR}/src/holosoma_retargeting/plot_root_trajectory_overlay.py"
  --motion-root-a "${MOTION_ROOT_A}"
  --motion-root-b "${MOTION_ROOT_B}"
  --label-a "${LABEL_A}"
  --label-b "${LABEL_B}"
  --output-path "${OUTPUT_PATH}"
  --bins "${BINS}"
  --blur-sigma-bins "${BLUR_SIGMA_BINS}"
  --title "${TITLE}"
)

if [[ -n "${BOUNDS_CSV}" ]]; then
  cmd+=(--bounds-csv "${BOUNDS_CSV}")
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
bool_arg "SHOW_START_MARKERS" "${SHOW_START_MARKERS}" "--show-start-markers" "--no-show-start-markers"
bool_arg "SHOW_TRAJECTORIES" "${SHOW_TRAJECTORIES}" "--show-trajectories" "--no-show-trajectories"

echo "[INFO] Plotting overlay top-down heatmap"
echo "[INFO] MOTION_ROOT_A=${MOTION_ROOT_A}"
echo "[INFO] MOTION_ROOT_B=${MOTION_ROOT_B}"
echo "[INFO] OUTPUT_PATH=${OUTPUT_PATH}"
if [[ -n "${BOUNDS_CSV}" ]]; then
  echo "[INFO] BOUNDS_CSV=${BOUNDS_CSV}"
fi
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
