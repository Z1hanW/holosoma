#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

PYTHON_BIN=${PYTHON_BIN:-python}
PORT=${PORT:-18080}
MOTION_DIRS_CSV=${MOTION_DIRS_CSV:-"${SCRIPT_DIR}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/omomo_carry_aug_extra,${SCRIPT_DIR}/src/holosoma_retargeting/demo_results_parallel/g1/object_interaction/behave_zup_sq_carry_aug_extra"}
DECISION_LOG_PATH=${DECISION_LOG_PATH:-"${SCRIPT_DIR}/logs/object_interaction_review_decisions.json"}
START_CLIP=${START_CLIP:-""}
AUTOPLAY=${AUTOPLAY:-False}
LOOP=${LOOP:-True}

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1; then
import numpy
import tyro
import viser
import yourdfpy
PY
  echo "[ERROR] Missing review viewer dependencies in ${PYTHON_BIN} environment (need: numpy, tyro, viser, yourdfpy)." >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/src/holosoma_retargeting/viser_review_bank.py"
  --motion-dirs-csv "${MOTION_DIRS_CSV}"
  --decision-log-path "${DECISION_LOG_PATH}"
  --port "${PORT}"
)

autoplay_lc=$(echo "${AUTOPLAY}" | tr '[:upper:]' '[:lower:]')
case "${autoplay_lc}" in
  1|true|yes|on) cmd+=(--autoplay) ;;
  0|false|no|off) cmd+=(--no-autoplay) ;;
  *)
    echo "[ERROR] AUTOPLAY must be true/false. Got: ${AUTOPLAY}" >&2
    exit 1
    ;;
esac

loop_lc=$(echo "${LOOP}" | tr '[:upper:]' '[:lower:]')
case "${loop_lc}" in
  1|true|yes|on) cmd+=(--loop) ;;
  0|false|no|off) cmd+=(--no-loop) ;;
  *)
    echo "[ERROR] LOOP must be true/false. Got: ${LOOP}" >&2
    exit 1
    ;;
esac

if [[ -n "${START_CLIP}" ]]; then
  cmd+=(--start-clip "${START_CLIP}")
fi

echo "[INFO] motion_dirs=${MOTION_DIRS_CSV}"
echo "[INFO] decision_log=${DECISION_LOG_PATH}"
echo "[INFO] viser=http://localhost:${PORT}"

exec "${cmd[@]}"
