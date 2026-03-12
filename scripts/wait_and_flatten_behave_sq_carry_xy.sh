#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/retgt/bin/python}

CONV_BEHAVE_ROOT=${CONV_BEHAVE_ROOT:-/data/holosoma_moved/src/holosoma_retargeting/converted_res/behave_sq_carry_xy_0p5_1p5}
FLAT_BEHAVE_DIR=${FLAT_BEHAVE_DIR:-/data/holosoma_moved/src/holosoma_retargeting/converted_res/behave_sq_carry_xy_0p5_1p5_flat}
LOG_DIR=${LOG_DIR:-/tmp/retarget_xy_scale_logs}
SLEEP_S=${SLEEP_S:-30}

TAGS=(${TAGS:-xy050 xy060 xy070 xy080 xy090 xy100 xy110 xy120 xy130 xy140 xy150})

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/behave_flatten.log"
: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[INFO] Waiting for completed scale markers under ${CONV_BEHAVE_ROOT}"
while true; do
  pending=()
  for tag in "${TAGS[@]}"; do
    if [[ ! -f "${CONV_BEHAVE_ROOT}/.${tag}.done" ]]; then
      pending+=("${tag}")
    fi
  done

  if [[ ${#pending[@]} -eq 0 ]]; then
    break
  fi

  echo "[INFO] Pending tags: ${pending[*]}"
  sleep "${SLEEP_S}"
done

echo "[INFO] All scale markers present. Flattening into ${FLAT_BEHAVE_DIR}"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/flatten_nested_xy_scale_dataset.py" \
  --input-root "${CONV_BEHAVE_ROOT}" \
  --output-dir "${FLAT_BEHAVE_DIR}" \
  --clean \
  --expected-tags "${TAGS[@]}"

echo "[INFO] Flattened BEHAVE XY bank ready: ${FLAT_BEHAVE_DIR}"
