#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 xy050" >&2
  exit 1
fi

TAG="$1"
case "${TAG}" in
  xy[0-9][0-9][0-9]) ;;
  *)
    echo "[ERROR] Invalid scale tag '${TAG}'. Expected form xy050." >&2
    exit 1
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}
RETARGET_ROOT=${RETARGET_ROOT:-${REPO_ROOT}/src/holosoma_retargeting}
PYTHON_BIN=${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/retgt/bin/python}

BEHAVE_SOURCE=${BEHAVE_SOURCE:-/data/behave/annotation_30fps_zup_carry}
BEHAVE_OBJECT_ROOT=${BEHAVE_OBJECT_ROOT:-/data/behave/objects}
RAW_BEHAVE_ROOT=${RAW_BEHAVE_ROOT:-${RETARGET_ROOT}/demo_results_parallel/g1/object_interaction/behave_zup_sq_carry_xy_0p5_1p5}
CONV_BEHAVE_ROOT=${CONV_BEHAVE_ROOT:-/data/holosoma_moved/src/holosoma_retargeting/converted_res/behave_sq_carry_xy_0p5_1p5}
LOG_DIR=${LOG_DIR:-/tmp/retarget_xy_scale_logs}

MAX_WORKERS_BEHAVE=${MAX_WORKERS_BEHAVE:-4}
EXPECTED_COUNT=${EXPECTED_COUNT:-10}
CLEAN=${CLEAN:-1}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[ERROR] PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ ! -d "${BEHAVE_SOURCE}" ]]; then
  echo "[ERROR] BEHAVE_SOURCE not found: ${BEHAVE_SOURCE}" >&2
  exit 1
fi
if [[ ! -d "${BEHAVE_OBJECT_ROOT}" ]]; then
  echo "[ERROR] BEHAVE_OBJECT_ROOT not found: ${BEHAVE_OBJECT_ROOT}" >&2
  exit 1
fi

RAW_DIR="${RAW_BEHAVE_ROOT}/${TAG}"
CONV_DIR="${CONV_BEHAVE_ROOT}/${TAG}"
MARKER_FILE="${CONV_BEHAVE_ROOT}/.${TAG}.done"
LOG_FILE="${LOG_DIR}/behave_${TAG}.log"

if [[ "${CLEAN}" == "1" ]]; then
  rm -rf "${RAW_DIR}" "${CONV_DIR}"
fi
rm -f "${MARKER_FILE}"
mkdir -p "${RAW_DIR}" "${CONV_DIR}" "${LOG_DIR}"
: > "${LOG_FILE}"
exec > >(tee -a "${LOG_FILE}") 2>&1

SCALE=$("${PYTHON_BIN}" - "${TAG}" <<'PY'
import sys
tag = sys.argv[1]
print(f"{int(tag[2:]) / 100.0:.1f}")
PY
)

echo "[INFO] === BEHAVE rebuild ${TAG} (scale=${SCALE}) ==="
echo "[INFO] Raw dir : ${RAW_DIR}"
echo "[INFO] Conv dir: ${CONV_DIR}"

(
  cd "${RETARGET_ROOT}" &&
  "${PYTHON_BIN}" examples/parallel_robot_retarget.py \
    --task-type object_interaction \
    --robot g1 \
    --data-format behave_zup \
    --data-dir "${BEHAVE_SOURCE}" \
    --save-dir "${RAW_DIR}" \
    --max-workers "${MAX_WORKERS_BEHAVE}" \
    --task-config.object-mesh-root "${BEHAVE_OBJECT_ROOT}" \
    --task-config.object-mesh-suffix "_f1000.ply" \
    --task-config.object-interaction-scale-augmented "${SCALE}" "${SCALE}" 1.0
)

RAW_COUNT=$(find "${RAW_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l | tr -d ' ')
echo "[INFO] Raw count for ${TAG}: ${RAW_COUNT}"
if [[ "${RAW_COUNT}" -ne "${EXPECTED_COUNT}" ]]; then
  echo "[ERROR] Expected ${EXPECTED_COUNT} raw npz files for ${TAG}, found ${RAW_COUNT}" >&2
  exit 1
fi

(
  cd "${REPO_ROOT}" &&
  INPUT_DIR="${RAW_DIR}" OUTPUT_DIR="${CONV_DIR}" ROBOT="g1" PYTHON_BIN="${PYTHON_BIN}" DATA_FORMAT="behave_zup" \
    bash retgt_post_behave.sh
)

CONV_COUNT=$(find "${CONV_DIR}" -maxdepth 1 -type f -name '*_mj_w_obj.npz' | wc -l | tr -d ' ')
echo "[INFO] Converted count for ${TAG}: ${CONV_COUNT}"
if [[ "${CONV_COUNT}" -ne "${EXPECTED_COUNT}" ]]; then
  echo "[ERROR] Expected ${EXPECTED_COUNT} converted npz files for ${TAG}, found ${CONV_COUNT}" >&2
  exit 1
fi

touch "${MARKER_FILE}"
echo "[INFO] ${TAG} complete. Marker: ${MARKER_FILE}"
