#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT="${SCRIPT_DIR}"

FIXTURE_ROOT=${FIXTURE_ROOT:-"${REPO_ROOT}/tests/fixtures/terrain_mini"}
MOTION_DIR=${MOTION_DIR:-"${FIXTURE_ROOT}/___crisp_clean_motion"}
OBJ_SOURCE=${OBJ_SOURCE:-"${FIXTURE_ROOT}/___crisp_clean_geometry"}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NPROC=${NPROC:-1}
PER_GPU_ENVS=${PER_GPU_ENVS:-2}
NUM_ENVS=${NUM_ENVS:-2}
HEADLESS=${HEADLESS:-True}
HOLOSOMA_EXPORT_ONNX_AT_END=${HOLOSOMA_EXPORT_ONNX_AT_END:-0}

PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-True}
USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-True}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_terrain_test3}
LOGGER_NAME=${LOGGER_NAME:-g1_terrain_test3}
NUM_ITERS=${NUM_ITERS:-2000}

if [[ ! -d "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -d "${OBJ_SOURCE}" ]]; then
  echo "[ERROR] OBJ_SOURCE not found: ${OBJ_SOURCE}" >&2
  exit 1
fi

case "${HEADLESS,,}" in
  0|false|off|no)
    HEADLESS_BOOL=False
    ;;
  1|true|on|yes)
    HEADLESS_BOOL=True
    ;;
  *)
    echo "[ERROR] HEADLESS must be one of: 0/1/false/true/off/on/no/yes. Got: ${HEADLESS}" >&2
    exit 1
    ;;
esac

motion_count=$(find "${MOTION_DIR}" -maxdepth 1 -type f -name "*.npz" | wc -l | tr -d '[:space:]')
obj_count=$(find "${OBJ_SOURCE}" -maxdepth 1 -type f -name "*.obj" | wc -l | tr -d '[:space:]')
if [[ "${motion_count}" != "3" || "${obj_count}" != "3" ]]; then
  echo "[ERROR] Expected exactly 3 motion npz and 3 obj files in fixture." >&2
  echo "        motion_count=${motion_count} obj_count=${obj_count}" >&2
  exit 1
fi

echo "[INFO] Using fixture root: ${FIXTURE_ROOT}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJ_SOURCE=${OBJ_SOURCE}"
echo "[INFO] NUM_ENVS=${NUM_ENVS} NPROC=${NPROC} PER_GPU_ENVS=${PER_GPU_ENVS} HEADLESS=${HEADLESS_BOOL}"

cd "${REPO_ROOT}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
NPROC="${NPROC}" \
PER_GPU_ENVS="${PER_GPU_ENVS}" \
NUM_ENVS="${NUM_ENVS}" \
HEADLESS="${HEADLESS_BOOL}" \
MOTION_DIR="${MOTION_DIR}" \
OBJ_SOURCE="${OBJ_SOURCE}" \
PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION}" \
USE_ADAPTIVE_TIMESTEPS_SAMPLER="${USE_ADAPTIVE_TIMESTEPS_SAMPLER}" \
HOLOSOMA_EXPORT_ONNX_AT_END="${HOLOSOMA_EXPORT_ONNX_AT_END}" \
TRAINING_NAME="${TRAINING_NAME}" \
LOGGER_NAME="${LOGGER_NAME}" \
bash ./train_terrain_generalist.sh heightmap \
  --training.export_onnx=False \
  --algo.config.num_learning_iterations="${NUM_ITERS}" \
  "$@"
