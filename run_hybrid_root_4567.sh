#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

SESSION_NAME=${SESSION_NAME:-${SESSION:-hybrid_root_4567}}
GPUS=${GPUS:-4,5,6,7}
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
NPROC=${NPROC:-${#GPU_LIST[@]}}
PER_GPU_ENVS=${PER_GPU_ENVS:-4096}
NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}
LOG_DIR=${LOG_DIR:-logs}
LOG_PATH="${LOG_DIR}/${SESSION_NAME}_$(date +%Y%m%d_%H%M%S).log"

PHYSX_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_FOUND_LOST_PAIRS_CAPACITY:-268435456}
PHYSX_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-268435456}

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [[ "${REPLACE_SESSION:-0}" == "1" ]]; then
    tmux kill-session -t "${SESSION_NAME}"
  else
    echo "[ERROR] tmux session '${SESSION_NAME}' already exists. Set REPLACE_SESSION=1 to replace it." >&2
    exit 1
  fi
fi

mkdir -p "${SCRIPT_DIR}/${LOG_DIR}"

cmd=(
  env
  "CUDA_VISIBLE_DEVICES=${GPUS}"
  "NPROC=${NPROC}"
  "NUM_ENVS=${NUM_ENVS}"
  "PER_GPU_ENVS=${PER_GPU_ENVS}"
  bash "${SCRIPT_DIR}/hybrid_root.sh"
  "--simulator.config.sim.physx.gpu-found-lost-pairs-capacity=${PHYSX_FOUND_LOST_PAIRS_CAPACITY}"
  "--simulator.config.sim.physx.gpu-found-lost-aggregate-pairs-capacity=${PHYSX_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY}"
  "$@"
)

printf -v quoted_cmd "%q " "${cmd[@]}"
printf -v quoted_log "%q" "${LOG_PATH}"

echo "[INFO] starting ${SESSION_NAME}"
echo "[INFO] gpus=${GPUS} nproc=${NPROC} num_envs=${NUM_ENVS} per_gpu_envs=${PER_GPU_ENVS}"
echo "[INFO] log=${LOG_PATH}"

tmux new-session -d -s "${SESSION_NAME}" -c "${SCRIPT_DIR}" "${quoted_cmd}2>&1 | tee ${quoted_log}"

echo "[INFO] tmux attach: tmux attach -t ${SESSION_NAME}"
echo "[INFO] tail log: tail -f ${SCRIPT_DIR}/${LOG_PATH}"
