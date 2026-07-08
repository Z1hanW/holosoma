#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:?REMOTE_HOST is required}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:?REMOTE_RUN_DIR is required}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:?TEACHER_CHECKPOINT is required}"

CHECKPOINT_ITERS="${CHECKPOINT_ITERS:-1000 3000 5000 6000 10000}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/holosoma_depth_distill_monitor}"
SLEEP_SECONDS="${SLEEP_SECONDS:-600}"
NUM_ENVS="${NUM_ENVS:-512}"
EVAL_STEPS="${EVAL_STEPS:-650}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-hssim}"

mkdir -p "${OUTPUT_DIR}"

source /home/ubuntu/.holosoma_deps/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
  || source /opt/conda/etc/profile.d/conda.sh

conda activate "${CONDA_ENV_NAME}"

while true; do
  date -u +"[%Y-%m-%dT%H:%M:%SZ] polling ${REMOTE_HOST}:${REMOTE_RUN_DIR}"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 "${REMOTE_HOST}" \
    "tail -n 5 '${REMOTE_RUN_DIR}'/*.log 2>/dev/null | grep 'iter=' | tail -n 3 || true"

  for iter in ${CHECKPOINT_ITERS}; do
    padded_iter="$(printf "%07d" "${iter}")"
    remote_ckpt="${REMOTE_RUN_DIR}/student_${padded_iter}.pt"
    local_ckpt="${OUTPUT_DIR}/student_${padded_iter}.pt"
    result_json="${OUTPUT_DIR}/student_${padded_iter}_ablation.json"

    if [[ -f "${result_json}" ]]; then
      continue
    fi
    if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8 "${REMOTE_HOST}" \
      "test -f '${remote_ckpt}'"; then
      continue
    fi

    rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8" \
      "${REMOTE_HOST}:${remote_ckpt}" "${local_ckpt}"

    source scripts/source_isaacsim_setup.sh
    PYTHONPATH="${PWD}/src/holosoma:${PYTHONPATH:-}" \
      python -m holosoma.eval_depth_student_ablation \
        --checkpoint "${local_ckpt}" \
        --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
        --num-envs "${NUM_ENVS}" \
        --steps "${EVAL_STEPS}" \
        --modes teacher normal zero shuffle \
        --output "${result_json}"
  done

  sleep "${SLEEP_SECONDS}"
done
