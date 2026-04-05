#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export NPROC=${NPROC:-4}
export PER_GPU_ENVS=${PER_GPU_ENVS:-8192}
export NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}

echo "[INFO] MINI launcher: NPROC=${NPROC} PER_GPU_ENVS=${PER_GPU_ENVS} NUM_ENVS=${NUM_ENVS}"

exec bash "${SCRIPT_DIR}/train_object_generalist_ds.sh" "$@"
