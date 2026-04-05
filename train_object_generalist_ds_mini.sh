#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NPROC=${NPROC:-4}
PER_GPU_ENVS=${PER_GPU_ENVS:-8192}
NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}

exec bash "${SCRIPT_DIR}/train_object_generalist_ds.sh" "$@"
