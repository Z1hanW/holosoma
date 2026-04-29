#!/usr/bin/env bash
set -euo pipefail

# DS-box object generalist training with DeFM depth perception.
#
# Keep train_object_generalist_ds.sh as the source of truth for all training,
# data-mode, naming, environment-count, camera, and export defaults. This
# wrapper only changes the perception preset.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export PERCEPTION="${PERCEPTION:-camera_depth_d435i_defm_efficientnet_b2}"

echo "[INFO] train_object_defm.sh -> train_object_generalist_ds.sh"
echo "[INFO] perception=${PERCEPTION}"

exec bash "${SCRIPT_DIR}/train_object_generalist_ds.sh" "$@"
