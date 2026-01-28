#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POST_SCENE_ROOT="/home/ubuntu/FAR/CRISP-Real2Sim/results/output/post_scene"
SEQ_NAME=${1:-${SEQ_NAME:-""}}
HMR_TYPE=${HMR_TYPE:-"gv"}
# Usage:
#   bash retgt_crisp.sh [seq_name]
# Example:
#   bash retgt_crisp.sh vmm_25
# Environment overrides:
#   HMR_TYPE=gv SEQ_NAME=vmm_25 bash retgt_crisp.sh
if [ "$#" -gt 1 ]; then
  echo "[ERROR] Too many arguments. Usage: bash retgt_crisp.sh [seq_name]" >&2
  exit 1
fi

bash "$SCRIPT_DIR/src/holosoma_retargeting/retgt_crisp.sh" "$POST_SCENE_ROOT" "$HMR_TYPE" "${SEQ_NAME}"
