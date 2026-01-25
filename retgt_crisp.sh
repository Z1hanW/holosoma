#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
POST_SCENE_ROOT=${1:-"/home/ubuntu/FAR/CRISP-Real2Sim/results/output/post_scene"}

bash "$SCRIPT_DIR/src/holosoma_retargeting/retgt_crisp.sh" "$POST_SCENE_ROOT"
