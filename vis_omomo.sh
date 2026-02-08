#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

DATA_DIR=${1:-"$SCRIPT_DIR/src/holosoma_retargeting/demo_data/OMOMO_new"}
GLOB=${2:-"*.pt"}

python "$SCRIPT_DIR/src/holosoma_retargeting/viser_omomo_player.py" \
    --data-dir "$DATA_DIR" \
    --glob "$GLOB"
