#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

python "$SCRIPT_DIR/behave/viser_annot_player.py" \
  "/data/behave/annotation_30fps_zup" \
  --annotation-root \
  --objects-root "/data/behave/objects" \
  --smpl-model-root "/data/behave/HMR"
