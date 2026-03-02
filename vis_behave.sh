#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ALLOWED_OBJECTS=${ALLOWED_OBJECTS:-"boxmedium,boxlarge"}

USE_SQ_OBJ=0
PASS_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --sqs)
      USE_SQ_OBJ=1
      ;;
    *)
      PASS_ARGS+=("$arg")
      ;;
  esac
done

cmd=(
  python "$SCRIPT_DIR/behave/viser_annot_player.py"
  "/data/behave/annotation_30fps_zup"
  --annotation-root
  --objects-root "/data/behave/objects"
  --smpl-model-root "/data/behave/HMR"
  --allowed-objects "${ALLOWED_OBJECTS}"
)

if [[ "${USE_SQ_OBJ}" == "1" ]]; then
  cmd+=(--use-sq-obj)
fi

cmd+=("${PASS_ARGS[@]}")
"${cmd[@]}"
