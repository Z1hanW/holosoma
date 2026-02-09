#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

NPZ_PATH=${1:-${NPZ_PATH:-""}}
DATASET_ROOT=${2:-${BEHAVE_ROOT:-""}}
SMPL_MODEL_ROOT=${3:-${SMPL_MODEL_ROOT:-""}}

if [[ -z "$NPZ_PATH" || -z "$DATASET_ROOT" || -z "$SMPL_MODEL_ROOT" ]]; then
  echo "Usage: $0 <npz_path_or_dir> <behave_dataset_root> <smpl_model_root>"
  echo "Or set env vars: NPZ_PATH, BEHAVE_ROOT, SMPL_MODEL_ROOT"
  exit 1
fi

python "$SCRIPT_DIR/behave/viser_annot_player.py" \
  "$NPZ_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --smpl-model-root "$SMPL_MODEL_ROOT"
