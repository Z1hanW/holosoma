#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
USER_OFFSET="${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET-}"; source scripts/source_inference_setup.sh "${1:-box_75}"
export HOLOSOMA_KEYBOARD_ROOT_COMMAND=0 HOLOSOMA_POLICY_MOTION_INDEX_OFFSET="${USER_OFFSET:-0}"
MODEL="$(python scripts/mj_resolve_wandb_model.py "${2:-$HOLOSOMA_MJ_MODEL}")"; PATCH_DIR="${PATCH_DIR:-logs/live_debug/mj_debug}"; PATCHED="${PATCH_DIR}/$(basename "${MODEL%.onnx}")__${HOLOSOMA_MJ_CLIP}.onnx"
mkdir -p "$PATCH_DIR"; [[ -f "$HOLOSOMA_MJ_MOTION" ]] || { echo "missing data_demo motion: $HOLOSOMA_MJ_MOTION" >&2; exit 1; }
[[ -f "$PATCHED" && "$PATCHED" -nt "$MODEL" && "$PATCHED" -nt "$HOLOSOMA_MJ_MOTION" ]] || python src/holosoma_inference/holosoma_inference/tools/patch_motion_onnx.py --model-path "$MODEL" --motion-file "$HOLOSOMA_MJ_MOTION" --output-path "$PATCHED" >/dev/null
python3 -u src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt-object-distill-mujoco --task.model-path "$PATCHED" --task.auto-start-policy --task.auto-start-motion-clip
