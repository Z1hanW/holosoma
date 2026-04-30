#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source scripts/source_inference_setup.sh "${1:-box_75}"
export HOLOSOMA_KEYBOARD_ROOT_COMMAND=1 HOLOSOMA_KEYBOARD_ROOT_COMMAND_VALUE=0 HOLOSOMA_KEYBOARD_ROOT_COMMAND_YAW_VALUE=0 HOLOSOMA_KEYBOARD_ROOT_COMMAND_MODE=manual HOLOSOMA_POLICY_CONTROL_PORT=0 HOLOSOMA_PREFILL_OBS_HISTORY_ON_MOTION_START=0
MODEL="$(python scripts/mj_resolve_wandb_model.py "${2:-$HOLOSOMA_MJ_MODEL}")"
CFG="${INFERENCE_CONFIG:-$(python scripts/mj_infer_inference_config.py "$MODEL")}"
[[ -f "$HOLOSOMA_MJ_MOTION" ]] || { echo "missing data_demo motion: $HOLOSOMA_MJ_MOTION" >&2; exit 1; }
python3 -u src/holosoma_inference/holosoma_inference/run_policy.py "inference:$CFG" --task.model-path "$MODEL" --task.motion-file "$HOLOSOMA_MJ_MOTION" --task.interface lo --task.use-sim-state --task.use-sim-time --task.sim-clock-port "$SIM_CLOCK_PORT" --task.sim-state-port "$SIM_STATE_PORT" --task.sim-control-port "$SIM_CONTROL_PORT" --task.use-zmq-lowcmd --task.use-split-perception-obs --task.perception-obs-port "$PERCEPTION_OBS_PORT" --task.use-split-perception-obs-shm --task.perception-obs-shm-name "$PERCEPTION_OBS_SHM_NAME" --task.no-auto-start-motion --task.auto-start-policy --task.auto-start-motion-clip --task.auto-start-stiff-hold-sec "${MJ_NOHIS_STIFF_HOLD_SEC:-2.0}" --task.auto-start-stiff-max-wait-sec "${MJ_NOHIS_STIFF_MAX_WAIT_SEC:-2.5}" --task.auto-start-stiff-pose-tolerance "$AUTO_START_STIFF_POSE_TOL" --task.policy-action-scale "$POLICY_ACTION_SCALE" --task.rl-rate "$POLICY_RL_RATE" --task.sim-object-name object --task.prefer-sim-ref-from-sim-state
