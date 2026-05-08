#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${HOLOSOMA_MJ_MOTION:-box_75}"
checkpoint="${HOLOSOMA_WANDB_CHECKPOINT:-latest}"
run_ref="${HOLOSOMA_WANDB_RUN:-zihanw22/boxer/w5qostjn}"
motion_init="${HOLOSOMA_MJ_MOTION_INIT:-0}"
auto_start="${HOLOSOMA_RO_AUTO_START:-0}"
auto_motion="${HOLOSOMA_RO_AUTO_MOTION:-0}"
use_sim_state="${HOLOSOMA_RO_USE_SIM_STATE:-1}"
explicit_motion_mode=0
clip_arg_seen=0
if [[ -n "${HOLOSOMA_MJ_MOTION_INIT:-}" ]]; then
  explicit_motion_mode=1
fi
positional=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --motion-init)
      motion_init=1
      explicit_motion_mode=1
      ;;
    --manual)
      motion_init=0
      explicit_motion_mode=1
      ;;
    --auto-start|--rollout)
      auto_start=1
      ;;
    --auto-motion)
      auto_start=1
      auto_motion=1
      ;;
    --use-sim-state)
      use_sim_state=1
      ;;
    --no-sim-state)
      use_sim_state=0
      ;;
    --clip)
      shift
      clip="$1"
      clip_arg_seen=1
      ;;
    --checkpoint)
      shift
      checkpoint="$1"
      ;;
    --run)
      shift
      run_ref="$1"
      ;;
    *)
      positional+=("$1")
      ;;
  esac
  shift
done

if (( ${#positional[@]} >= 1 )); then
  clip="${positional[0]}"
  clip_arg_seen=1
fi
if (( ${#positional[@]} >= 2 )); then
  checkpoint="${positional[1]}"
fi
if (( ${#positional[@]} >= 3 )); then
  run_ref="${positional[2]}"
fi

if [[ "$explicit_motion_mode" == "0" && "$clip_arg_seen" == "1" ]]; then
  motion_init=1
fi

motion_file="$clip"
if [[ "$clip" != *.npz && "$clip" != /* ]]; then
  motion_file="${ROOT_DIR}/data_demo/${clip}.npz"
fi
if [[ "$motion_file" != /* ]]; then
  motion_file="${ROOT_DIR}/${motion_file}"
fi

normalize_run_path() {
  local ref="$1"
  ref="${ref#wandb://}"
  if [[ "$ref" == https://wandb.ai/* ]]; then
    ref="${ref#https://wandb.ai/}"
  fi
  ref="${ref%%/files/*}"
  if [[ "$ref" == */runs/* ]]; then
    local entity project _runs run_id rest
    IFS=/ read -r entity project _runs run_id rest <<< "$ref"
    printf '%s/%s/%s\n' "$entity" "$project" "$run_id"
  elif [[ "$ref" != */* ]]; then
    printf 'zihanw22/boxer/%s\n' "$ref"
  else
    local entity project run_id rest
    IFS=/ read -r entity project run_id rest <<< "$ref"
    printf '%s/%s/%s\n' "$entity" "$project" "$run_id"
  fi
}

run_path="$(normalize_run_path "$run_ref")"
run_id="${run_path##*/}"

if [[ "$checkpoint" == /* || "$checkpoint" == ./* || "$checkpoint" == ../* ]]; then
  model_path="$checkpoint"
elif [[ "$checkpoint" == https://wandb.ai/* || "$checkpoint" == wandb://* ]]; then
  model_path="$checkpoint"
elif [[ "$checkpoint" =~ ^[0-9]+$ ]]; then
  model_path="wandb://${run_path}/model_${checkpoint}.onnx"
else
  model_path="wandb://${run_path}/${checkpoint:-latest}"
fi

inference_config="${HOLOSOMA_INFERENCE_CONFIG:-}"
if [[ -z "$inference_config" ]]; then
  case "$run_id" in
    w5qostjn)
      inference_config="g1-wbt-object-perception-no-linvel"
      ;;
    *)
      inference_config="g1-wbt-object-perception"
      ;;
  esac
fi

force_zero_sparse="${HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND:-}"
if [[ -z "$force_zero_sparse" ]]; then
  if [[ "$run_id" == "tvtwx4to" ]]; then
    force_zero_sparse=1
  else
    force_zero_sparse=0
  fi
fi

if [[ "${HOLOSOMA_SKIP_SOURCE_INFERENCE_SETUP:-0}" != "1" ]]; then
  source "${ROOT_DIR}/scripts/source_inference_setup.sh"
fi

export HOLOSOMA_MJ_MOTION="$motion_file"
export HOLOSOMA_WANDB_CHECKPOINT="$checkpoint"
export HOLOSOMA_WANDB_RUN="$run_path"
export HOLOSOMA_MJ_MOTION_INIT="$motion_init"
export HOLOSOMA_RO_AUTO_START="$auto_start"
export HOLOSOMA_RO_AUTO_MOTION="$auto_motion"
export HOLOSOMA_RO_USE_SIM_STATE="$use_sim_state"
export HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND="$force_zero_sparse"
export SIM_STATE_PORT="${SIM_STATE_PORT:-5557}"
if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]]; then
  if [[ "$(basename "$clip" .npz)" == "box_75" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1
  else
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=0
  fi
fi
export PYTHONPATH="${ROOT_DIR}/src/holosoma_inference:${ROOT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}"

echo "[mj_ro] Model path: $model_path"
echo "[mj_ro] Inference config: $inference_config"
echo "[mj_ro] Motion file: $motion_file"
echo "[mj_ro] Zero sparse root command: $HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND"
if [[ "$use_sim_state" == "1" ]]; then
  echo "[mj_ro] Use sim state: True"
else
  echo "[mj_ro] Use sim state: False"
fi

run_args=(
  python3 "${ROOT_DIR}/src/holosoma_inference/holosoma_inference/run_policy.py"
  "inference:${inference_config}"
  --task.model-path "$model_path"
  --task.no-use-joystick
  --task.use-sim-time
  --task.rl-rate 50
  --task.interface lo
  --task.motion-file "$motion_file"
)

if [[ "$use_sim_state" == "1" ]]; then
  run_args+=(--task.use-sim-state --task.sim-state-port "$SIM_STATE_PORT" --task.prefer-sim-ref-from-sim-state)
fi
if [[ "$auto_start" == "1" ]]; then
  run_args+=(--task.auto-start-policy)
fi
if [[ "$auto_motion" == "1" ]]; then
  run_args+=(--task.auto-start-motion-clip)
fi

exec "${run_args[@]}"
