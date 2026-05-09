#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${1:-${HOLOSOMA_MJ_MOTION:-box_75}}"
checkpoint="${2:-${HOLOSOMA_WANDB_CHECKPOINT:-latest}}"
run_ref="${3:-${HOLOSOMA_WANDB_RUN:-zihanw22/boxer/w5qostjn}}"

motion_file="$clip"
if [[ "$clip" != *.npz && "$clip" != /* ]]; then
  motion_file="${ROOT_DIR}/data_demo/${clip}.npz"
fi
if [[ "$motion_file" != /* ]]; then
  motion_file="${ROOT_DIR}/${motion_file}"
fi

run_path="${run_ref#wandb://}"
run_path="${run_path#https://wandb.ai/}"
run_path="${run_path%%/files/*}"
run_path="${run_path%/}"
run_path="${run_path/\/runs\//\/}"
if [[ "$run_path" != */* ]]; then
  run_path="zihanw22/boxer/$run_path"
fi
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

export HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND="$force_zero_sparse"
if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]]; then
  if [[ "$(basename "$clip" .npz)" == "box_75" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1
  else
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=0
  fi
fi
export PYTHONPATH="${ROOT_DIR}/src/holosoma_inference:${ROOT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}"

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

if [[ "${HOLOSOMA_MJ_RO_DEBUG:-0}" == "1" && "${HOLOSOMA_RO_USE_SIM_STATE:-1}" == "1" ]]; then
  export SIM_STATE_PORT="${SIM_STATE_PORT:-5557}"
  run_args+=(--task.use-sim-state --task.sim-state-port "$SIM_STATE_PORT" --task.prefer-sim-ref-from-sim-state)
fi
if [[ "${HOLOSOMA_MJ_RO_DEBUG:-0}" == "1" && "${HOLOSOMA_RO_AUTO_START:-0}" == "1" ]]; then
  run_args+=(--task.auto-start-policy)
fi
if [[ "${HOLOSOMA_MJ_RO_DEBUG:-0}" == "1" && "${HOLOSOMA_RO_AUTO_MOTION:-0}" == "1" ]]; then
  run_args+=(--task.auto-start-motion-clip)
fi

exec "${run_args[@]}"
