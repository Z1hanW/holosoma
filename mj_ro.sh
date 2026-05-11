#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${1:-${HOLOSOMA_MJ_MOTION:-box_75}}"
checkpoint="${2:-${HOLOSOMA_WANDB_CHECKPOINT:-latest}}"
run_ref="${3:-${HOLOSOMA_WANDB_RUN:-zihanw22/boxer/w5qostjn}}"

case "$run_ref" in
  ppo_first_contact_aware_h1)
    run_ref="6c7exbeq"
    ;;
  ppo_first_contact_aware_h5)
    run_ref="lk9ocrn6"
    ;;
  ppo_first_h1)
    run_ref="kxnhgj2v"
    ;;
  ppo_first_h5)
    run_ref="iepncc89"
    ;;
esac

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
    6c7exbeq)
      inference_config="g1-root_pos-contact-aware-actions-no-linvel-h1"
      ;;
    lk9ocrn6)
      inference_config="g1-root_pos-contact-aware-actions-no-linvel"
      ;;
    kxnhgj2v)
      inference_config="g1-root_pos-actions-no-linvel-h1"
      ;;
    iepncc89)
      inference_config="g1-root_pos-actions-no-linvel-h5"
      ;;
    w5qostjn)
      inference_config="g1-root_pos-object-perception-no-linvel"
      ;;
    *)
      inference_config="g1-root_pos-object-perception"
      ;;
  esac
fi

force_zero_sparse="${HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND:-}"
if [[ -z "$force_zero_sparse" ]]; then
  case "$run_id" in
    tvtwx4to)
      force_zero_sparse=1
      ;;
    *)
      force_zero_sparse=0
      ;;
  esac
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
)

external_root_pos_run=0
case "$run_id" in
  6c7exbeq|lk9ocrn6|kxnhgj2v|iepncc89)
    external_root_pos_run=1
    ;;
esac

if [[ "$external_root_pos_run" != "1" ]]; then
  run_args+=(--task.motion-file "$motion_file")
fi

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
