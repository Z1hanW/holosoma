#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
default_model_path="${ROOT_DIR}/_ckps/34qv1qqp_model_40000.onnx"
if [[ ! -f "$default_model_path" ]]; then
  default_model_path="wandb://zihanw22/carry-any/34qv1qqp/latest"
fi
clip="${1:-${HOLOSOMA_MJ_MOTION:-box_75}}"
model_path="${2:-${HOLOSOMA_MJ_MODEL_PATH:-$default_model_path}}"
model_run_id="${HOLOSOMA_MJ_MODEL_RUN_ID:-34qv1qqp}"
model_base="$(basename "$model_path")"
if [[ "$model_base" =~ ^([[:alnum:]]+)_model_[0-9]+\.onnx$ ]]; then
  model_run_id="${BASH_REMATCH[1]}"
elif [[ "$model_path" == wandb://* || "$model_path" == https://wandb.ai/* ]]; then
  model_path_has_checkpoint=0
  if [[ "$model_path" == */files/* ]]; then
    model_path_has_checkpoint=1
  elif [[ "$model_path" == wandb://* ]]; then
    IFS='/' read -r -a model_path_parts <<< "${model_path#wandb://}"
    if [[ "${#model_path_parts[@]}" -ge 4 ]]; then
      model_path_has_checkpoint=1
    fi
  fi

  model_run_path="${model_path#wandb://}"
  model_run_path="${model_run_path#https://wandb.ai/}"
  model_run_path="${model_run_path%%\?*}"
  model_run_path="${model_run_path%%/files/*}"
  model_run_path="${model_run_path%/}"
  model_run_path="${model_run_path/\/runs\//\/}"
  IFS='/' read -r -a model_run_parts <<< "$model_run_path"
  if [[ "${#model_run_parts[@]}" -ge 3 ]]; then
    model_run_id="${model_run_parts[2]}"
    if [[ "$model_path_has_checkpoint" != "1" ]]; then
      model_path="wandb://${model_run_parts[0]}/${model_run_parts[1]}/${model_run_parts[2]}/latest"
    fi
  fi
fi
run_id="$model_run_id"

motion_file="$clip"
if [[ "$clip" != *.npz && "$clip" != /* ]]; then
  motion_file="${ROOT_DIR}/data_demo/${clip}.npz"
fi
if [[ "$motion_file" != /* ]]; then
  motion_file="${ROOT_DIR}/${motion_file}"
fi

inference_config="${HOLOSOMA_INFERENCE_CONFIG:-}"
if [[ -z "$inference_config" ]]; then
  case "$model_run_id" in
    36k1vwdf|zzv6vtkk)
      inference_config="g1-root_pos-contact-aware-pickup-drop-button-actions-no-linvel-h1"
      ;;
    a1lh8uxa|d9m3z369|gjiefd3c|qihvpyqg|swl41n4x|9ez2ivr4|34qv1qqp|6urn4jvc|xm0hda83)
      inference_config="g1-root_pos-contact-aware-drop-button-actions-no-linvel-h1"
      ;;
    1j98x3g1|6c7exbeq)
      inference_config="g1-root_pos-contact-aware-actions-no-linvel-h1"
      ;;
    lk9ocrn6)
      inference_config="g1-root_pos-contact-aware-actions-no-linvel"
      ;;
    kxnhgj2v)
      inference_config="g1-root_pos-actions-no-linvel-h1"
      ;;
    z4arqumz)
      inference_config="g1-root_pos-actions-h1"
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
export HOLOSOMA_POLICY_PICKUP_BUTTON="${HOLOSOMA_POLICY_PICKUP_BUTTON:-1}"
export HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}"
export HOLOSOMA_POLICY_COMMAND_STATUS_PATH="${HOLOSOMA_POLICY_COMMAND_STATUS_PATH:-/tmp/holosoma_policy_command_status.json}"
export HOLOSOMA_POLICY_COMMAND_CONTROL_PATH="${HOLOSOMA_POLICY_COMMAND_CONTROL_PATH:-/tmp/holosoma_policy_command_control.json}"
if [[ -z "${HOLOSOMA_POLICY_MOTION_INDEX_OFFSET:-}" ]]; then
  if [[ "$(basename "$clip" .npz)" == "box_75" ]]; then
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1
  else
    export HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=0
  fi
fi
export PYTHONPATH="${ROOT_DIR}/src/holosoma_inference:${ROOT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${HOLOSOMA_INFERENCE_PYTHON:-}" ]]; then
  python_bin="$HOLOSOMA_INFERENCE_PYTHON"
elif [[ -x "/home/unitree/.holosoma_deps/miniconda3/envs/hsinference/bin/python3" ]]; then
  python_bin="/home/unitree/.holosoma_deps/miniconda3/envs/hsinference/bin/python3"
elif [[ -x "/home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python3" ]]; then
  python_bin="/home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python3"
else
  python_bin="python3"
fi

run_args=(
  "$python_bin" "${ROOT_DIR}/src/holosoma_inference/holosoma_inference/run_policy.py"
  "inference:${inference_config}"
  --task.model-path "$model_path"
  --task.no-use-joystick
  --task.use-sim-time
  --task.rl-rate 50
  --task.interface lo
)

external_root_pos_run=0
case "$model_run_id" in
  1j98x3g1|6c7exbeq|lk9ocrn6|kxnhgj2v|iepncc89|a1lh8uxa|d9m3z369|gjiefd3c|qihvpyqg|swl41n4x|9ez2ivr4|34qv1qqp|6urn4jvc|xm0hda83|36k1vwdf|zzv6vtkk)
    external_root_pos_run=1
    ;;
esac

derive_sparse_root_from_motion="${HOLOSOMA_DERIVE_SPARSE_ROOT_FROM_MOTION:-0}"
case "${derive_sparse_root_from_motion,,}" in
  1|true|yes|on)
    derive_sparse_root_from_motion=1
    ;;
  *)
    derive_sparse_root_from_motion=0
    ;;
esac

if [[ "$external_root_pos_run" != "1" || "$derive_sparse_root_from_motion" == "1" ]]; then
  run_args+=(--task.motion-file "$motion_file")
fi

requires_sim_state=0
case "$model_run_id" in
  z4arqumz)
    requires_sim_state=1
    ;;
esac

use_sim_state=0
if [[ "$requires_sim_state" == "1" ]]; then
  use_sim_state=1
elif [[ "${HOLOSOMA_MJ_RO_DEBUG:-0}" == "1" && "${HOLOSOMA_RO_USE_SIM_STATE:-1}" == "1" ]]; then
  use_sim_state=1
fi

if [[ "$use_sim_state" == "1" ]]; then
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
