#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

model_path="${ROOT_DIR}/_ckps_base/z4arqumz_model_39999.onnx"

if [[ "$#" -ne 0 ]]; then
  echo "mj_base_old.sh only runs ${model_path}; it does not accept run/checkpoint arguments." >&2
  exit 2
fi

if [[ ! -f "$model_path" ]]; then
  echo "mj_base_old.sh: missing checkpoint ${model_path}" >&2
  exit 1
fi

export HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}"
export HOLOSOMA_INFERENCE_CONFIG="g1-root_pos-actions-h1"
export HOLOSOMA_MJ_RO_DEBUG="${HOLOSOMA_MJ_RO_DEBUG:-1}"
export HOLOSOMA_RO_USE_SIM_STATE="${HOLOSOMA_RO_USE_SIM_STATE:-1}"

exec bash ./mj_ro.sh \
  scale__any_monitor_43 \
  "$model_path" \
  ppo_first_contact_aware_h1
