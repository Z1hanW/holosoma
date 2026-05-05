#!/usr/bin/env bash

SIM2REAL_REGISTRY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM2REAL_REGISTRY_FILE="${SIM2REAL_REGISTRY_DIR}/registry.tsv"

sim2real_known_policies() {
  awk 'NR > 1 {print $1}' "$SIM2REAL_REGISTRY_FILE"
}

sim2real_print_registry() {
  column -t -s $'\t' "$SIM2REAL_REGISTRY_FILE" 2>/dev/null || cat "$SIM2REAL_REGISTRY_FILE"
}

sim2real_load_policy() {
  local policy_name="$1"
  local row
  row="$(awk -F '\t' -v policy="$policy_name" 'NR > 1 && $1 == policy {print; exit}' "$SIM2REAL_REGISTRY_FILE")"
  if [[ -z "$row" ]]; then
    return 1
  fi

  IFS=$'\t' read -r SIM2REAL_POLICY_NAME SIM2REAL_RUN_ID SIM2REAL_INFERENCE_CONFIG SIM2REAL_MODEL_REF <<<"$row"
  export SIM2REAL_POLICY_NAME
  export SIM2REAL_RUN_ID
  export SIM2REAL_INFERENCE_CONFIG
  export SIM2REAL_MODEL_REF
}
