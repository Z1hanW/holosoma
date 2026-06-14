#!/usr/bin/env bash

default_python_bin() {
  local candidates=(
    "/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python3"
    "/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python"
    "/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python3"
    "/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python"
    "/home/ubuntu/miniconda3/envs/hssim/bin/python3"
    "/home/ubuntu/miniconda3/envs/hssim/bin/python"
    "/home/ubuntu/miniconda3/envs/sim/bin/python3"
    "/home/ubuntu/miniconda3/envs/sim/bin/python"
    "python3"
    "python"
  )

  local candidate
  local resolved
  for candidate in "${candidates[@]}"; do
    if [[ "${candidate}" == */* ]]; then
      resolved="${candidate}"
    else
      resolved="$(command -v "${candidate}" 2>/dev/null || true)"
    fi
    if [[ -n "${resolved}" && -x "${resolved}" ]]; then
      printf '%s\n' "${resolved}"
      return 0
    fi
  done

  return 1
}

if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$(default_python_bin)" || {
    echo "[ERROR] Unable to resolve a Python executable. Set PYTHON_BIN." >&2
    exit 1
  }
  export PYTHON_BIN
fi

detect_all_cuda_visible_devices() {
  local devices=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    devices="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d '[:blank:]' | paste -sd, -)"
  fi
  printf '%s\n' "${devices}"
}

default_cuda_visible_devices_all() {
  local current="${1:-}"
  if [[ -n "${current}" && "${current}" != "all" && "${current}" != "ALL" ]]; then
    printf '%s\n' "${current}"
    return
  fi

  local devices
  devices="$(detect_all_cuda_visible_devices)"
  if [[ -n "${devices}" ]]; then
    printf '%s\n' "${devices}"
  elif [[ -n "${current}" ]]; then
    printf '%s\n' "${current}"
  else
    printf '0\n'
  fi
}

count_cuda_visible_devices() {
  local devices="${1:-}"
  if [[ "${devices}" == "all" || "${devices}" == "ALL" ]]; then
    devices="$(detect_all_cuda_visible_devices)"
  fi
  devices="${devices//[[:space:]]/}"
  if [[ -z "${devices}" ]]; then
    printf '1\n'
    return
  fi

  local -a visible_gpus=()
  IFS=',' read -r -a visible_gpus <<< "${devices}"
  printf '%s\n' "${#visible_gpus[@]}"
}
