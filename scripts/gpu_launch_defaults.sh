#!/usr/bin/env bash

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
