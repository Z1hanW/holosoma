#!/usr/bin/env bash

# Python chooses its hash secret before user code runs.  A call to
# os.environ["PYTHONHASHSEED"] inside train_agent.py is therefore too late and
# gives parent/child processes different ordering semantics.  Every supported
# launcher exports one deterministic interpreter-level seed up front; callers
# may choose another explicit integer when it is part of their experiment.
PYTHONHASHSEED=${PYTHONHASHSEED:-0}
if [[ ! "${PYTHONHASHSEED}" =~ ^[0-9]+$ ]] \
  || (( ${#PYTHONHASHSEED} > 10 )) \
  || (( 10#${PYTHONHASHSEED} > 4294967295 )); then
  echo "[ERROR] PYTHONHASHSEED must be an integer in [0, 4294967295]. Got: ${PYTHONHASHSEED}" >&2
  exit 1
fi
export PYTHONHASHSEED

# cuBLAS chooses internal workspaces when its first handle is initialized.
# Export the reproducible multi-stream configuration before any Python/Isaac
# process starts; setting it later from seeding() can be after CUDA startup.
CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
case "${CUBLAS_WORKSPACE_CONFIG}" in
  :4096:8|:16:8)
    ;;
  *)
    echo "[ERROR] CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8. Got: ${CUBLAS_WORKSPACE_CONFIG}" >&2
    exit 1
    ;;
esac
export CUBLAS_WORKSPACE_CONFIG

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
    if [[ -n "${resolved}" && -x "${resolved}" ]] \
      && "${resolved}" -c 'import torch' >/dev/null 2>&1; then
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
fi

# Keep every nested launcher and legacy helper on the same interpreter.  Remote
# non-login shells do not necessarily inherit the conda PATH, so a bare
# `python3` in a child wrapper can otherwise silently select /usr/bin/python3
# even though the parent resolved the hssim interpreter correctly.
if [[ "${PYTHON_BIN}" == */* ]]; then
  _holosoma_python_candidate="${PYTHON_BIN}"
else
  _holosoma_python_candidate="$(command -v -- "${PYTHON_BIN}" 2>/dev/null || true)"
fi
if [[ -z "${_holosoma_python_candidate}" || ! -x "${_holosoma_python_candidate}" ]]; then
  echo "[ERROR] PYTHON_BIN is not an executable: ${PYTHON_BIN}" >&2
  unset _holosoma_python_candidate
  exit 1
fi
PYTHON_BIN="$(readlink -f -- "${_holosoma_python_candidate}")"
unset _holosoma_python_candidate
if ! "${PYTHON_BIN}" -c 'import torch' >/dev/null 2>&1; then
  echo "[ERROR] PYTHON_BIN cannot import torch: ${PYTHON_BIN}" >&2
  exit 1
fi
HOLOSOMA_PYTHON_PROFILE=${HOLOSOMA_PYTHON_PROFILE:-torch}
case "${HOLOSOMA_PYTHON_PROFILE}" in
  torch)
    ;;
  hssim)
    if ! "${PYTHON_BIN}" -c \
      'import pathlib, sys; raise SystemExit(0 if pathlib.Path(sys.prefix).resolve().name == "hssim" else 1)' \
      >/dev/null 2>&1; then
      echo "[ERROR] HOLOSOMA_PYTHON_PROFILE=hssim requires an hssim environment: ${PYTHON_BIN}" >&2
      exit 1
    fi
    ;;
  *)
    echo "[ERROR] HOLOSOMA_PYTHON_PROFILE must be torch or hssim. Got: ${HOLOSOMA_PYTHON_PROFILE}" >&2
    exit 1
    ;;
esac
export HOLOSOMA_PYTHON_PROFILE
export PYTHON_BIN
_holosoma_python_dir="$(dirname -- "${PYTHON_BIN}")"
if [[ "${PATH:-}:" != "${_holosoma_python_dir}:"* ]]; then
  export PATH="${_holosoma_python_dir}${PATH:+:${PATH}}"
fi
unset _holosoma_python_dir

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
  if [[ "${devices}" == "-1" || "${devices}" == ,* || "${devices}" == *, || "${devices}" == *,,* ]]; then
    echo "[ERROR] CUDA_VISIBLE_DEVICES must be a non-empty comma-separated list without empty/disabled entries. Got: ${1:-<empty>}" >&2
    return 1
  fi

  local -a visible_gpus=()
  IFS=',' read -r -a visible_gpus <<< "${devices}"
  local -a seen_visible_gpus=()
  local device seen_device
  for device in "${visible_gpus[@]}"; do
    for seen_device in "${seen_visible_gpus[@]}"; do
      if [[ "${device}" == "${seen_device}" ]]; then
        echo "[ERROR] CUDA_VISIBLE_DEVICES must not select the same GPU token twice: ${devices}" >&2
        return 1
      fi
    done
    seen_visible_gpus+=("${device}")
  done
  printf '%s\n' "${#visible_gpus[@]}"
}
