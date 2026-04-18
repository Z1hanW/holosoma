#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <train_log> <target_iteration> <monitor_log>" >&2
  exit 2
fi

TRAIN_LOG="$1"
TARGET_ITER="$2"
MONITOR_LOG="$3"

if [[ ! -f "${TRAIN_LOG}" ]]; then
  echo "[ERROR] Training log not found: ${TRAIN_LOG}" >&2
  exit 2
fi

touch "${MONITOR_LOG}"

last_iter="-1"
last_error_count="0"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

while true; do
  current_iter="$(
    rg -o 'Learning iteration [0-9]+' "${TRAIN_LOG}" 2>/dev/null \
      | awk '{print $3}' \
      | sort -n \
      | tail -n 1
  )"
  current_iter="${current_iter:-0}"

  error_count="$(
    rg -n 'NCCL|timeout|ChildFailedError|Traceback|RuntimeError|miss interactions|PhysX error|Watchdog caught collective operation timeout' \
      "${TRAIN_LOG}" -S 2>/dev/null \
      | wc -l
  )"
  error_count="${error_count//[[:space:]]/}"

  if [[ "${current_iter}" != "${last_iter}" ]]; then
    printf '[%s] iteration=%s target=%s\n' "$(timestamp)" "${current_iter}" "${TARGET_ITER}" >> "${MONITOR_LOG}"
    last_iter="${current_iter}"
  fi

  if [[ "${error_count}" != "${last_error_count}" ]]; then
    printf '[%s] warning_count_changed=%s\n' "$(timestamp)" "${error_count}" >> "${MONITOR_LOG}"
    last_error_count="${error_count}"
  fi

  if [[ "${current_iter}" =~ ^[0-9]+$ ]] && (( current_iter >= TARGET_ITER )); then
    printf '[%s] target_reached iteration=%s\n' "$(timestamp)" "${current_iter}" >> "${MONITOR_LOG}"
    exit 0
  fi

  sleep 30
done
