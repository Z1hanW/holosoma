#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

COMMAND_SWEEP=${COMMAND_SWEEP:-"0.05 0.10 0.15"}
RUN_TIMEOUT_S=${RUN_TIMEOUT_S:-180}
RUN_GAP_S=${RUN_GAP_S:-5}
VISER_PORT=${VISER_PORT:-7077}
POLICY_CHECKPOINT=${POLICY_CHECKPOINT:-outputs/checkpoints/swl41n4x/model_31500.pt}
PYTHON_BIN=${PYTHON_BIN:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hssim/bin/python}
OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${REPO_ROOT}/data/debug_auto_forward_after_lift_single_clip"}
OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${OMOMO_DATA_DIR}/_clip_object_urdf_map.json"}
OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-1}
OUT_ROOT=${OUT_ROOT:-"${REPO_ROOT}/logs/runtime/auto_forward_command_sweep_$(date +%Y%m%d_%H%M%S)"}

mkdir -p "${OUT_ROOT}"

cat > "${OUT_ROOT}/metadata.txt" <<EOF
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
repo_root=${REPO_ROOT}
command_sweep=${COMMAND_SWEEP}
run_timeout_s=${RUN_TIMEOUT_S}
run_gap_s=${RUN_GAP_S}
viser_port=${VISER_PORT}
policy_checkpoint=${POLICY_CHECKPOINT}
python_bin=${PYTHON_BIN}
omomo_data_dir=${OMOMO_DATA_DIR}
omomo_object_map=${OMOMO_OBJECT_MAP}
omomo_expected_total=${OMOMO_EXPECTED_TOTAL}
EOF

printf "cmd_x\tstatus\tpid\tlog_path\tjsonl_path\tstarted_utc\tended_utc\n" > "${OUT_ROOT}/status.tsv"
echo "[SWEEP] out_root=${OUT_ROOT}"

sanitize_label() {
  printf "%s" "$1" | sed 's/-/m/g; s/+//g; s/\\./p/g; s/,/_/g; s/:/_/g'
}

cleanup_visualize_processes() {
  local signal="${1:-TERM}"
  local pattern="[p]ython -m holosoma.visualize physics.*--motion-dir ${OMOMO_DATA_DIR}.*--viser-port ${VISER_PORT}"
  pkill "-${signal}" -f "${pattern}" 2>/dev/null || true
}

for command_spec in ${COMMAND_SWEEP}; do
  if [[ "${command_spec}" == *:* ]]; then
    label=$(sanitize_label "${command_spec%%:*}")
    command_csv="${command_spec#*:}"
  else
    command_csv="${command_spec}"
    label=$(sanitize_label "${command_spec}")
  fi
  IFS=',' read -r cmd_x cmd_y cmd_yaw <<< "${command_csv}"
  cmd_y=${cmd_y:-0}
  cmd_yaw=${cmd_yaw:-0}
  run_log="${OUT_ROOT}/infer_cmd_${label}.log"
  run_jsonl="${OUT_ROOT}/infer_cmd_${label}.jsonl"
  started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)

  cleanup_visualize_processes TERM
  sleep 2
  cleanup_visualize_processes KILL

  echo "[SWEEP] starting cmd=[${cmd_x},${cmd_y},${cmd_yaw}] label=${label} log=${run_log}"
  setsid bash -lc "
    cd '${REPO_ROOT}' && \
    PYTHON_BIN='${PYTHON_BIN}' \
    OMOMO_DATA_DIR='${OMOMO_DATA_DIR}' \
    OMOMO_OBJECT_MAP='${OMOMO_OBJECT_MAP}' \
    OMOMO_EXPECTED_TOTAL='${OMOMO_EXPECTED_TOTAL}' \
    AS_AUTO_FORWARD_AFTER_LIFT=1 \
    VISER_AUTO_FORWARD_AFTER_LIFT_COMMAND='${cmd_x},${cmd_y},${cmd_yaw}' \
    VISER_AUTO_FORWARD_AFTER_LIFT_DURATION_S='${VISER_AUTO_FORWARD_AFTER_LIFT_DURATION_S:-8.0}' \
    VISER_AUTO_FORWARD_AFTER_LIFT_LOG_PATH='${run_jsonl}' \
    VISER_SHOW_ROLLOUT_ROOT_TRAJECTORY=1 \
    VISER_SHOW_ROLLOUT_OBJECT_TRAJECTORY=1 \
    VISER_ROLLOUT_TRAJECTORY_MESH_WIDTH='${VISER_ROLLOUT_TRAJECTORY_MESH_WIDTH:-0.12}' \
    VISER_ROLLOUT_TRAJECTORY_Z_OFFSET='${VISER_ROLLOUT_TRAJECTORY_Z_OFFSET:-0.20}' \
    VISER_PORT='${VISER_PORT}' \
    LOGURU_LEVEL='${LOGURU_LEVEL:-INFO}' \
    PY_LOG_LEVEL='${PY_LOG_LEVEL:-INFO}' \
    bash infer_as_joystick.sh '${POLICY_CHECKPOINT}'
  " > "${run_log}" 2>&1 &
  pid=$!

  status="timeout"
  for _ in $(seq 1 "${RUN_TIMEOUT_S}"); do
    if rg -q "Auto-forward-after-lift duration complete" "${run_log}" 2>/dev/null; then
      status="complete"
      break
    fi
    if rg -q "Traceback|IndentationError|SyntaxError" "${run_log}" 2>/dev/null; then
      status="error"
      break
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      status="stopped"
      break
    fi
    sleep 1
  done

  echo "[SWEEP] stopping cmd=[${cmd_x},${cmd_y},${cmd_yaw}] label=${label} status=${status} pid=${pid}"
  if kill -0 "${pid}" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
    sleep 3
  fi
  if kill -0 "${pid}" 2>/dev/null; then
    kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
  fi
  wait "${pid}" 2>/dev/null || true
  cleanup_visualize_processes TERM
  sleep 2
  cleanup_visualize_processes KILL

  ended_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "${cmd_x}" "${status}" "${pid}" "${run_log}" "${run_jsonl}" "${started_utc}" "${ended_utc}" \
    >> "${OUT_ROOT}/status.tsv"

  sleep "${RUN_GAP_S}"
done

echo "[SWEEP] complete out_root=${OUT_ROOT}"
