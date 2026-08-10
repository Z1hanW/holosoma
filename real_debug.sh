#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

interface="${HOLOSOMA_REAL_INTERFACE:-eth0}"
default_model_path="${ROOT_DIR}/_ckps/sx9wctkd_model_40000.onnx"
model_path="${HOLOSOMA_REAL_DEBUG_MODEL_PATH:-${HOLOSOMA_REAL_MODEL_PATH:-$default_model_path}}"
depth_server_config="${HOLOSOMA_REAL_DEBUG_IMAGE_SERVER_CONFIG:-${HOLOSOMA_REAL_IMAGE_SERVER_CONFIG:-real_d435i_urdf}}"

log_dir="${ROOT_DIR}/logs/real_debug_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/run.log") 2>&1
command_status_path="${log_dir}/latest_command.json"
viser_pid=""
depth_pid=""
depth_process_group=""
sim_gt_pid=""
sim_gt_process_group=""

cleanup() {
  if [[ -n "$viser_pid" ]]; then
    kill "$viser_pid" 2>/dev/null || true
    wait "$viser_pid" 2>/dev/null || true
  fi
  if [[ -n "$depth_process_group" ]]; then
    kill -- "-$depth_process_group" 2>/dev/null || true
  elif [[ -n "$depth_pid" ]]; then
    kill "$depth_pid" 2>/dev/null || true
  fi
  if [[ -n "$depth_pid" ]]; then
    wait "$depth_pid" 2>/dev/null || true
  fi
  if [[ -n "$sim_gt_process_group" ]]; then
    kill -- "-$sim_gt_process_group" 2>/dev/null || true
  elif [[ -n "$sim_gt_pid" ]]; then
    kill "$sim_gt_pid" 2>/dev/null || true
  fi
  if [[ -n "$sim_gt_pid" ]]; then
    wait "$sim_gt_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[real_debug] log_dir=${log_dir}"
echo "[real_debug] model_path=${model_path}"
echo "[real_debug] interface=${interface}"
echo "[real_debug] target: Unitree L2+A suspended diagnostic position"
echo "[real_debug] joint target: all 29 G1 joints at 0 rad (zero-joint posture)"
echo "[real_debug] stiffness: high WBT stiff gains for straight standing legs/waist"
echo "[real_debug] transition: smooth 5-second move from the measured joint pose"
echo "[real_debug] policy and motion activation are locked out"
echo "[real_debug] keep the robot supported and press Enter only when the area is clear"

if [[ "${HOLOSOMA_REAL_DEBUG_DEPTH:-1}" != "0" ]]; then
  echo "[real_debug] depth config: ${depth_server_config} (0mcqao8k D435-URDF profile)"
  if command -v setsid >/dev/null 2>&1; then
    env HOLOSOMA_REAL_IMAGE_SERVER_CONFIG="$depth_server_config" setsid bash real_depth.sh &
    depth_pid=$!
    depth_process_group=$depth_pid
  else
    HOLOSOMA_REAL_IMAGE_SERVER_CONFIG="$depth_server_config" bash real_depth.sh &
    depth_pid=$!
  fi
  echo "[real_debug] depth server started pid=${depth_pid}"
else
  echo "[real_debug] depth server disabled by HOLOSOMA_REAL_DEBUG_DEPTH=0"
fi

if [[ "${HOLOSOMA_REAL_DEBUG_SIM_GT:-1}" != "0" ]]; then
  sim_gt_log="${log_dir}/sim_gt_depth.log"
  if command -v setsid >/dev/null 2>&1; then
    env HOLOSOMA_SIM_GT_STATE_PATH="$command_status_path" setsid bash sim_gt_depth.sh >"$sim_gt_log" 2>&1 &
    sim_gt_pid=$!
    sim_gt_process_group=$sim_gt_pid
  else
    HOLOSOMA_SIM_GT_STATE_PATH="$command_status_path" bash sim_gt_depth.sh >"$sim_gt_log" 2>&1 &
    sim_gt_pid=$!
  fi
  echo "[real_debug] MuJoCo sim GT started pid=${sim_gt_pid} log=${sim_gt_log}"
else
  echo "[real_debug] MuJoCo sim GT disabled by HOLOSOMA_REAL_DEBUG_SIM_GT=0"
fi

source scripts/source_inference_setup.sh

if [[ "${HOLOSOMA_REAL_VISER:-1}" != "0" ]]; then
  viser_browser_args=()
  if [[ "${HOLOSOMA_REAL_VISER_OPEN_BROWSER:-1}" != "0" ]] \
      && { [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; }; then
    viser_browser_args+=(--open-browser)
  fi
  PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
  python3 scripts/real_viser.py \
    --state-path "$command_status_path" \
    --host "${HOLOSOMA_REAL_VISER_HOST:-127.0.0.1}" \
    --port "${HOLOSOMA_REAL_VISER_PORT:-8080}" \
    --depth-profile "Real D435: 0mcqao8k processing" \
    --depth-source-height 60 \
    --depth-source-width 106 \
    --depth-crop-y-start 2 \
    --depth-crop-x-start 4 \
    --depth-crop-x-end -4 \
    --sim-gt-depth-shm-name "${HOLOSOMA_REAL_DEBUG_SIM_GT_SHM_NAME:-sim_gt_depth_raw_shm}" \
    "${viser_browser_args[@]}" &
  viser_pid=$!
  echo "[real_viser] debug viewer started pid=${viser_pid}"
else
  echo "[real_viser] disabled by HOLOSOMA_REAL_VISER=0"
fi

HOLOSOMA_POLICY_COMMAND_STATUS_PATH="$command_status_path" \
PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
  inference:g1-debug-diagnostic \
  --task.model-path "$model_path" \
  --task.use-joystick \
  --task.rl-rate 50 \
  --task.interface "$interface"
