#!/usr/bin/env bash
# Run the latest carry-any checkpoint on the real robot with FFS depth.
# Requires real_ffs_depth.sh (laptop) and stereo_relay_pub.py (robot) already up:
# the policy attaches to depth_img_shm at start and fails if it is missing.
#
# WARNING: run_policy.py sends a stiff hold-position command on every loop
# iteration from the moment it starts, before any key is pressed.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

interface="${HOLOSOMA_REAL_INTERFACE:-eth0}"
if ! ip link show "$interface" >/dev/null 2>&1; then
  echo "[real_ffs_run] ERROR: network interface '$interface' does not exist on this host." >&2
  echo "[real_ffs_run] Available: $(ip -brief link show | awk '$1!="lo"{printf "%s ", $1}')" >&2
  echo "[real_ffs_run] Set HOLOSOMA_REAL_INTERFACE=<name> to choose one." >&2
  exit 1
fi

# 4kocpixa (wandb zihanw22/carry-any) exports actor_obs=94 / perception_obs=5046,
# which matches exactly one inference preset.
checkpoint="${HOLOSOMA_REAL_MODEL_PATH:-_ckps/4kocpixa_model_25500.onnx}"
inference_config="${HOLOSOMA_INFERENCE_CONFIG:-g1-root_pos-contact-aware-drop-button-actions-no-linvel-h1}"

# The policy attaches to depth_img_shm at start and raises if it is missing, so
# wait for real_ffs_depth.sh to publish it instead of failing on an ordering race.
# It takes ~10 s to create (relay start + FFS model load), so a plain existence
# check at launch is almost always too early.
shm_wait="${HOLOSOMA_SHM_WAIT_S:-120}"
if [[ ! -e /dev/shm/depth_img_shm ]]; then
  if ! pgrep -f "image_server.py remote_ffs_d435i" >/dev/null 2>&1; then
    echo "[real_ffs_run] real_ffs_depth.sh is not running yet - start it in another terminal; waiting up to ${shm_wait}s for it..." >&2
  else
    echo "[real_ffs_run] waiting for real_ffs_depth.sh to publish depth_img_shm (up to ${shm_wait}s)..." >&2
  fi
  for ((t = 0; t < shm_wait; t++)); do
    [[ -e /dev/shm/depth_img_shm ]] && break
    (( t > 0 && t % 10 == 0 )) && echo "[real_ffs_run]   still waiting (${t}s)..." >&2
    sleep 1
  done
  if [[ ! -e /dev/shm/depth_img_shm ]]; then
    echo "[real_ffs_run] ERROR: depth_img_shm did not appear within ${shm_wait}s. Check the real_ffs_depth.sh terminal for errors." >&2
    exit 1
  fi
  echo "[real_ffs_run] depth_img_shm is up." >&2
fi

log_dir="${ROOT_DIR}/logs/real_ffs_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
exec > >(tee -a "${log_dir}/run.log") 2>&1

echo "[real_ffs_run] log_dir=${log_dir}"
echo "[real_ffs_run] checkpoint=${checkpoint}  config=${inference_config}  interface=${interface}"
python3 scripts/show_policy_command.py "${log_dir}/latest_command.json" &
command_window_pid=$!
trap 'kill "$command_window_pid" 2>/dev/null || true' EXIT
source scripts/source_inference_setup.sh
HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=0 \
HOLOSOMA_POLICY_DROP_BUTTON="${HOLOSOMA_POLICY_DROP_BUTTON:-0}" \
HOLOSOMA_POLICY_COMMAND_STATUS_PATH="${log_dir}/latest_command.json" \
HOLOSOMA_POLICY_DEBUG_INPUT_PATH="${log_dir}/depth_command.jsonl" \
HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT="${HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT:-100000}" \
PYTHONPATH=src/holosoma_inference:src/holosoma${PYTHONPATH:+:${PYTHONPATH}} \
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
  "inference:${inference_config}" \
  --task.model-path "$checkpoint" \
  --task.use-joystick \
  --task.rl-rate 50 \
  --task.interface "$interface"
