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

# The policy attaches to depth_img_shm at start and reads whatever is there. A
# segment left behind by a depth server that was killed (a killed process never
# unlinks it) still satisfies an existence check, and the policy then runs on a
# frozen frame from the previous session. That happened: two runs were fed the
# same stale ceiling image. So require the segment to be LIVE - written by a
# running server and changing - and drop a stale one rather than trust it.
shm_wait="${HOLOSOMA_SHM_WAIT_S:-120}"
depth_server_running() { pgrep -f "image_server.py remote_ffs_d435i" >/dev/null 2>&1; }

if [[ -e /dev/shm/depth_img_shm ]] && ! depth_server_running; then
  echo "[real_ffs_run] stale depth_img_shm from a previous session (no depth server running) - removing it." >&2
  rm -f /dev/shm/depth_img_shm
fi

depth_server_running || echo "[real_ffs_run] real_ffs_depth.sh is not running yet - start it in another terminal; waiting up to ${shm_wait}s..." >&2
for ((t = 0; t < shm_wait; t++)); do
  [[ -e /dev/shm/depth_img_shm ]] && depth_server_running && break
  (( t > 0 && t % 10 == 0 )) && echo "[real_ffs_run]   still waiting for depth (${t}s)..." >&2
  sleep 1
done
if [[ ! -e /dev/shm/depth_img_shm ]] || ! depth_server_running; then
  echo "[real_ffs_run] ERROR: no live depth server within ${shm_wait}s. Check the real_ffs_depth.sh terminal." >&2
  exit 1
fi

# Liveness and sanity: the frame must be updating and must not be almost all
# far-plane (camera pointed at the ceiling / nothing within 3 m). The policy was
# trained on scenes with floor and objects in view; starting it on such input
# has produced large actions within a second on this robot.
depth_check_py="${HOLOSOMA_INFERENCE_PYTHON:-$HOME/.holosoma_deps/miniconda3/envs/hsinference/bin/python3}"
[[ -x "$depth_check_py" ]] || depth_check_py=python3
if ! "$depth_check_py" - <<'EOF'
import sys, time, numpy as np
from multiprocessing import shared_memory, resource_tracker
s = shared_memory.SharedMemory(name="depth_img_shm"); resource_tracker.unregister(s._name, "shared_memory")
a = np.ndarray((1, 1, 58, 87), dtype=np.float32, buffer=s.buf)
snaps = set(); t0 = time.monotonic()
while time.monotonic() - t0 < 1.5:
    snaps.add(a.tobytes()); time.sleep(0.02)
frame = a.copy(); s.close()
far = float(np.isclose(frame, 0.5).mean()); lo, hi = float(frame.min()), float(frame.max())
ok = True
if len(snaps) < 5:
    print(f"[real_ffs_run] ERROR: depth frame is not updating ({len(snaps)} distinct frames in 1.5 s) - the depth server is stalled or this is a stale segment.", file=sys.stderr); ok = False
if not np.isfinite(frame).all() or lo < -0.5001 or hi > 0.5001:
    print(f"[real_ffs_run] ERROR: depth values out of range [{lo:.3f}, {hi:.3f}] / non-finite.", file=sys.stderr); ok = False
if far > 0.8:
    print(f"[real_ffs_run] ERROR: {far*100:.0f}% of the depth image is at the far plane (>3 m). The camera is not seeing the floor/objects - reposition the robot before starting the policy.", file=sys.stderr); ok = False
if ok:
    print(f"[real_ffs_run] depth is live: {len(snaps)} frames/1.5 s, far-plane {far*100:.0f}%, range [{lo:.2f}, {hi:.2f}]", file=sys.stderr)
sys.exit(0 if ok else 1)
EOF
then
  echo "[real_ffs_run] refusing to start the policy on this depth. Set HOLOSOMA_SKIP_DEPTH_CHECK=1 to override." >&2
  [[ "${HOLOSOMA_SKIP_DEPTH_CHECK:-0}" == "1" ]] || exit 1
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
