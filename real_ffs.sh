#!/usr/bin/env bash
# One command for the whole FFS deployment.
#
#   bash real_ffs.sh
#
# Starts real_ffs_depth.sh in the background (robot-side stereo relay over ssh,
# Fast-FoundationStereo on the local GPU, depth_img_shm), waits until the depth
# is actually being published, then runs real_ffs_run.sh in the foreground so the
# policy owns the terminal for keyboard/joystick control. One Ctrl-C tears
# everything down in order: policy first, then the depth server, which in turn
# stops the remote relay and restores any robot services it stopped.
#
# Defaults chosen for this robot; every one can be overridden in the environment:
#   HOLOSOMA_REAL_INTERFACE       auto-detected: the NIC holding a 192.168.123.x address
#   HOLOSOMA_RELAY_STOP_SERVICES  lsvla-vision  (autostarted service that holds the camera)
#   HOLOSOMA_REAL_MODEL_PATH      _ckps/4kocpixa_model_25500.onnx  (see real_ffs_run.sh)
#   HOLOSOMA_DRY_RUN=1            bring up depth only, do not launch the policy
#
# WARNING: as soon as the policy starts it sends a stiff hold-position command,
# before any key is pressed. Hand on the e-stop.
set -eo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export HOLOSOMA_RELAY_STOP_SERVICES="${HOLOSOMA_RELAY_STOP_SERVICES-lsvla-vision}"

if [[ -z "${HOLOSOMA_REAL_INTERFACE:-}" ]]; then
  HOLOSOMA_REAL_INTERFACE="$(ip -o -4 addr show 2>/dev/null | awk '/ 192\.168\.123\./{print $2; exit}')"
  if [[ -z "$HOLOSOMA_REAL_INTERFACE" ]]; then
    echo "[real_ffs] ERROR: no interface with a 192.168.123.x address; is the robot cable plugged in?" >&2
    echo "[real_ffs]        (set HOLOSOMA_REAL_INTERFACE to override)" >&2
    exit 1
  fi
  export HOLOSOMA_REAL_INTERFACE
fi

for h in 192.168.123.164; do
  if ! ping -c1 -W1 "$h" >/dev/null 2>&1; then
    echo "[real_ffs] ERROR: robot Jetson $h is not reachable" >&2
    exit 1
  fi
done

session_dir="${ROOT_DIR}/logs/real_ffs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$session_dir"
depth_log="${session_dir}/depth.stdout"
echo "[real_ffs] interface=${HOLOSOMA_REAL_INTERFACE}  stop_services='${HOLOSOMA_RELAY_STOP_SERVICES}'  session=${session_dir}"

depth_pid=""
policy_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$policy_pid" ]] && kill -0 "$policy_pid" 2>/dev/null; then
    echo "[real_ffs] stopping policy..."
    kill -INT "$policy_pid" 2>/dev/null
    for _ in $(seq 1 40); do kill -0 "$policy_pid" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$policy_pid" 2>/dev/null && kill -TERM "$policy_pid" 2>/dev/null
    wait "$policy_pid" 2>/dev/null || true
  fi
  if [[ -n "$depth_pid" ]] && kill -0 "$depth_pid" 2>/dev/null; then
    echo "[real_ffs] stopping depth server (and the remote relay)..."
    kill -INT "$depth_pid" 2>/dev/null
    for _ in $(seq 1 80); do kill -0 "$depth_pid" 2>/dev/null || break; sleep 0.25; done
    kill -0 "$depth_pid" 2>/dev/null && kill -TERM "$depth_pid" 2>/dev/null
    wait "$depth_pid" 2>/dev/null || true
  fi
  echo "[real_ffs] done. logs: ${session_dir}"
}
trap cleanup EXIT
trap 'echo; echo "[real_ffs] Ctrl-C"; cleanup; exit 130' INT TERM

# Depth in its own process group so the terminal's Ctrl-C reaches only this
# script, which then shuts things down in the right order.
rm -f /dev/shm/depth_img_shm
setsid bash "${ROOT_DIR}/real_ffs_depth.sh" > "$depth_log" 2>&1 < /dev/null &
depth_pid=$!
echo "[real_ffs] depth server starting (pid ${depth_pid}); log: ${depth_log}"

# Surface the milestones from the depth log while waiting for the segment.
shm_wait="${HOLOSOMA_SHM_WAIT_S:-120}"
seen=""
for ((t = 0; t < shm_wait; t++)); do
  if ! kill -0 "$depth_pid" 2>/dev/null; then
    echo "[real_ffs] ERROR: depth server exited before publishing depth. Its log:" >&2
    grep -vE "^\s*$|FutureWarning|autocast|warnings.warn" "$depth_log" | tail -15 | sed 's/^/    /' >&2
    exit 1
  fi
  while IFS= read -r line; do
    [[ "$seen" == *"|$line|"* ]] && continue
    seen+="|$line|"
    echo "  $line"
  done < <(grep -E "installing relay|stopped .* \(will restart|relay pid|calibration received|\[FFS\] Initialized|Created new shared memory|ERROR|did not start" "$depth_log" 2>/dev/null | cut -c1-140)
  # "Live" means the frame is changing, not merely that the segment exists: a
  # server whose receiver has died still creates the segment and never writes it.
  if [[ -e /dev/shm/depth_img_shm ]] && "${HOLOSOMA_INFERENCE_PYTHON:-$HOME/.holosoma_deps/miniconda3/envs/hsinference/bin/python3}" - <<'EOF' 2>/dev/null
import sys, time, numpy as np
from multiprocessing import shared_memory, resource_tracker
s = shared_memory.SharedMemory(name="depth_img_shm"); resource_tracker.unregister(s._name, "shared_memory")
a = np.ndarray((1, 1, 58, 87), dtype=np.float32, buffer=s.buf)
seen = set(); t0 = time.monotonic()
while time.monotonic() - t0 < 1.0:
    seen.add(a.tobytes()); time.sleep(0.02)
s.close(); sys.exit(0 if len(seen) >= 5 else 1)
EOF
  then break; fi
  sleep 1
done
if [[ ! -e /dev/shm/depth_img_shm ]] || ! kill -0 "$depth_pid" 2>/dev/null; then
  echo "[real_ffs] ERROR: live depth did not appear within ${shm_wait}s; see ${depth_log}" >&2
  exit 1
fi
echo "[real_ffs] depth is live (frames updating)."

if [[ "${HOLOSOMA_DRY_RUN:-0}" == "1" ]]; then
  echo "[real_ffs] DRY RUN: depth is up, not launching the policy. Ctrl-C to tear down."
  wait "$depth_pid" || true
  exit 0
fi

echo "[real_ffs] launching policy - the robot will stiffen NOW."
bash "${ROOT_DIR}/real_ffs_run.sh" &
policy_pid=$!
wait "$policy_pid" || true
policy_pid=""
