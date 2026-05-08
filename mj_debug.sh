#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${HOLOSOMA_MJ_MOTION:-box_75}"
run_ref="${HOLOSOMA_WANDB_RUN:-tvtwx4to}"
checkpoint="${HOLOSOMA_WANDB_CHECKPOINT:-latest}"
duration="${HOLOSOMA_DEBUG_DURATION:-45s}"
auto_motion="auto"
positional=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clip)
      shift
      clip="$1"
      ;;
    --run)
      shift
      run_ref="$1"
      ;;
    --checkpoint)
      shift
      checkpoint="$1"
      ;;
    --duration)
      shift
      duration="$1"
      ;;
    --auto-motion)
      auto_motion=1
      ;;
    --no-auto-motion)
      auto_motion=0
      ;;
    *)
      positional+=("$1")
      ;;
  esac
  shift
done

if (( ${#positional[@]} >= 1 )); then
  clip="${positional[0]}"
fi
if (( ${#positional[@]} >= 2 )); then
  run_ref="${positional[1]}"
fi
if (( ${#positional[@]} >= 3 )); then
  checkpoint="${positional[2]}"
fi

if [[ "$auto_motion" == "auto" ]]; then
  auto_motion=1
fi

run_id="$run_ref"
run_id="${run_id%%/files/*}"
run_id="${run_id##*/}"
if [[ "$run_id" == "runs" || -z "$run_id" ]]; then
  run_id="wandb"
fi

log_dir="${ROOT_DIR}/artifacts/mj_debug_${run_id}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
env_log="${log_dir}/env.log"
ro_log="${log_dir}/ro.log"
conda_sh="${HOLOSOMA_CONDA_SH:-/home/user/.holosoma_deps/miniconda3/etc/profile.d/conda.sh}"
if [[ -z "${HOLOSOMA_DDS_DOMAIN_ID:-}" ]]; then
  export HOLOSOMA_DDS_DOMAIN_ID="$((50 + ($(date +%s) % 100)))"
fi

env_pid=""
cleanup() {
  if [[ -n "$env_pid" ]] && kill -0 "$env_pid" 2>/dev/null; then
    kill "$env_pid" 2>/dev/null || true
    wait "$env_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[mj_debug] log_dir=$log_dir"
echo "[mj_debug] starting env: clip=$clip motion_init=1 domain=$HOLOSOMA_DDS_DOMAIN_ID"
(
  source "$conda_sh"
  conda activate hsmujoco
  export HOLOSOMA_MJ_DEBUG_LIFT_TELEMETRY=1
  bash "${ROOT_DIR}/mj_env.sh" "$clip" --motion-init
) >"$env_log" 2>&1 &
env_pid=$!

for _ in $(seq 1 80); do
  if ! kill -0 "$env_pid" 2>/dev/null; then
    echo "[mj_debug] mj_env.sh exited before readiness"
    tail -n 80 "$env_log" || true
    exit 1
  fi
  if grep -q "ImageServer initialized" "$env_log" && [[ -e /dev/shm/depth_img_shm ]]; then
    break
  fi
  sleep 0.5
done

if ! grep -q "ImageServer initialized" "$env_log"; then
  echo "[mj_debug] image server did not become ready"
  tail -n 80 "$env_log" || true
  exit 1
fi

(
  source "$conda_sh"
  conda activate hsinference
  python3 - <<'PY'
import os
import time
import numpy as np

shape = (1, 1, 58, 87)
path = "/dev/shm/depth_img_shm"
expected_bytes = int(np.prod(shape) * np.dtype(np.float32).itemsize)
actual_bytes = os.stat(path).st_size
assert actual_bytes == expected_bytes, f"depth shm bytes mismatch: {actual_bytes} != {expected_bytes}"
depth = np.memmap(path, dtype=np.float32, mode="r", shape=shape)
deadline = time.monotonic() + 20.0
while time.monotonic() < deadline:
    assert np.isfinite(depth).all(), "depth shm contains non-finite values"
    if float(np.max(np.abs(depth))) > 1e-6 and float(depth.max() - depth.min()) > 1e-6:
        break
    time.sleep(0.1)
else:
    raise AssertionError("depth shm stayed zero or constant before rollout")
print(
    f"[mj_debug] depth shm ok: shape={shape}, bytes={actual_bytes}, "
    f"min={float(depth.min()):.4f}, max={float(depth.max()):.4f}",
    flush=True,
)
PY
)

ro_args=("$clip" "$checkpoint" "$run_ref" --motion-init --auto-start)
if [[ "$auto_motion" == "1" ]]; then
  ro_args+=(--auto-motion)
fi

echo "[mj_debug] starting rollout: run=$run_ref checkpoint=$checkpoint auto_motion=$auto_motion duration=$duration"
set +e
(
  source "$conda_sh"
  conda activate hsinference
  export HOLOSOMA_POLICY_DEBUG_INPUT_PATH="${log_dir}/policy_debug.jsonl"
  export HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT="${HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT:-240}"
  timeout "$duration" bash "${ROOT_DIR}/mj_ro.sh" "${ro_args[@]}"
) >"$ro_log" 2>&1
ro_status=$?
set -e

if [[ "$ro_status" -ne 0 && "$ro_status" -ne 124 ]]; then
  echo "[mj_debug] mj_ro.sh failed with status $ro_status"
  tail -n 120 "$ro_log" || true
  exit "$ro_status"
fi

if ! grep -Eq "Received first external active lowcmd|Received first ZMQ lowcmd" "$env_log"; then
  echo "[mj_debug] rollout did not reach MuJoCo bridge"
  tail -n 80 "$env_log" || true
  tail -n 80 "$ro_log" || true
  exit 1
fi

echo "[mj_debug] rollout completed status=$ro_status"
tail -n 40 "$ro_log" || true
