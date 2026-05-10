#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
clip="${HOLOSOMA_MJ_MOTION:-box_75}"
run_ref="${HOLOSOMA_WANDB_RUN:-w5qostjn}"
checkpoint="${HOLOSOMA_WANDB_CHECKPOINT:-latest}"
duration="${HOLOSOMA_DEBUG_DURATION:-45s}"
auto_motion="auto"
use_sim_state="${HOLOSOMA_RO_USE_SIM_STATE:-0}"
record_video="${HOLOSOMA_MJ_DEBUG_RECORD_VIDEO:-0}"
record_fps="${HOLOSOMA_MJ_DEBUG_RECORD_FPS:-10}"
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
    --use-sim-state)
      use_sim_state=1
      ;;
    --no-sim-state)
      use_sim_state=0
      ;;
    --record|--record-mp4)
      record_video=1
      ;;
    --no-record)
      record_video=0
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
if [[ "$use_sim_state" != "1" ]]; then
  use_sim_state=0
fi

run_id="$run_ref"
run_id="${run_id%%/files/*}"
run_id="${run_id##*/}"
if [[ "$run_id" == "runs" || -z "$run_id" ]]; then
  run_id="wandb"
fi

sim_state_label="simstate_${use_sim_state}"
log_dir="${ROOT_DIR}/artifacts/mj_debug_${run_id}_${sim_state_label}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$log_dir"
export HOLOSOMA_POLICY_COMMAND_STATUS_PATH="${log_dir}/policy_command_status.json"
env_log="${log_dir}/env.log"
ro_log="${log_dir}/ro.log"
conda_sh="${HOLOSOMA_CONDA_SH:-/home/user/.holosoma_deps/miniconda3/etc/profile.d/conda.sh}"
bridge_bind_error_pattern="Address already in use|Failed to start (clock publisher|sim state publisher)"
if [[ -z "${HOLOSOMA_DDS_DOMAIN_ID:-}" ]]; then
  export HOLOSOMA_DDS_DOMAIN_ID="$((50 + ($(date +%s) % 100)))"
fi
if [[ -z "${SIM_STATE_PORT:-}" ]]; then
  auto_state_port="$(python3 - <<'PY'
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind(("", 0))
print(sock.getsockname()[1])
sock.close()
PY
)"
  export SIM_STATE_PORT="$auto_state_port"
else
  export SIM_STATE_PORT
fi

env_pid=""
depth_rec_pid=""
screen_rec_pid=""
depth_stop_path="${log_dir}/depth_record.stop"
screen_stop_path="${log_dir}/screen_record.stop"
stop_recorders() {
  if [[ -n "$depth_rec_pid" ]]; then
    if kill -0 "$depth_rec_pid" 2>/dev/null; then
      touch "$depth_stop_path" 2>/dev/null || true
      for _ in $(seq 1 100); do
        if ! kill -0 "$depth_rec_pid" 2>/dev/null; then
          break
        fi
        sleep 0.1
      done
    fi
    if kill -0 "$depth_rec_pid" 2>/dev/null; then
      kill "$depth_rec_pid" 2>/dev/null || true
    fi
    wait "$depth_rec_pid" 2>/dev/null || true
  fi
  if [[ -n "$screen_rec_pid" ]]; then
    if kill -0 "$screen_rec_pid" 2>/dev/null; then
      touch "$screen_stop_path" 2>/dev/null || true
      for _ in $(seq 1 100); do
        if ! kill -0 "$screen_rec_pid" 2>/dev/null; then
          break
        fi
        sleep 0.1
      done
    fi
    if kill -0 "$screen_rec_pid" 2>/dev/null; then
      kill "$screen_rec_pid" 2>/dev/null || true
    fi
    wait "$screen_rec_pid" 2>/dev/null || true
  fi
}
cleanup() {
  stop_recorders
  if [[ -n "$env_pid" ]] && kill -0 "$env_pid" 2>/dev/null; then
    kill "$env_pid" 2>/dev/null || true
    wait "$env_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "[mj_debug] log_dir=$log_dir"
env_args=("$clip" --motion-init)
ro_args=("$clip" "$checkpoint" "$run_ref")
(
  source "$conda_sh"
  conda activate hsmujoco
  export HOLOSOMA_MJ_DEBUG_LIFT_TELEMETRY=1
  bash "${ROOT_DIR}/mj_env.sh" "${env_args[@]}"
) >"$env_log" 2>&1 &
env_pid=$!

for _ in $(seq 1 80); do
  if ! kill -0 "$env_pid" 2>/dev/null; then
    echo "[mj_debug] mj_env.sh exited before readiness"
    tail -n 80 "$env_log" || true
    exit 1
  fi
  if grep -Eq "$bridge_bind_error_pattern" "$env_log"; then
    echo "[mj_debug] simulator bridge port bind failed; another MuJoCo env is probably still running"
    grep -E "$bridge_bind_error_pattern" "$env_log" || true
    pgrep -af "${ROOT_DIR}/src/holosoma/holosoma/run_sim.py" || true
    exit 1
  fi
  if grep -q "ImageServer initialized" "$env_log" && [[ -e /dev/shm/depth_img_shm ]]; then
    break
  fi
  sleep 0.5
done

if grep -Eq "$bridge_bind_error_pattern" "$env_log"; then
  echo "[mj_debug] simulator bridge port bind failed; another MuJoCo env is probably still running"
  grep -E "$bridge_bind_error_pattern" "$env_log" || true
  pgrep -af "${ROOT_DIR}/src/holosoma/holosoma/run_sim.py" || true
  exit 1
fi

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
PY
)

if [[ "$record_video" == "1" ]]; then
  echo "[mj_debug] recording policy depth mp4: ${log_dir}/depth_observation.mp4"
  rm -f "$depth_stop_path"
  rm -f "$screen_stop_path"
  (
    source "$conda_sh"
    conda activate hsmujoco
    DEPTH_RECORD_PATH="${log_dir}/depth_observation.mp4" \
    DEPTH_RECORD_STATS_PATH="${log_dir}/depth_record_stats.json" \
    DEPTH_RECORD_STOP_PATH="$depth_stop_path" \
    DEPTH_RECORD_FPS="$record_fps" \
    python3 - <<'PY'
import json
import os
import signal
import time

import cv2
import numpy as np

stop = False


def _stop(signum, frame):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

shape = (1, 1, 58, 87)
scale = 8
fps = float(os.environ.get("DEPTH_RECORD_FPS", "10"))
out_path = os.environ["DEPTH_RECORD_PATH"]
stats_path = os.environ["DEPTH_RECORD_STATS_PATH"]
stop_path = os.environ["DEPTH_RECORD_STOP_PATH"]
depth = np.memmap("/dev/shm/depth_img_shm", dtype=np.float32, mode="r", shape=shape)
writer = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (shape[-1] * scale, shape[-2] * scale),
    True,
)
if not writer.isOpened():
    raise RuntimeError(f"failed to open depth video writer: {out_path}")

count = 0
mins = []
maxs = []
means = []
period = 1.0 / fps
try:
    while not stop and not os.path.exists(stop_path):
        frame = np.array(depth[0, 0], copy=True)
        mins.append(float(np.min(frame)))
        maxs.append(float(np.max(frame)))
        means.append(float(np.mean(frame)))
        gray = np.clip((frame + 0.5) * 255.0, 0, 255).astype(np.uint8)
        gray = cv2.resize(gray, (shape[-1] * scale, shape[-2] * scale), interpolation=cv2.INTER_NEAREST)
        color = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
        writer.write(color)
        count += 1
        time.sleep(period)
finally:
    writer.release()
    summary = {
        "frames": count,
        "fps": fps,
        "path": out_path,
        "min": min(mins) if mins else None,
        "max": max(maxs) if maxs else None,
        "mean_min": min(means) if means else None,
        "mean_max": max(means) if means else None,
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
PY
  ) >"${log_dir}/depth_recorder.log" 2>&1 &
  depth_rec_pid=$!

  ffmpeg_bin=""
  if command -v ffmpeg >/dev/null 2>&1 && ffmpeg -hide_banner -formats 2>/dev/null | grep -q x11grab; then
    ffmpeg_bin="$(command -v ffmpeg)"
  elif [[ -x /usr/bin/ffmpeg ]] && /usr/bin/ffmpeg -hide_banner -formats 2>/dev/null | grep -q x11grab; then
    ffmpeg_bin="/usr/bin/ffmpeg"
  fi
  if [[ -n "${DISPLAY:-}" && -n "$ffmpeg_bin" ]] && command -v xwininfo >/dev/null 2>&1; then
    sleep 1
    window_line="$(xwininfo -root -tree 2>/dev/null | grep -Ei 'mujoco|MuJoCo' | head -n 1 || true)"
    if [[ "$window_line" =~ [[:space:]]([0-9]+)x([0-9]+)\+(-?[0-9]+)\+(-?[0-9]+)[[:space:]] ]]; then
      win_w="${BASH_REMATCH[1]}"
      win_h="${BASH_REMATCH[2]}"
      win_x="${BASH_REMATCH[3]}"
      win_y="${BASH_REMATCH[4]}"
      echo "[mj_debug] recording MuJoCo window mp4: ${log_dir}/viewer_capture.mp4 (${win_w}x${win_h}+${win_x},${win_y})"
      "$ffmpeg_bin" -hide_banner -loglevel warning -y \
        -f x11grab -framerate "$record_fps" -video_size "${win_w}x${win_h}" \
        -i "${DISPLAY}+${win_x},${win_y}" -pix_fmt yuv420p \
        "${log_dir}/viewer_capture.mp4" >"${log_dir}/viewer_capture.log" 2>&1 &
      screen_rec_pid=$!
    else
      echo "[mj_debug] MuJoCo window not found for x11 capture; depth mp4 will still be saved"
    fi
  elif [[ -n "${DISPLAY:-}" ]] && command -v xwininfo >/dev/null 2>&1; then
    sleep 1
    window_line="$(xwininfo -root -tree 2>/dev/null | grep -Ei 'mujoco|MuJoCo' | head -n 1 || true)"
    if [[ "$window_line" =~ [[:space:]]([0-9]+)x([0-9]+)\+(-?[0-9]+)\+(-?[0-9]+)[[:space:]] ]]; then
      win_w="${BASH_REMATCH[1]}"
      win_h="${BASH_REMATCH[2]}"
      win_x="${BASH_REMATCH[3]}"
      win_y="${BASH_REMATCH[4]}"
      echo "[mj_debug] recording MuJoCo window mp4: ${log_dir}/viewer_capture.mp4 (${win_w}x${win_h}+${win_x},${win_y}, PIL)"
      (
        source "$conda_sh"
        conda activate hsmujoco
        SCREEN_RECORD_PATH="${log_dir}/viewer_capture.mp4" \
        SCREEN_RECORD_STOP_PATH="$screen_stop_path" \
        SCREEN_RECORD_FPS="$record_fps" \
        SCREEN_RECORD_BBOX="${win_x},${win_y},${win_w},${win_h}" \
        python3 - <<'PY'
import json
import os
import signal
import time

import cv2
import numpy as np
from PIL import ImageGrab

stop = False


def _stop(signum, frame):
    global stop
    stop = True


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)

out_path = os.environ["SCREEN_RECORD_PATH"]
stop_path = os.environ["SCREEN_RECORD_STOP_PATH"]
fps = min(float(os.environ.get("SCREEN_RECORD_FPS", "10")), float(os.environ.get("SCREEN_RECORD_FPS_MAX", "5")))
x, y, w, h = [int(v) for v in os.environ["SCREEN_RECORD_BBOX"].split(",")]
max_width = int(os.environ.get("SCREEN_RECORD_MAX_WIDTH", "960"))
out_w = w
out_h = h
if out_w > max_width:
    scale = max_width / float(out_w)
    out_w = max_width
    out_h = int(round(out_h * scale))
out_w -= out_w % 2
out_h -= out_h % 2
writer = cv2.VideoWriter(
    out_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (out_w, out_h),
    True,
)
if not writer.isOpened():
    raise RuntimeError(f"failed to open screen video writer: {out_path}")
period = 1.0 / fps
count = 0
try:
    while not stop and not os.path.exists(stop_path):
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        frame = np.asarray(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
        count += 1
        time.sleep(period)
finally:
    writer.release()
    stats = {
        "frames": count,
        "fps": fps,
        "source_bbox": [x, y, w, h],
        "width": out_w,
        "height": out_h,
        "path": out_path,
    }
    with open(os.path.splitext(out_path)[0] + "_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True)
PY
      ) >"${log_dir}/viewer_capture.log" 2>&1 &
      screen_rec_pid=$!
    else
      echo "[mj_debug] MuJoCo window not found for PIL capture; depth mp4 will still be saved"
    fi
  fi
fi

if [[ "$auto_motion" == "1" ]]; then
  ro_auto_motion=1
else
  ro_auto_motion=0
fi

set +e
(
  source "$conda_sh"
  conda activate hsinference
  export HOLOSOMA_MJ_RO_DEBUG=1
  export HOLOSOMA_MJ_MOTION_INIT=1
  export HOLOSOMA_RO_AUTO_START=1
  export HOLOSOMA_RO_AUTO_MOTION="$ro_auto_motion"
  export HOLOSOMA_RO_USE_SIM_STATE="$use_sim_state"
  export HOLOSOMA_POLICY_DEBUG_INPUT_PATH="${log_dir}/policy_debug.jsonl"
  export HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT="${HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT:-240}"
  timeout "$duration" bash "${ROOT_DIR}/mj_ro.sh" "${ro_args[@]}" </dev/null
) >"$ro_log" 2>&1
ro_status=$?
set -e
stop_recorders

METRICS_PATH="${log_dir}/metrics.json" \
METRICS_CSV_PATH="${log_dir}/metrics.csv" \
METRICS_RUN_REF="$run_ref" \
METRICS_CHECKPOINT="$checkpoint" \
METRICS_CLIP="$clip" \
METRICS_DURATION="$duration" \
METRICS_USE_SIM_STATE="$use_sim_state" \
python3 - <<'PY'
import csv
import json
import os
import re
from pathlib import Path

log_dir = Path(os.environ["METRICS_PATH"]).parent
env_log = log_dir / "env.log"
ro_log = log_dir / "ro.log"
policy_debug = log_dir / "policy_debug.jsonl"


def strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", text)


env_text = strip_ansi(env_log.read_text(errors="replace") if env_log.exists() else "")
ro_text = strip_ansi(ro_log.read_text(errors="replace") if ro_log.exists() else "")

metrics = {
    "run_ref": os.environ["METRICS_RUN_REF"],
    "checkpoint": os.environ["METRICS_CHECKPOINT"],
    "clip": os.environ["METRICS_CLIP"],
    "duration": os.environ["METRICS_DURATION"],
    "use_sim_state_requested": os.environ["METRICS_USE_SIM_STATE"] == "1",
    "sim_state_subscriber_started": "Sim state subscriber started" in ro_text,
    "first_lowcmd_received": bool(
        re.search(r"Received first external active lowcmd|Received first ZMQ lowcmd", env_text)
    ),
}

metrics["use_sim_state_effective"] = metrics["sim_state_subscriber_started"]

bridge_line = next((line for line in env_text.splitlines() if "BridgeConfig(" in line), "")
metrics["bridge"] = {}
if bridge_line:
    for key in (
        "domain_id",
        "publish_sim_state",
        "sim_state_port",
        "ignore_default_idle_command",
        "hold_default_pose_until_first_command",
    ):
        match = re.search(rf"{key}=([^,\)]+)", bridge_line)
        if match:
            raw = match.group(1)
            if raw in {"True", "False"}:
                metrics["bridge"][key] = raw == "True"
            else:
                try:
                    metrics["bridge"][key] = int(raw)
                except ValueError:
                    metrics["bridge"][key] = raw

match = re.search(
    r"Matched motion-init low state: yaw current=([-+0-9.]+) deg expected=([-+0-9.]+) deg joint_max_err=([-+0-9.eE]+)",
    ro_text,
)
if match:
    metrics["motion_init_match"] = {
        "matched": True,
        "yaw_current_deg": float(match.group(1)),
        "yaw_expected_deg": float(match.group(2)),
        "joint_max_err": float(match.group(3)),
    }
else:
    timeout = re.search(
        r"Timed out waiting for motion-init low state yaw: last=([-+0-9.]+) deg, expected=([-+0-9.]+) deg, joint_max_err=([-+0-9.eE]+)",
        ro_text,
    )
    if timeout:
        metrics["motion_init_match"] = {
            "matched": False,
            "yaw_current_deg": float(timeout.group(1)),
            "yaw_expected_deg": float(timeout.group(2)),
            "joint_max_err": float(timeout.group(3)),
        }

telemetry = []
telemetry_re = re.compile(
    r"LiftTelemetry t=([-+0-9.]+).*?object_pos=\[([^\]]+)\] dz=([-+0-9.]+) max_dz=([-+0-9.]+).*?"
    r"contacts object=([0-9]+) robot=([0-9]+) terrain=([0-9]+).*?"
    r"robot_pos=\[([^\]]+)\] robot_yaw=([-+0-9.]+)"
)


def parse_vec(raw: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)]


for line in env_text.splitlines():
    m = telemetry_re.search(line)
    if not m:
        continue
    obj = parse_vec(m.group(2))
    robot = parse_vec(m.group(8))
    telemetry.append(
        {
            "t": float(m.group(1)),
            "object_pos": obj,
            "dz": float(m.group(3)),
            "max_dz": float(m.group(4)),
            "object_contacts": int(m.group(5)),
            "robot_contacts": int(m.group(6)),
            "terrain_contacts": int(m.group(7)),
            "robot_pos": robot,
            "robot_yaw": float(m.group(9)),
        }
    )

metrics["lift_telemetry_count"] = len(telemetry)
if telemetry:
    best = max(telemetry, key=lambda row: row["max_dz"])
    best_z = max(telemetry, key=lambda row: row["object_pos"][2] if len(row["object_pos"]) >= 3 else -999)
    robot_z = [row["robot_pos"][2] for row in telemetry if len(row["robot_pos"]) >= 3]
    metrics["lift"] = {
        "initial_object_z": telemetry[0]["object_pos"][2],
        "max_object_z": best_z["object_pos"][2],
        "max_dz": best["max_dz"],
        "t_at_max_dz": best["t"],
        "final_dz": telemetry[-1]["dz"],
        "final_object_z": telemetry[-1]["object_pos"][2],
        "robot_z_initial": robot_z[0] if robot_z else None,
        "robot_z_min": min(robot_z) if robot_z else None,
        "robot_drop_from_initial": (robot_z[0] - min(robot_z)) if robot_z else None,
    }

if policy_debug.exists():
    policy_rows = []
    for line in policy_debug.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            policy_rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    metrics["policy_debug_count"] = len(policy_rows)
    if policy_rows:
        def max_abs(path: tuple[str, ...]) -> float | None:
            vals = []
            for row in policy_rows:
                cur = row
                for key in path:
                    if not isinstance(cur, dict) or key not in cur:
                        cur = None
                        break
                    cur = cur[key]
                if isinstance(cur, dict) and "absmax" in cur:
                    vals.append(float(cur["absmax"]))
            return max(vals) if vals else None

        sparse_vals = []
        for row in policy_rows:
            for key in (
                "sparse_target_root_trajectory_command",
                "sparse_target_root_trajectory_command_contact_aware",
            ):
                val = row.get(key)
                if isinstance(val, list):
                    sparse_vals.extend(abs(float(x)) for x in val)
        metrics["policy"] = {
            "max_obs_absmax": max_abs(("input", "obs")),
            "max_perception_absmax": max_abs(("input", "perception_obs")),
            "max_policy_action_absmax": max_abs(("policy_action",)),
            "max_q_target_absmax": max_abs(("q_target",)),
            "max_sparse_root_command_abs": max(sparse_vals) if sparse_vals else None,
        }

video_paths = {}
for name in ("depth_observation.mp4", "viewer_capture.mp4"):
    path = log_dir / name
    if path.exists():
        video_paths[name] = str(path)
if video_paths:
    metrics["videos"] = video_paths

depth_stats = log_dir / "depth_record_stats.json"
if depth_stats.exists():
    try:
        metrics["depth_recording"] = json.loads(depth_stats.read_text())
    except json.JSONDecodeError:
        pass

Path(os.environ["METRICS_PATH"]).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
with open(os.environ["METRICS_CSV_PATH"], "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "run_ref",
            "checkpoint",
            "use_sim_state_effective",
            "sim_state_subscriber_started",
            "first_lowcmd_received",
            "motion_init_joint_max_err",
            "max_dz",
            "t_at_max_dz",
            "robot_drop_from_initial",
            "max_sparse_root_command_abs",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "run_ref": metrics.get("run_ref"),
            "checkpoint": metrics.get("checkpoint"),
            "use_sim_state_effective": metrics.get("use_sim_state_effective"),
            "sim_state_subscriber_started": metrics.get("sim_state_subscriber_started"),
            "first_lowcmd_received": metrics.get("first_lowcmd_received"),
            "motion_init_joint_max_err": metrics.get("motion_init_match", {}).get("joint_max_err"),
            "max_dz": metrics.get("lift", {}).get("max_dz"),
            "t_at_max_dz": metrics.get("lift", {}).get("t_at_max_dz"),
            "robot_drop_from_initial": metrics.get("lift", {}).get("robot_drop_from_initial"),
            "max_sparse_root_command_abs": metrics.get("policy", {}).get("max_sparse_root_command_abs"),
        }
    )
PY

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
echo "[mj_debug] metrics=${log_dir}/metrics.json"
