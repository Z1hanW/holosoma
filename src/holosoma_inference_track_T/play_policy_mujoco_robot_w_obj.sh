#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

EVAL_AGENT_SCRIPT="${REPO_ROOT}/src/holosoma/holosoma/eval_agent.py"
RUN_SIM_SCRIPT="${REPO_ROOT}/src/holosoma/holosoma/run_sim.py"
RUN_POLICY_SCRIPT="${REPO_ROOT}/src/holosoma_inference/holosoma_inference/run_policy.py"
DEFAULT_ONNX="/home/ANT.AMAZON.COM/zzzihanw/FAR/ckp/singlebox_07000.onnx"
DEFAULT_CKPT="/home/ANT.AMAZON.COM/zzzihanw/FAR/ckp/singlebox_07000.pt"
DEFAULT_MOTION="${REPO_ROOT}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
DEFAULT_OBJECT_URDF="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"

MODEL_OR_CKPT=${1:-${MODEL_OR_CKPT:-"${DEFAULT_ONNX}"}}
AUTO_LAUNCH=${AUTO_LAUNCH:-"True"}
INTERFACE=${INTERFACE:-"lo"}
INFERENCE_CONFIG=${INFERENCE_CONFIG:-"inference:g1-29dof-wbt-w-object"}
RL_RATE=${RL_RATE:-"50.0"}
USE_SIM_TIME=${USE_SIM_TIME:-"False"}
SIM_FPS=${SIM_FPS:-"200"}
SIM_CONTROL_DECIMATION=${SIM_CONTROL_DECIMATION:-"4"}
SIM_SUBSTEPS=${SIM_SUBSTEPS:-""}
MUJOCO_BACKEND=${MUJOCO_BACKEND:-""}
SIM_PHYSX_SOLVER_TYPE=${SIM_PHYSX_SOLVER_TYPE:-""}
SIM_PHYSX_NUM_POSITION_ITER=${SIM_PHYSX_NUM_POSITION_ITER:-""}
SIM_PHYSX_NUM_VELOCITY_ITER=${SIM_PHYSX_NUM_VELOCITY_ITER:-""}
TERRAIN_STATIC_FRICTION=${TERRAIN_STATIC_FRICTION:-""}
TERRAIN_DYNAMIC_FRICTION=${TERRAIN_DYNAMIC_FRICTION:-""}
USE_TRAINING_SIM_CONFIG=${USE_TRAINING_SIM_CONFIG:-"True"}
HEADLESS=${HEADLESS:-"False"}
AUTO_START=${AUTO_START:-"True"}
AUTO_START_STIFF_HOLD_SEC=${AUTO_START_STIFF_HOLD_SEC:-"0.0"}
AUTO_START_STIFF_MAX_WAIT_SEC=${AUTO_START_STIFF_MAX_WAIT_SEC:-"0.2"}
OBJECT_STATE_PORT=${OBJECT_STATE_PORT-}
TRACKING_VIZ_PORT=${TRACKING_VIZ_PORT-}
TRACKING_VIZ_ENABLED=${TRACKING_VIZ_ENABLED:-"True"}
TRACKING_VIZ_FUTURE_STEPS=${TRACKING_VIZ_FUTURE_STEPS:-"10"}
TRACKING_VIZ_MARKER_SCALE=${TRACKING_VIZ_MARKER_SCALE:-"2.5"}
MUJOCO_OBJECT_MASS_SCALE=${MUJOCO_OBJECT_MASS_SCALE:-"2.0"}
MUJOCO_OBJECT_MASS_OVERRIDE=${MUJOCO_OBJECT_MASS_OVERRIDE:-""}
ROLLOUT_LOG_EVERY=${ROLLOUT_LOG_EVERY:-"20"}
ROLLOUT_SUCCESS_MIN_STEP=${ROLLOUT_SUCCESS_MIN_STEP:-"2000"}
ROLLOUT_REQUIRE_SUCCESS_ROWS=${ROLLOUT_REQUIRE_SUCCESS_ROWS:-"1"}
ROLLOUT_REQUIRE_DIRECTIONAL_MOTION=${ROLLOUT_REQUIRE_DIRECTIONAL_MOTION:-"True"}
ROLLOUT_DIRECTION_MIN_STEP=${ROLLOUT_DIRECTION_MIN_STEP:-"0"}
ROLLOUT_DIRECTION_MIN_DURATION_SEC=${ROLLOUT_DIRECTION_MIN_DURATION_SEC:-"2.0"}
ROLLOUT_DIRECTION_MIN_COSINE=${ROLLOUT_DIRECTION_MIN_COSINE:-"0.5"}
ROLLOUT_DIRECTION_MIN_PAIR_COSINE=${ROLLOUT_DIRECTION_MIN_PAIR_COSINE:-"0.3"}
ROLLOUT_DIRECTION_MIN_ROOT_SPEED_MPS=${ROLLOUT_DIRECTION_MIN_ROOT_SPEED_MPS:-"0.05"}
ROLLOUT_DIRECTION_MIN_OBJECT_SPEED_MPS=${ROLLOUT_DIRECTION_MIN_OBJECT_SPEED_MPS:-"0.05"}
ROLLOUT_DIRECTION_MIN_ROOT_NET_DISP=${ROLLOUT_DIRECTION_MIN_ROOT_NET_DISP:-"0.25"}
ROLLOUT_DIRECTION_MIN_OBJECT_NET_DISP=${ROLLOUT_DIRECTION_MIN_OBJECT_NET_DISP:-"0.25"}
SIM_READY_TIMEOUT_SEC=${SIM_READY_TIMEOUT_SEC:-"40"}
SIM_READY_POLL_SEC=${SIM_READY_POLL_SEC:-"0.2"}

OBJECT_URDF=${OBJECT_URDF:-"${DEFAULT_OBJECT_URDF}"}
MOTION_FILE=${MOTION_FILE:-"${DEFAULT_MOTION}"}
TS=$(date -u +%Y%m%d_%H%M%S)
ROLLOUT_POSE_LOG=${ROLLOUT_POSE_LOG:-"logs/replay_motion_mujoco/policy_robot_w_obj_pose_${TS}.csv"}

find_free_tcp_port() {
  python3 - <<'PY'
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

is_tcp_port_busy() {
  local port="$1"
  python3 - "${port}" <<'PY'
import socket
import sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("0.0.0.0", port))
except OSError:
    print("1")
else:
    print("0")
finally:
    s.close()
PY
}

resolve_comm_ports() {
  local object_port_defaulted tracking_port_defaulted
  object_port_defaulted="0"
  tracking_port_defaulted="0"

  if [[ -z "${OBJECT_STATE_PORT:-}" ]]; then
    OBJECT_STATE_PORT="5557"
    object_port_defaulted="1"
  fi
  if [[ -z "${TRACKING_VIZ_PORT:-}" ]]; then
    TRACKING_VIZ_PORT="5560"
    tracking_port_defaulted="1"
  fi

  if [[ "${OBJECT_STATE_PORT}" == "${TRACKING_VIZ_PORT}" ]]; then
    if [[ "${tracking_port_defaulted}" == "1" ]]; then
      TRACKING_VIZ_PORT="$(find_free_tcp_port)"
      echo "[play_policy_mujoco_robot_w_obj] TRACKING_VIZ_PORT collided with OBJECT_STATE_PORT; reassigned to ${TRACKING_VIZ_PORT}"
    else
      echo "[play_policy_mujoco_robot_w_obj] ERROR: OBJECT_STATE_PORT and TRACKING_VIZ_PORT must differ (both=${OBJECT_STATE_PORT})" >&2
      exit 1
    fi
  fi

  local object_busy tracking_busy
  object_busy="$(is_tcp_port_busy "${OBJECT_STATE_PORT}")"
  tracking_busy="$(is_tcp_port_busy "${TRACKING_VIZ_PORT}")"

  if [[ "${object_busy}" == "1" ]]; then
    if [[ "${object_port_defaulted}" == "1" ]]; then
      OBJECT_STATE_PORT="$(find_free_tcp_port)"
      echo "[play_policy_mujoco_robot_w_obj] OBJECT_STATE_PORT busy; reassigned to ${OBJECT_STATE_PORT}"
    else
      echo "[play_policy_mujoco_robot_w_obj] ERROR: OBJECT_STATE_PORT=${OBJECT_STATE_PORT} is already in use." >&2
      exit 1
    fi
  fi

  if [[ "${tracking_busy}" == "1" ]]; then
    if [[ "${tracking_port_defaulted}" == "1" ]]; then
      TRACKING_VIZ_PORT="$(find_free_tcp_port)"
      echo "[play_policy_mujoco_robot_w_obj] TRACKING_VIZ_PORT busy; reassigned to ${TRACKING_VIZ_PORT}"
    else
      echo "[play_policy_mujoco_robot_w_obj] ERROR: TRACKING_VIZ_PORT=${TRACKING_VIZ_PORT} is already in use." >&2
      exit 1
    fi
  fi

  if [[ "${OBJECT_STATE_PORT}" == "${TRACKING_VIZ_PORT}" ]]; then
    TRACKING_VIZ_PORT="$(find_free_tcp_port)"
    echo "[play_policy_mujoco_robot_w_obj] adjusted TRACKING_VIZ_PORT to avoid collision: ${TRACKING_VIZ_PORT}"
  fi
}

report_model_motion_alignment() {
  local model_path="$1"
  local motion_path="$2"
  local repo_root="$3"
  python3 - "${model_path}" "${motion_path}" "${repo_root}" <<'PY'
import json
from pathlib import Path
import sys

model_path = Path(sys.argv[1]).expanduser()
motion_path = Path(sys.argv[2]).expanduser()
repo_root = Path(sys.argv[3]).expanduser()
prefix = "[play_policy_mujoco_robot_w_obj]"

if model_path.suffix.lower() != ".onnx":
    print(f"{prefix} model alignment check skipped (not ONNX): {model_path}")
    raise SystemExit(0)

try:
    import onnx
except Exception as exc:
    print(f"{prefix} model alignment check skipped (onnx import failed): {exc}")
    raise SystemExit(0)

if not model_path.is_file():
    print(f"{prefix} model alignment check skipped (model missing): {model_path}")
    raise SystemExit(0)

try:
    model = onnx.load(str(model_path))
except Exception as exc:
    print(f"{prefix} model alignment check skipped (onnx load failed): {exc}")
    raise SystemExit(0)

metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

exp_cfg = metadata.get("experiment_config")
motion_cfg = None
if isinstance(exp_cfg, dict):
    motion_cfg = (
        exp_cfg.get("command", {})
        .get("setup_terms", {})
        .get("motion_command", {})
        .get("params", {})
        .get("motion_config", {})
    )

if isinstance(motion_cfg, dict):
    print(f"{prefix} model metadata num_future_steps: {motion_cfg.get('num_future_steps')}")
    print(f"{prefix} model metadata target_pose_type: {motion_cfg.get('target_pose_type')}")
else:
    print(f"{prefix} model metadata motion_config: n/a")

metadata_motion_file = motion_cfg.get("motion_file") if isinstance(motion_cfg, dict) else None
if not metadata_motion_file:
    print(f"{prefix} model metadata motion_file: n/a")
    raise SystemExit(0)

metadata_motion_path = Path(str(metadata_motion_file)).expanduser()
is_same = str(metadata_motion_path) == str(motion_path)
exists = metadata_motion_path.exists()

print(f"{prefix} model metadata motion_file: {metadata_motion_path}")
print(f"{prefix} launcher motion_file: {motion_path}")
print(f"{prefix} motion source match: {is_same}")
print(f"{prefix} metadata motion path exists: {exists}")

if not exists:
    as_text = str(metadata_motion_path)
    if as_text.startswith("/home/ubuntu/FAR/holosoma/"):
        rel = as_text.replace("/home/ubuntu/FAR/holosoma/", "", 1)
        remap = repo_root / rel
        print(f"{prefix} metadata path remap candidate: {remap} (exists={remap.exists()})")
PY
}

is_truthy() {
  case "${1,,}" in
    1|true|yes|on)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

extract_training_sim_overrides() {
  local model_path="$1"
  python3 - "${model_path}" <<'PY'
import json
from pathlib import Path
import sys

model_path = Path(sys.argv[1]).expanduser()
if model_path.suffix.lower() != ".onnx" or not model_path.is_file():
    raise SystemExit(0)

try:
    import onnx
except Exception:
    raise SystemExit(0)

try:
    model = onnx.load(str(model_path))
except Exception:
    raise SystemExit(0)

metadata = {}
for prop in model.metadata_props:
    try:
        metadata[prop.key] = json.loads(prop.value)
    except Exception:
        metadata[prop.key] = prop.value

exp_cfg = metadata.get("experiment_config")
if not isinstance(exp_cfg, dict):
    raise SystemExit(0)

sim_cfg = {}
sim_parent = exp_cfg.get("simulator")
if isinstance(sim_parent, dict):
    sim_cfg = sim_parent.get("config") if isinstance(sim_parent.get("config"), dict) else {}
sim_cfg = sim_cfg if isinstance(sim_cfg, dict) else {}
sim = sim_cfg.get("sim") if isinstance(sim_cfg.get("sim"), dict) else {}
physx = sim.get("physx") if isinstance(sim.get("physx"), dict) else {}

terrain_term = {}
terrain_cfg = exp_cfg.get("terrain")
if isinstance(terrain_cfg, dict):
    terrain_term = terrain_cfg.get("terrain_term") if isinstance(terrain_cfg.get("terrain_term"), dict) else {}

def emit(key: str, value):
    if value is None:
        return
    if isinstance(value, bool):
        text = "True" if value else "False"
    elif isinstance(value, (int, float, str)):
        text = str(value)
    else:
        return
    print(f"{key}={text}")

emit("SIM_FPS", sim.get("fps"))
emit("SIM_CONTROL_DECIMATION", sim.get("control_decimation"))
emit("SIM_SUBSTEPS", sim.get("substeps"))
backend = sim_cfg.get("mujoco_backend")
if isinstance(backend, str):
    emit("MUJOCO_BACKEND", backend.upper())
emit("SIM_PHYSX_SOLVER_TYPE", physx.get("solver_type"))
emit("SIM_PHYSX_NUM_POSITION_ITER", physx.get("num_position_iterations"))
emit("SIM_PHYSX_NUM_VELOCITY_ITER", physx.get("num_velocity_iterations"))
emit("TERRAIN_STATIC_FRICTION", terrain_term.get("static_friction"))
emit("TERRAIN_DYNAMIC_FRICTION", terrain_term.get("dynamic_friction"))
PY
}

apply_training_sim_overrides() {
  local model_path="$1"
  if ! is_truthy "${USE_TRAINING_SIM_CONFIG}"; then
    echo "[play_policy_mujoco_robot_w_obj] training sim override disabled (USE_TRAINING_SIM_CONFIG=${USE_TRAINING_SIM_CONFIG})"
    return
  fi
  if [[ -z "${model_path}" || "${model_path}" != *.onnx ]]; then
    return
  fi

  local found="0"
  while IFS='=' read -r key value; do
    [[ -z "${key}" ]] && continue
    found="1"
    case "${key}" in
      SIM_FPS) SIM_FPS="${value}" ;;
      SIM_CONTROL_DECIMATION) SIM_CONTROL_DECIMATION="${value}" ;;
      SIM_SUBSTEPS) SIM_SUBSTEPS="${value}" ;;
      MUJOCO_BACKEND) MUJOCO_BACKEND="${value}" ;;
      SIM_PHYSX_SOLVER_TYPE) SIM_PHYSX_SOLVER_TYPE="${value}" ;;
      SIM_PHYSX_NUM_POSITION_ITER) SIM_PHYSX_NUM_POSITION_ITER="${value}" ;;
      SIM_PHYSX_NUM_VELOCITY_ITER) SIM_PHYSX_NUM_VELOCITY_ITER="${value}" ;;
      TERRAIN_STATIC_FRICTION) TERRAIN_STATIC_FRICTION="${value}" ;;
      TERRAIN_DYNAMIC_FRICTION) TERRAIN_DYNAMIC_FRICTION="${value}" ;;
      *) ;;
    esac
  done < <(extract_training_sim_overrides "${model_path}")

  if [[ "${found}" == "1" ]]; then
    echo "[play_policy_mujoco_robot_w_obj] applied training sim config: fps=${SIM_FPS}, decimation=${SIM_CONTROL_DECIMATION}, substeps=${SIM_SUBSTEPS:-n/a}, backend=${MUJOCO_BACKEND:-n/a}, physx_iters=${SIM_PHYSX_NUM_POSITION_ITER:-n/a}/${SIM_PHYSX_NUM_VELOCITY_ITER:-n/a}, terrain_friction=${TERRAIN_STATIC_FRICTION:-n/a}/${TERRAIN_DYNAMIC_FRICTION:-n/a}"
  else
    echo "[play_policy_mujoco_robot_w_obj] training sim config not found in ONNX metadata; keeping launcher defaults"
  fi
}

summarize_rollout_log() {
  local csv_path="$1"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[play_policy_mujoco_robot_w_obj] rollout log missing: ${csv_path}"
    return
  fi
  python3 - \
    "${csv_path}" \
    "${ROLLOUT_REQUIRE_SUCCESS_ROWS}" \
    "${ROLLOUT_REQUIRE_DIRECTIONAL_MOTION}" \
    "${ROLLOUT_DIRECTION_MIN_STEP}" \
    "${ROLLOUT_DIRECTION_MIN_DURATION_SEC}" \
    "${ROLLOUT_DIRECTION_MIN_COSINE}" \
    "${ROLLOUT_DIRECTION_MIN_PAIR_COSINE}" \
    "${ROLLOUT_DIRECTION_MIN_ROOT_SPEED_MPS}" \
    "${ROLLOUT_DIRECTION_MIN_OBJECT_SPEED_MPS}" \
    "${ROLLOUT_DIRECTION_MIN_ROOT_NET_DISP}" \
    "${ROLLOUT_DIRECTION_MIN_OBJECT_NET_DISP}" <<'PY'
import csv
import math
import sys

csv_path = sys.argv[1]
required_success = int(sys.argv[2])
require_direction = str(sys.argv[3]).strip().lower() in {"1", "true", "yes", "on"}
direction_min_step = int(float(sys.argv[4]))
direction_min_duration_sec = float(sys.argv[5])
direction_min_cos = float(sys.argv[6])
direction_min_pair_cos = float(sys.argv[7])
direction_min_root_speed = float(sys.argv[8])
direction_min_object_speed = float(sys.argv[9])
direction_min_root_net_disp = float(sys.argv[10])
direction_min_object_net_disp = float(sys.argv[11])
prefix = "[play_policy_mujoco_robot_w_obj]"

with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

if not rows:
    print(f"{prefix} rollout summary: rows=0, success_rows=0")
    print(f"{prefix} rollout verdict: FAIL (empty rollout csv)")
    raise SystemExit(0)

def count(field, value="1"):
    return sum(1 for row in rows if row.get(field) == value)

def minmax(field):
    vals = []
    for row in rows:
        raw = row.get(field)
        if raw is None:
            continue
        try:
            val = float(raw)
        except ValueError:
            continue
        if not math.isnan(val):
            vals.append(val)
    if not vals:
        return float("nan"), float("nan")
    return min(vals), max(vals)

def fval(row, field):
    raw = row.get(field)
    if raw is None or raw == "":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")

def ival(row, field):
    raw = row.get(field)
    if raw is None or raw == "":
        return -1
    try:
        return int(float(raw))
    except ValueError:
        return -1

rows_n = len(rows)
success_rows = count("success")
raw_success_rows = count("raw_success")
on_floor_rows = count("on_floor")
facing_up_rows = count("facing_up")
root_z_min, root_z_max = minmax("root_z")
up_dot_min, up_dot_max = minmax("up_dot")
foot_h_min, foot_h_max = minmax("max_foot_height")
last = rows[-1]

print(
    f"{prefix} rollout summary: rows={rows_n}, success_rows={success_rows}, raw_success_rows={raw_success_rows}, "
    f"on_floor_rows={on_floor_rows}, facing_up_rows={facing_up_rows}"
)
print(
    f"{prefix} rollout ranges: root_z=[{root_z_min:.3f},{root_z_max:.3f}] "
    f"up_dot=[{up_dot_min:.3f},{up_dot_max:.3f}] max_foot_height=[{foot_h_min:.3f},{foot_h_max:.3f}]"
)
print(
    f"{prefix} rollout last: step={last.get('step', '?')} root_z={last.get('root_z', '?')} "
    f"up_dot={last.get('up_dot', '?')} max_foot_height={last.get('max_foot_height', '?')} "
    f"on_floor={last.get('on_floor', '?')} facing_up={last.get('facing_up', '?')} "
    f"success={last.get('success', '?')} object_z={last.get('object_z', '?')} "
    f"init_dof_max_err={last.get('init_dof_max_err', 'n/a')}"
)

obj_qerr_min, obj_qerr_max = minmax("object_quat_err_init_deg")
obj_qerr_alt_min, obj_qerr_alt_max = minmax("object_quat_err_init_if_xyzw_deg")
obj_qnorm_min, obj_qnorm_max = minmax("object_quat_norm")
if not math.isnan(obj_qerr_min):
    print(
        f"{prefix} object quat vs init: err_if_motion_wxyz=[{obj_qerr_min:.2f},{obj_qerr_max:.2f}] deg "
        f"err_if_motion_xyzw=[{obj_qerr_alt_min:.2f},{obj_qerr_alt_max:.2f}] deg "
        f"quat_norm=[{obj_qnorm_min:.6f},{obj_qnorm_max:.6f}]"
    )

obj_robot_contacts_min, obj_robot_contacts_max = minmax("object_robot_contact_count")
obj_scene_contacts_min, obj_scene_contacts_max = minmax("object_scene_contact_count")
obj_robot_pen_min, obj_robot_pen_max = minmax("object_robot_max_pen")
obj_scene_pen_min, obj_scene_pen_max = minmax("object_scene_max_pen")
if not math.isnan(obj_robot_contacts_min):
    print(
        f"{prefix} object contacts: robot_count=[{obj_robot_contacts_min:.0f},{obj_robot_contacts_max:.0f}] "
        f"scene_count=[{obj_scene_contacts_min:.0f},{obj_scene_contacts_max:.0f}] "
        f"robot_pen=[{obj_robot_pen_min:.4f},{obj_robot_pen_max:.4f}] m "
        f"scene_pen=[{obj_scene_pen_min:.4f},{obj_scene_pen_max:.4f}] m"
    )

directional_reasons = []
directional_pass = True
if require_direction:
    eval_rows = [row for row in rows if ival(row, "step") >= direction_min_step]
    root_heading = (float("nan"), float("nan"))
    root_net_disp = float("nan")
    object_net_disp = float("nan")
    longest_good_duration = 0.0
    longest_good_rows = 0
    median_dt = float("nan")

    if len(eval_rows) < 2:
        directional_pass = False
        directional_reasons.append("direction_eval_rows_lt_2")
    else:
        dt_samples = []
        for idx in range(1, len(eval_rows)):
            dt = fval(eval_rows[idx], "sim_time") - fval(eval_rows[idx - 1], "sim_time")
            if dt > 0 and math.isfinite(dt):
                dt_samples.append(dt)
        if dt_samples:
            dt_samples.sort()
            median_dt = dt_samples[len(dt_samples) // 2]

        root_start = (fval(eval_rows[0], "root_x"), fval(eval_rows[0], "root_y"))
        root_end = (fval(eval_rows[-1], "root_x"), fval(eval_rows[-1], "root_y"))
        object_start = (fval(eval_rows[0], "object_x"), fval(eval_rows[0], "object_y"))
        object_end = (fval(eval_rows[-1], "object_x"), fval(eval_rows[-1], "object_y"))

        if any(not math.isfinite(v) for v in (*root_start, *root_end)):
            directional_pass = False
            directional_reasons.append("missing_root_xy")
        if any(not math.isfinite(v) for v in (*object_start, *object_end)):
            directional_pass = False
            directional_reasons.append("missing_object_xy")

        if directional_pass:
            root_dx = root_end[0] - root_start[0]
            root_dy = root_end[1] - root_start[1]
            object_dx = object_end[0] - object_start[0]
            object_dy = object_end[1] - object_start[1]
            root_net_disp = math.hypot(root_dx, root_dy)
            object_net_disp = math.hypot(object_dx, object_dy)

            if root_net_disp < direction_min_root_net_disp:
                directional_pass = False
                directional_reasons.append("root_net_disp_too_small")
            if object_net_disp < direction_min_object_net_disp:
                directional_pass = False
                directional_reasons.append("object_net_disp_too_small")

            if directional_pass:
                root_heading = (root_dx / (root_net_disp + 1e-9), root_dy / (root_net_disp + 1e-9))
                run_rows = 0
                run_duration = 0.0
                for idx in range(1, len(eval_rows)):
                    prev = eval_rows[idx - 1]
                    curr = eval_rows[idx]
                    dt = fval(curr, "sim_time") - fval(prev, "sim_time")
                    if not math.isfinite(dt) or dt <= 0:
                        dt = median_dt if math.isfinite(median_dt) and median_dt > 0 else 0.0

                    drx = fval(curr, "root_x") - fval(prev, "root_x")
                    dry = fval(curr, "root_y") - fval(prev, "root_y")
                    dox = fval(curr, "object_x") - fval(prev, "object_x")
                    doy = fval(curr, "object_y") - fval(prev, "object_y")
                    if not all(math.isfinite(v) for v in (drx, dry, dox, doy)):
                        good = False
                    else:
                        root_step_disp = math.hypot(drx, dry)
                        obj_step_disp = math.hypot(dox, doy)
                        root_speed = root_step_disp / max(dt, 1e-9)
                        obj_speed = obj_step_disp / max(dt, 1e-9)
                        root_dir_cos = (drx * root_heading[0] + dry * root_heading[1]) / max(root_step_disp, 1e-9)
                        obj_dir_cos = (dox * root_heading[0] + doy * root_heading[1]) / max(obj_step_disp, 1e-9)
                        pair_cos = (drx * dox + dry * doy) / max(root_step_disp * obj_step_disp, 1e-9)
                        stable = curr.get("raw_success", "1") == "1"
                        good = (
                            stable
                            and root_speed >= direction_min_root_speed
                            and obj_speed >= direction_min_object_speed
                            and root_dir_cos >= direction_min_cos
                            and obj_dir_cos >= direction_min_cos
                            and pair_cos >= direction_min_pair_cos
                        )

                    if good:
                        run_rows += 1
                        run_duration += max(dt, 0.0)
                    else:
                        if run_duration > longest_good_duration:
                            longest_good_duration = run_duration
                            longest_good_rows = run_rows
                        run_rows = 0
                        run_duration = 0.0

                if run_duration > longest_good_duration:
                    longest_good_duration = run_duration
                    longest_good_rows = run_rows

                if longest_good_duration < direction_min_duration_sec:
                    directional_pass = False
                    directional_reasons.append("directional_duration_too_short")

    print(
        f"{prefix} directional summary: enabled=True eval_rows={len(eval_rows)} eval_min_step={direction_min_step} "
        f"root_heading_xy=[{root_heading[0]:.3f},{root_heading[1]:.3f}] "
        f"root_net_disp={root_net_disp:.3f}m object_net_disp={object_net_disp:.3f}m "
        f"longest_good_duration={longest_good_duration:.3f}s longest_good_rows={longest_good_rows} "
        f"threshold_duration={direction_min_duration_sec:.3f}s"
    )
else:
    print(f"{prefix} directional summary: enabled=False")

overall_pass = success_rows >= required_success and directional_pass
if overall_pass:
    print(
        f"{prefix} rollout verdict: PASS (success_rows >= {required_success}"
        + ("" if not require_direction else ", directional_motion_ok")
        + ")"
    )
else:
    reasons = []
    if success_rows < required_success:
        if on_floor_rows == 0:
            reasons.append("never_on_floor")
        if facing_up_rows == 0:
            reasons.append("never_facing_up")
        if not reasons:
            if on_floor_rows < facing_up_rows:
                reasons.append("mostly_not_on_floor")
            elif facing_up_rows < on_floor_rows:
                reasons.append("mostly_not_facing_up")
            else:
                reasons.append("both_conditions_rare")
    if require_direction and not directional_pass:
        reasons.extend(directional_reasons)
    if not reasons:
        reasons.append("unknown")
    print(f"{prefix} rollout verdict: FAIL (reason={'+'.join(reasons)})")
PY
}

wait_for_sim_ready() {
  local timeout_sec="$1"
  local start_ts now elapsed
  start_ts=$(date +%s)
  while true; do
    if [[ -n "${SIM_PID}" ]] && ! kill -0 "${SIM_PID}" >/dev/null 2>&1; then
      echo "[play_policy_mujoco_robot_w_obj] simulator exited before ready"
      return 1
    fi
    if [[ -s "${ROLLOUT_POSE_LOG}" ]]; then
      local header
      header="$(head -n 1 "${ROLLOUT_POSE_LOG}" || true)"
      if [[ "${header}" == step,sim_time,* ]]; then
        echo "[play_policy_mujoco_robot_w_obj] simulator ready (rollout log initialized)"
        return 0
      fi
    fi
    now=$(date +%s)
    elapsed=$((now - start_ts))
    if (( elapsed >= timeout_sec )); then
      echo "[play_policy_mujoco_robot_w_obj] readiness wait timed out after ${timeout_sec}s; continuing"
      return 0
    fi
    sleep "${SIM_READY_POLL_SEC}"
  done
}

SIM_PID=""
POLICY_PID=""
SHOULD_SUMMARIZE="0"
CLEANUP_DONE="0"
cleanup_run() {
  if [[ "${CLEANUP_DONE}" == "1" ]]; then
    return
  fi
  CLEANUP_DONE="1"
  if [[ -n "${POLICY_PID}" ]] && kill -0 "${POLICY_PID}" >/dev/null 2>&1; then
    kill "${POLICY_PID}" >/dev/null 2>&1 || true
    wait "${POLICY_PID}" 2>/dev/null || true
  fi
  if [[ -n "${SIM_PID}" ]] && kill -0 "${SIM_PID}" >/dev/null 2>&1; then
    kill "${SIM_PID}" >/dev/null 2>&1 || true
    wait "${SIM_PID}" 2>/dev/null || true
  fi
  if [[ "${SHOULD_SUMMARIZE}" == "1" ]]; then
    summarize_rollout_log "${ROLLOUT_POSE_LOG}"
  fi
}
trap cleanup_run EXIT INT TERM

if [[ ! -f "${RUN_SIM_SCRIPT}" ]]; then
  echo "ERROR: Could not find run_sim script: ${RUN_SIM_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${RUN_POLICY_SCRIPT}" ]]; then
  echo "ERROR: Could not find run_policy script: ${RUN_POLICY_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${MODEL_OR_CKPT}" && "${MODEL_OR_CKPT}" != wandb://* ]]; then
  echo "Model/checkpoint not found: ${MODEL_OR_CKPT}" >&2
  exit 1
fi

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "Motion file not found: ${MOTION_FILE}" >&2
  exit 1
fi

resolve_comm_ports
echo "[play_policy_mujoco_robot_w_obj] comm ports: object_state=${OBJECT_STATE_PORT}, tracking_viz=${TRACKING_VIZ_PORT}"

MODEL_PATH=${MODEL_PATH:-""}
if [[ -z "${MODEL_PATH}" ]]; then
  if [[ "${MODEL_OR_CKPT}" == *.onnx ]]; then
    MODEL_PATH="${MODEL_OR_CKPT}"
  elif [[ "${MODEL_OR_CKPT}" == *.pt ]]; then
    CANDIDATE_ONNX="${MODEL_OR_CKPT%.pt}.onnx"
    if [[ -f "${CANDIDATE_ONNX}" ]]; then
      MODEL_PATH="${CANDIDATE_ONNX}"
    fi
  fi
fi

if [[ -n "${MODEL_PATH}" ]]; then
  report_model_motion_alignment "${MODEL_PATH}" "${MOTION_FILE}" "${REPO_ROOT}" || true
  apply_training_sim_overrides "${MODEL_PATH}"
fi

RUN_SIM_METADATA_ARGS=""
if [[ -n "${SIM_SUBSTEPS}" ]]; then
  RUN_SIM_METADATA_ARGS="${RUN_SIM_METADATA_ARGS} --simulator.config.sim.substeps ${SIM_SUBSTEPS}"
fi
if [[ -n "${MUJOCO_BACKEND}" ]]; then
  RUN_SIM_METADATA_ARGS="${RUN_SIM_METADATA_ARGS} --simulator.config.mujoco-backend ${MUJOCO_BACKEND}"
fi
if [[ -n "${SIM_PHYSX_SOLVER_TYPE}" ]]; then
  RUN_SIM_METADATA_ARGS="${RUN_SIM_METADATA_ARGS} --simulator.config.sim.physx.solver-type ${SIM_PHYSX_SOLVER_TYPE}"
fi
if [[ -n "${SIM_PHYSX_NUM_POSITION_ITER}" ]]; then
  RUN_SIM_METADATA_ARGS="${RUN_SIM_METADATA_ARGS} --simulator.config.sim.physx.num-position-iterations ${SIM_PHYSX_NUM_POSITION_ITER}"
fi
if [[ -n "${SIM_PHYSX_NUM_VELOCITY_ITER}" ]]; then
  RUN_SIM_METADATA_ARGS="${RUN_SIM_METADATA_ARGS} --simulator.config.sim.physx.num-velocity-iterations ${SIM_PHYSX_NUM_VELOCITY_ITER}"
fi
if [[ -n "${TERRAIN_STATIC_FRICTION}" ]]; then
  RUN_SIM_METADATA_ARGS="${RUN_SIM_METADATA_ARGS} --terrain.terrain-term.static-friction ${TERRAIN_STATIC_FRICTION}"
fi
if [[ -n "${TERRAIN_DYNAMIC_FRICTION}" ]]; then
  RUN_SIM_METADATA_ARGS="${RUN_SIM_METADATA_ARGS} --terrain.terrain-term.dynamic-friction ${TERRAIN_DYNAMIC_FRICTION}"
fi
if [[ -n "${MUJOCO_OBJECT_MASS_OVERRIDE}" ]]; then
  RUN_SIM_METADATA_ARGS="${RUN_SIM_METADATA_ARGS} --mujoco-object-mass-override ${MUJOCO_OBJECT_MASS_OVERRIDE}"
fi

RUN_SIM_ARGS_DEFAULT="--training.headless ${HEADLESS} --simulator.config.debug_viz True --simulator.config.sim.fps ${SIM_FPS} --simulator.config.sim.control_decimation ${SIM_CONTROL_DECIMATION} --simulator.config.virtual-gantry.enabled False --simulator.config.bridge.hold-until-first-command True --robot.object.enabled True --robot.object.object_urdf_path ${OBJECT_URDF} --initial-state-motion-file ${MOTION_FILE} --initial-state-motion-frame 0 --initial-state-include-object True --object-state-pub-enabled True --object-state-pub-port ${OBJECT_STATE_PORT} --tracking-viz-sub-enabled ${TRACKING_VIZ_ENABLED} --tracking-viz-sub-port ${TRACKING_VIZ_PORT} --tracking-viz-future-steps ${TRACKING_VIZ_FUTURE_STEPS} --tracking-viz-marker-scale ${TRACKING_VIZ_MARKER_SCALE} --mujoco-object-mass-scale ${MUJOCO_OBJECT_MASS_SCALE}${RUN_SIM_METADATA_ARGS} --rollout-pose-log-enabled True --rollout-pose-log-csv ${ROLLOUT_POSE_LOG} --rollout-pose-log-every-n-steps ${ROLLOUT_LOG_EVERY} --rollout-success-min-step ${ROLLOUT_SUCCESS_MIN_STEP}"
RUN_SIM_ARGS=${RUN_SIM_ARGS:-"${RUN_SIM_ARGS_DEFAULT}"}
TRACKING_VIZ_POLICY_FLAG="--task.no-tracking-viz-pub-enabled"
case "${TRACKING_VIZ_ENABLED,,}" in
  1|true|yes|on)
    TRACKING_VIZ_POLICY_FLAG="--task.tracking-viz-pub-enabled"
    ;;
esac
RUN_POLICY_ARGS_DEFAULT="--task.no-use-joystick --task.motion-future-target-poses-motion-file ${MOTION_FILE} --task.object-state-sub-enabled --task.object-state-sub-port ${OBJECT_STATE_PORT} ${TRACKING_VIZ_POLICY_FLAG} --task.tracking-viz-pub-port ${TRACKING_VIZ_PORT} --task.tracking-viz-future-steps ${TRACKING_VIZ_FUTURE_STEPS}"
case "${AUTO_START,,}" in
  1|true|yes|on)
    RUN_POLICY_ARGS_DEFAULT="${RUN_POLICY_ARGS_DEFAULT} --task.auto-start-policy --task.auto-start-motion-clip --task.auto-start-stiff-hold-sec ${AUTO_START_STIFF_HOLD_SEC} --task.auto-start-stiff-max-wait-sec ${AUTO_START_STIFF_MAX_WAIT_SEC}"
    ;;
esac
RUN_POLICY_ARGS=${RUN_POLICY_ARGS:-"${RUN_POLICY_ARGS_DEFAULT}"}

if [[ -n "${MODEL_PATH}" && "${MODEL_PATH}" == *.onnx ]]; then
  read -r -a RUN_SIM_EXTRA <<< "${RUN_SIM_ARGS}"
  read -r -a RUN_POLICY_EXTRA <<< "${RUN_POLICY_ARGS}"
  USE_SIM_TIME_FLAG="--task.no-use-sim-time"
  case "${USE_SIM_TIME,,}" in
    1|true|yes|on)
      USE_SIM_TIME_FLAG="--task.use-sim-time"
      ;;
  esac

  sim_cmd=(
    python3 "${RUN_SIM_SCRIPT}"
    simulator:mujoco
    robot:g1-29dof-w-object
    terrain:terrain_locomotion_plane
    --simulator.config.bridge.interface "${INTERFACE}"
  )
  sim_cmd+=("${RUN_SIM_EXTRA[@]}")

  policy_cmd=(
    python3 "${RUN_POLICY_SCRIPT}"
    "${INFERENCE_CONFIG}"
    --task.model-path "${MODEL_PATH}"
    --task.interface "${INTERFACE}"
    --task.rl-rate "${RL_RATE}"
    "${USE_SIM_TIME_FLAG}"
  )
  policy_cmd+=("${RUN_POLICY_EXTRA[@]}")

  echo "[play_policy_mujoco_robot_w_obj] ONNX direct mode"
  echo "[play_policy_mujoco_robot_w_obj] model=${MODEL_PATH}"
  echo "[play_policy_mujoco_robot_w_obj] rollout_pose_log=${ROLLOUT_POSE_LOG}"
  echo "[play_policy_mujoco_robot_w_obj] run_sim:   ${sim_cmd[*]}"
  echo "[play_policy_mujoco_robot_w_obj] run_policy:${policy_cmd[*]}"

  case "${AUTO_LAUNCH,,}" in
    1|true|yes|on)
      SHOULD_SUMMARIZE="1"
      "${sim_cmd[@]}" &
      sim_pid=$!
      SIM_PID="${sim_pid}"
      wait_for_sim_ready "${SIM_READY_TIMEOUT_SEC}" || exit 1

      set +e
      "${policy_cmd[@]}"
      policy_rc=$?
      set -e
      exit "${policy_rc}"
      ;;
    *)
      exit 0
      ;;
  esac
else
  if [[ ! -f "${EVAL_AGENT_SCRIPT}" ]]; then
    echo "ERROR: Could not find eval script: ${EVAL_AGENT_SCRIPT}" >&2
    exit 1
  fi
  CKPT="${MODEL_OR_CKPT}"
  if [[ ! -f "${CKPT}" && "${CKPT}" != wandb://* ]]; then
    echo "Checkpoint not found: ${CKPT}" >&2
    exit 1
  fi
  echo "[play_policy_mujoco_robot_w_obj] ONNX missing; fallback to eval_agent checkpoint flow"
  cmd=(
    python3 "${EVAL_AGENT_SCRIPT}"
    --checkpoint "${CKPT}"
    --sim2sim.enabled
    --sim2sim.simulator mujoco
    --sim2sim.interface "${INTERFACE}"
    --sim2sim.inference-config "${INFERENCE_CONFIG}"
    --sim2sim.run-sim-robot g1-29dof-w-object
    --sim2sim.run-sim-args "${RUN_SIM_ARGS}"
    --sim2sim.run-policy-args "${RUN_POLICY_ARGS}"
  )
  case "${AUTO_LAUNCH,,}" in
    1|true|yes|on) cmd+=(--sim2sim.auto-launch) ;;
    *) cmd+=(--sim2sim.no-auto-launch) ;;
  esac
  if [[ -n "${MODEL_PATH}" ]]; then
    cmd+=(--sim2sim.model-path "${MODEL_PATH}")
  fi
  "${cmd[@]}"
fi
