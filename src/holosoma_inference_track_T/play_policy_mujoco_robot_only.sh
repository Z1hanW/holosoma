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
DEFAULT_MOTION="${REPO_ROOT}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz"

MODEL_OR_CKPT=${1:-${MODEL_OR_CKPT:-"${DEFAULT_ONNX}"}}
AUTO_LAUNCH=${AUTO_LAUNCH:-"True"}
INTERFACE=${INTERFACE:-"lo"}
INFERENCE_CONFIG=${INFERENCE_CONFIG:-"inference:g1-29dof-wbt"}
RL_RATE=${RL_RATE:-"50.0"}
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
TRACKING_VIZ_PORT=${TRACKING_VIZ_PORT:-"5560"}
TRACKING_VIZ_ENABLED=${TRACKING_VIZ_ENABLED:-"True"}
TRACKING_VIZ_FUTURE_STEPS=${TRACKING_VIZ_FUTURE_STEPS:-"10"}
TRACKING_VIZ_MARKER_SCALE=${TRACKING_VIZ_MARKER_SCALE:-"2.5"}
ROLLOUT_LOG_EVERY=${ROLLOUT_LOG_EVERY:-"20"}
ROLLOUT_SUCCESS_MIN_STEP=${ROLLOUT_SUCCESS_MIN_STEP:-"2000"}
ROLLOUT_REQUIRE_SUCCESS_ROWS=${ROLLOUT_REQUIRE_SUCCESS_ROWS:-"1"}
SIM_READY_TIMEOUT_SEC=${SIM_READY_TIMEOUT_SEC:-"40"}
SIM_READY_POLL_SEC=${SIM_READY_POLL_SEC:-"0.2"}
MOTION_FILE=${MOTION_FILE:-"${DEFAULT_MOTION}"}
TS=$(date -u +%Y%m%d_%H%M%S)
ROLLOUT_POSE_LOG=${ROLLOUT_POSE_LOG:-"logs/replay_motion_mujoco/policy_robot_only_pose_${TS}.csv"}

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
    echo "[play_policy_mujoco_robot_only] training sim override disabled (USE_TRAINING_SIM_CONFIG=${USE_TRAINING_SIM_CONFIG})"
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
    echo "[play_policy_mujoco_robot_only] applied training sim config: fps=${SIM_FPS}, decimation=${SIM_CONTROL_DECIMATION}, substeps=${SIM_SUBSTEPS:-n/a}, backend=${MUJOCO_BACKEND:-n/a}, physx_iters=${SIM_PHYSX_NUM_POSITION_ITER:-n/a}/${SIM_PHYSX_NUM_VELOCITY_ITER:-n/a}, terrain_friction=${TERRAIN_STATIC_FRICTION:-n/a}/${TERRAIN_DYNAMIC_FRICTION:-n/a}"
  else
    echo "[play_policy_mujoco_robot_only] training sim config not found in ONNX metadata; keeping launcher defaults"
  fi
}

summarize_rollout_log() {
  local csv_path="$1"
  if [[ ! -f "${csv_path}" ]]; then
    echo "[play_policy_mujoco_robot_only] rollout log missing: ${csv_path}"
    return
  fi
  python3 - "${csv_path}" "${ROLLOUT_REQUIRE_SUCCESS_ROWS}" <<'PY'
import csv
import math
import sys

csv_path = sys.argv[1]
required_success = int(sys.argv[2])
prefix = "[play_policy_mujoco_robot_only]"

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

if success_rows >= required_success:
    print(f"{prefix} rollout verdict: PASS (success_rows >= {required_success})")
else:
    reasons = []
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
    print(f"{prefix} rollout verdict: FAIL (success_rows < {required_success}; reason={'+'.join(reasons)})")
PY
}

wait_for_sim_ready() {
  local timeout_sec="$1"
  local start_ts now elapsed
  start_ts=$(date +%s)
  while true; do
    if [[ -n "${SIM_PID}" ]] && ! kill -0 "${SIM_PID}" >/dev/null 2>&1; then
      echo "[play_policy_mujoco_robot_only] simulator exited before ready"
      return 1
    fi
    if [[ -s "${ROLLOUT_POSE_LOG}" ]]; then
      local header
      header="$(head -n 1 "${ROLLOUT_POSE_LOG}" || true)"
      if [[ "${header}" == step,sim_time,* ]]; then
        echo "[play_policy_mujoco_robot_only] simulator ready (rollout log initialized)"
        return 0
      fi
    fi
    now=$(date +%s)
    elapsed=$((now - start_ts))
    if (( elapsed >= timeout_sec )); then
      echo "[play_policy_mujoco_robot_only] readiness wait timed out after ${timeout_sec}s; continuing"
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

RUN_SIM_ARGS_DEFAULT="--training.headless ${HEADLESS} --simulator.config.debug_viz True --simulator.config.sim.fps ${SIM_FPS} --simulator.config.sim.control_decimation ${SIM_CONTROL_DECIMATION} --simulator.config.virtual-gantry.enabled False --simulator.config.bridge.hold-until-first-command True --initial-state-motion-file ${MOTION_FILE} --initial-state-motion-frame 0 --tracking-viz-sub-enabled ${TRACKING_VIZ_ENABLED} --tracking-viz-sub-port ${TRACKING_VIZ_PORT} --tracking-viz-future-steps ${TRACKING_VIZ_FUTURE_STEPS} --tracking-viz-marker-scale ${TRACKING_VIZ_MARKER_SCALE}${RUN_SIM_METADATA_ARGS} --rollout-pose-log-enabled True --rollout-pose-log-csv ${ROLLOUT_POSE_LOG} --rollout-pose-log-every-n-steps ${ROLLOUT_LOG_EVERY} --rollout-success-min-step ${ROLLOUT_SUCCESS_MIN_STEP}"
RUN_SIM_ARGS=${RUN_SIM_ARGS:-"${RUN_SIM_ARGS_DEFAULT}"}
TRACKING_VIZ_POLICY_FLAG="--task.no-tracking-viz-pub-enabled"
case "${TRACKING_VIZ_ENABLED,,}" in
  1|true|yes|on)
    TRACKING_VIZ_POLICY_FLAG="--task.tracking-viz-pub-enabled"
    ;;
esac
RUN_POLICY_ARGS_DEFAULT="--task.no-use-joystick --task.motion-future-target-poses-motion-file ${MOTION_FILE} ${TRACKING_VIZ_POLICY_FLAG} --task.tracking-viz-pub-port ${TRACKING_VIZ_PORT} --task.tracking-viz-future-steps ${TRACKING_VIZ_FUTURE_STEPS}"
case "${AUTO_START,,}" in
  1|true|yes|on)
    RUN_POLICY_ARGS_DEFAULT="${RUN_POLICY_ARGS_DEFAULT} --task.auto-start-policy --task.auto-start-motion-clip --task.auto-start-stiff-hold-sec ${AUTO_START_STIFF_HOLD_SEC} --task.auto-start-stiff-max-wait-sec ${AUTO_START_STIFF_MAX_WAIT_SEC}"
    ;;
esac
RUN_POLICY_ARGS=${RUN_POLICY_ARGS:-"${RUN_POLICY_ARGS_DEFAULT}"}

if [[ -n "${MODEL_PATH}" && "${MODEL_PATH}" == *.onnx ]]; then
  read -r -a RUN_SIM_EXTRA <<< "${RUN_SIM_ARGS}"
  read -r -a RUN_POLICY_EXTRA <<< "${RUN_POLICY_ARGS}"

  sim_cmd=(
    python3 "${RUN_SIM_SCRIPT}"
    simulator:mujoco
    robot:g1-29dof
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
    --task.use-sim-time
  )
  policy_cmd+=("${RUN_POLICY_EXTRA[@]}")

  echo "[play_policy_mujoco_robot_only] ONNX direct mode"
  echo "[play_policy_mujoco_robot_only] model=${MODEL_PATH}"
  echo "[play_policy_mujoco_robot_only] rollout_pose_log=${ROLLOUT_POSE_LOG}"
  echo "[play_policy_mujoco_robot_only] run_sim:   ${sim_cmd[*]}"
  echo "[play_policy_mujoco_robot_only] run_policy:${policy_cmd[*]}"

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
  echo "[play_policy_mujoco_robot_only] ONNX missing; fallback to eval_agent checkpoint flow"
  cmd=(
    python3 "${EVAL_AGENT_SCRIPT}"
    --checkpoint "${CKPT}"
    --sim2sim.enabled
    --sim2sim.simulator mujoco
    --sim2sim.interface "${INTERFACE}"
    --sim2sim.inference-config "${INFERENCE_CONFIG}"
    --sim2sim.run-sim-robot g1-29dof
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
