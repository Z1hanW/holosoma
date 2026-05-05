#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="${ROOT_DIR}/src/holosoma_inference${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_INPUT="${MODEL_INPUT:-${MODEL_PATH:-${MODEL_REF:-https://wandb.ai/zihanw22/boxer/runs/w5qostjn}}}"
if [[ "${1:-}" == wandb://* || "${1:-}" == https://* || "${1:-}" == *.onnx || "${1:-}" == *.pt ]]; then
  MODEL_INPUT="$1"
  shift
fi

INFERENCE_CONFIG="${INFERENCE_CONFIG:-g1-box-w5qostjn}"
INTERFACE="${INTERFACE:-lo}"
POLICY_PYTHON_BIN="${POLICY_PYTHON_BIN:-/home/user/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
if [[ ! -x "$POLICY_PYTHON_BIN" ]]; then
  POLICY_PYTHON_BIN="${PYTHON_BIN:-python3}"
fi
MUJOCO_PYTHON_BIN="${MUJOCO_PYTHON_BIN:-/home/user/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"

MJ_TRACK_MODE="${MJ_TRACK_MODE:-both}"
RUN_SECONDS="${RUN_SECONDS:-0}"
SIM_READY_TIMEOUT="${SIM_READY_TIMEOUT:-45}"
SIM_READY_PATTERN="${SIM_READY_PATTERN:-Starting direct simulation loop}"

RUN_NAME="${RUN_NAME:-g1_box_d435i}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/logs/direct_depth_runs/${RUN_NAME}}"
SIM_LOG="${SIM_LOG:-${RUN_DIR}/mujoco.log}"
POLICY_LOG="${POLICY_LOG:-${RUN_DIR}/policy.log}"
mkdir -p "$RUN_DIR"

ENV_CMD=(
  bash "$ROOT_DIR/mj_env.sh"
)
POLICY_CMD=(
  "$POLICY_PYTHON_BIN" -u src/holosoma_inference/holosoma_inference/run_policy.py "inference:${INFERENCE_CONFIG}"
  --task.interface "$INTERFACE"
  --task.model-path "$MODEL_INPUT"
)

echo "[INFO] launching direct MuJoCo DDS depth rollout"
echo "[INFO] model=${MODEL_INPUT}"
echo "[INFO] inference_config=${INFERENCE_CONFIG}"
echo "[INFO] interface=${INTERFACE}"
echo "[INFO] logs=${RUN_DIR}"
echo "[INFO] depth source is selected by mj_env.sh IMAGE_SERVER_CONFIG/IMAGE_DEPTH_SOURCE"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf '[INFO] env command:'
  printf ' %q' PYTHON_BIN="$MUJOCO_PYTHON_BIN" "${ENV_CMD[@]}" "$@"
  printf '\n'
  printf '[INFO] policy command:'
  printf ' %q' "${POLICY_CMD[@]}"
  printf '\n'
  exit 0
fi

SIM_PID=""
cleanup() {
  if [[ -n "${SIM_PID:-}" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
    kill "$SIM_PID" 2>/dev/null || true
    wait "$SIM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

wait_for_sim_ready() {
  local deadline=$((SECONDS + SIM_READY_TIMEOUT))
  while (( SECONDS < deadline )); do
    if [[ -n "${SIM_PID:-}" ]] && ! kill -0 "$SIM_PID" 2>/dev/null; then
      echo "[ERROR] MuJoCo exited during startup. See $SIM_LOG" >&2
      tail -n 60 "$SIM_LOG" >&2 || true
      return 1
    fi
    if [[ -f "$SIM_LOG" ]] && grep -qF "$SIM_READY_PATTERN" "$SIM_LOG"; then
      return 0
    fi
    sleep 0.5
  done
  echo "[ERROR] Timed out waiting for MuJoCo readiness pattern '$SIM_READY_PATTERN'. See $SIM_LOG" >&2
  tail -n 60 "$SIM_LOG" >&2 || true
  return 1
}

case "$(printf '%s' "$MJ_TRACK_MODE" | tr '[:upper:]' '[:lower:]')" in
  both|"")
    : >"$SIM_LOG"
    PYTHON_BIN="$MUJOCO_PYTHON_BIN" "${ENV_CMD[@]}" "$@" >"$SIM_LOG" 2>&1 &
    SIM_PID=$!
    wait_for_sim_ready
    ;;
  policy)
    echo "[INFO] MJ_TRACK_MODE=policy; not launching MuJoCo."
    ;;
  *)
    echo "[ERROR] Unsupported MJ_TRACK_MODE=${MJ_TRACK_MODE}; expected both or policy." >&2
    exit 2
    ;;
esac

if [[ "$RUN_SECONDS" != "0" ]]; then
  POLICY_CMD=(timeout --kill-after=5s --signal=INT "${RUN_SECONDS}s" "${POLICY_CMD[@]}")
fi

if [[ "${POLICY_STDIO:-inherit}" == "log" ]]; then
  : >"$POLICY_LOG"
  "${POLICY_CMD[@]}" >"$POLICY_LOG" 2>&1
else
  "${POLICY_CMD[@]}"
fi
