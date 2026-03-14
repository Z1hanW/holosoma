#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

MUJOCO_PY="${MUJOCO_PY:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python}"
INFER_PY="${INFER_PY:-/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx}"
VISER_URDF_PATH="${VISER_URDF_PATH:-$ROOT_DIR/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf}"
SIM_STATE_PORT="${SIM_STATE_PORT:-5557}"
SIM_CONTROL_PORT="${SIM_CONTROL_PORT:-5559}"
VISER_PORT="${VISER_PORT:-18080}"
RL_RATE="${RL_RATE:-50}"
SIM_STARTUP_WAIT="${SIM_STARTUP_WAIT:-3}"
HEADLESS="${HEADLESS:-False}"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/sim2sim_dancing_viser}"
mkdir -p "$LOG_DIR"
MUJOCO_LOG="$LOG_DIR/mujoco.log"

if [[ ! -x "$MUJOCO_PY" ]]; then
  echo "[ERROR] MuJoCo python not found: $MUJOCO_PY" >&2
  exit 1
fi

if [[ ! -x "$INFER_PY" ]]; then
  echo "[ERROR] Inference python not found: $INFER_PY" >&2
  exit 1
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[ERROR] Dancing ONNX not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$VISER_URDF_PATH" ]]; then
  echo "[ERROR] Viser URDF not found: $VISER_URDF_PATH" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${MUJOCO_PID:-}" ]] && kill -0 "$MUJOCO_PID" 2>/dev/null; then
    kill "$MUJOCO_PID" 2>/dev/null || true
    wait "$MUJOCO_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[INFO] Launching MuJoCo dance sim..."
"$MUJOCO_PY" "$ROOT_DIR/src/holosoma/holosoma/run_sim.py" \
  simulator:mujoco \
  robot:g1-29dof \
  --training.headless="$HEADLESS" \
  --simulator.config.virtual-gantry.enabled=False \
  --simulator.config.bridge.publish-sim-state=True \
  --simulator.config.bridge.sim-state-port="$SIM_STATE_PORT" \
  --simulator.config.bridge.listen-control=True \
  --simulator.config.bridge.control-port="$SIM_CONTROL_PORT" \
  >"$MUJOCO_LOG" 2>&1 &
MUJOCO_PID=$!

sleep "$SIM_STARTUP_WAIT"

if ! kill -0 "$MUJOCO_PID" 2>/dev/null; then
  echo "[ERROR] MuJoCo exited during startup. See $MUJOCO_LOG" >&2
  exit 1
fi

echo "[INFO] MuJoCo log:   $MUJOCO_LOG"
echo "[INFO] Viser URL:    http://localhost:$VISER_PORT"
echo "[INFO] Viser reset:  manual button in GUI; auto reset triggers at motion end."

"$INFER_PY" "$ROOT_DIR/src/holosoma_inference/holosoma_inference/run_policy.py" \
  inference:g1-29dof-wbt \
  --task.model-path "$MODEL_PATH" \
  --task.no-use-joystick \
  --task.use-sim-time \
  --task.use-sim-state \
  --task.sim-state-port "$SIM_STATE_PORT" \
  --task.sim-control-port "$SIM_CONTROL_PORT" \
  --task.rl-rate "$RL_RATE" \
  --task.interface lo \
  --viser.enabled \
  --viser.port "$VISER_PORT" \
  --viser.urdf-path "$VISER_URDF_PATH"
