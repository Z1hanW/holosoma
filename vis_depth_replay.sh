#!/usr/bin/env bash
set -euo pipefail

# Warp-sensors depth replay for Isaac Sim.
# Locked to perception:camera_depth_d435i on d435_joint.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP=${EXP:-"g1-29dof-wbt-videomimic-mlp"}
COMMAND_PRESET=${COMMAND_PRESET:-"g1-29dof-wbt"}
PERCEPTION_PRESET="camera_depth_d435i"

MOTION_FILE=${MOTION_FILE:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz"}
MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-""}
TERRAIN_OBJ=${TERRAIN_OBJ:-"${SCRIPT_DIR}/src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj"}
TERRAIN_NUM_ROWS=${TERRAIN_NUM_ROWS:-1}
TERRAIN_NUM_COLS=${TERRAIN_NUM_COLS:-1}

DEPTH_IMPL=${DEPTH_IMPL:-warp_sensors}
case "${DEPTH_IMPL}" in
  warp_like|warp_sensors)
    DEPTH_IMPL="warp_sensors"
    ;;
  *)
    echo "[ERROR] vis_depth_replay.sh is locked to DEPTH_IMPL=warp_sensors (or warp_like alias), got: ${DEPTH_IMPL}" >&2
    exit 1
    ;;
esac

IMAGE_WIDTH=${IMAGE_WIDTH:-106}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-60}
CAMERA_VFOV_DEG=${CAMERA_VFOV_DEG:-58.6}
CAMERA_HFOV_DEG=${CAMERA_HFOV_DEG:-89.5}
CAMERA_NEAR=${CAMERA_NEAR:-0.3}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-${CAMERA_FAR}}
CAMERA_PITCH_DEG=${CAMERA_PITCH_DEG:--20.0}
CAMERA_INCLUDE_ROBOT_MESH=${CAMERA_INCLUDE_ROBOT_MESH:-True}
SCANDOTS_STRIDE=${SCANDOTS_STRIDE:-1}
CAMERA_BODY_NAME=${CAMERA_BODY_NAME:-d435_joint}
CAMERA_FRAME_QUAT=${CAMERA_FRAME_QUAT:-"[0.5,-0.5,-0.5,0.5]"}

HEADLESS_RAW=${HEADLESS:-False}
headless_norm="$(printf '%s' "${HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${headless_norm}" in
  1|true|yes|on)
    TRAINING_HEADLESS="True"
    ISAAC_HEADLESS_ENV="1"
    ;;
  0|false|no|off)
    TRAINING_HEADLESS="False"
    ISAAC_HEADLESS_ENV="0"
    ;;
  *)
    echo "[ERROR] Invalid HEADLESS=${HEADLESS_RAW}. Use one of: True/False/1/0" >&2
    exit 1
    ;;
esac

if [[ "${TRAINING_HEADLESS}" == "True" && "${VISER_START_PAUSED:-1}" == "1" ]]; then
  echo "[WARN] HEADLESS=True with VISER_START_PAUSED=1 will pause replay until manually unpaused."
fi

if ! [[ "${TERRAIN_NUM_ROWS}" =~ ^[0-9]+$ ]] || (( TERRAIN_NUM_ROWS < 1 )); then
  echo "[ERROR] TERRAIN_NUM_ROWS must be a positive integer, got: ${TERRAIN_NUM_ROWS}" >&2
  exit 1
fi
if ! [[ "${TERRAIN_NUM_COLS}" =~ ^[0-9]+$ ]] || (( TERRAIN_NUM_COLS < 1 )); then
  echo "[ERROR] TERRAIN_NUM_COLS must be a positive integer, got: ${TERRAIN_NUM_COLS}" >&2
  exit 1
fi
TERRAIN_SLOT_COUNT=$((TERRAIN_NUM_ROWS * TERRAIN_NUM_COLS))
if [[ -z "${NUM_ENVS+x}" ]]; then
  NUM_ENVS="${TERRAIN_SLOT_COUNT}"
fi
if ! [[ "${NUM_ENVS}" =~ ^[0-9]+$ ]] || (( NUM_ENVS < 1 )); then
  echo "[ERROR] NUM_ENVS must be a positive integer, got: ${NUM_ENVS}" >&2
  exit 1
fi
SEED=${SEED:-42}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}

VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-True}
ISAAC_SHOW_SCANDOTS=${ISAAC_SHOW_SCANDOTS:-True}
VISER_STRICT_CAMERA_RAYS=${VISER_STRICT_CAMERA_RAYS:-1}
VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-0}
VISER_START_PAUSED=${VISER_START_PAUSED:-1}
VISER_PERCEPTION_IMAGE_MODE=${VISER_PERCEPTION_IMAGE_MODE:-depth}
VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE=${VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE:-0}
VISER_DISABLE_PERCEPTION_FRUSTUM=${VISER_DISABLE_PERCEPTION_FRUSTUM:-1}
VISER_DISABLE_CONTACT_FORCE_VIZ=${VISER_DISABLE_CONTACT_FORCE_VIZ:-1}
VISER_SCANDOTS_INCLUDE_MISSES=${VISER_SCANDOTS_INCLUDE_MISSES:-0}
VISER_SCANDOTS_USE_DEPTH_MASK=${VISER_SCANDOTS_USE_DEPTH_MASK:-1}
VISER_SCANDOTS_POINTS_FOLLOW_LINES=${VISER_SCANDOTS_POINTS_FOLLOW_LINES:-1}
VISER_SCANDOTS_SOURCE=${VISER_SCANDOTS_SOURCE:-isaac}
ISAAC_SCANDOTS_INCLUDE_MISSES=${ISAAC_SCANDOTS_INCLUDE_MISSES:-0}
ISAAC_SCANDOTS_USE_DEPTH_MASK=${ISAAC_SCANDOTS_USE_DEPTH_MASK:-1}
VISER_FAITHFUL_MODE=${VISER_FAITHFUL_MODE:-1}
VISER_PERCEPTION_IMAGE_FORMAT=${VISER_PERCEPTION_IMAGE_FORMAT:-png}
VISER_PERCEPTION_FLIP_VERTICAL=${VISER_PERCEPTION_FLIP_VERTICAL:-0}
VISER_DEPTH_COLORMAP=${VISER_DEPTH_COLORMAP:-fixed}

HOLOSOMA_REPLAY_KEEP_OPEN=${HOLOSOMA_REPLAY_KEEP_OPEN:-1}
HOLOSOMA_REPLAY_WANDB_ENABLE=${HOLOSOMA_REPLAY_WANDB_ENABLE:-1}
HOLOSOMA_REPLAY_WANDB_MODE=${HOLOSOMA_REPLAY_WANDB_MODE:-offline}
HOLOSOMA_REPLAY_WANDB_PROJECT=${HOLOSOMA_REPLAY_WANDB_PROJECT:-holosoma-depth-replay}
HOLOSOMA_REPLAY_WANDB_RUN_NAME=${HOLOSOMA_REPLAY_WANDB_RUN_NAME:-}
HOLOSOMA_REPLAY_WANDB_ENTITY=${HOLOSOMA_REPLAY_WANDB_ENTITY:-}
HOLOSOMA_REPLAY_WANDB_GROUP=${HOLOSOMA_REPLAY_WANDB_GROUP:-vis-depth-replay}
HOLOSOMA_REPLAY_WANDB_TAGS=${HOLOSOMA_REPLAY_WANDB_TAGS:-depth,replay}
HOLOSOMA_REPLAY_WANDB_DEPTH_EVERY=${HOLOSOMA_REPLAY_WANDB_DEPTH_EVERY:-10}
HOLOSOMA_REPLAY_WANDB_DEPTH_VIDEO=${HOLOSOMA_REPLAY_WANDB_DEPTH_VIDEO:-1}
HOLOSOMA_REPLAY_WANDB_DEPTH_VIDEO_MAX_FRAMES=${HOLOSOMA_REPLAY_WANDB_DEPTH_VIDEO_MAX_FRAMES:-1200}
HOLOSOMA_REPLAY_WANDB_ENV_ID=${HOLOSOMA_REPLAY_WANDB_ENV_ID:-0}

MOTION_USE_ADAPTIVE_TIMESTEP_SAMPLER=${MOTION_USE_ADAPTIVE_TIMESTEP_SAMPLER:-False}
MOTION_START_AT_ZERO_PROB=${MOTION_START_AT_ZERO_PROB:-1.0}
MOTION_ENABLE_DEFAULT_POSE_PREPEND=${MOTION_ENABLE_DEFAULT_POSE_PREPEND:-False}
MOTION_DEFAULT_POSE_PREPEND_DURATION_S=${MOTION_DEFAULT_POSE_PREPEND_DURATION_S:-0.0}
MOTION_ENABLE_DEFAULT_POSE_APPEND=${MOTION_ENABLE_DEFAULT_POSE_APPEND:-False}
MOTION_DEFAULT_POSE_APPEND_DURATION_S=${MOTION_DEFAULT_POSE_APPEND_DURATION_S:-0.0}
MOTION_INIT_NOISE_SCALE=${MOTION_INIT_NOISE_SCALE:-0.0}
MOTION_ALIGN_TO_INIT_YAW=${MOTION_ALIGN_TO_INIT_YAW:-False}
MOTION_PAIR_TERRAIN_WITH_MOTION=${MOTION_PAIR_TERRAIN_WITH_MOTION:-False}

DEBUG_MOTION_TERRAIN=${DEBUG_MOTION_TERRAIN:-1}
STRICT_OPTIONS=${STRICT_OPTIONS:-1}
DISABLE_RANDOMIZATION_AND_NOISE=${DISABLE_RANDOMIZATION_AND_NOISE:-1}

if [[ "$(uname -s)" == "Linux" ]] && ! command -v zenity >/dev/null 2>&1 && [[ -z "${BROWSER:-}" ]]; then
  export BROWSER=true
  echo "[WARN] zenity not found and BROWSER is unset; suppressing auto browser launch."
fi

if [[ ! -e "${MOTION_FILE}" ]]; then
  echo "[ERROR] MOTION_FILE not found: ${MOTION_FILE}" >&2
  exit 1
fi

if [[ -n "${TERRAIN_OBJ}" ]] && [[ ! -e "${TERRAIN_OBJ}" ]]; then
  echo "[ERROR] TERRAIN_OBJ not found: ${TERRAIN_OBJ}" >&2
  exit 1
fi
if [[ -n "${TERRAIN_OBJ}" ]]; then
  if (( TERRAIN_SLOT_COUNT < NUM_ENVS )); then
    echo "[WARN] terrain slots (${TERRAIN_NUM_ROWS}x${TERRAIN_NUM_COLS}=${TERRAIN_SLOT_COUNT}) are fewer than NUM_ENVS=${NUM_ENVS}; some envs will overlap." >&2
  elif (( TERRAIN_SLOT_COUNT > NUM_ENVS )); then
    echo "[WARN] terrain slots (${TERRAIN_NUM_ROWS}x${TERRAIN_NUM_COLS}=${TERRAIN_SLOT_COUNT}) exceed NUM_ENVS=${NUM_ENVS}; some terrain tiles will be unused." >&2
  fi
fi

if [[ "${STRICT_OPTIONS}" == "1" ]]; then
  if [[ "${EXP}" != "g1-29dof-wbt-videomimic-mlp" ]]; then
    echo "[ERROR] STRICT_OPTIONS=1 requires EXP=g1-29dof-wbt-videomimic-mlp, got: ${EXP}" >&2
    exit 1
  fi
  if [[ "${COMMAND_PRESET}" != "g1-29dof-wbt" ]]; then
    echo "[ERROR] STRICT_OPTIONS=1 requires COMMAND_PRESET=g1-29dof-wbt, got: ${COMMAND_PRESET}" >&2
    exit 1
  fi
  if [[ "${MOTION_FILE}" != *"/converted_res/"* ]]; then
    echo "[ERROR] STRICT_OPTIONS=1 requires MOTION_FILE under converted_res, got: ${MOTION_FILE}" >&2
    exit 1
  fi
  if [[ "${CAMERA_BODY_NAME}" != "d435_joint" ]]; then
    echo "[ERROR] STRICT_OPTIONS=1 requires CAMERA_BODY_NAME=d435_joint, got: ${CAMERA_BODY_NAME}" >&2
    exit 1
  fi
  if [[ "${CAMERA_FRAME_QUAT}" != "[0.5,-0.5,-0.5,0.5]" ]]; then
    echo "[ERROR] STRICT_OPTIONS=1 requires CAMERA_FRAME_QUAT=[0.5,-0.5,-0.5,0.5], got: ${CAMERA_FRAME_QUAT}" >&2
    exit 1
  fi
  if [[ "${IMAGE_WIDTH}" != "106" || "${IMAGE_HEIGHT}" != "60" ]]; then
    echo "[ERROR] STRICT_OPTIONS=1 requires IMAGE_WIDTH/HEIGHT=106/60, got ${IMAGE_WIDTH}/${IMAGE_HEIGHT}" >&2
    exit 1
  fi
  if [[ "${CAMERA_INCLUDE_ROBOT_MESH}" != "True" ]]; then
    echo "[ERROR] STRICT_OPTIONS=1 requires CAMERA_INCLUDE_ROBOT_MESH=True, got: ${CAMERA_INCLUDE_ROBOT_MESH}" >&2
    exit 1
  fi
fi

echo "[INFO] DEPTH_IMPL=${DEPTH_IMPL}"
echo "[INFO] PERCEPTION_PRESET=${PERCEPTION_PRESET} (locked)"
echo "[INFO] EXP=exp:${EXP}"
echo "[INFO] COMMAND_PRESET=command:${COMMAND_PRESET}"
echo "[INFO] MOTION_FILE=${MOTION_FILE}"
echo "[INFO] TERRAIN_OBJ=${TERRAIN_OBJ}"
echo "[INFO] TERRAIN_GRID rows=${TERRAIN_NUM_ROWS} cols=${TERRAIN_NUM_COLS}"
if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  echo "[INFO] MOTION_CLIP_NAME=${MOTION_CLIP_NAME}"
fi
echo "[INFO] CAMERA_BODY_NAME=${CAMERA_BODY_NAME}"
echo "[INFO] CAMERA_FRAME_QUAT=${CAMERA_FRAME_QUAT}"
echo "[INFO] CAMERA_CFG width=${IMAGE_WIDTH} height=${IMAGE_HEIGHT} vfov=${CAMERA_VFOV_DEG} hfov=${CAMERA_HFOV_DEG} near=${CAMERA_NEAR} far=${CAMERA_FAR} pitch=${CAMERA_PITCH_DEG} include_robot_mesh=${CAMERA_INCLUDE_ROBOT_MESH}"
echo "[INFO] CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE}"
echo "[INFO] VISER=http://localhost:${VISER_PORT}"
echo "[INFO] VISER_ENV_ID=${VISER_ENV_ID}"
echo "[INFO] VISER_START_PAUSED=${VISER_START_PAUSED}"
echo "[INFO] VISER_PERCEPTION_IMAGE_MODE=${VISER_PERCEPTION_IMAGE_MODE}"
echo "[INFO] VISER_STRICT_CAMERA_RAYS=${VISER_STRICT_CAMERA_RAYS}"
echo "[INFO] VISER_SCANDOTS_INCLUDE_MISSES=${VISER_SCANDOTS_INCLUDE_MISSES}"
echo "[INFO] VISER_SCANDOTS_USE_DEPTH_MASK=${VISER_SCANDOTS_USE_DEPTH_MASK}"
echo "[INFO] VISER_SCANDOTS_POINTS_FOLLOW_LINES=${VISER_SCANDOTS_POINTS_FOLLOW_LINES}"
echo "[INFO] VISER_SCANDOTS_SOURCE=${VISER_SCANDOTS_SOURCE}"
echo "[INFO] ISAAC_SCANDOTS_INCLUDE_MISSES=${ISAAC_SCANDOTS_INCLUDE_MISSES}"
echo "[INFO] ISAAC_SCANDOTS_USE_DEPTH_MASK=${ISAAC_SCANDOTS_USE_DEPTH_MASK}"
echo "[INFO] TRAINING_HEADLESS=${TRAINING_HEADLESS}"
echo "[INFO] HOLOSOMA_REPLAY_KEEP_OPEN=${HOLOSOMA_REPLAY_KEEP_OPEN}"
echo "[INFO] HOLOSOMA_REPLAY_WANDB_ENABLE=${HOLOSOMA_REPLAY_WANDB_ENABLE}"
echo "[INFO] HOLOSOMA_REPLAY_WANDB_MODE=${HOLOSOMA_REPLAY_WANDB_MODE}"
echo "[INFO] HOLOSOMA_REPLAY_WANDB_PROJECT=${HOLOSOMA_REPLAY_WANDB_PROJECT}"
echo "[INFO] MOTION_DEBUG use_adaptive=${MOTION_USE_ADAPTIVE_TIMESTEP_SAMPLER} start_at_zero_prob=${MOTION_START_AT_ZERO_PROB} prepend=${MOTION_ENABLE_DEFAULT_POSE_PREPEND}/${MOTION_DEFAULT_POSE_PREPEND_DURATION_S}s append=${MOTION_ENABLE_DEFAULT_POSE_APPEND}/${MOTION_DEFAULT_POSE_APPEND_DURATION_S}s init_noise=${MOTION_INIT_NOISE_SCALE} align_yaw=${MOTION_ALIGN_TO_INIT_YAW} pair_terrain=${MOTION_PAIR_TERRAIN_WITH_MOTION}"

if [[ "${DEBUG_MOTION_TERRAIN}" == "1" ]]; then
  python3 - <<'PY' "${MOTION_FILE}" "${TERRAIN_OBJ}"
import sys
from pathlib import Path

import numpy as np

motion_path = Path(sys.argv[1])
terrain_path = Path(sys.argv[2])

print(f"[DEBUG] motion={motion_path}")
with np.load(motion_path, allow_pickle=True) as data:
    body_pos = np.asarray(data["body_pos_w"], dtype=np.float32)
    body_names = [str(x) for x in np.asarray(data.get("body_names", []), dtype=object).tolist()]
    root_idx = 0
    root_name = "body[0]"
    for candidate in ("pelvis", "torso_link", "base_link"):
        if candidate in body_names:
            root_idx = int(body_names.index(candidate))
            root_name = candidate
            break
    root = body_pos[:, root_idx, :]
    print(f"[DEBUG] root_body={root_name} idx={root_idx}")
    print(f"[DEBUG] root start xyz={root[0].tolist()} end xyz={root[-1].tolist()}")
    print(f"[DEBUG] root min xyz={root.min(axis=0).tolist()} max xyz={root.max(axis=0).tolist()}")

print(f"[DEBUG] terrain={terrain_path}")
try:
    import trimesh

    mesh = trimesh.load(str(terrain_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        geoms = mesh.dump(concatenate=False)
        mesh = geoms[0] if geoms else None
    if mesh is None:
        print("[DEBUG] terrain stats unavailable: empty scene")
    else:
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        print(f"[DEBUG] terrain bounds min={bounds[0].tolist()} max={bounds[1].tolist()}")
except Exception as exc:
    print(f"[DEBUG] terrain stats unavailable: {exc}")
PY
fi

export VISER_ENABLE_CLIP_GUI
export VISER_START_PAUSED
export VISER_PERCEPTION_IMAGE_MODE
export VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE
export VISER_DISABLE_PERCEPTION_FRUSTUM
export VISER_DISABLE_CONTACT_FORCE_VIZ
export VISER_SCANDOTS_INCLUDE_MISSES
export VISER_SCANDOTS_USE_DEPTH_MASK
export VISER_SCANDOTS_POINTS_FOLLOW_LINES
export VISER_SCANDOTS_SOURCE
export ISAAC_SCANDOTS_INCLUDE_MISSES
export ISAAC_SCANDOTS_USE_DEPTH_MASK
export VISER_STRICT_CAMERA_RAYS
export VISER_FAITHFUL_MODE
export VISER_PERCEPTION_IMAGE_FORMAT
export VISER_PERCEPTION_FLIP_VERTICAL
export VISER_DEPTH_COLORMAP
export HOLOSOMA_REPLAY_KEEP_OPEN
export HOLOSOMA_REPLAY_WANDB_ENABLE
export HOLOSOMA_REPLAY_WANDB_MODE
export HOLOSOMA_REPLAY_WANDB_PROJECT
export HOLOSOMA_REPLAY_WANDB_RUN_NAME
export HOLOSOMA_REPLAY_WANDB_ENTITY
export HOLOSOMA_REPLAY_WANDB_GROUP
export HOLOSOMA_REPLAY_WANDB_TAGS
export HOLOSOMA_REPLAY_WANDB_DEPTH_EVERY
export HOLOSOMA_REPLAY_WANDB_DEPTH_VIDEO
export HOLOSOMA_REPLAY_WANDB_DEPTH_VIDEO_MAX_FRAMES
export HOLOSOMA_REPLAY_WANDB_ENV_ID
export HEADLESS="${ISAAC_HEADLESS_ENV}"

cmd=(
  python src/holosoma/holosoma/replay.py
  "exp:${EXP}"
  "command:${COMMAND_PRESET}"
  "perception:${PERCEPTION_PRESET}"
  --training.seed="${SEED}"
  --training.headless="${TRAINING_HEADLESS}"
  --training.num-envs="${NUM_ENVS}"
  --training.enable-viser=True
  --training.viser-port="${VISER_PORT}"
  --training.viser-env-id="${VISER_ENV_ID}"
  --training.viser-show-scandots="${VISER_SHOW_SCANDOTS}"
  --training.isaac-show-scandots="${ISAAC_SHOW_SCANDOTS}"
  --training.isaac-scandots-point-size=3.0
  --simulator.config.debug-viz=True
  --simulator.config.contact-force-viz=False
  --simulator.config.scene.env-spacing=0.0
  --command.setup-terms.motion-command.params.motion-config.motion-file "${MOTION_FILE}"
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler "${MOTION_USE_ADAPTIVE_TIMESTEP_SAMPLER}"
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob "${MOTION_START_AT_ZERO_PROB}"
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend "${MOTION_ENABLE_DEFAULT_POSE_PREPEND}"
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s "${MOTION_DEFAULT_POSE_PREPEND_DURATION_S}"
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append "${MOTION_ENABLE_DEFAULT_POSE_APPEND}"
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s "${MOTION_DEFAULT_POSE_APPEND_DURATION_S}"
  --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale "${MOTION_INIT_NOISE_SCALE}"
  --command.setup-terms.motion-command.params.motion-config.align-motion-to-init-yaw "${MOTION_ALIGN_TO_INIT_YAW}"
  --command.setup-terms.motion-command.params.motion-config.pair-terrain-with-motion "${MOTION_PAIR_TERRAIN_WITH_MOTION}"
  --perception.camera-width="${IMAGE_WIDTH}"
  --perception.camera-height="${IMAGE_HEIGHT}"
  --perception.camera-vfov-deg="${CAMERA_VFOV_DEG}"
  --perception.camera-hfov-deg="${CAMERA_HFOV_DEG}"
  --perception.camera-near="${CAMERA_NEAR}"
  --perception.camera-far="${CAMERA_FAR}"
  --perception.max-distance="${CAMERA_MAX_DISTANCE}"
  --perception.camera-pitch-deg="${CAMERA_PITCH_DEG}"
  --perception.camera-include-robot-mesh="${CAMERA_INCLUDE_ROBOT_MESH}"
  --perception.camera-scandots-stride="${SCANDOTS_STRIDE}"
  --perception.camera-body-name "${CAMERA_BODY_NAME}"
  --perception.camera-frame-quat "${CAMERA_FRAME_QUAT}"
)

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(--command.setup-terms.motion-command.params.motion-config.motion-clip-name "${MOTION_CLIP_NAME}")
fi

cmd+=("$@")

if [[ -n "${TERRAIN_OBJ}" ]]; then
  cmd+=(
    terrain:terrain-load-obj
    --terrain.terrain-term.obj-file-path "${TERRAIN_OBJ}"
    --terrain.terrain-term.num-rows="${TERRAIN_NUM_ROWS}"
    --terrain.terrain-term.num-cols="${TERRAIN_NUM_COLS}"
  )
fi

cmd+=("$@")

if [[ "${DISABLE_RANDOMIZATION_AND_NOISE}" == "1" ]]; then
  # Enforce at the end so user-provided overrides in "$@" cannot re-enable.
  cmd+=(
    randomization:disabled
  )
fi

echo "[INFO] replay command:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
