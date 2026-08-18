#!/usr/bin/env bash
set -euo pipefail

# Isaac Sim <-> Viser synchronized visualization entrypoint.
# Examples:
#   ./sync_isaacsim_viser.sh
#   ./sync_isaacsim_viser.sh heightmap
#   ./sync_isaacsim_viser.sh camera_depth_d435i --training.headless=True
#   HEADLESS=True VISER_PORT=9001 ./sync_isaacsim_viser.sh
#   PERCEPTION=heightmap ./sync_isaacsim_viser.sh
#
# Optional env overrides:
#   ISAACSIM_CONDA_ENV=sim|hssim
#   EXP=g1-29dof-wbt
#   COMMAND_PRESET=g1-29dof-wbt
#   PERCEPTION=camera_depth_d435i|heightmap
#   MOTION_FILE=src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz
#   TERRAIN_OBJ=src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj
#   TERRAIN_NUM_ROWS=1
#   TERRAIN_NUM_COLS=1
#   ALLOW_OBJECT_MOTION=0|1
#   NUM_ENVS=1
#   HEADLESS=False
#   VISER_PORT=xxxx
#   VISER_ENV_ID=0
#   VISER_UPDATE_HZ=30
#   VISER_RECENTER=True
#   VISER_SHOW_SCANDOTS=True
#   VISER_SHOW_PERCEPTION_FRUSTUM=0|1
#   VISER_DISABLE_PERCEPTION_FRUSTUM=0|1
#   VISER_STRICT_CAMERA_RAYS=0|1
#   VISER_SCANDOTS_INCLUDE_MISSES=1|0
#   ISAAC_SCANDOTS_INCLUDE_MISSES=1|0
#   CAMERA_WIDTH=106
#   CAMERA_HEIGHT=60
#   CAMERA_BODY_NAME=torso_link
#   CAMERA_SENSOR_OFFSET=[0.0576235,0.01753,0.42987]
#   CAMERA_MOUNT_QUAT=[0.0,0.40354529635239006,0.0,0.9149596678498247]
#   CAMERA_FRAME_QUAT=[-0.5,0.5,-0.5,0.5]
#   CAMERA_PITCH_DEG=0.0
#   CAMERA_NEAR=0.3
#   CAMERA_FAR=3.0
#   CAMERA_MAX_DISTANCE=3.0
#   CAMERA_INCLUDE_ROBOT_MESH=True
#   DISABLE_CAMERA_RANDOMIZATION=1
#   HOLOSOMA_CAMERA_AUTOFIX_BACKWARD=0
#   HOLOSOMA_CAMERA_DISABLE_OFFSETS=1
#   HOLOSOMA_CAMERA_STRICT_WARP=1
#   HOLOSOMA_CAMERA_EXTRA_YAW_DEG=0.0
#   HOLOSOMA_CAMERA_BACKWARD_RATIO_THRESHOLD=0.6
#   KEEP_OPEN=1|0

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
cd "${SCRIPT_DIR}"

EXP=${EXP:-g1-29dof-wbt}
COMMAND_PRESET=${COMMAND_PRESET:-g1-29dof-wbt}
PERCEPTION=${PERCEPTION:-camera_depth_d435i}
MOTION_FILE=${MOTION_FILE:-src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz}
TERRAIN_OBJ=${TERRAIN_OBJ:-src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj}
TERRAIN_NUM_ROWS=${TERRAIN_NUM_ROWS:-1}
TERRAIN_NUM_COLS=${TERRAIN_NUM_COLS:-1}
PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-0}
ALLOW_OBJECT_MOTION=${ALLOW_OBJECT_MOTION:-0}
NUM_ENVS=${NUM_ENVS:-1}
HEADLESS=${HEADLESS:-0}
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-True}
VISER_SHOW_PERCEPTION_FRUSTUM=${VISER_SHOW_PERCEPTION_FRUSTUM:-0}
VISER_DISABLE_PERCEPTION_FRUSTUM=${VISER_DISABLE_PERCEPTION_FRUSTUM:-0}
VISER_STRICT_CAMERA_RAYS=${VISER_STRICT_CAMERA_RAYS:-1}
VISER_SCANDOTS_INCLUDE_MISSES=${VISER_SCANDOTS_INCLUDE_MISSES:-1}
ISAAC_SCANDOTS_INCLUDE_MISSES=${ISAAC_SCANDOTS_INCLUDE_MISSES:-1}
ISAAC_SHOW_SCANDOTS=${ISAAC_SHOW_SCANDOTS:-False}
ISAAC_SCANDOTS_POINT_SIZE=${ISAAC_SCANDOTS_POINT_SIZE:-3.0}
KEEP_OPEN=${KEEP_OPEN:-1}

headless_norm="$(printf '%s' "${HEADLESS}" | tr '[:upper:]' '[:lower:]')"
case "${headless_norm}" in
  1|true|yes|on)
    HEADLESS_ENV=1
    TRAINING_HEADLESS=True
    ;;
  0|false|no|off|"")
    HEADLESS_ENV=0
    TRAINING_HEADLESS=False
    ;;
  *)
    echo "[ERROR] Invalid HEADLESS=${HEADLESS}. Use one of: 0/1/True/False" >&2
    exit 1
    ;;
esac

pair_norm="$(printf '%s' "${PAIR_TERRAIN_WITH_MOTION}" | tr '[:upper:]' '[:lower:]')"
case "${pair_norm}" in
  1|true|yes|on)
    PAIR_TERRAIN_WITH_MOTION_CLI=True
    ;;
  0|false|no|off|"")
    PAIR_TERRAIN_WITH_MOTION_CLI=False
    ;;
  *)
    echo "[ERROR] Invalid PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION}. Use one of: 0/1/True/False" >&2
    exit 1
    ;;
esac

# d435 defaults (only applied when PERCEPTION=camera_depth_d435i)
CAMERA_WIDTH=${CAMERA_WIDTH:-106}
CAMERA_HEIGHT=${CAMERA_HEIGHT:-60}
CAMERA_BODY_NAME=${CAMERA_BODY_NAME:-torso_link}
CAMERA_SENSOR_OFFSET=${CAMERA_SENSOR_OFFSET:-"[0.0576235,0.01753,0.42987]"}
CAMERA_MOUNT_QUAT=${CAMERA_MOUNT_QUAT:-"[0.0,0.40354529635239006,0.0,0.9149596678498247]"}
CAMERA_FRAME_QUAT=${CAMERA_FRAME_QUAT:-"[-0.5,0.5,-0.5,0.5]"}
CAMERA_PITCH_DEG=${CAMERA_PITCH_DEG:-0.0}
CAMERA_NEAR=${CAMERA_NEAR:-0.3}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-3.0}
CAMERA_INCLUDE_ROBOT_MESH=${CAMERA_INCLUDE_ROBOT_MESH:-True}
DISABLE_CAMERA_RANDOMIZATION=${DISABLE_CAMERA_RANDOMIZATION:-1}
HOLOSOMA_CAMERA_AUTOFIX_BACKWARD=${HOLOSOMA_CAMERA_AUTOFIX_BACKWARD:-0}
HOLOSOMA_CAMERA_DISABLE_OFFSETS=${HOLOSOMA_CAMERA_DISABLE_OFFSETS:-1}
HOLOSOMA_CAMERA_STRICT_WARP=${HOLOSOMA_CAMERA_STRICT_WARP:-1}

# Optional first positional argument as perception type.
# Supported: heightmap | camera_depth_d435i | perception:heightmap | perception:camera_depth_d435i
if [[ $# -gt 0 ]]; then
  case "$1" in
    heightmap|camera_depth_d435i)
      PERCEPTION="$1"
      shift
      ;;
    perception:heightmap|perception:camera_depth_d435i)
      PERCEPTION="${1#perception:}"
      shift
      ;;
  esac
fi

if [[ ! -e "${MOTION_FILE}" ]]; then
  echo "[ERROR] MOTION_FILE not found: ${MOTION_FILE}" >&2
  exit 1
fi
if [[ -d "${MOTION_FILE}" ]]; then
  MOTION_CLIP_COUNT=$(find "${MOTION_FILE}" -maxdepth 1 -type f -name '*.npz' | wc -l)
  if [[ "${MOTION_CLIP_COUNT}" -eq 0 ]]; then
    echo "[ERROR] MOTION_FILE directory has no .npz clips: ${MOTION_FILE}" >&2
    exit 1
  fi
fi

if [[ -n "${TERRAIN_OBJ}" && ! -e "${TERRAIN_OBJ}" ]]; then
  echo "[ERROR] TERRAIN_OBJ not found: ${TERRAIN_OBJ}" >&2
  exit 1
fi
if [[ -d "${TERRAIN_OBJ}" ]]; then
  TERRAIN_OBJ_COUNT=$(find "${TERRAIN_OBJ}" -maxdepth 1 -type f \( -name '*.obj' -o -name '*.OBJ' \) | wc -l)
  if [[ "${TERRAIN_OBJ_COUNT}" -eq 0 ]]; then
    echo "[ERROR] TERRAIN_OBJ directory has no .obj tiles: ${TERRAIN_OBJ}" >&2
    exit 1
  fi
fi

if [[ "${ALLOW_OBJECT_MOTION}" != "1" && "${MOTION_FILE}" == *"_w_obj.npz" ]]; then
  echo "[ERROR] MOTION_FILE appears to require object actor: ${MOTION_FILE}" >&2
  echo "        Use a robot-only motion file, or set ALLOW_OBJECT_MOTION=1 intentionally." >&2
  exit 1
fi

# Prefer user override; otherwise auto-detect existing IsaacSim env name.
eval "$(conda shell.bash hook)"
if [[ -n "${ISAACSIM_CONDA_ENV:-}" ]]; then
  conda activate "${ISAACSIM_CONDA_ENV}"
elif conda env list | awk 'NF > 0 && $1 !~ /^#/ {print $1}' | grep -qx "sim"; then
  conda activate sim
elif conda env list | awk 'NF > 0 && $1 !~ /^#/ {print $1}' | grep -qx "hssim"; then
  conda activate hssim
else
  echo "[ERROR] Could not find conda env 'sim' or 'hssim'." >&2
  echo "        Set ISAACSIM_CONDA_ENV explicitly, e.g.:" >&2
  echo "        ISAACSIM_CONDA_ENV=sim ./sync_isaacsim_viser.sh" >&2
  exit 1
fi

export OMNI_KIT_ACCEPT_EULA=1
export HOLOSOMA_VISER_PORT="${VISER_PORT}"
export VISER_SHOW_PERCEPTION_FRUSTUM
export VISER_DISABLE_PERCEPTION_FRUSTUM
export VISER_STRICT_CAMERA_RAYS
export VISER_SCANDOTS_INCLUDE_MISSES
export ISAAC_SCANDOTS_INCLUDE_MISSES
export HOLOSOMA_REPLAY_KEEP_OPEN="${HOLOSOMA_REPLAY_KEEP_OPEN:-${KEEP_OPEN}}"
export HOLOSOMA_CAMERA_AUTOFIX_BACKWARD
export HOLOSOMA_CAMERA_DISABLE_OFFSETS
export HOLOSOMA_CAMERA_STRICT_WARP
export HEADLESS="${HEADLESS_ENV}"

echo "[INFO] Isaac Sim + Viser sync launch"
echo "[INFO] EXP=${EXP}"
echo "[INFO] COMMAND_PRESET=${COMMAND_PRESET}"
echo "[INFO] PERCEPTION=${PERCEPTION}"
echo "[INFO] MOTION_FILE=${MOTION_FILE}"
if [[ -d "${MOTION_FILE}" ]]; then
  echo "[INFO] MOTION_FILE mode=directory (multi-clip), clips=${MOTION_CLIP_COUNT}"
fi
echo "[INFO] TERRAIN_OBJ=${TERRAIN_OBJ}"
if [[ -d "${TERRAIN_OBJ}" ]]; then
  echo "[INFO] TERRAIN_OBJ mode=directory (tile bank), tiles=${TERRAIN_OBJ_COUNT}"
fi
echo "[INFO] PAIR_TERRAIN_WITH_MOTION(raw)=${PAIR_TERRAIN_WITH_MOTION}"
echo "[INFO] PAIR_TERRAIN_WITH_MOTION(cli)=${PAIR_TERRAIN_WITH_MOTION_CLI}"
echo "[INFO] HEADLESS(raw)=${HEADLESS}"
echo "[INFO] HEADLESS(env/int)=${HEADLESS_ENV}"
echo "[INFO] training.headless=${TRAINING_HEADLESS}"
echo "[INFO] NUM_ENVS=${NUM_ENVS}"
echo "[INFO] VISER=http://localhost:${VISER_PORT}"
echo "[INFO] VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS}"
echo "[INFO] VISER_SHOW_PERCEPTION_FRUSTUM=${VISER_SHOW_PERCEPTION_FRUSTUM}"
echo "[INFO] VISER_DISABLE_PERCEPTION_FRUSTUM=${VISER_DISABLE_PERCEPTION_FRUSTUM}"
echo "[INFO] VISER_STRICT_CAMERA_RAYS=${VISER_STRICT_CAMERA_RAYS}"
echo "[INFO] VISER_SCANDOTS_INCLUDE_MISSES=${VISER_SCANDOTS_INCLUDE_MISSES}"
echo "[INFO] ISAAC_SHOW_SCANDOTS=${ISAAC_SHOW_SCANDOTS}"
echo "[INFO] ISAAC_SCANDOTS_POINT_SIZE=${ISAAC_SCANDOTS_POINT_SIZE}"
echo "[INFO] HOLOSOMA_REPLAY_KEEP_OPEN=${HOLOSOMA_REPLAY_KEEP_OPEN}"
if [[ "${PERCEPTION}" == "camera_depth_d435i" ]]; then
  echo "[INFO] D435 camera: ${CAMERA_HEIGHT}x${CAMERA_WIDTH}, body=${CAMERA_BODY_NAME}, offset=${CAMERA_SENSOR_OFFSET}, pitch=${CAMERA_PITCH_DEG}, mount_quat=${CAMERA_MOUNT_QUAT}, frame_quat=${CAMERA_FRAME_QUAT}"
  echo "[INFO] D435 depth range: near=${CAMERA_NEAR}, far=${CAMERA_FAR}, max_distance=${CAMERA_MAX_DISTANCE}"
  echo "[INFO] D435 include_robot_mesh=${CAMERA_INCLUDE_ROBOT_MESH}"
  echo "[INFO] DISABLE_CAMERA_RANDOMIZATION=${DISABLE_CAMERA_RANDOMIZATION}"
  echo "[INFO] HOLOSOMA_CAMERA_AUTOFIX_BACKWARD=${HOLOSOMA_CAMERA_AUTOFIX_BACKWARD}"
  echo "[INFO] HOLOSOMA_CAMERA_DISABLE_OFFSETS=${HOLOSOMA_CAMERA_DISABLE_OFFSETS}"
  echo "[INFO] HOLOSOMA_CAMERA_STRICT_WARP=${HOLOSOMA_CAMERA_STRICT_WARP}"
fi

cmd=(
  python src/holosoma/holosoma/replay.py
  "exp:${EXP}"
  "command:${COMMAND_PRESET}"
  simulator:isaacsim
  "perception:${PERCEPTION}"
  --training.num_envs="${NUM_ENVS}"
  --training.headless="${TRAINING_HEADLESS}"
  --training.enable_viser=True
  --training.viser_port="${VISER_PORT}"
  --training.viser_env_id="${VISER_ENV_ID}"
  --training.viser_update_hz="${VISER_UPDATE_HZ}"
  --training.viser_recenter="${VISER_RECENTER}"
  --training.viser_show_scandots="${VISER_SHOW_SCANDOTS}"
  --training.isaac_show_scandots="${ISAAC_SHOW_SCANDOTS}"
  --training.isaac_scandots_point_size="${ISAAC_SCANDOTS_POINT_SIZE}"
  --simulator.config.scene.env_spacing=0.0
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}"
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion="${PAIR_TERRAIN_WITH_MOTION_CLI}"
)

if [[ "${PERCEPTION}" == "camera_depth_d435i" ]]; then
  cmd+=(
    --perception.camera_width "${CAMERA_WIDTH}"
    --perception.camera_height "${CAMERA_HEIGHT}"
    --perception.camera_body_name "${CAMERA_BODY_NAME}"
    --perception.sensor_offset "${CAMERA_SENSOR_OFFSET}"
    --perception.camera_mount_quat "${CAMERA_MOUNT_QUAT}"
    --perception.camera_frame_quat "${CAMERA_FRAME_QUAT}"
    --perception.camera_pitch_deg "${CAMERA_PITCH_DEG}"
    --perception.camera_near "${CAMERA_NEAR}"
    --perception.camera_far "${CAMERA_FAR}"
    --perception.max_distance "${CAMERA_MAX_DISTANCE}"
    --perception.camera_include_robot_mesh "${CAMERA_INCLUDE_ROBOT_MESH}"
  )

  if [[ "${DISABLE_CAMERA_RANDOMIZATION}" == "1" ]]; then
    cmd+=(
      --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled=False
      --randomization.reset_terms.randomize_camera_raycast.params.enabled=False
    )
  fi
fi

if [[ -n "${TERRAIN_OBJ}" ]]; then
  cmd+=(
    terrain:terrain-load-obj
    --terrain.terrain-term.obj-file-path "${TERRAIN_OBJ}"
    --terrain.terrain-term.num-rows "${TERRAIN_NUM_ROWS}"
    --terrain.terrain-term.num-cols "${TERRAIN_NUM_COLS}"
  )
fi

cmd+=("$@")
"${cmd[@]}"
