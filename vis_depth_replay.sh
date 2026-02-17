#!/usr/bin/env bash
set -euo pipefail

# Minimal Isaac Sim kinematic replay for depth-camera sanity check.
# Default behavior:
# - Kinematic replay (motion command)
# - Scene scandots in Isaac Sim
# - Perception camera/frustum + depth image in Viser GUI (realtime)
#
# Usage:
#   ./vis_depth_replay.sh
#   MOTION_FILE=/abs/or/rel/path/to/motion_or_dir TERRAIN_OBJ=/abs/or/rel/path/to/scene.obj ./vis_depth_replay.sh
#   DEPTH_IMPL=rendered ./vis_depth_replay.sh   # rendered|depth_sensor|raycast|scandots

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP=${EXP:-"g1-29dof-wbt-videomimic-mlp"}
MOTION_FILE=${MOTION_FILE:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz"}
MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-""}
TERRAIN_OBJ=${TERRAIN_OBJ:-"${SCRIPT_DIR}/src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj"}

DEPTH_IMPL=${DEPTH_IMPL:-scandots}
case "${DEPTH_IMPL}" in
  scandots)
    PERCEPTION_PRESET="camera_depth_d435i_scandots"
    ;;
  rendered)
    PERCEPTION_PRESET="camera_depth_d435i_rendered"
    ;;
  depth_sensor)
    PERCEPTION_PRESET="camera_depth_d435i_depth_sensor"
    ;;
  raycast)
    PERCEPTION_PRESET="camera_depth_d435i"
    ;;
  *)
    echo "[ERROR] Unknown DEPTH_IMPL=${DEPTH_IMPL}. Use scandots|rendered|depth_sensor|raycast." >&2
    exit 1
    ;;
esac

IMAGE_WIDTH=${IMAGE_WIDTH:-640}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-360}
SCANDOTS_STRIDE=${SCANDOTS_STRIDE:-4}
CAMERA_BODY_NAME=${CAMERA_BODY_NAME:-d435_joint}
HEADLESS=${HEADLESS:-False}
NUM_ENVS=${NUM_ENVS:-1}
SEED=${SEED:-42}

# Motion alignment debug defaults (deterministic + no synthetic transitions).
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

VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}

if [[ ! -e "${MOTION_FILE}" ]]; then
  echo "[ERROR] MOTION_FILE not found: ${MOTION_FILE}" >&2
  exit 1
fi

if [[ -n "${TERRAIN_OBJ}" ]] && [[ ! -e "${TERRAIN_OBJ}" ]]; then
  echo "[ERROR] TERRAIN_OBJ not found: ${TERRAIN_OBJ}" >&2
  exit 1
fi

if [[ "${DEPTH_IMPL}" != "scandots" ]]; then
  echo "[WARN] DEPTH_IMPL=${DEPTH_IMPL}: scene scan points may disappear. Use DEPTH_IMPL=scandots for scan-point debugging."
fi

if [[ "${STRICT_OPTIONS}" == "1" ]]; then
  if [[ "${EXP}" != "g1-29dof-wbt-videomimic-mlp" ]]; then
    echo "[ERROR] STRICT_OPTIONS=1 requires EXP=g1-29dof-wbt-videomimic-mlp, got: ${EXP}" >&2
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
  if [[ "${DEPTH_IMPL}" != "scandots" ]]; then
    echo "[ERROR] STRICT_OPTIONS=1 requires DEPTH_IMPL=scandots, got: ${DEPTH_IMPL}" >&2
    exit 1
  fi
  if [[ -f "${MOTION_FILE}" && "${MOTION_FILE}" == *.npz && -f "${TERRAIN_OBJ}" && "${TERRAIN_OBJ}" == *.obj ]]; then
    motion_stem="$(basename "${MOTION_FILE}" .npz)"
    terrain_stem="$(basename "${TERRAIN_OBJ}" .obj)"
    if [[ "${motion_stem}" != "${terrain_stem}" ]]; then
      echo "[ERROR] STRICT_OPTIONS=1 requires terrain stem to match motion stem: motion=${motion_stem}, terrain=${terrain_stem}" >&2
      exit 1
    fi
  fi
fi

echo "[INFO] EXP=exp:${EXP}"
echo "[INFO] MOTION_FILE=${MOTION_FILE}"
echo "[INFO] TERRAIN_OBJ=${TERRAIN_OBJ}"
if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  echo "[INFO] MOTION_CLIP_NAME=${MOTION_CLIP_NAME}"
fi
echo "[INFO] CAMERA_BODY_NAME=${CAMERA_BODY_NAME}"
echo "[INFO] PERCEPTION_PRESET=${PERCEPTION_PRESET}"
echo "[INFO] VISER=http://localhost:${VISER_PORT}"
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
    root = body_pos[:, 0, :]
    print(f"[DEBUG] root start xyz={root[0].tolist()} end xyz={root[-1].tolist()}")
    print(f"[DEBUG] root min xyz={root.min(axis=0).tolist()} max xyz={root.max(axis=0).tolist()}")
    if "object_pos_w" in data:
        obj = np.asarray(data["object_pos_w"], dtype=np.float32)
        print(f"[DEBUG] object start xyz={obj[0].tolist()} end xyz={obj[-1].tolist()}")
        print(f"[DEBUG] object min xyz={obj.min(axis=0).tolist()} max xyz={obj.max(axis=0).tolist()}")

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

cmd=(
  python src/holosoma/holosoma/replay.py
  "exp:${EXP}"
  "perception:${PERCEPTION_PRESET}"
  --training.seed="${SEED}"
  --training.headless="${HEADLESS}"
  --training.num_envs="${NUM_ENVS}"
  --training.enable_viser=True
  --training.viser_port="${VISER_PORT}"
  --training.viser_show_scandots=True
  --training.isaac_show_scandots=True
  --training.isaac_scandots_point_size=3.0
  --simulator.config.debug_viz=True
  --simulator.config.scene.env_spacing=0.0
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_FILE}"
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler "${MOTION_USE_ADAPTIVE_TIMESTEP_SAMPLER}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${MOTION_START_AT_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend "${MOTION_ENABLE_DEFAULT_POSE_PREPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s "${MOTION_DEFAULT_POSE_PREPEND_DURATION_S}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append "${MOTION_ENABLE_DEFAULT_POSE_APPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s "${MOTION_DEFAULT_POSE_APPEND_DURATION_S}"
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale "${MOTION_INIT_NOISE_SCALE}"
  --command.setup_terms.motion_command.params.motion_config.align_motion_to_init_yaw "${MOTION_ALIGN_TO_INIT_YAW}"
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion "${MOTION_PAIR_TERRAIN_WITH_MOTION}"
  --perception.camera_width="${IMAGE_WIDTH}"
  --perception.camera_height="${IMAGE_HEIGHT}"
  --perception.camera_scandots_stride="${SCANDOTS_STRIDE}"
  --perception.camera_body_name "${CAMERA_BODY_NAME}"
)

if [[ -n "${MOTION_CLIP_NAME}" ]]; then
  cmd+=(--command.setup_terms.motion_command.params.motion_config.motion_clip_name "${MOTION_CLIP_NAME}")
fi

if [[ -n "${TERRAIN_OBJ}" ]]; then
  cmd+=(
    terrain:terrain-load-obj
    --terrain.terrain-term.obj-file-path "${TERRAIN_OBJ}"
    --terrain.terrain-term.num_rows=1
    --terrain.terrain-term.num_cols=1
  )
fi

cmd+=("$@")
"${cmd[@]}"
