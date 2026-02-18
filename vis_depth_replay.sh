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
#   DEPTH_IMPL=warp_like ./vis_depth_replay.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

EXP=${EXP:-"g1-29dof-wbt-videomimic-mlp"}
MOTION_FILE=${MOTION_FILE:-"${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/far_robot_mj.npz"}
MOTION_CLIP_NAME=${MOTION_CLIP_NAME:-""}
TERRAIN_OBJ=${TERRAIN_OBJ:-"${SCRIPT_DIR}/src/holosoma_retargeting/demo_data/far_robot/far_robot/stairs.obj"}

DEPTH_IMPL=${DEPTH_IMPL:-warp_like}
if [[ "${DEPTH_IMPL}" != "warp_like" ]]; then
  echo "[ERROR] vis_depth_replay.sh is locked to DEPTH_IMPL=warp_like, got: ${DEPTH_IMPL}" >&2
  exit 1
fi
PERCEPTION_PRESET="camera_depth_d435i"

IMAGE_WIDTH=${IMAGE_WIDTH:-106}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-60}
CAMERA_VFOV_DEG=${CAMERA_VFOV_DEG:-58.6}
CAMERA_HFOV_DEG=${CAMERA_HFOV_DEG:-89.5}
CAMERA_NEAR=${CAMERA_NEAR:-0.3}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_INCLUDE_ROBOT_MESH=${CAMERA_INCLUDE_ROBOT_MESH:-True}

SCANDOTS_STRIDE=${SCANDOTS_STRIDE:-1}
CAMERA_BODY_NAME=${CAMERA_BODY_NAME:-d435_joint}
# Match warp_sensors offset_rot_base=[-90,0,-90] in xyzw convention.
CAMERA_FRAME_QUAT=${CAMERA_FRAME_QUAT:-"[-0.5,0.5,-0.5,0.5]"}
HEADLESS=${HEADLESS:-False}
NUM_ENVS=${NUM_ENVS:-1}
SEED=${SEED:-42}
# Enable Viser ray/hit debug overlay for warp-like mesh raycast depth.
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-True}
ISAAC_SHOW_SCANDOTS=${ISAAC_SHOW_SCANDOTS:-False}

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
VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-0}
VISER_START_PAUSED=${VISER_START_PAUSED:-1}
VISER_PERCEPTION_IMAGE_MODE=${VISER_PERCEPTION_IMAGE_MODE:-depth}
VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE=${VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE:-0}
VISER_DISABLE_PERCEPTION_FRUSTUM=${VISER_DISABLE_PERCEPTION_FRUSTUM:-1}
VISER_DISABLE_CONTACT_FORCE_VIZ=${VISER_DISABLE_CONTACT_FORCE_VIZ:-1}
VISER_SCANDOTS_INCLUDE_MISSES=${VISER_SCANDOTS_INCLUDE_MISSES:-0}
VISER_SCANDOTS_RAY_LEN=${VISER_SCANDOTS_RAY_LEN:-2.0}

VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}

if [[ ! -e "${MOTION_FILE}" ]]; then
  echo "[ERROR] MOTION_FILE not found: ${MOTION_FILE}" >&2
  exit 1
fi

if [[ -n "${TERRAIN_OBJ}" ]] && [[ ! -e "${TERRAIN_OBJ}" ]]; then
  echo "[ERROR] TERRAIN_OBJ not found: ${TERRAIN_OBJ}" >&2
  exit 1
fi

echo "[INFO] DEPTH_IMPL=${DEPTH_IMPL}: full-image perception mode with Viser ray/hit overlay enabled."

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
  if [[ "${DEPTH_IMPL}" == "warp_like" ]]; then
    if [[ "${IMAGE_WIDTH}" != "106" || "${IMAGE_HEIGHT}" != "60" ]]; then
      echo "[ERROR] STRICT_OPTIONS=1 requires IMAGE_WIDTH/HEIGHT=106/60 for DEPTH_IMPL=warp_like, got ${IMAGE_WIDTH}/${IMAGE_HEIGHT}" >&2
      exit 1
    fi
    if [[ "${CAMERA_INCLUDE_ROBOT_MESH}" != "True" ]]; then
      echo "[ERROR] STRICT_OPTIONS=1 requires CAMERA_INCLUDE_ROBOT_MESH=True for DEPTH_IMPL=warp_like, got: ${CAMERA_INCLUDE_ROBOT_MESH}" >&2
      exit 1
    fi
  fi
  # Strict terrain check:
  # - If TERRAIN_OBJ is a directory, require a clip-matching OBJ file.
  # - If TERRAIN_OBJ is a single OBJ file, treat it as an explicit global terrain.
  if [[ -d "${TERRAIN_OBJ}" ]]; then
    motion_stem="$(basename "${MOTION_FILE}")"
    motion_stem="${motion_stem%.npz}"
    if [[ -n "${MOTION_CLIP_NAME}" ]]; then
      motion_stem="${MOTION_CLIP_NAME}"
    fi
    if [[ ! -f "${TERRAIN_OBJ}/${motion_stem}.obj" && ! -f "${TERRAIN_OBJ}/${motion_stem}.OBJ" ]]; then
      echo "[ERROR] STRICT_OPTIONS=1 requires ${TERRAIN_OBJ}/${motion_stem}.obj (or .OBJ) for motion clip '${motion_stem}'" >&2
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
echo "[INFO] CAMERA_FRAME_QUAT=${CAMERA_FRAME_QUAT}"
echo "[INFO] PERCEPTION_PRESET=${PERCEPTION_PRESET}"
echo "[INFO] DEPTH_IMPL=${DEPTH_IMPL}"
echo "[INFO] CAMERA_CFG width=${IMAGE_WIDTH} height=${IMAGE_HEIGHT} vfov=${CAMERA_VFOV_DEG} hfov=${CAMERA_HFOV_DEG} near=${CAMERA_NEAR} far=${CAMERA_FAR} include_robot_mesh=${CAMERA_INCLUDE_ROBOT_MESH}"
echo "[INFO] VISER=http://localhost:${VISER_PORT}"
echo "[INFO] VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI}"
echo "[INFO] VISER_START_PAUSED=${VISER_START_PAUSED}"
echo "[INFO] VISER_PERCEPTION_IMAGE_MODE=${VISER_PERCEPTION_IMAGE_MODE}"
echo "[INFO] VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE=${VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE}"
echo "[INFO] VISER_DISABLE_PERCEPTION_FRUSTUM=${VISER_DISABLE_PERCEPTION_FRUSTUM}"
echo "[INFO] VISER_DISABLE_CONTACT_FORCE_VIZ=${VISER_DISABLE_CONTACT_FORCE_VIZ}"
echo "[INFO] VISER_SCANDOTS_INCLUDE_MISSES=${VISER_SCANDOTS_INCLUDE_MISSES}"
echo "[INFO] VISER_SCANDOTS_RAY_LEN=${VISER_SCANDOTS_RAY_LEN}"
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

export VISER_ENABLE_CLIP_GUI
export VISER_START_PAUSED
export VISER_PERCEPTION_IMAGE_MODE
export VISER_DISABLE_PERCEPTION_IMAGE_PIPELINE
export VISER_DISABLE_PERCEPTION_FRUSTUM
export VISER_DISABLE_CONTACT_FORCE_VIZ
export VISER_SCANDOTS_INCLUDE_MISSES
export VISER_SCANDOTS_RAY_LEN

cmd=(
  python src/holosoma/holosoma/replay.py
  "exp:${EXP}"
  "perception:${PERCEPTION_PRESET}"
  --training.seed="${SEED}"
  --training.headless="${HEADLESS}"
  --training.num_envs="${NUM_ENVS}"
  --training.enable_viser=True
  --training.viser_port="${VISER_PORT}"
  --training.viser_show_scandots="${VISER_SHOW_SCANDOTS}"
  --training.isaac_show_scandots="${ISAAC_SHOW_SCANDOTS}"
  --training.isaac_scandots_point_size=3.0
  --simulator.config.debug_viz=True
  --simulator.config.contact_force_viz=False
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
  --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled=False
  --randomization.reset_terms.randomize_camera_raycast.params.enabled=False
  --randomization.setup_terms.push_randomizer_state.params.enabled=False
  --randomization.setup_terms.setup_dof_pos_bias.params.enabled=False
  --randomization.reset_terms.randomize_dof_state.params.randomize_dof_pos_bias=False
  --randomization.reset_terms.randomize_dof_state.params.joint_pos_bias_range="[0.0,0.0]"
  --perception.camera_width="${IMAGE_WIDTH}"
  --perception.camera_height="${IMAGE_HEIGHT}"
  --perception.camera_vfov_deg="${CAMERA_VFOV_DEG}"
  --perception.camera_hfov_deg="${CAMERA_HFOV_DEG}"
  --perception.camera_near="${CAMERA_NEAR}"
  --perception.camera_far="${CAMERA_FAR}"
  --perception.camera_include_robot_mesh="${CAMERA_INCLUDE_ROBOT_MESH}"
  --perception.camera_scandots_stride="${SCANDOTS_STRIDE}"
  --perception.camera_body_name "${CAMERA_BODY_NAME}"
  --perception.camera_frame_quat "${CAMERA_FRAME_QUAT}"
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
