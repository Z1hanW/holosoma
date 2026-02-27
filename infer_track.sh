#!/usr/bin/env bash
set -euo pipefail

# Minimal unified physics inference:
#   infer_physics obj <ckpt> <motion_path> <object_urdf> [extra tyro args...]
#   infer_physics terrain [<ckpt> <motion_path> <terrain_obj_or_dir>] [extra tyro args...]

usage() {
  cat <<'EOF'
Usage:
  infer_physics obj <ckpt> <motion_path> <object_urdf> [extra tyro args...]
  infer_physics terrain
  infer_physics terrain <motion_path> <terrain_obj_or_dir> [extra tyro args...]
  infer_physics terrain <ckpt> <motion_path> <terrain_obj_or_dir> [extra tyro args...]

Notes:
  - terrain branch defaults to perception:heightmap
  - terrain branch default ckpt:
    /home/ubuntu/FAR/holosoma/logs/WholeBodyTracking/20260224_204920-g1_29dof_wbt_videomimic_mlp-locomotion/model_10000.pt
  - terrain branch default motion:
    /data/terrain/___crisp_clean_motion
  - terrain branch default terrain:
    /data/terrain/___crisp_clean_geometry
  - Viser clip dropdown is enabled
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE="$1"
shift

DEFAULT_TERRAIN_CKPT="/home/ubuntu/FAR/holosoma/logs/WholeBodyTracking/20260224_204920-g1_29dof_wbt_videomimic_mlp-locomotion/model_10000.pt"
DEFAULT_TERRAIN_MOTION="/data/terrain/___crisp_clean_motion"
DEFAULT_TERRAIN_ASSET="/data/terrain/___crisp_clean_geometry"

case "${MODE}" in
  obj|terrain) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    echo "[ERROR] mode must be 'obj' or 'terrain', got: ${MODE}" >&2
    exit 1
    ;;
esac

if [[ "${MODE}" == "obj" ]]; then
  if [[ $# -lt 3 ]]; then
    usage
    exit 1
  fi
  CKPT="$1"
  MOTION_PATH="$2"
  ASSET_PATH="$3"
  shift 3
else
  if [[ $# -eq 0 ]]; then
    CKPT="${DEFAULT_TERRAIN_CKPT}"
    MOTION_PATH="${DEFAULT_TERRAIN_MOTION}"
    ASSET_PATH="${DEFAULT_TERRAIN_ASSET}"
  elif [[ "$1" == --* ]]; then
    CKPT="${DEFAULT_TERRAIN_CKPT}"
    MOTION_PATH="${DEFAULT_TERRAIN_MOTION}"
    ASSET_PATH="${DEFAULT_TERRAIN_ASSET}"
  elif [[ $# -eq 2 ]]; then
    CKPT="${DEFAULT_TERRAIN_CKPT}"
    MOTION_PATH="$1"
    ASSET_PATH="$2"
    shift 2
  elif [[ $# -ge 3 ]]; then
    CKPT="$1"
    MOTION_PATH="$2"
    ASSET_PATH="$3"
    shift 3
  else
    usage
    exit 1
  fi
fi
EXTRA_ARGS=("$@")

if [[ "${CKPT}" != wandb://* ]] && [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ "${MOTION_PATH}" != s3://* ]] && [[ ! -e "${MOTION_PATH}" ]]; then
  echo "[ERROR] motion path not found: ${MOTION_PATH}" >&2
  exit 1
fi
if [[ ! -e "${ASSET_PATH}" ]]; then
  echo "[ERROR] asset path not found: ${ASSET_PATH}" >&2
  exit 1
fi

VISER_PORT=$((RANDOM % 8976 + 1024))
export VISER_ENABLE_CLIP_GUI=1
export VISER_START_PAUSED=1

if [[ "${MODE}" == "obj" ]]; then
  if [[ ! -f "${ASSET_PATH}" ]]; then
    echo "[ERROR] object_urdf must be a file: ${ASSET_PATH}" >&2
    exit 1
  fi

  cmd=(
    python -m holosoma.visualize physics
    --checkpoint "${CKPT}"
    --motion-dir "${MOTION_PATH}"
    --num-envs 1
    --headless True
    --viser-port "${VISER_PORT}"
    --viser-env-id 0
    --viser-update-hz 30
    --viser-recenter True
    --simulator.config.sim.physx.gpu_collision_stack_size 268435456
    --robot.object.enabled True
    --robot.object.object_urdf_path "${ASSET_PATH}"
  )
else
  cmd=(
    python -m holosoma.visualize physics
    --checkpoint "${CKPT}"
    --motion-dir "${MOTION_PATH}"
    --geometry-dir "${ASSET_PATH}"
    --num-envs 1
    --headless True
    --pair-terrain-with-motion True
    --viser-port "${VISER_PORT}"
    --viser-env-id 0
    --viser-update-hz 30
    --viser-recenter True
    --simulator.config.sim.physx.gpu_collision_stack_size 268435456
    perception:heightmap
  )
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] mode=${MODE}"
echo "[INFO] ckpt=${CKPT}"
echo "[INFO] motion=${MOTION_PATH}"
echo "[INFO] asset=${ASSET_PATH}"
echo "[INFO] viser=http://localhost:${VISER_PORT}"

"${cmd[@]}"
