#!/usr/bin/env bash
set -euo pipefail

# Unified IsaacSim + Viser interactive inference for distilled box-carry policies.
#
# Features:
# - Two branches: mocap | depth
# - Viser clip selection GUI
# - Viser manual command GUI (root-frame XY + yaw), VideoMimic-style workflow
# - Optional hardware joystick via pygame/bridge backend
#
# Usage:
#   bash infer_box_joystick.sh <mocap|depth> [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/.../files] [extra tyro args...]
#
# Examples:
#   bash infer_box_joystick.sh mocap /abs/path/model.pt
#   bash infer_box_joystick.sh depth /abs/path/model.pt --viser-port 18080
#   bash infer_box_joystick.sh mocap https://wandb.ai/zihanw22/WholeBodyTracking/runs/d20ktze6/files

usage() {
  cat <<'EOF'
Usage:
  bash infer_box_joystick.sh <mocap|depth> [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/.../files] [extra tyro args...]

Modes:
  mocap   Distilled policy with box-state (mocap) actor observation
  depth   Distilled policy with camera depth perception (D435i)

Optional env vars:
  INFER_DATASET           (default: omomo; options: omomo|behave|mixed)
  MOTION_DIR              (optional override; default chosen by INFER_DATASET)
  OBJECT_URDF             (optional override; default chosen by INFER_DATASET)
  GEOMETRY_DIR            (optional; OBJ file/dir for terrain visualization)
  PAIR_TERRAIN_WITH_MOTION (default: False)
  NUM_ENVS                (default: 1)
  HEADLESS                (default: True; accepts 0/1/true/false/yes/no/on/off)
  VISER_PORT              (default: random)
  VISER_ENV_ID            (default: 0)
  VISER_UPDATE_HZ         (default: 30)
  VISER_RECENTER          (default: True)
  WANDB_MODEL_FILE        (default varies by mode; used when checkpoint is a wandb run URL)
  MOCAP_PERCEPTION_PRESET (default: checkpoint; checkpoint|none|heightmap)
  DEPTH_PERCEPTION_PRESET (default: checkpoint; checkpoint|d435i)

Hardware joystick (optional):
  VISER_MANUAL_USE_HW_JOYSTICK=1
  VISER_MANUAL_HW_BACKEND=auto|pygame|bridge
  VISER_MANUAL_HW_DEVICE=0
  VISER_MANUAL_HW_TYPE=xbox|switch
  USE_HW_JOYSTICK_BRIDGE=True   # optional legacy bridge joystick path (requires real joystick)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MODE_INPUT="$1"
shift

case "${MODE_INPUT}" in
  mocap|depth) ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  mixed_goal)
    echo "[ERROR] mixed_goal inference has been removed. Use mode=depth or mode=mocap." >&2
    exit 1
    ;;
  *)
    echo "[ERROR] mode must be one of: mocap|depth. Got: ${MODE_INPUT}" >&2
    exit 1
    ;;
esac

MODE="${MODE_INPUT}"

LOG_ROOT="/data/logs_new/WholeBodyTracking"
MOCAP_TRAINING_NAME_DEFAULT="g1_29dof_wbt_w_object_distill_box_mocap_access_to_mocap_data"
DEPTH_TRAINING_NAME_DEFAULT="g1_29dof_wbt_w_object_distill_box_perception_access_to_depth"
MOCAP_CHECKPOINT_DEFAULT=${MOCAP_CHECKPOINT_DEFAULT:-"wandb://zihanw22/WholeBodyTracking/d20ktze6/model_00800.pt"}
DEPTH_CHECKPOINT_DEFAULT=${DEPTH_CHECKPOINT_DEFAULT:-"wandb://zihanw22/WholeBodyTracking/xplmudrp/model_01000.pt"}
if [[ -n "${WANDB_MODEL_FILE+x}" && -n "${WANDB_MODEL_FILE}" ]]; then
  WANDB_MODEL_FILE_FROM_ENV=1
else
  WANDB_MODEL_FILE_FROM_ENV=0
  if [[ "${MODE}" == "depth" ]]; then
    WANDB_MODEL_FILE="model_01000.pt"
  else
    WANDB_MODEL_FILE="model_00800.pt"
  fi
fi

parse_wandb_run_url() {
  local ref="$1"
  local clean_ref="${ref%%\?*}"
  if [[ "${clean_ref}" != https://wandb.ai/*/runs/* ]]; then
    return 1
  fi

  local trimmed="${clean_ref#https://wandb.ai/}"
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  IFS='/' read -r -a parts <<< "${trimmed}"
  if [[ "${#parts[@]}" -lt 4 || "${parts[2]}" != "runs" ]]; then
    return 1
  fi

  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[3]}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi

  if [[ "${#parts[@]}" -ge 6 && "${parts[4]}" == "files" ]]; then
    explicit_file="${trimmed#${entity}/${project}/runs/${run_id}/files/}"
  fi

  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

extract_wandb_run_id_from_url() {
  local ref="$1"
  local parsed=""
  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -n "${parsed}" ]]; then
    IFS=$'\t' read -r _entity _project run_id _explicit_file <<< "${parsed}"
    echo "${run_id}"
    return 0
  fi
  echo ""
}

resolve_remote_wandb_checkpoint_name() {
  local entity="$1"
  local project="$2"
  local run_id="$3"

  python - "${entity}" "${project}" "${run_id}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
sanitized_sys_path: list[str] = []
for path_entry in sys.path:
    if path_entry in {"", "."}:
        continue
    try:
        if Path(path_entry).resolve() == repo_root:
            continue
    except Exception:
        pass
    sanitized_sys_path.append(path_entry)
sys.path = sanitized_sys_path

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id = sys.argv[1:4]
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
model_pattern = re.compile(r"^model_(\d+)\.pt$")
latest_step = -1
latest_name = ""
for file_obj in run.files():
    name = getattr(file_obj, "name", "")
    match = model_pattern.match(name)
    if not match:
        continue
    step = int(match.group(1))
    if step >= latest_step:
        latest_step = step
        latest_name = name
if latest_name:
    print(latest_name)
PY
}

normalize_checkpoint_ref() {
  local ref="$1"
  if [[ "${ref}" == https://wandb.ai/*/runs/* ]]; then
    local parsed=""
    local entity=""
    local project=""
    local run_id=""
    local explicit_file=""
    local model_file="${WANDB_MODEL_FILE}"

    parsed="$(parse_wandb_run_url "${ref}" || true)"
    if [[ -n "${parsed}" ]]; then
      IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
      if [[ -n "${explicit_file}" ]]; then
        model_file="${explicit_file}"
      elif [[ "${WANDB_MODEL_FILE_FROM_ENV}" != "1" ]]; then
        local remote_model_file=""
        remote_model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}")"
        if [[ -n "${remote_model_file}" ]]; then
          model_file="${remote_model_file}"
          echo "[INFO] Resolved wandb run URL to latest remote checkpoint: ${model_file}" >&2
        fi
      fi
      if [[ -n "${entity}" && -n "${project}" && -n "${run_id}" && -n "${model_file}" ]]; then
        ref="wandb://${entity}/${project}/${run_id}/${model_file}"
      fi
    fi
  fi
  echo "${ref}"
}

resolve_local_checkpoint_from_run_url() {
  local ref="$1"
  local parsed=""
  local run_id=""
  local explicit_file=""
  local wandb_run_dir=""
  local run_log_dir=""
  local local_ckpt=""

  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo ""
    return 0
  fi
  IFS=$'\t' read -r _entity _project run_id explicit_file <<< "${parsed}"

  wandb_run_dir=$(find /data/logs_new -maxdepth 8 -type d -name "run-*-${run_id}" 2>/dev/null | head -n 1 || true)
  if [[ -z "${wandb_run_dir}" ]]; then
    echo ""
    return 0
  fi

  run_log_dir="$(dirname "$(dirname "$(dirname "${wandb_run_dir}")")")"
  if [[ -n "${explicit_file}" && -f "${run_log_dir}/${explicit_file}" ]]; then
    local_ckpt="${run_log_dir}/${explicit_file}"
  else
    local_ckpt=$(ls -1 "${run_log_dir}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  fi
  echo "${local_ckpt}"
}

find_latest_ckpt() {
  local root="$1"
  local training_name="$2"
  local latest_run=""
  local latest_ckpt=""

  latest_run=$(ls -dt "${root}"/*-"${training_name}"* 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest_run}" ]]; then
    echo ""
    return 0
  fi

  latest_ckpt=$(ls -1 "${latest_run}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  echo "${latest_ckpt}"
}

CKPT=""
if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == https://wandb.ai/* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    CKPT="$1"
    shift
  fi
fi

if [[ -n "${CKPT}" ]]; then
  if [[ "${CKPT}" == https://wandb.ai/*/runs/* ]]; then
    LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_run_url "${CKPT}")"
    if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
      CKPT="${LOCAL_WANDB_CKPT}"
      echo "[INFO] Resolved wandb run URL to local checkpoint: ${CKPT}"
    fi
  fi
  CKPT="$(normalize_checkpoint_ref "${CKPT}")"
fi

if [[ -z "${CKPT}" ]]; then
  if [[ "${MODE}" == "mocap" ]]; then
    if [[ "${MOCAP_CHECKPOINT_DEFAULT}" == wandb://* ]] || [[ -f "${MOCAP_CHECKPOINT_DEFAULT}" ]]; then
      CKPT="${MOCAP_CHECKPOINT_DEFAULT}"
    else
      CKPT="$(find_latest_ckpt "${LOG_ROOT}" "${MOCAP_TRAINING_NAME_DEFAULT}")"
    fi
  else
    if [[ "${DEPTH_CHECKPOINT_DEFAULT}" == wandb://* ]] || [[ -f "${DEPTH_CHECKPOINT_DEFAULT}" ]]; then
      CKPT="${DEPTH_CHECKPOINT_DEFAULT}"
    else
      CKPT="$(find_latest_ckpt "${LOG_ROOT}" "${DEPTH_TRAINING_NAME_DEFAULT}")"
    fi
  fi
fi

if [[ -z "${CKPT}" ]]; then
  echo "[ERROR] Could not auto-resolve checkpoint. Please pass checkpoint path explicitly." >&2
  exit 1
fi
if [[ "${CKPT}" != wandb://* ]] && [[ ! -f "${CKPT}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT}" >&2
  exit 1
fi

INFER_DATASET_DEFAULT="omomo"
INFER_DATASET=${INFER_DATASET:-${INFER_DATASET_DEFAULT}}
INFER_DATASET=$(echo "${INFER_DATASET}" | tr '[:upper:]' '[:lower:]' | tr -d '[][:space:]')
case "${INFER_DATASET}" in
  omomo|behave|mixed) ;;
  *)
    echo "[ERROR] INFER_DATASET must be one of: omomo|behave|mixed. Got: ${INFER_DATASET}" >&2
    exit 2
    ;;
esac

DEFAULT_OMOMO_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
DEFAULT_BEHAVE_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/behave_sq_carry"
DEFAULT_MIXED_MOTION_DIR="${SCRIPT_DIR}/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml"
DEFAULT_OMOMO_URDF="${SCRIPT_DIR}/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
DEFAULT_BEHAVE_MAP_FILE="${DEFAULT_BEHAVE_MOTION_DIR}/_clip_object_urdf_map.json"
DEFAULT_MIXED_MAP_FILE="${DEFAULT_MIXED_MOTION_DIR}/_clip_object_urdf_map.json"

if [[ -z "${MOTION_DIR+x}" ]]; then
  case "${INFER_DATASET}" in
    omomo) MOTION_DIR="${DEFAULT_OMOMO_MOTION_DIR}" ;;
    behave) MOTION_DIR="${DEFAULT_BEHAVE_MOTION_DIR}" ;;
    mixed) MOTION_DIR="${DEFAULT_MIXED_MOTION_DIR}" ;;
  esac
fi
if [[ -z "${OBJECT_URDF+x}" ]]; then
  case "${INFER_DATASET}" in
    omomo) OBJECT_URDF="${DEFAULT_OMOMO_URDF}" ;;
    behave) OBJECT_URDF="${DEFAULT_BEHAVE_MAP_FILE}" ;;
    mixed) OBJECT_URDF="${DEFAULT_MIXED_MAP_FILE}" ;;
  esac
fi

GEOMETRY_DIR=${GEOMETRY_DIR:-}

PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-False}
NUM_ENVS=${NUM_ENVS:-1}
HEADLESS_RAW=${HEADLESS:-True}
HEADLESS_NORM=$(echo "${HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')
case "${HEADLESS_NORM}" in
  1|true|yes|on)
    HEADLESS_FLAG=True
    export HEADLESS=1
    ;;
  0|false|no|off|"")
    HEADLESS_FLAG=False
    export HEADLESS=0
    ;;
  *)
    echo "[ERROR] HEADLESS must be one of: 0/1/true/false/yes/no/on/off. Got: ${HEADLESS_RAW}" >&2
    exit 2
    ;;
esac
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_RECENTER=${VISER_RECENTER:-True}

START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-1.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
RESET_NOISE_SCALE=${RESET_NOISE_SCALE:-0.0}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}

DISABLE_RANDOMIZATION=${DISABLE_RANDOMIZATION:-True}
MAX_EPISODE_LENGTH_S=${MAX_EPISODE_LENGTH_S:-1000000}

IMAGE_WIDTH=${IMAGE_WIDTH:-106}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-60}
CAMERA_NEAR=${CAMERA_NEAR:-0.001}
CAMERA_FAR=${CAMERA_FAR:-3.0}
CAMERA_MAX_DISTANCE=${CAMERA_MAX_DISTANCE:-3.0}
MOCAP_PERCEPTION_PRESET=${MOCAP_PERCEPTION_PRESET:-checkpoint}
DEPTH_PERCEPTION_PRESET=${DEPTH_PERCEPTION_PRESET:-checkpoint}

if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -f "${OBJECT_URDF}" ]]; then
  echo "[ERROR] OBJECT_URDF not found: ${OBJECT_URDF}" >&2
  exit 1
fi
if [[ -n "${GEOMETRY_DIR}" && ! -e "${GEOMETRY_DIR}" ]]; then
  echo "[ERROR] GEOMETRY_DIR not found: ${GEOMETRY_DIR}" >&2
  exit 1
fi

# Viser GUI defaults aligned with VideoMimic-style manual + clip control.
export VISER_ENABLE_CLIP_GUI=${VISER_ENABLE_CLIP_GUI:-1}
export VISER_ENABLE_MANUAL_GUI=${VISER_ENABLE_MANUAL_GUI:-1}
export VISER_MANUAL_CONTROL_DEFAULT=${VISER_MANUAL_CONTROL_DEFAULT:-1}
export VISER_FORCE_MANUAL_CONTROL=${VISER_FORCE_MANUAL_CONTROL:-0}
export VISER_SHOW_TARGET_KEYPOINTS=${VISER_SHOW_TARGET_KEYPOINTS:-1}
export VISER_START_PAUSED=${VISER_START_PAUSED:-0}
export VISER_MANUAL_USE_HW_JOYSTICK=${VISER_MANUAL_USE_HW_JOYSTICK:-0}
export VISER_MANUAL_HW_DEADZONE=${VISER_MANUAL_HW_DEADZONE:-0.08}
export VISER_CLIP_LOCK_DEFAULT=${VISER_CLIP_LOCK_DEFAULT:-1}

# VideoMimic-style default:
# - manual control comes from Viser GUI toggles.
# - hardware joystick is optional and should not crash startup when no device exists.
USE_HW_JOYSTICK_BRIDGE=${USE_HW_JOYSTICK_BRIDGE:-False}
export VISER_MANUAL_HW_BACKEND=${VISER_MANUAL_HW_BACKEND:-auto}
export VISER_MANUAL_HW_DEVICE=${VISER_MANUAL_HW_DEVICE:-0}
export VISER_MANUAL_HW_TYPE=${VISER_MANUAL_HW_TYPE:-xbox}
# Keep noisy third-party debug logs off by default.
export LOGURU_LEVEL=${LOGURU_LEVEL:-WARNING}
export PY_LOG_LEVEL=${PY_LOG_LEVEL:-WARNING}

EXTRA_ARGS=("$@")

cmd=(
  python -m holosoma.visualize physics
  --checkpoint "${CKPT}"
  --motion-dir "${MOTION_DIR}"
  --num-envs "${NUM_ENVS}"
  --headless "${HEADLESS_FLAG}"
  --pair-terrain-with-motion "${PAIR_TERRAIN_WITH_MOTION}"
  --viser-port "${VISER_PORT}"
  --viser-env-id "${VISER_ENV_ID}"
  --viser-update-hz "${VISER_UPDATE_HZ}"
  --viser-recenter "${VISER_RECENTER}"
  --simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
  --robot.object.enabled True
  --robot.object.object_urdf_path "${OBJECT_URDF}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob "${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale "${RESET_NOISE_SCALE}"
  --algo.config.distill.bc_loss_coef 0.0
  --algo.config.distill.loss_coef 0.0
  --algo.config.distill.switch_to_rl_after -1
  --algo.config.distill.take_teacher_actions False
  --algo.config.distill.teacher_action_mix_ratio 0.0
  # Hard-disable distillation teacher loading during inference.
  --algo.config.distill.enabled False
  --algo.config.distill.mode mse
  --algo.config.distill.ppo_start_epoch -1
  --algo.config.distill.dagger_end_epoch -1
)

if [[ -n "${GEOMETRY_DIR}" ]]; then
  cmd+=(--geometry-dir "${GEOMETRY_DIR}")
fi

if [[ "${DISABLE_RANDOMIZATION}" == "True" || "${DISABLE_RANDOMIZATION}" == "true" ]]; then
  cmd+=(
    --randomization.setup_terms.push_randomizer_state.params.enabled False
    --randomization.reset_terms.randomize_push_schedule.params.enabled False
    --randomization.step_terms.apply_pushes.params.enabled False
    --randomization.setup_terms.actuator_randomizer_state.params.enable_pd_gain False
    --randomization.setup_terms.actuator_randomizer_state.params.enable_rfi_lim False
    --randomization.setup_terms.setup_action_delay_buffers.params.enabled False
    --randomization.reset_terms.randomize_action_delay.params.enabled False
    --randomization.setup_terms.randomize_robot_rigid_body_material_startup.params.enabled False
    --randomization.setup_terms.randomize_base_com_startup.params.enabled False
    --randomization.setup_terms.setup_dof_pos_bias.params.enabled False
    --randomization.reset_terms.randomize_dof_state.params.randomize_dof_pos_bias False
    --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled False
    --randomization.reset_terms.randomize_camera_raycast.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_material_startup.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_mass_startup.params.enabled False
    --randomization.setup_terms.randomize_object_rigid_body_inertia_startup.params.enabled False
  )
fi

if [[ "${USE_HW_JOYSTICK_BRIDGE}" == "True" || "${USE_HW_JOYSTICK_BRIDGE}" == "true" || "${VISER_MANUAL_HW_BACKEND:-auto}" == "bridge" ]]; then
  cmd+=(
    --simulator.config.bridge.enabled True
    --simulator.config.bridge.use_joystick True
    --simulator.config.bridge.joystick_device "${VISER_MANUAL_HW_DEVICE}"
    --simulator.config.bridge.joystick_type "${VISER_MANUAL_HW_TYPE}"
  )
fi

if [[ "${MODE}" == "mocap" ]]; then
  case "$(echo "${MOCAP_PERCEPTION_PRESET}" | tr '[:upper:]' '[:lower:]')" in
    checkpoint|auto|"")
      # Keep checkpoint-saved perception config (required by some distill checkpoints).
      ;;
    none)
      cmd+=(perception:none)
      ;;
    heightmap)
      cmd+=(perception:heightmap)
      ;;
    *)
      echo "[ERROR] MOCAP_PERCEPTION_PRESET must be one of: checkpoint|none|heightmap. Got: ${MOCAP_PERCEPTION_PRESET}" >&2
      exit 2
      ;;
  esac
else
  # Depth branch defaults to checkpoint-native perception so actor dims match checkpoint.
  export VISER_PERCEPTION_IMAGE_MODE=${VISER_PERCEPTION_IMAGE_MODE:-depth}
  export VISER_SHOW_PERCEPTION_FRUSTUM=${VISER_SHOW_PERCEPTION_FRUSTUM:-1}
  case "$(echo "${DEPTH_PERCEPTION_PRESET}" | tr '[:upper:]' '[:lower:]')" in
    checkpoint|auto|"")
      # Keep checkpoint-saved perception config.
      ;;
    d435i)
      cmd+=(
        perception:camera-depth-d435i
        --perception.camera_width "${IMAGE_WIDTH}"
        --perception.camera_height "${IMAGE_HEIGHT}"
        --perception.camera_near "${CAMERA_NEAR}"
        --perception.camera_far "${CAMERA_FAR}"
        --perception.max_distance "${CAMERA_MAX_DISTANCE}"
      )
      ;;
    *)
      echo "[ERROR] DEPTH_PERCEPTION_PRESET must be one of: checkpoint|d435i. Got: ${DEPTH_PERCEPTION_PRESET}" >&2
      exit 2
      ;;
  esac
fi

if [[ "${#EXTRA_ARGS[@]}" -gt 0 ]]; then
  cmd+=("${EXTRA_ARGS[@]}")
fi

echo "[INFO] mode_input=${MODE_INPUT} runtime_mode=${MODE}"
echo "[INFO] infer_dataset=${INFER_DATASET}"
echo "[INFO] checkpoint=${CKPT}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] object_urdf=${OBJECT_URDF}"
if [[ -n "${GEOMETRY_DIR}" ]]; then
  echo "[INFO] geometry_dir=${GEOMETRY_DIR}"
fi
echo "[INFO] headless=${HEADLESS_FLAG} (env HEADLESS=${HEADLESS})"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] manual_gui=${VISER_ENABLE_MANUAL_GUI} clip_gui=${VISER_ENABLE_CLIP_GUI}"
echo "[INFO] manual_control_default=${VISER_MANUAL_CONTROL_DEFAULT} force_manual=${VISER_FORCE_MANUAL_CONTROL}"
echo "[INFO] hw_joystick=${VISER_MANUAL_USE_HW_JOYSTICK}"
echo "[INFO] hw_backend=${VISER_MANUAL_HW_BACKEND:-auto} bridge_joystick=${USE_HW_JOYSTICK_BRIDGE}"
if [[ "${MODE}" == "mocap" ]]; then
  echo "[INFO] mocap_perception_preset=${MOCAP_PERCEPTION_PRESET}"
else
  echo "[INFO] depth_perception_preset=${DEPTH_PERCEPTION_PRESET}"
fi
echo "[INFO] Viser controls:"
echo "  1) Open 'Manual Control' and enable 'Enable Manual Root Command'."
echo "  2) Use Move +X/-X/+Y/-Y and Yaw +/- to command root trajectory."
echo "  3) Tune 'XY Command (m)' and 'Yaw Command (rad)' sliders."
echo "  4) Use 'Clip Playback' to select clip/start frame and click 'Apply Clip'."
echo "  5) Use 'Advanced > Simulation Control' for Play/Step/Reset."
if command -v hostname >/dev/null 2>&1; then
  HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}' || true)"
  if [[ -n "${HOST_IP}" ]]; then
    echo "[INFO] Remote URL: http://${HOST_IP}:${VISER_PORT}"
    echo "[INFO] SSH tunnel example: ssh -N -L ${VISER_PORT}:localhost:${VISER_PORT} <user>@<host>"
  fi
fi

"${cmd[@]}"
