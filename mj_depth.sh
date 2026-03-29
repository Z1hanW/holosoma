#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_MOTION_FILE="${DEFAULT_MOTION_FILE:-$ROOT_DIR/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz}"
LOG_ROOT="${LOG_ROOT:-/data/logs_new/WholeBodyTracking}"
LOG_ROOTS_DEFAULT="${LOG_ROOTS_DEFAULT:-${LOG_ROOT}:/data/logs_new/boxer}"
DEPTH_TRAINING_NAME_DEFAULT="${DEPTH_TRAINING_NAME_DEFAULT:-g1_29dof_wbt_w_object_distill_box_perception_access_to_depth}"
DEPTH_CHECKPOINT_DEFAULT="${DEPTH_CHECKPOINT_DEFAULT:-wandb://zihanw22/boxer/0z2aggr2/model_05000.pt}"
DEFAULT_SIM_PY="/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python"
if [[ -z "${PYTHON_BIN+x}" || -z "${PYTHON_BIN}" ]]; then
  if [[ -x "${DEFAULT_SIM_PY}" ]]; then
    PYTHON_BIN="${DEFAULT_SIM_PY}"
  else
    PYTHON_BIN="python3"
  fi
fi
if [[ "${PYTHON_BIN}" == "${DEFAULT_SIM_PY}" ]]; then
  export PATH="$(dirname "${DEFAULT_SIM_PY}"):${PATH}"
fi

usage() {
  cat <<EOF
Usage:
  bash mj_depth.sh [--viewer sim_state|mjviser] [--depth-source rendered|warp] [--default-pose-init|--no-default-pose-init] [--motion-file clip.npz] [checkpoint.pt|model.onnx|wandb://...|https://wandb.ai/.../runs/...]

Purpose:
  Launch the depth distill box-carry policy through split MuJoCo sim2sim:
  simulator + policy in separate processes, with authoritative sim-state viewed in viser.

Defaults:
  motion       = ${DEFAULT_MOTION_FILE}
  depth_source = warp
  viewer       = sim_state
  checkpoint   = ${DEPTH_CHECKPOINT_DEFAULT}
                 (if that run exists locally with ONNX, prefer it; otherwise fall back to latest local depth distill ONNX under ${LOG_ROOTS_DEFAULT})

Useful env vars:
  VISER_PORT                Optional fixed sim-state viewer port.
  HEADLESS                  True/False; forwarded as split training_headless.
  TRAINING_HEADLESS         Explicit override for split training_headless.
  MUJOCO_BACKEND            classic|warp
  SIM_PERCEPTION_PORT       Default: 5658
  DEPTH_CHECKPOINT_DEFAULT  Fallback checkpoint/model ref.
  DEPTH_TRAINING_NAME_DEFAULT

Examples:
  bash mj_depth.sh
  bash mj_depth.sh --motion-file /abs/path/clip.npz
  bash mj_depth.sh --depth-source rendered /abs/path/model_01000.onnx
  VISER_PORT=18080 bash mj_depth.sh
EOF
}

resolve_existing_path() {
  local raw="$1"
  if [[ -f "$raw" ]]; then
    python3 - <<'PY' "$raw"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
    return 0
  fi

  local candidate="$ROOT_DIR/$raw"
  if [[ -f "$candidate" ]]; then
    python3 - <<'PY' "$candidate"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
    return 0
  fi

  return 1
}

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

parse_wandb_uri() {
  local ref="$1"
  if [[ "${ref}" != wandb://* ]]; then
    return 1
  fi

  local trimmed="${ref#wandb://}"
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  IFS='/' read -r -a parts <<< "${trimmed}"
  if [[ "${#parts[@]}" -lt 4 ]]; then
    return 1
  fi

  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[2]}"
  explicit_file="${trimmed#${entity}/${project}/${run_id}/}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" || -z "${explicit_file}" ]]; then
    return 1
  fi

  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

find_local_run_log_dir() {
  local run_id="$1"
  local wandb_run_dir=""
  wandb_run_dir=$(find /data/logs_new -maxdepth 8 -type d -name "run-*-${run_id}" 2>/dev/null | head -n 1 || true)
  if [[ -z "${wandb_run_dir}" ]]; then
    echo ""
    return 0
  fi
  dirname "$(dirname "$(dirname "${wandb_run_dir}")")"
}

prefer_existing_model_file() {
  local candidate="$1"
  if [[ -z "${candidate}" ]]; then
    echo ""
    return 0
  fi
  if [[ -f "${candidate}" ]]; then
    if [[ "${candidate}" == *.onnx ]]; then
      echo "${candidate}"
      return 0
    fi
    if [[ "${candidate}" == *.pt ]]; then
      local sibling="${candidate%.pt}.onnx"
      if [[ -f "${sibling}" ]]; then
        echo "${sibling}"
        return 0
      fi
      echo ""
      return 0
    fi
  fi
  echo ""
}

resolve_local_model_from_run_ref() {
  local run_id="$1"
  local explicit_file="$2"
  local run_log_dir=""
  run_log_dir="$(find_local_run_log_dir "${run_id}")"
  if [[ -z "${run_log_dir}" ]]; then
    echo ""
    return 0
  fi

  if [[ -n "${explicit_file}" ]]; then
    local preferred=""
    preferred="$(prefer_existing_model_file "${run_log_dir}/${explicit_file}")"
    if [[ -n "${preferred}" ]]; then
      echo "${preferred}"
      return 0
    fi
  fi

  local latest_onnx=""
  latest_onnx=$(ls -1 "${run_log_dir}"/model_*.onnx 2>/dev/null | sort -V | tail -n 1 || true)
  if [[ -n "${latest_onnx}" ]]; then
    echo "${latest_onnx}"
    return 0
  fi

  local latest_pt=""
  latest_pt=$(ls -1 "${run_log_dir}"/model_*.pt 2>/dev/null | sort -V | tail -n 1 || true)
  echo "${latest_pt}"
}

find_latest_local_model() {
  local training_name="$1"
  local roots_raw="${LOG_ROOTS:-$LOG_ROOTS_DEFAULT}"
  local latest_onnx=""
  local latest_ts="-1"
  local roots=()
  local root=""
  local run_dir=""
  local candidate=""
  local candidate_ts=""

  IFS=':' read -r -a roots <<< "${roots_raw}"
  for root in "${roots[@]}"; do
    [[ -d "${root}" ]] || continue
    while IFS= read -r -d '' run_dir; do
      candidate=$(ls -1 "${run_dir}"/model_*.onnx 2>/dev/null | sort -V | tail -n 1 || true)
      [[ -n "${candidate}" ]] || continue
      candidate_ts="$(stat -c '%Y' "${candidate}" 2>/dev/null || echo 0)"
      if [[ "${candidate_ts}" -gt "${latest_ts}" ]]; then
        latest_ts="${candidate_ts}"
        latest_onnx="${candidate}"
      fi
    done < <(find "${root}" -maxdepth 1 -mindepth 1 -type d -name "*-${training_name}*" -print0 2>/dev/null)
  done

  echo "${latest_onnx}"
}

resolve_model_input() {
  local raw="$1"
  local resolved=""

  if [[ -n "${raw}" ]]; then
    resolved="$(resolve_existing_path "${raw}" || true)"
    if [[ -n "${resolved}" ]]; then
      echo "${resolved}"
      return 0
    fi

    if [[ "${raw}" == https://wandb.ai/*/runs/* ]]; then
      local parsed=""
      parsed="$(parse_wandb_run_url "${raw}" || true)"
      if [[ -n "${parsed}" ]]; then
        IFS=$'\t' read -r _entity _project run_id explicit_file <<< "${parsed}"
        resolved="$(resolve_local_model_from_run_ref "${run_id}" "${explicit_file}")"
        if [[ -n "${resolved}" ]]; then
          echo "${resolved}"
          return 0
        fi
      fi
    fi

    if [[ "${raw}" == wandb://* ]]; then
      local parsed_uri=""
      parsed_uri="$(parse_wandb_uri "${raw}" || true)"
      if [[ -n "${parsed_uri}" ]]; then
        IFS=$'\t' read -r _entity _project run_id explicit_file <<< "${parsed_uri}"
        resolved="$(resolve_local_model_from_run_ref "${run_id}" "${explicit_file}")"
        if [[ -n "${resolved}" ]]; then
          echo "${resolved}"
          return 0
        fi
      fi
    fi

    echo ""
    return 0
  fi

  if [[ -n "${DEPTH_CHECKPOINT_DEFAULT}" ]]; then
    resolved="$(resolve_model_input "${DEPTH_CHECKPOINT_DEFAULT}")"
    if [[ -n "${resolved}" ]]; then
      echo "${resolved}"
      return 0
    fi
  fi

  resolved="$(find_latest_local_model "${DEPTH_TRAINING_NAME_DEFAULT}")"
  if [[ -n "${resolved}" ]]; then
    echo "${resolved}"
    return 0
  fi

  echo ""
}

normalize_bool_flag() {
  local raw="$1"
  local norm
  norm="$(echo "${raw}" | tr '[:upper:]' '[:lower:]')"
  case "${norm}" in
    1|true|yes|on) echo "True" ;;
    0|false|no|off|"") echo "False" ;;
    *)
      echo "[ERROR] Boolean value must be one of: 0/1/true/false/yes/no/on/off. Got: ${raw}" >&2
      exit 2
      ;;
  esac
}

if [[ "${HOLOSOMA_MJ_DEPTH_INTERNAL_CORE:-0}" != "1" ]]; then
  export DEPTH_CHECKPOINT_DEFAULT="${DEPTH_CHECKPOINT_DEFAULT:-wandb://zihanw22/boxer/0z2aggr2/model_05000.pt}"
  export INFER_DATASET="${INFER_DATASET:-omomo}"
  export NUM_ENVS="${NUM_ENVS:-1}"
  export HEADLESS="${HEADLESS:-True}"
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export DISABLE_RANDOMIZATION="${DISABLE_RANDOMIZATION:-True}"
  export VISER_ENABLE_CLIP_GUI="${VISER_ENABLE_CLIP_GUI:-0}"
  export VISER_ENABLE_MANUAL_GUI="${VISER_ENABLE_MANUAL_GUI:-1}"
  export VISER_MANUAL_CONTROL_DEFAULT="${VISER_MANUAL_CONTROL_DEFAULT:-0}"
  export VISER_ENABLE_OBJECT_RESET_OVERRIDE="${VISER_ENABLE_OBJECT_RESET_OVERRIDE:-0}"
  export VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS:-1}"
  export VISER_START_PAUSED="${VISER_START_PAUSED:-0}"
  export VISER_PERCEPTION_DEPTH_SOURCE="${VISER_PERCEPTION_DEPTH_SOURCE:-obs}"
  export VISER_PERCEPTION_FLIP_VERTICAL="${VISER_PERCEPTION_FLIP_VERTICAL:-0}"
  export START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-1.0}"
  export FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}"
  export POLICY_CONTROL_PORT="${POLICY_CONTROL_PORT:-5660}"

  MUJOCO_BACKEND_NORM="$(echo "${MUJOCO_BACKEND:-classic}" | tr '[:upper:]' '[:lower:]')"
  case "${MUJOCO_BACKEND_NORM}" in
    classic|warp) export MUJOCO_BACKEND="${MUJOCO_BACKEND_NORM}" ;;
    *)
      echo "[ERROR] MUJOCO_BACKEND must be one of: classic|warp. Got: ${MUJOCO_BACKEND:-}" >&2
      exit 2
      ;;
  esac

  DEPTH_SOURCE_RAW="${MJ_DEPTH_CAMERA_SOURCE:-warp}"
  VIEWER_KIND="${MJ_VIEWER:-sim_state}"
  DEFAULT_POSE_INIT_OVERRIDE=""
  MOTION_FILE_RAW="${MOTION_FILE:-$DEFAULT_MOTION_FILE}"
  MODEL_REF=""
  HELP_REQUESTED=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help|help)
        HELP_REQUESTED=1
        shift
        ;;
      --depth-source)
        if [[ $# -lt 2 ]]; then
          echo "[ERROR] --depth-source requires a value: rendered|warp" >&2
          exit 2
        fi
        DEPTH_SOURCE_RAW="$2"
        shift 2
        ;;
      --depth-source=*)
        DEPTH_SOURCE_RAW="${1#*=}"
        shift
        ;;
      --viewer)
        if [[ $# -lt 2 ]]; then
          echo "[ERROR] --viewer requires a value: sim_state|mjviser" >&2
          exit 2
        fi
        VIEWER_KIND="$2"
        shift 2
        ;;
      --viewer=*)
        VIEWER_KIND="${1#*=}"
        shift
        ;;
      --default-pose-init)
        DEFAULT_POSE_INIT_OVERRIDE="1"
        shift
        ;;
      --no-default-pose-init)
        DEFAULT_POSE_INIT_OVERRIDE="0"
        shift
        ;;
      --motion-file)
        if [[ $# -lt 2 ]]; then
          echo "[ERROR] --motion-file requires a .npz path" >&2
          exit 2
        fi
        MOTION_FILE_RAW="$2"
        shift 2
        ;;
      --motion-file=*)
        MOTION_FILE_RAW="${1#*=}"
        shift
        ;;
      wandb://*|https://wandb.ai/*|/*|./*|../*|*.pt|*.onnx)
        if [[ -n "${MODEL_REF}" ]]; then
          echo "[ERROR] Multiple model refs provided: ${MODEL_REF} and $1" >&2
          exit 2
        fi
        MODEL_REF="$1"
        shift
        ;;
      *)
        echo "[ERROR] Unsupported split mj_depth argument: $1" >&2
        exit 2
        ;;
    esac
  done

  DEPTH_SOURCE="$(echo "${DEPTH_SOURCE_RAW}" | tr '[:upper:]' '[:lower:]')"
  case "${DEPTH_SOURCE}" in
    rendered|warp) ;;
    *)
      echo "[ERROR] depth source must be one of: rendered|warp. Got: ${DEPTH_SOURCE_RAW}" >&2
      exit 2
      ;;
  esac

  VIEWER_KIND="$(echo "${VIEWER_KIND}" | tr '[:upper:]' '[:lower:]')"
  case "${VIEWER_KIND}" in
    sim_state|mjviser) ;;
    *)
      echo "[ERROR] viewer must be one of: sim_state|mjviser. Got: ${VIEWER_KIND}" >&2
      exit 2
      ;;
  esac

  if [[ "${HELP_REQUESTED}" == "1" ]]; then
    usage
    exit 0
  fi

  if [[ -n "${DEFAULT_POSE_INIT_OVERRIDE}" ]]; then
    export HOLOSOMA_DEFAULT_POSE_INIT="${DEFAULT_POSE_INIT_OVERRIDE}"
    export HOLOSOMA_RESET_TO_DEFAULT_POSE="${DEFAULT_POSE_INIT_OVERRIDE}"
    if [[ "${DEFAULT_POSE_INIT_OVERRIDE}" == "1" ]]; then
      export SIM_MOTION_INIT_MODE="training_default_pose"
    else
      export SIM_MOTION_INIT_MODE="raw_motion"
    fi
  fi

  MOTION_FILE_RESOLVED="$(resolve_existing_path "${MOTION_FILE_RAW}" || true)"
  if [[ -z "${MOTION_FILE_RESOLVED}" ]]; then
    echo "[ERROR] motion file not found: ${MOTION_FILE_RAW}" >&2
    exit 1
  fi

  MODEL_INPUT_RESOLVED="$(resolve_model_input "${MODEL_REF}")"
  if [[ -z "${MODEL_INPUT_RESOLVED}" ]]; then
    echo "[ERROR] Could not resolve a usable local depth ONNX. Pass a local .onnx path, or a .pt with sibling .onnx, or make sure a matching run exists under ${LOG_ROOTS_DEFAULT}." >&2
    exit 1
  fi

  TRAINING_HEADLESS_FLAG="$(normalize_bool_flag "${TRAINING_HEADLESS:-${HEADLESS:-True}}")"

  export MJ_DEPTH_CAMERA_SOURCE="${DEPTH_SOURCE}"
  export MJ_DEPTH_MOTION_FILE="${MOTION_FILE_RESOLVED}"
  export MJ_DEPTH_MODEL_INPUT="${MODEL_INPUT_RESOLVED}"
  export HOLOSOMA_MJ_DEPTH_INTERNAL_CORE=1
  export PYTHONPATH="$ROOT_DIR/src/holosoma:$ROOT_DIR/src/holosoma_inference${PYTHONPATH:+:$PYTHONPATH}"
  VIEWER_SCRIPT="$ROOT_DIR/src/holosoma/holosoma/viser_mujoco_sim_state.py"
  if [[ "${VIEWER_KIND}" == "mjviser" ]]; then
    VIEWER_SCRIPT="$ROOT_DIR/src/holosoma/holosoma/mjviser_mujoco_sim_state.py"
  fi

  VIEWER_CMD=(
    "$PYTHON_BIN" "$VIEWER_SCRIPT"
    --launch-rollout
    --run-script "$ROOT_DIR/mj_depth.sh"
  )
  if [[ "${TRAINING_HEADLESS_FLAG}" == "True" ]]; then
    VIEWER_CMD+=(--training-headless)
  else
    VIEWER_CMD+=(--no-training-headless)
  fi
  if [[ -n "${VISER_PORT:-}" ]]; then
    VIEWER_CMD+=(--port "${VISER_PORT}")
  fi

  echo "[INFO] launcher=mj_depth.sh"
  echo "[INFO] mode=split_sim2sim"
  echo "[INFO] checkpoint_default=${DEPTH_CHECKPOINT_DEFAULT}"
  echo "[INFO] model_input=${MODEL_INPUT_RESOLVED}"
  echo "[INFO] motion_file=${MOTION_FILE_RESOLVED}"
  echo "[INFO] depth_source=${DEPTH_SOURCE}"
  echo "[INFO] viewer=${VIEWER_KIND}"
  echo "[INFO] training_headless=${TRAINING_HEADLESS_FLAG}"
  echo "[INFO] mujoco_backend=${MUJOCO_BACKEND}"
  echo "[INFO] manual_gui=${VISER_ENABLE_MANUAL_GUI}"
  echo "[INFO] manual_control_default=${VISER_MANUAL_CONTROL_DEFAULT}"

  exec "${VIEWER_CMD[@]}"
fi

MODEL_INPUT="${MJ_DEPTH_MODEL_INPUT:-}"
MOTION_FILE="${MJ_DEPTH_MOTION_FILE:-$DEFAULT_MOTION_FILE}"
DEPTH_SOURCE="$(echo "${MJ_DEPTH_CAMERA_SOURCE:-warp}" | tr '[:upper:]' '[:lower:]')"

if [[ -z "${MODEL_INPUT}" ]]; then
  echo "[ERROR] MJ_DEPTH_MODEL_INPUT is not set for internal split rollout launch." >&2
  exit 1
fi

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "[ERROR] MJ_DEPTH_MOTION_FILE is not a file: ${MOTION_FILE}" >&2
  exit 1
fi

case "${DEPTH_SOURCE}" in
  rendered)
    export SIM_PERCEPTION_CAMERA_SOURCE_OVERRIDE="rendered"
    ;;
  warp)
    export SIM_PERCEPTION_CAMERA_SOURCE_OVERRIDE="far_tracking_warp"
    ;;
  *)
    echo "[ERROR] Unsupported MJ_DEPTH_CAMERA_SOURCE=${DEPTH_SOURCE}" >&2
    exit 2
    ;;
esac

export POLICY_DEFER_UNTIL_VALID_STATE="${POLICY_DEFER_UNTIL_VALID_STATE:-1}"
export HOLOSOMA_MJ_TRACK_INTERNAL_CORE=1

echo "[INFO] split_rollout=mj_depth.sh -> mj_track.sh"
echo "[INFO] model_input=${MODEL_INPUT}"
echo "[INFO] motion_file=${MOTION_FILE}"
echo "[INFO] split_perception_camera_source=${SIM_PERCEPTION_CAMERA_SOURCE_OVERRIDE}"

exec bash "$ROOT_DIR/mj_track.sh" "$MOTION_FILE" "$MODEL_INPUT"
