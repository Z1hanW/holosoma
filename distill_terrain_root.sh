#!/usr/bin/env bash
set -euo pipefail

# Distill terrain tracking teacher -> sparse-root student.
#
# Student actor observation:
# - actor_obs_root: sparse root command [rel_xy(2), rel_yaw(1)]
# - actor_obs_proprio: base_lin_vel, base_ang_vel, dof_pos, dof_vel; actor_obs_actions carries single-step action
# - optional perception_obs: injected automatically when PERCEPTION_PRESET != none
#
# Teacher observation defaults:
# - actor_obs
# - actor_obs_target
#
# Terrain comes from the same motion/geometry pairing used by terrain tracking.

usage() {
  cat <<'EOF'
Usage:
  bash distill_terrain_root.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra train_agent.py args...]

Optional env vars:
  TEACHER_CHECKPOINT / CKPT
  WANDB_MODEL_FILE
  ALLOW_TEACHER_PERCEPTION_CHECKPOINT
                               (default: 0; set 1 only to bypass tracking-teacher validation)
  EXP                          (default: g1-29dof-wbt-terrain-distill-sparse-root-cmd)
  MOTION_DIR                   (default: /data/terrain/___crisp_clean_motion)
  OBJ_SOURCE                   (default: /data/terrain/___crisp_clean_geometry)
  OBJ_META_PATH                (optional metadata override)
  NUM_ROWS / NUM_COLS          (optional terrain layout override)
  PERCEPTION_PRESET            (default: camera_depth_d435i_17x17; options: none|camera_depth_d435i|camera_depth_d435i_17x17|heightmap)
  TEACHER_OBS_KEYS             (default: actor_obs,actor_obs_target)
  NUM_ENVS                     (default: NPROC * PER_GPU_ENVS)
  PER_GPU_ENVS                 (default: 8192)
  CUDA_VISIBLE_DEVICES         (default: 0,1,2,3)
  TRAINING_PROJECT             (default: terrain-aware)
  RUN_NAME                     (default: g1_terrain_distill_root_access_to_depth)
  TRAINING_NAME                (default: g1_29dof_wbt_terrain_distill_root_access_to_depth)
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SIM_ENV_BIN=/home/ubuntu/miniconda3/envs/sim/bin
if ! command -v torchrun >/dev/null 2>&1 && [[ -x "${SIM_ENV_BIN}/torchrun" ]]; then
  export PATH="${SIM_ENV_BIN}:${PATH}"
fi
if [[ -x "${SIM_ENV_BIN}/python" ]]; then
  DEFAULT_PYTHON_BIN="${SIM_ENV_BIN}/python"
else
  DEFAULT_PYTHON_BIN="$(command -v python)"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON_BIN}}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
  usage
  exit 0
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

resolve_remote_wandb_checkpoint_name() {
  local entity="$1"
  local project="$2"
  local run_id="$3"
  "${PYTHON_BIN}" - "${entity}" "${project}" "${run_id}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
sanitized_sys_path = []
for entry in sys.path:
    if entry in {"", "."}:
        continue
    try:
        if Path(entry).resolve() == repo_root:
            continue
    except Exception:
        pass
    sanitized_sys_path.append(entry)
sys.path = sanitized_sys_path

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id = sys.argv[1:4]
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
pattern = re.compile(r"^model_(\d+)\.pt$")
latest_step = -1
latest_name = ""
for file_obj in run.files():
    name = getattr(file_obj, "name", "")
    match = pattern.match(name)
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
  if [[ "${ref}" != https://wandb.ai/*/runs/* ]]; then
    echo "${ref}"
    return 0
  fi

  local parsed=""
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  local model_file="${WANDB_MODEL_FILE:-}"

  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo "${ref}"
    return 0
  fi

  IFS=$'\t' read -r entity project run_id explicit_file <<< "${parsed}"
  if [[ -n "${explicit_file}" ]]; then
    model_file="${explicit_file}"
  elif [[ -z "${model_file}" ]]; then
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved wandb run URL to latest remote checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine checkpoint for W&B run URL: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL or set WANDB_MODEL_FILE." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

resolve_local_checkpoint_from_run_url() {
  local ref="$1"
  local preferred_model_file="${2:-}"
  local parsed=""
  local run_id=""
  local explicit_file=""
  local target_model_file=""
  local latest_ckpt=""
  parsed="$(parse_wandb_run_url "${ref}" || true)"
  if [[ -z "${parsed}" ]]; then
    echo ""
    return 0
  fi
  IFS=$'\t' read -r _entity _project run_id explicit_file <<< "${parsed}"
  target_model_file="${explicit_file:-${preferred_model_file}}"

  while IFS= read -r run_dir; do
    [[ -z "${run_dir}" ]] && continue
    if [[ -n "${target_model_file}" && -f "${run_dir}/${target_model_file}" ]]; then
      echo "${run_dir}/${target_model_file}"
      return 0
    fi
    latest_ckpt="$(find "${run_dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n1 || true)"
    if [[ -n "${latest_ckpt}" ]]; then
      echo "${latest_ckpt}"
      return 0
    fi
  done < <(find /data/logs_new -maxdepth 2 -type d -name "*${run_id}*" 2>/dev/null | sort)

  echo ""
}

local_checkpoint_is_tracking_only() {
  local checkpoint_path="$1"
  "${PYTHON_BIN}" - "${checkpoint_path}" <<'PY' 2>/dev/null || true
import sys
import torch

cfg = torch.load(sys.argv[1], map_location="cpu").get("experiment_config", {})
perception_cfg = cfg.get("perception") if isinstance(cfg, dict) else None
enabled = perception_cfg.get("enabled") if isinstance(perception_cfg, dict) else False
print("1" if not bool(enabled) else "0")
PY
}

find_latest_terrain_teacher_ckpt() {
  local latest_ckpt=""
  local latest_mtime=0
  local ckpt=""
  local mtime=0
  local root=""
  IFS=',' read -r -a roots <<< "${TEACHER_LOG_ROOTS:-/data/logs_new/boxer,/data/logs_new/terrain-aware}"
  for root in "${roots[@]}"; do
    [[ -d "${root}" ]] || continue
    while IFS= read -r dir; do
      ckpt="$(find "${dir}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V | tail -n1 || true)"
      [[ -n "${ckpt}" ]] || continue
      [[ "$(local_checkpoint_is_tracking_only "${ckpt}")" == "1" ]] || continue
      mtime="$(stat -c %Y "${ckpt}" 2>/dev/null || echo 0)"
      if [[ "${mtime}" -gt "${latest_mtime}" ]]; then
        latest_mtime="${mtime}"
        latest_ckpt="${ckpt}"
      fi
    done < <(find "${root}" -maxdepth 1 -type d \
      \( -iname '*terrain*' -o -iname '*wbt*terrain*' \) \
      ! -iname '*distill*' 2>/dev/null)
  done
  echo "${latest_ckpt}"
}

read_meta_rows_cols() {
  local meta_path="$1"
  "${PYTHON_BIN}" - "${meta_path}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as handle:
    meta = json.load(handle)
rows = int(meta.get("tile_rows", 1) or 1)
cols = int(meta.get("tile_cols", 1) or 1)
print(rows, cols)
PY
}

extract_checkpoint_teacher_summary() {
  local checkpoint_ref="$1"
  "${PYTHON_BIN}" - "${checkpoint_ref}" <<'PY' 2>/dev/null || true
import sys
import tempfile
from pathlib import Path

import torch


def parse_ref(reference: str) -> tuple[str, str]:
    remainder = reference[len("wandb://") :]
    parts = remainder.split("/")
    entity, project = parts[0], parts[1]
    run_id_index = 2
    if len(parts) > 4 and parts[2] == "runs":
        run_id_index = 3
    run_id = parts[run_id_index]
    ckpt_name = "/".join(parts[run_id_index + 1 :]).strip()
    return f"{entity}/{project}/{run_id}", ckpt_name


def load_payload(reference: str):
    if reference.startswith("wandb://"):
        import wandb

        run_path, ckpt_name = parse_ref(reference)
        run = wandb.Api(timeout=30).run(run_path)
        with tempfile.TemporaryDirectory() as tmp_dir:
            downloaded = run.file(ckpt_name).download(root=tmp_dir, replace=True)
            path = Path(downloaded.name)
            if not path.is_absolute():
                path = (Path.cwd() / path).resolve()
            return torch.load(path, map_location="cpu")
    return torch.load(reference, map_location="cpu")


payload = load_payload(sys.argv[1])
cfg = payload.get("experiment_config")
if not isinstance(cfg, dict):
    sys.exit(0)

perception_cfg = cfg.get("perception")
obs_cfg = cfg.get("observation")
groups = obs_cfg.get("groups", {}) if isinstance(obs_cfg, dict) else {}
training_cfg = cfg.get("training")

values = [
    perception_cfg.get("enabled") if isinstance(perception_cfg, dict) else None,
    perception_cfg.get("output_mode") if isinstance(perception_cfg, dict) else None,
    "perception_obs" in groups if isinstance(groups, dict) else None,
    training_cfg.get("name") if isinstance(training_cfg, dict) else None,
]
for value in values:
    print("" if value is None else str(value))
PY
}

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${CKPT:-${DEFAULT_TEACHER_CHECKPOINT:-}}}"
if [[ $# -gt 0 ]]; then
  if [[ "$1" == wandb://* || "$1" == https://wandb.ai/*/runs/* || "$1" == /* || "$1" == ./* || "$1" == ../* || "$1" == *.pt ]]; then
    TEACHER_CHECKPOINT="$1"
    shift
  fi
fi

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  TEACHER_CHECKPOINT="$(find_latest_terrain_teacher_ckpt)"
fi

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "[ERROR] Could not auto-resolve terrain teacher checkpoint. Pass one explicitly." >&2
  exit 1
fi

if [[ "${TEACHER_CHECKPOINT}" == https://wandb.ai/*/runs/* ]]; then
  LOCAL_WANDB_CKPT="$(resolve_local_checkpoint_from_run_url "${TEACHER_CHECKPOINT}" "${WANDB_MODEL_FILE:-}")"
  if [[ -n "${LOCAL_WANDB_CKPT}" && -f "${LOCAL_WANDB_CKPT}" ]]; then
    TEACHER_CHECKPOINT="${LOCAL_WANDB_CKPT}"
    echo "[INFO] Resolved wandb run URL to local checkpoint: ${TEACHER_CHECKPOINT}"
  else
    TEACHER_CHECKPOINT="$(normalize_checkpoint_ref "${TEACHER_CHECKPOINT}")"
  fi
fi

if [[ "${TEACHER_CHECKPOINT}" != wandb://* ]] && [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "[ERROR] Teacher checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
  exit 1
fi

ALLOW_TEACHER_PERCEPTION_CHECKPOINT_RAW="${ALLOW_TEACHER_PERCEPTION_CHECKPOINT:-0}"
case "$(echo "${ALLOW_TEACHER_PERCEPTION_CHECKPOINT_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    ALLOW_TEACHER_PERCEPTION_CHECKPOINT_FLAG=1
    ;;
  0|false|no|off|"")
    ALLOW_TEACHER_PERCEPTION_CHECKPOINT_FLAG=0
    ;;
  *)
    echo "[ERROR] ALLOW_TEACHER_PERCEPTION_CHECKPOINT must be one of: 0/1/true/false/yes/no/on/off. Got: ${ALLOW_TEACHER_PERCEPTION_CHECKPOINT_RAW}" >&2
    exit 2
    ;;
esac

mapfile -t teacher_summary_lines < <(extract_checkpoint_teacher_summary "${TEACHER_CHECKPOINT}")
TEACHER_PERCEPTION_ENABLED="${teacher_summary_lines[0]:-}"
TEACHER_PERCEPTION_OUTPUT_MODE="${teacher_summary_lines[1]:-}"
TEACHER_HAS_PERCEPTION_OBS="${teacher_summary_lines[2]:-}"
TEACHER_TRAINING_NAME="${teacher_summary_lines[3]:-}"

if [[ "${ALLOW_TEACHER_PERCEPTION_CHECKPOINT_FLAG}" != "1" && "$(echo "${TEACHER_PERCEPTION_ENABLED}" | tr '[:upper:]' '[:lower:]')" == "true" ]]; then
  echo "[ERROR] distill_terrain_root.sh expects a tracking-only teacher checkpoint, but teacher perception.enabled=True: ${TEACHER_CHECKPOINT}" >&2
  echo "[ERROR] Training name: ${TEACHER_TRAINING_NAME:-<unknown>}, perception mode: ${TEACHER_PERCEPTION_OUTPUT_MODE:-<unknown>}, has perception_obs: ${TEACHER_HAS_PERCEPTION_OBS:-<unknown>}" >&2
  echo "[ERROR] Tracking should stay perception-free; distillation is the stage that adds perception." >&2
  echo "[ERROR] If you really need to bypass this, set ALLOW_TEACHER_PERCEPTION_CHECKPOINT=1." >&2
  exit 2
fi

PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i_17x17}"
case "${PERCEPTION_PRESET}" in
  none|camera_depth_d435i|camera_depth_d435i_17x17|heightmap)
    ;;
  *)
    echo "[ERROR] Unsupported PERCEPTION_PRESET=${PERCEPTION_PRESET}. Use none|camera_depth_d435i|camera_depth_d435i_17x17|heightmap." >&2
    exit 2
    ;;
esac

EXP="${EXP:-g1-29dof-wbt-terrain-distill-sparse-root-cmd}"
if [[ "${EXP}" == exp:* ]]; then
  EXP_ARG="${EXP}"
else
  EXP_ARG="exp:${EXP}"
fi

DEFAULT_CUDA_VISIBLE_DEVICES="${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}"
if [[ -z "${NPROC:-}" ]]; then
  IFS=',' read -r -a _visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC="${#_visible_gpus[@]}"
fi
NPROC="${NPROC:-1}"
PER_GPU_ENVS="${PER_GPU_ENVS:-8192}"
NUM_ENVS="${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC:-1800}"

TRAINING_PROJECT="${TRAINING_PROJECT:-terrain-aware}"
RUN_NAME="${RUN_NAME:-g1_terrain_distill_root_access_to_depth}"
TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_terrain_distill_root_access_to_depth}"
LOGGER="${LOGGER:-logger:wandb}"

NUM_LEARNING_ITERATIONS="${NUM_LEARNING_ITERATIONS:-3000}"
ACTOR_LR="${ACTOR_LR:-7e-5}"
CRITIC_LR="${CRITIC_LR:-7e-5}"
ACTOR_MIN_NOISE_STD="${ACTOR_MIN_NOISE_STD:-0.01}"
INIT_NOISE_STD="${INIT_NOISE_STD:-0.01}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-536870912}"
BC_LOSS_COEF="${BC_LOSS_COEF:-1.0}"
TEACHER_ACTION_MIX_RATIO="${TEACHER_ACTION_MIX_RATIO:-0.0}"
PPO_START_EPOCH="${PPO_START_EPOCH:-1000}"
DAGGER_END_EPOCH="${DAGGER_END_EPOCH:-2000}"
DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF:-5.0}"
DISTILL_LOSS_TYPE="${DISTILL_LOSS_TYPE:-mse}"
DAGGER_IGNORE_ZERO_TEACHER_ACTIONS="${DAGGER_IGNORE_ZERO_TEACHER_ACTIONS:-True}"
STRICT_TEACHER_LOAD="${STRICT_TEACHER_LOAD:-True}"
TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS:-actor_obs,actor_obs_target}"
PERCEPTION_INTO_POLICY_MODULES="${PERCEPTION_INTO_POLICY_MODULES:-True}"
CLIP_TEACHER_ACTIONS="${CLIP_TEACHER_ACTIONS:-True}"
CLIP_ACTIONS_THRESHOLD="${CLIP_ACTIONS_THRESHOLD:-8.0}"

MOTION_DIR="${MOTION_DIR:-/data/terrain/___crisp_clean_motion}"
OBJ_SOURCE="${OBJ_SOURCE:-/data/terrain/___crisp_clean_geometry}"
OBJ_META_PATH="${OBJ_META_PATH:-}"
NUM_ROWS="${NUM_ROWS:-}"
NUM_COLS="${NUM_COLS:-}"
REBUILD_FUSED="${REBUILD_FUSED:-0}"
FUSED_OUT_DIR="${FUSED_OUT_DIR:-${SCRIPT_DIR}/multi-terrain/generated}"
FUSED_PREFIX="${FUSED_PREFIX:-terrain_distill}"

PAIR_TERRAIN_WITH_MOTION="${PAIR_TERRAIN_WITH_MOTION:-True}"
START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-0.7}"
ENABLE_DEFAULT_POSE_APPEND="${ENABLE_DEFAULT_POSE_APPEND:-False}"
DEFAULT_POSE_APPEND_DURATION_S="${DEFAULT_POSE_APPEND_DURATION_S:-0}"
ENABLE_DEFAULT_POSE_PREPEND="${ENABLE_DEFAULT_POSE_PREPEND:-False}"
DEFAULT_POSE_PREPEND_DURATION_S="${DEFAULT_POSE_PREPEND_DURATION_S:-0}"
RESET_NOISE_SCALE="${RESET_NOISE_SCALE:-1.0}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
LOAD_OPTIMIZER="${LOAD_OPTIMIZER:-False}"

IMAGE_WIDTH="${IMAGE_WIDTH:-17}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-17}"
CAMERA_NEAR="${CAMERA_NEAR:-0.3}"
CAMERA_FAR="${CAMERA_FAR:-3.0}"
CAMERA_MAX_DISTANCE="${CAMERA_MAX_DISTANCE:-3.0}"
CAMERA_WARP_PREPROCESS="${CAMERA_WARP_PREPROCESS:-True}"

if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi
if [[ ! -e "${OBJ_SOURCE}" ]]; then
  echo "[ERROR] OBJ_SOURCE not found: ${OBJ_SOURCE}" >&2
  exit 1
fi

OBJ_PATH="${OBJ_SOURCE}"
if [[ -d "${OBJ_SOURCE}" ]]; then
  mapfile -t OBJ_FILES < <(find "${OBJ_SOURCE}" -maxdepth 1 -type f \( -name "*.obj" -o -name "*.OBJ" \) | sort)
  NUM_TILES="${#OBJ_FILES[@]}"
  if [[ "${NUM_TILES}" -eq 0 ]]; then
    echo "[ERROR] No OBJ files found in ${OBJ_SOURCE}" >&2
    exit 1
  fi
  if [[ -z "${NUM_ROWS}" ]]; then
    NUM_ROWS=1
  fi
  mkdir -p "${FUSED_OUT_DIR}"
  FUSED_OBJ="${FUSED_OUT_DIR}/${FUSED_PREFIX}_${NUM_ROWS}x${NUM_TILES}.obj"
  FUSED_META="${FUSED_OUT_DIR}/${FUSED_PREFIX}_${NUM_ROWS}x${NUM_TILES}.json"

  NEEDS_REBUILD=0
  if [[ ! -f "${FUSED_OBJ}" || ! -f "${FUSED_META}" ]]; then
    NEEDS_REBUILD=1
  else
    read -r META_ROWS META_COLS < <(read_meta_rows_cols "${FUSED_META}")
    if [[ "${META_ROWS}" != "${NUM_ROWS}" || "${META_COLS}" != "${NUM_TILES}" ]]; then
      NEEDS_REBUILD=1
    fi
  fi

  if [[ "${REBUILD_FUSED}" == "1" || "${NEEDS_REBUILD}" == "1" ]]; then
    "${PYTHON_BIN}" preprocess/build_obj_terrain_tiles.py \
      --obj-dir "${OBJ_SOURCE}" \
      --out-obj "${FUSED_OBJ}" \
      --out-meta "${FUSED_META}" \
      --num-rows "${NUM_ROWS}"
  fi

  OBJ_PATH="${FUSED_OBJ}"
  OBJ_META_PATH="${OBJ_META_PATH:-${FUSED_META}}"
fi

if [[ -z "${OBJ_META_PATH}" && -f "${OBJ_PATH%.*}.json" ]]; then
  OBJ_META_PATH="${OBJ_PATH%.*}.json"
fi

if [[ -n "${OBJ_META_PATH}" ]]; then
  if [[ ! -f "${OBJ_META_PATH}" ]]; then
    echo "[ERROR] OBJ_META_PATH not found: ${OBJ_META_PATH}" >&2
    exit 1
  fi
  read -r META_ROWS META_COLS < <(read_meta_rows_cols "${OBJ_META_PATH}")
  NUM_ROWS="${NUM_ROWS:-${META_ROWS}}"
  NUM_COLS="${NUM_COLS:-${META_COLS}}"
fi

NUM_ROWS="${NUM_ROWS:-1}"
NUM_COLS="${NUM_COLS:-1}"

PERCEPTION_OVERRIDES=()
if [[ "${PERCEPTION_PRESET}" == "camera_depth_d435i" || "${PERCEPTION_PRESET}" == "camera_depth_d435i_17x17" ]]; then
  PERCEPTION_OVERRIDES=(
    --perception.camera_width="${IMAGE_WIDTH}"
    --perception.camera_height="${IMAGE_HEIGHT}"
    --perception.camera_near="${CAMERA_NEAR}"
    --perception.camera_far="${CAMERA_FAR}"
    --perception.max_distance="${CAMERA_MAX_DISTANCE}"
    --perception.camera_warp_preprocess="${CAMERA_WARP_PREPROCESS}"
  )
fi

EXTRA_ARGS=("$@")

echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS} strict_teacher_load=${STRICT_TEACHER_LOAD}"
echo "[INFO] teacher_perception=${TEACHER_PERCEPTION_ENABLED:-<unknown>}/${TEACHER_PERCEPTION_OUTPUT_MODE:-<unknown>} perception_obs=${TEACHER_HAS_PERCEPTION_OBS:-<unknown>}"
echo "[INFO] exp=${EXP_ARG}"
echo "[INFO] perception=${PERCEPTION_PRESET} inject_into_policy_modules=${PERCEPTION_INTO_POLICY_MODULES}"
echo "[INFO] motion_dir=${MOTION_DIR}"
echo "[INFO] obj_path=${OBJ_PATH}"
if [[ -n "${OBJ_META_PATH}" ]]; then
  echo "[INFO] obj_meta_path=${OBJ_META_PATH}"
fi
echo "[INFO] terrain_grid=${NUM_ROWS}x${NUM_COLS}"
echo "[INFO] total_envs=${NUM_ENVS} world_size=${NPROC} envs_per_rank=$((NUM_ENVS / NPROC))"
echo "[INFO] bc_loss_coef=${BC_LOSS_COEF} dagger_loss_coef=${DAGGER_LOSS_COEF} teacher_action_mix_ratio=${TEACHER_ACTION_MIX_RATIO}"
echo "[INFO] ppo_start_epoch=${PPO_START_EPOCH} dagger_end_epoch=${DAGGER_END_EPOCH}"

cmd=(
  torchrun
  --nnodes="${NNODES}"
  --node_rank="${NODE_RANK}"
  --master_addr="${MASTER_ADDR}"
  --nproc_per_node="${NPROC}"
  --max_restarts="${MAX_RESTARTS}"
  --master_port="${MASTER_PORT}"
  src/holosoma/holosoma/train_agent.py
  "${EXP_ARG}"
  "perception:${PERCEPTION_PRESET}"
  terrain:terrain-load-obj
  --algo.config.distill.enabled=True
  --algo.config.distill.mode=dagger
  --algo.config.distill.policy_to_clone="${TEACHER_CHECKPOINT}"
  --algo.config.distill.bc_loss_coef="${BC_LOSS_COEF}"
  --algo.config.distill.clip_teacher_actions="${CLIP_TEACHER_ACTIONS}"
  --algo.config.distill.clip_actions_threshold="${CLIP_ACTIONS_THRESHOLD}"
  --algo.config.distill.teacher_obs_keys="${TEACHER_OBS_KEYS}"
  --algo.config.distill.strict_teacher_load="${STRICT_TEACHER_LOAD}"
  --algo.config.distill.teacher_action_mix_ratio="${TEACHER_ACTION_MIX_RATIO}"
  --algo.config.distill.ppo_start_epoch="${PPO_START_EPOCH}"
  --algo.config.distill.dagger_end_epoch="${DAGGER_END_EPOCH}"
  --algo.config.distill.dagger_loss_coef="${DAGGER_LOSS_COEF}"
  --algo.config.distill.distill_loss_type="${DISTILL_LOSS_TYPE}"
  --algo.config.distill.dagger_ignore_zero_teacher_actions="${DAGGER_IGNORE_ZERO_TEACHER_ACTIONS}"
  --training.num_envs="${NUM_ENVS}"
  --training.project="${TRAINING_PROJECT}"
  --training.name="${TRAINING_NAME}"
  --training.multigpu=$([[ "${NPROC}" -gt 1 || "${NNODES}" -gt 1 ]] && echo True || echo False)
  --algo.config.num_learning_iterations="${NUM_LEARNING_ITERATIONS}"
  --algo.config.actor_learning_rate="${ACTOR_LR}"
  --algo.config.critic_learning_rate="${CRITIC_LR}"
  --algo.config.init_noise_std="${INIT_NOISE_STD}"
  --algo.config.module_dict.actor.min_noise_std="${ACTOR_MIN_NOISE_STD}"
  --algo.config.normalize_actor_obs=False
  --algo.config.normalize_critic_obs=False
  --algo.config.save_interval="${SAVE_INTERVAL}"
  --algo.config.load_optimizer="${LOAD_OPTIMIZER}"
  --simulator.config.scene.env_spacing=0.0
  --simulator.config.sim.physx.gpu_collision_stack_size="${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --terrain.terrain_term.obj_file_path "${OBJ_PATH}"
  --terrain.terrain_term.num_rows "${NUM_ROWS}"
  --terrain.terrain_term.num_cols "${NUM_COLS}"
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}"
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion="${PAIR_TERRAIN_WITH_MOTION}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob="${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.noise_to_initial_pose.overall_noise_scale="${RESET_NOISE_SCALE}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append="${ENABLE_DEFAULT_POSE_APPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s="${DEFAULT_POSE_APPEND_DURATION_S}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend="${ENABLE_DEFAULT_POSE_PREPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s="${DEFAULT_POSE_PREPEND_DURATION_S}"
  "${LOGGER}"
)

if [[ -n "${OBJ_META_PATH}" ]]; then
  cmd+=(--terrain.terrain_term.obj_metadata_path "${OBJ_META_PATH}")
fi

cmd+=("${PERCEPTION_OVERRIDES[@]}")

if [[ "${LOGGER}" != "logger:disabled" ]]; then
  cmd+=(
    --logger.name="${RUN_NAME}"
    --logger.video.enabled=False
    --logger.headless_recording=False
    --logger.video.upload_to_wandb=False
  )
fi

cmd+=("${EXTRA_ARGS[@]}")

HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES="${PERCEPTION_INTO_POLICY_MODULES}" \
TORCH_DIST_TIMEOUT_SEC="${TORCH_DIST_TIMEOUT_SEC}" \
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
"${cmd[@]}"
