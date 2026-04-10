#!/usr/bin/env bash
set -euo pipefail

# Terrain generalist training entrypoint.
# Default flow uses heightmap perception unless explicitly overridden.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
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
PYTHON_BIN=${PYTHON_BIN:-"${DEFAULT_PYTHON_BIN}"}

DEFAULT_CUDA_VISIBLE_DEVICES=${DEFAULT_CUDA_VISIBLE_DEVICES:-0,1,2,3}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-${DEFAULT_CUDA_VISIBLE_DEVICES}}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES//[[:space:]]/}
export CUDA_VISIBLE_DEVICES

detect_nproc() {
  local gpu_count=""
  if gpu_count="$("${PYTHON_BIN}" - <<'PY' 2>/dev/null
import torch

print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)"; then
    gpu_count="${gpu_count//[[:space:]]/}"
  fi

  if [[ ! "${gpu_count}" =~ ^[0-9]+$ || "${gpu_count}" == "0" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d '[:space:]')"
    fi
  fi

  if [[ ! "${gpu_count}" =~ ^[0-9]+$ || "${gpu_count}" == "0" ]]; then
    echo "[ERROR] Failed to detect available CUDA GPUs. Set NPROC explicitly." >&2
    exit 1
  fi

  echo "${gpu_count}"
}

PERCEPTION_PRESET=${1:-${PERCEPTION_PRESET:-heightmap}}
case "${PERCEPTION_PRESET}" in
  none|camera_depth_d435i|heightmap)
    if [[ $# -gt 0 ]]; then
      shift
    fi
    ;;
  *)
    echo "[ERROR] Unknown PERCEPTION_PRESET=${PERCEPTION_PRESET}. Use none|camera_depth_d435i|heightmap." >&2
    exit 1
    ;;
esac

EXP=${EXP:-g1-29dof-wbt-terrain-transformer}
if [[ "${EXP}" == exp:* ]]; then
  EXP_ARG="${EXP}"
else
  EXP_ARG="exp:${EXP}"
fi

AVAILABLE_GPU_COUNT=$(detect_nproc)
NPROC=${NPROC:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES}")}
if [[ ! "${NPROC}" =~ ^[0-9]+$ || "${NPROC}" -lt 1 ]]; then
  echo "[ERROR] NPROC must be a positive integer. Got: ${NPROC}" >&2
  exit 1
fi
if [[ "${NPROC}" -gt "${AVAILABLE_GPU_COUNT}" ]]; then
  echo "[ERROR] Requested NPROC=${NPROC}, but only ${AVAILABLE_GPU_COUNT} visible CUDA GPU(s) are available." >&2
  exit 1
fi
PER_GPU_ENVS=${PER_GPU_ENVS:-2048}
NUM_ENVS=${NUM_ENVS:-$((NPROC * PER_GPU_ENVS))}
MASTER_PORT=${MASTER_PORT:-$((29500 + RANDOM % 1000))}
PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT:-33554432}
PHYSX_GPU_MAX_RIGID_PATCH_COUNT=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-655360}
PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-134217728}
PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-134217728}
PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-16777216}

WANDB_PROJECT=${WANDB_PROJECT:-terrain-aware}
TRAINING_NAME=${TRAINING_NAME:-g1_29dof_wbt_terrain_generalist}
LOGGER_NAME=${LOGGER_NAME:-g1_terrain_generalist}
RESUME_CKPT=${RESUME_CKPT:-}

ACTOR_LR=${ACTOR_LR:-1e-3}
CRITIC_LR=${CRITIC_LR:-1e-3}
NORMALIZE_ACTOR_OBS=${NORMALIZE_ACTOR_OBS:-True}
NORMALIZE_CRITIC_OBS=${NORMALIZE_CRITIC_OBS:-True}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
LOAD_OPTIMIZER=${LOAD_OPTIMIZER:-False}
HOLOSOMA_EXPORT_ONNX_DURING_TRAIN=${HOLOSOMA_EXPORT_ONNX_DURING_TRAIN:-0}
HOLOSOMA_EXPORT_ONNX_AT_END=${HOLOSOMA_EXPORT_ONNX_AT_END:-1}
HOLOSOMA_WANDB_SAVE_FILES=${HOLOSOMA_WANDB_SAVE_FILES:-0}
PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE:-67108864}
PHYSX_GPU_HEAP_CAPACITY=${PHYSX_GPU_HEAP_CAPACITY:-67108864}
PHYSX_GPU_TEMP_BUFFER_CAPACITY=${PHYSX_GPU_TEMP_BUFFER_CAPACITY:-16777216}

MOTION_DIR=${MOTION_DIR:-${SCRIPT_DIR}/data/ds_crisp_data/___crisp_clean_motion}
OBJ_SOURCE=${OBJ_SOURCE:-${SCRIPT_DIR}/data/ds_crisp_data/___crisp_clean_geometry}
OBJ_META_PATH=${OBJ_META_PATH:-}
NUM_ROWS=${NUM_ROWS:-}
NUM_COLS=${NUM_COLS:-}
SINGLE_TERRAIN_ID=${SINGLE_TERRAIN_ID:-}
SINGLE_TERRAIN_OBJ=${SINGLE_TERRAIN_OBJ:-}
FORCE_SINGLE_TERRAIN=${FORCE_SINGLE_TERRAIN:-0}
REBUILD_FUSED=${REBUILD_FUSED:-0}
GENERATED_DATA_ROOT=${GENERATED_DATA_ROOT:-${SCRIPT_DIR}/data/ds_crisp_data/_generated}
FUSED_OUT_DIR=${FUSED_OUT_DIR:-${GENERATED_DATA_ROOT}/fused}
FUSED_PREFIX_EXPLICIT=0
if [[ -n "${FUSED_PREFIX+x}" ]]; then
  FUSED_PREFIX_EXPLICIT=1
fi
FUSED_PREFIX=${FUSED_PREFIX:-terrain_generalist}
PAIRED_MANIFEST_PATH=${PAIRED_MANIFEST_PATH:-${PAIRED_DATA_MANIFEST:-}}
PAIRED_DS_CRISP_DATA_ROOT=${PAIRED_DS_CRISP_DATA_ROOT:-}
PAIRED_STAGE_OUT_DIR=${PAIRED_STAGE_OUT_DIR:-${GENERATED_DATA_ROOT}/staged}

PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION:-True}
USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-True}
ADD_GROUND_PLANE_COLLISION=${ADD_GROUND_PLANE_COLLISION:-True}
START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB:-0.0}
FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}
ENABLE_DEFAULT_POSE_APPEND=${ENABLE_DEFAULT_POSE_APPEND:-False}
DEFAULT_POSE_APPEND_DURATION_S=${DEFAULT_POSE_APPEND_DURATION_S:-0}
ENABLE_DEFAULT_POSE_PREPEND=${ENABLE_DEFAULT_POSE_PREPEND:-False}
DEFAULT_POSE_PREPEND_DURATION_S=${DEFAULT_POSE_PREPEND_DURATION_S:-0}

# Early-termination tolerances for terrain tracking. Relaxed defaults reduce
# premature resets on uneven terrain and keep episode length stable.
BAD_TRACKING_REF_POS_THRESHOLD=${BAD_TRACKING_REF_POS_THRESHOLD:-1.0}
BAD_TRACKING_REF_ORI_THRESHOLD=${BAD_TRACKING_REF_ORI_THRESHOLD:-1.2}
BAD_TRACKING_BODY_POS_THRESHOLD=${BAD_TRACKING_BODY_POS_THRESHOLD:-0.55}
ALLOW_TERRAIN_SLOT_OVERLAP=${ALLOW_TERRAIN_SLOT_OVERLAP:-0}
DRY_RUN=${DRY_RUN:-0}

HEADLESS=${HEADLESS:-True}
ENABLE_VISER=${ENABLE_VISER:-0}
VISER_PORT_SET=0
if [[ -n "${VISER_PORT+x}" ]]; then
  VISER_PORT_SET=1
fi
VISER_PORT=${VISER_PORT:-$((RANDOM % 8976 + 1024))}
VISER_ENV_ID=${VISER_ENV_ID:-0}
VISER_ENV_COUNT_SET=0
if [[ -n "${VISER_ENV_COUNT+x}" ]]; then
  VISER_ENV_COUNT_SET=1
fi
VISER_ENV_COUNT=${VISER_ENV_COUNT:-${NUM_ENVS}}
VISER_UPDATE_HZ_SET=0
if [[ -n "${VISER_UPDATE_HZ+x}" ]]; then
  VISER_UPDATE_HZ_SET=1
fi
VISER_UPDATE_HZ=${VISER_UPDATE_HZ:-30}
VISER_SYNC_TO_SIM=${VISER_SYNC_TO_SIM:-True}
VISER_FORCE_DT=${VISER_FORCE_DT:-True}
VISER_RECENTER=${VISER_RECENTER:-True}
VISER_SHOW_SCANDOTS=${VISER_SHOW_SCANDOTS:-False}
VISER_MULTI_ENV_SPACING_SET=0
if [[ -n "${VISER_MULTI_ENV_SPACING+x}" ]]; then
  VISER_MULTI_ENV_SPACING_SET=1
fi
VISER_MULTI_ENV_SPACING=${VISER_MULTI_ENV_SPACING:-0.0}
TRAIN_DEBUG_VISER=${TRAIN_DEBUG_VISER:-0}
DEBUG_VISER_ENV_COUNT=${DEBUG_VISER_ENV_COUNT:-4}
DEBUG_VISER_UPDATE_HZ=${DEBUG_VISER_UPDATE_HZ:-30}
DEBUG_VISER_MULTI_ENV_COLS=${DEBUG_VISER_MULTI_ENV_COLS:-2}
DEBUG_VISER_PORT=${DEBUG_VISER_PORT:-}

IMAGE_WIDTH=${IMAGE_WIDTH:-106}
IMAGE_HEIGHT=${IMAGE_HEIGHT:-60}
CAMERA_WARP_PREPROCESS=${CAMERA_WARP_PREPROCESS:-True}
DISABLE_CAMERA_RANDOMIZATION=${DISABLE_CAMERA_RANDOMIZATION:-0}

read_meta_rows_cols() {
  local meta_path="$1"
  "${PYTHON_BIN}" - "${meta_path}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as handle:
    meta = json.load(handle)
rows = int(meta.get("tile_rows", 1) or 1)
cols = int(meta.get("tile_cols", 1) or 1)
print(rows, cols)
PY
}

canonicalize_path() {
  local path_str="$1"
  "${PYTHON_BIN}" - "${path_str}" "${SCRIPT_DIR}" <<'PY'
from pathlib import Path
import sys

path_str = sys.argv[1]
script_dir = Path(sys.argv[2]).resolve()
p = Path(path_str).expanduser()
if not p.is_absolute():
    p = script_dir / p
print(p.resolve())
PY
}

is_true() {
  case "${1:-}" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

normalize_extra_cli_args() {
  local -a input_args=("$@")
  local -a output_args=()
  local expecting_value=0
  local arg=""

  for arg in "${input_args[@]}"; do
    if [[ "${expecting_value}" == "1" ]]; then
      output_args+=("${arg}")
      expecting_value=0
      continue
    fi

    case "${arg}" in
      --training.healdless=*|--training.headles=*|--training.headlesss=*)
        echo "[WARN] Normalizing typo option '${arg%%=*}' -> --training.headless" >&2
        output_args+=(--training.headless="${arg#*=}")
        ;;
      --training.healdless|--training.headles|--training.headlesss)
        echo "[WARN] Normalizing typo option '${arg}' -> --training.headless" >&2
        output_args+=(--training.headless)
        expecting_value=1
        ;;
      *)
        output_args+=("${arg}")
        ;;
    esac
  done

  NORMALIZED_EXTRA_CLI_ARGS=("${output_args[@]}")
}

if is_true "${TRAIN_DEBUG_VISER}"; then
  ENABLE_VISER=1
  export HOLOSOMA_DEBUG_TILE_LAYOUT=1
  if [[ -z "${HOLOSOMA_DEBUG_PAIR_ALIGNMENT+x}" ]]; then
    export HOLOSOMA_DEBUG_PAIR_ALIGNMENT=1
  fi
  if [[ -z "${HOLOSOMA_DEBUG_PAIR_ALIGNMENT_RESETS+x}" ]]; then
    export HOLOSOMA_DEBUG_PAIR_ALIGNMENT_RESETS=3
  fi
  if [[ -z "${VISER_DISABLE_CONTACT_FORCE_VIZ+x}" ]]; then
    export VISER_DISABLE_CONTACT_FORCE_VIZ=1
  fi
  if [[ "${VISER_ENV_COUNT_SET}" -eq 0 ]]; then
    VISER_ENV_COUNT="${DEBUG_VISER_ENV_COUNT}"
  fi
  if [[ "${VISER_UPDATE_HZ_SET}" -eq 0 ]]; then
    VISER_UPDATE_HZ="${DEBUG_VISER_UPDATE_HZ}"
  fi
  if [[ "${VISER_MULTI_ENV_SPACING_SET}" -eq 0 ]]; then
    VISER_MULTI_ENV_SPACING="0.0"
  fi
  if [[ "${VISER_PORT_SET}" -eq 0 ]]; then
    if [[ -n "${DEBUG_VISER_PORT}" ]]; then
      VISER_PORT="${DEBUG_VISER_PORT}"
    else
      VISER_PORT="$((20000 + RANDOM % 10000))"
    fi
  fi
  export VISER_MULTI_ENV_COLS="${VISER_MULTI_ENV_COLS:-${DEBUG_VISER_MULTI_ENV_COLS}}"
fi

preflight_pairing_assets() {
  local motion_path="$1"
  local obj_path="$2"
  local obj_meta_path="${3:-}"

  "${PYTHON_BIN}" - "${motion_path}" "${obj_path}" "${obj_meta_path}" <<'PY'
import json
import sys
from pathlib import Path


def _decode_strings(values):
    decoded = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def _list_motion_clips(path_str: str) -> list[str]:
    path = Path(path_str)
    if path.is_dir():
        names = []
        for candidate in sorted(path.iterdir()):
            if candidate.is_file() and candidate.suffix.lower() in {".npz", ".h5", ".hdf5"}:
                names.append(candidate.stem)
        if not names:
            raise FileNotFoundError(f"No motion clips found under {path}")
        return names
    if not path.exists():
        raise FileNotFoundError(f"Motion path not found: {path}")
    if path.suffix.lower() == ".npz":
        return [path.stem]
    if path.suffix.lower() in {".h5", ".hdf5"}:
        try:
            import h5py  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError("h5py is required to inspect HDF5 motion clips.") from exc
        with h5py.File(path, "r") as h5f:
            clips = h5f.get("clips")
            if clips is not None and "clip_ids" in clips:
                clip_ids = _decode_strings(clips["clip_ids"][()])
                if clip_ids:
                    return clip_ids
        return [path.stem]
    return [path.stem]


def _list_tile_names(obj_path_str: str, meta_path_str: str) -> list[str]:
    if meta_path_str:
        meta_path = Path(meta_path_str)
        if not meta_path.exists():
            raise FileNotFoundError(f"OBJ metadata path not found: {meta_path}")
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        tile_names = [str(name) for name in meta.get("tile_names", [])]
        if tile_names:
            return sorted(tile_names)

    obj_path = Path(obj_path_str)
    if obj_path.is_dir():
        tile_names = sorted(p.stem for p in obj_path.iterdir() if p.is_file() and p.suffix.lower() == ".obj")
        if tile_names:
            return tile_names
    return []


motion_clips = sorted(_list_motion_clips(sys.argv[1]))
tile_names = _list_tile_names(sys.argv[2], sys.argv[3])
if not tile_names:
    print("[INFO] Pairing preflight skipped: no named terrain tile set available.")
    raise SystemExit(0)

motion_set = set(motion_clips)
tile_set = set(tile_names)
missing_tiles = sorted(motion_set - tile_set)
unused_tiles = sorted(tile_set - motion_set)
if missing_tiles or unused_tiles:
    if missing_tiles:
        print(f"[ERROR] Terrain tiles missing for motion clips: {missing_tiles[:10]}")
    if unused_tiles:
        print(f"[ERROR] Terrain tiles without matching motion clips: {unused_tiles[:10]}")
    raise SystemExit(1)

print(f"[INFO] Pairing preflight passed: {len(motion_clips)} motion clips match {len(tile_names)} terrain tiles.")
PY
}

resolve_stage_info() {
  local source_path="$1"
  local stage_out_dir="$2"

  "${PYTHON_BIN}" - "${source_path}" "${stage_out_dir}" <<'PY'
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

source_path = Path(sys.argv[1]).expanduser().resolve()
stage_out_dir = Path(sys.argv[2]).expanduser().resolve()
if source_path.is_file():
    digest = hashlib.sha1(source_path.read_bytes()).hexdigest()[:12]
else:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_path.stem).strip("._") or "paired"
print(stage_out_dir / f"{stem}_{digest}")
print(f"terrain_generalist_{stem}_{digest[:8]}")
PY
}

if [[ -n "${PAIRED_MANIFEST_PATH}" && -n "${PAIRED_DS_CRISP_DATA_ROOT}" ]]; then
  echo "[ERROR] Set at most one of PAIRED_MANIFEST_PATH and PAIRED_DS_CRISP_DATA_ROOT." >&2
  exit 1
fi

if [[ -n "${PAIRED_MANIFEST_PATH}" || -n "${PAIRED_DS_CRISP_DATA_ROOT}" ]]; then
  PAIRED_SOURCE_PATH="${PAIRED_MANIFEST_PATH}"
  if [[ -z "${PAIRED_SOURCE_PATH}" ]]; then
    PAIRED_SOURCE_PATH="${PAIRED_DS_CRISP_DATA_ROOT}"
  fi
  if [[ ! -e "${PAIRED_SOURCE_PATH}" ]]; then
    echo "[ERROR] Paired staging source not found: ${PAIRED_SOURCE_PATH}" >&2
    exit 1
  fi
  if [[ -n "${PAIRED_MANIFEST_PATH}" && ! -f "${PAIRED_MANIFEST_PATH}" ]]; then
    echo "[ERROR] PAIRED_MANIFEST_PATH not found: ${PAIRED_MANIFEST_PATH}" >&2
    exit 1
  fi
  mkdir -p "${PAIRED_STAGE_OUT_DIR}"
  mapfile -t MANIFEST_STAGE_INFO < <(resolve_stage_info "${PAIRED_SOURCE_PATH}" "${PAIRED_STAGE_OUT_DIR}")
  PAIRED_STAGE_ROOT="${MANIFEST_STAGE_INFO[0]}"
  if [[ "${FUSED_PREFIX_EXPLICIT}" -eq 0 ]]; then
    FUSED_PREFIX="${MANIFEST_STAGE_INFO[1]}"
  fi

  if [[ -n "${PAIRED_MANIFEST_PATH}" ]]; then
    "${PYTHON_BIN}" preprocess/stage_paired_motion_terrain_manifest.py \
      --manifest "${PAIRED_MANIFEST_PATH}" \
      --out-root "${PAIRED_STAGE_ROOT}"
  else
    "${PYTHON_BIN}" preprocess/stage_paired_motion_terrain_manifest.py \
      --ds-crisp-data-root "${PAIRED_DS_CRISP_DATA_ROOT}" \
      --out-root "${PAIRED_STAGE_ROOT}"
  fi

  MOTION_DIR="${PAIRED_STAGE_ROOT}/___crisp_clean_motion"
  OBJ_SOURCE="${PAIRED_STAGE_ROOT}/___crisp_clean_geometry"
  OBJ_META_PATH=""

  if [[ -n "${PAIRED_MANIFEST_PATH}" ]]; then
    echo "[INFO] PAIRED_MANIFEST_PATH=${PAIRED_MANIFEST_PATH}"
  else
    echo "[INFO] PAIRED_DS_CRISP_DATA_ROOT=${PAIRED_DS_CRISP_DATA_ROOT}"
  fi
  echo "[INFO] PAIRED_STAGE_ROOT=${PAIRED_STAGE_ROOT}"
fi

MOTION_DIR="$(canonicalize_path "${MOTION_DIR}")"
OBJ_SOURCE="$(canonicalize_path "${OBJ_SOURCE}")"
if [[ -n "${OBJ_META_PATH}" ]]; then
  OBJ_META_PATH="$(canonicalize_path "${OBJ_META_PATH}")"
fi

if is_true "${FORCE_SINGLE_TERRAIN}" || [[ -n "${SINGLE_TERRAIN_ID}" || -n "${SINGLE_TERRAIN_OBJ}" ]]; then
  OBJ_PARENT_DIR="${OBJ_SOURCE}"
  if [[ -f "${OBJ_PARENT_DIR}" ]]; then
    OBJ_PARENT_DIR="$(dirname "${OBJ_PARENT_DIR}")"
  fi

  SELECTED_SINGLE_TERRAIN=""
  if [[ -n "${SINGLE_TERRAIN_OBJ}" ]]; then
    SELECTED_SINGLE_TERRAIN="$(canonicalize_path "${SINGLE_TERRAIN_OBJ}")"
  elif [[ -n "${SINGLE_TERRAIN_ID}" ]]; then
    SELECTED_SINGLE_TERRAIN="$(canonicalize_path "${OBJ_PARENT_DIR}/${SINGLE_TERRAIN_ID}.obj")"
  else
    mapfile -t _single_obj_candidates < <(find "${OBJ_PARENT_DIR}" -maxdepth 1 \( -type f -o -type l \) \( -name "*.obj" -o -name "*.OBJ" \) | sort)
    if [[ "${#_single_obj_candidates[@]}" -eq 0 ]]; then
      echo "[ERROR] FORCE_SINGLE_TERRAIN is set but no OBJ files found in ${OBJ_PARENT_DIR}" >&2
      exit 1
    fi
    SELECTED_SINGLE_TERRAIN="${_single_obj_candidates[0]}"
    SELECTED_SINGLE_TERRAIN="$(canonicalize_path "${SELECTED_SINGLE_TERRAIN}")"
  fi

  if [[ ! -f "${SELECTED_SINGLE_TERRAIN}" ]]; then
    echo "[ERROR] Single terrain OBJ not found: ${SELECTED_SINGLE_TERRAIN}" >&2
    exit 1
  fi

  OBJ_SOURCE="${SELECTED_SINGLE_TERRAIN}"
  OBJ_META_PATH=""
  PAIR_TERRAIN_WITH_MOTION=False
  NUM_ROWS=1
  NUM_COLS=1

  if [[ -z "${SUPPORT_MASK_DIR:-}" ]]; then
    SUPPORT_MASK_DIR="$(dirname "${SELECTED_SINGLE_TERRAIN}")"
  fi

  echo "[INFO] FORCE_SINGLE_TERRAIN enabled: using ${SELECTED_SINGLE_TERRAIN}"
  echo "[INFO] FORCE_SINGLE_TERRAIN overrides: PAIR_TERRAIN_WITH_MOTION=False NUM_ROWS=1 NUM_COLS=1"
fi

if [[ ! -e "${OBJ_SOURCE}" ]]; then
  echo "[ERROR] OBJ_SOURCE not found: ${OBJ_SOURCE}" >&2
  exit 1
fi
if [[ ! -e "${MOTION_DIR}" ]]; then
  echo "[ERROR] MOTION_DIR not found: ${MOTION_DIR}" >&2
  exit 1
fi

SUPPORT_MASK_DIR=${SUPPORT_MASK_DIR:-}
if [[ -z "${SUPPORT_MASK_DIR}" && -d "${OBJ_SOURCE}" ]]; then
  SUPPORT_MASK_DIR="${OBJ_SOURCE}"
elif [[ -n "${SUPPORT_MASK_DIR}" ]]; then
  SUPPORT_MASK_DIR="$(canonicalize_path "${SUPPORT_MASK_DIR}")"
fi

SUPPORT_MASK_DIR_OPTION_SUPPORTED=0
TERRAIN_CFG_FILE="${SCRIPT_DIR}/src/holosoma/holosoma/config_types/terrain.py"
if [[ -f "${TERRAIN_CFG_FILE}" ]] && grep -Eq '^[[:space:]]*support_mask_dir:[[:space:]]' "${TERRAIN_CFG_FILE}"; then
  SUPPORT_MASK_DIR_OPTION_SUPPORTED=1
fi

OBJ_PATH="${OBJ_SOURCE}"
if [[ -d "${OBJ_SOURCE}" ]]; then
  mapfile -t OBJ_FILES < <(find "${OBJ_SOURCE}" -maxdepth 1 \( -type f -o -type l \) \( -name "*.obj" -o -name "*.OBJ" \) | sort)
  NUM_TILES=${#OBJ_FILES[@]}
  if [[ "${NUM_TILES}" -eq 0 ]]; then
    echo "[ERROR] No OBJ files found in ${OBJ_SOURCE}" >&2
    exit 1
  fi

  if [[ -z "${NUM_ROWS}" ]]; then
    if [[ "${PAIR_TERRAIN_WITH_MOTION}" == "True" || "${PAIR_TERRAIN_WITH_MOTION}" == "true" ]]; then
      PER_RANK_ENVS=$(((NUM_ENVS + NPROC - 1) / NPROC))
      NUM_ROWS=$(((PER_RANK_ENVS + NUM_TILES - 1) / NUM_TILES))
      if [[ "${NUM_ROWS}" -lt 1 ]]; then
        NUM_ROWS=1
      fi
      echo "[INFO] Auto-selected NUM_ROWS=${NUM_ROWS} so terrain slots cover ${PER_RANK_ENVS} envs/rank across ${NUM_TILES} tiles."
    else
      NUM_ROWS=1
    fi
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
  OBJ_META_PATH="${FUSED_META}"
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
  NUM_ROWS=${NUM_ROWS:-${META_ROWS}}
  NUM_COLS=${NUM_COLS:-${META_COLS}}
fi

NUM_ROWS=${NUM_ROWS:-1}
NUM_COLS=${NUM_COLS:-1}

if [[ "${PAIR_TERRAIN_WITH_MOTION}" == "True" || "${PAIR_TERRAIN_WITH_MOTION}" == "true" ]]; then
  PER_RANK_ENVS=$(((NUM_ENVS + NPROC - 1) / NPROC))
  TERRAIN_SLOT_CAPACITY=$((NUM_ROWS * NUM_COLS))
  if [[ "${TERRAIN_SLOT_CAPACITY}" -lt "${PER_RANK_ENVS}" ]]; then
    if is_true "${ALLOW_TERRAIN_SLOT_OVERLAP}"; then
      echo "[WARN] Terrain slot capacity (${NUM_ROWS}x${NUM_COLS}=${TERRAIN_SLOT_CAPACITY}) is smaller than envs per rank (${PER_RANK_ENVS})." >&2
      echo "[WARN] Multiple envs will overlap the same paired terrain tile because ALLOW_TERRAIN_SLOT_OVERLAP=${ALLOW_TERRAIN_SLOT_OVERLAP}." >&2
    else
      echo "[ERROR] Terrain slot capacity (${NUM_ROWS}x${NUM_COLS}=${TERRAIN_SLOT_CAPACITY}) is smaller than envs per rank (${PER_RANK_ENVS})." >&2
      echo "[ERROR] Refusing to launch because paired terrain overlap would corrupt the run. Increase NUM_ROWS/NUM_COLS, reduce PER_GPU_ENVS, or set ALLOW_TERRAIN_SLOT_OVERLAP=1 to override." >&2
      exit 1
    fi
  fi
  preflight_pairing_assets "${MOTION_DIR}" "${OBJ_SOURCE}" "${OBJ_META_PATH}"
fi

PERCEPTION_OVERRIDES=()
if [[ "${PERCEPTION_PRESET}" == "camera_depth_d435i" ]]; then
  PERCEPTION_OVERRIDES=(
    --perception.camera_width="${IMAGE_WIDTH}"
    --perception.camera_height="${IMAGE_HEIGHT}"
    --perception.camera_warp_preprocess="${CAMERA_WARP_PREPROCESS}"
    --perception.camera_warp_freq_ratio=1
    --perception.camera_warp_latency_frame=0
    --perception.camera_warp_buffer_len=3
    --perception.camera_warp_crop_top=2
    --perception.camera_warp_crop_bottom=0
    --perception.camera_warp_crop_left=4
    --perception.camera_warp_crop_right=4
    --perception.camera_warp_min_valid_depth=0.15
    --perception.camera_warp_normalize=True
    --perception.camera_warp_edge_noise=True
    --perception.camera_warp_edge_border=3
    --perception.camera_warp_edge_shuffle_prob=0.9
    --perception.camera_warp_edge_empty_prob=0.7
    --perception.camera_warp_edge_thresh_primary=1.0
    --perception.camera_warp_edge_thresh_secondary=0.6
    --perception.camera_warp_edge_far_depth_thresh=2.5
    --perception.camera_warp_enable_holes=False
    --perception.camera_warp_hole_prob=0.0
  )
fi

RANDOMIZATION_OVERRIDES=()
if [[ "${DISABLE_CAMERA_RANDOMIZATION}" == "1" ]]; then
  RANDOMIZATION_OVERRIDES=(
    --randomization.setup_terms.setup_camera_raycast_randomization.params.enabled=False
    --randomization.reset_terms.randomize_camera_raycast.params.enabled=False
  )
fi

VISER_OVERRIDES=()
if [[ "${ENABLE_VISER}" == "1" ]]; then
  VISER_OVERRIDES=(
    --training.enable_viser=True
    --training.viser_port="${VISER_PORT}"
    --training.viser_env_id="${VISER_ENV_ID}"
    --training.viser_env_count="${VISER_ENV_COUNT}"
    --training.viser_update_hz="${VISER_UPDATE_HZ}"
    --training.viser_sync_to_sim="${VISER_SYNC_TO_SIM}"
    --training.viser_force_dt="${VISER_FORCE_DT}"
    --training.viser_recenter="${VISER_RECENTER}"
    --training.viser_show_scandots="${VISER_SHOW_SCANDOTS}"
    --training.viser_multi_env_spacing="${VISER_MULTI_ENV_SPACING}"
  )
fi

CHECKPOINT_OVERRIDES=()
if [[ -n "${RESUME_CKPT}" ]]; then
  CHECKPOINT_OVERRIDES=(
    --training.checkpoint "${RESUME_CKPT}"
  )
fi

echo "[INFO] EXP=${EXP_ARG}"
echo "[INFO] PERCEPTION=${PERCEPTION_PRESET}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] NPROC=${NPROC} PER_GPU_ENVS=${PER_GPU_ENVS} NUM_ENVS=${NUM_ENVS}"
echo "[INFO] PhysX gpu_max_rigid_contact_count=${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT} gpu_max_rigid_patch_count=${PHYSX_GPU_MAX_RIGID_PATCH_COUNT} gpu_found_lost_pairs_capacity=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}"
echo "[INFO] PhysX gpu_found_lost_aggregate_pairs_capacity=${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY} gpu_total_aggregate_pairs_capacity=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY} gpu_collision_stack_size=${PHYSX_GPU_COLLISION_STACK_SIZE} gpu_heap_capacity=${PHYSX_GPU_HEAP_CAPACITY} gpu_temp_buffer_capacity=${PHYSX_GPU_TEMP_BUFFER_CAPACITY}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJ_PATH=${OBJ_PATH}"
if [[ -n "${OBJ_META_PATH}" ]]; then
  echo "[INFO] OBJ_META_PATH=${OBJ_META_PATH}"
fi
echo "[INFO] TERRAIN_GRID=${NUM_ROWS}x${NUM_COLS}"
echo "[INFO] SCENE_LOAD_MODE=terrain-load-obj(static /World/ground mesh)"
echo "[INFO] PAIR_TERRAIN_WITH_MOTION=${PAIR_TERRAIN_WITH_MOTION}"
echo "[INFO] USE_ADAPTIVE_TIMESTEPS_SAMPLER=${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
echo "[INFO] ADD_GROUND_PLANE_COLLISION=${ADD_GROUND_PLANE_COLLISION}"
echo "[INFO] START_AT_TIMESTEP_ZERO_PROB=${START_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] FREEZE_AT_TIMESTEP_ZERO_PROB=${FREEZE_AT_TIMESTEP_ZERO_PROB}"
echo "[INFO] NORMALIZE_ACTOR_OBS=${NORMALIZE_ACTOR_OBS}"
echo "[INFO] NORMALIZE_CRITIC_OBS=${NORMALIZE_CRITIC_OBS}"
echo "[INFO] HOLOSOMA_EXPORT_ONNX_DURING_TRAIN=${HOLOSOMA_EXPORT_ONNX_DURING_TRAIN}"
echo "[INFO] HOLOSOMA_EXPORT_ONNX_AT_END=${HOLOSOMA_EXPORT_ONNX_AT_END}"
echo "[INFO] HOLOSOMA_WANDB_SAVE_FILES=${HOLOSOMA_WANDB_SAVE_FILES}"
if is_true "${TRAIN_DEBUG_VISER}"; then
  echo "[INFO] TRAIN_DEBUG_VISER=1"
  echo "[INFO] HOLOSOMA_DEBUG_TILE_LAYOUT=${HOLOSOMA_DEBUG_TILE_LAYOUT:-0}"
  echo "[INFO] HOLOSOMA_DEBUG_PAIR_ALIGNMENT=${HOLOSOMA_DEBUG_PAIR_ALIGNMENT:-0} resets=${HOLOSOMA_DEBUG_PAIR_ALIGNMENT_RESETS:-0}"
  echo "[INFO] VISER_MULTI_ENV_COLS=${VISER_MULTI_ENV_COLS:-<unset>}"
  echo "[INFO] VISER_DISABLE_CONTACT_FORCE_VIZ=${VISER_DISABLE_CONTACT_FORCE_VIZ:-0}"
fi
echo "[INFO] BAD_TRACKING_THRESHOLDS ref_pos=${BAD_TRACKING_REF_POS_THRESHOLD} ref_ori=${BAD_TRACKING_REF_ORI_THRESHOLD} body_pos=${BAD_TRACKING_BODY_POS_THRESHOLD}"
if [[ -n "${SUPPORT_MASK_DIR}" ]]; then
  if [[ "${SUPPORT_MASK_DIR_OPTION_SUPPORTED}" == "1" ]]; then
    echo "[INFO] SUPPORT_MASK_DIR=${SUPPORT_MASK_DIR}"
  else
    echo "[WARN] support_mask_dir option is not available in this checkout; ignoring SUPPORT_MASK_DIR=${SUPPORT_MASK_DIR}" >&2
  fi
else
  echo "[WARN] SUPPORT_MASK_DIR is empty. Support-aware terrain rewards require support sidecars via support_mask_dir or metadata source_obj_dir." >&2
fi
if [[ "${ENABLE_VISER}" == "1" ]]; then
  echo "[INFO] VISER=http://localhost:${VISER_PORT}"
fi

cmd=(
  torchrun
  --nproc_per_node="${NPROC}"
  --master_port="${MASTER_PORT}"
  src/holosoma/holosoma/train_agent.py
  "${EXP_ARG}"
  "perception:${PERCEPTION_PRESET}"
  terrain:terrain-load-obj
  --training.project="${WANDB_PROJECT}"
  --training.name="${TRAINING_NAME}"
  --training.num_envs="${NUM_ENVS}"
  --training.headless="${HEADLESS}"
  --simulator.config.scene.env_spacing=0.0
  --simulator.config.sim.physx.gpu_max_rigid_contact_count="${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT}"
  --simulator.config.sim.physx.gpu_max_rigid_patch_count="${PHYSX_GPU_MAX_RIGID_PATCH_COUNT}"
  --simulator.config.sim.physx.gpu_collision_stack_size="${PHYSX_GPU_COLLISION_STACK_SIZE}"
  --simulator.config.sim.physx.gpu_found_lost_pairs_capacity="${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu_found_lost_aggregate_pairs_capacity="${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu_total_aggregate_pairs_capacity="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY}"
  --simulator.config.sim.physx.gpu_heap_capacity="${PHYSX_GPU_HEAP_CAPACITY}"
  --simulator.config.sim.physx.gpu_temp_buffer_capacity="${PHYSX_GPU_TEMP_BUFFER_CAPACITY}"
  --terrain.terrain-term.obj-file-path "${OBJ_PATH}"
  --terrain.terrain-term.num-rows "${NUM_ROWS}"
  --terrain.terrain-term.num-cols "${NUM_COLS}"
  --terrain.terrain-term.add-ground-plane-collision="${ADD_GROUND_PLANE_COLLISION}"
  --algo.config.actor_learning_rate="${ACTOR_LR}"
  --algo.config.critic_learning_rate="${CRITIC_LR}"
  --algo.config.normalize_actor_obs="${NORMALIZE_ACTOR_OBS}"
  --algo.config.normalize_critic_obs="${NORMALIZE_CRITIC_OBS}"
  --algo.config.load_optimizer="${LOAD_OPTIMIZER}"
  --algo.config.save_interval="${SAVE_INTERVAL}"
  --command.setup_terms.motion_command.params.motion_config.motion_file "${MOTION_DIR}"
  --command.setup_terms.motion_command.params.motion_config.pair_terrain_with_motion="${PAIR_TERRAIN_WITH_MOTION}"
  --command.setup_terms.motion_command.params.motion_config.use_adaptive_timesteps_sampler="${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
  --command.setup_terms.motion_command.params.motion_config.start_at_timestep_zero_prob="${START_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.freeze_at_timestep_zero_prob="${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_append="${ENABLE_DEFAULT_POSE_APPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_append_duration_s="${DEFAULT_POSE_APPEND_DURATION_S}"
  --command.setup_terms.motion_command.params.motion_config.enable_default_pose_prepend="${ENABLE_DEFAULT_POSE_PREPEND}"
  --command.setup_terms.motion_command.params.motion_config.default_pose_prepend_duration_s="${DEFAULT_POSE_PREPEND_DURATION_S}"
  --termination.terms.bad_tracking.params.bad_ref_pos_threshold="${BAD_TRACKING_REF_POS_THRESHOLD}"
  --termination.terms.bad_tracking.params.bad_ref_ori_threshold="${BAD_TRACKING_REF_ORI_THRESHOLD}"
  --termination.terms.bad_tracking.params.bad_motion_body_pos_threshold="${BAD_TRACKING_BODY_POS_THRESHOLD}"
)

if [[ -n "${OBJ_META_PATH}" ]]; then
  cmd+=(--terrain.terrain-term.obj-metadata-path "${OBJ_META_PATH}")
fi
if [[ -n "${SUPPORT_MASK_DIR}" && "${SUPPORT_MASK_DIR_OPTION_SUPPORTED}" == "1" ]]; then
  cmd+=(--terrain.terrain-term.support-mask-dir "${SUPPORT_MASK_DIR}")
fi

cmd+=("${PERCEPTION_OVERRIDES[@]}")
cmd+=("${RANDOMIZATION_OVERRIDES[@]}")
cmd+=("${VISER_OVERRIDES[@]}")
cmd+=("${CHECKPOINT_OVERRIDES[@]}")
NORMALIZED_EXTRA_CLI_ARGS=()
normalize_extra_cli_args "$@"
cmd+=("${NORMALIZED_EXTRA_CLI_ARGS[@]}")
cmd+=(
  logger:wandb
  --logger.video.enabled=False
  --logger.headless_recording=False
  --logger.video.upload_to_wandb=False
  --logger.name="${LOGGER_NAME}"
)

if is_true "${DRY_RUN}"; then
  echo "[INFO] DRY_RUN=${DRY_RUN}; resolved launch command:"
  printf '  %q\n' "${cmd[@]}"
  exit 0
fi

export HOLOSOMA_EXPORT_ONNX_DURING_TRAIN
export HOLOSOMA_EXPORT_ONNX_AT_END
export HOLOSOMA_WANDB_SAVE_FILES

"${cmd[@]}"
