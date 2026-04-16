#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash debug_box.sh

Environment overrides:
  DATA_MODE                 Default: pure-sd
                            Same source-mode names as train_object_generalist_ds.sh:
                            pure-sd | pure-real | mix-naive | mix-curriculum
  MOTION_ROOT               Optional explicit source dir for replay bank
  MOTION_DIR                Alias for MOTION_ROOT
  VIEW_SCOPE                Default: all
                            all  : replay clips from the whole source bank
                            omomo: replay only OMOMO clips matched by OMOMO_CLIP_REGEX
  OMOMO_CLIP_REGEX          Default: ^sub
  REPLAY_SUBSET_DIR         Default: <motion_root>_replay_debug_<scope>
  GROUP_INDEX               Default: 0
  CLIP_LIST                 Default: empty
                            Optional comma-separated clip ids to keep in the replay bank.
  OBJECT_MAP                Default: <replay_subset_dir>/_clip_object_urdf_map.json
  DS_DATA_ROOT              Default: ./data/ds_box_data
  NUM_ENVS                  Default: 4
  VISER_PORT                Default: 18085
  VISER_ENV_ID              Default: 0
  VISER_GRID_SPACING        Default: 2.8
  VISER_MULTI_ENV_COLS      Default: 2
  VISER_UPDATE_HZ           Default: 30
  VISER_START_PAUSED        Default: 0
  VISER_ENV_SEQUENCE_LABEL_HEIGHT
                            Default: 1.6
  HEADLESS                  Default: False (training.headless)
  ISAAC_APP_HEADLESS        Default: auto (1 when no DISPLAY, else follows HEADLESS)
  GPU_ID                    Default: auto
  DRY_RUN                   Default: 0

Notes:
  - Source motion root now follows the same DATA_MODE/MOTION_DIR logic as train_object_generalist_ds.sh.
  - VIEW_SCOPE=omomo filters clips by OMOMO_CLIP_REGEX inside the training source path.
  - The replay subset only materializes the currently selected group (or CLIP_LIST), so object geometry
    stays aligned with the visible clips under heterogeneous object banks.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

source "${ROOT_DIR}/scripts/source_common.sh"
source "${ROOT_DIR}/scripts/source_isaacsim_setup.sh"
source "${ROOT_DIR}/scripts/object_generalist_ds_paths.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
DS_DATA_ROOT="${DS_DATA_ROOT:-${ROOT_DIR}/data/ds_box_data}"
DATA_MODE_RAW="${DATA_MODE:-pure-sd}"
DATA_MODE="$(ogds_normalize_data_mode "${DATA_MODE_RAW}")"

if [[ -n "${MOTION_ROOT:-}" ]]; then
  MOTION_SOURCE_ROOT="${MOTION_ROOT}"
elif [[ -n "${MOTION_DIR:-}" ]]; then
  MOTION_SOURCE_ROOT="${MOTION_DIR}"
else
  if ! MOTION_SOURCE_ROOT="$(ogds_default_motion_dir "${DS_DATA_ROOT}" "${DATA_MODE}")"; then
    echo "[ERROR] Unsupported DATA_MODE for debug_box.sh: ${DATA_MODE_RAW}" >&2
    exit 2
  fi
fi

VIEW_SCOPE_RAW="${VIEW_SCOPE:-all}"
VIEW_SCOPE="$(printf '%s' "${VIEW_SCOPE_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${VIEW_SCOPE}" in
  all|omomo) ;;
  *)
    echo "[ERROR] VIEW_SCOPE must be one of: all, omomo. Got: ${VIEW_SCOPE_RAW}" >&2
    exit 2
    ;;
esac

OMOMO_CLIP_REGEX="${OMOMO_CLIP_REGEX:-^sub}"
GROUP_INDEX="${GROUP_INDEX:-0}"
CLIP_LIST="${CLIP_LIST:-}"
NUM_ENVS="${NUM_ENVS:-4}"
VISER_PORT="${VISER_PORT:-18085}"
VISER_ENV_ID="${VISER_ENV_ID:-0}"
VISER_GRID_SPACING="${VISER_GRID_SPACING:-2.8}"
VISER_MULTI_ENV_COLS="${VISER_MULTI_ENV_COLS:-2}"
VISER_UPDATE_HZ="${VISER_UPDATE_HZ:-30}"
VISER_START_PAUSED="${VISER_START_PAUSED:-0}"
TRAINING_NAME="${TRAINING_NAME:-debug_box_replay}"
GPU_ID="${GPU_ID:-auto}"
DRY_RUN="${DRY_RUN:-0}"

if [[ ! "${NUM_ENVS}" =~ ^[0-9]+$ || "${NUM_ENVS}" == "0" ]]; then
  echo "[ERROR] NUM_ENVS must be a positive integer. Got: ${NUM_ENVS}" >&2
  exit 2
fi
if [[ ! "${VISER_MULTI_ENV_COLS}" =~ ^[0-9]+$ || "${VISER_MULTI_ENV_COLS}" == "0" ]]; then
  echo "[ERROR] VISER_MULTI_ENV_COLS must be a positive integer. Got: ${VISER_MULTI_ENV_COLS}" >&2
  exit 2
fi

REPLAY_SUBSET_DIR_DEFAULT="$(ogds_default_replay_subset_dir "${MOTION_SOURCE_ROOT}" "debug_${VIEW_SCOPE}")"
REPLAY_SUBSET_DIR="${REPLAY_SUBSET_DIR:-${REPLAY_SUBSET_DIR_DEFAULT}}"
OBJECT_MAP="${OBJECT_MAP:-${REPLAY_SUBSET_DIR}/_clip_object_urdf_map.json}"

TRAINING_HEADLESS_RAW="${HEADLESS:-False}"
case "$(printf '%s' "${TRAINING_HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TRAINING_HEADLESS="True"
    ;;
  0|false|no|off)
    TRAINING_HEADLESS="False"
    ;;
  *)
    echo "[ERROR] HEADLESS must be True/False/1/0, got: ${TRAINING_HEADLESS_RAW}" >&2
    exit 2
    ;;
esac

APP_HEADLESS_RAW="${ISAAC_APP_HEADLESS:-auto}"
case "$(printf '%s' "${APP_HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    HEADLESS_ENV=1
    ;;
  0|false|no|off)
    HEADLESS_ENV=0
    ;;
  auto|"")
    if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
      if [[ "${TRAINING_HEADLESS}" == "True" ]]; then
        HEADLESS_ENV=1
      else
        HEADLESS_ENV=0
      fi
    else
      HEADLESS_ENV=1
    fi
    ;;
  *)
    echo "[ERROR] ISAAC_APP_HEADLESS must be auto/True/False/1/0, got: ${APP_HEADLESS_RAW}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${MOTION_SOURCE_ROOT}" ]]; then
  echo "[ERROR] MOTION source root not found: ${MOTION_SOURCE_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MOTION_SOURCE_ROOT}/_clip_object_urdf_map.json" ]]; then
  echo "[ERROR] clip-object map missing under source root: ${MOTION_SOURCE_ROOT}" >&2
  exit 1
fi

mkdir -p "${REPLAY_SUBSET_DIR}"

BANK_INFO="$("${PYTHON_BIN}" - <<'PY' "${MOTION_SOURCE_ROOT}" "${REPLAY_SUBSET_DIR}" "${VIEW_SCOPE}" "${OMOMO_CLIP_REGEX}" "${GROUP_INDEX}" "${CLIP_LIST}" "${NUM_ENVS}"
import json
import math
import re
import sys
from pathlib import Path

motion_root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
view_scope = sys.argv[3].strip().lower()
clip_regex = sys.argv[4]
group_index_raw = sys.argv[5].strip()
clip_list_raw = sys.argv[6].strip()
group_size = int(sys.argv[7])

try:
    group_index = int(group_index_raw or "0")
except ValueError as exc:
    raise SystemExit(f"GROUP_INDEX must be an integer, got: {group_index_raw}") from exc
if group_index < 0:
    raise SystemExit(f"GROUP_INDEX must be >= 0, got: {group_index}")
if group_size <= 0:
    raise SystemExit(f"NUM_ENVS must be > 0, got: {group_size}")

map_path = motion_root / "_clip_object_urdf_map.json"
payload = json.loads(map_path.read_text(encoding="utf-8"))
clips_map = payload["clips"] if isinstance(payload, dict) and "clips" in payload else payload
if not isinstance(clips_map, dict) or not clips_map:
    raise SystemExit(f"Invalid clip map: {map_path}")

pattern = re.compile(clip_regex)
available = []
for clip_name in sorted(clips_map):
    clip_path = motion_root / f"{clip_name}.npz"
    if not clip_path.is_file():
        continue
    if view_scope == "omomo" and not pattern.search(clip_name):
        continue
    available.append(clip_name)

if not available:
    if view_scope == "omomo":
        raise SystemExit(
            f"No OMOMO clips matching /{clip_regex}/ found under {motion_root}"
        )
    raise SystemExit(f"No replayable clips found under {motion_root}")

if clip_list_raw:
    bank_clips = [item.strip() for item in clip_list_raw.split(",") if item.strip()]
    missing = [name for name in bank_clips if name not in clips_map or not (motion_root / f"{name}.npz").is_file()]
    if missing:
        raise SystemExit(f"Unknown or missing clip ids in CLIP_LIST: {missing}")
    if view_scope == "omomo":
        non_omomo = [name for name in bank_clips if not pattern.search(name)]
        if non_omomo:
            raise SystemExit(f"CLIP_LIST contains non-OMOMO clips while VIEW_SCOPE=omomo: {non_omomo}")
    total_groups = 1
    initial_visible = list(bank_clips[:group_size])
    replay_clips = list(bank_clips)
else:
    bank_clips = list(available)
    total_groups = int(math.ceil(len(bank_clips) / float(group_size)))
    if group_index >= total_groups:
        raise SystemExit(f"GROUP_INDEX {group_index} out of range for {len(bank_clips)} clips ({total_groups} groups)")
    start = group_index * group_size
    initial_visible = bank_clips[start : start + group_size]
    if not initial_visible:
        raise SystemExit(f"Group {group_index} resolved to an empty initial selection.")
    replay_clips = list(initial_visible)

out_dir.mkdir(parents=True, exist_ok=True)
for old_npz in out_dir.glob("*.npz"):
    old_npz.unlink()

subset_map = {}
for clip_name in replay_clips:
    src = motion_root / f"{clip_name}.npz"
    dst = out_dir / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    subset_map[clip_name] = clips_map[clip_name]

(out_dir / "_clip_object_urdf_map.json").write_text(
    json.dumps({"clips": subset_map}, indent=2, sort_keys=True),
    encoding="utf-8",
)

print(f"view_scope={view_scope}")
print(f"bank_count={len(bank_clips)}")
print(f"replay_count={len(replay_clips)}")
print(f"group_size={group_size}")
if clip_list_raw:
    print("mode=manual")
else:
    print(f"mode=group index={group_index}/{total_groups - 1}")
for idx, clip_name in enumerate(initial_visible):
    print(f"{idx}\t{clip_name}")
PY
)"

if [[ -z "${BANK_INFO}" ]]; then
  echo "[ERROR] No replay bank prepared." >&2
  exit 1
fi
if [[ ! -f "${OBJECT_MAP}" ]]; then
  echo "[ERROR] object map not found after subset generation: ${OBJECT_MAP}" >&2
  exit 1
fi

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  if [[ "${GPU_ID}" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      gpu_pick="$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2n | head -n1 | cut -d',' -f1 | xargs)"
      if [[ -n "${gpu_pick}" ]]; then
        export CUDA_VISIBLE_DEVICES="${gpu_pick}"
      fi
    fi
  else
    export CUDA_VISIBLE_DEVICES="${GPU_ID}"
  fi
fi

export PYTHONUNBUFFERED=1
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export HEADLESS="${HEADLESS_ENV}"
export VISER_MESH_SOURCE="${VISER_MESH_SOURCE:-sim}"
export VISER_MESH_MODE="${VISER_MESH_MODE:-both}"
export VISER_LOAD_URDF=0
export VISER_MULTI_ENV_COLS="${VISER_MULTI_ENV_COLS}"
export VISER_START_PAUSED="${VISER_START_PAUSED}"
export VISER_PLAY_RESTARTS_VISIBLE_REPLAY="${VISER_PLAY_RESTARTS_VISIBLE_REPLAY:-1}"
export VISER_RESET_RESTARTS_VISIBLE_REPLAY="${VISER_RESET_RESTARTS_VISIBLE_REPLAY:-1}"
export VISER_ENABLE_CLIP_GUI="${VISER_ENABLE_CLIP_GUI:-0}"
export VISER_ENABLE_CLIP_GROUP_GUI="${VISER_ENABLE_CLIP_GROUP_GUI:-0}"
export VISER_ENABLE_MANUAL_GUI="${VISER_ENABLE_MANUAL_GUI:-0}"
export VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS:-0}"
export VISER_SHOW_ENV_SEQUENCE_LABELS="${VISER_SHOW_ENV_SEQUENCE_LABELS:-1}"
export VISER_ENV_SEQUENCE_LABEL_HEIGHT="${VISER_ENV_SEQUENCE_LABEL_HEIGHT:-1.6}"
export VISER_CLIP_GROUP_SIZE="${VISER_CLIP_GROUP_SIZE:-${NUM_ENVS}}"
export VISER_INITIAL_CLIP_GROUP_INDEX="${VISER_INITIAL_CLIP_GROUP_INDEX:-0}"
export HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT="${HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT:-1}"
export HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_START="${HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_START:-0}"
export HOLOSOMA_REPLAY_KEEP_OPEN="${HOLOSOMA_REPLAY_KEEP_OPEN:-1}"

cmd=(
  "${PYTHON_BIN}" src/holosoma/holosoma/replay.py
  exp:g1-29dof-wbt-w-object-generalist
  randomization:disabled
  logger:disabled
  --training.name="${TRAINING_NAME}"
  --training.headless="${TRAINING_HEADLESS}"
  --training.debug=True
  --training.num-envs="${NUM_ENVS}"
  --training.enable-viser=True
  --training.viser-port="${VISER_PORT}"
  --training.viser-env-id="${VISER_ENV_ID}"
  --training.viser-env-count="${NUM_ENVS}"
  --training.viser-multi-env-spacing="${VISER_GRID_SPACING}"
  --training.viser-update-hz="${VISER_UPDATE_HZ}"
  --training.viser-sync-to-sim=True
  --training.viser-force-dt=True
  --training.viser-recenter=True
  --training.viser-show-scandots=False
  --command.setup-terms.motion-command.params.motion-config.motion-file "${REPLAY_SUBSET_DIR}"
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler False
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob 1.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend False
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s 0.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append False
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s 0.0
  --robot.object.object-urdf-path "${OBJECT_MAP}"
)

echo "[INFO] data_mode=${DATA_MODE}"
echo "[INFO] motion_root=${MOTION_SOURCE_ROOT}"
echo "[INFO] view_scope=${VIEW_SCOPE}"
echo "[INFO] replay_subset_dir=${REPLAY_SUBSET_DIR}"
echo "[INFO] object_map=${OBJECT_MAP}"
echo "[INFO] replay bank prepared:"
printf '%s\n' "${BANK_INFO}"
echo "[INFO] viser=http://localhost:${VISER_PORT}"
echo "[INFO] training_headless=${TRAINING_HEADLESS} isaac_app_headless=${HEADLESS_ENV}"
echo "[INFO] mesh_source=${VISER_MESH_SOURCE} mesh_mode=${VISER_MESH_MODE}"
echo "[INFO] env_sequence_labels=${VISER_SHOW_ENV_SEQUENCE_LABELS} label_height=${VISER_ENV_SEQUENCE_LABEL_HEIGHT}"
echo "[INFO] group_gui=${VISER_ENABLE_CLIP_GROUP_GUI} initial_group=${VISER_INITIAL_CLIP_GROUP_INDEX} group_size=${VISER_CLIP_GROUP_SIZE}"
echo "[INFO] force_round_robin_clip_assignment=${HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT}"
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$(printf '%s' "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
  exit 0
fi

"${cmd[@]}"
