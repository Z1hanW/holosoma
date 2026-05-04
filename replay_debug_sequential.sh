#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEBUG_ROOT="${DEBUG_ROOT:-${SCRIPT_DIR}/data/ds_box_debug/debug_data/v1_named_sequences_20260502T054723Z}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<EOF
Usage:
  bash replay_debug_sequential.sh <mode> [replay.py extra args...]
  bash replay_debug_sequential.sh --list

Environment overrides:
  DEBUG_ROOT        Default: ${DEBUG_ROOT}
  CLIP_LIST         Optional comma-separated clip ids to replay.
  START_INDEX       Default: 0
  MAX_CLIPS         Optional max number of clips to replay.
  NUM_ENVS          Default: 1, must stay 1 for sequential replay.
  HEADLESS          Default: False
  ISAAC_APP_HEADLESS Default: auto
  VISER_PORT        Default: 18085
  VISER_START_PAUSED Default: 0
  REPLAY_WORK_ROOT  Default: <repo>/data/ds_box_debug/replay_sequential
  DRY_RUN           Default: 0
EOF
}

list_modes() {
  find "${DEBUG_ROOT}" -mindepth 1 -maxdepth 1 -type d -print \
    | while IFS= read -r dir; do
        if find "${dir}" -maxdepth 1 -name '*.npz' -print -quit | grep -q .; then
          basename "${dir}"
        fi
      done \
    | sort
}

if [[ ! -d "${DEBUG_ROOT}" ]]; then
  echo "[ERROR] Debug data root does not exist: ${DEBUG_ROOT}" >&2
  exit 2
fi

if [[ "$#" -lt 1 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  echo
  echo "Available modes:"
  list_modes | sed 's/^/  /'
  exit 0
fi

if [[ "${1}" == "--list" ]]; then
  list_modes
  exit 0
fi

MODE="$1"
shift
MOTION_SOURCE_ROOT="${DEBUG_ROOT}/${MODE}"
SOURCE_OBJECT_MAP="${MOTION_SOURCE_ROOT}/_clip_object_urdf_map.json"

if [[ ! -d "${MOTION_SOURCE_ROOT}" ]]; then
  echo "[ERROR] Unknown debug mode: ${MODE}" >&2
  list_modes | sed 's/^/[ERROR]   /' >&2
  exit 2
fi
if [[ ! -f "${SOURCE_OBJECT_MAP}" ]]; then
  echo "[ERROR] Missing object map: ${SOURCE_OBJECT_MAP}" >&2
  exit 2
fi
if [[ ! -d "${MOTION_SOURCE_ROOT}/objects" ]]; then
  echo "[ERROR] Missing objects directory: ${MOTION_SOURCE_ROOT}/objects" >&2
  exit 2
fi

NUM_ENVS="${NUM_ENVS:-1}"
if [[ "${NUM_ENVS}" != "1" ]]; then
  echo "[ERROR] sequential replay requires NUM_ENVS=1, got ${NUM_ENVS}" >&2
  exit 2
fi

START_INDEX="${START_INDEX:-0}"
MAX_CLIPS="${MAX_CLIPS:-}"
CLIP_LIST="${CLIP_LIST:-}"
REPLAY_WORK_ROOT="${REPLAY_WORK_ROOT:-${SCRIPT_DIR}/data/ds_box_debug/replay_sequential}"
REPLAY_MODE_ROOT="${REPLAY_WORK_ROOT}/${MODE}"
VISER_PORT="${VISER_PORT:-18085}"
VISER_UPDATE_HZ="${VISER_UPDATE_HZ:-30}"
VISER_START_PAUSED="${VISER_START_PAUSED:-0}"
VISER_GRID_SPACING="${VISER_GRID_SPACING:-2.8}"
TRAINING_NAME_PREFIX="${TRAINING_NAME_PREFIX:-debug_${MODE}_sequential}"
DRY_RUN="${DRY_RUN:-0}"

TRAINING_HEADLESS_RAW="${HEADLESS:-False}"
case "$(printf '%s' "${TRAINING_HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) TRAINING_HEADLESS=True ;;
  0|false|no|off) TRAINING_HEADLESS=False ;;
  *)
    echo "[ERROR] HEADLESS must be True/False/1/0, got: ${TRAINING_HEADLESS_RAW}" >&2
    exit 2
    ;;
esac

APP_HEADLESS_RAW="${ISAAC_APP_HEADLESS:-auto}"
case "$(printf '%s' "${APP_HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on) HEADLESS_ENV=1 ;;
  0|false|no|off) HEADLESS_ENV=0 ;;
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

CLIPS_FILE="${REPLAY_MODE_ROOT}/_clips.txt"
mkdir -p "${REPLAY_MODE_ROOT}"

"${PYTHON_BIN}" - <<'PY' "${MOTION_SOURCE_ROOT}" "${REPLAY_MODE_ROOT}" "${CLIPS_FILE}" "${CLIP_LIST}" "${START_INDEX}" "${MAX_CLIPS}"
import json
import shutil
import sys
from pathlib import Path

motion_root = Path(sys.argv[1]).resolve()
out_root = Path(sys.argv[2]).resolve()
clips_file = Path(sys.argv[3]).resolve()
clip_list_raw = sys.argv[4].strip()
start_index = int(sys.argv[5] or "0")
max_clips_raw = sys.argv[6].strip()
max_clips = int(max_clips_raw) if max_clips_raw else None

payload = json.loads((motion_root / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))
clip_map = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clip_map, dict) or not clip_map:
    raise SystemExit(f"Invalid object map: {motion_root / '_clip_object_urdf_map.json'}")

available = sorted(
    (p.stem for p in motion_root.glob("*.npz")),
    key=lambda name: (name.startswith("sub"), name),
)
available = [name for name in available if name in clip_map]

if clip_list_raw:
    selected = [item.strip() for item in clip_list_raw.split(",") if item.strip()]
    missing = [name for name in selected if name not in available]
    if missing:
        raise SystemExit(f"CLIP_LIST contains unknown clips: {missing}")
else:
    if start_index < 0:
        raise SystemExit(f"START_INDEX must be >= 0, got {start_index}")
    selected = available[start_index:]
    if max_clips is not None:
        selected = selected[:max_clips]

if not selected:
    raise SystemExit("No clips selected for sequential replay.")

objects_src = motion_root / "objects"
for index, clip_name in enumerate(selected):
    clip_dir = out_root / f"{index:04d}_{clip_name}"
    clip_dir.mkdir(parents=True, exist_ok=True)

    for old_npz in clip_dir.glob("*.npz"):
        old_npz.unlink()
    dst_npz = clip_dir / f"{clip_name}.npz"
    if dst_npz.exists() or dst_npz.is_symlink():
        dst_npz.unlink()
    dst_npz.symlink_to(motion_root / f"{clip_name}.npz")

    objects_dst = clip_dir / "objects"
    if objects_dst.exists() or objects_dst.is_symlink():
        if objects_dst.is_symlink() or objects_dst.is_file():
            objects_dst.unlink()
        else:
            shutil.rmtree(objects_dst)
    objects_dst.symlink_to(objects_src)

    (clip_dir / "_clip_object_urdf_map.json").write_text(
        json.dumps({"clips": {clip_name: clip_map[clip_name]}}, indent=2, sort_keys=True),
        encoding="utf-8",
    )

clips_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
print(f"selected_count={len(selected)}")
for index, clip_name in enumerate(selected):
    print(f"{index}\t{clip_name}")
PY

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" || -z "${CUDA_VISIBLE_DEVICES}" ]]; then
  GPU_ID="${GPU_ID:-auto}"
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
export VISER_START_PAUSED="${VISER_START_PAUSED}"
export HOLOSOMA_REPLAY_KEEP_OPEN=0
export HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT=1
export HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_START=0

echo "[INFO] Sequential replay mode: ${MODE}"
echo "[INFO] Source motion root: ${MOTION_SOURCE_ROOT}"
echo "[INFO] Work root: ${REPLAY_MODE_ROOT}"
echo "[INFO] Clips:"
sed 's/^/[INFO]   /' "${CLIPS_FILE}"
echo "[INFO] NUM_ENVS=1 training.headless=${TRAINING_HEADLESS} isaac_app_headless=${HEADLESS_ENV}"
echo "[INFO] Viser: http://localhost:${VISER_PORT}"

if [[ "$(printf '%s' "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
  echo "[INFO] DRY_RUN enabled; not launching replay."
  exit 0
fi

SELECTED_BANK_DIR="${REPLAY_MODE_ROOT}/_selected_bank"
SELECTED_OBJECT_MAP="${SELECTED_BANK_DIR}/_clip_object_urdf_map.json"
"${PYTHON_BIN}" - <<'PY' "${MOTION_SOURCE_ROOT}" "${SELECTED_BANK_DIR}" "${CLIPS_FILE}"
import json
import shutil
import sys
from pathlib import Path

motion_root = Path(sys.argv[1]).resolve()
bank_dir = Path(sys.argv[2]).resolve()
clips_file = Path(sys.argv[3]).resolve()
clip_names = [line.strip() for line in clips_file.read_text(encoding="utf-8").splitlines() if line.strip()]

payload = json.loads((motion_root / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))
clip_map = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clip_map, dict):
    raise SystemExit(f"Invalid object map: {motion_root / '_clip_object_urdf_map.json'}")

bank_dir.mkdir(parents=True, exist_ok=True)
for old_npz in bank_dir.glob("*.npz"):
    old_npz.unlink()
objects_dst = bank_dir / "objects"
if objects_dst.exists() or objects_dst.is_symlink():
    if objects_dst.is_symlink() or objects_dst.is_file():
        objects_dst.unlink()
    else:
        shutil.rmtree(objects_dst)
objects_dst.symlink_to(motion_root / "objects")

selected_map = {}
for clip_name in clip_names:
    src = motion_root / f"{clip_name}.npz"
    if not src.is_file():
        raise SystemExit(f"Missing clip file: {src}")
    if clip_name not in clip_map:
        raise SystemExit(f"Missing clip map entry: {clip_name}")
    dst = bank_dir / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    selected_map[clip_name] = clip_map[clip_name]

(bank_dir / "_clip_object_urdf_map.json").write_text(
    json.dumps({"clips": selected_map}, indent=2, sort_keys=True),
    encoding="utf-8",
)
print(f"[INFO] Selected single-window bank: {bank_dir} ({len(clip_names)} clips)")
PY

export HOLOSOMA_REPLAY_SEQUENTIAL_CLIPS=1
export HOLOSOMA_REPLAY_SEQUENTIAL_START_INDEX=0
export HOLOSOMA_REPLAY_SEQUENTIAL_MAX_CLIPS="${MAX_CLIPS}"
export HOLOSOMA_DISABLE_CLIP_END_RESET=1

cmd=(
  "${PYTHON_BIN}" src/holosoma/holosoma/replay.py
  exp:g1-29dof-wbt-w-object-generalist
  randomization:disabled
  logger:disabled
  --training.name="${TRAINING_NAME_PREFIX}"
  --training.headless="${TRAINING_HEADLESS}"
  --training.debug=True
  --training.num-envs=1
  --training.enable-viser=True
  --training.viser-port="${VISER_PORT}"
  --training.viser-env-id=0
  --training.viser-env-count=1
  --training.viser-multi-env-spacing="${VISER_GRID_SPACING}"
  --training.viser-update-hz="${VISER_UPDATE_HZ}"
  --training.viser-sync-to-sim=True
  --training.viser-force-dt=True
  --training.viser-recenter=True
  --training.viser-show-scandots=False
  --command.setup-terms.motion-command.params.motion-config.motion-file "${SELECTED_BANK_DIR}"
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler False
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob 1.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend False
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s 0.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append False
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s 0.0
  --robot.object.object-urdf-path "${SELECTED_OBJECT_MAP}"
)

echo "[INFO] Replaying all selected clips in one window/process."
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}" "$@"

echo "[INFO] Sequential replay complete."
