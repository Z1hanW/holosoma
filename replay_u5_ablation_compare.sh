#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DEBUG_ROOT="${DEBUG_ROOT:-${SCRIPT_DIR}/data/ds_box_debug/debug_data/v1_named_sequences_20260502T054723Z}"
U5_MODE="${U5_MODE:-local_u5_v1_copy}"
REF_MODE="${REF_MODE:-current_train_data}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<EOF
Usage:
  bash replay_u5_ablation_compare.sh [replay.py extra args...]

Builds a two-env, 1-to-1 comparison replay bank:
  env0: ${U5_MODE}
  env1: ${REF_MODE}

Environment overrides:
  DEBUG_ROOT          Default: ${DEBUG_ROOT}
  U5_MODE             Default: ${U5_MODE}
  REF_MODE            Default: ${REF_MODE}
  CLIP_LIST           Optional comma-separated clip ids to compare.
  START_INDEX         Default: 0
  MAX_CLIPS           Optional max matched clips.
  HEADLESS            Default: False
  ISAAC_APP_HEADLESS  Default: auto
  VISER_PORT          Default: 18086
  VISER_START_PAUSED  Default: 0
  COMPARE_WORK_ROOT   Default: <repo>/data/ds_box_debug/replay_compare
  DRY_RUN             Default: 0
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

U5_ROOT="${DEBUG_ROOT}/${U5_MODE}"
REF_ROOT="${DEBUG_ROOT}/${REF_MODE}"
for path in "${U5_ROOT}" "${REF_ROOT}"; do
  if [[ ! -d "${path}" ]]; then
    echo "[ERROR] Missing motion mode directory: ${path}" >&2
    exit 2
  fi
  if [[ ! -f "${path}/_clip_object_urdf_map.json" ]]; then
    echo "[ERROR] Missing object map: ${path}/_clip_object_urdf_map.json" >&2
    exit 2
  fi
  if [[ ! -d "${path}/objects" ]]; then
    echo "[ERROR] Missing objects directory: ${path}/objects" >&2
    exit 2
  fi
done

START_INDEX="${START_INDEX:-0}"
MAX_CLIPS="${MAX_CLIPS:-}"
CLIP_LIST="${CLIP_LIST:-}"
COMPARE_WORK_ROOT="${COMPARE_WORK_ROOT:-${SCRIPT_DIR}/data/ds_box_debug/replay_compare}"
COMPARE_NAME="${COMPARE_NAME:-${U5_MODE}_vs_${REF_MODE}}"
COMPARE_BANK_DIR="${COMPARE_WORK_ROOT}/${COMPARE_NAME}/_paired_bank"
CLIPS_FILE="${COMPARE_WORK_ROOT}/${COMPARE_NAME}/_clips.txt"
VISER_PORT="${VISER_PORT:-18086}"
VISER_UPDATE_HZ="${VISER_UPDATE_HZ:-30}"
VISER_START_PAUSED="${VISER_START_PAUSED:-0}"
VISER_GRID_SPACING="${VISER_GRID_SPACING:-2.8}"
TRAINING_NAME_PREFIX="${TRAINING_NAME_PREFIX:-debug_${COMPARE_NAME}_1x1}"
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

mkdir -p "${COMPARE_BANK_DIR}"

"${PYTHON_BIN}" - <<'PY' "${U5_ROOT}" "${REF_ROOT}" "${COMPARE_BANK_DIR}" "${CLIPS_FILE}" "${CLIP_LIST}" "${START_INDEX}" "${MAX_CLIPS}" "${U5_MODE}" "${REF_MODE}"
import json
import shutil
import sys
from pathlib import Path

u5_root = Path(sys.argv[1]).resolve()
ref_root = Path(sys.argv[2]).resolve()
bank_dir = Path(sys.argv[3]).resolve()
clips_file = Path(sys.argv[4]).resolve()
clip_list_raw = sys.argv[5].strip()
start_index = int(sys.argv[6] or "0")
max_clips_raw = sys.argv[7].strip()
max_clips = int(max_clips_raw) if max_clips_raw else None
u5_label = sys.argv[8]
ref_label = sys.argv[9]

def load_map(root: Path) -> dict:
    payload = json.loads((root / "_clip_object_urdf_map.json").read_text(encoding="utf-8"))
    clip_map = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
    if not isinstance(clip_map, dict) or not clip_map:
        raise SystemExit(f"Invalid object map: {root / '_clip_object_urdf_map.json'}")
    return clip_map

u5_map = load_map(u5_root)
ref_map = load_map(ref_root)
u5_clips = {p.stem for p in u5_root.glob("*.npz")}
ref_clips = {p.stem for p in ref_root.glob("*.npz")}
available = sorted(u5_clips & ref_clips & set(u5_map) & set(ref_map))

if clip_list_raw:
    selected = [item.strip() for item in clip_list_raw.split(",") if item.strip()]
    missing = [name for name in selected if name not in available]
    if missing:
        raise SystemExit(f"CLIP_LIST contains clips without a complete pair: {missing}")
else:
    if start_index < 0:
        raise SystemExit(f"START_INDEX must be >= 0, got {start_index}")
    selected = available[start_index:]
    if max_clips is not None:
        selected = selected[:max_clips]

if not selected:
    raise SystemExit("No matched clips selected for comparison.")

for old_npz in bank_dir.glob("*.npz"):
    old_npz.unlink()
for old_link in ("objects_u5", "objects_ref", "objects"):
    target = bank_dir / old_link
    if target.exists() or target.is_symlink():
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

(bank_dir / "objects_u5").symlink_to(u5_root / "objects")
(bank_dir / "objects_ref").symlink_to(ref_root / "objects")
# Keep a conventional objects/ link for tools that expect it.
(bank_dir / "objects").symlink_to(u5_root / "objects")

paired_map = {}
for pair_idx, clip_name in enumerate(selected):
    entries = (
        ("0_u5", "u5", u5_root, u5_map, u5_label),
        ("1_ref", "ref", ref_root, ref_map, ref_label),
    )
    for side_prefix, side, root, clip_map, label in entries:
        paired_name = f"{pair_idx:04d}_{side_prefix}_{clip_name}"
        src = root / f"{clip_name}.npz"
        dst = bank_dir / f"{paired_name}.npz"
        dst.symlink_to(src)
        entry = dict(clip_map[clip_name])
        raw_urdf = str(entry.get("object_urdf_path", "")).strip()
        if raw_urdf and not Path(raw_urdf).is_absolute():
            # The paired bank has side-specific object roots; rewrite relative
            # object paths so each side keeps its own exact URDF.
            parts = Path(raw_urdf).parts
            if parts and parts[0] == "objects":
                entry["object_urdf_path"] = str(Path(f"objects_{side}", *parts[1:]))
            else:
                entry["object_urdf_path"] = str(Path(f"objects_{side}") / raw_urdf)
        entry["compare_source_mode"] = label
        entry["compare_source_clip"] = clip_name
        entry["compare_pair_index"] = pair_idx
        entry["compare_pair_side"] = side
        paired_map[paired_name] = entry

(bank_dir / "_clip_object_urdf_map.json").write_text(
    json.dumps({"clips": paired_map}, indent=2, sort_keys=True),
    encoding="utf-8",
)
clips_file.parent.mkdir(parents=True, exist_ok=True)
clips_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
print(f"selected_pairs={len(selected)}")
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
export HOLOSOMA_REPLAY_KEEP_OPEN="${HOLOSOMA_REPLAY_KEEP_OPEN:-1}"
export HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_ASSIGNMENT=1
export HOLOSOMA_FORCE_ROUND_ROBIN_CLIP_START=0
export HOLOSOMA_DISABLE_CLIP_END_RESET=1
export VISER_ENABLE_CLIP_GUI="${VISER_ENABLE_CLIP_GUI:-0}"
export VISER_ENABLE_CLIP_GROUP_GUI="${VISER_ENABLE_CLIP_GROUP_GUI:-1}"
export VISER_CLIP_GROUP_SIZE=2
export VISER_INITIAL_CLIP_GROUP_INDEX="${VISER_INITIAL_CLIP_GROUP_INDEX:-0}"
export VISER_SHOW_ENV_SEQUENCE_LABELS="${VISER_SHOW_ENV_SEQUENCE_LABELS:-1}"
export VISER_MULTI_ENV_COLS="${VISER_MULTI_ENV_COLS:-1}"

echo "[INFO] U5 mode: ${U5_MODE} (${U5_ROOT})"
echo "[INFO] Reference mode: ${REF_MODE} (${REF_ROOT})"
echo "[INFO] Paired bank: ${COMPARE_BANK_DIR}"
echo "[INFO] Clips:"
sed 's/^/[INFO]   /' "${CLIPS_FILE}"
echo "[INFO] Viser: http://localhost:${VISER_PORT}"
echo "[INFO] Use Group Playback Prev/Next to advance one matched pair at a time."

cmd=(
  "${PYTHON_BIN}" src/holosoma/holosoma/replay.py
  exp:g1-29dof-wbt-w-object-generalist
  randomization:disabled
  logger:disabled
  --training.name="${TRAINING_NAME_PREFIX}"
  --training.headless="${TRAINING_HEADLESS}"
  --training.debug=True
  --training.num-envs=2
  --training.enable-viser=True
  --training.viser-port="${VISER_PORT}"
  --training.viser-env-id=0
  --training.viser-env-count=2
  --training.viser-multi-env-spacing="${VISER_GRID_SPACING}"
  --training.viser-update-hz="${VISER_UPDATE_HZ}"
  --training.viser-sync-to-sim=True
  --training.viser-force-dt=True
  --training.viser-recenter=True
  --training.viser-show-scandots=False
  --command.setup-terms.motion-command.params.motion-config.motion-file "${COMPARE_BANK_DIR}"
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler False
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob 1.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend False
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s 0.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append False
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s 0.0
  --robot.object.object-urdf-path "${COMPARE_BANK_DIR}/_clip_object_urdf_map.json"
)

printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "$(printf '%s' "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
  echo "[INFO] DRY_RUN enabled; not launching replay."
  exit 0
fi

"${cmd[@]}" "$@"
