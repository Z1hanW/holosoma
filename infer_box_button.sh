#!/usr/bin/env bash
set -euo pipefail

# Interactive inference for the contact-aware drop-button box policy.
#
# Default run:
#   https://wandb.ai/zihanw22/boxer/runs/d9m3z369
#
# Training contract for this run:
# - actor inputs: actor_obs_root_contact_aware, actor_obs_drop_button,
#   actor_obs_proprio_with_actions_no_linvel
# - actor_obs_drop_button is 0 before carry-end t2 and 1 from t2 to clip end.
#   In this launcher, Viser owns that scalar explicitly: it starts at 0, the
#   Drop Button sets it to 1, and reset sets it back to 0.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_URL_DEFAULT="${RUN_URL_DEFAULT:-https://wandb.ai/zihanw22/boxer/runs/d9m3z369}"
FALLBACK_MODEL_FILE_DEFAULT="${FALLBACK_MODEL_FILE_DEFAULT:-model_12500.pt}"
TRAINING_MOTION_DIR_DEFAULT="${TRAINING_MOTION_DIR_DEFAULT:-${SCRIPT_DIR}/outputs/motion_bank_success_box_0_92_0p3}"
TRAINING_OBJECT_MAP_DEFAULT="${TRAINING_OBJECT_MAP_DEFAULT:-${TRAINING_MOTION_DIR_DEFAULT}/_clip_object_urdf_map.json}"

usage() {
  cat <<EOF
Usage:
  bash infer_box_button.sh [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra infer_box_joystick.sh args...]

Defaults:
  checkpoint: ${RUN_URL_DEFAULT}
  motion_dir: ${TRAINING_MOTION_DIR_DEFAULT}
  object_map: ${TRAINING_OBJECT_MAP_DEFAULT}

Useful env vars:
  WANDB_MODEL_FILE              checkpoint file for W&B run URLs
  INFER_DATASET                 default: rollout-ref
  MOTION_DIR                    default: training 28-clip rollout-ref bank
  OBJECT_URDF                   default: training per-clip object map
  VISER_MANUAL_CONTROL_DEFAULT  default: 1
  VISER_DROP_BUTTON_DEFAULT     default: 0
  DRY_RUN=1                     print delegated command
EOF
}

is_checkpoint_ref() {
  local ref="${1:-}"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

extract_wandb_run_id() {
  local ref="$1"
  local clean_ref="${ref%%#*}"
  clean_ref="${clean_ref%%\?*}"
  [[ "${clean_ref}" == https://wandb.ai/*/runs/* ]] || return 1
  local trimmed="${clean_ref#https://wandb.ai/}"
  local parts=()
  IFS='/' read -r -a parts <<< "${trimmed}"
  [[ "${#parts[@]}" -ge 4 && "${parts[2]}" == "runs" && -n "${parts[3]}" ]] || return 1
  printf '%s\n' "${parts[3]}"
}

resolve_latest_remote_model_file() {
  local run_ref="$1"
  "${PYTHON_BIN}" - "${run_ref}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
sys.path = [
    entry
    for entry in sys.path
    if entry not in {"", "."} and (not entry or Path(entry).resolve() != repo_root)
]

try:
    import wandb
except Exception:
    sys.exit(0)

ref = sys.argv[1].split("?", 1)[0].split("#", 1)[0]
parts = ref.removeprefix("https://wandb.ai/").split("/")
if len(parts) < 4 or parts[2] != "runs":
    sys.exit(0)
entity, project, run_id = parts[0], parts[1], parts[3]
pattern = re.compile(r"^model_(\d+)\.pt$")
try:
    run = wandb.Api(timeout=30).run(f"{entity}/{project}/{run_id}")
except Exception:
    sys.exit(0)
best: tuple[int, str] | None = None
for file_obj in run.files():
    name = str(getattr(file_obj, "name", "") or "")
    match = pattern.match(name)
    if match is None:
        continue
    candidate = (int(match.group(1)), name)
    if best is None or candidate[0] > best[0]:
        best = candidate
if best is not None:
    print(best[1])
PY
}

case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

CKPT="${CKPT:-${CHECKPOINT:-}}"
if [[ $# -gt 0 && -z "${CKPT}" ]]; then
  if is_checkpoint_ref "$1"; then
    CKPT="$1"
    shift
  fi
fi
CKPT="${CKPT:-${RUN_URL_DEFAULT}}"

run_id="$(extract_wandb_run_id "${CKPT}" || true)"
if [[ "${run_id}" == "d9m3z369" && -z "${WANDB_MODEL_FILE:-}" ]]; then
  latest_model_file="$(resolve_latest_remote_model_file "${CKPT}")"
  export WANDB_MODEL_FILE="${latest_model_file:-${FALLBACK_MODEL_FILE_DEFAULT}}"
fi

export INFER_DATASET="${INFER_DATASET:-rollout-ref}"
export MOTION_DIR="${MOTION_DIR:-${TRAINING_MOTION_DIR_DEFAULT}}"
export OBJECT_URDF="${OBJECT_URDF:-${TRAINING_OBJECT_MAP_DEFAULT}}"
export DEPTH_PERCEPTION_PRESET="${DEPTH_PERCEPTION_PRESET:-checkpoint}"
export VISER_ENABLE_MANUAL_GUI="${VISER_ENABLE_MANUAL_GUI:-1}"
export VISER_ENABLE_MANUAL_ROOT_GUI="${VISER_ENABLE_MANUAL_ROOT_GUI:-1}"
export VISER_MANUAL_CONTROL_DEFAULT="${VISER_MANUAL_CONTROL_DEFAULT:-1}"
export VISER_ENABLE_DROP_BUTTON_GUI="${VISER_ENABLE_DROP_BUTTON_GUI:-1}"
export VISER_DROP_BUTTON_DEFAULT="${VISER_DROP_BUTTON_DEFAULT:-0}"
export VISER_SHOW_TARGET_KEYPOINTS="${VISER_SHOW_TARGET_KEYPOINTS:-0}"
export HOLOSOMA_DISABLE_AUTO_RESET="${HOLOSOMA_DISABLE_AUTO_RESET:-1}"
export HOLOSOMA_DISABLE_MOTION_END_RESET="${HOLOSOMA_DISABLE_MOTION_END_RESET:-1}"
export HOLOSOMA_DISABLE_CLIP_END_RESET="${HOLOSOMA_DISABLE_CLIP_END_RESET:-1}"

echo "[INFO] button_run=${RUN_URL_DEFAULT}"
echo "[INFO] checkpoint=${CKPT}"
if [[ -n "${WANDB_MODEL_FILE:-}" ]]; then
  echo "[INFO] wandb_model_file=${WANDB_MODEL_FILE}"
fi
echo "[INFO] drop_button_semantics=explicit_viser_0_until_pressed_1_until_reset"

exec bash "${SCRIPT_DIR}/infer_box_joystick.sh" depth "${CKPT}" "$@"
