#!/usr/bin/env bash
set -euo pipefail

# AS drop-button distillation on the solid-object subset only.
#
# This does not create a filtered motion bank. It keeps the normal AS button
# source bank selection (default: success133, 133 clips) and asks the motion
# loader to include only these object categories:
#   box, bin, barrel, ball

usage() {
  cat <<'EOF'
Usage:
  bash distill_as_button_solid.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  CHECK_ONLY=1 bash distill_as_button_solid.sh
  bash distill_as_button_solid.sh --check-only

Allowed object categories:
  box, bin, barrel, ball

Behavior:
  Uses the normal distill_as_button.sh AS bank selection. By default that is the
  success133 teacher-rollout filtered AS bank, still loaded from the 133-clip
  source directory. This wrapper only passes an allowed-object-categories filter
  into MotionLoader, so non-solid categories are not sampled/loaded.

Useful env vars:
  SOLID_ALLOWED_OBJECT_CATEGORIES='["box","bin","barrel","ball"]'  optional subset of these four
  CHECK_ONLY=1               count matching clips in the selected source bank
  RESUME_FROM_BOX=1          optionally initialize from the box-button policy
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

CHECK_ONLY=${CHECK_ONLY:-0}
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
    --check-only|check-only|check_only)
      CHECK_ONLY=1
      shift
      ;;
    solid|solid-only|solid_only|box-bin-barrel-ball|box_bin_barrel_ball)
      shift
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

normalize_bool() {
  local name="$1"
  local value="$2"
  case "$(echo "${value}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on)
      echo 1
      ;;
    0|false|no|off|"")
      echo 0
      ;;
    *)
      echo "[ERROR] ${name} must be a boolean. Got: ${value}" >&2
      exit 2
      ;;
  esac
}

CHECK_ONLY="$(normalize_bool CHECK_ONLY "${CHECK_ONLY}")"

export AS_SUCCESS133_FINAL0P5="${AS_SUCCESS133_FINAL0P5:-1}"
export RESUME_FROM_BOX="${RESUME_FROM_BOX:-0}"
SOLID_ALLOWED_OBJECT_CATEGORIES=${SOLID_ALLOWED_OBJECT_CATEGORIES:-'["box","bin","barrel","ball"]'}
SOLID_ALLOWED_OBJECT_CATEGORIES=$(
  python3 - "${SOLID_ALLOWED_OBJECT_CATEGORIES}" <<'PY'
from __future__ import annotations

import json
import sys

allowed_universe = {"box", "bin", "barrel", "ball"}
aliases = {
    "boxes": "box",
    "cube": "box",
    "cubes": "box",
    "largebox": "box",
    "largeboxes": "box",
    "trash": "bin",
    "trashcan": "bin",
    "trashcans": "bin",
    "basket": "bin",
    "baskets": "bin",
    "bins": "bin",
    "barrels": "barrel",
    "sphere": "ball",
    "spheres": "ball",
    "balls": "ball",
}
try:
    raw = json.loads(sys.argv[1])
except Exception as exc:
    raise SystemExit(f"[ERROR] SOLID_ALLOWED_OBJECT_CATEGORIES must be a JSON list: {exc}")
if not isinstance(raw, list):
    raise SystemExit("[ERROR] SOLID_ALLOWED_OBJECT_CATEGORIES must be a JSON list")

normalized = []
for value in raw:
    category = aliases.get(str(value).strip().lower().replace("-", "_"), str(value).strip().lower())
    if not category:
        continue
    if category not in allowed_universe:
        raise SystemExit(
            "[ERROR] distill_as_button_solid.sh only allows box/bin/barrel/ball. "
            f"Got: {category}"
        )
    if category not in normalized:
        normalized.append(category)
if not normalized:
    raise SystemExit("[ERROR] SOLID_ALLOWED_OBJECT_CATEGORIES cannot be empty")
print(json.dumps(normalized))
PY
)

AS_SUCCESS133_BANK_NAME=${AS_SUCCESS133_BANK_NAME:-carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5}
SOLID_SOURCE_BANK=${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/${AS_SUCCESS133_BANK_NAME}"}
SOLID_SOURCE_MAP=${OMOMO_OBJECT_MAP:-"${SOLID_SOURCE_BANK}/_clip_object_urdf_map.json"}

if [[ "${CHECK_ONLY}" == "1" ]]; then
  python3 - "${SOLID_SOURCE_BANK}" "${SOLID_SOURCE_MAP}" "${SOLID_ALLOWED_OBJECT_CATEGORIES}" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

motion_dir = Path(sys.argv[1]).expanduser().resolve()
map_path = Path(sys.argv[2]).expanduser().resolve()
allowed_raw = sys.argv[3]

if not motion_dir.is_dir():
    raise SystemExit(f"[ERROR] Motion source bank does not exist: {motion_dir}")
if not map_path.is_file():
    raise SystemExit(f"[ERROR] Object map does not exist: {map_path}")

allowed = set(json.loads(allowed_raw))
payload = json.loads(map_path.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clips, dict):
    raise SystemExit(f"[ERROR] Invalid object map: {map_path}")


def category_for(clip_id: str, entry: object) -> str:
    parts = [clip_id]
    if isinstance(entry, dict):
        for key in ("object_name", "object_urdf_path", "object_mesh_path", "object_category", "category", "object_type"):
            value = str(entry.get(key, "")).strip()
            if value:
                parts.append(value)
    else:
        parts.append(str(entry))
    raw = " ".join(parts).lower()
    if "barrel" in raw:
        return "barrel"
    if "bin" in raw or "trash" in raw or "basket" in raw:
        return "bin"
    if "ball" in raw or "sphere" in raw:
        return "ball"
    if "box" in raw or "cube" in raw or "largebox" in raw:
        return "box"
    return "other"


counts = Counter()
missing_npz = []
for clip_id, entry in clips.items():
    category = category_for(clip_id, entry)
    counts[category] += 1
    if category in allowed and not (motion_dir / f"{clip_id}.npz").is_file():
        missing_npz.append(clip_id)

if missing_npz:
    preview = ", ".join(missing_npz[:20])
    raise SystemExit(f"[ERROR] Allowed clips missing .npz files: {preview}")

selected = sum(counts[category] for category in allowed)
print(f"[INFO] source_motion_dir={motion_dir}")
print(f"[INFO] source_object_map={map_path}")
print(f"[INFO] total_clips={len(clips)} selected_solid_clips={selected}")
print("[INFO] category_counts=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))
PY
  exit 0
fi

export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_button_solid_box_bin_barrel_ball}"
export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_button_solid_box_bin_barrel_ball_depth}"
export PPO_START_EPOCH="${PPO_START_EPOCH:-0}"
export PPO_START_COEFF="${PPO_START_COEFF:-0.01}"
export PPO_TARGET_COEFF="${PPO_TARGET_COEFF:-0.9}"
export PPO_SCHEDULE_STEP_EPOCHS="${PPO_SCHEDULE_STEP_EPOCHS:-1000}"
export DAGGER_END_EPOCH="${DAGGER_END_EPOCH:-9000}"
export DAGGER_LOSS_COEF="${DAGGER_LOSS_COEF:-1.0}"
export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_solid_box_bin_barrel_ball_sparse_root_ppo001_to09_9k_contact_drop_button}"
export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS drop-button distillation restricted by MotionLoader to solid object categories box/bin/barrel/ball while keeping the normal 133-clip success-filtered source bank. PPO+DAgger are active from iteration 0; PPO coeff starts at 0.01, staircase-updates every 1000 iterations, and reaches 0.9 at iteration 9000, so the effective DAgger weight decreases from 0.99 to 0.1.}"

echo "[INFO] solid_allowed_object_categories=${SOLID_ALLOWED_OBJECT_CATEGORIES}"
echo "[INFO] source_bank_selection=distill_as_button default success133 unless overridden"
echo "[INFO] resume_from_box=${RESUME_FROM_BOX}"
echo "[INFO] solid_ppo_schedule=${PPO_START_EPOCH}->${DAGGER_END_EPOCH} start=${PPO_START_COEFF} target=${PPO_TARGET_COEFF} step_epochs=${PPO_SCHEDULE_STEP_EPOCHS} dagger_loss_coef=${DAGGER_LOSS_COEF}"

exec bash "${SCRIPT_DIR}/distill_as_button.sh" "${POSITIONAL[@]}" \
  --command.setup-terms.motion-command.params.motion-config.allowed-object-categories="${SOLID_ALLOWED_OBJECT_CATEGORIES}"
