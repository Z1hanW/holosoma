#!/usr/bin/env bash
set -euo pipefail

# AS drop-button distillation on the solid-object subset only.
#
# This prepares a filtered motion bank view before launch. The simulator object
# bank and MotionLoader must see the same URDF set, otherwise fixed
# env-to-clip assignment will fail for single-slot AS training.
#
# The default source prefers the CoRL solid80 bank produced by cp_corl.sh when
# it exists locally. Otherwise it falls back to the teacher-rollout success155
# final0p5 primitive-proj bank and filters it down to solid clips:
#   strict success_contact_and_final_position
#   box/bin/barrel/ball only
#   excludes scale__any_bin_8, box_21, box_39 falldown/suspect clips

usage() {
  cat <<'EOF'
Usage:
  bash distill_as_button_solid.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  bash distill_as_button_solid.sh --resume-from-box [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  CHECK_ONLY=1 bash distill_as_button_solid.sh
  bash distill_as_button_solid.sh --check-only

Allowed object categories:
  box, bin, barrel, ball

Behavior:
  Prefers the repo-local CoRL solid80 bank copied by cp_corl.sh. If that bank
  is unavailable, uses the normal distill_as_button.sh AS bank selection as the
  source, then creates a repo-local solid-only symlink bank and launches from it.
  This keeps simulator object assignment and MotionLoader clip filtering
  consistent.

Useful env vars:
  SOLID_ALLOWED_OBJECT_CATEGORIES='["box","bin","barrel","ball"]'  optional subset of these four
  SOLID_CLIP_LIST=<file>      optional one-clip-id-per-line allowlist; default is clean80
  SOLID_TARGET_BANK_NAME=<name>  override generated filtered bank name
  CORL_SOLID80_BANK_NAME=<name>  override preferred cp_corl.sh bank name
  CHECK_ONLY=1               count matching clips in the selected source bank
  RESUME_FROM_BOX=1          initialize policy weights from box-button; default d9m3z369/model_17000.pt
  BOX_RESUME_CKPT=<checkpoint>  override the box policy initializer
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
    --resume-from-box|--resume_from_box|resume-from-box|resume_from_box|resume-from-box-button|resume_from_box_button|init-box-button|init_box_button)
      RESUME_FROM_BOX=1
      shift
      ;;
    --no-resume-from-box|--no_resume_from_box|no-resume-from-box|no_resume_from_box)
      RESUME_FROM_BOX=0
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
RESUME_FROM_BOX="$(normalize_bool RESUME_FROM_BOX "${RESUME_FROM_BOX:-0}")"

export AS_SUCCESS133_FINAL0P5="${AS_SUCCESS133_FINAL0P5:-1}"
export RESUME_FROM_BOX
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  DEFAULT_BOX_RESUME_RUN=${DEFAULT_BOX_RESUME_RUN:-"https://wandb.ai/zihanw22/boxer/runs/d9m3z369"}
  DEFAULT_BOX_RESUME_MODEL_FILE=${DEFAULT_BOX_RESUME_MODEL_FILE:-model_17000.pt}
  BOX_RESUME_MODEL_FILE=${BOX_RESUME_MODEL_FILE:-${DEFAULT_BOX_RESUME_MODEL_FILE}}
  DEFAULT_BOX_RESUME_CHECKPOINT=${DEFAULT_BOX_RESUME_CHECKPOINT:-"${DEFAULT_BOX_RESUME_RUN}/files/${BOX_RESUME_MODEL_FILE}"}
  BOX_RESUME_CKPT=${BOX_RESUME_CKPT:-${RESUME_FROM_BOX_CKPT:-${DEFAULT_BOX_RESUME_CHECKPOINT}}}
  export DEFAULT_BOX_RESUME_RUN
  export DEFAULT_BOX_RESUME_MODEL_FILE
  export BOX_RESUME_MODEL_FILE
  export DEFAULT_BOX_RESUME_CHECKPOINT
  export BOX_RESUME_CKPT
fi
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

DEFAULT_AS_SUCCESS155_BANK_NAME="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_primitiveproj"
CORL_SOLID80_BANK_NAME=${CORL_SOLID80_BANK_NAME:-"${DEFAULT_AS_SUCCESS155_BANK_NAME}_solid80_clean_box_bin_barrel_ball"}
USER_SET_AS_SUCCESS133_BANK_NAME=${AS_SUCCESS133_BANK_NAME+x}
USER_SET_OMOMO_DATA_DIR=${OMOMO_DATA_DIR+x}
AS_SUCCESS133_BANK_NAME=${AS_SUCCESS133_BANK_NAME:-"${DEFAULT_AS_SUCCESS155_BANK_NAME}"}
if [[ -z "${USER_SET_AS_SUCCESS133_BANK_NAME}" && -z "${USER_SET_OMOMO_DATA_DIR}" ]]; then
  CORL_SOLID80_BANK="${SCRIPT_DIR}/data/ds_as_data/${CORL_SOLID80_BANK_NAME}"
  if [[ -d "${CORL_SOLID80_BANK}" ]]; then
    AS_SUCCESS133_BANK_NAME="${CORL_SOLID80_BANK_NAME}"
  fi
fi
SOLID_SOURCE_BANK=${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/${AS_SUCCESS133_BANK_NAME}"}
SOLID_SOURCE_MAP=${OMOMO_OBJECT_MAP:-"${SOLID_SOURCE_BANK}/_clip_object_urdf_map.json"}
SOLID_CONTACT_EXPORT_NAME=${SOLID_CONTACT_EXPORT_NAME:-contact_export_from_teacher_success133_final0p5}
DEFAULT_SOLID_CLIP_LIST="${SOLID_SOURCE_BANK}/clean80_strict_success_solid_no_falldown_clips.txt"
if [[ -z "${SOLID_CLIP_LIST:-}" && -f "${DEFAULT_SOLID_CLIP_LIST}" ]]; then
  SOLID_CLIP_LIST="${DEFAULT_SOLID_CLIP_LIST}"
fi
SOLID_CLIP_LIST=${SOLID_CLIP_LIST:-}
if [[ -n "${SOLID_CLIP_LIST}" ]]; then
  SOLID_TARGET_BANK_NAME=${SOLID_TARGET_BANK_NAME:-"${AS_SUCCESS133_BANK_NAME}_solid80_clean_box_bin_barrel_ball"}
fi
SOLID_TARGET_BANK_NAME=${SOLID_TARGET_BANK_NAME:-}

if [[ "${CHECK_ONLY}" == "1" ]]; then
  python3 - "${SOLID_SOURCE_BANK}" "${SOLID_SOURCE_MAP}" "${SOLID_ALLOWED_OBJECT_CATEGORIES}" "${SOLID_CLIP_LIST}" <<'PY'
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

motion_dir = Path(sys.argv[1]).expanduser().resolve()
map_path = Path(sys.argv[2]).expanduser().resolve()
allowed_raw = sys.argv[3]
clip_list_raw = sys.argv[4].strip()

if not motion_dir.is_dir():
    raise SystemExit(f"[ERROR] Motion source bank does not exist: {motion_dir}")
if not map_path.is_file():
    raise SystemExit(f"[ERROR] Object map does not exist: {map_path}")
if clip_list_raw:
    clip_list_path = Path(clip_list_raw).expanduser().resolve()
    if not clip_list_path.is_file():
        raise SystemExit(f"[ERROR] SOLID_CLIP_LIST does not exist: {clip_list_path}")
    allowed_clip_ids = {
        line.strip()
        for line in clip_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not allowed_clip_ids:
        raise SystemExit(f"[ERROR] SOLID_CLIP_LIST is empty: {clip_list_path}")
else:
    clip_list_path = None
    allowed_clip_ids = None

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
                if key.endswith("_path"):
                    path = Path(value)
                    parts.extend([path.name, path.stem])
                else:
                    parts.append(value)
    else:
        path = Path(str(entry).strip())
        parts.extend([path.name, path.stem])
    raw = " ".join(parts).lower().replace("-", "_")
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
selected = []
list_missing_from_map = set(allowed_clip_ids or ())
for clip_id, entry in clips.items():
    category = category_for(clip_id, entry)
    counts[category] += 1
    if allowed_clip_ids is not None:
        if clip_id in allowed_clip_ids:
            list_missing_from_map.discard(clip_id)
        else:
            continue
    if category not in allowed:
        continue
    if not (motion_dir / f"{clip_id}.npz").is_file():
        missing_npz.append(clip_id)
        continue
    selected.append(clip_id)

if list_missing_from_map:
    preview = ", ".join(sorted(list_missing_from_map)[:20])
    raise SystemExit(f"[ERROR] SOLID_CLIP_LIST contains clips missing from object map: {preview}")

if missing_npz:
    preview = ", ".join(missing_npz[:20])
    raise SystemExit(f"[ERROR] Allowed clips missing .npz files: {preview}")

print(f"[INFO] source_motion_dir={motion_dir}")
print(f"[INFO] source_object_map={map_path}")
if clip_list_path is not None:
    print(f"[INFO] solid_clip_list={clip_list_path}")
print(f"[INFO] total_clips={len(clips)} selected_solid_clips={len(selected)}")
print("[INFO] category_counts=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))
PY
  exit 0
fi

SOLID_PREP_OUTPUT=$(
  python3 - "${SOLID_SOURCE_BANK}" "${SOLID_SOURCE_MAP}" "${SOLID_ALLOWED_OBJECT_CATEGORIES}" "${SOLID_CONTACT_EXPORT_NAME}" "${SOLID_CLIP_LIST}" "${SOLID_TARGET_BANK_NAME}" <<'PY'
from __future__ import annotations

import json
import os
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

source_bank = Path(sys.argv[1]).expanduser().resolve()
source_map = Path(sys.argv[2]).expanduser().resolve()
allowed = set(json.loads(sys.argv[3]))
contact_export_name = sys.argv[4].strip() or "contact_export_from_teacher_success133_final0p5"
clip_list_raw = sys.argv[5].strip()
target_bank_name_raw = sys.argv[6].strip()

if not source_bank.is_dir():
    raise SystemExit(f"[ERROR] Solid source bank does not exist: {source_bank}")
if not source_map.is_file():
    raise SystemExit(f"[ERROR] Solid source object map does not exist: {source_map}")
if clip_list_raw:
    clip_list_path = Path(clip_list_raw).expanduser().resolve()
    if not clip_list_path.is_file():
        raise SystemExit(f"[ERROR] SOLID_CLIP_LIST does not exist: {clip_list_path}")
    allowed_clip_ids = {
        line.strip()
        for line in clip_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not allowed_clip_ids:
        raise SystemExit(f"[ERROR] SOLID_CLIP_LIST is empty: {clip_list_path}")
else:
    clip_list_path = None
    allowed_clip_ids = None

payload = json.loads(source_map.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clips, dict) or not clips:
    raise SystemExit(f"[ERROR] Invalid object map: {source_map}")


def category_for(clip_id: str, entry: object) -> str:
    parts = [clip_id]
    if isinstance(entry, dict):
        for key in ("object_name", "object_urdf_path", "object_mesh_path", "object_category", "category", "object_type"):
            value = str(entry.get(key, "")).strip()
            if not value:
                continue
            if key.endswith("_path"):
                path = Path(value)
                parts.extend([path.name, path.stem])
            else:
                parts.append(value)
    else:
        path = Path(str(entry).strip())
        parts.extend([path.name, path.stem])
    raw = " ".join(parts).lower().replace("-", "_")
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
selected: dict[str, object] = {}
missing_npz: list[str] = []
list_missing_from_map = set(allowed_clip_ids or ())
for clip_id, entry in clips.items():
    category = category_for(clip_id, entry)
    counts[category] += 1
    if allowed_clip_ids is not None:
        if clip_id in allowed_clip_ids:
            list_missing_from_map.discard(clip_id)
        else:
            continue
    if category not in allowed:
        continue
    source_npz = source_bank / f"{clip_id}.npz"
    if not source_npz.is_file():
        missing_npz.append(clip_id)
        continue
    selected[clip_id] = entry

if list_missing_from_map:
    preview = ", ".join(sorted(list_missing_from_map)[:20])
    raise SystemExit(f"[ERROR] SOLID_CLIP_LIST contains clips missing from object map: {preview}")
if missing_npz:
    preview = ", ".join(missing_npz[:20])
    raise SystemExit(f"[ERROR] Allowed solid clips missing .npz files: {preview}")
if not selected:
    raise SystemExit(f"[ERROR] No clips matched allowed solid categories {sorted(allowed)} in {source_bank}")

slug = "_".join(category for category in ("box", "bin", "barrel", "ball") if category in allowed)
safe_source_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", source_bank.name).strip("_")
if target_bank_name_raw:
    target_bank = source_bank.parent / target_bank_name_raw
else:
    target_bank = source_bank.parent / f"{safe_source_name}_solid_{slug}"
marker = target_bank / ".generated_by_distill_as_button_solid"

if target_bank.exists():
    if not marker.exists():
        raise SystemExit(
            f"[ERROR] Refusing to overwrite non-generated solid bank: {target_bank}. "
            "Remove it manually or choose another source bank."
        )
    for child in target_bank.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
else:
    target_bank.mkdir(parents=True)

for clip_id in sorted(selected):
    source_npz = (source_bank / f"{clip_id}.npz").resolve()
    target_npz = target_bank / f"{clip_id}.npz"
    target_npz.symlink_to(os.path.relpath(source_npz, start=target_bank))

filtered_payload = {"clips": {clip_id: selected[clip_id] for clip_id in sorted(selected)}}
(target_bank / "_clip_object_urdf_map.json").write_text(
    json.dumps(filtered_payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

for metadata_name in ("teacher_export_summary.json", "teacher_export_summary.csv", "source_teacher_export.txt"):
    source_metadata = source_bank / metadata_name
    if source_metadata.exists():
        (target_bank / metadata_name).symlink_to(os.path.relpath(source_metadata.resolve(), start=target_bank))

source_contact_root_raw = os.environ.get("AS_CONTACT_EXPORT_ROOT", "").strip()
source_contact_root = Path(source_contact_root_raw).expanduser()
if not source_contact_root_raw:
    source_contact_root = source_bank / contact_export_name
source_contact_root = source_contact_root.resolve()
if not source_contact_root.is_dir():
    raise SystemExit(f"[ERROR] Missing contact export root for solid bank: {source_contact_root}")
target_contact_root = target_bank / contact_export_name
target_contact_root.symlink_to(os.path.relpath(source_contact_root, start=target_bank))

marker.write_text(
    "generated by distill_as_button_solid.sh\n"
    f"source_bank={source_bank}\n"
    f"allowed={json.dumps(sorted(allowed))}\n"
    f"clip_list={clip_list_path if clip_list_path is not None else ''}\n"
    f"selected={len(selected)}\n",
    encoding="utf-8",
)

print(f"SOLID_BANK_NAME={target_bank.name}")
print(f"SOLID_BANK_DIR={target_bank}")
print(f"SOLID_OBJECT_MAP={target_bank / '_clip_object_urdf_map.json'}")
print(f"SOLID_SELECTED_CLIP_COUNT={len(selected)}")
print("SOLID_CATEGORY_COUNTS=" + ",".join(f"{key}:{counts[key]}" for key in sorted(counts)))
PY
)

while IFS='=' read -r key value; do
  case "${key}" in
    SOLID_BANK_NAME|SOLID_BANK_DIR|SOLID_OBJECT_MAP|SOLID_SELECTED_CLIP_COUNT|SOLID_CATEGORY_COUNTS)
      printf -v "${key}" '%s' "${value}"
      ;;
  esac
done <<< "${SOLID_PREP_OUTPUT}"

export AS_SUCCESS133_BANK_NAME="${SOLID_BANK_NAME}"
export OMOMO_DATA_DIR="${SOLID_BANK_DIR}"
export OMOMO_OBJECT_MAP="${SOLID_OBJECT_MAP}"
export OMOMO_EXPECTED_TOTAL="${SOLID_SELECTED_CLIP_COUNT}"
export RESUME_FROM_BOX_EXPECTED_TOTAL="${SOLID_SELECTED_CLIP_COUNT}"
export AS_CONTACT_EXPORT_ROOT="${SOLID_BANK_DIR}/${SOLID_CONTACT_EXPORT_NAME}"

echo "[INFO] solid_allowed_object_categories=${SOLID_ALLOWED_OBJECT_CATEGORIES}"
echo "[INFO] source_bank=${SOLID_SOURCE_BANK}"
echo "[INFO] solid_clip_list=${SOLID_CLIP_LIST:-<none>}"
echo "[INFO] prepared_solid_bank=${SOLID_BANK_DIR}"
echo "[INFO] selected_solid_clips=${SOLID_SELECTED_CLIP_COUNT} category_counts=${SOLID_CATEGORY_COUNTS}"
echo "[INFO] resume_from_box=${RESUME_FROM_BOX}"
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  echo "[INFO] box_policy_init_checkpoint=${BOX_RESUME_CKPT}"
fi

exec bash "${SCRIPT_DIR}/distill_as_button.sh" "${POSITIONAL[@]}"
