#!/usr/bin/env bash
set -euo pipefail

# AS drop-button distillation on the solid-object subset only.
#
# This prepares a filtered motion bank view before launch. The simulator object
# bank and MotionLoader must see the same URDF set, otherwise fixed
# env-to-clip assignment will fail for single-slot AS training.
#
# The default source prefers the mesh-physics solid bank when it exists locally.
# Otherwise it falls back to the normal success133 AS distill bank and filters it
# down to solid clips:
#   strict success_contact_and_final_position
#   box/bin/barrel/ball only
#   excludes scale__any_bin_3, scale__any_bin_8, box_21, box_39 falldown/suspect clips

usage() {
  cat <<'EOF'
Usage:
  bash distill_as_button_solid.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  bash distill_as_button_solid.sh --resume-from-box [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  bash distill_as_button_solid.sh --resume-from-previous [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  CHECK_ONLY=1 bash distill_as_button_solid.sh
  bash distill_as_button_solid.sh --check-only

Allowed object categories:
  box, bin, barrel, ball

Behavior:
  Prefers the repo-local mesh-physics solid bank. If that bank is unavailable,
  uses the normal distill_as_button.sh AS bank selection as the source, then
  creates a repo-local solid-only symlink bank and launches from it.
  This keeps simulator object assignment and MotionLoader clip filtering
  consistent.

Useful env vars:
  SOLID_ALLOWED_OBJECT_CATEGORIES='["box","bin","barrel","ball"]'  optional subset of these four
  SOLID_CLIP_LIST=<file>      optional one-clip-id-per-line allowlist; default is clean80
  SOLID_TARGET_BANK_NAME=<name>  override generated filtered bank name
  CORL_SOLID80_BANK_NAME=<name>  override preferred cp_corl.sh bank name
  CHECK_ONLY=1               count matching clips in the selected source bank
  RESUME_FROM_BOX=1          initialize policy weights from an architecture-compatible box-button checkpoint
  BOX_RESUME_CKPT=<checkpoint>  override the box policy initializer; actor keys/shapes must match exactly
  RESUME_FROM_PREVIOUS=1     initialize actor policy weights from previous AS distill run
  PREVIOUS_RESUME_RUN=<url>  previous run URL; default swl41n4x
  PREVIOUS_RESUME_CKPT=<checkpoint>  explicit previous checkpoint; otherwise latest model_*.pt is used
EOF
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
source "${SCRIPT_DIR}/scripts/gpu_launch_defaults.sh"

CHECK_ONLY=${CHECK_ONLY:-0}
RESUME_FROM_PREVIOUS=${RESUME_FROM_PREVIOUS:-0}
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
    --resume-from-previous|--resume_from_previous|resume-from-previous|resume_from_previous|previous|previous-run)
      RESUME_FROM_PREVIOUS=1
      shift
      ;;
    --no-resume-from-box|--no_resume_from_box|no-resume-from-box|no_resume_from_box)
      RESUME_FROM_BOX=0
      shift
      ;;
    --no-resume-from-previous|--no_resume_from_previous|no-resume-from-previous|no_resume_from_previous)
      RESUME_FROM_PREVIOUS=0
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
RESUME_FROM_PREVIOUS="$(normalize_bool RESUME_FROM_PREVIOUS "${RESUME_FROM_PREVIOUS:-0}")"

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
  if [[ "${#parts[@]}" -lt 3 ]]; then
    return 1
  fi
  entity="${parts[0]}"
  project="${parts[1]}"
  run_id="${parts[2]}"
  if [[ -z "${entity}" || -z "${project}" || -z "${run_id}" ]]; then
    return 1
  fi
  if [[ "${#parts[@]}" -gt 3 ]]; then
    explicit_file="${trimmed#${entity}/${project}/${run_id}/}"
  fi
  printf '%s\t%s\t%s\t%s\n' "${entity}" "${project}" "${run_id}" "${explicit_file}"
}

parse_wandb_reference() {
  local ref="$1"
  parse_wandb_run_url "${ref}" || parse_wandb_uri "${ref}"
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
sys.path = [
    entry
    for entry in sys.path
    if entry not in {"", "."} and Path(entry).resolve() != repo_root
]

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id = sys.argv[1:4]
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
pattern = re.compile(r"^model_(\d+)\.pt$")
best: tuple[int, str] | None = None
for file_obj in run.files():
    name = str(getattr(file_obj, "name", "") or "")
    match = pattern.match(name)
    if match is None:
        continue
    try:
        size = int(getattr(file_obj, "size", 0) or 0)
    except Exception:
        size = 0
    if size <= 0:
        continue
    candidate = (int(match.group(1)), name)
    if best is None or candidate[0] > best[0]:
        best = candidate

if best is not None:
    print(best[1])
PY
}

normalize_wandb_checkpoint_ref() {
  local ref="$1"
  local requested_model_file="${2:-}"
  local parsed=""
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  local model_file="${requested_model_file}"

  parsed="$(parse_wandb_reference "${ref}" || true)"
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
      echo "[INFO] Resolved previous W&B run to latest checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B reference: ${ref}" >&2
    echo "[ERROR] Pass /files/<checkpoint>.pt or set PREVIOUS_RESUME_MODEL_FILE/PREVIOUS_RESUME_CKPT." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

if [[ "${RESUME_FROM_PREVIOUS}" == "1" && "${RESUME_FROM_BOX}" == "1" ]]; then
  echo "[ERROR] --resume-from-previous and --resume-from-box are mutually exclusive." >&2
  exit 2
fi
if [[ "${RESUME_FROM_PREVIOUS}" == "1" ]]; then
  _previous_student_policy_type="$(echo "${STUDENT_POLICY_TYPE:-mlp}" | tr '[:upper:]' '[:lower:]' | tr '-' '_')"
  if [[ "${_previous_student_policy_type}" != "mlp" ]]; then
    echo "[ERROR] RESUME_FROM_PREVIOUS=1 uses the saved single-button MLP policy profile and cannot initialize STUDENT_POLICY_TYPE=${_previous_student_policy_type}." >&2
    echo "[ERROR] Use an architecture-matched full flow checkpoint through RESUME_CKPT." >&2
    exit 2
  fi
fi

export AS_SUCCESS133_FINAL0P5="${AS_SUCCESS133_FINAL0P5:-1}"
export RESUME_FROM_BOX
export RESUME_FROM_PREVIOUS
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  DEFAULT_BOX_RESUME_RUN=${DEFAULT_BOX_RESUME_RUN:-"https://wandb.ai/zihanw22/boxer/runs/d9m3z369-recovered"}
  DEFAULT_BOX_RESUME_MODEL_FILE=${DEFAULT_BOX_RESUME_MODEL_FILE:-model_22000.pt}
  BOX_RESUME_MODEL_FILE=${BOX_RESUME_MODEL_FILE:-${DEFAULT_BOX_RESUME_MODEL_FILE}}
  DEFAULT_BOX_RESUME_CHECKPOINT=${DEFAULT_BOX_RESUME_CHECKPOINT:-"${DEFAULT_BOX_RESUME_RUN}/files/${BOX_RESUME_MODEL_FILE}"}
  BOX_RESUME_CKPT=${BOX_RESUME_CKPT:-${RESUME_FROM_BOX_CKPT:-${DEFAULT_BOX_RESUME_CHECKPOINT}}}
  export DEFAULT_BOX_RESUME_RUN
  export DEFAULT_BOX_RESUME_MODEL_FILE
  export BOX_RESUME_MODEL_FILE
  export DEFAULT_BOX_RESUME_CHECKPOINT
  export BOX_RESUME_CKPT
fi
if [[ "${RESUME_FROM_PREVIOUS}" == "1" ]]; then
  DEFAULT_PREVIOUS_RESUME_RUN=${DEFAULT_PREVIOUS_RESUME_RUN:-"https://wandb.ai/zihanw22/carry-any/runs/swl41n4x"}
  PREVIOUS_RESUME_RUN=${PREVIOUS_RESUME_RUN:-${DEFAULT_PREVIOUS_RESUME_RUN}}
  PREVIOUS_RESUME_CKPT=${PREVIOUS_RESUME_CKPT:-${RESUME_FROM_PREVIOUS_CKPT:-${PREVIOUS_RESUME_RUN}}}
  PREVIOUS_RESUME_CKPT="$(normalize_wandb_checkpoint_ref "${PREVIOUS_RESUME_CKPT}" "${PREVIOUS_RESUME_MODEL_FILE:-}")"
  case "${PREVIOUS_RESUME_CKPT}" in
    wandb://*|*.pt)
      ;;
    *)
      echo "[ERROR] PREVIOUS_RESUME_CKPT must resolve to a .pt checkpoint. Got: ${PREVIOUS_RESUME_CKPT}" >&2
      exit 2
      ;;
  esac
  PREVIOUS_POLICY_INIT_CACHE_ROOT=${PREVIOUS_POLICY_INIT_CACHE_ROOT:-"${HOME}/.cache/holosoma/policy_init"}
  PREVIOUS_RESUME_CKPT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/resolve_exact_checkpoint.py" \
    --ref "${PREVIOUS_RESUME_CKPT}" \
    --cache-root "${PREVIOUS_POLICY_INIT_CACHE_ROOT}")
  export POLICY_INIT_CKPT="${PREVIOUS_RESUME_CKPT}"
  export AS_POLICY_INIT_PROFILE=drop_button_mlp_perception
  unset POLICY_INIT_CHECKPOINT
  unset RESUME_CKPT
  unset RESUME_CHECKPOINT
  unset WANDB_RUN_ID
  unset RESUME_WANDB_ID
  unset WANDB_RESUME
  export WANDB_RESUME_SAME_RUN=0
  export DEFAULT_PREVIOUS_RESUME_RUN
  export PREVIOUS_RESUME_RUN
  export PREVIOUS_RESUME_CKPT
fi
SOLID_ALLOWED_OBJECT_CATEGORIES=${SOLID_ALLOWED_OBJECT_CATEGORIES:-'["box","bin","barrel","ball"]'}
SOLID_ALLOWED_OBJECT_CATEGORIES=$(
  "${PYTHON_BIN}" - "${SOLID_ALLOWED_OBJECT_CATEGORIES}" <<'PY'
from __future__ import annotations

import json
import sys

allowed_universe = {"box", "bin", "barrel", "ball", "anything"}
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

DEFAULT_AS_SUCCESS133_BANK_NAME="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5"
DEFAULT_MESHPHYS_SOLID_BANK_NAME="carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success155_bcleb5oi58000_final0p5_solid80_clean_box_bin_barrel_ball_meshphys_v1"
CORL_SOLID80_BANK_NAME=${CORL_SOLID80_BANK_NAME:-"${DEFAULT_MESHPHYS_SOLID_BANK_NAME}"}
USER_SET_AS_SUCCESS133_BANK_NAME=${AS_SUCCESS133_BANK_NAME+x}
USER_SET_OMOMO_DATA_DIR=${OMOMO_DATA_DIR+x}
AS_SUCCESS133_BANK_NAME=${AS_SUCCESS133_BANK_NAME:-"${DEFAULT_AS_SUCCESS133_BANK_NAME}"}
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
  "${PYTHON_BIN}" - "${SOLID_SOURCE_BANK}" "${SOLID_SOURCE_MAP}" "${SOLID_ALLOWED_OBJECT_CATEGORIES}" "${SOLID_CLIP_LIST}" <<'PY'
from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
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


def validate_mesh_geometry(clip_id: str, entry: object) -> None:
    if not isinstance(entry, dict):
        raise SystemExit(f"[ERROR] Solid object map entry for {clip_id} must contain object_urdf_path metadata")
    raw_urdf = str(entry.get("object_urdf_path", "")).strip()
    if not raw_urdf:
        raise SystemExit(f"[ERROR] Solid clip {clip_id} is missing object_urdf_path")
    urdf = Path(raw_urdf).expanduser()
    if not urdf.is_absolute():
        urdf = (map_path.parent / urdf).resolve()
    if not urdf.is_file():
        raise SystemExit(f"[ERROR] Solid clip {clip_id} object URDF is missing: {urdf}")
    try:
        root = ET.parse(urdf).getroot()
    except Exception as exc:
        raise SystemExit(f"[ERROR] Invalid object URDF for {clip_id}: {urdf}: {exc}") from exc
    primitive_tags = [name for name in ("box", "sphere", "cylinder", "capsule") if root.findall(f".//{name}")]
    if primitive_tags:
        raise SystemExit(
            f"[ERROR] Solid clip {clip_id} uses primitive geometry {primitive_tags} in {urdf}; "
            "the bank name is not used to infer geometry"
        )
    mesh_paths = []
    for mesh in root.findall(".//mesh"):
        raw_mesh = str(mesh.get("filename", "")).strip()
        if not raw_mesh:
            raise SystemExit(f"[ERROR] Empty mesh filename for {clip_id} in {urdf}")
        mesh_path = Path(raw_mesh).expanduser()
        if not mesh_path.is_absolute():
            mesh_path = (urdf.parent / mesh_path).resolve()
        mesh_paths.append(mesh_path)
    if not mesh_paths:
        raise SystemExit(f"[ERROR] Solid clip {clip_id} has no mesh geometry in {urdf}")
    missing = [path for path in mesh_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"[ERROR] Solid clip {clip_id} references missing mesh assets: {missing[:6]}")


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
    validate_mesh_geometry(clip_id, entry)
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

# Publish a new content-addressed generation instead of clearing a directory
# that an active MotionLoader may still be consuming.
SOLID_PREP_ARGS=(
  --source-bank "${SOLID_SOURCE_BANK}"
  --source-map "${SOLID_SOURCE_MAP}"
  --allowed-categories-json "${SOLID_ALLOWED_OBJECT_CATEGORIES}"
  --contact-export-name "${SOLID_CONTACT_EXPORT_NAME}"
)
if [[ -n "${SOLID_CLIP_LIST}" ]]; then
  SOLID_PREP_ARGS+=(--clip-list "${SOLID_CLIP_LIST}")
fi
if [[ -n "${SOLID_TARGET_BANK_NAME}" ]]; then
  SOLID_PREP_ARGS+=(--target-bank-name "${SOLID_TARGET_BANK_NAME}")
fi
if [[ -n "${AS_CONTACT_EXPORT_ROOT:-}" ]]; then
  SOLID_PREP_ARGS+=(--contact-root "${AS_CONTACT_EXPORT_ROOT}")
fi
SOLID_PREP_OUTPUT=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_immutable_solid_bank.py" "${SOLID_PREP_ARGS[@]}")


while IFS='=' read -r key value; do
  case "${key}" in
    SOLID_BANK_NAME|SOLID_BANK_DIR|SOLID_OBJECT_MAP|SOLID_SELECTED_CLIP_COUNT|SOLID_CATEGORY_COUNTS|SOLID_SOURCE_DIGEST)
      printf -v "${key}" '%s' "${value}"
      ;;
  esac
done <<< "${SOLID_PREP_OUTPUT}"

if ! [[ "${SOLID_SOURCE_DIGEST:-}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "[ERROR] Immutable solid-AS preparation returned a malformed source digest: ${SOLID_SOURCE_DIGEST:-<empty>}" >&2
  exit 2
fi
if ! [[ "${SOLID_SELECTED_CLIP_COUNT:-}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] Immutable solid-AS preparation returned an invalid selected clip count: ${SOLID_SELECTED_CLIP_COUNT:-<empty>}" >&2
  exit 2
fi

# batch_ne.sh seals these two values only after every launch node has built and
# verified the same external-AS closure.  Re-check the values returned by the
# real wrapper materialization so a source-bank mutation between that barrier
# and this entrypoint cannot redirect training onto a newly generated view.
if [[ -n "${HOLOSOMA_EXTERNAL_AS_SOLID_SOURCE_DIGEST:-}" \
      || -n "${HOLOSOMA_EXTERNAL_AS_SELECTED_CLIP_COUNT:-}" ]]; then
  if ! [[ "${HOLOSOMA_EXTERNAL_AS_SOLID_SOURCE_DIGEST:-}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "[ERROR] Sealed external-AS contract has a malformed solid source digest." >&2
    exit 2
  fi
  if ! [[ "${HOLOSOMA_EXTERNAL_AS_SELECTED_CLIP_COUNT:-}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Sealed external-AS contract has an invalid selected clip count." >&2
    exit 2
  fi
  if [[ "${SOLID_SOURCE_DIGEST}" != "${HOLOSOMA_EXTERNAL_AS_SOLID_SOURCE_DIGEST}" ]]; then
    echo "[ERROR] Effective solid-AS source changed after the all-node barrier: actual=${SOLID_SOURCE_DIGEST} expected=${HOLOSOMA_EXTERNAL_AS_SOLID_SOURCE_DIGEST}" >&2
    exit 2
  fi
  if [[ "${SOLID_SELECTED_CLIP_COUNT}" != "${HOLOSOMA_EXTERNAL_AS_SELECTED_CLIP_COUNT}" ]]; then
    echo "[ERROR] Effective solid-AS clip count changed after the all-node barrier: actual=${SOLID_SELECTED_CLIP_COUNT} expected=${HOLOSOMA_EXTERNAL_AS_SELECTED_CLIP_COUNT}" >&2
    exit 2
  fi
fi

export AS_SUCCESS133_BANK_NAME="${SOLID_BANK_NAME}"
export OMOMO_DATA_DIR="${SOLID_BANK_DIR}"
export OMOMO_OBJECT_MAP="${SOLID_OBJECT_MAP}"
export OMOMO_EXPECTED_TOTAL="${SOLID_SELECTED_CLIP_COUNT}"
export RESUME_FROM_BOX_EXPECTED_TOTAL="${SOLID_SELECTED_CLIP_COUNT}"
export AS_CONTACT_EXPORT_ROOT="${SOLID_BANK_DIR}/${SOLID_CONTACT_EXPORT_NAME}"
export HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS="${HOLOSOMA_REQUIRE_OBJECT_MESH_ASSETS:-1}"

echo "[INFO] solid_allowed_object_categories=${SOLID_ALLOWED_OBJECT_CATEGORIES}"
echo "[INFO] source_bank=${SOLID_SOURCE_BANK}"
echo "[INFO] solid_clip_list=${SOLID_CLIP_LIST:-<none>}"
echo "[INFO] prepared_solid_bank=${SOLID_BANK_DIR}"
echo "[INFO] solid_source_digest=${SOLID_SOURCE_DIGEST}"
echo "[INFO] selected_solid_clips=${SOLID_SELECTED_CLIP_COUNT} category_counts=${SOLID_CATEGORY_COUNTS}"
echo "[INFO] resume_from_box=${RESUME_FROM_BOX}"
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  echo "[INFO] box_policy_init_checkpoint=${BOX_RESUME_CKPT}"
fi
echo "[INFO] resume_from_previous=${RESUME_FROM_PREVIOUS}"
if [[ "${RESUME_FROM_PREVIOUS}" == "1" ]]; then
  echo "[INFO] previous_policy_init_checkpoint=${PREVIOUS_RESUME_CKPT}"
fi

exec bash "${SCRIPT_DIR}/distill_as_button.sh" "${POSITIONAL[@]}"
