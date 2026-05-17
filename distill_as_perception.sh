#!/usr/bin/env bash
set -euo pipefail

# Distill an AS/OMOMO real-mesh generalist teacher into a depth-perception student.
#
# The teacher is expected to be a checkpoint produced by train_as_general.sh.
# This wrapper mirrors train_as_general.sh's local AS/OMOMO data validation and
# delegates the actual perception distillation launch to distill_box_perception.sh.
#
# Usage:
#   bash distill_as_perception.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [schedule/run_name/extra args...]
#   TEACHER_CHECKPOINT=<teacher_checkpoint> bash distill_as_perception.sh [schedule/run_name/extra args...]

usage() {
  cat <<'EOF'
Usage:
  bash distill_as_perception.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  TEACHER_CHECKPOINT=<teacher_checkpoint> bash distill_as_perception.sh [extra args...]
  RESUME_FROM_BOX=1 bash distill_as_perception.sh [extra args...]
  bash distill_as_perception.sh success133 [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]

Examples:
  bash distill_as_perception.sh /data/logs_new/carry-any/<run>/model_01000.pt
  bash distill_as_perception.sh wandb://<entity>/carry-any/<run_id>/model_01000.pt
  bash distill_as_perception.sh https://wandb.ai/<entity>/carry-any/runs/<run_id>
  bash distill_as_perception.sh /abs/model.pt ppo-first run:as_depth_student
  RESUME_FROM_BOX=1 bash distill_as_perception.sh

This launcher defaults to PPO+DAgger ppo-first distillation and always uses
the repo-local AS/OMOMO real-mesh bank by default:
  OMOMO_DATA_DIR=./data/ds_as_data/omomo
  OMOMO_OBJECT_MAP=./data/ds_as_data/omomo/_clip_object_urdf_map.json

If no teacher checkpoint is passed, the default teacher is the latest model
from:
  https://wandb.ai/zihanw22/carry-any/runs/bcleb5oi

With RESUME_FROM_BOX=1, the student policy parameters are initialized from the
checkpoint in:
  https://wandb.ai/zihanw22/boxer/runs/6c7exbeq
Training still starts from iteration 0 with a new/current run; only actor
policy parameters are loaded, not optimizer, critic, env state, or W&B resume
state.
and the motion/contact bank switches to the repo-local keep169 AS bank with
retargeted contact sidecars. Run ./cp_as.sh first to copy them from NFS:
  data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513
  data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513/contact_export_from_retarget

For the teacher-rollout filtered 133-clip AS bank:
  bash cp_tao.sh success133
  bash distill_as_perception.sh success133
The success133 mode uses contact-aware student inputs by default. Add
RESUME_FROM_BOX=1 or the resume-from-box alias only when you also want box
policy parameter initialization.
EOF
}

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/*/runs/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
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
  local requested_step="${4:-}"
  "${PYTHON_BIN:-python}" - "${entity}" "${project}" "${run_id}" "${requested_step}" <<'PY' 2>/dev/null || true
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

entity, project, run_id, requested_step = sys.argv[1:5]
requested_step_int = int(requested_step) if requested_step else None
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
pattern = re.compile(r"^model_(\d+)\.pt$")
best: tuple[int, str] | None = None
for file_obj in run.files():
    name = str(getattr(file_obj, "name", "") or "")
    match = pattern.match(name)
    if match is None:
        continue
    step = int(match.group(1))
    if requested_step_int is not None:
        if step == requested_step_int:
            print(name)
            sys.exit(0)
        continue
    try:
        size = int(getattr(file_obj, "size", 0) or 0)
    except Exception:
        size = 0
    if size <= 0:
        continue
    candidate = (step, name)
    if best is None or candidate[0] > best[0]:
        best = candidate

if best is not None and requested_step_int is None:
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
    model_file="$(resolve_remote_wandb_checkpoint_name "${entity}" "${project}" "${run_id}" "${RESUME_STEP:-}")"
    if [[ -n "${model_file}" ]]; then
      echo "[INFO] Resolved W&B reference to checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B reference: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL, set BOX_RESUME_MODEL_FILE/WANDB_MODEL_FILE, or set RESUME_STEP." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"
AS_SUCCESS133_FINAL0P5=${AS_SUCCESS133_FINAL0P5:-0}
AS_CONTACT_AWARE=${AS_CONTACT_AWARE:-${CONTACT_AWARE:-}}

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

# Accept harmless AS/dataset aliases for muscle memory; this wrapper owns the
# actual data selection and always launches pure-real AS/OMOMO distillation.
while [[ $# -gt 0 ]]; do
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    resume-from-box|resume_from_box)
      RESUME_FROM_BOX=1
      shift
      ;;
    success133|as-success133|as_success133|success133-final0p5|success133_final0p5)
      AS_SUCCESS133_FINAL0P5=1
      shift
      ;;
    contact-aware|contact_aware|contactaware)
      AS_CONTACT_AWARE=1
      shift
      ;;
    no-contact-aware|no_contact_aware|no-contactaware|no_contactaware)
      AS_CONTACT_AWARE=0
      shift
      ;;
    as|as-perception|as_perception|omomo|omomo-real|omomo_real|pure-real|pure_real|pure-omomo|pure_omomo|real)
      shift
      ;;
    *)
      break
      ;;
  esac
done

DEFAULT_AS_TEACHER_CHECKPOINT=${DEFAULT_AS_TEACHER_CHECKPOINT:-"https://wandb.ai/zihanw22/carry-any/runs/bcleb5oi"}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-${CKPT:-${DEFAULT_AS_TEACHER_CHECKPOINT}}}
if [[ $# -gt 0 ]] && is_checkpoint_ref "$1"; then
  TEACHER_CHECKPOINT="$1"
  shift
fi

if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing teacher checkpoint from train_as_general.sh." >&2
  usage >&2
  exit 1
fi

PYTHON_BIN=${PYTHON_BIN:-python}
WANDB_PROJECT=${WANDB_PROJECT:-carry-any}
case "$(echo "${AS_SUCCESS133_FINAL0P5}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    AS_SUCCESS133_FINAL0P5=1
    ;;
  0|false|no|off|"")
    AS_SUCCESS133_FINAL0P5=0
    ;;
  *)
    echo "[ERROR] AS_SUCCESS133_FINAL0P5 must be a boolean. Got: ${AS_SUCCESS133_FINAL0P5}" >&2
    exit 2
    ;;
esac
RESUME_FROM_BOX=${RESUME_FROM_BOX:-0}
case "$(echo "${RESUME_FROM_BOX}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    RESUME_FROM_BOX=1
    ;;
  0|false|no|off|"")
    RESUME_FROM_BOX=0
    ;;
  *)
    echo "[ERROR] RESUME_FROM_BOX must be a boolean. Got: ${RESUME_FROM_BOX}" >&2
    exit 2
    ;;
esac
if [[ -z "${AS_CONTACT_AWARE}" ]]; then
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" || "${RESUME_FROM_BOX}" == "1" ]]; then
    AS_CONTACT_AWARE=1
  else
    AS_CONTACT_AWARE=0
  fi
fi
case "$(echo "${AS_CONTACT_AWARE}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    AS_CONTACT_AWARE=1
    ;;
  0|false|no|off|"")
    AS_CONTACT_AWARE=0
    ;;
  *)
    echo "[ERROR] AS_CONTACT_AWARE must be a boolean. Got: ${AS_CONTACT_AWARE}" >&2
    exit 2
    ;;
esac

DEFAULT_RESUME_FROM_BOX_AS_BANK=${DEFAULT_RESUME_FROM_BOX_AS_BANK:-carryany_filter_scale_noscale_keep169_20260513}
DEFAULT_RESUME_FROM_BOX_LOCAL_DATA_DIR="${SCRIPT_DIR}/data/ds_as_data/${DEFAULT_RESUME_FROM_BOX_AS_BANK}"
DEFAULT_RESUME_FROM_BOX_LOCAL_CONTACT_ROOT="${DEFAULT_RESUME_FROM_BOX_LOCAL_DATA_DIR}/contact_export_from_retarget"
DEFAULT_RESUME_FROM_BOX_CONTACT_ROOT="${DEFAULT_RESUME_FROM_BOX_LOCAL_CONTACT_ROOT}"
DEFAULT_RESUME_FROM_BOX_DATA_DIR="${DEFAULT_RESUME_FROM_BOX_LOCAL_DATA_DIR}"
AS_SUCCESS133_BANK_NAME=${AS_SUCCESS133_BANK_NAME:-carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout_success133_final0p5}
AS_SUCCESS133_DATA_DIR="${SCRIPT_DIR}/data/ds_as_data/${AS_SUCCESS133_BANK_NAME}"
AS_SUCCESS133_CONTACT_EXPORT_ROOT="${AS_SUCCESS133_DATA_DIR}/contact_export_from_teacher_success133_final0p5"
DEFAULT_BOX_RESUME_RUN=${DEFAULT_BOX_RESUME_RUN:-"https://wandb.ai/zihanw22/boxer/runs/6c7exbeq"}
DEFAULT_BOX_RESUME_MODEL_FILE=${DEFAULT_BOX_RESUME_MODEL_FILE:-model_11000.pt}
BOX_RESUME_MODEL_FILE=${BOX_RESUME_MODEL_FILE:-${WANDB_MODEL_FILE:-${DEFAULT_BOX_RESUME_MODEL_FILE}}}
DEFAULT_BOX_RESUME_CHECKPOINT=${DEFAULT_BOX_RESUME_CHECKPOINT:-"${DEFAULT_BOX_RESUME_RUN}/files/${BOX_RESUME_MODEL_FILE}"}
BOX_RESUME_CKPT=${BOX_RESUME_CKPT:-${RESUME_FROM_BOX_CKPT:-${DEFAULT_BOX_RESUME_CHECKPOINT}}}
if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
  RESUME_FROM_BOX_AS_DATA_DIR=${RESUME_FROM_BOX_AS_DATA_DIR:-${AS_RESUME_DATA_DIR:-"${AS_SUCCESS133_DATA_DIR}"}}
  RESUME_FROM_BOX_AS_OBJECT_MAP=${RESUME_FROM_BOX_AS_OBJECT_MAP:-${AS_RESUME_OBJECT_MAP:-"${RESUME_FROM_BOX_AS_DATA_DIR}/_clip_object_urdf_map.json"}}
  RESUME_FROM_BOX_CONTACT_EXPORT_ROOT=${RESUME_FROM_BOX_CONTACT_EXPORT_ROOT:-${AS_CONTACT_EXPORT_ROOT:-"${AS_SUCCESS133_CONTACT_EXPORT_ROOT}"}}
  RESUME_FROM_BOX_EXPECTED_TOTAL=${RESUME_FROM_BOX_EXPECTED_TOTAL:-133}
else
  RESUME_FROM_BOX_AS_DATA_DIR=${RESUME_FROM_BOX_AS_DATA_DIR:-${AS_RESUME_DATA_DIR:-"${DEFAULT_RESUME_FROM_BOX_DATA_DIR}"}}
  RESUME_FROM_BOX_AS_OBJECT_MAP=${RESUME_FROM_BOX_AS_OBJECT_MAP:-${AS_RESUME_OBJECT_MAP:-"${RESUME_FROM_BOX_AS_DATA_DIR}/_clip_object_urdf_map.json"}}
  RESUME_FROM_BOX_CONTACT_EXPORT_ROOT=${RESUME_FROM_BOX_CONTACT_EXPORT_ROOT:-${AS_CONTACT_EXPORT_ROOT:-"${DEFAULT_RESUME_FROM_BOX_CONTACT_ROOT}"}}
  RESUME_FROM_BOX_EXPECTED_TOTAL=${RESUME_FROM_BOX_EXPECTED_TOTAL:-169}
fi

if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
  OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${AS_SUCCESS133_DATA_DIR}"}
  OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${AS_SUCCESS133_DATA_DIR}/_clip_object_urdf_map.json"}
  OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-133}
elif [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${RESUME_FROM_BOX_AS_DATA_DIR}"}
  OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${RESUME_FROM_BOX_AS_OBJECT_MAP}"}
  OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-"${RESUME_FROM_BOX_EXPECTED_TOTAL}"}
else
  OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/omomo"}
  OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${OMOMO_DATA_DIR}/_clip_object_urdf_map.json"}
  OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-45}
fi

if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  BOX_RESUME_CKPT="$(normalize_wandb_checkpoint_ref "${BOX_RESUME_CKPT}" "${BOX_RESUME_MODEL_FILE:-}")"
  case "${BOX_RESUME_CKPT}" in
    wandb://*|*.pt)
      ;;
    *)
      echo "[ERROR] BOX_RESUME_CKPT must resolve to a .pt checkpoint. Got: ${BOX_RESUME_CKPT}" >&2
      exit 2
      ;;
  esac
  if [[ -n "${RESUME_CKPT:-}" || -n "${RESUME_CHECKPOINT:-}" ]]; then
    echo "[ERROR] RESUME_FROM_BOX initializes policy parameters only; do not also set RESUME_CKPT/RESUME_CHECKPOINT." >&2
    echo "[ERROR] Use BOX_RESUME_CKPT to choose the box policy initializer." >&2
    exit 2
  fi
fi

LOCAL_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data")
OMOMO_DATA_DIR=$(realpath -m "${OMOMO_DATA_DIR}")
OMOMO_OBJECT_MAP=$(realpath -m "${OMOMO_OBJECT_MAP}")

case "${OMOMO_DATA_DIR}" in
  /nfs|/nfs/*)
    echo "[ERROR] OMOMO_DATA_DIR must be local, not NFS: ${OMOMO_DATA_DIR}" >&2
    if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
      echo "[ERROR] Run ./cp_as.sh first; it copies keep169 and contact_export_from_retarget under ${SCRIPT_DIR}/data/ds_as_data." >&2
    else
      echo "[ERROR] Run ./cp_real.sh first and distill from ${SCRIPT_DIR}/data/ds_as_data/omomo." >&2
    fi
    exit 2
    ;;
esac
case "${OMOMO_DATA_DIR}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] OMOMO_DATA_DIR must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${OMOMO_DATA_DIR}" >&2
    exit 2
    ;;
esac
case "${OMOMO_OBJECT_MAP}" in
  /nfs|/nfs/*)
    echo "[ERROR] OMOMO_OBJECT_MAP must be local, not NFS: ${OMOMO_OBJECT_MAP}" >&2
    if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
      echo "[ERROR] Run ./cp_as.sh first and use the copied map under ${SCRIPT_DIR}/data." >&2
    else
      echo "[ERROR] Run ./cp_real.sh first and use the copied map under ${SCRIPT_DIR}/data." >&2
    fi
    exit 2
    ;;
esac
case "${OMOMO_OBJECT_MAP}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] OMOMO_OBJECT_MAP must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${OMOMO_OBJECT_MAP}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${OMOMO_DATA_DIR}" ]]; then
  echo "[ERROR] OMOMO_DATA_DIR does not exist: ${OMOMO_DATA_DIR}" >&2
  if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
    echo "[ERROR] Run ./cp_as.sh first; it copies the keep169 bank and contact_export_from_retarget under data/ds_as_data/." >&2
  else
    echo "[ERROR] Run ./cp_real.sh first, or set OMOMO_DATA_DIR to a prepared motion bank." >&2
  fi
  exit 2
fi

if ! compgen -G "${OMOMO_DATA_DIR}/*.npz" >/dev/null; then
  echo "[ERROR] No .npz files found in OMOMO_DATA_DIR: ${OMOMO_DATA_DIR}" >&2
  if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
    echo "[ERROR] Run ./cp_as.sh first; it copies the keep169 bank and contact_export_from_retarget under data/ds_as_data/." >&2
  else
    echo "[ERROR] Run ./cp_real.sh first, or set OMOMO_DATA_DIR to a prepared motion bank." >&2
  fi
  exit 2
fi

if [[ ! -f "${OMOMO_OBJECT_MAP}" ]]; then
  echo "[ERROR] Missing clip-object URDF map: ${OMOMO_OBJECT_MAP}" >&2
  exit 2
fi

OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE:-${HOLOSOMA_OBJECT_SPAWN_MODE:-single_slot_multi_urdf}}
case "$(echo "${OBJECT_SPAWN_MODE}" | tr '[:upper:]' '[:lower:]')" in
  single_slot_multi_urdf|single-slot-multi-urdf|single_slot|single-slot|heterogeneous_single_slot|heterogeneous-single-slot)
    OBJECT_SPAWN_MODE=single_slot_multi_urdf
    ;;
  *)
    echo "[ERROR] distill_as_perception.sh only supports OBJECT_SPAWN_MODE=single_slot_multi_urdf." >&2
    echo "[ERROR] Legacy urdf bank and primitive/box modes are disabled for AS to prevent object-slot explosion." >&2
    echo "[ERROR] Got OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE}" >&2
    exit 2
    ;;
esac
OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-mesh}
case "$(echo "${OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
  mesh|urdf)
    OBJECT_GEOMETRY_MODE=mesh
    ;;
  *)
    echo "[ERROR] distill_as_perception.sh only supports mesh object geometry." >&2
    echo "[ERROR] Primitive/box/disabled geometry is not allowed for AS real-mesh training." >&2
    echo "[ERROR] Got OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE}" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" - "${OMOMO_DATA_DIR}" "${OMOMO_OBJECT_MAP}" "${OMOMO_EXPECTED_TOTAL}" <<'PY'
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

motion_dir = Path(sys.argv[1]).expanduser().resolve()
map_path = Path(sys.argv[2]).expanduser().resolve()
expected_raw = sys.argv[3].strip()
expected = int(expected_raw) if expected_raw else None

npz_files = sorted(motion_dir.glob("*.npz"))
if expected is not None and len(npz_files) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} .npz clips under {motion_dir}, found {len(npz_files)}")
if not npz_files:
    raise SystemExit(f"[ERROR] No .npz clips found under {motion_dir}")

payload = json.loads(map_path.read_text(encoding="utf-8"))
clips = payload["clips"] if isinstance(payload, dict) and isinstance(payload.get("clips"), dict) else payload
if not isinstance(clips, dict) or not clips:
    raise SystemExit(f"[ERROR] Invalid or empty object map: {map_path}")
if expected is not None and len(clips) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} object-map entries in {map_path}, found {len(clips)}")

missing_entries = [p.stem for p in npz_files if p.stem not in clips]
if missing_entries:
    preview = ", ".join(missing_entries[:10])
    raise SystemExit(f"[ERROR] Missing object-map entries for {len(missing_entries)} clip(s): {preview}")


def resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(str(raw).strip()).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


bad = []
unique_urdfs = {}
for clip_id, entry in clips.items():
    if not isinstance(entry, dict):
        bad.append(f"{clip_id}: map entry is not a dict")
        continue
    urdf_path = resolve_path(entry.get("object_urdf_path", ""), map_path.parent)
    mesh_path_raw = str(entry.get("object_mesh_path", "")).strip()
    mesh_path = resolve_path(mesh_path_raw, map_path.parent) if mesh_path_raw else None
    if not urdf_path.is_file():
        bad.append(f"{clip_id}: missing URDF {urdf_path}")
        continue
    if mesh_path is not None and not mesh_path.is_file():
        bad.append(f"{clip_id}: missing mesh {mesh_path}")
    unique_urdfs[str(urdf_path)] = clip_id

for urdf_raw, clip_id in sorted(unique_urdfs.items()):
    urdf_path = Path(urdf_raw)
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception as exc:
        bad.append(f"{clip_id}: failed to parse URDF {urdf_path}: {exc}")
        continue
    mesh_tags = root.findall(".//mesh")
    if not mesh_tags:
        bad.append(f"{clip_id}: URDF has no <mesh> geometry: {urdf_path}")
        continue
    for tag in mesh_tags:
        filename = str(tag.get("filename", "")).strip()
        if not filename:
            bad.append(f"{clip_id}: URDF mesh tag has empty filename: {urdf_path}")
            continue
        mesh_path = resolve_path(filename, urdf_path.parent)
        if not mesh_path.is_file():
            bad.append(f"{clip_id}: URDF mesh file missing: {mesh_path}")

if bad:
    raise SystemExit("[ERROR] Real-mesh OMOMO validation failed:\n  " + "\n  ".join(bad[:20]))

print(
    f"[INFO] Validated real-mesh OMOMO bank: {motion_dir} "
    f"({len(npz_files)} clips, {len(unique_urdfs)} unique URDF mesh asset(s))"
)
PY

AS_SINGLE_SLOT_MOTION_DIR=${AS_SINGLE_SLOT_MOTION_DIR:-"${OMOMO_DATA_DIR}/_single_slot_motion_bank"}
AS_SINGLE_SLOT_MOTION_DIR_ABS=$(realpath -m "${AS_SINGLE_SLOT_MOTION_DIR}")
case "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] Generated AS single-slot motion bank must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${AS_SINGLE_SLOT_MOTION_DIR_ABS}" >&2
    exit 2
    ;;
esac

python3 - "${OMOMO_DATA_DIR}" "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" <<'PY'
import os
import shutil
import sys
from pathlib import Path

source_dir = Path(sys.argv[1]).resolve()
view_dir = Path(sys.argv[2]).resolve()
if view_dir == source_dir or source_dir not in view_dir.parents:
    raise SystemExit(f"[ERROR] Refusing unexpected generated motion view path: {view_dir}")

marker = view_dir / ".generated_by_train_as_general"
if view_dir.exists():
    if not marker.exists():
        raise SystemExit(
            f"[ERROR] Refusing to clean non-generated AS motion view: {view_dir}. "
            "Choose an empty AS_SINGLE_SLOT_MOTION_DIR or remove it manually."
        )
    for child in view_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
else:
    view_dir.mkdir(parents=True)

for npz_path in sorted(source_dir.glob("*.npz")):
    target = view_dir / npz_path.name
    target.symlink_to(os.path.relpath(npz_path.resolve(), start=view_dir))
marker.write_text("generated by distill_as_perception.sh\n", encoding="utf-8")
PY

AS_SINGLE_SLOT_OBJECT_MAP="${AS_SINGLE_SLOT_MOTION_DIR_ABS}/_clip_object_urdf_map.json"
OMOMO_OBJECT_MAP=$(python3 "${SCRIPT_DIR}/scripts/prepare_single_slot_object_map.py" \
  --motion-dir "${AS_SINGLE_SLOT_MOTION_DIR_ABS}" \
  --object-map "${OMOMO_OBJECT_MAP}" \
  --output-map "${AS_SINGLE_SLOT_OBJECT_MAP}")
OMOMO_OBJECT_MAP=$(realpath -m "${OMOMO_OBJECT_MAP}")
case "${OMOMO_OBJECT_MAP}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*)
    ;;
  *)
    echo "[ERROR] Generated AS single-slot object map must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${OMOMO_OBJECT_MAP}" >&2
    exit 2
    ;;
esac
OMOMO_DATA_DIR="${AS_SINGLE_SLOT_MOTION_DIR_ABS}"

CONTACT_EXPORT_ROOT=""
CONTACT_EXPORT_CLIPS_ROOT=""
if [[ "${AS_CONTACT_AWARE}" == "1" ]]; then
  CONTACT_EXPORT_ROOT=$(realpath -m "${RESUME_FROM_BOX_CONTACT_EXPORT_ROOT}")
  CONTACT_EXPORT_CLIPS_ROOT=$(realpath -m "${CONTACT_EXPORT_ROOT}/clips")
  if [[ ! -d "${CONTACT_EXPORT_CLIPS_ROOT}" ]]; then
    CONTACT_EXPORT_CLIPS_ROOT="${CONTACT_EXPORT_ROOT}"
  fi
  if [[ ! -d "${CONTACT_EXPORT_CLIPS_ROOT}" ]]; then
    echo "[ERROR] Contact export root does not exist: ${CONTACT_EXPORT_ROOT}" >&2
    echo "[ERROR] Run ./cp_as.sh first; it copies contact_export_from_retarget into the repo-local keep169 bank." >&2
    exit 2
  fi
  case "${CONTACT_EXPORT_CLIPS_ROOT}" in
    /nfs|/nfs/*)
      echo "[ERROR] Contact export root must be local, not NFS: ${CONTACT_EXPORT_CLIPS_ROOT}" >&2
      echo "[ERROR] Run ./cp_as.sh first; it copies contact_export_from_retarget into the repo-local keep169 bank." >&2
      exit 2
      ;;
    *)
      CONTACT_EXPORT_CLIPS_ROOT=$("${PYTHON_BIN}" - "${OMOMO_DATA_DIR}" "${CONTACT_EXPORT_ROOT}" "${OMOMO_EXPECTED_TOTAL}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

motion_dir = Path(sys.argv[1]).expanduser().resolve()
contact_root = Path(sys.argv[2]).expanduser().resolve()
expected_raw = sys.argv[3].strip()
expected = int(expected_raw) if expected_raw else None
clips_root = contact_root / "clips" if (contact_root / "clips").is_dir() else contact_root

if not clips_root.is_dir():
    raise SystemExit(f"[ERROR] Contact export root does not exist: {contact_root}")

motion_ids = {path.stem for path in motion_dir.glob("*.npz")}
if expected is not None and len(motion_ids) != expected:
    raise SystemExit(f"[ERROR] Expected {expected} AS keep clips under {motion_dir}, found {len(motion_ids)}")
if not motion_ids:
    raise SystemExit(f"[ERROR] No .npz clips found under AS keep motion dir: {motion_dir}")

def infer_clip_id(dir_name: str) -> str:
    return dir_name.split("_", 1)[1].strip() if "_" in dir_name else dir_name.strip()

contact_ids: set[str] = set()
missing_files: list[str] = []
required_files = (
    "left_wrist_contact_points.npy",
    "left_wrist_contact_point_counts.npy",
    "left_wrist_contact_interval_steps.npy",
    "right_wrist_contact_points.npy",
    "right_wrist_contact_point_counts.npy",
    "right_wrist_contact_interval_steps.npy",
)
for clip_dir in sorted(path for path in clips_root.iterdir() if path.is_dir()):
    clip_id = infer_clip_id(clip_dir.name)
    contact_ids.add(clip_id)
    for file_name in required_files:
        if not (clip_dir / file_name).is_file():
            missing_files.append(f"{clip_id}:{file_name}")

missing_contacts = sorted(motion_ids.difference(contact_ids))
if missing_contacts:
    preview = ", ".join(missing_contacts[:20])
    raise SystemExit(f"[ERROR] Contact export missing {len(missing_contacts)} active clip(s): {preview}")
if missing_files:
    preview = ", ".join(missing_files[:20])
    raise SystemExit(f"[ERROR] Contact export has incomplete wrist sidecars: {preview}")

print(str(clips_root))
PY
      )
      ;;
  esac
  export CONTACT_EXPORT_ROOT
  export ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT="${CONTACT_EXPORT_CLIPS_ROOT}"
fi

if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  export POLICY_INIT_CKPT="${BOX_RESUME_CKPT}"
  unset RESUME_CKPT
  unset RESUME_CHECKPOINT
  unset WANDB_RUN_ID
  unset RESUME_WANDB_ID
  unset WANDB_RESUME
  export WANDB_RESUME_SAME_RUN=0
fi

export WANDB_PROJECT
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  export DATA_MODE="${DATA_MODE:-mix-naive}"
else
  export DATA_MODE=pure-real
fi
export DS_DATA_ROOT="${SCRIPT_DIR}/data/ds_as_data"
export MOTION_DIR="${OMOMO_DATA_DIR}"
export OBJECT_SPEC_PATH="${OMOMO_OBJECT_MAP}"
export OBJECT_URDF="${OMOMO_OBJECT_MAP}"
export AUTO_PREP_DS_BANK=0
export STRICT_DEFAULT_DS_BANK_VALIDATION=0
export USE_LEGACY_DS=0

export OBJECT_SPAWN_MODE
export OBJECT_GEOMETRY_MODE
export HOLOSOMA_OBJECT_SPAWN_MODE="${OBJECT_SPAWN_MODE}"
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${OBJECT_GEOMETRY_MODE}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE="${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}"
export HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS="${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS:-0}"
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK="${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-1}"
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1
export VISER_LOAD_URDF="${VISER_LOAD_URDF:-1}"

export DEFAULT_TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT}"
export TEACHER_CHECKPOINT
export TEACHER_COMPAT_PROFILE="${TEACHER_COMPAT_PROFILE:-none}"
export TEACHER_OBS_KEYS="${TEACHER_OBS_KEYS:-actor_obs}"
case "${TEACHER_CHECKPOINT}" in
  *"zihanw22/carry-any/runs/bcleb5oi"*|*"zihanw22/carry-any/bcleb5oi"*)
    export TEACHER_ACTOR_OBS_HISTORY_LENGTH="${TEACHER_ACTOR_OBS_HISTORY_LENGTH:-1}"
    ;;
esac
export TEACHER_PERCEPTION_PRESET="${TEACHER_PERCEPTION_PRESET:-none}"
export TEACHER_PERCEPTION_OBS_KEY="${TEACHER_PERCEPTION_OBS_KEY:-}"
export TRACKER_PROFILE="${TRACKER_PROFILE:-as-general}"
export SCHEDULE_VARIANT="${SCHEDULE_VARIANT:-ppo_first}"

if [[ "${AS_CONTACT_AWARE}" == "1" ]]; then
  export EXP="${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd-r2s-contact}"
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" && "${RESUME_FROM_BOX}" == "1" ]]; then
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_perception_init_box}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_perception_init_box}"
  elif [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_perception_contact}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_perception_contact}"
  else
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_keep169_perception_init_box}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_keep169_perception_init_box}"
  fi
else
  export EXP="${EXP:-g1-29dof-wbt-w-object-distill-sparse-root-cmd}"
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_success133_final0p5_perception}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_success133_final0p5_perception}"
  else
    export RUN_NAME="${RUN_NAME:-g1_w_object_distill_as_perception}"
    export TRAINING_NAME="${TRAINING_NAME:-g1_29dof_wbt_w_object_distill_as_real_mesh_perception}"
  fi
fi
export TRAINING_PROJECT="${TRAINING_PROJECT:-${WANDB_PROJECT}}"
export PERCEPTION_PRESET="${PERCEPTION_PRESET:-camera_depth_d435i}"
if [[ "${AS_CONTACT_AWARE}" == "1" ]]; then
  export ROOT_COMMAND_MODE="${ROOT_COMMAND_MODE:-contact-aware}"
  export STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-['actor_obs_root_contact_aware','actor_obs_proprio_with_actions_no_linvel']}"
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" && "${RESUME_FROM_BOX}" == "1" ]]; then
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_init_box_sparse_root_ppo_first_contact}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill initialized from actor policy parameters in zihanw22/boxer/6c7exbeq. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5, use teacher-exported contact sidecars for offline contact guidance and adaptive contact-window sampling, and keep the PPO+DAgger hybrid active from iteration 0.}"
  elif [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_contact}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill with contact-aware sparse root. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5, use teacher-exported contact sidecars for offline contact guidance and adaptive contact-window sampling, and keep the PPO+DAgger hybrid active from iteration 0.}"
  else
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_keep169_init_box_sparse_root_ppo_first_contact}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS keep169 real-mesh perception distill initialized from actor policy parameters in zihanw22/boxer/6c7exbeq. Training starts from iteration 0 with current AS data/contact/schedule, uses retarget-exported left/right wrist contact sidecars for offline contact guidance and adaptive contact-window sampling, and keeps the PPO+DAgger hybrid active from iteration 0.}"
  fi
else
  export STUDENT_ACTOR_INPUTS="${STUDENT_ACTOR_INPUTS:-['actor_obs_root','actor_obs_proprio','actor_obs_actions']}"
  if [[ "${AS_SUCCESS133_FINAL0P5}" == "1" ]]; then
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_success133_final0p5_sparse_root_ppo_first_step_mix}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS teacher-rollout filtered 133-clip real-mesh perception distill from train_as_general.sh teacher. Clips satisfy stable_contact_success=True and final_object_position_error_m<=0.5. PPO+DAgger hybrid by default; teacher consumes actor_obs without perception, student consumes sparse root command, proprio/action history, and depth perception.}"
  else
    export SCHEDULE_NAME="${SCHEDULE_NAME:-as_real_mesh_sparse_root_ppo_first_step_mix}"
    export SCHEDULE_NOTES="${SCHEDULE_NOTES:-AS/OMOMO real-mesh perception distill from train_as_general.sh teacher. PPO+DAgger hybrid by default: PPO is active from iteration 0, ramps from 0.1 to 0.9 by iteration 4000, and the effective DAgger BC weight decreases from 0.9 to 0.1. Teacher consumes actor_obs without perception; student consumes sparse root command, proprio/action history, and depth perception.}"
  fi
fi

echo "[INFO] Launching AS/OMOMO real-mesh perception distillation"
echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
echo "[INFO] resume_from_box=${RESUME_FROM_BOX}"
echo "[INFO] as_contact_aware=${AS_CONTACT_AWARE}"
if [[ "${RESUME_FROM_BOX}" == "1" ]]; then
  echo "[INFO] student_policy_init_checkpoint=${POLICY_INIT_CKPT}"
fi
if [[ "${AS_CONTACT_AWARE}" == "1" ]]; then
  echo "[INFO] contact_export_root=${CONTACT_EXPORT_ROOT}"
  echo "[INFO] adaptive_sampling_contact_interval_root=${ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT}"
fi
echo "[INFO] teacher_obs_keys=${TEACHER_OBS_KEYS} teacher_perception=${TEACHER_PERCEPTION_PRESET}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJECT_URDF=${OBJECT_URDF}"
echo "[INFO] EXP=${EXP} perception=${PERCEPTION_PRESET}"
echo "[INFO] RUN_NAME=${RUN_NAME} TRAINING_PROJECT=${TRAINING_PROJECT}"
echo "[INFO] student_actor_inputs=${STUDENT_ACTOR_INPUTS}"
echo "[INFO] schedule_variant=${SCHEDULE_VARIANT} schedule_name=${SCHEDULE_NAME}"
echo "[INFO] HOLOSOMA_OBJECT_SPAWN_MODE=${HOLOSOMA_OBJECT_SPAWN_MODE}"
echo "[INFO] HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE}"
echo "[INFO] HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=${HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS}"
echo "[INFO] HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK}"
echo "[INFO] HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=${HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS}"
if [[ -n "${TEACHER_ACTOR_OBS_HISTORY_LENGTH:-}" ]]; then
  echo "[INFO] teacher_actor_obs_history_length=${TEACHER_ACTOR_OBS_HISTORY_LENGTH}"
fi

exec bash "${SCRIPT_DIR}/distill_box_perception.sh" "$@"
