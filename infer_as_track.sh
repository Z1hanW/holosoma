#!/usr/bin/env bash
set -euo pipefail

# Teacher-policy inference for AS/OMOMO real-mesh object tracking.
#
# This mirrors train_as_general.sh and delegates the actual inference launch to
# infer_box_tracking.sh so checkpoint/W&B/Viser behavior stays consistent.
#
# Usage:
#   bash infer_as_track.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]
#
# Optional env vars:
#   TEACHER_CHECKPOINT / CKPT  Optional checkpoint override. If unset, infer_box_tracking.sh
#                             tries the latest local generalist checkpoint under LOG_ROOT.
#   LOG_ROOT                  Default: /data/logs_new/${WANDB_PROJECT}
#   WANDB_PROJECT             Default: carry-any
#   OMOMO_DATA_DIR            Default: ./data/ds_as_data/omomo
#   OMOMO_OBJECT_MAP          Default: ${OMOMO_DATA_DIR}/_clip_object_urdf_map.json
#   OMOMO_EXPECTED_TOTAL      Optional exact clip count check. Default: auto
#   MOTION_CLIP_NAME          Optional: pin a single clip
#   NUM_ENVS                  Default inherited from infer_box_tracking.sh: 1
#   HEADLESS                  Default inherited from infer_box_tracking.sh: True
#   VISER_PORT                Default inherited from infer_box_tracking.sh: random

usage() {
  cat <<'EOF'
Usage:
  bash infer_as_track.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]

Examples:
  bash infer_as_track.sh
  bash infer_as_track.sh /data/logs_new/carry-any/<run>/model_00500.pt
  bash infer_as_track.sh https://wandb.ai/<entity>/carry-any/runs/<run_id>
  MOTION_CLIP_NAME=<clip_name> bash infer_as_track.sh /abs/path/to/model.pt
  HEADLESS=False bash infer_as_track.sh /abs/path/to/model.pt

This launcher always uses the repo-local AS/OMOMO real-mesh bank by default:
  OMOMO_DATA_DIR=./data/ds_as_data/omomo
  OMOMO_OBJECT_MAP=./data/ds_as_data/omomo/_clip_object_urdf_map.json
EOF
}

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/*/runs/* || "${ref}" == http://wandb.ai/*/runs/* || "${ref}" == wandb.ai/*/runs/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

canonicalize_wandb_run_url_ref() {
  local ref="$1"
  if [[ "${ref}" == wandb.ai/*/runs/* ]]; then
    echo "https://${ref}"
  elif [[ "${ref}" == http://wandb.ai/*/runs/* ]]; then
    echo "https://${ref#http://}"
  else
    echo "${ref}"
  fi
}

parse_wandb_run_url() {
  local ref="$1"
  local clean_ref
  clean_ref="$(canonicalize_wandb_run_url_ref "${ref}")"
  clean_ref="${clean_ref%%#*}"
  clean_ref="${clean_ref%%\?*}"
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

known_as_track_wandb_checkpoint_name() {
  local entity="$1"
  local project="$2"
  local run_id="$3"

  case "${entity}/${project}/${run_id}" in
    zihanw22/carry-any/bcleb5oi)
      echo "model_45000.pt"
      ;;
  esac
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

normalize_checkpoint_ref() {
  local ref="$1"
  local parsed=""
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  local model_file="${WANDB_MODEL_FILE:-}"

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
      echo "[INFO] Resolved W&B reference to latest checkpoint: ${model_file}" >&2
    fi
    if [[ -z "${model_file}" && -z "${RESUME_STEP:-}" ]]; then
      model_file="$(known_as_track_wandb_checkpoint_name "${entity}" "${project}" "${run_id}")"
      if [[ -n "${model_file}" ]]; then
        echo "[WARN] W&B API did not return model files; using known AS tracking checkpoint for ${entity}/${project}/${run_id}: ${model_file}" >&2
      fi
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B reference: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL, set WANDB_MODEL_FILE, or set RESUME_STEP." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

# Accept harmless AS/dataset aliases for muscle memory, then keep this wrapper
# responsible for the actual dataset selection.
if [[ $# -gt 0 ]]; then
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    as|as-track|as_tracking|omomo|omomo-real|omomo_real|pure-real|pure_real|pure-omomo|pure_omomo|real)
      shift
      ;;
  esac
fi

PYTHON_BIN=${PYTHON_BIN:-python}
WANDB_PROJECT=${WANDB_PROJECT:-carry-any}
LOG_ROOT=${LOG_ROOT:-"/data/logs_new/${WANDB_PROJECT}"}
TEACHER_CHECKPOINT=${TEACHER_CHECKPOINT:-${CKPT:-${CHECKPOINT:-}}}

if [[ $# -gt 0 ]] && is_checkpoint_ref "$1"; then
  TEACHER_CHECKPOINT="$1"
  shift
fi

if [[ -n "${TEACHER_CHECKPOINT}" ]]; then
  TEACHER_CHECKPOINT="$(normalize_checkpoint_ref "${TEACHER_CHECKPOINT}")"
  if [[ "${TEACHER_CHECKPOINT}" != wandb://* && ! -f "${TEACHER_CHECKPOINT}" ]]; then
    echo "[ERROR] teacher checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
    exit 1
  fi
fi

OMOMO_DATA_DIR=${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/omomo"}
OMOMO_OBJECT_MAP=${OMOMO_OBJECT_MAP:-"${OMOMO_DATA_DIR}/_clip_object_urdf_map.json"}
OMOMO_EXPECTED_TOTAL=${OMOMO_EXPECTED_TOTAL:-}

LOCAL_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data")
OMOMO_DATA_DIR=$(realpath -m "${OMOMO_DATA_DIR}")
OMOMO_OBJECT_MAP=$(realpath -m "${OMOMO_OBJECT_MAP}")

case "${OMOMO_DATA_DIR}" in
  /nfs|/nfs/*)
    echo "[ERROR] OMOMO_DATA_DIR must be local, not NFS: ${OMOMO_DATA_DIR}" >&2
    echo "[ERROR] Run ./cp_as.sh first and infer from ${SCRIPT_DIR}/data/ds_as_data/omomo." >&2
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
    echo "[ERROR] Run ./cp_as.sh first and use the copied map under ${SCRIPT_DIR}/data." >&2
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
  echo "[ERROR] Run ./cp_as.sh first, or set OMOMO_DATA_DIR to a prepared motion bank." >&2
  exit 2
fi

if ! compgen -G "${OMOMO_DATA_DIR}/*.npz" >/dev/null; then
  echo "[ERROR] No .npz files found in OMOMO_DATA_DIR: ${OMOMO_DATA_DIR}" >&2
  echo "[ERROR] Run ./cp_as.sh first, or set OMOMO_DATA_DIR to a prepared motion bank." >&2
  exit 2
fi

if [[ ! -f "${OMOMO_OBJECT_MAP}" ]]; then
  echo "[ERROR] Missing clip-object URDF map: ${OMOMO_OBJECT_MAP}" >&2
  exit 2
fi

OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE:-urdf}
OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-mesh}
case "$(echo "${OBJECT_SPAWN_MODE}" | tr '[:upper:]' '[:lower:]')" in
  urdf|mesh)
    OBJECT_SPAWN_MODE=urdf
    ;;
  *)
    echo "[ERROR] infer_as_track.sh requires real URDF mesh spawning." >&2
    echo "[ERROR] Do not use primitive/box mode here. Got OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE}" >&2
    exit 2
    ;;
esac
case "$(echo "${OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
  mesh|urdf|off|disable|disabled|0|false|no)
    OBJECT_GEOMETRY_MODE=mesh
    ;;
  *)
    echo "[ERROR] infer_as_track.sh requires mesh object geometry." >&2
    echo "[ERROR] Do not use primitive/box geometry here. Got OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE}" >&2
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
missing_npz = [clip_id for clip_id in sorted(clips) if not (motion_dir / f"{clip_id}.npz").is_file()]
if missing_npz:
    preview = ", ".join(missing_npz[:10])
    raise SystemExit(f"[ERROR] Missing .npz files for {len(missing_npz)} object-map entries: {preview}")


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
    raise SystemExit("[ERROR] Real-mesh AS/OMOMO validation failed:\n  " + "\n  ".join(bad[:20]))

print(
    f"[INFO] Validated real-mesh AS/OMOMO bank: {motion_dir} "
    f"({len(npz_files)} clips, {len(unique_urdfs)} unique URDF mesh asset(s))"
)
PY

export WANDB_PROJECT
export LOG_ROOT
export DATA_MODE=pure-real
export DS_DATA_ROOT="${SCRIPT_DIR}/data/ds_as_data"
export MOTION_DIR="${OMOMO_DATA_DIR}"
export OBJECT_SPEC_PATH="${OMOMO_OBJECT_MAP}"
export OBJECT_URDF="${OMOMO_OBJECT_MAP}"

export OBJECT_SPAWN_MODE
export OBJECT_GEOMETRY_MODE
export HOLOSOMA_OBJECT_SPAWN_MODE="${OBJECT_SPAWN_MODE}"
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${OBJECT_GEOMETRY_MODE}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}
export VISER_LOAD_URDF=${VISER_LOAD_URDF:-1}

echo "[INFO] Launching AS/OMOMO real-mesh co-tracking inference"
echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT:-<auto>}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJECT_SPEC_PATH=${OBJECT_SPEC_PATH}"
echo "[INFO] WANDB_PROJECT=${WANDB_PROJECT}"
echo "[INFO] LOG_ROOT=${LOG_ROOT}"
echo "[INFO] HOLOSOMA_OBJECT_SPAWN_MODE=${HOLOSOMA_OBJECT_SPAWN_MODE}"
echo "[INFO] HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE}"
echo "[INFO] HOLOSOMA_OBJECT_COLLIDER_TYPE=${HOLOSOMA_OBJECT_COLLIDER_TYPE}"

infer_cmd=(bash "${SCRIPT_DIR}/infer_box_tracking.sh" real)
if [[ -n "${TEACHER_CHECKPOINT}" ]]; then
  infer_cmd+=("${TEACHER_CHECKPOINT}")
fi
infer_cmd+=("$@")

case "$(echo "${DRY_RUN:-0}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    printf '[INFO] final_infer_command:'
    printf ' %q' "${infer_cmd[@]}"
    printf '\n'
    exit 0
    ;;
esac

exec "${infer_cmd[@]}"
