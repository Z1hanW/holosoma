#!/usr/bin/env bash
set -euo pipefail

# Interactive depth-policy inference on the 197-clip AS real-mesh training bank.
#
# This launcher mirrors train_as_general.sh's AS bank preparation, then delegates
# the actual depth/Viser runtime to infer_box_joystick.sh.

usage() {
  cat <<'EOF'
Usage:
  bash infer_as_depth.sh [policy_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra args...]
  POLICY_CHECKPOINT=<checkpoint> bash infer_as_depth.sh [extra args...]

Examples:
  bash infer_as_depth.sh
  bash infer_as_depth.sh https://wandb.ai/zihanw22/carry-any/runs/b38z5iok
  MOTION_CLIP_NAME=scale__any_lamp_28 HEADLESS=False bash infer_as_depth.sh
  DRY_RUN=1 bash infer_as_depth.sh

Defaults:
  checkpoint = https://wandb.ai/zihanw22/carry-any/runs/b38z5iok
  AS_DATA_DIR = ./data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout
  AS_EXPECTED_TOTAL = 197

The checkpoint run URL is resolved to an explicit latest model_*.pt. If W&B
cannot provide a checkpoint and none is passed explicitly, this script exits.
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
    size = int(getattr(file_obj, "size", 0) or 0)
    if size <= 0:
        continue
    if best is None or step > best[0]:
        best = (step, name)

if best is not None and requested_step_int is None:
    print(best[1])
PY
}

normalize_wandb_checkpoint_ref() {
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
      echo "[INFO] Resolved W&B reference to checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine a .pt checkpoint for W&B reference: ${ref}" >&2
    echo "[ERROR] Pass a /files/model_*.pt URL, set WANDB_MODEL_FILE, or set RESUME_STEP." >&2
    return 2
  fi
  case "${model_file}" in
    *.pt) ;;
    *)
      echo "[ERROR] W&B checkpoint must be a .pt file. Got: ${model_file}" >&2
      return 2
      ;;
  esac

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

while [[ $# -gt 0 ]]; do
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    as|as-depth|as_depth|depth|perception|omomo|omomo-real|omomo_real|pure-real|pure_real|real)
      shift
      ;;
    *)
      break
      ;;
  esac
done

PYTHON_BIN=${PYTHON_BIN:-python}
DEFAULT_AS_DEPTH_CHECKPOINT=${DEFAULT_AS_DEPTH_CHECKPOINT:-"https://wandb.ai/zihanw22/carry-any/runs/b38z5iok"}
POLICY_CHECKPOINT=${POLICY_CHECKPOINT:-${AS_DEPTH_CHECKPOINT:-${CKPT:-${CHECKPOINT:-${DEFAULT_AS_DEPTH_CHECKPOINT}}}}}

if [[ $# -gt 0 ]] && is_checkpoint_ref "$1"; then
  POLICY_CHECKPOINT="$1"
  shift
fi

if [[ -z "${POLICY_CHECKPOINT}" ]]; then
  echo "[ERROR] Missing policy checkpoint." >&2
  usage >&2
  exit 1
fi

if [[ "${POLICY_CHECKPOINT}" == wandb://* || "${POLICY_CHECKPOINT}" == https://wandb.ai/*/runs/* || "${POLICY_CHECKPOINT}" == http://wandb.ai/*/runs/* || "${POLICY_CHECKPOINT}" == wandb.ai/*/runs/* ]]; then
  POLICY_CHECKPOINT="$(normalize_wandb_checkpoint_ref "${POLICY_CHECKPOINT}")"
fi
if [[ "${POLICY_CHECKPOINT}" != wandb://* && ! -f "${POLICY_CHECKPOINT}" ]]; then
  echo "[ERROR] policy checkpoint not found: ${POLICY_CHECKPOINT}" >&2
  exit 1
fi

DEFAULT_AS_BANK=carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout
AS_DATA_DIR=${AS_DATA_DIR:-${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/${DEFAULT_AS_BANK}"}}
AS_OBJECT_MAP=${AS_OBJECT_MAP:-${OMOMO_OBJECT_MAP:-"${AS_DATA_DIR}/_clip_object_urdf_map.json"}}
AS_EXPECTED_TOTAL=${AS_EXPECTED_TOTAL:-${OMOMO_EXPECTED_TOTAL:-197}}

LOCAL_DATA_ROOT=$(realpath -m "${SCRIPT_DIR}/data")
AS_DATA_DIR=$(realpath -m "${AS_DATA_DIR}")
AS_OBJECT_MAP=$(realpath -m "${AS_OBJECT_MAP}")

case "${AS_DATA_DIR}" in
  /nfs|/nfs/*)
    echo "[ERROR] AS_DATA_DIR must be local, not NFS: ${AS_DATA_DIR}" >&2
    echo "[ERROR] Run ./cp_as.sh first and infer from the copied AS bank under ${SCRIPT_DIR}/data." >&2
    exit 2
    ;;
esac
case "${AS_DATA_DIR}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*) ;;
  *)
    echo "[ERROR] AS_DATA_DIR must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${AS_DATA_DIR}" >&2
    exit 2
    ;;
esac
case "${AS_OBJECT_MAP}" in
  /nfs|/nfs/*)
    echo "[ERROR] AS_OBJECT_MAP must be local, not NFS: ${AS_OBJECT_MAP}" >&2
    echo "[ERROR] Run ./cp_as.sh first and use the copied map under ${SCRIPT_DIR}/data." >&2
    exit 2
    ;;
esac
case "${AS_OBJECT_MAP}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*) ;;
  *)
    echo "[ERROR] AS_OBJECT_MAP must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${AS_OBJECT_MAP}" >&2
    exit 2
    ;;
esac

if [[ ! -d "${AS_DATA_DIR}" ]]; then
  echo "[ERROR] AS_DATA_DIR does not exist: ${AS_DATA_DIR}" >&2
  echo "[ERROR] Run ./cp_as.sh first, or set AS_DATA_DIR to a prepared motion bank." >&2
  exit 2
fi
if ! compgen -G "${AS_DATA_DIR}/*.npz" >/dev/null; then
  echo "[ERROR] No .npz clips found in AS_DATA_DIR: ${AS_DATA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${AS_OBJECT_MAP}" ]]; then
  echo "[ERROR] Missing clip-object URDF map: ${AS_OBJECT_MAP}" >&2
  exit 2
fi

AS_SINGLE_SLOT_MOTION_DIR=${AS_SINGLE_SLOT_MOTION_DIR:-"${AS_DATA_DIR}/_single_slot_motion_bank"}
AS_SINGLE_SLOT_MOTION_DIR=$(realpath -m "${AS_SINGLE_SLOT_MOTION_DIR}")
case "${AS_SINGLE_SLOT_MOTION_DIR}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*) ;;
  *)
    echo "[ERROR] Generated AS single-slot motion bank must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${AS_SINGLE_SLOT_MOTION_DIR}" >&2
    exit 2
    ;;
esac

"${PYTHON_BIN}" - "${AS_DATA_DIR}" "${AS_SINGLE_SLOT_MOTION_DIR}" <<'PY'
import shutil
import sys
from pathlib import Path

source_dir = Path(sys.argv[1]).resolve()
view_dir = Path(sys.argv[2]).resolve()
if view_dir == source_dir or source_dir not in view_dir.parents:
    raise SystemExit(f"[ERROR] Refusing unexpected generated AS motion view path: {view_dir}")

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
    target.symlink_to(npz_path.resolve())
marker.write_text("generated by infer_as_depth.sh using train_as_general layout\n", encoding="utf-8")
PY

AS_SINGLE_SLOT_OBJECT_MAP="${AS_SINGLE_SLOT_MOTION_DIR}/_clip_object_urdf_map.json"
AS_OBJECT_MAP=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_single_slot_object_map.py" \
  --motion-dir "${AS_SINGLE_SLOT_MOTION_DIR}" \
  --object-map "${AS_OBJECT_MAP}" \
  --output-map "${AS_SINGLE_SLOT_OBJECT_MAP}")
AS_OBJECT_MAP=$(realpath -m "${AS_OBJECT_MAP}")
AS_DATA_DIR="${AS_SINGLE_SLOT_MOTION_DIR}"

"${PYTHON_BIN}" - "${AS_DATA_DIR}" "${AS_OBJECT_MAP}" "${AS_EXPECTED_TOTAL}" <<'PY'
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
    raise SystemExit("[ERROR] Real-mesh AS validation failed:\n  " + "\n  ".join(bad[:20]))

print(
    f"[INFO] Validated real-mesh AS depth bank: {motion_dir} "
    f"({len(npz_files)} clips, {len(unique_urdfs)} unique URDF mesh asset(s))"
)
PY

export WANDB_PROJECT=${WANDB_PROJECT:-carry-any}
export LOG_ROOT=${LOG_ROOT:-"/data/logs_new/${WANDB_PROJECT}"}
export INFER_DATASET=omomo
export MOTION_DIR="${AS_DATA_DIR}"
export OBJECT_URDF="${AS_OBJECT_MAP}"
export OBJECT_SPEC_PATH="${AS_OBJECT_MAP}"
export OBJECT_GEOMETRY_MODE="${OBJECT_GEOMETRY_MODE:-mesh}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE="${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}"
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE:-mesh}"
export VISER_LOAD_URDF="${VISER_LOAD_URDF:-1}"
export DEFAULT_DISTILL_PROPRIO_HISTORY_ONLY="${DEFAULT_DISTILL_PROPRIO_HISTORY_ONLY:-0}"
export DEFAULT_DISTILL_PROPRIO_HISTORY_LENGTH="${DEFAULT_DISTILL_PROPRIO_HISTORY_LENGTH:-1}"
export DEPTH_PERCEPTION_PRESET="${DEPTH_PERCEPTION_PRESET:-checkpoint}"
export HOLOSOMA_RESET_TO_DEFAULT_POSE="${HOLOSOMA_RESET_TO_DEFAULT_POSE:-0}"
export PHYSX_GPU_MAX_RIGID_CONTACT_COUNT="${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT:-4194304}"
export PHYSX_GPU_MAX_RIGID_PATCH_COUNT="${PHYSX_GPU_MAX_RIGID_PATCH_COUNT:-524288}"
export PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY="${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY:-16777216}"
export PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY="${PHYSX_GPU_FOUND_LOST_AGGREGATE_PAIRS_CAPACITY:-16777216}"
export PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY="${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY:-4194304}"
export PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-67108864}"
export PHYSX_GPU_HEAP_CAPACITY="${PHYSX_GPU_HEAP_CAPACITY:-67108864}"
export PHYSX_GPU_TEMP_BUFFER_CAPACITY="${PHYSX_GPU_TEMP_BUFFER_CAPACITY:-16777216}"

echo "[INFO] Launching AS real-mesh depth inference"
echo "[INFO] checkpoint=${POLICY_CHECKPOINT}"
echo "[INFO] MOTION_DIR=${MOTION_DIR}"
echo "[INFO] OBJECT_URDF=${OBJECT_URDF}"
echo "[INFO] AS_EXPECTED_TOTAL=${AS_EXPECTED_TOTAL}"
echo "[INFO] DEPTH_PERCEPTION_PRESET=${DEPTH_PERCEPTION_PRESET}"
echo "[INFO] DEFAULT_DISTILL_PROPRIO_HISTORY_ONLY=${DEFAULT_DISTILL_PROPRIO_HISTORY_ONLY}"
echo "[INFO] DEFAULT_DISTILL_PROPRIO_HISTORY_LENGTH=${DEFAULT_DISTILL_PROPRIO_HISTORY_LENGTH}"
echo "[INFO] HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE=${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE}"
echo "[INFO] PHYSX_GPU_MAX_RIGID_CONTACT_COUNT=${PHYSX_GPU_MAX_RIGID_CONTACT_COUNT}"
echo "[INFO] PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY=${PHYSX_GPU_FOUND_LOST_PAIRS_CAPACITY}"
echo "[INFO] PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY=${PHYSX_GPU_TOTAL_AGGREGATE_PAIRS_CAPACITY}"
echo "[INFO] PHYSX_GPU_COLLISION_STACK_SIZE=${PHYSX_GPU_COLLISION_STACK_SIZE}"

exec bash "${SCRIPT_DIR}/infer_box_joystick.sh" depth "${POLICY_CHECKPOINT}" "$@"
