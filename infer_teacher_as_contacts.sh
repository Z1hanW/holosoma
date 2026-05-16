#!/usr/bin/env bash
set -euo pipefail

# Policy contact export for AS real-mesh sequences.
#
# This is the AS counterpart of infer_teacher_box_contacts.sh. It is meant to
# run a policy checkpoint produced by train_as_general.sh over the same
# repo-local AS real-mesh bank, then export the same contact point/reference
# layout consumed by rollout-ref/contact-aware code.
#
# Usage:
#   bash infer_teacher_as_contacts.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]
#   bash infer_teacher_as_contacts.sh view [existing_output_dir]
#
# Examples:
#   bash infer_teacher_as_contacts.sh
#   NUM_ENVS=8 bash infer_teacher_as_contacts.sh
#   DRY_RUN=1 bash infer_teacher_as_contacts.sh https://wandb.ai/zihanw22/carry-any/runs/gml45u7p
#   bash infer_teacher_as_contacts.sh view

usage() {
  cat <<'EOF'
Usage:
  bash infer_teacher_as_contacts.sh [teacher_checkpoint.pt|wandb://...|https://wandb.ai/.../runs/...] [extra tyro args...]
  bash infer_teacher_as_contacts.sh view [existing_output_dir]

Optional env vars:
  TEACHER_CHECKPOINT        Policy checkpoint to export; default parsed from distill_as_perception.sh
  POLICY_CHECKPOINT         Alias for TEACHER_CHECKPOINT
  WANDB_MODEL_FILE          Optional; used when checkpoint is a W&B run URL without /files/<checkpoint>
  AS_DATA_DIR               Default: ./data/ds_as_data/carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout
  AS_OBJECT_MAP             Default: AS_DATA_DIR/_clip_object_urdf_map.json
  AS_EXPECTED_TOTAL         Default: 197; set empty to disable count check
  AS_SINGLE_SLOT_MOTION_DIR Default: AS_DATA_DIR/_single_slot_motion_bank
  OMOMO_DATA_DIR            Legacy alias for AS_DATA_DIR
  OMOMO_OBJECT_MAP          Legacy alias for AS_OBJECT_MAP
  OMOMO_EXPECTED_TOTAL      Legacy alias for AS_EXPECTED_TOTAL
  NUM_ENVS                  Default: 8
  HEADLESS                  Default: True
  OUTPUT_DIR                Export mode default: outputs/teacher_as_contacts/<utc timestamp>.
                            View mode: existing export dir; default is latest outputs/teacher_as_contacts/*
  VIEW_ONLY                 Default: 0; set 1 to skip rollout and only launch the viewer
  PUBLISH_FOR_INFER_BOX     Export default: 1; view default: 0; copy clips/motion_bank into legacy outputs paths
  PUBLISH_CLIPS_DIR         Default: ./outputs/clips
  PUBLISH_MOTION_BANK_DIR   Default: ./outputs/motion_bank
  LAUNCH_VISER              Default: 1; start rollout/contact viewer after export
  VIEWER_BACKGROUND         Default: 1; keep viewer running in background after export
  VISER_PORT                Default: random
  VISER_HOST                Default: 0.0.0.0
  VIEWER_SEQUENCE           Optional initial clip id / clip directory name
  VIEWER_SHOW_ORIGINAL_MOTION Default: 1; show input reference motion at startup
  VIEWER_SHOW_PRIMITIVE_BOX Default: 0; show AABB primitive-box overlay at startup
  VIEWER_SHOW_ROBOT         Default: 1; draw training G1 overlay
  VIEWER_LOG                Default: logs/runtime/infer_teacher_as_contacts_viewer_<timestamp>.log
  MIN_CONTACT_FRAMES        Default: 10
  CONTACT_FORCE_THRESHOLD   Default: 1.0
  CONTACT_VOXEL_SIZE        Default: 0.01
  SUCCESS_POSITION_THRESHOLD Default: 0.10
  MAX_ROLLOUT_STEPS         Optional per-clip step cap
  DISABLE_RANDOMIZATION     Default: True
  START_AT_TIMESTEP_ZERO_PROB Default: 1.0
  FREEZE_AT_TIMESTEP_ZERO_PROB Default: 0.0
  RESET_NOISE_SCALE         Default: 0.0
  USE_ADAPTIVE_TIMESTEPS_SAMPLER Default: False
  MAX_EPISODE_LENGTH_S      Default: 1000000
  PHYSX_GPU_COLLISION_STACK_SIZE Default: 268435456
  VALIDATE_OUTPUT_FORMAT    Default: 1; verify exported clips match rollout-ref/contact-aware layout
  DRY_RUN                   Default: 0

By default the raw export is kept under OUTPUT_DIR and the generated
OUTPUT_DIR/clips plus OUTPUT_DIR/motion_bank are also copied into outputs/clips
and outputs/motion_bank so existing rollout-ref/contact-aware interfaces can
find the point files without changing their config.

Export mode mirrors train_as_general.sh's AS layout: it builds a repo-local
single-slot motion/object view and uses mesh URDF assets, not primitive boxes.

View-only mode never launches IsaacSim or loads the teacher checkpoint. It only
starts the rollout/contact Viser viewer on an existing export directory.
EOF
}

is_checkpoint_ref() {
  local ref="$1"
  [[ "${ref}" == wandb://* || "${ref}" == https://wandb.ai/*/runs/* || "${ref}" == /* || "${ref}" == ./* || "${ref}" == ../* || "${ref}" == *.pt ]]
}

is_truthy() {
  case "$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

resolve_latest_teacher_as_contacts_output() {
  find "${SCRIPT_DIR}/outputs/teacher_as_contacts" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | while IFS= read -r candidate; do
        if [[ -d "${candidate}/clips" ]]; then
          printf '%s\n' "${candidate}"
        fi
      done \
    | sort \
    | tail -n 1
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|help)
      usage
      exit 0
      ;;
  esac
fi

# Accept harmless aliases so the wrapper can be called with the same muscle
# memory as distill_as_perception.sh / infer_as_joystick.sh.
while [[ $# -gt 0 ]]; do
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    as|as-contacts|as_contacts|omomo|omomo-real|omomo_real|pure-real|pure_real|pure-omomo|pure_omomo|real)
      shift
      ;;
    *)
      break
      ;;
  esac
done

VIEW_ONLY="${VIEW_ONLY:-0}"
if [[ $# -gt 0 ]]; then
  case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in
    view|view-only|view_only|viewer|viser|open)
      VIEW_ONLY=1
      shift
      if [[ $# -gt 0 && "$1" != -* ]]; then
        OUTPUT_DIR="$1"
        shift
      fi
      ;;
  esac
fi

extract_default_teacher_checkpoint_from_distill_as_perception() {
  "${PYTHON_BIN}" - "${SCRIPT_DIR}/distill_as_perception.sh" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    text = path.read_text(encoding="utf-8")
except Exception:
    sys.exit(0)

match = re.search(r'^DEFAULT_AS_TEACHER_CHECKPOINT=\$\{DEFAULT_AS_TEACHER_CHECKPOINT:-"([^"]+)"\}', text, re.M)
if match:
    print(match.group(1))
PY
}

parse_wandb_run_url() {
  local ref="$1"
  local clean_ref="${ref%%#*}"
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

resolve_remote_wandb_checkpoint_name() {
  local entity="$1"
  local project="$2"
  local run_id="$3"

  "${PYTHON_BIN}" - "${entity}" "${project}" "${run_id}" <<'PY' 2>/dev/null || true
import re
import sys
from pathlib import Path

repo_root = Path.cwd().resolve()
sanitized_sys_path = []
for path_entry in sys.path:
    if path_entry in {"", "."}:
        continue
    try:
        if Path(path_entry).resolve() == repo_root:
            continue
    except Exception:
        pass
    sanitized_sys_path.append(path_entry)
sys.path = sanitized_sys_path

try:
    import wandb
except Exception:
    sys.exit(0)

entity, project, run_id = sys.argv[1:4]
api = wandb.Api(timeout=30)
run = api.run(f"{entity}/{project}/{run_id}")
model_pattern = re.compile(r"^model_(\d+)\.pt$")
latest_step = -1
latest_name = ""
for file_obj in run.files():
    name = getattr(file_obj, "name", "")
    match = model_pattern.match(name)
    if not match:
        continue
    step = int(match.group(1))
    if step >= latest_step:
        latest_step = step
        latest_name = name
if latest_name:
    print(latest_name)
PY
}

normalize_checkpoint_ref() {
  local ref="$1"
  if [[ "${ref}" != https://wandb.ai/*/runs/* ]]; then
    echo "${ref}"
    return 0
  fi

  local parsed=""
  local entity=""
  local project=""
  local run_id=""
  local explicit_file=""
  local model_file="${WANDB_MODEL_FILE:-}"

  parsed="$(parse_wandb_run_url "${ref}" || true)"
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
      echo "[INFO] Resolved W&B run URL to remote checkpoint: ${model_file}" >&2
    fi
  fi

  if [[ -z "${model_file}" ]]; then
    echo "[ERROR] Could not determine checkpoint for W&B run URL: ${ref}" >&2
    echo "[ERROR] Pass a /files/<checkpoint>.pt URL or set WANDB_MODEL_FILE." >&2
    return 2
  fi

  echo "wandb://${entity}/${project}/${run_id}/${model_file}"
}

normalize_bool_flag() {
  local value="${1:-}"
  case "$(echo "${value}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) echo "True" ;;
    0|false|no|off) echo "False" ;;
    *)
      echo "[ERROR] Invalid boolean value: ${value}" >&2
      exit 2
      ;;
  esac
}

TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${AS_POLICY_CHECKPOINT:-${POLICY_CHECKPOINT:-${CHECKPOINT:-${CKPT:-}}}}}"
if ! is_truthy "${VIEW_ONLY}"; then
  DISTILL_DEFAULT_TEACHER_CHECKPOINT="$(extract_default_teacher_checkpoint_from_distill_as_perception)"
  DISTILL_DEFAULT_TEACHER_CHECKPOINT="${DISTILL_DEFAULT_TEACHER_CHECKPOINT:-https://wandb.ai/zihanw22/carry-any/runs/gml45u7p}"
  TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${DISTILL_DEFAULT_TEACHER_CHECKPOINT}}"

  if [[ $# -gt 0 ]] && is_checkpoint_ref "$1"; then
    TEACHER_CHECKPOINT="$1"
    shift
  fi

  if [[ "${TEACHER_CHECKPOINT}" == https://wandb.ai/*/runs/* ]]; then
    TEACHER_CHECKPOINT="$(normalize_checkpoint_ref "${TEACHER_CHECKPOINT}")"
  fi

  if [[ -z "${TEACHER_CHECKPOINT}" ]]; then
    echo "[ERROR] Missing AS teacher checkpoint." >&2
    usage >&2
    exit 2
  fi
  if [[ "${TEACHER_CHECKPOINT}" != wandb://* ]] && [[ ! -f "${TEACHER_CHECKPOINT}" ]]; then
    echo "[ERROR] checkpoint not found: ${TEACHER_CHECKPOINT}" >&2
    exit 1
  fi
fi

WANDB_PROJECT=${WANDB_PROJECT:-carry-any}
DEFAULT_AS_BANK=carryany_filter_scale_noscale_keep169_20260513_plus_box_teacher_rollout
AS_DATA_DIR=${AS_DATA_DIR:-${OMOMO_DATA_DIR:-"${SCRIPT_DIR}/data/ds_as_data/${DEFAULT_AS_BANK}"}}
AS_OBJECT_MAP=${AS_OBJECT_MAP:-${OMOMO_OBJECT_MAP:-"${AS_DATA_DIR}/_clip_object_urdf_map.json"}}
AS_EXPECTED_TOTAL=${AS_EXPECTED_TOTAL:-${OMOMO_EXPECTED_TOTAL:-197}}

LOCAL_DATA_ROOT="$(realpath -m "${SCRIPT_DIR}/data")"
AS_DATA_DIR="$(realpath -m "${AS_DATA_DIR}")"
AS_OBJECT_MAP="$(realpath -m "${AS_OBJECT_MAP}")"

case "${AS_DATA_DIR}" in
  /nfs|/nfs/*)
    echo "[ERROR] AS_DATA_DIR must be local, not NFS: ${AS_DATA_DIR}" >&2
    echo "[ERROR] Run ./cp_as.sh first and export from the repo-local AS bank under ${SCRIPT_DIR}/data." >&2
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
  echo "[ERROR] No .npz files found in AS_DATA_DIR: ${AS_DATA_DIR}" >&2
  exit 2
fi
if [[ ! -f "${AS_OBJECT_MAP}" ]]; then
  echo "[ERROR] Missing clip-object URDF map: ${AS_OBJECT_MAP}" >&2
  exit 2
fi

OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE:-${HOLOSOMA_OBJECT_SPAWN_MODE:-single_slot_multi_urdf}}
OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE:-mesh}
case "$(echo "${OBJECT_SPAWN_MODE}" | tr '[:upper:]' '[:lower:]')" in
  single_slot_multi_urdf|single-slot-multi-urdf|single_slot|single-slot|heterogeneous_single_slot|heterogeneous-single-slot)
    OBJECT_SPAWN_MODE=single_slot_multi_urdf
    ;;
  *)
    echo "[ERROR] infer_teacher_as_contacts.sh requires OBJECT_SPAWN_MODE=single_slot_multi_urdf for train_as_general.sh checkpoints." >&2
    echo "[ERROR] Legacy all-object URDF spawning is disabled for AS contact export to avoid object-slot explosion." >&2
    echo "[ERROR] Got OBJECT_SPAWN_MODE=${OBJECT_SPAWN_MODE}" >&2
    exit 2
    ;;
esac
case "$(echo "${OBJECT_GEOMETRY_MODE}" | tr '[:upper:]' '[:lower:]')" in
  mesh|urdf)
    OBJECT_GEOMETRY_MODE=mesh
    ;;
  *)
    echo "[ERROR] infer_teacher_as_contacts.sh requires mesh object geometry." >&2
    echo "[ERROR] Got OBJECT_GEOMETRY_MODE=${OBJECT_GEOMETRY_MODE}" >&2
    exit 2
    ;;
esac

AS_SINGLE_SLOT_MOTION_DIR=${AS_SINGLE_SLOT_MOTION_DIR:-"${AS_DATA_DIR}/_single_slot_motion_bank"}
AS_SINGLE_SLOT_MOTION_DIR="$(realpath -m "${AS_SINGLE_SLOT_MOTION_DIR}")"
case "${AS_SINGLE_SLOT_MOTION_DIR}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*) ;;
  *)
    echo "[ERROR] Generated AS single-slot motion bank must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${AS_SINGLE_SLOT_MOTION_DIR}" >&2
    exit 2
    ;;
esac

if [[ "${AS_DATA_DIR}" != "${AS_SINGLE_SLOT_MOTION_DIR}" ]]; then
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
marker.write_text("generated by infer_teacher_as_contacts.sh using train_as_general layout\n", encoding="utf-8")
PY

  AS_SINGLE_SLOT_OBJECT_MAP="${AS_SINGLE_SLOT_MOTION_DIR}/_clip_object_urdf_map.json"
  AS_OBJECT_MAP=$("${PYTHON_BIN}" "${SCRIPT_DIR}/scripts/prepare_single_slot_object_map.py" \
    --motion-dir "${AS_SINGLE_SLOT_MOTION_DIR}" \
    --object-map "${AS_OBJECT_MAP}" \
    --output-map "${AS_SINGLE_SLOT_OBJECT_MAP}")
  AS_OBJECT_MAP="$(realpath -m "${AS_OBJECT_MAP}")"
  AS_DATA_DIR="${AS_SINGLE_SLOT_MOTION_DIR}"
fi

case "${AS_OBJECT_MAP}" in
  "${LOCAL_DATA_ROOT}"|"${LOCAL_DATA_ROOT}"/*) ;;
  *)
    echo "[ERROR] Generated AS single-slot object map must live under repo-local data root: ${LOCAL_DATA_ROOT}" >&2
    echo "[ERROR] Got: ${AS_OBJECT_MAP}" >&2
    exit 2
    ;;
esac

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
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


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
    f"[INFO] Validated real-mesh AS bank: {motion_dir} "
    f"({len(npz_files)} clips, {len(unique_urdfs)} unique URDF mesh asset(s))"
)
PY

OMOMO_DATA_DIR="${AS_DATA_DIR}"
OMOMO_OBJECT_MAP="${AS_OBJECT_MAP}"
OMOMO_EXPECTED_TOTAL="${AS_EXPECTED_TOTAL}"

NUM_ENVS="${NUM_ENVS:-8}"
HEADLESS="${HEADLESS:-True}"
if is_truthy "${VIEW_ONLY}"; then
  OUTPUT_DIR="${OUTPUT_DIR:-$(resolve_latest_teacher_as_contacts_output)}"
  if [[ -z "${OUTPUT_DIR}" ]]; then
    echo "[ERROR] VIEW_ONLY=1 could not find an existing outputs/teacher_as_contacts/* export." >&2
    echo "[ERROR] Pass one explicitly: bash infer_teacher_as_contacts.sh view /path/to/export" >&2
    exit 2
  fi
else
  OUTPUT_DIR="${OUTPUT_DIR:-"${SCRIPT_DIR}/outputs/teacher_as_contacts/$(date -u +%Y%m%d_%H%M%S)"}"
fi
OUTPUT_DIR="$(realpath -m "${OUTPUT_DIR}")"
MIN_CONTACT_FRAMES="${MIN_CONTACT_FRAMES:-10}"
CONTACT_FORCE_THRESHOLD="${CONTACT_FORCE_THRESHOLD:-1.0}"
CONTACT_VOXEL_SIZE="${CONTACT_VOXEL_SIZE:-0.01}"
SUCCESS_POSITION_THRESHOLD="${SUCCESS_POSITION_THRESHOLD:-0.10}"
DISABLE_RANDOMIZATION="${DISABLE_RANDOMIZATION:-True}"
START_AT_TIMESTEP_ZERO_PROB="${START_AT_TIMESTEP_ZERO_PROB:-1.0}"
FREEZE_AT_TIMESTEP_ZERO_PROB="${FREEZE_AT_TIMESTEP_ZERO_PROB:-0.0}"
RESET_NOISE_SCALE="${RESET_NOISE_SCALE:-0.0}"
USE_ADAPTIVE_TIMESTEPS_SAMPLER="${USE_ADAPTIVE_TIMESTEPS_SAMPLER:-False}"
MAX_EPISODE_LENGTH_S="${MAX_EPISODE_LENGTH_S:-1000000}"
PHYSX_GPU_COLLISION_STACK_SIZE="${PHYSX_GPU_COLLISION_STACK_SIZE:-268435456}"
MAX_ROLLOUT_STEPS="${MAX_ROLLOUT_STEPS:-}"
VALIDATE_OUTPUT_FORMAT="${VALIDATE_OUTPUT_FORMAT:-1}"
DRY_RUN="${DRY_RUN:-0}"
if [[ -z "${PUBLISH_FOR_INFER_BOX+x}" ]]; then
  if is_truthy "${VIEW_ONLY}"; then
    PUBLISH_FOR_INFER_BOX=0
  else
    PUBLISH_FOR_INFER_BOX=1
  fi
fi
PUBLISH_CLIPS_DIR="${PUBLISH_CLIPS_DIR:-"${SCRIPT_DIR}/outputs/clips"}"
PUBLISH_MOTION_BANK_DIR="${PUBLISH_MOTION_BANK_DIR:-"${SCRIPT_DIR}/outputs/motion_bank"}"
LAUNCH_VISER="${LAUNCH_VISER:-1}"
VIEWER_BACKGROUND="${VIEWER_BACKGROUND:-1}"
VISER_HOST="${VISER_HOST:-0.0.0.0}"
VISER_PORT="${VISER_PORT:-$((RANDOM % 8976 + 1024))}"
VIEWER_SEQUENCE="${VIEWER_SEQUENCE:-}"
VIEWER_SHOW_ORIGINAL_MOTION="${VIEWER_SHOW_ORIGINAL_MOTION:-1}"
VIEWER_SHOW_PRIMITIVE_BOX="${VIEWER_SHOW_PRIMITIVE_BOX:-0}"
VIEWER_SHOW_ROBOT="${VIEWER_SHOW_ROBOT:-1}"
VIEWER_LOG="${VIEWER_LOG:-"${SCRIPT_DIR}/logs/runtime/infer_teacher_as_contacts_viewer_$(date -u +%Y%m%d_%H%M%S).log"}"
ROBOT_URDF="${ROBOT_URDF:-"${SCRIPT_DIR}/src/holosoma/holosoma/data/robots/g1/g1_29dof.urdf"}"
ORIGINAL_MOTION_DIR="${ORIGINAL_MOTION_DIR:-"${OMOMO_DATA_DIR}"}"

HEADLESS_FLAG="$(normalize_bool_flag "${HEADLESS}")"
if [[ "${HEADLESS_FLAG}" == "True" ]]; then
  export HEADLESS=1
else
  export HEADLESS=0
fi

export WANDB_PROJECT
unset HOLOSOMA_DISABLE_HETEROGENEOUS_OBJECT_SINGLE_SLOT
export HOLOSOMA_OBJECT_SPAWN_MODE="${OBJECT_SPAWN_MODE}"
export HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE="${OBJECT_GEOMETRY_MODE}"
export HOLOSOMA_OBJECT_COLLIDER_TYPE="${HOLOSOMA_OBJECT_COLLIDER_TYPE:-convex_decomposition}"
export HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK="${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK:-1}"
export HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS=1
export VISER_LOAD_URDF="${VISER_LOAD_URDF:-1}"

if is_truthy "${VIEW_ONLY}"; then
  if [[ ! -d "${OUTPUT_DIR}/clips" ]]; then
    echo "[ERROR] View-only output dir is missing clips/: ${OUTPUT_DIR}" >&2
    exit 2
  fi
  if ! compgen -G "${OUTPUT_DIR}/clips/*/teacher_rollout_reference.npz" >/dev/null; then
    echo "[ERROR] View-only output dir has no teacher_rollout_reference.npz files under ${OUTPUT_DIR}/clips" >&2
    exit 2
  fi
  echo "[INFO] view_only=1"
  echo "[INFO] output_dir=${OUTPUT_DIR}"
  echo "[INFO] publish_for_infer_box=${PUBLISH_FOR_INFER_BOX}"
  echo "[INFO] launch_viser=${LAUNCH_VISER}"
else
  cmd=(
    "${PYTHON_BIN}" src/holosoma/holosoma/export_teacher_box_contacts.py
    --checkpoint "${TEACHER_CHECKPOINT}"
    --output-dir "${OUTPUT_DIR}"
    --min-contact-frames "${MIN_CONTACT_FRAMES}"
    --contact-force-threshold "${CONTACT_FORCE_THRESHOLD}"
    --contact-voxel-size "${CONTACT_VOXEL_SIZE}"
    --success-position-threshold "${SUCCESS_POSITION_THRESHOLD}"
    --training.num-envs "${NUM_ENVS}"
    --training.headless "${HEADLESS_FLAG}"
    --simulator.config.sim.max_episode_length_s "${MAX_EPISODE_LENGTH_S}"
    --simulator.config.sim.physx.gpu_collision_stack_size "${PHYSX_GPU_COLLISION_STACK_SIZE}"
    --robot.object.enabled True
    --robot.object.object-urdf-path "${OMOMO_OBJECT_MAP}"
    --command.setup-terms.motion-command.params.motion-config.motion-file "${OMOMO_DATA_DIR}"
    --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler "${USE_ADAPTIVE_TIMESTEPS_SAMPLER}"
    --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob "${START_AT_TIMESTEP_ZERO_PROB}"
    --command.setup-terms.motion-command.params.motion-config.freeze-at-timestep-zero-prob "${FREEZE_AT_TIMESTEP_ZERO_PROB}"
    --command.setup-terms.motion-command.params.motion-config.noise-to-initial-pose.overall-noise-scale "${RESET_NOISE_SCALE}"
    --perception.object-geometry-mode "${OBJECT_GEOMETRY_MODE}"
  )

  if [[ -n "${MAX_ROLLOUT_STEPS}" ]]; then
    cmd+=(--max-rollout-steps "${MAX_ROLLOUT_STEPS}")
  fi

  if is_truthy "${DISABLE_RANDOMIZATION}"; then
    cmd+=(randomization:disabled)
  fi

  if [[ $# -gt 0 ]]; then
    cmd+=("$@")
  fi

  echo "[INFO] teacher_checkpoint=${TEACHER_CHECKPOINT}"
  echo "[INFO] motion_dir=${OMOMO_DATA_DIR}"
  echo "[INFO] object_urdf=${OMOMO_OBJECT_MAP}"
  echo "[INFO] as_expected_total=${AS_EXPECTED_TOTAL}"
  echo "[INFO] num_envs=${NUM_ENVS}"
  echo "[INFO] object_spawn_mode=${HOLOSOMA_OBJECT_SPAWN_MODE}"
  echo "[INFO] require_single_slot_objects=${HOLOSOMA_REQUIRE_SINGLE_SLOT_OBJECTS}"
  echo "[INFO] shard_object_assets_by_rank=${HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK}"
  echo "[INFO] object_geometry_mode=${HOLOSOMA_PERCEPTION_OBJECT_GEOMETRY_MODE}"
  echo "[INFO] start_at_timestep_zero_prob=${START_AT_TIMESTEP_ZERO_PROB}"
  echo "[INFO] freeze_at_timestep_zero_prob=${FREEZE_AT_TIMESTEP_ZERO_PROB}"
  echo "[INFO] reset_noise_scale=${RESET_NOISE_SCALE}"
  echo "[INFO] disable_randomization=${DISABLE_RANDOMIZATION}"
  echo "[INFO] output_dir=${OUTPUT_DIR}"
  echo "[INFO] publish_for_infer_box=${PUBLISH_FOR_INFER_BOX}"
  echo "[INFO] launch_viser=${LAUNCH_VISER}"

  if is_truthy "${DRY_RUN}"; then
    printf '%q ' "${cmd[@]}"
    printf '\n'
    if is_truthy "${PUBLISH_FOR_INFER_BOX}"; then
      echo "[INFO] dry_run_publish_clips_dir=${PUBLISH_CLIPS_DIR}"
      echo "[INFO] dry_run_publish_motion_bank_dir=${PUBLISH_MOTION_BANK_DIR}"
    fi
    if is_truthy "${LAUNCH_VISER}"; then
      echo "[INFO] dry_run_viewer_data_root=${OUTPUT_DIR}"
      echo "[INFO] dry_run_viewer_original_motion_dir=${ORIGINAL_MOTION_DIR}"
      echo "[INFO] dry_run_viewer_url=http://localhost:${VISER_PORT}"
    fi
    exit 0
  fi

  "${cmd[@]}"

  if is_truthy "${VALIDATE_OUTPUT_FORMAT}"; then
    "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${AS_EXPECTED_TOTAL}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

output_root = Path(sys.argv[1]).expanduser().resolve()
expected_raw = sys.argv[2].strip()
expected = int(expected_raw) if expected_raw else None
clips_root = output_root / "clips"
motion_bank = output_root / "motion_bank"

if not clips_root.is_dir():
    raise SystemExit(f"[ERROR] Export format validation failed: missing clips dir {clips_root}")
if not motion_bank.is_dir():
    raise SystemExit(f"[ERROR] Export format validation failed: missing motion_bank dir {motion_bank}")

clip_dirs = sorted(path for path in clips_root.iterdir() if path.is_dir())
motion_files = sorted(motion_bank.glob("*.npz"))
if expected is not None and len(clip_dirs) != expected:
    raise SystemExit(
        f"[ERROR] Export format validation failed: expected {expected} clip dirs, found {len(clip_dirs)}"
    )
if expected is not None and len(motion_files) != expected:
    raise SystemExit(
        f"[ERROR] Export format validation failed: expected {expected} rollout motion files, found {len(motion_files)}"
    )
if not clip_dirs:
    raise SystemExit(f"[ERROR] Export format validation failed: no clip dirs under {clips_root}")

required_files = (
    "teacher_rollout_reference.npz",
    "primitive_contact_points.npy",
    "primitive_contact_point_counts.npy",
    "contact_intervals.json",
    "contact_intervals.npz",
)
region_labels = (
    "left_wrist",
    "right_wrist",
    "arm",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "torso",
)

missing: list[str] = []
bad_shapes: list[str] = []
for clip_dir in clip_dirs:
    clip_required_files = list(required_files)
    for label in region_labels:
        clip_required_files.extend(
            (
                f"{label}_contact_points.npy",
                f"{label}_contact_point_counts.npy",
                f"{label}_contact_interval_steps.npy",
            )
        )
    for file_name in clip_required_files:
        path = clip_dir / file_name
        if not path.is_file():
            missing.append(f"{clip_dir.name}/{file_name}")
            continue
        if file_name.endswith("_contact_points.npy"):
            arr = np.load(path)
            if arr.ndim != 2 or arr.shape[1] != 3:
                bad_shapes.append(f"{clip_dir.name}/{file_name}:{arr.shape}")
        elif file_name.endswith("_contact_point_counts.npy"):
            arr = np.load(path)
            if arr.ndim != 1:
                bad_shapes.append(f"{clip_dir.name}/{file_name}:{arr.shape}")
        elif file_name.endswith("_contact_interval_steps.npy"):
            arr = np.load(path)
            if arr.shape != (2,):
                bad_shapes.append(f"{clip_dir.name}/{file_name}:{arr.shape}")
    interval_payload = json.loads((clip_dir / "contact_intervals.json").read_text(encoding="utf-8"))
    if not isinstance(interval_payload, dict):
        bad_shapes.append(f"{clip_dir.name}/contact_intervals.json:not_dict")

if missing:
    preview = ", ".join(missing[:20])
    raise SystemExit(f"[ERROR] Export format validation failed: missing files: {preview}")
if bad_shapes:
    preview = ", ".join(bad_shapes[:20])
    raise SystemExit(f"[ERROR] Export format validation failed: invalid array/json shapes: {preview}")

print(
    f"[INFO] Validated AS contact export format: {len(clip_dirs)} clips, "
    f"{len(motion_files)} rollout motion files -> {output_root}"
)
PY
  fi
fi

if is_truthy "${PUBLISH_FOR_INFER_BOX}"; then
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" "${PUBLISH_CLIPS_DIR}" "${PUBLISH_MOTION_BANK_DIR}" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

source_root = Path(sys.argv[1]).expanduser().resolve()
clips_dst = Path(sys.argv[2]).expanduser().resolve()
motion_dst = Path(sys.argv[3]).expanduser().resolve()

source_clips = source_root / "clips"
source_motion = source_root / "motion_bank"
if not source_clips.is_dir():
    raise SystemExit(f"[ERROR] export did not create clips dir: {source_clips}")
if not source_motion.is_dir():
    raise SystemExit(f"[ERROR] export did not create motion_bank dir: {source_motion}")

clips_dst.mkdir(parents=True, exist_ok=True)
motion_dst.mkdir(parents=True, exist_ok=True)

clip_count = 0
for clip_dir in sorted(source_clips.iterdir()):
    if not clip_dir.is_dir():
        continue
    dst = clips_dst / clip_dir.name
    shutil.copytree(clip_dir, dst, dirs_exist_ok=True)
    clip_count += 1

motion_count = 0
for path in sorted(source_motion.iterdir()):
    if path.is_file():
        shutil.copy2(path, motion_dst / path.name)
        if path.suffix == ".npz":
            motion_count += 1

manifest = {
    "source_root": str(source_root),
    "clips_dir": str(clips_dst),
    "motion_bank_dir": str(motion_dst),
    "clip_count": clip_count,
    "motion_count": motion_count,
}
(clips_dst / "_as_teacher_contacts_publish_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"[INFO] Published AS contact clips: {clip_count} -> {clips_dst}")
print(f"[INFO] Published AS rollout motion clips: {motion_count} -> {motion_dst}")
PY
fi

if is_truthy "${LAUNCH_VISER}"; then
  export PYTHONPATH="${SCRIPT_DIR}/src/holosoma:${SCRIPT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

  viewer_cmd=(
    "${PYTHON_BIN}" -m holosoma.debug_rollout_viewer
    --data-root "${OUTPUT_DIR}"
    --vis-root "${OUTPUT_DIR}"
    --stats-root "${OUTPUT_DIR}"
    --original-motion-dir "${ORIGINAL_MOTION_DIR}"
    --host "${VISER_HOST}"
    --port "${VISER_PORT}"
  )

  if is_truthy "${VIEWER_SHOW_ROBOT}"; then
    viewer_cmd+=(--robot-urdf "${ROBOT_URDF}")
  else
    viewer_cmd+=(--no-robot)
  fi
  if is_truthy "${VIEWER_SHOW_ORIGINAL_MOTION}"; then
    viewer_cmd+=(--show-original-motion)
  fi
  if is_truthy "${VIEWER_SHOW_PRIMITIVE_BOX}"; then
    viewer_cmd+=(--show-primitive-box)
  fi
  if [[ -n "${VIEWER_SEQUENCE}" ]]; then
    viewer_cmd+=(--sequence "${VIEWER_SEQUENCE}")
  fi

  echo "[INFO] Launching Viser rollout/contact viewer"
  echo "[INFO] viewer_data_root=${OUTPUT_DIR}"
  echo "[INFO] viewer_original_motion_dir=${ORIGINAL_MOTION_DIR}"
  echo "[INFO] viewer_url=http://localhost:${VISER_PORT}"

  if is_truthy "${DRY_RUN}"; then
    printf '[DRY_RUN] viewer'
    printf ' %q' "${viewer_cmd[@]}"
    printf '\n'
    exit 0
  fi

  if is_truthy "${VIEWER_BACKGROUND}"; then
    mkdir -p "$(dirname "${VIEWER_LOG}")"
    {
      printf '[INFO] command:'
      printf ' %q' "${viewer_cmd[@]}"
      printf '\n'
    } > "${VIEWER_LOG}"
    setsid "${viewer_cmd[@]}" >> "${VIEWER_LOG}" 2>&1 < /dev/null &
    viewer_pid=$!
    echo "[INFO] viewer_pid=${viewer_pid}"
    echo "[INFO] viewer_log=${VIEWER_LOG}"
    echo "[INFO] ssh_tunnel=ssh -N -L ${VISER_PORT}:localhost:${VISER_PORT} <user>@<host>"
  else
    exec "${viewer_cmd[@]}"
  fi
fi
