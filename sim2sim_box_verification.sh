#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash sim2sim_box_verification.sh <motion.npz> [checkpoint.pt|wandb://...|https://wandb.ai/.../runs/.../files] [extra infer_box_joystick args...]

Purpose:
  Reuse the current `infer_box_joystick.sh` checkpoint + a single motion clip,
  but switch the simulator to MuJoCo for sim2sim verification.

Defaults:
  SIM2SIM_MODE=mocap
  NUM_ENVS=1
  HEADLESS=True
  BACKGROUND=0

Useful env vars:
  OBJECT_URDF        Override the object URDF for this motion.
  BACKGROUND=1       Run under nohup and write a log file.
  SIM2SIM_LOG_DIR    Log directory for background runs (default: ./logs/sim2sim).
  SIM2SIM_LOG_FILE   Explicit log file path for background runs.
  SIM2SIM_MODE       mocap|depth (default: mocap).

Examples:
  bash sim2sim_box_verification.sh /abs/path/sub3_largebox_016_mj_w_obj.npz
  BACKGROUND=1 bash sim2sim_box_verification.sh /abs/path/clip.npz wandb://zihanw22/WholeBodyTracking/d20ktze6/model_00800.pt
  HEADLESS=False bash sim2sim_box_verification.sh /abs/path/clip.npz --viser-port 18080
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

case "${1}" in
  -h|--help|help)
    usage
    exit 0
    ;;
esac

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

MOTION_FILE_RAW="$1"
shift

CHECKPOINT=""
if [[ $# -gt 0 ]]; then
  case "$1" in
    wandb://*|https://wandb.ai/*|/*|./*|../*|*.pt)
      CHECKPOINT="$1"
      shift
      ;;
  esac
fi

BACKGROUND_FLAG="${BACKGROUND:-0}"
FORWARD_ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == "--background" ]]; then
    BACKGROUND_FLAG=1
    continue
  fi
  FORWARD_ARGS+=("${arg}")
done

MOTION_FILE="$(python - "${MOTION_FILE_RAW}" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

if [[ ! -f "${MOTION_FILE}" ]]; then
  echo "[ERROR] motion file not found: ${MOTION_FILE}" >&2
  exit 1
fi

resolve_object_urdf() {
  python - "${SCRIPT_DIR}" "${MOTION_FILE}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


repo_root = Path(sys.argv[1]).resolve()
motion_path = Path(sys.argv[2]).resolve()
clip_id = motion_path.stem

DEFAULT_URDFS = {
    "objects_largebox": repo_root / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
    "largebox": repo_root / "src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
    "boxlarge": repo_root / "src/holosoma_retargeting/models/behave_objects/boxlarge/boxlarge.urdf",
    "boxmedium": repo_root / "src/holosoma_retargeting/models/behave_objects/boxmedium/boxmedium.urdf",
    "boxsmall": repo_root / "src/holosoma_retargeting/models/behave_objects/boxsmall/boxsmall.urdf",
    "boxtiny": repo_root / "src/holosoma_retargeting/models/behave_objects/boxtiny/boxtiny.urdf",
    "boxlong": repo_root / "src/holosoma_retargeting/models/behave_objects/boxlong/boxlong.urdf",
}


def scalar_str(value: object) -> str:
    arr = np.asarray(value)
    if arr.size == 0:
        return ""
    item = arr.reshape(-1)[0]
    if hasattr(item, "item"):
        item = item.item()
    return str(item).strip()


def resolve_candidate(raw_path: str, *, base_dir: Path) -> Path | None:
    raw_path = raw_path.strip()
    if not raw_path:
        return None

    if raw_path.startswith("holosoma/data/"):
        candidate = (repo_root / "src/holosoma" / raw_path).resolve()
        return candidate if candidate.exists() else None

    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve() if path.exists() else None

    relative_to_base = (base_dir / path).resolve()
    if relative_to_base.exists():
        return relative_to_base

    relative_to_repo = (repo_root / path).resolve()
    if relative_to_repo.exists():
        return relative_to_repo

    resolved = path.resolve()
    return resolved if resolved.exists() else None


def fallback_from_name(name: str) -> Path | None:
    key = name.strip().lower()
    if not key:
        return None
    for token, path in DEFAULT_URDFS.items():
        if key == token or key.endswith(f"_{token}") or token in key:
            return path.resolve() if path.exists() else None
    return None


def resolve_from_npz() -> tuple[Path | None, str]:
    object_name = ""
    with np.load(motion_path, allow_pickle=True) as data:
        if "object_urdf_path" in data:
            candidate = resolve_candidate(scalar_str(data["object_urdf_path"]), base_dir=motion_path.parent)
            if candidate is not None:
                return candidate, scalar_str(data["object_name"]) if "object_name" in data else ""
        if "object_name" in data:
            object_name = scalar_str(data["object_name"])
    return None, object_name


resolved_path, object_name = resolve_from_npz()
if resolved_path is not None:
    print(resolved_path)
    raise SystemExit(0)

for map_name in ("_clip_object_urdf_map.json", "clip_object_urdf_map.json"):
    map_path = motion_path.parent / map_name
    if not map_path.is_file():
        continue
    payload = json.loads(map_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("clips"), dict):
        payload = payload["clips"]
    if not isinstance(payload, dict):
        continue
    entry = payload.get(clip_id)
    if entry is None:
        continue
    raw_path = entry.strip() if isinstance(entry, str) else str(entry.get("object_urdf_path", "")).strip()
    if isinstance(entry, dict) and not object_name:
        object_name = str(entry.get("object_name", "")).strip()
    candidate = resolve_candidate(raw_path, base_dir=map_path.parent)
    if candidate is not None:
        print(candidate)
        raise SystemExit(0)

for hint in (object_name, clip_id):
    candidate = fallback_from_name(hint)
    if candidate is not None:
        print(candidate)
        raise SystemExit(0)
PY
}

OBJECT_URDF_RESOLVED="${OBJECT_URDF:-}"
if [[ -z "${OBJECT_URDF_RESOLVED}" ]]; then
  OBJECT_URDF_RESOLVED="$(resolve_object_urdf || true)"
fi

if [[ -z "${OBJECT_URDF_RESOLVED}" ]]; then
  echo "[ERROR] Could not resolve OBJECT_URDF for motion: ${MOTION_FILE}" >&2
  echo "[ERROR] Set OBJECT_URDF=/abs/path/to/object.urdf explicitly." >&2
  exit 1
fi
if [[ ! -f "${OBJECT_URDF_RESOLVED}" ]]; then
  echo "[ERROR] object URDF not found: ${OBJECT_URDF_RESOLVED}" >&2
  exit 1
fi

export PYTHONPATH="${SCRIPT_DIR}/src/holosoma${PYTHONPATH:+:${PYTHONPATH}}"
export MOTION_DIR="${MOTION_FILE}"
export OBJECT_URDF="${OBJECT_URDF_RESOLVED}"
export NUM_ENVS="${NUM_ENVS:-1}"
export HEADLESS="${HEADLESS:-True}"

MODE="${SIM2SIM_MODE:-mocap}"

if ! python - <<'PY' >/dev/null 2>&1
import mujoco  # noqa: F401
PY
then
  echo "[ERROR] Python environment is missing MuJoCo. Install/activate an env with both 'mujoco' and local holosoma deps." >&2
  exit 1
fi

if ! python - <<'PY' >/dev/null 2>&1
import holosoma  # noqa: F401
PY
then
  echo "[ERROR] Python environment cannot import holosoma even after setting PYTHONPATH." >&2
  exit 1
fi

CMD=(bash "${SCRIPT_DIR}/infer_box_joystick.sh" "${MODE}")
if [[ -n "${CHECKPOINT}" ]]; then
  CMD+=("${CHECKPOINT}")
fi
CMD+=(
  simulator:mujoco
  --simulator.config.mujoco-backend classic
  "${FORWARD_ARGS[@]}"
)

echo "[INFO] sim2sim mode=${MODE}"
echo "[INFO] motion=${MOTION_FILE}"
echo "[INFO] object_urdf=${OBJECT_URDF_RESOLVED}"
if [[ -n "${CHECKPOINT}" ]]; then
  echo "[INFO] checkpoint=${CHECKPOINT}"
fi
echo "[INFO] headless=${HEADLESS}"
echo "[INFO] command=${CMD[*]}"

if [[ "${BACKGROUND_FLAG}" == "1" || "${BACKGROUND_FLAG}" == "true" ]]; then
  LOG_DIR="${SIM2SIM_LOG_DIR:-${SCRIPT_DIR}/logs/sim2sim}"
  mkdir -p "${LOG_DIR}"
  LOG_FILE="${SIM2SIM_LOG_FILE:-${LOG_DIR}/$(basename "${MOTION_FILE}" .npz)_$(date -u +%Y%m%dT%H%M%SZ).log}"
  nohup "${CMD[@]}" >"${LOG_FILE}" 2>&1 &
  echo "[INFO] background pid=$!"
  echo "[INFO] log=${LOG_FILE}"
  exit 0
fi

exec "${CMD[@]}"
