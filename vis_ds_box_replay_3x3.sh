#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_ID="${GPU_ID:-7}"
VISER_PORT="${VISER_PORT:-18085}"
VISER_ENV_ID="${VISER_ENV_ID:-0}"
VISER_GRID_SPACING="${VISER_GRID_SPACING:-2.4}"
VISER_MULTI_ENV_COLS="${VISER_MULTI_ENV_COLS:-3}"
VISER_START_PAUSED="${VISER_START_PAUSED:-1}"
HOLOSOMA_REPLAY_KEEP_OPEN="${HOLOSOMA_REPLAY_KEEP_OPEN:-1}"
NUM_ENVS="${NUM_ENVS:-9}"
TRAINING_NAME="${TRAINING_NAME:-ds_box_replay_3x3}"
MOTION_ROOT="${MOTION_ROOT:-${ROOT_DIR}/data/ds_box_data/train_g1_w_obj_prepared}"
REPLAY_SUBSET_DIR="${REPLAY_SUBSET_DIR:-${ROOT_DIR}/data/ds_box_data/train_g1_w_obj_prepared_replay3x3}"
CLIP_LIST="${CLIP_LIST:-}"
DRY_RUN="${DRY_RUN:-0}"

HEADLESS_RAW="${HEADLESS:-True}"
case "$(printf '%s' "${HEADLESS_RAW}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    TRAINING_HEADLESS="True"
    ;;
  0|false|no|off)
    TRAINING_HEADLESS="False"
    ;;
  *)
    echo "[ERROR] HEADLESS must be True/False/1/0, got: ${HEADLESS_RAW}" >&2
    exit 1
    ;;
esac

if [[ "${NUM_ENVS}" != "9" ]]; then
  echo "[ERROR] vis_ds_box_replay_3x3.sh is fixed to NUM_ENVS=9, got ${NUM_ENVS}" >&2
  exit 2
fi

if [[ ! -d "${MOTION_ROOT}" ]]; then
  echo "[ERROR] MOTION_ROOT not found: ${MOTION_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${MOTION_ROOT}/_clip_object_urdf_map.json" ]]; then
  echo "[ERROR] clip-object map missing under MOTION_ROOT: ${MOTION_ROOT}" >&2
  exit 1
fi

mkdir -p "${REPLAY_SUBSET_DIR}"

SELECTED_CLIPS="$(${PYTHON_BIN} - <<'PY' "${MOTION_ROOT}" "${REPLAY_SUBSET_DIR}" "${CLIP_LIST}"
import json
import sys
from pathlib import Path

motion_root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
clip_list_raw = sys.argv[3].strip()
map_path = motion_root / '_clip_object_urdf_map.json'
payload = json.loads(map_path.read_text(encoding='utf-8'))
clips_map = payload['clips'] if isinstance(payload, dict) and 'clips' in payload else payload
if not isinstance(clips_map, dict) or not clips_map:
    raise SystemExit(f'Invalid clip map: {map_path}')

rows = []
for clip, entry in clips_map.items():
    size = entry.get('object_size') if isinstance(entry, dict) else None
    if size is None or len(size) != 3:
        continue
    sx, sy, sz = [float(v) for v in size]
    vol = sx * sy * sz
    aspect = max(sx, sy, sz) / max(min(sx, sy, sz), 1.0e-8)
    rows.append({
        'clip': clip,
        'size': (sx, sy, sz),
        'vol': vol,
        'aspect': aspect,
        'entry': entry,
    })
if len(rows) < 9:
    raise SystemExit(f'Need at least 9 clips with object_size, found {len(rows)}')

if clip_list_raw:
    selected_names = [item.strip() for item in clip_list_raw.split(',') if item.strip()]
    if len(selected_names) != 9:
        raise SystemExit(f'CLIP_LIST must contain exactly 9 comma-separated clip ids, got {len(selected_names)}')
    missing = [name for name in selected_names if name not in clips_map]
    if missing:
        raise SystemExit(f'Unknown clip ids in CLIP_LIST: {missing}')
    row_by_name = {row['clip']: row for row in rows}
    selected = [row_by_name[name] for name in selected_names]
else:
    rows.sort(key=lambda row: (row['vol'], row['aspect'], row['clip']))
    bands = [rows[0:14], rows[14:29], rows[29:43]]
    selected = []
    for band in bands:
        band_sorted = sorted(band, key=lambda row: (row['aspect'], row['vol'], row['clip']))
        for idx in (0, len(band_sorted) // 2, len(band_sorted) - 1):
            selected.append(band_sorted[idx])

out_dir.mkdir(parents=True, exist_ok=True)
for old_npz in out_dir.glob('*.npz'):
    old_npz.unlink()
subset_map = {}
for item in selected:
    clip = item['clip']
    src = motion_root / f'{clip}.npz'
    if not src.is_file():
        raise SystemExit(f'Missing clip file: {src}')
    dst = out_dir / src.name
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)
    subset_map[clip] = item['entry']
(out_dir / '_clip_object_urdf_map.json').write_text(
    json.dumps({'clips': subset_map}, indent=2, sort_keys=True),
    encoding='utf-8',
)
for item in selected:
    sx, sy, sz = item['size']
    print(f"{item['clip']}\t{sx:.4f},{sy:.4f},{sz:.4f}\tvol={item['vol']:.4f}\taspect={item['aspect']:.3f}")
PY
)"

if [[ -z "${SELECTED_CLIPS}" ]]; then
  echo "[ERROR] No clips selected for 3x3 replay." >&2
  exit 1
fi

echo "[INFO] Selected 3x3 clips:"
printf '%s\n' "${SELECTED_CLIPS}"
echo "[INFO] Replay subset dir: ${REPLAY_SUBSET_DIR}"

if [[ "$(printf '%s' "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')" =~ ^(1|true|yes|on)$ ]]; then
  echo "[INFO] DRY_RUN enabled; not launching replay."
  exit 0
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1
export LOGURU_LEVEL="${LOGURU_LEVEL:-INFO}"
export VISER_MULTI_ENV_COLS="${VISER_MULTI_ENV_COLS}"
export VISER_START_PAUSED="${VISER_START_PAUSED}"
export HOLOSOMA_REPLAY_KEEP_OPEN="${HOLOSOMA_REPLAY_KEEP_OPEN}"

cmd=(
  "${PYTHON_BIN}" src/holosoma/holosoma/replay.py
  exp:g1-29dof-wbt-w-object-generalist
  randomization:disabled
  logger:disabled
  --training.name="${TRAINING_NAME}"
  --training.headless="${TRAINING_HEADLESS}"
  --training.debug=True
  --training.num-envs=9
  --training.enable-viser=True
  --training.viser-port="${VISER_PORT}"
  --training.viser-env-id="${VISER_ENV_ID}"
  --training.viser-env-count=9
  --training.viser-multi-env-spacing="${VISER_GRID_SPACING}"
  --training.viser-update-hz=30
  --training.viser-sync-to-sim=True
  --training.viser-force-dt=True
  --training.viser-recenter=True
  --training.viser-show-scandots=False
  --command.setup-terms.motion-command.params.motion-config.motion-file "${REPLAY_SUBSET_DIR}"
  --command.setup-terms.motion-command.params.motion-config.use-adaptive-timesteps-sampler False
  --command.setup-terms.motion-command.params.motion-config.start-at-timestep-zero-prob 1.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-prepend False
  --command.setup-terms.motion-command.params.motion-config.default-pose-prepend-duration-s 0.0
  --command.setup-terms.motion-command.params.motion-config.enable-default-pose-append False
  --command.setup-terms.motion-command.params.motion-config.default-pose-append-duration-s 0.0
  --robot.object.object-urdf-path "${REPLAY_SUBSET_DIR}/_clip_object_urdf_map.json"
)

echo "[INFO] Running DS box 3x3 replay"
echo "[INFO] Open: http://localhost:${VISER_PORT}"
printf '[INFO] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
