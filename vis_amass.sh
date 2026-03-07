#!/usr/bin/env bash
set -euo pipefail

# Kinematic Viser viewer for AMASS-style LAFAN clips.
#
# This script builds a lightweight cache of proxy .npz files with qpos/qvel so
# existing holosoma Viser motion viewer can load them directly.
#
# Default source:
#   /home/ubuntu/FAR/holosoma/amass/LAFAN1_npz
#
# Usage:
#   bash vis_amass.sh
#   AMASS_SRC_DIR=/abs/path/to/amass_folder bash vis_amass.sh
#   START_CLIP=dance1_subject1 PORT=8090 bash vis_amass.sh
#
# Optional env vars:
#   AMASS_SRC_DIR   source folder with .npz clips
#   CACHE_DIR       output proxy folder
#   REF_LAFAN_DIR   reference converted lafan folder for auto 29dof remap
#   PORT            viser port
#   START_CLIP      initial clip stem
#   AUTOPLAY        True|False
#   LOOP            True|False
#   PRELOAD         True|False
#   FPS             override playback fps (empty = from file)
#   ROBOT           robot preset in viewer (default: g1_29dof)
#   MAX_CLIPS       limit number of clips converted (0 = all)
#   AUTO_MAP        True|False (learn 29->29 mapping from REF_LAFAN_DIR)
#   ORDER_MODE      amass_csv|auto_ref|identity (default: amass_csv)
#   WRIST_POLICY    mapped|zero (default: mapped)
#   CONVERT_ONLY    True|False (default: False)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

AMASS_SRC_DIR=${AMASS_SRC_DIR:-"${SCRIPT_DIR}/amass/LAFAN1_npz"}
CACHE_DIR=${CACHE_DIR:-"${SCRIPT_DIR}/.cache/vis_amass_lafan1_proxy"}
REF_LAFAN_DIR=${REF_LAFAN_DIR:-"${SCRIPT_DIR}/src/holosoma_retargeting_my/converted_res/robot_only/lafan"}
PORT=${PORT:-"$((RANDOM % 8976 + 1024))"}
START_CLIP=${START_CLIP:-""}
AUTOPLAY=${AUTOPLAY:-"True"}
LOOP=${LOOP:-"True"}
PRELOAD=${PRELOAD:-"False"}
FPS=${FPS:-""}
ROBOT=${ROBOT:-"g1_29dof"}
MAX_CLIPS=${MAX_CLIPS:-0}
AUTO_MAP=${AUTO_MAP:-"True"}
ORDER_MODE=${ORDER_MODE:-"amass_csv"}
WRIST_POLICY=${WRIST_POLICY:-"mapped"}
CONVERT_ONLY=${CONVERT_ONLY:-"False"}
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ ! -d "${AMASS_SRC_DIR}" ]]; then
  echo "[ERROR] AMASS_SRC_DIR not found: ${AMASS_SRC_DIR}" >&2
  exit 1
fi

mkdir -p "${CACHE_DIR}"

echo "[INFO] Source AMASS dir : ${AMASS_SRC_DIR}"
echo "[INFO] Proxy cache dir  : ${CACHE_DIR}"
echo "[INFO] Ref LAFAN dir    : ${REF_LAFAN_DIR}"
echo "[INFO] Order mode       : ${ORDER_MODE}"
echo "[INFO] Building proxy clips (qpos/qvel)..."

"${PYTHON_BIN}" - <<'PY' "${AMASS_SRC_DIR}" "${CACHE_DIR}" "${MAX_CLIPS}" "${REF_LAFAN_DIR}" "${AUTO_MAP}" "${ORDER_MODE}" "${WRIST_POLICY}"
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

src_dir = Path(sys.argv[1]).resolve()
dst_dir = Path(sys.argv[2]).resolve()
max_clips = int(sys.argv[3])
ref_dir = Path(sys.argv[4]).resolve()
auto_map = sys.argv[5].strip().lower() in {"1", "true", "yes", "on"}
order_mode = sys.argv[6].strip().lower()
wrist_policy = sys.argv[7].strip().lower()
if wrist_policy not in {"mapped", "zero"}:
    raise SystemExit(f"[ERROR] WRIST_POLICY must be mapped|zero, got: {wrist_policy}")
if order_mode not in {"amass_csv", "auto_ref", "identity"}:
    raise SystemExit(f"[ERROR] ORDER_MODE must be amass_csv|auto_ref|identity, got: {order_mode}")

# Backward compatibility: if ORDER_MODE is explicitly identity/auto_ref, keep behavior.
# If ORDER_MODE stays default amass_csv, this takes priority over AUTO_MAP.
if order_mode == "identity" and auto_map:
    order_mode = "auto_ref"

WRIST_IDXS = [19, 20, 21, 26, 27, 28]

files = sorted(src_dir.glob("*.npz"))
if max_clips > 0:
    files = files[:max_clips]

if not files:
    raise SystemExit(f"[ERROR] No .npz files found in {src_dir}")

converted = 0
skipped = 0
force_rebuild = False

def _base_key(stem: str) -> str:
    suffixes = ["_original_mj_fps50", "_original", "_mj_fps50"]
    for s in suffixes:
        if stem.endswith(s):
            return stem[: -len(s)]
    return stem

def _extract_ref_29(path: Path) -> np.ndarray | None:
    with np.load(path, allow_pickle=True) as data:
        if "joint_pos" in data:
            jp = np.asarray(data["joint_pos"])
            if jp.ndim == 2 and jp.shape[1] == 36:
                return jp[:, 7:36].astype(np.float64, copy=False)
            if jp.ndim == 2 and jp.shape[1] == 29:
                return jp.astype(np.float64, copy=False)
        if "qpos" in data:
            qp = np.asarray(data["qpos"])
            if qp.ndim == 2 and qp.shape[1] >= 36:
                return qp[:, 7:36].astype(np.float64, copy=False)
    return None

def _extract_src_29(path: Path) -> np.ndarray | None:
    with np.load(path, allow_pickle=True) as data:
        if "joint_pos" not in data:
            return None
        jp = np.asarray(data["joint_pos"])
        if jp.ndim != 2:
            return None
        if jp.shape[1] == 29:
            return jp.astype(np.float64, copy=False)
        if jp.shape[1] == 36:
            return jp[:, 7:36].astype(np.float64, copy=False)
    return None

def _learn_mapping(src_paths: list[Path], ref_dir: Path):
    if not ref_dir.exists():
        print(f"[WARN] REF_LAFAN_DIR not found, skip AUTO_MAP: {ref_dir}")
        return None
    ref_files = sorted(ref_dir.glob("*.npz"))
    if not ref_files:
        print(f"[WARN] REF_LAFAN_DIR has no npz, skip AUTO_MAP: {ref_dir}")
        return None

    ref_map = {_base_key(p.stem): p for p in ref_files}
    src_map = {_base_key(p.stem): p for p in src_paths}
    keys = sorted(set(ref_map) & set(src_map))
    if not keys:
        print("[WARN] No clip name overlap between source and REF_LAFAN_DIR, skip AUTO_MAP.")
        return None

    d = 29
    sum_xy = np.zeros((d, d), dtype=np.float64)
    sum_x2 = np.zeros((d, d), dtype=np.float64)
    sum_y2 = np.zeros((d, d), dtype=np.float64)
    used = 0

    for k in keys:
        x = _extract_ref_29(ref_map[k])
        y = _extract_src_29(src_map[k])
        if x is None or y is None:
            continue
        n = min(x.shape[0], y.shape[0])
        if n < 8:
            continue
        x = x[:n] - x[:n].mean(axis=0, keepdims=True)
        y = y[:n] - y[:n].mean(axis=0, keepdims=True)
        for i in range(d):
            xi = x[:, i]
            x2 = float(np.dot(xi, xi))
            for j in range(d):
                yj = y[:, j]
                sum_xy[i, j] += float(np.dot(xi, yj))
                sum_x2[i, j] += x2
                sum_y2[i, j] += float(np.dot(yj, yj))
        used += 1

    if used == 0:
        print("[WARN] AUTO_MAP found overlaps but no valid paired arrays, skip.")
        return None

    corr = sum_xy / np.sqrt(np.maximum(sum_x2 * sum_y2, 1e-12))
    abs_corr = np.abs(corr)

    pairs = sorted(
        ((float(abs_corr[i, j]), i, j) for i in range(d) for j in range(d)),
        reverse=True,
    )
    dst_to_src = np.full((d,), -1, dtype=np.int64)
    signs = np.ones((d,), dtype=np.float32)
    confidence = np.zeros((d,), dtype=np.float32)
    used_dst: set[int] = set()
    used_src: set[int] = set()
    for score, i, j in pairs:
        if i in used_dst or j in used_src:
            continue
        dst_to_src[i] = j
        signs[i] = 1.0 if corr[i, j] >= 0 else -1.0
        confidence[i] = float(abs(corr[i, j]))
        used_dst.add(i)
        used_src.add(j)
        if len(used_dst) == d:
            break

    if np.any(dst_to_src < 0):
        print("[WARN] AUTO_MAP incomplete assignment, fallback to identity mapping.")
        return None

    print(f"[INFO] AUTO_MAP learned from {used} paired clips (overlap={len(keys)}).")
    wrist_conf = [float(confidence[i]) for i in WRIST_IDXS]
    print("[INFO] AUTO_MAP wrist confidence:", ", ".join(f"{c:.3f}" for c in wrist_conf))

    return dst_to_src, signs, confidence

def _fixed_amass_csv_mapping():
    # Canonical G1 order <- AMASS npz order (verified against LAFAN1/TWIST2 csv).
    dst_to_src = np.array(
        [
            0, 3, 6, 9, 13, 17,
            1, 4, 7, 10, 14, 18,
            2, 5, 8,
            11, 15, 19, 21, 23, 25, 27,
            12, 16, 20, 22, 24, 26, 28,
        ],
        dtype=np.int64,
    )
    signs = np.ones((29,), dtype=np.float32)
    confidence = np.ones((29,), dtype=np.float32)
    return dst_to_src, signs, confidence

if order_mode == "amass_csv":
    mapping_payload = _fixed_amass_csv_mapping()
    print("[INFO] ORDER_MODE=amass_csv: using fixed joint reorder map.")
elif order_mode == "auto_ref":
    src_for_map = files
    mapping_payload = _learn_mapping(src_for_map, ref_dir)
    if mapping_payload is None:
        print("[WARN] ORDER_MODE=auto_ref unavailable. Using identity order.")
else:
    mapping_payload = None

config_path = dst_dir / "_proxy_config.json"
curr_cfg = {
    "source_dir": str(src_dir),
    "ref_dir": str(ref_dir),
    "auto_map": bool(auto_map),
    "order_mode": order_mode,
    "wrist_policy": wrist_policy,
    "version": 3,
}
if config_path.exists():
    try:
        prev_cfg = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        prev_cfg = None
    if prev_cfg != curr_cfg:
        force_rebuild = True
        print("[INFO] Proxy config changed, forcing cache rebuild.")
else:
    force_rebuild = True

def _map_29(joint_29: np.ndarray) -> np.ndarray:
    if mapping_payload is None:
        out = joint_29.copy()
    else:
        dst_to_src, signs, _ = mapping_payload
        out = np.empty_like(joint_29, dtype=np.float32)
        for dst_i in range(29):
            src_i = int(dst_to_src[dst_i])
            out[:, dst_i] = joint_29[:, src_i] * signs[dst_i]
    if wrist_policy == "zero":
        out[:, WRIST_IDXS] = 0.0
    return out

for src in files:
    dst = dst_dir / src.name
    if (not force_rebuild) and dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        skipped += 1
        continue

    with np.load(src, allow_pickle=True) as data:
        fps = np.asarray(data.get("fps", np.array([50], dtype=np.int64))).reshape(-1)
        fps = np.array([int(fps[0])], dtype=np.int64)

        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float32)
        else:
            required = ("body_pos_w", "body_quat_w", "joint_pos")
            missing = [k for k in required if k not in data]
            if missing:
                raise ValueError(f"{src.name}: missing keys for proxy qpos build: {missing}")

            body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
            joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
            if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
                raise ValueError(f"{src.name}: invalid body_pos_w shape {body_pos_w.shape}")
            if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
                raise ValueError(f"{src.name}: invalid body_quat_w shape {body_quat_w.shape}")
            if joint_pos.ndim != 2:
                raise ValueError(f"{src.name}: invalid joint_pos shape {joint_pos.shape}")
            if joint_pos.shape[1] not in (29, 36):
                raise ValueError(
                    f"{src.name}: unexpected joint_pos dim {joint_pos.shape[1]} (expected 29 or 36)"
                )

            if joint_pos.shape[1] == 36:
                qpos = joint_pos
            else:
                mapped_joint_pos = _map_29(joint_pos)
                root_pos = body_pos_w[:, 0, :]
                root_quat_wxyz = body_quat_w[:, 0, :]
                qpos = np.concatenate([root_pos, root_quat_wxyz, mapped_joint_pos], axis=1)

        if "qvel" in data:
            qvel = np.asarray(data["qvel"], dtype=np.float32)
        elif "joint_vel" in data:
            joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
            if joint_vel.shape[1] in (35, 29):
                if joint_vel.shape[1] == 35:
                    qvel = joint_vel
                else:
                    mapped_joint_vel = _map_29(joint_vel)
                    body_lin_vel = np.asarray(data.get("body_lin_vel_w", np.zeros((joint_vel.shape[0], 1, 3))), dtype=np.float32)
                    body_ang_vel = np.asarray(data.get("body_ang_vel_w", np.zeros((joint_vel.shape[0], 1, 3))), dtype=np.float32)
                    root_lin = body_lin_vel[:, 0, :] if body_lin_vel.ndim == 3 and body_lin_vel.shape[-1] == 3 else np.zeros((joint_vel.shape[0], 3), dtype=np.float32)
                    root_ang = body_ang_vel[:, 0, :] if body_ang_vel.ndim == 3 and body_ang_vel.shape[-1] == 3 else np.zeros((joint_vel.shape[0], 3), dtype=np.float32)
                    qvel = np.concatenate([root_lin, root_ang, mapped_joint_vel], axis=1)
            else:
                qvel = np.zeros((qpos.shape[0], max(0, qpos.shape[1] - 1)), dtype=np.float32)
        else:
            qvel = np.zeros((qpos.shape[0], max(0, qpos.shape[1] - 1)), dtype=np.float32)

    np.savez_compressed(dst, qpos=qpos.astype(np.float32), qvel=qvel.astype(np.float32), fps=fps)
    converted += 1

if mapping_payload is not None:
    dst_to_src, signs, confidence = mapping_payload
    with (dst_dir / "_auto_map_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_dir": str(src_dir),
                "ref_dir": str(ref_dir),
                "order_mode": order_mode,
                "wrist_policy": wrist_policy,
                "dst_to_src": dst_to_src.tolist(),
                "signs": [float(x) for x in signs.tolist()],
                "confidence": [float(x) for x in confidence.tolist()],
                "wrist_indices": WRIST_IDXS,
            },
            f,
            indent=2,
        )

config_path.write_text(json.dumps(curr_cfg, indent=2), encoding="utf-8")

print(f"[INFO] Proxy conversion done. converted={converted}, skipped={skipped}, total={len(files)}")
PY

if [[ "${CONVERT_ONLY,,}" == "true" || "${CONVERT_ONLY}" == "1" ]]; then
  echo "[INFO] CONVERT_ONLY=True, skip launching viewer."
  exit 0
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1; then
import trimesh
import viser
import tyro
PY
  echo "[ERROR] Missing viewer dependencies in ${PYTHON_BIN} environment (need: trimesh, viser, tyro)." >&2
  echo "        Try switching python env, or run with PYTHON_BIN pointing to your training env python." >&2
  exit 1
fi

cmd=(
  "${PYTHON_BIN}" src/holosoma/holosoma/viser_motion_geometry.py
  --motion-dir "${CACHE_DIR}"
  --geometry-dir ""
  --robot "${ROBOT}"
  --port "${PORT}"
  --autoplay "${AUTOPLAY}"
  --loop "${LOOP}"
  --preload "${PRELOAD}"
  --show-geometry False
  --show-object False
)

if [[ -n "${START_CLIP}" ]]; then
  cmd+=(--start-clip "${START_CLIP}")
fi

if [[ -n "${FPS}" ]]; then
  cmd+=(--fps "${FPS}")
fi

echo "[INFO] Viser URL: http://localhost:${PORT}"
echo "[INFO] Running kinematic viewer..."
"${cmd[@]}"
