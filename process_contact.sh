#!/usr/bin/env bash
set -euo pipefail

# Process BEHAVE z-up sequences and plot velocity signals.
#
# Outputs:
#   velocity/<SEQ_NAME>/hands_velocity.png
#   velocity/<SEQ_NAME>/relative_velocity.png
#   velocity/<SEQ_NAME>/hands_velocity.npy
#   velocity/<SEQ_NAME>/relative_velocity.npy
#
# Environment overrides:
#   DATA_ROOT=/data/behave/annotation_30fps_zup
#   OUT_ROOT=./velocity
#   FPS=30
#   SEQ_NAME=Date03_Sub03_boxlarge   # optional single sequence

DATA_ROOT=${DATA_ROOT:-"/data/behave/annotation_30fps_zup"}
OUT_ROOT=${OUT_ROOT:-"velocity"}
FPS=${FPS:-"30"}
SEQ_NAME=${SEQ_NAME:-""}

python - <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - defensive
    raise RuntimeError("matplotlib is required for plotting velocity curves.") from exc

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/data/behave/annotation_30fps_zup"))
OUT_ROOT = Path(os.environ.get("OUT_ROOT", "velocity"))
fps_raw = os.environ.get("FPS", "30")
SEQ_NAME = os.environ.get("SEQ_NAME", "").strip()
try:
    FPS = float(fps_raw)
except ValueError as exc:
    raise ValueError(f"Invalid FPS value: {fps_raw}") from exc

if not DATA_ROOT.exists():
    raise FileNotFoundError(f"Missing DATA_ROOT: {DATA_ROOT}")

def _load_npz(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _get_object_trans(obj: dict[str, object]) -> np.ndarray:
    trans = obj.get("trans")
    if trans is None:
        trans = obj.get("obj_trans")
    if trans is None:
        raise KeyError("object_fit_all.npz missing trans/obj_trans")
    trans = np.asarray(trans, dtype=np.float32)
    if trans.ndim == 1:
        trans = trans[None, :]
    return trans


def _ensure_length(arr: np.ndarray, length: int, name: str) -> np.ndarray:
    if arr.shape[0] == length:
        return arr
    if arr.shape[0] == 1:
        return np.repeat(arr, length, axis=0)
    raise ValueError(f"{name} length {arr.shape[0]} does not match {length}")


def _velocity(pos: np.ndarray, dt: float) -> np.ndarray:
    vel = np.zeros_like(pos)
    vel[1:] = (pos[1:] - pos[:-1]) / dt
    return vel


def _plot_series(time_s: np.ndarray, series: dict[str, np.ndarray], title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    for name, values in series.items():
        plt.plot(time_s, values, label=name)
    plt.xlabel("time (s)")
    plt.ylabel("speed (m/s)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _process_sequence(seq_dir: Path) -> None:
    smpl_path = seq_dir / "smpl_fit_all.npz"
    obj_path = seq_dir / "object_fit_all.npz"
    if not smpl_path.exists() or not obj_path.exists():
        return

    smpl = _load_npz(smpl_path)
    if "global_joint_positions" not in smpl:
        raise KeyError(f"global_joint_positions not found in {smpl_path}")
    joints = np.asarray(smpl["global_joint_positions"], dtype=np.float32)
    if joints.ndim != 3 or joints.shape[2] != 3:
        raise ValueError(f"Invalid joints shape {joints.shape} in {smpl_path}")

    obj = _load_npz(obj_path)
    obj_trans = _get_object_trans(obj)

    num_frames = joints.shape[0]
    obj_trans = _ensure_length(obj_trans, num_frames, "object_trans")

    # SMPL 22 joint indices: 20 = L_Wrist, 21 = R_Wrist, 0 = Pelvis
    l_wrist = joints[:, 20]
    r_wrist = joints[:, 21]
    hand_mid = 0.5 * (l_wrist + r_wrist)

    dt = 1.0 / FPS
    l_vel = _velocity(l_wrist, dt)
    r_vel = _velocity(r_wrist, dt)
    l_speed = np.linalg.norm(l_vel, axis=1)
    r_speed = np.linalg.norm(r_vel, axis=1)

    rel_pos = hand_mid - obj_trans
    rel_vel = _velocity(rel_pos, dt)
    rel_speed = np.linalg.norm(rel_vel, axis=1)

    t = np.arange(num_frames, dtype=np.float32) * dt

    out_dir = OUT_ROOT / seq_dir.name
    _plot_series(
        t,
        {"L_Wrist": l_speed, "R_Wrist": r_speed},
        f"{seq_dir.name} hand speed",
        out_dir / "hands_velocity.png",
    )
    _plot_series(
        t,
        {"hands_mid_rel_box_center": rel_speed},
        f"{seq_dir.name} hands-mid rel-box speed",
        out_dir / "relative_velocity.png",
    )

    np.save(out_dir / "hands_velocity.npy", np.stack([l_speed, r_speed], axis=1))
    np.save(out_dir / "relative_velocity.npy", rel_speed)


seq_dirs = []
if SEQ_NAME:
    seq_dir = DATA_ROOT / SEQ_NAME
    if not seq_dir.exists():
        raise FileNotFoundError(f"Missing sequence: {seq_dir}")
    seq_dirs = [seq_dir]
else:
    seq_dirs = sorted(p for p in DATA_ROOT.iterdir() if p.is_dir())

for seq_dir in seq_dirs:
    _process_sequence(seq_dir)

print(f"Saved velocity plots to: {OUT_ROOT}")
PY
