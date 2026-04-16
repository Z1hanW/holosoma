#!/usr/bin/env python3
"""Verify object geometry metadata for a motion bank.

This is a training-path check, not a Viser check. It reads the motion bank and
clip-object map directly, validates that each clip's object_size matches its
URDF extents, and checks that selected clips start with grounded box primitives.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from holosoma.holosoma.utils.viser_live import load_urdf_geometry_extents  # noqa: E402


def _load_clip_map(motion_root: Path, map_path: Path | None) -> dict[str, Any]:
    resolved = map_path or motion_root / "_clip_object_urdf_map.json"
    if not resolved.is_file():
        raise FileNotFoundError(f"clip-object map not found: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    clips = payload["clips"] if isinstance(payload, dict) and "clips" in payload else payload
    if not isinstance(clips, dict) or not clips:
        raise ValueError(f"invalid clip-object map: {resolved}")
    return clips


def _resolve_urdf(raw: str, motion_root: Path) -> Path:
    raw = str(raw).strip()
    if not raw:
        raise ValueError("empty object_urdf_path")
    path = Path(raw)
    if not path.is_absolute():
        path = motion_root / path
    return path.resolve()


def _extract_object_size(data: np.lib.npyio.NpzFile, clip_name: str) -> np.ndarray:
    if "object_size" not in data:
        raise ValueError(f"{clip_name}: missing object_size")
    raw = np.asarray(data["object_size"], dtype=np.float64)
    if raw.ndim == 0:
        return np.full(3, float(raw), dtype=np.float64)
    if raw.ndim == 1:
        if raw.shape[0] == 1:
            return np.full(3, float(raw[0]), dtype=np.float64)
        if raw.shape[0] >= 3:
            return raw[:3].astype(np.float64)
    if raw.ndim >= 2:
        first = np.asarray(raw[0], dtype=np.float64)
        if first.ndim == 0:
            return np.full(3, float(first), dtype=np.float64)
        if first.shape[0] == 1:
            return np.full(3, float(first[0]), dtype=np.float64)
        if first.shape[0] >= 3:
            return first[:3].astype(np.float64)
    raise ValueError(f"{clip_name}: unsupported object_size shape {raw.shape}")


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError(f"invalid quaternion: {q}")
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _box_bottom_z(pos_w: np.ndarray, quat_wxyz: np.ndarray, extents_xyz: np.ndarray) -> float:
    rot = _quat_wxyz_to_matrix(quat_wxyz)
    support = float(np.abs(rot[2]) @ (0.5 * np.asarray(extents_xyz, dtype=np.float64)))
    return float(np.asarray(pos_w, dtype=np.float64)[2] - support)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--motion-root",
        default=str(REPO_ROOT / "data/ds_box_data/train_g1_w_obj_prepared_plus_omomo_orig"),
        help="Motion bank directory to verify.",
    )
    parser.add_argument("--map", dest="map_path", default=None, help="Optional clip-object map override.")
    parser.add_argument("--ground-prefix", default="box_", help="Clip prefix requiring first-frame grounding.")
    parser.add_argument("--size-tol", type=float, default=1.0e-4)
    parser.add_argument("--ground-tol", type=float, default=1.0e-3)
    parser.add_argument("--max-samples", type=int, default=12)
    args = parser.parse_args()

    motion_root = Path(args.motion_root).resolve()
    clips = _load_clip_map(motion_root, Path(args.map_path).resolve() if args.map_path else None)

    checked = 0
    size_bad: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    ground_bad: list[tuple[str, float]] = []
    ground_values: list[float] = []
    missing_bad: list[str] = []
    extents_bad: list[str] = []

    for clip_name in sorted(clips):
        npz_path = motion_root / f"{clip_name}.npz"
        if not npz_path.is_file():
            missing_bad.append(clip_name)
            continue
        entry = clips[clip_name]
        urdf_raw = entry.get("object_urdf_path", "") if isinstance(entry, dict) else str(entry)
        try:
            urdf_path = _resolve_urdf(urdf_raw, motion_root)
            extents = load_urdf_geometry_extents(str(urdf_path))
            if extents is None:
                raise ValueError(f"no extents resolved from {urdf_path}")
            extents_arr = np.asarray(extents, dtype=np.float64).reshape(3)
        except Exception as exc:
            extents_bad.append(f"{clip_name}: {exc}")
            continue

        with np.load(npz_path, allow_pickle=True) as data:
            if "object_pos_w" not in data or "object_quat_w" not in data:
                missing_bad.append(f"{clip_name}: missing object pose fields")
                continue
            size = _extract_object_size(data, clip_name)
            size_err = float(np.max(np.abs(size - extents_arr)))
            if size_err > args.size_tol:
                size_bad.append((clip_name, size, extents_arr, size_err))
            if args.ground_prefix and clip_name.startswith(args.ground_prefix):
                bottom = _box_bottom_z(data["object_pos_w"][0], data["object_quat_w"][0], extents_arr)
                ground_values.append(float(bottom))
                if abs(bottom) > args.ground_tol:
                    ground_bad.append((clip_name, bottom))
        checked += 1

    print(f"motion_root={motion_root}")
    print(f"checked_clips={checked}")
    print(f"missing_or_invalid_clips={len(missing_bad)}")
    print(f"bad_extents={len(extents_bad)}")
    print(f"object_size_mismatch={len(size_bad)}")
    print(f"{args.ground_prefix}first_frame_grounding_bad={len(ground_bad)}")
    if ground_values:
        print(
            f"{args.ground_prefix}first_frame_bottom_range="
            f"[{min(ground_values):+.8f}, {max(ground_values):+.8f}]"
        )

    if missing_bad:
        print("missing_or_invalid_samples:")
        for item in missing_bad[: args.max_samples]:
            print(f"  {item}")
    if extents_bad:
        print("bad_extents_samples:")
        for item in extents_bad[: args.max_samples]:
            print(f"  {item}")
    if size_bad:
        print("object_size_mismatch_samples:")
        for clip_name, size, extents, err in size_bad[: args.max_samples]:
            print(f"  {clip_name}: size={size.tolist()} urdf_extents={extents.tolist()} max_err={err:.6g}")
    if ground_bad:
        print("grounding_bad_samples:")
        for clip_name, bottom in ground_bad[: args.max_samples]:
            print(f"  {clip_name}: bottom_z={bottom:+.8f}")

    return 1 if (missing_bad or extents_bad or size_bad or ground_bad) else 0


if __name__ == "__main__":
    raise SystemExit(main())
