#!/usr/bin/env python3
"""Standalone Intel RealSense D435i depth viewer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show live D435i depth frames.")
    parser.add_argument("--serial", default="", help="RealSense serial number. Empty uses the first device.")
    parser.add_argument("--width", type=int, default=848, help="Depth stream width.")
    parser.add_argument("--height", type=int, default=480, help="Depth stream height.")
    parser.add_argument("--fps", type=int, default=30, help="Depth stream FPS.")
    parser.add_argument("--near", type=float, default=0.3, help="Depth visualization near range in meters.")
    parser.add_argument("--far", type=float, default=3.0, help="Depth visualization far range in meters.")
    parser.add_argument(
        "--emitter",
        choices=("on", "off", "unchanged"),
        default="on",
        help="IR emitter setting. Use 'unchanged' to leave device state untouched.",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Optional directory for saving raw .npy and visualized .png frames with the 's' key.",
    )
    return parser.parse_args()


def _depth_to_color(depth_m: np.ndarray, near: float, far: float) -> np.ndarray:
    clipped = np.clip(depth_m, near, far)
    normalized = ((clipped - near) / (far - near) * 255.0).astype(np.uint8)
    invalid = depth_m <= 0.0
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[invalid] = (0, 0, 0)
    return colored


def _stats_text(depth_m: np.ndarray) -> tuple[str, str]:
    valid = np.isfinite(depth_m) & (depth_m > 0.0) & (depth_m < 65.0)
    valid_frac = float(valid.mean())
    if not valid.any():
        return "valid=0.0%", "no valid depth"

    valid_depth = depth_m[valid]
    q50, q90, q99 = np.quantile(valid_depth, [0.5, 0.9, 0.99])
    line1 = f"valid={valid_frac * 100:.1f}% min={valid_depth.min():.3f}m max={valid_depth.max():.3f}m"
    line2 = f"median={q50:.3f}m p90={q90:.3f}m p99={q99:.3f}m"
    return line1, line2


def _put_status(image: np.ndarray, lines: tuple[str, str], emitter: str) -> None:
    text_lines = (
        "D435i depth | q/esc quit | s save",
        f"emitter={emitter}",
        *lines,
    )
    y = 26
    for line in text_lines:
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


def main() -> int:
    args = _parse_args()

    try:
        import pyrealsense2 as rs
    except ImportError:
        print("pyrealsense2 is not installed in this Python environment.", file=sys.stderr)
        return 1

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    if args.emitter != "unchanged" and depth_sensor.supports(rs.option.emitter_enabled):
        depth_sensor.set_option(rs.option.emitter_enabled, 1.0 if args.emitter == "on" else 0.0)

    print(
        f"D435i viewer started: {args.width}x{args.height}@{args.fps}, "
        f"depth_scale={depth_scale:.6f}, emitter={args.emitter}"
    )

    frame_idx = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue

            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            view = _depth_to_color(depth_m, args.near, args.far)
            _put_status(view, _stats_text(depth_m), args.emitter)

            cv2.imshow("D435i Depth", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s") and save_dir is not None:
                stamp = time.strftime("%Y%m%d_%H%M%S")
                raw_path = save_dir / f"d435i_depth_{frame_idx:06d}_{stamp}.npy"
                png_path = save_dir / f"d435i_depth_{frame_idx:06d}_{stamp}.png"
                np.save(raw_path, depth_m)
                cv2.imwrite(str(png_path), view)
                print(f"saved {raw_path} and {png_path}")
            frame_idx += 1
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
