#!/usr/bin/env python3
"""Publish RealSense depth as model-ready perception_obs for sim2real."""

from __future__ import annotations

import argparse
import time
from multiprocessing import resource_tracker
from multiprocessing import shared_memory

import cv2
import numpy as np


class PerceptionObsShmPublisher:
    """Write the latest flattened perception_obs tensor into POSIX shared memory."""

    def __init__(self, name: str, dim: int, *, fill_value: float = 0.5) -> None:
        self.name = str(name)
        self.dim = int(dim)
        self.size = self.dim * np.dtype(np.float32).itemsize
        self.shm: shared_memory.SharedMemory | None = None
        self.array: np.ndarray | None = None
        self._created = False
        self._attach_or_create()
        if self.array is not None:
            self.array[:] = np.float32(fill_value)

    def _attach_or_create(self) -> None:
        try:
            self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
            self._created = True
            print(f"[INFO] created perception_obs shared memory: name={self.name} values={self.dim}")
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=self.name, create=False)
            if len(existing.buf) != self.size:
                existing.close()
                stale = shared_memory.SharedMemory(name=self.name, create=False)
                stale.unlink()
                stale.close()
                self.shm = shared_memory.SharedMemory(name=self.name, create=True, size=self.size)
                self._created = True
                print(f"[INFO] recreated perception_obs shared memory: name={self.name} values={self.dim}")
            else:
                self.shm = existing
                try:
                    resource_tracker.unregister(self.shm._name, "shared_memory")
                except Exception:
                    pass
                print(f"[INFO] connected to perception_obs shared memory: name={self.name} values={self.dim}")

        self.array = np.ndarray((self.dim,), dtype=np.float32, buffer=self.shm.buf)

    def publish(self, obs_img: np.ndarray) -> None:
        if self.array is None:
            return
        self.array[:] = np.asarray(obs_img, dtype=np.float32).reshape(-1)

    def close(self) -> None:
        shm = self.shm
        self.shm = None
        self.array = None
        if shm is None:
            return
        if self._created:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
        shm.close()


def _process_depth(
    depth_m: np.ndarray,
    *,
    camera_height: int,
    camera_width: int,
    resize_height: int,
    resize_width: int,
    crop_top: int,
    crop_bottom: int,
    crop_left: int,
    crop_right: int,
    near: float,
    far: float,
    min_valid_depth: float,
) -> np.ndarray:
    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, far)

    # Match training camera preprocessing: camera frame -> crop -> bicubic resize -> normalize.
    depth = cv2.resize(depth, (int(camera_width), int(camera_height)), interpolation=cv2.INTER_AREA)
    h, w = depth.shape
    top = min(max(int(crop_top), 0), max(h - 1, 0))
    bottom = min(max(int(crop_bottom), 0), max(h - top - 1, 0))
    left = min(max(int(crop_left), 0), max(w - 1, 0))
    right = min(max(int(crop_right), 0), max(w - left - 1, 0))
    depth = depth[top : max(top + 1, h - bottom), left : max(left + 1, w - right)]
    depth = cv2.resize(depth, (int(resize_width), int(resize_height)), interpolation=cv2.INTER_CUBIC)
    depth = np.clip(depth, near, far)
    if min_valid_depth > 0.0:
        depth = np.where(depth < min_valid_depth, far, depth)
    depth = (depth - near) / max(1.0e-6, far - near) - 0.5
    return np.clip(depth, -0.5, 0.5).astype(np.float32, copy=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm-name", default="depth_img_shm")
    parser.add_argument("--serial", default="")
    parser.add_argument("--stream-width", type=int, default=640)
    parser.add_argument("--stream-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-width", type=int, default=106)
    parser.add_argument("--camera-height", type=int, default=60)
    parser.add_argument("--resize-width", type=int, default=87)
    parser.add_argument("--resize-height", type=int, default=58)
    parser.add_argument("--crop-top", type=int, default=2)
    parser.add_argument("--crop-bottom", type=int, default=0)
    parser.add_argument("--crop-left", type=int, default=4)
    parser.add_argument("--crop-right", type=int, default=4)
    parser.add_argument("--near", type=float, default=0.3)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument("--min-valid-depth", type=float, default=0.15)
    parser.add_argument("--shm-fill-value", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    try:
        import pyrealsense2 as rs
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "pyrealsense2 is not installed in this Python environment. "
            "Install it in the env used for sim2real, then rerun this publisher."
        ) from exc

    expected_dim = int(args.resize_height) * int(args.resize_width)
    publisher = PerceptionObsShmPublisher(
        args.shm_name,
        expected_dim,
        fill_value=float(args.shm_fill_value),
    )

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.depth, args.stream_width, args.stream_height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())

    print(
        f"[INFO] publishing RealSense perception_obs to shared memory '{args.shm_name}'; "
        f"raw={args.stream_width}x{args.stream_height}@{args.fps}, "
        f"obs={args.resize_height}x{args.resize_width}, depth_scale={depth_scale}"
    )
    time.sleep(0.5)

    frame_idx = 0
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                continue
            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale
            obs_img = _process_depth(
                depth_m,
                camera_height=args.camera_height,
                camera_width=args.camera_width,
                resize_height=args.resize_height,
                resize_width=args.resize_width,
                crop_top=args.crop_top,
                crop_bottom=args.crop_bottom,
                crop_left=args.crop_left,
                crop_right=args.crop_right,
                near=args.near,
                far=args.far,
                min_valid_depth=args.min_valid_depth,
            )
            publisher.publish(obs_img)

            if args.preview:
                preview = ((obs_img + 0.5) * 255.0).clip(0, 255).astype(np.uint8)
                cv2.imshow("perception_obs", cv2.resize(preview, (348, 232), interpolation=cv2.INTER_NEAREST))
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            if args.log_every > 0 and frame_idx % args.log_every == 0:
                print(
                    "[INFO]",
                    f"frame={frame_idx}",
                    f"min={float(obs_img.min()):.4f}",
                    f"max={float(obs_img.max()):.4f}",
                    f"mean={float(obs_img.mean()):.4f}",
                    f"std={float(obs_img.std()):.4f}",
                )
            frame_idx += 1
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        publisher.close()
        if args.preview:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
