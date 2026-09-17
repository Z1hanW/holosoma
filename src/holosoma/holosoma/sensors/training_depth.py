"""Checkpoint-bound, deterministic real-depth processing for box23K PPO.

Training noise is domain randomization, not a real-camera postprocess. Native
RealSense misses are translated to far depth before the training operations.
"""

from __future__ import annotations

from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import cv2


def contract_digest(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_training_depth_profile(profile: dict) -> dict:
    """Reject unsupported profiles; never infer processing from camera pitch."""
    contract = profile["training_depth_contract"]
    if contract_digest(contract) != profile["training_depth_contract_sha256"]:
        raise ValueError("Depth contract digest mismatch")
    geometry = contract["camera_geometry"]
    schema = contract["effective_observation_schema"]
    expected = {
        "version": 2, "output_mode": "camera_depth", "camera_source": "far_tracking_warp",
        "camera_shape": [60, 106], "camera_obs_shape": [58, 87],
        "camera_warp_preprocess": True, "camera_warp_freq_ratio": 1,
        "camera_warp_buffer_len": 6, "camera_warp_latency_frame_range": [3, 4],
    }
    for key, value in expected.items():
        if contract[key] != value:
            raise ValueError(f"Unsupported training depth {key}: {contract[key]!r}")
    for key, value in {"near": 0.3, "far": 3.0, "max_distance": 3.0, "fps": 30.0,
                       "body_name": "torso_link", "pitch_deg": 10.0,
                       "hfov_deg": 89.5, "vfov_deg": 58.6,
                       "distortion": [0.0] * 5, "target_pitch_deg": None,
                       "frame_quaternion": [-0.5, 0.5, -0.5, 0.5],
                       "use_mount_quaternion": True, "use_frame_quaternion": True,
                       "auto_fix_backward": False}.items():
        if geometry[key] != value:
            raise ValueError(f"Unsupported camera geometry {key}")
    for key, value in {"crop": [2, 0, 4, 4], "resize": [58, 87], "normalize": True,
                       "min_valid_depth": 0.15}.items():
        if schema[key] != value:
            raise ValueError(f"Unsupported depth preprocessing {key}")
    if not np.allclose(schema["sensor_offset"], [0.01, 0.01, 0.44], atol=1e-7, rtol=0):
        raise ValueError("Expected historical torso-relative camera position")
    if not np.allclose(geometry["mount_quaternion"],
                       [0.00644801, 0.23350163, 0.00644801, 0.97231365], atol=1e-7, rtol=0):
        raise ValueError("Expected historical 27 degree mount plus 10 degree pitch")
    digest = profile["checkpoint"]["sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or len(bytes.fromhex(digest)) != 32:
        raise ValueError("Invalid checkpoint SHA256")
    return contract


def load_training_depth_profile(path: str | Path, model_path: str | Path | None = None) -> dict:
    profile = json.loads(Path(path).read_text())
    validate_training_depth_profile(profile)
    if model_path is not None:
        if hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != profile["checkpoint"]["sha256"]:
            raise ValueError("Policy and depth server must use the same exact ONNX")
    return profile


def preprocess_real_depth(frame: np.ndarray, profile: dict) -> np.ndarray:
    """Native 848x480 -> virtual 106x60 -> training crop/resize/normalize.

The 106x60 input is also accepted for exact simulator parity tests. Full-frame
resampling assumes matching optical FOV; it does not calibrate a physical mount.
"""
    contract = profile["training_depth_contract"]
    frame = np.asarray(frame, dtype=np.float32)
    if frame.shape not in ((480, 848), (60, 106)):
        raise ValueError(f"Unsupported raw depth shape: {frame.shape}")
    near, far = contract["camera_geometry"]["near"], contract["camera_geometry"]["far"]
    frame = np.where((frame == 0) | ~np.isfinite(frame) | (frame > far), far, frame)
    depth = np.clip(frame, near, far)
    if frame.shape != (60, 106):
        depth = cv2.resize(depth, (106, 60), interpolation=cv2.INTER_CUBIC)
        depth = np.clip(depth, near, far)
    depth = depth[2:, 4:-4]
    # OpenCV uses the same half-pixel coordinates and cubic coefficient (-0.75)
    # as torch bicubic/align_corners=False; tests bound CPU float rounding error.
    depth = cv2.resize(depth, (87, 58), interpolation=cv2.INTER_CUBIC)
    depth = np.clip(depth, near, far)
    depth = np.where(depth < 0.15, far, depth)
    result = ((depth - near) / (far - near) - 0.5)[None]
    if not np.isfinite(result).all():
        raise ValueError("Non-finite processed depth")
    return result


class TimestampedDepthBuffer:
    """Select at-or-before capture time, including measured hardware latency."""

    def __init__(self, profile: dict):
        contract = validate_training_depth_profile(profile)
        self.frames = deque(maxlen=contract["camera_warp_buffer_len"])
        self.fps = contract["camera_geometry"]["fps"]
        self.delays = tuple(contract["camera_warp_latency_frame_range"])

    def append(self, image: np.ndarray, capture_time: float, now: float) -> None:
        if not math.isfinite(capture_time) or not math.isfinite(now) or not 0 <= now - capture_time <= 0.2:
            raise ValueError("Missing/invalid/stale RealSense capture timestamp")
        if self.frames and capture_time <= self.frames[-1][0]:
            raise ValueError("RealSense capture timestamps must increase")
        self.frames.append((capture_time, image.copy()))

    def select(self, now: float, delay_frames: int) -> tuple[float, np.ndarray] | None:
        if delay_frames not in self.delays:
            raise ValueError("Unsupported training depth latency")
        deadline = now - delay_frames / self.fps
        for captured_at, image in reversed(self.frames):
            if captured_at <= deadline:
                if now - captured_at > 0.2:
                    raise RuntimeError("Depth capture is too old")
                return captured_at, image
        if len(self.frames) == self.frames.maxlen:
            raise RuntimeError("Depth history cannot satisfy training latency")
        return None  # Warm-up: no synthetic frame is published.


def write_depth_status(path: str, status: dict) -> None:
    target = Path(path)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(status, allow_nan=False))
    os.replace(temporary, target)


def read_bound_depth(array: np.ndarray, status_path: str, profile: dict, shm_name: str) -> np.ndarray:
    """Read only a complete, fresh publication from this checkpoint's producer."""
    path = Path(status_path)
    for _ in range(5):
        before = json.loads(path.read_text())
        if before["sequence"] % 2:
            time.sleep(0.001)
            continue
        image = array.copy()
        after = json.loads(path.read_text())
        if before != after:
            time.sleep(0.001)
            continue
        if (before["checkpoint_sha256"] != profile["checkpoint"]["sha256"]
                or before["depth_contract_sha256"] != profile["training_depth_contract_sha256"]
                or before["shm_name"] != shm_name or before["shape"] != list(array.shape)):
            raise RuntimeError("Depth publication does not match the policy contract")
        now = time.monotonic()
        if not 0 <= now - before["published_at_monotonic"] <= 0.1:
            raise RuntimeError("Depth producer has stopped or is stale")
        if not 0 <= now - before["captured_at_monotonic"] <= 0.25:
            raise RuntimeError("Policy depth capture is stale")
        if not np.isfinite(image).all() or image.min() < -0.50001 or image.max() > 0.50001:
            raise RuntimeError("Invalid normalized depth publication")
        return image
    raise RuntimeError("Depth publication changed during read")
