from __future__ import annotations

import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from holosoma.sensors.training_depth import (
    TimestampedDepthBuffer, contract_digest, load_training_depth_profile,
    preprocess_real_depth, read_bound_depth, validate_training_depth_profile, write_depth_status,
)


@pytest.fixture
def profile():
    contract = {
        "version": 2, "output_mode": "camera_depth", "camera_source": "far_tracking_warp",
        "camera_shape": [60, 106], "camera_obs_shape": [58, 87],
        "camera_warp_preprocess": True, "camera_warp_freq_ratio": 1,
        "camera_warp_buffer_len": 6, "camera_warp_latency_frame_range": [3, 4],
        "camera_geometry": {"near": 0.3, "far": 3.0, "max_distance": 3.0,
                            "fps": 30.0, "body_name": "torso_link", "pitch_deg": 10.0,
                            "hfov_deg": 89.5, "vfov_deg": 58.6, "distortion": [0.0] * 5,
                            "target_pitch_deg": None, "frame_quaternion": [-0.5, 0.5, -0.5, 0.5],
                            "use_mount_quaternion": True, "use_frame_quaternion": True,
                            "auto_fix_backward": False,
                            "mount_quaternion": [0.00644801, 0.23350163, 0.00644801, 0.97231365]},
        "effective_observation_schema": {"crop": [2, 0, 4, 4], "resize": [58, 87],
                                         "normalize": True, "min_valid_depth": 0.15,
                                         "sensor_offset": [0.01, 0.01, 0.44]},
    }
    return {"training_depth_contract": contract,
            "training_depth_contract_sha256": contract_digest(contract),
            "checkpoint": {"sha256": "a" * 64}}


@pytest.mark.parametrize("pattern", ["random", "edges", "constant", "invalid"])
def test_deterministic_training_tensor_parity(profile, pattern):
    frame = np.random.default_rng(42).uniform(0.1, 4, (60, 106)).astype(np.float32)
    if pattern == "edges":
        frame[:] = 0.3
        frame[:, 47:] = 3.0
    elif pattern == "constant":
        frame[:] = 1.25
    elif pattern == "invalid":
        frame.flat[::7] = 0
        frame.flat[::11] = np.nan
        frame.flat[::13] = np.inf
    clean = np.where((frame == 0) | ~np.isfinite(frame) | (frame > 3), 3, frame)
    depth = torch.from_numpy(clean).clamp(0.3, 3)[None, None, 2:, 4:-4]
    expected = (F.interpolate(depth, (58, 87), mode="bicubic", align_corners=False)
                .clamp(0.3, 3) - 0.3) / 2.7 - 0.5
    actual = preprocess_real_depth(frame, profile)
    np.testing.assert_allclose(actual, expected[0].numpy(), atol=2e-5, rtol=0)
    # Integer 8x enlargement preserves sampling centers in the virtual raster.
    sensor = np.repeat(np.repeat(frame, 8, axis=0), 8, axis=1)
    np.testing.assert_allclose(preprocess_real_depth(sensor, profile), actual, atol=1e-6, rtol=0)


def test_opencv_path_really_uses_cubic():
    from holosoma.config_types.image_server import ImageServerConfig
    from holosoma.sensors.image_server import ImageServer

    server = ImageServer.__new__(ImageServer)
    server.cfg = ImageServerConfig(near_clip=0.3, far_clip=3, resized_height=11, resized_width=17)
    frame = np.random.default_rng(3).uniform(0.3, 3, (23, 41)).astype(np.float32)
    expected = cv2.resize(frame, (17, 11), interpolation=cv2.INTER_CUBIC).clip(0.3, 3)
    expected = (expected - 0.3) / 2.7 - 0.5
    np.testing.assert_array_equal(server._resize_clip_expand_transpose(frame)[0], expected)


@pytest.mark.parametrize("field,value", [("camera_obs_shape", [58, 88]),
                                         ("camera_warp_latency_frame_range", [2, 3]),
                                         ("camera_source", "rendered")])
def test_unsupported_contract_rejected_even_with_correct_hash(profile, field, value):
    profile["training_depth_contract"][field] = value
    profile["training_depth_contract_sha256"] = contract_digest(profile["training_depth_contract"])
    with pytest.raises(ValueError, match="Unsupported"):
        validate_training_depth_profile(profile)


def test_corrupt_contract_and_wrong_model_rejected(profile, tmp_path):
    changed = copy.deepcopy(profile)
    changed["training_depth_contract"]["version"] = 7
    with pytest.raises(ValueError, match="digest"):
        validate_training_depth_profile(changed)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    model = tmp_path / "model.onnx"
    model.write_bytes(b"not-the-bound-model")
    with pytest.raises(ValueError, match="same exact ONNX"):
        load_training_depth_profile(path, model)


def test_timestamp_latency_includes_hardware_delay(profile):
    buffer = TimestampedDepthBuffer(profile)
    image = np.zeros((1, 1, 58, 87), np.float32)
    assert buffer.select(10, 3) is None
    for i in range(6):
        now = 10 + i / 30
        buffer.append(image + i, now - 0.039, now)
    for delay in (3, 4):
        captured_at, result = buffer.select(now, delay)
        age = now - captured_at
        assert delay / 30 <= age < (delay + 1) / 30
        assert np.isfinite(result).all()
    with pytest.raises(ValueError, match="increase"):
        buffer.append(image, now - 0.039, now)
    with pytest.raises(RuntimeError, match="old"):
        buffer.select(now + 1, 3)
    with pytest.raises(ValueError):
        buffer.append(image, float("nan"), now)


def test_bound_publication_rejects_stale_wrong_and_partial_frames(profile, tmp_path):
    path = str(tmp_path / "status.json")
    image = np.zeros((1, 1, 58, 87), np.float32)
    now = time.monotonic()
    status = {"sequence": 2, "checkpoint_sha256": "a" * 64,
              "depth_contract_sha256": profile["training_depth_contract_sha256"],
              "shm_name": "test_depth", "shape": list(image.shape),
              "published_at_monotonic": now, "captured_at_monotonic": now - 0.1}
    write_depth_status(path, status)
    np.testing.assert_array_equal(read_bound_depth(image, path, profile, "test_depth"), image)
    for key, value in [("sequence", 3), ("checkpoint_sha256", "b" * 64),
                       ("captured_at_monotonic", now - 10),
                       ("published_at_monotonic", now - 10), ("shm_name", "other")]:
        write_depth_status(path, {**status, key: value})
        with pytest.raises(RuntimeError):
            read_bound_depth(image, path, profile, "test_depth")
    write_depth_status(path, status)
    image.flat[0] = np.nan
    with pytest.raises(RuntimeError, match="Invalid"):
        read_bound_depth(image, path, profile, "test_depth")


def test_launcher_requires_explicit_checkpoint_and_does_not_stop_services():
    text = (Path(__file__).resolve().parents[2] / "real_training.sh").read_text()
    assert "--preflight-only" in text
    assert "systemctl" not in text
    assert "n94vaeq7" not in text
    assert "pkill" not in text
    assert 'HOLOSOMA_TRAINING_DEPTH_PROFILE="$log_dir/camera_profile.json"' in text


def test_global_timestamp_age_does_not_double_count_capture_wait(monkeypatch):
    from types import SimpleNamespace
    from holosoma.sensors.realsense import RealSenseCamera, RealSenseCameraConfig

    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera.config = RealSenseCameraConfig(require_global_time=True)
    camera.rs = SimpleNamespace(timestamp_domain=SimpleNamespace(global_time=7))
    monkeypatch.setattr(time, "time", lambda: 1000.0)
    # A 30 ms blocking wait must not turn a 40 ms-old image into a 70 ms-old one.
    assert camera._compute_latency(999960.0, 7, 999995.0, 1.0, 1.035) == 40.0
    assert camera._compute_latency(999960.0, 8, 999995.0, 1.0, 1.035) is None


def test_image_server_full_publication_without_hardware(profile, tmp_path, monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace
    import uuid
    from holosoma.config_values.image_server import real_d435i
    from holosoma.sensors import image_server

    clock = [100.0]
    monkeypatch.setattr(image_server, "time", SimpleNamespace(
        monotonic=lambda: clock[0], perf_counter=time.perf_counter))

    class Limiter:
        def __init__(self, frequency):
            self.frequency = frequency

        def sleep(self):
            clock[0] += 1 / self.frequency

        def get_stats(self):
            return {}

        def reset(self):
            pass

    class Camera:
        num_cameras = 1
        calls = 0

        def get_frames(self):
            self.calls += 1
            if self.calls > 8:
                raise StopIteration
            return {"depth": {"d435": np.full((480, 848), 1.2, np.float32)},
                    "total_latency_ms": 39.0}

    monkeypatch.setattr(image_server, "RateLimiter", Limiter)
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    status_path = tmp_path / "status.json"
    config = replace(real_d435i, save_images=False, training_depth_profile=str(path),
                     shared_memory_name="test_depth_" + uuid.uuid4().hex,
                     depth_status_path=str(status_path))
    server = image_server.ImageServer(Camera(), config)
    try:
        with pytest.raises(StopIteration):
            server.send_process()
        status = json.loads(status_path.read_text())
        assert status["sequence"] > 0 and status["sequence"] % 2 == 0
        assert status["checkpoint_sha256"] == profile["checkpoint"]["sha256"]
        assert 0.1 <= status["published_at_monotonic"] - status["captured_at_monotonic"] < 0.17
        np.testing.assert_allclose(server.img_array, (1.2 - 0.3) / 2.7 - 0.5, atol=1e-6)
    finally:
        server.image_shm.close()
        server.image_shm.unlink()
