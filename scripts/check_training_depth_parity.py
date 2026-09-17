#!/usr/bin/env python3
"""Compare real preprocessing against exact training source and ONNX on CPU."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from checkpoint_camera_profile import extract_profile, _atomic_write
from holosoma.sensors.training_depth import preprocess_real_depth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = extract_profile(args.model_path, training_depth=True)
    source = args.training_source / "src/holosoma/holosoma/managers/perception/manager.py"
    # Execute only the actual deterministic methods, without importing simulator,
    # CUDA, or the training environment. No duplicated preprocessing reference.
    names = {"_clamp_camera_depth_to_sensor_range", "_process_camera_depth_for_obs",
             "_crop_camera_depth", "_normalize_camera_depth_for_obs"}
    tree = ast.parse(source.read_text())
    methods = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name in names]
    if {method.name for method in methods} != names:
        raise ValueError("Training source methods are missing")
    module = ast.Module(body=[ast.ClassDef(name="Reference", bases=[], keywords=[],
                                         body=methods, decorator_list=[])], type_ignores=[])
    namespace = {"torch": torch, "F": F}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    reference = namespace["Reference"]()
    reference.cfg = SimpleNamespace(camera_near=0.3, max_distance=3.0)
    attributes = {"_camera_warp_preprocess": True, "_camera_obs_height": 58, "_camera_obs_width": 87,
                  "_camera_warp_crop_top": 2, "_camera_warp_crop_bottom": 0,
                  "_camera_warp_crop_left": 4, "_camera_warp_crop_right": 4,
                  "_camera_warp_min_valid_depth": 0.15, "_camera_warp_normalize": True,
                  "_camera_warp_edge_noise": False, "_camera_warp_enable_holes": False,
                  "_camera_warp_additive_noise_std": 0, "_camera_warp_depth_offset_std": 0}
    for key, value in attributes.items():
        setattr(reference, key, value)
    rng = np.random.default_rng(20260917)
    frames = rng.uniform(-0.1, 4.0, (64, 60, 106)).astype(np.float32)
    frames[0] = 0.0
    frames[1] = 0.3
    frames[2] = 3.0
    frames[3, :, :53], frames[3, :, 53:] = 0.3, 3.0
    frames[4, ::2] = np.nan
    frames[5, ::2] = np.inf
    sensor_misses_to_far = np.where(frames == 0, 3, frames)
    with torch.inference_mode():
        expected = reference._process_camera_depth_for_obs(
            reference._clamp_camera_depth_to_sensor_range(torch.from_numpy(sensor_misses_to_far))).numpy()
    actual = np.stack([preprocess_real_depth(frame, profile)[0] for frame in frames])
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=0)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(args.model_path), options, providers=["CPUExecutionProvider"])
    actor = rng.normal(0, 0.1, (len(frames), 94)).astype(np.float32)
    expected_action = session.run(None, {"actor_obs": actor, "perception_obs": expected.reshape(-1, 5046)})[0]
    actual_action = session.run(None, {"actor_obs": actor, "perception_obs": actual.reshape(-1, 5046)})[0]
    if not np.isfinite(actual_action).all() or not np.isfinite(expected_action).all():
        raise ValueError("Non-finite ONNX actions")
    np.testing.assert_allclose(actual_action, expected_action, atol=1e-4, rtol=1e-4)
    result = {
        "accepted": True, "checkpoint": profile["checkpoint"], "probe_frames": len(frames),
        "training_commit": subprocess.check_output(
            ["git", "-C", str(args.training_source), "rev-parse", "HEAD"], text=True).strip(),
        "training_perception_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "normalized_depth_max_abs": float(np.max(np.abs(actual - expected))),
        "onnx_action_max_abs": float(np.max(np.abs(actual_action - expected_action))),
        "scope": "deterministic_pixel_processing_and_action_sensitivity_not_real_sensor_calibration",
        "training_noise_disabled_for_parity_only": True,
        "physical_camera_opened": False,
    }
    _atomic_write(args.output, json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
