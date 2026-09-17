#!/usr/bin/env python3
"""CPU-only validation for an explicitly selected current-training ONNX."""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from checkpoint_camera_profile import _atomic_write, _resolve_model_path, extract_profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    model = _resolve_model_path(args.model_path, args.output_dir / "checkpoint")
    profile = extract_profile(model, training_depth=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    feed = {"actor_obs": np.zeros((1, 94), dtype=np.float32),
            "perception_obs": np.full((1, 5046), 0.5, dtype=np.float32)}
    outputs = session.run(None, feed)
    if len(outputs) != 1 or outputs[0].shape != (1, 29) or not np.isfinite(outputs[0]).all():
        raise ValueError("Expected a finite 29-action ONNX output")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(args.output_dir / "camera_profile.json", json.dumps(profile, indent=2) + "\n")
    _atomic_write(args.output_dir / "resolved_model_path.txt", str(model) + "\n")
    print(json.dumps({"accepted": True, "checkpoint": profile["checkpoint"],
                      "camera": profile["camera"],
                      "command_mode": profile["deployment_command_mode"],
                      "hardware_opened": False}, indent=2))


if __name__ == "__main__":
    main()
