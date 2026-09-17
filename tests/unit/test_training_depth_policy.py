import json
from types import SimpleNamespace
import time

import numpy as np
import pytest

from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy


def test_policy_consumes_bound_depth_and_rejects_wrong_producer(tmp_path, monkeypatch):
    policy = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
    policy.obs_dims = {"cam_depth": 5046}
    props = SimpleNamespace(resized_height=58, resized_width=87)
    policy.config = SimpleNamespace(camera=SimpleNamespace(poses=[object()], props=props))
    policy._depth_img_array = np.full((1, 1, 58, 87), 0.25, np.float32)
    policy._training_depth_profile = {"checkpoint": {"sha256": "a" * 64},
                                      "training_depth_contract_sha256": "b" * 64}
    status = {"sequence": 2, "checkpoint_sha256": "a" * 64, "depth_contract_sha256": "b" * 64,
              "shm_name": "unique", "shape": [1, 1, 58, 87],
              "published_at_monotonic": time.monotonic(),
              "captured_at_monotonic": time.monotonic() - 0.1}
    path = tmp_path / "status.json"
    path.write_text(json.dumps(status))
    monkeypatch.setenv("HOLOSOMA_DEPTH_STATUS_PATH", str(path))
    monkeypatch.setenv("HOLOSOMA_DEPTH_SHM_NAME", "unique")
    np.testing.assert_array_equal(policy._get_depth_image_obs(), np.full((1, 5046), 0.25))
    status["checkpoint_sha256"] = "c" * 64
    path.write_text(json.dumps(status))
    with pytest.raises(RuntimeError, match="does not match"):
        policy._get_depth_image_obs()


def test_model_hash_rejected_before_ort_or_robot_model(tmp_path, monkeypatch):
    from holosoma.sensors import training_depth

    policy = WholeBodyTrackingPolicy.__new__(WholeBodyTrackingPolicy)
    monkeypatch.setenv("HOLOSOMA_TRAINING_DEPTH_PROFILE", "bound_profile.json")

    def reject(path, model_path):
        assert path == "bound_profile.json"
        raise ValueError("wrong model")

    monkeypatch.setattr(training_depth, "load_training_depth_profile", reject)
    with pytest.raises(ValueError, match="wrong model"):
        policy.setup_policy(str(tmp_path / "wrong.onnx"))
