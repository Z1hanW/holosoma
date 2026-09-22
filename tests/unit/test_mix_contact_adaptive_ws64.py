from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import contact_sampling_factorial as recipe
import mix_contact_adaptive_ws64 as launch
from prepare_mix216_contact_labels import interval_json


def test_legacy_interval_conversion_is_exact_and_never_fabricates(tmp_path):
    np.save(tmp_path / "left_wrist_contact_interval_steps.npy", np.array([-1, -1]))
    np.save(tmp_path / "right_wrist_contact_interval_steps.npy", np.array([25, 200]))
    assert interval_json(tmp_path) == {"left_wrist": [-1, -1], "right_wrist": [25, 200]}
    np.save(tmp_path / "left_wrist_contact_interval_steps.npy", np.array([5, 4]))
    with pytest.raises(ValueError, match="bounds"):
        interval_json(tmp_path)
    (tmp_path / "left_wrist_contact_interval_steps.npy").unlink()
    with pytest.raises(FileNotFoundError):
        interval_json(tmp_path)


def _options(args):
    pairs = [a[2:].split("=", 1) for a in args if a.startswith("--")]
    assert len(pairs) == len({k for k, _ in pairs})
    return dict(pairs)


def test_recipe_changes_only_dataset_topology_name_and_runtime_append():
    campaign = {"persist": "/audit", "dataset": {"bank": "/new216", "contact_root": "/new216/contact"}}
    before = _options(recipe.training_args(launch.ARM, "formal", "/audit", "abcd1234"))
    after = _options(launch.training_args(campaign, "formal", "abcd1234"))
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    assert changed == {
        "training.name", "logger.name", "training.num-envs",
        recipe.MOTION_PREFIX + "motion-file",
        recipe.MOTION_PREFIX + "adaptive-sampling-contact-interval-root",
        recipe.MOTION_PREFIX + "runtime-default-pose-append-duration-s",
        "reward.terms.offline-contact-guidance.params.contact-export-root",
        "robot.object.object-urdf-path",
    }
    assert after["training.num-envs"] == "131072"
    assert after["training.export-onnx"] == "True"
    assert after["algo.config.distill.ppo-start-coeff"] == "0.01"
    assert after["algo.config.save-interval"] == "500"
    assert after["algo.config.module-dict.actor.layer-config.hidden-dims"] == "[512,256,128]"


def test_canary_has_fresh_identity_and_original_initializer():
    campaign = {"persist": "/audit", "dataset": {"bank": "/new216", "contact_root": "/new216/contact"}}
    options = _options(launch.training_args(campaign, "canary"))
    assert "logger.id" not in options
    assert options["logger.mode"] == "offline"
    assert options["algo.config.num-learning-iterations"] == "2"
    assert options["training.policy-init-checkpoint"] == recipe.INIT
    assert options["algo.config.distill.policy-to-clone"] == recipe.TEACHER
