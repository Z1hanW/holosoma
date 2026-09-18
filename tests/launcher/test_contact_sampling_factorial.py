"""Factor isolation and launch identity for the eight independent experiments."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("factorial", ROOT / "scripts/contact_sampling_factorial.py")
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


def options(arm, mode="canary"):
    cli = EXP.training_args(arm, mode, "/canonical", "abcdefgh" if mode == "formal" else None)
    parsed = dict(arg[2:].split("=", 1) for arg in cli if arg.startswith("--"))
    assert len(parsed) == sum(arg.startswith("--") for arg in cli)
    return parsed


@pytest.mark.parametrize("arm", EXP.ARMS)
def test_exact_experiment_contract(arm):
    args = options(arm)
    mix, contact, adaptive = EXP.arm_flags(arm)
    assert args["training.num-envs"] == "16384"
    assert args["training.export-onnx"] == "True"
    assert args["algo.config.distill.enabled"] == str(mix)
    assert args["reward.terms.offline-contact-guidance.weight"] == str(float(contact))
    for key in ("contact-weight", "wrist-weight"):
        assert args["reward.terms.offline-contact-guidance.params." + key] == "1.0"
    assert args[EXP.MOTION_PREFIX + "use-adaptive-timesteps-sampler"] == str(adaptive)
    assert args[EXP.MOTION_PREFIX + "uniform-t1-window-sampling-enabled"] == "False"
    assert args[EXP.MOTION_PREFIX + "clip-weighting-strategy"] == "uniform_clip"
    assert args[EXP.MOTION_PREFIX + "contact-aware-button-window-mode"] == "peak_height"
    if mix:
        assert args["algo.config.distill.policy-to-clone"] == EXP.TEACHER
        assert args["algo.config.distill.ppo-start-coeff"] == "0.01"
        assert args["algo.config.distill.take-teacher-actions"] == "False"
    else:
        assert "algo.config.distill.policy-to-clone" not in args
    formal = options(arm, "formal")
    assert formal["algo.config.save-interval"] == "500"
    assert formal["algo.config.num-learning-iterations"] == "40000"
    assert formal["logger.resume"] == "never"
    assert formal["logger.mode"] == "online"


def test_factor_isolation():
    normalized = []
    for arm in EXP.ARMS:
        values = options(arm)
        for key in list(values):
            if key.startswith("algo.config.distill.") or key in {
                "training.name", "logger.name", "logger.base-dir",
                "reward.terms.offline-contact-guidance.weight",
                EXP.MOTION_PREFIX + "use-adaptive-timesteps-sampler",
            }:
                del values[key]
        normalized.append(values)
    assert all(value == normalized[0] for value in normalized)


def test_formal_requires_identity():
    with pytest.raises(ValueError, match="fresh run ID"):
        EXP.training_args(EXP.ARMS[0], "formal", "/canonical")
    with pytest.raises(ValueError, match="Unknown arm"):
        EXP.training_args("bad-arm", "canary", "/canonical")


@pytest.mark.parametrize("arm", EXP.ARMS)
def test_canary_and_formal_scientific_validation(arm):
    spec = importlib.util.spec_from_file_location("cli_validator", ROOT / "scripts/validate_train_cli.py")
    validator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validator)
    for mode in ("canary", "formal"):
        validator.parse_and_validate_train_cli(EXP.training_args(arm, mode, "/canonical", "abcdefgh" if mode == "formal" else None))
    short, full = options(arm), options(arm, "formal")
    prefix = EXP.MOTION_PREFIX + "start-at-timestep-zero-prob-"
    for iteration in (0, 1):
        short_p = float(short[prefix + "end"]) * iteration / int(short[prefix + "end-iter"])
        full_p = float(full[prefix + "end"]) * iteration / int(full[prefix + "end-iter"])
        assert short_p == full_p
