from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.utils.contact_sidecar_contract import (
    EMBEDDED_CONTACT_SIDECAR_CONTRACT_KEY,
    EMBEDDED_CONTACT_SIDECAR_CONTRACT_SHA256_KEY,
    _verified_contact_manifest,
    build_verified_contact_sidecar_contract,
    embedded_contact_sidecar_contract_from_metadata,
    policy_requires_contact_window,
)
from holosoma_inference.utils.policy_contract import PolicyContractError
from scripts.compute_training_provenance import _contact_manifest_digest


REQUIRED_ARRAYS = (
    "left_wrist_contact_points.npy",
    "left_wrist_contact_point_counts.npy",
    "left_wrist_contact_interval_steps.npy",
    "right_wrist_contact_points.npy",
    "right_wrist_contact_point_counts.npy",
    "right_wrist_contact_interval_steps.npy",
)


def _bank(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    motion_bank = tmp_path / "motions"
    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "clips" / "0000_clip"
    motion_bank.mkdir()
    clip_dir.mkdir(parents=True)
    motion_path = motion_bank / "clip.npz"
    motion_path.write_bytes(b"exact-selected-motion-bytes")
    (clip_dir / "teacher_rollout_reference.npz").write_bytes(b"rollout")
    for name in REQUIRED_ARRAYS:
        value = (
            np.asarray([2, 5], dtype=np.int64)
            if name.endswith("_interval_steps.npy")
            else np.asarray([1], dtype=np.int64)
        )
        np.save(clip_dir / name, value)
    (clip_dir / "metadata.json").write_text(
        json.dumps({"clip_id": "clip", "contact_interval_fps": 25}),
        encoding="utf-8",
    )
    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_wrist": [2, 5], "right_wrist": [3, 7]}),
        encoding="utf-8",
    )
    return motion_bank, contact_root, clip_dir, motion_path


def _metadata(contact_manifest_sha256: str) -> dict:
    return {
        "training_provenance": {
            "contact_sidecar_manifest_sha256": contact_manifest_sha256,
            "motion_shard_manifest_sha256": "d" * 64,
        },
        "motion_transition_contract_sha256": "a" * 64,
        "experiment_config": {
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "contact_interval_runtime_prepend_compensation": True,
                                "use_adaptive_timesteps_sampler": True,
                            }
                        }
                    }
                }
            }
        },
    }


def _build(
    *,
    motion_bank: Path,
    contact_root: Path,
    motion_path: Path,
    metadata: dict | None = None,
) -> tuple[dict, str, dict]:
    manifest_sha256 = _contact_manifest_digest(motion_bank, contact_root)
    metadata = _metadata(manifest_sha256) if metadata is None else metadata
    motion_payload = motion_path.read_bytes()
    contract, digest = build_verified_contact_sidecar_contract(
        metadata=metadata,
        motion_path=motion_path,
        motion_bank_dir=motion_bank,
        contact_root=contact_root,
        source_motion_sha256=hashlib.sha256(motion_payload).hexdigest(),
        source_motion_size=len(motion_payload),
        source_frame_count=20,
        motion_fps=50.0,
        verified_training_motion_manifest_sha256="d" * 64,
    )
    return contract, digest, metadata


def _attached_metadata(contract: dict, digest: str, metadata: dict) -> dict:
    result = copy.deepcopy(metadata)
    motion_cfg = result["experiment_config"]["command"]["setup_terms"]["motion_command"][
        "params"
    ]["motion_config"]
    motion_cfg.update(
        {
            "motion_file": "/portable/clip.npz",
            "motion_clip_id": 0,
            "motion_clip_name": "clip",
        }
    )
    result[EMBEDDED_CONTACT_SIDECAR_CONTRACT_KEY] = copy.deepcopy(contract)
    result[EMBEDDED_CONTACT_SIDECAR_CONTRACT_SHA256_KEY] = digest
    return result


def test_builder_matches_training_v3_manifest_and_binds_active_bytes(tmp_path: Path) -> None:
    motion_bank, contact_root, _clip_dir, motion_path = _bank(tmp_path)
    training_digest = _contact_manifest_digest(motion_bank, contact_root)
    helper_digest, _records, _payloads, _dir, bank_member = _verified_contact_manifest(
        motion_bank_dir=motion_bank,
        contact_root=contact_root,
        active_clip_id="clip",
    )
    assert helper_digest == training_digest

    contract, digest, metadata = _build(
        motion_bank=motion_bank,
        contact_root=contact_root,
        motion_path=motion_path,
    )
    assert contract["training_contact_sidecar_manifest_sha256"] == training_digest
    assert contract["motion_bank_member"] == bank_member
    assert contract["selected_raw_interval"] == [2, 7]
    assert contract["contact_interval_fps"] == 25.0
    assert embedded_contact_sidecar_contract_from_metadata(
        _attached_metadata(contract, digest, metadata)
    ) == contract


def test_builder_rejects_same_named_motion_with_different_bytes(tmp_path: Path) -> None:
    motion_bank, contact_root, _clip_dir, _bank_motion = _bank(tmp_path)
    selected_dir = tmp_path / "selected"
    selected_dir.mkdir()
    selected_motion = selected_dir / "clip.npz"
    selected_motion.write_bytes(b"different-motion-with-the-same-name")

    with pytest.raises(PolicyContractError, match="same-named member"):
        _build(
            motion_bank=motion_bank,
            contact_root=contact_root,
            motion_path=selected_motion,
        )


def test_builder_accepts_symlinked_bank_member_without_losing_logical_clip_id(
    tmp_path: Path,
) -> None:
    motion_bank, contact_root, _clip_dir, motion_path = _bank(tmp_path)
    target = tmp_path / "physical_payload_name.npz"
    target.write_bytes(motion_path.read_bytes())
    motion_path.unlink()
    motion_path.symlink_to(target)

    contract, _digest, _metadata_value = _build(
        motion_bank=motion_bank,
        contact_root=contact_root,
        motion_path=motion_path,
    )
    assert contract["clip_id"] == "clip"
    assert contract["motion_bank_member"]["name"] == "clip.npz"
    assert contract["motion_bank_member"]["sha256"] == hashlib.sha256(
        target.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    "raw_interval",
    ([1.5, 7], [True, 7], [1, 7, 9]),
)
def test_builder_rejects_lossy_or_overspecified_json_intervals(
    tmp_path: Path,
    raw_interval: list[object],
) -> None:
    motion_bank, contact_root, clip_dir, motion_path = _bank(tmp_path)
    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_wrist": raw_interval}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="No valid contact interval"):
        _build(
            motion_bank=motion_bank,
            contact_root=contact_root,
            motion_path=motion_path,
        )


@pytest.mark.parametrize(
    ("timebase_metadata", "expected_key", "expected_fps"),
    [
        ({}, None, None),
        ({"fps": 25}, "fps", 25.0),
        (
            {"contact_interval_fps": None, "fps": 25},
            "contact_interval_fps",
            None,
        ),
        (
            {"contact_interval_fps": 20, "fps": 25},
            "contact_interval_fps",
            20.0,
        ),
    ],
)
def test_contact_fps_precedence_matches_training_runtime(
    tmp_path: Path,
    timebase_metadata: dict[str, object],
    expected_key: str | None,
    expected_fps: float | None,
) -> None:
    motion_bank, contact_root, clip_dir, motion_path = _bank(tmp_path)
    (clip_dir / "metadata.json").write_text(
        json.dumps({"clip_id": "clip", **timebase_metadata}),
        encoding="utf-8",
    )
    contract, _digest, _metadata_value = _build(
        motion_bank=motion_bank,
        contact_root=contact_root,
        motion_path=motion_path,
    )
    assert contract["contact_interval_fps_key"] == expected_key
    assert contract["contact_interval_fps"] == expected_fps


@pytest.mark.parametrize("invalid_fps", ["bad", "25", True])
def test_invalid_primary_contact_fps_does_not_fall_back_to_fps(
    tmp_path: Path,
    invalid_fps: object,
) -> None:
    motion_bank, contact_root, clip_dir, motion_path = _bank(tmp_path)
    (clip_dir / "metadata.json").write_text(
        json.dumps({"clip_id": "clip", "contact_interval_fps": invalid_fps, "fps": 25}),
        encoding="utf-8",
    )
    with pytest.raises(PolicyContractError, match="finite positive"):
        _build(
            motion_bank=motion_bank,
            contact_root=contact_root,
            motion_path=motion_path,
        )


@pytest.mark.parametrize(
    ("actor_term", "adaptive", "uniform"),
    [
        ("drop_button", False, False),
        ("dof_pos", True, False),
        ("dof_pos", False, True),
    ],
)
def test_shared_contact_requirement_covers_every_runtime_consumer(
    actor_term: str,
    adaptive: bool,
    uniform: bool,
) -> None:
    metadata = {
        "experiment_config": {
            "algo": {
                "config": {
                    "module_dict": {"actor": {"input_dim": ["actor_obs"]}}
                }
            },
            "observation": {
                "groups": {"actor_obs": {"terms": {actor_term: {}}}}
            },
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "use_adaptive_timesteps_sampler": adaptive,
                                "uniform_t1_window_sampling_enabled": uniform,
                            }
                        }
                    }
                }
            },
        }
    }
    assert policy_requires_contact_window(metadata) is True


@pytest.mark.parametrize("invalid_flag", ["False", 0, 1, None])
def test_shared_contact_requirement_rejects_non_boolean_sampler_flags(
    invalid_flag: object,
) -> None:
    metadata = {
        "experiment_config": {
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "use_adaptive_timesteps_sampler": invalid_flag,
                                "uniform_t1_window_sampling_enabled": False,
                            }
                        }
                    }
                }
            }
        }
    }
    with pytest.raises(PolicyContractError, match="must be boolean"):
        policy_requires_contact_window(metadata)


@pytest.mark.parametrize(
    "tamper",
    ["manifest", "motion_manifest", "transition", "compensation"],
)
def test_runtime_contract_parser_fails_closed_on_external_binding_tamper(
    tmp_path: Path,
    tamper: str,
) -> None:
    motion_bank, contact_root, _clip_dir, motion_path = _bank(tmp_path)
    contract, digest, metadata = _build(
        motion_bank=motion_bank,
        contact_root=contact_root,
        motion_path=motion_path,
    )
    attached = _attached_metadata(contract, digest, metadata)
    if tamper == "manifest":
        attached["training_provenance"]["contact_sidecar_manifest_sha256"] = "b" * 64
    elif tamper == "motion_manifest":
        attached["training_provenance"]["motion_shard_manifest_sha256"] = "b" * 64
    elif tamper == "transition":
        attached["motion_transition_contract_sha256"] = "b" * 64
    else:
        attached["experiment_config"]["command"]["setup_terms"]["motion_command"][
            "params"
        ]["motion_config"]["contact_interval_runtime_prepend_compensation"] = False

    with pytest.raises(PolicyContractError):
        embedded_contact_sidecar_contract_from_metadata(attached)


def test_runtime_consumes_embedded_window_without_contact_bank(tmp_path: Path) -> None:
    motion_bank, contact_root, _clip_dir, motion_path = _bank(tmp_path)
    contract, digest, metadata = _build(
        motion_bank=motion_bank,
        contact_root=contact_root,
        motion_path=motion_path,
    )
    attached = _attached_metadata(contract, digest, metadata)

    policy = object.__new__(WholeBodyTrackingPolicy)
    policy._onnx_metadata = attached
    policy._motion_data = SimpleNamespace(
        motion_path=Path("/portable/clip.npz"),
        source_sha256=contract["source_motion_sha256"],
        source_size=contract["source_motion_size"],
        source_frame_count=20,
        frame_count=20,
        fps=50.0,
        has_object=True,
    )
    policy._motion_cfg = {
        "contact_interval_runtime_prepend_compensation": True,
        "use_adaptive_timesteps_sampler": True,
        "adaptive_sampling_contact_interval_root": "/deliberately/not/present",
    }
    policy._uses_contact_window_observation = False
    policy._motion_transition_prepend_steps = 0
    policy._effective_motion_transition_settings = {
        "source_semantics": "single_clip_static",
    }

    # [2, 7) at 25 Hz becomes [4, 14) at the 50 Hz motion rate.  No external
    # sidecar path is opened after the verified contract has been embedded.
    assert policy._load_contact_aware_button_window(Path("/portable/policy.onnx")) == (4, 14)
