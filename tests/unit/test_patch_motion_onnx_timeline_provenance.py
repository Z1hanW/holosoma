from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

import holosoma_inference.policies.wbt as inference_wbt_module
import holosoma_inference.tools.patch_motion_onnx as patch_motion_onnx_module
import scripts.compute_training_provenance as training_provenance_module
from holosoma.managers.command.terms.wbt import (
    MotionCommand,
    MotionLoader,
    _kinematic_lift_window_from_rel_z,
)
from holosoma_inference.policies.base import BasePolicy
from holosoma_inference.policies import base as base_policy_module
from holosoma_inference.tools.patch_motion_onnx import patch_model
from holosoma_inference.utils.embedded_motion_timeline import (
    EMBEDDED_MOTION_TIMELINE_CONTRACT_KEY,
    embedded_motion_timeline_contract_from_metadata,
    validate_embedded_motion_timeline_model,
)
from holosoma_inference.utils.contact_sidecar_contract import (
    EMBEDDED_CONTACT_SIDECAR_CONTRACT_KEY,
    embedded_contact_sidecar_contract_from_metadata,
)
from holosoma_inference.utils.button_window_contract import (
    EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY,
    embedded_button_window_contract_from_metadata,
)
from holosoma_inference.utils.policy_contract import PolicyContractError
from scripts.compute_training_provenance import (
    _contact_manifest_digest,
    _motion_manifest_digest,
)


def _transition_metadata() -> dict[str, object]:
    contract = {
        "version": 1,
        "control_dt_s": 0.02,
        "source_semantics": "global_multi_clip_runtime",
        "prepend": {
            "implementation": "runtime_hold",
            "applied": True,
            "steps": 2,
        },
        "append": {"implementation": "none", "applied": False, "steps": 0},
    }
    transition_sha = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return {
        "dof_names": ["j0", "j1"],
        "robot_urdf": "<robot name='fixture'/>",
        "motion_transition_contract": contract,
        "motion_transition_contract_sha256": transition_sha,
        "experiment_config": {
            "simulator": {
                "_target_": "holosoma.simulator.isaacsim.isaacsim.IsaacSim",
                "config": {
                    "name": "isaacsim",
                    "sim": {"fps": 50, "control_decimation": 1},
                },
            },
            "robot": {
                "init_state": {
                    "pos": [0.0, 0.0, 1.0],
                    "rot": [0.0, 0.0, 0.0, 1.0],
                    "default_joint_angles": {"j0": 0.0, "j1": 0.0},
                }
            },
            "command": {
                "setup_terms": {
                    "motion_command": {
                        "params": {
                            "motion_config": {
                                "motion_file": "motion_bank",
                                "motion_clip_id": None,
                                "motion_clip_name": None,
                                "body_name_ref": ["torso_link"],
                                "enable_default_pose_prepend": True,
                                "default_pose_prepend_duration_s": 0.04,
                                # Requested global append is intentionally not effective.
                                "enable_default_pose_append": True,
                                "default_pose_append_duration_s": 0.04,
                            }
                        }
                    }
                }
            },
        },
    }


def _attach_metadata(model: onnx.ModelProto, metadata: dict[str, object]) -> None:
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = json.dumps(value, allow_nan=False)


def _constant(name: str, output: str, value: np.ndarray) -> onnx.NodeProto:
    return helper.make_node(
        "Constant",
        [],
        [output],
        name=name,
        value=numpy_helper.from_array(value),
    )


def _write_source_model(
    path: Path,
    metadata: dict[str, object] | None = None,
    *,
    real_export_topology: bool = False,
) -> None:
    if real_export_topology:
        time_step = helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1])
        nodes = [
            helper.make_node("Cast", ["time_step"], ["cast_time_step"], to=TensorProto.INT64),
            _constant("squeeze_axes", "squeeze_axes_value", np.asarray([1], dtype=np.int64)),
            helper.make_node(
                "Squeeze",
                ["cast_time_step", "squeeze_axes_value"],
                ["squeezed_time_step"],
            ),
            _constant("max", "max_idx", np.asarray([1], dtype=np.int64)),
            # PyTorch's real opset-13 export leaves the optional Clip minimum empty.
            helper.make_node("Clip", ["squeezed_time_step", "", "max_idx"], ["clipped_idx"]),
        ]
    else:
        time_step = helper.make_tensor_value_info("time_step", TensorProto.INT64, [1])
        nodes = [
            _constant("min", "min_idx", np.asarray([0], dtype=np.int64)),
            _constant("max", "max_idx", np.asarray([1], dtype=np.int64)),
            helper.make_node("Clip", ["time_step", "min_idx", "max_idx"], ["clipped_idx"]),
        ]
    outputs = []
    widths = {"joint_pos": 2, "joint_vel": 2, "ref_pos_xyz": 3, "ref_quat_xyzw": 4}
    for offset, (name, width) in enumerate(widths.items()):
        value = np.arange(2 * width, dtype=np.float32).reshape(2, width) + offset
        nodes.append(_constant(f"{name}_constant", f"{name}_data", value))
        nodes.append(
            helper.make_node("Gather", [f"{name}_data", "clipped_idx"], [name], axis=0)
        )
        outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, width]))
    model = helper.make_model(
        helper.make_graph(nodes, "motion_patch_fixture", [time_step], outputs),
        opset_imports=[helper.make_opsetid("", 13)],
    )
    _attach_metadata(model, metadata or _transition_metadata())
    onnx.save(model, path)


def _write_motion(path: Path) -> None:
    body_quat = np.zeros((2, 2, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez(
        path,
        joint_names=np.asarray(["j0", "j1"]),
        body_names=np.asarray(["pelvis", "torso_link"]),
        joint_pos=np.asarray([[2.0, 2.0], [4.0, 4.0]], dtype=np.float32),
        joint_vel=np.zeros((2, 2), dtype=np.float32),
        body_pos_w=np.asarray(
            [
                [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
                [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        ),
        body_quat_w=body_quat,
        fps=np.asarray([50.0], dtype=np.float32),
    )


def _write_object_motion(path: Path) -> None:
    _write_motion(path)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload["object_pos_w"] = np.zeros((2, 3), dtype=np.float32)
    payload["object_quat_w"] = np.asarray(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    payload["object_size"] = np.ones((2, 3), dtype=np.float32)
    np.savez(path, **payload)


def _write_kinematic_object_motion(path: Path) -> np.ndarray:
    rel_z = np.asarray(
        [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    frame_count = int(rel_z.size)
    body_pos = np.zeros((frame_count, 2, 3), dtype=np.float32)
    body_pos[:, :, 2] = 1.0
    body_quat = np.zeros((frame_count, 2, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    object_pos = np.zeros((frame_count, 3), dtype=np.float32)
    object_pos[:, 2] = 1.0 + rel_z
    object_quat = np.zeros((frame_count, 4), dtype=np.float32)
    object_quat[:, 0] = 1.0
    np.savez(
        path,
        joint_names=np.asarray(["j0", "j1"]),
        body_names=np.asarray(["pelvis", "torso_link"]),
        joint_pos=np.zeros((frame_count, 2), dtype=np.float32),
        joint_vel=np.zeros((frame_count, 2), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        object_pos_w=object_pos,
        object_quat_w=object_quat,
        object_size=np.ones((frame_count, 3), dtype=np.float32),
        fps=np.asarray([50.0], dtype=np.float32),
    )
    return rel_z


def _single_static_kinematic_metadata() -> dict[str, object]:
    metadata = _transition_metadata()
    motion_cfg = metadata["experiment_config"]["command"]["setup_terms"][
        "motion_command"
    ]["params"]["motion_config"]
    motion_cfg.update(
        {
            "contact_aware_button_window_mode": "kinematic_lift",
            "enable_default_pose_prepend": True,
            "default_pose_prepend_duration_s": 0.2,
            "enable_default_pose_append": True,
            "default_pose_append_duration_s": 0.2,
        }
    )
    metadata["experiment_config"]["robot"]["init_state"]["pos"] = [0.0, 0.0, 0.5]
    contract = {
        "version": 1,
        "control_dt_s": 0.02,
        "source_semantics": "single_clip_static",
        "prepend": {
            "implementation": "static_splice",
            "applied": True,
            "steps": 10,
        },
        "append": {
            "implementation": "static_splice",
            "applied": True,
            "steps": 10,
        },
    }
    metadata["motion_transition_contract"] = contract
    metadata["motion_transition_contract_sha256"] = hashlib.sha256(
        json.dumps(
            contract,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return metadata


def _metadata(model: onnx.ModelProto) -> dict[str, object]:
    return {prop.key: json.loads(prop.value) for prop in model.metadata_props}


@pytest.fixture
def fake_pinocchio(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePinocchioRobot:
        def __init__(self, *_args, **_kwargs) -> None:
            self.real2pinocchio_index = np.asarray([0, 1])

        @staticmethod
        def fk_and_get_ref_body_pose_in_world(configuration):
            return (
                np.asarray(configuration[:3], dtype=np.float32),
                np.asarray(configuration[3:7], dtype=np.float32),
            )

    monkeypatch.setattr(inference_wbt_module, "PinocchioRobot", _FakePinocchioRobot)


def _native_training_static_button_window(motion_path: Path) -> tuple[int, int]:
    with np.load(motion_path, allow_pickle=False) as data:
        loader = object.__new__(MotionLoader)
        loader._joint_pos = torch.from_numpy(np.asarray(data["joint_pos"], dtype=np.float32))
        loader._joint_vel = torch.from_numpy(np.asarray(data["joint_vel"], dtype=np.float32))
        loader._body_pos_w = torch.from_numpy(np.asarray(data["body_pos_w"], dtype=np.float32))
        body_quat_wxyz = torch.from_numpy(np.asarray(data["body_quat_w"], dtype=np.float32))
        loader._body_quat_w = body_quat_wxyz[..., [1, 2, 3, 0]]
        loader._body_lin_vel_w = torch.zeros_like(loader._body_pos_w)
        loader._body_ang_vel_w = torch.zeros_like(loader._body_pos_w)
        loader._object_pos_w = torch.from_numpy(np.asarray(data["object_pos_w"], dtype=np.float32))
        object_quat_wxyz = torch.from_numpy(np.asarray(data["object_quat_w"], dtype=np.float32))
        loader._object_quat_w = object_quat_wxyz[..., [1, 2, 3, 0]]
        loader._object_lin_vel_w = torch.zeros_like(loader._object_pos_w)
        loader._object_size = torch.from_numpy(np.asarray(data["object_size"], dtype=np.float32))
    loader.has_object = True
    loader.num_clips = 1
    loader.clip_offsets = torch.tensor([0], dtype=torch.long)
    loader.clip_lengths = torch.tensor([loader._joint_pos.shape[0]], dtype=torch.long)
    loader.time_step_total = int(loader._joint_pos.shape[0])

    command = object.__new__(MotionCommand)
    command.motion = loader

    def motion_state(index: int) -> dict[str, torch.Tensor]:
        return {
            "joint_pos": loader._joint_pos[index].clone(),
            "joint_vel": loader._joint_vel[index].clone(),
            "body_pos": loader._body_pos_w[index].clone(),
            "body_quat": loader._body_quat_w[index].clone(),
            "body_lin_vel": loader._body_lin_vel_w[index].clone(),
            "body_ang_vel": loader._body_ang_vel_w[index].clone(),
            "object_pos": loader._object_pos_w[index].clone(),
            "object_quat": loader._object_quat_w[index].clone(),
            "object_lin_vel": loader._object_lin_vel_w[index].clone(),
            "object_size": loader._object_size[index].clone(),
        }

    def default_state(index: int) -> dict[str, torch.Tensor]:
        state = motion_state(index)
        state["joint_pos"].zero_()
        state["joint_vel"].zero_()
        state["body_pos"][:, 2] = 0.5
        state["body_lin_vel"].zero_()
        state["body_ang_vel"].zero_()
        return state

    command._build_and_apply_transition(
        start_state=default_state(0),
        target_state=motion_state(0),
        num_steps=10,
        prepend=True,
        drop_first=False,
        drop_last=True,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    command._build_and_apply_transition(
        start_state=motion_state(-1),
        target_state=default_state(-1),
        num_steps=10,
        prepend=False,
        drop_first=True,
        drop_last=False,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    return _kinematic_lift_window_from_rel_z(
        loader._object_pos_w[:, 2] - loader._body_pos_w[:, 0, 2]
    )


def test_patch_default_materializes_and_authenticates_effective_timeline(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(motion)

    patch_model(source, motion, output)

    model = onnx.load(output)
    metadata = _metadata(model)
    contract = validate_embedded_motion_timeline_model(model, metadata)
    assert contract is not None
    assert contract == embedded_motion_timeline_contract_from_metadata(metadata, required=True)
    assert contract["materialization"] == "effective_training_timeline"
    assert contract["source_motion_sha256"] == hashlib.sha256(motion.read_bytes()).hexdigest()
    assert contract["source_frame_count"] == 2
    assert contract["embedded_frame_count"] == 4
    assert contract["effective_prepend_steps"] == 2
    assert contract["effective_append_steps"] == 0


def test_single_static_kinematic_button_contract_matches_native_training_and_inference(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    metadata = _single_static_kinematic_metadata()
    _write_source_model(source, metadata)
    source_rel_z = _write_kinematic_object_motion(motion)

    expected_training_window = _native_training_static_button_window(motion)
    patch_model(source, motion, output)

    patched_metadata = _metadata(onnx.load(output))
    assert EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY in patched_metadata
    contract = embedded_button_window_contract_from_metadata(
        patched_metadata,
        required=True,
    )
    assert contract is not None
    assert contract["source_window"] == list(
        _kinematic_lift_window_from_rel_z(torch.from_numpy(source_rel_z.copy()))
    )
    assert contract["materialized_window"] == list(expected_training_window)
    assert contract["effective_prepend_steps"] == 10
    assert contract["effective_append_steps"] == 10

    policy = object.__new__(inference_wbt_module.WholeBodyTrackingPolicy)
    policy.config = SimpleNamespace(
        task=SimpleNamespace(apply_training_motion_transitions=True),
        robot=SimpleNamespace(dof_names=["j0", "j1"]),
    )
    policy._motion_data = inference_wbt_module.MotionData(
        motion,
        ["j0", "j1"],
        "torso_link",
    )
    policy.pinocchio_robot = inference_wbt_module.PinocchioRobot()
    policy._effective_motion_transition_settings = (
        inference_wbt_module._validated_runtime_motion_transition_settings(
            patched_metadata,
            apply_training_motion_transitions=True,
        )
    )
    policy._motion_transition_prepend_steps = (
        policy._maybe_apply_training_motion_transitions_to_motion_data(
            patched_metadata,
            "torso_link",
        )
    )
    policy._motion_cfg = patched_metadata["experiment_config"]["command"][
        "setup_terms"
    ]["motion_command"]["params"]["motion_config"]
    policy._onnx_metadata = patched_metadata

    assert policy._load_kinematic_button_window() == expected_training_window


def test_kinematic_button_patcher_rejects_motion_without_object_trajectory(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source, _single_static_kinematic_metadata())
    _write_motion(motion)

    with pytest.raises(ValueError, match="requires an object trajectory"):
        patch_model(source, motion, output)

    assert not output.exists()


def test_patcher_preserves_symlink_clip_id_while_hashing_target_bytes(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    target = tmp_path / "physical_payload_name.npz"
    link_dir = tmp_path / "logical"
    link_dir.mkdir()
    logical_motion = link_dir / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(target)
    logical_motion.symlink_to(target)

    patch_model(source, logical_motion, output)

    metadata = _metadata(onnx.load(output))
    contract = embedded_motion_timeline_contract_from_metadata(metadata, required=True)
    assert contract is not None
    assert contract["source_motion_sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    motion_cfg = metadata["experiment_config"]["command"]["setup_terms"]["motion_command"][
        "params"
    ]["motion_config"]
    assert Path(motion_cfg["motion_file"]).name == "clip.npz"
    assert motion_cfg["motion_clip_name"] == "clip"


def _mark_contact_window_policy(metadata: dict[str, object]) -> None:
    experiment = metadata["experiment_config"]
    experiment["algo"] = {
        "config": {
            "module_dict": {"actor": {"input_dim": ["actor_obs_drop_button"]}}
        }
    }
    experiment["observation"] = {
        "groups": {
            "actor_obs_drop_button": {"terms": {"drop_button": {}}},
        }
    }


def test_patcher_requires_explicit_unsafe_mode_for_unprovenanced_legacy_contact_policy(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    metadata = _transition_metadata()
    _mark_contact_window_policy(metadata)
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source, metadata)
    _write_object_motion(motion)

    with pytest.raises(ValueError, match="unsafe_allow_unbound_contact_sidecar"):
        patch_model(source, motion, output)
    assert not output.exists()

    patch_model(
        source,
        motion,
        output,
        unsafe_allow_unbound_contact_sidecar=True,
    )
    assert output.is_file()


def test_patcher_rejects_provenanced_contact_policy_without_full_contact_root(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    metadata = _transition_metadata()
    _mark_contact_window_policy(metadata)
    metadata["training_provenance"] = {
        "contact_sidecar_manifest_sha256": "b" * 64,
        "motion_shard_manifest_sha256": "c" * 64,
    }
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source, metadata)
    _write_object_motion(motion)

    with pytest.raises(ValueError, match="complete contact sidecar root"):
        patch_model(source, motion, output)
    assert not output.exists()


def test_patcher_publishes_verified_active_contact_contract(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    motion_bank = tmp_path / "motion_bank"
    contact_root = tmp_path / "contact_bank"
    clip_dir = contact_root / "clips" / "0000_clip"
    motion_bank.mkdir()
    clip_dir.mkdir(parents=True)
    motion = motion_bank / "clip.npz"
    _write_object_motion(motion)
    object_mesh = motion_bank / "object.obj"
    object_mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    object_urdf = motion_bank / "object.urdf"
    object_urdf.write_text(
        "<robot name='object'><link name='object'><visual><geometry>"
        "<mesh filename='object.obj'/></geometry></visual></link></robot>",
        encoding="utf-8",
    )
    object_map = motion_bank / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps(
            {"clips": {"clip": {"object_urdf_path": str(object_urdf)}}}
        ),
        encoding="utf-8",
    )
    (clip_dir / "teacher_rollout_reference.npz").write_bytes(b"rollout")
    for name in (
        "left_wrist_contact_points.npy",
        "left_wrist_contact_point_counts.npy",
        "left_wrist_contact_interval_steps.npy",
        "right_wrist_contact_points.npy",
        "right_wrist_contact_point_counts.npy",
        "right_wrist_contact_interval_steps.npy",
    ):
        value = (
            np.asarray([0, 2], dtype=np.int64)
            if name.endswith("_interval_steps.npy")
            else np.asarray([1], dtype=np.int64)
        )
        np.save(clip_dir / name, value)
    (clip_dir / "metadata.json").write_text(
        json.dumps({"clip_id": "clip", "fps": 50}),
        encoding="utf-8",
    )
    (clip_dir / "contact_intervals.json").write_text(
        json.dumps({"left_wrist": [0, 2]}),
        encoding="utf-8",
    )

    metadata = _transition_metadata()
    _mark_contact_window_policy(metadata)
    motion_cfg = metadata["experiment_config"]["command"]["setup_terms"]["motion_command"][
        "params"
    ]["motion_config"]
    motion_cfg["motion_file"] = str(motion_bank)
    motion_cfg["adaptive_sampling_contact_interval_root"] = str(contact_root)
    metadata["training_provenance"] = {
        "contact_sidecar_manifest_sha256": _contact_manifest_digest(
            motion_bank,
            contact_root,
        ),
        "motion_shard_manifest_sha256": _motion_manifest_digest(
            motion_bank,
            object_map,
            None,
        ),
    }
    source = tmp_path / "source.onnx"
    output = tmp_path / "patched.onnx"
    _write_source_model(source, metadata)

    patch_model(source, motion, output)

    patched_metadata = _metadata(onnx.load(output))
    assert EMBEDDED_CONTACT_SIDECAR_CONTRACT_KEY in patched_metadata
    contact_contract = embedded_contact_sidecar_contract_from_metadata(
        patched_metadata,
        required=True,
    )
    assert contact_contract is not None
    assert contact_contract["clip_id"] == "clip"
    assert contact_contract["selected_raw_interval"] == [0, 2]
    assert contact_contract["source_motion_sha256"] == hashlib.sha256(
        motion.read_bytes()
    ).hexdigest()

    wrong_object_map = tmp_path / "wrong_object_map.json"
    wrong_object_map.write_text(object_map.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="do not match training provenance"):
        patch_model(
            source,
            motion,
            tmp_path / "wrong-map.onnx",
            training_object_map=wrong_object_map,
        )

    moved_object_map = tmp_path / "moved_object_map.json"
    object_map.rename(moved_object_map)
    with pytest.raises(FileNotFoundError, match="sole default candidate"):
        patch_model(source, motion, tmp_path / "missing-map.onnx")
    moved_object_map.rename(object_map)

    with np.load(motion, allow_pickle=False) as data:
        changed_motion = {key: np.asarray(data[key]) for key in data.files}
    changed_motion["joint_pos"] = changed_motion["joint_pos"].copy()
    changed_motion["joint_pos"][0, 0] += 0.25
    np.savez(motion, **changed_motion)
    with pytest.raises(ValueError, match="do not match training provenance"):
        # The selected path and current bank member changed together.  Contact
        # v3 still matches, but the training motion-manifest digest must not.
        patch_model(source, motion, tmp_path / "changed-motion.onnx")


def test_training_motion_manifest_verification_preserves_exact_shard_none_semantics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = "d" * 64
    metadata = {
        "training_provenance": {"motion_shard_manifest_sha256": expected}
    }
    motion_bank = tmp_path / "motions"
    motion_bank.mkdir()
    object_map = tmp_path / "map.json"
    object_map.write_text("{}", encoding="utf-8")
    exact_shard = tmp_path / "exact_manifest.json"
    wrong_shard = tmp_path / "wrong_manifest.json"
    exact_shard.write_text("{}", encoding="utf-8")
    wrong_shard.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        training_provenance_module,
        "_motion_manifest_digest",
        lambda _bank, _map, shard: expected if shard == exact_shard else "e" * 64,
    )
    assert patch_motion_onnx_module._verify_training_motion_manifest(
        metadata=metadata,
        motion_bank_dir=motion_bank,
        object_map_path=object_map,
        shard_manifest_path=exact_shard,
    ) == expected
    with pytest.raises(ValueError, match="do not match training provenance"):
        patch_motion_onnx_module._verify_training_motion_manifest(
            metadata=metadata,
            motion_bank_dir=motion_bank,
            object_map_path=object_map,
            shard_manifest_path=None,
        )
    with pytest.raises(ValueError, match="do not match training provenance"):
        patch_motion_onnx_module._verify_training_motion_manifest(
            metadata=metadata,
            motion_bank_dir=motion_bank,
            object_map_path=object_map,
            shard_manifest_path=wrong_shard,
        )

    monkeypatch.setattr(
        training_provenance_module,
        "_motion_manifest_digest",
        lambda _bank, _map, shard: expected if shard is None else "e" * 64,
    )
    assert patch_motion_onnx_module._verify_training_motion_manifest(
        metadata=metadata,
        motion_bank_dir=motion_bank,
        object_map_path=object_map,
        shard_manifest_path=None,
    ) == expected


def test_patch_canonicalizes_real_export_cast_squeeze_optional_clip_minimum(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "real-export.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source, real_export_topology=True)
    _write_motion(motion)

    patch_model(source, motion, output)

    model = onnx.load(output)
    metadata = _metadata(model)
    assert validate_embedded_motion_timeline_model(model, metadata) is not None
    gather = next(
        node for node in model.graph.node if node.op_type == "Gather" and "joint_pos" in node.output
    )
    clip = next(
        node for node in model.graph.node if node.op_type == "Clip" and gather.input[1] in node.output
    )
    assert len(clip.input) == 3
    assert clip.input[1]
    minimum = next(
        node
        for node in model.graph.node
        if node.op_type == "Constant" and clip.input[1] in node.output
    )
    minimum_value = next(attribute for attribute in minimum.attribute if attribute.name == "value")
    np.testing.assert_array_equal(
        numpy_helper.to_array(minimum_value.t),
        np.asarray([0], dtype=np.int64),
    )


def test_validator_rejects_constant_motion_index_bypass(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(motion)
    patch_model(source, motion, output)
    model = onnx.load(output)
    model.graph.node.append(
        _constant("forced_index", "forced_index_value", np.asarray([0], dtype=np.int64))
    )
    clip = next(node for node in model.graph.node if node.op_type == "Clip")
    clip.input[0] = "forced_index_value"

    with pytest.raises(PolicyContractError, match="found 'Constant'"):
        validate_embedded_motion_timeline_model(model, _metadata(model))


def test_validator_rejects_arithmetic_motion_index_bypass(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(motion)
    patch_model(source, motion, output)
    model = onnx.load(output)
    model.graph.node.extend(
        [
            _constant("index_delta", "index_delta_value", np.asarray([0], dtype=np.int64)),
            helper.make_node(
                "Add",
                ["time_step", "index_delta_value"],
                ["arithmetic_index"],
            ),
        ]
    )
    clip = next(node for node in model.graph.node if node.op_type == "Clip")
    clip.input[0] = "arithmetic_index"

    with pytest.raises(PolicyContractError, match="found 'Add'"):
        validate_embedded_motion_timeline_model(model, _metadata(model))


def test_validator_rejects_time_step_initializer_bypass(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(motion)
    patch_model(source, motion, output)
    model = onnx.load(output)
    model.graph.initializer.append(
        numpy_helper.from_array(np.asarray([0], dtype=np.int64), name="time_step")
    )

    with pytest.raises(PolicyContractError, match="must not also be an initializer"):
        validate_embedded_motion_timeline_model(model, _metadata(model))


def test_raw_patch_requires_two_explicit_unsafe_decisions_and_cannot_masquerade(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "unsafe.onnx"
    _write_source_model(source)
    _write_motion(motion)

    with pytest.raises(ValueError, match="unsafe_allow_raw_motion_timeline=True"):
        patch_model(
            source,
            motion,
            output,
            apply_training_motion_transitions=False,
        )

    patch_model(
        source,
        motion,
        output,
        apply_training_motion_transitions=False,
        unsafe_allow_raw_motion_timeline=True,
    )
    model = onnx.load(output)
    metadata = _metadata(model)
    contract = embedded_motion_timeline_contract_from_metadata(metadata, required=True)
    assert contract is not None
    assert contract["materialization"] == "raw_unsafe_diagnostic"
    assert contract["source_frame_count"] == contract["embedded_frame_count"] == 2
    with pytest.raises(PolicyContractError, match="raw_unsafe_diagnostic"):
        validate_embedded_motion_timeline_model(model, metadata)
    assert validate_embedded_motion_timeline_model(
        model,
        metadata,
        allow_unsafe_diagnostic=True,
    ) == contract

    session_payloads: list[bytes] = []
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            base_policy_module.onnxruntime,
            "InferenceSession",
            lambda payload: session_payloads.append(payload),
        )
        with pytest.raises(PolicyContractError, match="raw_unsafe_diagnostic"):
            object.__new__(BasePolicy)._load_onnx_session_and_metadata(str(output))
        assert session_payloads == []
        monkeypatch.setenv(
            "HOLOSOMA_ALLOW_UNSAFE_RAW_EMBEDDED_MOTION_TIMELINE",
            "1",
        )
        object.__new__(BasePolicy)._load_onnx_session_and_metadata(str(output))
        assert session_payloads == [output.read_bytes()]
    finally:
        monkeypatch.undo()


def test_repatch_rebuilds_from_raw_source_once_and_rejects_tampered_parent(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    first = tmp_path / "first.onnx"
    second = tmp_path / "second.onnx"
    _write_source_model(source)
    _write_motion(motion)
    patch_model(source, motion, first)

    patch_model(first, motion, second)
    first_contract = embedded_motion_timeline_contract_from_metadata(
        _metadata(onnx.load(first)), required=True
    )
    second_contract = embedded_motion_timeline_contract_from_metadata(
        _metadata(onnx.load(second)), required=True
    )
    assert first_contract == second_contract
    assert second_contract is not None and second_contract["embedded_frame_count"] == 4

    tampered = onnx.load(first)
    for node in tampered.graph.node:
        if "joint_pos" in node.output and node.op_type == "Gather":
            data_name = node.input[0]
            break
    else:  # pragma: no cover - fixture invariant
        raise AssertionError("joint_pos Gather missing")
    for node in tampered.graph.node:
        if data_name in node.output and node.op_type == "Constant":
            for attribute in node.attribute:
                if attribute.name == "value":
                    value = numpy_helper.to_array(attribute.t).copy()
                    value[0, 0] += 1.0
                    attribute.t.CopyFrom(numpy_helper.from_array(value))
                    break
            break
    onnx.save(tampered, first)
    with pytest.raises(PolicyContractError, match="do not match their SHA-256"):
        patch_model(first, motion, second)


def test_base_loader_validates_constants_from_same_onnx_payload_before_session(
    tmp_path: Path,
    fake_pinocchio: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(motion)
    patch_model(source, motion, output)
    tampered = onnx.load(output)
    for node in tampered.graph.node:
        if node.name == "joint_pos_constant":
            attribute = next(item for item in node.attribute if item.name == "value")
            value = numpy_helper.to_array(attribute.t).copy()
            value[-1, -1] += 1.0
            attribute.t.CopyFrom(numpy_helper.from_array(value))
            break
    tampered_payload = tampered.SerializeToString()
    output.write_bytes(tampered_payload)

    session_payloads: list[bytes] = []
    monkeypatch.setattr(
        base_policy_module.onnxruntime,
        "InferenceSession",
        lambda payload: session_payloads.append(payload),
    )
    with pytest.raises(PolicyContractError, match="do not match their SHA-256"):
        object.__new__(BasePolicy)._load_onnx_session_and_metadata(str(output))
    assert session_payloads == []


def test_loader_rejects_clip_bound_that_hides_authenticated_frames(
    tmp_path: Path,
    fake_pinocchio: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(motion)
    patch_model(source, motion, output)
    model = onnx.load(output)
    for node in model.graph.node:
        if node.name == "max":
            attribute = next(item for item in node.attribute if item.name == "value")
            attribute.t.CopyFrom(numpy_helper.from_array(np.asarray([1], dtype=np.int64)))
            break
    output.write_bytes(model.SerializeToString())

    session_payloads: list[bytes] = []
    monkeypatch.setattr(
        base_policy_module.onnxruntime,
        "InferenceSession",
        lambda payload: session_payloads.append(payload),
    )
    with pytest.raises(PolicyContractError, match="Clip maximum contradicts"):
        object.__new__(BasePolicy)._load_onnx_session_and_metadata(str(output))
    assert session_payloads == []


def test_legacy_patch_signature_fails_closed_but_patcher_can_repair_it(
    tmp_path: Path,
    fake_pinocchio: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = _transition_metadata()
    motion_cfg = metadata["experiment_config"]["command"]["setup_terms"]["motion_command"][
        "params"
    ]["motion_config"]
    motion_cfg.update(
        {
            "motion_file": "/legacy/clip.npz",
            "motion_clip_id": 0,
            "motion_clip_name": "clip",
        }
    )
    source = tmp_path / "legacy.onnx"
    motion = tmp_path / "clip.npz"
    repaired = tmp_path / "repaired.onnx"
    _write_source_model(source, metadata)
    _write_motion(motion)
    model = onnx.load(source)
    with pytest.raises(PolicyContractError, match="Legacy patched ONNX"):
        validate_embedded_motion_timeline_model(model, _metadata(model))
    session_payloads: list[bytes] = []
    monkeypatch.setattr(
        base_policy_module.onnxruntime,
        "InferenceSession",
        lambda payload: session_payloads.append(payload),
    )
    with pytest.raises(PolicyContractError, match="Legacy patched ONNX"):
        object.__new__(BasePolicy)._load_onnx_session_and_metadata(str(source))
    assert session_payloads == []

    patch_model(source, motion, repaired)
    repaired_model = onnx.load(repaired)
    assert validate_embedded_motion_timeline_model(
        repaired_model,
        _metadata(repaired_model),
    )["embedded_frame_count"] == 4


def test_unmarked_legacy_export_without_patch_provenance_remains_compatible(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.onnx"
    _write_source_model(source)
    model = onnx.load(source)
    metadata = _metadata(model)
    assert EMBEDDED_MOTION_TIMELINE_CONTRACT_KEY not in metadata
    assert validate_embedded_motion_timeline_model(model, metadata) is None


@pytest.mark.parametrize("name_field", ["joint_names", "body_names"])
def test_patcher_rejects_pickled_object_name_arrays(
    name_field: str,
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "object-names.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(motion)
    with np.load(motion, allow_pickle=False) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    payload[name_field] = payload[name_field].astype(object)
    np.savez(motion, **payload)

    with pytest.raises(ValueError, match="pickled/object|Object arrays"):
        patch_model(source, motion, output)
    assert not output.exists()


def test_motion_data_binds_one_stable_external_payload_to_expected_sha256(
    tmp_path: Path,
) -> None:
    motion = tmp_path / "clip.npz"
    _write_motion(motion)
    expected_sha256 = hashlib.sha256(motion.read_bytes()).hexdigest()

    loaded = inference_wbt_module.MotionData(
        motion,
        ["j0", "j1"],
        "torso_link",
        expected_source_sha256=expected_sha256,
    )

    assert loaded.source_sha256 == expected_sha256
    with pytest.raises(ValueError, match="does not match patched ONNX provenance"):
        inference_wbt_module.MotionData(
            motion,
            ["j0", "j1"],
            "torso_link",
            expected_source_sha256="0" * 64,
        )


def test_atomic_publish_preserves_previous_output_when_save_fails(
    tmp_path: Path,
    fake_pinocchio: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(motion)
    prior_payload = b"previous-complete-artifact"
    output.write_bytes(prior_payload)
    monkeypatch.setattr(
        patch_motion_onnx_module.onnx,
        "save",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("injected save failure")),
    )

    with pytest.raises(OSError, match="injected save failure"):
        patch_model(source, motion, output)

    assert output.read_bytes() == prior_payload
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_full_onnx_checker_rejects_invalid_graph_before_atomic_publish(
    tmp_path: Path,
    fake_pinocchio: None,
) -> None:
    source = tmp_path / "source.onnx"
    motion = tmp_path / "clip.npz"
    output = tmp_path / "patched.onnx"
    _write_source_model(source)
    _write_motion(motion)
    malformed = onnx.load(source)
    malformed.graph.node.append(
        helper.make_node("Identity", ["missing_value"], ["unused_invalid_value"])
    )
    onnx.save(malformed, source)
    prior_payload = b"previous-complete-artifact"
    output.write_bytes(prior_payload)

    with pytest.raises(onnx.checker.ValidationError):
        patch_model(source, motion, output)

    assert output.read_bytes() == prior_payload
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []
