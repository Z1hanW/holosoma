#!/usr/bin/env python3
"""Patch a Holosoma WBT ONNX to use a single motion clip."""

from __future__ import annotations

import argparse
import io
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import numpy_helper

from holosoma_inference.utils.contact_sidecar_contract import (
    EMBEDDED_CONTACT_SIDECAR_CONTRACT_KEY,
    EMBEDDED_CONTACT_SIDECAR_CONTRACT_SHA256_KEY,
    build_verified_contact_sidecar_contract,
    embedded_contact_sidecar_contract_from_metadata,
    policy_requires_contact_window,
)
from holosoma_inference.utils.button_window_contract import (
    EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY,
    EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY,
    build_kinematic_button_window_contract,
    embedded_button_window_contract_from_metadata,
    kinematic_lift_window_from_rel_z_np,
    validated_contact_aware_button_window_mode,
)
from holosoma_inference.utils.embedded_motion_timeline import (
    EMBEDDED_MOTION_TIMELINE_CONTRACT_KEY,
    EMBEDDED_MOTION_TIMELINE_CONTRACT_SHA256_KEY,
    EMBEDDED_MOTION_TENSOR_NAMES,
    build_embedded_motion_timeline_contract,
    embedded_motion_tensors_sha256,
    read_stable_regular_file_bytes,
    validate_embedded_motion_timeline_model,
)
from holosoma_inference.utils.policy_contract import (
    effective_motion_transition_settings_from_metadata,
)


def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()
    if path.suffix == ".pt":
        candidate = path.with_suffix(".onnx")
        if not candidate.is_file():
            raise FileNotFoundError(f"Expected sibling ONNX next to checkpoint: {candidate}")
        return candidate
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _absolute_without_symlink_resolution(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else (Path.cwd() / expanded).absolute()


def _load_motion_clip(
    motion_path: Path,
    dof_names: list[str],
    ref_body_name: str,
    *,
    motion_payload: bytes | None = None,
) -> tuple[dict[str, np.ndarray], str]:
    payload = (
        read_stable_regular_file_bytes(motion_path, label="Motion source")
        if motion_payload is None
        else motion_payload
    )
    # Reuse the deployment loader so patching and runtime accept exactly the
    # same non-pickled schema, numeric dtypes, shapes, names, and quaternions.
    from holosoma_inference.policies.wbt import MotionData

    motion_data = MotionData(
        motion_path,
        dof_names,
        ref_body_name,
        motion_payload=payload,
    )

    motion = {
        "joint_pos": motion_data.joint_pos,
        "joint_vel": motion_data.joint_vel,
        "ref_pos_xyz": motion_data.ref_pos_w,
        "ref_quat_xyzw": motion_data.ref_quat_w[:, [1, 2, 3, 0]],
        "root_pos_w": motion_data.root_pos_w,
        "root_quat_wxyz": motion_data.root_quat_w,
    }
    if motion_data.has_object and motion_data.object_pos_w is not None:
        motion["object_pos_w"] = motion_data.object_pos_w
        motion["object_quat_wxyz"] = motion_data.object_quat_w
        motion["object_size"] = motion_data.object_size
    return motion, motion_data.source_sha256


def _extract_motion_cfg(metadata: dict[str, object]) -> dict | None:
    experiment_config = metadata.get("experiment_config")
    if not isinstance(experiment_config, dict):
        return None
    motion_cfg = (
        experiment_config.get("command", {})
        .get("setup_terms", {})
        .get("motion_command", {})
        .get("params", {})
        .get("motion_config", {})
    )
    return motion_cfg if isinstance(motion_cfg, dict) else None


def _resolve_directory_input(
    raw_path: str | Path,
    *,
    model_path: Path,
    label: str,
) -> Path:
    unresolved = Path(raw_path).expanduser()
    candidates = [unresolved]
    if not unresolved.is_absolute():
        candidates.append(model_path.parent / unresolved)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} directory does not exist: {raw_path}")


def _resolve_file_input(
    raw_path: str | Path,
    *,
    model_path: Path,
    label: str,
) -> Path:
    unresolved = Path(raw_path).expanduser()
    candidates = [unresolved]
    if not unresolved.is_absolute():
        candidates.append(model_path.parent / unresolved)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"{label} file does not exist: {raw_path}")


def _verify_training_motion_manifest(
    *,
    metadata: dict[str, object],
    motion_bank_dir: Path,
    object_map_path: Path,
    shard_manifest_path: Path | None,
) -> str:
    training_provenance = metadata.get("training_provenance")
    expected = (
        training_provenance.get("motion_shard_manifest_sha256")
        if isinstance(training_provenance, dict)
        else None
    )
    if (
        not isinstance(expected, str)
        or len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError(
            "training_provenance.motion_shard_manifest_sha256 must be a lowercase SHA-256 digest."
        )
    try:
        # Import the training implementation rather than duplicating its
        # object-map/URDF/mesh/shard closure algorithm in the inference package.
        from scripts.compute_training_provenance import _motion_manifest_digest
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "Scientific contact-sidecar patching requires the repository training provenance "
            "module to verify motion NPZ/object/URDF/shard bytes. Run patch_motion_onnx from the "
            "Holosoma source checkout with src/holosoma on PYTHONPATH; inference-only installs "
            "cannot publish this digest-bound artifact."
        ) from exc
    actual = _motion_manifest_digest(
        motion_bank_dir,
        object_map_path,
        shard_manifest_path,
    )
    if actual != expected:
        shard_description = (
            "disabled (training must also have used no shard manifest)"
            if shard_manifest_path is None
            else str(shard_manifest_path)
        )
        raise ValueError(
            "Motion bank/object assets/shard manifest do not match training provenance: "
            f"declared={expected}, actual={actual}, motion_bank={motion_bank_dir}, "
            f"object_map={object_map_path}, shard_manifest={shard_description}. Provide the exact "
            "--training-object-map and --motion-shard-manifest inputs used by training."
        )
    return expected


def _motion_runtime_metadata_from_payload(
    payload: bytes,
    *,
    motion_path: Path,
) -> tuple[float, bool]:
    try:
        archive = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"Motion source is not a non-pickled NPZ archive: {motion_path}") from exc
    if not isinstance(archive, np.lib.npyio.NpzFile):
        raise ValueError(f"Motion source must be an NPZ archive: {motion_path}")
    with archive as data:
        if "fps" not in data:
            raise ValueError(f"Motion source is missing scalar fps metadata: {motion_path}")
        values = np.asarray(data["fps"]).reshape(-1)
        if values.size != 1 or values.dtype.kind not in {"i", "u", "f"}:
            raise ValueError(f"Motion source fps must be one real numeric scalar: {motion_path}")
        fps = float(values[0])
        has_object = "object_pos_w" in data and "object_quat_w" in data
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"Motion source fps must be finite and positive: {motion_path}")
    return fps, has_object


def _extract_robot_init_state(metadata: dict[str, object]) -> dict | None:
    experiment_config = metadata.get("experiment_config")
    if not isinstance(experiment_config, dict):
        return None
    robot_cfg = experiment_config.get("robot", {})
    if not isinstance(robot_cfg, dict):
        return None
    init_state = robot_cfg.get("init_state")
    return init_state if isinstance(init_state, dict) else None


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return np.divide(quat, norm, out=quat, where=norm > 0)


def _slerp_quat_wxyz(start: np.ndarray, end: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    start = _normalize_quat_wxyz(np.asarray(start, dtype=np.float32).reshape(4))
    end = _normalize_quat_wxyz(np.asarray(end, dtype=np.float32).reshape(4))
    alphas = np.asarray(alphas, dtype=np.float32).reshape(-1)
    if alphas.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    dot = float(np.dot(start, end))
    if dot < 0.0:
        end = -end
        dot = -dot

    if dot > 0.9995:
        blended = start[None, :] + (end - start)[None, :] * alphas[:, None]
        return _normalize_quat_wxyz(blended)

    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * alphas
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, None] * start[None, :]) + (s1[:, None] * end[None, :])


def _apply_transition(
    motion: dict[str, np.ndarray],
    *,
    start_state: dict[str, np.ndarray],
    target_state: dict[str, np.ndarray],
    num_steps: int,
    prepend: bool,
    drop_first: bool,
    drop_last: bool,
) -> None:
    if num_steps <= 0:
        return

    alphas = np.linspace(0.0, 1.0, num_steps + 1, dtype=np.float32)
    if drop_first:
        alphas = alphas[1:]
    if drop_last:
        alphas = alphas[:-1]
    if alphas.size == 0:
        return

    def _lerp(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        view = alphas.reshape(-1, *([1] * a.ndim))
        return a + view * (b - a)

    segment_joint_pos = _lerp(start_state["joint_pos"], target_state["joint_pos"])
    segment_joint_vel = _lerp(start_state["joint_vel"], target_state["joint_vel"])
    segment_root_pos = _lerp(start_state["root_pos"], target_state["root_pos"])
    segment_ref_pos = _lerp(start_state["ref_pos"], target_state["ref_pos"])
    segment_root_quat = _slerp_quat_wxyz(start_state["root_quat"], target_state["root_quat"], alphas)
    segment_ref_quat = _slerp_quat_wxyz(start_state["ref_quat"], target_state["ref_quat"], alphas)
    has_object = "object_pos" in start_state and "object_pos" in target_state
    if has_object:
        segment_object_pos = _lerp(start_state["object_pos"], target_state["object_pos"])
        segment_object_quat = _slerp_quat_wxyz(
            start_state["object_quat"],
            target_state["object_quat"],
            alphas,
        )
        segment_object_size = _lerp(start_state["object_size"], target_state["object_size"])

    if prepend:
        motion["joint_pos"] = np.concatenate([segment_joint_pos, motion["joint_pos"]], axis=0)
        motion["joint_vel"] = np.concatenate([segment_joint_vel, motion["joint_vel"]], axis=0)
        motion["root_pos_w"] = np.concatenate([segment_root_pos, motion["root_pos_w"]], axis=0)
        motion["root_quat_wxyz"] = np.concatenate([segment_root_quat, motion["root_quat_wxyz"]], axis=0)
        motion["ref_pos_xyz"] = np.concatenate([segment_ref_pos, motion["ref_pos_xyz"]], axis=0)
        motion["ref_quat_xyzw"] = np.concatenate([segment_ref_quat[:, [1, 2, 3, 0]], motion["ref_quat_xyzw"]], axis=0)
        if has_object:
            motion["object_pos_w"] = np.concatenate(
                [segment_object_pos, motion["object_pos_w"]], axis=0
            )
            motion["object_quat_wxyz"] = np.concatenate(
                [segment_object_quat, motion["object_quat_wxyz"]], axis=0
            )
            motion["object_size"] = np.concatenate(
                [segment_object_size, motion["object_size"]], axis=0
            )
    else:
        motion["joint_pos"] = np.concatenate([motion["joint_pos"], segment_joint_pos], axis=0)
        motion["joint_vel"] = np.concatenate([motion["joint_vel"], segment_joint_vel], axis=0)
        motion["root_pos_w"] = np.concatenate([motion["root_pos_w"], segment_root_pos], axis=0)
        motion["root_quat_wxyz"] = np.concatenate([motion["root_quat_wxyz"], segment_root_quat], axis=0)
        motion["ref_pos_xyz"] = np.concatenate([motion["ref_pos_xyz"], segment_ref_pos], axis=0)
        motion["ref_quat_xyzw"] = np.concatenate([motion["ref_quat_xyzw"], segment_ref_quat[:, [1, 2, 3, 0]]], axis=0)
        if has_object:
            motion["object_pos_w"] = np.concatenate(
                [motion["object_pos_w"], segment_object_pos], axis=0
            )
            motion["object_quat_wxyz"] = np.concatenate(
                [motion["object_quat_wxyz"], segment_object_quat], axis=0
            )
            motion["object_size"] = np.concatenate(
                [motion["object_size"], segment_object_size], axis=0
            )


def _maybe_apply_training_motion_transitions(
    motion: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    dof_names: list[str],
    ref_body_name: str,
) -> None:
    transition_settings = effective_motion_transition_settings_from_metadata(metadata)
    init_state = _extract_robot_init_state(metadata)
    robot_urdf = metadata.get("robot_urdf")

    prepend_contract = transition_settings["prepend"]
    append_contract = transition_settings["append"]
    needs_prepend = bool(prepend_contract["applied"])
    needs_append = bool(append_contract["applied"])
    if not needs_prepend and not needs_append:
        return
    if not isinstance(init_state, dict) or not isinstance(robot_urdf, str):
        raise ValueError(
            "Applied motion transitions require serialized robot init_state and robot_urdf metadata."
        )

    from holosoma_inference.policies.wbt import PinocchioRobot
    from holosoma_inference.utils.math.quat import quat_to_rpy, rpy_to_quat, wxyz_to_xyzw, xyzw_to_wxyz

    pin_robot = PinocchioRobot(
        SimpleNamespace(dof_names=tuple(dof_names), motion={"body_name_ref": [ref_body_name]}),
        robot_urdf,
    )

    default_joint_angles = init_state.get("default_joint_angles")
    default_dof = np.zeros((len(dof_names),), dtype=np.float32)
    if isinstance(default_joint_angles, dict):
        for i, name in enumerate(dof_names):
            if name in default_joint_angles:
                default_dof[i] = float(default_joint_angles[name])
    else:
        default_dof = motion["joint_pos"][0].astype(np.float32, copy=True)

    def _build_default_state(use_motion_end: bool) -> dict[str, np.ndarray]:
        motion_idx = -1 if use_motion_end else 0
        motion_root_pos = motion["root_pos_w"][motion_idx]
        motion_root_quat = motion["root_quat_wxyz"][motion_idx]
        _, _, motion_yaw = quat_to_rpy(motion_root_quat)

        init_pos = np.asarray(init_state.get("pos", [0.0, 0.0, motion_root_pos[2]]), dtype=np.float32)
        init_rot_xyzw = np.asarray(init_state.get("rot", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32).reshape(1, 4)
        init_rot_wxyz = xyzw_to_wxyz(init_rot_xyzw)[0]
        init_roll, init_pitch, _ = quat_to_rpy(init_rot_wxyz)

        default_root_pos = np.asarray([motion_root_pos[0], motion_root_pos[1], init_pos[2]], dtype=np.float32)
        default_root_quat = rpy_to_quat((float(init_roll), float(init_pitch), float(motion_yaw))).astype(np.float32)

        root_quat_xyzw = wxyz_to_xyzw(default_root_quat.reshape(1, 4))[0]
        dof_pos_pin = default_dof[pin_robot.real2pinocchio_index]
        configuration = np.concatenate([default_root_pos, root_quat_xyzw, dof_pos_pin], axis=0)
        ref_pos, ref_quat_xyzw = pin_robot.fk_and_get_ref_body_pose_in_world(configuration)
        ref_quat_wxyz = xyzw_to_wxyz(ref_quat_xyzw.reshape(1, 4))[0]
        state = {
            "joint_pos": default_dof.astype(np.float32, copy=True),
            "joint_vel": np.zeros_like(default_dof, dtype=np.float32),
            "root_pos": default_root_pos.astype(np.float32, copy=False),
            "root_quat": default_root_quat.astype(np.float32, copy=False),
            "ref_pos": ref_pos.astype(np.float32, copy=False),
            "ref_quat": ref_quat_wxyz.astype(np.float32, copy=False),
        }
        if "object_pos_w" in motion:
            state.update(
                {
                    "object_pos": motion["object_pos_w"][motion_idx].astype(
                        np.float32, copy=False
                    ),
                    "object_quat": motion["object_quat_wxyz"][motion_idx].astype(
                        np.float32, copy=False
                    ),
                    "object_size": motion["object_size"][motion_idx].astype(
                        np.float32, copy=False
                    ),
                }
            )
        return state

    def _motion_state(idx: int) -> dict[str, np.ndarray]:
        state = {
            "joint_pos": motion["joint_pos"][idx].astype(np.float32, copy=False),
            "joint_vel": motion["joint_vel"][idx].astype(np.float32, copy=False),
            "root_pos": motion["root_pos_w"][idx].astype(np.float32, copy=False),
            "root_quat": motion["root_quat_wxyz"][idx].astype(np.float32, copy=False),
            "ref_pos": motion["ref_pos_xyz"][idx].astype(np.float32, copy=False),
            "ref_quat": motion["ref_quat_xyzw"][idx][[3, 0, 1, 2]].astype(np.float32, copy=False),
        }
        if "object_pos_w" in motion:
            state.update(
                {
                    "object_pos": motion["object_pos_w"][idx].astype(
                        np.float32, copy=False
                    ),
                    "object_quat": motion["object_quat_wxyz"][idx].astype(
                        np.float32, copy=False
                    ),
                    "object_size": motion["object_size"][idx].astype(
                        np.float32, copy=False
                    ),
                }
            )
        return state

    if needs_prepend:
        prepend_steps = int(prepend_contract["steps"])
        _apply_transition(
            motion,
            start_state=_build_default_state(use_motion_end=False),
            target_state=_motion_state(0),
            num_steps=prepend_steps,
            prepend=True,
            drop_first=False,
            drop_last=True,
        )

    if needs_append:
        append_steps = int(append_contract["steps"])
        _apply_transition(
            motion,
            start_state=_motion_state(-1),
            target_state=_build_default_state(use_motion_end=True),
            num_steps=append_steps,
            prepend=False,
            drop_first=True,
            drop_last=False,
        )


def _find_node_by_output(model: onnx.ModelProto, output_name: str, op_type: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if output_name in node.output and node.op_type == op_type:
            return node
    raise KeyError(f"Could not find {op_type} node producing '{output_name}'")


def _find_constant_node(model: onnx.ModelProto, const_output_name: str) -> onnx.NodeProto:
    for node in model.graph.node:
        if const_output_name in node.output and node.op_type == "Constant":
            return node
    raise KeyError(f"Could not find Constant node for '{const_output_name}'")


def _set_constant_tensor(const_node: onnx.NodeProto, value: np.ndarray) -> None:
    tensor = numpy_helper.from_array(np.asarray(value))
    for attr in const_node.attribute:
        if attr.name == "value":
            attr.t.CopyFrom(tensor)
            return
    raise KeyError(f"Constant node '{const_node.name}' has no 'value' attribute")


def _unique_graph_name(model: onnx.ModelProto, base: str) -> str:
    used = {
        name
        for node in model.graph.node
        for name in (*node.input, *node.output)
        if name
    }
    used.update(item.name for item in model.graph.input)
    used.update(item.name for item in model.graph.output)
    used.update(item.name for item in model.graph.value_info)
    used.update(item.name for item in model.graph.initializer)
    if base not in used:
        return base
    suffix = 1
    while f"{base}_{suffix}" in used:
        suffix += 1
    return f"{base}_{suffix}"


def _canonicalize_clip_bound(
    model: onnx.ModelProto,
    clip_node: onnx.NodeProto,
    *,
    input_index: int,
    value: np.ndarray,
    value_name_base: str,
    node_name_base: str,
) -> None:
    """Give one Clip bound a dedicated Constant without mutating shared values."""

    if len(clip_node.input) != 3:
        raise ValueError(
            "Motion timeline Clip must have exactly data, minimum, and maximum input slots."
        )
    bound_input = clip_node.input[input_index]
    if bound_input:
        producers = [
            node
            for node in model.graph.node
            if bound_input in node.output
        ]
        constants = [node for node in producers if node.op_type == "Constant"]
        consumer_slots = [
            (node, index)
            for node in model.graph.node
            for index, input_name in enumerate(node.input)
            if input_name == bound_input
        ]
        if (
            len(producers) == 1
            and len(constants) == 1
            and len(consumer_slots) == 1
            and consumer_slots[0][0] is clip_node
            and consumer_slots[0][1] == input_index
        ):
            _set_constant_tensor(constants[0], value)
            return

    output_name = _unique_graph_name(
        model,
        value_name_base,
    )
    node_names = {node.name for node in model.graph.node if node.name}
    node_name = node_name_base
    suffix = 1
    while node_name in node_names:
        node_name = f"{node_name_base}_{suffix}"
        suffix += 1
    bound_node = onnx.helper.make_node(
        "Constant",
        [],
        [output_name],
        name=node_name,
        value=numpy_helper.from_array(value),
    )
    clip_index = next(
        index for index, node in enumerate(model.graph.node) if node is clip_node
    )
    model.graph.node.insert(clip_index, bound_node)
    clip_node.input[input_index] = output_name


def _canonicalize_clip_bounds(
    model: onnx.ModelProto,
    clip_node: onnx.NodeProto,
    *,
    maximum: int,
) -> None:
    """Canonicalize optional/shared Clip bounds to explicit dedicated INT64 values."""

    _canonicalize_clip_bound(
        model,
        clip_node,
        input_index=1,
        value=np.asarray([0], dtype=np.int64),
        value_name_base="holosoma_embedded_motion_clip_minimum",
        node_name_base="HolosomaEmbeddedMotionClipMinimum",
    )
    _canonicalize_clip_bound(
        model,
        clip_node,
        input_index=2,
        value=np.asarray([maximum], dtype=np.int64),
        value_name_base="holosoma_embedded_motion_clip_maximum",
        node_name_base="HolosomaEmbeddedMotionClipMaximum",
    )


def _atomic_save_model(model: onnx.ModelProto, output_path: Path) -> None:
    """Publish a complete ONNX artifact with one same-filesystem replace."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        onnx.save(model, temporary_path)
        with temporary_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_path, output_path)
        directory_fd = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary_path.unlink(missing_ok=True)


def _decode_model_metadata(model: onnx.ModelProto) -> dict[str, object]:
    metadata: dict[str, object] = {}

    def _reject_nonfinite_json(constant: str) -> None:
        raise ValueError(f"non-finite JSON constant {constant!r}")

    for prop in model.metadata_props:
        if not prop.key:
            raise ValueError("ONNX contains an empty metadata key.")
        if prop.key in metadata:
            raise ValueError(f"ONNX contains ambiguous duplicate metadata key {prop.key!r}.")
        try:
            metadata[prop.key] = json.loads(
                prop.value,
                parse_constant=_reject_nonfinite_json,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"ONNX metadata {prop.key!r} is not strict finite JSON.") from exc
    return metadata


def _patch_metadata(
    model: onnx.ModelProto,
    metadata: dict[str, object],
    motion_file: str,
    *,
    embedded_timeline_contract: dict[str, object],
    embedded_timeline_contract_sha256: str,
    embedded_contact_sidecar_contract: dict[str, object] | None,
    embedded_contact_sidecar_contract_sha256: str | None,
    embedded_button_window_contract: dict[str, object] | None,
    embedded_button_window_contract_sha256: str | None,
    action_scale_override: float | None = None,
) -> None:
    metadata = dict(metadata)

    experiment_config = metadata.get("experiment_config")
    if isinstance(experiment_config, dict):
        motion_cfg = (
            experiment_config.setdefault("command", {})
            .setdefault("setup_terms", {})
            .setdefault("motion_command", {})
            .setdefault("params", {})
            .setdefault("motion_config", {})
        )
        if isinstance(motion_cfg, dict):
            motion_cfg["motion_file"] = motion_file
            motion_cfg["motion_clip_id"] = 0
            motion_cfg["motion_clip_name"] = Path(motion_file).stem
        if action_scale_override is not None:
            control_cfg = experiment_config.setdefault("robot", {}).setdefault("control", {})
            if isinstance(control_cfg, dict):
                control_cfg["action_scale"] = float(action_scale_override)

    metadata[EMBEDDED_MOTION_TIMELINE_CONTRACT_KEY] = embedded_timeline_contract
    metadata[EMBEDDED_MOTION_TIMELINE_CONTRACT_SHA256_KEY] = (
        embedded_timeline_contract_sha256
    )
    metadata.pop(EMBEDDED_CONTACT_SIDECAR_CONTRACT_KEY, None)
    metadata.pop(EMBEDDED_CONTACT_SIDECAR_CONTRACT_SHA256_KEY, None)
    if embedded_contact_sidecar_contract is not None:
        if embedded_contact_sidecar_contract_sha256 is None:
            raise ValueError(
                "Embedded contact-sidecar contract requires its SHA-256 digest."
            )
        metadata[EMBEDDED_CONTACT_SIDECAR_CONTRACT_KEY] = (
            embedded_contact_sidecar_contract
        )
        metadata[EMBEDDED_CONTACT_SIDECAR_CONTRACT_SHA256_KEY] = (
            embedded_contact_sidecar_contract_sha256
        )
    elif embedded_contact_sidecar_contract_sha256 is not None:
        raise ValueError(
            "Embedded contact-sidecar SHA-256 requires its serialized contract."
        )
    metadata.pop(EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY, None)
    metadata.pop(EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY, None)
    if embedded_button_window_contract is not None:
        if embedded_button_window_contract_sha256 is None:
            raise ValueError(
                "Embedded button-window contract requires its SHA-256 digest."
            )
        metadata[EMBEDDED_BUTTON_WINDOW_CONTRACT_KEY] = (
            embedded_button_window_contract
        )
        metadata[EMBEDDED_BUTTON_WINDOW_CONTRACT_SHA256_KEY] = (
            embedded_button_window_contract_sha256
        )
    elif embedded_button_window_contract_sha256 is not None:
        raise ValueError(
            "Embedded button-window SHA-256 requires its serialized contract."
        )

    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value, allow_nan=False)


def patch_model(
    model_path: Path,
    motion_path: Path,
    output_path: Path,
    *,
    apply_training_motion_transitions: bool = True,
    unsafe_allow_raw_motion_timeline: bool = False,
    action_scale_override: float | None = None,
    contact_interval_root: Path | None = None,
    contact_motion_bank_dir: Path | None = None,
    unsafe_allow_unbound_contact_sidecar: bool = False,
    training_object_map: Path | None = None,
    motion_shard_manifest: Path | None = None,
) -> Path:
    if type(apply_training_motion_transitions) is not bool:
        raise ValueError("apply_training_motion_transitions must be boolean.")
    if type(unsafe_allow_raw_motion_timeline) is not bool:
        raise ValueError("unsafe_allow_raw_motion_timeline must be boolean.")
    if type(unsafe_allow_unbound_contact_sidecar) is not bool:
        raise ValueError("unsafe_allow_unbound_contact_sidecar must be boolean.")
    if apply_training_motion_transitions and unsafe_allow_raw_motion_timeline:
        raise ValueError(
            "unsafe_allow_raw_motion_timeline is incompatible with applying training transitions."
        )
    if action_scale_override is not None:
        if isinstance(action_scale_override, bool):
            raise ValueError("action_scale_override must be a finite real number.")
        try:
            action_scale_override = float(action_scale_override)
        except (TypeError, ValueError) as exc:
            raise ValueError("action_scale_override must be a finite real number.") from exc
        if not math.isfinite(action_scale_override):
            raise ValueError("action_scale_override must be a finite real number.")

    model_path = Path(model_path).expanduser().resolve()
    # The logical basename is the active clip ID.  Keep it across symlinks;
    # stable descriptor reads and the bank-membership digest verify the target
    # bytes without replacing that semantic ID by the target's filename.
    motion_path = _absolute_without_symlink_resolution(motion_path)
    output_path = Path(output_path).expanduser().resolve()
    model_payload = read_stable_regular_file_bytes(model_path, label="Source ONNX")
    model = onnx.load_model_from_string(model_payload)
    metadata = _decode_model_metadata(model)
    # A re-patch always rebuilds constants from the separately hashed source
    # motion. Validate any prior provenance before replacing it so a tampered
    # patched artifact cannot be silently laundered into a new one.
    validate_embedded_motion_timeline_model(
        model,
        metadata,
        allow_unsafe_diagnostic=True,
        allow_legacy_unprovenanced_repatch_source=True,
    )
    embedded_contact_sidecar_contract_from_metadata(metadata)
    embedded_button_window_contract_from_metadata(metadata)
    dof_names = list(metadata["dof_names"])
    motion_cfg = _extract_motion_cfg(metadata) or {}
    body_name_ref = motion_cfg.get("body_name_ref", ["torso_link"])
    ref_body_name = body_name_ref[0] if isinstance(body_name_ref, list) and body_name_ref else "torso_link"

    transition_settings = effective_motion_transition_settings_from_metadata(metadata)
    prepend_steps = int(transition_settings["prepend"]["steps"])
    append_steps = int(transition_settings["append"]["steps"])
    has_effective_transition = bool(prepend_steps or append_steps)
    if (
        has_effective_transition
        and not apply_training_motion_transitions
        and not unsafe_allow_raw_motion_timeline
    ):
        raise ValueError(
            "Refusing to embed a raw motion timeline that skips the authenticated effective "
            "training transition. Set unsafe_allow_raw_motion_timeline=True only for an "
            "explicitly non-scientific diagnostic artifact."
        )
    if unsafe_allow_raw_motion_timeline and not has_effective_transition:
        raise ValueError(
            "unsafe_allow_raw_motion_timeline is meaningful only when an effective training "
            "transition would actually be skipped."
        )

    motion_payload = read_stable_regular_file_bytes(motion_path, label="Motion source")
    motion, source_motion_sha256 = _load_motion_clip(
        motion_path,
        dof_names,
        ref_body_name,
        motion_payload=motion_payload,
    )
    motion_fps, motion_has_object = _motion_runtime_metadata_from_payload(
        motion_payload,
        motion_path=motion_path,
    )
    source_frame_count = int(motion["joint_pos"].shape[0])
    button_window_mode = validated_contact_aware_button_window_mode(motion_cfg)
    button_window_contract: dict[str, object] | None = None
    button_window_contract_sha256: str | None = None
    source_window: tuple[int, int] | None = None
    if button_window_mode == "kinematic_lift":
        if not motion_has_object:
            raise ValueError(
                "Kinematic button-window mode requires an object trajectory in the "
                "selected motion; refusing to publish an artifact without a digest-bound "
                "pickup/drop window."
            )
        object_pos_w = motion.get("object_pos_w")
        if object_pos_w is None:
            raise ValueError(
                "Kinematic button-window mode requires object_pos_w in the selected motion."
            )
        source_window = kinematic_lift_window_from_rel_z_np(
            np.asarray(object_pos_w[:, 2] - motion["root_pos_w"][:, 2], dtype=np.float32)
        )
    if apply_training_motion_transitions:
        _maybe_apply_training_motion_transitions(
            motion,
            metadata,
            dof_names=dof_names,
            ref_body_name=ref_body_name,
        )
    if source_window is not None:
        applied_prepend_steps = prepend_steps if apply_training_motion_transitions else 0
        applied_append_steps = append_steps if apply_training_motion_transitions else 0
        materialized_button_window: tuple[int, int] | None = None
        if str(transition_settings["source_semantics"]) == "single_clip_static":
            materialized_button_window = kinematic_lift_window_from_rel_z_np(
                np.asarray(
                    motion["object_pos_w"][:, 2] - motion["root_pos_w"][:, 2],
                    dtype=np.float32,
                )
            )
        button_window_contract, button_window_contract_sha256 = (
            build_kinematic_button_window_contract(
                clip_id=motion_path.stem,
                source_motion_sha256=source_motion_sha256,
                source_motion_size=len(motion_payload),
                source_frame_count=source_frame_count,
                motion_fps=motion_fps,
                source_window=source_window,
                motion_transition_contract_sha256=str(
                    transition_settings["contract_sha256"]
                ),
                source_semantics=str(transition_settings["source_semantics"]),
                effective_prepend_steps=applied_prepend_steps,
                effective_append_steps=applied_append_steps,
                materialized_window=materialized_button_window,
            )
        )
    embedded_tensors = {
        name: motion[name].astype(np.float32, copy=False)
        for name in EMBEDDED_MOTION_TENSOR_NAMES
    }
    embedded_tensors_sha256, embedded_frame_count = embedded_motion_tensors_sha256(
        embedded_tensors
    )
    materialization = (
        "raw_unsafe_diagnostic"
        if has_effective_transition and not apply_training_motion_transitions
        else "effective_training_timeline"
    )
    embedded_contract, embedded_contract_sha256 = build_embedded_motion_timeline_contract(
        source_motion_sha256=source_motion_sha256,
        source_frame_count=source_frame_count,
        embedded_tensors_sha256=embedded_tensors_sha256,
        embedded_frame_count=embedded_frame_count,
        motion_transition_contract_sha256=transition_settings["contract_sha256"],
        effective_prepend_steps=prepend_steps,
        effective_append_steps=append_steps,
        materialization=materialization,
    )
    contact_contract: dict[str, object] | None = None
    contact_contract_sha256: str | None = None
    configured_contact_root = str(
        motion_cfg.get("adaptive_sampling_contact_interval_root") or ""
    ).strip()
    if contact_motion_bank_dir is not None and contact_interval_root is None and not configured_contact_root:
        raise ValueError(
            "contact_motion_bank_dir requires contact_interval_root or a serialized contact root."
        )
    if (
        training_object_map is not None or motion_shard_manifest is not None
    ) and contact_interval_root is None and not configured_contact_root:
        raise ValueError(
            "Training motion provenance overrides require a contact interval root because they "
            "are used to publish an active contact-sidecar contract."
        )
    raw_contact_root: str | Path | None = (
        contact_interval_root if contact_interval_root is not None else configured_contact_root or None
    )
    training_provenance = metadata.get("training_provenance")
    requires_contact_contract = bool(
        motion_has_object and policy_requires_contact_window(metadata)
    )
    if requires_contact_contract and not isinstance(training_provenance, dict):
        if not unsafe_allow_unbound_contact_sidecar:
            raise ValueError(
                "Contact-window policy has no training_provenance needed to bind its active "
                "sidecar. Use unsafe_allow_unbound_contact_sidecar=True only for an explicitly "
                "legacy, non-scientific diagnostic artifact."
            )
    if requires_contact_contract and isinstance(training_provenance, dict) and raw_contact_root is None:
        raise ValueError(
            "Digest-provenanced contact-window policy requires the complete contact sidecar root "
            "while patching. Provide --contact-interval-root if the serialized path moved."
        )
    if (
        raw_contact_root is not None
        and isinstance(training_provenance, dict)
        and (requires_contact_contract or contact_interval_root is not None)
    ):
        resolved_contact_root = _resolve_directory_input(
            raw_contact_root,
            model_path=model_path,
            label="Contact interval root",
        )
        if contact_motion_bank_dir is not None:
            resolved_motion_bank = _resolve_directory_input(
                contact_motion_bank_dir,
                model_path=model_path,
                label="Contact motion bank",
            )
        else:
            raw_motion_bank = motion_cfg.get("motion_file")
            try:
                resolved_motion_bank = _resolve_directory_input(
                    raw_motion_bank,
                    model_path=model_path,
                    label="Serialized motion bank",
                ) if isinstance(raw_motion_bank, str) and raw_motion_bank.strip() else motion_path.parent
            except FileNotFoundError:
                resolved_motion_bank = motion_path.parent
        if training_object_map is None:
            canonical_object_map = resolved_motion_bank / "_clip_object_urdf_map.json"
            if not canonical_object_map.is_file():
                raise FileNotFoundError(
                    "Scientific contact-sidecar patching requires the exact training object map. "
                    f"The sole default candidate is absent: {canonical_object_map}. Provide "
                    "--training-object-map explicitly."
                )
            resolved_object_map = canonical_object_map.resolve()
        else:
            resolved_object_map = _resolve_file_input(
                training_object_map,
                model_path=model_path,
                label="Training object map",
            )
        resolved_shard_manifest = (
            None
            if motion_shard_manifest is None
            else _resolve_file_input(
                motion_shard_manifest,
                model_path=model_path,
                label="Motion shard manifest",
            )
        )
        verified_motion_manifest_sha256 = _verify_training_motion_manifest(
            metadata=metadata,
            motion_bank_dir=resolved_motion_bank,
            object_map_path=resolved_object_map,
            shard_manifest_path=resolved_shard_manifest,
        )
        contact_contract, contact_contract_sha256 = build_verified_contact_sidecar_contract(
            metadata=metadata,
            motion_path=motion_path,
            motion_bank_dir=resolved_motion_bank,
            contact_root=resolved_contact_root,
            source_motion_sha256=source_motion_sha256,
            source_motion_size=len(motion_payload),
            source_frame_count=source_frame_count,
            motion_fps=motion_fps,
            verified_training_motion_manifest_sha256=(
                verified_motion_manifest_sha256
            ),
        )
    elif contact_interval_root is not None and not unsafe_allow_unbound_contact_sidecar:
        raise ValueError(
            "An explicit contact_interval_root requires training_provenance so the complete "
            "contact bank can be verified before an active sidecar is embedded."
        )
    for output_name, value in motion.items():
        if output_name not in {"joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"}:
            continue
        gather_node = _find_node_by_output(model, output_name, "Gather")
        const_node = _find_constant_node(model, gather_node.input[0])
        _set_constant_tensor(const_node, value.astype(np.float32, copy=False))

    joint_gather = _find_node_by_output(model, "joint_pos", "Gather")
    clip_node = _find_node_by_output(model, joint_gather.input[1], "Clip")
    _canonicalize_clip_bounds(
        model,
        clip_node,
        maximum=motion["joint_pos"].shape[0] - 1,
    )

    _patch_metadata(
        model,
        metadata,
        str(motion_path),
        embedded_timeline_contract=embedded_contract,
        embedded_timeline_contract_sha256=embedded_contract_sha256,
        embedded_contact_sidecar_contract=contact_contract,
        embedded_contact_sidecar_contract_sha256=contact_contract_sha256,
        embedded_button_window_contract=button_window_contract,
        embedded_button_window_contract_sha256=button_window_contract_sha256,
        action_scale_override=action_scale_override,
    )
    patched_metadata = _decode_model_metadata(model)
    validate_embedded_motion_timeline_model(
        model,
        patched_metadata,
        allow_unsafe_diagnostic=True,
    )
    embedded_contact_sidecar_contract_from_metadata(
        patched_metadata,
        required=contact_contract is not None,
    )
    embedded_button_window_contract_from_metadata(
        patched_metadata,
        required=button_window_contract is not None,
    )
    if requires_contact_contract and isinstance(training_provenance, dict) and contact_contract is None:
        raise RuntimeError(
            "Refusing to publish a digest-provenanced contact-window artifact without its active "
            "contact-sidecar contract."
        )
    # The provenance validator intentionally covers only the authenticated
    # timeline subgraph.  Refuse to publish an otherwise malformed ONNX graph.
    onnx.checker.check_model(model)
    _atomic_save_model(model, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Source .onnx path or sibling .pt checkpoint path")
    parser.add_argument("--motion-file", required=True, help="Single .npz motion clip")
    parser.add_argument("--output-path", required=True, help="Patched .onnx output path")
    parser.add_argument(
        "--unsafe-skip-training-motion-transitions",
        action="store_true",
        help=(
            "Produce an explicitly non-scientific raw-timeline diagnostic artifact instead of "
            "materializing the authenticated effective training transition."
        ),
    )
    parser.add_argument(
        "--action-scale-override",
        type=float,
        default=None,
        help="Override experiment_config.robot.control.action_scale in the patched ONNX metadata.",
    )
    parser.add_argument(
        "--contact-interval-root",
        default=None,
        help=(
            "Override the serialized full contact-sidecar bank root. When training provenance is "
            "present, the complete bank is verified and the active clip contract is embedded."
        ),
    )
    parser.add_argument(
        "--contact-motion-bank-dir",
        default=None,
        help=(
            "Override the full motion bank whose clip set defined the training contact manifest. "
            "This bank is needed only while patching, not during inference."
        ),
    )
    parser.add_argument(
        "--unsafe-allow-unbound-contact-sidecar",
        action="store_true",
        help=(
            "Permit a legacy contact-window policy without training provenance to use mutable "
            "external sidecars. The resulting rollout is explicitly non-scientific diagnostic."
        ),
    )
    parser.add_argument(
        "--training-object-map",
        default=None,
        help=(
            "Exact object map used by training. Defaults only to "
            "<contact-motion-bank-dir>/_clip_object_urdf_map.json; no glob guessing is used."
        ),
    )
    parser.add_argument(
        "--motion-shard-manifest",
        default=None,
        help=(
            "Exact rank-shard manifest used by training. Omit only when training also used no "
            "shard manifest; omission is part of the verified digest contract."
        ),
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    motion_path = _absolute_without_symlink_resolution(args.motion_file)
    output_path = Path(args.output_path).expanduser().resolve()
    patched = patch_model(
        model_path,
        motion_path,
        output_path,
        apply_training_motion_transitions=not args.unsafe_skip_training_motion_transitions,
        unsafe_allow_raw_motion_timeline=args.unsafe_skip_training_motion_transitions,
        action_scale_override=args.action_scale_override,
        contact_interval_root=(
            None
            if args.contact_interval_root is None
            else Path(args.contact_interval_root).expanduser()
        ),
        contact_motion_bank_dir=(
            None
            if args.contact_motion_bank_dir is None
            else Path(args.contact_motion_bank_dir).expanduser()
        ),
        unsafe_allow_unbound_contact_sidecar=args.unsafe_allow_unbound_contact_sidecar,
        training_object_map=(
            None
            if args.training_object_map is None
            else Path(args.training_object_map).expanduser()
        ),
        motion_shard_manifest=(
            None
            if args.motion_shard_manifest is None
            else Path(args.motion_shard_manifest).expanduser()
        ),
    )
    print(patched)


if __name__ == "__main__":
    main()
