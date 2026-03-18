#!/usr/bin/env python3
"""Patch a Holosoma WBT ONNX to use a single motion clip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _decode_names(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for item in values.tolist():
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


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


def _load_motion_clip(motion_path: Path, dof_names: list[str], ref_body_name: str) -> dict[str, np.ndarray]:
    with np.load(motion_path, allow_pickle=True) as data:
        joint_names = _decode_names(np.asarray(data["joint_names"]))
        body_names = _decode_names(np.asarray(data["body_names"]))

        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        if joint_pos.shape[1] == len(joint_names) + 7:
            joint_pos = joint_pos[:, 7:]
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        if joint_vel.shape[1] == len(joint_names) + 6:
            joint_vel = joint_vel[:, 6:]

        joint_indices = [joint_names.index(name) for name in dof_names]
        joint_pos = joint_pos[:, joint_indices]
        joint_vel = joint_vel[:, joint_indices]

        ref_idx = body_names.index(ref_body_name)
        root_idx = _resolve_root_body_index(body_names)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)

    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "ref_pos_xyz": body_pos_w[:, ref_idx, :],
        "ref_quat_xyzw": body_quat_w[:, ref_idx, :][:, [1, 2, 3, 0]],
        "root_pos_w": body_pos_w[:, root_idx, :],
        "root_quat_wxyz": body_quat_w[:, root_idx, :],
    }


def _resolve_root_body_index(body_names: list[str]) -> int:
    for candidate in ("pelvis", "pelvis_link", "base_link", "torso_link"):
        if candidate in body_names:
            return body_names.index(candidate)
    for idx, name in enumerate(body_names):
        if name.lower() != "world":
            return idx
    return 0


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


def _extract_robot_init_state(metadata: dict[str, object]) -> dict | None:
    experiment_config = metadata.get("experiment_config")
    if not isinstance(experiment_config, dict):
        return None
    robot_cfg = experiment_config.get("robot", {})
    if not isinstance(robot_cfg, dict):
        return None
    init_state = robot_cfg.get("init_state")
    return init_state if isinstance(init_state, dict) else None


def _extract_control_dt(metadata: dict[str, object]) -> float | None:
    experiment_config = metadata.get("experiment_config")
    if not isinstance(experiment_config, dict):
        return None
    sim_cfg = experiment_config.get("simulator", {}).get("config", {}).get("sim", {})
    if not isinstance(sim_cfg, dict):
        return None
    fps = float(sim_cfg.get("fps", 0.0) or 0.0)
    control_decimation = float(sim_cfg.get("control_decimation", 0.0) or 0.0)
    if fps <= 0.0 or control_decimation <= 0.0:
        return None
    return control_decimation / fps


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

    if prepend:
        motion["joint_pos"] = np.concatenate([segment_joint_pos, motion["joint_pos"]], axis=0)
        motion["joint_vel"] = np.concatenate([segment_joint_vel, motion["joint_vel"]], axis=0)
        motion["root_pos_w"] = np.concatenate([segment_root_pos, motion["root_pos_w"]], axis=0)
        motion["root_quat_wxyz"] = np.concatenate([segment_root_quat, motion["root_quat_wxyz"]], axis=0)
        motion["ref_pos_xyz"] = np.concatenate([segment_ref_pos, motion["ref_pos_xyz"]], axis=0)
        motion["ref_quat_xyzw"] = np.concatenate([segment_ref_quat[:, [1, 2, 3, 0]], motion["ref_quat_xyzw"]], axis=0)
    else:
        motion["joint_pos"] = np.concatenate([motion["joint_pos"], segment_joint_pos], axis=0)
        motion["joint_vel"] = np.concatenate([motion["joint_vel"], segment_joint_vel], axis=0)
        motion["root_pos_w"] = np.concatenate([motion["root_pos_w"], segment_root_pos], axis=0)
        motion["root_quat_wxyz"] = np.concatenate([motion["root_quat_wxyz"], segment_root_quat], axis=0)
        motion["ref_pos_xyz"] = np.concatenate([motion["ref_pos_xyz"], segment_ref_pos], axis=0)
        motion["ref_quat_xyzw"] = np.concatenate([motion["ref_quat_xyzw"], segment_ref_quat[:, [1, 2, 3, 0]]], axis=0)


def _maybe_apply_training_motion_transitions(
    motion: dict[str, np.ndarray],
    metadata: dict[str, object],
    *,
    dof_names: list[str],
    ref_body_name: str,
) -> None:
    motion_cfg = _extract_motion_cfg(metadata)
    init_state = _extract_robot_init_state(metadata)
    control_dt = _extract_control_dt(metadata)
    robot_urdf = metadata.get("robot_urdf")

    if not isinstance(motion_cfg, dict) or not isinstance(init_state, dict) or not isinstance(robot_urdf, str):
        return
    if control_dt is None or control_dt <= 0.0:
        return

    needs_prepend = bool(motion_cfg.get("enable_default_pose_prepend", False))
    needs_append = bool(motion_cfg.get("enable_default_pose_append", False))
    if not needs_prepend and not needs_append:
        return

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
        return {
            "joint_pos": default_dof.astype(np.float32, copy=True),
            "joint_vel": np.zeros_like(default_dof, dtype=np.float32),
            "root_pos": default_root_pos.astype(np.float32, copy=False),
            "root_quat": default_root_quat.astype(np.float32, copy=False),
            "ref_pos": ref_pos.astype(np.float32, copy=False),
            "ref_quat": ref_quat_wxyz.astype(np.float32, copy=False),
        }

    def _motion_state(idx: int) -> dict[str, np.ndarray]:
        return {
            "joint_pos": motion["joint_pos"][idx].astype(np.float32, copy=False),
            "joint_vel": motion["joint_vel"][idx].astype(np.float32, copy=False),
            "root_pos": motion["root_pos_w"][idx].astype(np.float32, copy=False),
            "root_quat": motion["root_quat_wxyz"][idx].astype(np.float32, copy=False),
            "ref_pos": motion["ref_pos_xyz"][idx].astype(np.float32, copy=False),
            "ref_quat": motion["ref_quat_xyzw"][idx][[3, 0, 1, 2]].astype(np.float32, copy=False),
        }

    if needs_prepend:
        prepend_duration = float(motion_cfg.get("default_pose_prepend_duration_s", 0.0) or 0.0)
        prepend_steps = round(prepend_duration / control_dt)
        if prepend_steps > 1:
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
        append_duration = float(motion_cfg.get("default_pose_append_duration_s", 0.0) or 0.0)
        append_steps = round(append_duration / control_dt)
        if append_steps > 1:
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


def _patch_metadata(model: onnx.ModelProto, motion_file: str) -> None:
    metadata: dict[str, object] = {}
    for prop in model.metadata_props:
        try:
            metadata[prop.key] = json.loads(prop.value)
        except Exception:
            metadata[prop.key] = prop.value

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

    del model.metadata_props[:]
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = json.dumps(value)


def patch_model(
    model_path: Path,
    motion_path: Path,
    output_path: Path,
    *,
    apply_training_motion_transitions: bool = False,
) -> Path:
    model = onnx.load(model_path)
    metadata = {prop.key: json.loads(prop.value) for prop in model.metadata_props}
    dof_names = list(metadata["dof_names"])
    motion_cfg = _extract_motion_cfg(metadata) or {}
    body_name_ref = motion_cfg.get("body_name_ref", ["torso_link"])
    ref_body_name = body_name_ref[0] if isinstance(body_name_ref, list) and body_name_ref else "torso_link"

    motion = _load_motion_clip(motion_path, dof_names, ref_body_name)
    if apply_training_motion_transitions:
        _maybe_apply_training_motion_transitions(
            motion,
            metadata,
            dof_names=dof_names,
            ref_body_name=ref_body_name,
        )
    for output_name, value in motion.items():
        if output_name not in {"joint_pos", "joint_vel", "ref_pos_xyz", "ref_quat_xyzw"}:
            continue
        gather_node = _find_node_by_output(model, output_name, "Gather")
        const_node = _find_constant_node(model, gather_node.input[0])
        _set_constant_tensor(const_node, value.astype(np.float32, copy=False))

    joint_gather = _find_node_by_output(model, "joint_pos", "Gather")
    clip_node = _find_node_by_output(model, joint_gather.input[1], "Clip")
    max_const = _find_constant_node(model, clip_node.input[2])
    _set_constant_tensor(max_const, np.array([motion["joint_pos"].shape[0] - 1], dtype=np.int64))

    _patch_metadata(model, str(motion_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Source .onnx path or sibling .pt checkpoint path")
    parser.add_argument("--motion-file", required=True, help="Single .npz motion clip")
    parser.add_argument("--output-path", required=True, help="Patched .onnx output path")
    parser.add_argument(
        "--apply-training-motion-transitions",
        action="store_true",
        help="Apply training-time default-pose prepend/append transitions from ONNX metadata before patching constants.",
    )
    args = parser.parse_args()

    model_path = _resolve_model_path(args.model_path)
    motion_path = Path(args.motion_file).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    patched = patch_model(
        model_path,
        motion_path,
        output_path,
        apply_training_motion_transitions=args.apply_training_motion_transitions,
    )
    print(patched)


if __name__ == "__main__":
    main()
