#!/usr/bin/env python3
"""Stage MuJoCo web-demo assets for train_object_extend tracking rollouts."""

from __future__ import annotations

import argparse
from dataclasses import replace as dataclass_replace
import hashlib
import io
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import onnx
import onnxruntime

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_inference"))

from holosoma_inference.tools.patch_motion_onnx import patch_model
from holosoma_inference.utils.embedded_motion_timeline import (
    embedded_motion_timeline_contract_from_metadata,
    read_stable_regular_file_bytes,
)
from holosoma_inference.utils.policy_contract import (
    effective_motion_transition_settings_from_metadata,
)


DEFAULT_MODEL = Path(
    "/data/logs_new/boxer/20260316_200048-g1_29dof_wbt_w_object_extend_20260316_200027_s01_scale_1p0-"
    "g1_29dof_wbt_w_object_extend_20260316_200027/model_23500.onnx"
)
DEFAULT_MOTION = Path(
    "/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz"
)
DEFAULT_MOTION_DIR = Path(
    "/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry"
)
DEFAULT_MAX_CLIPS = 1


def _decode_names(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for item in values.tolist():
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def _resolve_root_body_index(body_names: list[str]) -> int:
    for candidate in ("pelvis", "pelvis_link", "base_link", "torso_link"):
        if candidate in body_names:
            return body_names.index(candidate)
    for idx, name in enumerate(body_names):
        if name.lower() != "world":
            return idx
    return 0


def _normalize_object_size_array(raw: np.ndarray | None, length: int) -> np.ndarray:
    if raw is None:
        return np.ones((length, 3), dtype=np.float32)
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 0:
        return np.full((length, 3), float(arr), dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] == 1:
            return np.full((length, 3), float(arr[0]), dtype=np.float32)
        if arr.shape[0] == 3:
            return np.repeat(arr.reshape(1, 3), repeats=length, axis=0)
        if arr.shape[0] == length:
            return np.repeat(arr.reshape(length, 1), repeats=3, axis=1)
    if arr.ndim == 2:
        if arr.shape == (1, 3):
            return np.repeat(arr, repeats=length, axis=0)
        if arr.shape == (length, 1):
            return np.repeat(arr, repeats=3, axis=1)
        if arr.shape == (length, 3):
            return arr
    raise ValueError(
        f"Unsupported object-size shape {arr.shape}; expected scalar, (3,), (T,), (T,3), (1,3), or (T,1)."
    )


def _read_onnx_metadata(model_path: Path) -> dict:
    model = onnx.load(model_path)
    metadata = {}
    for prop in model.metadata_props:
        metadata[prop.key] = json.loads(prop.value)
    return metadata


def _read_onnx_io(model_path: Path) -> tuple[list[dict], list[str]]:
    model = onnx.load(model_path)
    inputs = []
    for value in model.graph.input:
        shape = []
        for dim in value.type.tensor_type.shape.dim:
            shape.append(int(dim.dim_value) if dim.dim_value else 0)
        inputs.append({"name": value.name, "shape": shape})
    outputs = [value.name for value in model.graph.output]
    return inputs, outputs


def _extract_default_joint_angles(metadata: dict) -> list[float]:
    exp_cfg = metadata.get("experiment_config", {})
    robot_cfg = exp_cfg.get("robot", {})
    init_state = robot_cfg.get("init_state", {})
    dof_names = list(metadata["dof_names"])
    default_joint_angles = init_state.get("default_joint_angles", {})
    return [float(default_joint_angles.get(name, 0.0)) for name in dof_names]


def _extract_robot_init_state(metadata: dict) -> dict:
    exp_cfg = metadata.get("experiment_config", {})
    robot_cfg = exp_cfg.get("robot", {})
    init_state = robot_cfg.get("init_state", {})
    return {
        "pos": list(init_state.get("pos", [0.0, 0.0, 0.76])),
        "rot": list(init_state.get("rot", [0.0, 0.0, 0.0, 1.0])),
        "lin_vel": list(init_state.get("lin_vel", [0.0, 0.0, 0.0])),
        "ang_vel": list(init_state.get("ang_vel", [0.0, 0.0, 0.0])),
        "default_joint_angles": init_state.get("default_joint_angles", {}),
    }


def _extract_motion_cfg(metadata: dict) -> dict:
    return (
        metadata.get("experiment_config", {})
        .get("command", {})
        .get("setup_terms", {})
        .get("motion_command", {})
        .get("params", {})
        .get("motion_config", {})
    )


def _extract_control_cfg(metadata: dict) -> dict:
    exp_cfg = metadata.get("experiment_config", {})
    robot_cfg = exp_cfg.get("robot", {})
    control_cfg = robot_cfg.get("control", {})
    simulator_cfg = exp_cfg.get("simulator", {}).get("config", {}).get("sim", {})
    action_scale = float(control_cfg.get("action_scale", 1.0))
    sim_fps = int(simulator_cfg.get("fps", 200))
    control_decimation = int(simulator_cfg.get("control_decimation", 4))
    return {
        "action_scale": action_scale,
        "policy_action_scale": action_scale,
        "policy_hz": float(sim_fps) / float(control_decimation),
        "sim_fps": sim_fps,
        "control_decimation": control_decimation,
        "clip_actions_threshold": float(
            exp_cfg.get("algo", {}).get("config", {}).get("distill", {}).get("clip_actions_threshold", 100.0)
        ),
    }


def _extract_observation_cfg(metadata: dict) -> dict:
    exp_cfg = metadata.get("experiment_config", {})
    actor_group = exp_cfg.get("observation", {}).get("groups", {}).get("actor_obs", {})
    terms = actor_group.get("terms", {})
    actor_obs_terms_sorted = sorted(terms.keys()) if isinstance(terms, dict) else []
    return {
        "actor_obs_terms_sorted": actor_obs_terms_sorted,
        "actor_obs_history_length": int(actor_group.get("history_length", 1)),
        "actor_obs_concatenate": bool(actor_group.get("concatenate", True)),
    }


def _extract_terrain_cfg(metadata: dict) -> dict:
    terrain_term = metadata.get("experiment_config", {}).get("terrain", {}).get("terrain_term", {})
    if not isinstance(terrain_term, dict):
        terrain_term = {}
    return {
        "name": str(terrain_term.get("name", "floor")),
        "static_friction": float(terrain_term.get("static_friction", 1.0)),
        "dynamic_friction": float(terrain_term.get("dynamic_friction", terrain_term.get("static_friction", 1.0))),
        "restitution": float(terrain_term.get("restitution", 0.0)),
    }


def _transition_step_counts(metadata: dict) -> tuple[int, int]:
    """Return only digest-authenticated transitions actually used in training.

    Raw MotionConfig flags are requests, not an effective timeline: global
    multi-clip training intentionally ignores the requested append and may
    disable a requested prepend when the runtime implementation is unavailable.
    ``patch_model`` uses the same authenticated contract for policy reference
    tensors, so the web-demo object tracks and frame count must use it too.
    """

    settings = effective_motion_transition_settings_from_metadata(metadata)
    return int(settings["prepend"]["steps"]), int(settings["append"]["steps"])


def _quat_wxyz_to_rpy(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = [float(value) for value in np.asarray(quat_wxyz, dtype=np.float64)]
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _quat_from_rpy_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )


def _repeat_endpoints(sequence: np.ndarray | None, prepend_steps: int, append_steps: int) -> np.ndarray | None:
    if sequence is None:
        return None
    arr = np.asarray(sequence, dtype=np.float32)
    parts: list[np.ndarray] = []
    if prepend_steps > 0:
        parts.append(np.repeat(arr[:1], prepend_steps, axis=0))
    parts.append(arr)
    if append_steps > 0:
        parts.append(np.repeat(arr[-1:], append_steps, axis=0))
    return np.concatenate(parts, axis=0)


def _run_onnx_bootstrap(patched_model_path: Path, obs_dim: int) -> dict[str, np.ndarray]:
    session = onnxruntime.InferenceSession(str(patched_model_path))
    outputs = session.run(
        None,
        {
            "obs": np.zeros((1, obs_dim), dtype=np.float32),
            "time_step": np.zeros((1, 1), dtype=np.float32),
        },
    )
    output_names = [item.name for item in session.get_outputs()]
    output_map = {name: value for name, value in zip(output_names, outputs, strict=False)}
    return {
        "joint_pos": np.asarray(output_map["joint_pos"], dtype=np.float32).reshape(-1),
        "joint_vel": np.asarray(output_map["joint_vel"], dtype=np.float32).reshape(-1),
        "ref_pos_xyz": np.asarray(output_map["ref_pos_xyz"], dtype=np.float32).reshape(-1),
        "ref_quat_xyzw": np.asarray(output_map["ref_quat_xyzw"], dtype=np.float32).reshape(-1),
    }


def _extract_effort_limits(scene_xml_path: Path) -> dict[str, float]:
    root = ET.parse(scene_xml_path).getroot()
    actuator_root = root.find("actuator")
    if actuator_root is None:
        return {}

    limits: dict[str, float] = {}
    for actuator in actuator_root:
        joint_name = actuator.attrib.get("joint")
        ctrlrange = actuator.attrib.get("ctrlrange")
        if not joint_name or not ctrlrange:
            continue
        parts = [float(value) for value in ctrlrange.split()]
        if len(parts) != 2:
            continue
        limits[joint_name] = max(abs(parts[0]), abs(parts[1]))
    return limits


def _resolve_policy_action_scales(
    *,
    dof_names: list[str],
    kp: list[float],
    base_action_scale: float,
    effort_limits: dict[str, float],
) -> list[float]:
    scales: list[float] = []
    for joint_name, stiffness in zip(dof_names, kp, strict=False):
        effort = float(effort_limits.get(joint_name, 0.0))
        stiffness = float(stiffness)
        if effort > 0.0 and stiffness > 0.0:
            scales.append(base_action_scale * effort / stiffness)
        else:
            scales.append(base_action_scale)
    return scales


def _read_motion_summary(
    motion_path: Path,
    dof_names: list[str],
    ref_body_name: str,
    *,
    metadata: dict,
    patched_model_path: Path,
    obs_dim: int,
    apply_training_motion_transitions: bool,
) -> dict:
    motion_payload = read_stable_regular_file_bytes(
        motion_path,
        label="Demo motion source",
    )
    with np.load(io.BytesIO(motion_payload), allow_pickle=False) as data:
        body_names = _decode_names(np.asarray(data["body_names"]))
        joint_names = _decode_names(np.asarray(data["joint_names"]))
        root_idx = _resolve_root_body_index(body_names)
        ref_idx = body_names.index(ref_body_name)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float32) if "object_pos_w" in data else None
        object_quat_w = np.asarray(data["object_quat_w"], dtype=np.float32) if "object_quat_w" in data else None
        object_lin_vel_w = np.asarray(data["object_lin_vel_w"], dtype=np.float32) if "object_lin_vel_w" in data else None
        object_ang_vel_w = np.asarray(data["object_ang_vel_w"], dtype=np.float32) if "object_ang_vel_w" in data else None
        raw_object_size = None
        for key in ("object_size", "box_size", "object_scale", "box_scale"):
            if key in data:
                raw_object_size = np.asarray(data[key], dtype=np.float32)
                break
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    timeline_contract = embedded_motion_timeline_contract_from_metadata(
        metadata,
        required=True,
    )
    assert timeline_contract is not None
    if hashlib.sha256(motion_payload).hexdigest() != timeline_contract["source_motion_sha256"]:
        raise RuntimeError(
            "Motion source changed between ONNX patching and demo-config materialization."
        )
    expected_materialization = (
        "effective_training_timeline"
        if apply_training_motion_transitions
        else "raw_unsafe_diagnostic"
    )
    if timeline_contract["materialization"] != expected_materialization:
        raise RuntimeError(
            "Demo-config transition mode diverged from patched ONNX timeline provenance."
        )

    if joint_pos.shape[1] == len(joint_names) + 7:
        joint_pos = joint_pos[:, 7:]
    if joint_vel.shape[1] == len(joint_names) + 6:
        joint_vel = joint_vel[:, 6:]

    joint_indices = [joint_names.index(name) for name in dof_names]
    joint_pos = joint_pos[:, joint_indices]
    joint_vel = joint_vel[:, joint_indices]
    object_size = _normalize_object_size_array(raw_object_size, joint_pos.shape[0])
    bootstrap = _run_onnx_bootstrap(patched_model_path, obs_dim)

    prepend_steps = 0
    append_steps = 0
    if apply_training_motion_transitions:
        prepend_steps, append_steps = _transition_step_counts(metadata)

    init_state = _extract_robot_init_state(metadata)
    if prepend_steps > 0 and init_state:
        init_root_pos = np.asarray(init_state.get("pos", [0.0, 0.0, body_pos_w[0, root_idx, 2]]), dtype=np.float32)
        init_root_rot_xyzw = np.asarray(init_state.get("rot", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)
        init_root_quat_wxyz = np.asarray(
            [init_root_rot_xyzw[3], init_root_rot_xyzw[0], init_root_rot_xyzw[1], init_root_rot_xyzw[2]],
            dtype=np.float32,
        )
        init_roll, init_pitch, _ = _quat_wxyz_to_rpy(init_root_quat_wxyz)
        _, _, motion_yaw = _quat_wxyz_to_rpy(body_quat_w[0, root_idx, :])
        initial_root_pos_w = np.asarray([body_pos_w[0, root_idx, 0], body_pos_w[0, root_idx, 1], init_root_pos[2]], dtype=np.float32)
        initial_root_quat_wxyz = _quat_from_rpy_wxyz(init_roll, init_pitch, motion_yaw)
        initial_root_lin_vel_w = np.asarray(init_state.get("lin_vel", [0.0, 0.0, 0.0]), dtype=np.float32)
        initial_root_ang_vel_w = np.asarray(init_state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=np.float32)
    else:
        initial_root_pos_w = body_pos_w[0, root_idx, :].astype(np.float32, copy=True)
        initial_root_quat_wxyz = body_quat_w[0, root_idx, :].astype(np.float32, copy=True)
        initial_root_lin_vel_w = body_lin_vel_w[0, root_idx, :].astype(np.float32, copy=True)
        initial_root_ang_vel_w = body_ang_vel_w[0, root_idx, :].astype(np.float32, copy=True)

    object_pos_aug = _repeat_endpoints(object_pos_w, prepend_steps, append_steps)
    object_quat_aug = _repeat_endpoints(object_quat_w, prepend_steps, append_steps)
    object_lin_vel_aug = _repeat_endpoints(object_lin_vel_w, prepend_steps, append_steps)
    object_ang_vel_aug = _repeat_endpoints(object_ang_vel_w, prepend_steps, append_steps)
    object_size_aug = _repeat_endpoints(object_size, prepend_steps, append_steps)
    frame_count = int(joint_pos.shape[0] + prepend_steps + append_steps)
    if frame_count != int(timeline_contract["embedded_frame_count"]):
        raise RuntimeError(
            "Demo-config frame count diverged from patched ONNX timeline provenance."
        )

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": float(frame_count / fps) if fps > 0.0 else 0.0,
        "transition_prepend_steps": prepend_steps,
        "transition_append_steps": append_steps,
        "initial_root_pos_w": initial_root_pos_w.astype(np.float32).tolist(),
        "initial_root_quat_wxyz": initial_root_quat_wxyz.astype(np.float32).tolist(),
        "initial_root_lin_vel_w": initial_root_lin_vel_w.astype(np.float32).tolist(),
        "initial_root_ang_vel_w": initial_root_ang_vel_w.astype(np.float32).tolist(),
        # MuJoCo split sim initializes robot DOFs from the raw clip frame, while the
        # policy-side motion command / ref outputs come from an ONNX zero-obs bootstrap.
        "reset_joint_pos": joint_pos[0].astype(np.float32).tolist(),
        "reset_joint_vel": joint_vel[0].astype(np.float32).tolist(),
        "initial_ref_pos_w": bootstrap["ref_pos_xyz"].astype(np.float32).tolist(),
        "initial_ref_quat_wxyz": np.asarray(
            [
                bootstrap["ref_quat_xyzw"][3],
                bootstrap["ref_quat_xyzw"][0],
                bootstrap["ref_quat_xyzw"][1],
                bootstrap["ref_quat_xyzw"][2],
            ],
            dtype=np.float32,
        ).tolist(),
        "initial_joint_pos": bootstrap["joint_pos"].astype(np.float32).tolist(),
        "initial_joint_vel": bootstrap["joint_vel"].astype(np.float32).tolist(),
        "object_pos_w": object_pos_aug.astype(np.float32).tolist() if object_pos_aug is not None else None,
        "object_quat_wxyz": object_quat_aug.astype(np.float32).tolist() if object_quat_aug is not None else None,
        "object_lin_vel_w": object_lin_vel_aug.astype(np.float32).tolist() if object_lin_vel_aug is not None else None,
        "object_ang_vel_w": object_ang_vel_aug.astype(np.float32).tolist() if object_ang_vel_aug is not None else None,
        "object_size": object_size_aug.astype(np.float32).tolist() if object_size_aug is not None else None,
        "initial_object_pos_w": object_pos_aug[0].astype(np.float32).tolist() if object_pos_aug is not None else None,
        "initial_object_quat_wxyz": object_quat_aug[0].astype(np.float32).tolist() if object_quat_aug is not None else None,
        "initial_object_lin_vel_w": object_lin_vel_aug[0].astype(np.float32).tolist() if object_lin_vel_aug is not None else None,
        "initial_object_ang_vel_w": object_ang_vel_aug[0].astype(np.float32).tolist() if object_ang_vel_aug is not None else None,
        "initial_object_size": object_size_aug[0].astype(np.float32).tolist() if object_size_aug is not None else [1.0, 1.0, 1.0],
    }


def _slugify(value: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "-", value)
    safe = safe.strip("._-")
    return safe or "clip"


def _bundle_id_for_motion_path(motion_path: Path, motion_root: Path | None) -> str:
    if motion_root is not None:
        try:
            rel = motion_path.resolve().relative_to(motion_root.resolve()).with_suffix("")
            return _slugify("__".join(rel.parts))
        except ValueError:
            pass
    return _slugify(motion_path.stem)


def _ordered_motion_paths(
    *,
    motion_dir: Path,
    motion_glob: str,
    recursive: bool,
    max_clips: int,
    preferred_clip_stem: str | None,
) -> list[Path]:
    iterator = motion_dir.rglob(motion_glob) if recursive else motion_dir.glob(motion_glob)
    paths = sorted(path.resolve() for path in iterator if path.is_file())
    if preferred_clip_stem:
        preferred = [path for path in paths if path.stem == preferred_clip_stem]
        non_preferred = [path for path in paths if path.stem != preferred_clip_stem]
        paths = preferred + non_preferred
    if max_clips > 0:
        paths = paths[:max_clips]
    return paths


def _collect_motion_paths(args: argparse.Namespace) -> tuple[list[Path], Path | None]:
    if args.motion_file is not None:
        motion_file = args.motion_file.expanduser().resolve()
        if not motion_file.is_file():
            raise FileNotFoundError(motion_file)
        return [motion_file], motion_file.parent

    if args.motion_dir is None:
        motion_file = DEFAULT_MOTION.expanduser().resolve()
        if not motion_file.is_file():
            raise FileNotFoundError(motion_file)
        return [motion_file], motion_file.parent

    motion_dir = args.motion_dir.expanduser().resolve()
    if not motion_dir.is_dir():
        raise FileNotFoundError(motion_dir)
    motion_paths = _ordered_motion_paths(
        motion_dir=motion_dir,
        motion_glob=args.motion_glob,
        recursive=args.recursive,
        max_clips=args.max_clips,
        preferred_clip_stem=args.preferred_clip_stem,
    )
    if not motion_paths:
        raise RuntimeError(f"No motion clips matched {args.motion_glob!r} under {motion_dir}")
    return motion_paths, motion_dir


def _write_shared_scene_assets(
    output_dir: Path,
    model_path: Path,
) -> Path:
    from holosoma.config_values import robot as robot_values
    from holosoma.config_values import simulator as sim_values
    from holosoma.simulator.mujoco.scene_manager import MujocoSceneManager

    metadata = _read_onnx_metadata(model_path)
    control_cfg = _extract_control_cfg(metadata)
    terrain_cfg = _extract_terrain_cfg(metadata)
    simulator_cfg = dataclass_replace(
        sim_values.mujoco.config,
        sim=dataclass_replace(sim_values.mujoco.config.sim, fps=int(control_cfg["sim_fps"])),
        robot_mjcf_filter=dataclass_replace(sim_values.mujoco.config.robot_mjcf_filter, enable=True),
    )

    class _PlaneTerrain:
        name = str(terrain_cfg["name"])
        mesh_type = "plane"
        static_friction = float(terrain_cfg["static_friction"])
        dynamic_friction = float(terrain_cfg["dynamic_friction"])
        restitution = float(terrain_cfg["restitution"])

    output_assets_dir = output_dir / "assets"
    output_assets_dir.mkdir(parents=True, exist_ok=True)

    robot_cfg = dataclass_replace(
        robot_values.g1_29dof_w_object,
        object=dataclass_replace(
            robot_values.g1_29dof_w_object.object,
            enabled=True,
            object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf",
            mujoco_use_training_urdf_scene=True,
            mujoco_object_contact_body_name_markers=["wrist", "hand"],
            mujoco_add_default_actuators=True,
            mujoco_copy_joint_defaults_from_robot_xml=True,
            mujoco_copy_tendons_from_robot_xml=True,
            mujoco_copy_collision_geoms_from_robot_xml=True,
            mujoco_copy_contact_pairs_from_robot_xml=True,
        ),
    )

    scene_manager = MujocoSceneManager(simulator_cfg)
    terrain = _PlaneTerrain()
    scene_manager.add_materials()
    scene_manager.add_lighting()
    scene_manager.add_terrain(terrain, num_envs=1)
    scene_manager.add_robot(terrain, robot_cfg, xml_filter=simulator_cfg.robot_mjcf_filter, prefix="")

    scene_xml_path = output_dir / "scene.xml"
    composite_scene_path = Path(scene_manager.robot_model_path).resolve()
    composite_scene_root = ET.parse(composite_scene_path).getroot()
    composite_meshdir_raw = composite_scene_root.find("compiler")
    composite_meshdir = (
        composite_meshdir_raw.attrib.get("meshdir", "") if composite_meshdir_raw is not None else ""
    ).replace("\\", "/")
    composite_mesh_root = (composite_scene_path.parent / composite_meshdir).resolve()

    scene_root = ET.fromstring(scene_manager.world_spec.to_xml())
    compiler = scene_root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(scene_root, "compiler")
    compiler.set("meshdir", "assets")

    def _remove_empty_default_nodes(node: ET.Element) -> None:
        for child in list(node):
            _remove_empty_default_nodes(child)
            if child.tag != "default":
                continue
            has_text = bool((child.text or "").strip())
            if len(child) == 0 and not child.attrib and not has_text:
                node.remove(child)

    _remove_empty_default_nodes(scene_root)
    top_default = scene_root.find("default")
    if top_default is not None and len(top_default) == 0 and not top_default.attrib and not (top_default.text or "").strip():
        scene_root.remove(top_default)

    staged_mesh_sources: dict[str, Path] = {}

    def _resolve_source_mesh(raw_path: str) -> Path:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (composite_mesh_root / raw_path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Generated MuJoCo scene referenced missing mesh: {raw_path} -> {resolved}")
        return resolved

    def _stage_mesh_path(raw_path: str) -> str:
        candidate = Path(raw_path)
        if candidate.is_absolute():
            rel_path = Path(candidate.name)
        else:
            rel_parts = [part for part in candidate.parts if part not in ("", ".", "..")]
            rel_path = Path(*rel_parts) if rel_parts else Path(candidate.name)
        if not rel_path.suffix:
            rel_path = rel_path.with_suffix(Path(raw_path).suffix)
        rel_posix = rel_path.as_posix()
        source_path = _resolve_source_mesh(raw_path)
        previous_source = staged_mesh_sources.get(rel_posix)
        if previous_source is not None and previous_source != source_path:
            rel_posix = f"{_slugify(source_path.parent.name)}__{source_path.name}"
        staged_mesh_sources[rel_posix] = source_path
        return rel_posix

    for mesh in scene_root.findall(".//mesh[@file]"):
        raw_path = mesh.attrib["file"]
        staged_relpath = _stage_mesh_path(raw_path)
        mesh.set("file", staged_relpath)

    for rel_path, source_path in staged_mesh_sources.items():
        destination = output_assets_dir / rel_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)

    scene_tree = ET.ElementTree(scene_root)
    ET.indent(scene_tree, space="  ")
    scene_tree.write(scene_xml_path, encoding="utf-8", xml_declaration=False)
    return scene_xml_path


def _stage_clip_bundle(
    *,
    model_path: Path,
    motion_path: Path,
    bundle_dir: Path,
    scene_path: Path,
    effort_limits: dict[str, float],
    unsafe_skip_training_motion_transitions: bool,
) -> tuple[dict, dict]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    patched_model_path = bundle_dir / "policy.onnx"
    apply_training_motion_transitions = not unsafe_skip_training_motion_transitions
    patch_model(
        model_path.resolve(),
        motion_path.resolve(),
        patched_model_path.resolve(),
        apply_training_motion_transitions=apply_training_motion_transitions,
        unsafe_allow_raw_motion_timeline=unsafe_skip_training_motion_transitions,
    )

    metadata = _read_onnx_metadata(patched_model_path)
    onnx_inputs, onnx_outputs = _read_onnx_io(patched_model_path)
    dof_names = list(metadata["dof_names"])
    motion_cfg = _extract_motion_cfg(metadata)
    observation_cfg = _extract_observation_cfg(metadata)
    ref_body_names = motion_cfg.get("body_name_ref", ["torso_link"])
    ref_body_name = ref_body_names[0] if isinstance(ref_body_names, list) and ref_body_names else "torso_link"
    obs_shape = next((entry["shape"] for entry in onnx_inputs if entry["name"] in {"obs", "actor_obs"}), None)
    motion_summary = _read_motion_summary(
        motion_path.resolve(),
        dof_names,
        ref_body_name,
        metadata=metadata,
        patched_model_path=patched_model_path,
        obs_dim=int(obs_shape[1]) if obs_shape and len(obs_shape) > 1 else 0,
        apply_training_motion_transitions=apply_training_motion_transitions,
    )
    control_cfg = _extract_control_cfg(metadata)
    policy_action_scales = _resolve_policy_action_scales(
        dof_names=dof_names,
        kp=list(metadata["kp"]),
        base_action_scale=float(control_cfg["action_scale"]),
        effort_limits=effort_limits,
    )

    clip_config = {
        "model_path": str(patched_model_path.relative_to(bundle_dir.parents[1])),
        "scene_path": str(scene_path.relative_to(bundle_dir.parents[1])),
        "motion_file": str(motion_path.resolve()),
        "checkpoint": {
            "source_model_path": str(model_path.resolve()),
            "patched_model_path": str(patched_model_path.resolve()),
            "model_name": model_path.name,
            "wandb_run_path": metadata.get("wandb_run_path"),
            "iteration": metadata.get("iteration"),
        },
        "clip_name": motion_path.stem,
        "dof_names": dof_names,
        "default_dof_angles": _extract_default_joint_angles(metadata),
        "robot_init_state": _extract_robot_init_state(metadata),
        "kp": list(metadata["kp"]),
        "kd": list(metadata["kd"]),
        "reset_mode_default": "demo_raw",
        "reset_modes": ["demo_raw", "isaac_training_default_pose"],
        "ref_body_name": ref_body_name,
        "object_body_name": "largebox_link",
        "motion_alignment_enabled": bool(motion_cfg.get("align_motion_to_init_yaw", False)),
        # Match the split MuJoCo tracking launcher defaults in mj_track.sh.
        "use_root_reference_at_clip_start": True,
        "prefer_sim_ref_from_sim_state": True,
        "apply_training_motion_transitions": bool(apply_training_motion_transitions),
        "observation": observation_cfg,
        "control": {
            **control_cfg,
            "effort_limits": effort_limits,
            "policy_action_scales": policy_action_scales,
        },
        "onnx": {
            "inputs": onnx_inputs,
            "outputs": onnx_outputs,
            "obs_dim": int(obs_shape[1]) if obs_shape and len(obs_shape) > 1 else None,
        },
        "motion": motion_summary,
    }
    config_path = bundle_dir / "demo-config.json"
    config_path.write_text(json.dumps(clip_config, indent=2), encoding="utf-8")

    clip_manifest_entry = {
        "id": bundle_dir.name,
        "label": motion_path.stem,
        "config_path": str(config_path.relative_to(bundle_dir.parents[1])),
        "motion_file": str(motion_path.resolve()),
        "frame_count": motion_summary["frame_count"],
        "duration_s": motion_summary["duration_s"],
    }
    return clip_config, clip_manifest_entry


def stage_assets(
    *,
    model_path: Path,
    motion_paths: list[Path],
    motion_root: Path | None,
    output_dir: Path,
    unsafe_skip_training_motion_transitions: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    scene_xml_path = _write_shared_scene_assets(output_dir, model_path)
    effort_limits = _extract_effort_limits(scene_xml_path)

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_clips: list[dict] = []
    bundle_ids: set[str] = set()
    default_clip_id: str | None = None
    default_label: str | None = None

    for motion_path in motion_paths:
        base_id = _bundle_id_for_motion_path(motion_path, motion_root)
        bundle_id = base_id
        suffix = 2
        while bundle_id in bundle_ids:
            bundle_id = f"{base_id}-{suffix}"
            suffix += 1
        bundle_ids.add(bundle_id)

        _clip_config, clip_manifest_entry = _stage_clip_bundle(
            model_path=model_path,
            motion_path=motion_path,
            bundle_dir=clips_dir / bundle_id,
            scene_path=scene_xml_path,
            effort_limits=effort_limits,
            unsafe_skip_training_motion_transitions=unsafe_skip_training_motion_transitions,
        )
        manifest_clips.append(clip_manifest_entry)
        if default_clip_id is None:
            default_clip_id = clip_manifest_entry["id"]
            default_label = clip_manifest_entry["label"]

    manifest = {
        "version": 1,
        "scene_path": str(scene_xml_path.relative_to(output_dir)),
        "clip_count": len(manifest_clips),
        "default_clip_id": default_clip_id,
        "default_clip_label": default_label,
        "clips": manifest_clips,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    motion_group = parser.add_mutually_exclusive_group()
    motion_group.add_argument("--motion-file", type=Path, default=None)
    motion_group.add_argument("--motion-dir", type=Path, default=None)
    parser.add_argument("--motion-glob", type=str, default="*_w_obj.npz")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS)
    parser.add_argument("--preferred-clip-stem", type=str, default=DEFAULT_MOTION.stem)
    parser.add_argument(
        "--unsafe-skip-training-motion-transitions",
        dest="unsafe_skip_training_motion_transitions",
        action="store_true",
        help=(
            "Build an explicitly non-scientific diagnostic bundle whose embedded motion skips "
            "the authenticated effective training transition."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "demo-assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_paths, motion_root = _collect_motion_paths(args)
    manifest = stage_assets(
        model_path=args.model_path.expanduser().resolve(),
        motion_paths=motion_paths,
        motion_root=motion_root,
        output_dir=args.output_dir.expanduser().resolve(),
        unsafe_skip_training_motion_transitions=bool(
            args.unsafe_skip_training_motion_transitions
        ),
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "clip_count": manifest["clip_count"]}, indent=2))


if __name__ == "__main__":
    main()
