#!/usr/bin/env python3
"""Stage MuJoCo web-demo assets for one or more w-object motion clips."""

from __future__ import annotations

import argparse
from dataclasses import replace as dataclass_replace
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma"))
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_inference"))

from holosoma_inference.tools.patch_motion_onnx import patch_model


DEFAULT_MODEL = Path(
    "/data/logs_new/boxer/20260415_014803-g1_29dof_wbt_w_object_distill_box_perception_sparse_root_cmd_access_to_depth-locomotion/model_03999.onnx"
)
DEFAULT_MOTION = Path(
    "/home/ubuntu/FAR/holosoma/outputs/motion_bank_success_box_0_92_0p3/box_74.npz"
)
DEFAULT_MOTION_DIR = Path(
    "/home/ubuntu/FAR/holosoma/outputs/motion_bank_success_box_0_92_0p3"
)
DEFAULT_SCENE = Path("/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml")
DEFAULT_ACTUATOR_SOURCE = Path("/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.xml")
DEFAULT_MAX_CLIPS = 0
DEFAULT_OBJECT_MASS_KG = 1.4
MUJOCO_OBJECT_CONDIM = 6
MUJOCO_OBJECT_SLIDE_FRICTION_MIN = 0.6
MUJOCO_OBJECT_SPIN_FRICTION = 0.02
MUJOCO_OBJECT_ROLL_FRICTION = 0.005
MUJOCO_RUBBER_HAND_SLIDE_FRICTION = 0.8
MUJOCO_RUBBER_HAND_SPIN_FRICTION = 0.02
MUJOCO_RUBBER_HAND_ROLL_FRICTION = 0.005


def _decode_names(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for item in values.tolist():
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            decoded.append(item.decode("utf-8"))
        else:
            decoded.append(str(item))
    return decoded


def _scalar_str(value: np.ndarray | object) -> str:
    arr = np.asarray(value)
    item = arr.reshape(-1)[0] if arr.ndim > 0 else arr.item()
    if isinstance(item, (bytes, bytearray, np.bytes_)):
        return item.decode("utf-8")
    return str(item)


def _mj_float(value: float) -> str:
    return f"{float(value):.9g}"


def _resolve_root_body_index(body_names: list[str]) -> int:
    for candidate in ("pelvis", "pelvis_link", "base_link", "torso_link"):
        if candidate in body_names:
            return body_names.index(candidate)
    for idx, name in enumerate(body_names):
        if name.lower() != "world":
            return idx
    return 0


def _first_object_size(raw: np.ndarray | None) -> list[float] | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 0:
        value = float(arr)
        return [value, value, value]
    if arr.ndim == 1:
        if arr.shape[0] == 3:
            return arr.astype(np.float32).tolist()
        if arr.shape[0] == 1:
            value = float(arr[0])
            return [value, value, value]
    if arr.ndim >= 2:
        first = np.asarray(arr[0], dtype=np.float32).reshape(-1)
        if first.shape[0] == 3:
            return first.tolist()
        if first.shape[0] == 1:
            value = float(first[0])
            return [value, value, value]
    return None


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
    }


def _extract_perception_cfg(metadata: dict, perception_dim: int | None = None) -> dict:
    exp_cfg = metadata.get("experiment_config", {})
    perception = exp_cfg.get("perception", {})
    camera_width = int(perception.get("camera_width", 17))
    camera_height = int(perception.get("camera_height", 17))
    resize = perception.get("camera_warp_resize")
    if isinstance(resize, list) and len(resize) == 2:
        obs_height = int(resize[0])
        obs_width = int(resize[1])
    else:
        crop_top = int(perception.get("camera_warp_crop_top", 0) or 0)
        crop_bottom = int(perception.get("camera_warp_crop_bottom", 0) or 0)
        crop_left = int(perception.get("camera_warp_crop_left", 0) or 0)
        crop_right = int(perception.get("camera_warp_crop_right", 0) or 0)
        obs_height = max(1, camera_height - crop_top - crop_bottom)
        obs_width = max(1, camera_width - crop_left - crop_right)
    if perception_dim is not None and obs_height * obs_width != int(perception_dim):
        obs_height = int(round(float(perception_dim) ** 0.5))
        while obs_height > 1 and int(perception_dim) % obs_height != 0:
            obs_height -= 1
        obs_width = int(perception_dim) // max(1, obs_height)
    return {
        "camera_strict_warp": bool(perception.get("camera_strict_warp", True)),
        "camera_width": camera_width,
        "camera_height": camera_height,
        "observation_width": obs_width,
        "observation_height": obs_height,
        "observation_dim": int(perception_dim or (obs_width * obs_height)),
        "camera_vfov_deg": float(perception.get("camera_vfov_deg", 58.6)),
        "camera_hfov_deg": float(perception.get("camera_hfov_deg", 89.5)),
        "camera_near": float(perception.get("camera_near", 0.001)),
        "camera_far": float(perception.get("camera_far", 3.0)),
        "max_distance": float(perception.get("max_distance", 3.0)),
        "camera_warp_normalize": bool(perception.get("camera_warp_normalize", False)),
        "camera_warp_preprocess": bool(perception.get("camera_warp_preprocess", False)),
        "camera_warp_crop_top": int(perception.get("camera_warp_crop_top", 0) or 0),
        "camera_warp_crop_bottom": int(perception.get("camera_warp_crop_bottom", 0) or 0),
        "camera_warp_crop_left": int(perception.get("camera_warp_crop_left", 0) or 0),
        "camera_warp_crop_right": int(perception.get("camera_warp_crop_right", 0) or 0),
        "camera_warp_min_valid_depth": float(perception.get("camera_warp_min_valid_depth", 0.15)),
        "web_depth_include_visual_meshes": bool(perception.get("web_depth_include_visual_meshes", True)),
        "web_depth_mesh_mode": str(perception.get("web_depth_mesh_mode", "bounds")),
        "camera_body_name": str(perception.get("camera_body_name", "torso_link")),
        "sensor_offset": list(perception.get("sensor_offset", [0.01, 0.01, 0.44])),
        "camera_mount_quat": list(
            perception.get("camera_mount_quat", [0.00644801, 0.23350163, 0.00644801, 0.97231365])
        ),
        "camera_frame_quat": list(perception.get("camera_frame_quat", [-0.5, 0.5, -0.5, 0.5])),
    }


def _extract_observation_cfg(metadata: dict, *, obs_dim: int) -> dict:
    dof_dim = len(metadata["dof_names"])
    term_dims = {
        "sparse_target_root_trajectory_command": 3,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "dof_pos": dof_dim,
        "dof_vel": dof_dim,
        "actions": dof_dim,
    }
    groups_raw = (
        metadata.get("experiment_config", {})
        .get("observation", {})
        .get("groups", {})
    )
    if not isinstance(groups_raw, dict):
        groups_raw = {}

    def make_group(group_name: str) -> dict | None:
        raw = groups_raw.get(group_name)
        if not isinstance(raw, dict):
            return None
        terms_raw = raw.get("terms", {})
        if not isinstance(terms_raw, dict):
            return None
        terms = []
        for term_name in sorted(terms_raw.keys()):
            if term_name not in term_dims:
                return None
            term_raw = terms_raw.get(term_name, {})
            scale = term_raw.get("scale", 1.0) if isinstance(term_raw, dict) else 1.0
            terms.append({"name": term_name, "dim": int(term_dims[term_name]), "scale": float(scale)})
        return {
            "name": group_name,
            "history_length": int(raw.get("history_length", 1)),
            "terms": terms,
        }

    def total_dim(groups: list[dict]) -> int:
        return sum(
            int(group["history_length"]) * sum(int(term["dim"]) for term in group["terms"])
            for group in groups
        )

    candidates = [
        ["actor_obs_root", "actor_obs_proprio_no_linvel"],
        ["actor_obs_root", "actor_obs_proprio_no_linvel", "actor_obs_actions"],
        ["actor_obs_root", "actor_obs_proprio"],
        ["actor_obs_root", "actor_obs_proprio", "actor_obs_actions"],
        ["actor_obs"],
    ]
    for candidate in candidates:
        groups = [make_group(name) for name in candidate]
        if all(group is not None for group in groups) and total_dim(groups) == int(obs_dim):
            return {"actor_groups": groups, "obs_dim": int(obs_dim)}

    fallback_groups = [
        {
            "name": "actor_obs_root",
            "history_length": 1,
            "terms": [
                {"name": "sparse_target_root_trajectory_command", "dim": 3, "scale": 1.0},
            ],
        },
        {
            "name": "actor_obs_proprio_no_linvel",
            "history_length": 5,
            "terms": [
                {"name": "base_ang_vel", "dim": 3, "scale": 1.0},
                {"name": "dof_pos", "dim": dof_dim, "scale": 1.0},
                {"name": "dof_vel", "dim": dof_dim, "scale": 1.0},
            ],
        },
    ]
    fallback_dim = total_dim(fallback_groups)
    if fallback_dim != int(obs_dim):
        raise RuntimeError(
            f"Unable to derive browser actor observation groups for ONNX obs dim {obs_dim}; "
            f"fallback sparse-root dim would be {fallback_dim}."
        )
    return {"actor_groups": fallback_groups, "obs_dim": int(obs_dim)}


def _extract_control_cfg(metadata: dict) -> dict:
    exp_cfg = metadata.get("experiment_config", {})
    robot_cfg = exp_cfg.get("robot", {})
    control_cfg = robot_cfg.get("control", {})
    simulator_cfg = exp_cfg.get("simulator", {}).get("config", {}).get("sim", {})
    action_scale = float(control_cfg.get("action_scale", 0.25))
    action_clip_value = float(control_cfg.get("action_clip_value", 100.0))
    sim_fps = int(simulator_cfg.get("fps", 200))
    control_decimation = int(simulator_cfg.get("control_decimation", 4))
    return {
        "action_scale": action_scale,
        "policy_action_scale": action_scale,
        "policy_hz": float(sim_fps) / float(control_decimation),
        "sim_fps": sim_fps,
        "control_decimation": control_decimation,
        "clip_actions_threshold": action_clip_value,
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


def _extract_joint_physics_cfg(metadata: dict) -> dict[str, dict[str, float]]:
    robot_cfg = metadata.get("experiment_config", {}).get("robot", {})
    dof_names = list(robot_cfg.get("dof_names") or metadata.get("dof_names") or [])
    effort = list(robot_cfg.get("dof_effort_limit_list") or [])
    armature = list(robot_cfg.get("dof_armature_list") or [])
    friction = list(robot_cfg.get("dof_joint_friction_list") or [])

    physics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(dof_names):
        values: dict[str, float] = {}
        if index < len(effort):
            values["effort"] = float(effort[index])
        if index < len(armature):
            values["armature"] = float(armature[index])
        if index < len(friction):
            values["frictionloss"] = float(friction[index])
        physics[str(name)] = values
    return physics


def _read_object_urdf_physics(object_urdf_path: str | Path) -> dict[str, object]:
    root = ET.parse(Path(object_urdf_path)).getroot()
    link = root.find("link")
    if link is None:
        raise RuntimeError(f"Object URDF has no link: {object_urdf_path}")

    def _float_attr(element: ET.Element | None, name: str, default: float) -> float:
        if element is None:
            return default
        raw = element.attrib.get(name)
        if raw is None:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    inertial = link.find("inertial")
    mass_el = inertial.find("mass") if inertial is not None else None
    inertia_el = inertial.find("inertia") if inertial is not None else None
    mass = _float_attr(mass_el, "value", 0.1)
    inertia = {
        "ixx": _float_attr(inertia_el, "ixx", 1.0e-4),
        "iyy": _float_attr(inertia_el, "iyy", 1.0e-4),
        "izz": _float_attr(inertia_el, "izz", 1.0e-4),
        "ixy": _float_attr(inertia_el, "ixy", 0.0),
        "ixz": _float_attr(inertia_el, "ixz", 0.0),
        "iyz": _float_attr(inertia_el, "iyz", 0.0),
    }

    contact = link.find("contact")
    root_dynamics = root.find("dynamics")
    lateral = contact.find("lateral_friction") if contact is not None else None
    rolling = contact.find("rolling_friction") if contact is not None else None
    stiffness = contact.find("stiffness") if contact is not None else None
    damping = contact.find("damping") if contact is not None else None
    restitution = contact.find("restitution") if contact is not None else None
    lateral_friction = _float_attr(lateral, "value", _float_attr(root_dynamics, "friction", 0.9))
    rolling_friction = _float_attr(rolling, "value", 0.001)

    return {
        "mass": mass,
        "inertia": inertia,
        "friction": [lateral_friction, 0.005, rolling_friction],
        "contact_stiffness": _float_attr(stiffness, "value", 0.0),
        "contact_damping": _float_attr(damping, "value", 0.0),
        "restitution": _float_attr(restitution, "value", 0.0),
    }


def _setup_term_params(metadata: dict, term_name: str) -> dict:
    term = (
        metadata.get("experiment_config", {})
        .get("randomization", {})
        .get("setup_terms", {})
        .get(term_name, {})
    )
    if not isinstance(term, dict):
        return {}
    params = term.get("params", {})
    if not isinstance(params, dict) or params.get("enabled", True) is False:
        return {}
    return params


def _range_midpoint(values: object) -> float | None:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        return None
    try:
        lo = float(values[0])
        hi = float(values[1])
    except (TypeError, ValueError):
        return None
    return 0.5 * (lo + hi)


def _scale_inertia(inertia: dict[str, float], scale: float) -> dict[str, float]:
    return {key: float(value) * float(scale) for key, value in inertia.items()}


def _resolve_distill_object_physics(
    metadata: dict,
    object_urdf_path: str | Path,
    *,
    fixed_mass_kg: float | None = DEFAULT_OBJECT_MASS_KG,
) -> dict[str, object]:
    """Pick one deterministic MuJoCo object setting from the distill training distribution."""
    base = _read_object_urdf_physics(object_urdf_path)
    base_mass = float(base["mass"])
    inertia = dict(base["inertia"])
    assert isinstance(inertia, dict)

    target_mass = base_mass
    mass_params = _setup_term_params(metadata, "randomize_object_rigid_body_mass_startup")
    mass_delta_mid = _range_midpoint(mass_params.get("mass_distribution_params"))
    if fixed_mass_kg is not None:
        target_mass = float(fixed_mass_kg)
        if target_mass <= 0.0:
            raise ValueError(f"fixed object mass must be positive, got {target_mass}")
        if base_mass > 0.0:
            inertia = _scale_inertia(inertia, target_mass / base_mass)
    elif mass_delta_mid is not None:
        # distill uses operation="add" in randomize_object_rigid_body_mass_startup.
        target_mass = base_mass + mass_delta_mid
        if base_mass > 0.0:
            inertia = _scale_inertia(inertia, target_mass / base_mass)

    inertia_params = _setup_term_params(metadata, "randomize_object_rigid_body_inertia_startup")
    inertia_ranges = inertia_params.get("inertia_distribution_params_dict", {})
    if isinstance(inertia_ranges, dict):
        for random_key, inertia_key in (
            ("Ixx", "ixx"),
            ("Iyy", "iyy"),
            ("Izz", "izz"),
            ("Ixy", "ixy"),
            ("Iyz", "iyz"),
            ("Ixz", "ixz"),
        ):
            midpoint = _range_midpoint(inertia_ranges.get(random_key))
            if midpoint is not None and inertia_key in inertia:
                inertia[inertia_key] = float(inertia[inertia_key]) * midpoint

    material_params = _setup_term_params(metadata, "randomize_object_rigid_body_material_startup")
    dynamic_mid = _range_midpoint(material_params.get("dynamic_friction_range"))
    static_mid = _range_midpoint(material_params.get("static_friction_range"))
    if dynamic_mid is not None:
        slide_friction = dynamic_mid
    elif static_mid is not None:
        slide_friction = static_mid
    else:
        slide_friction = float(base["friction"][0])  # type: ignore[index]
    slide_friction = max(slide_friction, MUJOCO_OBJECT_SLIDE_FRICTION_MIN)

    restitution_mid = _range_midpoint(material_params.get("restitution_range"))

    return {
        **base,
        "mass": target_mass,
        "base_mass": base_mass,
        "inertia": inertia,
        # Isaac Sim object material randomization has static/dynamic friction, but no MuJoCo-style spin/roll.
        "friction": [slide_friction, MUJOCO_OBJECT_SPIN_FRICTION, MUJOCO_OBJECT_ROLL_FRICTION],
        "restitution": float(base["restitution"]) if restitution_mid is None else restitution_mid,
        "distill_mass_delta_mid": mass_delta_mid,
        "fixed_mass_kg": fixed_mass_kg,
        "distill_material_friction_mid": slide_friction,
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


def _read_motion_summary(motion_path: Path, dof_names: list[str]) -> dict:
    with np.load(motion_path, allow_pickle=True) as data:
        body_names = _decode_names(np.asarray(data["body_names"]))
        joint_names = _decode_names(np.asarray(data["joint_names"]))
        root_idx = _resolve_root_body_index(body_names)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32) if "body_lin_vel_w" in data else None
        body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32) if "body_ang_vel_w" in data else None
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float32) if "object_pos_w" in data else None
        object_quat_w = np.asarray(data["object_quat_w"], dtype=np.float32) if "object_quat_w" in data else None
        object_lin_vel_w = np.asarray(data["object_lin_vel_w"], dtype=np.float32) if "object_lin_vel_w" in data else None
        object_ang_vel_w = np.asarray(data["object_ang_vel_w"], dtype=np.float32) if "object_ang_vel_w" in data else None
        object_size = np.asarray(data["object_size"], dtype=np.float32) if "object_size" in data else None
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    if joint_pos.shape[1] == len(joint_names) + 7:
        joint_pos = joint_pos[:, 7:]
    if joint_vel.shape[1] == len(joint_names) + 6:
        joint_vel = joint_vel[:, 6:]

    joint_indices = [joint_names.index(name) for name in dof_names]
    joint_pos = joint_pos[:, joint_indices]
    joint_vel = joint_vel[:, joint_indices]

    return {
        "fps": fps,
        "frame_count": int(body_pos_w.shape[0]),
        "duration_s": float(body_pos_w.shape[0] / fps) if fps > 0.0 else 0.0,
        "root_pos_w": body_pos_w[:, root_idx, :].astype(np.float32).tolist(),
        "root_quat_wxyz": body_quat_w[:, root_idx, :].astype(np.float32).tolist(),
        "initial_root_pos_w": body_pos_w[0, root_idx, :].astype(np.float32).tolist(),
        "initial_root_quat_wxyz": body_quat_w[0, root_idx, :].astype(np.float32).tolist(),
        "initial_root_lin_vel_w": (
            body_lin_vel_w[0, root_idx, :].astype(np.float32).tolist()
            if body_lin_vel_w is not None
            else [0.0, 0.0, 0.0]
        ),
        "initial_root_ang_vel_w": (
            body_ang_vel_w[0, root_idx, :].astype(np.float32).tolist()
            if body_ang_vel_w is not None
            else [0.0, 0.0, 0.0]
        ),
        "reset_joint_pos": joint_pos[0].astype(np.float32).tolist(),
        "reset_joint_vel": joint_vel[0].astype(np.float32).tolist(),
        "initial_joint_pos": joint_pos[0].astype(np.float32).tolist(),
        "initial_joint_vel": joint_vel[0].astype(np.float32).tolist(),
        "initial_object_pos_w": object_pos_w[0].astype(np.float32).tolist() if object_pos_w is not None else None,
        "initial_object_quat_wxyz": object_quat_w[0].astype(np.float32).tolist() if object_quat_w is not None else None,
        "initial_object_lin_vel_w": (
            object_lin_vel_w[0].astype(np.float32).tolist()
            if object_lin_vel_w is not None
            else [0.0, 0.0, 0.0]
        ),
        "initial_object_ang_vel_w": (
            object_ang_vel_w[0].astype(np.float32).tolist()
            if object_ang_vel_w is not None
            else [0.0, 0.0, 0.0]
        ),
        "initial_object_size": _first_object_size(object_size),
    }


def _read_motion_object_urdf_path(motion_path: Path) -> str:
    default_urdf = "holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
    with np.load(motion_path, allow_pickle=True) as data:
        if "object_urdf_path" not in data:
            return default_urdf
        raw = _scalar_str(data["object_urdf_path"]).strip()
    if not raw:
        return default_urdf
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    if raw.startswith("holosoma/data"):
        return raw
    return str((motion_path.parent / candidate).resolve())


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _patch_scene_xml(scene_path: Path, output_scene_path: Path, actuator_source_path: Path) -> None:
    scene_tree = ET.parse(scene_path)
    scene_root = scene_tree.getroot()

    compiler = scene_root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(scene_root, "compiler")
    compiler.set("meshdir", "assets")

    for mesh in scene_root.findall(".//mesh"):
        file_path = mesh.attrib.get("file")
        if file_path == "../../largebox/largebox.obj":
            mesh.set("file", "largebox/largebox.obj")

    if scene_root.find("actuator") is None:
        actuator_tree = ET.parse(actuator_source_path)
        actuator_root = actuator_tree.getroot().find("actuator")
        if actuator_root is None:
            raise RuntimeError(f"Actuator source is missing <actuator>: {actuator_source_path}")
        scene_root.append(actuator_root)

    ET.indent(scene_tree, space="  ")
    scene_tree.write(output_scene_path, encoding="utf-8", xml_declaration=False)


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


def _motion_sort_key(path: Path) -> tuple:
    parts: list[object] = []
    for part in re.split(r"(\d+)", path.stem):
        if part.isdigit():
            parts.append(int(part))
        elif part:
            parts.append(part)
    return tuple(parts)


def _ordered_motion_paths(
    *,
    motion_dir: Path,
    motion_glob: str,
    recursive: bool,
    max_clips: int,
    preferred_clip_stem: str | None,
) -> list[Path]:
    iterator = motion_dir.rglob(motion_glob) if recursive else motion_dir.glob(motion_glob)
    paths = sorted((path.resolve() for path in iterator if path.is_file()), key=_motion_sort_key)
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

    if args.motion_dir is None and DEFAULT_MOTION_DIR.expanduser().is_dir():
        motion_dir = DEFAULT_MOTION_DIR.expanduser().resolve()
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
    *,
    scene_path: Path,
    actuator_source_path: Path,
    output_dir: Path,
    model_path: Path,
    object_urdf_path: str,
    scene_xml_path: Path | None = None,
) -> Path:
    del scene_path, actuator_source_path

    os.environ.setdefault("HOLOSOMA_W_OBJECT_URDF", "g1/g1_29dof.urdf")
    os.environ.setdefault("HOLOSOMA_MUJOCO_LOAD_ROBOT_VISUAL_MESHES", "1")

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

    robot_cfg = dataclass_replace(
        robot_values.g1_29dof_w_object,
        object=dataclass_replace(
            robot_values.g1_29dof_w_object.object,
            object_urdf_path=object_urdf_path,
            enabled=True,
            mujoco_use_training_urdf_scene=True,
            mujoco_object_contact_body_name_markers=["rubber_hand"],
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

    output_assets_dir = output_dir / "assets"
    output_assets_dir.mkdir(parents=True, exist_ok=True)
    scene_xml_path = scene_xml_path or (output_dir / "scene.xml")
    scene_xml_path.parent.mkdir(parents=True, exist_ok=True)
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
    option = scene_root.find("option")
    if option is None:
        option = ET.SubElement(scene_root, "option")
    option.set("timestep", _mj_float(1.0 / float(control_cfg["sim_fps"])))
    option.set("gravity", "0 0 -9.81")

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
        object_urdf_parent = Path(object_urdf_path).expanduser().resolve().parent
        if candidate.is_absolute():
            candidates = [candidate.resolve()]
        else:
            candidates = [
                (composite_mesh_root / raw_path).resolve(),
                (object_urdf_parent / raw_path).resolve(),
                (object_urdf_parent / "meshes" / raw_path).resolve(),
                (composite_scene_path.parent / "meshes" / raw_path).resolve(),
                (REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "robots" / "g1" / "meshes" / raw_path).resolve(),
                (REPO_ROOT / "src" / "holosoma" / "holosoma" / "data" / "motions" / "g1_29dof" / "whole_body_tracking" / raw_path).resolve(),
                (REPO_ROOT / "src" / "holosoma_retargeting" / "models" / "largebox" / Path(raw_path).name).resolve(),
            ]
        for resolved in candidates:
            if resolved.is_file():
                return resolved
        raise FileNotFoundError(
            f"Generated MuJoCo scene referenced missing mesh: {raw_path} -> "
            f"{', '.join(str(path) for path in candidates)}"
        )

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
        mesh.set("file", _stage_mesh_path(raw_path))

    parent_map = {child: parent for parent in scene_root.iter() for child in parent}
    joint_physics = _extract_joint_physics_cfg(metadata)
    object_physics = _resolve_distill_object_physics(metadata, object_urdf_path)
    rubber_hand_geom_names: list[str] = []

    for joint in scene_root.findall(".//joint"):
        name = joint.attrib.get("name", "")
        physics = joint_physics.get(name)
        if not physics:
            continue
        if "armature" in physics:
            joint.set("armature", _mj_float(physics["armature"]))
        if "frictionloss" in physics:
            joint.set("frictionloss", _mj_float(physics["frictionloss"]))
        if "effort" in physics:
            effort = abs(float(physics["effort"]))
            joint.set("actuatorfrcrange", f"{_mj_float(-effort)} {_mj_float(effort)}")

    for actuator in scene_root.findall(".//actuator/*"):
        joint_name = actuator.attrib.get("joint") or actuator.attrib.get("name", "")
        physics = joint_physics.get(joint_name)
        if physics and "effort" in physics:
            effort = abs(float(physics["effort"]))
            actuator.set("ctrlrange", f"{_mj_float(-effort)} {_mj_float(effort)}")
            actuator.set("ctrllimited", "true")

    def _is_object_body(body: ET.Element) -> bool:
        body_name = body.attrib.get("name", "")
        return body_name.startswith("object_") or body_name in {"object", "object_baseLink"}

    def _set_object_body_inertial(body: ET.Element) -> None:
        for inertial in list(body.findall("inertial")):
            body.remove(inertial)
        inertia = object_physics["inertia"]
        assert isinstance(inertia, dict)
        inertial = ET.Element(
            "inertial",
            {
                "pos": "0 0 0",
                "mass": _mj_float(float(object_physics["mass"])),
            },
        )
        if any(abs(float(inertia[key])) > 0.0 for key in ("ixy", "ixz", "iyz")):
            inertial.set(
                "fullinertia",
                " ".join(
                    _mj_float(float(inertia[key]))
                    for key in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
                ),
            )
        else:
            inertial.set(
                "diaginertia",
                " ".join(_mj_float(float(inertia[key])) for key in ("ixx", "iyy", "izz")),
            )
        body.insert(0, inertial)

    for body in scene_root.findall(".//body"):
        if _is_object_body(body):
            _set_object_body_inertial(body)

    def _is_object_geom(geom: ET.Element) -> bool:
        name = geom.attrib.get("name", "")
        mesh = geom.attrib.get("mesh", "")
        if "largebox" in name or "object" in name or "largebox" in mesh or "object" in mesh:
            return True
        parent = parent_map.get(geom)
        while parent is not None:
            if parent.tag == "body":
                body_name = parent.attrib.get("name", "")
                if body_name.startswith("object_") or body_name in {"object", "object_baseLink"}:
                    return True
            parent = parent_map.get(parent)
        return False

    def _owning_body_name(element: ET.Element) -> str:
        parent = parent_map.get(element)
        while parent is not None:
            if parent.tag == "body":
                return parent.attrib.get("name", "")
            parent = parent_map.get(parent)
        return ""

    def _is_enabled_collision_geom(geom: ET.Element) -> bool:
        return geom.attrib.get("contype", "1") != "0" and geom.attrib.get("conaffinity", "1") != "0"

    def _format_friction(values: object) -> str:
        return " ".join(_mj_float(value) for value in values)

    def _configure_rubber_hand_geom(geom: ET.Element) -> None:
        body_name = _owning_body_name(geom)
        body_name_lower = body_name.lower()
        if "rubber_hand" not in body_name_lower or not _is_enabled_collision_geom(geom):
            return
        if body_name_lower.startswith("left_"):
            side = "left"
        elif body_name_lower.startswith("right_"):
            side = "right"
        else:
            side = body_name
        if "name" not in geom.attrib:
            geom.set("name", f"{side}_rubber_hand_collision")
        geom.set("condim", str(MUJOCO_OBJECT_CONDIM))
        geom.set(
            "friction",
            _format_friction(
                [
                    MUJOCO_RUBBER_HAND_SLIDE_FRICTION,
                    MUJOCO_RUBBER_HAND_SPIN_FRICTION,
                    MUJOCO_RUBBER_HAND_ROLL_FRICTION,
                ]
            ),
        )
        geom.set("solref", "0.01 1")
        geom.attrib.pop("solimp", None)
        name = geom.attrib["name"]
        if name not in rubber_hand_geom_names:
            rubber_hand_geom_names.append(name)

    object_geom_names: list[str] = []

    for geom in scene_root.findall(".//geom"):
        if _is_object_geom(geom):
            if "name" not in geom.attrib:
                geom.set("name", "object_collision")
            object_geom_names.append(geom.attrib["name"])
            geom.set("rgba", "0.95 0.42 0.16 1")
            geom.set("density", "0")
            geom.set("friction", _format_friction(object_physics["friction"]))
            geom.set("condim", str(MUJOCO_OBJECT_CONDIM))
            geom.set("solref", "0.01 1")
            geom.attrib.pop("solimp", None)
            geom.set("contype", "4")
            geom.set("conaffinity", "11")
            geom.attrib.pop("group", None)
            continue
        _configure_rubber_hand_geom(geom)
        if geom.attrib.get("group") is not None:
            continue
        name = geom.attrib.get("name", "")
        mesh = geom.attrib.get("mesh", "")
        if name == str(terrain_cfg["name"]):
            terrain_slide = float(terrain_cfg["static_friction"])
            geom.set("friction", f"{_mj_float(terrain_slide)} 0.005 0.001")
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
            geom.set("solref", "0.01 1")
            geom.attrib.pop("solimp", None)
            continue
        if name == "floor" or "largebox" in name or "object" in mesh:
            if "largebox" in name or "object" in mesh:
                geom.set("rgba", "0.7 0.8 0.9 0.7")
            continue
        contype = geom.attrib.get("contype")
        conaffinity = geom.attrib.get("conaffinity")
        if (contype and contype != "0") or (conaffinity and conaffinity != "0"):
            geom.set("group", "3")

    if object_geom_names and rubber_hand_geom_names:
        contact_root = scene_root.find("contact")
        if contact_root is None:
            contact_root = ET.SubElement(scene_root, "contact")
        existing_pair_names = {
            pair.attrib.get("name", "")
            for pair in contact_root.findall("pair")
            if pair.attrib.get("name")
        }
        existing_pair_keys = {
            (pair.attrib.get("geom1", ""), pair.attrib.get("geom2", ""))
            for pair in contact_root.findall("pair")
        }
        pair_friction = _format_friction(
            [
                MUJOCO_RUBBER_HAND_SLIDE_FRICTION,
                MUJOCO_RUBBER_HAND_SLIDE_FRICTION,
                MUJOCO_RUBBER_HAND_SPIN_FRICTION,
                MUJOCO_RUBBER_HAND_ROLL_FRICTION,
                MUJOCO_RUBBER_HAND_ROLL_FRICTION,
            ]
        )
        for hand_geom_name in rubber_hand_geom_names:
            for object_geom_name in object_geom_names:
                if (hand_geom_name, object_geom_name) in existing_pair_keys or (
                    object_geom_name,
                    hand_geom_name,
                ) in existing_pair_keys:
                    continue
                base_pair_name = f"{hand_geom_name}_{object_geom_name}"
                pair_name = base_pair_name
                suffix = 2
                while pair_name in existing_pair_names:
                    pair_name = f"{base_pair_name}_{suffix}"
                    suffix += 1
                existing_pair_names.add(pair_name)
                existing_pair_keys.add((hand_geom_name, object_geom_name))
                ET.SubElement(
                    contact_root,
                    "pair",
                    {
                        "name": pair_name,
                        "geom1": hand_geom_name,
                        "geom2": object_geom_name,
                        "condim": str(MUJOCO_OBJECT_CONDIM),
                        "friction": pair_friction,
                        "solref": "0.01 1",
                    },
                )

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
) -> tuple[dict, dict]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    patched_model_path = bundle_dir / "policy.onnx"
    patch_model(model_path.resolve(), motion_path.resolve(), patched_model_path.resolve())

    metadata = _read_onnx_metadata(patched_model_path)
    onnx_inputs, onnx_outputs = _read_onnx_io(patched_model_path)
    obs_shape = next((entry["shape"] for entry in onnx_inputs if entry["name"] in {"obs", "actor_obs"}), None)
    perception_shape = next((entry["shape"] for entry in onnx_inputs if entry["name"] == "perception_obs"), None)
    obs_dim = int(obs_shape[1]) if obs_shape and len(obs_shape) > 1 else 0
    perception_dim = int(perception_shape[1]) if perception_shape and len(perception_shape) > 1 else None
    dof_names = list(metadata["dof_names"])
    motion_summary = _read_motion_summary(motion_path.resolve(), dof_names)
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
        "clip_name": motion_path.stem,
        "root_body_name": "pelvis",
        "object_body_name": "object_baseLink",
        "dof_names": dof_names,
        "default_dof_angles": _extract_default_joint_angles(metadata),
        "robot_init_state": _extract_robot_init_state(metadata),
        "kp": list(metadata["kp"]),
        "kd": list(metadata["kd"]),
        "reset_mode_default": "demo_raw",
        "reset_modes": ["demo_raw", "isaac_training_default_pose"],
        "perception": _extract_perception_cfg(metadata, perception_dim=perception_dim),
        "observation": _extract_observation_cfg(metadata, obs_dim=obs_dim),
        "control": {
            **control_cfg,
            "effort_limits": effort_limits,
            "policy_action_scales": policy_action_scales,
        },
        "onnx": {
            "inputs": onnx_inputs,
            "outputs": onnx_outputs,
            "obs_dim": obs_dim,
            "perception_obs_dim": perception_dim,
        },
        "web_control": {
            "xy_speed_mps": 0.7,
            "yaw_speed_radps": 1.0,
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
    scene_path: Path,
    actuator_source_path: Path,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    manifest_clips: list[dict] = []
    bundle_ids: set[str] = set()
    default_clip_id: str | None = None
    default_label: str | None = None
    default_scene_path: str | None = None

    for motion_path in motion_paths:
        base_id = _bundle_id_for_motion_path(motion_path, motion_root)
        bundle_id = base_id
        suffix = 2
        while bundle_id in bundle_ids:
            bundle_id = f"{base_id}-{suffix}"
            suffix += 1
        bundle_ids.add(bundle_id)
        bundle_dir = clips_dir / bundle_id
        scene_xml_path = _write_shared_scene_assets(
            scene_path=scene_path,
            actuator_source_path=actuator_source_path,
            output_dir=output_dir,
            model_path=model_path,
            object_urdf_path=_read_motion_object_urdf_path(motion_path),
            scene_xml_path=bundle_dir / "scene.xml",
        )
        effort_limits = _extract_effort_limits(scene_xml_path)

        _clip_config, clip_manifest_entry = _stage_clip_bundle(
            model_path=model_path,
            motion_path=motion_path,
            bundle_dir=bundle_dir,
            scene_path=scene_xml_path,
            effort_limits=effort_limits,
        )
        manifest_clips.append(clip_manifest_entry)
        if default_clip_id is None:
            default_clip_id = clip_manifest_entry["id"]
            default_label = clip_manifest_entry["label"]
            default_scene_path = str(scene_xml_path.relative_to(output_dir))

    manifest = {
        "version": 1,
        "scene_path": default_scene_path,
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
    parser.add_argument("--motion-glob", type=str, default="*.npz")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-clips", type=int, default=DEFAULT_MAX_CLIPS)
    parser.add_argument("--preferred-clip-stem", type=str, default=DEFAULT_MOTION.stem)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--actuator-source-xml", type=Path, default=DEFAULT_ACTUATOR_SOURCE)
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
        scene_path=args.scene_xml.expanduser().resolve(),
        actuator_source_path=args.actuator_source_xml.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "clip_count": manifest["clip_count"]}, indent=2))


if __name__ == "__main__":
    main()
