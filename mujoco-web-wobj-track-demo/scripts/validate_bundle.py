#!/usr/bin/env python3
"""Validate the staged tracking web bundle against the train_object_extend observation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import onnx


def _load_manifest(asset_root: Path) -> dict:
    manifest_path = asset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_clip_config(asset_root: Path, clip_id: str | None) -> tuple[str, dict]:
    manifest = _load_manifest(asset_root)
    clips = manifest.get("clips", [])
    if not clips:
        raise RuntimeError(f"No clips found under {asset_root}")
    selected_id = clip_id or manifest.get("default_clip_id") or clips[0]["id"]
    for clip in clips:
        if clip["id"] == selected_id:
            config_path = asset_root / clip["config_path"]
            return clip["id"], json.loads(config_path.read_text(encoding="utf-8"))
    raise RuntimeError(f"Unknown clip id {selected_id!r}")


def _shape_of(value_info) -> list[int]:
    shape = []
    for dim in value_info.type.tensor_type.shape.dim:
        shape.append(int(dim.dim_value) if dim.dim_value else 0)
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "public" / "demo-assets",
    )
    parser.add_argument("--clip-id", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.expanduser().resolve()
    clip_id, config = _load_clip_config(asset_root, args.clip_id)
    model_path = asset_root / config["model_path"]
    scene_path = asset_root / config["scene_path"]
    motion_cfg = config["motion"]

    if not scene_path.is_file():
        raise FileNotFoundError(scene_path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)

    model = onnx.load(model_path)
    input_shapes = {value.name: _shape_of(value) for value in model.graph.input}
    output_names = [value.name for value in model.graph.output]
    scene_root = ET.parse(scene_path).getroot()

    expected_terms = {
        "motion_command": 58,
        "motion_ref_ori_b": 6,
        "base_ang_vel": 3,
        "dof_pos": 29,
        "dof_vel": 29,
        "actions": 29,
        "obj_target_pose_size_b": 12,
        "obj_pos_b": 3,
        "obj_ori_b": 6,
        "obj_lin_vel_b": 3,
        "obj_ang_vel_b": 3,
    }
    expected_term_order = sorted(expected_terms.keys())
    expected_obs_dim = sum(expected_terms.values())

    assert input_shapes.get("obs") == [1, expected_obs_dim], input_shapes
    assert input_shapes.get("time_step") == [1, 1], input_shapes
    for name in ("actions", "joint_pos", "joint_vel", "ref_quat_xyzw"):
        assert name in output_names, output_names
    assert len(config["dof_names"]) == 29, len(config["dof_names"])
    assert len(config["default_dof_angles"]) == 29, len(config["default_dof_angles"])
    assert len(config["kp"]) == 29 and len(config["kd"]) == 29
    assert len(config["control"]["policy_action_scales"]) == 29
    assert len(config["control"]["effort_limits"]) == 29, config["control"]["effort_limits"]
    assert config["onnx"]["obs_dim"] == expected_obs_dim, config["onnx"]
    assert config["observation"]["actor_obs_terms_sorted"] == expected_term_order, config["observation"]
    assert config["observation"]["actor_obs_history_length"] == 1, config["observation"]
    assert config["observation"]["actor_obs_concatenate"] is True, config["observation"]
    assert config["use_root_reference_at_clip_start"] is True, config
    assert config["prefer_sim_ref_from_sim_state"] is True, config
    assert config["apply_training_motion_transitions"] is False, config
    assert motion_cfg["frame_count"] > 0
    assert motion_cfg["object_pos_w"] is not None
    assert motion_cfg["object_quat_wxyz"] is not None
    assert len(motion_cfg["object_size"]) == motion_cfg["frame_count"]
    assert len(motion_cfg["initial_joint_pos"]) == 29
    assert len(motion_cfg["initial_joint_vel"]) == 29
    assert len(motion_cfg["reset_joint_pos"]) == 29
    assert len(motion_cfg["reset_joint_vel"]) == 29
    compiler = scene_root.find("compiler")
    assert compiler is not None and compiler.attrib.get("meshdir") == "assets"
    scene_mesh_files = [mesh.attrib.get("file") for mesh in scene_root.findall(".//mesh[@file]")]
    assert "half_sphere.obj" in scene_mesh_files, scene_mesh_files
    assert "largebox/largebox.obj" in scene_mesh_files, scene_mesh_files
    actuator_names = [actuator.attrib.get("name") for actuator in scene_root.findall("./actuator/*")]
    assert len(actuator_names) == 29, actuator_names
    assert "left_hip_pitch_joint" in actuator_names, actuator_names
    tendons = scene_root.findall("./tendon/*")
    assert len(tendons) >= 8, len(tendons)
    pair_names = [pair.attrib.get("name", "") for pair in scene_root.findall("./contact/pair")]
    assert any("sim2sim" in name for name in pair_names), pair_names
    largebox_geoms = [geom for geom in scene_root.findall(".//geom") if geom.attrib.get("name") == "largebox"]
    assert largebox_geoms, "largebox geom missing from staged scene"
    assert any(geom.attrib.get("friction") == "0.4 0.005 0.001" for geom in largebox_geoms), largebox_geoms
    base_action_scale = float(config["control"]["action_scale"])
    expected_action_scales = [
        base_action_scale * float(config["control"]["effort_limits"][name]) / float(kp)
        if float(kp) != 0.0
        else 0.0
        for name, kp in zip(config["dof_names"], config["kp"], strict=False)
    ]
    for expected, actual in zip(expected_action_scales, config["control"]["policy_action_scales"], strict=False):
        assert abs(float(actual) - expected) < 1e-6, (expected, actual)
    assert len({round(float(value), 6) for value in config["control"]["policy_action_scales"]}) > 1

    print(f"Clip: {clip_id}")
    print(f"Scene: {scene_path}")
    print(f"Model: {model_path}")
    print(f"obs input shape: {input_shapes['obs']}")
    print(f"time_step shape: {input_shapes['time_step']}")
    print(f"outputs: {', '.join(output_names)}")
    print(f"actor_obs terms: {', '.join(config['observation']['actor_obs_terms_sorted'])}")
    print(f"frame_count: {motion_cfg['frame_count']}")
    print(f"ref_body_name: {config['ref_body_name']}")
    print(f"object_body_name: {config['object_body_name']}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
