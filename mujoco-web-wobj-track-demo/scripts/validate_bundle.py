#!/usr/bin/env python3
"""Validate the staged tracking web bundle against the train_object_extend observation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    expected_obs_dim = sum(expected_terms.values())

    assert input_shapes.get("obs") == [1, expected_obs_dim], input_shapes
    assert input_shapes.get("time_step") == [1, 1], input_shapes
    for name in ("actions", "joint_pos", "joint_vel", "ref_quat_xyzw"):
        assert name in output_names, output_names
    assert len(config["dof_names"]) == 29, len(config["dof_names"])
    assert len(config["default_dof_angles"]) == 29, len(config["default_dof_angles"])
    assert len(config["kp"]) == 29 and len(config["kd"]) == 29
    assert len(config["control"]["policy_action_scales"]) == 29
    assert config["onnx"]["obs_dim"] == expected_obs_dim, config["onnx"]
    assert motion_cfg["frame_count"] > 0
    assert motion_cfg["object_pos_w"] is not None
    assert motion_cfg["object_quat_wxyz"] is not None
    assert len(motion_cfg["object_size"]) == motion_cfg["frame_count"]
    assert len(motion_cfg["initial_joint_pos"]) == 29
    assert len(motion_cfg["initial_joint_vel"]) == 29

    print(f"Clip: {clip_id}")
    print(f"Scene: {scene_path}")
    print(f"Model: {model_path}")
    print(f"obs input shape: {input_shapes['obs']}")
    print(f"time_step shape: {input_shapes['time_step']}")
    print(f"outputs: {', '.join(output_names)}")
    print(f"frame_count: {motion_cfg['frame_count']}")
    print(f"ref_body_name: {config['ref_body_name']}")
    print(f"object_body_name: {config['object_body_name']}")
    print("Validation: OK")


if __name__ == "__main__":
    main()
