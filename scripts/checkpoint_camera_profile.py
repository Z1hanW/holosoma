#!/usr/bin/env python3
"""Extract the deployed depth-camera contract from an ONNX checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

import onnx


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolved-model-path-output", type=Path)
    parser.add_argument("--image-server-config-output", type=Path)
    parser.add_argument("--download-dir", type=Path, default=Path("/tmp/holosoma_checkpoints"))
    parser.add_argument("--training-depth", action="store_true",
                        help="Validate current box23K PPO depth, not a pitch-based legacy preset")
    return parser.parse_args(argv)


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _quat_multiply_xyzw(left: Sequence[float], right: Sequence[float]) -> list[float]:
    lx, ly, lz, lw = (float(value) for value in left)
    rx, ry, rz, rw = (float(value) for value in right)
    return [
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ]


def _normalized_quaternion(value: Sequence[float]) -> list[float]:
    quaternion = [float(item) for item in value]
    if len(quaternion) != 4 or not all(math.isfinite(item) for item in quaternion):
        raise ValueError(f"Invalid camera quaternion: {value}")
    norm = math.sqrt(sum(item * item for item in quaternion))
    if norm < 1.0e-8:
        raise ValueError("Camera quaternion has zero length")
    return [item / norm for item in quaternion]


def _euler_xyz_deg_from_quaternion_xyzw(value: Sequence[float]) -> list[float]:
    x, y, z, w = _normalized_quaternion(value)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def _effective_mount_quaternion(mount: Sequence[float], pitch_deg: float) -> list[float]:
    half_pitch = math.radians(float(pitch_deg)) * 0.5
    pitch_quaternion = [0.0, math.sin(half_pitch), 0.0, math.cos(half_pitch)]
    return _normalized_quaternion(_quat_multiply_xyzw(pitch_quaternion, mount))


def _input_width(model: onnx.ModelProto, name: str) -> int | None:
    for value in model.graph.input:
        if value.name != name:
            continue
        dimensions = value.type.tensor_type.shape.dim
        if dimensions and dimensions[-1].HasField("dim_value"):
            return int(dimensions[-1].dim_value)
    return None


def _profile_from_v2(contract: dict[str, Any]) -> dict[str, Any]:
    geometry = contract["camera_geometry"]
    schema = contract.get("effective_observation_schema", {})
    raw_height, raw_width = (int(value) for value in contract["camera_shape"])
    output_height, output_width = (int(value) for value in contract["camera_obs_shape"])
    crop = schema.get("crop", [raw_height - output_height, 0, 4, 4])
    position = schema.get("sensor_offset", geometry.get("body_offset_position", [0.0, 0.0, 0.0]))
    mount = geometry["mount_quaternion"]
    pitch_deg = float(geometry.get("pitch_deg", 0.0) or 0.0)
    return {
        "metadata_source": "perception_observation_contract",
        "camera_source": contract.get("camera_source", "unknown"),
        "raw_shape": [raw_height, raw_width],
        "policy_shape": [output_height, output_width],
        "crop": [int(value) for value in crop],
        "near": float(geometry["near"]),
        "far": float(geometry["far"]),
        "horizontal_fov_deg": float(geometry["hfov_deg"]),
        "vertical_fov_deg": float(geometry["vfov_deg"]),
        "fps": float(geometry["fps"]),
        "position_xyz": [float(value) for value in position],
        "mount_quaternion_xyzw": _effective_mount_quaternion(mount, pitch_deg),
    }


def _profile_from_legacy(config: dict[str, Any]) -> dict[str, Any]:
    perception = config["perception"]
    raw_height = int(perception["camera_height"])
    raw_width = int(perception["camera_width"])
    output_height, output_width = (int(value) for value in perception["camera_warp_resize"])
    crop = [
        int(perception.get("camera_warp_crop_top", 0)),
        int(perception.get("camera_warp_crop_bottom", 0)),
        int(perception.get("camera_warp_crop_left", 0)),
        int(perception.get("camera_warp_crop_right", 0)),
    ]
    mount = perception["camera_mount_quat"]
    pitch_deg = float(perception.get("camera_pitch_deg", 0.0) or 0.0)
    return {
        "metadata_source": "experiment_config.perception",
        "camera_source": perception.get("camera_source", "unknown"),
        "raw_shape": [raw_height, raw_width],
        "policy_shape": [output_height, output_width],
        "crop": crop,
        "near": float(perception["camera_near"]),
        "far": float(perception["camera_far"]),
        "horizontal_fov_deg": float(perception["camera_hfov_deg"]),
        "vertical_fov_deg": float(perception["camera_vfov_deg"]),
        "fps": float(perception["camera_fps"]),
        # Legacy checkpoints predate an exported sensor offset. This is the
        # matching D435 inference mount from camera.single_d435i_depth.
        "position_xyz": [0.01, 0.01, 0.44],
        "mount_quaternion_xyzw": _effective_mount_quaternion(mount, pitch_deg),
    }


def extract_profile(model_path: Path, *, training_depth: bool = False) -> dict[str, Any]:
    model = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(model)
    metadata = {item.key: _json_value(item.value) for item in model.metadata_props}
    if isinstance(metadata.get("perception_observation_contract"), dict):
        camera = _profile_from_v2(metadata["perception_observation_contract"])
    elif isinstance(metadata.get("experiment_config"), dict):
        camera = _profile_from_legacy(metadata["experiment_config"])
    else:
        raise ValueError("Checkpoint has no supported camera observation contract")

    expected_width = camera["policy_shape"][0] * camera["policy_shape"][1]
    model_width = _input_width(model, "perception_obs")
    if model_width != expected_width:
        raise ValueError(
            f"Camera contract produces {expected_width} values, but perception_obs expects {model_width}"
        )
    camera["rotation_xyz_deg"] = _euler_xyz_deg_from_quaternion_xyzw(camera["mount_quaternion_xyzw"])
    run_path = metadata.get("wandb_run_path", "unknown")
    if not isinstance(run_path, str):
        run_path = str(run_path)
    run_id = run_path.rstrip("/").rsplit("/", 1)[-1]
    camera["label"] = f"{run_id}: D435 {camera['rotation_xyz_deg'][1]:.1f} deg down"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    profile = {
        "version": 1,
        "checkpoint": {
            "path": str(model_path),
            "sha256": digest,
            "wandb_run_path": run_path,
        },
        "camera": camera,
    }
    if training_depth:
        from holosoma.sensors.training_depth import validate_training_depth_profile

        profile["training_depth_contract"] = metadata["perception_observation_contract"]
        profile["training_depth_contract_sha256"] = metadata["perception_observation_contract_sha256"]
        validate_training_depth_profile(profile)
        if {value.name for value in model.graph.input} != {"actor_obs", "perception_obs"}:
            raise ValueError("Expected non-recurrent command/depth actor inputs")
        if _input_width(model, "actor_obs") != 94:
            raise ValueError("Expected 94D command/drop/proprio actor")
        config = metadata["experiment_config"]
        actor = config["algo"]["config"]["module_dict"]["actor"]
        expected_groups = ["actor_obs_root_contact_aware", "actor_obs_drop_button",
                           "actor_obs_proprio_with_actions_no_linvel"]
        if actor["input_dim"] != expected_groups:
            raise ValueError("Unsupported actor observation group order")
        profile["deployment_command_mode"] = "explicit_manual_robot_heading_command_not_native_rollout"
        profile["synthetic_training_noise_applied"] = False
        profile["physical_mount_and_intrinsics_verified"] = False
        profile["depth_latency"] = "capture_timestamp_at_or_before_now_minus_3_or_4_over_30_seconds"
    return profile


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _resolve_model_path(model_path: str, download_dir: Path) -> Path:
    if model_path.startswith(("wandb://", "https://")):
        from holosoma_inference.utils.wandb import load_checkpoint

        return load_checkpoint(None, model_path, str(download_dir)).resolve()
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    model_path = _resolve_model_path(args.model_path, args.download_dir)
    profile = extract_profile(model_path, training_depth=args.training_depth)
    _atomic_write(args.output, json.dumps(profile, indent=2, sort_keys=True) + "\n")
    if args.resolved_model_path_output is not None:
        _atomic_write(args.resolved_model_path_output, f"{model_path}\n")
    camera = profile["camera"]
    if args.image_server_config_output is not None:
        camera_pitch_deg = float(camera["rotation_xyz_deg"][1])
        image_server_config = "real_d435i_urdf" if camera_pitch_deg > 42.0 else "real_d435i"
        _atomic_write(args.image_server_config_output, f"{image_server_config}\n")
    print(
        f"[camera_profile] {camera['label']} raw={camera['raw_shape'][1]}x{camera['raw_shape'][0]} "
        f"policy={camera['policy_shape'][1]}x{camera['policy_shape'][0]} "
        f"position={camera['position_xyz']} metadata={camera['metadata_source']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
