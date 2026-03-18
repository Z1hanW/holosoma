#!/usr/bin/env python3
"""Stage MuJoCo web-demo assets for train_object_extend tracking rollouts."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import onnx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src" / "holosoma_inference"))

from holosoma_inference.tools.patch_motion_onnx import patch_model


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
DEFAULT_SCENE = Path("/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml")
DEFAULT_ACTUATOR_SOURCE = Path("/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/robots/g1/g1_29dof.xml")
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


def _extract_effort_limits(scene_xml_path: Path) -> dict[str, float]:
    root = ET.parse(scene_xml_path).getroot()
    actuator_root = root.find("actuator")
    if actuator_root is None:
        return {}

    limits: dict[str, float] = {}
    for motor in actuator_root.findall("motor"):
        joint_name = motor.attrib.get("joint")
        ctrlrange = motor.attrib.get("ctrlrange")
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


def _read_motion_summary(motion_path: Path, dof_names: list[str], ref_body_name: str) -> dict:
    with np.load(motion_path, allow_pickle=True) as data:
        body_names = _decode_names(np.asarray(data["body_names"]))
        joint_names = _decode_names(np.asarray(data["joint_names"]))
        root_idx = _resolve_root_body_index(body_names)
        ref_idx = body_names.index(ref_body_name)
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float32) if "object_pos_w" in data else None
        object_quat_w = np.asarray(data["object_quat_w"], dtype=np.float32) if "object_quat_w" in data else None
        raw_object_size = None
        for key in ("object_size", "box_size", "object_scale", "box_scale"):
            if key in data:
                raw_object_size = np.asarray(data[key], dtype=np.float32)
                break
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])

    if joint_pos.shape[1] == len(joint_names) + 7:
        joint_pos = joint_pos[:, 7:]
    if joint_vel.shape[1] == len(joint_names) + 6:
        joint_vel = joint_vel[:, 6:]

    joint_indices = [joint_names.index(name) for name in dof_names]
    joint_pos = joint_pos[:, joint_indices]
    joint_vel = joint_vel[:, joint_indices]
    object_size = _normalize_object_size_array(raw_object_size, joint_pos.shape[0])

    return {
        "fps": fps,
        "frame_count": int(body_pos_w.shape[0]),
        "duration_s": float(body_pos_w.shape[0] / fps) if fps > 0.0 else 0.0,
        "initial_root_pos_w": body_pos_w[0, root_idx, :].astype(np.float32).tolist(),
        "initial_root_quat_wxyz": body_quat_w[0, root_idx, :].astype(np.float32).tolist(),
        "initial_ref_pos_w": body_pos_w[0, ref_idx, :].astype(np.float32).tolist(),
        "initial_ref_quat_wxyz": body_quat_w[0, ref_idx, :].astype(np.float32).tolist(),
        "initial_joint_pos": joint_pos[0].astype(np.float32).tolist(),
        "initial_joint_vel": joint_vel[0].astype(np.float32).tolist(),
        "object_pos_w": object_pos_w.astype(np.float32).tolist() if object_pos_w is not None else None,
        "object_quat_wxyz": object_quat_w.astype(np.float32).tolist() if object_quat_w is not None else None,
        "object_size": object_size.astype(np.float32).tolist(),
        "initial_object_pos_w": object_pos_w[0].astype(np.float32).tolist() if object_pos_w is not None else None,
        "initial_object_quat_wxyz": object_quat_w[0].astype(np.float32).tolist() if object_quat_w is not None else None,
        "initial_object_size": object_size[0].astype(np.float32).tolist(),
    }


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
    *,
    scene_path: Path,
    actuator_source_path: Path,
    output_dir: Path,
) -> Path:
    assets_src = scene_path.parent / "assets"
    largebox_src = scene_path.parent.parent / "largebox"
    _copy_tree(assets_src, output_dir / "assets")
    _copy_tree(largebox_src, output_dir / "assets" / "largebox")
    scene_xml_path = output_dir / "scene.xml"
    _patch_scene_xml(scene_path.resolve(), scene_xml_path, actuator_source_path.resolve())
    return scene_xml_path


def _stage_clip_bundle(
    *,
    model_path: Path,
    motion_path: Path,
    bundle_dir: Path,
    scene_path: Path,
    effort_limits: dict[str, float],
    apply_training_motion_transitions: bool,
) -> tuple[dict, dict]:
    bundle_dir.mkdir(parents=True, exist_ok=True)
    patched_model_path = bundle_dir / "policy.onnx"
    patch_model(
        model_path.resolve(),
        motion_path.resolve(),
        patched_model_path.resolve(),
        apply_training_motion_transitions=apply_training_motion_transitions,
    )

    metadata = _read_onnx_metadata(patched_model_path)
    onnx_inputs, onnx_outputs = _read_onnx_io(patched_model_path)
    dof_names = list(metadata["dof_names"])
    motion_cfg = _extract_motion_cfg(metadata)
    ref_body_names = motion_cfg.get("body_name_ref", ["torso_link"])
    ref_body_name = ref_body_names[0] if isinstance(ref_body_names, list) and ref_body_names else "torso_link"
    motion_summary = _read_motion_summary(motion_path.resolve(), dof_names, ref_body_name)
    control_cfg = _extract_control_cfg(metadata)
    policy_action_scales = _resolve_policy_action_scales(
        dof_names=dof_names,
        kp=list(metadata["kp"]),
        base_action_scale=float(control_cfg["action_scale"]),
        effort_limits=effort_limits,
    )
    obs_shape = next((entry["shape"] for entry in onnx_inputs if entry["name"] in {"obs", "actor_obs"}), None)

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
    scene_path: Path,
    actuator_source_path: Path,
    output_dir: Path,
    apply_training_motion_transitions: bool,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    scene_xml_path = _write_shared_scene_assets(
        scene_path=scene_path,
        actuator_source_path=actuator_source_path,
        output_dir=output_dir,
    )
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
            apply_training_motion_transitions=apply_training_motion_transitions,
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
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--actuator-source-xml", type=Path, default=DEFAULT_ACTUATOR_SOURCE)
    parser.add_argument("--apply-training-motion-transitions", action="store_true")
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
        apply_training_motion_transitions=bool(args.apply_training_motion_transitions),
    )
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "clip_count": manifest["clip_count"]}, indent=2))


if __name__ == "__main__":
    main()
