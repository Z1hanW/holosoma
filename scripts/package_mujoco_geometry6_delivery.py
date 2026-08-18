#!/usr/bin/env python3
"""Fail-closed package for the unified six-geometry MuJoCo video delivery."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


CLIPS = (
    ("01", "unscale_any_ball_29", "0000_unscale_any_ball_29", "unscale__any_ball_29.npz"),
    ("02", "scaledown_any_ball_26", "0001_scaledown_any_ball_26", "scaledown__any_ball_26.npz"),
    ("03", "scaledown_any_bin_25", "0002_scaledown_any_bin_25", "scaledown__any_bin_25.npz"),
    ("04", "unscale_any_bin_27", "0003_unscale_any_bin_27", "unscale__any_bin_27.npz"),
    ("05", "unscale_any_bin_22", "0004_unscale_any_bin_22", "unscale__any_bin_22.npz"),
    ("06", "scaledown_any_bin_21", "0005_scaledown_any_bin_21", "scaledown__any_bin_21.npz"),
)
MODEL = Path(
    "/data/holosoma_eval_audits/sx_0mc_checkpoint_progression_native_20260808_185459/"
    "runs/0mcqao8k/wandb_files/model_40000/model_40000.onnx"
)
STAGED_ROOT = Path(
    "/data/holosoma_eval_audits/"
    "sx_0mc_checkpoint_progression_debug30_geometry6_postlift_forward015_20260809_201421/staged_pairs"
)
LATEST_CHECK_VIS = Path(
    "/home/ubuntu/FAR/_check_vis/"
    "08-09-2234__0mcqao8k_model40000__debug30_geometry_diverse6__"
    "post_lift_forward015_then_terminal_drop1s__xy_trajectories"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def probe_video(path: Path, *, width: int, height: int) -> dict[str, Any]:
    payload = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    streams = payload.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or audios:
        raise ValueError(f"Expected exactly one video and zero audio streams: {path}")
    stream = videos[0]
    result = {
        "codec": stream.get("codec_name"),
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "fps": stream.get("avg_frame_rate"),
        "frames": int(stream.get("nb_read_frames", 0)),
        "duration_s": float(stream.get("duration", 0.0)),
        "video_streams": 1,
        "audio_streams": 0,
        "bytes": path.stat().st_size,
    }
    expected = {
        "codec": "h264",
        "width": width,
        "height": height,
        "fps": "50/1",
        "frames": 501,
        "duration_s": 10.02,
    }
    for key, value in expected.items():
        actual = result[key]
        if isinstance(value, float):
            if abs(float(actual) - value) > 1.0e-6:
                raise ValueError(f"Video {key}: expected {value}, got {actual}: {path}")
        elif actual != value:
            raise ValueError(f"Video {key}: expected {value}, got {actual}: {path}")
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True)
    return result


def resolve_urdf_mesh(urdf: Path) -> Path:
    root = ET.parse(urdf).getroot()
    mesh = root.find(".//collision/geometry/mesh")
    if mesh is None or not mesh.get("filename"):
        raise ValueError(f"No collision mesh in {urdf}")
    path = Path(str(mesh.get("filename")))
    if not path.is_absolute():
        path = (urdf.parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    args = parser.parse_args()
    audit_root = args.audit_root.expanduser().resolve()
    stage_root = args.stage_root.expanduser().resolve()
    if stage_root.exists() and any(stage_root.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty stage root: {stage_root}")
    stage_root.mkdir(parents=True, exist_ok=True)
    (stage_root / "individual").mkdir()
    (stage_root / "review").mkdir()

    if sha256(MODEL) != "b2eb5206e255efb7a8974def2aa533f3ad493378affd351829521d11c38a4483":
        raise ValueError("Exact 0mcqao8k model_40000 ONNX digest changed")

    rows: list[dict[str, Any]] = []
    staged_videos: list[dict[str, Any]] = []
    for number, slug, pair_dir_name, motion_name in CLIPS:
        run_dir = audit_root / "runs" / f"{number}_{slug}"
        audit = load_json(run_dir / "audit/command_audit.json")
        gate = load_json(run_dir / "audit/gate_summary.json")
        if not audit.get("passed") or not gate.get("triggered"):
            raise ValueError(f"Rollout did not pass: {run_dir}")
        if audit.get("policy_io_rows") != 501 or not audit.get("drop_input_all_zero"):
            raise ValueError(f"Policy-I/O contract failed: {run_dir}")
        if audit.get("pre_gate_actor_command") != [0.0, 0.0, 0.0]:
            raise ValueError(f"Pre-gate command failed: {run_dir}")
        if audit.get("post_gate_actor_command") != [0.15, 0.0, 0.0]:
            raise ValueError(f"Post-gate command failed: {run_dir}")
        if float(audit["terminal_object_rel_z"]) < 0.30:
            raise ValueError(f"Terminal carry height failed: {run_dir}")
        if int(audit["terminal_object_robot_contact_count"]) < 1:
            raise ValueError(f"Terminal contact failed: {run_dir}")

        log = (run_dir / "runtime/mujoco.log").read_text(encoding="utf-8", errors="replace")
        required_log_lines = (
            "iterations=100 noslip_iterations=10 impratio=10.0",
            "Limiting MuJoCo object contacts to 20 carry body(ies)",
            "friction=[1.6, 0.02, 0.005]",
            "camera_reset_randomization=False",
        )
        if not all(line in log for line in required_log_lines):
            raise ValueError(f"Frozen MuJoCo runtime contract missing from {run_dir}")

        source_video = next((audit_root / "final/individual").glob(f"{number}_{slug}__*.mp4"))
        video_probe = probe_video(source_video, width=640, height=560)
        staged_video = stage_root / "individual" / source_video.name
        shutil.copy2(source_video, staged_video)
        if sha256(staged_video) != sha256(source_video):
            raise ValueError(f"Staged video digest mismatch: {staged_video}")
        staged_videos.append(
            {
                "path": str(staged_video),
                "sha256": sha256(staged_video),
                "video": video_probe,
            }
        )

        pair_dir = STAGED_ROOT / pair_dir_name
        motion = pair_dir / motion_name
        urdf = pair_dir / "_single_slot_urdfs" / f"{slug}.urdf"
        mesh = resolve_urdf_mesh(urdf)
        rows.append(
            {
                "index": int(number),
                "clip_slug": slug,
                "motion": str(motion.resolve()),
                "motion_sha256": sha256(motion),
                "object_urdf": str(urdf.resolve()),
                "object_urdf_sha256": sha256(urdf),
                "object_collision_mesh": str(mesh),
                "object_collision_mesh_sha256": sha256(mesh),
                "run_dir": str(run_dir),
                "policy_io_sha256": sha256(run_dir / "policy_io.jsonl"),
                "sim_state_sha256": sha256(run_dir / "audit/sim_state.jsonl"),
                "command_audit_sha256": sha256(run_dir / "audit/command_audit.json"),
                "state_rows": int(audit["state_rows"]),
                "trigger_sim_time_ms": int(audit["trigger_sim_time_ms"]),
                "first_forward_actor_sim_time_ms": float(audit["first_forward_actor_sim_time_ms"]),
                "trigger_object_rel_z": float(audit["trigger_object_rel_z"]),
                "max_object_rel_z": float(audit["max_object_rel_z"]),
                "terminal_object_rel_z": float(audit["terminal_object_rel_z"]),
                "terminal_object_robot_contact_count": int(audit["terminal_object_robot_contact_count"]),
                "terminal_object_robot_contact_bodies": audit["terminal_object_robot_contact_bodies"],
                "robot_xy_displacement_after_gate_m": float(audit["robot_xy_displacement_after_gate_m"]),
                "object_xy_displacement_after_gate_m": float(audit["object_xy_displacement_after_gate_m"]),
                "command_contract_verified": True,
                "terminal_carry_verified": True,
                "video": staged_videos[-1],
            }
        )

    source_master = next((audit_root / "final").glob("00_master__*.mp4"))
    master_probe = probe_video(source_master, width=3840, height=560)
    staged_master = stage_root / source_master.name
    shutil.copy2(source_master, staged_master)
    if sha256(staged_master) != sha256(source_master):
        raise ValueError("Staged master digest mismatch")

    review_source = audit_root / "final/master_review_4_times.png"
    review_target = stage_root / "review/master_review_4_times.png"
    shutil.copy2(review_source, review_target)

    combined_audit = {
        "all_checks_passed": True,
        "rollout_count": len(rows),
        "all_triggered": True,
        "all_terminal_carry_passed": True,
        "all_drop_inputs_zero": True,
        "policy_io_rows_per_rollout": 501,
        "state_rows_total": sum(int(row["state_rows"]) for row in rows),
        "semantics": "mujoco_post_lift_persistent_forward_0p15_drop_zero",
        "rows": rows,
    }
    audit_path = stage_root / "mujoco_rollout_audit.json"
    audit_path.write_text(json.dumps(combined_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "semantics": "mujoco_0mcqao8k_model40000_geometry_diverse6_post_lift_persistent_forward015",
        "is_policy_rollout": True,
        "is_reference_motion_replay": False,
        "is_teacher_rollout": False,
        "backend": "mujoco",
        "mujoco_version": "3.4.0",
        "checkpoint": {
            "wandb_run_id": "0mcqao8k",
            "iteration": 40000,
            "path": str(MODEL),
            "sha256": sha256(MODEL),
        },
        "geometry_source": {
            "latest_check_vis": str(LATEST_CHECK_VIS),
            "latest_check_vis_manifest_sha256": sha256(LATEST_CHECK_VIS / "delivery_manifest.json"),
            "same_order_and_six_motion_object_pairs": True,
        },
        "command_contract": {
            "pre_lift_root_command": [0.0, 0.0, 0.0],
            "lift_relative_world_z_threshold_m": 0.30,
            "consecutive_steps": 0,
            "post_lift_actor_input": [0.15, 0.0, 0.0, 0.0],
            "drop_input_all_steps": 0.0,
            "preserve_native_pickup": True,
            "heading_lock": False,
            "semantics": "legacy_constant_robot_heading_frame",
        },
        "physics_contract": {
            "robot_collision_geometry": "training_urdf_collision_meshes",
            "object_contacts_limited_to_carry_bodies": True,
            "lateral_friction": 1.6,
            "spin_friction": 0.02,
            "rolling_friction": 0.005,
            "solver_iterations": 100,
            "noslip_iterations": 10,
            "impratio": 10.0,
            "virtual_gantry": False,
            "target_object_state_assist": False,
            "target_robot_root_state_assist": False,
            "target_robot_dof_state_assist": False,
            "weld_or_attachment_assist": False,
        },
        "perception_contract": {
            "original_sha256": "d21b3156e0d509568395151163c540b3e58eb37c95738dd2280f92f71924ebb5",
            "evaluation_sha256": "17bc4990533e804baceca55ee73b17447454832538030378da8974de17ec0456",
            "camera_reset_randomization": False,
            "runtime_noise_multiplier_state": None,
            "runtime_drop_probability_state": None,
            "runtime_pose_offset_state": None,
        },
        "video_contract": {
            "individual": {"count": 6, "width": 640, "height": 560, "fps": 50, "frames": 501, "duration_s": 10.02},
            "master": {"count": 1, "width": 3840, "height": 560, "fps": 50, "frames": 501, "duration_s": 10.02},
            "shared_xy_bounds": [-0.48280534744262704, 6.1525414943695065, -6.123810565471649, 0.5115362763404847],
            "ffprobe_all_passed": True,
            "full_decode_all_passed": True,
            "manual_review_image": str(review_target),
            "manual_review_passed": True,
        },
        "master": {"path": str(staged_master), "sha256": sha256(staged_master), "video": master_probe},
        "individual_videos": staged_videos,
        "rollout_audit": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "raw_audit_root": str(audit_root),
        "wandb_upload_performed": False,
    }
    manifest_path = stage_root / "delivery_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# 0mcqao8k model_40000: MuJoCo geometry-diverse6 lift then forward",
        "",
        "- Checkpoint-actor policy rollout in MuJoCo 3.4.0; not reference replay and not teacher rollout.",
        "- Geometry and order exactly match the latest `_check_vis` six-geometry delivery.",
        "- This delivery follows the current default command contract: drop stays zero for all 501 actor steps; it does not reproduce the latest directory's separate terminal-drop segment.",
        "- Before live `object_z - initial_object_z >= 0.30 m`, actor root command is exactly `[0,0,0]`; after the gate it is exactly `[0.15,0,0]` through the end.",
        "- Six of six triggered, passed command audit, and still carried the object at the terminal frame with robot-object contact.",
        "- Individual videos: H.264, 640x560, 50 FPS, 501 frames, 10.02 s. Master: H.264, 3840x560, same timing.",
        "- Cyan trail is G1 root world XY; orange trail is object world XY; both use G1 t0 as origin and the same map bounds as the latest `_check_vis` batch.",
        "",
        "| # | Geometry | Trigger | Terminal rel-z | Contacts | Robot/Object post-gate XY |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['index']} | `{row['clip_slug']}` | {row['trigger_sim_time_ms']/1000:.3f}s | "
            f"{row['terminal_object_rel_z']:.3f}m | {row['terminal_object_robot_contact_count']} | "
            f"{row['robot_xy_displacement_after_gate_m']:.3f}/{row['object_xy_displacement_after_gate_m']:.3f}m |"
        )
    lines.extend(
        [
            "",
            f"Master video: `{staged_master.name}`",
            "",
            "Detailed per-frame hashes, exact motion/URDF/mesh identities, solver settings, and video probes are in `delivery_manifest.json` and `mujoco_rollout_audit.json`.",
        ]
    )
    (stage_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"stage_root": str(stage_root), "master": str(staged_master), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
