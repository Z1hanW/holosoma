#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parent
RUN_ROOT = Path("/home/ubuntu/FAR/holosoma_runs/formal_teacher_rl_model2k_eval3_20260720")
ENTITY = "zihanw22"
PROJECT = "carry-any"
RUN_ID = "crjedc0u"
RUN_PATH = f"{ENTITY}/{PROJECT}/{RUN_ID}"
CHECKPOINT_REF = "wandb://zihanw22/carry-any/crjedc0u/model_02000.pt"
CHECKPOINT_SHA256 = "4e4f8739a332abfd49c23760e07c8413d56441793eaa07167ef80a8762a8ba19"
SOURCE_ID = "src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399"
SOURCE_SHA256 = "a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399"


@dataclass(frozen=True)
class Role:
    name: str
    media_key: str
    motion_clip: str
    motion_sha256: str
    object_map_sha256: str
    object_urdf_sha256: str
    object_mesh_sha256: str
    video_name: str
    video_sha256: str
    visual_review_result: str


ROLES = (
    Role(
        name="ball",
        media_key="vis/evaluation_checkpoint_actor_ball",
        motion_clip="unscale__any_ball_29",
        motion_sha256="87644e984e4af1e7b75f3e3f83d822a0ee18e2b7604968d38f16d5b80cae46bb",
        object_map_sha256="30fb9d3f81ceceef22fc6aa5df83778aac34ba65d2d653c13584fd7feb5aa92d",
        object_urdf_sha256="2e9d3d7c47f5915415e2aefa0b76b7294a829bfb5bb74dc637466a40dc38d556",
        object_mesh_sha256="9734a65b4cd1127c96fad2b499832cbe5f5c7608200c593127c45db31b92d5b9",
        video_name="model_02000_checkpoint_actor_ball.mp4",
        video_sha256="fd30c7d11dc504683274db375186e340b12bcb9500d080c3fd75ca8b17dac5df",
        visual_review_result="initial_carry_and_walk_then_late_horizon_fall",
    ),
    Role(
        name="bin",
        media_key="vis/evaluation_checkpoint_actor_bin",
        motion_clip="unscale__any_bin_29",
        motion_sha256="9c981f20edb97a9d598fee7beb15c42278ad4bb0bc1725812540fd243f35adb4",
        object_map_sha256="b8cb4c033916af0b598f5966f026f54fb6ab54ec79005ddbef2338f1da2aec5f",
        object_urdf_sha256="c2009cb217f8157bfd581bceffe1ad422d98722b425af8341057674118e1c385",
        object_mesh_sha256="daae95872696e55484f37a166978fca182303ce1bb73b26d851b0d085784890d",
        video_name="model_02000_checkpoint_actor_bin.mp4",
        video_sha256="a837dc2203ea5fc33e5ebdaacb8ce902f7cd1f9092149d20889cc134bd73c893",
        visual_review_result="initial_carry_and_walk_then_late_horizon_fall",
    ),
    Role(
        name="barrel",
        media_key="vis/evaluation_checkpoint_actor_barrel",
        motion_clip="scaledown__any_barrel_25",
        motion_sha256="382e0aaffc8a6e4dd4c1906eaed50c5ed3e244bdd3e769e5581f374e60f06126",
        object_map_sha256="9e7bf562367cc32802c251d843c6d79218be9dcd23626bbedc9df5482718cdd3",
        object_urdf_sha256="0a69875d5a2d4ed19a62040a1ee326e81f68b49c934f4ed659d21b23828843e3",
        object_mesh_sha256="24d046ad6047fa8f33c63138c9be35975d5a0e078bc6df321b2068442d64f4c5",
        video_name="model_02000_checkpoint_actor_barrel.mp4",
        video_sha256="660289767f6929e7040bebac65db670b4797ff58170ebfea828687d890f9da2e",
        visual_review_result="initial_carry_and_walk_then_late_horizon_fall",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected exactly one video stream in {path}, got {streams!r}")
    stream = streams[0]
    expected = {
        "codec_name": "h264",
        "width": 640,
        "height": 360,
        "pix_fmt": "yuv420p",
        "r_frame_rate": "50/1",
        "nb_read_frames": "501",
        "duration": "10.020000",
    }
    if any(stream.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Video contract mismatch for {path}: {stream!r}")
    return stream


def matching_files(run, role: Role):
    token = f"evaluation_checkpoint_actor_{role.name}"
    return [file_obj for file_obj in run.files() if token in file_obj.name and file_obj.name.endswith(".mp4")]


def local_payload(*, include_media: bool = True) -> tuple[dict[str, object], dict[str, Path]]:
    payload: dict[str, object] = {
        "evaluation/policy_role": "checkpoint_actor",
        "evaluation/policy_type": "privileged_teacher_rl",
        "evaluation/checkpoint_ref": CHECKPOINT_REF,
        "evaluation/checkpoint_sha256": CHECKPOINT_SHA256,
        "evaluation/checkpoint_completed_iteration": 1999,
        "evaluation/checkpoint_step": 2000,
        "evaluation/source_snapshot_id": SOURCE_ID,
        "evaluation/source_manifest_sha256": SOURCE_SHA256,
        "evaluation/num_envs": 1,
        "evaluation/robot_count": 1,
        "evaluation/randomization_preset": "disabled",
        "evaluation/randomization_active_terms": 0,
        "evaluation/camera_pose_randomization": False,
        "evaluation/depth_multiplicative_noise": False,
        "evaluation/depth_dropout": False,
        "evaluation/start_at_timestep_zero_prob": 1.0,
        "evaluation/freeze_at_timestep_zero_prob": 0.0,
        "evaluation/initial_pose_noise_scale": 0.0,
        "evaluation/auto_reset_disabled": True,
        "evaluation/motion_end_reset_disabled": True,
        "evaluation/clip_end_reset_disabled": True,
        "evaluation/rollout_reference_rewards_disabled": True,
        "evaluation/max_eval_steps": 500,
        "evaluation/video_num_frames": 501,
        "evaluation/video_fps": 50,
        "evaluation/video_duration_seconds": 10.02,
        "evaluation/video_codec": "h264",
        "evaluation/visual_review_result": "initial_carry_and_walk_then_late_horizon_fall_all_three",
    }
    paths: dict[str, Path] = {}
    for role in ROLES:
        path = RUN_ROOT / "videos" / "final" / role.video_name
        if not path.is_file():
            raise FileNotFoundError(path)
        probe(path)
        actual_sha256 = sha256(path)
        if actual_sha256 != role.video_sha256:
            raise RuntimeError(f"Local video SHA mismatch for {role.name}: {actual_sha256} != {role.video_sha256}")
        paths[role.name] = path
        if include_media:
            payload[role.media_key] = wandb.Video(str(path), format="mp4", fps=50)
        prefix = f"evaluation/{role.name}"
        payload[f"{prefix}/motion_clip"] = role.motion_clip
        payload[f"{prefix}/motion_sha256"] = role.motion_sha256
        payload[f"{prefix}/object_map_sha256"] = role.object_map_sha256
        payload[f"{prefix}/object_urdf_sha256"] = role.object_urdf_sha256
        payload[f"{prefix}/object_mesh_sha256"] = role.object_mesh_sha256
        payload[f"{prefix}/video_sha256"] = role.video_sha256
        payload[f"{prefix}/video_size_bytes"] = path.stat().st_size
        payload[f"{prefix}/visual_review_result"] = role.visual_review_result
    return payload, paths


def verify_remote(paths: dict[str, Path]) -> dict[str, object]:
    deadline = time.monotonic() + 60
    while True:
        run = wandb.Api(timeout=90).run(RUN_PATH)
        visible = {role.name: matching_files(run, role) for role in ROLES}
        if all(len(files) == 1 for files in visible.values()):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "Timed out waiting for the W&B file index: "
                + repr({name: [file.name for file in files] for name, files in visible.items()})
            )
        time.sleep(2)
    if run.state != "running":
        raise RuntimeError(f"Run state changed after upload: {run.state!r}")

    remote_records: list[dict[str, object]] = []
    for role in ROLES:
        files = matching_files(run, role)
        if len(files) != 1:
            raise RuntimeError(f"Expected one remote video for {role.name}, got {[file.name for file in files]}")
        remote = files[0]
        with tempfile.TemporaryDirectory(prefix=f"model2k_{role.name}_") as temp_dir:
            downloaded = Path(remote.download(root=temp_dir, replace=True).name)
            remote_sha256 = sha256(downloaded)
            if remote_sha256 != role.video_sha256:
                raise RuntimeError(
                    f"Remote video SHA mismatch for {role.name}: {remote_sha256} != {role.video_sha256}"
                )
            probe(downloaded)
        remote_records.append(
            {
                "role": role.name,
                "name": remote.name,
                "size": remote.size,
                "sha256": remote_sha256,
                "local_path": str(paths[role.name]),
            }
        )

    history_keys = [
        "_step",
        *(role.media_key for role in ROLES),
        "evaluation/checkpoint_ref",
        "evaluation/checkpoint_completed_iteration",
        "evaluation/checkpoint_step",
        "evaluation/checkpoint_sha256",
        "evaluation/policy_role",
        "evaluation/policy_type",
        "evaluation/source_snapshot_id",
        "evaluation/source_manifest_sha256",
        "evaluation/num_envs",
        "evaluation/robot_count",
        "evaluation/randomization_preset",
        "evaluation/randomization_active_terms",
        "evaluation/max_eval_steps",
        "evaluation/video_num_frames",
        "evaluation/video_fps",
        "evaluation/video_duration_seconds",
        "evaluation/visual_review_result",
        *(f"evaluation/{role.name}/motion_clip" for role in ROLES),
        *(f"evaluation/{role.name}/video_sha256" for role in ROLES),
    ]
    rows = list(run.scan_history(keys=history_keys, page_size=1000))
    rows = [row for row in rows if any(row.get(role.media_key) for role in ROLES)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one evaluation history row, got {len(rows)}")
    row = rows[0]
    if not all(row.get(role.media_key) for role in ROLES):
        raise RuntimeError(f"Evaluation history row is missing media: {row!r}")
    expected_row = {
        "evaluation/checkpoint_ref": CHECKPOINT_REF,
        "evaluation/checkpoint_completed_iteration": 1999,
        "evaluation/checkpoint_step": 2000,
        "evaluation/checkpoint_sha256": CHECKPOINT_SHA256,
        "evaluation/policy_role": "checkpoint_actor",
        "evaluation/policy_type": "privileged_teacher_rl",
        "evaluation/source_snapshot_id": SOURCE_ID,
        "evaluation/source_manifest_sha256": SOURCE_SHA256,
        "evaluation/num_envs": 1,
        "evaluation/robot_count": 1,
        "evaluation/randomization_preset": "disabled",
        "evaluation/randomization_active_terms": 0,
        "evaluation/max_eval_steps": 500,
        "evaluation/video_num_frames": 501,
        "evaluation/video_fps": 50,
        "evaluation/video_duration_seconds": 10.02,
        "evaluation/visual_review_result": "initial_carry_and_walk_then_late_horizon_fall_all_three",
    }
    for role in ROLES:
        expected_row[f"evaluation/{role.name}/motion_clip"] = role.motion_clip
        expected_row[f"evaluation/{role.name}/video_sha256"] = role.video_sha256
    for key, value in expected_row.items():
        if row.get(key) != value:
            raise RuntimeError(f"History metadata mismatch for {key}: {row.get(key)!r} != {value!r}")

    return {"run_state": run.state, "history_step": row["_step"], "remote_videos": remote_records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify an existing upload without creating another history row.",
    )
    args = parser.parse_args()

    if args.verify_only:
        _, paths = local_payload(include_media=False)
        result = verify_remote(paths)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    api = wandb.Api(timeout=90)
    before = api.run(RUN_PATH)
    if before.state != "running":
        raise RuntimeError(f"Refusing upload: {RUN_PATH} state is {before.state!r}, expected 'running'")
    existing = {role.name: [file.name for file in matching_files(before, role)] for role in ROLES}
    if any(existing.values()):
        raise RuntimeError(f"Refusing duplicate evaluation upload: {existing!r}")

    payload, paths = local_payload()
    secondary = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=RUN_ID,
        resume="must",
        reinit=True,
        settings=wandb.Settings(x_primary=False, x_update_finish_state=False),
    )
    try:
        secondary.log(payload)
    finally:
        secondary.finish(quiet=True)

    result = verify_remote(paths)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
