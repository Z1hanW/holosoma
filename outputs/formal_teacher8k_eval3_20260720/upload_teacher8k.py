#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import wandb


ROOT = Path(__file__).resolve().parent
ENTITY = "zihanw22"
PROJECT = "carry-any"
MEDIA_KEY = "vis/evaluation_distill_label_teacher"
OLD_MEDIA_KEY = "vis/evaluation_motion_generator_teacher"
TEACHER_REF = "wandb://zihanw22/carry-any/li1gcc1v/model_08000.pt"
TEACHER_SHA256 = "a6093a6fbfb84932517002323fab735aff4759214d3b56acd65e8db934929124"
GENERATOR_REF = "wandb://zihanw22/carry-any/u8udzw0u/model_05000.pt"
GENERATOR_SHA256 = "80cb13e13590239d015ba0a29bdbae901b7785a9789d144745c7bd330059cd68"
SOURCE_ID = "src-6a871a6c74d045b8ff1686002f6dcc8eacae438022151013e6630b3227a28eca"
SOURCE_SHA256 = "6a871a6c74d045b8ff1686002f6dcc8eacae438022151013e6630b3227a28eca"


def summary_metadata(*, retired: bool) -> dict[str, object]:
    return {
        "evaluation/primary_teacher_key": MEDIA_KEY,
        "evaluation/primary_teacher_role": "distill_label_teacher",
        "evaluation/primary_teacher_checkpoint_ref": TEACHER_REF,
        "evaluation/primary_teacher_checkpoint_sha256": TEACHER_SHA256,
        "evaluation/motion_generator_teacher_checkpoint_ref": GENERATOR_REF,
        "evaluation/motion_generator_teacher_checkpoint_sha256": GENERATOR_SHA256,
        "evaluation/distill_label_teacher_same_as_generator": False,
        "evaluation/visible_media_contract": (
            "vis/replay + vis/evaluation_distill_label_teacher + vis/evaluation_student"
        ),
        "evaluation/retired_motion_generator_teacher_media": retired,
    }


@dataclass(frozen=True)
class Role:
    name: str
    run_id: str
    motion_clip: str
    motion_sha256: str
    object_map_sha256: str
    object_urdf_sha256: str
    object_mesh_sha256: str
    video_name: str
    video_sha256: str
    old_video_sha256: str


ROLES = (
    Role(
        name="ball",
        run_id="q9qn6xb9",
        motion_clip="unscale__any_ball_29",
        motion_sha256="87644e984e4af1e7b75f3e3f83d822a0ee18e2b7604968d38f16d5b80cae46bb",
        object_map_sha256="30fb9d3f81ceceef22fc6aa5df83778aac34ba65d2d653c13584fd7feb5aa92d",
        object_urdf_sha256="2e9d3d7c47f5915415e2aefa0b76b7294a829bfb5bb74dc637466a40dc38d556",
        object_mesh_sha256="9734a65b4cd1127c96fad2b499832cbe5f5c7608200c593127c45db31b92d5b9",
        video_name="ball_teacher8k_motion_frames_0004_0318.mp4",
        video_sha256="b3349bb2589d0657f05365f7bc3fb231e0f8a4363edeb11fb34558e42addcb75",
        old_video_sha256="fd0f9edf5bb9d9b2446c972a1f82b5a59b07360f6fcca2ea657a0a4dd7e3d58d",
    ),
    Role(
        name="bin",
        run_id="b72gh1wx",
        motion_clip="unscale__any_bin_29",
        motion_sha256="9c981f20edb97a9d598fee7beb15c42278ad4bb0bc1725812540fd243f35adb4",
        object_map_sha256="b8cb4c033916af0b598f5966f026f54fb6ab54ec79005ddbef2338f1da2aec5f",
        object_urdf_sha256="c2009cb217f8157bfd581bceffe1ad422d98722b425af8341057674118e1c385",
        object_mesh_sha256="daae95872696e55484f37a166978fca182303ce1bb73b26d851b0d085784890d",
        video_name="bin_teacher8k_motion_frames_0004_0318.mp4",
        video_sha256="acba4e0f1095569d3fac41ed25387d4697c5cdb72a744461fbfbdfa16bb97ec1",
        old_video_sha256="a7ba356d7c5f604156565c7695ae8bd64da7864dd5f689a152e441352c7bf8a9",
    ),
    Role(
        name="barrel",
        run_id="5utvhw89",
        motion_clip="scaledown__any_barrel_25",
        motion_sha256="382e0aaffc8a6e4dd4c1906eaed50c5ed3e244bdd3e769e5581f374e60f06126",
        object_map_sha256="9e7bf562367cc32802c251d843c6d79218be9dcd23626bbedc9df5482718cdd3",
        object_urdf_sha256="0a69875d5a2d4ed19a62040a1ee326e81f68b49c934f4ed659d21b23828843e3",
        object_mesh_sha256="24d046ad6047fa8f33c63138c9be35975d5a0e078bc6df321b2068442d64f4c5",
        video_name="barrel_teacher8k_motion_frames_0004_0318.mp4",
        video_sha256="954d20817633d79e59c65f737c4bd27a11df6e806ca5ab020a94f5bde2f97e7d",
        old_video_sha256="695baeacc554e4e43ec795aaea5d53fddbce0d4028cc8a971cea464e372cd7a8",
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
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
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
        "r_frame_rate": "50/1",
        "nb_read_frames": "315",
        "duration": "6.300000",
    }
    if any(stream.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Video contract mismatch for {path}: {stream!r}")
    return stream


def fresh_run(api: wandb.Api, run_id: str):
    return api.run(f"{ENTITY}/{PROJECT}/{run_id}")


def matching_files(run, needle: str):
    return [file_obj for file_obj in run.files() if needle in file_obj.name and file_obj.name.endswith(".mp4")]


def verify_remote(role: Role, *, retired: bool) -> dict[str, object]:
    api = wandb.Api(timeout=90)
    run = fresh_run(api, role.run_id)
    if run.state != "running":
        raise RuntimeError(f"Refusing verification: {role.run_id} state changed to {run.state!r}")

    files = matching_files(run, "evaluation_distill_label_teacher")
    if len(files) != 1:
        raise RuntimeError(f"Expected one new teacher media file for {role.run_id}, got {[f.name for f in files]}")
    remote = files[0]
    with tempfile.TemporaryDirectory(prefix=f"teacher8k_{role.name}_") as temp_dir:
        downloaded = Path(remote.download(root=temp_dir, replace=True).name)
        remote_sha256 = sha256(downloaded)
        if remote_sha256 != role.video_sha256:
            raise RuntimeError(
                f"Remote video SHA mismatch for {role.run_id}: {remote_sha256} != {role.video_sha256}"
            )
        probe(downloaded)

    rows = list(
        run.scan_history(
            keys=[
                "_step",
                MEDIA_KEY,
                "evaluation/checkpoint_step",
                "evaluation/teacher_checkpoint_sha256",
                "evaluation/teacher_role",
                "evaluation/video_sha256",
            ],
            page_size=1000,
        )
    )
    rows = [row for row in rows if row.get(MEDIA_KEY)]
    if len(rows) != 1:
        raise RuntimeError(f"Expected one history row for {MEDIA_KEY} in {role.run_id}, got {len(rows)}")
    row = rows[0]
    expected_row = {
        "evaluation/checkpoint_step": 8000,
        "evaluation/teacher_checkpoint_sha256": TEACHER_SHA256,
        "evaluation/teacher_role": "distill_label_teacher",
        "evaluation/video_sha256": role.video_sha256,
    }
    for key, value in expected_row.items():
        if row.get(key) != value:
            raise RuntimeError(f"History metadata mismatch for {role.run_id} {key}: {row.get(key)!r} != {value!r}")
    expected_summary = summary_metadata(retired=retired)
    actual_summary = {key: run.summary.get(key) for key in expected_summary}
    if actual_summary != expected_summary:
        raise RuntimeError(
            f"Summary metadata mismatch for {role.run_id}: {actual_summary!r} != {expected_summary!r}"
        )
    return {
        "run_id": role.run_id,
        "state": run.state,
        "history_step": row["_step"],
        "remote_name": remote.name,
        "remote_size": remote.size,
        "remote_sha256": remote_sha256,
    }


def log_role(role: Role) -> None:
    path = ROOT / "videos" / "final" / role.video_name
    if not path.is_file():
        raise FileNotFoundError(path)
    probe(path)
    if sha256(path) != role.video_sha256:
        raise RuntimeError(f"Local video SHA mismatch: {path}")

    api = wandb.Api(timeout=90)
    before = fresh_run(api, role.run_id)
    if before.state != "running":
        raise RuntimeError(f"Refusing upload: {role.run_id} state is {before.state!r}, expected running")
    if matching_files(before, "evaluation_distill_label_teacher"):
        raise RuntimeError(f"Refusing duplicate upload: {role.run_id} already has {MEDIA_KEY}")
    old_files = matching_files(before, "evaluation_motion_generator_teacher")
    if len(old_files) != 1 or role.old_video_sha256[:20] not in old_files[0].name:
        raise RuntimeError(f"Old generator media identity mismatch for {role.run_id}: {[f.name for f in old_files]}")

    payload = {
        MEDIA_KEY: wandb.Video(str(path), format="mp4", fps=50),
        "evaluation/teacher_role": "distill_label_teacher",
        "evaluation/teacher_checkpoint_ref": TEACHER_REF,
        "evaluation/teacher_checkpoint_sha256": TEACHER_SHA256,
        "evaluation/checkpoint_completed_iteration": 7999,
        "evaluation/checkpoint_step": 8000,
        "evaluation/motion_generator_teacher_ref": GENERATOR_REF,
        "evaluation/motion_generator_teacher_sha256": GENERATOR_SHA256,
        "evaluation/distill_label_teacher_same_as_generator": False,
        "evaluation/source_snapshot_id": SOURCE_ID,
        "evaluation/source_manifest_sha256": SOURCE_SHA256,
        "evaluation/motion_clip": role.motion_clip,
        "evaluation/motion_sha256": role.motion_sha256,
        "evaluation/object_map_sha256": role.object_map_sha256,
        "evaluation/object_urdf_sha256": role.object_urdf_sha256,
        "evaluation/object_mesh_sha256": role.object_mesh_sha256,
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
        "evaluation/video_source_frame_start": 4,
        "evaluation/video_source_frame_end": 318,
        "evaluation/video_num_frames": 315,
        "evaluation/video_fps": 50,
        "evaluation/video_duration_seconds": 6.3,
        "evaluation/video_sha256": role.video_sha256,
        "evaluation/video_size_bytes": path.stat().st_size,
        "evaluation/primary_teacher_key": MEDIA_KEY,
    }

    secondary = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=role.run_id,
        resume="must",
        reinit=True,
        settings=wandb.Settings(x_primary=False, x_update_finish_state=False),
    )
    try:
        secondary.log(payload)
    finally:
        secondary.finish(quiet=True)

    # x_primary=False intentionally prevents the media writer from owning the
    # live run's summary.  Commit role metadata through the Public API instead;
    # this mutates only the named summary keys and cannot finish the run.
    public_run = fresh_run(wandb.Api(timeout=90), role.run_id)
    if public_run.state != "running":
        raise RuntimeError(f"Run state changed after media upload: {role.run_id} -> {public_run.state!r}")
    public_run.summary.update(summary_metadata(retired=False))


def retire_old_media() -> list[dict[str, object]]:
    retired: list[dict[str, object]] = []
    api = wandb.Api(timeout=90)
    for role in ROLES:
        run = fresh_run(api, role.run_id)
        if run.state != "running":
            raise RuntimeError(f"Refusing retirement: {role.run_id} state changed to {run.state!r}")
        old_files = matching_files(run, "evaluation_motion_generator_teacher")
        if len(old_files) != 1 or role.old_video_sha256[:20] not in old_files[0].name:
            raise RuntimeError(f"Old media identity mismatch before retirement for {role.run_id}")
        old_file = old_files[0]
        record = {"run_id": role.run_id, "name": old_file.name, "size": old_file.size, "sha256": role.old_video_sha256}
        old_file.delete()
        run.summary.update(summary_metadata(retired=True))
        retired.append(record)

    fresh_api = wandb.Api(timeout=90)
    for role in ROLES:
        run = fresh_run(fresh_api, role.run_id)
        if run.state != "running":
            raise RuntimeError(f"Run state changed after retirement: {role.run_id} -> {run.state!r}")
        if matching_files(run, "evaluation_motion_generator_teacher"):
            raise RuntimeError(f"Old generator media still present after retirement: {role.run_id}")
        logical_media = [
            file_obj.name
            for file_obj in run.files()
            if file_obj.name.endswith(".mp4")
            and any(token in file_obj.name for token in ("replay", "evaluation_student", "evaluation_distill_label_teacher"))
        ]
        if len(logical_media) != 3:
            raise RuntimeError(f"Expected exactly three retained logical videos for {role.run_id}, got {logical_media}")
    return retired


def main() -> None:
    for role in ROLES:
        log_role(role)

    verified = [verify_remote(role, retired=False) for role in ROLES]
    retired = retire_old_media()
    final_verified = [verify_remote(role, retired=True) for role in ROLES]
    output = {"verified_before_retirement": verified, "retired": retired, "final_verified": final_verified}
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
