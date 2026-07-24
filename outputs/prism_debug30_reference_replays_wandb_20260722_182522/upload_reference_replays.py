#!/usr/bin/env python3
from __future__ import annotations

import datetime as dt
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import wandb


RUN_ROOT = Path(
    os.environ.get(
        "REFERENCE_RUN_ROOT",
        "/home/ubuntu/FAR/holosoma/outputs/prism_debug30_reference_replays_wandb_20260722_182522",
    )
)
MOTION_VIEW = Path(
    os.environ.get(
        "REFERENCE_MOTION_VIEW",
        "/home/ubuntu/FAR/holosoma/data/ds_as_data/"
        "prism_debug30_convexhull_allmesh_solid_box_bin_barrel_ball/"
        "_scientific_teacher64_single_slot/by-source/"
        "77738917deb60e578dc695841b3a07b10ad4f50371d3c0500474f41c78f71f90",
    )
)
EXPECTED_TOTAL = int(os.environ.get("REFERENCE_EXPECTED_TOTAL", "30"))
ENTITY = os.environ.get("REFERENCE_WANDB_ENTITY", "zihanw22")
PROJECT = os.environ.get("REFERENCE_WANDB_PROJECT", "carry-any")
PARENT_TRAINING_RUN = os.environ.get("REFERENCE_PARENT_TRAINING_RUN", "3m8lkcxf")
RUN_NAME = os.environ.get(
    "REFERENCE_WANDB_RUN_NAME",
    "3m8lkcxf_prism_debug30_reference_replays_20260722_182522",
)
BANK_NAME = os.environ.get(
    "REFERENCE_BANK_NAME", "prism_debug30_convexhull_allmesh_solid_box_bin_barrel_ball"
)
DATASET_LABEL = os.environ.get("REFERENCE_DATASET_LABEL", "prism_debug30")
ARTIFACT_SLUG = os.environ.get("REFERENCE_ARTIFACT_SLUG", "prism-debug30-reference-replays")
ARTIFACT_ALIAS = os.environ.get("REFERENCE_ARTIFACT_ALIAS", "prism-debug30")
RUN_TAGS = [
    value.strip()
    for value in os.environ.get(
        "REFERENCE_WANDB_TAGS",
        "reference-replay,prism-debug30,30-clips,3m8lkcxf-context",
    ).split(",")
    if value.strip()
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_video(path: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    frames = int(stream["nb_read_frames"])
    width = int(stream["width"])
    height = int(stream["height"])
    fps = float(Fraction(stream["r_frame_rate"]))
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration"))
    if frames <= 0 or width <= 0 or height <= 0 or duration <= 0:
        raise RuntimeError(f"Invalid video metadata for {path}: {stream}")
    return {
        "codec_name": str(stream["codec_name"]),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frames,
        "duration_s": duration,
    }


def collect_videos() -> list[dict[str, Any]]:
    expected = sorted(path.stem for path in MOTION_VIEW.glob("*.npz"))
    if len(expected) != EXPECTED_TOTAL:
        raise RuntimeError(f"Expected {EXPECTED_TOTAL} source motions, found {len(expected)}")

    rows: dict[str, tuple[Path, Path]] = {}
    for manifest_path in sorted(RUN_ROOT.glob("shard_*/manifest.tsv")):
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            fields = line.split("\t")
            if len(fields) != 6:
                raise RuntimeError(f"Malformed row in {manifest_path}: {line}")
            clip_id = fields[0]
            if clip_id in rows:
                raise RuntimeError(f"Duplicate clip across shards: {clip_id}")
            video_dir = Path(fields[5])
            videos = sorted(video_dir.glob("*.mp4"))
            if len(videos) != 1:
                raise RuntimeError(f"Expected one MP4 for {clip_id}, found {len(videos)}")
            rows[clip_id] = (Path(fields[4]), videos[0])

    if sorted(rows) != expected:
        missing = sorted(set(expected) - set(rows))
        extra = sorted(set(rows) - set(expected))
        raise RuntimeError(f"Clip cover mismatch: missing={missing}, extra={extra}")

    items: list[dict[str, Any]] = []
    for index, clip_id in enumerate(expected):
        motion_path, video_path = rows[clip_id]
        items.append(
            {
                "index": index,
                "clip_id": clip_id,
                "motion_sha256": sha256_file(motion_path.resolve()),
                "video_path": video_path,
                "video_relpath": str(video_path.relative_to(RUN_ROOT)),
                "video_sha256": sha256_file(video_path),
                "video_size_bytes": video_path.stat().st_size,
                "ffprobe": probe_video(video_path),
            }
        )
    return items


def main() -> None:
    items = collect_videos()
    source_map = MOTION_VIEW / "_clip_object_urdf_map.json"
    manifest_path = RUN_ROOT / "reference_replay_manifest.json"
    manifest_payload = {
        "version": 1,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "semantics": {
            "kind": "reference_motion_replay",
            "is_policy_rollout": False,
            "policy_checkpoint": None,
            "parent_training_run_context": PARENT_TRAINING_RUN,
            "randomization_disabled": True,
            "start_at_timestep_zero": True,
            "reset_noise_scale": 0.0,
        },
        "source": {
            "bank": BANK_NAME,
            "single_slot_view_digest": MOTION_VIEW.name,
            "object_map_sha256": sha256_file(source_map),
            "clip_count": len(items),
        },
        "videos": [
            {key: value for key, value in item.items() if key != "video_path"}
            for item in items
        ],
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_sha256 = sha256_file(manifest_path)

    wandb_dir = RUN_ROOT / "wandb_runtime"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        name=RUN_NAME,
        job_type="reference-replay",
        tags=RUN_TAGS,
        dir=str(wandb_dir),
        resume="never",
        config={
            "semantics": "reference_motion_replay_not_policy_rollout",
            "is_policy_rollout": False,
            "policy_checkpoint": None,
            "parent_training_run_context": PARENT_TRAINING_RUN,
            "source_bank": manifest_payload["source"]["bank"],
            "single_slot_view_digest": MOTION_VIEW.name,
            "source_object_map_sha256": manifest_payload["source"]["object_map_sha256"],
            "clip_count": len(items),
            "randomization_disabled": True,
            "start_at_timestep_zero": True,
            "reset_noise_scale": 0.0,
            "reference_replay_manifest_sha256": manifest_sha256,
        },
        settings=wandb.Settings(init_timeout=180),
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")

    run_url = run.url
    table = wandb.Table(
        columns=["index", "clip_id", "video", "frame_count", "duration_s", "sha256"]
    )
    for item in items:
        video = wandb.Video(str(item["video_path"]), format="mp4", caption=item["clip_id"])
        table.add_data(
            item["index"],
            item["clip_id"],
            video,
            item["ffprobe"]["frame_count"],
            item["ffprobe"]["duration_s"],
            item["video_sha256"],
        )
        run.log(
            {f"reference_replay/{item['index']:02d}_{item['clip_id']}": video},
            step=item["index"],
        )
        print(
            f"[UPLOAD] {item['index'] + 1:02d}/{len(items)} {item['clip_id']}",
            flush=True,
        )

    run.log({"reference_replays/table": table}, step=len(items))
    run.summary["reference_replay/count"] = len(items)
    run.summary["reference_replay/is_policy_rollout"] = False
    run.summary["reference_replay/manifest_sha256"] = manifest_sha256
    run.summary["reference_replay/source_view_digest"] = MOTION_VIEW.name

    artifact = wandb.Artifact(
        name=f"{ARTIFACT_SLUG}-{run.id}",
        type="reference-replay-bundle",
        description=(
            f"{len(items)} deterministic {DATASET_LABEL} reference motion replays; "
            "not policy rollouts."
        ),
        metadata={
            "clip_count": len(items),
            "is_policy_rollout": False,
            "parent_training_run_context": PARENT_TRAINING_RUN,
            "manifest_sha256": manifest_sha256,
        },
    )
    artifact.add_file(str(manifest_path), name="reference_replay_manifest.json")
    for item in items:
        artifact.add_file(
            str(item["video_path"]),
            name=f"videos/{item['index']:02d}_{item['clip_id']}.mp4",
        )
    run.log_artifact(artifact, aliases=["latest", ARTIFACT_ALIAS])
    run_id = run.id
    run.finish()

    result = {
        "entity": ENTITY,
        "project": PROJECT,
        "run_id": run_id,
        "run_url": run_url,
        "run_name": RUN_NAME,
        "clip_count": len(items),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
    }
    result_path = RUN_ROOT / "wandb_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
