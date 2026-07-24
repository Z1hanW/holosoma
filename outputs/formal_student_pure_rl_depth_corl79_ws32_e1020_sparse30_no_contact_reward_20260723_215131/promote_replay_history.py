#!/usr/bin/env python3
"""Promote the reviewed Rule-90 replay from prebind summary to one history row."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile

import wandb


ENTITY = "zihanw22"
PROJECT = "carry-any"
RUN_ID = "ptabkuyq"
RUN_PATH = f"{ENTITY}/{PROJECT}/{RUN_ID}"
MEDIA_KEY = "vis/replay"
VIDEO = Path(
    "/home/ubuntu/FAR/holosoma/outputs/"
    "formal_student_pure_rl_depth_corl79_ws32_e1020_sparse30_no_contact_reward_"
    "20260723_215131/replay/capture/videos/0000_box_10/"
    "episode_0_1784843837.mp4"
)
VIDEO_SHA256 = "43de32d93bbcd389000bfab335521ebbc6c33a094ffb2a0fbbcd4081238da761"
VIDEO_SIZE = 1_533_187
PREBIND_PATH = "media/videos/vis/replay_summary_43de32d93bbcd389000b.mp4"
MANIFEST_SHA256 = "2765b91f3bd4c07883581af4ff1ca01475ebc062f00bf920510636d441b8b1dc"
RUN_CONTRACT_SHA256 = "804b56ad8fd740e6282c8fc96159d814f4aa26706ccb18238cfbd578c71f3cec"
RANK_SHARD_MANIFEST_SHA256 = (
    "6861cb9b62547c8d16f68d7759344805b9684a6335fe32923f90f8acd54d799c"
)
SOURCE_SNAPSHOT_ID = (
    "src-a8879814e247e78b0e686fb8bdf94b985d008b2960aa187b482ef8de7b644399"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fresh_run() -> wandb.apis.public.Run:
    return wandb.Api(timeout=90).run(RUN_PATH)


def media_value(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        result = dict(value)
    elif hasattr(value, "items"):
        result = dict(value.items())  # type: ignore[union-attr]
    else:
        raise RuntimeError(f"Unexpected W&B media value: {value!r}")
    if result.get("_type") != "video-file":
        raise RuntimeError(f"Expected video-file media, got {result!r}")
    return result


def replay_history_rows(run: wandb.apis.public.Run) -> list[dict[str, object]]:
    rows = list(
        run.scan_history(
            keys=[
                "_step",
                MEDIA_KEY,
                "replay_history/video_sha256",
                "replay_history/manifest_sha256",
            ],
            page_size=1000,
        )
    )
    return [row for row in rows if row.get(MEDIA_KEY)]


def replay_files(run: wandb.apis.public.Run) -> list[wandb.apis.public.File]:
    return sorted(
        [
            file_obj
            for file_obj in run.files()
            if file_obj.name.startswith("media/videos/vis/replay")
            and file_obj.name.endswith(".mp4")
        ],
        key=lambda file_obj: file_obj.name,
    )


def verify_remote_file(file_obj: wandb.apis.public.File) -> None:
    if file_obj.size != VIDEO_SIZE:
        raise RuntimeError(
            f"Remote replay size mismatch: {file_obj.name} -> {file_obj.size}"
        )
    with tempfile.TemporaryDirectory(prefix="ptabkuyq-replay-verify-") as temp_root:
        downloaded = file_obj.download(root=temp_root, replace=True)
        path = Path(downloaded.name)
        if path.stat().st_size != VIDEO_SIZE or sha256(path) != VIDEO_SHA256:
            raise RuntimeError(f"Remote replay bytes mismatch: {file_obj.name}")


def main() -> None:
    if (
        not VIDEO.is_file()
        or VIDEO.stat().st_size != VIDEO_SIZE
        or sha256(VIDEO) != VIDEO_SHA256
    ):
        raise RuntimeError("Reviewed local replay bytes changed before promotion")

    before = fresh_run()
    if before.state != "running":
        raise RuntimeError(f"Refusing promotion while run state is {before.state!r}")
    if replay_history_rows(before):
        raise RuntimeError("Refusing duplicate vis/replay history promotion")
    before_summary = media_value(before.summary.get(MEDIA_KEY))
    if (
        before_summary.get("path") != PREBIND_PATH
        or before_summary.get("sha256") != VIDEO_SHA256
        or int(before_summary.get("size", -1)) != VIDEO_SIZE
    ):
        raise RuntimeError(f"Prebind summary identity mismatch: {before_summary!r}")
    before_files = replay_files(before)
    if [file_obj.name for file_obj in before_files] != [PREBIND_PATH]:
        raise RuntimeError(
            f"Expected only the exact prebind replay file, got {before_files!r}"
        )

    payload = {
        MEDIA_KEY: wandb.Video(str(VIDEO), format="mp4"),
        "replay_history/video_sha256": VIDEO_SHA256,
        "replay_history/video_size_bytes": VIDEO_SIZE,
        "replay_history/manifest_sha256": MANIFEST_SHA256,
        "replay_history/run_contract_sha256": RUN_CONTRACT_SHA256,
        "replay_history/rank_shard_manifest_sha256": RANK_SHARD_MANIFEST_SHA256,
        "replay_history/source_snapshot_id": SOURCE_SNAPSHOT_ID,
        "replay_history/motion_clip_id": "box_10",
        "replay_history/world_size": 32,
        "replay_history/visual_review_passed": True,
    }
    secondary = wandb.init(
        entity=ENTITY,
        project=PROJECT,
        id=RUN_ID,
        resume="must",
        reinit=True,
        settings=wandb.Settings(
            x_primary=False,
            x_update_finish_state=False,
            console="off",
        ),
    )
    try:
        secondary.log(payload)
    finally:
        secondary.finish(quiet=True)

    after_history = fresh_run()
    if after_history.state != "running":
        raise RuntimeError(
            f"Run state changed during history promotion: {after_history.state!r}"
        )
    rows = replay_history_rows(after_history)
    if len(rows) != 1:
        raise RuntimeError(f"Expected one vis/replay history row, got {len(rows)}")
    row = rows[0]
    if (
        row.get("replay_history/video_sha256") != VIDEO_SHA256
        or row.get("replay_history/manifest_sha256") != MANIFEST_SHA256
    ):
        raise RuntimeError(f"Replay history metadata mismatch: {row!r}")
    history_media = media_value(row[MEDIA_KEY])
    history_path = history_media.get("path")
    if (
        not isinstance(history_path, str)
        or history_path == PREBIND_PATH
        or history_media.get("sha256") != VIDEO_SHA256
        or int(history_media.get("size", -1)) != VIDEO_SIZE
    ):
        raise RuntimeError(f"History media identity mismatch: {history_media!r}")
    matching_history_file = [
        file_obj
        for file_obj in replay_files(after_history)
        if file_obj.name == history_path
    ]
    if len(matching_history_file) != 1:
        raise RuntimeError(f"History replay file is missing: {history_path!r}")
    verify_remote_file(matching_history_file[0])

    # Bind summary to the already verified history media without making the
    # secondary writer primary or changing the live run's finish state.
    after_history.summary[MEDIA_KEY] = history_media
    after_history.summary["replay_history/history_step"] = int(row["_step"])
    after_history.summary["replay_history/video_sha256"] = VIDEO_SHA256
    after_history.summary["replay_history/manifest_sha256"] = MANIFEST_SHA256
    after_history.summary.update()

    rebound = fresh_run()
    if media_value(rebound.summary.get(MEDIA_KEY)).get("path") != history_path:
        raise RuntimeError("Summary did not bind to the history-backed replay")
    old_files = [
        file_obj for file_obj in replay_files(rebound) if file_obj.name == PREBIND_PATH
    ]
    if len(old_files) != 1:
        raise RuntimeError("Exact summary-only replay file disappeared unexpectedly")
    old_files[0].delete()

    final = fresh_run()
    final_rows = replay_history_rows(final)
    final_files = replay_files(final)
    final_summary = media_value(final.summary.get(MEDIA_KEY))
    if final.state != "running":
        raise RuntimeError(f"Run state changed after cleanup: {final.state!r}")
    if len(final_rows) != 1 or len(final_files) != 1:
        raise RuntimeError(
            f"Final unique replay contract failed: rows={len(final_rows)} "
            f"files={[file_obj.name for file_obj in final_files]!r}"
        )
    if (
        final_summary.get("path") != history_path
        or final_files[0].name != history_path
    ):
        raise RuntimeError("Final summary/history/file paths do not agree")
    verify_remote_file(final_files[0])
    print(
        json.dumps(
            {
                "status": "ok",
                "run_path": RUN_PATH,
                "run_state": final.state,
                "history_step": final_rows[0]["_step"],
                "history_path": history_path,
                "video_sha256": VIDEO_SHA256,
                "video_size_bytes": VIDEO_SIZE,
                "history_rows": len(final_rows),
                "replay_files": len(final_files),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
