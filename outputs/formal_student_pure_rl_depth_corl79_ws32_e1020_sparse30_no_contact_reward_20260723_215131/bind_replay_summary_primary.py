#!/usr/bin/env python3
"""Bind vis/replay in the exact live primary W&B service, then detach safely."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


RUN_ID = "ptabkuyq"
ENTITY = "zihanw22"
PROJECT = "carry-any"
MEDIA_KEY = "vis/replay"
HISTORY_STEP = 20
HISTORY_PATH = "media/videos/vis/replay_20_43de32d93bbcd389000b.mp4"
VIDEO_SHA256 = "43de32d93bbcd389000bfab335521ebbc6c33a094ffb2a0fbbcd4081238da761"
VIDEO_SIZE = 1_533_187
MANIFEST_SHA256 = "2765b91f3bd4c07883581af4ff1ca01475ebc062f00bf920510636d441b8b1dc"


def service_token(endpoint: str) -> str:
    if endpoint.startswith("unix="):
        return f"3-{os.getpid()}-unix-{endpoint.removeprefix('unix=')}"
    if endpoint.startswith("sock="):
        port = int(endpoint.removeprefix("sock="))
        return f"3-{os.getpid()}-tcp-localhost-{port}"
    port_file = Path(endpoint)
    lines = port_file.read_text(encoding="utf-8").splitlines()
    if not lines or lines[-1] != "EOF":
        raise RuntimeError("W&B service port file is incomplete")
    for line in lines:
        if line.startswith("unix="):
            return f"3-{os.getpid()}-unix-{line.removeprefix('unix=')}"
        if line.startswith("sock="):
            port = int(line.removeprefix("sock="))
            return f"3-{os.getpid()}-tcp-localhost-{port}"
    raise RuntimeError("W&B service port file has no supported endpoint")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            f"usage: {sys.argv[0]} WANDB_CORE_PORT_FILE|unix=PATH|sock=PORT"
        )
    os.environ["WANDB_SERVICE"] = service_token(sys.argv[1])

    # Import only after the exact live service token is installed.
    import wandb

    run = wandb._attach(run_id=RUN_ID)
    if run is None:
        raise RuntimeError("Unable to attach to the exact live primary run")
    if run.id != RUN_ID or run.entity != ENTITY or run.project != PROJECT:
        raise RuntimeError(
            f"Attached to the wrong run: {run.entity}/{run.project}/{run.id}"
        )
    media = {
        "_type": "video-file",
        "path": HISTORY_PATH,
        "sha256": VIDEO_SHA256,
        "size": VIDEO_SIZE,
    }
    run.summary.update(
        {
            MEDIA_KEY: media,
            "replay_history/history_step": HISTORY_STEP,
            "replay_history/video_sha256": VIDEO_SHA256,
            "replay_history/manifest_sha256": MANIFEST_SHA256,
            "replay_history/history_backed": True,
        }
    )

    # This synchronous request is an ordering barrier for the preceding summary
    # updates.  Do not call finish(): that would send an exit record to the
    # live primary stream.  os._exit bypasses W&B's process atexit hook.
    run.status()
    sys.stdout.write(
        json.dumps(
            {
                "status": "ok",
                "run": f"{ENTITY}/{PROJECT}/{RUN_ID}",
                "history_step": HISTORY_STEP,
                "history_path": HISTORY_PATH,
            },
            sort_keys=True,
        )
        + "\n"
    )
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
