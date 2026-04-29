#!/usr/bin/env python3
"""Resolve an ONNX policy from a local path or W&B run file."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


ENTITY = os.environ.get("MJ_WANDB_ENTITY", "zihanw22")
PROJECT = os.environ.get("MJ_WANDB_PROJECT", "boxer")
DEFAULT_RUN = os.environ.get("MJ_DEMO_MODEL_RUN", f"{ENTITY}/{PROJECT}/shoo7sr1")
CACHE_ROOT = Path(os.environ.get("MJ_WANDB_CACHE", "logs/wandb_assets/run_files"))


def _remote(ref: str) -> bool:
    return ref.startswith("wandb://") or ref.startswith("https://wandb.ai/")


def _parse_ref(ref: str) -> tuple[str, str]:
    if ref.startswith("wandb://"):
        parts = ref[len("wandb://") :].split("/")
        run_idx = 3 if len(parts) > 4 and parts[2] == "runs" else 2
        if len(parts) <= run_idx + 1:
            raise SystemExit("[ERROR] expected wandb://<entity>/<project>/<run_id>/<file>")
        return f"{parts[0]}/{parts[1]}/{parts[run_idx]}", "/".join(parts[run_idx + 1 :])

    clean = ref.split("#", 1)[0].split("?", 1)[0]
    parts = clean[len("https://wandb.ai/") :].split("/")
    if len(parts) < 4 or parts[2] != "runs":
        raise SystemExit("[ERROR] expected https://wandb.ai/<entity>/<project>/runs/<run_id>[/files/<file>]")
    filename = "/".join(parts[5:] if len(parts) >= 6 and parts[4] == "files" else parts[4:])
    if not filename:
        raise SystemExit("[ERROR] W&B run URL must include a file")
    return f"{parts[0]}/{parts[1]}/{parts[3]}", filename


def _local_default(path: Path) -> tuple[str, str]:
    parts = path.parts
    if "wandb_runs" in parts:
        return f"{ENTITY}/{PROJECT}/{parts[parts.index('wandb_runs') + 1]}", path.name
    return DEFAULT_RUN, path.name


def _download(run_path: str, filename: str, dest: Path) -> Path:
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise SystemExit("[ERROR] missing python package 'wandb'; install it or put the ONNX file locally") from exc

    run = wandb.Api(timeout=30).run(run_path)
    if filename.lower() in {"latest", "latest.onnx"}:
        onnx_files = [file_obj for file_obj in run.files() if str(file_obj.name).endswith(".onnx")]
        if not onnx_files:
            raise SystemExit(f"[ERROR] no ONNX files found in W&B run {run_path}")
        filename = str(max(onnx_files, key=lambda file_obj: ((file_obj.updated_at or ""), file_obj.name)).name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    downloaded = run.file(filename).download(root=str(dest.parent), replace=True)
    downloaded_path = Path(downloaded.name)
    if not downloaded_path.is_absolute():
        cwd_candidate = downloaded_path.resolve()
        downloaded_path = cwd_candidate if cwd_candidate.is_file() else (dest.parent / downloaded_path).resolve()
    if downloaded_path.resolve() != dest.resolve() and downloaded_path.is_file():
        shutil.copy2(downloaded_path, dest)
    if not dest.is_file():
        raise SystemExit(f"[ERROR] W&B download did not create expected file: {dest}")
    print(f"[INFO] fetched {dest} from W&B {run_path}/{filename}", file=sys.stderr)
    return dest


def main() -> None:
    ref = sys.argv[1] if len(sys.argv) > 1 else "logs/wandb_runs/shoo7sr1/model_29999.onnx"
    if _remote(ref):
        run_path, filename = _parse_ref(ref)
        path = CACHE_ROOT.expanduser().resolve() / run_path / filename
    else:
        path = Path(ref).expanduser()
        if path.is_file() and path.stat().st_size > 0:
            print(path)
            return
        run_path, filename = _local_default(path)
    print(_download(run_path, filename, path))


if __name__ == "__main__":
    main()
