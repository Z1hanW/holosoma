#!/usr/bin/env python3
"""Resolve native MuJoCo demo assets from local files or W&B."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


DEFAULT_ENTITY = os.environ.get("MJ_WANDB_ENTITY", "zihanw22")
DEFAULT_PROJECT = os.environ.get("MJ_WANDB_PROJECT", "boxer")
DEFAULT_ASSET_ARTIFACT = os.environ.get(
    "MJ_DEMO_ASSET_ARTIFACT",
    f"{DEFAULT_ENTITY}/{DEFAULT_PROJECT}/holosoma-mj-demo-assets:latest",
)
DEFAULT_MODEL_RUN = os.environ.get("MJ_DEMO_MODEL_RUN", f"{DEFAULT_ENTITY}/{DEFAULT_PROJECT}/shoo7sr1")
DEFAULT_CACHE_ROOT = Path(os.environ.get("MJ_WANDB_CACHE", "logs/wandb_assets"))
_ARTIFACT_ROOT: Path | None = None


def _log(message: str) -> None:
    print(f"[INFO] {message}", file=sys.stderr)


def _is_remote(ref: str) -> bool:
    return ref.startswith("wandb://") or ref.startswith("https://wandb.ai/")


def _import_wandb():
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise SystemExit("[ERROR] Missing python package 'wandb'; install it or prefetch the assets locally.") from exc
    return wandb


def _artifact_root() -> Path:
    global _ARTIFACT_ROOT
    if _ARTIFACT_ROOT is not None:
        return _ARTIFACT_ROOT
    wandb = _import_wandb()
    cache_root = DEFAULT_CACHE_ROOT.expanduser().resolve()
    artifact_root = cache_root / "artifacts" / DEFAULT_ASSET_ARTIFACT.replace("/", "_").replace(":", "_")
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact = wandb.Api(timeout=30).artifact(DEFAULT_ASSET_ARTIFACT)
    downloaded = artifact.download(root=str(artifact_root))
    _ARTIFACT_ROOT = Path(downloaded).resolve()
    return _ARTIFACT_ROOT


def _repo_relative(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    try:
        return resolved.relative_to(cwd)
    except ValueError:
        return path


def _copy_artifact_file(relative_name: str, destination: Path) -> None:
    root = _artifact_root()
    source = root / relative_name
    if not source.is_file():
        raise SystemExit(f"[ERROR] W&B artifact {DEFAULT_ASSET_ARTIFACT} does not contain {relative_name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    _log(f"Fetched {destination} from W&B artifact {DEFAULT_ASSET_ARTIFACT}")


def _ensure_artifact_file(destination: Path, relative_name: str) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    _copy_artifact_file(relative_name, destination)


def _parse_remote_file_ref(ref: str) -> tuple[str, str]:
    if ref.startswith("wandb://"):
        parts = ref[len("wandb://") :].split("/")
        if len(parts) < 4:
            raise SystemExit("[ERROR] Expected wandb://<entity>/<project>/<run_id>/<file>")
        run_idx = 3 if len(parts) > 4 and parts[2] == "runs" else 2
        if len(parts) <= run_idx + 1:
            raise SystemExit("[ERROR] W&B file reference must include a filename")
        return f"{parts[0]}/{parts[1]}/{parts[run_idx]}", "/".join(parts[run_idx + 1 :])

    clean = ref.split("#", 1)[0].split("?", 1)[0]
    parts = clean[len("https://wandb.ai/") :].split("/")
    if len(parts) < 4 or parts[2] != "runs":
        raise SystemExit("[ERROR] Expected https://wandb.ai/<entity>/<project>/runs/<run_id>[/files/<file>]")
    filename_parts = parts[5:] if len(parts) >= 6 and parts[4] == "files" else parts[4:]
    if not filename_parts:
        raise SystemExit("[ERROR] W&B run URL must include a filename")
    return f"{parts[0]}/{parts[1]}/{parts[3]}", "/".join(filename_parts)


def _latest_onnx_name(run: object) -> str:
    onnx_files = [file_obj for file_obj in run.files() if str(file_obj.name).endswith(".onnx")]
    if not onnx_files:
        raise SystemExit("[ERROR] No ONNX files found in W&B run")
    return str(max(onnx_files, key=lambda file_obj: ((file_obj.updated_at or ""), file_obj.name)).name)


def _download_run_file(run_path: str, filename: str, destination: Path) -> Path:
    wandb = _import_wandb()
    api = wandb.Api(timeout=30)
    run = api.run(run_path)
    if filename.lower() in {"latest", "latest.onnx"}:
        filename = _latest_onnx_name(run)
    destination.parent.mkdir(parents=True, exist_ok=True)
    downloaded = run.file(filename).download(root=str(destination.parent), replace=True)
    downloaded_path = Path(downloaded.name)
    if not downloaded_path.is_absolute():
        cwd_candidate = downloaded_path.resolve()
        root_candidate = (destination.parent / downloaded_path).resolve()
        downloaded_path = cwd_candidate if cwd_candidate.is_file() else root_candidate
    if downloaded_path.resolve() != destination.resolve() and downloaded_path.is_file():
        shutil.copy2(downloaded_path, destination)
    if not destination.is_file():
        raise SystemExit(f"[ERROR] W&B download did not create expected file: {destination}")
    _log(f"Fetched {destination} from W&B run {run_path}/{filename}")
    return destination


def _infer_local_model_ref(path: Path) -> tuple[str, str]:
    parts = path.parts
    if "wandb_runs" in parts:
        run_id = parts[parts.index("wandb_runs") + 1]
        return f"{DEFAULT_ENTITY}/{DEFAULT_PROJECT}/{run_id}", path.name
    return DEFAULT_MODEL_RUN, path.name


def _resolve_model(model: str) -> Path:
    if _is_remote(model):
        run_path, filename = _parse_remote_file_ref(model)
        destination = DEFAULT_CACHE_ROOT.expanduser().resolve() / "run_files" / run_path / filename
        if destination.is_file() and destination.stat().st_size > 0:
            return destination
        return _download_run_file(run_path, filename, destination)

    path = Path(model).expanduser()
    if path.is_file() and path.stat().st_size > 0:
        return path
    run_path, filename = _infer_local_model_ref(path)
    return _download_run_file(run_path, filename, path)


def _ensure_clip_assets(clip: str, motion: str | None, object_urdf: str | None) -> None:
    clip = clip.removesuffix(".npz")
    if motion:
        motion_path = Path(motion).expanduser()
        _ensure_artifact_file(motion_path, str(_repo_relative(motion_path)))
    if object_urdf:
        urdf_path = Path(object_urdf).expanduser()
        _ensure_artifact_file(urdf_path, str(_repo_relative(urdf_path)))
        object_path = urdf_path.with_suffix(".obj")
        _ensure_artifact_file(object_path, f"data_demo/objects/{clip}.obj")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", required=True)
    parser.add_argument("--motion")
    parser.add_argument("--object-urdf")
    parser.add_argument("--model")
    parser.add_argument("--print-model", action="store_true")
    args = parser.parse_args()

    _ensure_clip_assets(args.clip, args.motion, args.object_urdf)
    if args.model:
        model_path = _resolve_model(args.model)
        if args.print_model:
            print(model_path)


if __name__ == "__main__":
    main()
