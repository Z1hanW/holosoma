"""Path resolution utilities for package data files."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 9):
    from importlib.resources import files
else:
    from importlib_resources import files


def _repo_root_data_fallback(path_obj: Path) -> Path | None:
    """Resolve relocated repo-local DS assets embedded as absolute paths."""
    parts = path_obj.expanduser().parts
    try:
        repo_root = Path(__file__).resolve().parents[4]
    except IndexError:
        return None

    for idx, part in enumerate(parts[:-1]):
        if part != "data":
            continue
        tail = parts[idx:]
        if len(tail) < 2 or tail[1] != "ds_box_data":
            continue

        candidate = repo_root.joinpath(*tail)
        if candidate.exists():
            return candidate

    return None


def resolve_data_file_path(file_path: str) -> str:
    """
    Resolve a data file path.

    Handles multiple path formats:
    1. S3 paths: "s3://bucket/path/to/file.npz" -> return as-is
    2. Package data paths: "holosoma/data/.../file.npz" -> resolved via importlib.resources
    3. Absolute paths: "/path/to/file.npz" -> returned as-is
    4. Relative paths: "./data/file.npz" or "../data/file.npz" -> resolved relative to CWD

    Args:
        file_path: The path to resolve

    Returns:
        The resolved absolute path as a string

    Examples:
        >>> # Package data
        >>> path = resolve_data_file_path("holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz")
        >>> print(path)
        /path/to/installed/holosoma/data/motions/g1_29dof/whole_body_tracking/motion_crawl_slope.npz

        >>> # User's custom file (absolute)
        >>> path = resolve_data_file_path("/home/user/my_motions/custom.npz")
        >>> print(path)
        /home/user/my_motions/custom.npz

        >>> # User's custom file (relative)
        >>> path = resolve_data_file_path("./my_data/custom.npz")
        >>> print(path)
        /current/working/dir/my_data/custom.npz
    """
    # 1. If it's an S3 path, return as-is
    if file_path.startswith("s3://"):
        return file_path
    # 2. If starts with "holosoma/data", use importlib.resources
    if file_path.startswith("holosoma/data"):
        suffix = file_path[13:].lstrip("/")  # Remove "holosoma/data" and leading slashes
        base = files("holosoma.data")
        return str(base / suffix) if suffix else str(base)

    # 3. If it's an absolute path, return as-is unless it is a relocated repo-local DS asset.
    path_obj = Path(file_path)
    if path_obj.is_absolute():
        if path_obj.exists():
            return file_path
        fallback = _repo_root_data_fallback(path_obj)
        if fallback is not None:
            return str(fallback)
        return file_path

    # 4. Otherwise, resolve relative path to absolute (relative to CWD)
    resolved = path_obj.resolve()
    return str(resolved)
