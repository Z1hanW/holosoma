#!/usr/bin/env python3
"""Hold one bounded, no-follow flock until stdin closes."""

from __future__ import annotations

import argparse
import fcntl
import os
import stat
import sys
import time
from pathlib import Path


def _open_or_create_directory_no_follow(path: Path) -> int:
    """Open an absolute directory one component at a time without symlinks."""

    root = Path(os.path.abspath(path.expanduser()))
    if root == Path(root.anchor):
        raise SystemExit(f"[ERROR] Refusing filesystem root as launch lock root: {root}")
    flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | os.O_NOFOLLOW
    )
    descriptor = os.open(root.anchor, flags)
    try:
        for component in root.parts[1:]:
            try:
                child_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    # A concurrent creator won.  The no-follow open below still
                    # authenticates the resulting directory before it is used.
                    pass
                child_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    args = parser.parse_args()

    if Path(args.name).name != args.name or args.name in {".", ".."}:
        raise SystemExit("[ERROR] Lock name must be one safe path component.")
    if not (0.0 <= args.timeout_seconds <= 60.0):
        raise SystemExit("[ERROR] Lock timeout must be in [0, 60] seconds.")
    if not hasattr(os, "O_NOFOLLOW"):
        raise SystemExit("[ERROR] O_NOFOLLOW is required for teacher rollout launch locking.")

    root = Path(os.path.abspath(args.root.expanduser()))
    try:
        root_descriptor = _open_or_create_directory_no_follow(root)
    except OSError as exc:
        raise SystemExit(f"[ERROR] Unable to open no-follow lock root {root}: {exc}") from exc
    try:
        try:
            descriptor = os.open(
                args.name,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW,
                0o600,
                dir_fd=root_descriptor,
            )
        except OSError as exc:
            raise SystemExit(f"[ERROR] Unable to open no-follow launch lock {args.name!r}: {exc}") from exc
    finally:
        os.close(root_descriptor)

    try:
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode) or identity.st_nlink != 1:
            raise SystemExit("[ERROR] Refusing non-regular or multiply-linked launch lock.")
        deadline = time.monotonic() + args.timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise SystemExit("[ERROR] Teacher rollout launch scope is already locked.")
                time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        print("LOCKED", flush=True)
        sys.stdin.buffer.read()
    finally:
        os.close(descriptor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
