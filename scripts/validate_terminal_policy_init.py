#!/usr/bin/env python3
"""Validate one private controller-local terminal policy initializer."""

from __future__ import annotations

import argparse
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from holosoma.utils.checkpoint_validation import (
    CheckpointFileSecurityContract,
    load_verified_torch_checkpoint,
)
from holosoma.utils.policy_init_preflight import (
    validate_policy_init_terminal_source_payload,
)
from holosoma.utils.training_provenance import validate_training_provenance


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_SNAPSHOT_ID_RE = re.compile(r"^src-[0-9a-f]{64}$")


def _absolute_lexical(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _directory_identity(result: os.stat_result) -> tuple[int, ...]:
    return (
        int(result.st_dev),
        int(result.st_ino),
        int(result.st_mode),
        int(result.st_nlink),
        int(result.st_uid),
        int(result.st_gid),
        int(result.st_mtime_ns),
        int(result.st_ctime_ns),
    )


def _require_real_private_directory(
    path: Path,
    *,
    expected_mode: int,
    role: str,
) -> os.stat_result:
    try:
        result = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"{role} is missing: {path}") from exc
    if (
        not stat.S_ISDIR(result.st_mode)
        or stat.S_ISLNK(result.st_mode)
        or int(result.st_uid) != os.geteuid()
        or stat.S_IMODE(result.st_mode) != expected_mode
        or path.resolve() != path
    ):
        raise ValueError(
            f"{role} must be a real current-UID directory with mode {expected_mode:04o}: {path}"
        )
    return result


def _validate_fresh_source_provenance(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = checkpoint.get("training_provenance")
    if not isinstance(provenance, dict):
        raise ValueError(
            "Fresh terminal policy-init source must contain finalized training provenance."
        )
    canonical = validate_training_provenance(
        provenance,
        require_finalized=True,
    )
    if canonical.get("policy_init_enabled") is not False:
        raise ValueError(
            "Fresh terminal policy-init source unexpectedly used policy initialization."
        )
    if canonical.get("training_resume_enabled") is not False:
        raise ValueError(
            "Fresh terminal policy-init source unexpectedly used full training resume."
        )
    return canonical


def validate_controller_terminal_policy_init(
    checkpoint_path: Path,
    *,
    cache_root: Path,
    expected_sha256: str,
    required_target: int,
    expected_world_size: int | None = None,
    expected_wandb_run_path: str | None = None,
    require_fresh_source: bool = False,
    expected_source_snapshot_id: str | None = None,
) -> dict[str, Any]:
    """Validate private file identity and the complete terminal source proof."""

    if not isinstance(expected_sha256, str) or _SHA256_RE.fullmatch(expected_sha256) is None:
        raise ValueError("Expected checkpoint SHA256 must be lowercase hexadecimal.")
    if type(required_target) is not int or required_target < 1:
        raise ValueError("Required terminal target must be a positive integer.")
    if expected_world_size is not None and (
        type(expected_world_size) is not int or expected_world_size < 1
    ):
        raise ValueError("Expected world size must be a positive integer.")
    if expected_wandb_run_path is not None and (
        not isinstance(expected_wandb_run_path, str)
        or not expected_wandb_run_path
        or expected_wandb_run_path != expected_wandb_run_path.strip()
    ):
        raise ValueError("Expected W&B run path must be one non-empty stripped string.")
    if type(require_fresh_source) is not bool:
        raise ValueError("require_fresh_source must be boolean.")
    if expected_source_snapshot_id is not None:
        if (
            not isinstance(expected_source_snapshot_id, str)
            or _SOURCE_SNAPSHOT_ID_RE.fullmatch(expected_source_snapshot_id) is None
        ):
            raise ValueError(
                "Expected source snapshot ID must have format src-<64 lowercase hexadecimal characters>."
            )
        if not require_fresh_source:
            raise ValueError(
                "Expected source snapshot ID requires fresh-source provenance validation."
            )

    checkpoint_path = _absolute_lexical(checkpoint_path)
    cache_root = _absolute_lexical(cache_root)
    if checkpoint_path.name != f"{expected_sha256}.pt":
        raise ValueError(
            "Controller checkpoint basename must equal its authenticated content digest."
        )
    if checkpoint_path.parent.parent != cache_root:
        raise ValueError(
            "Controller checkpoint must be exactly one sealed object directory below its private cache root."
        )

    root_before = _require_real_private_directory(
        cache_root,
        expected_mode=0o700,
        role="controller checkpoint cache root",
    )
    object_before = _require_real_private_directory(
        checkpoint_path.parent,
        expected_mode=0o500,
        role="controller checkpoint sealed object directory",
    )
    checkpoint, actual_sha256 = load_verified_torch_checkpoint(
        checkpoint_path,
        expected_sha256=expected_sha256,
        map_location="cpu",
        file_security=CheckpointFileSecurityContract(
            owner_uid=os.geteuid(),
            mode=0o400,
            link_count=1,
            minimum_size=1,
            bind_pathname=True,
        ),
    )
    if actual_sha256 != expected_sha256:
        raise RuntimeError("Verified checkpoint loader returned an inconsistent digest.")
    root_after = _require_real_private_directory(
        cache_root,
        expected_mode=0o700,
        role="controller checkpoint cache root",
    )
    object_after = _require_real_private_directory(
        checkpoint_path.parent,
        expected_mode=0o500,
        role="controller checkpoint sealed object directory",
    )
    if _directory_identity(root_before) != _directory_identity(root_after):
        raise RuntimeError("Controller checkpoint cache root changed during validation.")
    if _directory_identity(object_before) != _directory_identity(object_after):
        raise RuntimeError("Controller checkpoint object directory changed during validation.")
    if not isinstance(checkpoint, dict):
        raise ValueError("Terminal policy-init checkpoint payload must be a mapping.")

    terminal_state = validate_policy_init_terminal_source_payload(
        checkpoint,
        required_target=required_target,
    )
    if (
        expected_world_size is not None
        and terminal_state["world_size"] != expected_world_size
    ):
        raise ValueError(
            "Terminal policy-init source world size differs from the expected canary topology: "
            f"checkpoint={terminal_state['world_size']}, expected={expected_world_size}."
        )
    if expected_wandb_run_path is not None:
        actual_run_path = checkpoint.get("wandb_run_path")
        if actual_run_path != expected_wandb_run_path:
            raise ValueError(
                "Terminal policy-init source W&B identity differs from the selected canary run."
            )
    if require_fresh_source:
        fresh_provenance = _validate_fresh_source_provenance(checkpoint)
        if (
            expected_source_snapshot_id is not None
            and fresh_provenance.get("source_snapshot_id")
            != expected_source_snapshot_id
        ):
            raise ValueError(
                "Terminal policy-init source snapshot identity differs from the expected immutable source snapshot."
            )
    return terminal_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--require-terminal-target", required=True, type=int)
    parser.add_argument("--expected-world-size", type=int)
    parser.add_argument("--expected-wandb-run-path")
    parser.add_argument("--require-fresh-source", action="store_true")
    parser.add_argument("--expected-source-snapshot-id")
    args = parser.parse_args()

    terminal_state = validate_controller_terminal_policy_init(
        args.checkpoint,
        cache_root=args.cache_root,
        expected_sha256=args.expected_sha256,
        required_target=args.require_terminal_target,
        expected_world_size=args.expected_world_size,
        expected_wandb_run_path=args.expected_wandb_run_path,
        require_fresh_source=args.require_fresh_source,
        expected_source_snapshot_id=args.expected_source_snapshot_id,
    )
    print(
        "[INFO] controller_terminal_policy_init_verified "
        f"completed_iteration={terminal_state['completed_iteration']} "
        f"next_iteration={terminal_state['next_iteration']} "
        f"target_iteration={terminal_state['run_target_iteration']} "
        f"world_size={terminal_state['world_size']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
