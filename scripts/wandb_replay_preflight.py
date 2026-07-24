#!/usr/bin/env python3
"""Pre-bind and verify one reviewed replay video on a fresh W&B run.

This helper is deliberately independent from the training launcher.  It has two
subcommands:

``upload``
    Validate an immutable local manifest and its sole MP4, create the manifest's
    *fresh* W&B run with ``resume="never"``, and bind the MP4 directly to the
    run summary under the exact key ``vis/replay``.  It never writes a history
    row, so training iteration zero remains available to the formal run.

``verify``
    Re-run every local validation and prove through the W&B public API that the
    expected run contains exactly one non-empty ``video-file`` summary value,
    at ``vis/replay``, backed by exactly one matching MP4 run file.

Canonical v1 manifest schema (additional non-secret fields are permitted)::

    {
      "version": 1,
      "run": {
        "fresh": true,
        "entity": "entity",
        "project": "carry-any",
        "run_id": "fresh-id",
        "name": "descriptive-name"
      },
      "source": {
        "snapshot_id": "src-<64 lowercase hex>",
        "archive_sha256": "<64 lowercase hex>"
      },
      "inputs": {
        "world_size": 8,
        "motion_clip_id": "unscale__any_ball_29",
        "motion_npz_sha256": "<64 lowercase hex>",
        "object_map_sha256": "<64 lowercase hex>",
        "object_urdf_sha256": "<64 lowercase hex>",
        "object_mesh_sha256": "<64 lowercase hex>",
        "single_slot_source_digest": "<64 lowercase hex>",
        "single_slot_view_digest": "<64 lowercase hex>",
        "rank_shard_source_digest": "<64 lowercase hex>",
        "transition_digest": "<64 lowercase hex>"
      },
      "video": {
        "path": "/absolute/path/to/the-only-video.mp4",
        "sha256": "<64 lowercase hex>",
        "size_bytes": 123456,
        "ffprobe": {
          "width": 1280,
          "height": 720,
          "fps": 50.0,
          "frame_count": 327,
          "duration_s": 6.54
        }
      },
      "visual_review": {
        "passed": true,
        "video_sha256": "<same digest as video.sha256>",
        "reviewer": "reviewer-id",
        "reviewed_at_utc": "2026-07-17T12:00:00Z"
      }
    }

Every digest leaf below ``source``, ``inputs``, and ``video`` is mirrored to
both W&B config and summary.  Local paths and review notes are intentionally
not uploaded.  The manifest itself must not contain credential-like keys.

Version 2 is an additive, fail-closed Rule-90 contract for corrected fresh
dual-button launches.  It retains every v1 field and additionally binds the
authenticated archive member used as the training entrypoint, the exact 95D
actor observation order, the contact-union and kinematic-button semantics,
both source/materialized button windows, transition padding, root-carry mode,
button boundary values, and the burned-in replay overlay.  A canonical
identity/semantics digest must also be embedded in the actual MP4 ``comment``
metadata as ``holosoma_rule90_v2_binding_sha256=<digest>``.  Consequently an
old video or a video made for another source/run cannot be relabelled as a v2
capture without changing the media artifact and repeating visual review.

The canonical v2 additions are::

    "source": {
      "source_manifest_sha256": "<snapshot-id digest>",
      "entrypoint": {
        "archive_member": "distill_as_dual_button_solid.sh",
        "sha256": "<authenticated member digest>"
      }
    },
    "run": {
      "rule90": {
        "actor": {
          "ordered_groups": ["actor_obs_root_contact_aware",
                             "actor_obs_pickup_button",
                             "actor_obs_drop_button",
                             "actor_obs_proprio_with_actions_no_linvel"],
          "input_dim": 95,
          "history_length": 1
        },
        "contact_selector": {
          "algorithm": "all_carry_regions_union",
          "version": 2
        },
        "button_window": {
          "mode": "kinematic_lift",
          "algorithm": "object_root_rel_z_v1",
          "lift_height_threshold_m": 0.10,
          "lift_range_ratio": 0.35,
          "sustained_frames": 5,
          "source_semantics": "global_multi_clip_runtime",
          "motion_fps": 50.0,
          "source_motion_sha256": "<inputs.motion_npz_sha256>",
          "motion_transition_contract_sha256": "<inputs.transition_digest>",
          "source_window": {"frame_count": 317, "t1": 59, "t2": 232},
          "materialized_window": {"frame_count": 327, "t1": 69, "t2": 242},
          "effective_prepend_frames": 10,
          "effective_append_frames": 0,
          "boundary_values": {
            "pickup_at_t1_minus_1": 1, "pickup_at_t1": 0,
            "drop_at_t2_minus_1": 0, "drop_at_t2": 1
          }
        },
        "root_carry_mode": "peak_height"
      },
      "capture": {
        "fresh": true,
        "run_id": "<same run.run_id>",
        "source_snapshot_id": "<same source.snapshot_id>",
        "source_archive_sha256": "<same source.archive_sha256>",
        "entrypoint_archive_member": "<same source.entrypoint.archive_member>",
        "entrypoint_sha256": "<same source.entrypoint.sha256>",
        "video_sha256": "<same video.sha256>",
        "captured_at_utc": "<timezone-aware ISO-8601>",
        "semantic_binding_sha256": "<canonical v2 binding digest>"
      }
    },
    "video": {
      "ffprobe": {
        "rule90_v2_binding_sha256": "<same canonical v2 binding digest>"
      },
      "overlay": {
        "burned_in": true,
        "fields": ["frame", "index", "pickup", "drop"],
        "frame_value_source": "source_motion_frame",
        "index_value_source": "materialized_zero_based_index",
        "button_value_source": "materialized_kinematic_lift_window"
      }
    },
    "visual_review": {
      "overlay_verified": true,
      "run_id": "<same run.run_id>",
      "source_snapshot_id": "<same source.snapshot_id>",
      "semantic_binding_sha256": "<same run.capture.semantic_binding_sha256>"
    }
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar


_SUPPORTED_SCHEMA_VERSIONS = frozenset({1, 2})
_RULE90_V2_SCHEMA_VERSION = 2
_SUMMARY_VIDEO_KEY = "vis/replay"
_METADATA_PREFIX = "replay_preflight"
_WANDB_BASE_URL = "https://api.wandb.ai"
_MAX_MANIFEST_BYTES = 4 * 1024 * 1024
_MAX_FFPROBE_OUTPUT_BYTES = 4 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID_RE = re.compile(r"^src-([0-9a-f]{64})$")
_IDENTITY_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RULE90_V2_MP4_COMMENT_RE = re.compile(
    r"^holosoma_rule90_v2_binding_sha256=([0-9a-f]{64})$"
)
_DIGEST_KEY_RE = re.compile(r"(?:^|_)(?:sha256|digest)$", re.IGNORECASE)
_SAFE_DIGEST_SEGMENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|bearer|credential|password|secret|token|cookie)",
    re.IGNORECASE,
)
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Network exceptions can embed signed S3/GCS URLs.  The URL is not needed
    # for this fail-closed gate, so remove it wholesale rather than attempting
    # to enumerate every temporary credential query parameter.
    (re.compile(r"(?i)https?://[^\s,;]+"), "<redacted-url>"),
    (
        re.compile(r"(?i)(WANDB_API_KEY\s*[=:]\s*)[^\s,;]+"),
        r"\1<redacted>",
    ),
    (
        re.compile(r"(?i)(authorization\s*[=:]\s*bearer\s+)[^\s,;]+"),
        r"\1<redacted>",
    ),
    (re.compile(r"(?i)(api[_-]?key[=:/\s]+)[A-Za-z0-9_-]{16,}"), r"\1<redacted>"),
    # W&B API keys are commonly 40-character opaque strings.  Redact a bare
    # token too, in case an upstream exception omitted the field label.
    (re.compile(r"(?<![A-Za-z0-9])[A-Za-z0-9]{40}(?![A-Za-z0-9])"), "<redacted>"),
)

_RULE90_V2_ACTOR_GROUPS = [
    "actor_obs_root_contact_aware",
    "actor_obs_pickup_button",
    "actor_obs_drop_button",
    "actor_obs_proprio_with_actions_no_linvel",
]
_RULE90_V2_OVERLAY_FIELDS = ["frame", "index", "pickup", "drop"]
_RULE90_V2_ENTRYPOINT = "distill_as_dual_button_solid.sh"


class PreflightError(RuntimeError):
    """A fail-closed validation or remote verification error."""


@dataclass(frozen=True)
class FileSnapshot:
    path: Path
    size: int
    sha256: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class VideoDeclaration:
    path: Path
    sha256: str
    size_bytes: int
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec_name: str | None
    rule90_v2_binding_sha256: str | None


@dataclass(frozen=True)
class ProbeResult:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float
    codec_name: str
    rule90_v2_binding_sha256: str | None


@dataclass(frozen=True)
class ValidatedManifest:
    path: Path
    sha256: str
    schema_version: int
    payload: dict[str, Any]
    entity: str
    project: str
    run_id: str
    run_name: str
    source_snapshot_id: str
    world_size: int
    motion_clip_id: str
    video: VideoDeclaration
    video_snapshot: FileSnapshot
    metadata: dict[str, Any]

    @property
    def run_path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"


def _safe_message(value: object) -> str:
    """Render an exception without echoing common credential forms."""

    message = str(value).replace("\n", " ").replace("\r", " ")
    for pattern, replacement in _REDACTIONS:
        message = pattern.sub(replacement, message)
    # Errors are diagnostic, not a channel for dumping an upstream response.
    return message[:1000]


def _require_sha256(value: object, *, role: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PreflightError(f"{role} must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_nonempty_string(value: object, *, role: str, max_length: int = 512) -> str:
    if not isinstance(value, str):
        raise PreflightError(f"{role} must be a string")
    if value != value.strip() or not value:
        raise PreflightError(f"{role} must be non-empty and have no surrounding whitespace")
    if len(value) > max_length or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise PreflightError(f"{role} contains control characters or exceeds {max_length} characters")
    return value


def _require_mapping(value: object, *, role: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PreflightError(f"{role} must be a JSON object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any], *, role: str, expected: set[str]
) -> None:
    actual = set(value)
    if actual != expected:
        raise PreflightError(
            f"{role} keys must be canonical: expected={sorted(expected)!r}, "
            f"actual={sorted(actual)!r}"
        )


def _require_exact_int(
    value: object,
    *,
    role: str,
    minimum: int = 1,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:  # bool is intentionally rejected.
        raise PreflightError(f"{role} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"{minimum}..{maximum}" if maximum is not None else f">={minimum}"
        raise PreflightError(f"{role} must be in {bound}")
    return value


def _require_literal_int(value: object, *, role: str, expected: int) -> int:
    if type(value) is not int or value != expected:  # bool is intentionally rejected.
        raise PreflightError(f"{role} must be the JSON integer {expected}")
    return value


def _require_finite_number(
    value: object,
    *,
    role: str,
    minimum_exclusive: float = 0.0,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PreflightError(f"{role} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result <= minimum_exclusive:
        raise PreflightError(f"{role} must be finite and > {minimum_exclusive}")
    if maximum is not None and result > maximum:
        raise PreflightError(f"{role} must be <= {maximum}")
    return result


def _reject_json_constant(raw: str) -> None:
    raise PreflightError(f"non-standard JSON constant is forbidden: {raw}")


def _canonical_json_sha256(value: object, *, role: str) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PreflightError(f"{role} is not strict finite JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PreflightError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _stat_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_regular_nofollow(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PreflightError(f"cannot open regular file {path}: {_safe_message(exc)}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise PreflightError(f"expected a regular file: {path}")
    return descriptor, info


def _resolve_regular_path(raw_path: Path, *, role: str) -> Path:
    expanded = raw_path.expanduser()
    if expanded.is_symlink():
        raise PreflightError(f"{role} must not be a symlink: {expanded}")
    try:
        resolved = expanded.resolve(strict=True)
    except OSError as exc:
        raise PreflightError(f"{role} does not resolve to an existing file: {_safe_message(exc)}") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise PreflightError(f"{role} must resolve to a regular non-symlink file: {resolved}")
    return resolved


def _stable_file_snapshot(path: Path, *, role: str, max_bytes: int | None = None) -> tuple[FileSnapshot, bytes | None]:
    resolved = _resolve_regular_path(path, role=role)
    descriptor, before = _open_regular_nofollow(resolved)
    try:
        if max_bytes is not None and before.st_size > max_bytes:
            raise PreflightError(f"{role} exceeds the {max_bytes}-byte limit")
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if max_bytes is not None else None
        while True:
            chunk = os.read(descriptor, 4 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _stat_identity(before) != _stat_identity(after):
        raise PreflightError(f"{role} changed while it was being hashed")
    snapshot = FileSnapshot(
        path=resolved,
        size=int(after.st_size),
        sha256=digest.hexdigest(),
        identity=_stat_identity(after),
    )
    return snapshot, (b"".join(chunks) if chunks is not None else None)


def _assert_file_unchanged(snapshot: FileSnapshot, *, role: str) -> None:
    descriptor, info = _open_regular_nofollow(snapshot.path)
    os.close(descriptor)
    if _stat_identity(info) != snapshot.identity:
        raise PreflightError(f"{role} changed after validation: {snapshot.path}")


def _assert_manifest_bytes_unchanged(validated: ValidatedManifest) -> None:
    current, _ = _stable_file_snapshot(
        validated.path,
        role="manifest",
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    if current.sha256 != validated.sha256:
        raise PreflightError("manifest bytes changed after local validation")


def _reject_sensitive_keys(value: object, *, path: tuple[str, ...] = ()) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise PreflightError("manifest object keys must be strings")
            if _SENSITIVE_KEY_RE.search(key):
                location = ".".join((*path, key))
                raise PreflightError(f"credential-like manifest key is forbidden: {location}")
            _reject_sensitive_keys(child, path=(*path, key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_sensitive_keys(child, path=(*path, str(index)))


def _load_manifest(path: Path, expected_sha256: str) -> tuple[Path, str, dict[str, Any]]:
    expected = _require_sha256(expected_sha256, role="--expected-manifest-sha256")
    snapshot, raw = _stable_file_snapshot(path, role="manifest", max_bytes=_MAX_MANIFEST_BYTES)
    assert raw is not None
    if snapshot.size == 0:
        raise PreflightError("manifest is empty")
    if snapshot.sha256 != expected:
        raise PreflightError(
            f"manifest SHA-256 mismatch: expected={expected}, actual={snapshot.sha256}"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreflightError("manifest must be UTF-8 JSON") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except PreflightError:
        raise
    except json.JSONDecodeError as exc:
        raise PreflightError(f"invalid manifest JSON at line {exc.lineno}, column {exc.colno}") from exc
    if not isinstance(payload, dict):
        raise PreflightError("manifest root must be a JSON object")
    _reject_sensitive_keys(payload)
    return snapshot.path, snapshot.sha256, payload


def _digest_leaves(section: Mapping[str, Any], *, section_name: str) -> dict[str, str]:
    leaves: dict[str, str] = {}

    def visit(value: object, path: tuple[str, ...], in_digest_container: bool) -> None:
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str) or _SAFE_DIGEST_SEGMENT_RE.fullmatch(key) is None:
                    raise PreflightError(
                        f"{section_name} contains a key that cannot be mirrored safely: {key!r}"
                    )
                child = value[key]
                next_container = in_digest_container or key.lower() in {"digest", "digests"}
                if isinstance(child, (dict, list)):
                    visit(child, (*path, key), next_container)
                    continue
                is_digest = bool(_DIGEST_KEY_RE.search(key)) or in_digest_container
                if not is_digest:
                    continue
                digest = _require_sha256(child, role=f"{section_name}.{'.'.join((*path, key))}")
                relative = "/".join((*path, key))
                if len(relative) > 512:
                    raise PreflightError(f"{section_name} digest path exceeds 512 characters")
                if relative in leaves:
                    raise PreflightError(f"duplicate flattened digest path: {section_name}/{relative}")
                leaves[relative] = digest
            return
        if isinstance(value, list):
            for index, child in enumerate(value):
                child_path = (*path, str(index))
                if isinstance(child, (dict, list)):
                    visit(child, child_path, in_digest_container)
                elif in_digest_container:
                    digest = _require_sha256(
                        child,
                        role=f"{section_name}.{'.'.join(child_path)}",
                    )
                    relative = "/".join(child_path)
                    if len(relative) > 512:
                        raise PreflightError(
                            f"{section_name} digest path exceeds 512 characters"
                        )
                    leaves[relative] = digest

    visit(section, (), False)
    return leaves


def _parse_timestamp(raw: object, *, role: str) -> tuple[str, dt.datetime]:
    value = _require_nonempty_string(raw, role=role, max_length=64)
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise PreflightError(f"{role} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PreflightError(f"{role} must include an explicit timezone")
    return value, parsed.astimezone(dt.timezone.utc)


def _parse_review_timestamp(raw: object) -> str:
    value, _ = _parse_timestamp(raw, role="visual_review.reviewed_at_utc")
    return value


def _video_field(video: Mapping[str, Any], ffprobe: Mapping[str, Any], key: str, *aliases: str) -> object:
    candidates: list[tuple[str, object]] = []
    for name in (key, *aliases):
        if name in ffprobe:
            candidates.append((f"video.ffprobe.{name}", ffprobe[name]))
        if name in video:
            candidates.append((f"video.{name}", video[name]))
    if not candidates:
        raise PreflightError(f"missing video ffprobe declaration: {key}")
    first_name, first_value = candidates[0]
    for other_name, other_value in candidates[1:]:
        if other_value != first_value:
            raise PreflightError(f"conflicting declarations: {first_name} and {other_name}")
    return first_value


def _parse_video_declaration(video: dict[str, Any], *, manifest_dir: Path) -> VideoDeclaration:
    raw_path = _require_nonempty_string(video.get("path"), role="video.path", max_length=4096)
    declared_path = Path(raw_path).expanduser()
    if not declared_path.is_absolute():
        declared_path = manifest_dir / declared_path
    resolved = _resolve_regular_path(declared_path, role="video.path")
    if resolved.suffix.lower() != ".mp4":
        raise PreflightError("video.path must end in .mp4")

    sha256 = _require_sha256(video.get("sha256"), role="video.sha256")
    size_bytes = _require_exact_int(
        video.get("size_bytes"), role="video.size_bytes", minimum=1, maximum=2**63 - 1
    )
    ffprobe = _require_mapping(video.get("ffprobe", {}), role="video.ffprobe")
    width = _require_exact_int(
        _video_field(video, ffprobe, "width"), role="video.ffprobe.width", maximum=32768
    )
    height = _require_exact_int(
        _video_field(video, ffprobe, "height"), role="video.ffprobe.height", maximum=32768
    )
    fps = _require_finite_number(
        _video_field(video, ffprobe, "fps"), role="video.ffprobe.fps", maximum=1000.0
    )
    frame_count = _require_exact_int(
        _video_field(video, ffprobe, "frame_count", "frames"),
        role="video.ffprobe.frame_count",
        maximum=100_000_000,
    )
    duration_s = _require_finite_number(
        _video_field(video, ffprobe, "duration_s", "duration_seconds"),
        role="video.ffprobe.duration_s",
        maximum=7 * 24 * 3600.0,
    )
    codec_raw = ffprobe.get("codec_name", video.get("codec_name"))
    codec_name = (
        _require_nonempty_string(codec_raw, role="video.ffprobe.codec_name", max_length=64)
        if codec_raw is not None
        else None
    )
    binding_raw = (
        _video_field(video, ffprobe, "rule90_v2_binding_sha256")
        if "rule90_v2_binding_sha256" in ffprobe
        or "rule90_v2_binding_sha256" in video
        else None
    )
    rule90_v2_binding_sha256 = (
        _require_sha256(
            binding_raw,
            role="video.ffprobe.rule90_v2_binding_sha256",
        )
        if binding_raw is not None
        else None
    )

    expected_duration = frame_count / fps
    duration_tolerance = max(0.001, 0.51 / fps)
    if abs(duration_s - expected_duration) > duration_tolerance:
        raise PreflightError(
            "video declaration is internally inconsistent: "
            f"duration_s={duration_s}, frame_count/fps={expected_duration}"
        )
    return VideoDeclaration(
        path=resolved,
        sha256=sha256,
        size_bytes=size_bytes,
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
        codec_name=codec_name,
        rule90_v2_binding_sha256=rule90_v2_binding_sha256,
    )


def _assert_unique_local_mp4(video_path: Path) -> None:
    candidates: list[Path] = []
    try:
        children = list(video_path.parent.iterdir())
    except OSError as exc:
        raise PreflightError(f"cannot enumerate replay video directory: {_safe_message(exc)}") from exc
    for child in children:
        if child.suffix.lower() != ".mp4":
            continue
        if child.is_symlink() or not child.is_file():
            raise PreflightError(f"replay directory contains a non-regular MP4 entry: {child.name}")
        if child.stat().st_size <= 0:
            raise PreflightError(f"replay directory contains an empty MP4: {child.name}")
        candidates.append(child.resolve(strict=True))
    if candidates != [video_path]:
        names = sorted(path.name for path in candidates)
        raise PreflightError(
            "video.path parent must contain exactly the declared MP4; "
            f"found={names!r}, declared={video_path.name!r}"
        )


def _fraction_to_float(raw: object, *, role: str) -> float:
    value = _require_nonempty_string(raw, role=role, max_length=64)
    try:
        result = float(Fraction(value))
    except (ValueError, ZeroDivisionError) as exc:
        raise PreflightError(f"{role} is not a valid rational frame rate") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise PreflightError(f"{role} must be positive and finite")
    return result


def _probe_video(path: Path, *, ffprobe_binary: str, timeout_s: float) -> ProbeResult:
    executable = shutil.which(ffprobe_binary)
    if executable is None:
        raise PreflightError(f"ffprobe executable was not found: {ffprobe_binary!r}")
    command = [
        executable,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v",
        "-show_entries",
        (
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,"
            "nb_frames,nb_read_frames,duration:format=duration,size"
            ":format_tags=comment"
        ),
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise PreflightError(f"ffprobe exceeded its {timeout_s:.1f}s timeout") from exc
    except OSError as exc:
        raise PreflightError(f"failed to execute ffprobe: {_safe_message(exc)}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="replace")
        raise PreflightError(f"ffprobe failed: {_safe_message(stderr)}")
    if len(completed.stdout) > _MAX_FFPROBE_OUTPUT_BYTES:
        raise PreflightError("ffprobe output exceeded the safety limit")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError("ffprobe returned invalid JSON") from exc
    streams = payload.get("streams") if isinstance(payload, dict) else None
    if not isinstance(streams, list) or len(streams) != 1 or not isinstance(streams[0], dict):
        count = len(streams) if isinstance(streams, list) else 0
        raise PreflightError(f"MP4 must contain exactly one video stream; found={count}")
    stream = streams[0]
    width = _require_exact_int(stream.get("width"), role="ffprobe.width", maximum=32768)
    height = _require_exact_int(stream.get("height"), role="ffprobe.height", maximum=32768)
    avg_rate = stream.get("avg_frame_rate")
    raw_rate = avg_rate if isinstance(avg_rate, str) and avg_rate != "0/0" else stream.get("r_frame_rate")
    fps = _fraction_to_float(raw_rate, role="ffprobe.avg_frame_rate")

    raw_frames = stream.get("nb_read_frames")
    if raw_frames in (None, "N/A"):
        raw_frames = stream.get("nb_frames")
    try:
        frame_count = int(str(raw_frames))
    except (TypeError, ValueError) as exc:
        raise PreflightError("ffprobe did not report a usable decoded frame count") from exc
    frame_count = _require_exact_int(frame_count, role="ffprobe.frame_count", maximum=100_000_000)

    raw_duration = stream.get("duration")
    if raw_duration in (None, "N/A"):
        format_record = payload.get("format", {})
        raw_duration = format_record.get("duration") if isinstance(format_record, dict) else None
    try:
        duration_s = float(str(raw_duration))
    except (TypeError, ValueError) as exc:
        raise PreflightError("ffprobe did not report a usable duration") from exc
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise PreflightError("ffprobe duration must be positive and finite")
    codec_name = _require_nonempty_string(stream.get("codec_name"), role="ffprobe.codec_name", max_length=64)
    format_record = payload.get("format", {})
    tags = format_record.get("tags", {}) if isinstance(format_record, dict) else {}
    comment = tags.get("comment") if isinstance(tags, dict) else None
    comment_match = (
        _RULE90_V2_MP4_COMMENT_RE.fullmatch(comment)
        if isinstance(comment, str)
        else None
    )
    rule90_v2_binding_sha256 = comment_match.group(1) if comment_match else None
    return ProbeResult(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_s=duration_s,
        codec_name=codec_name,
        rule90_v2_binding_sha256=rule90_v2_binding_sha256,
    )


def _assert_probe_matches(declared: VideoDeclaration, actual: ProbeResult) -> None:
    if (actual.width, actual.height) != (declared.width, declared.height):
        raise PreflightError(
            "ffprobe dimensions differ from the manifest: "
            f"declared={declared.width}x{declared.height}, actual={actual.width}x{actual.height}"
        )
    if actual.frame_count != declared.frame_count:
        raise PreflightError(
            "ffprobe frame count differs from the manifest: "
            f"declared={declared.frame_count}, actual={actual.frame_count}"
        )
    fps_tolerance = max(1.0e-6, declared.fps * 1.0e-6)
    if abs(actual.fps - declared.fps) > fps_tolerance:
        raise PreflightError(
            f"ffprobe FPS differs from the manifest: declared={declared.fps}, actual={actual.fps}"
        )
    duration_tolerance = max(0.001, 0.51 / declared.fps)
    if abs(actual.duration_s - declared.duration_s) > duration_tolerance:
        raise PreflightError(
            "ffprobe duration differs from the manifest: "
            f"declared={declared.duration_s}, actual={actual.duration_s}"
        )
    if declared.codec_name is not None and actual.codec_name != declared.codec_name:
        raise PreflightError(
            "ffprobe codec differs from the manifest: "
            f"declared={declared.codec_name!r}, actual={actual.codec_name!r}"
        )
    if (
        declared.rule90_v2_binding_sha256 is not None
        and actual.rule90_v2_binding_sha256 != declared.rule90_v2_binding_sha256
    ):
        raise PreflightError(
            "MP4 Rule-90 v2 binding metadata differs from the manifest: "
            f"declared={declared.rule90_v2_binding_sha256!r}, "
            f"actual={actual.rule90_v2_binding_sha256!r}"
        )


def _validate_identity_component(value: object, *, role: str, run_id: bool = False) -> str:
    result = _require_nonempty_string(value, role=role, max_length=128)
    pattern = _RUN_ID_RE if run_id else _IDENTITY_COMPONENT_RE
    if pattern.fullmatch(result) is None or "/" in result or result in {".", ".."}:
        raise PreflightError(f"{role} contains unsupported characters")
    return result


def _rule90_v2_window(
    raw: object,
    *,
    role: str,
) -> tuple[dict[str, Any], int, int, int]:
    window = _require_mapping(raw, role=role)
    _require_exact_keys(
        window,
        role=role,
        expected={"frame_count", "t1", "t2"},
    )
    frame_count = _require_exact_int(
        window.get("frame_count"), role=f"{role}.frame_count", maximum=100_000_000
    )
    t1 = _require_exact_int(
        window.get("t1"), role=f"{role}.t1", minimum=1, maximum=100_000_000
    )
    t2 = _require_exact_int(
        window.get("t2"), role=f"{role}.t2", minimum=1, maximum=100_000_000
    )
    # Both t1-1 and t2 are real replay indices because v2 visibly proves all
    # four boundary values in the sole MP4.  Empty windows and an end boundary
    # outside the materialized trace therefore fail closed.
    if not 1 <= t1 < t2 < frame_count:
        raise PreflightError(
            f"{role} must satisfy 1 <= t1 < t2 < frame_count; "
            f"got t1={t1}, t2={t2}, frame_count={frame_count}"
        )
    return window, frame_count, t1, t2


def _validate_rule90_v2(
    *,
    args: argparse.Namespace,
    run: Mapping[str, Any],
    source: Mapping[str, Any],
    inputs: Mapping[str, Any],
    video_section: Mapping[str, Any],
    video: VideoDeclaration,
    review: Mapping[str, Any],
    source_snapshot_id: str,
    snapshot_id_digest: str,
) -> dict[str, Any]:
    archive_sha256 = _require_sha256(
        source.get("archive_sha256"), role="source.archive_sha256"
    )
    source_manifest_sha256 = _require_sha256(
        source.get("source_manifest_sha256"), role="source.source_manifest_sha256"
    )
    if source_manifest_sha256 != snapshot_id_digest:
        raise PreflightError(
            "v2 source.source_manifest_sha256 must equal the digest embedded in "
            "source.snapshot_id"
        )

    expected_archive_sha256 = _require_sha256(
        getattr(args, "expected_source_archive_sha256", None),
        role="--expected-source-archive-sha256",
    )
    if archive_sha256 != expected_archive_sha256:
        raise PreflightError(
            "v2 source archive SHA-256 differs from its controller expectation"
        )

    entrypoint = _require_mapping(source.get("entrypoint"), role="source.entrypoint")
    _require_exact_keys(
        entrypoint,
        role="source.entrypoint",
        expected={"archive_member", "sha256"},
    )
    archive_member = _require_nonempty_string(
        entrypoint.get("archive_member"),
        role="source.entrypoint.archive_member",
        max_length=256,
    )
    member_path = PurePosixPath(archive_member)
    if (
        member_path.is_absolute()
        or len(member_path.parts) != 1
        or member_path.as_posix() != archive_member
        or archive_member != _RULE90_V2_ENTRYPOINT
    ):
        raise PreflightError(
            "source.entrypoint.archive_member must be the exact root-level formal-dual "
            f"snapshot entrypoint {_RULE90_V2_ENTRYPOINT!r}"
        )
    entrypoint_sha256 = _require_sha256(
        entrypoint.get("sha256"), role="source.entrypoint.sha256"
    )
    expected_entrypoint_member = _require_nonempty_string(
        getattr(args, "expected_entrypoint_archive_member", None),
        role="--expected-entrypoint-archive-member",
        max_length=256,
    )
    expected_entrypoint_sha256 = _require_sha256(
        getattr(args, "expected_entrypoint_sha256", None),
        role="--expected-entrypoint-sha256",
    )
    if archive_member != expected_entrypoint_member:
        raise PreflightError(
            "v2 source entrypoint archive member differs from its controller expectation"
        )
    if entrypoint_sha256 != expected_entrypoint_sha256:
        raise PreflightError(
            "v2 source entrypoint SHA-256 differs from its controller expectation"
        )

    rule90 = _require_mapping(run.get("rule90"), role="run.rule90")
    _require_exact_keys(
        rule90,
        role="run.rule90",
        expected={"actor", "contact_selector", "button_window", "root_carry_mode"},
    )

    actor = _require_mapping(rule90.get("actor"), role="run.rule90.actor")
    _require_exact_keys(
        actor,
        role="run.rule90.actor",
        expected={"ordered_groups", "input_dim", "history_length"},
    )
    ordered_groups = actor.get("ordered_groups")
    if type(ordered_groups) is not list or ordered_groups != _RULE90_V2_ACTOR_GROUPS:
        raise PreflightError(
            "run.rule90.actor.ordered_groups must equal the exact ordered dual-button "
            f"95D contract: {_RULE90_V2_ACTOR_GROUPS!r}"
        )
    _require_literal_int(
        actor.get("input_dim"), role="run.rule90.actor.input_dim", expected=95
    )
    _require_literal_int(
        actor.get("history_length"),
        role="run.rule90.actor.history_length",
        expected=1,
    )

    selector = _require_mapping(
        rule90.get("contact_selector"), role="run.rule90.contact_selector"
    )
    _require_exact_keys(
        selector,
        role="run.rule90.contact_selector",
        expected={"algorithm", "version"},
    )
    if selector.get("algorithm") != "all_carry_regions_union":
        raise PreflightError(
            "run.rule90.contact_selector.algorithm must equal 'all_carry_regions_union'"
        )
    _require_literal_int(
        selector.get("version"),
        role="run.rule90.contact_selector.version",
        expected=2,
    )

    button = _require_mapping(
        rule90.get("button_window"), role="run.rule90.button_window"
    )
    _require_exact_keys(
        button,
        role="run.rule90.button_window",
        expected={
            "mode",
            "algorithm",
            "lift_height_threshold_m",
            "lift_range_ratio",
            "sustained_frames",
            "source_semantics",
            "motion_fps",
            "source_motion_sha256",
            "motion_transition_contract_sha256",
            "source_window",
            "materialized_window",
            "effective_prepend_frames",
            "effective_append_frames",
            "boundary_values",
        },
    )
    literal_button_values: tuple[tuple[str, object], ...] = (
        ("mode", "kinematic_lift"),
        ("algorithm", "object_root_rel_z_v1"),
        ("source_semantics", "global_multi_clip_runtime"),
    )
    for key, expected in literal_button_values:
        if button.get(key) != expected or type(button.get(key)) is not type(expected):
            raise PreflightError(
                f"run.rule90.button_window.{key} must equal {expected!r}"
            )
    for key, expected in (
        ("lift_height_threshold_m", 0.10),
        ("lift_range_ratio", 0.35),
    ):
        value = button.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) != expected
        ):
            raise PreflightError(
                f"run.rule90.button_window.{key} must equal the finite number {expected}"
            )
    _require_literal_int(
        button.get("sustained_frames"),
        role="run.rule90.button_window.sustained_frames",
        expected=5,
    )
    motion_fps = button.get("motion_fps")
    if (
        isinstance(motion_fps, bool)
        or not isinstance(motion_fps, (int, float))
        or not math.isfinite(float(motion_fps))
        or float(motion_fps) <= 0.0
        or float(motion_fps) != video.fps
    ):
        raise PreflightError(
            "run.rule90.button_window.motion_fps must be finite, positive, and "
            "equal to the replay video FPS"
        )
    button_motion_sha = _require_sha256(
        button.get("source_motion_sha256"),
        role="run.rule90.button_window.source_motion_sha256",
    )
    button_transition_sha = _require_sha256(
        button.get("motion_transition_contract_sha256"),
        role="run.rule90.button_window.motion_transition_contract_sha256",
    )
    if button_motion_sha != inputs.get("motion_npz_sha256"):
        raise PreflightError(
            "v2 button-window source motion differs from inputs.motion_npz_sha256"
        )
    if button_transition_sha != inputs.get("transition_digest"):
        raise PreflightError(
            "v2 button-window transition contract differs from inputs.transition_digest"
        )

    source_window, source_frames, source_t1, source_t2 = _rule90_v2_window(
        button.get("source_window"), role="run.rule90.button_window.source_window"
    )
    materialized_window, materialized_frames, materialized_t1, materialized_t2 = (
        _rule90_v2_window(
            button.get("materialized_window"),
            role="run.rule90.button_window.materialized_window",
        )
    )
    prepend = _require_exact_int(
        button.get("effective_prepend_frames"),
        role="run.rule90.button_window.effective_prepend_frames",
        minimum=0,
        maximum=100_000_000,
    )
    append = _require_exact_int(
        button.get("effective_append_frames"),
        role="run.rule90.button_window.effective_append_frames",
        minimum=0,
        maximum=100_000_000,
    )
    if materialized_frames != source_frames + prepend + append:
        raise PreflightError(
            "v2 materialized frame_count must equal source frame_count + prepend + append"
        )
    if video.frame_count != materialized_frames:
        raise PreflightError(
            "v2 replay video frame_count must equal the materialized motion frame_count"
        )
    if (materialized_t1, materialized_t2) != (
        source_t1 + prepend,
        source_t2 + prepend,
    ):
        raise PreflightError(
            "v2 global-runtime materialized t1/t2 must equal source t1/t2 + prepend"
        )

    boundaries = _require_mapping(
        button.get("boundary_values"),
        role="run.rule90.button_window.boundary_values",
    )
    expected_boundaries = {
        "pickup_at_t1_minus_1": 1,
        "pickup_at_t1": 0,
        "drop_at_t2_minus_1": 0,
        "drop_at_t2": 1,
    }
    _require_exact_keys(
        boundaries,
        role="run.rule90.button_window.boundary_values",
        expected=set(expected_boundaries),
    )
    for key, expected in expected_boundaries.items():
        _require_literal_int(
            boundaries.get(key),
            role=f"run.rule90.button_window.boundary_values.{key}",
            expected=expected,
        )
    if rule90.get("root_carry_mode") != "peak_height":
        raise PreflightError("run.rule90.root_carry_mode must equal 'peak_height'")

    overlay = _require_mapping(video_section.get("overlay"), role="video.overlay")
    _require_exact_keys(
        overlay,
        role="video.overlay",
        expected={
            "burned_in",
            "fields",
            "frame_value_source",
            "index_value_source",
            "button_value_source",
        },
    )
    if overlay.get("burned_in") is not True:
        raise PreflightError("video.overlay.burned_in must be the JSON boolean true")
    overlay_fields = overlay.get("fields")
    if type(overlay_fields) is not list or overlay_fields != _RULE90_V2_OVERLAY_FIELDS:
        raise PreflightError(
            "video.overlay.fields must equal the exact ordered burned-in fields "
            f"{_RULE90_V2_OVERLAY_FIELDS!r}"
        )
    expected_overlay_sources = {
        "frame_value_source": "source_motion_frame",
        "index_value_source": "materialized_zero_based_index",
        "button_value_source": "materialized_kinematic_lift_window",
    }
    for key, expected in expected_overlay_sources.items():
        if type(overlay.get(key)) is not str or overlay.get(key) != expected:
            raise PreflightError(f"video.overlay.{key} must equal {expected!r}")

    capture = _require_mapping(run.get("capture"), role="run.capture")
    _require_exact_keys(
        capture,
        role="run.capture",
        expected={
            "fresh",
            "run_id",
            "source_snapshot_id",
            "source_archive_sha256",
            "entrypoint_archive_member",
            "entrypoint_sha256",
            "video_sha256",
            "captured_at_utc",
            "semantic_binding_sha256",
        },
    )
    if capture.get("fresh") is not True:
        raise PreflightError("run.capture.fresh must be the JSON boolean true")
    capture_pairs = (
        ("run_id", capture.get("run_id"), run.get("run_id")),
        ("source_snapshot_id", capture.get("source_snapshot_id"), source_snapshot_id),
        ("source_archive_sha256", capture.get("source_archive_sha256"), archive_sha256),
        ("entrypoint_archive_member", capture.get("entrypoint_archive_member"), archive_member),
        ("entrypoint_sha256", capture.get("entrypoint_sha256"), entrypoint_sha256),
        ("video_sha256", capture.get("video_sha256"), video.sha256),
    )
    for key, actual, expected in capture_pairs:
        if type(actual) is not str or actual != expected:
            raise PreflightError(f"run.capture.{key} differs from the bound v2 identity")
    captured_at_utc, captured_at = _parse_timestamp(
        capture.get("captured_at_utc"), role="run.capture.captured_at_utc"
    )
    _, reviewed_at = _parse_timestamp(
        review.get("reviewed_at_utc"), role="visual_review.reviewed_at_utc"
    )
    if reviewed_at < captured_at:
        raise PreflightError("visual review timestamp precedes the v2 replay capture")

    binding_inputs = {
        key: inputs.get(key)
        for key in (
            "world_size",
            "motion_clip_id",
            "motion_npz_sha256",
            "object_map_sha256",
            "object_urdf_sha256",
            "object_mesh_sha256",
            "single_slot_source_digest",
            "single_slot_view_digest",
            "rank_shard_source_digest",
            "transition_digest",
        )
    }
    binding_payload = {
        "version": _RULE90_V2_SCHEMA_VERSION,
        "run": {
            "entity": run.get("entity"),
            "project": run.get("project"),
            "run_id": run.get("run_id"),
            "name": run.get("name"),
        },
        "source": {
            "snapshot_id": source_snapshot_id,
            "archive_sha256": archive_sha256,
            "source_manifest_sha256": source_manifest_sha256,
            "entrypoint": entrypoint,
        },
        "inputs": binding_inputs,
        "rule90": rule90,
        "overlay": overlay,
        "captured_at_utc": captured_at_utc,
    }
    expected_binding_sha256 = _canonical_json_sha256(
        binding_payload, role="Rule-90 v2 semantic binding"
    )
    declared_binding_sha256 = _require_sha256(
        capture.get("semantic_binding_sha256"),
        role="run.capture.semantic_binding_sha256",
    )
    if declared_binding_sha256 != expected_binding_sha256:
        raise PreflightError(
            "run.capture.semantic_binding_sha256 does not match the canonical v2 identity"
        )
    if video.rule90_v2_binding_sha256 != expected_binding_sha256:
        raise PreflightError(
            "the actual MP4 is not metadata-bound to this Rule-90 v2 run/source/semantics"
        )
    if review.get("overlay_verified") is not True:
        raise PreflightError(
            "visual_review.overlay_verified must be the JSON boolean true"
        )
    review_pairs = (
        ("run_id", review.get("run_id"), run.get("run_id")),
        ("source_snapshot_id", review.get("source_snapshot_id"), source_snapshot_id),
        (
            "semantic_binding_sha256",
            review.get("semantic_binding_sha256"),
            expected_binding_sha256,
        ),
    )
    for key, actual, expected in review_pairs:
        if type(actual) is not str or actual != expected:
            raise PreflightError(
                f"visual_review.{key} differs from the reviewed v2 capture identity"
            )

    return {
        "version": _RULE90_V2_SCHEMA_VERSION,
        "semantic_binding_sha256": expected_binding_sha256,
        "source_entrypoint": dict(entrypoint),
        "actor": dict(actor),
        "contact_selector": dict(selector),
        "button_window": {
            **dict(button),
            "source_window": dict(source_window),
            "materialized_window": dict(materialized_window),
            "boundary_values": dict(boundaries),
        },
        "root_carry_mode": "peak_height",
        "overlay": dict(overlay),
        "capture": dict(capture),
        "visual_review": {
            "overlay_verified": True,
            "run_id": run.get("run_id"),
            "source_snapshot_id": source_snapshot_id,
            "semantic_binding_sha256": expected_binding_sha256,
        },
    }


def _metadata_payload(
    *,
    schema_version: int,
    manifest_sha256: str,
    entity: str,
    project: str,
    run_id: str,
    run_name: str,
    source_snapshot_id: str,
    world_size: int,
    motion_clip_id: str,
    video: VideoDeclaration,
    reviewed_at_utc: str,
    digest_sections: Mapping[str, Mapping[str, str]],
    rule90_v2_contract: Mapping[str, Any] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        f"{_METADATA_PREFIX}/schema_version": schema_version,
        f"{_METADATA_PREFIX}/manifest_sha256": manifest_sha256,
        f"{_METADATA_PREFIX}/entity": entity,
        f"{_METADATA_PREFIX}/project": project,
        f"{_METADATA_PREFIX}/run_id": run_id,
        f"{_METADATA_PREFIX}/run_name": run_name,
        f"{_METADATA_PREFIX}/source_snapshot_id": source_snapshot_id,
        f"{_METADATA_PREFIX}/world_size": world_size,
        f"{_METADATA_PREFIX}/motion_clip_id": motion_clip_id,
        f"{_METADATA_PREFIX}/visual_review_passed": True,
        f"{_METADATA_PREFIX}/visual_reviewed_at_utc": reviewed_at_utc,
        f"{_METADATA_PREFIX}/video_width": video.width,
        f"{_METADATA_PREFIX}/video_height": video.height,
        f"{_METADATA_PREFIX}/video_fps": video.fps,
        f"{_METADATA_PREFIX}/video_frame_count": video.frame_count,
        f"{_METADATA_PREFIX}/video_duration_s": video.duration_s,
        f"{_METADATA_PREFIX}/video_size_bytes": video.size_bytes,
    }
    for section_name in ("source", "inputs", "video"):
        for relative_path, digest in sorted(digest_sections[section_name].items()):
            key = f"{_METADATA_PREFIX}/digest/{section_name}/{relative_path}"
            if key in metadata:
                raise PreflightError(f"duplicate W&B metadata key: {key}")
            metadata[key] = digest
    if rule90_v2_contract is not None:
        metadata[f"{_METADATA_PREFIX}/rule90_v2_contract"] = dict(
            rule90_v2_contract
        )
    return metadata


def _validate_manifest(args: argparse.Namespace) -> ValidatedManifest:
    path, manifest_sha256, payload = _load_manifest(args.manifest, args.expected_manifest_sha256)
    schema_version = payload.get("version")
    if type(schema_version) is not int or schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise PreflightError(
            f"manifest.version must be one of {sorted(_SUPPORTED_SCHEMA_VERSIONS)!r}"
        )
    required_version = getattr(args, "required_manifest_version", None)
    if required_version is not None:
        if type(required_version) is not int or required_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise PreflightError(
                "--required-manifest-version must be an exact supported schema integer"
            )
        if schema_version != required_version:
            raise PreflightError(
                f"manifest.version must equal controller-required version {required_version}"
            )

    run = _require_mapping(payload.get("run"), role="run")
    if run.get("fresh") is not True:
        raise PreflightError("run.fresh must be the JSON boolean true")
    entity = _validate_identity_component(run.get("entity"), role="run.entity")
    project = _validate_identity_component(run.get("project"), role="run.project")
    run_id = _validate_identity_component(run.get("run_id"), role="run.run_id", run_id=True)
    run_name = _require_nonempty_string(run.get("name"), role="run.name", max_length=256)

    expected_entity = _validate_identity_component(args.expected_entity, role="--expected-entity")
    expected_project = _validate_identity_component(args.expected_project, role="--expected-project")
    expected_run_id = _validate_identity_component(
        args.expected_run_id, role="--expected-run-id", run_id=True
    )
    expected_run_name = _require_nonempty_string(
        args.expected_run_name, role="--expected-run-name", max_length=256
    )
    identity_pairs = (
        ("entity", entity, expected_entity),
        ("project", project, expected_project),
        ("run_id", run_id, expected_run_id),
        ("run_name", run_name, expected_run_name),
    )
    for role, actual, expected in identity_pairs:
        if actual != expected:
            raise PreflightError(f"manifest {role} differs from its controller expectation")

    source = _require_mapping(payload.get("source"), role="source")
    source_snapshot_id = _require_nonempty_string(
        source.get("snapshot_id"), role="source.snapshot_id", max_length=128
    )
    snapshot_match = _SNAPSHOT_ID_RE.fullmatch(source_snapshot_id)
    if snapshot_match is None:
        raise PreflightError("source.snapshot_id must have canonical form src-<64 lowercase hex>")
    expected_snapshot_id = _require_nonempty_string(
        args.expected_source_snapshot_id,
        role="--expected-source-snapshot-id",
        max_length=128,
    )
    if source_snapshot_id != expected_snapshot_id:
        raise PreflightError("manifest source snapshot differs from its controller expectation")

    inputs = _require_mapping(payload.get("inputs"), role="inputs")
    world_size = _require_exact_int(inputs.get("world_size"), role="inputs.world_size", maximum=4096)
    expected_world_size = _require_exact_int(
        args.expected_world_size, role="--expected-world-size", maximum=4096
    )
    if world_size != expected_world_size:
        raise PreflightError("manifest world size differs from its controller expectation")
    motion_clip_id = _require_nonempty_string(
        inputs.get("motion_clip_id"), role="inputs.motion_clip_id", max_length=256
    )
    if _IDENTITY_COMPONENT_RE.fullmatch(motion_clip_id) is None:
        raise PreflightError("inputs.motion_clip_id contains unsupported characters")

    video_section = _require_mapping(payload.get("video"), role="video")
    video = _parse_video_declaration(video_section, manifest_dir=path.parent)
    _assert_unique_local_mp4(video.path)
    video_snapshot, _ = _stable_file_snapshot(video.path, role="replay MP4")
    if video_snapshot.size != video.size_bytes:
        raise PreflightError(
            f"video size mismatch: declared={video.size_bytes}, actual={video_snapshot.size}"
        )
    if video_snapshot.sha256 != video.sha256:
        raise PreflightError(
            f"video SHA-256 mismatch: declared={video.sha256}, actual={video_snapshot.sha256}"
        )
    actual_probe = _probe_video(
        video.path,
        ffprobe_binary=args.ffprobe,
        timeout_s=args.ffprobe_timeout_seconds,
    )
    _assert_probe_matches(video, actual_probe)
    _assert_file_unchanged(video_snapshot, role="replay MP4")

    review = _require_mapping(payload.get("visual_review"), role="visual_review")
    if review.get("passed") is not True:
        raise PreflightError("visual_review.passed must be the JSON boolean true")
    review_video_sha = _require_sha256(
        review.get("video_sha256"), role="visual_review.video_sha256"
    )
    if review_video_sha != video.sha256:
        raise PreflightError("visual_review is not bound to the declared MP4 SHA-256")
    _require_nonempty_string(review.get("reviewer"), role="visual_review.reviewer", max_length=128)
    reviewed_at_utc = _parse_review_timestamp(review.get("reviewed_at_utc"))

    digest_sections = {
        "source": _digest_leaves(source, section_name="source"),
        "inputs": _digest_leaves(inputs, section_name="inputs"),
        "video": _digest_leaves(video_section, section_name="video"),
    }
    required_source_digests = ("archive_sha256",)
    required_input_digests = (
        "motion_npz_sha256",
        "object_map_sha256",
        "object_urdf_sha256",
        "object_mesh_sha256",
        "single_slot_source_digest",
        "single_slot_view_digest",
        "rank_shard_source_digest",
        "transition_digest",
    )
    for key in required_source_digests:
        if key not in digest_sections["source"]:
            raise PreflightError(f"source is missing required digest field: {key}")
    for key in required_input_digests:
        if key not in digest_sections["inputs"]:
            raise PreflightError(f"inputs is missing required digest field: {key}")
    if digest_sections["video"].get("sha256") != video.sha256:
        raise PreflightError("video.sha256 was not captured as the canonical video digest")
    # Preserve the digest embedded in the snapshot identity as an independently
    # comparable value even when the source also declares an archive digest.
    snapshot_id_digest = snapshot_match.group(1)
    declared_snapshot_id_digest = digest_sections["source"].get("snapshot_id_digest")
    if (
        declared_snapshot_id_digest is not None
        and declared_snapshot_id_digest != snapshot_id_digest
    ):
        raise PreflightError(
            "source.snapshot_id_digest conflicts with the digest embedded in source.snapshot_id"
        )
    digest_sections["source"]["snapshot_id_digest"] = snapshot_id_digest

    rule90_v2_contract: dict[str, Any] | None = None
    if schema_version == _RULE90_V2_SCHEMA_VERSION:
        rule90_v2_contract = _validate_rule90_v2(
            args=args,
            run=run,
            source=source,
            inputs=inputs,
            video_section=video_section,
            video=video,
            review=review,
            source_snapshot_id=source_snapshot_id,
            snapshot_id_digest=snapshot_id_digest,
        )

    metadata = _metadata_payload(
        schema_version=schema_version,
        manifest_sha256=manifest_sha256,
        entity=entity,
        project=project,
        run_id=run_id,
        run_name=run_name,
        source_snapshot_id=source_snapshot_id,
        world_size=world_size,
        motion_clip_id=motion_clip_id,
        video=video,
        reviewed_at_utc=reviewed_at_utc,
        digest_sections=digest_sections,
        rule90_v2_contract=rule90_v2_contract,
    )
    return ValidatedManifest(
        path=path,
        sha256=manifest_sha256,
        schema_version=schema_version,
        payload=payload,
        entity=entity,
        project=project,
        run_id=run_id,
        run_name=run_name,
        source_snapshot_id=source_snapshot_id,
        world_size=world_size,
        motion_clip_id=motion_clip_id,
        video=video,
        video_snapshot=video_snapshot,
        metadata=metadata,
    )


def _import_wandb() -> Any:
    configured_base_url = os.environ.get("WANDB_BASE_URL", "").strip().rstrip("/")
    if configured_base_url and configured_base_url != _WANDB_BASE_URL:
        raise PreflightError(
            "WANDB_BASE_URL differs from the pinned official W&B API endpoint"
        )
    try:
        import wandb  # type: ignore[import-not-found]
    except Exception as exc:
        raise PreflightError(f"wandb is unavailable: {_safe_message(exc)}") from exc
    return wandb


def _json_equal(left: object, right: object) -> bool:
    # W&B normalizes integral JSON floats such as ``50.0`` to the JSON number
    # ``50`` on round-trip.  JSON has one numeric type, so compare finite
    # non-boolean numbers by value while retaining exact recursive structure
    # and strict boolean/string/null semantics for every other field.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is bool and type(right) is bool and left is right
    if type(left) in (int, float) and type(right) in (int, float):
        return math.isfinite(float(left)) and math.isfinite(float(right)) and left == right
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False
        return all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _json_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    try:
        return json.dumps(left, sort_keys=True, separators=(",", ":"), allow_nan=False) == json.dumps(
            right, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError):
        return False


def _find_summary_videos(value: object, *, path: tuple[str, ...] = ()) -> list[tuple[str, dict[str, Any]]]:
    matches: list[tuple[str, dict[str, Any]]] = []
    if isinstance(value, dict):
        if value.get("_type") == "video-file":
            matches.append(("/".join(path), value))
        for key, child in value.items():
            matches.extend(_find_summary_videos(child, path=(*path, str(key))))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            matches.extend(_find_summary_videos(child, path=(*path, str(index))))
    return matches


def _remote_run_identity(remote: Any) -> tuple[str, str, str, str]:
    return (
        str(getattr(remote, "entity", "")),
        str(getattr(remote, "project", "")),
        str(getattr(remote, "id", "")),
        str(getattr(remote, "name", "")),
    )


def _plain_remote_value(value: object, *, role: str) -> Any:
    """Convert W&B's dict-like HTTPSummary wrappers into plain JSON values."""

    if isinstance(value, Mapping):
        return {str(key): _plain_remote_value(child, role=role) for key, child in value.items()}
    if isinstance(value, list):
        return [_plain_remote_value(child, role=role) for child in value]
    keys = getattr(value, "keys", None)
    getitem = getattr(value, "__getitem__", None)
    if callable(keys) and callable(getitem):
        try:
            return {
                str(key): _plain_remote_value(value[key], role=role)
                for key in keys()
            }
        except Exception as exc:
            raise PreflightError(f"could not decode remote W&B {role}") from exc
    return value


def _verify_rule90_v2_remote_prebind_only(
    remote: Any,
    *,
    summary: Mapping[str, Any],
) -> None:
    """Prove that a v2 W&B identity has never contained a training row.

    ``upload`` intentionally finishes the summary-only prebind before formal
    workers resume the same run.  Therefore the only admissible v2 state at
    launch time is ``finished`` with no summary step marker, no valid
    ``lastHistoryStep``, and an empty bounded history scan.  These independent
    checks close the old-run reuse gap without relying on private GraphQL APIs.
    """

    state = getattr(remote, "state", None)
    if type(state) is not str or state != "finished":
        raise PreflightError(
            "Rule-90 v2 remote run must be in summary-only prebind state 'finished'; "
            f"got {state!r}"
        )
    if "_step" in summary:
        raise PreflightError(
            "Rule-90 v2 remote summary contains _step and is not prebind-only"
        )

    last_history_step = getattr(remote, "lastHistoryStep", None)
    if last_history_step is not None:
        if (
            isinstance(last_history_step, bool)
            or not isinstance(last_history_step, int)
            or last_history_step != -1
        ):
            raise PreflightError(
                "Rule-90 v2 remote run has a valid or malformed lastHistoryStep "
                f"marker: {last_history_step!r}"
            )

    scan_history = getattr(remote, "scan_history", None)
    if not callable(scan_history):
        raise PreflightError(
            "Rule-90 v2 remote run does not expose the required bounded history scan"
        )
    try:
        first_rows = list(itertools.islice(scan_history(page_size=1), 1))
    except Exception as exc:
        raise PreflightError(
            "Rule-90 v2 remote history could not be proven empty"
        ) from exc
    if first_rows:
        raise PreflightError(
            "Rule-90 v2 remote run already contains a history row and cannot be reused"
        )


def _verify_remote_once(validated: ValidatedManifest, *, api_timeout_seconds: float) -> dict[str, Any]:
    wandb = _import_wandb()
    api = wandb.Api(
        overrides={"base_url": _WANDB_BASE_URL},
        timeout=api_timeout_seconds,
    )
    remote = api.run(validated.run_path)
    actual_identity = _remote_run_identity(remote)
    expected_identity = (
        validated.entity,
        validated.project,
        validated.run_id,
        validated.run_name,
    )
    if actual_identity != expected_identity:
        raise PreflightError("remote W&B run identity differs from the manifest/controller identity")

    # ``Run.summary`` is an HTTPSummary duck type rather than a
    # collections.abc.Mapping in W&B 0.26.1.  ``summary_metrics`` is the raw
    # backend JSON; retain the fallback for compatible public-API versions.
    raw_summary = getattr(remote, "summary_metrics", None)
    if not isinstance(raw_summary, Mapping):
        raw_summary = getattr(remote, "summary", None)
    raw_config = getattr(remote, "config", None)
    summary = _plain_remote_value(raw_summary, role="summary")
    config = _plain_remote_value(raw_config, role="config")
    if not isinstance(summary, dict) or not isinstance(config, dict):
        raise PreflightError("remote W&B run did not expose object-valued summary and config")
    if validated.schema_version == _RULE90_V2_SCHEMA_VERSION:
        _verify_rule90_v2_remote_prebind_only(remote, summary=summary)
    for key, expected in validated.metadata.items():
        if key not in summary or not _json_equal(summary[key], expected):
            raise PreflightError(f"remote W&B summary metadata mismatch at {key}")
        if key not in config or not _json_equal(config[key], expected):
            raise PreflightError(f"remote W&B config metadata mismatch at {key}")

    video_value = summary.get(_SUMMARY_VIDEO_KEY)
    if not isinstance(video_value, dict) or video_value.get("_type") != "video-file":
        raise PreflightError(f"remote summary key {_SUMMARY_VIDEO_KEY!r} is not a video-file")
    remote_sha = video_value.get("sha256")
    remote_path = video_value.get("path")
    remote_size = video_value.get("size")
    if remote_sha != validated.video.sha256:
        raise PreflightError("remote vis/replay SHA-256 differs from the reviewed MP4")
    if not isinstance(remote_path, str) or not remote_path.startswith("media/videos/") or not remote_path.endswith(".mp4"):
        raise PreflightError("remote vis/replay has an invalid or empty media path")
    if Path(remote_path).is_absolute() or ".." in Path(remote_path).parts:
        raise PreflightError("remote vis/replay media path is not a safe run-relative path")
    if type(remote_size) is not int or remote_size != validated.video.size_bytes or remote_size <= 0:
        raise PreflightError("remote vis/replay size differs from the non-empty reviewed MP4")

    summary_videos = _find_summary_videos(summary)
    if len(summary_videos) != 1 or summary_videos[0][0] != _SUMMARY_VIDEO_KEY:
        found = [path for path, _value in summary_videos]
        raise PreflightError(
            "remote summary must contain exactly one video-file at vis/replay; "
            f"found={found!r}"
        )

    # Limit iteration to two entries: one is required, and seeing a second is
    # already sufficient to fail the uniqueness contract.
    remote_mp4s = list(
        itertools.islice(remote.files(pattern="%.mp4", per_page=50), 2)
    )
    if len(remote_mp4s) != 1:
        raise PreflightError(
            f"remote W&B run must contain exactly one MP4 file; observed_at_least={len(remote_mp4s)}"
        )
    remote_file = remote_mp4s[0]
    if str(getattr(remote_file, "name", "")) != remote_path:
        raise PreflightError("remote MP4 run file does not match the vis/replay summary path")
    try:
        file_size = int(getattr(remote_file, "size"))
    except (TypeError, ValueError) as exc:
        raise PreflightError("remote MP4 run file has no usable size") from exc
    if file_size != validated.video.size_bytes or file_size <= 0:
        raise PreflightError("remote MP4 run file size differs from the reviewed local MP4")

    # ``run.files`` exposes path and size, while the summary exposes SHA-256.
    # Download the sole remote object and hash its actual bytes as well, closing
    # the same-size overwrite/stale-summary gap without trusting a second piece
    # of metadata from the same backend record.
    with tempfile.TemporaryDirectory(prefix="wandb-replay-verify-") as download_root:
        try:
            downloaded = remote_file.download(
                root=download_root,
                replace=True,
                api=api,
            )
            downloaded_path = Path(downloaded.name)
            downloaded.close()
        except Exception as exc:
            raise PreflightError(
                "could not download remote replay MP4 "
                f"({type(exc).__name__}; remote URL intentionally redacted)"
            ) from exc
        remote_snapshot, _ = _stable_file_snapshot(
            downloaded_path,
            role="downloaded remote replay MP4",
        )
        if (
            remote_snapshot.sha256 != validated.video.sha256
            or remote_snapshot.size != validated.video.size_bytes
        ):
            raise PreflightError("downloaded remote replay MP4 bytes differ from the reviewed local MP4")
    return {
        "run_path": validated.run_path,
        "summary_path": remote_path,
        "manifest_sha256": validated.sha256,
        "video_sha256": validated.video.sha256,
        "video_size_bytes": validated.video.size_bytes,
    }


T = TypeVar("T")


def _finite_retry(
    operation: Callable[[], T],
    *,
    attempts: int,
    initial_delay_seconds: float,
    role: str,
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # W&B exposes several network exception classes.
            last_error = exc
            if attempt == attempts:
                break
            delay = min(30.0, initial_delay_seconds * (2 ** (attempt - 1)))
            if delay > 0.0:
                time.sleep(delay)
    assert last_error is not None
    raise PreflightError(
        f"{role} failed after {attempts} bounded attempt(s): {_safe_message(last_error)}"
    ) from last_error


def _verify_remote(validated: ValidatedManifest, args: argparse.Namespace) -> dict[str, Any]:
    result = _finite_retry(
        lambda: _verify_remote_once(
            validated,
            api_timeout_seconds=args.api_timeout_seconds,
        ),
        attempts=args.attempts,
        initial_delay_seconds=args.retry_initial_seconds,
        role="W&B replay verification",
    )
    _assert_manifest_bytes_unchanged(validated)
    _assert_file_unchanged(validated.video_snapshot, role="replay MP4")
    return result


def _upload(validated: ValidatedManifest, args: argparse.Namespace) -> dict[str, Any]:
    wandb = _import_wandb()
    # Do not inherit an offline/disabled mode from a shell profile: this gate is
    # specifically a remote pre-bind contract.  Credentials remain exclusively
    # in W&B's normal environment/netrc handling and are never printed here.
    with tempfile.TemporaryDirectory(prefix="wandb-replay-preflight-") as wandb_dir:
        settings = wandb.Settings(
            base_url=_WANDB_BASE_URL,
            init_timeout=args.wandb_init_timeout_seconds,
            silent=True,
            console="off",
        )
        run = None
        finished = False
        try:
            run = wandb.init(
                entity=validated.entity,
                project=validated.project,
                id=validated.run_id,
                name=validated.run_name,
                resume="never",
                mode="online",
                dir=wandb_dir,
                settings=settings,
                config=validated.metadata,
                # This is the final formal training run, merely pre-bound by
                # the replay gate.  Keep its durable W&B semantics as training.
                job_type="training",
            )
            if run is None:
                raise PreflightError("wandb.init returned no run")
            if bool(getattr(run, "resumed", False)):
                raise PreflightError("W&B resumed an existing run despite the fresh-run contract")
            actual_identity = (
                str(getattr(run, "entity", "")),
                str(getattr(run, "project", "")),
                str(getattr(run, "id", "")),
                str(getattr(run, "name", "")),
            )
            expected_identity = (
                validated.entity,
                validated.project,
                validated.run_id,
                validated.run_name,
            )
            if actual_identity != expected_identity:
                raise PreflightError("new W&B run identity differs from the manifest/controller identity")

            # Summary assignment binds and uploads media without committing a
            # W&B history row.  Do not replace this with a history-writing API.
            run.summary[_SUMMARY_VIDEO_KEY] = wandb.Video(
                str(validated.video.path), format="mp4"
            )
            run.summary.update(validated.metadata)
            _assert_file_unchanged(validated.video_snapshot, role="replay MP4")
            run.finish(exit_code=0)
            finished = True
        except Exception:
            if run is not None and not finished:
                try:
                    run.finish(exit_code=1)
                except Exception:
                    pass
            raise

    _assert_file_unchanged(validated.video_snapshot, role="replay MP4")
    return _verify_remote(validated, args)


def _positive_bounded_int(raw: str, *, minimum: int, maximum: int, role: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{role} must be an integer") from exc
    if value < minimum or value > maximum:
        raise argparse.ArgumentTypeError(f"{role} must be in {minimum}..{maximum}")
    return value


def _bounded_float(raw: str, *, minimum: float, maximum: float, role: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{role} must be numeric") from exc
    if not math.isfinite(value) or value < minimum or value > maximum:
        raise argparse.ArgumentTypeError(f"{role} must be in {minimum}..{maximum}")
    return value


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-source-snapshot-id", required=True)
    parser.add_argument(
        "--required-manifest-version",
        type=int,
        choices=sorted(_SUPPORTED_SCHEMA_VERSIONS),
        help="require one exact replay schema; formal-fresh dual launchers pass 2",
    )
    parser.add_argument(
        "--expected-source-archive-sha256",
        help="v2-only controller binding for the authenticated source archive",
    )
    parser.add_argument(
        "--expected-entrypoint-archive-member",
        help="v2-only exact root-level member selected from the source archive",
    )
    parser.add_argument(
        "--expected-entrypoint-sha256",
        help="v2-only digest of the selected authenticated archive member",
    )
    parser.add_argument("--expected-entity", required=True)
    parser.add_argument("--expected-project", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-run-name", required=True)
    parser.add_argument(
        "--expected-world-size",
        required=True,
        type=lambda raw: _positive_bounded_int(
            raw, minimum=1, maximum=4096, role="--expected-world-size"
        ),
    )
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument(
        "--ffprobe-timeout-seconds",
        default=60.0,
        type=lambda raw: _bounded_float(
            raw, minimum=1.0, maximum=300.0, role="--ffprobe-timeout-seconds"
        ),
    )
    parser.add_argument(
        "--api-timeout-seconds",
        default=30.0,
        type=lambda raw: _bounded_float(
            raw, minimum=1.0, maximum=120.0, role="--api-timeout-seconds"
        ),
    )
    parser.add_argument(
        "--attempts",
        default=5,
        type=lambda raw: _positive_bounded_int(raw, minimum=1, maximum=10, role="--attempts"),
    )
    parser.add_argument(
        "--retry-initial-seconds",
        default=2.0,
        type=lambda raw: _bounded_float(
            raw, minimum=0.0, maximum=30.0, role="--retry-initial-seconds"
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    upload = subparsers.add_parser(
        "upload",
        help="validate locally, create a fresh run, bind summary vis/replay, and verify remotely",
    )
    _add_common_arguments(upload)
    upload.add_argument(
        "--wandb-init-timeout-seconds",
        default=60.0,
        type=lambda raw: _bounded_float(
            raw, minimum=1.0, maximum=300.0, role="--wandb-init-timeout-seconds"
        ),
    )
    verify = subparsers.add_parser(
        "verify",
        help="validate local inputs and prove the existing remote vis/replay contract",
    )
    _add_common_arguments(verify)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        validated = _validate_manifest(args)
        if args.action == "upload":
            result = _upload(validated, args)
        elif args.action == "verify":
            result = _verify_remote(validated, args)
        else:  # argparse enforces this, but keep the dispatch fail-closed.
            raise PreflightError(f"unsupported action: {args.action!r}")
    except PreflightError as exc:
        print(f"[ERROR] wandb replay preflight: {_safe_message(exc)}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("[ERROR] wandb replay preflight: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            f"[ERROR] wandb replay preflight: unexpected failure: {_safe_message(exc)}",
            file=sys.stderr,
        )
        return 3

    output = {
        "action": args.action,
        "status": "ok",
        "run_path": result["run_path"],
        "manifest_sha256": result["manifest_sha256"],
        "video_sha256": result["video_sha256"],
        "video_size_bytes": result["video_size_bytes"],
        "summary_key": _SUMMARY_VIDEO_KEY,
        "summary_path": result["summary_path"],
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
