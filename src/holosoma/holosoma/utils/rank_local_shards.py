from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from holosoma.utils.motion_transition_source import (
    MOTION_TRANSITION_SOURCE_KEY,
    canonical_motion_transition_source,
)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_OBJECT_MAP_NAME = "_clip_object_urdf_map.json"
_MANIFEST_NAME = "manifest.json"
_MANIFEST_VERSION = 3
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    raise RuntimeError(
        f"{name} must be a boolean (0/1/false/true/no/yes/off/on), "
        f"got {os.environ.get(name)!r}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, *, role: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse {role} from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{role} must be a JSON object: {path}")
    return payload


def _require_sha256(value: Any, *, role: str) -> str:
    digest = str(value)
    if _SHA256_RE.fullmatch(digest) is None:
        raise RuntimeError(f"{role} must be a 64-character lowercase SHA256 digest")
    return digest


def _require_finite_positive_float(value: Any, *, role: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{role} must be a finite positive number")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{role} must be a finite positive number") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise RuntimeError(f"{role} must be a finite positive number")
    return parsed


def validate_rank_local_shard_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    expected_clip_ids: set[str] | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    """Validate a format-v3 rank-shard assignment/content manifest.

    This validates the complete assignment table and its declared per-shard
    content records.  Worker-side validation below additionally hashes the
    selected shard's actual files before returning any rank-local input path.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Rank-local shard manifest does not exist: {path}")
    manifest = _load_json_object(path, role="rank-local shard manifest")
    try:
        version = int(manifest["version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Rank-local shard manifest predates the content-closed format-v{_MANIFEST_VERSION} schema: {path}"
        ) from exc
    if version != _MANIFEST_VERSION:
        raise RuntimeError(
            f"Unsupported rank-local shard manifest version {version}; expected {_MANIFEST_VERSION}: {path}"
        )

    try:
        world_size = int(manifest["world_size"])
        clip_count = int(manifest["clip_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Rank-local shard manifest has invalid world_size/clip_count: {path}") from exc
    if world_size < 1 or clip_count < 1:
        raise RuntimeError(f"Rank-local shard manifest world_size/clip_count must be positive: {path}")
    if expected_world_size is not None and world_size != expected_world_size:
        raise RuntimeError(
            "Rank-local shard world size does not match the active distributed launch: "
            f"runtime={expected_world_size}, manifest={world_size}, path={path}. "
            "Regenerate shards for the current WORLD_SIZE."
        )
    try:
        manifest_transition_source = canonical_motion_transition_source(
            manifest.get(MOTION_TRANSITION_SOURCE_KEY),
            active_clip_count=clip_count,
            role=f"rank-local manifest {MOTION_TRANSITION_SOURCE_KEY}",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc

    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != world_size:
        raise RuntimeError(
            f"Rank-local shard manifest must contain exactly {world_size} shard records: {path}"
        )

    ranks: set[int] = set()
    computed_cover_counts: dict[str, int] = {}
    for shard_index, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise RuntimeError(f"Rank-local shard record {shard_index} must be a JSON object: {path}")
        try:
            rank = int(shard["rank"])
            local_clip_count = int(shard["clip_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Rank-local shard record {shard_index} has invalid rank/clip_count: {path}") from exc
        if rank < 0 or rank >= world_size or rank in ranks:
            raise RuntimeError(f"Rank-local shard manifest has invalid or duplicate rank {rank}: {path}")
        ranks.add(rank)

        clip_ids = shard.get("clip_ids")
        if (
            not isinstance(clip_ids, list)
            or not clip_ids
            or not all(isinstance(clip_id, str) and clip_id for clip_id in clip_ids)
            or clip_ids != sorted(set(clip_ids))
            or local_clip_count != len(clip_ids)
        ):
            raise RuntimeError(f"Rank-local shard record for rank {rank} has invalid exact clip_ids: {path}")
        for clip_id in clip_ids:
            computed_cover_counts[clip_id] = computed_cover_counts.get(clip_id, 0) + 1

        _require_sha256(shard.get("object_map_sha256"), role=f"rank {rank} object_map_sha256")
        npz_files = shard.get("npz_files")
        if not isinstance(npz_files, list) or len(npz_files) != len(clip_ids):
            raise RuntimeError(f"Rank-local shard record for rank {rank} has invalid npz_files: {path}")
        expected_names = [f"{clip_id}.npz" for clip_id in clip_ids]
        actual_names: list[str] = []
        for file_index, file_record in enumerate(npz_files):
            if not isinstance(file_record, dict):
                raise RuntimeError(
                    f"Rank-local shard NPZ record {rank}:{file_index} must be a JSON object: {path}"
                )
            name = file_record.get("name")
            try:
                size = int(file_record["size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Rank-local shard NPZ record {rank}:{file_index} has invalid size: {path}"
                ) from exc
            if not isinstance(name, str) or size < 0:
                raise RuntimeError(f"Rank-local shard NPZ record {rank}:{file_index} is invalid: {path}")
            _require_sha256(file_record.get("sha256"), role=f"rank {rank} NPZ {name!r} sha256")
            actual_names.append(name)
        if actual_names != expected_names:
            raise RuntimeError(
                f"Rank-local shard NPZ records do not match exact clip_ids for rank {rank}: {path}"
            )
        stored_npz_digest = _require_sha256(
            shard.get("npz_content_sha256"), role=f"rank {rank} npz_content_sha256"
        )
        computed_npz_digest = _sha256_json(npz_files)
        if stored_npz_digest != computed_npz_digest:
            raise RuntimeError(
                f"Rank-local shard NPZ aggregate digest is inconsistent for rank {rank}: {path}"
            )

    if ranks != set(range(world_size)):
        raise RuntimeError(f"Rank-local shard manifest does not cover ranks [0, {world_size}): {path}")
    if len(computed_cover_counts) != clip_count:
        raise RuntimeError(
            "Rank-local shard manifest clip_count does not match the union of exact clip_ids: "
            f"declared={clip_count}, actual={len(computed_cover_counts)}, path={path}"
        )
    if expected_clip_ids is not None and set(computed_cover_counts) != expected_clip_ids:
        raise RuntimeError(
            "Rank-local shard manifest does not cover the active motion bank exactly: "
            f"manifest_only={sorted(set(computed_cover_counts) - expected_clip_ids)}, "
            f"motion_only={sorted(expected_clip_ids - set(computed_cover_counts))}, path={path}"
        )
    declared_cover_counts = manifest.get("clip_cover_counts")
    if declared_cover_counts != computed_cover_counts:
        raise RuntimeError(
            f"Rank-local shard manifest clip_cover_counts do not match its shard assignments: {path}"
        )
    exact_partition = all(count == 1 for count in computed_cover_counts.values())
    if manifest.get("exact_clip_partition") is not exact_partition:
        raise RuntimeError(f"Rank-local shard manifest exact_clip_partition is inconsistent: {path}")
    duplicated = any(count > 1 for count in computed_cover_counts.values())
    if manifest.get("duplicated_to_fill_empty_ranks") is not duplicated:
        raise RuntimeError(
            f"Rank-local shard manifest duplicated_to_fill_empty_ranks is inconsistent: {path}"
        )
    return manifest


def rank_local_sharding_enabled() -> bool:
    root = os.environ.get("HOLOSOMA_RANK_LOCAL_MOTION_ROOT", "").strip()
    if not root:
        return False
    return _env_flag("HOLOSOMA_RANK_LOCAL_SHARDING_ENABLED", default=True)


def current_rank_local_shard_dir() -> Path | None:
    if not rank_local_sharding_enabled():
        return None

    global_rank_raw = os.environ.get("RANK", "")
    local_rank_raw = os.environ.get("LOCAL_RANK", "")
    world_size_raw = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(world_size_raw or "1")
    except ValueError as exc:
        raise RuntimeError(
            f"Invalid WORLD_SIZE for rank-local shard selection: {world_size_raw!r}"
        ) from exc
    if world_size < 1:
        raise RuntimeError(
            f"Invalid non-positive WORLD_SIZE for rank-local shard selection: {world_size}"
        )

    # Multi-node torchrun sets RANK to the global rank and LOCAL_RANK to the
    # node-local GPU index. Rank-local shards are generated for the global
    # world, so RANK is authoritative whenever torchrun supplied it.  This is
    # also correct for a one-node torchrun launch where global and local ranks
    # happen to be equal.
    rank_raw = global_rank_raw or local_rank_raw or "0"
    try:
        rank = int(rank_raw)
    except ValueError as exc:
        raise RuntimeError(f"Invalid rank for rank-local shard selection: {rank_raw!r}") from exc
    if rank < 0:
        raise RuntimeError(f"Invalid negative rank for rank-local shard selection: {rank}")
    if os.environ.get("WORLD_SIZE", "").strip() and rank >= world_size:
        raise RuntimeError(
            "Rank-local shard rank is outside the active distributed world: "
            f"rank={rank}, WORLD_SIZE={world_size}"
        )

    root = Path(os.environ["HOLOSOMA_RANK_LOCAL_MOTION_ROOT"]).expanduser().resolve()
    shard_dir = root / f"rank_{rank}"
    if not shard_dir.is_dir():
        raise FileNotFoundError(
            f"Rank-local shard directory does not exist for rank {rank}: {shard_dir}. "
            "Check HOLOSOMA_RANK_LOCAL_MOTION_ROOT and NPROC/WORLD_SIZE."
        )
    return shard_dir


def _strict_content_provenance_required() -> bool:
    # torchrun always provides WORLD_SIZE, including one-worker scientific
    # launches.  Legacy manifests remain usable only for explicitly
    # non-distributed inspection/inference where WORLD_SIZE is absent.  An
    # operator may opt that path into strict validation with the explicit flag.
    return bool(os.environ.get("WORLD_SIZE", "").strip()) or _env_flag(
        "HOLOSOMA_REQUIRE_RANK_LOCAL_SHARD_PROVENANCE",
        default=False,
    )


@lru_cache(maxsize=None)
def _validated_rank_local_shard_payload(
    shard_dir_text: str,
    *,
    strict: bool,
) -> dict[str, Any]:
    shard_dir = Path(shard_dir_text)
    object_map = shard_dir / _OBJECT_MAP_NAME
    if not object_map.is_file():
        raise FileNotFoundError(f"Rank-local object map does not exist for shard {shard_dir}: {object_map}")
    payload = _load_json_object(object_map, role="rank-local object map")

    manifest_path = shard_dir.parent / _MANIFEST_NAME
    if not manifest_path.is_file():
        if strict:
            raise RuntimeError(
                f"Distributed/scientific rank-local shard use requires a content-closed format-v{_MANIFEST_VERSION} manifest; "
                f"missing {manifest_path}. Regenerate shards with scripts/prepare_as_rank_shards.py."
            )
        return payload

    # A manifest created by the pre-closure implementation is accepted only by
    # the non-distributed legacy path described above.  If a current manifest is
    # present, validate it even for local inspection so corruption is never
    # silently ignored.
    raw_manifest = _load_json_object(manifest_path, role="rank-local shard manifest")
    try:
        manifest_version = int(raw_manifest["version"])
    except (KeyError, TypeError, ValueError):
        if strict:
            raise RuntimeError(
                "Distributed/scientific rank-local shard use rejects legacy manifests without "
                f"content digests: {manifest_path}. Regenerate the shards."
            ) from None
        return payload
    if manifest_version != _MANIFEST_VERSION and not strict:
        return payload

    runtime_world_size: int | None = None
    runtime_world_size_raw = os.environ.get("WORLD_SIZE", "").strip()
    if runtime_world_size_raw:
        try:
            runtime_world_size = int(runtime_world_size_raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid WORLD_SIZE for rank-local shard validation: {runtime_world_size_raw!r}") from exc
    manifest = validate_rank_local_shard_manifest(
        manifest_path,
        expected_world_size=runtime_world_size,
    )

    try:
        selected_rank = int(shard_dir.name.removeprefix("rank_"))
    except ValueError as exc:
        raise RuntimeError(f"Invalid rank-local shard directory name: {shard_dir}") from exc
    shard_records = [shard for shard in manifest["shards"] if int(shard["rank"]) == selected_rank]
    if len(shard_records) != 1:
        raise RuntimeError(
            f"Rank-local shard manifest has no unique record for selected rank {selected_rank}: {manifest_path}"
        )
    shard_record = shard_records[0]
    expected_clip_ids = list(shard_record["clip_ids"])

    actual_npz_paths = sorted(shard_dir.glob("*.npz"), key=lambda path: path.name)
    actual_npz_names = [path.name for path in actual_npz_paths]
    expected_npz_names = [f"{clip_id}.npz" for clip_id in expected_clip_ids]
    if actual_npz_names != expected_npz_names:
        raise RuntimeError(
            "Rank-local shard NPZ set does not match its manifest: "
            f"rank={selected_rank}, expected={expected_npz_names}, actual={actual_npz_names}"
        )
    for path in actual_npz_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Rank-local shard NPZ is missing or has a dangling symlink: {path}")

    clips = payload.get("clips")
    if not isinstance(clips, dict) or sorted(str(clip_id) for clip_id in clips) != expected_clip_ids:
        raise RuntimeError(
            "Rank-local object-map clips do not match the manifest's exact clip_ids: "
            f"rank={selected_rank}, expected={expected_clip_ids}, "
            f"actual={sorted(str(clip_id) for clip_id in clips) if isinstance(clips, dict) else '<invalid>'}"
        )
    metadata = payload.get("rank_local_shard")
    try:
        metadata_rank = int(metadata["rank"])
        metadata_world_size = int(metadata["world_size"])
        metadata_global_clip_count = int(metadata["global_clip_count"])
        metadata_cover_counts = metadata["clip_cover_counts"]
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Rank-local object map has invalid assignment metadata for rank {selected_rank}: {object_map}"
        ) from exc
    expected_local_cover_counts = {
        clip_id: int(manifest["clip_cover_counts"][clip_id]) for clip_id in expected_clip_ids
    }
    if (
        metadata_rank != selected_rank
        or metadata_world_size != int(manifest["world_size"])
        or metadata_global_clip_count != int(manifest["clip_count"])
        or metadata_cover_counts != expected_local_cover_counts
    ):
        raise RuntimeError(
            "Rank-local object-map assignment metadata does not match the manifest: "
            f"rank={selected_rank}, map={object_map}"
        )

    try:
        map_transition_source = canonical_motion_transition_source(
            payload.get(MOTION_TRANSITION_SOURCE_KEY),
            active_clip_count=int(manifest["clip_count"]),
            role=f"rank-local object map {MOTION_TRANSITION_SOURCE_KEY}",
        )
        rank_transition_source = canonical_motion_transition_source(
            metadata.get(MOTION_TRANSITION_SOURCE_KEY),
            active_clip_count=int(manifest["clip_count"]),
            role=f"rank_local_shard.{MOTION_TRANSITION_SOURCE_KEY}",
        )
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    manifest_transition_source = manifest[MOTION_TRANSITION_SOURCE_KEY]
    if not (
        map_transition_source == rank_transition_source == manifest_transition_source
    ):
        raise RuntimeError(
            "Rank-local motion transition provenance does not match across manifest, "
            f"object-map root, and rank metadata: rank={selected_rank}, map={object_map}"
        )

    inverse_cover_mass = _require_finite_positive_float(
        metadata.get("inverse_cover_mass"),
        role=f"rank {selected_rank} inverse_cover_mass",
    )
    distributed_loss_weight = _require_finite_positive_float(
        metadata.get("distributed_loss_weight"),
        role=f"rank {selected_rank} distributed_loss_weight",
    )
    expected_inverse_cover_mass = sum(
        1.0 / float(expected_local_cover_counts[clip_id]) for clip_id in expected_clip_ids
    )
    expected_distributed_loss_weight = (
        float(manifest["world_size"])
        * expected_inverse_cover_mass
        / float(manifest["clip_count"])
    )
    if not math.isclose(
        inverse_cover_mass,
        expected_inverse_cover_mass,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "Rank-local inverse_cover_mass does not match the global clip-cover formula: "
            f"rank={selected_rank}, actual={inverse_cover_mass}, "
            f"expected={expected_inverse_cover_mass}, map={object_map}"
        )
    if not math.isclose(
        distributed_loss_weight,
        expected_distributed_loss_weight,
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "Rank-local distributed_loss_weight does not match the world/global cover formula: "
            f"rank={selected_rank}, actual={distributed_loss_weight}, "
            f"expected={expected_distributed_loss_weight}, map={object_map}"
        )

    actual_object_map_sha256 = _sha256_file(object_map)
    if actual_object_map_sha256 != shard_record["object_map_sha256"]:
        raise RuntimeError(
            "Rank-local object-map content digest mismatch: "
            f"rank={selected_rank}, actual={actual_object_map_sha256}, "
            f"expected={shard_record['object_map_sha256']}, map={object_map}"
        )

    actual_npz_records = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in actual_npz_paths
    ]
    if actual_npz_records != shard_record["npz_files"]:
        raise RuntimeError(
            "Rank-local NPZ content records do not match the manifest: "
            f"rank={selected_rank}, shard={shard_dir}"
        )
    actual_npz_content_sha256 = _sha256_json(actual_npz_records)
    if actual_npz_content_sha256 != shard_record["npz_content_sha256"]:
        raise RuntimeError(
            "Rank-local NPZ aggregate content digest mismatch: "
            f"rank={selected_rank}, actual={actual_npz_content_sha256}, "
            f"expected={shard_record['npz_content_sha256']}"
        )
    return payload


def current_rank_local_shard_metadata() -> dict[str, Any] | None:
    """Load and validate metadata embedded in the selected rank-local shard."""
    shard_dir = current_rank_local_shard_dir()
    if shard_dir is None:
        return None
    object_map = shard_dir / _OBJECT_MAP_NAME
    payload = _validated_rank_local_shard_payload(
        str(shard_dir),
        strict=_strict_content_provenance_required(),
    )
    metadata = payload.get("rank_local_shard") if isinstance(payload, dict) else None
    if not isinstance(metadata, dict):
        raise RuntimeError(f"Rank-local object map is missing rank_local_shard metadata: {object_map}")

    try:
        selected_rank = int(shard_dir.name.removeprefix("rank_"))
        metadata_rank = int(metadata["rank"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Rank-local object map has invalid rank metadata: {object_map}") from exc
    if metadata_rank != selected_rank:
        raise RuntimeError(
            "Rank-local shard metadata does not match the selected shard directory: "
            f"directory_rank={selected_rank}, metadata_rank={metadata_rank}, map={object_map}"
        )

    # During distributed training WORLD_SIZE is authoritative.  A stale shard
    # root generated for a different launch size changes both clip coverage and
    # the per-rank gradient multiplier, so silently accepting it biases the
    # global objective.  WORLD_SIZE may be absent for single-process inference;
    # in that case loading rank_0 for inspection remains supported.
    runtime_world_size_raw = os.environ.get("WORLD_SIZE", "").strip()
    if runtime_world_size_raw:
        try:
            runtime_world_size = int(runtime_world_size_raw)
            metadata_world_size = int(metadata["world_size"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Rank-local object map has invalid world-size metadata: {object_map}") from exc
        if runtime_world_size != metadata_world_size:
            raise RuntimeError(
                "Rank-local shard world size does not match the active distributed launch: "
                f"runtime={runtime_world_size}, metadata={metadata_world_size}, map={object_map}. "
                "Regenerate shards for the current WORLD_SIZE."
            )
    return metadata


def build_clip_weighted_object_assignment(
    available_urdf_paths: list[str],
    per_clip_urdf_sequence: list[str] | None,
    *,
    num_envs: int,
) -> list[str]:
    """Assign objects by clip frequency while keeping spawned assets unique."""
    if not available_urdf_paths or num_envs <= 0:
        return []
    available = set(available_urdf_paths)
    sequence = [path for path in (per_clip_urdf_sequence or []) if path in available]
    source = sequence or available_urdf_paths
    return [source[env_id % len(source)] for env_id in range(num_envs)]


def resolve_rank_local_motion_path(path: str | os.PathLike[str]) -> str:
    shard_dir = current_rank_local_shard_dir()
    if shard_dir is None:
        return str(path)
    _validated_rank_local_shard_payload(
        str(shard_dir),
        strict=_strict_content_provenance_required(),
    )

    original = Path(path).expanduser()
    if original.is_dir():
        return str(shard_dir)
    return str(path)


def resolve_rank_local_object_map(path: str | os.PathLike[str]) -> str:
    shard_dir = current_rank_local_shard_dir()
    if shard_dir is None:
        return str(path)
    _validated_rank_local_shard_payload(
        str(shard_dir),
        strict=_strict_content_provenance_required(),
    )

    object_map = shard_dir / _OBJECT_MAP_NAME
    if not object_map.is_file():
        raise FileNotFoundError(
            f"Rank-local object map does not exist for shard {shard_dir}: {object_map}"
        )
    return str(object_map)
