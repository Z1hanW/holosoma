#!/usr/bin/env python3
"""Stage the exact scientific Python dependency closure for AS training.

The launcher intentionally does not mutate node-local Conda environments.  A
controller builds this overlay from its validated hssim interpreter, seals the
complete tree, and distributes the resulting content-addressed archive. The
roots cover numerical arrays and configuration semantics. ``attrs`` is explicit because OmegaConf probes
it at import time even though OmegaConf does not declare it in Requires-Dist.
The Hugging Face network client is absent because scientific HoloSoma DeFM
launches require a local authenticated checkpoint and never download weights.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import importlib.machinery
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import sysconfig
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

_SCHEMA_PATH = Path(__file__).absolute().with_name("python_runtime_schema.py")
if not _SCHEMA_PATH.is_file() or _SCHEMA_PATH.is_symlink():
    raise RuntimeError(f"Python runtime schema is missing or aliased: {_SCHEMA_PATH}")
_SCHEMA_NAMESPACE: dict[str, Any] = {}
exec(
    compile(_SCHEMA_PATH.read_bytes(), str(_SCHEMA_PATH), "exec"),
    _SCHEMA_NAMESPACE,
)
DISTRIBUTION_CONTRACT_NAME = _SCHEMA_NAMESPACE["DISTRIBUTION_CONTRACT_NAME"]
DISTRIBUTION_CONTRACT_VERSION = _SCHEMA_NAMESPACE["DISTRIBUTION_CONTRACT_VERSION"]
ROOT_DISTRIBUTIONS = _SCHEMA_NAMESPACE["ROOT_DISTRIBUTIONS"]
RUNTIME_PROFILE = _SCHEMA_NAMESPACE["RUNTIME_PROFILE"]
if (
    not isinstance(DISTRIBUTION_CONTRACT_NAME, str)
    or not isinstance(DISTRIBUTION_CONTRACT_VERSION, int)
    or not isinstance(ROOT_DISTRIBUTIONS, tuple)
    or not ROOT_DISTRIBUTIONS
    or not all(isinstance(name, str) for name in ROOT_DISTRIBUTIONS)
    or not isinstance(RUNTIME_PROFILE, str)
):
    raise RuntimeError(f"Python runtime schema constants are malformed: {_SCHEMA_PATH}")

_SAFE_VERSION_RE = re.compile(r"^[\x21-\x7e]{1,128}$")
_IMPORT_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BYTECODE_SUFFIXES = (".pyc", ".pyo")
_FORBIDDEN_TOP_LEVEL_FILES = {"sitecustomize.py", "usercustomize.py"}


class StagingError(RuntimeError):
    """The active interpreter cannot produce an exact scientific overlay."""


def _inside(path: Path, root: Path) -> Path | None:
    try:
        return path.relative_to(root)
    except ValueError:
        return None


def _stable_copy(source: Path, destination: Path) -> None:
    before = os.stat(source, follow_symlinks=True)
    if not stat.S_ISREG(before.st_mode):
        raise StagingError(f"distribution payload is not a regular file: {source}")

    source_digest = hashlib.sha256()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink():
            raise StagingError(f"overlay payload collision is not a regular file: {destination}")
        with source.open("rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                source_digest.update(chunk)
        destination_digest = hashlib.sha256(destination.read_bytes()).digest()
        if source_digest.digest() != destination_digest:
            raise StagingError(
                f"distributions own different bytes at the same overlay path: {destination}"
            )
    else:
        with source.open("rb") as input_file, destination.open("xb") as output_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                source_digest.update(chunk)
                output_file.write(chunk)

    after = os.stat(source, follow_symlinks=True)
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise StagingError(f"distribution payload changed while being copied: {source}")


def _active_requirement(raw: str) -> Requirement | None:
    try:
        requirement = Requirement(raw)
    except InvalidRequirement as exc:
        raise StagingError(f"invalid Requires-Dist entry {raw!r}") from exc
    if requirement.marker is None:
        return requirement
    environment = default_environment()
    environment["extra"] = ""
    return requirement if requirement.marker.evaluate(environment) else None


def _resolve_distributions() -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    pending: deque[tuple[str, Requirement | None]] = deque(
        (name, None) for name in ROOT_DISTRIBUTIONS
    )
    while pending:
        requested_name, incoming = pending.popleft()
        try:
            distribution = importlib.metadata.distribution(requested_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise StagingError(
                f"required scientific distribution is missing: {requested_name}"
            ) from exc
        declared_name = str(distribution.metadata.get("Name") or requested_name)
        canonical_name = canonicalize_name(declared_name)
        version_text = str(distribution.version)
        if not _SAFE_VERSION_RE.fullmatch(version_text):
            raise StagingError(
                f"distribution {declared_name!r} has an unsafe/empty version: {version_text!r}"
            )
        try:
            parsed_version = Version(version_text)
        except InvalidVersion as exc:
            raise StagingError(
                f"distribution {declared_name!r} has an invalid version: {version_text!r}"
            ) from exc
        if incoming is not None and not incoming.specifier.contains(
            parsed_version, prereleases=True
        ):
            raise StagingError(
                f"installed {declared_name}=={version_text} does not satisfy {incoming}"
            )
        if canonical_name in records:
            continue

        requirements: list[dict[str, str]] = []
        for raw_requirement in distribution.requires or ():
            requirement = _active_requirement(raw_requirement)
            if requirement is None:
                continue
            dependency_name = canonicalize_name(requirement.name)
            requirements.append(
                {"name": dependency_name, "specifier": str(requirement.specifier)}
            )
            pending.append((requirement.name, requirement))
        records[canonical_name] = {
            "distribution": distribution,
            "canonical_name": canonical_name,
            "version": version_text,
            "requirements": sorted(
                requirements, key=lambda value: (value["name"], value["specifier"])
            ),
        }

    roots = sorted(canonicalize_name(name) for name in ROOT_DISTRIBUTIONS)
    missing_roots = sorted(set(roots).difference(records))
    if missing_roots:
        raise StagingError(f"scientific overlay root closure is incomplete: {missing_roots}")
    return records, roots


def _source_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        value = sysconfig.get_path(key)
        if not value:
            continue
        path = Path(value).resolve()
        if path not in roots:
            roots.append(path)
    if not roots:
        raise StagingError("active interpreter exposes no purelib/platlib path")
    return tuple(roots)


def _import_root_for_file(relative: PurePosixPath) -> str | None:
    parts = relative.parts
    if not parts:
        return None
    top = parts[0]
    if len(parts) > 1:
        if len(parts) == 2 and parts[1] == "__init__.py" and _IMPORT_NAME_RE.fullmatch(top):
            return top
        return None

    name = relative.name
    if name.endswith(".py"):
        candidate = name[:-3]
        return candidate if _IMPORT_NAME_RE.fullmatch(candidate) else None
    for suffix in sorted(importlib.machinery.EXTENSION_SUFFIXES, key=len, reverse=True):
        if name.endswith(suffix):
            candidate = name[: -len(suffix)]
            return candidate if _IMPORT_NAME_RE.fullmatch(candidate) else None
    return None


def stage(site_packages: Path) -> dict[str, Any]:
    try:
        root_mode = os.lstat(site_packages).st_mode
    except FileNotFoundError as exc:
        raise StagingError(f"overlay staging root is missing: {site_packages}") from exc
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        raise StagingError(f"overlay staging root is not a real directory: {site_packages}")
    if any(site_packages.iterdir()):
        raise StagingError(f"overlay staging root must initially be empty: {site_packages}")

    records, roots = _resolve_distributions()
    source_roots = _source_roots()
    prefix_bin = (Path(sys.prefix) / "bin").resolve()
    omitted_console_scripts: list[dict[str, str]] = []
    copied_paths: dict[str, str] = {}

    for canonical_name in sorted(records):
        record = records[canonical_name]
        distribution = record["distribution"]
        declared_files = distribution.files
        if declared_files is None:
            raise StagingError(
                f"distribution {canonical_name!r} exposes no installed-file manifest"
            )
        owned_paths: set[str] = set()
        import_roots: set[str] = set()
        for declared_path in sorted(declared_files, key=lambda value: str(value)):
            # Preserve the RECORD path lexically. Resolving a declared symlink
            # before deriving the destination can move its payload to the
            # target's name in the overlay.
            source = Path(
                os.path.abspath(os.fspath(distribution.locate_file(declared_path)))
            )
            relative: Path | None = None
            for source_root in source_roots:
                relative = _inside(source, source_root)
                if relative is not None:
                    break
            if relative is None:
                console_relative = _inside(source, prefix_bin)
                if console_relative is None or len(console_relative.parts) != 1:
                    raise StagingError(
                        f"distribution {canonical_name!r} owns an unsupported payload outside "
                        f"site-packages: {declared_path}"
                    )
                omitted_console_scripts.append(
                    {
                        "distribution": canonical_name,
                        "path": str(declared_path).replace(os.sep, "/"),
                    }
                )
                continue

            pure_relative = PurePosixPath(relative.as_posix())
            # Bytecode is cache material, not an input. Some installers retain
            # stale RECORD rows after cache cleanup, so classify it before
            # requiring the source path to exist.
            if (
                "__pycache__" in pure_relative.parts
                or pure_relative.name.endswith(_BYTECODE_SUFFIXES)
            ):
                continue
            try:
                source_mode = os.lstat(source).st_mode
            except FileNotFoundError as exc:
                raise StagingError(
                    f"distribution {canonical_name!r} declares a missing payload: "
                    f"{declared_path}"
                ) from exc
            if stat.S_ISLNK(source_mode) or not stat.S_ISREG(source_mode):
                raise StagingError(
                    f"distribution {canonical_name!r} declares a symlink/non-file payload: "
                    f"{declared_path}"
                )
            if len(pure_relative.parts) == 1 and (
                pure_relative.name in _FORBIDDEN_TOP_LEVEL_FILES
                or pure_relative.name.endswith(".pth")
            ):
                raise StagingError(
                    f"scientific dependency would inject startup code: {pure_relative}"
                )
            destination = site_packages.joinpath(*pure_relative.parts)
            previous_owner = copied_paths.get(pure_relative.as_posix())
            _stable_copy(source, destination)
            if previous_owner is None:
                copied_paths[pure_relative.as_posix()] = canonical_name
            owned_paths.add(pure_relative.as_posix())
            import_root = _import_root_for_file(pure_relative)
            if import_root is not None:
                import_roots.add(import_root)

        if not owned_paths:
            raise StagingError(f"distribution {canonical_name!r} contributed no payload files")
        if not import_roots:
            raise StagingError(
                f"distribution {canonical_name!r} exposes no auditable top-level import"
            )
        record["payload_file_count"] = len(owned_paths)
        record["import_roots"] = sorted(import_roots)

    contract = {
        "version": DISTRIBUTION_CONTRACT_VERSION,
        "runtime_profile": RUNTIME_PROFILE,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_version": platform.python_version(),
        "python_soabi": str(sysconfig.get_config_var("SOABI") or ""),
        "platform_machine": platform.machine(),
        "root_distributions": roots,
        "distributions": [
            {
                "canonical_name": canonical_name,
                "version": records[canonical_name]["version"],
                "requirements": records[canonical_name]["requirements"],
                "import_roots": records[canonical_name]["import_roots"],
                "payload_file_count": records[canonical_name]["payload_file_count"],
            }
            for canonical_name in sorted(records)
        ],
        "omitted_console_scripts": sorted(
            omitted_console_scripts,
            key=lambda value: (value["distribution"], value["path"]),
        ),
    }
    contract_path = site_packages / DISTRIBUTION_CONTRACT_NAME
    contract_path.write_text(
        json.dumps(contract, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", required=True, type=Path)
    args = parser.parse_args()
    try:
        contract = stage(args.site_packages.absolute())
    except (OSError, StagingError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    print(
        "[INFO] python_runtime_distribution_closure_staged="
        f"{args.site_packages.absolute()} distributions={len(contract['distributions'])} "
        f"roots={','.join(contract['root_distributions'])}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
