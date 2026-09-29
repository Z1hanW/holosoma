"""Identify installed policy code and verify Git-installed runtime packages."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess


PACKAGES = {"holosoma": "holosoma", "holosoma-inference": "holosoma_inference"}


def _package_root(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    locations = list(spec.submodule_search_locations or []) if spec else []
    if len(locations) != 1:
        raise ValueError(f"Expected one installed package directory for {name}")
    return Path(locations[0]).resolve()


def _source_files(root: Path) -> dict[str, str]:
    result = {}
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Runtime source must be a regular file: {path}")
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not result:
        raise ValueError(f"No runtime source files found: {root}")
    return result


def installed_source_manifest() -> dict[str, dict[str, str]]:
    """Hash the code Python actually resolves, independently of checkout paths."""
    return {name: _source_files(_package_root(name)) for name in PACKAGES.values()}


def verify_runtime_installation(checkout: Path, commit: str, remote_url: str) -> dict:
    """Require both wheels to match one previously verified clean Git checkout.

    PEP 610 metadata authenticates the installer source. A complete Python-file
    comparison also rejects stale editable installs, import shadowing, changed
    installed code and missing modules in a wheel.
    """
    checkout = checkout.resolve()
    tracked = subprocess.check_output(
        ["git", "-C", str(checkout), "ls-files", "-z"], text=True,
    ).split("\0")
    report = {}
    for distribution_name, package_name in PACKAGES.items():
        distribution = importlib.metadata.distribution(distribution_name)
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        vcs = direct.get("vcs_info", {})
        subdirectory = f"src/{package_name}"
        if (direct.get("url", "").removesuffix(".git") != remote_url.removesuffix(".git")
                or vcs.get("vcs") != "git" or vcs.get("commit_id") != commit
                or direct.get("subdirectory") != subdirectory):
            raise ValueError(f"{distribution_name} must be pip-installed from {remote_url}@{commit}")
        package_root = _package_root(package_name)
        if package_root != Path(distribution.locate_file(package_name)).resolve():
            raise ValueError(f"{package_name} import is shadowed by another source directory")
        source_prefix = f"{subdirectory}/{package_name}/"
        expected = {
            name.removeprefix(source_prefix): hashlib.sha256((checkout / name).read_bytes()).hexdigest()
            for name in tracked if name.startswith(source_prefix) and name.endswith(".py")
        }
        actual = _source_files(package_root)
        if not expected or actual != expected:
            differing = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
            raise ValueError(f"{distribution_name} installed code differs from Git: {differing[:10]}")
        digest = hashlib.sha256(json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        report[distribution_name] = {"commit_sha": commit, "remote_url": remote_url,
                                     "subdirectory": subdirectory, "file_count": len(actual),
                                     "source_sha256": digest}
    return report
