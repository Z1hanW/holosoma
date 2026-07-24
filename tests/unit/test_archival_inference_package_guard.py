from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_ARCHIVES = (
    _REPO_ROOT / "src" / "holosoma_inference_wrong",
    _REPO_ROOT / "src" / "holosoma_inference_track_T",
)
_CANONICAL = _REPO_ROOT / "src" / "holosoma_inference"
_OVERRIDE = "HOLOSOMA_ALLOW_UNSAFE_ARCHIVAL_INFERENCE"


def _python_env(package_root: Path, *, allow_archive: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(package_root)
    if allow_archive:
        env[_OVERRIDE] = "1"
    else:
        env.pop(_OVERRIDE, None)
    return env


@pytest.mark.parametrize("archive_root", _ARCHIVES, ids=lambda path: path.name)
def test_archival_package_import_fails_closed_by_default(
    archive_root: Path,
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-S", "-c", "import holosoma_inference"],
        cwd=tmp_path,
        env=_python_env(archive_root),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported archival snapshot" in result.stderr.lower()
    assert "src/holosoma_inference" in result.stderr


@pytest.mark.parametrize("archive_root", _ARCHIVES, ids=lambda path: path.name)
def test_archival_setup_metadata_path_refuses_same_name_install_by_default(
    archive_root: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "setup.py", "--name"],
        cwd=archive_root,
        env=_python_env(archive_root),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "refusing to install unsupported archival package" in result.stderr.lower()
    assert "canonical holosoma-inference distribution" in result.stderr


@pytest.mark.parametrize("archive_root", _ARCHIVES, ids=lambda path: path.name)
def test_exact_forensics_override_allows_archival_package_import(
    archive_root: Path,
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import holosoma_inference; print(holosoma_inference.__file__)",
        ],
        cwd=tmp_path,
        env=_python_env(archive_root, allow_archive=True),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(archive_root.resolve()) in result.stdout


def test_canonical_inference_package_is_not_guarded(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import holosoma_inference; print(holosoma_inference.__file__)",
        ],
        cwd=tmp_path,
        env=_python_env(_CANONICAL),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert str(_CANONICAL.resolve()) in result.stdout

