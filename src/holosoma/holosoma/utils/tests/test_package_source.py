import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from holosoma.utils import package_source


@pytest.fixture
def installation(tmp_path, monkeypatch):
    checkout = tmp_path / "checkout"
    installed = tmp_path / "site-packages"
    commit = "a" * 40
    remote = "https://example.org/holosoma.git"
    metadata = {}
    tracked = []
    for distribution, package in package_source.PACKAGES.items():
        relative = f"src/{package}/{package}/__init__.py"
        tracked.append(relative)
        for path in (checkout / relative, installed / package / "__init__.py"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('VALUE = "released"\n')
        metadata[distribution] = {"url": remote, "subdirectory": f"src/{package}",
                                  "vcs_info": {"vcs": "git", "commit_id": commit}}
    monkeypatch.setattr(package_source.subprocess, "check_output", lambda *a, **kw: "\0".join(tracked))
    monkeypatch.setattr(package_source, "_package_root", lambda name: installed / name)
    monkeypatch.setattr(package_source.importlib.metadata, "distribution", lambda name: SimpleNamespace(
        read_text=lambda _: json.dumps(metadata[name]), locate_file=lambda path: installed / path,
    ))
    return checkout, installed, commit, remote, metadata


def test_matching_git_installs_are_verified(installation):
    checkout, _, commit, remote, _ = installation
    result = package_source.verify_runtime_installation(checkout, commit, remote)
    assert set(result) == {"holosoma", "holosoma-inference"}
    assert all(r["file_count"] == 1 and r["commit_sha"] == commit for r in result.values())


@pytest.mark.parametrize("mode", ["modified", "missing", "extra"])
def test_installed_source_drift_is_rejected(installation, mode):
    checkout, installed, commit, remote, _ = installation
    path = installed / "holosoma_inference" / "__init__.py"
    if mode == "modified":
        path.write_text("VALUE = 'stale'\n")
    elif mode == "missing":
        path.unlink()
    else:
        path.with_name("unexpected.py").write_text("VALUE = 1\n")
    with pytest.raises(ValueError):
        package_source.verify_runtime_installation(checkout, commit, remote)


@pytest.mark.parametrize("field", ["commit", "url", "subdirectory", "editable"])
def test_wrong_install_origin_is_rejected(installation, field):
    checkout, _, commit, remote, metadata = installation
    direct = metadata["holosoma-inference"]
    if field == "commit":
        direct["vcs_info"]["commit_id"] = "b" * 40
    elif field == "editable":
        direct.clear()
        direct["dir_info"] = {"editable": True}
    else:
        direct[field] = "wrong"
    with pytest.raises(ValueError, match="must be pip-installed"):
        package_source.verify_runtime_installation(checkout, commit, remote)


def test_import_shadowing_is_rejected(installation, monkeypatch):
    checkout, installed, commit, remote, _ = installation
    monkeypatch.setattr(package_source, "_package_root", lambda name: installed / "shadow" / name)
    with pytest.raises(ValueError, match="shadowed"):
        package_source.verify_runtime_installation(checkout, commit, remote)
