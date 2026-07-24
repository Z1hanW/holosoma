from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import sysconfig

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VERIFIER = REPO_ROOT / "scripts" / "verify_python_runtime_overlay.py"
MANIFEST = ".holosoma-runtime-manifest.sha256"


def _seal_overlay(root: Path) -> str:
    payload = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name != MANIFEST
    )
    rows = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.relative_to(root).as_posix()}"
        for path in payload
    ]
    manifest = root / MANIFEST
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)
    return hashlib.sha256(manifest.read_bytes()).hexdigest()


def _verify(
    root: Path,
    digest: str,
    *,
    strict: bool = False,
    current_runtime: bool = False,
    pythonpath: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable]
    if not current_runtime:
        command.extend(["-I", "-S"])
    command.extend([
        str(VERIFIER),
        "--site-packages",
        str(root),
        "--manifest-sha256",
        digest,
    ])
    if strict:
        command.append("--require-distribution-closure")
    if current_runtime:
        command.append("--require-current-runtime-binding")
    env = os.environ.copy()
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _strict_overlay(
    tmp_path: Path,
    *,
    missing_metadata: str | None = None,
    missing_package: str | None = None,
    cache_tag: str | None = None,
    soabi: str | None = None,
    omit_contract_distribution: str | None = None,
    import_rogue_optional: bool = False,
    include_orphan_distribution: bool = False,
) -> tuple[Path, str]:
    root = tmp_path / "site-packages"
    packages = {
        "attrs": "__version__ = '1.0'\n",
        "numpy": (
            "__version__ = '1.0'\n"
            "def asarray(value):\n"
            "    return value\n"
            "def dot(left, right):\n"
            "    return sum(a * b for a, b in zip(left, right))\n"
        ),
        "omegaconf": (
            "__version__ = '1.0'\n"
            "class OmegaConf:\n"
            "    @staticmethod\n"
            "    def create(value):\n"
            "        return dict(value)\n"
            "    @staticmethod\n"
            "    def to_container(value, resolve=False):\n"
            "        result = dict(value)\n"
            "        if resolve and result.get('resolved') == '${base}':\n"
            "            result['resolved'] = result['base']\n"
            "        return result\n"
        ),
        "pyyaml": "__version__ = '1.0'\n",
    }
    module_names = {
        "attrs": "attrs",
        "numpy": "numpy",
        "omegaconf": "omegaconf",
        "pyyaml": "yaml",
    }
    if include_orphan_distribution:
        packages["orphan"] = "__version__ = '1.0'\n"
        module_names["orphan"] = "orphan"
    if import_rogue_optional:
        packages["omegaconf"] = "import rogue_optional\n" + packages["omegaconf"]
    records = []
    for distribution_name in sorted(packages):
        module_name = module_names[distribution_name]
        owned: list[str] = []
        if missing_package != distribution_name:
            package = root / module_name
            package.mkdir(parents=True)
            (package / "__init__.py").write_text(packages[distribution_name])
            owned.append(f"{module_name}/__init__.py")
        if missing_metadata != distribution_name:
            dist_info = root / f"{distribution_name}-1.0.dist-info"
            dist_info.mkdir(parents=True)
            metadata = dist_info / "METADATA"
            metadata.write_text(
                "Metadata-Version: 2.1\n"
                f"Name: {distribution_name}\n"
                "Version: 1.0\n"
            )
            owned.append(f"{dist_info.name}/METADATA")
            record_path = dist_info / "RECORD"
            owned.append(f"{dist_info.name}/RECORD")
            record_path.write_text("".join(f"{path},,\n" for path in owned))
        if distribution_name != omit_contract_distribution:
            records.append(
                {
                    "canonical_name": distribution_name,
                    "version": "1.0",
                    "requirements": (
                        [{"name": "pyyaml", "specifier": ">=1"}]
                        if distribution_name == "omegaconf"
                        else []
                    ),
                    "import_roots": [module_name],
                    "payload_file_count": len(owned),
                }
            )
    contract = {
        "version": 2,
        "runtime_profile": "as-core-v1",
        "python_cache_tag": cache_tag or sys.implementation.cache_tag,
        "python_version": platform.python_version(),
        "python_soabi": soabi or str(sysconfig.get_config_var("SOABI") or ""),
        "platform_machine": platform.machine(),
        "root_distributions": [
            "attrs",
            "numpy",
            "omegaconf",
        ],
        "distributions": records,
        "omitted_console_scripts": [],
    }
    (root / ".holosoma-runtime-distributions.json").write_text(
        json.dumps(contract, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return root, _seal_overlay(root)


def test_exact_read_only_overlay_is_accepted(tmp_path: Path) -> None:
    root = tmp_path / "site-packages"
    (root / "numpy").mkdir(parents=True)
    (root / "numpy" / "__init__.py").write_text("__version__ = 'test'\n")
    digest = _seal_overlay(root)

    result = _verify(root, digest)

    assert result.returncode == 0, result.stderr
    assert "python_runtime_exact_closure_verified=" in result.stdout
    assert "payload_files=1" in result.stdout


@pytest.mark.parametrize(
    ("relative_path", "contents"),
    [
        ("sitecustomize.py", b"raise RuntimeError('poisoned')\n"),
        ("numpy/__pycache__/__init__.cpython-311.pyc", b"unlisted bytecode"),
    ],
)
def test_unlisted_importable_file_is_rejected(
    tmp_path: Path, relative_path: str, contents: bytes
) -> None:
    root = tmp_path / "site-packages"
    (root / "numpy").mkdir(parents=True)
    (root / "numpy" / "__init__.py").write_text("pass\n")
    digest = _seal_overlay(root)
    root.chmod(0o755)
    extra = root / relative_path
    for parent in [root, *(root / relative_path).parents]:
        if parent == tmp_path:
            break
        if parent.exists() and parent.is_dir():
            parent.chmod(0o755)
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(contents)
    for path in [extra, *extra.parents]:
        if path == tmp_path:
            break
        path.chmod(0o444 if path.is_file() else 0o555)

    result = _verify(root, digest)

    assert result.returncode != 0
    assert "Python runtime overlay path closure mismatch" in result.stderr


def test_unlisted_empty_namespace_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "site-packages"
    (root / "numpy").mkdir(parents=True)
    (root / "numpy" / "__init__.py").write_text("pass\n")
    digest = _seal_overlay(root)
    root.chmod(0o755)
    (root / "unlisted_namespace").mkdir()
    (root / "unlisted_namespace").chmod(0o555)
    root.chmod(0o555)

    result = _verify(root, digest)

    assert result.returncode != 0
    assert "Python runtime overlay path closure mismatch" in result.stderr


def test_writable_overlay_directory_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "site-packages"
    (root / "numpy").mkdir(parents=True)
    (root / "numpy" / "__init__.py").write_text("pass\n")
    digest = _seal_overlay(root)
    (root / "numpy").chmod(0o755)

    result = _verify(root, digest)

    assert result.returncode != 0
    assert "runtime overlay path is writable: ./numpy" in result.stderr


def test_strict_distribution_closure_and_import_smokes_are_accepted(
    tmp_path: Path,
) -> None:
    root, digest = _strict_overlay(tmp_path)

    result = _verify(root, digest, strict=True)

    assert result.returncode == 0, result.stderr
    assert "distribution_closure=4" in result.stdout


def test_strict_closure_rejects_metadata_fallback_to_base_environment(
    tmp_path: Path,
) -> None:
    root, digest = _strict_overlay(tmp_path, missing_metadata="attrs")

    result = _verify(root, digest, strict=True)

    assert result.returncode != 0
    assert "distribution metadata escaped the overlay" in result.stderr


def test_strict_closure_rejects_import_fallback_to_base_environment(
    tmp_path: Path,
) -> None:
    root, digest = _strict_overlay(tmp_path, missing_package="attrs")

    result = _verify(root, digest, strict=True)

    assert result.returncode != 0
    assert "Python import root escaped the overlay" in result.stderr


def test_strict_closure_rejects_omitted_transitive_distribution(
    tmp_path: Path,
) -> None:
    root, digest = _strict_overlay(
        tmp_path,
        omit_contract_distribution="pyyaml",
    )

    result = _verify(root, digest, strict=True)

    assert result.returncode != 0
    assert "dependency closure omits 'pyyaml'" in result.stderr


def test_strict_closure_rejects_unreachable_extra_distribution(
    tmp_path: Path,
) -> None:
    root, digest = _strict_overlay(
        tmp_path,
        include_orphan_distribution=True,
    )

    result = _verify(root, digest, strict=True)

    assert result.returncode != 0
    assert "packages unreachable from the scientific roots: ['orphan']" in result.stderr


def test_strict_closure_rejects_python_abi_drift(tmp_path: Path) -> None:
    root, digest = _strict_overlay(tmp_path, cache_tag="cpython-incompatible")

    result = _verify(root, digest, strict=True)

    assert result.returncode != 0
    assert "cache tag differs from the overlay contract" in result.stderr


def test_strict_closure_rejects_python_soabi_drift(tmp_path: Path) -> None:
    root, digest = _strict_overlay(tmp_path, soabi="cpython-incompatible-soabi")

    result = _verify(root, digest, strict=True)

    assert result.returncode != 0
    assert "SOABI differs from the overlay contract" in result.stderr


def test_strict_current_runtime_binding_is_accepted(tmp_path: Path) -> None:
    root, digest = _strict_overlay(tmp_path)

    result = _verify(
        root,
        digest,
        strict=True,
        current_runtime=True,
        pythonpath=str(root),
    )

    assert result.returncode == 0, result.stderr
    assert "current_runtime_binding=1" in result.stdout


def test_strict_current_runtime_rejects_preloaded_contracted_module(
    tmp_path: Path,
) -> None:
    root, digest = _strict_overlay(tmp_path)
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "attrs.py").write_text("__version__ = '1.0'\n", encoding="utf-8")
    (poison / "sitecustomize.py").write_text("import attrs\n", encoding="utf-8")

    result = _verify(
        root,
        digest,
        strict=True,
        current_runtime=True,
        pythonpath=os.pathsep.join((str(poison), str(root))),
    )

    assert result.returncode != 0
    assert (
        "Python import root escaped the overlay" in result.stderr
        or "contracted Python module escaped the overlay" in result.stderr
    )


def test_strict_current_runtime_rejects_new_pythonpath_dependency(
    tmp_path: Path,
) -> None:
    root, digest = _strict_overlay(tmp_path, import_rogue_optional=True)
    poison = tmp_path / "poison"
    poison.mkdir()
    (poison / "rogue_optional.py").write_text("VALUE = 1\n", encoding="utf-8")

    result = _verify(
        root,
        digest,
        strict=True,
        current_runtime=True,
        pythonpath=os.pathsep.join((str(root), str(poison))),
    )

    assert result.returncode != 0
    assert "outside the overlay/stdlib allowlist" in result.stderr
