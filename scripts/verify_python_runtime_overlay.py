#!/usr/bin/env python3
"""Fail-closed verification for a content-addressed Python runtime overlay.

The checksum manifest authenticates every payload file.  This verifier also
compares the complete on-disk file/directory set with the paths implied by the
manifest, so an unlisted module, native library, ``sitecustomize.py``, or
``__pycache__`` entry cannot affect imports.  Empty directories are forbidden
because a directory on ``sys.path`` can itself change namespace-package
resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat
import sys
import sysconfig

_SCHEMA_PATH = Path(__file__).absolute().with_name("python_runtime_schema.py")
if not _SCHEMA_PATH.is_file() or _SCHEMA_PATH.is_symlink():
    raise RuntimeError(f"Python runtime schema is missing or aliased: {_SCHEMA_PATH}")
_SCHEMA_NAMESPACE: dict[str, object] = {}
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

MANIFEST_NAME = ".holosoma-runtime-manifest.sha256"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANONICAL_DISTRIBUTION_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_IMPORT_ROOT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OverlayVerificationError(RuntimeError):
    """The runtime overlay is not the exact immutable tree in its manifest."""


def _relative_path(raw: bytes) -> str:
    if not raw or raw.startswith(b"\\"):
        raise OverlayVerificationError(
            "runtime manifest uses an empty or checksum-escaped path"
        )
    try:
        value = os.fsdecode(raw)
    except UnicodeError as exc:
        raise OverlayVerificationError("runtime manifest path is not decodable") from exc
    if not value.startswith("./") or "\n" in value or "\r" in value:
        raise OverlayVerificationError(
            f"runtime manifest path must be a canonical ./ relative path: {value!r}"
        )
    relative = PurePosixPath(value[2:])
    if (
        not relative.parts
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value[2:]
    ):
        raise OverlayVerificationError(
            f"runtime manifest path is unsafe or non-canonical: {value!r}"
        )
    if relative.as_posix() == MANIFEST_NAME:
        raise OverlayVerificationError("runtime manifest cannot checksum itself")
    return f"./{relative.as_posix()}"


def _read_manifest(manifest: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    raw = manifest.read_bytes()
    if not raw or not raw.endswith(b"\n"):
        raise OverlayVerificationError("runtime manifest must be non-empty and newline-terminated")
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if len(line) < 67:
            raise OverlayVerificationError(
                f"malformed runtime manifest line {line_number}"
            )
        try:
            digest = line[:64].decode("ascii")
        except UnicodeDecodeError as exc:
            raise OverlayVerificationError(
                f"non-ASCII digest on runtime manifest line {line_number}"
            ) from exc
        if not _SHA256_RE.fullmatch(digest) or line[64:66] not in {b"  ", b" *"}:
            raise OverlayVerificationError(
                f"malformed runtime manifest line {line_number}"
            )
        relative = _relative_path(line[66:])
        if relative in entries:
            raise OverlayVerificationError(
                f"duplicate runtime manifest path: {relative}"
            )
        entries[relative] = digest
    return entries


def _scan_tree(root: Path) -> tuple[set[str], set[str]]:
    try:
        root_mode = os.lstat(root).st_mode
    except FileNotFoundError as exc:
        raise OverlayVerificationError(f"runtime overlay is missing: {root}") from exc
    if not stat.S_ISDIR(root_mode) or stat.S_ISLNK(root_mode):
        raise OverlayVerificationError(f"runtime overlay root is not a real directory: {root}")
    if root_mode & 0o222:
        raise OverlayVerificationError(f"runtime overlay root is writable: {root}")

    files: set[str] = set()
    directories: set[str] = set()
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in sorted((*dirnames, *filenames)):
            path = current_path / name
            mode = os.lstat(path).st_mode
            relative = f"./{path.relative_to(root).as_posix()}"
            if mode & 0o222:
                raise OverlayVerificationError(f"runtime overlay path is writable: {relative}")
            if stat.S_ISDIR(mode):
                directories.add(relative)
            elif stat.S_ISREG(mode):
                files.add(relative)
            else:
                raise OverlayVerificationError(
                    f"runtime overlay contains a symlink or special file: {relative}"
                )
    return files, directories


def _implied_directories(files: set[str]) -> set[str]:
    directories: set[str] = set()
    for relative in files:
        parent = PurePosixPath(relative[2:]).parent
        while parent != PurePosixPath("."):
            directories.add(f"./{parent.as_posix()}")
            parent = parent.parent
    return directories


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _module_paths(module: object) -> list[Path]:
    paths: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if isinstance(module_file, str):
        paths.append(Path(module_file).resolve())
    module_path = getattr(module, "__path__", None)
    if module_path is not None:
        for entry in module_path:
            if isinstance(entry, str):
                paths.append(Path(entry).resolve())
    return paths


def _require_overlay_module(
    module_name: str,
    module: object,
    overlay_root: Path,
) -> None:
    paths = _module_paths(module)
    if not paths:
        raise OverlayVerificationError(
            f"contracted Python module has no auditable path: {module_name}"
        )
    for path in paths:
        if not _inside(path, overlay_root):
            raise OverlayVerificationError(
                f"contracted Python module escaped the overlay: {module_name} -> {path}"
            )


def _require_smoke_module_path(
    module_name: str,
    module: object,
    *,
    overlay_root: Path,
    stdlib_roots: set[Path],
    base_site_roots: set[Path],
) -> None:
    # Newly imported scientific code may come only from the authenticated
    # overlay or from this interpreter's standard library. Check site-package
    # roots first because a venv's purelib is normally nested below stdlib.
    for path in _module_paths(module):
        if _inside(path, overlay_root):
            continue
        if any(_inside(path, base_root) for base_root in base_site_roots):
            raise OverlayVerificationError(
                "scientific Python import smoke escaped the declared overlay closure: "
                f"module={module_name} path={path}"
            )
        if any(_inside(path, stdlib_root) for stdlib_root in stdlib_roots):
            continue
        raise OverlayVerificationError(
            "scientific Python import smoke loaded code outside the overlay/stdlib allowlist: "
            f"module={module_name} path={path}"
        )


def _distribution_contract(site_packages: Path) -> tuple[list[dict[str, object]], set[str]]:
    contract_path = site_packages / DISTRIBUTION_CONTRACT_NAME
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise OverlayVerificationError(
            f"Python runtime distribution contract is missing or malformed: {contract_path}"
        ) from exc
    expected_top_level = {
        "version",
        "runtime_profile",
        "python_cache_tag",
        "python_version",
        "python_soabi",
        "platform_machine",
        "root_distributions",
        "distributions",
        "omitted_console_scripts",
    }
    if not isinstance(contract, dict) or set(contract) != expected_top_level:
        raise OverlayVerificationError(
            "Python runtime distribution contract has an unexpected top-level schema"
        )
    if contract["version"] != DISTRIBUTION_CONTRACT_VERSION:
        raise OverlayVerificationError(
            f"unsupported Python runtime distribution contract version: {contract['version']!r}"
        )
    if contract["runtime_profile"] != RUNTIME_PROFILE:
        raise OverlayVerificationError(
            "Python runtime profile differs from the scientific schema: "
            f"actual={contract['runtime_profile']!r} expected={RUNTIME_PROFILE!r}"
        )
    if contract["python_cache_tag"] != sys.implementation.cache_tag:
        raise OverlayVerificationError(
            "Python runtime cache tag differs from the overlay contract: "
            f"actual={sys.implementation.cache_tag!r} "
            f"expected={contract['python_cache_tag']!r}"
        )
    if contract["python_version"] != platform.python_version():
        raise OverlayVerificationError(
            "Python runtime version differs from the overlay contract: "
            f"actual={platform.python_version()!r} "
            f"expected={contract['python_version']!r}"
        )
    actual_soabi = str(sysconfig.get_config_var("SOABI") or "")
    if not actual_soabi or contract["python_soabi"] != actual_soabi:
        raise OverlayVerificationError(
            "Python runtime SOABI differs from the overlay contract: "
            f"actual={actual_soabi!r} expected={contract['python_soabi']!r}"
        )
    if contract["platform_machine"] != platform.machine():
        raise OverlayVerificationError(
            "Python runtime machine architecture differs from the overlay contract: "
            f"actual={platform.machine()!r} expected={contract['platform_machine']!r}"
        )
    roots = contract["root_distributions"]
    if roots != list(ROOT_DISTRIBUTIONS):
        raise OverlayVerificationError(
            "Python runtime distribution roots must exactly match the scientific schema: "
            f"actual={roots!r} expected={list(ROOT_DISTRIBUTIONS)!r}"
        )
    distributions = contract["distributions"]
    if not isinstance(distributions, list) or not distributions:
        raise OverlayVerificationError(
            "Python runtime distribution contract must contain a non-empty closure"
        )

    names: set[str] = set()
    import_roots: set[str] = set()
    expected_record_keys = {
        "canonical_name",
        "version",
        "requirements",
        "import_roots",
        "payload_file_count",
    }
    for record in distributions:
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise OverlayVerificationError(
                "Python runtime distribution record has an unexpected schema"
            )
        name = record["canonical_name"]
        version = record["version"]
        roots_for_distribution = record["import_roots"]
        payload_file_count = record["payload_file_count"]
        requirements = record["requirements"]
        if not isinstance(name, str) or _CANONICAL_DISTRIBUTION_RE.fullmatch(name) is None:
            raise OverlayVerificationError(
                f"invalid canonical Python distribution name: {name!r}"
            )
        if name in names:
            raise OverlayVerificationError(f"duplicate Python distribution record: {name}")
        names.add(name)
        if (
            not isinstance(version, str)
            or not 1 <= len(version) <= 128
            or any(ord(character) < 0x21 or ord(character) > 0x7E for character in version)
        ):
            raise OverlayVerificationError(
                f"invalid Python distribution version for {name}: {version!r}"
            )
        if not isinstance(payload_file_count, int) or isinstance(payload_file_count, bool) or payload_file_count < 1:
            raise OverlayVerificationError(
                f"invalid payload file count for Python distribution {name}: {payload_file_count!r}"
            )
        if not isinstance(roots_for_distribution, list) or not roots_for_distribution:
            raise OverlayVerificationError(
                f"Python distribution {name} has no declared import roots"
            )
        for import_root in roots_for_distribution:
            if not isinstance(import_root, str) or _IMPORT_ROOT_RE.fullmatch(import_root) is None:
                raise OverlayVerificationError(
                    f"invalid import root for Python distribution {name}: {import_root!r}"
                )
            if import_root in import_roots:
                raise OverlayVerificationError(
                    f"multiple Python distributions claim import root {import_root!r}"
                )
            import_roots.add(import_root)
        if not isinstance(requirements, list):
            raise OverlayVerificationError(
                f"Python distribution {name} requirements must be a list"
            )
        for requirement in requirements:
            if (
                not isinstance(requirement, dict)
                or set(requirement) != {"name", "specifier"}
                or not isinstance(requirement["name"], str)
                or _CANONICAL_DISTRIBUTION_RE.fullmatch(requirement["name"]) is None
                or not isinstance(requirement["specifier"], str)
            ):
                raise OverlayVerificationError(
                    f"malformed active requirement for Python distribution {name}: {requirement!r}"
                )

    if not set(ROOT_DISTRIBUTIONS).issubset(names):
        raise OverlayVerificationError(
            "Python runtime distribution closure omits a required root distribution"
        )
    requirements_by_name: dict[str, set[str]] = {}
    for record in distributions:
        record_name = str(record["canonical_name"])
        requirements_by_name[record_name] = set()
        for requirement in record["requirements"]:  # type: ignore[index]
            if requirement["name"] not in names:  # type: ignore[index]
                raise OverlayVerificationError(
                    f"Python runtime dependency closure omits {requirement['name']!r}"
                )
            requirements_by_name[record_name].add(requirement["name"])  # type: ignore[index]

    reachable = set(ROOT_DISTRIBUTIONS)
    pending = list(ROOT_DISTRIBUTIONS)
    while pending:
        current = pending.pop()
        for dependency in requirements_by_name.get(current, set()):
            if dependency not in reachable:
                reachable.add(dependency)
                pending.append(dependency)
    if reachable != names:
        raise OverlayVerificationError(
            "Python runtime distribution closure contains packages unreachable "
            f"from the scientific roots: {sorted(names.difference(reachable))}"
        )

    omitted = contract["omitted_console_scripts"]
    if not isinstance(omitted, list):
        raise OverlayVerificationError(
            "Python runtime omitted console-script records must be a list"
        )
    for record in omitted:
        if (
            not isinstance(record, dict)
            or set(record) != {"distribution", "path"}
            or record["distribution"] not in names
            or not isinstance(record["path"], str)
            or "\n" in record["path"]
            or "\r" in record["path"]
        ):
            raise OverlayVerificationError(
                f"malformed omitted Python console-script record: {record!r}"
            )
    return distributions, import_roots


def verify_distribution_closure(
    site_packages: Path,
    *,
    current_runtime_binding: bool = False,
) -> int:
    root = site_packages.resolve()
    distributions, import_roots = _distribution_contract(root)

    # Model the launcher's actual import precedence without executing .pth
    # startup hooks: exact overlay first, then the interpreter's base
    # site-packages.  Every contracted module and metadata record must still
    # resolve from the overlay, never from a heterogeneous node environment.
    original_sys_path = list(sys.path)
    base_site_roots = {
        Path(value).resolve()
        for key in ("purelib", "platlib")
        if (value := sysconfig.get_path(key))
    }
    stdlib_roots = {
        Path(value).resolve()
        for key in ("stdlib", "platstdlib")
        if (value := sysconfig.get_path(key))
    }
    if not stdlib_roots:
        raise OverlayVerificationError("active interpreter exposes no stdlib path")
    if current_runtime_binding:
        resolved_search_entries = {
            Path(entry).resolve()
            for entry in original_sys_path
            if entry
        }
        if root not in resolved_search_entries:
            raise OverlayVerificationError(
                f"Python runtime overlay is absent from the live interpreter search path: {root}"
            )
    else:
        search_path = [str(root)]
        search_path.extend(
            entry for entry in original_sys_path if entry and entry not in search_path
        )
        for base_site_root in sorted(base_site_roots):
            if str(base_site_root) not in search_path:
                search_path.append(str(base_site_root))
        sys.path[:] = search_path
        sys.path_importer_cache.clear()
    importlib.invalidate_caches()

    expected_versions: dict[str, str] = {}
    for record in distributions:
        name = str(record["canonical_name"])
        expected_version = str(record["version"])
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise OverlayVerificationError(
                f"contracted Python distribution is not importable: {name}"
            ) from exc
        metadata_path = Path(distribution._path).resolve()  # type: ignore[attr-defined]
        if not _inside(metadata_path, root):
            raise OverlayVerificationError(
                f"Python distribution metadata escaped the overlay: {name} -> {metadata_path}"
            )
        if str(distribution.version) != expected_version:
            raise OverlayVerificationError(
                f"Python distribution version differs from its overlay contract: "
                f"{name} actual={distribution.version!r} expected={expected_version!r}"
            )
        declared_files = distribution.files
        if declared_files is None:
            raise OverlayVerificationError(
                f"contracted Python distribution exposes no installed-file manifest: {name}"
            )
        actual_payload_count = 0
        for declared_path in declared_files:
            payload = Path(distribution.locate_file(declared_path)).resolve()
            if payload.is_file() and _inside(payload, root):
                if "__pycache__" not in payload.parts and payload.suffix not in {".pyc", ".pyo"}:
                    actual_payload_count += 1
        if actual_payload_count != record["payload_file_count"]:
            raise OverlayVerificationError(
                f"Python distribution payload count differs from its overlay contract: "
                f"{name} actual={actual_payload_count} expected={record['payload_file_count']}"
            )
        expected_versions[name] = expected_version

    for import_root in sorted(import_roots):
        spec = importlib.util.find_spec(import_root)
        if spec is None:
            raise OverlayVerificationError(
                f"contracted Python import root is unavailable: {import_root}"
            )
        if spec.origin not in {None, "built-in", "frozen"}:
            origin = Path(spec.origin).resolve()
            if not _inside(origin, root):
                raise OverlayVerificationError(
                    f"Python import root escaped the overlay: {import_root} -> {origin}"
                )
        locations = list(spec.submodule_search_locations or ())
        if not locations and spec.origin is None:
            raise OverlayVerificationError(
                f"Python import root has neither an origin nor package locations: {import_root}"
            )
        for location in locations:
            resolved_location = Path(location).resolve()
            if not _inside(resolved_location, root):
                raise OverlayVerificationError(
                    f"Python namespace/package root is split outside the overlay: "
                    f"{import_root} -> {resolved_location}"
                )

    # A normal interpreter processes .pth files before this verifier starts.
    # Reject a preloaded base/PYTHONPATH copy of any contracted package instead
    # of letting the later smoke import reuse it from sys.modules.
    for module_name, module in sorted(sys.modules.items()):
        if (
            module is not None
            and module_name.split(".", 1)[0] in import_roots
        ):
            _require_overlay_module(module_name, module, root)

    modules_before_smoke = set(sys.modules)
    try:
        import attrs
        import numpy
        import onnx
        import onnxruntime
        from omegaconf import OmegaConf
        import omegaconf
    except Exception as exc:
        raise OverlayVerificationError(
            f"scientific Python runtime import smoke failed: {type(exc).__name__}: {exc}"
        ) from exc
    for name, module in (
        ("attrs", attrs),
        ("numpy", numpy),
        ("omegaconf", omegaconf),
        ("onnx", onnx),
        ("onnxruntime", onnxruntime),
    ):
        module_file = getattr(module, "__file__", None)
        if not isinstance(module_file, str) or not _inside(Path(module_file).resolve(), root):
            raise OverlayVerificationError(
                f"scientific Python module escaped the overlay: {name} -> {module_file!r}"
            )
        module_version = getattr(module, "__version__", None)
        if module_version is not None and str(module_version) != expected_versions[name]:
            raise OverlayVerificationError(
                f"scientific Python module version mismatch: {name} "
                f"actual={module_version!r} expected={expected_versions[name]!r}"
            )
    for module_name in sorted(set(sys.modules).difference(modules_before_smoke)):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        _require_smoke_module_path(
            module_name,
            module,
            overlay_root=root,
            stdlib_roots=stdlib_roots,
            base_site_roots=base_site_roots,
        )
    if float(numpy.dot(numpy.asarray([1.0, 2.0]), numpy.asarray([3.0, 4.0]))) != 11.0:
        raise OverlayVerificationError("NumPy scientific runtime smoke produced the wrong result")
    try:
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [onnx.helper.make_node("Identity", ["input"], ["output"])],
                "holosoma-runtime-smoke",
                [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
                [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2])],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 17)],
            ir_version=9,
        )
        onnx.checker.check_model(model)
        session = onnxruntime.InferenceSession(
            model.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
        expected = numpy.asarray([[1.25, -2.5]], dtype=numpy.float32)
        outputs = session.run(None, {"input": expected})
    except Exception as exc:
        raise OverlayVerificationError(
            f"ONNX checker/runtime smoke failed: {type(exc).__name__}: {exc}"
        ) from exc
    if len(outputs) != 1 or not numpy.array_equal(outputs[0], expected):
        raise OverlayVerificationError("ONNX Runtime scientific smoke produced the wrong result")
    config = OmegaConf.create({"base": 3, "resolved": "${base}"})
    resolved = OmegaConf.to_container(config, resolve=True)
    if not isinstance(resolved, dict) or resolved.get("resolved") != 3:
        raise OverlayVerificationError("OmegaConf scientific runtime smoke produced the wrong result")
    return len(distributions)


def verify_overlay(site_packages: Path, expected_manifest_sha256: str) -> int:
    if not _SHA256_RE.fullmatch(expected_manifest_sha256):
        raise OverlayVerificationError(
            "expected runtime manifest SHA256 must be 64 lowercase hex characters"
        )
    root = site_packages.absolute()
    manifest = root / MANIFEST_NAME
    files, directories = _scan_tree(root)
    if f"./{MANIFEST_NAME}" not in files:
        raise OverlayVerificationError(f"runtime manifest is missing: {manifest}")

    manifest_bytes = manifest.read_bytes()
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise OverlayVerificationError(
            "Python runtime manifest SHA256 mismatch: "
            f"actual={actual_manifest_sha256} expected={expected_manifest_sha256}"
        )

    entries = _read_manifest(manifest)
    expected_files = set(entries) | {f"./{MANIFEST_NAME}"}
    expected_directories = _implied_directories(expected_files)
    if files != expected_files or directories != expected_directories:
        extra = sorted((files - expected_files) | (directories - expected_directories))
        missing = sorted((expected_files - files) | (expected_directories - directories))
        raise OverlayVerificationError(
            "Python runtime overlay path closure mismatch: "
            f"extra={extra[:8]} missing={missing[:8]}"
        )

    for relative, expected_digest in entries.items():
        digest = hashlib.sha256((root / relative[2:]).read_bytes()).hexdigest()
        if digest != expected_digest:
            raise OverlayVerificationError(
                f"Python runtime payload SHA256 mismatch: path={relative} "
                f"actual={digest} expected={expected_digest}"
            )
    return len(entries)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-packages", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--require-distribution-closure", action="store_true")
    parser.add_argument("--require-current-runtime-binding", action="store_true")
    args = parser.parse_args()
    try:
        if args.require_current_runtime_binding and not args.require_distribution_closure:
            raise OverlayVerificationError(
                "--require-current-runtime-binding requires --require-distribution-closure"
            )
        file_count = verify_overlay(args.site_packages, args.manifest_sha256)
        distribution_count = (
            verify_distribution_closure(
                args.site_packages,
                current_runtime_binding=args.require_current_runtime_binding,
            )
            if args.require_distribution_closure
            else 0
        )
    except (OSError, OverlayVerificationError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc
    print(
        "[INFO] python_runtime_exact_closure_verified="
        f"{args.site_packages.absolute()} manifest_sha256={args.manifest_sha256} "
        f"payload_files={file_count} distribution_closure={distribution_count} "
        f"current_runtime_binding={int(args.require_current_runtime_binding)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
