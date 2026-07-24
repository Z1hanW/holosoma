#!/usr/bin/env python3
"""Install one authenticated Python runtime overlay without partial publication."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"^[0-9a-f]{64}$")
_RUNTIME_PREFIX = "python-runtime-v2-"
_MAX_ARCHIVE_BYTES = 4 * 1024 * 1024 * 1024
_MAX_MEMBER_COUNT = 250_000
_MAX_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TOTAL_BYTES = 8 * 1024 * 1024 * 1024


class InstallError(RuntimeError):
    """The archive or destination violates the immutable runtime contract."""


class MissingRuntime(InstallError):
    """The requested content-addressed runtime has not been published."""


def _canonical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _mode(stat_result: os.stat_result) -> int:
    return stat.S_IMODE(stat_result.st_mode)


def _require_real_directory(path: Path, *, mode: int, role: str) -> os.stat_result:
    path = _canonical_absolute(path)
    try:
        result = os.lstat(path)
    except FileNotFoundError as exc:
        raise InstallError(f"{role} is missing: {path}") from exc
    if (
        not stat.S_ISDIR(result.st_mode)
        or stat.S_ISLNK(result.st_mode)
        or result.st_uid != os.geteuid()
        or _mode(result) != mode
        or path.resolve() != path
    ):
        raise InstallError(
            f"{role} must be a real current-UID directory with mode {mode:04o}: {path}"
        )
    return result


def _require_verifier(path: Path) -> Path:
    path = _canonical_absolute(path)
    try:
        result = os.lstat(path)
    except FileNotFoundError as exc:
        raise InstallError(f"runtime verifier is missing: {path}") from exc
    if (
        not stat.S_ISREG(result.st_mode)
        or stat.S_ISLNK(result.st_mode)
        or result.st_nlink != 1
        or result.st_uid != os.geteuid()
        or _mode(result) != 0o444
        or path.resolve() != path
    ):
        raise InstallError(
            f"runtime verifier must be a sealed current-UID 0444 regular file: {path}"
        )
    schema = path.with_name("python_runtime_schema.py")
    try:
        schema_stat = os.lstat(schema)
    except FileNotFoundError as exc:
        raise InstallError(f"runtime verifier schema is missing: {schema}") from exc
    if (
        not stat.S_ISREG(schema_stat.st_mode)
        or stat.S_ISLNK(schema_stat.st_mode)
        or schema_stat.st_nlink != 1
        or schema_stat.st_uid != os.geteuid()
        or _mode(schema_stat) != 0o444
    ):
        raise InstallError(
            f"runtime verifier schema must be a sealed current-UID 0444 regular file: {schema}"
        )
    return path


def _stable_fields(result: os.stat_result) -> tuple[int, ...]:
    return (
        result.st_dev,
        result.st_ino,
        result.st_mode,
        result.st_nlink,
        result.st_uid,
        result.st_size,
        result.st_mtime_ns,
        result.st_ctime_ns,
    )


def _bind_archive(
    archive: Path,
    *,
    runtime_root: Path,
    runtime_id: str,
    archive_sha256: str,
) -> tuple[int, os.stat_result]:
    archive = _canonical_absolute(archive)
    incoming_root = runtime_root / ".incoming"
    _require_real_directory(incoming_root, mode=0o700, role="runtime incoming root")
    transfer_root = archive.parent
    _require_real_directory(transfer_root, mode=0o700, role="runtime transfer root")
    if transfer_root.parent != incoming_root or _TOKEN_RE.fullmatch(transfer_root.name) is None:
        raise InstallError(
            f"runtime archive is not inside one token-bound incoming directory: {archive}"
        )
    expected_name = f"{runtime_id}.{archive_sha256}.tar.gz"
    if archive.name != expected_name:
        raise InstallError(
            f"runtime archive basename does not bind runtime/archive identity: {archive.name!r}"
        )

    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(archive, flags)
    except OSError as exc:
        raise InstallError(f"cannot open runtime archive without following aliases: {archive}") from exc
    try:
        before = os.fstat(descriptor)
        path_before = os.lstat(archive)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.geteuid()
            or _mode(before) != 0o400
            or not 0 < before.st_size <= _MAX_ARCHIVE_BYTES
            or _stable_fields(before) != _stable_fields(path_before)
        ):
            raise InstallError(
                f"runtime archive must be a bound current-UID single-link 0400 file: {archive}"
            )
        digest = hashlib.sha256()
        os.lseek(descriptor, 0, os.SEEK_SET)
        while chunk := os.read(descriptor, 4 * 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(descriptor)
        path_after = os.lstat(archive)
        if (
            _stable_fields(before) != _stable_fields(after)
            or _stable_fields(before) != _stable_fields(path_after)
        ):
            raise InstallError(f"runtime archive changed while being authenticated: {archive}")
        if digest.hexdigest() != archive_sha256:
            raise InstallError(f"runtime archive SHA256 mismatch: {archive}")
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, before
    except BaseException:
        os.close(descriptor)
        raise


def _require_bound_archive_path(
    archive: Path, *, bound_fields: tuple[int, ...]
) -> None:
    """Cancel a pre-lock archive FD if its authenticated pathname was revoked."""
    archive = _canonical_absolute(archive)
    try:
        current = os.lstat(archive)
    except FileNotFoundError as exc:
        raise InstallError(
            "runtime archive transfer was revoked before the install transaction"
        ) from exc
    if _stable_fields(current) != bound_fields:
        raise InstallError(
            "runtime archive identity changed before the install transaction"
        )


def _canonical_member_name(member: tarfile.TarInfo) -> str:
    name = member.name
    if (
        not isinstance(name, str)
        or not name
        or name.startswith("/")
        or "\\" in name
        or any(ord(character) < 0x20 or ord(character) == 0x7F for character in name)
    ):
        raise InstallError(f"runtime archive contains an unsafe member name: {name!r}")
    value = name[:-1] if member.isdir() and name.endswith("/") else name
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or relative.as_posix() != value
        or relative.parts[0] != "site-packages"
    ):
        raise InstallError(f"runtime archive member is non-canonical: {name!r}")
    return relative.as_posix()


def _validated_members(
    archive: tarfile.TarFile,
) -> tuple[list[tuple[str, tarfile.TarInfo]], set[str]]:
    members: list[tuple[str, tarfile.TarInfo]] = []
    directories: set[str] = set()
    names: set[str] = set()
    total_bytes = 0
    for index, member in enumerate(archive, start=1):
        if index > _MAX_MEMBER_COUNT:
            raise InstallError("runtime archive exceeds the member-count limit")
        name = _canonical_member_name(member)
        if name in names:
            raise InstallError(f"runtime archive contains duplicate member: {name}")
        names.add(name)
        if getattr(member, "sparse", None):
            raise InstallError(f"runtime archive contains a sparse member: {name}")
        if member.isdir():
            if member.size != 0 or stat.S_IMODE(member.mode) != 0o555:
                raise InstallError(f"runtime archive directory has non-canonical metadata: {name}")
            directories.add(name)
        elif member.isreg():
            if (
                member.size < 0
                or member.size > _MAX_MEMBER_BYTES
                or stat.S_IMODE(member.mode) != 0o444
            ):
                raise InstallError(f"runtime archive file has non-canonical metadata: {name}")
            total_bytes += member.size
            if total_bytes > _MAX_TOTAL_BYTES:
                raise InstallError("runtime archive exceeds the extracted-size limit")
        else:
            raise InstallError(f"runtime archive contains a link or special member: {name}")
        if member.uid != 0 or member.gid != 0 or int(member.mtime) != 0:
            raise InstallError(f"runtime archive member ownership/time is non-canonical: {name}")
        members.append((name, member))

    if "site-packages" not in directories:
        raise InstallError("runtime archive omits its site-packages root directory")
    if not members:
        raise InstallError("runtime archive is empty")
    for name, _member in members:
        parent = PurePosixPath(name).parent
        while parent != PurePosixPath("."):
            if parent.as_posix() not in directories:
                raise InstallError(
                    f"runtime archive omits declared parent directory: {parent.as_posix()}"
                )
            parent = parent.parent
    return members, directories


def _extract_candidate(
    descriptor: int,
    *,
    runtime_root: Path,
    runtime_id: str,
) -> Path:
    candidate = Path(
        tempfile.mkdtemp(prefix=f".{runtime_id}.candidate.", dir=runtime_root)
    )
    os.chmod(candidate, 0o700)
    try:
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as stream:
            stream.seek(0)
            with tarfile.open(fileobj=stream, mode="r:gz") as archive:
                members, directories = _validated_members(archive)
                for name in sorted(directories, key=lambda value: (value.count("/"), value)):
                    destination = candidate.joinpath(*PurePosixPath(name).parts)
                    destination.mkdir(mode=0o700)
                for name, member in members:
                    if member.isdir():
                        continue
                    destination = candidate.joinpath(*PurePosixPath(name).parts)
                    source = archive.extractfile(member)
                    if source is None:
                        raise InstallError(f"runtime archive file cannot be read: {name}")
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                    output_fd = os.open(destination, flags, 0o600)
                    written = 0
                    try:
                        with source:
                            while chunk := source.read(1024 * 1024):
                                written += len(chunk)
                                if written > member.size:
                                    raise InstallError(
                                        f"runtime archive expanded beyond declared size: {name}"
                                    )
                                view = memoryview(chunk)
                                while view:
                                    consumed = os.write(output_fd, view)
                                    view = view[consumed:]
                        if written != member.size:
                            raise InstallError(
                                f"runtime archive file is truncated: {name}"
                            )
                        os.fsync(output_fd)
                        os.fchmod(output_fd, 0o444)
                    finally:
                        os.close(output_fd)
                for name in sorted(
                    directories,
                    key=lambda value: (value.count("/"), value),
                    reverse=True,
                ):
                    os.chmod(candidate.joinpath(*PurePosixPath(name).parts), 0o555)
        if {path.name for path in candidate.iterdir()} != {"site-packages"}:
            raise InstallError("runtime candidate has an unexpected top-level entry")
        os.chmod(candidate, 0o555)
        return candidate
    except BaseException:
        _remove_candidate(candidate)
        raise


def _remove_candidate(candidate: Path) -> None:
    if not os.path.lexists(candidate):
        return
    for current, dirnames, filenames in os.walk(candidate, topdown=False, followlinks=False):
        for filename in filenames:
            path = Path(current) / filename
            try:
                os.chmod(path, 0o600, follow_symlinks=False)
            except OSError:
                pass
        for dirname in dirnames:
            path = Path(current) / dirname
            try:
                os.chmod(path, 0o700, follow_symlinks=False)
            except OSError:
                pass
    try:
        os.chmod(candidate, 0o700, follow_symlinks=False)
    except OSError:
        pass
    shutil.rmtree(candidate)


def _open_lock(path: Path, timeout_seconds: int) -> int:
    flags = os.O_RDWR | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise InstallError(f"runtime install lock cannot be opened safely: {path}") from exc
    try:
        result = os.fstat(descriptor)
        path_result = os.lstat(path)
        if (
            not stat.S_ISREG(result.st_mode)
            or result.st_nlink != 1
            or result.st_uid != os.geteuid()
            or _mode(result) != 0o600
            or result.st_size != 0
            or (result.st_dev, result.st_ino) != (path_result.st_dev, path_result.st_ino)
        ):
            raise InstallError(f"runtime install lock is aliased or malformed: {path}")
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return descriptor
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise InstallError(f"timed out acquiring runtime install lock: {path}")
                time.sleep(0.05)
    except BaseException:
        os.close(descriptor)
        raise


def _run_verifier(
    final_root: Path,
    *,
    manifest_sha256: str,
    verifier: Path,
) -> None:
    try:
        outer = os.lstat(final_root)
    except FileNotFoundError as exc:
        raise MissingRuntime(f"Python runtime is not installed: {final_root}") from exc
    if (
        not stat.S_ISDIR(outer.st_mode)
        or stat.S_ISLNK(outer.st_mode)
        or outer.st_uid != os.geteuid()
        or _mode(outer) != 0o555
        or final_root.resolve() != final_root
        or {path.name for path in final_root.iterdir()} != {"site-packages"}
    ):
        raise InstallError(f"published Python runtime root is malformed: {final_root}")
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["LC_ALL"] = "C"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(verifier),
            "--site-packages",
            str(final_root / "site-packages"),
            "--manifest-sha256",
            manifest_sha256,
            "--require-distribution-closure",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        diagnostic = (result.stderr or result.stdout).strip()
        raise InstallError(
            f"published Python runtime failed strict verification: {diagnostic}"
        )


def _validate_runtime_namespace(runtime_root: Path) -> tuple[Path, Path]:
    runtime_root = _canonical_absolute(runtime_root)
    _require_real_directory(runtime_root, mode=0o700, role="Python runtime root")
    lock_root = runtime_root / ".locks"
    _require_real_directory(lock_root, mode=0o700, role="Python runtime lock root")
    return runtime_root, lock_root


def _remove_stale_candidates(runtime_root: Path, *, runtime_id: str) -> None:
    """Reap interrupted extractions while holding this runtime's lock."""
    prefix = f".{runtime_id}.candidate."
    candidates = sorted(
        path for path in runtime_root.iterdir() if path.name.startswith(prefix)
    )
    if len(candidates) > 1024:
        raise InstallError("runtime namespace contains too many stale candidates")
    for candidate in candidates:
        suffix = candidate.name[len(prefix) :]
        try:
            result = os.lstat(candidate)
        except FileNotFoundError:
            continue
        if (
            not suffix
            or len(suffix) > 128
            or re.fullmatch(r"[A-Za-z0-9_.-]+", suffix) is None
            or not stat.S_ISDIR(result.st_mode)
            or stat.S_ISLNK(result.st_mode)
            or result.st_uid != os.geteuid()
            or candidate.resolve() != candidate
        ):
            raise InstallError(
                f"stale Python runtime candidate is aliased or malformed: {candidate}"
            )
    for candidate in candidates:
        if os.path.lexists(candidate):
            _remove_candidate(candidate)


def _safe_remove_transfer(
    archive: Path,
    *,
    runtime_root: Path,
    runtime_id: str,
    archive_sha256: str,
    bound_fields: tuple[int, ...],
) -> None:
    """Remove only the exact incoming file authenticated by ``_bind_archive``.

    In particular, never treat an arbitrary user-supplied ``--archive`` path as
    cleanup authority.  Every namespace and inode property is rechecked so an
    early validation failure or a later path replacement cannot delete an
    unrelated current-UID file.
    """
    archive = _canonical_absolute(archive)
    runtime_root = _canonical_absolute(runtime_root)
    incoming_root = runtime_root / ".incoming"
    transfer_root = archive.parent
    try:
        _require_real_directory(
            runtime_root, mode=0o700, role="Python runtime root during cleanup"
        )
        _require_real_directory(
            incoming_root, mode=0o700, role="runtime incoming root during cleanup"
        )
        _require_real_directory(
            transfer_root, mode=0o700, role="runtime transfer root during cleanup"
        )
        if (
            transfer_root.parent != incoming_root
            or _TOKEN_RE.fullmatch(transfer_root.name) is None
            or archive.name != f"{runtime_id}.{archive_sha256}.tar.gz"
        ):
            return
        result = os.lstat(archive)
        if (
            stat.S_ISREG(result.st_mode)
            and not stat.S_ISLNK(result.st_mode)
            and result.st_nlink == 1
            and result.st_uid == os.geteuid()
            and _mode(result) == 0o400
            and _stable_fields(result) == bound_fields
        ):
            os.unlink(archive)
        else:
            return
    except (FileNotFoundError, InstallError, OSError):
        return
    try:
        transfer_stat = os.lstat(transfer_root)
        if (
            stat.S_ISDIR(transfer_stat.st_mode)
            and not stat.S_ISLNK(transfer_stat.st_mode)
            and transfer_stat.st_uid == os.geteuid()
            and _mode(transfer_stat) == 0o700
            and not any(transfer_root.iterdir())
        ):
            transfer_root.rmdir()
    except (FileNotFoundError, OSError):
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--archive-sha256")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--lock-timeout-seconds", type=int, default=60)
    args = parser.parse_args()

    archive_descriptor: int | None = None
    bound_archive_fields: tuple[int, ...] | None = None
    runtime_root: Path | None = None
    runtime_id: str | None = None
    candidate: Path | None = None
    try:
        if _SHA256_RE.fullmatch(args.manifest_sha256) is None:
            raise InstallError("runtime manifest SHA256 must be 64 lowercase hex characters")
        if not 1 <= args.lock_timeout_seconds <= 3600:
            raise InstallError("runtime install lock timeout must be in [1, 3600]")
        runtime_root, lock_root = _validate_runtime_namespace(args.runtime_root)
        verifier = _require_verifier(args.verifier)
        runtime_id = f"{_RUNTIME_PREFIX}{args.manifest_sha256}"
        final_root = runtime_root / runtime_id
        lock_path = lock_root / f"{runtime_id}.lock"

        if args.probe_only:
            if args.archive is not None or args.archive_sha256 is not None:
                raise InstallError("probe-only mode does not accept an archive")
            lock_descriptor = _open_lock(lock_path, args.lock_timeout_seconds)
            try:
                _remove_stale_candidates(runtime_root, runtime_id=runtime_id)
                _run_verifier(
                    final_root,
                    manifest_sha256=args.manifest_sha256,
                    verifier=verifier,
                )
            finally:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            print(f"[INFO] reused_verified_python_runtime={final_root}")
            return 0

        if args.archive is None or args.archive_sha256 is None:
            raise InstallError("publish mode requires archive and archive SHA256")
        if _SHA256_RE.fullmatch(args.archive_sha256) is None:
            raise InstallError("runtime archive SHA256 must be 64 lowercase hex characters")
        archive_descriptor, archive_stat = _bind_archive(
            args.archive,
            runtime_root=runtime_root,
            runtime_id=runtime_id,
            archive_sha256=args.archive_sha256,
        )
        bound_archive_fields = _stable_fields(archive_stat)
        lock_descriptor = _open_lock(lock_path, args.lock_timeout_seconds)
        try:
            _remove_stale_candidates(runtime_root, runtime_id=runtime_id)
            _require_bound_archive_path(
                args.archive,
                bound_fields=bound_archive_fields,
            )
            if os.path.lexists(final_root):
                _run_verifier(
                    final_root,
                    manifest_sha256=args.manifest_sha256,
                    verifier=verifier,
                )
                outcome = "reused_verified_python_runtime"
            else:
                candidate = _extract_candidate(
                    archive_descriptor,
                    runtime_root=runtime_root,
                    runtime_id=runtime_id,
                )
                _run_verifier(
                    candidate,
                    manifest_sha256=args.manifest_sha256,
                    verifier=verifier,
                )
                try:
                    os.rename(candidate, final_root)
                    candidate = None
                except FileExistsError:
                    _run_verifier(
                        final_root,
                        manifest_sha256=args.manifest_sha256,
                        verifier=verifier,
                    )
                    _remove_candidate(candidate)
                    candidate = None
                _run_verifier(
                    final_root,
                    manifest_sha256=args.manifest_sha256,
                    verifier=verifier,
                )
                outcome = "installed_verified_python_runtime"
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        print(f"[INFO] {outcome}={final_root}")
        return 0
    except MissingRuntime as exc:
        print(f"[MISSING] {exc}", file=sys.stderr)
        return 3
    except (OSError, tarfile.TarError, InstallError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    finally:
        if archive_descriptor is not None:
            os.close(archive_descriptor)
        if candidate is not None:
            _remove_candidate(candidate)
        if (
            args.archive is not None
            and args.archive_sha256 is not None
            and runtime_root is not None
            and runtime_id is not None
            and bound_archive_fields is not None
        ):
            _safe_remove_transfer(
                args.archive,
                runtime_root=runtime_root,
                runtime_id=runtime_id,
                archive_sha256=args.archive_sha256,
                bound_fields=bound_archive_fields,
            )


if __name__ == "__main__":
    raise SystemExit(main())
