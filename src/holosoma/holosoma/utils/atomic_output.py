"""Small process-shared stdout records used by launcher health gates."""

from __future__ import annotations

import os
import sys


_POSIX_MINIMUM_PIPE_BUF = 512


def emit_atomic_stdout_record(record: str) -> None:
    """Emit one short machine record without cross-process line fusion.

    Batch workers share the stdout pipe consumed by ``tee``.  ``print`` writes
    the payload and newline separately, so two workers can permanently fuse
    otherwise valid readiness records.  POSIX guarantees that one write of at
    most 512 bytes cannot interleave on a pipe; every launch record using this
    helper is deliberately kept below that portable minimum.

    The leading newline is part of the same write.  It makes the record begin
    at a logical line boundary even if another writer previously left an
    unterminated diagnostic fragment in the pipe.

    Test/capture streams may not expose a file descriptor.  They are not a
    process-shared launch pipe, so one checked high-level write is the faithful
    fallback for those streams.
    """

    if not isinstance(record, str) or not record:
        raise ValueError("Atomic stdout records must be non-empty strings.")
    if "\n" in record or "\r" in record:
        raise ValueError("Atomic stdout records must contain exactly one logical line.")
    payload = f"\n{record}\n".encode("utf-8")
    if len(payload) > _POSIX_MINIMUM_PIPE_BUF:
        raise ValueError(
            "Atomic stdout record exceeds the portable POSIX PIPE_BUF minimum: "
            f"bytes={len(payload)} limit={_POSIX_MINIMUM_PIPE_BUF}."
        )

    try:
        descriptor = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):
        text = payload.decode("utf-8")
        written = sys.stdout.write(text)
        if written != len(text):
            raise RuntimeError(
                "Atomic stdout fallback record was only partially written: "
                f"written={written!r} expected={len(text)}."
            )
        sys.stdout.flush()
        return

    # Preserve the ordering of earlier buffered Python diagnostics before
    # bypassing TextIOWrapper for the process-shared machine record.
    sys.stdout.flush()
    written = os.write(descriptor, payload)
    if written != len(payload):
        raise RuntimeError(
            "Atomic stdout record was only partially written: "
            f"written={written} expected={len(payload)}."
        )
