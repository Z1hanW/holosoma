"""Unsupported archival inference snapshot; use ``src/holosoma_inference``."""

from __future__ import annotations

import os


if os.environ.get("HOLOSOMA_ALLOW_UNSAFE_ARCHIVAL_INFERENCE") != "1":
    raise RuntimeError(
        "src/holosoma_inference_wrong is an unsupported archival snapshot with known "
        "scientific/deployment contract bugs. Use src/holosoma_inference. Set "
        "HOLOSOMA_ALLOW_UNSAFE_ARCHIVAL_INFERENCE=1 only for explicit historical forensics."
    )
