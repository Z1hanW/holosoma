"""Process-local initialization contract for lazy DeFM policy encoders."""

from __future__ import annotations

import os


DEFM_MATERIALIZATION_MODE_ENV = "HOLOSOMA_DEFM_MATERIALIZATION_MODE"
DEFM_MATERIALIZATION_MODES = frozenset({"fresh", "policy_init", "full_resume"})


def set_defm_materialization_mode(mode: str) -> str:
    """Install an authoritative mode, overriding any ambient shell value."""

    normalized = str(mode).strip().lower()
    if normalized not in DEFM_MATERIALIZATION_MODES:
        raise ValueError(
            f"DeFM materialization mode must be one of {sorted(DEFM_MATERIALIZATION_MODES)}, "
            f"got {mode!r}."
        )
    os.environ[DEFM_MATERIALIZATION_MODE_ENV] = normalized
    return normalized


def set_defm_checkpoint_restore_mode() -> str:
    """Make an evaluation/export setup construct architecture without external weights."""

    return set_defm_materialization_mode("full_resume")

