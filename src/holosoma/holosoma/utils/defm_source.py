"""Resolve the single pinned DeFM source tree used for hashing and imports."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path


def _is_defm_source_root(path: Path) -> bool:
    return (path / "defm" / "model_factory.py").is_file()


def resolve_defm_source_root(
    *,
    environ: Mapping[str, str] | None = None,
    anchor: Path | None = None,
) -> Path:
    """Return one unambiguous local DeFM checkout or fail closed."""

    if environ is None:
        environ = os.environ
    configured = str(environ.get("HOLOSOMA_DEFM_ROOT", "") or "").strip()
    if configured:
        resolved = Path(configured).expanduser().resolve()
        if not _is_defm_source_root(resolved):
            raise FileNotFoundError(
                "HOLOSOMA_DEFM_ROOT must identify a directory containing "
                f"defm/model_factory.py, got {resolved}."
            )
        return resolved

    search_anchor = (anchor or Path(__file__)).expanduser().resolve()
    roots: list[Path] = []
    seen: set[Path] = set()
    base = search_anchor if search_anchor.is_dir() else search_anchor.parent
    for parent in (base, *base.parents):
        for candidate in (parent / "submodules" / "defm", parent / "defm"):
            resolved = candidate.resolve()
            if resolved in seen or not _is_defm_source_root(resolved):
                continue
            seen.add(resolved)
            roots.append(resolved)
    if not roots:
        raise FileNotFoundError(
            "Unable to locate the pinned DeFM source tree. Initialize submodules/defm or set "
            "HOLOSOMA_DEFM_ROOT to a directory containing defm/model_factory.py."
        )
    if len(roots) != 1:
        raise RuntimeError(
            "Multiple local DeFM source trees are visible, so provenance hashing and Python import "
            f"could select different implementations: {[str(path) for path in roots]}. Set "
            "HOLOSOMA_DEFM_ROOT explicitly to the pinned checkout."
        )
    return roots[0]


def require_module_within_defm_root(module: object, source_root: Path, *, name: str) -> None:
    """Prove that an imported DeFM module came from the hashed checkout."""

    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise RuntimeError(f"Imported {name} has no auditable __file__ path.")
    resolved_module = Path(module_file).resolve()
    try:
        resolved_module.relative_to(source_root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            f"Imported {name} came from {resolved_module}, outside the pinned/hashed DeFM root "
            f"{source_root.resolve()}. Restart with a clean Python process and correct "
            "HOLOSOMA_DEFM_ROOT/PYTHONPATH."
        ) from exc
