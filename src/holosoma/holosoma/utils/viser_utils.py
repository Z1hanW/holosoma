from __future__ import annotations

import os
import random
import sys
from pathlib import Path


def ensure_viser_on_path() -> None:
    """Ensure the local Viser sources are importable if vendored in the repo."""
    here = Path(__file__).resolve()
    candidates: list[Path] = []
    for parent in here.parents:
        candidates.append(parent / "viser" / "src")
        candidates.append(parent / "holosoma" / "viser" / "src")

    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def random_viser_port() -> int:
    """Return a random 4-digit port (1024-9999)."""
    return random.randint(1024, 9999)


def resolve_viser_port(port: int | None = None, *, env_var: str = "HOLOSOMA_VISER_PORT") -> int:
    """Resolve a Viser port from env or default, falling back to random."""
    env_value = os.environ.get(env_var)
    if env_value:
        try:
            return int(env_value)
        except ValueError:
            pass

    if port is None:
        return random_viser_port()
    try:
        port_val = int(port)
    except (TypeError, ValueError):
        return random_viser_port()
    if port_val <= 0:
        return random_viser_port()
    return port_val
