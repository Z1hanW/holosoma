from __future__ import annotations

import os
import random


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
