"""Robot communication package."""
from __future__ import annotations
from importlib.metadata import entry_points

# Auto-discover SDK interfaces from installed packages
_registry = {ep.name: ep.load() for ep in entry_points(group="holosoma.sdk")}


def create_interface(robot_config, domain_id=0, interface_str=None, use_joystick=True):
    """Create interface from registry."""
    if robot_config.sdk_type not in _registry:
        raise ValueError(f"Unknown sdk_type: {robot_config.sdk_type}. Available: {sorted(_registry.keys())}")
    return _registry[robot_config.sdk_type](robot_config, domain_id, interface_str, use_joystick)
