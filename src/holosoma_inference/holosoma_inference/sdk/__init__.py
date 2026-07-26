"""Robot communication package."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import entry_points

# Auto-discover SDK interfaces from installed packages using lazy loading.
# Lazy loading is to avoid errors from SDK dependencies from extensions (e.g. ROS2) when working with other SDKs.
_entry_points = {ep.name: ep for ep in entry_points(group="holosoma.sdk")}
_registry = {}  # Cache for loaded interfaces
_builtin_interfaces = {
    "unitree": "holosoma_inference.sdk.unitree.unitree_interface:UnitreeInterface",
    "booster": "holosoma_inference.sdk.booster.booster_interface:BoosterInterface",
}


def _load_builtin_interface(qualified_name: str):
    module_name, attr_name = qualified_name.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def create_interface(robot_config, domain_id=0, interface_str=None, use_joystick=True):
    """Create interface from registry."""
    sdk_type = robot_config.sdk_type
    available_sdk_types = sorted(set(_entry_points) | set(_builtin_interfaces))
    if sdk_type not in _entry_points and sdk_type not in _builtin_interfaces:
        raise ValueError(f"Unknown sdk_type: {sdk_type}. Available: {available_sdk_types}")

    # Lazy load: only load the entry point when actually needed.
    if sdk_type not in _registry:
        if sdk_type in _entry_points:
            _registry[sdk_type] = _entry_points[sdk_type].load()
        else:
            _registry[sdk_type] = _load_builtin_interface(_builtin_interfaces[sdk_type])

    return _registry[sdk_type](robot_config, domain_id, interface_str, use_joystick)
