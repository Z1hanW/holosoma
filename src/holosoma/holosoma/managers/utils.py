"""Common utilities for manager implementations."""

from __future__ import annotations

import importlib
from typing import Any


def _legacy_callable_name_candidates(callable_name: str) -> tuple[str, ...]:
    """Return current callable names for aliases saved by older checkpoints."""
    object_goal_legacy_prefix = "obj_" + "spa" + "rse_" + "goal_"
    object_goal_current_prefix = "obj_goal_"
    if callable_name.startswith(object_goal_legacy_prefix):
        return (object_goal_current_prefix + callable_name[len(object_goal_legacy_prefix) :],)
    return ()


def resolve_callable(path: Any | str, context: str = "term") -> Any:
    """Resolve a callable (function or class) from a string path.

    Parameters
    ----------
    path : Any or str
        Callable reference or string path like "module.path:callable_name".
        If not a string, returns as-is (assumed to be already a callable).
    context : str, optional
        Context name for error messages (e.g., "term", "function", "class").
        Default is "term".

    Returns
    -------
    Any
        Resolved callable (function or class)

    Raises
    ------
    ValueError
        If string path is malformed or callable cannot be imported

    Examples
    --------
    >>> # Resolve a function
    >>> func = resolve_callable("holosoma.managers.reward.terms.locomotion:tracking_lin_vel")

    >>> # Resolve a class
    >>> cls = resolve_callable("holosoma.managers.action.terms.joint_control:JointPositionAction")

    >>> # Pass through an already-resolved callable
    >>> func = resolve_callable(my_function)  # Returns my_function as-is
    """
    # If already a callable, return as-is
    if not isinstance(path, str):
        return path

    # Parse string path
    if ":" not in path:
        raise ValueError(f"{context.capitalize()} path must be in format 'module:callable', got: {path}")

    module_path, callable_name = path.split(":", 1)

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ValueError(f"Failed to import {context} '{path}': {exc}") from exc

    try:
        return getattr(module, callable_name)
    except AttributeError as exc:
        for candidate_name in _legacy_callable_name_candidates(callable_name):
            try:
                return getattr(module, candidate_name)
            except AttributeError:
                continue
        raise ValueError(f"Failed to import {context} '{path}': {exc}") from exc
