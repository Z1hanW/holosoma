"""Pure helpers for IsaacSim joint hot-path configuration.

This module intentionally has no Isaac Sim or IsaacLab imports so its ordering
and selector contracts can be tested without starting the simulator.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import re
from typing import TypeVar

import torch


_ActuatorCfgT = TypeVar("_ActuatorCfgT")
_ALL_JOINTS_ACTUATOR_GROUP = "all_joints"


def _exact_joint_patterns(joint_names: Sequence[str]) -> list[str]:
    names = [str(name) for name in joint_names]
    if not names:
        raise ValueError("At least one joint is required to build an actuator group.")
    if len(set(names)) != len(names):
        raise ValueError(f"Joint names must be unique, got {names!r}.")
    # IsaacLab treats names as regular expressions and resolves them with
    # re.fullmatch. Escaping makes these expressions exact even for unusual
    # URDF joint names containing regex metacharacters.
    return [re.escape(name) for name in names]


def _per_joint_values(
    exact_joint_patterns: Sequence[str],
    values: Sequence[float],
    *,
    property_name: str,
) -> dict[str, float]:
    if len(values) != len(exact_joint_patterns):
        raise ValueError(
            f"{property_name} must contain one value per joint: "
            f"got {len(values)} values for {len(exact_joint_patterns)} joints."
        )
    return dict(zip(exact_joint_patterns, values, strict=True))


def build_ideal_pd_actuator_groups(
    *,
    actuator_cfg_type: Callable[..., _ActuatorCfgT],
    joint_names: Sequence[str],
    effort_limits: Sequence[float],
    velocity_limits: Sequence[float],
    armatures: Sequence[float],
    frictions: Sequence[float],
) -> dict[str, _ActuatorCfgT]:
    """Build one Ideal-PD group while preserving every per-joint property.

    IsaacLab's articulation loop dispatches once per actuator group. The old
    layout created one group per joint even though every group used the same
    explicit Ideal-PD model with zero gains. A single all-joints group lets
    IsaacLab use ``slice(None)`` internally while exact-name dictionaries keep
    the original heterogeneous limits and physical properties.
    """

    exact_patterns = _exact_joint_patterns(joint_names)
    actuator = actuator_cfg_type(
        joint_names_expr=list(exact_patterns),
        effort_limit=_per_joint_values(
            exact_patterns, effort_limits, property_name="effort_limits"
        ),
        velocity_limit=_per_joint_values(
            exact_patterns, velocity_limits, property_name="velocity_limits"
        ),
        stiffness=0,
        damping=0,
        armature=_per_joint_values(
            exact_patterns, armatures, property_name="armatures"
        ),
        friction=_per_joint_values(
            exact_patterns, frictions, property_name="frictions"
        ),
    )
    return {_ALL_JOINTS_ACTUATOR_GROUP: actuator}


def cached_dof_selector(
    dof_ids: Sequence[int], *, total_num_dofs: int
) -> Sequence[int] | slice:
    """Return a zero-copy selector only for a full-articulation identity map.

    Non-identity mappings must retain their original selector object so the
    Holosoma/config ordering remains unchanged.  A leading identity subset
    (for example ``[0, 1]`` out of a four-joint articulation) must not become
    ``slice(None)`` because that would silently expose the unconfigured joints.
    """

    if total_num_dofs < 0:
        raise ValueError(f"total_num_dofs must be non-negative, got {total_num_dofs}.")
    if len(dof_ids) == total_num_dofs and all(
        int(dof_id) == index for index, dof_id in enumerate(dof_ids)
    ):
        return slice(None)
    return dof_ids


def select_dof_write_batch(
    current_pos: torch.Tensor,
    current_vel: torch.Tensor,
    env_ids: torch.Tensor,
    dof_states: torch.Tensor | None,
    *,
    num_envs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select one IsaacSim joint-state write without assembling all DOFs.

    ``dof_states=None`` is the reset hot path: positions and velocities already
    live in separate IsaacLab-backed tensors, so stacking a full
    ``[num_envs, num_dofs, 2]`` state only to select a few rows is unnecessary.
    Explicit state accepts either a full environment batch or an already
    compact batch matching ``env_ids``; the latter is the shape documented by
    :meth:`IsaacSim.set_dof_state_tensor_robots`.
    """

    if dof_states is None:
        return current_pos[env_ids], current_vel[env_ids]
    if dof_states.ndim != 3 or dof_states.shape[-1] != 2:
        raise ValueError(
            "Expected dof_states shape [N, num_dofs, 2], "
            f"got {tuple(dof_states.shape)}"
        )
    if dof_states.shape[1] != current_pos.shape[1]:
        raise ValueError(
            f"Expected {current_pos.shape[1]} DOFs, got {dof_states.shape[1]}."
        )

    selected_count = env_ids.numel()
    if dof_states.shape[0] == num_envs:
        selected = dof_states[env_ids]
    elif dof_states.shape[0] == selected_count:
        selected = dof_states
    else:
        raise ValueError(
            "Expected dof_states batch size to match env_ids "
            f"({selected_count}) or num_envs ({num_envs}), got {dof_states.shape[0]}."
        )
    return selected[..., 0], selected[..., 1]
