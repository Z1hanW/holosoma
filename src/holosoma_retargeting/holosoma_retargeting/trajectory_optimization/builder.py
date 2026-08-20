"""Whole-trajectory QP construction with vector-valued scale fields."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .problem import SparseQuadraticBuilder, SparseQuadraticProblem


def piecewise_linear_scale_basis(frame_count: int, knot_count: int) -> np.ndarray:
    """Return a local-support basis mapping scale knots to every frame."""

    if frame_count <= 0 or knot_count <= 0:
        raise ValueError("frame_count and knot_count must be positive")
    if knot_count == 1:
        return np.ones((frame_count, 1), dtype=np.float64)
    knot_positions = np.linspace(0.0, frame_count - 1, knot_count)
    frame_positions = np.arange(frame_count, dtype=np.float64)
    basis = np.zeros((frame_count, knot_count), dtype=np.float64)
    for frame, position in enumerate(frame_positions):
        right = int(np.searchsorted(knot_positions, position, side="right"))
        right = min(max(right, 1), knot_count - 1)
        left = right - 1
        interval = knot_positions[right] - knot_positions[left]
        alpha = (position - knot_positions[left]) / interval
        basis[frame, left] = 1.0 - alpha
        basis[frame, right] = alpha
    return basis


@dataclass(frozen=True)
class LinearDynamics:
    """Time-varying affine dynamics ``x[t+1] = A[t] x[t] + B[t] u[t] + c[t]``."""

    transition: np.ndarray
    control: np.ndarray
    offset: np.ndarray
    active: np.ndarray | None = None


@dataclass(frozen=True)
class LinearizedConstraint:
    indices: np.ndarray
    matrix: np.ndarray
    lower: np.ndarray
    upper: np.ndarray


@dataclass
class WholeTrajectorySpec:
    """Inputs for a joint trajectory, scale-field, and control optimization."""

    state_reference: np.ndarray
    scale_reference: np.ndarray
    scale_basis: np.ndarray
    tracking_state_jacobian: np.ndarray
    tracking_scale_jacobian: np.ndarray
    tracking_target: np.ndarray
    tracking_offset: np.ndarray | None = None
    control_reference: np.ndarray | None = None
    dynamics: LinearDynamics | None = None
    dynamics_weight: float | np.ndarray | None = None
    dynamics_soft_formulation: str = "slack"
    normalize_dynamics_rows: bool = True
    initial_state: np.ndarray | None = None
    tracking_weight: float | np.ndarray = 1.0
    state_prior_weight: float | np.ndarray = 1e-3
    scale_prior_weight: float | np.ndarray = 1e-3
    control_weight: float | np.ndarray = 1e-3
    state_velocity_weight: float = 1.0
    state_acceleration_weight: float = 1.0
    state_velocity_target: np.ndarray | None = None
    state_acceleration_target: np.ndarray | None = None
    scale_smoothness_weight: float = 1.0
    state_lower: float | np.ndarray = -np.inf
    state_upper: float | np.ndarray = np.inf
    scale_lower: float | np.ndarray = 0.25
    scale_upper: float | np.ndarray = 4.0
    control_lower: float | np.ndarray = -np.inf
    control_upper: float | np.ndarray = np.inf
    extra_constraints: list[LinearizedConstraint] = field(default_factory=list)
    diagonal_regularization: float = 1e-8


@dataclass(frozen=True)
class VariableLayout:
    frame_count: int
    state_dimension: int
    knot_count: int
    scale_dimension: int
    control_dimension: int
    dynamics_slack_dimension: int = 0

    @property
    def state_count(self) -> int:
        return self.frame_count * self.state_dimension

    @property
    def scale_count(self) -> int:
        return self.knot_count * self.scale_dimension

    @property
    def control_count(self) -> int:
        return max(0, self.frame_count - 1) * self.control_dimension

    @property
    def dynamics_slack_count(self) -> int:
        return (
            max(0, self.frame_count - 1)
            * self.dynamics_slack_dimension
        )

    @property
    def variable_count(self) -> int:
        return (
            self.state_count
            + self.scale_count
            + self.control_count
            + self.dynamics_slack_count
        )

    def state_indices(self, frame: int) -> np.ndarray:
        start = frame * self.state_dimension
        return np.arange(start, start + self.state_dimension)

    def scale_indices(self, knot: int) -> np.ndarray:
        start = self.state_count + knot * self.scale_dimension
        return np.arange(start, start + self.scale_dimension)

    def control_indices(self, frame: int) -> np.ndarray:
        start = self.state_count + self.scale_count + frame * self.control_dimension
        return np.arange(start, start + self.control_dimension)

    def dynamics_slack_indices(self, frame: int) -> np.ndarray:
        start = (
            self.state_count
            + self.scale_count
            + self.control_count
            + frame * self.dynamics_slack_dimension
        )
        return np.arange(start, start + self.dynamics_slack_dimension)

    @property
    def all_state_indices(self) -> np.ndarray:
        return np.arange(self.state_count)

    @property
    def all_scale_indices(self) -> np.ndarray:
        return np.arange(self.state_count, self.state_count + self.scale_count)

    @property
    def all_control_indices(self) -> np.ndarray:
        start = self.state_count + self.scale_count
        return np.arange(
            start,
            start + self.control_count,
        )

    @property
    def all_dynamics_slack_indices(self) -> np.ndarray:
        start = (
            self.state_count
            + self.scale_count
            + self.control_count
        )
        return np.arange(
            start,
            start + self.dynamics_slack_count,
        )


@dataclass(frozen=True)
class WholeTrajectorySolution:
    states: np.ndarray
    scale_knots: np.ndarray
    frame_scales: np.ndarray
    controls: np.ndarray | None
    dynamics_slacks: np.ndarray | None


def _broadcast(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        result = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
    except ValueError as exc:
        raise ValueError(f"{name} cannot broadcast to {shape}") from exc
    if np.isnan(result).any():
        raise ValueError(f"{name} must not contain NaN")
    return result


def _weight_rows(weight, frame: int, frame_count: int, row_count: int):
    array = np.asarray(weight, dtype=np.float64)
    if array.ndim == 0:
        return float(array)
    if array.shape == (row_count,):
        return array
    if array.shape == (frame_count, row_count):
        return array[frame]
    raise ValueError(
        f"tracking_weight must be scalar, ({row_count},), or "
        f"({frame_count}, {row_count}); got {array.shape}"
    )


class WholeTrajectoryProblem:
    """Build and unpack a horizon-wide sparse QP."""

    def __init__(self, spec: WholeTrajectorySpec) -> None:
        self.spec = spec
        state_reference = np.asarray(spec.state_reference, dtype=np.float64)
        scale_reference = np.asarray(spec.scale_reference, dtype=np.float64)
        scale_basis = np.asarray(spec.scale_basis, dtype=np.float64)
        if state_reference.ndim != 2:
            raise ValueError("state_reference must have shape (T, nx)")
        if scale_reference.ndim != 2:
            raise ValueError("scale_reference must have shape (K, ns)")
        frame_count, state_dimension = state_reference.shape
        knot_count, scale_dimension = scale_reference.shape
        if scale_basis.shape != (frame_count, knot_count):
            raise ValueError(
                f"scale_basis must have shape {(frame_count, knot_count)}, got {scale_basis.shape}"
            )
        if not np.allclose(scale_basis.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("every scale_basis row must sum to one")
        tracking_state = np.asarray(spec.tracking_state_jacobian, dtype=np.float64)
        tracking_scale = np.asarray(spec.tracking_scale_jacobian, dtype=np.float64)
        tracking_target = np.asarray(spec.tracking_target, dtype=np.float64)
        if tracking_state.ndim != 3 or tracking_state.shape[0] != frame_count:
            raise ValueError("tracking_state_jacobian must have shape (T, M, nx)")
        tracking_rows = tracking_state.shape[1]
        if tracking_state.shape[2] != state_dimension:
            raise ValueError("tracking state dimension disagrees with state_reference")
        if tracking_scale.shape != (frame_count, tracking_rows, scale_dimension):
            raise ValueError("tracking_scale_jacobian must have shape (T, M, ns)")
        if tracking_target.shape != (frame_count, tracking_rows):
            raise ValueError("tracking_target must have shape (T, M)")
        if spec.tracking_offset is None:
            tracking_offset = np.zeros_like(tracking_target)
        else:
            tracking_offset = np.asarray(spec.tracking_offset, dtype=np.float64)
            if tracking_offset.shape != tracking_target.shape:
                raise ValueError("tracking_offset shape disagrees with tracking_target")

        control_dimension = 0
        control_reference = None
        if spec.control_reference is not None:
            control_reference = np.asarray(spec.control_reference, dtype=np.float64)
            if control_reference.ndim != 2 or control_reference.shape[0] != frame_count - 1:
                raise ValueError("control_reference must have shape (T-1, nu)")
            control_dimension = control_reference.shape[1]
        if spec.dynamics is not None and control_reference is None:
            raise ValueError("dynamics requires control_reference")
        if spec.dynamics is None and spec.dynamics_weight is not None:
            raise ValueError("dynamics_weight requires dynamics")
        if spec.dynamics_soft_formulation not in {"slack", "penalty"}:
            raise ValueError(
                "dynamics_soft_formulation must be 'slack' or 'penalty'"
            )
        self.layout = VariableLayout(
            frame_count=frame_count,
            state_dimension=state_dimension,
            knot_count=knot_count,
            scale_dimension=scale_dimension,
            control_dimension=control_dimension,
            dynamics_slack_dimension=(
                state_dimension
                if spec.dynamics is not None
                and spec.dynamics_weight is not None
                and spec.dynamics_soft_formulation == "slack"
                else 0
            ),
        )
        self._state_reference = state_reference
        self._scale_reference = scale_reference
        self._scale_basis = scale_basis
        self._control_reference = control_reference
        self._tracking_state = tracking_state
        self._tracking_scale = tracking_scale
        self._tracking_target = tracking_target
        self._tracking_offset = tracking_offset
        if spec.state_velocity_target is None:
            self._state_velocity_target = np.zeros(
                (max(0, frame_count - 1), state_dimension),
                dtype=np.float64,
            )
        else:
            self._state_velocity_target = np.asarray(
                spec.state_velocity_target,
                dtype=np.float64,
            )
            expected = (max(0, frame_count - 1), state_dimension)
            if self._state_velocity_target.shape != expected:
                raise ValueError(
                    f"state_velocity_target must have shape {expected}"
                )
        if spec.state_acceleration_target is None:
            self._state_acceleration_target = np.zeros(
                (max(0, frame_count - 2), state_dimension),
                dtype=np.float64,
            )
        else:
            self._state_acceleration_target = np.asarray(
                spec.state_acceleration_target,
                dtype=np.float64,
            )
            expected = (max(0, frame_count - 2), state_dimension)
            if self._state_acceleration_target.shape != expected:
                raise ValueError(
                    f"state_acceleration_target must have shape {expected}"
                )

    def build(self) -> SparseQuadraticProblem:
        spec = self.spec
        layout = self.layout
        builder = SparseQuadraticBuilder(layout.variable_count)
        frame_count = layout.frame_count

        for frame in range(frame_count):
            active_knots = np.flatnonzero(np.abs(self._scale_basis[frame]) > 1e-14)
            local_indices = [layout.state_indices(frame)]
            local_matrix = [self._tracking_state[frame]]
            for knot in active_knots:
                local_indices.append(layout.scale_indices(int(knot)))
                local_matrix.append(
                    self._tracking_scale[frame] * self._scale_basis[frame, knot]
                )
            builder.add_least_squares(
                np.concatenate(local_indices),
                np.hstack(local_matrix),
                self._tracking_target[frame] - self._tracking_offset[frame],
                _weight_rows(
                    spec.tracking_weight,
                    frame,
                    frame_count,
                    self._tracking_target.shape[1],
                ),
            )
            state_indices = layout.state_indices(frame)
            builder.add_least_squares(
                state_indices,
                np.eye(layout.state_dimension),
                self._state_reference[frame],
                spec.state_prior_weight,
            )
            if frame:
                difference_indices = np.concatenate(
                    (layout.state_indices(frame - 1), state_indices)
                )
                difference_matrix = np.hstack(
                    (-np.eye(layout.state_dimension), np.eye(layout.state_dimension))
                )
                builder.add_least_squares(
                    difference_indices,
                    difference_matrix,
                    self._state_velocity_target[frame - 1],
                    spec.state_velocity_weight,
                )
            if frame >= 2:
                acceleration_indices = np.concatenate(
                    (
                        layout.state_indices(frame - 2),
                        layout.state_indices(frame - 1),
                        state_indices,
                    )
                )
                acceleration_matrix = np.hstack(
                    (
                        np.eye(layout.state_dimension),
                        -2.0 * np.eye(layout.state_dimension),
                        np.eye(layout.state_dimension),
                    )
                )
                builder.add_least_squares(
                    acceleration_indices,
                    acceleration_matrix,
                    self._state_acceleration_target[frame - 2],
                    spec.state_acceleration_weight,
                )

        for knot in range(layout.knot_count):
            scale_indices = layout.scale_indices(knot)
            builder.add_least_squares(
                scale_indices,
                np.eye(layout.scale_dimension),
                self._scale_reference[knot],
                spec.scale_prior_weight,
            )
            if knot:
                builder.add_least_squares(
                    np.concatenate((layout.scale_indices(knot - 1), scale_indices)),
                    np.hstack(
                        (
                            -np.eye(layout.scale_dimension),
                            np.eye(layout.scale_dimension),
                        )
                    ),
                    np.zeros(layout.scale_dimension),
                    spec.scale_smoothness_weight,
                )

        if self._control_reference is not None:
            for frame in range(frame_count - 1):
                builder.add_least_squares(
                    layout.control_indices(frame),
                    np.eye(layout.control_dimension),
                    self._control_reference[frame],
                    spec.control_weight,
                )
        if layout.dynamics_slack_dimension:
            dynamics_active = (
                np.ones(frame_count - 1, dtype=bool)
                if spec.dynamics.active is None
                else np.asarray(spec.dynamics.active, dtype=bool)
            )
            for frame in range(frame_count - 1):
                if not dynamics_active[frame]:
                    continue
                builder.add_least_squares(
                    layout.dynamics_slack_indices(frame),
                    np.eye(layout.dynamics_slack_dimension),
                    np.zeros(layout.dynamics_slack_dimension),
                    spec.dynamics_weight,
                )

        state_lower = _broadcast(
            spec.state_lower, self._state_reference.shape, "state_lower"
        )
        state_upper = _broadcast(
            spec.state_upper, self._state_reference.shape, "state_upper"
        )
        builder.add_variable_bounds(
            layout.all_state_indices,
            state_lower.reshape(-1),
            state_upper.reshape(-1),
        )
        scale_lower = _broadcast(
            spec.scale_lower, self._scale_reference.shape, "scale_lower"
        )
        scale_upper = _broadcast(
            spec.scale_upper, self._scale_reference.shape, "scale_upper"
        )
        builder.add_variable_bounds(
            layout.all_scale_indices,
            scale_lower.reshape(-1),
            scale_upper.reshape(-1),
        )
        if self._control_reference is not None:
            control_lower = _broadcast(
                spec.control_lower,
                self._control_reference.shape,
                "control_lower",
            )
            control_upper = _broadcast(
                spec.control_upper,
                self._control_reference.shape,
                "control_upper",
            )
            builder.add_variable_bounds(
                layout.all_control_indices,
                control_lower.reshape(-1),
                control_upper.reshape(-1),
            )

        if spec.initial_state is not None:
            initial_state = _broadcast(
                spec.initial_state, (layout.state_dimension,), "initial_state"
            )
            builder.add_linear_constraint(
                layout.state_indices(0),
                np.eye(layout.state_dimension),
                initial_state,
                initial_state,
            )

        if spec.dynamics is not None:
            dynamics = spec.dynamics
            transition = np.asarray(dynamics.transition, dtype=np.float64)
            control = np.asarray(dynamics.control, dtype=np.float64)
            offset = np.asarray(dynamics.offset, dtype=np.float64)
            expected_a = (
                frame_count - 1,
                layout.state_dimension,
                layout.state_dimension,
            )
            expected_b = (
                frame_count - 1,
                layout.state_dimension,
                layout.control_dimension,
            )
            expected_c = (frame_count - 1, layout.state_dimension)
            if transition.shape != expected_a:
                raise ValueError(f"dynamics transition must have shape {expected_a}")
            if control.shape != expected_b:
                raise ValueError(f"dynamics control must have shape {expected_b}")
            if offset.shape != expected_c:
                raise ValueError(f"dynamics offset must have shape {expected_c}")
            if dynamics.active is None:
                dynamics_active = np.ones(frame_count - 1, dtype=bool)
            else:
                dynamics_active = np.asarray(
                    dynamics.active,
                    dtype=bool,
                )
                if dynamics_active.shape != (frame_count - 1,):
                    raise ValueError(
                        "dynamics active mask must have shape "
                        f"{(frame_count - 1,)}"
                    )
            for frame in range(frame_count - 1):
                if not dynamics_active[frame]:
                    continue
                indices = np.concatenate(
                    (
                        layout.state_indices(frame),
                        layout.state_indices(frame + 1),
                        layout.control_indices(frame),
                        (
                            layout.dynamics_slack_indices(frame)
                            if layout.dynamics_slack_dimension
                            else np.empty(0, dtype=np.int64)
                        ),
                    )
                )
                blocks = [
                    -transition[frame],
                    np.eye(layout.state_dimension),
                    -control[frame],
                ]
                penalty_formulation = (
                    spec.dynamics_weight is not None
                    and spec.dynamics_soft_formulation == "penalty"
                )
                if layout.dynamics_slack_dimension:
                    blocks.append(-np.eye(layout.state_dimension))
                matrix = np.hstack(blocks)
                if penalty_formulation:
                    builder.add_least_squares(
                        indices,
                        matrix,
                        offset[frame],
                        spec.dynamics_weight,
                    )
                    continue
                right_hand_side = np.asarray(offset[frame])
                if spec.normalize_dynamics_rows:
                    row_scale = np.maximum(
                        1.0,
                        np.max(np.abs(matrix), axis=1),
                    )
                    matrix = matrix / row_scale[:, None]
                    right_hand_side = right_hand_side / row_scale
                builder.add_linear_constraint(
                    indices,
                    matrix,
                    right_hand_side,
                    right_hand_side,
                )

        for constraint in spec.extra_constraints:
            builder.add_linear_constraint(
                constraint.indices,
                constraint.matrix,
                constraint.lower,
                constraint.upper,
            )
        return builder.build(
            diagonal_regularization=spec.diagonal_regularization,
            metadata={
                "frame_count": layout.frame_count,
                "state_dimension": layout.state_dimension,
                "scale_dimension": layout.scale_dimension,
                "scale_knot_count": layout.knot_count,
                "control_dimension": layout.control_dimension,
                "dynamics_slack_dimension": (
                    layout.dynamics_slack_dimension
                ),
                "dynamics_mode": (
                    "none"
                    if spec.dynamics is None
                    else (
                        "hard"
                        if spec.dynamics_weight is None
                        else f"soft-{spec.dynamics_soft_formulation}"
                    )
                ),
                "dynamics_rows_normalized": bool(
                    spec.dynamics is not None
                    and spec.normalize_dynamics_rows
                ),
                "active_dynamics_transitions": (
                    0
                    if spec.dynamics is None
                    else int(np.count_nonzero(dynamics_active))
                ),
            },
        )

    def unpack(self, solution: np.ndarray) -> WholeTrajectorySolution:
        solution = np.asarray(solution, dtype=np.float64).reshape(-1)
        if solution.shape != (self.layout.variable_count,):
            raise ValueError("solution shape disagrees with trajectory layout")
        states = solution[self.layout.all_state_indices].reshape(
            self.layout.frame_count, self.layout.state_dimension
        )
        scale_knots = solution[self.layout.all_scale_indices].reshape(
            self.layout.knot_count, self.layout.scale_dimension
        )
        controls = None
        if self.layout.control_dimension:
            controls = solution[self.layout.all_control_indices].reshape(
                self.layout.frame_count - 1,
                self.layout.control_dimension,
            )
        dynamics_slacks = None
        if self.layout.dynamics_slack_dimension:
            dynamics_slacks = solution[
                self.layout.all_dynamics_slack_indices
            ].reshape(
                self.layout.frame_count - 1,
                self.layout.dynamics_slack_dimension,
            )
        return WholeTrajectorySolution(
            states=states,
            scale_knots=scale_knots,
            frame_scales=self._scale_basis @ scale_knots,
            controls=controls,
            dynamics_slacks=dynamics_slacks,
        )

    def reference_vector(self) -> np.ndarray:
        parts = [self._state_reference.reshape(-1), self._scale_reference.reshape(-1)]
        if self._control_reference is not None:
            parts.append(self._control_reference.reshape(-1))
        if self.layout.dynamics_slack_dimension:
            parts.append(np.zeros(self.layout.dynamics_slack_count))
        return np.concatenate(parts)
