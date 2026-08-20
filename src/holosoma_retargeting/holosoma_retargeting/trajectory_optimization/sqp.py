"""Sequential convex TrajOpt loop for nonlinear retargeting models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .problem import SparseQuadraticProblem
from .solvers import SolveResult


class SparseQPSolver(Protocol):
    def solve(
        self,
        problem: SparseQuadraticProblem,
        warm_start: np.ndarray | None = None,
    ) -> SolveResult: ...


@dataclass(frozen=True)
class TrajOptSettings:
    max_iterations: int = 20
    initial_trust_radius: float = 0.2
    minimum_trust_radius: float = 1e-4
    maximum_trust_radius: float = 1.0
    line_search_steps: int = 10
    step_tolerance: float = 1e-5
    merit_tolerance: float = 1e-7


@dataclass(frozen=True)
class TrajOptIteration:
    iteration: int
    merit: float
    step_norm: float
    step_scale: float
    trust_radius: float
    qp_status: str
    qp_iterations: int


@dataclass(frozen=True)
class TrajOptResult:
    value: np.ndarray
    status: str
    iterations: tuple[TrajOptIteration, ...]


class SequentialTrajectoryOptimizer:
    """Trust-region sequential convex optimization over a flattened trajectory.

    ``linearize`` returns a QP whose variable is a delta from the current
    iterate. It is responsible for adding the current trust-region and
    linearized collision/dynamics constraints.
    """

    def __init__(
        self,
        solver: SparseQPSolver,
        settings: TrajOptSettings | None = None,
    ) -> None:
        self.solver = solver
        self.settings = TrajOptSettings() if settings is None else settings

    def optimize(
        self,
        initial: np.ndarray,
        linearize: Callable[[np.ndarray, float], SparseQuadraticProblem],
        merit: Callable[[np.ndarray], float],
    ) -> TrajOptResult:
        current = np.asarray(initial, dtype=np.float64).reshape(-1).copy()
        current_merit = float(merit(current))
        trust_radius = self.settings.initial_trust_radius
        history: list[TrajOptIteration] = []
        status = "maximum iterations reached"
        for iteration in range(self.settings.max_iterations):
            problem = linearize(current, trust_radius)
            qp_result = self.solver.solve(problem)
            if not qp_result.success:
                trust_radius *= 0.5
                if trust_radius < self.settings.minimum_trust_radius:
                    status = f"qp failed: {qp_result.status}"
                    break
                continue
            step = qp_result.solution
            step_norm = float(np.linalg.norm(step, ord=np.inf))
            accepted_scale = 0.0
            candidate_merit = current_merit
            for line_step in range(self.settings.line_search_steps):
                scale = 0.5**line_step
                candidate = current + scale * step
                value = float(merit(candidate))
                if np.isfinite(value) and value < current_merit:
                    accepted_scale = scale
                    candidate_merit = value
                    current = candidate
                    break
            history.append(
                TrajOptIteration(
                    iteration=iteration,
                    merit=candidate_merit,
                    step_norm=step_norm,
                    step_scale=accepted_scale,
                    trust_radius=trust_radius,
                    qp_status=qp_result.status,
                    qp_iterations=qp_result.iterations,
                )
            )
            if accepted_scale == 0.0:
                trust_radius *= 0.5
                if trust_radius < self.settings.minimum_trust_radius:
                    status = "line search stalled"
                    break
                continue
            improvement = current_merit - candidate_merit
            current_merit = candidate_merit
            if accepted_scale == 1.0:
                trust_radius = min(
                    self.settings.maximum_trust_radius, trust_radius * 1.5
                )
            if (
                accepted_scale * step_norm <= self.settings.step_tolerance
                or improvement <= self.settings.merit_tolerance
            ):
                status = "converged"
                break
        return TrajOptResult(
            value=current,
            status=status,
            iterations=tuple(history),
        )
