"""Whole-trajectory retargeting and sparse solver experiments."""

from .builder import (
    LinearDynamics,
    LinearizedConstraint,
    WholeTrajectoryProblem,
    WholeTrajectorySolution,
    WholeTrajectorySpec,
    piecewise_linear_scale_basis,
)
from .mujoco_dynamics import (
    MujocoDynamicsLinearization,
    MujocoDynamicsLinearizer,
    MujocoNominalTrajectory,
)
from .problem import SparseQuadraticBuilder, SparseQuadraticProblem
from .solvers import (
    AutoSparseSolver,
    OSQPSolver,
    SolveResult,
    TorchADMMSettings,
    TorchSparseADMMSolver,
)
from .sqp import (
    SequentialTrajectoryOptimizer,
    TrajOptIteration,
    TrajOptResult,
    TrajOptSettings,
)

__all__ = [
    "LinearDynamics",
    "LinearizedConstraint",
    "AutoSparseSolver",
    "MujocoDynamicsLinearization",
    "MujocoDynamicsLinearizer",
    "MujocoNominalTrajectory",
    "OSQPSolver",
    "SequentialTrajectoryOptimizer",
    "SolveResult",
    "SparseQuadraticBuilder",
    "SparseQuadraticProblem",
    "TorchADMMSettings",
    "TorchSparseADMMSolver",
    "TrajOptIteration",
    "TrajOptResult",
    "TrajOptSettings",
    "WholeTrajectoryProblem",
    "WholeTrajectorySolution",
    "WholeTrajectorySpec",
    "piecewise_linear_scale_basis",
]
