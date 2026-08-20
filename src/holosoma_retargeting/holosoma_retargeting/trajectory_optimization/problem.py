"""Sparse quadratic problem primitives used by whole-trajectory optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import sparse as sp


def _as_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {result.shape}")
    if np.isnan(result).any():
        raise ValueError(f"{name} must not contain NaN")
    return result


@dataclass(frozen=True)
class SparseQuadraticProblem:
    """Convex QP in OSQP form.

    Minimize ``0.5 * x.T @ H @ x + g.T @ x`` subject to
    ``lower <= A @ x <= upper``.
    """

    hessian: sp.spmatrix
    gradient: np.ndarray
    constraint_matrix: sp.spmatrix
    lower: np.ndarray
    upper: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        hessian = sp.csc_matrix(self.hessian, dtype=np.float64)
        if hessian.shape[0] != hessian.shape[1]:
            raise ValueError(f"hessian must be square, got {hessian.shape}")
        variable_count = hessian.shape[0]
        gradient = _as_vector(self.gradient, variable_count, "gradient")
        constraint_matrix = sp.csr_matrix(
            self.constraint_matrix, shape=(self.constraint_matrix.shape[0], variable_count), dtype=np.float64
        )
        lower = _as_vector(self.lower, constraint_matrix.shape[0], "lower")
        upper = _as_vector(self.upper, constraint_matrix.shape[0], "upper")
        if np.any(lower > upper):
            raise ValueError("constraint lower bounds must not exceed upper bounds")
        asymmetry = hessian - hessian.T
        if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-9:
            raise ValueError("hessian must be symmetric")
        if not np.isfinite(hessian.data).all() or not np.isfinite(gradient).all():
            raise ValueError("objective must contain only finite values")
        if not np.isfinite(constraint_matrix.data).all():
            raise ValueError("constraint matrix must contain only finite values")
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "constraint_matrix", constraint_matrix)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def variable_count(self) -> int:
        return self.hessian.shape[0]

    @property
    def constraint_count(self) -> int:
        return self.constraint_matrix.shape[0]

    def objective(self, x: np.ndarray) -> float:
        value = _as_vector(x, self.variable_count, "x")
        return float(0.5 * value @ (self.hessian @ value) + self.gradient @ value)

    def max_constraint_violation(self, x: np.ndarray) -> float:
        if self.constraint_count == 0:
            return 0.0
        value = self.constraint_matrix @ _as_vector(x, self.variable_count, "x")
        below = np.maximum(self.lower - value, 0.0)
        above = np.maximum(value - self.upper, 0.0)
        return float(max(np.max(below), np.max(above)))


class SparseQuadraticBuilder:
    """Incrementally assemble a sparse QP from small local blocks."""

    def __init__(self, variable_count: int) -> None:
        if variable_count <= 0:
            raise ValueError("variable_count must be positive")
        self.variable_count = int(variable_count)
        self._h_rows: list[np.ndarray] = []
        self._h_cols: list[np.ndarray] = []
        self._h_values: list[np.ndarray] = []
        self._gradient = np.zeros(self.variable_count, dtype=np.float64)
        self._a_rows: list[np.ndarray] = []
        self._a_cols: list[np.ndarray] = []
        self._a_values: list[np.ndarray] = []
        self._lower: list[np.ndarray] = []
        self._upper: list[np.ndarray] = []
        self._constraint_count = 0

    def add_quadratic(
        self,
        indices: np.ndarray,
        hessian: np.ndarray,
        gradient: np.ndarray | None = None,
    ) -> None:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.variable_count):
            raise ValueError("quadratic indices are out of range")
        local_hessian = np.asarray(hessian, dtype=np.float64)
        if local_hessian.shape != (len(indices), len(indices)):
            raise ValueError("local hessian shape disagrees with indices")
        local_rows, local_cols = np.nonzero(local_hessian)
        if len(local_rows):
            self._h_rows.append(indices[local_rows])
            self._h_cols.append(indices[local_cols])
            self._h_values.append(local_hessian[local_rows, local_cols])
        if gradient is not None:
            local_gradient = _as_vector(gradient, len(indices), "local gradient")
            np.add.at(self._gradient, indices, local_gradient)

    def add_least_squares(
        self,
        indices: np.ndarray,
        matrix: np.ndarray,
        target: np.ndarray,
        weight: float | np.ndarray = 1.0,
    ) -> None:
        """Add ``||W**0.5 * (matrix @ x[indices] - target)||**2``."""

        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(indices):
            raise ValueError("least-squares matrix shape disagrees with indices")
        target = _as_vector(target, matrix.shape[0], "least-squares target")
        if np.isscalar(weight):
            scalar_weight = float(weight)
            if not np.isfinite(scalar_weight) or scalar_weight < 0.0:
                raise ValueError("least-squares weight must be finite and non-negative")
            weighted_matrix = matrix * np.sqrt(scalar_weight)
            weighted_target = target * np.sqrt(scalar_weight)
        else:
            weight_vector = _as_vector(np.asarray(weight), matrix.shape[0], "least-squares weight")
            if np.any(weight_vector < 0.0) or not np.isfinite(weight_vector).all():
                raise ValueError("least-squares weight must be finite and non-negative")
            sqrt_weight = np.sqrt(weight_vector)
            weighted_matrix = matrix * sqrt_weight[:, None]
            weighted_target = target * sqrt_weight
        self.add_quadratic(
            indices,
            2.0 * weighted_matrix.T @ weighted_matrix,
            -2.0 * weighted_matrix.T @ weighted_target,
        )

    def add_linear_constraint(
        self,
        indices: np.ndarray,
        matrix: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(indices):
            raise ValueError("constraint matrix shape disagrees with indices")
        row_count = matrix.shape[0]
        lower = _as_vector(lower, row_count, "constraint lower")
        upper = _as_vector(upper, row_count, "constraint upper")
        if np.any(lower > upper):
            raise ValueError("constraint lower bounds must not exceed upper bounds")
        rows, local_cols = np.nonzero(matrix)
        self._a_rows.append(rows.astype(np.int64) + self._constraint_count)
        self._a_cols.append(indices[local_cols])
        self._a_values.append(matrix[rows, local_cols])
        self._lower.append(lower)
        self._upper.append(upper)
        self._constraint_count += row_count

    def add_variable_bounds(
        self,
        indices: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.variable_count):
            raise ValueError("bound indices are out of range")
        lower = _as_vector(lower, len(indices), "bound lower")
        upper = _as_vector(upper, len(indices), "bound upper")
        if np.any(lower > upper):
            raise ValueError("bound lower values must not exceed upper values")
        rows = np.arange(len(indices), dtype=np.int64) + self._constraint_count
        self._a_rows.append(rows)
        self._a_cols.append(indices)
        self._a_values.append(np.ones(len(indices), dtype=np.float64))
        self._lower.append(lower)
        self._upper.append(upper)
        self._constraint_count += len(indices)

    def build(
        self,
        *,
        diagonal_regularization: float = 1e-9,
        metadata: dict[str, Any] | None = None,
    ) -> SparseQuadraticProblem:
        if diagonal_regularization < 0.0 or not np.isfinite(diagonal_regularization):
            raise ValueError("diagonal_regularization must be finite and non-negative")
        if self._h_values:
            rows = np.concatenate(self._h_rows)
            cols = np.concatenate(self._h_cols)
            values = np.concatenate(self._h_values)
            hessian = sp.coo_matrix(
                (values, (rows, cols)),
                shape=(self.variable_count, self.variable_count),
            ).tocsc()
        else:
            hessian = sp.csc_matrix((self.variable_count, self.variable_count))
        if diagonal_regularization:
            hessian = hessian + diagonal_regularization * sp.eye(
                self.variable_count, format="csc"
            )
        if self._a_values:
            a_rows = np.concatenate(self._a_rows)
            a_cols = np.concatenate(self._a_cols)
            a_values = np.concatenate(self._a_values)
            constraint_matrix = sp.coo_matrix(
                (a_values, (a_rows, a_cols)),
                shape=(self._constraint_count, self.variable_count),
            ).tocsr()
            lower = np.concatenate(self._lower)
            upper = np.concatenate(self._upper)
        else:
            constraint_matrix = sp.csr_matrix((0, self.variable_count))
            lower = np.empty(0, dtype=np.float64)
            upper = np.empty(0, dtype=np.float64)
        hessian.sum_duplicates()
        constraint_matrix.sum_duplicates()
        return SparseQuadraticProblem(
            hessian=hessian,
            gradient=self._gradient,
            constraint_matrix=constraint_matrix,
            lower=lower,
            upper=upper,
            metadata={} if metadata is None else metadata,
        )
