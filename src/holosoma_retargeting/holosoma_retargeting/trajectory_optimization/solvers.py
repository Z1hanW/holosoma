"""Sparse CPU and GPU solvers for trajectory QPs."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable

import numpy as np
from scipy import sparse as sp

from .problem import SparseQuadraticProblem


@dataclass(frozen=True)
class SolveResult:
    solution: np.ndarray
    status: str
    iterations: int
    solve_time_s: float
    objective: float
    max_constraint_violation: float
    backend: str
    diagnostics: dict[str, float | int | str] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status in {"optimal", "solved", "solved inaccurate"}


@dataclass(frozen=True)
class TorchADMMSettings:
    max_iterations: int = 2000
    absolute_tolerance: float = 1e-5
    relative_tolerance: float = 1e-5
    rho: float = 1.0
    sigma: float = 1e-6
    pcg_max_iterations: int = 500
    pcg_relative_tolerance: float = 1e-7
    adaptive_rho_interval: int = 25
    verbose: bool = False


def _torch_sparse(matrix: sp.spmatrix, *, device: str, dtype):
    import torch

    coo = sp.coo_matrix(matrix)
    indices = torch.as_tensor(
        np.vstack((coo.row, coo.col)),
        dtype=torch.int64,
        device=device,
    )
    values = torch.as_tensor(coo.data, dtype=dtype, device=device)
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=coo.shape,
            dtype=dtype,
            device=device,
        ).coalesce()


def _sparse_matvec(matrix, vector):
    import torch

    return torch.sparse.mm(matrix, vector.reshape(-1, 1)).reshape(-1)


def _pcg(
    operator: Callable,
    rhs,
    initial,
    inverse_diagonal,
    *,
    max_iterations: int,
    relative_tolerance: float,
):
    import torch

    x = initial.clone()
    residual = rhs - operator(x)
    rhs_norm = float(torch.linalg.vector_norm(rhs).item())
    threshold = max(1e-12, relative_tolerance * max(rhs_norm, 1.0))
    residual_norm = float(torch.linalg.vector_norm(residual).item())
    if residual_norm <= threshold:
        return x, 0, residual_norm
    preconditioned = inverse_diagonal * residual
    direction = preconditioned.clone()
    residual_dot = torch.dot(residual, preconditioned)
    completed = 0
    for iteration in range(1, max_iterations + 1):
        operator_direction = operator(direction)
        denominator = torch.dot(direction, operator_direction)
        if float(denominator.item()) <= 1e-30:
            break
        alpha = residual_dot / denominator
        x = x + alpha * direction
        residual = residual - alpha * operator_direction
        residual_norm = float(torch.linalg.vector_norm(residual).item())
        completed = iteration
        if residual_norm <= threshold:
            break
        preconditioned = inverse_diagonal * residual
        next_residual_dot = torch.dot(residual, preconditioned)
        beta = next_residual_dot / residual_dot
        direction = preconditioned + beta * direction
        residual_dot = next_residual_dot
    return x, completed, residual_norm


class TorchSparseADMMSolver:
    """OSQP-style ADMM with matrix-free sparse PCG x-updates.

    Both CPU and CUDA paths use sparse ``H @ x``, ``A @ x``, and
    ``A.T @ y`` products. ``A.T @ A`` is never materialized.
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        settings: TorchADMMSettings | None = None,
    ) -> None:
        self.device = device
        self.settings = TorchADMMSettings() if settings is None else settings

    def _resolve_device(self) -> str:
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA solver requested but torch.cuda.is_available() is false")
        return self.device

    def solve(
        self,
        problem: SparseQuadraticProblem,
        warm_start: np.ndarray | None = None,
    ) -> SolveResult:
        import torch

        settings = self.settings
        device = self._resolve_device()
        dtype = torch.float64
        started = time.perf_counter()
        hessian = _torch_sparse(problem.hessian, device=device, dtype=dtype)
        constraints = _torch_sparse(problem.constraint_matrix, device=device, dtype=dtype)
        constraints_t = constraints.transpose(0, 1).coalesce()
        gradient = torch.as_tensor(problem.gradient, dtype=dtype, device=device)
        lower = torch.as_tensor(problem.lower, dtype=dtype, device=device)
        upper = torch.as_tensor(problem.upper, dtype=dtype, device=device)
        if warm_start is None:
            x = torch.zeros(problem.variable_count, dtype=dtype, device=device)
        else:
            warm_start = np.asarray(warm_start, dtype=np.float64).reshape(-1)
            if warm_start.shape != (problem.variable_count,):
                raise ValueError("warm_start shape disagrees with problem")
            x = torch.as_tensor(warm_start, dtype=dtype, device=device).clone()

        hessian_diagonal = torch.as_tensor(
            np.asarray(problem.hessian.diagonal()).reshape(-1),
            dtype=dtype,
            device=device,
        )
        column_square_sum = torch.as_tensor(
            np.asarray(problem.constraint_matrix.power(2).sum(axis=0)).reshape(-1),
            dtype=dtype,
            device=device,
        )
        rho = float(settings.rho)
        sigma = float(settings.sigma)
        if rho <= 0.0 or sigma <= 0.0:
            raise ValueError("rho and sigma must be positive")

        if problem.constraint_count:
            ax = _sparse_matvec(constraints, x)
            z = torch.clamp(ax, min=lower, max=upper)
            scaled_dual = torch.zeros_like(z)
        else:
            ax = z = scaled_dual = torch.empty(0, dtype=dtype, device=device)

        status = "maximum iterations reached"
        primal_norm = dual_norm = np.inf
        total_pcg_iterations = 0
        completed_iterations = 0
        for iteration in range(1, settings.max_iterations + 1):
            previous_x = x.clone()
            previous_z = z.clone()

            def normal_operator(value):
                result = _sparse_matvec(hessian, value) + sigma * value
                if problem.constraint_count:
                    av = _sparse_matvec(constraints, value)
                    result = result + rho * _sparse_matvec(constraints_t, av)
                return result

            rhs = -gradient + sigma * previous_x
            if problem.constraint_count:
                rhs = rhs + rho * _sparse_matvec(
                    constraints_t, z - scaled_dual
                )
            inverse_diagonal = torch.reciprocal(
                torch.clamp(
                    hessian_diagonal + rho * column_square_sum + sigma,
                    min=1e-12,
                )
            )
            x, pcg_iterations, pcg_residual = _pcg(
                normal_operator,
                rhs,
                previous_x,
                inverse_diagonal,
                max_iterations=settings.pcg_max_iterations,
                relative_tolerance=settings.pcg_relative_tolerance,
            )
            total_pcg_iterations += pcg_iterations
            completed_iterations = iteration

            if not problem.constraint_count:
                stationarity = _sparse_matvec(hessian, x) + gradient
                primal_norm = 0.0
                dual_norm = float(torch.linalg.vector_norm(stationarity, ord=np.inf).item())
                threshold = settings.absolute_tolerance
                threshold += settings.relative_tolerance * max(
                    float(torch.linalg.vector_norm(gradient, ord=np.inf).item()),
                    float(
                        torch.linalg.vector_norm(
                            _sparse_matvec(hessian, x), ord=np.inf
                        ).item()
                    ),
                )
                if dual_norm <= threshold:
                    status = "optimal"
                    break
                continue

            ax = _sparse_matvec(constraints, x)
            z = torch.clamp(ax + scaled_dual, min=lower, max=upper)
            scaled_dual = scaled_dual + ax - z
            primal = ax - z
            dual = rho * _sparse_matvec(constraints_t, z - previous_z)
            primal_norm = float(torch.linalg.vector_norm(primal, ord=np.inf).item())
            dual_norm = float(torch.linalg.vector_norm(dual, ord=np.inf).item())
            primal_tolerance = settings.absolute_tolerance
            primal_tolerance += settings.relative_tolerance * max(
                float(torch.linalg.vector_norm(ax, ord=np.inf).item()),
                float(torch.linalg.vector_norm(z, ord=np.inf).item()),
            )
            dual_reference = rho * _sparse_matvec(constraints_t, scaled_dual)
            dual_tolerance = settings.absolute_tolerance
            dual_tolerance += settings.relative_tolerance * float(
                torch.linalg.vector_norm(dual_reference, ord=np.inf).item()
            )
            if primal_norm <= primal_tolerance and dual_norm <= dual_tolerance:
                status = "optimal"
                break

            interval = settings.adaptive_rho_interval
            if interval > 0 and iteration % interval == 0:
                old_rho = rho
                if primal_norm > 10.0 * max(dual_norm, 1e-16):
                    rho *= 2.0
                elif dual_norm > 10.0 * max(primal_norm, 1e-16):
                    rho *= 0.5
                if rho != old_rho:
                    scaled_dual *= old_rho / rho
            if settings.verbose and (iteration == 1 or iteration % 100 == 0):
                print(
                    f"[torch-admm] iter={iteration} primal={primal_norm:.3e} "
                    f"dual={dual_norm:.3e} rho={rho:.3e} pcg={pcg_iterations} "
                    f"pcg_residual={pcg_residual:.3e}"
                )

        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        solution = x.detach().cpu().numpy()
        elapsed = time.perf_counter() - started
        return SolveResult(
            solution=solution,
            status=status,
            iterations=completed_iterations,
            solve_time_s=elapsed,
            objective=problem.objective(solution),
            max_constraint_violation=problem.max_constraint_violation(solution),
            backend=f"torch-sparse-admm:{device}",
            diagnostics={
                "primal_residual_inf": primal_norm,
                "dual_residual_inf": dual_norm,
                "rho": rho,
                "pcg_iterations": total_pcg_iterations,
                "hessian_nnz": problem.hessian.nnz,
                "constraint_nnz": problem.constraint_matrix.nnz,
            },
        )


class OSQPSolver:
    """Sparse CPU reference backend."""

    def __init__(
        self,
        *,
        absolute_tolerance: float = 1e-6,
        relative_tolerance: float = 1e-6,
        max_iterations: int = 20000,
        polish: bool = True,
        verbose: bool = False,
    ) -> None:
        self.absolute_tolerance = absolute_tolerance
        self.relative_tolerance = relative_tolerance
        self.max_iterations = max_iterations
        self.polish = polish
        self.verbose = verbose

    def solve(
        self,
        problem: SparseQuadraticProblem,
        warm_start: np.ndarray | None = None,
    ) -> SolveResult:
        import osqp

        solver = osqp.OSQP()
        started = time.perf_counter()
        solver.setup(
            P=sp.triu(problem.hessian, format="csc"),
            q=problem.gradient,
            A=problem.constraint_matrix.tocsc(),
            l=problem.lower,
            u=problem.upper,
            eps_abs=self.absolute_tolerance,
            eps_rel=self.relative_tolerance,
            max_iter=self.max_iterations,
            polishing=self.polish,
            verbose=self.verbose,
        )
        if warm_start is not None:
            solver.warm_start(x=np.asarray(warm_start, dtype=np.float64))
        raw = solver.solve()
        elapsed = time.perf_counter() - started
        if raw.x is None:
            solution = np.full(problem.variable_count, np.nan)
        else:
            solution = np.asarray(raw.x, dtype=np.float64)
        raw_status = str(raw.info.status).lower()
        status = "optimal" if raw_status == "solved" else raw_status
        objective = problem.objective(solution) if np.isfinite(solution).all() else np.inf
        violation = (
            problem.max_constraint_violation(solution)
            if np.isfinite(solution).all()
            else np.inf
        )
        return SolveResult(
            solution=solution,
            status=status,
            iterations=int(raw.info.iter),
            solve_time_s=elapsed,
            objective=objective,
            max_constraint_violation=violation,
            backend="osqp:cpu",
            diagnostics={
                "primal_residual": float(raw.info.prim_res),
                "dual_residual": float(raw.info.dual_res),
                "hessian_nnz": problem.hessian.nnz,
                "constraint_nnz": problem.constraint_matrix.nnz,
            },
        )


class AutoSparseSolver:
    """Route small QPs to OSQP and large QPs to CUDA with CPU fallback."""

    def __init__(
        self,
        *,
        gpu_minimum_variables: int = 20_000,
        gpu_settings: TorchADMMSettings | None = None,
        fallback_to_osqp: bool = True,
    ) -> None:
        if gpu_minimum_variables <= 0:
            raise ValueError("gpu_minimum_variables must be positive")
        self.gpu_minimum_variables = int(gpu_minimum_variables)
        self.gpu_settings = (
            TorchADMMSettings() if gpu_settings is None else gpu_settings
        )
        self.fallback_to_osqp = fallback_to_osqp

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return bool(torch.cuda.is_available())
        except ImportError:
            return False

    def _cpu_solver(self) -> OSQPSolver:
        return OSQPSolver(
            absolute_tolerance=self.gpu_settings.absolute_tolerance,
            relative_tolerance=self.gpu_settings.relative_tolerance,
            max_iterations=max(20_000, self.gpu_settings.max_iterations),
        )

    def solve(
        self,
        problem: SparseQuadraticProblem,
        warm_start: np.ndarray | None = None,
    ) -> SolveResult:
        use_gpu = (
            problem.variable_count >= self.gpu_minimum_variables
            and self._cuda_available()
        )
        if not use_gpu:
            reason = (
                "problem below GPU crossover threshold"
                if problem.variable_count < self.gpu_minimum_variables
                else "CUDA unavailable"
            )
            result = self._cpu_solver().solve(problem, warm_start)
            return SolveResult(
                **{
                    **result.__dict__,
                    "diagnostics": {
                        **result.diagnostics,
                        "auto_selection": reason,
                    },
                }
            )

        gpu_result = TorchSparseADMMSolver(
            device="cuda",
            settings=self.gpu_settings,
        ).solve(problem, warm_start)
        constraint_values = problem.constraint_matrix @ gpu_result.solution
        projected_values = np.minimum(
            np.maximum(constraint_values, problem.lower),
            problem.upper,
        )
        constraint_scale = max(
            1.0,
            float(np.max(np.abs(constraint_values), initial=0.0)),
            float(np.max(np.abs(projected_values), initial=0.0)),
        )
        maximum_allowed_violation = (
            self.gpu_settings.absolute_tolerance
            + self.gpu_settings.relative_tolerance
            * constraint_scale
        )
        gpu_accepted = (
            gpu_result.success
            and np.isfinite(gpu_result.solution).all()
            and gpu_result.max_constraint_violation
            <= 1.05 * maximum_allowed_violation
        )
        if gpu_accepted or not self.fallback_to_osqp:
            return SolveResult(
                **{
                    **gpu_result.__dict__,
                    "diagnostics": {
                        **gpu_result.diagnostics,
                        "auto_selection": "problem above GPU crossover threshold",
                    },
                }
            )

        cpu_result = self._cpu_solver().solve(problem, gpu_result.solution)
        return SolveResult(
            **{
                **cpu_result.__dict__,
                "diagnostics": {
                    **cpu_result.diagnostics,
                    "auto_selection": "GPU result rejected; fell back to OSQP",
                    "gpu_status": gpu_result.status,
                    "gpu_max_constraint_violation": (
                        gpu_result.max_constraint_violation
                    ),
                    "gpu_solve_time_s": gpu_result.solve_time_s,
                },
            }
        )
