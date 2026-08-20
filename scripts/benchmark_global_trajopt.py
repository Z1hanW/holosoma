#!/usr/bin/env python3
"""Benchmark sparse whole-trajectory QP backends on a reproducible problem."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import time

import numpy as np
import torch

from holosoma_retargeting.trajectory_optimization import (
    OSQPSolver,
    TorchADMMSettings,
    TorchSparseADMMSolver,
    WholeTrajectoryProblem,
    WholeTrajectorySpec,
    piecewise_linear_scale_basis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--scale-dim", type=int, default=6)
    parser.add_argument("--scale-knots", type=int, default=16)
    parser.add_argument("--tracking-rows", type=int, default=48)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--active-bounds", action="store_true")
    parser.add_argument("--absolute-tolerance", type=float, default=1e-6)
    parser.add_argument("--relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["osqp", "torch-cpu", "torch-cuda"],
        choices=["osqp", "torch-cpu", "torch-cuda"],
    )
    return parser.parse_args()


def build_problem(args: argparse.Namespace) -> WholeTrajectoryProblem:
    rng = np.random.default_rng(args.seed)
    frame_count = args.frames
    state_dimension = args.state_dim
    scale_dimension = args.scale_dim
    tracking_rows = args.tracking_rows
    basis = piecewise_linear_scale_basis(frame_count, args.scale_knots)
    state_time = np.linspace(0.0, 2.0 * np.pi, frame_count)
    frequencies = np.linspace(0.5, 2.5, state_dimension)
    true_states = np.sin(state_time[:, None] * frequencies[None, :]) * 0.15
    true_scale_knots = 1.0 + rng.normal(
        scale=0.12, size=(args.scale_knots, scale_dimension)
    )
    state_jacobian = rng.normal(
        scale=1.0 / np.sqrt(state_dimension),
        size=(frame_count, tracking_rows, state_dimension),
    )
    scale_jacobian = rng.normal(
        scale=0.4 / np.sqrt(scale_dimension),
        size=(frame_count, tracking_rows, scale_dimension),
    )
    frame_scales = basis @ true_scale_knots
    target = np.einsum("tmn,tn->tm", state_jacobian, true_states)
    target += np.einsum("tms,ts->tm", scale_jacobian, frame_scales)
    state_reference = true_states + rng.normal(scale=0.04, size=true_states.shape)
    state_limit = 0.10 if args.active_bounds else 2.0
    scale_lower = 0.95 if args.active_bounds else 0.5
    scale_upper = 1.05 if args.active_bounds else 1.5
    return WholeTrajectoryProblem(
        WholeTrajectorySpec(
            state_reference=state_reference,
            scale_reference=np.ones_like(true_scale_knots),
            scale_basis=basis,
            tracking_state_jacobian=state_jacobian,
            tracking_scale_jacobian=scale_jacobian,
            tracking_target=target,
            tracking_weight=20.0,
            state_prior_weight=1.0,
            scale_prior_weight=0.1,
            state_velocity_weight=0.2,
            state_acceleration_weight=0.05,
            scale_smoothness_weight=0.1,
            state_lower=-state_limit,
            state_upper=state_limit,
            scale_lower=scale_lower,
            scale_upper=scale_upper,
        )
    )


def main() -> int:
    args = parse_args()
    trajectory_problem = build_problem(args)
    build_started = time.perf_counter()
    problem = trajectory_problem.build()
    build_time_s = time.perf_counter() - build_started
    output = {
        "system": {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        "problem": {
            **problem.metadata,
            "variables": problem.variable_count,
            "constraints": problem.constraint_count,
            "hessian_nnz": problem.hessian.nnz,
            "constraint_nnz": problem.constraint_matrix.nnz,
            "active_bounds_requested": args.active_bounds,
            "build_time_s": build_time_s,
            "peak_rss_mb_after_build": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
        },
        "results": [],
    }
    for backend in args.backends:
        if backend == "torch-cuda" and not torch.cuda.is_available():
            output["results"].append(
                {"backend": backend, "status": "skipped: CUDA unavailable"}
            )
            continue
        if backend == "osqp":
            solver = OSQPSolver(
                absolute_tolerance=args.absolute_tolerance,
                relative_tolerance=args.relative_tolerance,
            )
        else:
            solver = TorchSparseADMMSolver(
                device="cuda" if backend == "torch-cuda" else "cpu",
                settings=TorchADMMSettings(
                    max_iterations=3000,
                    absolute_tolerance=args.absolute_tolerance,
                    relative_tolerance=args.relative_tolerance,
                    pcg_relative_tolerance=1e-7,
                ),
            )
        result = solver.solve(problem)
        output["results"].append(
            {
                "backend": result.backend,
                "status": result.status,
                "iterations": result.iterations,
                "solve_time_s": result.solve_time_s,
                "objective": result.objective,
                "max_constraint_violation": result.max_constraint_violation,
                "diagnostics": result.diagnostics,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
