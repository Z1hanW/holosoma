#!/usr/bin/env python3
"""Run a dynamics-constrained refinement of a geometric PRISM result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from holosoma_retargeting.config_types.data_type import JOINTS_MAPPINGS
from holosoma_retargeting.trajectory_optimization import (
    AutoSparseSolver,
    CuPySparseDirectADMMSolver,
    MujocoDynamicsStageSettings,
    MujocoDynamicsTrajectoryOptimizer,
    MujocoNominalTrajectory,
    MujocoObjectFrameKinematics,
    MujocoTrajectoryCollision,
    MujocoTrajectoryDynamicsAuditor,
    OSQPSolver,
    ProxQPSolver,
    TorchADMMSettings,
    TorchSparseADMMSolver,
    build_mocap_object_model,
    decode_object_poses,
    piecewise_linear_scale_basis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--geometric-result", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--qpos-trust-radius", type=float, default=0.2)
    parser.add_argument("--qvel-trust-radius", type=float, default=50.0)
    parser.add_argument("--dynamics-position-weight", type=float, default=100.0)
    parser.add_argument("--dynamics-velocity-weight", type=float, default=0.1)
    parser.add_argument("--dynamics-epsilon", type=float, default=1e-5)
    parser.add_argument(
        "--dynamics-formulation",
        choices=("slack", "penalty"),
        default="slack",
    )
    parser.add_argument(
        "--max-transition-coefficient",
        type=float,
        default=1_000.0,
    )
    parser.add_argument(
        "--backend",
        choices=(
            "auto",
            "osqp",
            "proxqp",
            "torch-cuda",
            "cupy-direct-cuda",
        ),
        default="auto",
    )
    parser.add_argument("--gpu-minimum-variables", type=int, default=20_000)
    parser.add_argument("--solver-rho", type=float, default=1.0)
    parser.add_argument("--solver-absolute-tolerance", type=float, default=1e-5)
    parser.add_argument("--solver-relative-tolerance", type=float, default=1e-5)
    parser.add_argument("--solver-relaxation", type=float, default=1.6)
    parser.add_argument("--solver-adaptive-rho-interval", type=int, default=25)
    parser.add_argument("--solver-time-limit", type=float, default=30.0)
    parser.add_argument("--solver-verbose", action="store_true")
    parser.add_argument(
        "--collision-regression-tolerance",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--maximum-collision-violation",
        type=float,
        default=None,
    )
    parser.add_argument("--line-search-steps", type=int, default=8)
    parser.add_argument(
        "--maximum-qvel-consistency",
        type=float,
        default=None,
    )
    parser.add_argument("--allow-inaccurate-qp", action="store_true")
    parser.add_argument(
        "--maximum-inaccurate-qp-violation",
        type=float,
        default=1.0,
    )
    parser.add_argument("--project-inaccurate-qp", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    result_path = args.geometric_result.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    with np.load(input_path, allow_pickle=True) as data:
        human_joints = np.asarray(data["human_joints"], dtype=np.float64)
        object_poses = np.asarray(data["object_poses"], dtype=np.float64)
        joint_names = np.asarray(data["joint_names"]).astype(str).tolist()
        quaternion_order = str(
            np.asarray(data["object_pose_quat_order"]).item()
        )
        mesh_path = Path(str(np.asarray(data["mesh_file"]).item())).resolve()
        contact_start = int(np.asarray(data["contact_start_idx"]).item())
        contact_end = int(np.asarray(data["contact_end_idx"]).item())
        sequence = str(np.asarray(data["sequence"]).item())
    transforms = decode_object_poses(
        object_poses,
        quaternion_order=quaternion_order,
        pose_layout="auto",
    )
    with np.load(result_path) as data:
        qpos = np.asarray(data["qpos_optimized"], dtype=np.float64)
        scale_knots = np.asarray(data["scale_knots"], dtype=np.float64)
        saved_qvel = (
            np.asarray(data["qvel"], dtype=np.float64)
            if "qvel" in data.files
            else None
        )
        saved_controls = (
            np.asarray(data["controls"], dtype=np.float64)
            if "controls" in data.files
            else None
        )

    model = build_mocap_object_model(model_path, mesh_path)
    mapping = JOINTS_MAPPINGS[("seedance", "g1")]
    kinematics = MujocoObjectFrameKinematics(model, joint_names, mapping)
    collision = MujocoTrajectoryCollision(model)
    initial_dynamics = MujocoTrajectoryDynamicsAuditor(
        model,
        fps=args.fps,
    ).audit(
        qpos,
        object_poses,
        quaternion_order=quaternion_order,
        pose_layout=transforms.pose_layout,
    )
    expected_qvel_shape = (len(qpos), model.nv)
    expected_control_shape = (len(qpos) - 1, model.nu)
    use_saved_nominal = (
        saved_qvel is not None
        and saved_controls is not None
        and saved_qvel.shape == expected_qvel_shape
        and saved_controls.shape == expected_control_shape
        and np.isfinite(saved_qvel).all()
        and np.isfinite(saved_controls).all()
    )
    if use_saved_nominal:
        nominal_qvel = saved_qvel.copy()
        controls = saved_controls.copy()
        nominal_state_source = "saved dynamics result"
    else:
        nominal_qvel = initial_dynamics.qvel
        controls = initial_dynamics.controls[:-1].copy()
        nominal_state_source = "qpos-derived inverse dynamics"
    limited = np.asarray(model.actuator_ctrllimited, dtype=bool)
    controls[:, limited] = np.clip(
        controls[:, limited],
        model.actuator_ctrlrange[limited, 0],
        model.actuator_ctrlrange[limited, 1],
    )
    mocap_positions = np.zeros((len(qpos), model.nmocap, 3))
    mocap_quaternions = np.zeros((len(qpos), model.nmocap, 4))
    mocap_quaternions[:, :, 0] = 1.0
    object_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "trajopt_object_geom",
    )
    object_body_id = int(model.geom_bodyid[object_geom_id])
    object_mocap_id = int(model.body_mocapid[object_body_id])
    mocap_positions[:, object_mocap_id] = transforms.positions
    mocap_quaternions[:, object_mocap_id] = transforms.quaternions_wxyz
    nominal = MujocoNominalTrajectory(
        qpos=qpos,
        qvel=nominal_qvel,
        controls=controls,
        mocap_positions=mocap_positions,
        mocap_quaternions=mocap_quaternions,
    )

    tracking_weight = np.full((len(qpos), 3 * len(joint_names)), 20.0)
    wrist_indices = [
        joint_names.index("L_Wrist"),
        joint_names.index("R_Wrist"),
    ]
    contact_slice = slice(contact_start, contact_end + 1)
    for wrist in wrist_indices:
        tracking_weight[
            contact_slice,
            3 * wrist : 3 * (wrist + 1),
        ] = 100.0
    scale_basis = piecewise_linear_scale_basis(
        len(qpos),
        len(scale_knots),
    )
    solver_settings = TorchADMMSettings(
        max_iterations=5000,
        absolute_tolerance=args.solver_absolute_tolerance,
        relative_tolerance=args.solver_relative_tolerance,
        rho=args.solver_rho,
        pcg_relative_tolerance=1e-7,
        adaptive_rho_interval=args.solver_adaptive_rho_interval,
        relaxation=args.solver_relaxation,
        max_solve_time_s=args.solver_time_limit,
        verbose=args.solver_verbose,
    )
    if args.backend == "auto":
        solver = AutoSparseSolver(
            gpu_minimum_variables=args.gpu_minimum_variables,
            gpu_settings=solver_settings,
        )
    elif args.backend == "osqp":
        solver = OSQPSolver(
            absolute_tolerance=1e-5,
            relative_tolerance=1e-5,
        )
    elif args.backend == "proxqp":
        solver = ProxQPSolver(
            absolute_tolerance=1e-5,
            relative_tolerance=1e-5,
        )
    elif args.backend == "cupy-direct-cuda":
        solver = CuPySparseDirectADMMSolver(
            settings=solver_settings,
        )
    else:
        solver = TorchSparseADMMSolver(
            device="cuda",
            settings=solver_settings,
        )
    settings = MujocoDynamicsStageSettings(
        fps=args.fps,
        qpos_trust_radius=args.qpos_trust_radius,
        qvel_trust_radius=args.qvel_trust_radius,
        dynamics_position_weight=args.dynamics_position_weight,
        dynamics_velocity_weight=args.dynamics_velocity_weight,
        dynamics_epsilon=args.dynamics_epsilon,
        dynamics_soft_formulation=args.dynamics_formulation,
        max_transition_coefficient=args.max_transition_coefficient,
        collision_regression_tolerance=(
            args.collision_regression_tolerance
        ),
        maximum_collision_violation=(
            args.maximum_collision_violation
        ),
        line_search_steps=args.line_search_steps,
        maximum_qvel_consistency=args.maximum_qvel_consistency,
        allow_inaccurate_qp=args.allow_inaccurate_qp,
        maximum_inaccurate_qp_violation=(
            args.maximum_inaccurate_qp_violation
        ),
        project_inaccurate_qp=args.project_inaccurate_qp,
    )
    result = MujocoDynamicsTrajectoryOptimizer(
        model,
        kinematics,
        collision,
        solver,
        settings,
    ).optimize(
        nominal,
        scale_knots,
        human_joints,
        object_poses,
        quaternion_order=quaternion_order,
        pose_layout=transforms.pose_layout,
        scale_basis=scale_basis,
        tracking_weight=tracking_weight,
    )

    errors = kinematics.object_frame_error(
        result.nominal.qpos,
        human_joints,
        object_poses,
        result.frame_scales,
        quaternion_order=quaternion_order,
        pose_layout=transforms.pose_layout,
    )
    collision_summary = result.collision_audit.summary(
        settings.minimum_collision_distance
    )
    final_qpos_dynamics = MujocoTrajectoryDynamicsAuditor(
        model,
        fps=args.fps,
    ).audit(
        result.nominal.qpos,
        object_poses,
        quaternion_order=quaternion_order,
        pose_layout=transforms.pose_layout,
    )
    qvel_consistency = np.linalg.norm(
        result.nominal.qvel - final_qpos_dynamics.qvel,
        axis=1,
    )
    report = {
        "sequence": sequence,
        "input": str(input_path),
        "geometric_result": str(result_path),
        "nominal_state_source": nominal_state_source,
        "frames": len(qpos),
        "state_dimension": 2 * model.nv + model.na,
        "control_dimension": model.nu,
        "variables": result.variable_count,
        "constraints": result.constraint_count,
        "hessian_nnz": result.hessian_nnz,
        "constraint_nnz": result.constraint_nnz,
        "linearization_time_s": result.linearization_time_s,
        "build_time_s": result.build_time_s,
        "qp": {
            "backend": result.qp_result.backend,
            "status": result.qp_result.status,
            "iterations": result.qp_result.iterations,
            "solve_time_s": result.qp_result.solve_time_s,
            "objective": result.qp_result.objective,
            "max_constraint_violation": (
                result.qp_result.max_constraint_violation
            ),
            "diagnostics": result.qp_result.diagnostics,
            "used_inaccurate_direction": result.used_inaccurate_qp,
        },
        "direction_projection": (
            None
            if result.direction_projection is None
            else {
                "backend": result.direction_projection.backend,
                "status": result.direction_projection.status,
                "iterations": result.direction_projection.iterations,
                "solve_time_s": result.direction_projection.solve_time_s,
                "max_constraint_violation": (
                    result.direction_projection.max_constraint_violation
                ),
                "diagnostics": result.direction_projection.diagnostics,
            }
        ),
        "initial_inverse_dynamics_audit": initial_dynamics.summary(),
        "final_qpos_inverse_dynamics_audit": final_qpos_dynamics.summary(),
        "initial_linearized_defect_mean": float(
            np.mean(result.initial_defect_norm)
        ),
        "initial_active_defect_mean": float(
            np.mean(
                result.initial_defect_norm[
                    result.active_dynamics_mask
                ]
            )
        ),
        "initial_linearized_defect_max": float(
            np.max(result.initial_defect_norm)
        ),
        "transition_max_abs": result.transition_max_abs,
        "active_dynamics_transitions": (
            result.active_dynamics_transitions
        ),
        "skipped_dynamics_transitions": (
            result.skipped_dynamics_transitions
        ),
        "qp_soft_defect_mean": float(np.mean(result.qp_defect_norm)),
        "qp_active_defect_mean": float(
            np.mean(
                result.qp_defect_norm[
                    result.active_dynamics_mask
                ]
            )
        ),
        "qp_soft_defect_max": float(np.max(result.qp_defect_norm)),
        "qp_dynamics_equality_violation": (
            result.qp_dynamics_equality_violation
        ),
        "accepted_step_size": result.accepted_step_size,
        "line_search_trials": result.line_search_trials,
        "line_search": [
            {
                "step_size": step_size,
                "dynamics_mean": dynamics_mean,
                "collision_violation_m": collision_violation,
                "qvel_consistency_max": qvel_consistency_max,
            }
            for (
                step_size,
                dynamics_mean,
                collision_violation,
                qvel_consistency_max,
            ) in zip(
                result.line_search_step_sizes,
                result.line_search_dynamics_means,
                result.line_search_collision_violations,
                result.line_search_qvel_consistency_maxes,
            )
        ],
        "initial_collision_violation_m": (
            result.initial_collision_violation
        ),
        "final_collision_violation_m": (
            result.final_collision_violation
        ),
        "collision_violation_limit_m": (
            result.collision_violation_limit
        ),
        "line_search_final_qvel_consistency_max": (
            result.final_qvel_consistency_max
        ),
        "final_nonlinear_defect_mean": float(
            np.mean(result.final_defect_norm)
        ),
        "final_active_defect_mean": float(
            np.mean(
                result.final_defect_norm[
                    result.active_dynamics_mask
                ]
            )
        ),
        "final_nonlinear_defect_max": float(
            np.max(result.final_defect_norm)
        ),
        "final_qvel_consistency_mean": float(
            np.mean(qvel_consistency)
        ),
        "final_qvel_consistency_max": float(
            np.max(qvel_consistency)
        ),
        "final_max_abs_control": float(
            np.max(np.abs(result.nominal.controls))
        ),
        "final_mean_keypoint_error_m": float(np.mean(errors)),
        "final_contact_wrist_error_m": float(
            np.mean(errors[contact_slice][:, wrist_indices])
        ),
        "collision": collision_summary,
        "scale_min": np.min(result.frame_scales, axis=0).tolist(),
        "scale_max": np.max(result.frame_scales, axis=0).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos=result.nominal.qpos,
        qpos_optimized=result.nominal.qpos,
        qvel=result.nominal.qvel,
        controls=result.nominal.controls,
        scale_knots=result.scale_knots,
        frame_scales=result.frame_scales,
        initial_defect_norm=result.initial_defect_norm,
        qp_defect_norm=result.qp_defect_norm,
        final_defect_norm=result.final_defect_norm,
        qpos_derived_qvel=final_qpos_dynamics.qvel,
        qvel_consistency_norm=qvel_consistency,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
