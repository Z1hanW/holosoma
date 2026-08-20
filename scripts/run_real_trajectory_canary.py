#!/usr/bin/env python3
"""Run nonlinear whole-trajectory optimization on a real PRISM input."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from holosoma_retargeting.config_types.data_type import JOINTS_MAPPINGS
from holosoma_retargeting.trajectory_optimization import (
    AutoSparseSolver,
    MujocoInteractionTrajectoryOptimizer,
    MujocoInteractionTrajOptSettings,
    MujocoObjectFrameKinematics,
    MujocoTrajectoryCollision,
    MujocoTrajectoryDynamicsAuditor,
    OSQPSolver,
    TorchADMMSettings,
    TorchSparseADMMSolver,
    build_mocap_object_model,
    decode_object_poses,
    piecewise_linear_scale_basis,
)


def sequence_signature(sequence: str) -> tuple[str | None, str | None]:
    parts = sequence.split("_")
    object_class = parts[2] if len(parts) > 2 else None
    motion = next(
        (part for part in parts if len(part) > 1 and part[0] == "m" and part[1:].isdigit()),
        None,
    )
    return object_class, motion


def select_seed(
    candidates: np.ndarray,
    source_sequences: np.ndarray,
    *,
    sequence: str,
    kinematics: MujocoObjectFrameKinematics,
    collision: MujocoTrajectoryCollision,
    human_joints: np.ndarray,
    object_poses: np.ndarray,
    quaternion_order: str,
    pose_layout: str,
    contact_slice: slice,
    wrist_indices: np.ndarray,
    collision_candidates: int,
    minimum_collision_distance: float,
) -> tuple[int, np.ndarray, dict]:
    """Rank all seed trajectories, then collision-audit the best few."""

    target_class, target_motion = sequence_signature(sequence)
    frame_scales = np.ones((len(human_joints), 3), dtype=np.float64)
    ranked: list[dict] = []
    aligned_candidates: dict[int, np.ndarray] = {}
    for index, candidate in enumerate(candidates):
        qpos = np.asarray(candidate[:, : kinematics.model.nq], dtype=np.float64)
        if len(qpos) != len(human_joints) or not np.isfinite(qpos).all():
            continue
        aligned = kinematics.align_seed_root(qpos, human_joints)
        errors = kinematics.object_frame_error(
            aligned,
            human_joints,
            object_poses,
            frame_scales,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        mean_error = float(np.mean(errors))
        contact_wrist_error = float(
            np.mean(errors[contact_slice][:, wrist_indices])
        )
        source_class, source_motion = sequence_signature(
            str(source_sequences[index])
        )
        compatibility_penalty = 0.0
        if target_class != source_class:
            compatibility_penalty += 0.10
        if target_motion != source_motion:
            compatibility_penalty += 0.025
        tracking_score = (
            mean_error + 2.0 * contact_wrist_error + compatibility_penalty
        )
        aligned_candidates[index] = aligned
        ranked.append(
            {
                "index": index,
                "source_sequence": str(source_sequences[index]),
                "mean_keypoint_error_m": mean_error,
                "contact_wrist_error_m": contact_wrist_error,
                "compatibility_penalty": compatibility_penalty,
                "tracking_score": tracking_score,
            }
        )
    if not ranked:
        raise ValueError("seed bank has no finite frame-compatible candidates")

    ranked.sort(key=lambda item: item["tracking_score"])
    audit_count = min(max(collision_candidates, 1), len(ranked))
    for item in ranked[:audit_count]:
        audit = collision.audit(
            aligned_candidates[item["index"]],
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        ground_violation = float(
            np.max(
                np.maximum(
                    minimum_collision_distance
                    - audit.ground_minimum_distance,
                    0.0,
                )
            )
        )
        object_violation = float(
            np.max(
                np.maximum(
                    minimum_collision_distance
                    - audit.object_minimum_distance,
                    0.0,
                )
            )
        )
        item["ground_collision_violation_m"] = ground_violation
        item["object_collision_violation_m"] = object_violation
        item["selection_score"] = (
            item["tracking_score"]
            + 25.0 * ground_violation
            + 25.0 * object_violation
        )
    chosen = min(
        ranked[:audit_count],
        key=lambda item: item["selection_score"],
    )
    report = {
        "mode": "auto",
        "candidate_count": len(ranked),
        "collision_audited_candidates": audit_count,
        "chosen_index": chosen["index"],
        "chosen_source_sequence": chosen["source_sequence"],
        "top_candidates": ranked[:audit_count],
    }
    return chosen["index"], aligned_candidates[chosen["index"]], report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--seed-bank", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mesh", type=Path)
    parser.add_argument(
        "--seed-index",
        default="25",
        help="integer seed index or 'auto'",
    )
    parser.add_argument("--seed-collision-candidates", type=int, default=5)
    parser.add_argument("--scale-knots", type=int, default=8)
    parser.add_argument("--scale-lower", type=float, default=0.85)
    parser.add_argument("--scale-upper", type=float, default=1.15)
    parser.add_argument("--scale-prior-weight", type=float, default=50.0)
    parser.add_argument("--velocity-weight", type=float, default=2.0)
    parser.add_argument("--acceleration-weight", type=float, default=20.0)
    parser.add_argument(
        "--scale-mode",
        choices=("vector-field", "isotropic-field", "single-scalar"),
        default="vector-field",
    )
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--initial-trust-radius", type=float, default=0.25)
    parser.add_argument("--minimum-trust-radius", type=float, default=0.01)
    parser.add_argument("--maximum-trust-radius", type=float, default=0.6)
    parser.add_argument("--line-search-steps", type=int, default=8)
    parser.add_argument("--collision-activation-distance", type=float, default=0.08)
    parser.add_argument("--minimum-collision-distance", type=float, default=-1e-3)
    parser.add_argument("--collision-restoration-fraction", type=float, default=0.5)
    parser.add_argument(
        "--minimum-collision-restoration-fraction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--collision-linearization-margin",
        type=float,
        default=5e-4,
    )
    parser.add_argument(
        "--pose-layout",
        choices=("auto", "quat_pos", "pos_quat"),
        default="auto",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("auto", "osqp", "torch-cuda"),
        default="auto",
    )
    parser.add_argument("--gpu-minimum-variables", type=int, default=20_000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--skip-dynamics-audit", action="store_true")
    return parser.parse_args()


def trajectory_step_norms(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    result = np.empty(len(qpos) - 1, dtype=np.float64)
    velocity = np.empty(model.nv, dtype=np.float64)
    for frame in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(
            model,
            velocity,
            1.0,
            qpos[frame],
            qpos[frame + 1],
        )
        result[frame] = np.linalg.norm(velocity)
    return result


def trajectory_delta(
    model: mujoco.MjModel,
    reference: np.ndarray,
    value: np.ndarray,
) -> np.ndarray:
    result = np.empty((len(value), model.nv), dtype=np.float64)
    for frame in range(len(value)):
        mujoco.mj_differentiatePos(
            model,
            result[frame],
            1.0,
            reference[frame],
            value[frame],
        )
    return result


def evaluation_report(evaluation, minimum_distance: float) -> dict:
    audit = evaluation.collision_audit
    result = {
        "merit": evaluation.merit,
        "tracking_objective": evaluation.tracking_objective,
        "regularization_objective": evaluation.regularization_objective,
        "collision_objective": evaluation.collision_objective,
        "mean_keypoint_error_m": evaluation.mean_keypoint_error_m,
        "contact_wrist_error_m": evaluation.contact_wrist_error_m,
        "maximum_collision_violation_m": (
            evaluation.maximum_collision_violation_m
        ),
        **audit.summary(minimum_distance),
        "ground_limiting_geoms": dict(Counter(audit.ground_limiting_geoms)),
        "object_limiting_geoms": dict(Counter(audit.object_limiting_geoms)),
    }
    return result


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    seed_bank_path = args.seed_bank.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    with np.load(input_path, allow_pickle=True) as data:
        human_joints = np.asarray(data["human_joints"], dtype=np.float64)
        object_poses = np.asarray(data["object_poses"], dtype=np.float64)
        joint_names = np.asarray(data["joint_names"]).astype(str).tolist()
        quaternion_order = str(
            np.asarray(data["object_pose_quat_order"]).item()
        )
        contact_start = int(np.asarray(data["contact_start_idx"]).item())
        contact_end = int(np.asarray(data["contact_end_idx"]).item())
        sequence = str(np.asarray(data["sequence"]).item())
        input_mesh_path = Path(str(np.asarray(data["mesh_file"]).item()))
    mesh_path = (
        args.mesh.expanduser().resolve()
        if args.mesh is not None
        else input_mesh_path.expanduser().resolve()
    )
    transforms = decode_object_poses(
        object_poses,
        quaternion_order=quaternion_order,
        pose_layout=args.pose_layout,
    )
    pose_layout = transforms.pose_layout

    with np.load(seed_bank_path, allow_pickle=True) as data:
        candidates = np.asarray(data["qpos_candidates"], dtype=np.float64)
        source_sequences = np.asarray(data["source_sequences"]).astype(str)

    model = build_mocap_object_model(model_path, mesh_path)
    mapping = JOINTS_MAPPINGS[("seedance", "g1")]
    kinematics = MujocoObjectFrameKinematics(model, joint_names, mapping)
    collision = MujocoTrajectoryCollision(model)

    frame_count = len(human_joints)
    row_count = 3 * len(joint_names)
    tracking_weight = np.full((frame_count, row_count), 20.0)
    wrist_indices = np.asarray(
        [joint_names.index("L_Wrist"), joint_names.index("R_Wrist")],
        dtype=np.int32,
    )
    contact_slice = slice(contact_start, contact_end + 1)
    for wrist in wrist_indices:
        tracking_weight[
            contact_slice,
            3 * wrist : 3 * (wrist + 1),
        ] = 100.0
    if args.seed_index == "auto":
        seed_index, qpos_seed, seed_selection = select_seed(
            candidates,
            source_sequences,
            sequence=sequence,
            kinematics=kinematics,
            collision=collision,
            human_joints=human_joints,
            object_poses=object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
            contact_slice=contact_slice,
            wrist_indices=wrist_indices,
            collision_candidates=args.seed_collision_candidates,
            minimum_collision_distance=args.minimum_collision_distance,
        )
    else:
        try:
            seed_index = int(args.seed_index)
        except ValueError as exc:
            raise ValueError("--seed-index must be an integer or 'auto'") from exc
        if not 0 <= seed_index < len(candidates):
            raise ValueError("--seed-index is outside the seed bank")
        qpos_seed = candidates[seed_index, :, : model.nq].copy()
        if len(qpos_seed) != frame_count:
            raise ValueError("seed and real input frame counts disagree")
        qpos_seed = kinematics.align_seed_root(qpos_seed, human_joints)
        seed_selection = {
            "mode": "explicit",
            "candidate_count": len(candidates),
            "chosen_index": seed_index,
            "chosen_source_sequence": str(source_sequences[seed_index]),
        }
    scale_basis = piecewise_linear_scale_basis(
        frame_count,
        args.scale_knots,
    )
    solver_settings = TorchADMMSettings(
        max_iterations=4000,
        absolute_tolerance=1e-6,
        relative_tolerance=1e-6,
        pcg_relative_tolerance=1e-8,
    )
    if args.backend == "auto":
        solver = AutoSparseSolver(
            gpu_minimum_variables=args.gpu_minimum_variables,
            gpu_settings=solver_settings,
        )
    elif args.backend == "osqp":
        solver = OSQPSolver(
            absolute_tolerance=1e-6,
            relative_tolerance=1e-6,
        )
    else:
        solver = TorchSparseADMMSolver(
            device="cuda",
            settings=solver_settings,
        )
    settings = MujocoInteractionTrajOptSettings(
        max_iterations=args.max_iterations,
        initial_trust_radius=args.initial_trust_radius,
        minimum_trust_radius=args.minimum_trust_radius,
        maximum_trust_radius=args.maximum_trust_radius,
        line_search_steps=args.line_search_steps,
        scale_lower=args.scale_lower,
        scale_upper=args.scale_upper,
        scale_prior_weight=args.scale_prior_weight,
        state_velocity_weight=args.velocity_weight,
        state_acceleration_weight=args.acceleration_weight,
        scale_mode=args.scale_mode,
        collision_activation_distance=args.collision_activation_distance,
        minimum_collision_distance=args.minimum_collision_distance,
        collision_restoration_fraction=(
            args.collision_restoration_fraction
        ),
        minimum_collision_restoration_fraction=(
            args.minimum_collision_restoration_fraction
        ),
        collision_linearization_margin=(
            args.collision_linearization_margin
        ),
    )
    optimizer = MujocoInteractionTrajectoryOptimizer(
        model,
        kinematics,
        collision,
        solver,
        settings,
    )
    result = optimizer.optimize(
        qpos_seed,
        human_joints,
        object_poses,
        quaternion_order=quaternion_order,
        pose_layout=pose_layout,
        scale_basis=scale_basis,
        tracking_weight=tracking_weight,
        contact_slice=contact_slice,
        wrist_indices=wrist_indices,
    )

    deltas = trajectory_delta(model, qpos_seed, result.qpos)
    steps_before = trajectory_step_norms(model, qpos_seed)
    steps_after = trajectory_step_norms(model, result.qpos)
    initial_audit = result.initial_evaluation.collision_audit
    final_audit = result.final_evaluation.collision_audit
    dynamics_report = None
    dynamics_arrays = {}
    if not args.skip_dynamics_audit:
        dynamics_audit = MujocoTrajectoryDynamicsAuditor(
            model,
            fps=args.fps,
        ).audit(
            result.qpos,
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        dynamics_report = dynamics_audit.summary()
        dynamics_arrays = {
            "dynamics_qvel": dynamics_audit.qvel,
            "dynamics_qacc": dynamics_audit.qacc,
            "dynamics_controls": dynamics_audit.controls,
            "dynamics_force_residual_norm": (
                dynamics_audit.generalized_force_residual_norm
            ),
            "dynamics_control_violation": dynamics_audit.control_violation,
            "dynamics_rollout_qpos_defect_norm": (
                dynamics_audit.rollout_qpos_defect_norm
            ),
            "dynamics_rollout_qvel_defect_norm": (
                dynamics_audit.rollout_qvel_defect_norm
            ),
        }
    initial_contact_object_distance = initial_audit.object_minimum_distance[
        contact_slice
    ]
    final_contact_object_distance = final_audit.object_minimum_distance[
        contact_slice
    ]
    contact_proximity_fraction = float(
        np.mean(initial_contact_object_distance < 0.1)
    )
    interaction_geometry_status = (
        "active"
        if contact_proximity_fraction >= 0.05
        else "no_robot_object_proximity_in_input"
    )
    report = {
        "sequence": sequence,
        "input": str(input_path),
        "mesh": str(mesh_path),
        "seed_bank": str(seed_bank_path),
        "seed_index": seed_index,
        "seed_sequence": source_sequences[seed_index],
        "seed_selection": seed_selection,
        "frames": frame_count,
        "model_nq": model.nq,
        "model_nv": model.nv,
        "model_nu": model.nu,
        "pose_layout": pose_layout,
        "quaternion_order": quaternion_order,
        "collision_geometry": "MuJoCo convex collision representation",
        "interaction_geometry_status": interaction_geometry_status,
        "contact_object_proximity_fraction": contact_proximity_fraction,
        "collision_audit_distance_limit_m": 0.2,
        "contact_object_distance_initial_min_m": float(
            np.min(initial_contact_object_distance)
        ),
        "contact_object_distance_initial_median_m": float(
            np.median(initial_contact_object_distance)
        ),
        "contact_object_distance_final_min_m": float(
            np.min(final_contact_object_distance)
        ),
        "contact_object_distance_final_median_m": float(
            np.median(final_contact_object_distance)
        ),
        "optimizer_status": result.status,
        "settings": asdict(settings),
        "initial": evaluation_report(
            result.initial_evaluation,
            settings.minimum_collision_distance,
        ),
        "final": evaluation_report(
            result.final_evaluation,
            settings.minimum_collision_distance,
        ),
        "mean_tangent_step_before": float(np.mean(steps_before)),
        "mean_tangent_step_after": float(np.mean(steps_after)),
        "max_total_tangent_delta": float(np.max(np.abs(deltas))),
        "scale_min": np.min(result.frame_scales, axis=0).tolist(),
        "scale_max": np.max(result.frame_scales, axis=0).tolist(),
        "collision_feasible": (
            result.final_evaluation.maximum_collision_violation_m
            <= settings.collision_feasibility_tolerance
        ),
        "tracking_improved": (
            result.final_evaluation.tracking_objective
            < result.initial_evaluation.tracking_objective
        ),
        "dynamics_audit": dynamics_report,
        "iterations": [asdict(item) for item in result.iterations],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        qpos_seed=qpos_seed,
        qpos_optimized=result.qpos,
        tangent_deltas=deltas,
        scale_knots=result.scale_knots,
        frame_scales=result.frame_scales,
        initial_ground_distance_m=initial_audit.ground_minimum_distance,
        initial_object_distance_m=initial_audit.object_minimum_distance,
        final_ground_distance_m=final_audit.ground_minimum_distance,
        final_object_distance_m=final_audit.object_minimum_distance,
        source_input=np.asarray(str(input_path)),
        source_mesh=np.asarray(str(mesh_path)),
        **dynamics_arrays,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
