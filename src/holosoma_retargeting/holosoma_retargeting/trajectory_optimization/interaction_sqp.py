"""Nonlinear whole-trajectory interaction retargeting with sparse SQP."""

from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco
import numpy as np

from .builder import (
    LinearizedConstraint,
    WholeTrajectoryProblem,
    WholeTrajectorySpec,
)
from .mujoco_collision import (
    MujocoTrajectoryCollision,
    TrajectoryCollisionAudit,
)
from .mujoco_kinematics import MujocoObjectFrameKinematics
from .solvers import SolveResult
from .sqp import SparseQPSolver


@dataclass(frozen=True)
class MujocoInteractionTrajOptSettings:
    max_iterations: int = 8
    initial_trust_radius: float = 0.25
    minimum_trust_radius: float = 0.01
    maximum_trust_radius: float = 0.6
    line_search_steps: int = 8
    step_tolerance: float = 2e-4
    merit_tolerance: float = 1e-5
    state_prior_weight: float = 0.5
    state_velocity_weight: float = 2.0
    state_acceleration_weight: float = 20.0
    scale_prior_weight: float = 50.0
    scale_smoothness_weight: float = 20.0
    scale_lower: float = 0.85
    scale_upper: float = 1.15
    scale_mode: str = "vector-field"
    collision_activation_distance: float = 0.08
    minimum_collision_distance: float = -1e-3
    collision_restoration_fraction: float = 0.5
    minimum_collision_restoration_fraction: float = 0.05
    collision_linearization_margin: float = 5e-4
    collision_penalty: float = 1e7
    collision_feasibility_tolerance: float = 5e-5
    maximum_inaccurate_qp_violation: float = 1e-4


@dataclass(frozen=True)
class InteractionTrajOptEvaluation:
    merit: float
    tracking_objective: float
    regularization_objective: float
    collision_objective: float
    mean_keypoint_error_m: float
    contact_wrist_error_m: float
    maximum_collision_violation_m: float
    collision_audit: TrajectoryCollisionAudit


@dataclass(frozen=True)
class InteractionTrajOptIteration:
    iteration: int
    accepted: bool
    step_scale: float
    trust_radius: float
    state_step_inf: float
    state_bound_saturation_fraction: float
    scale_bound_saturation_fraction: float
    collision_rows: int
    ground_collision_rows: int
    object_collision_rows: int
    build_time_s: float
    qp_backend: str
    qp_status: str
    qp_iterations: int
    qp_solve_time_s: float
    qp_objective: float
    qp_max_constraint_violation: float
    merit_before: float
    merit_after: float
    mean_keypoint_error_m: float
    contact_wrist_error_m: float
    maximum_collision_violation_m: float
    collision_restoration_fraction: float


@dataclass(frozen=True)
class MujocoInteractionTrajOptResult:
    qpos: np.ndarray
    scale_knots: np.ndarray
    frame_scales: np.ndarray
    status: str
    initial_evaluation: InteractionTrajOptEvaluation
    final_evaluation: InteractionTrajOptEvaluation
    iterations: tuple[InteractionTrajOptIteration, ...]


def mujoco_tangent_bounds(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    trust_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build tangent trust bounds intersected with scalar joint limits."""

    if not np.isfinite(trust_radius) or trust_radius <= 0.0:
        raise ValueError("trust_radius must be positive and finite")
    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape (T, {model.nq})")
    lower = np.full((len(qpos), model.nv), -trust_radius, dtype=np.float64)
    upper = np.full((len(qpos), model.nv), trust_radius, dtype=np.float64)
    for joint_id in range(model.njnt):
        if not model.jnt_limited[joint_id]:
            continue
        if model.jnt_type[joint_id] not in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ):
            continue
        qpos_index = int(model.jnt_qposadr[joint_id])
        dof_index = int(model.jnt_dofadr[joint_id])
        lower[:, dof_index] = np.maximum(
            lower[:, dof_index],
            model.jnt_range[joint_id, 0] - qpos[:, qpos_index],
        )
        upper[:, dof_index] = np.minimum(
            upper[:, dof_index],
            model.jnt_range[joint_id, 1] - qpos[:, qpos_index],
        )
    if np.any(lower > upper + 1e-10):
        raise ValueError("current qpos lies outside a scalar joint limit")
    return lower, upper


class MujocoInteractionTrajectoryOptimizer:
    """Jointly optimize a MuJoCo trajectory and anisotropic scale field."""

    def __init__(
        self,
        model: mujoco.MjModel,
        kinematics: MujocoObjectFrameKinematics,
        collision: MujocoTrajectoryCollision,
        solver: SparseQPSolver,
        settings: MujocoInteractionTrajOptSettings | None = None,
    ) -> None:
        self.model = model
        self.kinematics = kinematics
        self.collision = collision
        self.solver = solver
        self.settings = (
            MujocoInteractionTrajOptSettings()
            if settings is None
            else settings
        )
        self._validate_settings()

    def _validate_settings(self) -> None:
        settings = self.settings
        if settings.max_iterations <= 0 or settings.line_search_steps <= 0:
            raise ValueError("iteration counts must be positive")
        if not (
            0.0 < settings.minimum_trust_radius
            <= settings.initial_trust_radius
            <= settings.maximum_trust_radius
        ):
            raise ValueError("trust radii must be positive and ordered")
        if not (
            0.0
            < settings.minimum_collision_restoration_fraction
            <= settings.collision_restoration_fraction
            <= 1.0
        ):
            raise ValueError(
                "collision restoration fractions must be in (0, 1] and ordered"
            )
        if (
            not np.isfinite(settings.collision_linearization_margin)
            or settings.collision_linearization_margin < 0.0
        ):
            raise ValueError(
                "collision_linearization_margin must be finite and non-negative"
            )
        if (
            not np.isfinite(settings.maximum_inaccurate_qp_violation)
            or settings.maximum_inaccurate_qp_violation < 0.0
        ):
            raise ValueError(
                "maximum_inaccurate_qp_violation must be finite and non-negative"
            )
        if settings.scale_lower >= settings.scale_upper:
            raise ValueError("scale_lower must be below scale_upper")
        if settings.scale_mode not in {
            "vector-field",
            "isotropic-field",
            "single-scalar",
        }:
            raise ValueError(
                "scale_mode must be 'vector-field', 'isotropic-field', "
                "or 'single-scalar'"
            )

    def _trajectory_correction(
        self,
        reference_qpos: np.ndarray,
        qpos: np.ndarray,
    ) -> np.ndarray:
        correction = np.empty((len(qpos), self.model.nv), dtype=np.float64)
        for frame in range(len(qpos)):
            mujoco.mj_differentiatePos(
                self.model,
                correction[frame],
                1.0,
                reference_qpos[frame],
                qpos[frame],
            )
        return correction

    def _trajectory_motion(
        self,
        qpos: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        velocity = np.empty(
            (max(0, len(qpos) - 1), self.model.nv),
            dtype=np.float64,
        )
        for frame in range(len(qpos) - 1):
            mujoco.mj_differentiatePos(
                self.model,
                velocity[frame],
                1.0,
                qpos[frame],
                qpos[frame + 1],
            )
        acceleration = np.diff(velocity, axis=0)
        return velocity, acceleration

    def _evaluate(
        self,
        qpos: np.ndarray,
        scale_knots: np.ndarray,
        *,
        reference_qpos: np.ndarray,
        human_points_world: np.ndarray,
        object_poses: np.ndarray,
        quaternion_order: str,
        pose_layout: str,
        scale_basis: np.ndarray,
        tracking_weight: np.ndarray,
        contact_slice: slice,
        wrist_indices: np.ndarray,
    ) -> InteractionTrajOptEvaluation:
        settings = self.settings
        frame_scales = scale_basis @ scale_knots
        residual = self.kinematics.object_frame_residual(
            qpos,
            human_points_world,
            object_poses,
            frame_scales,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        errors = np.linalg.norm(residual, axis=2)
        tracking_objective = float(
            np.sum(tracking_weight * residual.reshape(len(qpos), -1) ** 2)
        )
        correction = self._trajectory_correction(reference_qpos, qpos)
        velocity, acceleration = self._trajectory_motion(qpos)
        regularization = settings.state_prior_weight * float(
            np.sum(correction**2)
        )
        if len(velocity):
            regularization += settings.state_velocity_weight * float(
                np.sum(velocity**2)
            )
        if len(acceleration):
            regularization += settings.state_acceleration_weight * float(
                np.sum(acceleration**2)
            )
        regularization += settings.scale_prior_weight * float(
            np.sum((scale_knots - 1.0) ** 2)
        )
        if len(scale_knots) > 1:
            regularization += settings.scale_smoothness_weight * float(
                np.sum(np.diff(scale_knots, axis=0) ** 2)
            )
        audit = self.collision.audit(
            qpos,
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        ground_violation = np.maximum(
            settings.minimum_collision_distance
            - audit.ground_minimum_distance,
            0.0,
        )
        object_violation = np.maximum(
            settings.minimum_collision_distance
            - audit.object_minimum_distance,
            0.0,
        )
        maximum_collision_violation = float(
            max(
                np.max(ground_violation, initial=0.0),
                np.max(object_violation, initial=0.0),
            )
        )
        collision_objective = settings.collision_penalty * float(
            np.sum(ground_violation**2) + np.sum(object_violation**2)
        )
        return InteractionTrajOptEvaluation(
            merit=tracking_objective + regularization + collision_objective,
            tracking_objective=tracking_objective,
            regularization_objective=regularization,
            collision_objective=collision_objective,
            mean_keypoint_error_m=float(np.mean(errors)),
            contact_wrist_error_m=float(
                np.mean(errors[contact_slice][:, wrist_indices])
            ),
            maximum_collision_violation_m=maximum_collision_violation,
            collision_audit=audit,
        )

    def optimize(
        self,
        qpos_seed: np.ndarray,
        human_points_world: np.ndarray,
        object_poses: np.ndarray,
        *,
        quaternion_order: str,
        pose_layout: str,
        scale_basis: np.ndarray,
        tracking_weight: np.ndarray,
        contact_slice: slice,
        wrist_indices: np.ndarray,
    ) -> MujocoInteractionTrajOptResult:
        settings = self.settings
        qpos_seed = np.asarray(qpos_seed, dtype=np.float64)
        current_qpos = qpos_seed.copy()
        scale_basis = np.asarray(scale_basis, dtype=np.float64)
        scale_knots = np.ones((scale_basis.shape[1], 3), dtype=np.float64)
        current = self._evaluate(
            current_qpos,
            scale_knots,
            reference_qpos=qpos_seed,
            human_points_world=human_points_world,
            object_poses=object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
            scale_basis=scale_basis,
            tracking_weight=tracking_weight,
            contact_slice=contact_slice,
            wrist_indices=wrist_indices,
        )
        initial_evaluation = current
        trust_radius = settings.initial_trust_radius
        collision_restoration_fraction = (
            settings.collision_restoration_fraction
        )
        history: list[InteractionTrajOptIteration] = []
        status = "maximum iterations reached"

        for iteration in range(settings.max_iterations):
            linearization = self.kinematics.linearize(
                current_qpos,
                human_points_world,
                object_poses,
                quaternion_order=quaternion_order,
                pose_layout=pose_layout,
            )
            collision_linearization = self.collision.linearize(
                current_qpos,
                object_poses,
                quaternion_order=quaternion_order,
                pose_layout=pose_layout,
                activation_distance=settings.collision_activation_distance,
            )
            state_lower, state_upper = mujoco_tangent_bounds(
                self.model,
                current_qpos,
                trust_radius,
            )
            correction = self._trajectory_correction(qpos_seed, current_qpos)
            velocity, acceleration = self._trajectory_motion(current_qpos)
            extra_constraints = []
            linearized_minimum_distance = (
                settings.minimum_collision_distance
                + settings.collision_linearization_margin
            )
            for frame in np.unique(collision_linearization.frames):
                mask = collision_linearization.frames == frame
                distances = collision_linearization.distances[mask]
                restored_targets = distances + (
                    collision_restoration_fraction
                    * np.maximum(
                        linearized_minimum_distance - distances,
                        0.0,
                    )
                )
                required_distances = np.where(
                    distances < linearized_minimum_distance,
                    restored_targets,
                    linearized_minimum_distance,
                )
                extra_constraints.append(
                    LinearizedConstraint(
                        indices=np.arange(
                            int(frame) * self.model.nv,
                            (int(frame) + 1) * self.model.nv,
                        ),
                        matrix=collision_linearization.jacobians[mask],
                        lower=required_distances - distances,
                        upper=np.full(np.count_nonzero(mask), np.inf),
                    )
                )
            scale_start = current_qpos.shape[0] * self.model.nv
            if settings.scale_mode != "vector-field":
                for knot in range(len(scale_knots)):
                    indices = np.arange(
                        scale_start + 3 * knot,
                        scale_start + 3 * (knot + 1),
                    )
                    extra_constraints.append(
                        LinearizedConstraint(
                            indices=indices,
                            matrix=np.asarray(
                                [
                                    [1.0, -1.0, 0.0],
                                    [0.0, 1.0, -1.0],
                                ]
                            ),
                            lower=np.zeros(2),
                            upper=np.zeros(2),
                        )
                    )
            if settings.scale_mode == "single-scalar":
                first_indices = np.arange(scale_start, scale_start + 3)
                for knot in range(1, len(scale_knots)):
                    knot_indices = np.arange(
                        scale_start + 3 * knot,
                        scale_start + 3 * (knot + 1),
                    )
                    extra_constraints.append(
                        LinearizedConstraint(
                            indices=np.concatenate(
                                (first_indices, knot_indices)
                            ),
                            matrix=np.hstack(
                                (-np.eye(3), np.eye(3))
                            ),
                            lower=np.zeros(3),
                            upper=np.zeros(3),
                        )
                    )
            trajectory_problem = WholeTrajectoryProblem(
                WholeTrajectorySpec(
                    state_reference=-correction,
                    scale_reference=np.ones_like(scale_knots),
                    scale_basis=scale_basis,
                    tracking_state_jacobian=linearization.state_jacobian,
                    tracking_scale_jacobian=linearization.scale_jacobian,
                    tracking_target=linearization.target,
                    tracking_weight=tracking_weight,
                    state_prior_weight=settings.state_prior_weight,
                    scale_prior_weight=settings.scale_prior_weight,
                    state_velocity_weight=settings.state_velocity_weight,
                    state_acceleration_weight=settings.state_acceleration_weight,
                    state_velocity_target=-velocity,
                    state_acceleration_target=-acceleration,
                    scale_smoothness_weight=settings.scale_smoothness_weight,
                    state_lower=state_lower,
                    state_upper=state_upper,
                    scale_lower=settings.scale_lower,
                    scale_upper=settings.scale_upper,
                    extra_constraints=extra_constraints,
                )
            )
            build_started = time.perf_counter()
            problem = trajectory_problem.build()
            build_time = time.perf_counter() - build_started
            warm_start = np.concatenate(
                (
                    np.zeros(current_qpos.shape[0] * self.model.nv),
                    scale_knots.reshape(-1),
                )
            )
            qp_result: SolveResult = self.solver.solve(problem, warm_start)
            qp_usable = (
                qp_result.success
                or (
                    qp_result.status
                    in {"time limit reached", "maximum iterations reached"}
                    and np.isfinite(qp_result.solution).all()
                    and qp_result.max_constraint_violation
                    <= settings.maximum_inaccurate_qp_violation
                )
            )
            accepted = False
            accepted_scale = 0.0
            candidate_evaluation = current
            state_step_inf = 0.0
            state_saturation = 0.0
            scale_saturation = 0.0
            if qp_usable:
                unpacked = trajectory_problem.unpack(qp_result.solution)
                state_step_inf = float(np.max(np.abs(unpacked.states)))
                finite_bounds = np.isfinite(state_lower) & np.isfinite(state_upper)
                active_lower = np.isclose(
                    unpacked.states,
                    state_lower,
                    atol=1e-5,
                    rtol=0.0,
                )
                active_upper = np.isclose(
                    unpacked.states,
                    state_upper,
                    atol=1e-5,
                    rtol=0.0,
                )
                state_saturation = float(
                    np.mean((active_lower | active_upper)[finite_bounds])
                )
                scale_saturation = float(
                    np.mean(
                        np.isclose(
                            unpacked.scale_knots,
                            settings.scale_lower,
                            atol=1e-5,
                            rtol=0.0,
                        )
                        | np.isclose(
                            unpacked.scale_knots,
                            settings.scale_upper,
                            atol=1e-5,
                            rtol=0.0,
                        )
                    )
                )
                scale_step = unpacked.scale_knots - scale_knots
                minimum_improvement = max(
                    settings.merit_tolerance,
                    1e-8 * abs(current.merit),
                )
                allowed_collision_violation = max(
                    current.maximum_collision_violation_m + 1e-8,
                    settings.collision_feasibility_tolerance,
                )
                for line_step in range(settings.line_search_steps):
                    step_scale = 0.5**line_step
                    candidate_qpos = self.kinematics.retract(
                        current_qpos,
                        step_scale * unpacked.states,
                    )
                    candidate_scale_knots = (
                        scale_knots + step_scale * scale_step
                    )
                    trial = self._evaluate(
                        candidate_qpos,
                        candidate_scale_knots,
                        reference_qpos=qpos_seed,
                        human_points_world=human_points_world,
                        object_poses=object_poses,
                        quaternion_order=quaternion_order,
                        pose_layout=pose_layout,
                        scale_basis=scale_basis,
                        tracking_weight=tracking_weight,
                        contact_slice=contact_slice,
                        wrist_indices=wrist_indices,
                    )
                    if (
                        np.isfinite(trial.merit)
                        and trial.merit < current.merit - minimum_improvement
                        and trial.maximum_collision_violation_m
                        <= allowed_collision_violation
                    ):
                        accepted = True
                        accepted_scale = step_scale
                        candidate_evaluation = trial
                        current_qpos = candidate_qpos
                        scale_knots = candidate_scale_knots
                        break

            ground_rows = collision_linearization.kinds.count("ground")
            object_rows = collision_linearization.kinds.count("object")
            history.append(
                InteractionTrajOptIteration(
                    iteration=iteration,
                    accepted=accepted,
                    step_scale=accepted_scale,
                    trust_radius=trust_radius,
                    state_step_inf=state_step_inf,
                    state_bound_saturation_fraction=state_saturation,
                    scale_bound_saturation_fraction=scale_saturation,
                    collision_rows=len(collision_linearization.frames),
                    ground_collision_rows=ground_rows,
                    object_collision_rows=object_rows,
                    build_time_s=build_time,
                    qp_backend=qp_result.backend,
                    qp_status=qp_result.status,
                    qp_iterations=qp_result.iterations,
                    qp_solve_time_s=qp_result.solve_time_s,
                    qp_objective=qp_result.objective,
                    qp_max_constraint_violation=qp_result.max_constraint_violation,
                    merit_before=current.merit,
                    merit_after=candidate_evaluation.merit,
                    mean_keypoint_error_m=(
                        candidate_evaluation.mean_keypoint_error_m
                    ),
                    contact_wrist_error_m=(
                        candidate_evaluation.contact_wrist_error_m
                    ),
                    maximum_collision_violation_m=(
                        candidate_evaluation.maximum_collision_violation_m
                    ),
                    collision_restoration_fraction=(
                        collision_restoration_fraction
                    ),
                )
            )
            if not qp_usable:
                if (
                    current.maximum_collision_violation_m
                    > settings.collision_feasibility_tolerance
                    and trust_radius < settings.maximum_trust_radius
                ):
                    trust_radius = min(
                        settings.maximum_trust_radius,
                        trust_radius * 1.5,
                    )
                    continue
                if (
                    current.maximum_collision_violation_m
                    > settings.collision_feasibility_tolerance
                    and collision_restoration_fraction
                    > settings.minimum_collision_restoration_fraction
                ):
                    collision_restoration_fraction = max(
                        settings.minimum_collision_restoration_fraction,
                        0.5 * collision_restoration_fraction,
                    )
                    continue
                status = f"qp failed: {qp_result.status}"
                break
            if not accepted:
                trust_radius *= 0.5
                if trust_radius < settings.minimum_trust_radius:
                    status = "line search stalled"
                    break
                continue
            improvement = current.merit - candidate_evaluation.merit
            current = candidate_evaluation
            if (
                current.maximum_collision_violation_m
                < 0.05
                and collision_restoration_fraction
                < settings.collision_restoration_fraction
            ):
                collision_restoration_fraction = min(
                    settings.collision_restoration_fraction,
                    1.5 * collision_restoration_fraction,
                )
            if accepted_scale == 1.0:
                trust_radius = min(
                    settings.maximum_trust_radius,
                    trust_radius * 1.4,
                )
            else:
                trust_radius = max(
                    settings.minimum_trust_radius,
                    trust_radius * 0.75,
                )
            if (
                accepted_scale * state_step_inf <= settings.step_tolerance
                and current.maximum_collision_violation_m
                <= settings.collision_feasibility_tolerance
            ):
                status = "converged"
                break
            if (
                improvement <= settings.merit_tolerance
                and current.maximum_collision_violation_m
                <= settings.collision_feasibility_tolerance
            ):
                status = "converged"
                break
        return MujocoInteractionTrajOptResult(
            qpos=current_qpos,
            scale_knots=scale_knots,
            frame_scales=scale_basis @ scale_knots,
            status=status,
            initial_evaluation=initial_evaluation,
            final_evaluation=current,
            iterations=tuple(history),
        )
