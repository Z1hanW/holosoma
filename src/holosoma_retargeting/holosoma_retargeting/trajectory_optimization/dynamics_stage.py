"""Single-stage dynamics-constrained TrajOpt refinement for real trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import time

import mujoco
import numpy as np

from .builder import (
    LinearDynamics,
    LinearizedConstraint,
    WholeTrajectoryProblem,
    WholeTrajectorySpec,
)
from .interaction_sqp import mujoco_tangent_bounds
from .mujoco_collision import (
    MujocoTrajectoryCollision,
    TrajectoryCollisionAudit,
)
from .mujoco_dynamics import (
    MujocoDynamicsLinearizer,
    MujocoNominalTrajectory,
)
from .mujoco_kinematics import MujocoObjectFrameKinematics
from .object_pose import decode_object_poses
from .solvers import SolveResult
from .sqp import SparseQPSolver


@dataclass(frozen=True)
class MujocoDynamicsStageSettings:
    fps: float = 30.0
    qpos_trust_radius: float = 0.2
    qvel_trust_radius: float = 50.0
    qpos_prior_weight: float = 1.0
    qvel_prior_weight: float = 1e-3
    control_weight: float = 1e-3
    dynamics_position_weight: float = 100.0
    dynamics_velocity_weight: float = 0.1
    dynamics_activation_weight: float = 0.1
    dynamics_soft_formulation: str = "slack"
    max_transition_coefficient: float = 1_000.0
    scale_prior_weight: float = 50.0
    scale_smoothness_weight: float = 20.0
    scale_lower: float = 0.85
    scale_upper: float = 1.15
    collision_activation_distance: float = 0.08
    minimum_collision_distance: float = -1e-3
    collision_regression_tolerance: float = 1e-6
    maximum_collision_violation: float | None = None
    line_search_steps: int = 8
    minimum_dynamics_improvement: float = 1e-6
    allow_inaccurate_qp: bool = False
    maximum_inaccurate_qp_violation: float = 1.0
    dynamics_epsilon: float = 1e-5


@dataclass(frozen=True)
class MujocoDynamicsStageResult:
    nominal: MujocoNominalTrajectory
    scale_knots: np.ndarray
    frame_scales: np.ndarray
    qp_result: SolveResult
    variable_count: int
    constraint_count: int
    hessian_nnz: int
    constraint_nnz: int
    build_time_s: float
    linearization_time_s: float
    initial_defect_norm: np.ndarray
    qp_defect_norm: np.ndarray
    qp_dynamics_equality_violation: float
    final_defect_norm: np.ndarray
    transition_max_abs: float
    active_dynamics_mask: np.ndarray
    active_dynamics_transitions: int
    skipped_dynamics_transitions: int
    accepted_step_size: float
    line_search_trials: int
    line_search_step_sizes: tuple[float, ...]
    line_search_dynamics_means: tuple[float, ...]
    line_search_collision_violations: tuple[float, ...]
    initial_collision_violation: float
    final_collision_violation: float
    collision_violation_limit: float
    used_inaccurate_qp: bool
    collision_audit: TrajectoryCollisionAudit


class MujocoDynamicsTrajectoryOptimizer:
    """Refine qpos, qvel, controls, and scale under linearized dynamics."""

    def __init__(
        self,
        model: mujoco.MjModel,
        kinematics: MujocoObjectFrameKinematics,
        collision: MujocoTrajectoryCollision,
        solver: SparseQPSolver,
        settings: MujocoDynamicsStageSettings | None = None,
    ) -> None:
        self.model = model
        self.kinematics = kinematics
        self.collision = collision
        self.solver = solver
        self.settings = (
            MujocoDynamicsStageSettings()
            if settings is None
            else settings
        )

    @staticmethod
    def _maximum_collision_violation(
        audit: TrajectoryCollisionAudit,
        minimum_distance: float,
    ) -> float:
        return float(
            max(
                np.max(
                    np.maximum(
                        minimum_distance - audit.ground_minimum_distance,
                        0.0,
                    ),
                    initial=0.0,
                ),
                np.max(
                    np.maximum(
                        minimum_distance - audit.object_minimum_distance,
                        0.0,
                    ),
                    initial=0.0,
                ),
            )
        )

    def optimize(
        self,
        nominal: MujocoNominalTrajectory,
        scale_knots: np.ndarray,
        human_points_world: np.ndarray,
        object_poses: np.ndarray,
        *,
        quaternion_order: str,
        pose_layout: str,
        scale_basis: np.ndarray,
        tracking_weight: np.ndarray,
    ) -> MujocoDynamicsStageResult:
        settings = self.settings
        nominal.validate(self.model)
        frame_count = len(nominal.qpos)
        state_dimension = 2 * self.model.nv + self.model.na
        transforms = decode_object_poses(
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        kinematic_linearization = self.kinematics.linearize(
            nominal.qpos,
            human_points_world,
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        tracking_state_jacobian = np.zeros(
            (
                frame_count,
                kinematic_linearization.target.shape[1],
                state_dimension,
            ),
            dtype=np.float64,
        )
        tracking_state_jacobian[:, :, : self.model.nv] = (
            kinematic_linearization.state_jacobian
        )
        collision_linearization = self.collision.linearize(
            nominal.qpos,
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
            activation_distance=settings.collision_activation_distance,
        )

        original_timestep = float(self.model.opt.timestep)
        self.model.opt.timestep = 1.0 / settings.fps
        linearizer = MujocoDynamicsLinearizer(
            self.model,
            epsilon=settings.dynamics_epsilon,
            centered=False,
        )
        linearization_started = time.perf_counter()
        try:
            dynamics_linearization = linearizer.linearize(nominal)
        finally:
            self.model.opt.timestep = original_timestep
        linearization_time = time.perf_counter() - linearization_started
        transition_max_per_frame = np.max(
            np.abs(dynamics_linearization.dynamics.transition),
            axis=(1, 2),
        )
        dynamics_active = (
            transition_max_per_frame
            <= settings.max_transition_coefficient
        )
        filtered_dynamics = LinearDynamics(
            transition=dynamics_linearization.dynamics.transition,
            control=dynamics_linearization.dynamics.control,
            offset=dynamics_linearization.dynamics.offset,
            active=dynamics_active,
        )

        qpos_lower, qpos_upper = mujoco_tangent_bounds(
            self.model,
            nominal.qpos,
            settings.qpos_trust_radius,
        )
        state_lower = np.full(
            (frame_count, state_dimension),
            -settings.qvel_trust_radius,
            dtype=np.float64,
        )
        state_upper = np.full(
            (frame_count, state_dimension),
            settings.qvel_trust_radius,
            dtype=np.float64,
        )
        state_lower[:, : self.model.nv] = qpos_lower
        state_upper[:, : self.model.nv] = qpos_upper
        state_prior_weight = np.concatenate(
            (
                np.full(self.model.nv, settings.qpos_prior_weight),
                np.full(self.model.nv, settings.qvel_prior_weight),
                np.full(self.model.na, settings.qvel_prior_weight),
            )
        )

        extra_constraints = []
        for frame in np.unique(collision_linearization.frames):
            mask = collision_linearization.frames == frame
            matrix = np.zeros(
                (np.count_nonzero(mask), state_dimension),
                dtype=np.float64,
            )
            matrix[:, : self.model.nv] = (
                collision_linearization.jacobians[mask]
            )
            distances = collision_linearization.distances[mask]
            extra_constraints.append(
                LinearizedConstraint(
                    indices=np.arange(
                        int(frame) * state_dimension,
                        (int(frame) + 1) * state_dimension,
                    ),
                    matrix=matrix,
                    lower=settings.minimum_collision_distance - distances,
                    upper=np.full(len(distances), np.inf),
                )
            )

        control_lower = np.full_like(nominal.controls, -np.inf)
        control_upper = np.full_like(nominal.controls, np.inf)
        limited = np.asarray(self.model.actuator_ctrllimited, dtype=bool)
        control_lower[:, limited] = (
            self.model.actuator_ctrlrange[limited, 0]
            - nominal.controls[:, limited]
        )
        control_upper[:, limited] = (
            self.model.actuator_ctrlrange[limited, 1]
            - nominal.controls[:, limited]
        )
        trajectory_problem = WholeTrajectoryProblem(
            WholeTrajectorySpec(
                state_reference=np.zeros((frame_count, state_dimension)),
                scale_reference=np.asarray(scale_knots, dtype=np.float64),
                scale_basis=scale_basis,
                tracking_state_jacobian=tracking_state_jacobian,
                tracking_scale_jacobian=(
                    kinematic_linearization.scale_jacobian
                ),
                tracking_target=kinematic_linearization.target,
                tracking_weight=tracking_weight,
                state_prior_weight=state_prior_weight,
                scale_prior_weight=settings.scale_prior_weight,
                state_velocity_weight=0.0,
                state_acceleration_weight=0.0,
                scale_smoothness_weight=settings.scale_smoothness_weight,
                state_lower=state_lower,
                state_upper=state_upper,
                scale_lower=settings.scale_lower,
                scale_upper=settings.scale_upper,
                control_reference=np.zeros_like(nominal.controls),
                control_weight=settings.control_weight,
                control_lower=control_lower,
                control_upper=control_upper,
                dynamics=filtered_dynamics,
                dynamics_weight=np.concatenate(
                    (
                        np.full(
                            self.model.nv,
                            settings.dynamics_position_weight,
                        ),
                        np.full(
                            self.model.nv,
                            settings.dynamics_velocity_weight,
                        ),
                        np.full(
                            self.model.na,
                            settings.dynamics_activation_weight,
                        ),
                    )
                ),
                dynamics_soft_formulation=(
                    settings.dynamics_soft_formulation
                ),
                extra_constraints=extra_constraints,
            )
        )
        build_started = time.perf_counter()
        problem = trajectory_problem.build()
        build_time = time.perf_counter() - build_started
        warm_start = trajectory_problem.reference_vector()
        warm_start[: frame_count * state_dimension] = 0.0
        warm_start[trajectory_problem.layout.all_control_indices] = 0.0
        qp_result = self.solver.solve(problem, warm_start)
        inaccurate_qp_usable = (
            settings.allow_inaccurate_qp
            and qp_result.status
            in {"time limit reached", "maximum iterations reached"}
            and np.isfinite(qp_result.solution).all()
            and qp_result.max_constraint_violation
            <= settings.maximum_inaccurate_qp_violation
        )
        if not qp_result.success and not inaccurate_qp_usable:
            raise RuntimeError(
                "dynamics QP failed: "
                f"{qp_result.status}; diagnostics={qp_result.diagnostics}"
            )
        unpacked = trajectory_problem.unpack(qp_result.solution)
        qp_defect = (
            unpacked.states[1:]
            - np.einsum(
                "tij,tj->ti",
                dynamics_linearization.dynamics.transition,
                unpacked.states[:-1],
            )
            - np.einsum(
                "tij,tj->ti",
                dynamics_linearization.dynamics.control,
                unpacked.controls,
            )
            - dynamics_linearization.dynamics.offset
        )
        if unpacked.dynamics_slacks is None:
            dynamics_equality_violation = 0.0
        else:
            dynamics_equality_violation = float(
                np.max(
                    np.abs(
                        qp_defect - unpacked.dynamics_slacks
                    )[dynamics_active],
                    initial=0.0,
                )
            )
        initial_collision_audit = self.collision.audit(
            nominal.qpos,
            object_poses,
            quaternion_order=quaternion_order,
            pose_layout=pose_layout,
        )
        initial_collision_violation = self._maximum_collision_violation(
            initial_collision_audit,
            settings.minimum_collision_distance,
        )
        collision_violation_limit = (
            initial_collision_violation
            + settings.collision_regression_tolerance
        )
        if settings.maximum_collision_violation is not None:
            collision_violation_limit = min(
                collision_violation_limit,
                settings.maximum_collision_violation,
            )
        initial_defect_mean = float(
            np.mean(dynamics_linearization.defect_norms)
        )
        accepted_step_size = 0.0
        line_search_trials = 0
        line_search_step_sizes: list[float] = []
        line_search_dynamics_means: list[float] = []
        line_search_collision_violations: list[float] = []
        updated = nominal
        accepted_scale_knots = np.asarray(scale_knots, dtype=np.float64)
        final_defect = dynamics_linearization.dynamics.offset
        collision_audit = initial_collision_audit
        final_collision_violation = initial_collision_violation
        original_timestep = float(self.model.opt.timestep)
        self.model.opt.timestep = 1.0 / settings.fps
        try:
            for step_index in range(settings.line_search_steps):
                line_search_trials += 1
                step_size = 0.5**step_index
                candidate_control_deltas = (
                    step_size * unpacked.controls
                )
                candidate_controls = (
                    nominal.controls + candidate_control_deltas
                )
                candidate_controls[:, limited] = np.clip(
                    candidate_controls[:, limited],
                    self.model.actuator_ctrlrange[limited, 0],
                    self.model.actuator_ctrlrange[limited, 1],
                )
                candidate = linearizer.apply_deltas(
                    nominal,
                    step_size * unpacked.states,
                    candidate_controls - nominal.controls,
                )
                candidate_defect = linearizer.rollout_defects(candidate)
                candidate_collision = self.collision.audit(
                    candidate.qpos,
                    object_poses,
                    quaternion_order=quaternion_order,
                    pose_layout=pose_layout,
                )
                candidate_collision_violation = (
                    self._maximum_collision_violation(
                        candidate_collision,
                        settings.minimum_collision_distance,
                    )
                )
                candidate_dynamics_mean = float(
                    np.mean(np.linalg.norm(candidate_defect, axis=1))
                )
                line_search_step_sizes.append(step_size)
                line_search_dynamics_means.append(candidate_dynamics_mean)
                line_search_collision_violations.append(
                    candidate_collision_violation
                )
                dynamics_improved = (
                    candidate_dynamics_mean
                    <= initial_defect_mean
                    - settings.minimum_dynamics_improvement
                )
                collision_preserved = (
                    candidate_collision_violation
                    <= collision_violation_limit
                )
                if dynamics_improved and collision_preserved:
                    accepted_step_size = step_size
                    updated = candidate
                    final_defect = candidate_defect
                    collision_audit = candidate_collision
                    final_collision_violation = (
                        candidate_collision_violation
                    )
                    accepted_scale_knots = (
                        np.asarray(scale_knots, dtype=np.float64)
                        + step_size
                        * (
                            unpacked.scale_knots
                            - np.asarray(scale_knots, dtype=np.float64)
                        )
                    )
                    break
        finally:
            self.model.opt.timestep = original_timestep
        return MujocoDynamicsStageResult(
            nominal=updated,
            scale_knots=accepted_scale_knots,
            frame_scales=scale_basis @ accepted_scale_knots,
            qp_result=qp_result,
            variable_count=problem.variable_count,
            constraint_count=problem.constraint_count,
            hessian_nnz=problem.hessian.nnz,
            constraint_nnz=problem.constraint_matrix.nnz,
            build_time_s=build_time,
            linearization_time_s=linearization_time,
            initial_defect_norm=dynamics_linearization.defect_norms,
            qp_defect_norm=np.linalg.norm(qp_defect, axis=1),
            qp_dynamics_equality_violation=(
                dynamics_equality_violation
            ),
            final_defect_norm=np.linalg.norm(final_defect, axis=1),
            transition_max_abs=float(
                np.max(
                    np.abs(
                        dynamics_linearization.dynamics.transition
                    ),
                    initial=0.0,
                )
            ),
            active_dynamics_mask=dynamics_active,
            active_dynamics_transitions=int(
                np.count_nonzero(dynamics_active)
            ),
            skipped_dynamics_transitions=int(
                np.count_nonzero(~dynamics_active)
            ),
            accepted_step_size=accepted_step_size,
            line_search_trials=line_search_trials,
            line_search_step_sizes=tuple(line_search_step_sizes),
            line_search_dynamics_means=tuple(
                line_search_dynamics_means
            ),
            line_search_collision_violations=tuple(
                line_search_collision_violations
            ),
            initial_collision_violation=initial_collision_violation,
            final_collision_violation=final_collision_violation,
            collision_violation_limit=collision_violation_limit,
            used_inaccurate_qp=not qp_result.success,
            collision_audit=collision_audit,
        )
