from __future__ import annotations

import numpy as np
import pytest
import mujoco

from holosoma_retargeting.trajectory_optimization import (
    AutoSparseSolver,
    CuPySparseDirectADMMSolver,
    LinearDynamics,
    MujocoObjectFrameKinematics,
    MujocoTrajectoryCollision,
    MujocoTrajectoryDynamicsAuditor,
    MujocoDynamicsLinearizer,
    MujocoNominalTrajectory,
    OSQPSolver,
    ProxQPSolver,
    SparseQuadraticBuilder,
    TorchADMMSettings,
    TorchSparseADMMSolver,
    WholeTrajectoryProblem,
    WholeTrajectorySpec,
    decode_object_poses,
    piecewise_linear_scale_basis,
)


def test_sparse_admm_solves_box_qp() -> None:
    builder = SparseQuadraticBuilder(2)
    builder.add_quadratic(
        np.arange(2),
        np.eye(2),
        np.array([-1.0, -2.0]),
    )
    builder.add_variable_bounds(
        np.arange(2),
        np.zeros(2),
        np.array([0.5, 10.0]),
    )
    problem = builder.build()
    result = TorchSparseADMMSolver(
        device="cpu",
        settings=TorchADMMSettings(
            max_iterations=1000,
            absolute_tolerance=1e-7,
            relative_tolerance=1e-7,
        ),
    ).solve(problem)
    assert result.success
    np.testing.assert_allclose(result.solution, [0.5, 2.0], atol=2e-5)
    assert result.max_constraint_violation < 2e-6


def test_cupy_direct_admm_solves_box_qp() -> None:
    cupy = pytest.importorskip("cupy")
    if cupy.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("CUDA is unavailable")
    builder = SparseQuadraticBuilder(2)
    builder.add_quadratic(
        np.arange(2),
        np.eye(2),
        np.array([-1.0, -2.0]),
    )
    builder.add_variable_bounds(
        np.arange(2),
        np.zeros(2),
        np.array([0.5, 10.0]),
    )
    result = CuPySparseDirectADMMSolver(
        settings=TorchADMMSettings(
            max_iterations=1000,
            absolute_tolerance=1e-7,
            relative_tolerance=1e-7,
        ),
    ).solve(builder.build())
    assert result.success
    np.testing.assert_allclose(result.solution, [0.5, 2.0], atol=2e-5)
    assert result.max_constraint_violation < 2e-6
    assert result.diagnostics["relaxation"] == 1.6
    assert result.diagnostics["refactorizations"] >= 1


def test_large_variable_bounds_are_assembled_sparsely() -> None:
    variable_count = 100_000
    builder = SparseQuadraticBuilder(variable_count)
    builder.add_variable_bounds(
        np.arange(variable_count),
        np.full(variable_count, -1.0),
        np.full(variable_count, 1.0),
    )
    problem = builder.build()
    assert problem.constraint_matrix.shape == (variable_count, variable_count)
    assert problem.constraint_matrix.nnz == variable_count


def test_proxqp_solves_equality_and_box_qp() -> None:
    builder = SparseQuadraticBuilder(2)
    builder.add_quadratic(
        np.arange(2),
        np.eye(2),
        np.array([-1.0, -2.0]),
    )
    builder.add_linear_constraint(
        np.arange(2),
        np.ones((1, 2)),
        np.ones(1),
        np.ones(1),
    )
    builder.add_variable_bounds(
        np.arange(2),
        np.zeros(2),
        np.full(2, 10.0),
    )
    result = ProxQPSolver().solve(builder.build())
    assert result.success
    np.testing.assert_allclose(result.solution, [0.0, 1.0], atol=2e-5)
    assert result.max_constraint_violation < 2e-5


def test_diagonal_quadratic_does_not_store_dense_zeros() -> None:
    variable_count = 1_000
    builder = SparseQuadraticBuilder(variable_count)
    builder.add_quadratic(
        np.arange(variable_count),
        np.eye(variable_count),
    )
    problem = builder.build(diagonal_regularization=0.0)
    assert problem.hessian.nnz == variable_count


def test_auto_solver_routes_small_problem_to_osqp() -> None:
    builder = SparseQuadraticBuilder(2)
    builder.add_quadratic(np.arange(2), np.eye(2), np.array([-1.0, -2.0]))
    problem = builder.build()
    result = AutoSparseSolver(gpu_minimum_variables=100).solve(problem)
    assert result.success
    assert result.backend == "osqp:cpu"
    assert result.diagnostics["auto_selection"] == (
        "problem below GPU crossover threshold"
    )


@pytest.mark.skipif(
    pytest.importorskip("torch").cuda.is_available() is False,
    reason="CUDA is unavailable",
)
def test_auto_solver_routes_large_problem_to_gpu() -> None:
    builder = SparseQuadraticBuilder(2)
    builder.add_quadratic(np.arange(2), np.eye(2), np.array([-1.0, -2.0]))
    builder.add_variable_bounds(
        np.arange(2),
        np.zeros(2),
        np.full(2, 10.0),
    )
    problem = builder.build()
    result = AutoSparseSolver(
        gpu_minimum_variables=1,
        gpu_settings=TorchADMMSettings(
            absolute_tolerance=1e-7,
            relative_tolerance=1e-7,
        ),
    ).solve(problem)
    assert result.success
    assert result.backend == "torch-sparse-admm:cuda"
    assert result.diagnostics["auto_selection"] == (
        "problem above GPU crossover threshold"
    )


def _joint_scale_problem() -> tuple[WholeTrajectoryProblem, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    frame_count = 18
    state_dimension = 3
    scale_dimension = 3
    knot_count = 4
    tracking_rows = 7
    basis = piecewise_linear_scale_basis(frame_count, knot_count)
    true_scale_knots = np.array(
        [
            [0.85, 1.05, 1.15],
            [0.95, 1.15, 1.05],
            [1.10, 0.95, 0.90],
            [1.20, 0.90, 1.00],
        ]
    )
    true_states = np.column_stack(
        (
            np.linspace(-0.3, 0.4, frame_count),
            0.2 * np.sin(np.linspace(0.0, np.pi, frame_count)),
            0.1 * np.cos(np.linspace(0.0, 2.0 * np.pi, frame_count)),
        )
    )
    state_jacobian = rng.normal(size=(frame_count, tracking_rows, state_dimension))
    scale_jacobian = rng.normal(size=(frame_count, tracking_rows, scale_dimension))
    frame_scales = basis @ true_scale_knots
    target = np.einsum("tmn,tn->tm", state_jacobian, true_states)
    target += np.einsum("tms,ts->tm", scale_jacobian, frame_scales)
    noisy_state_reference = true_states + rng.normal(scale=0.08, size=true_states.shape)
    spec = WholeTrajectorySpec(
        state_reference=noisy_state_reference,
        scale_reference=np.ones_like(true_scale_knots),
        scale_basis=basis,
        tracking_state_jacobian=state_jacobian,
        tracking_scale_jacobian=scale_jacobian,
        tracking_target=target,
        tracking_weight=100.0,
        state_prior_weight=2.0,
        scale_prior_weight=0.05,
        state_velocity_weight=0.05,
        state_acceleration_weight=0.01,
        scale_smoothness_weight=0.02,
        state_lower=-2.0,
        state_upper=2.0,
        scale_lower=0.5,
        scale_upper=1.5,
    )
    return WholeTrajectoryProblem(spec), true_states, true_scale_knots


def test_joint_trajectory_and_vector_scale_recovery() -> None:
    trajectory_problem, true_states, true_scale_knots = _joint_scale_problem()
    problem = trajectory_problem.build()
    result = OSQPSolver().solve(problem)
    assert result.success
    solution = trajectory_problem.unpack(result.solution)
    reference = trajectory_problem.unpack(trajectory_problem.reference_vector())
    assert np.linalg.norm(solution.states - true_states) < np.linalg.norm(
        reference.states - true_states
    )
    assert np.linalg.norm(solution.scale_knots - true_scale_knots) < np.linalg.norm(
        reference.scale_knots - true_scale_knots
    )
    assert result.max_constraint_violation < 1e-6


def test_dynamics_constraints_are_jointly_enforced() -> None:
    frame_count = 10
    dt = 0.1
    transition = np.broadcast_to(
        np.array([[1.0, dt], [0.0, 1.0]]),
        (frame_count - 1, 2, 2),
    ).copy()
    control = np.broadcast_to(
        np.array([[0.5 * dt**2], [dt]]),
        (frame_count - 1, 2, 1),
    ).copy()
    offset = np.zeros((frame_count - 1, 2))
    state_jacobian = np.zeros((frame_count, 1, 2))
    state_jacobian[:, 0, 0] = 1.0
    target = np.linspace(0.0, 1.0, frame_count)[:, None]
    trajectory_problem = WholeTrajectoryProblem(
        WholeTrajectorySpec(
            state_reference=np.zeros((frame_count, 2)),
            scale_reference=np.ones((1, 2)),
            scale_basis=np.ones((frame_count, 1)),
            tracking_state_jacobian=state_jacobian,
            tracking_scale_jacobian=np.zeros((frame_count, 1, 2)),
            tracking_target=target,
            control_reference=np.zeros((frame_count - 1, 1)),
            dynamics=LinearDynamics(transition, control, offset),
            initial_state=np.zeros(2),
            tracking_weight=100.0,
            state_prior_weight=1e-4,
            scale_prior_weight=1.0,
            control_weight=0.1,
            state_velocity_weight=0.0,
            state_acceleration_weight=0.0,
            scale_smoothness_weight=0.0,
            state_lower=-10.0,
            state_upper=10.0,
            control_lower=-20.0,
            control_upper=20.0,
        )
    )
    problem = trajectory_problem.build()
    result = OSQPSolver().solve(problem)
    assert result.success
    solution = trajectory_problem.unpack(result.solution)
    for frame in range(frame_count - 1):
        predicted = (
            transition[frame] @ solution.states[frame]
            + control[frame] @ solution.controls[frame]
        )
        np.testing.assert_allclose(solution.states[frame + 1], predicted, atol=2e-6)
    np.testing.assert_allclose(solution.states[0], 0.0, atol=2e-6)
    assert solution.states[-1, 0] > 0.9


def test_soft_dynamics_remain_feasible_with_conflicting_bounds() -> None:
    trajectory_problem = WholeTrajectoryProblem(
        WholeTrajectorySpec(
            state_reference=np.zeros((2, 1)),
            scale_reference=np.ones((1, 1)),
            scale_basis=np.ones((2, 1)),
            tracking_state_jacobian=np.zeros((2, 1, 1)),
            tracking_scale_jacobian=np.zeros((2, 1, 1)),
            tracking_target=np.zeros((2, 1)),
            control_reference=np.zeros((1, 1)),
            dynamics=LinearDynamics(
                transition=np.ones((1, 1, 1)),
                control=np.zeros((1, 1, 1)),
                offset=np.ones((1, 1)),
            ),
            dynamics_weight=10.0,
            state_lower=0.0,
            state_upper=0.0,
            state_velocity_weight=0.0,
            state_acceleration_weight=0.0,
            scale_smoothness_weight=0.0,
        )
    )
    result = OSQPSolver().solve(trajectory_problem.build())
    assert result.success
    solution = trajectory_problem.unpack(result.solution)
    np.testing.assert_allclose(solution.states, 0.0, atol=1e-7)
    np.testing.assert_allclose(
        solution.dynamics_slacks,
        -1.0,
        atol=1e-7,
    )


def test_penalty_dynamics_do_not_add_equality_rows() -> None:
    trajectory_problem = WholeTrajectoryProblem(
        WholeTrajectorySpec(
            state_reference=np.zeros((2, 1)),
            scale_reference=np.ones((1, 1)),
            scale_basis=np.ones((2, 1)),
            tracking_state_jacobian=np.zeros((2, 1, 1)),
            tracking_scale_jacobian=np.zeros((2, 1, 1)),
            tracking_target=np.zeros((2, 1)),
            control_reference=np.zeros((1, 1)),
            dynamics=LinearDynamics(
                transition=np.ones((1, 1, 1)),
                control=np.zeros((1, 1, 1)),
                offset=np.ones((1, 1)),
            ),
            dynamics_weight=10.0,
            dynamics_soft_formulation="penalty",
            state_lower=-np.inf,
            state_upper=np.inf,
            control_lower=-np.inf,
            control_upper=np.inf,
            state_velocity_weight=0.0,
            state_acceleration_weight=0.0,
            scale_smoothness_weight=0.0,
        )
    )
    problem = trajectory_problem.build()
    solution = trajectory_problem.unpack(
        OSQPSolver().solve(problem).solution
    )
    equality_rows = (
        np.isfinite(problem.lower)
        & np.isfinite(problem.upper)
        & np.isclose(problem.lower, problem.upper)
    )
    assert not np.any(equality_rows)
    assert solution.dynamics_slacks is None


def test_inactive_dynamics_transition_is_omitted() -> None:
    trajectory_problem = WholeTrajectoryProblem(
        WholeTrajectorySpec(
            state_reference=np.zeros((2, 1)),
            scale_reference=np.ones((1, 1)),
            scale_basis=np.ones((2, 1)),
            tracking_state_jacobian=np.zeros((2, 1, 1)),
            tracking_scale_jacobian=np.zeros((2, 1, 1)),
            tracking_target=np.zeros((2, 1)),
            control_reference=np.zeros((1, 1)),
            dynamics=LinearDynamics(
                transition=np.ones((1, 1, 1)),
                control=np.zeros((1, 1, 1)),
                offset=np.ones((1, 1)),
                active=np.zeros(1, dtype=bool),
            ),
            state_velocity_weight=0.0,
            state_acceleration_weight=0.0,
            scale_smoothness_weight=0.0,
        )
    )
    problem = trajectory_problem.build()
    assert problem.metadata["active_dynamics_transitions"] == 0


def test_mujoco_dynamics_linearization_uses_tangent_state() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.02"/>
          <worldbody>
            <body>
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <motor joint="slide" gear="1"/>
          </actuator>
        </mujoco>
        """
    )
    frame_count = 8
    controls = np.linspace(0.0, 1.0, frame_count - 1)[:, None]
    qpos = np.empty((frame_count, model.nq))
    qvel = np.empty((frame_count, model.nv))
    data = mujoco.MjData(model)
    for frame in range(frame_count):
        qpos[frame] = data.qpos
        qvel[frame] = data.qvel
        if frame < frame_count - 1:
            data.ctrl[:] = controls[frame]
            mujoco.mj_step(model, data)
    nominal = MujocoNominalTrajectory(qpos=qpos, qvel=qvel, controls=controls)
    linearizer = MujocoDynamicsLinearizer(model)
    result = linearizer.linearize(nominal)
    assert result.dynamics.transition.shape == (frame_count - 1, 2, 2)
    assert result.dynamics.control.shape == (frame_count - 1, 2, 1)
    assert np.max(result.defect_norms) < 1e-10
    np.testing.assert_allclose(
        linearizer.rollout_defects(nominal),
        result.dynamics.offset,
        atol=1e-12,
    )

    state_deltas = np.zeros((frame_count, 2))
    state_deltas[:, 0] = 0.1
    updated = linearizer.apply_deltas(
        nominal,
        state_deltas,
        np.zeros_like(controls),
    )
    np.testing.assert_allclose(updated.qpos[:, 0], nominal.qpos[:, 0] + 0.1)


def test_object_pose_layout_and_kinematic_jacobian() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="pelvis">
              <freejoint/>
              <geom type="sphere" size="0.05"/>
              <body name="wrist" pos="0.3 0 0">
                <geom type="sphere" size="0.03"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    qpos = np.array([[0.1, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0]])
    human = np.array([[[0.1, -0.2, 0.8], [0.45, -0.2, 0.8]]])
    object_poses = np.array([[0.0, 0.0, 0.0, 1.0, -0.3, 0.2, 0.1]])
    transforms = decode_object_poses(
        object_poses,
        quaternion_order="xyzw",
    )
    assert transforms.pose_layout == "quat_pos"
    np.testing.assert_allclose(transforms.positions[0], [-0.3, 0.2, 0.1])

    kinematics = MujocoObjectFrameKinematics(
        model,
        ["Pelvis", "L_Wrist"],
        {"Pelvis": "pelvis", "L_Wrist": "wrist"},
    )
    linearization = kinematics.linearize(
        qpos,
        human,
        object_poses,
        quaternion_order="xyzw",
    )
    direction = np.array([[0.2, -0.1, 0.05, 0.0, 0.0, 0.3]])
    epsilon = 1e-7
    perturbed = kinematics.retract(qpos, epsilon * direction)
    finite_difference = (
        kinematics.mapped_points(perturbed) - kinematics.mapped_points(qpos)
    ) / epsilon
    predicted = np.einsum(
        "tjdn,tn->tjd",
        linearization.state_jacobian.reshape(1, 2, 3, model.nv),
        direction,
    )
    np.testing.assert_allclose(predicted, finite_difference, atol=2e-7)

    qp_residual = (
        np.einsum(
            "tms,ts->tm",
            linearization.scale_jacobian,
            np.ones((1, 3)),
        )
        - linearization.target
    ).reshape(1, 2, 3)
    expected_residual = (
        linearization.robot_points_object
        - linearization.human_points_object
    )
    np.testing.assert_allclose(qp_residual, expected_residual, atol=1e-12)


def test_mujoco_collision_audit_and_linearization() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="ground" type="plane" size="2 2 0.1"/>
            <body name="robot">
              <freejoint/>
              <geom name="robot_ball" type="sphere" size="0.1"/>
            </body>
            <body name="trajopt_object" mocap="true">
              <geom name="trajopt_object_geom" type="sphere" size="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    qpos = np.array([[0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]])
    object_poses = np.array([[0.0, 0.0, 0.0, 1.0, 0.3, 0.0, 0.1]])
    collision = MujocoTrajectoryCollision(model)
    audit = collision.audit(
        qpos,
        object_poses,
        quaternion_order="xyzw",
    )
    np.testing.assert_allclose(audit.ground_minimum_distance, 0.0, atol=1e-12)
    np.testing.assert_allclose(audit.object_minimum_distance, 0.1, atol=1e-12)

    linearization = collision.linearize(
        qpos,
        object_poses,
        quaternion_order="xyzw",
        activation_distance=0.2,
        include_ground=False,
    )
    assert len(linearization.frames) == 1
    direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    epsilon = 1e-7
    perturbed = qpos.copy()
    mujoco.mj_integratePos(model, perturbed[0], direction, epsilon)
    perturbed_audit = collision.audit(
        perturbed,
        object_poses,
        quaternion_order="xyzw",
    )
    finite_difference = (
        perturbed_audit.object_minimum_distance[0]
        - audit.object_minimum_distance[0]
    ) / epsilon
    np.testing.assert_allclose(
        linearization.jacobians[0] @ direction,
        finite_difference,
        atol=2e-7,
    )


def test_mujoco_trajectory_dynamics_audit_recovers_motor_force() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body>
              <joint name="slide" type="slide" axis="1 0 0"/>
              <geom type="sphere" size="0.05" mass="1"/>
            </body>
          </worldbody>
          <actuator>
            <motor joint="slide" ctrllimited="true" ctrlrange="-10 10"/>
          </actuator>
        </mujoco>
        """
    )
    fps = 100.0
    times = np.arange(12) / fps
    acceleration = 2.0
    qpos = (0.5 * acceleration * times**2)[:, None]
    audit = MujocoTrajectoryDynamicsAuditor(model, fps=fps).audit(qpos)
    assert np.max(audit.generalized_force_residual_norm) < 1e-10
    np.testing.assert_allclose(
        audit.controls[2:-2, 0],
        acceleration,
        atol=1e-9,
    )
    assert np.max(audit.control_violation) == 0.0


@pytest.mark.skipif(
    pytest.importorskip("torch").cuda.is_available() is False,
    reason="CUDA is unavailable",
)
def test_gpu_matches_osqp_on_joint_problem() -> None:
    trajectory_problem, _, _ = _joint_scale_problem()
    problem = trajectory_problem.build()
    reference = OSQPSolver().solve(problem)
    gpu = TorchSparseADMMSolver(
        device="cuda",
        settings=TorchADMMSettings(
            max_iterations=3000,
            absolute_tolerance=2e-5,
            relative_tolerance=2e-5,
            pcg_relative_tolerance=1e-8,
        ),
    ).solve(problem)
    assert gpu.success
    assert gpu.max_constraint_violation < 2e-4
    relative_objective_gap = abs(gpu.objective - reference.objective) / max(
        1.0, abs(reference.objective)
    )
    assert relative_objective_gap < 2e-3


@pytest.mark.skipif(
    pytest.importorskip("torch").cuda.is_available() is False,
    reason="CUDA is unavailable",
)
def test_gpu_respects_active_bounds_per_variable() -> None:
    variable_count = 256
    builder = SparseQuadraticBuilder(variable_count)
    builder.add_quadratic(
        np.arange(variable_count),
        np.eye(variable_count),
        -np.linspace(1.0, 2.0, variable_count),
    )
    upper = np.linspace(0.1, 0.5, variable_count)
    builder.add_variable_bounds(
        np.arange(variable_count),
        -upper,
        upper,
    )
    problem = builder.build()
    result = TorchSparseADMMSolver(
        device="cuda",
        settings=TorchADMMSettings(
            max_iterations=3000,
            absolute_tolerance=1e-6,
            relative_tolerance=1e-6,
        ),
    ).solve(problem)
    assert result.success
    assert result.max_constraint_violation < 3e-6
    np.testing.assert_allclose(result.solution, upper, atol=3e-6)
