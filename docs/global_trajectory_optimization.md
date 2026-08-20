# Whole-Trajectory Retargeting Experiment

## Scope

This branch adds an experimental horizon-wide optimization path without
changing the accepted frame-by-frame Clarabel retargeter.

The optimization variables are:

- robot state or tangent-state trajectory `x[0:T]`;
- vector-valued scale knots `s[0:K]`, where each knot may represent XYZ scale,
  body-group scale, contact scale, or another user-defined scale vector;
- optional controls `u[0:T-1]`.

A local-support piecewise-linear basis maps `K` scale knots to every frame.
This supports anisotropic and time-varying scale fields without introducing a
free scale variable at every frame.

## Sparse formulation

The QP uses OSQP form:

```text
minimize  0.5 z' H z + g' z
subject to lower <= A z <= upper
```

The builder adds local tracking blocks, state velocity and acceleration
regularization, scale-knot smoothness, variable bounds, arbitrary linearized
constraints, and affine dynamics. It emits SciPy sparse matrices and never
materializes horizon-wide dense identity or `A' A` matrices.

Available solvers:

- `OSQPSolver`: sparse CPU reference and preferred small-problem backend.
- `TorchSparseADMMSolver`: CUDA or CPU ADMM with matrix-free sparse PCG.
  Each PCG iteration evaluates `H x`, `A x`, and `A' y`; it does not form
  `A' A`.
- `AutoSparseSolver`: routes problems below 20,000 variables to OSQP and
  larger problems to CUDA. A failed or over-tolerance GPU result falls back to
  OSQP with the reason recorded in diagnostics.

GPU termination uses per-constraint infinity norms. L2 termination was tested
and rejected because it allowed an individual active bound to exceed its
tolerance as the horizon grew.

## Dynamics-level TrajOpt

`MujocoDynamicsLinearizer` calls `mjd_transitionFD` at each nominal frame. The
linearized state is MuJoCo's tangent state:

```text
[delta_q_tangent, delta_qvel, delta_activation]
```

This avoids treating free-joint quaternions as Euclidean coordinates. The
linearizer also computes rollout defects and emits:

```text
delta_x[t+1] = A[t] delta_x[t] + B[t] delta_u[t] + defect[t]
```

`SequentialTrajectoryOptimizer` supplies the trust-region and line-search
outer loop needed to refresh collision, tracking, scale, and dynamics
linearizations.

The actuated G1 canary must load the complete
`scene_g1_29dof_wbt_plane.xml`, not the bare robot XML whose contact pairs
refer to a scene-owned floor. The scene canary reports `nq=36`, `nv=35`,
`nu=29`, tangent-state dimension 70, `B.shape=(3, 70, 29)`, finite matrices,
zero rollout defect, and `0.017 s` linearization time for three transitions.

## 2026-08-20 benchmark

Host: 8x NVIDIA L40S. Precision: float64.

Inactive-bound problem:

```text
frames                 1024
state dimension        64
scale dimension        8
scale knots            32
variables              65,792
H nonzeros             6,526,464
QP build time           1.683 s
peak RSS after build    1.12 GB
CPU OSQP                7.236 s
CUDA sparse ADMM/PCG    0.469 s
speedup                 15.4x
max violation          0
relative objective gap about 2e-9
```

At 8,288 variables, CPU OSQP remains faster. With active bounds and `1e-6`
absolute/relative tolerances, OSQP took `0.222 s`; CUDA took `2.888 s` and its
maximum per-variable violation was `1.91e-6`, within the combined tolerance.
This motivates automatic routing.

At 65,792 variables with active bounds, OSQP took `12.532 s`; CUDA took
`4.163 s` with a maximum per-variable violation of `1.63e-6`. The large
active-set case therefore still benefits from CUDA by about `3.0x`.

An active-bound benchmark exposed and drove two fixes:

1. variable bounds now assemble directly as sparse diagonal entries instead
   of a dense `np.eye(N)`;
2. GPU convergence now uses infinity-norm residuals.

## Production boundary

This code is not yet a replacement for the accepted strict retargeter.
Production promotion still requires:

1. extracting per-frame interaction-mesh, contact, foot, joint-limit, and
   collision Jacobians around an accepted sequential seed;
2. parameterizing each scale dimension with an explicit physical meaning and
   keeping visual, collision, solver, and saved-metadata scale contracts
   consistent;
3. exact nonlinear MuJoCo collision acceptance after every TrajOpt step;
4. a real G1 sequence canary comparing completion, penetration, contact error,
   jitter, dynamics defects, wall time, and GPU memory against the accepted
   baseline.

Until those gates pass, the existing strict Clarabel path remains authoritative.
