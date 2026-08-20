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
- `CuPySparseDirectADMMSolver`: sparse ADMM with one CPU SuperLU
  factorization and repeated GPU cuSPARSE triangular solves. It supports
  over-relaxation and residual-based `rho` refactorization. This is a hybrid
  CPU/GPU backend, not a GPU sparse factorizer.
- `ProxQPSolver`: sparse CPU proximal-QP reference for equality-heavy
  formulations.
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

The real dynamics stage supports explicit dynamics slacks or a quadratic
dynamics penalty. Contact finite differences may create isolated transition
coefficients several orders of magnitude larger than the median, so the stage
records the maximum coefficient and can omit transitions above an explicit
stability threshold. A nonlinear line search rejects every update that fails
to reduce rollout defect or increases exact collision violation beyond the
incoming trajectory's value.

`MujocoInteractionTrajectoryOptimizer` applies that structure to real PRISM
interaction data. It relinearizes mapped robot points and MuJoCo distances
around every accepted trajectory, retracts tangent updates with
`mj_integratePos`, and accepts steps only when the nonlinear merit improves
without increasing exact collision violation. Ground and moving-object
distances are checked against the compiled MuJoCo collision geometry after
every line-search trial.

PRISM input poses use `[quaternion, translation]`, while some downstream
interfaces use `[translation, quaternion]`. `decode_object_poses` validates
quaternion norms and resolves this layout explicitly. This guard is required:
the first real prototype silently interpreted an `xyzw + xyz` sequence as
`xyz + xyzw`, so its object-frame metrics were invalid.

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

## Real interaction canary

The collision-active `prism_cf_bin_m1_v8` source sequence has 217 frames,
7,619 kinematic QP variables, 15 mapped joints, and eight XYZ scale knots.
Eighteen nonlinear SQP iterations with CPU OSQP produced:

```text
mean keypoint error           141.5 mm -> 34.9 mm
contact-window wrist error    121.7 mm -> 17.5 mm
maximum collision violation   184.1 mm -> 0.002 mm
ground violating frames       12 -> 0
nonlinear collision gate      pass (0.050 mm tolerance)
```

Every QP solved to optimality. Sixteen updates were accepted; the final two
were rejected by the nonlinear merit gate. The real geometric problem remains
below the measured GPU crossover, so automatic routing correctly selects
OSQP; the 65,792-variable synthetic horizon remains the CUDA validation case.

The later endpoint-mask-IoU variant is not a valid interaction canary. Its
object translation moved about one meter away from the robot and its contact
window has no robot-object proximity. The runner records this as
`no_robot_object_proximity_in_input` instead of presenting collision-free
output as successful interaction retargeting.

## Real dynamics canary

The dynamics stage jointly optimizes 217 tangent states, 216 controls, and
eight three-axis scale knots: 20,182 variables in total. Inverse dynamics,
one-step rollout, and `mjd_transitionFD` must all use the sequence timestep
(`1/30 s`). Earlier runs estimated controls with the XML default timestep and
then linearized at `1/30 s`; those runs are invalid negative controls and must
not be used as dynamics evidence.

With the timestep contract fixed and finite-difference epsilon `1e-3`, the
maximum transition coefficient is about `4.71e4`. Stability thresholds cover:

```text
threshold       active transitions
100             130 / 216
300             134 / 216
1000            164 / 216
```

The corrected nominal trajectory has mean nonlinear one-step defect `52.15`
and maximum defect `231.85` in tangent-state norm. Strict `1e-5` ADMM
tolerances did not converge within 90 seconds on the coupled 2.95-million-NNZ
penalty Hessian. The canary can explicitly use a finite, bounded time-limit
iterate as a search direction, while retaining its non-optimal QP status.
Controls are projected to actuator limits and every direction is then subject
to the nonlinear dynamics/collision line search.

A 30-second gate-100 direction passed after backtracking to `1/2048`:

```text
mean nonlinear defect         52.15205 -> 52.14590
collision violation           2.029e-6 -> 2.801e-6 m
contact wrist error            17.504 -> 17.499 mm
QP status                     time limit reached
```

The iteration driver preserves saved `qvel` and controls instead of
re-estimating them from `qpos`. One chained iteration then reduced defect from
`52.14590` to `52.14452` at step `1/8192`, with collision violation
`2.989e-6 m`. A fixed absolute collision cap of `3.029e-6 m` rejected the next
iteration and stopped the driver automatically. This is measurable but very
small dynamics progress, not a converged dynamics retargeting result.

## Production boundary

This code is not yet a replacement for the accepted strict retargeter.
Production promotion still requires:

1. adding explicit hand-surface contact targets and foot sticking constraints
   in addition to the current keypoint and exact non-penetration terms;
2. parameterizing each scale dimension with an explicit physical meaning and
   keeping visual, collision, solver, and saved-metadata scale contracts
   consistent;
3. completing a dynamics-constrained real-sequence stage with controls,
   actuator limits, moving-object contacts, and a known frame rate;
4. broadening real canaries beyond one collision-active sequence and comparing
   completion, penetration, contact error, jitter, dynamics defects, wall
   time, and GPU memory against the accepted baseline.

Until those gates pass, the existing strict Clarabel path remains authoritative.
