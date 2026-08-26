# Hybrid Motion Imitation with the HoloSoma depth interface

This branch selectively adapts the Hybrid Motion Imitation (HMI) training
contract from
[`jiashunwang/Hybrid-Motion-Imitation`](https://github.com/jiashunwang/Hybrid-Motion-Imitation),
pinned for this port at commit
`c353731999b3578c41ad5a00f896415b45e6a9f5` (Apache-2.0).

The implementation is an adapter, not a source-tree merge. The upstream and
this repository have unrelated histories, and HoloSoma's production policy
interface is intentionally kept stable.

## Fixed policy/deployment interface

- `actor_obs`: 94 scalars in the order
  `hmi_goal_command(3) + drop_button(1) + proprio_with_actions_no_linvel(90)`
- `perception_obs`: normalized D435i depth, flattened `58 x 87 = 5046`
- depth encoder: `far_tracking_cnn_small`, 32-dimensional embedding
- actor output: 29 joint actions
- ONNX inputs/outputs and depth preprocessing remain the existing HoloSoma
  contract.

HMI upstream uses richer terminal-goal observations. To fit the immutable
three-scalar command slot, the adapter exposes the terminal object pose as
`[goal_x, goal_y, goal_yaw]` in the current robot-heading frame. Current object
state comes from depth. The existing drop slot is always zero so the actor
cannot infer the reference-motion phase from a native drop cue.

## Algorithm contract

- Every environment is assigned once to a deterministic track or generation
  partition and never changes mode mid-run.
- Track rows retain adaptive reset-frame sampling; generation rows use uniform
  reset-frame sampling with the released `0.20` frame-zero mixture.
- The actor never observes a per-step reference pose. It receives only current
  proprioception/depth and the terminal object goal.
- Tracking environments use the original dense robot/object imitation rewards
  and strict full-XYZ tracking termination.
- Generation environments mask dense reference rewards and reference-only
  critic observations. As in the released implementation, masked exponential
  rewards become an action-independent constant `0.5`. Generation rows receive
  a one-time object-goal bonus of `3.0` at position error `<= 0.20 m` and
  orientation error `<= pi/6`, plus the shared regularizers.
- Generation termination keeps HMI's relaxed safety gates: root height
  `0.45 m`, reference-z `0.40 m`, reference-XYZ `100 m`, object-z `0.60 m`, and
  object orientation `1.0 rad`.
- Motion completion holds the terminal reference until the normal 10-second
  timeout; it does not end or silently resample an HMI episode.
- Stage 1 is track-only for 15,000 iterations. Stage 2 is a fresh formal
  identity initialized from the Stage-1 actor, with a fixed 50/50 track/gen
  partition for 20,000 iterations. Stage 2 uses the released object-goal recipe
  (`0.5 m` XY / `0.4 rad` yaw standard deviations) with its success-driven
  `0.3 -> 1.0` curriculum, globally synchronized at 20-iteration boundaries.
  Reset and step-zero perturbations match the upstream object task. Geometry
  size randomization is deliberately disabled in this first depth-controlled
  comparison.

Both experiment presets set `training.export_onnx=true`:

- `exp:g1-29dof-wbt-w-object-hmi-depth-stage1`
- `exp:g1-29dof-wbt-w-object-hmi-depth-stage2`

The checked-in, mesh-based `data_demo/sub10_largebox_032_mj_w_obj.npz` clip is
the runnable preset default and has matching object metadata. Production
launches must override `motion_file` with the intended immutable motion view,
override `robot.object.object_urdf_path` with that view's object map, and
preserve the complete object-map and real-mesh contracts.

Before a formal run, the launcher must still bind an immutable remote Git SHA,
immutable motion/object manifests, the exact depth observation contract, and a
real PyTorch/ONNX Runtime parity report. Stage 2 must use an explicit audited
policy-init migration; it must not silently resume a Stage-1 run under changed
reward and partition semantics.
