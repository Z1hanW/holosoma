#!/usr/bin/env bash
set -euo pipefail
#### 7. Object And Contact Physics Were Fixed Last

After observations and commands were aligned, the remaining failure was physics: the policy contacted the box but did not lift it.

The important discovery was the object mass. The URDF box mass is `0.1kg`, but MuJoCo telemetry showed about `397N` support force at rest, which means the box was effectively around `40kg`. The non-colliding visual mesh was still contributing default MuJoCo mass.

Fixes:

- The object visual mesh geom is now massless.
- Only the collision geom carries the object mass from the URDF.
- Training-style object contact pairs are added for carry bodies such as hands, wrists, elbows, shoulders, and torso against the `largebox` collision geom.
- Hand collision geometry was aligned back to the reference values used by the original Holosoma assets.

After this, the rest support force became about `0.98N`, matching the `0.1kg` object.


cat <<'EOF'
# MuJoCo WBT Box Rollout Solved Notes

## 1. Motion-Init Reset Was Added First

The first issue was that MuJoCo reset and backspace reset could put the robot/object at the origin or at a generic robot default instead of the first frame of the selected motion file. Motion-init mode fixes that.

- `mj_env.sh` is manual by default.
- `bash mj_env.sh box_75 --motion-init` enables motion-init mode.
- Motion-init reads frame 0 from the selected motion file and applies:
  - robot X/Y position and yaw from the motion file,
  - robot root height, roll/pitch, and joints from the G1 WBT default standing pose,
  - object position and orientation.
- MuJoCo `qpos0` is also updated for robot root and object root, so backspace reset returns to the same motion-frame-0 state.
- `HOLOSOMA_MUJOCO_HOLD_MOTION_INIT_UNTIL_COMMAND=1` is enabled in motion-init mode so the robot/object do not drift before rollout commands arrive.
- An orange origin marker is added to make origin/reset mistakes visible.

For `box_75`, the expected motion-init state is not world origin. The robot and object should initialize at the position specified by `data_demo/box_75.npz`.

## 2. The Launch Scripts Were Split Into Manual And Debug Paths

After reset was reliable, the scripts were made explicit:

```bash
# Manual environment. No motion-init unless requested.
bash mj_env.sh box_75

# Manual rollout. Latest w5qostjn unless overridden.
bash mj_ro.sh

# Fully automated debug rollout. Always uses motion-init environment setup.
bash mj_debug.sh --run tvtwx4to --checkpoint latest --duration 120s
```

Useful variants:

```bash
# Motion-init environment, but rollout still launched manually.
bash mj_env.sh box_75 --motion-init

# Specific checkpoint.
bash mj_debug.sh --run tvtwx4to --checkpoint 20000 --duration 120s

# Another W&B run. This one is expected to use motion-derived/root command.
bash mj_debug.sh --run w5qostjn --checkpoint latest --duration 120s
```

`mj_env.sh` launches `robot:g1-29dof-w-object`, `camera:single_d435i_depth`, and `image_server:mujoco_d435i`. The virtual gantry is disabled, and manual mode stays on the plain DDS bridge path.

`mj_debug.sh` starts `mj_env.sh` in `hsmujoco`, waits for the image server and `/dev/shm/depth_img_shm`, checks the depth buffer, then starts `mj_ro.sh` in `hsinference`. Logs go to `artifacts/mj_debug_{run_id}_.../`.

## 4. W&B Checkpoint And ONNX Handling Were Made Explicit

`mj_ro.sh` now handles W&B checkpoint selection and ONNX setup directly.

- It accepts run ids, W&B URLs, `wandb://` paths, `latest`, numeric checkpoints like `20000`, or explicit ONNX filenames.
- Latest checkpoint is selected by the largest `model_{step}.onnx` step.
- Downloaded checkpoints are renamed to `/tmp/{run_id}_{model_name}.onnx`, for example `/tmp/tvtwx4to_model_20000.onnx`.
- The renamed checkpoint path is asserted after download to avoid confusing stale `/tmp/model_*.onnx` files.
- ONNX motion constants are patched from the selected motion `.npz`.
- Patched ONNX motion constants are asserted against the motion file.
- The observation config is adapted to the ONNX actor input dimensions and metadata, then asserted against ONNX input shapes.

Run-specific behavior:

- `tvtwx4to` must lift with zero sparse/root command. `mj_ro.sh` forces `HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=1` for this run.
- `w5qostjn` is not forced to zero command; it is expected to use the motion-derived/root command path.
- For `tvtwx4to`, policy debug should show `rootca_absmax=0.0`.

## 5. WBT Policy Initialization Was Cleaned Up

Several WBT rollout mismatches were fixed in the inference policy.

- `motion_yaw_offset` and `robot_yaw_offset` are initialized and reset consistently, fixing the earlier missing-attribute crash.
- Motion-init auto-start waits for the low state to match the expected motion-frame-0 yaw and G1 WBT default joint pose before starting policy control.
- `box_75` uses `HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1` to align the policy motion index with the motion sequence.
- Per-joint action scaling is read from ONNX metadata when training requested `action_scales_by_effort_limit_over_p_gain`.
- KP/KD are loaded from the WBT training-aligned config/metadata path and validated by shape.

Control behavior:

- `q_target` is not clipped in inference.
- Only PD torque is clipped before applying MuJoCo actuator torques.

## 9. Current Verified Result

Latest local verification:

```bash
HOLOSOMA_DEBUG_DURATION=120s HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT=300 \
  bash mj_debug.sh --run tvtwx4to --checkpoint latest --duration 120s
```

Result from `artifacts/mj_debug_tvtwx4to_20260507_233657`:

- `rootca_absmax=0.0`, confirming `tvtwx4to` used zero sparse/root command.
- Object support force at rest was about `0.98N`, matching the `0.1kg` URDF object.
- The box lifted to about `z=0.932m`.
- `max_dz=0.764m`.
- Sampled `LowCmdTelemetry` had no torque saturation.
EOF
