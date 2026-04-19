# MuJoCo WASM Box Depth Track

Browser version of the `mj_box_depth_track.sh` depth-tracking flow for the
`g1_29dof` w-object large-box scene.

## What This Folder Does

- stages MuJoCo XML scenes, meshes, success rollout motion resets, and patched ONNX policies into `public/demo-assets/`
- runs official `@mujoco/mujoco` WASM in the browser
- runs `onnxruntime-web/wasm` policy inference in the browser
- builds `obs` from sparse root command + proprio history and `perception_obs` as an `87x58` depth observation
- defaults to the `infer_box_joystick.sh`-style manual root command and default-pose reset; the panel can switch back to clip-tracking command and raw motion init:
  - `W/S`: command forward/back
  - `A/D`: command left/right
  - `Q/E`: command yaw
  - `Space`: start, then pause/resume
  - `Backspace`: reset

## Default Assets

- checkpoint: `/home/ubuntu/FAR/holosoma/.teacher_checkpoints/model_07000.onnx` (`zihanw22/boxer/shoo7sr1`, iteration 7000)
- clips: success rollouts from `/home/ubuntu/FAR/holosoma/outputs/motion_bank_success_box_0_92_0p3`
- default clip: `box_74`
- default command/reset: `manual_root` command with `isaac_training_default_pose` reset
- scene xml: generated per clip from the G1 training URDF scene and each clip's `object_urdf_path`

## Usage

```bash
cd /home/ubuntu/FAR/holosoma/mujoco-web-wobj-depth-demo
npm install
npm run prepare-assets
npm run dev
```

Open `http://localhost:4173`.

`npm run dev` and `npm run build` copy the MuJoCo WASM runtime and ONNX Runtime
WASM assets into `public/runtime/` before serving or building.

## Useful Variants

Stage a single exact clip:

```bash
/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python scripts/prepare_demo_assets.py \
  --motion-file /home/ubuntu/FAR/holosoma/data/ds_box_data/train_g1_w_obj_prepared/box_74.npz
```

Stage clips from a directory:

```bash
/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python scripts/prepare_demo_assets.py \
  --motion-dir /home/ubuntu/FAR/holosoma/outputs/motion_bank_success_box_0_92_0p3 \
  --motion-glob "*.npz" \
  --preferred-clip-stem box_74 \
  --max-clips 0
```

Use the Clip dropdown in the browser to switch between staged success rollout
sequences. Each sequence has its own `clips/<id>/scene.xml`, `demo-config.json`,
and patched `policy.onnx`, so clips with different object URDFs load the matching
box geometry and reset state.

Compare browser observations against the Python reference assembly:

```bash
npm run compare-obs -- --steps 2
```

The comparer launches the running dev server in Playwright, snapshots the JS
state, and verifies sparse-root/proprio/action history assembly plus depth
crop/resize/normalize post-processing.

## Control Diagnostics

The browser panel shows two live control checks:

- `Action Scale`: the effective per-joint policy target scale range. The exported base `action_scale` is `0.25`, but the browser uses `control.policy_action_scales` when present.
- `Torque Sat`: the number of joints whose raw PD torque is within 98% of the MuJoCo actuator `ctrlrange`, plus the joint with the largest torque/limit ratio.

On clip load, the console prints a control table with per-joint action scale,
`kp`, `kd`, exported effort limit, compiled MuJoCo actuator limit, and whether
the limits match. During rollout, saturated joints are logged at most once per
second so weak grasps can be traced to specific arm/wrist/leg torque limits.

The exported dynamics treat `infer_box_joystick.sh` checkpoint metadata as the
source of truth. `control.effort_limits` comes from
`experiment_config.robot.dof_effort_limit_list`, effective action scales are
computed as `action_scale * effort_limit / kp`, and asset preparation fails if
the generated MuJoCo actuator `ctrlrange` differs from those checkpoint limits.
The `dynamics` block in each `demo-config.json` records the simulator/control
source, fixed object mass, object/rubber-hand friction, `condim`, and terrain
settings used to generate that clip scene.

Camera pitch is explicitly overridden to `10 deg`, matching the
`distill_box_perception.sh` training override. Each clip config keeps the
checkpoint value as `camera_pitch_deg_checkpoint` so accidental regressions back
to the checkpoint's `0 deg` metadata are visible.

## Notes

- this path does not use `viser`
- `manual_root` sends `[dx, dy, dyaw]` directly in the current root frame; `clip-tracking` sends the motion-derived root-frame command plus keyboard offsets
- `default-pose` reset keeps the selected clip's root XY/yaw and object state, but uses the training default root height/joints/velocities; `motion-init` reset uses the raw first frame from the selected rollout
- MuJoCo uses the official multi-threaded WASM build so XML compile has a pre-warmed worker pool
- depth observation raycasts the rendered visual geoms as BVH-accelerated triangles, then falls back to the object box proxy and analytic ground when no visual surface is hit
- staged assets are written under `public/demo-assets/`; generated runtime files are written under `public/runtime/`

## Alignment Lessons

- Keep command coordinates in the sparse root frame. The motion-derived sparse command and the keyboard offsets both need to be expressed relative to the current root yaw; applying keyboard `W/A/S/D` in the world frame makes "forward" depend on the global heading instead of the robot heading.
- Treat the depth map as camera data, not a canvas image. The policy path expects metric depth after the same crop, bicubic resize, clamp, min-valid replacement, and normalization as `infer_box_drop.sh`; do not vertically flip the array just because the debug canvas origin differs.
- Avoid JavaScript numeric coercion for calibration defaults. `Number(null)` becomes `0`, which can silently replace camera intrinsics with zero and produce black or invalid depth. Use explicit finite-number fallbacks and require positive focal lengths.
- Load object visuals from each clip's `object_urdf_path`. The collision box used for contacts is only an approximation; the visible object mesh must be added separately as a non-contact visual geom, while the collision geom can stay hidden for physics.
- Do not use visual-mesh bounding boxes for robot depth. Torso-mounted cameras are easily self-occluded by large oriented bounds. Raycast the actual triangles instead.
- Do not use Three.js's default triangle raycast for the depth camera. Thousands of rays over the robot visual meshes are too slow without `three-mesh-bvh`; build a bounds tree for each geometry and use `Raycaster.firstHitOnly`.
- Filter depth geometry by MuJoCo visual/contact intent. Include visible non-contact mesh visual geoms (`type=mesh`, `contype=0`, `conaffinity=0`) and exclude collision geoms such as the rubber-hand contact meshes, even when those collision geoms are visible in the browser scene.
- Hide disabled URDF collision primitives after replacing collisions from the MuJoCo reference XML. Wrist-yaw cylinder primitives can otherwise remain visible with `contype=0`, `conaffinity=0` and look like extra hand geometry even though they are no longer active contacts.
- Match the distillation physics, not just the URDF. The hand/object contact setup uses rubber-hand collision geoms, `condim=6`, trained friction/solver settings, and fixed object mass; editing only the URDF visual tree can make grasping look right while the dynamics still diverge.
