# Sim2Sim Verification

> **See also:** [Inference & Deployment Guide](../../README.md)

## What Sim2Sim Means Here

Sim2Sim means taking the same task from Simulator A and replaying it in Simulator B without changing the task semantics.

For this repository:

1. Simulator A is the existing whole-body tracking training / inference stack.
2. Simulator B is MuJoCo.
3. We keep the same checkpoint.
4. We keep the same single motion clip.
5. We keep the command source derived from the motion clip.
6. We only add new configuration paths for MuJoCo verification instead of changing existing default workflows.

## Verified Split Workflows

There are now three split launchers, depending on which training configuration the checkpoint came from:

```bash
bash sim2sim_box_split.sh <motion.npz> <checkpoint.pt|model.onnx>
bash sim2sim_box_split_tracking.sh <motion.npz> <checkpoint.pt|model.onnx>
bash sim2sim_box_split_depth.sh <motion.npz> <checkpoint.pt|model.onnx>
```

Use:

- `sim2sim_box_split.sh` for the box-state / mocap distill student
- `sim2sim_box_split_tracking.sh` for the object-generalist tracking teacher / teacher-export ONNX
- `sim2sim_box_split_depth.sh` for the depth-distill student trained from `distill_box_perception.sh`

All three launchers run:

- MuJoCo simulation in `hsmujoco`
- policy inference in `hsinference`
- ONNX patching for a single motion clip
- sim-state publishing from MuJoCo to inference
- motion-frame initialization for robot + object
- object-scene actuator injection through a MuJoCo-only config flag
- training-aligned MuJoCo timing defaults: `sim.fps=200`, `control_decimation=4`, virtual gantry disabled

The depth launcher additionally runs:

- `perception:camera_depth_d435i` inside `run_sim.py`
- perception-observation publishing from MuJoCo to inference
- the depth-distill inference preset `inference:g1-29dof-wbt-object-distill-depth`
- training-aligned D435 overrides from the checkpoint metadata: `17x17`, `near=0.001`, `far=max_distance=3.0`

The split path was verified with:

- `hsmujoco`: `/home/ubuntu/.holosoma_deps/miniconda3/envs/hsmujoco/bin/python`
- `hsinference`: `/home/ubuntu/.holosoma_deps/miniconda3/envs/hsinference/bin/python`

When the motion clip contains retargeting metadata, the split launchers now auto-resolve:

- `object_urdf_path`

from the `.npz` before falling back to the default `largebox` object scene. Manual `OBJECT_URDF=...`
still overrides the clip metadata.

## What Stays Compatible

The original code path stays intact. The Sim2Sim changes only activate through new scripts or new config values:

- `sim2sim_box_split.sh`
- `sim2sim_box_split_tracking.sh`
- `sim2sim_box_split_depth.sh`
- `--robot.object.mujoco-add-default-actuators=True`
- `--motion-init.*`
- `--simulator.config.bridge.publish-sim-state=True`
- `--simulator.config.bridge.publish-perception-obs=True`
- `inference:g1-29dof-wbt-object-distill`
- `inference:g1-29dof-wbt-object-distill-depth`

Existing defaults are not overwritten for normal training, IsaacGym, IsaacSim, or older inference flows.
The MuJoCo object-mesh fallback for perception is explicitly gated to the MuJoCo simulator path only.

## Tracking Teacher Notes

For the object-generalist tracking teacher:

- the first MuJoCo state should match the first motion-frame state from the clip
- use `SIM_MOTION_INIT_MODE=raw_motion`
- keep `APPLY_TRAINING_MOTION_TRANSITIONS=0` for clip-faithful sim2sim verification
- start inference after MuJoCo is already up, so the bridge can hold the initial pose before the first active command
- do not force a default-pose hold before the first command unless debugging startup only

This is now the default behavior of `sim2sim_box_split_tracking.sh`.

## Supported G1 Object Scenes

Current MuJoCo object-carry scene mapping for `g1_29dof`:

- `objects_largebox.urdf` / `largebox` -> `g1_29dof_w_largebox.xml`
- `boxlarge` -> `g1_29dof_w_boxlarge.xml`
- `boxmedium` -> `g1_29dof_w_boxmedium.xml`
- `boxsmall` -> `g1_29dof_w_boxsmall.xml`
- `boxtiny` -> `g1_29dof_w_boxtiny.xml`
- `boxlong` -> `g1_29dof_w_boxlong.xml`

For object scenes that do not define MuJoCo actuators in XML, the split launcher enables MuJoCo-only default actuator injection so the robot can actually receive commands.

## Recommended Commands

Box-state / mocap distill:

```bash
cd /home/ubuntu/FAR/holosoma
RUN_SECONDS=12 SIM_STARTUP_WAIT=5 bash ./sim2sim_box_split.sh \
  /home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml/omomo__sub3_largebox_015_mj_w_obj.npz \
  /data/logs_new/WholeBodyTracking/20260307_220516-g1_29dof_wbt_w_object_distill_goal_box_mocap_curriculum_stageA_20260307_220425-locomotion/model_00800.pt
```

Depth distill from `distill_box_perception.sh`:

```bash
cd /home/ubuntu/FAR/holosoma
RUN_SECONDS=8 SIM_STARTUP_WAIT=6 bash ./sim2sim_box_split_depth.sh \
  /home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml/omomo__sub3_largebox_015_mj_w_obj.npz \
  /data/logs_new/boxer/20260312_184258-g1_29dof_wbt_w_object_distill_box_perception_access_to_depth-locomotion/model_01000.pt
```

Object-generalist tracking teacher:

```bash
cd /home/ubuntu/FAR/holosoma
RUN_SECONDS=12 bash ./sim2sim_box_split_tracking.sh \
  /home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_behave_sq_carry_aug_mix_ml/omomo__sub3_largebox_015_mj_w_obj.npz \
  /home/ubuntu/FAR/holosoma/logs/sim2sim_exports/kge4jozt_model_12000.onnx
```

What these commands do:

1. patches the ONNX to the selected motion clip
2. launches `run_sim.py` in MuJoCo with `robot:g1_29dof_w_object`
3. initializes robot and object from motion frame 0
4. publishes sim state and clock from MuJoCo
5. launches `run_policy.py` with the matching inference preset

The depth command additionally matches training-side perception semantics:

1. MuJoCo uses `camera_depth_d435i`
2. it overrides the preset to the student-training camera settings stored in the ONNX metadata: `camera_width=17`, `camera_height=17`, `camera_near=0.001`, `camera_far=3.0`, `max_distance=3.0`
3. the registered `object` mesh is included in the far-tracking raycast
4. MuJoCo publishes `5046`-dim `perception_obs`
5. inference feeds `obs + time_step + perception_obs` into the patched depth ONNX

## Teacher Reference

The default teacher loaded by `distill_box_perception.sh` is:

- `wandb://zihanw22/boxer/kge4jozt/model_12000.pt`

Its exported ONNX metadata shows:

- training name: `g1_29dof_wbt_w_object_generalist`
- simulator: `IsaacSim`
- simulator timing: `sim.fps=200`, `control_decimation=4`
- inputs: `obs[1,181]` and `time_step[1,1]`
- no perception input on the teacher policy

This is why the MuJoCo split launchers now pin timing to `200 / 4`, while the depth student launcher additionally pins the D435 settings to the student checkpoint metadata.

## Findings As Of 2026-03-14

The current findings for the object-generalist tracking teacher are:

- `raw_motion` is the correct split initialization mode when the first simulated state must equal the first motion-frame state from the clip
- for tracking-teacher sim2sim, training-time default-pose prepend/append transitions must stay disabled or the ONNX motion command is shifted away from clip frame `0`
- `training_default_pose` can be useful for debugging startup only, but it is not the training-aligned tracking initialization
- MuJoCo with `robot + object + raw_motion` is stable without policy control
- the patched tracking-teacher ONNX matches the motion clip exactly for early `joint_pos` frames; the remaining failure is not in motion patching
- a clean split run must not share `lo` / `5555` / `5557` / `5559` with older MuJoCo or inference background jobs
- MuJoCo-only bridge hold works in isolation: `hold_initial_pose_until_first_command` can keep the raw-motion init pose stable when no policy is attached
- with clean ports and raw-clip patching, startup ONNX alignment improves to about `+3` frames with best joint error around `0.15 rad`
- the early failure happens after policy control starts, not during pure scene/object initialization
- startup synchronization and deferred auto-start reduce false negatives from stale startup state, but they do not remove the core instability
- the current blocker is MuJoCo early control instability after the first active teacher actions, typically `QACC unstable` within about `0.145s` to `0.24s` in the cleaned tracking path
- object loading, actor registration, and sim-state publishing work, but reliable object carry has not been validated because the robot diverges too early
- current live checks still show `object_disp_norm = 0.0`, so the box is not yet being stably carried in MuJoCo

Representative logs:

- `logs/sim2sim_runs/debug_raw_motion_series_mujoco.log`
- `logs/sim2sim_runs/debug_hold_series_mujoco.log`
- `logs/sim2sim_runs/debug_policy_first_raw_mujoco_v3.log`
- `logs/sim2sim_runs/debug_policy_first_raw_policy_v3.log`
- `logs/sim2sim_runs/hold_only_mujoco.log`

## Verified Runtime Signals

During the verified run:

- MuJoCo injected `29` torque actuators for the G1 object scene
- MuJoCo reported `Total actuators: 29`
- MuJoCo ran around `1340-1405 FPS`
- inference ran around `50 RL FPS`
- motion timestep advanced normally from frame `0`
- DDS watcher observed non-zero `lowcmd` traffic during the split run

During the verified depth split run:

- MuJoCo published `5046`-dim perception observations
- inference subscribed to those observations and logged repeated perception message milestones
- the registered `object` was added to the far-tracking perception raycast
- the MuJoCo split path required `--device cuda:0` for continuous depth updates

Logs are written to:

```bash
logs/sim2sim_runs/<motion_stem>/mujoco.log
logs/sim2sim_runs/<motion_stem>/policy.log
```

The patched ONNX is written to:

```bash
logs/sim2sim_exports/<model>__<motion>.onnx
```

## What To Verify Physically

When you use this path for physics verification, inspect:

- whether the robot remains upright
- whether the box follows the reference pose reasonably
- whether contacts explode or drift badly
- whether resets put both robot and object back into valid states
- whether MuJoCo diverges immediately from the source simulator

## Troubleshooting

`MuJoCo simulator exited during startup`

- Check `logs/sim2sim_runs/<motion_stem>/mujoco.log`

`Policy run failed`

- Check `logs/sim2sim_runs/<motion_stem>/policy.log`

`Failed to publish perception obs: Must be a CUDA device`

- Use the depth launcher `sim2sim_box_split_depth.sh`
- Keep `SIM_DEVICE=cuda:0` (or another CUDA device)

`Unsupported object URDF ... for robot 'g1_29dof'`

- Add the object mapping in `src/holosoma/holosoma/simulator/mujoco/scene_manager.py`
- Make sure the object body name in the MuJoCo composite scene matches the registered actor name

`No actuators found in MuJoCo model`

- Enable `--robot.object.mujoco-add-default-actuators=True`
- Or add explicit actuators to the composite MuJoCo XML

`QACC unstable` shortly after policy control starts

- For the tracking teacher, make sure the split run starts from `SIM_MOTION_INIT_MODE=raw_motion`
- Keep `APPLY_TRAINING_MOTION_TRANSITIONS=0` when the first MuJoCo state must equal clip frame `0`
- Kill unrelated MuJoCo / inference background processes before re-running if they share `lo`, `5555`, `5557`, or `5559`
- Confirm the checkpoint is the actor-only object-generalist teacher / teacher-export ONNX, not the depth student
- If MuJoCo is stable without policy but diverges as soon as active commands arrive, treat this as a control/dynamics mismatch, not an object-scene loading failure
- Check `logs/sim2sim_runs/debug_raw_motion_series_mujoco.log` and `logs/sim2sim_runs/debug_policy_first_raw_mujoco_v3.log`

`ModuleNotFoundError: mujoco`

- Use the `hsmujoco` environment

`ModuleNotFoundError` for inference dependencies

- Use the `hsinference` environment

## Legacy Wrapper

`sim2sim_box_verification.sh` is still kept for the older inference-wrapper path, but the verified and recommended workflows for checkpoint + motion replay are now:

- `sim2sim_box_split.sh` for box-state students
- `sim2sim_box_split_tracking.sh` for the object-generalist tracking teacher
- `sim2sim_box_split_depth.sh` for depth students
