# MuJoCo Web W-Obj Track Demo

Standalone non-Viser browser demo for the `train_object_extend.sh` tracking policy on the shared `w-obj` MuJoCo scene.

## What This Folder Does

- stages a sim2sim-style `w-obj` MuJoCo scene generated from `holosoma` scene manager plus one or more `*_w_obj.npz` motion clips into `public/demo-assets/`
- patches a `train_object_extend.sh` ONNX checkpoint per clip so browser MuJoCo uses the same clip-specific motion constants as split sim2sim
- runs `mujoco-js + onnxruntime-web + Three.js` entirely in the browser
- builds the object-generalist tracking observation used by the current `holosoma_inference` code:
  - `motion_command(58)`
  - `motion_ref_ori_b(6)`
  - `base_ang_vel(3)`
  - `dof_pos(29)`
  - `dof_vel(29)`
  - `actions(29)`
  - `obj_target_pose_size_b(12)`
  - `obj_pos_b(3)`
  - `obj_ori_b(6)`
  - `obj_lin_vel_b(3)`
  - `obj_ang_vel_b(3)`
- exposes clip selection and reset mode toggles without using `viser`

## Default Assets

- checkpoint: `/data/logs_new/boxer/20260316_200048-g1_29dof_wbt_w_object_extend_20260316_200027_s01_scale_1p0-g1_29dof_wbt_w_object_extend_20260316_200027/model_23500.onnx`
- W&B run: `https://wandb.ai/zihanw22/boxer/runs/j21xgvcb`
- default preferred clip: `/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz`
- scene xml: generated at `public/demo-assets/scene.xml` from the same MuJoCo object-scene patch path used by `mj_track.sh`

## Usage

```bash
cd /home/ubuntu/FAR/holosoma/mujoco-web-wobj-track-demo
npm install
npm run clean-local-cache
npm run prepare-assets
npm run dev
```

Then open `http://localhost:4174`.

`npm run dev` and `npm run build` auto-copy the ONNX Runtime wasm binary into `public/runtime/ort/` before starting, so the browser does not depend on ORT's internal wasm path inference.

By default `prepare-assets` stages only this single clip and keeps the raw motion clip timing used by the current split MuJoCo tracking launcher:

```bash
/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz
```

## Useful Variants

Stage one exact clip:

```bash
/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python scripts/prepare_demo_assets.py \
  --motion-file /home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz
```

Use a different tracking checkpoint:

```bash
/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python scripts/prepare_demo_assets.py \
  --model-path /path/to/model_05000.onnx
```

Use the raw motion clip without the training-time prepend/append transitions:

```bash
/home/ubuntu/.holosoma_deps/miniconda3/envs/sim/bin/python scripts/prepare_demo_assets.py \
  --no-apply-training-motion-transitions
```

Validate the staged bundle:

```bash
npm run validate-bundle
npm run compare-motion-fk
npm run analyze-reset
npm run smoke-rollout
npm run compare-holosoma-inference
npm run js-runtime-smoke
```

Confirm you are looking at a fresh build:

```bash
curl -I http://127.0.0.1:4174/
curl -s http://127.0.0.1:4174/demo-assets/clips/sub3_largebox_003_mj_w_obj/demo-config.json | jq '.checkpoint.model_name, .checkpoint.iteration, .motion_file'
```

The page header and Runtime panel also show a `Build` timestamp. If that timestamp changes after restart, you are not looking at an old JS bundle.

## Notes

- this path does not use `viser`
- the front end feeds only `obs + time_step`; there is no `perception_obs` path in this demo
- `obj_lin_vel_b` intentionally mirrors the current `holosoma_inference` implementation so browser rollout matches the active inference code path
- the front end keeps the shared MuJoCo scene loaded and swaps only per-clip `demo-config.json + policy.onnx`
- the staged scene is no longer the bare retargeting XML; it now includes the same MuJoCo object-scene patches used by split sim2sim: default actuators, copied joint defaults, copied tendons, largebox mass/friction override, and object-terrain pair override
- `policy_action_scales` in the staged bundle now follow the training logic `action_scale * effort_limit / kp`, using the MuJoCo actuator `ctrlrange` values from the generated `w-obj` scene
- `WholeBodyTrackingPolicy` now reloads ONNX metadata and applies the same per-joint action scaling, so the browser demo and `holosoma_inference` agree on `raw action -> q_target -> torque`
- `npm run compare-holosoma-inference` is the main parity check: it walks the web rollout path and compares `obs`, `raw action`, `q_target`, and `torque` against `holosoma_inference` on the exact same MuJoCo state
- the runtime stats panel now shows the checkpoint file name, W&B run path, and iteration so you can verify you are not accidentally looking at the wrong export
- before ORT session creation, the browser explicitly loads `/runtime/ort/ort-wasm-simd-threaded.jsep.wasm` and checks the `\0asm` magic bytes, so wrong static paths fail immediately
- if a browser-side inference step throws, the page now stops the loop and writes the failure into the stats panel instead of silently freezing on the last rendered frame
- after regenerating assets or changing the front-end code, hard-refresh the browser so it does not keep serving a cached `demo-config.json` or JS bundle
