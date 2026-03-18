# MuJoCo Web W-Obj Track Demo

Standalone non-Viser browser demo for the `train_object_extend.sh` tracking policy on the shared `w-obj` MuJoCo scene.

## What This Folder Does

- stages the shared `w-obj` MuJoCo scene plus one or more `*_w_obj.npz` motion clips into `public/demo-assets/`
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
- scene xml: `/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml`

## Usage

```bash
cd /home/ubuntu/FAR/holosoma/mujoco-web-wobj-track-demo
npm install
npm run prepare-assets
npm run dev
```

Then open `http://localhost:4174`.

`npm run dev` and `npm run build` auto-copy the ONNX Runtime wasm binary into `public/runtime/ort/` before starting, so the browser does not depend on ORT's internal wasm path inference.

By default `prepare-assets` stages only this single clip:

```bash
/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz
```

## Useful Variants

Stage one exact clip:

```bash
python3 scripts/prepare_demo_assets.py \
  --motion-file /home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz
```

Use a different tracking checkpoint:

```bash
python3 scripts/prepare_demo_assets.py \
  --model-path /path/to/model_05000.onnx
```

Apply training-time default-pose prepend/append transitions before patching:

```bash
python3 scripts/prepare_demo_assets.py \
  --apply-training-motion-transitions
```

Validate the staged bundle:

```bash
npm run validate-bundle
npm run compare-motion-fk
npm run analyze-reset
```

## Notes

- this path does not use `viser`
- the front end feeds only `obs + time_step`; there is no `perception_obs` path in this demo
- `obj_lin_vel_b` intentionally mirrors the current `holosoma_inference` implementation so browser rollout matches the active inference code path
- the front end keeps the shared MuJoCo scene loaded and swaps only per-clip `demo-config.json + policy.onnx`
- the runtime stats panel now shows the checkpoint file name, W&B run path, and iteration so you can verify you are not accidentally looking at the wrong export
- before ORT session creation, the browser explicitly loads `/runtime/ort/ort-wasm-simd-threaded.jsep.wasm` and checks the `\0asm` magic bytes, so wrong static paths fail immediately
