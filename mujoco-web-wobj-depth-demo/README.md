# MuJoCo WASM Box Depth Track

Browser version of the `mj_box_depth_track.sh` depth-tracking flow for the
`g1_29dof` w-object large-box scene.

## What This Folder Does

- stages MuJoCo XML scenes, meshes, success rollout motion resets, and patched ONNX policies into `public/demo-assets/`
- runs official `@mujoco/mujoco` WASM in the browser
- runs `onnxruntime-web/wasm` policy inference in the browser
- builds `obs` from sparse root command + proprio history and `perception_obs` as an `87x58` depth observation
- maps the motion clip root trajectory to the sparse root command, with keyboard offsets:
  - `W/S`: root target x
  - `A/D`: root target y
  - `Q/E`: root target yaw
  - `Space`: start, then pause/resume
  - `Backspace`: reset

## Default Assets

- checkpoint: `/data/logs_new/boxer/20260415_014803-g1_29dof_wbt_w_object_distill_box_perception_sparse_root_cmd_access_to_depth-locomotion/model_03999.onnx`
- clips: success rollouts from `/home/ubuntu/FAR/holosoma/outputs/motion_bank_success_box_0_92_0p3`
- default clip: `box_74`
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

## Notes

- this path does not use `viser`
- MuJoCo uses the official multi-threaded WASM build so XML compile has a pre-warmed worker pool
- depth observation uses the rendered visual bodies through oriented bounding-box ray hits by default (`web_depth_mesh_mode: "bounds"`), with analytic ground as a fallback; set `web_depth_mesh_mode: "triangles"` only for slow offline checks
- staged assets are written under `public/demo-assets/`; generated runtime files are written under `public/runtime/`
