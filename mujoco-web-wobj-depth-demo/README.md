# MuJoCo Web W-Obj Depth Demo

Standalone non-Viser browser demo for the `distill_box_perception.sh` depth student on the `w-obj` scene.

## What This Folder Does

- stages the shared `w-obj` MuJoCo scene plus a selectable set of `w_obj` motion clips into `public/demo-assets/`
- runs MuJoCo + ONNX inference in the browser
- renders the main scene and a side depth panel using the same 17x17 depth observation shape expected by the exported student ONNX
- exposes a front-end clip selector without using `viser`

## Default Assets

- checkpoint: `/data/logs_new/boxer/20260317_111305-g1_29dof_wbt_w_object_distill_box_perception_access_to_depth-locomotion/model_00800.onnx`
- default preferred clip: `/home/ubuntu/FAR/holosoma/src/holosoma/holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz`
- default clip source dir: `/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry`
- scene xml: `/home/ubuntu/FAR/holosoma/src/holosoma_retargeting/models/g1/g1_29dof_w_largebox.xml`

## Usage

```bash
cd /home/ubuntu/FAR/holosoma/mujoco-web-wobj-depth-demo
npm install
npm run prepare-assets
npm run dev
```

Then open `http://localhost:4173`.

`npm run dev` and `npm run build` now auto-copy the ONNX Runtime wasm binary into `public/runtime/ort/` before starting, so the browser no longer depends on ORT's internal path inference.

By default `prepare-assets` stages up to `12` clips from `omomo_carry`, with `sub3_largebox_003_mj_w_obj` forced to the front if present.

## Useful Variants

Stage a single exact clip:

```bash
python3 scripts/prepare_demo_assets.py \
  --motion-file /home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry/sub3_largebox_003_mj_w_obj.npz
```

Stage more clips from a directory:

```bash
python3 scripts/prepare_demo_assets.py \
  --motion-dir /home/ubuntu/FAR/holosoma/src/holosoma_retargeting/converted_res/object_interaction/omomo_carry \
  --max-clips 24
```

Check the staged strict camera / Three camera alignment:

```bash
npm run check-camera
```

The `check-camera` npm script uses the repo's MuJoCo-enabled Python env directly, so it does not depend on your currently activated shell env.

Compare the current web-demo reset against Isaac's `training_default_pose` reset:

```bash
npm run analyze-reset
```

## Notes

- this path does not use `viser`
- the asset prep script writes a shared `scene.xml`, shared meshes, and per-clip `policy.onnx + demo-config.json` bundles under `public/demo-assets/clips/`
- the front-end selector swaps clip bundles by reloading only the config and ONNX, not the MuJoCo scene
- before ORT session creation, the browser explicitly loads `/runtime/ort/ort-wasm-simd-threaded.jsep.wasm` and checks the `\0asm` magic bytes, so HTML fallback / wrong static path failures surface immediately
- the current rollout path is tied to the observation structure of the depth-distill object-carry checkpoint:
  - `obs = sparse_root_cmd(3) + proprio(93)`
  - `perception_obs = normalized depth 17x17`
