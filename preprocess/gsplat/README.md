# Gsplat PLY Preprocess (Legacy Heightfield)

Note: For Isaac Sim NuRec scenes (USD/USDZ assets), use
`preprocess/nurec/export_nurec_usdz.py`. The script in this folder only builds
a terrain heightfield mesh and does not produce NuRec-compatible assets.

This folder contains utilities to convert 3D Gaussian Splatting PLY exports into
a mesh that the simulator can load via `terrain_load_obj`.

## Expected PLY format (gsplat)

The converter reads PLY files in the gsplat export layout, which typically
includes these vertex properties in order:

- `x`, `y`, `z` (float)
- `f_dc_*` and `f_rest_*` (float, SH coefficients)
- `opacity` (float)
- `scale_0`, `scale_1`, `scale_2` (float)
- `rot_0`, `rot_1`, `rot_2`, `rot_3` (float)

Only `x`, `y`, `z` are required. If `opacity` is present you can use
`--opacity-min` to filter points.

## Output

The script produces an OBJ heightfield mesh suitable for
`terrain:terrain-load-obj` in holosoma.

Example:

```bash
python preprocess/gsplat/gsplat_ply_to_obj.py path/to/point_cloud.ply \
  --grid-res 0.05 \
  --align-ground \
  --output path/to/point_cloud_terrain.obj
```
