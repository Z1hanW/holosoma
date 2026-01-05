# NuRec Export (3D Gaussian PLY -> USDZ)

This folder provides a thin wrapper around external NuRec exporters (for example
3DGruT) to convert a gsplat PLY into a USD/USDZ scene that Isaac Sim can load.

Isaac Sim expects NuRec scenes as standard USD assets (often `stage.usdz`).
The exporter does the heavy lifting; this script validates the gsplat PLY header,
invokes the exporter, and creates a `stage.usdz` or `stage.usda` you can point
`SceneFileConfig.usd_path` at.

## Usage

```bash
python preprocess/nurec/export_nurec_usdz.py \
  --input-ply path/to/scene.ply \
  --output-dir path/to/output_scene \
  --export-command "3dgrut export --input {input} --output {output_dir}"
```

Notes:
- The `--export-command` string is a template. `{input}` expands to the PLY
  path and `{output_dir}` to the output directory.
- If your exporter writes to a known file, you can pass `--exported-usd` to
  point to it explicitly.
- If your exporter already ran, use `--skip-export` and only create the stage.

Output:
- When the exporter emits a `.usdz`, the script copies it to `stage.usdz`.
- For `.usd/.usda/.usdc` outputs, the script writes a small `stage.usda`
  referencing the exported asset.

## Loading in Isaac Sim (Holosoma)

Point your scene config to the generated stage file:

- `scene.scene_files[0].usd_path = "path/to/output_scene/stage.usdz"`

If you have a `.usda` stage instead, use that path. The loader already supports
standard USD assets.
