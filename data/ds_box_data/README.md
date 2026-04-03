# DS Box Data

This directory contains the final trainable `g1-w-obj` dataset exported from:

- `/home/ubuntu/FAR/CRISP-Real2Sim-Obj/vis_scripts/results/train_g1_w_obj`
- `/home/ubuntu/FAR/CRISP-Real2Sim-Obj/vis_scripts/results/train_g1_w_obj_geometry`

Contents:

- `train_g1_w_obj/`: motion bank `.npz` files for training
- `train_g1_w_obj_geometry/`: matching per-sequence `.obj` box geometry
- `train_g1_w_obj_status.csv`: machine-readable status table
- `train_g1_w_obj_status.md`: human-readable status table

Notes:

- Bad sequences removed from final results: `box_64`, `box_66`, `box_69`, `box_81`
- Current converted trainable clips: `43`

Example training commands:

```bash
MOTION_DIR=/nfs/zzzihanw/ds_box_data/train_g1_w_obj \
OBJ_DIR=/nfs/zzzihanw/ds_box_data/train_g1_w_obj_geometry \
bash /home/ubuntu/FAR/holosoma/train_multi_perception.sh
```

```bash
MOTION_DIR=/nfs/zzzihanw/ds_box_data/train_g1_w_obj \
bash /home/ubuntu/FAR/holosoma/train_object_generalist.sh
```
