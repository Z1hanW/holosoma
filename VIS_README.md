# Physics Inference With Viser

Use this file as the task-wise launcher reference for local physics inference.

## Common Setup

```bash
cd /home/ubuntu/FAR/holosoma
source scripts/source_inference_setup.sh
export PYTHONPATH=src/holosoma
```

Checkpoint rule:

- `model_*.pt`: tracking expert / teacher policy. Use `scripts/viser_current_physics_rollout.py`.
- `student_*.pt`: depth student policy. Use `scripts/viser_depth_student_physics_eval.py`.

## Carry-Object

### Expert / Teacher Physics Inference

Use this for TokenHSI object policies such as `rrfbwfnq`. This path shows robot motion, object mesh, object motion, reference object, terrain, and supports sequence switching.

```bash
python scripts/viser_current_physics_rollout.py \
  --checkpoint wandb://zihanw22/holosomatest/rrfbwfnq/model_08000.pt \
  --port 2099 \
  --env-id 0 \
  --sequence-envs 64 \
  --disable-randomization \
  --no-red-points
```

Open `http://localhost:2099`. Use the Viser `Sequence` dropdown to switch clips.

Carry-object checks:

- Use object-capable expert checkpoints from `g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target`.
- The checkpoint config must load `/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry`.
- The object map must exist: `/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry/_clip_object_urdf_map.json`.
- The object URDF must exist: `holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf`.
- If object mesh or object motion is missing, first check checkpoint type, object map, object URDF, and saved experiment config.

### Depth Student Physics Inference

Use this only after distilling the object expert to a `student_*.pt`. Do not pass `rrfbwfnq/model_08000.pt` to the depth-student script.

```bash
CKPT=/path/to/student_XXXXXXX.pt

python scripts/viser_depth_student_physics_eval.py \
  --checkpoint "$CKPT" \
  --port 2106 \
  --num-envs 1 \
  --env-id 0 \
  --gui-command \
  --depth-hits \
  --no-red-points \
  --no-motion-ref \
  --disable-randomization \
  --log-every 100
```

For joystick control, replace `--gui-command` with `--joystick`.

## Terrain-Traversal

### Tracking Expert Physics Inference

Use this for terrain WBT teacher checkpoints (`model_*.pt`) when checking true PhysX rollout on OBJ terrain.

```bash
python scripts/viser_current_physics_rollout.py \
  --checkpoint /path/to/model_XXXXX.pt \
  --port 2099 \
  --env-id 0 \
  --sequence-envs 16 \
  --disable-randomization \
  --red-points
```

For W&B checkpoints, use a `wandb://` URI, for example:

```bash
python scripts/viser_current_physics_rollout.py \
  --checkpoint wandb://zihanw22/holosomatest/btoe97gr/model_XXXXX.pt \
  --port 2099 \
  --env-id 0 \
  --sequence-envs 16 \
  --disable-randomization \
  --red-points
```

Terrain checks:

- For fused CRISP stairs, the NPZ must contain `terrain_origins`.
- Do not use kinematic replay to judge policy quality.
- Red height-scan points are diagnostics for teacher/heightmap policies, not depth-student inputs.

### Depth Student Physics Inference

Use this for terrain depth-student checkpoints (`student_*.pt`), including root-command students.

```bash
CKPT=/path/to/student_XXXXXXX.pt

python scripts/viser_depth_student_physics_eval.py \
  --checkpoint "$CKPT" \
  --port 2106 \
  --num-envs 1 \
  --env-id 0 \
  --gui-command \
  --depth-hits \
  --no-red-points \
  --no-motion-ref \
  --disable-randomization \
  --log-every 100
```

The Viser `Command` folder writes the first three student inputs:

- `Root target x (m)`
- `Root target y (m)`
- `Root target yaw (rad)`

To inspect another sampled rollout:

```bash
python scripts/viser_depth_student_physics_eval.py \
  --checkpoint "$CKPT" \
  --port 2106 \
  --num-envs 8 \
  --env-id 3 \
  --gui-command \
  --depth-hits \
  --no-red-points \
  --no-motion-ref \
  --disable-randomization
```

## Download Latest Student Checkpoint From W&B

Use this for either task when the target run contains `student_*.pt` files.

```bash
RUN_PATH=zihanw22/holosomatest/your_run_id
OUT_DIR=artifacts/wandb_checkpoints/${RUN_PATH//\//_}
mkdir -p "$OUT_DIR"
export RUN_PATH OUT_DIR

python - <<'PY'
import os
import re
from pathlib import Path

import wandb

run_path = os.environ["RUN_PATH"]
out_dir = Path(os.environ["OUT_DIR"])
run = wandb.Api().run(run_path)
ckpts = [f for f in run.files() if re.fullmatch(r"student_\d+\.pt", f.name)]
if not ckpts:
    raise SystemExit(f"No student_*.pt checkpoint found in {run_path}")
latest = max(ckpts, key=lambda f: int(re.search(r"\d+", f.name).group()))
latest.download(root=str(out_dir), replace=True)
path = out_dir / latest.name
(out_dir / "LATEST_STUDENT_CHECKPOINT").write_text(str(path) + "\n")
print(path)
PY

CKPT=$(cat "$OUT_DIR/LATEST_STUDENT_CHECKPOINT")
```

## Student Input Contract

Depth-student inference input:

```text
root_target_xy_yaw
+ projected_gravity
+ base_ang_vel
+ joint_pos
+ joint_vel
+ last_action
+ processed_depth
```

`projected_gravity` comes from current robot/base orientation. On the real robot, provide the equivalent value from IMU/base attitude.

## Common Mistakes

- Do not run `scripts/viser_depth_student_physics_eval.py` with `model_*.pt`.
- Do not run `scripts/viser_current_physics_rollout.py` with `student_*.pt`.
- Do not use `simulator.config.bridge` for GUI-command student visualization; use `--gui-command`.
- Do not show red height-scan points or target ghost by default for depth students.
- Do not treat heightmap, height-scan, or full target G1 as student inputs.
- Do not debug carry-object without confirming object map, object URDF, and object-capable checkpoint config.
