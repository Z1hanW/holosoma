# Viser Inference Quick Reference

This note records the correct local Viser inference path for depth-student checkpoints.

## Correct Entry Point

Use `scripts/viser_depth_student_physics_eval.py` for depth-student visualization and debugging. This path rebuilds the environment from the saved checkpoint config, computes the same low-dimensional proprioception and processed depth input used during training, and steps the simulator with the student's actions.

Use a distilled `student_*.pt` checkpoint. Do not pass a privileged teacher `model_*.pt` checkpoint to this script.

## Download The Latest Student Checkpoint

Replace `RUN_PATH` with the target W&B run. Examples:

- `zihanw22/holosomatest/rrfbwfnq`
- `zihanw22/carry-any/hxtcnu9p`

```bash
cd /home/ubuntu/FAR/holosoma

RUN_PATH=zihanw22/holosomatest/rrfbwfnq
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
api = wandb.Api()
run = api.run(run_path)
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
echo "$CKPT"
```

If the checkpoint is already local, skip the W&B block and set:

```bash
CKPT=/path/to/student_XXXXXXX.pt
```

## Launch Viser Inference

Use GUI command controls for normal local debugging:

```bash
cd /home/ubuntu/FAR/holosoma
PYTHONPATH=src/holosoma ./scripts/viser_depth_student_physics_eval.py \
  --checkpoint "$CKPT" \
  --num-envs 1 \
  --env-id 0 \
  --port 2106 \
  --gui-command \
  --depth-hits \
  --no-red-points \
  --no-motion-ref \
  --disable-randomization
```

Open `http://localhost:2106`. The right-side `Command` folder exposes:

- `Root target x (m)`
- `Root target y (m)`
- `Root target yaw (rad)`
- `Zero command`

These GUI values directly replace the first three `root_target_xy_yaw` policy inputs. Do not use `simulator.config.bridge` for this; the bridge is the SDK/DDS low-level robot-control path and does not set the student's command observation.

For local gamepad control, use `--joystick` instead of `--gui-command`.

## Student Input Contract

The expected inference input is:

```text
root_target_xy_yaw
+ projected_gravity
+ base_ang_vel
+ joint_pos
+ joint_vel
+ last_action
+ processed_depth
```

`projected_gravity` is proprioception computed from the current robot/base orientation. On the real robot, the equivalent value must come from the IMU/base attitude estimate.

The student does not consume heightmap/height-scan points, the full teacher tracking observation, or the full target ghost G1. Keep `--no-red-points` and `--no-motion-ref` for deployment-like visualization. Enable `--red-points` or `--motion-ref` only as diagnostics.

## HOI/Object Student Notes

For HOI/object students, keep the checkpoint's saved experiment config intact so the motion bank, object URDF metadata, simulator object state, and dynamic-object depth raycast match training.

If object mesh or object motion is missing in visualization, first check that:

- the checkpoint is a HOI `student_*.pt` from the intended run,
- it is not a teacher `model_*.pt`,
- it is not a non-object depth student,
- the checkpoint config still points to the object motion bank and object URDF metadata used during training.

## Switching Sequence

The current script samples sequences through the environment. To inspect another sampled rollout, launch more envs and change `--env-id`:

```bash
PYTHONPATH=src/holosoma ./scripts/viser_depth_student_physics_eval.py \
  --checkpoint "$CKPT" \
  --num-envs 8 \
  --env-id 3 \
  --port 2106 \
  --gui-command \
  --depth-hits \
  --no-red-points \
  --no-motion-ref \
  --disable-randomization
```

Use `--env-id 0..7` to view different sampled rollouts from the same run.
