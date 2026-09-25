# HoloSoma: teacher training and distillation

Two training entry points:

- `train_teacher.py`: privileged-state MLP teacher, PPO from scratch (4 nodes × 8 GPUs).
- `train_distillation.py`: depth student, online teacher labels and PPO, using the `4kocpixa` recipe (1 node × 8 GPUs).

Each GPU uses 2,048 environments. Both scripts train for 40,000 iterations and save PT/ONNX pairs. Training parameters are defined directly in the scripts; shared algorithms, environments and inference code live in `src/`.

Use Python 3.11 with PyTorch 2.7.0, Isaac Sim 5.1.0, IsaacLab 0.47.2 and NumPy 1.26.0, then install:

```bash
pip install -e src/holosoma -e src/holosoma_inference
```

Data and checkpoints are external inputs. Paths below refer to the prepared single-slot motion banks, including object maps and rank shards. Distillation additionally requires the contact sidecars, robot assets, `ch2ckwzw` teacher at 40K and box actor initializer at 23K used by `4kocpixa`.

Run teacher training on each node, setting `--node-rank` to 0–3:

```bash
python train_teacher.py \
  --motion-bank /path/to/teacher_motion_bank \
  --entity YOUR_ENTITY --name teacher \
  --node-rank 0 --master-addr NODE_0_IP \
  --source-commit FULL_GIT_SHA
```

Run distillation:

```bash
python train_distillation.py \
  --motion-bank /path/to/student_motion_bank \
  --contact-bank /path/to/contact_sidecars \
  --robot-assets /path/to/robot_assets \
  --teacher-checkpoint /path/to/teacher_40000.pt \
  --initializer-checkpoint /path/to/box_23000.pt \
  --entity YOUR_ENTITY --name student \
  --source-commit FULL_GIT_SHA
```

Use `--help` for launch options or append `--check` to validate and print the CLI without training. Launches verify a clean checkout fetched directly from `origin`; use `--source-ref` if the published branch is renamed. All teacher nodes must use the same full SHA. Use a new `--output` directory for each launch.

The scripts retain the original data, reward and camera choices and explicitly select `peak_height` button labels for new training. They do not resume an existing run. GPU training has not been rerun for this release.
