# HoloSoma: teacher → rollout → student

Three shell entry points at the repository root:

- `train_teacher.sh`: privileged-state MLP teacher, PPO from scratch (4 nodes × 8 GPUs).
- `rollout.sh`: checkpoint-native teacher trajectories and mesh contact sidecars.
- `train_student.sh`: depth student, online teacher labels and PPO, using the `4kocpixa` recipe (1 node × 8 GPUs).

Each GPU uses 2,048 environments. Both scripts train for 40,000 iterations and save PT/ONNX pairs. Training parameters live in `scripts/_teacher.py` and `scripts/_student.py`; shared algorithms, environments and inference code live in `src/`.

Use Python 3.11 with PyTorch 2.7.0, Isaac Sim 5.1.0, IsaacLab 0.47.2 and NumPy 1.26.0, then install:

```bash
pip install -e src/holosoma -e src/holosoma_inference
```

Data and checkpoints are external inputs. Paths below refer to the prepared single-slot motion banks, including object maps and rank shards. Distillation additionally requires the contact sidecars, robot assets, `ch2ckwzw` teacher at 40K and box actor initializer at 23K used by `4kocpixa`.

Run teacher training on each node, setting `--node-rank` to 0–3:

```bash
bash train_teacher.sh \
  --motion-bank /path/to/teacher_motion_bank \
  --entity YOUR_ENTITY --name teacher \
  --node-rank 0 --master-addr NODE_0_IP \
  --source-commit FULL_GIT_SHA
```

Collect teacher trajectories and contacts (one environment per clip in the supplied bank):

```bash
bash rollout.sh \
  --checkpoint /path/to/teacher_40000.pt \
  --motion-bank /path/to/teacher_motion_bank \
  --output /path/to/new_rollout --gpu 0
```

This is native policy rollout for training data. It exports `motion_bank/`, `clips/` and success/failure summaries, retaining every clip. A prepared rank shard can be supplied instead of the full bank to reproduce the original 8-shard collection sizes (32/32/16/16/16/16/8/1). Output is raw rollout/contact data; the command-bank and rank-shard preparation utilities are in `scripts/`. The `4kocpixa` student recipe expects its prepared command/contact banks.

Run distillation:

```bash
bash train_student.sh \
  --motion-bank /path/to/student_motion_bank \
  --contact-bank /path/to/contact_sidecars \
  --robot-assets /path/to/robot_assets \
  --teacher-checkpoint /path/to/teacher_40000.pt \
  --initializer-checkpoint /path/to/box_23000.pt \
  --entity YOUR_ENTITY --name student \
  --source-commit FULL_GIT_SHA
```

Use your activated Python environment, or set `PYTHON_BIN=/path/to/python`. Use `--help` for launch options or append `--check` to validate and print the CLI without training. Training launches verify a clean checkout fetched directly from `origin`; use `--source-ref` if the published branch is renamed. All teacher nodes must use the same full SHA. Use a new `--output` directory for each launch.

The scripts retain the original data, reward and camera choices and explicitly select `peak_height` button labels for new training. They do not resume an existing run. GPU training has not been rerun for this release.
