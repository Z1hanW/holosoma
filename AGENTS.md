# Agent Notes

## Style Requirement

- Be precise and concise.
- User requirement: "precise 且 concise"; answer directly, with concrete commands and paths.
- State concrete commands, paths, branches, run IDs, and dates when they matter.
- Do not add motivational filler or long narrative.
- Inspect the repo before editing. Keep edits scoped.
- Do not commit secrets, W&B keys, logs, or large generated data.

## Skill Map

- `terrain-traversal`: slope, stairs, heightmap, multi-terrain WBT, foothold penalty, depth-student distillation for terrain traversal.
- `carry-object`: HOI/object tracking, TokenHSI object policies, object-aware depth distillation. Use `hoi-track-readme.md` for the current carry-object notes.
- Current conversation is mainly `terrain-traversal`.

## Current Repo State

- Branch: `distill-ppo-teacher`.
- Git remote: `https://github.com/Z1hanW/holosoma.git`.
- Pushed branch: `origin/distill-ppo-teacher`.
- Local untracked `data/` is generated and large; do not commit it by default.
- Main terrain docs are in `README.md`, especially `CSP WBT Stair45 Debug Runs`, `CSP Multi-Terrain Heightmap WBT`, and `Depth-Dist`.
- Main carry-object docs are in `hoi-track-readme.md`.

## Data Locations

### Terrain-Traversal Data

Use the local repo copy for scripts and default launches:

```text
/home/ubuntu/FAR/holosoma/crisp_stairs
```

The canonical NFS copy that corresponds to this local `crisp_stairs` subset is:

```text
/nfs/zzzihanw/crisp_stairs
```

This is the same cleaned 16-clip subset used by the local repo copy. Some `.obj` files may differ by harmless trailing whitespace/newline between local and NFS; compare semantics, not raw byte identity, for text OBJ files.

Direct mapping:

- Local motion dir: `crisp_stairs/___crisp_clean_motion`
- NFS motion dir: `/nfs/zzzihanw/crisp_stairs/___crisp_clean_motion`
- Local terrain dir: `crisp_stairs/___crisp_clean_geometry`
- NFS terrain dir: `/nfs/zzzihanw/crisp_stairs/___crisp_clean_geometry`
- Local manifest: `crisp_stairs/terrain_traversal_manifest.json`
- NFS manifest: `/nfs/zzzihanw/crisp_stairs/terrain_traversal_manifest.json`

The older/source dataset root for this cleaned subset is:

```text
/nfs/zzzihanw/ds_crisp_data_vggtomega_crisp_terrain_g1
```

The raw VGGT/Omega source roots are:

```text
/nfs/zzzihanw/FAR_stairs_vggt_omega_live/hmr_vggt_omega
/nfs/zzzihanw/FAR_stairs_vggt_omega_live/scene_vggt_omega_consistent_camera_min1
```

Do not train directly from the raw VGGT/Omega roots. Use `crisp_stairs` or `/nfs/zzzihanw/crisp_stairs`; those contain the cleaned `.npz` motions and exported `.obj` terrain used by current scripts.

The current 16 terrain-traversal clips are:

```text
stair_45
stair_3
56_outdoor_stairs_up_down
78_outdoor_stairs_up_down
stair_48
stair_50
stair_51
stair_53
stair_54
stair_61
stair_69
stair_75
stair_78
stair_83
stair_95
stair_101
```

Example pair:

```text
local motion:  crisp_stairs/___crisp_clean_motion/stair_45.npz
local terrain: crisp_stairs/___crisp_clean_geometry/stair_45.obj
NFS motion:    /nfs/zzzihanw/crisp_stairs/___crisp_clean_motion/stair_45.npz
NFS terrain:   /nfs/zzzihanw/crisp_stairs/___crisp_clean_geometry/stair_45.obj
```

Fused multi-terrain outputs are generated locally by `scripts/fuse_crisp_stairs_multiterrain.py`:

```text
crisp_stairs/_fused/motion_stairs_16_multiterrain.npz
crisp_stairs/_fused/motion_stairs_16_multiterrain.obj
crisp_stairs/_fused/motion_stairs_16_multiterrain.json
```

### Carry-Object Data

Canonical carry-object data is:

```text
/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry
```

This is separate from terrain-traversal `crisp_stairs`. Do not confuse it with local generated `data/` debug banks.

## Carry-Object / HOI Tracking Reproduction

Use `hoi-track-readme.md` as the source of truth for current HOI tracking details.

### 1. Data And Object Assets

Canonical carry-object data is the OMOMO largebox set under `/nfs/zzzihanw`:

```bash
/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry
```

This is the dataset used by the successful HOI tracking run. Do not substitute local `data/` debug banks unless intentionally doing a smoke test.

Required checks:

```bash
find /nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry -maxdepth 1 -name '*.npz' | wc -l
test -f /nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry/_clip_object_urdf_map.json
test -f holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf
```

Expected data state:

- Exactly 62 `.npz` clips in the current `/nfs/zzzihanw` copy.
- Object map: `/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry/_clip_object_urdf_map.json`.
- Object map schema: top-level `clips` dict, one entry per clip stem.
- Object URDF: `holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf`.
- Object: `largebox`.
- Example clip stem: `sub10_largebox_032_mj_w_obj`.

### 2. Current Successful Expert Setup

Current reference run:

- W&B: `https://wandb.ai/zihanw22/holosomatest/runs/rrfbwfnq`
- Project: `zihanw22/holosomatest`
- Exp: `g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target`
- Architecture: TokenHSI transformer, not MLP.
- Nodes: `10.0.90.122`, `10.0.123.134`.
- Scale: 2 nodes x 8 GPUs x 4096 envs/GPU = 65536 envs.
- Data: OMOMO carry path above.

Launch equivalent run:

```bash
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

NODE_HOSTS="10.0.90.122 10.0.123.134" \
NNODES=2 \
GPUS_PER_NODE=8 \
ENVS_PER_GPU=4096 \
EXP_NAME=g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target \
MOTION_DIR=/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry \
OBJECT_URDF_MAP=/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry/_clip_object_urdf_map.json \
WANDB_ENTITY=zihanw22 \
WANDB_PROJECT=holosomatest \
RUN_NAME="omomo62_object_tokenhsi_nexttarget_2x8_16gpu_4096envpergpu_65536env_40000iter_${TIMESTAMP}" \
LOGGER_GROUP="omomo62-object-tokenhsi-nexttarget-4096envpergpu-2x8-${TIMESTAMP}" \
SESSION="object_tokenhsi_nexttarget_omomo62_2x8_4096envpergpu_${TIMESTAMP}" \
REMOTE_REPO=/home/ubuntu/FAR/holosoma_object_tokenhsi_20260704_211351 \
SYNC_SCRIPT=1 \
KILL_EXISTING=0 \
./object_tokenhsi_multinode.sh
```

Use W&B project `holosomatest`, not `carry-any`.

### 3. Required Log Checks

After launch, verify nodes directly:

```bash
for h in 10.0.90.122 10.0.123.134; do
  ssh "$h" "tmux ls | grep ${SESSION} || true"
  ssh "$h" "pgrep -af 'train_agent|torchrun' || true"
  ssh "$h" "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
done
```

Remote logs must show:

- `Loaded object URDF`
- `MultiMotionLoader: found 62 .npz files`
- `Loaded clip-object metadata map`
- `Fixed env-to-clip assignment`
- W&B run under `zihanw22/holosomatest`

### 4. Policy Inputs And Rewards

Current TokenHSI is modality-token, not per-joint-token:

- robot/proprio token: `actor_obs`
- height scan token: `actor_height_scan`
- object token: `actor_object_obs`

Object observation includes current object state, current target, and next-frame target:

- `obj_pos_b`, `obj_ori_b`, `obj_lin_vel_b`
- `obj_ref_pos_b`, `obj_ref_ori_b`, `obj_ref_lin_vel_b`
- `obj_ref_pos_next_b`, `obj_ref_ori_next_b`, `obj_ref_lin_vel_next_b`

Object rewards to confirm:

- `Episode/rew_object_global_ref_position_error_exp`
- `Episode/rew_object_global_ref_orientation_error_exp`

### 5. Object-Aware Depth Distillation

Use a HOI expert checkpoint, for example:

```bash
wandb://zihanw22/holosomatest/rrfbwfnq/model_08000.pt
```

Single-node smoke run:

```bash
TIMESTAMP=$(date -u +%Y%m%d_%H%M%S)

TEACHER_CHECKPOINT=wandb://zihanw22/holosomatest/rrfbwfnq/model_08000.pt \
WANDB_ENTITY=zihanw22 \
WANDB_PROJECT=holosomatest \
DISTILL_TAG=hoi_omomo62_rrfbwfnq_model08000_objdepth \
NUM_GPUS=1 \
ENVS_PER_GPU=1024 \
NUM_ITERATIONS=20000 \
SAVE_INTERVAL=1000 \
RUN_NAME="hoi_depth_student_rrfbwfnq_model08000_objdepth_1gpu_${TIMESTAMP}" \
SESSION="hoi_depth_student_rrfbwfnq_model08000_objdepth_1gpu_${TIMESTAMP}" \
./csp_depth_distill.sh
```

## Terrain-Traversal Reproduction

### 1. Stair45 Blind Debug

Use blind WBT first to verify motion and OBJ terrain without heightmap:

```bash
cd /home/ubuntu/FAR/holosoma
./csp_blindwbt.sh
```

Key defaults:

- 8 GPUs.
- 4096 envs/GPU.
- checkpoint save interval `1000`.
- local motion: `crisp_stairs/___crisp_clean_motion/stair_45.npz`.
- local terrain: `crisp_stairs/___crisp_clean_geometry/stair_45.obj`.
- NFS motion: `/nfs/zzzihanw/crisp_stairs/___crisp_clean_motion/stair_45.npz`.
- NFS terrain: `/nfs/zzzihanw/crisp_stairs/___crisp_clean_geometry/stair_45.obj`.
- true IsaacSim/PhysX rollout.

### 2. Stair45 Heightmap Debug

Then run the heightmap-aware single-stair version:

```bash
NUM_GPUS=4 ENVS_PER_GPU=4096 ./csp_heightmapwbt.sh
```

This uses `exp:g1-29dof-wbt-height-scan`, enables `simulator.config.height_scanner`, adds `height_scan` to actor/critic obs, and adds a flat floor patch under the finite OBJ terrain.

### 3. Multi-Terrain Heightmap Tracking

Build fused CRISP stair assets:

```bash
python scripts/fuse_crisp_stairs_multiterrain.py
```

Expected outputs:

- `crisp_stairs/_fused/motion_stairs_16_multiterrain.npz`
- `crisp_stairs/_fused/motion_stairs_16_multiterrain.obj`
- `crisp_stairs/_fused/motion_stairs_16_multiterrain.json`

These are generated local training artifacts. The input pairs come from local `crisp_stairs/___crisp_clean_*`, corresponding to `/nfs/zzzihanw/crisp_stairs/___crisp_clean_*`.

Run training:

```bash
./csp_multiterrain_heightmapwbt.sh
```

Important implementation detail: the fused NPZ must carry `terrain_origins`. On reset, `MotionCommand` writes the sampled motion's terrain origin into `scene.env_origins`, `simulator.env_origins`, and locomotion terrain state so each motion aligns with its translated terrain tile.

### 4. Zhen Penalty

Enable the far-tracking-style foothold support penalty only when intentionally testing it:

```bash
ENABLE_ZHEN_PENALTY=1 ZHEN_PENALTY_WEIGHT=-10.0 ./csp_multiterrain_heightmapwbt.sh
```

It uses foot RayCaster sensors and penalizes contacting soles that are not sufficiently supported by the static triangle-mesh terrain. The reward term exists with weight `0.0` by default.

### 5. 32-GPU Multi-Node Tracking

Validated remote topology excludes local `10.0.73.59`:

```bash
NODE_HOSTS="10.0.74.86 10.0.100.200 10.0.72.226 10.0.90.122" \
./csp_multinode_multiterrain_heightmapwbt.sh
```

This is `4 nodes x 8 GPUs x 4096 envs/GPU = 131072 envs`.

After launch, verify all nodes:

```bash
for h in 10.0.74.86 10.0.100.200 10.0.72.226 10.0.90.122; do
  echo "== $h =="
  ssh "$h" 'tmux ls | grep csp_multinode_heightmapwbt || true; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | head -n 8'
done
```

Validated W&B reference: `https://wandb.ai/zihanw22/holosomatest/runs/btoe97gr`.

### 6. Depth-Dist For Terrain

Depth-dist is hybrid DAgger + PPO with true physics rollout, not pure BC and not kinematics replay.

Student input contract:

- `root_target_xy_yaw`
- projected gravity
- base angular velocity
- joint position
- joint velocity
- last action
- processed depth image

Single-node:

```bash
TEACHER_CHECKPOINT=logs/holosomatest/.../model_01000.pt ./csp_depth_distill.sh
```

Two remote nodes:

```bash
NODE_HOSTS="10.0.100.200 10.0.72.226" \
DISTILL_TAG=multiterrain_teacher09999 \
TEACHER_CHECKPOINT=logs/holosomatest/.../model_09999.pt \
./csp_multinode_depth_distill.sh
```

Visualize a distilled student checkpoint:

```bash
PYTHONPATH=src/holosoma ./scripts/viser_depth_student_physics_eval.py \
  --checkpoint /path/to/student_XXXXXXX.pt \
  --num-envs 1 \
  --env-id 0 \
  --port 2106 \
  --gui-command \
  --depth-hits \
  --no-red-points \
  --no-motion-ref \
  --disable-randomization
```

Use `student_*.pt`, not teacher `model_*.pt`.

## Physics Inference / Visualization

Use `VIS_README.md` for expanded commands. The rule is task-wise and checkpoint-wise:

- `model_*.pt`: expert / teacher physics rollout via `scripts/viser_current_physics_rollout.py`.
- `student_*.pt`: depth-student physics rollout via `scripts/viser_depth_student_physics_eval.py`.

Common local setup:

```bash
cd /home/ubuntu/FAR/holosoma
source scripts/source_inference_setup.sh
export PYTHONPATH=src/holosoma
```

### Carry-Object

Expert / teacher inference for object TokenHSI, e.g. `rrfbwfnq`:

```bash
python scripts/viser_current_physics_rollout.py \
  --checkpoint wandb://zihanw22/holosomatest/rrfbwfnq/model_08000.pt \
  --port 2099 \
  --env-id 0 \
  --sequence-envs 64 \
  --disable-randomization \
  --no-red-points
```

This is the correct path for object mesh, object motion, reference object, and Viser `Sequence` switching. Before debugging visuals, confirm:

- data: `/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry`
- object map: `/nfs/zzzihanw/amass/converted_res/object_interaction/omomo_carry/_clip_object_urdf_map.json`
- object URDF: `holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf`
- exp: `g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target`

Carry-object depth student inference:

```bash
python scripts/viser_depth_student_physics_eval.py \
  --checkpoint /path/to/student_XXXXXXX.pt \
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

Use `--joystick` instead of `--gui-command` only for local gamepad control.

### Terrain-Traversal

Terrain expert / teacher inference:

```bash
python scripts/viser_current_physics_rollout.py \
  --checkpoint /path/to/model_XXXXX.pt \
  --port 2099 \
  --env-id 0 \
  --sequence-envs 16 \
  --disable-randomization \
  --red-points
```

For W&B:

```bash
python scripts/viser_current_physics_rollout.py \
  --checkpoint wandb://zihanw22/holosomatest/btoe97gr/model_XXXXX.pt \
  --port 2099 \
  --env-id 0 \
  --sequence-envs 16 \
  --disable-randomization \
  --red-points
```

Terrain depth-student inference:

```bash
python scripts/viser_depth_student_physics_eval.py \
  --checkpoint /path/to/student_XXXXXXX.pt \
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

For sampled rollout switching in depth-student eval, increase `--num-envs` and change `--env-id`.

Depth-student input contract for both tasks:

- `root_target_xy_yaw`
- `projected_gravity`
- `base_ang_vel`
- `joint_pos`
- `joint_vel`
- `last_action`
- `processed_depth`

`projected_gravity` must come from current robot/base orientation; on real robot use IMU/base attitude.

## Common Mistakes

- Do not put carry-object W&B runs under `carry-any`; use `holosomatest`.
- Do not split nodes into separate experiments when the request is one multi-node experiment.
- Do not assume object tracking is active unless logs show object URDF and object map loaded.
- Do not assume policy sees next-frame target unless using `g1-29dof-wbt-w-object-height-scan-tokenhsi-next-target`.
- Do not call the current TokenHSI policy per-joint-token; it is modality-token.
- Do not trust W&B alone for remote liveness. Check SSH, tmux, process list, GPU utilization, and logs.
- Do not use cross-region 10.x private IPs for distributed training unless routing is explicitly configured.
- Do not call depth-student visualization with a teacher checkpoint.
- Do not treat heightmap/height-scan red points or target ghost G1 as student inputs.
- Do not enable `simulator.config.bridge` for GUI-command student visualization; use `--gui-command`.
- Do not show red heightmap points or target ghost by default in student visualization.
- Do not call pure BC runs successful terrain distillation; intended mode is hybrid DAgger + PPO.
- Do not enable the old global adaptive timestep sampler for fused multi-terrain unless intentionally debugging it. It collapsed distribution in run `h5xzojtc`.
- Do not use `num_rows/num_cols` terrain tiling for the fused OBJ; the fused mesh is already the full world.
- Do not use 512MB PhysX collision stack for fused stairs; use the 1GB setting from the multi-terrain script.
- Do not assume remote runs are alive from W&B alone. Check SSH, tmux, GPU utilization, and W&B heartbeat.
- Do not reuse a remote checkout with the wrong repository or stale editable install. Confirm `git rev-parse --short HEAD` and `python -c "import holosoma; print(holosoma.__file__)"`.
- Do not commit local generated `data/`, logs, checkpoints, W&B metadata, or credentials.
- Do not expose W&B API keys in docs, commands, commits, or chat.

## Run Status Notes

- `y9zvox6k`: old pure-BC path; useful ablation, not final terrain depth-dist.
- `bji6ir93`: failure mode for missing KL/std control.
- `vj7urlp6`: root-command multi-terrain student run; W&B marked `crashed`, last logged iteration `7025`, local synced checkpoint `student_0005000.pt`.
