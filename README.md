# Holosoma

Holosoma (Greek: "whole-body") is a comprehensive humanoid robotics framework for training and deploying reinforcement learning policies on humanoid robots, as well as motion retargeting. Supports locomotion (velocity tracking) and whole-body tracking tasks across multiple simulators (IsaacGym, IsaacSim, MJWarp, MuJoCo) with algorithms like PPO and FastSAC.

## Features

- **Multi-simulator support**: IsaacGym, IsaacSim, MuJoCo Warp (MJWarp), and MuJoCo (inference only)
- **Multiple RL algorithms**: PPO and FastSAC
- **Robot support**: Unitree G1 and Booster T1 humanoids
- **Task types**: Locomotion (velocity tracking) and whole-body tracking
- **Sim-to-sim and sim-to-real deployment**: Shared inference pipeline across simulation and real robot control
- **Motion retargeting**: Convert human motion capture data to robot motions while preserving interactions with objects and terrain
- **Wandb integration**: Video logging, automatic ONNX checkpoint uploads, and direct checkpoint loading from Wandb

## Repository Structure

```
src/
├── holosoma/              # Core training framework (locomotion & whole-body tracking)
├── holosoma_inference/    # Inference and deployment pipeline
└── holosoma_retargeting/  # Motion retargeting from human motion data to robots
```

## Documentation

- **[Training Guide](src/holosoma/README.md)** - Train locomotion and whole-body tracking policies in IsaacGym/IsaacSim
- **[Inference & Deployment Guide](src/holosoma_inference/README.md)** - Deploy policies to real robots or evaluate in MuJoCo simulation
- **[Retargeting Guide](src/holosoma_retargeting/README.md)** - Convert human motion capture data to robot motions

## Quick Start

### Setup

Choose the appropriate setup script based on your use case:

```bash
# For IsaacGym training
bash scripts/setup_isaacgym.sh

# For IsaacSim training
# Requires Ubuntu 22.04 or later due to IsaacSim dependencies
bash scripts/setup_isaacsim.sh

# For MJWarp training and MuJoCo simulation (inference)
bash scripts/setup_mujoco.sh

# For inference/deployment
bash scripts/setup_inference.sh

# For motion retargeting
bash scripts/setup_retargeting.sh
```

### Training

Train a G1 robot with FastSAC on IsaacGym:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1
```

> **Note:** For headless servers, see the [training guide](src/holosoma/README.md#video-recording) for video recording configuration.

See the [Training Guide](src/holosoma/README.md) for more examples and configuration options.

### No Silent Fallbacks

Do not add fallback behavior that silently substitutes missing or invalid data,
geometry, checkpoints, config, logging, or distributed-training state. If a
required artifact cannot be loaded, the code must fail loudly with a clear error.

Any fallback must be explicitly approved by the project owner before it is added,
and the code path must make that fallback visible in logs or UI so it cannot be
mistaken for the intended data or behavior.

Policy initialization is strict. `--training.policy-init-checkpoint` must match
the current actor state dict exactly, including all keys and tensor shapes.
Partial actor initialization is forbidden: do not load only the compatible
subset of a checkpoint and continue training. If the checkpoint architecture
does not match, start without policy init or use a checkpoint trained with the
exact same actor architecture.

### Required Realmesh Debug Sequences

The AS realmesh debug rollout/original comparison depends on a fixed set of
sequences. Do not prune, rename, replace with cuboid fallback geometry, or
regenerate these realmesh debug sequences without explicit owner approval.

The current comparison viewer uses:

```text
realmesh rollout bank:
data/ds_as_data/debug39_realmesh_rollout_u8udzw0u_model05000_retake4gpu_20260706_0205_target/_single_slot_motion_bank

realmesh rollout contact export:
data/ds_as_data/debug39_realmesh_rollout_u8udzw0u_model05000_retake4gpu_20260706_0205_target/contact_export_from_teacher_realmesh_rollout

original realmesh packed motion bank:
/nfs/zzzihanw/ds_as_data/debug/_single_slot_motion_bank
```

These debug banks currently resolve their object mesh assets through
`/nfs/zzzihanw/ds_as_data/debug/objects/*/object_mesh_yup.obj`. The rollout
target directory is not a self-contained geometry package; if `/nfs/zzzihanw`
is unavailable, the strict viewer must fail instead of substituting boxes.

The convex-hull-all-mesh version of the 30 rollout debug sequences is stored at:

```text
/nfs/zzzihanw/prism-debug/debug39_realmesh_rollout_u8udzw0u_model05000_retake4gpu_20260706_0205_target_convexhull_allmesh
```

In that package, URDF visual meshes, URDF collision meshes, clip-map
`object_mesh_path`, clip-map `object_collision_mesh_path`, motion-bank metadata,
and contact-export metadata all point to `objects_convex_hull/*.obj`. It is
intended to fail validation if any object falls back to real mesh or cuboid
geometry.

The 30 required debug comparison clip ids are:

```text
scaledown__any_ball_24
scaledown__any_ball_26
scaledown__any_ball_28
scaledown__any_barrel_25
scaledown__any_bin_16
scaledown__any_bin_17
scaledown__any_bin_19
scaledown__any_bin_20
scaledown__any_bin_21
scaledown__any_bin_22
scaledown__any_bin_24
scaledown__any_bin_25
scaledown__any_bin_27
scaledown__any_bin_28
scaledown__any_bin_29
unscale__any_ball_24
unscale__any_ball_29
unscale__any_bin_16
unscale__any_bin_17
unscale__any_bin_18
unscale__any_bin_19
unscale__any_bin_20
unscale__any_bin_22
unscale__any_bin_23
unscale__any_bin_24
unscale__any_bin_25
unscale__any_bin_27
unscale__any_bin_28
unscale__any_bin_29
unscale__any_bin_31
```

To replay the strict rollout-vs-original comparison in `viser`:

```bash
PYTHONPATH=src python3 -m holosoma.debug_rollout_grid_viewer \
  --data-root data/ds_as_data/debug39_realmesh_rollout_u8udzw0u_model05000_retake4gpu_20260706_0205_target/contact_export_from_teacher_realmesh_rollout \
  --host 0.0.0.0 \
  --port 7082 \
  --group-size 1 \
  --cols 1 \
  --spacing 3.6 \
  --playback-fps 30 \
  --robot-mode urdf \
  --object-visual surface-points \
  --object-point-count 4000 \
  --original-motion-dir /nfs/zzzihanw/ds_as_data/debug/_single_slot_motion_bank
```

### Checkpoint Uploads

Training runs that use W&B logging should upload `.pt` checkpoints by default.
Do not set `HOLOSOMA_SKIP_WANDB_CHECKPOINT_UPLOAD` or
`HOLOSOMA_SKIP_WANDB_FILE_UPLOAD` for normal training launches. The multi-node
`batch_ne.sh` launcher clears those variables before starting training, and its
default checkpoint save interval is `SAVE_INTERVAL=500`.

### Pure RL Bootstrap

Use `debug_pure_rl_bootstrap.sh` for pure PPO warm-up runs. This launcher is
separate from distillation launchers and must not be used with teacher policy,
DAgger, box-policy init, or checkpoint resume.

Stage A is the default early-learning setup: fixed `3e-4` actor/critic learning
rates, reduced exploration noise, no entropy bonus, two PPO epochs per rollout,
`clip_param=0.1`, `max_grad_norm=0.5`, start from timestep zero, reduced reset
noise, no adaptive timestep sampler, no push randomization, relaxed
tracking/object termination thresholds, and softer contact guidance with lower
contact weights. It is intended to make early episodes long enough for PPO to
learn useful locomotion and pickup behavior without the action-rate/critic
explosion seen with the earlier aggressive `1e-3` setting.

Later stages are explicit:

```bash
PURE_RL_BOOTSTRAP_STAGE=B ./debug_pure_rl_bootstrap.sh
PURE_RL_BOOTSTRAP_STAGE=C ./debug_pure_rl_bootstrap.sh
```

Stage B tightens thresholds and guidance weights. Stage C restores the formal
termination thresholds, original contact weights, force-gated contact guidance,
and push randomization. The launcher still delegates to `debug_pure_rl.sh`, so
missing real mesh assets or accidental checkpoint init must fail loudly.

For multi-GPU pure-RL bootstrap runs, keep the launcher defaults:
`HOLOSOMA_GLOO_SMALL_COLLECTIVES=1`, `HOLOSOMA_GLOO_GRAD_REDUCE=1`,
`HOLOSOMA_SYNC_AFTER_GRAD_ALLREDUCE=1`, and
`HOLOSOMA_SYNC_EACH_ITERATION=0`. This avoids mixing NCCL gradient all-reduce
with Gloo/CPU small collectives in the PPO update, which previously caused
silent rank waits around iteration boundaries. On 2026-07-09, the verified
8-GPU Stage-A run used 2048 envs/GPU, the 5-layer actor
`[2048,1024,512,256,128]`, no distillation, and no checkpoint init.

### CoRL Baseline Real-Data Training

All training data must be read from repo-local relative paths under `data/`.
Do not point training launchers directly at `/nfs`. Before launching the CoRL
real-data baseline, copy the prepared data into this repository:

```bash
bash cp_baseline.sh
```

This installs the two required banks under `data/corl_numbers/`:

```text
data/corl_numbers/omomo_z0p4_nofoot_bimanual161_training_ready
data/corl_numbers/behave_z0p4_first_lift_run_bimanual56_w_obj_training_ready
```

Then launch the training wrapper:

```bash
./corl_numbers/train_as_general_realdata.sh
```

The wrapper builds its generated union view under `data/corl_numbers/realdata_union`
and delegates to `train_as_general.sh`. It will fail if the source data is not
under `data/`.

### Quick Demo

We provide scripts to run the complete pipeline: (data downloading and processing for LAFAN), retargeting, data conversion, and whole-body tracking policy training.

```bash
# Run retargeting and whole-body tracking policy training using OMOMO data
bash demo_scripts/demo_omomo_wb_tracking.sh

# Run retargeting and whole-body tracking policy training using LAFAN data
bash demo_scripts/demo_lafan_wb_tracking.sh
```

### Deployment & Evaluation

After training, deploy your policies:

- **Real Robot**: See [Real Robot Locomotion](src/holosoma_inference/docs/workflows/real-robot-locomotion.md) or [Real Robot WBT](src/holosoma_inference/docs/workflows/real-robot-wbt.md)
- **MuJoCo Simulation**: See [Sim-to-Sim Locomotion](src/holosoma_inference/docs/workflows/sim-to-sim-locomotion.md) or [Sim-to-Sim WBT](src/holosoma_inference/docs/workflows/sim-to-sim-wbt.md)
- **G1 dance replay with existing `viser`**: Run `bash ./sim2sim_dancing_viser.sh` from the repo root to launch the packaged MuJoCo + inference dance workflow. This path disables the virtual gantry by default, reuses the existing inference-side `viser`, exposes `Reset sim + motion` plus `Auto reset on motion end`, and currently renders the robot with a grid rather than the full MuJoCo scene.
- **Depth/perception policy checks**: When comparing actor or critic inputs, do not infer depth usage from `module_dict.actor.input_dim` alone. See the [Perception Policy Input Audit](src/holosoma_inference/README.md#perception-policy-input-audit) checklist.

Or browse all deployment options in the [Inference & Deployment Guide](src/holosoma_inference/README.md).

### Demo Videos

Watch real-world deployments of Holosoma policies *(click thumbnails to play)*

<table>
  <tr>
    <th>G1 Locomotion</th>
    <th>T1 Locomotion</th>
    <th>G1 Dancing</th>
  </tr>
  <tr>
    <td width="33%">
      <a href="https://youtu.be/YYMgj5BDIMI">
        <img src="https://img.youtube.com/vi/YYMgj5BDIMI/hqdefault.jpg" width="100%" alt="▶ G1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/Q6rNHJZ2a6Y">
        <img src="https://img.youtube.com/vi/Q6rNHJZ2a6Y/hqdefault.jpg" width="100%" alt="▶ T1 Locomotion">
      </a>
    </td>
    <td width="33%">
      <a href="https://youtu.be/ouPk69_eFfE">
        <img src="https://img.youtube.com/vi/ouPk69_eFfE/hqdefault.jpg" width="100%" alt="▶ G1 Dancing">
      </a>
    </td>
  </tr>
</table>


## Issue Reporting

We welcome feedback and issue reports to help improve holosoma. Please use issues to:

- Report bugs and technical issues
- Request new features

## Support

If you need help with anything aside from issues feel free to join our [discord server](https://discord.gg/TPupMvpqHc).

Use the discord to discuss larger plans and other more involved problems.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## Citation

If you use Holosoma in your research, please cite it according to the "Cite this repository" panel on the right sidebar of the Github repo.

## License

This project is licensed under the Apache-2.0 License.
