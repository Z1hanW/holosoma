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

Or browse all deployment options in the [Inference & Deployment Guide](src/holosoma_inference/README.md).

### Real Depth Experiment Viewer

For the two-terminal real depth experiment, start the camera and policy as usual:

```bash
bash real_depth.sh
bash real_drop.sh
```

`real_drop.sh` and `real_debug.sh` start a Viser dashboard at `http://127.0.0.1:8080` and open it automatically when a desktop display is available. The dashboard shows the measured G1 pose, target-pose overlay, controller mode, joint-error telemetry, the exact normalized D435i policy input, and a depth point cloud.

Set `HOLOSOMA_REAL_VISER=0` to disable the viewer, `HOLOSOMA_REAL_VISER_PORT` to choose another port, or `HOLOSOMA_REAL_VISER_OPEN_BROWSER=0` to keep it from opening a browser. When running over SSH, forward the port (for example, `ssh -L 8080:localhost:8080 ...`) and open the same URL locally.

### Real Unitree Diagnostic-Pose Debug

To reproduce Unitree's suspended `L2+A` diagnostic position, keep the G1 supported on the gantry, put it in damping/develop mode with `L2+R2`, and run:

```bash
bash real_debug.sh
```

`real_debug.sh` automatically starts `real_depth.sh` with the `real_d435i_urdf` latency preset used for `0mcqao8k`, plus an isolated render-only MuJoCo server for simulation ground-truth depth. Viser shows separate **Real D435** and **MuJoCo sim GT** image/point-cloud panels. The GT server holds the all-zero diagnostic pose, uses the `box_75` motion-frame-0 scene by default, renders the 106x60 D435-URDF camera, then applies the same crop, cubic resize, depth range, and normalization to produce the 58x87 comparison view. It has no Unitree/DDS bridge and cannot command the real robot.

Set `HOLOSOMA_REAL_DEBUG_SIM_GT_MOTION` or `HOLOSOMA_REAL_DEBUG_SIM_GT_FRAME` to select another simulation reference. Set `HOLOSOMA_REAL_DEBUG_SIM_GT=0` to disable GT, `HOLOSOMA_REAL_DEBUG_IMAGE_SERVER_CONFIG` to override the real image-server preset, or `HOLOSOMA_REAL_DEBUG_DEPTH=0` when a separate real depth server is already running.

After the controller loads, verify the area is clear and press Enter to confirm stiff mode. It transitions from the measured joint positions to the straight, high-standing all-zero 29-DOF diagnostic posture over five seconds, then holds the legs and waist with the established high WBT stiff gains. Policy and motion activation are locked out in this configuration; joystick A/Start cannot leave the hold. Use `L1+R1` or `Ctrl+C` to exit. `HOLOSOMA_REAL_INTERFACE` and `HOLOSOMA_REAL_DEBUG_MODEL_PATH` override the default interface and initialization checkpoint.

### MuJoCo WBT Box Rollout Debug Log

This section keeps the remaining MuJoCo WBT box-lifting notes. The solved items moved to `solved_readme.sh` are sections 1, 2, 4, 5, and 9.

#### 3. Manual Path Stays Plain DDS

The normal MuJoCo rollout path should stay close to the plain commands:

```bash
python src/holosoma/holosoma/run_sim.py robot:g1-29dof-w-object camera:single_d435i_depth image_server:mujoco_d435i \
  --simulator.config.bridge.enabled=True

python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-wbt-distillation \
  --task.interface lo \
  --task.use-sim-time \
  --task.rl-rate 50 \
  --task.model-path /path/to/model.onnx
```

`mj_env.sh` and `mj_ro.sh` keep the command path plain DDS. MuJoCo still publishes clock/sim-state so policies that need measured root state can use it.

Motion-init debugging now uses DDS lowcmd by default:

```bash
bash mj_debug.sh --run tvtwx4to --checkpoint latest --duration 120s
```

Motion-init initializes the robot in the G1 WBT default standing pose at the motion-frame X/Y/yaw. The object still uses the motion-frame object pose.

#### 6. Depth Was Aligned To The Checkpoint

The WBT checkpoints expect the D435i depth path.

- The MuJoCo path uses `image_server:mujoco_d435i`.
- The debug script expects depth shared memory shape `(1, 1, 58, 87)`.
- `near_clip=0.3`, `far_clip=3.0`, and `min_valid_depth=0.15` match the checkpoint/distillation depth config.
- Image-server preprocessing now matches `/home/user/FAR/holosoma` distillation: crop/resize, clamp to `[near_clip, far_clip]`, apply `min_valid_depth`, then normalize to `[-0.5, 0.5]`.
- With the current `near_clip=0.3` and `min_valid_depth=0.15`, finite depth below `0.3m` becomes near depth (`-0.5`), while invalid/no-hit/far depth becomes far depth (`+0.5`).
- `mj_debug.sh` fails early if depth shared memory has the wrong byte size, non-finite values, or constant/zero values before rollout.



#### 8. Debug Telemetry Was Added To Verify Each Hypothesis

`HOLOSOMA_MJ_DEBUG_LIFT_TELEMETRY=1` enables debug logs without changing normal rollout behavior.

- `LiftTelemetry`: object position, lift delta, max lift delta, object/robot/terrain contacts, contact geoms, robot pose, and contact force summaries.
- `LowCmdTelemetry`: `q_target`/`q_actual` ranges, tracking error, raw torque, clipped torque, and torque saturation count.
- `HOLOSOMA_POLICY_DEBUG_INPUT_PATH`: policy-side JSONL with observation stats, sparse root command, depth quantiles, policy action stats, scaled action stats, and `q_target` stats.

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
