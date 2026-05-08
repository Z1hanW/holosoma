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

### MuJoCo WBT Box Rollout Debug Log

This section records the debugging path for G1 WBT box lifting in MuJoCo, starting from the motion-init/reset patch. The intent is to make the current behavior easy to reproduce and to explain why each change was made.

#### 1. Motion-Init Reset Was Added First

The first issue was that MuJoCo reset and backspace reset could put the robot/object at the origin or at a generic robot default instead of the first frame of the selected motion file. Motion-init mode fixes that.

- `mj_env.sh` is manual by default.
- `bash mj_env.sh box_75 --motion-init` enables motion-init mode.
- Motion-init reads frame 0 from the selected motion file and applies:
  - robot root position and orientation,
  - robot joint pose,
  - object position and orientation.
- MuJoCo `qpos0` is also updated for robot root and object root, so backspace reset returns to the same motion-frame-0 state.
- `HOLOSOMA_MUJOCO_HOLD_MOTION_INIT_UNTIL_COMMAND=1` is enabled in motion-init mode so the robot/object do not drift before rollout commands arrive.
- An orange origin marker is added to make origin/reset mistakes visible.

For `box_75`, the expected motion-init state is not world origin. The robot and object should initialize at the position specified by `data_demo/box_75.npz`.

#### 2. The Launch Scripts Were Split Into Manual And Debug Paths

After reset was reliable, the scripts were made explicit:

```bash
# Manual environment. No motion-init unless requested.
bash mj_env.sh box_75

# Manual rollout. Latest tvtwx4to unless overridden.
bash mj_ro.sh

# Fully automated debug rollout. Always uses motion-init environment setup.
bash mj_debug.sh --run tvtwx4to --checkpoint latest --duration 120s
```

Useful variants:

```bash
# Motion-init environment, but rollout still launched manually.
bash mj_env.sh box_75 --motion-init

# Specific checkpoint.
bash mj_debug.sh --run tvtwx4to --checkpoint 20000 --duration 120s

# Another W&B run. This one is expected to use motion-derived/root command.
bash mj_debug.sh --run w5qostjn --checkpoint latest --duration 120s
```

`mj_env.sh` launches `robot:g1-29dof-w-object`, `camera:single_d435i_depth`, and `image_server:mujoco_d435i`. The virtual gantry is disabled.

`mj_debug.sh` starts `mj_env.sh` in `hsmujoco`, waits for the image server and `/dev/shm/depth_img_shm`, checks the depth buffer, then starts `mj_ro.sh` in `hsinference`. Logs go to `artifacts/mj_debug_{run_id}_.../`.

#### 3. Split Sim-State And Lowcmd Were Added

Motion-init rollout needs the inference side to see the simulator's actual state. The debug path therefore uses split ZMQ sim-state and lowcmd channels when motion-init is enabled.

- `mj_env.sh --motion-init` publishes sim state and listens for lowcmd/control requests.
- `mj_ro.sh --motion-init` reads simulator state and sends lowcmd over ZMQ.
- The inference policy augments root pose, root velocity, DOF position, and DOF velocity from sim state.
- The simulator publishes the measured ref-body pose so WBT can use MuJoCo's actual torso/reference orientation.
- Root velocity in sim-state is now produced with MuJoCo `mj_objectVelocity`, then inference rotates it into the body frame for `base_lin_vel` and `base_ang_vel`.

This avoided mismatches between raw freejoint velocity, measured body velocity, and the WBT observation terms used in training.

#### 4. W&B Checkpoint And ONNX Handling Were Made Explicit

`mj_ro.sh` now handles W&B checkpoint selection and ONNX setup directly.

- It accepts run ids, W&B URLs, `wandb://` paths, `latest`, numeric checkpoints like `20000`, or explicit ONNX filenames.
- Latest checkpoint is selected by the largest `model_{step}.onnx` step.
- Downloaded checkpoints are renamed to `/tmp/{run_id}_{model_name}.onnx`, for example `/tmp/tvtwx4to_model_20000.onnx`.
- The renamed checkpoint path is asserted after download to avoid confusing stale `/tmp/model_*.onnx` files.
- ONNX motion constants are patched from the selected motion `.npz`.
- Patched ONNX motion constants are asserted against the motion file.
- The observation config is adapted to the ONNX actor input dimensions and metadata, then asserted against ONNX input shapes.

Run-specific behavior:

- `tvtwx4to` must lift with zero sparse/root command. `mj_ro.sh` forces `HOLOSOMA_FORCE_ZERO_SPARSE_ROOT_COMMAND=1` for this run.
- `w5qostjn` is not forced to zero command; it is expected to use the motion-derived/root command path.
- For `tvtwx4to`, policy debug should show `rootca_absmax=0.0`.

#### 5. WBT Policy Initialization Was Cleaned Up

Several WBT rollout mismatches were fixed in the inference policy.

- `motion_yaw_offset` and `robot_yaw_offset` are initialized and reset consistently, fixing the earlier missing-attribute crash.
- Motion-init auto-start waits for the low state to match the expected motion-frame-0 yaw and joint pose before starting policy control.
- `box_75` uses `HOLOSOMA_POLICY_MOTION_INDEX_OFFSET=1` to align the policy motion index with the motion sequence.
- Per-joint action scaling is read from ONNX metadata when training requested `action_scales_by_effort_limit_over_p_gain`.
- KP/KD are loaded from the WBT training-aligned config/metadata path and validated by shape.

Control behavior:

- `q_target` is not clipped in inference.
- Only PD torque is clipped before applying MuJoCo actuator torques.

#### 6. Depth Was Aligned To The Checkpoint

The WBT checkpoints expect the D435i depth path.

- The MuJoCo path uses `image_server:mujoco_d435i`.
- The debug script expects depth shared memory shape `(1, 1, 58, 87)`.
- `min_valid_depth=0.15` matches training invalid-depth handling.
- `near_clip` remains the clamp floor and is not used as the invalid-depth threshold.
- `mj_debug.sh` fails early if depth shared memory has the wrong byte size, non-finite values, or constant/zero values before rollout.

#### 7. Object And Contact Physics Were Fixed Last

After observations and commands were aligned, the remaining failure was physics: the policy contacted the box but did not lift it.

The important discovery was the object mass. The URDF box mass is `0.1kg`, but MuJoCo telemetry showed about `397N` support force at rest, which means the box was effectively around `40kg`. The non-colliding visual mesh was still contributing default MuJoCo mass.

Fixes:

- The object visual mesh geom is now massless.
- Only the collision geom carries the object mass from the URDF.
- Training-style object contact pairs are added for carry bodies such as hands, wrists, elbows, shoulders, and torso against the `largebox` collision geom.
- Hand collision geometry was aligned back to the reference values used by the original Holosoma assets.

After this, the rest support force became about `0.98N`, matching the `0.1kg` object.

#### 8. Debug Telemetry Was Added To Verify Each Hypothesis

`HOLOSOMA_MJ_DEBUG_LIFT_TELEMETRY=1` enables debug logs without changing normal rollout behavior.

- `LiftTelemetry`: object position, lift delta, max lift delta, object/robot/terrain contacts, contact geoms, robot pose, and contact force summaries.
- `LowCmdTelemetry`: `q_target`/`q_actual` ranges, tracking error, raw torque, clipped torque, and torque saturation count.
- `HOLOSOMA_POLICY_DEBUG_INPUT_PATH`: policy-side JSONL with observation stats, sparse root command, depth quantiles, policy action stats, scaled action stats, and `q_target` stats.

#### 9. Current Verified Result

Latest local verification:

```bash
HOLOSOMA_DEBUG_DURATION=120s HOLOSOMA_POLICY_DEBUG_INPUT_LIMIT=300 \
  bash mj_debug.sh --run tvtwx4to --checkpoint latest --duration 120s
```

Result from `artifacts/mj_debug_tvtwx4to_20260507_233657`:

- `rootca_absmax=0.0`, confirming `tvtwx4to` used zero sparse/root command.
- Object support force at rest was about `0.98N`, matching the `0.1kg` URDF object.
- The box lifted to about `z=0.932m`.
- `max_dz=0.764m`.
- Sampled `LowCmdTelemetry` had no torque saturation.

The main improvements were motion-frame-0 reset correctness, split sim-state/lowcmd rollout, ONNX/checkpoint assertions, depth alignment, no `q_target` clipping, corrected root velocity publishing, and the massless visual object fix.

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
