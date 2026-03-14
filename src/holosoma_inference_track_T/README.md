# Holosoma Inference

Policy inference for humanoid robot policies.

## Supported Policies

| Robot      | Locomotion | WBT |
|:----------:|:----------:|:---:|
| Unitree G1 | ✅         | ✅  |
| Booster T1 | ✅         | ❌  |

| Simulator               | Locomotion | WBT |
|:-----------------------:|:----------:|:---:|
| IsaacGym                | ✅         | ❌  |
| IsaacSim                | ✅         | ✅  |
| MuJoCo (inference only) | ✅         | ✅  |

- ✅ (full support)
- 🚧 (in progress/partial support)
- ❌ (no support)

## Quick Start

### Setup the Environment

```bash
cd ~/holosoma
bash scripts/setup_inference.sh             # Create a virtual environment with all dependencies
source scripts/source_inference_setup.sh    # Activate the virtual environment
```

### Choose Your Workflow

Select the appropriate workflow guide based on your setup:

#### Real Robot Deployment
- **[Real Robot Locomotion](docs/workflows/real-robot-locomotion.md)** - Run locomotion policies on physical Unitree G1 or Booster T1 robots
- **[Real Robot Whole Body Tracking](docs/workflows/real-robot-wbt.md)** - Run WBT policies on physical Unitree G1 robots

#### Simulation (MuJoCo)
- **[Sim-to-Sim Locomotion](docs/workflows/sim-to-sim-locomotion.md)** - Test locomotion policies in MuJoCo simulation
- **[Sim-to-Sim Whole Body Tracking](docs/workflows/sim-to-sim-wbt.md)** - Test WBT policies in MuJoCo simulation

Each workflow guide includes:
- Hardware/environment setup instructions
- Step-by-step commands
- Control references
- Deployment options (offboard/onboard/Docker)
- Troubleshooting tips

## MuJoCo Replay/Sim2Sim Helpers

For quick local MuJoCo runs with fixed motion/checkpoint presets:

```bash
# Replay motion only (robot-only)
bash src/holosoma_inference/replay_mujoco_robot_only.sh

# Replay motion only (robot + object)
bash src/holosoma_inference/replay_mujoco_robot_w_obj.sh

# Play policy from .onnx model via sim2sim (robot-only)
bash src/holosoma_inference/play_policy_mujoco_robot_only.sh /path/to/model.onnx

# Play policy from .onnx model via sim2sim (robot + object)
bash src/holosoma_inference/play_policy_mujoco_robot_w_obj.sh /path/to/model.onnx

```

Policy helpers are ONNX-first: when an ONNX path is provided (or discovered next to a `.pt`), they run `run_sim.py` + `run_policy.py` directly in MuJoCo. `.pt` is only used as fallback for `eval_agent.py` flow when ONNX is not available.

For WBT policy helper scripts, rollout auto-start is enabled by default with a staged sequence: stiff-hold at initialized start pose, then policy start, then motion clip start. Virtual gantry is disabled in helper defaults, and run-sim now also uses `--simulator.config.bridge.hold-until-first-command True` so physics stays pinned at the initialized frame until the first valid policy command arrives. When auto-start is enabled, the usual stiff-hold `Enter` confirmation prompt is skipped so rollout begins automatically in interactive terminals too. Disable auto-start with `AUTO_START=False` if you want manual key controls (`]`, then `s`). You can tune timing via `AUTO_START_STIFF_HOLD_SEC` (default `0.0`, one control tick) and `AUTO_START_STIFF_MAX_WAIT_SEC` (default `0.2`).

Policy helpers now also emit rollout pose CSV logs by default (`ROLLOUT_POSE_LOG`), with the same success criterion as replay (`on_floor && facing_up`). Success counting is gated by `ROLLOUT_SUCCESS_MIN_STEP` (default `2000`) so frame-0 initialization does not count as rollout success. Use `ROLLOUT_LOG_EVERY` to adjust cadence.
Run-sim startup now logs motion-initialization diagnostics at two checkpoints (`post_write_state_updates` and `post_episode_start`) with root/object position-quaternion errors and max joint error, so you can verify that robot/object are still at the intended start pose before rollout.
When `AUTO_LAUNCH=True`, helpers print a richer rollout summary at exit (`rows`, `success_rows`, `raw_success_rows`, `on_floor_rows`, `facing_up_rows`, value ranges, final row fields) plus explicit `PASS/FAIL` and failure reason based on `ROLLOUT_REQUIRE_SUCCESS_ROWS` (default `1`).
For object rollouts, verdicts now also require sustained directional motion by default (`ROLLOUT_REQUIRE_DIRECTIONAL_MOTION=True`): robot and box must keep moving in a consistent heading for at least `ROLLOUT_DIRECTION_MIN_DURATION_SEC` after `ROLLOUT_DIRECTION_MIN_STEP`. Tune with `ROLLOUT_DIRECTION_MIN_COSINE`, `ROLLOUT_DIRECTION_MIN_PAIR_COSINE`, speed thresholds, and net-displacement thresholds.
For MuJoCo object tasks, `play_policy_mujoco_robot_w_obj.sh` now supports runtime object-mass tuning without editing URDF: `MUJOCO_OBJECT_MASS_SCALE` (default `0.5`) and optional `MUJOCO_OBJECT_MASS_OVERRIDE` (absolute kg).

Policy helpers are MuJoCo-only: they force `--sim2sim.simulator mujoco` and accept either `.pt` or `.onnx` checkpoints. If no ONNX path is provided, `eval_agent.py` exports from the `.pt` checkpoint when needed.

Replay scripts write per-frame pose logs to `logs/replay_motion_mujoco/*.csv` and report success when the robot is both:
- touching floor (via foot height threshold), and
- upright/facing up (root up-vector threshold).

---

# Policy Controls

Commands for controlling policies during execution.

**Important**: All policy controls that use keyboard should be entered in the **policy terminal** (where you ran `run_policy.py`), not in the MuJoCo window. MuJoCo has separate controls for simulation (see workflow docs).

## General Controls

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start the policy | `]` | A button |
| Stop the policy | `o` | B button |
| Set robot to default pose | `i` | Y button |
| Kill controller program | - | L1 (LB) + R1 (RB) |

## Locomotion (Velocity Tracking)

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Switch walking/standing | `=` | Start button |
| Adjust linear velocity | `w` `a` `s` `d` | Left stick |
| Adjust angular velocity | `q` `e` | Right stick |

**Default pose**: Standing pose

## Whole-Body Tracking

| Action | Keyboard | Joystick |
|--------|----------|----------|
| Start the policy | `]` | A button |
| Start motion clip | `s` | Start button |

**Default pose**: Standing with raised arms

## Joystick-Only Features

- **Select button**: Switch between policies (when multiple policies are loaded)


# Configuration Overrides

## Loading ONNX Checkpoints from Wandb

You can load ONNX checkpoints directly from Wandb without manually downloading them first. This is useful for quickly testing models from training runs.

**Syntax:**
```bash
--task.model-path wandb://entity/project_name/run_id/model.onnx
```

**Example with G1 locomotion:**
```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path wandb://my-username/my-project/run-abc123/fastsac_g1_29dof.onnx \
    --task.use-joystick \
    --task.interface eth0
```

**Example with Wandb HTTPS URL:**
```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path https://wandb.ai/username/project/runs/abc123/files/model.onnx \
    --task.use-joystick \
    --task.interface eth0
```

The model will be automatically downloaded and cached locally. The entity is your Wandb username or organization name.

## Finding Your Network Interface

The `--task.interface` parameter specifies which network interface to use for communicating with the robot. The correct interface name varies by computer and network card.

**Common interface names:**
- `eth0` - Common Ethernet interface name
- `enp0s31f6` - Modern Linux Ethernet naming
- `lo` - Loopback interface (for sim2sim)

**To find your interface name:**
```bash
ifconfig
```

Look for the interface connected to your robot's network. For real robot deployments, use the interface with an IP address on the same subnet as your robot. For sim2sim deployments, use `lo` (loopback).

## Observation History Length (> 1)

If a policy was trained with stacked observations (e.g., history length 4), you must pass the same history length at inference time so the observation tensor matches the model's expected input size.

Example:

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path <path-to-model>.onnx \
    --task.interface eth0 \
    --observation.history_length_dict.actor_obs=4
```

The override updates the `actor_obs` buffer before the ONNX session is initialized, so any policy (locomotion or WBT) can run with longer observation histories as long as the underlying model was trained that way.


## Overriding Control Gains

By default, control gains (kp/kd) are loaded from ONNX model metadata. You can override these values in your configuration:

### G1 29-DOF with Custom Control Gains

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --robot.motor-kp 40.2 99.1 40.2 99.1 28.5 28.5 40.2 99.1 40.2 99.1 28.5 28.5 40.2 28.5 28.5 14.3 14.3 14.3 14.3 14.3 16.8 16.8 14.3 14.3 14.3 14.3 14.3 16.8 16.8 \
    --robot.motor-kd 2.6 6.3 2.6 6.3 1.8 1.8 2.6 6.3 2.6 6.3 1.8 1.8 2.6 1.8 1.8 0.9 0.9 0.9 0.9 0.9 1.1 1.1 0.9 0.9 0.9 0.9 0.9 1.1 1.1
```

### T1 29-DOF with Custom Control Gains

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/t1_29dof/ppo_t1_29dof.onnx \
    --robot.motor-kp 5.0 5.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 200.0 200.0 200.0 200.0 200.0 50.0 50.0 200.0 200.0 200.0 200.0 50.0 50.0 \
    --robot.motor-kd 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 5.0 5.0 5.0 5.0 5.0 3.0 3.0 5.0 5.0 5.0 5.0 3.0 3.0
```

**Note**: When control gains are not specified, they will be automatically loaded from the ONNX model metadata. This is the recommended approach as it ensures the gains match those used during training.
