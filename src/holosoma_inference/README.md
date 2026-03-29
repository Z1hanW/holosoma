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

## Split MuJoCo Sim2Sim Notes

The main lessons from debugging G1 object-tracking on MuJoCo split sim2sim are:

1. Treat split MuJoCo as the source of truth. Debug `run_sim.py + run_policy.py + sim-state` before trusting web or `viser` views.
2. If the robot does not move, first verify the simulator actually receives ZMQ lowcmd over the split `sim-control` channel. A running policy alone is not evidence that MuJoCo is actuated.
3. Object-tracking inference must stay aligned with training semantics: use simulator clock, simulator state, simulator-measured ref body when available, and training-aligned per-joint action scales from ONNX metadata.
4. If G1 moves but the object does not move with it, inspect authoritative split sim traces first. This is usually a MuJoCo contact/material issue, not a frontend rendering issue.
5. Reset should mean `sim-control reset -> simulator reset -> clock rewind -> motion clip restarts from frame 0`. If reset takes seconds, you are probably restarting the whole split sim pipeline rather than resetting the simulator.
6. Keep ports `5655/5657/5659` clean. Stale split sim processes can make new viewers or tools read old `sim-state`, which looks like broken reset behavior.

Useful entry points in this repo:

- `./mj_track.sh [--viewer sim_state|mjviser] [motion.npz] [model.onnx]`: tracking launcher with split MuJoCo visualization
- `./mj_depth.sh [--viewer sim_state|mjviser]`: MuJoCo joystick/manual-control launcher for the depth box-carry policy
- `./vis_mujoco_sim_state.sh`: `viser` viewer that reads split MuJoCo `sim-state` and can trigger reset over `sim-control`

### Training-Aligned Invariants

For current G1 WBT/object-tracking split sim2sim, the minimum configuration baseline is:

- Simulator side:
  `use_zmq_lowcmd=True` must also start the split `sim-control` subscriber, not just the lowcmd publisher path.
- Inference side:
  `use_sim_time=True`, `use_sim_state=True`, `prefer_sim_ref_from_sim_state=True`, and `restart_motion_on_clock_reset=True`.
- Action scaling:
  per-joint policy action scales must be restored from ONNX metadata, not replaced with a flat fallback.
- Startup behavior:
  prefer `freeze_until_first_command=1` over long startup holds so rollout semantics match training more closely.
- Object carry:
  when carry quality is wrong, debug `sim-state` traces and contact bodies before changing web or `viser` rendering.

### Reset Semantics And Timing

Correct reset means:

1. send `{"action": "reset"}` over split `sim-control`
2. simulator rewinds to motion-init state and resets its clock
3. policy sees the clock jump backwards and restarts the motion clip from frame `0`
4. viewers only declare reset complete after seeing a rewound `sim_time_ms`

Measured reference sequence for the fixed path on March 19, 2026:

- `mujoco.log`: reset queued at `01:06:51.909`
- `mujoco.log`: simulator reset at `01:06:51.917`
- `policy.log`: motion clip restart triggered by clock rewind at `01:06:51.930`
- `viser`: first post-reset rewound state at `01:06:51.944`

That is about `35.5 ms` end to end. Multi-second reset behavior indicates whole-process restart, not a real simulator reset.

### Practical Debug Order

1. Kill stale split sim processes on ports `5655/5657/5659`.
2. Run `./mj_track.sh [motion.npz] [model.onnx]` and confirm authoritative MuJoCo behavior first.
3. Check `logs/sim2sim_runs/.../mujoco.log` for first lowcmd reception and reset handling.
4. Check `logs/sim2sim_runs/.../policy.log` for motion timestep progress and clock-rewind restart.
5. If carry looks wrong, enable `HOLOSOMA_SPLIT_SIM_STATE_TRACE_PATH` and inspect `object_robot_contact_count` and contact bodies.
6. Only after split sim looks correct, debug web or `viser` presentation layers.

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
