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

## Getting Started

### 1. Setup the Robot

- Hang the robot on the gantry, turn on the robot and the controller.
- Connect the robot to your laptop with an Ethernet cable.
- Configure your laptop's network interface to a static IP address `192.168.123.224` with netmask `255.255.255.0`.
- Put the robot in damping mode, then press `L2+R2` on the controller to enter development mode.

For more detailed instructions, refer to the [Unitree Quick Start page](https://support.unitree.com/home/en/G1_developer/quick_start).

### 2. Setup the Environment

On your laptop, run the following commands to set up the environment:

```bash
cd ~/holosoma
bash scripts/setup_inference.sh             # Create a virtual environment with all dependencies
source scripts/source_inference_setup.sh    # Activate the virtual environment
```

### 3. Run a Locomotion Policy

Within the `(hsinference)` virtualenv, run the following command. Be sure to set `--task.interface` to your network interface name where the robot is connected (use `ifconfig` to list available interfaces).

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.interface eth0 \
    --task.use-joystick
```

### 4. Control the Robot

- Press `A` on the controller to start the policy.
- Press `Start` to enter walking mode.
- Use the left joystick to move forward/backward/left/right.
- Use the right joystick to turn.

For detailed controls, refer to the [Policy Controls](#policy-controls) section below.


### (Optional) Run Onboard

To run the policy onboard the robot's Jetson:

1. Follow **Step 1** above.
2. SSH to the onboard Jetson:
    ```bash
    ssh unitree@192.168.123.164     # Default password is '123'
    sudo jetson_clocks              # Set Jetson to maximum performance
    ```
3. Follow **Steps 2, 3, 4** above. Use `--task.interface eth0` in Step 3.

### (Optional) Run Inside Docker

To run inside Docker (onboard or offboard):

1. Follow **Step 1** above.
2. Create the Docker container:
    ```bash
    bash holosoma/src/holosoma_inference/docker/build.sh   # Build the Docker image
    bash holosoma/src/holosoma_inference/docker/run.sh     # Create and enter the Docker container
    ```
3. Follow **Steps 3 and 4** inside the Docker container. Skip Step 2, as the environment is pre-built in the image.
4. Use the same interface as your host system in Step 3 (`eth0` on Jetson, or check with `ifconfig` on your laptop).




# Examples: Locomotion

## Locomotion Unitree G1 (real robot)

1. Launch the policy:

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.use-joystick \
    --task.interface eth0
```

## Locomotion Unitree G1 (sim2sim)

1. Start MuJoCo environment:

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

2. Launch the policy:

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx
```

## Locomotion Booster T1 (real robot)

1. Launch the policy:

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/t1_29dof/ppo_t1_29dof.onnx \
    --task.use-joystick \
    --task.interface eth0
```

## Locomotion Booster T1 (sim2sim)

1. Launch the simulation:

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:t1-29dof-waist-wrist
```

2. Launch the policy:

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path src/holosoma_inference/holosoma_inference/models/loco/t1_29dof/ppo_t1_29dof.onnx \
    --task.no-use-joystick \
    --task.interface lo
```

# Examples: Whole Body Tracking

## Whole Body Tracking Unitree G1 (real robot)

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.no-use-joystick
    --task.interface=eth0
```


## Whole Body Tracking Unitree G1 (sim2sim)

1. Start MuJoCo environment:

```bash
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:g1-29dof
```

2. Launch the policy:

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.no-use-joystick \
    --task.use-sim-time \
    --task.rl-rate 50
```

3. Control the robot:

- **Step 1**: In policy terminal, press `Enter` when prompted. The robot enters a stiff control mode.
- **Step 2**: In MuJoCo window, press `8` to lower the gantry, let the robot touch the ground
- **Step 3**: In MuJoCo window, press `9` to remove the gantry and let the stiff controller stabilize the robot
- **Step 4**: In policy terminal, press `] (A in joystick)` to start the policy
- **Step 5**: In policy terminal, press `s (Start in joystick)` to start the motion clip

## Whole Body Tracking Unitree G1 (sim2real)

1. Launch the policy:

```bash
source scripts/source_inference_setup.sh
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.use-joystick \
    --task.rl-rate 50 \
    --task.interface eth0
```

2. Control the robot:

- **Step 1**: In policy terminal, press `Enter` when prompted. The robot enters a stiff control mode.
- **Step 4**: In policy terminal, press `] (A in joystick)` to start the policy
- **Step 5**: In policy terminal, press `s (Start in joystick)` to start the motion clip

# Policy Controls

Commands for controlling policies during execution:

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

## Loading ONNX Checkpoints from WandB

You can load ONNX checkpoints directly from WandB without manually downloading them first. This is useful for quickly testing models from training runs.

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

**Example with WandB HTTPS URL:**
```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path https://wandb.ai/username/project/runs/abc123/files/model.onnx \
    --task.use-joystick \
    --task.interface eth0
```

The model will be automatically downloaded and cached locally. The entity is your WandB username or organization name.

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



# Known Issues

## History Length > 1 Not Supported

**Warning**: Policies trained with `history_length > 1` are currently not supported for inference. This is a known limitation and will be fixed soon.
