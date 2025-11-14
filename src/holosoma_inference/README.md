# Holosoma Inference

Policy inference for humanoid robot policies.

## Supported Policies

| Robot      | Locomotion | WBT |
|:----------:|:----------:|:---:|
| Unitree G1 | ✅         | ✅  |
| Booster T1 | ✅         | ❌  |

- ✅ (full support)
- 🚧 (in progress/partial support)
- ❌ (no support)

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
python src/holosoma/holosoma/run_sim.py robot:g1-29dof --simulator.config.bridge.enabled=True
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
    --task.rl-rate 50
```

3. Control the robot:

- **Step 1**: In policy terminal, press `Enter` when prompted. The robot enters a stiff control mode.
- **Step 2**: In MuJoCo window, press `7` to lower the gantry, let the robot touch the ground
- **Step 3**: In MuJoCo window, press `9` to remove the gantry and let the stiff controller stabilize the robot
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


