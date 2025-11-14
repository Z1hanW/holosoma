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
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path wandb://far-wandb/nightly-g1_29dof_manager-multigpu/3vbl6vnz/model_04999.onnx \
    --task.use-joystick \
    --task.interface eth0
```

## Locomotion Unitree G1 (sim2sim)

1. Start MuJoCo environment:

```bash
python3 src/holosoma_inference/holosoma_inference/run_sim.py \
    task=loco/loco \
    robot=g1/g1_29dof \
    obs=loco/g1_29dof \
    task.USE_JOYSTICK=false \
    task.INTERFACE="lo"
```

2. Launch the policy:

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path wandb://far-wandb/nightly-g1_29dof_manager-multigpu/3vbl6vnz/model_04999.onnx
```

## Locomotion Booster T1 (real robot)

1. Launch the policy:

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path https://far.wandb.io/far-wandb/nightly-t1_29dof_manager/runs/taks33kw/files/model_04999.onnx \
    --task.use-joystick \
    --task.interface eth0
```

## Locomotion Booster T1 (sim2sim)

1. Launch the simulation:

```bash
python3 src/holosoma_inference/holosoma_inference/run_sim.py \
    task=loco/loco \
    robot=t1/t1_29dof \
    obs=loco/t1_29dof
```

2. Launch the policy:

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path https://far.wandb.io/far-wandb/nightly-t1_29dof_manager/runs/taks33kw/files/model_04999.onnx \
    --task.no-use-joystick \
    --task.interface lo
```

# Examples: Whole Body Tracking

## Whole Body Tracking Unitree G1 (real robot)

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
    task=wbt/wbt \
    robot=g1/g1_29dof \
    obs=wbt/wbt \
    task.USE_JOYSTICK=true \
    task.INTERFACE="enp0s31f6" \
    task.policy.rl_rate=50 \
    model_path=src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx
```


## Whole Body Tracking Unitree G1 (sim2sim)

1. Start MuJoCo environment:

```bash
python3 src/holosoma_inference/holosoma_inference/run_sim.py \
    task=wbt/wbt \
    robot=g1/g1_29dof \
    obs=wbt/wbt \
    task.USE_JOYSTICK=false
```

2. Launch the policy:

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-wbt \
    --task.model-path src/holosoma_inference/holosoma_inference/models/wbt/fastsac_g1_29dof_dancing.onnx \
    --task.no-use-joystick \
    --task.rl-rate 50
```

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py \
    task=wbt/wbt \
    robot=g1/g1_29dof \
    obs=wbt/wbt \
    task.USE_JOYSTICK=true \
    task.INTERFACE="enp0s31f6" \ # NOTE: set to apropriate interface
    task.policy.rl_rate=50 \
    model_path='[stand_policy.onnx,dance.onnx]' # to cycle through multiple policies
```


3. Control the robot:

   - **Step 1**: In policy terminal, press `i` to gradually move the robot to the first movement of the motion clip
   - **Step 2**: In MuJoCo window, press `7` to lower the gantry, let the robot touch the ground
   - **Step 3**: In policy terminal, press `]` to start the policy
   - **Step 4**: In MuJoCo window, press `9` to remove the gantry and let the policy stabilize the robot
   - **Step 5**: In policy terminal, press `s` to start the motion clip

> **TODO**: Replace the policy and commit a working policy
# Configuration Overrides

## Overriding Control Gains

By default, control gains (kp/kd) are loaded from ONNX model metadata. You can override these values in your configuration:

### G1 29-DOF with Custom Control Gains

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:g1-29dof-loco \
    --task.model-path wandb://far-wandb/nightly-g1_29dof_manager-multigpu/3vbl6vnz/model_04999.onnx \
    --robot.motor-kp 40.2 99.1 40.2 99.1 28.5 28.5 40.2 99.1 40.2 99.1 28.5 28.5 40.2 28.5 28.5 14.3 14.3 14.3 14.3 14.3 16.8 16.8 14.3 14.3 14.3 14.3 14.3 16.8 16.8 \
    --robot.motor-kd 2.6 6.3 2.6 6.3 1.8 1.8 2.6 6.3 2.6 6.3 1.8 1.8 2.6 1.8 1.8 0.9 0.9 0.9 0.9 0.9 1.1 1.1 0.9 0.9 0.9 0.9 0.9 1.1 1.1
```

### T1 29-DOF with Custom Control Gains

```bash
python3 src/holosoma_inference/holosoma_inference/run_policy.py inference:t1-29dof-loco \
    --task.model-path https://far.wandb.io/far-wandb/nightly-t1_29dof_manager/runs/taks33kw/files/model_04999.onnx \
    --robot.motor-kp 5.0 5.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 20.0 200.0 200.0 200.0 200.0 200.0 50.0 50.0 200.0 200.0 200.0 200.0 50.0 50.0 \
    --robot.motor-kd 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 5.0 5.0 5.0 5.0 5.0 3.0 3.0 5.0 5.0 5.0 5.0 3.0 3.0
```

**Note**: When control gains are not specified, they will be automatically loaded from the ONNX model metadata. This is the recommended approach as it ensures the gains match those used during training.


