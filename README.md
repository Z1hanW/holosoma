# Holosoma

Holosoma (Greek: "whole-body") is a comprehensive humanoid robotics framework for training and deploying reinforcement learning policies on humanoid robots, as well as motion retargeting. Supports locomotion (velocity tracking) and whole-body tracking tasks across multiple simulators (IsaacGym, IsaacSim, Mujoco) with algorithms like PPO and FastSAC.

## Features

- **Multi-simulator support**: IsaacGym, IsaacSim, and Mujoco (inference only)
- **Multiple RL algorithms**: PPO and FastSAC
- **Robot support**: Unitree G1 and Booster T1 humanoids
- **Task types**: Locomotion (velocity tracking) and whole-body tracking
- **Sim-to-sim and sim-to-real deployment**: Shared inference pipeline across simulation and real robot control
- **Motion retargeting**: Convert human motion capture data to robot motions while preserving interactions with objects and terrain

## Repository Structure

```
src/
├── holosoma/              # Core training framework (locomotion & whole-body tracking)
├── holosoma_inference/    # Inference and deployment pipeline
└── holosoma_retargeting/  # Motion retargeting from human motion data to robots
```

## Documentation

- [Training Guide](src/holosoma/README.md) - Detailed training instructions and configurations
- [Inference Guide](src/holosoma_inference/holosoma_inference/README.md) - Deployment and inference instructions
- [Retargeting Guide](src/holosoma_retargeting/README.md) - Motion retargeting workflow

## Quick Start

### Setup

Choose the appropriate setup script based on your use case:

```bash
# For IsaacGym training
source scripts/setup_isaacgym.sh

# For IsaacSim training
source scripts/setup_isaacsim.sh

# For Mujoco simulation
source scripts/setup_mujoco.sh

# For inference/deployment
source scripts/setup_inference.sh

# For motion retargeting
source scripts/setup_retargeting.sh
```

### Training Example (Locomotion)

Train a G1 robot with FastSAC on IsaacGym:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --training.seed 1
```

### Sim-to-Sim Evaluation

Run trained policy in Mujoco with joystick control:

```bash
# Terminal 1: Start simulation
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py \
    robot:g1-29dof \
    --simulator.config.bridge.enabled=True \
    --simulator.config.bridge.use-joystick=True

# Terminal 2: This is handled by holosoma_inference - see [inference README](src/holosoma_inference/holosoma_inference/README.md)
source scripts/source_inference_setup.sh
python src/holosoma_inference/holosoma_inference/run_policy.py \
    inference:g1-29dof-loco \
    --task.model-path=src/holosoma_inference/holosoma_inference/models/loco/g1_29dof/fastsac_g1_29dof.onnx \
    --task.use-joystick \
    --task.interface=lo
```

See [Policy Controls](src/holosoma_inference/README.md#policy-controls) for keyboard and joystick commands.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

