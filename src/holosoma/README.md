# Holosoma Training Framework

Core training framework for humanoid robot reinforcement learning with support for locomotion (velocity tracking) and whole-body tracking tasks.

| **Category** | **Supported Options** |
|-------------|----------------------|
| **Simulators** | IsaacGym, IsaacSim (training) \| Mujoco (evaluation) |
| **Algorithms** | PPO, FastSAC |
| **Robots** | Unitree G1, Booster T1 |

## Training

All training/eval scripts support `--help` for discovering available flags, e.g. `python src/holosoma/holosoma/train_agent.py --help`.

### Locomotion (Velocity Tracking)

Train robots to track velocity commands.

```bash
# G1 with FastSAC on IsaacGym
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1

# T1 with PPO on IsaacSim
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof \
    simulator:isaacsim \
    logger:wandb \
    --training.seed 1
```

Once checkpoints are saved to Wandb or locally, you can evaluate these locomotion policies using either the [Sim-to-Sim Evaluation](#sim-to-sim-evaluation) workflow or the [In-Training Evaluation](#in-training-evaluation) flow described below.

### Whole-Body Tracking

Train robots to track full-body motion sequences.

**Note**: Currently only supported for Unitree G1 / IsaacSim.

```bash
# G1 with PPO (default)
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    logger:wandb \
    exp:g1-29dof-wbt

# G1 with FastSAC
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    logger:wandb \
    exp:g1-29dof-wbt-fast-sac

# Custom motion file
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt \
    logger:wandb \
    --command.setup_terms.motion_command.params.motion_config.motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/<your file>.npz"
```

Just like locomotion checkpoints, whole-body tracking checkpoints saved in Wandb or on disk can be evaluated via [Sim-to-Sim Evaluation](#sim-to-sim-evaluation) (Mujoco-only) or [In-Training Evaluation](#in-training-evaluation) (IsaacSim-only), depending on whether you need cross-simulator testing or same-simulator validation.

## Sim-to-Sim Evaluation

Run trained policies in Mujoco simulation:

```bash
# Terminal 1: Start Mujoco simulation
source scripts/source_mujoco_setup.sh
python src/holosoma/holosoma/run_sim.py robot:t1-29dof-waist-wrist

# Terminal 2: This is handled by holosoma_inference - see inference README
```

### Mujoco Controls

**Gantry:**
- `7`: Lift the gantry
- `8`: Lower the gantry
- `9`: Disable/remove the gantry

**General:**
- `Backspace`: Reset simulation

## In-Training Evaluation

For evaluating policies with the exact same configuration used during training (same simulator, environment settings, etc.):

```bash
# Evaluate checkpoint from Wandb
python src/holosoma/holosoma/eval_agent.py \
    --wandb_run_path=<ENTITY>/<PROJECT_NAME>/<RUN_NAME> \
    --checkpoint=<CHECKPOINT_NAME>

# Evaluate local checkpoint
python src/holosoma/holosoma/eval_agent.py \
    --checkpoint=<CHECKPOINT_PATH>
```

This evaluation mode:
- Automatically loads the training configuration from the checkpoint
- Runs evaluation in the same simulator and environment as training
- Can export policies to ONNX format (via `--training.export_onnx=True`)

**Note**: For sim-to-sim evaluation (testing policies across simulators), use the workflow described in the "Sim-to-Sim Evaluation" section above. ONNX policies are typically exported alongside `.pt` checkpoints during training, but can also be generated using this script.

## Advanced Configuration

The training system uses a hierarchical configuration system. The `exp` config serves as the main entry point with default configurations tuned for each algorithm and robot. You can customize training by overriding parameters on the command line.

### Logging with Weights & Biases

```bash
source scripts/source_isaacsim_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof \
    simulator:isaacsim \
    --training.seed 1 \
    --algo.config.use-symmetry=False \
    logger:wandb \
    --logger.project locomotion-g1-29dof-ppo \
    --logger.name ppo-without-symmetry-seed1
```

### Video Logging

Enable video recording during training. Videos are saved locally and uploaded to Wandb if enabled.

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --logger.video.enabled=True \
    --logger.video.interval=5
```

**Video logging parameters:**
- `--logger.video.enabled=True` - Enable video recording
- `--logger.video.interval=5` - Record video every 5 episodes
- Videos are automatically uploaded to Wandb when `logger:wandb` is used

### Terrain

```bash
# Use plane terrain instead of mixed terrain
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    terrain:terrain-locomotion-plane
```

### Multi-GPU Training

```bash
source scripts/source_isaacgym_setup.sh
torchrun --nproc_per_node=4 src/holosoma/holosoma/train_agent.py \
    exp:t1-29dof-fast-sac \
    simulator:isaacgym \
    --training.num-envs 16384  # global/total number of environments
```

### Custom Reward Weights

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --reward.terms.tracking-lin-vel.weight=2.5 \
    --reward.terms.feet-phase.params.swing-height=0.12
```

### Observation Noise

```bash
# Disable observation noise
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --observation.groups.actor-obs.enable-noise=False
```

### Observation History Length

Some policies benefit from stacking multiple timesteps of observations. You can increase the history length used during training with:

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --observation.groups.actor_obs.history-length 4
```

Make sure to pass the same history length when running inference so the exported ONNX policy receives inputs with the correct shape.

### Curriculum Learning

```bash
# Disable curriculum
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --curriculum.setup-terms.penalty-curriculum.params.enabled=False

# Custom curriculum threshold (for shorter episodes)
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --simulator.config.sim.max-episode-length-s=10.0 \
    --curriculum.setup-terms.penalty-curriculum.params.level-up-threshold=350
```

### Domain Randomization

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    --randomization.setup-terms.push-randomizer-state.params.enabled=False \
    --randomization.setup-terms.randomize-base-com-startup.params.enabled=True \
    --randomization.setup-terms.mass-randomizer.params.added-mass-range=[-1.0,3.0]
```
