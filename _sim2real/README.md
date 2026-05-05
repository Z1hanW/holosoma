# Sim2Real G1 Box Policies

These launchers run the same registered box policies as `_sim2sim`, but through the real Unitree G1 interface.

## Run

Terminal 1: start the real depth publisher.

```bash
bash _sim2real/depth_realsense.sh --preview
```

Terminal 2: verify the policy can receive model-ready depth.

```bash
python _sim2real/check_perception_obs.py --port 5658 --once
```

Terminal 3: start the current policy.

```bash
UNITREE_INTERFACE=eth0 bash _sim2real/current.sh
```

`current.sh` matches the raw `mj_rollout.sh` default model source:
`https://wandb.ai/zihanw22/boxer/runs/w5qostjn`.

Or choose a registered policy:

```bash
UNITREE_INTERFACE=eth0 bash _sim2real/w5qostjn_linvel.sh
UNITREE_INTERFACE=eth0 bash _sim2real/run.sh w5qostjn_linvel
```

Replace `eth0` with the robot network interface from `ip link`.

## Remote Control

The launcher enables `HOLOSOMA_JOYSTICK_ROOT_COMMAND=1`.

- `A`: start policy
- `Start`: not required for sparse-root sim2real
- `B`: stop policy
- `Y`: init/default pose
- `L1+R1`: kill controller
- left stick: sparse root `x/y`
- right stick horizontal: sparse root `yaw`

Default full-stick command is `x/y=0.2` and `yaw=17 deg`. The x/y command is clamped to `[-0.5, 0.5]`. Tune without editing code:

```bash
HOLOSOMA_JOYSTICK_ROOT_COMMAND_VALUE=0.15 \
HOLOSOMA_JOYSTICK_ROOT_COMMAND_XY_MAX=0.5 \
HOLOSOMA_JOYSTICK_ROOT_COMMAND_YAW_DEGREES=10 \
UNITREE_INTERFACE=eth0 bash _sim2real/current.sh
```

If an axis sign is reversed:

```bash
HOLOSOMA_JOYSTICK_ROOT_COMMAND_X_SIGN=-1 \
HOLOSOMA_JOYSTICK_ROOT_COMMAND_Y_SIGN=1 \
HOLOSOMA_JOYSTICK_ROOT_COMMAND_YAW_SIGN=1 \
UNITREE_INTERFACE=eth0 bash _sim2real/current.sh
```

## Perception Input

The ONNX policies still consume `perception_obs [5046]`. Before deployment, run the real depth/perception publisher on the same machine and check that the policy can receive it:

```bash
python _sim2real/check_perception_obs.py --port 5658 --once
```

The expected tensor is the training-aligned flattened depth observation, shape `58x87 = 5046`, published as JSON key `perception_obs` over ZMQ on `PERCEPTION_OBS_PORT` (default `5658`).

`depth_realsense.sh` uses the current `w5qostjn` training depth defaults: RealSense depth in meters -> `106x60` camera frame -> crop top `2`, left/right `4` -> bicubic resize to `58x87` -> normalize with near `0.3`, far `3.0`.

## Registered Policies

See `_sim2real/registry.tsv`. `MODEL_INPUT` can override the W&B run URL with a local `.onnx` path.

For command inspection without touching hardware:

```bash
DRY_RUN=1 UNITREE_INTERFACE=eth0 bash _sim2real/current.sh
```
