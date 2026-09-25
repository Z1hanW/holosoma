"""
Configuration values for run_sim.py - sim2sim-optimized simulator defaults.

These configurations extend the base simulator configs with sim2sim-specific settings:
- Bridge enabled by default
- Virtual gantry enabled
- High FPS for real-time robot control
- DOF force sensors enabled (for IsaacGym)
"""

import dataclasses

import holosoma.config_values.simulator
from holosoma.config_types.simulator import BridgeConfig, VirtualGantryCfg

# IsaacGym with sim2sim optimizations
isaacgym = dataclasses.replace(
    holosoma.config_values.simulator.isaacgym,
    config=dataclasses.replace(
        holosoma.config_values.simulator.isaacgym.config,
        bridge=BridgeConfig(enabled=True),
        virtual_gantry=VirtualGantryCfg(enabled=True),
        sim=dataclasses.replace(
            holosoma.config_values.simulator.isaacgym.config.sim,
            fps=1000,  # Gym on CPU can reach ~900 on 4090 (for sim2sim i.e, num_envs=1)
            physx=dataclasses.replace(
                holosoma.config_values.simulator.isaacgym.config.sim.physx,
                enable_dof_force_sensors=True,  # required for force controls
            ),
        ),
    ),
)

# IsaacSim with sim2sim optimizations
isaacsim = dataclasses.replace(
    holosoma.config_values.simulator.isaacsim,
    config=dataclasses.replace(
        holosoma.config_values.simulator.isaacsim.config,
        bridge=BridgeConfig(enabled=True),
        virtual_gantry=VirtualGantryCfg(enabled=True),
        sim=dataclasses.replace(
            holosoma.config_values.simulator.isaacsim.config.sim,
            fps=200,
        ),
    ),
)

# MuJoCo with sim2sim optimizations
mujoco = dataclasses.replace(
    holosoma.config_values.simulator.mujoco,
    config=dataclasses.replace(
        holosoma.config_values.simulator.mujoco.config,
        bridge=BridgeConfig(enabled=True),
        virtual_gantry=VirtualGantryCfg(enabled=True),
        sim=dataclasses.replace(
            holosoma.config_values.simulator.mujoco.config.sim,
            fps=2000,  # mujoco can run faster
        ),
    ),
)

mujoco_split = dataclasses.replace(
    mujoco,
    config=dataclasses.replace(
        mujoco.config,
        bridge=BridgeConfig(
            enabled=True,
            interface="lo",
            publish_sim_state=True,
            listen_control=True,
            use_zmq_lowcmd=True,
            publish_perception_obs_shm=True,
            ignore_default_idle_command=True,
            freeze_until_first_command=True,
        ),
        sim=dataclasses.replace(
            mujoco.config.sim,
            fps=500,
            control_decimation=10,
            substeps=1,
        ),
    ),
)

mujoco_split_debug = dataclasses.replace(
    mujoco_split,
    config=dataclasses.replace(
        mujoco_split.config,
        virtual_gantry=VirtualGantryCfg(enabled=False),
    ),
)

# MuJoCo Warp with sim2sim optimizations
mjwarp = dataclasses.replace(
    holosoma.config_values.simulator.mjwarp,
    config=dataclasses.replace(
        holosoma.config_values.simulator.mjwarp.config,
        bridge=BridgeConfig(enabled=True),
        virtual_gantry=VirtualGantryCfg(enabled=True),
        sim=dataclasses.replace(
            holosoma.config_values.simulator.mjwarp.config.sim,
            fps=400,
        ),
    ),
)

DEFAULTS = {
    "isaacgym": isaacgym,
    "isaacsim": isaacsim,
    "mujoco": mujoco,
    "mujoco_split": mujoco_split,
    "mujoco_split_debug": mujoco_split_debug,
    "mjwarp": mjwarp,
}
