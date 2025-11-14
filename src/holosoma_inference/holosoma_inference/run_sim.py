#!/usr/bin/env python3
"""
Sim2Real Hydra Runner Script

This script uses Hydra configuration system to run different simulation environments.

Usage:
    python run_sim.py robot=g1 task=loco/loco obs=loco/g1_29dof
    python run_sim.py robot=g1 task=loco_manip/loco_manip obs=loco_manip/g1_29dof robot.use_joystick=true
"""

import sys
import traceback

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from holosoma_inference.utils.hydra import register_omegaconf_resolvers

register_omegaconf_resolvers()


def run_sim(config: DictConfig):
    print("🚀 Starting Simulator with Hydra configuration...")
    print(f"🤖 Robot: {config.task.sim_env.config.ROBOT_CFG.ROBOT_TYPE}")
    print(f"📋 Task: {config.task.type}")
    print(f"⚙️ Config file: {config.task.name}")
    # Use Hydra's instantiate to create the simulator
    try:
        # Create simulator with config parameters
        # instantiate internally converts DictConfig to dict
        simulator = instantiate(config.task.sim_env)

        simulator.sim_thread.start()
        print("✅ Simulation started successfully!")
        if simulator.record_thread:
            simulator.record_thread.start()
            print("✅ Video recording started successfully!")

    except Exception as e:
        print(f"❌ Error starting simulation: {e}")
        traceback.print_exc()
        sys.exit(1)


@hydra.main(version_base="1.1", config_path="config", config_name="sim_base")
def main(config: DictConfig):
    """Main function using Hydra configuration."""
    run_sim(config)


if __name__ == "__main__":
    main()
