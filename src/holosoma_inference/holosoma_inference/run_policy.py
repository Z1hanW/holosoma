#!/usr/bin/env python3
"""
Policy Runner Script with Tyro Configuration

This script uses Tyro configuration system to run different policy types.

Usage:
    python run_policy.py inference:g1-29dof-loco --task.model-path path/to/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path wandb://project/run/model.onnx
    python run_policy.py inference:g1-29dof-loco --task.model-path https://wandb-url/files/model.onnx
"""

import sys
import traceback

import tyro
from loguru import logger

from holosoma_inference.config2.config_types.inference import InferenceConfig
from holosoma_inference.config2.config_values.inference import AnnotatedInferenceConfig
from holosoma_inference.config2.utils import TYRO_CONFIG
from holosoma_inference.policies.locomotion import LocomotionPolicy
from holosoma_inference.policies.wbt import WholeBodyTrackingPolicy
from holosoma_inference.utils.misc import restore_terminal_settings


def run_policy(config: InferenceConfig):
    """Run policy with Tyro configuration."""
    logger.info("🚀 Starting Policy with Tyro configuration...")
    logger.info(f"🤖 Robot: {config.robot.robot_type}")
    logger.info(f"📋 Observation groups: {list(config.observation.obs_dict.keys())}")
    logger.info(f"⚙️ RL Rate: {config.task.rl_rate} Hz")
    logger.info(f"📁 Model path: {config.task.model_path}")

    try:
        # Determine policy class based on observation type
        actor_obs = config.observation.obs_dict.get("actor_obs", [])
        policy_class = WholeBodyTrackingPolicy if "motion_command" in actor_obs else LocomotionPolicy
        logger.info(f"Using {policy_class.__name__}")
        policy: LocomotionPolicy | WholeBodyTrackingPolicy = policy_class(config=config)

        logger.info("✅ Policy initialized successfully!")
        policy.run()
        logger.info("✅ Policy execution completed!")

    except Exception as e:
        logger.error(f"❌ Error running policy: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        restore_terminal_settings()


def main():
    config = tyro.cli(
        AnnotatedInferenceConfig,
        config=TYRO_CONFIG,
    )
    run_policy(config)


if __name__ == "__main__":
    main()
