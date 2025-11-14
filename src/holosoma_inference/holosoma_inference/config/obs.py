from __future__ import annotations

from pathlib import Path

import yaml


def get_obs_config(robot_type: str) -> dict | None:
    # TODO:  we should have a way to auto-generate this config file during policy training
    current_dir = Path(__file__).parent
    config_path = current_dir / "config" / f"{robot_type}_obs.yaml"

    # Load and return the YAML configuration
    with open(config_path) as f:
        return yaml.safe_load(f)

    return None
