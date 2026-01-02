import hydra
from hydra.utils import instantiate
import os
import sys
from pathlib import Path
for arg in sys.argv:
    # This hack ensures that isaacgym is imported before any torch modules.
    # The reason it is here (and not in the main func) is due to pytorch lightning multi-gpu behavior.
    if 'backbone' in arg and 'isaacgym' in arg.split('=')[-1]:
        import isaacgym

from utils.config_utils import *


@hydra.main(config_path="config", config_name="base_eval")
def main(override_config: OmegaConf):
    os.chdir(hydra.utils.get_original_cwd())

    if not (override_config.config_path is None and override_config.checkpoint is None):
        if override_config.config_path is not None:
            config_path = override_config.config_path
        else:
            checkpoint = Path(override_config.checkpoint)
            config_path = checkpoint.parent / "config.yaml"
            if not config_path.exists():
                config_path = checkpoint.parent.parent / "config.yaml"
                if not config_path.exists():
                    raise ValueError(f"Could not find config path: {config_path}")

        print(f"Loading training config file from {config_path}")
        with open(config_path) as file:
            train_config = OmegaConf.load(file)

        if train_config.eval_overrides is not None:
            train_config = OmegaConf.merge(train_config, train_config.eval_overrides)

        config = OmegaConf.merge(train_config, override_config)

        algo = instantiate(config.algo_inference).to("cuda:0")

    else:
        if override_config.eval_overrides is not None:
            config = OmegaConf.merge(override_config, override_config.eval_overrides)
        else:
            config = override_config

        # When both config path and checkpoint are none, for example
        # if one just wants to play back a motion.
        algo = instantiate(config.algo).to("cuda:0")

    algo.evaluate_policy()


if __name__ == "__main__":
    main()