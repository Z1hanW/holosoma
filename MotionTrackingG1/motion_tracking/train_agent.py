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

from pytorch_lightning import Trainer, seed_everything
from lightning_fabric.utilities.rank_zero import _get_rank
from motion_tracking.agents.ppo import PPO
import torch

from utils.config_utils import *
from agents.callbacks.adlr_autoresume import AdlrAutoresume


@hydra.main(config_path="config", config_name="base")
def main(config: OmegaConf):
    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation
    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    os.chdir(hydra.utils.get_original_cwd())

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    if config.seed is not None:
        rank = _get_rank()
        if rank is None:
            rank = 0
        seed_everything(config.seed + rank)

    autoresume = AdlrAutoresume()
    id = autoresume.details.get("id")
    if id is not None and "wandb_id" in config:
        config = OmegaConf.merge(config, OmegaConf.create({"wandb_id": id}))

    algo: PPO = instantiate(config.algo)
    max_num_batches = algo.max_num_batches()
    print("MAX NUM BATCHES:", max_num_batches)
    config.data.config.num_batches = max_num_batches
    data = instantiate(config.data)

    trainer: Trainer = instantiate(config.trainer)

    save_dir = Path(trainer.loggers[0].log_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"Saving config file to {save_dir}")
    with open(save_dir / "config.yaml", "w") as file:
        OmegaConf.save(unresolved_conf, file)
    trainer.fit(algo, datamodule=data, ckpt_path=config.checkpoint)


if __name__ == "__main__":
    main()
