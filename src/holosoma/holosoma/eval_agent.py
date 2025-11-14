from __future__ import annotations

import os

import tyro
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.utils.helpers import get_class
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import (
    CheckpointConfig,
    init_eval_logging,
    load_checkpoint,
    load_saved_experiment_config,
)
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.inference_helpers import (
    export_motion_and_policy_as_onnx,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from holosoma.utils.sim_utils import (
    close_simulation_app,
    setup_simulation_environment,
)
from holosoma.utils.tyro_utils import TYRO_CONIFG


def run_eval_with_tyro(tyro_config: ExperimentConfig, checkpoint_cfg: CheckpointConfig):
    # Use shared simulation environment setup
    env, device, simulation_app = setup_simulation_environment(tyro_config)

    eval_log_dir = get_experiment_dir(tyro_config.logger, tyro_config.training, get_timestamp(), task_name="eval")
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving eval logs to {eval_log_dir}")
    tyro_config.save_config(str(eval_log_dir / CONFIG_NAME))

    assert checkpoint_cfg.checkpoint is not None
    checkpoint = load_checkpoint(checkpoint_cfg.wandb_run_path, checkpoint_cfg.checkpoint, str(eval_log_dir))
    checkpoint_path = str(checkpoint)

    algo_class = get_class(tyro_config.algo._target_)
    algo: BaseAlgo = algo_class(
        device=device,
        env=env,
        config=tyro_config.algo.config,
        log_dir=str(eval_log_dir),
        multi_gpu_cfg=None,
    )
    algo.setup()
    algo.load(checkpoint_path)

    checkpoint_dir = os.path.dirname(checkpoint_path)

    exported_policy_dir_path = os.path.join(checkpoint_dir, "exported")
    os.makedirs(exported_policy_dir_path, exist_ok=True)
    exported_policy_name = checkpoint_path.split("/")[-1]  # example: model_5000.pt
    exported_onnx_name = exported_policy_name.replace(".pt", ".onnx")  # example: model_5000.onnx

    if tyro_config.training.export_policy:
        export_policy_as_jit(algo.alg.actor_critic, exported_policy_dir_path, exported_policy_name)  # type: ignore[attr-defined]
        logger.info("Exported policy as jit script to: ", os.path.join(exported_policy_dir_path, exported_policy_name))
    if tyro_config.training.export_onnx:
        example_obs_dict = algo.get_example_obs()  # type: ignore[attr-defined]

        # Automatically export motion if available (following fast_sac_agent pattern)
        motion_command = env.command_manager.get_state("motion_command")
        if motion_command is not None:
            export_motion_and_policy_as_onnx(
                algo.actor_onnx_wrapper,
                motion_command,
                os.path.join(exported_policy_dir_path, "motion_command.onnx"),
                device,
            )
            logger.info(
                "Exported policy and motion as onnx to: "
                + os.path.join(exported_policy_dir_path, "motion_command.onnx")
            )
        else:
            export_policy_as_onnx(
                wrapper=algo.actor_onnx_wrapper,
                onnx_file_path=os.path.join(exported_policy_dir_path, exported_onnx_name),
                example_obs_dict=example_obs_dict,
            )
            logger.info(f"Exported policy as onnx to: {os.path.join(exported_policy_dir_path, exported_onnx_name)}")

    algo.evaluate_policy(
        max_eval_steps=tyro_config.training.max_eval_steps,
    )

    # Cleanup simulation app
    if simulation_app:
        close_simulation_app(simulation_app)


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, remaining_args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    saved_cfg = load_saved_experiment_config(checkpoint_cfg)
    eval_cfg = saved_cfg.get_eval_config() if saved_cfg is not None else tyro._singleton.MISSING_NONPROP
    overwritten_tyro_config = tyro.cli(
        ExperimentConfig,
        default=eval_cfg,
        args=remaining_args,
        description="Overriding config on top of what's loaded.",
        config=TYRO_CONIFG,
    )
    print("overwritten_tyro_config: ", overwritten_tyro_config)
    run_eval_with_tyro(overwritten_tyro_config, checkpoint_cfg)


if __name__ == "__main__":
    main()
