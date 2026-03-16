from __future__ import annotations

import dataclasses

import tyro

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.experiment import AnnotatedExperimentConfig
from holosoma.utils.eval_utils import (
    init_sim_imports,
)
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app
from holosoma.utils.tyro_utils import TYRO_CONIFG


def _resolve_wandb_registry(tyro_config: ExperimentConfig) -> ExperimentConfig:
    """Resolve WandB registry data and override motion/terrain config.

    If ``training.registry_name`` is set, downloads motion + terrain data from
    the WandB registry. When ``training.add_onpath_obstacle`` is also set,
    obstacle poses are sampled and a terrain OBJ mesh is generated.

    This mirrors the registry resolution logic in ``train_agent.py``.
    """
    registry_name = tyro_config.training.registry_name
    if not registry_name:
        return tyro_config

    from holosoma.utils.multi_motion_helpers import pull_paired_from_wandb_registry

    motion_counts, motion_file, terrain_npy_file, terrain_obj_files = (
        pull_paired_from_wandb_registry(registry_name)
    )

    if tyro_config.training.add_onpath_obstacle and terrain_npy_file:
        from holosoma.config_types.terrain import MeshType
        from holosoma.utils.obstacle_helpers import add_onpath_obstacle_standalone

        num_variants = tyro_config.training.num_variants or tyro_config.training.num_envs
        env_spacing, motion_file_str, terrain_mesh, _ = add_onpath_obstacle_standalone(
            num_variants, str(motion_file), str(terrain_npy_file), tyro_config.training.obstacle_seed
        )
        import tempfile as _tempfile

        terrain_obj_path = _tempfile.mktemp(suffix=".obj")
        terrain_mesh.export(terrain_obj_path)
        tyro_config = dataclasses.replace(
            tyro_config,
            terrain=dataclasses.replace(
                tyro_config.terrain,
                terrain_term=dataclasses.replace(
                    tyro_config.terrain.terrain_term,
                    mesh_type=MeshType.LOAD_OBJ,
                    obj_file_path=terrain_obj_path,
                ),
            ),
            simulator=dataclasses.replace(
                tyro_config.simulator,
                config=dataclasses.replace(
                    tyro_config.simulator.config,
                    scene=dataclasses.replace(
                        tyro_config.simulator.config.scene,
                        env_spacing=env_spacing,
                    ),
                ),
            ),
        )
        motion_file = motion_file_str
    elif terrain_obj_files:
        # Use the first OBJ terrain file directly
        from holosoma.config_types.terrain import MeshType

        tyro_config = dataclasses.replace(
            tyro_config,
            terrain=dataclasses.replace(
                tyro_config.terrain,
                terrain_term=dataclasses.replace(
                    tyro_config.terrain.terrain_term,
                    mesh_type=MeshType.LOAD_OBJ,
                    obj_file_path=str(terrain_obj_files[0]),
                ),
            ),
            simulator=dataclasses.replace(
                tyro_config.simulator,
                config=dataclasses.replace(
                    tyro_config.simulator.config,
                    scene=dataclasses.replace(
                        tyro_config.simulator.config.scene,
                        env_spacing=0.0,
                    ),
                ),
            ),
        )

    # Update motion file path in command config
    if tyro_config.command is not None:
        setup_terms = tyro_config.command.setup_terms
        if setup_terms and "motion_command" in setup_terms:
            mc_term = setup_terms["motion_command"]
            mc_params = dict(mc_term.params) if mc_term.params else {}
            if "motion_config" in mc_params:
                mc_params["motion_config"] = dataclasses.replace(
                    mc_params["motion_config"],
                    motion_file=str(motion_file),
                )
                new_mc_term = dataclasses.replace(mc_term, params=mc_params)
                new_setup_terms = dict(setup_terms)
                new_setup_terms["motion_command"] = new_mc_term
                tyro_config = dataclasses.replace(
                    tyro_config,
                    command=dataclasses.replace(tyro_config.command, setup_terms=new_setup_terms),
                )

    print(f"[INFO] Registry resolution complete. Motion file: {motion_file}")
    return tyro_config


def replay(tyro_config: ExperimentConfig):
    tyro_config = _resolve_wandb_registry(tyro_config)
    simulation_app = init_sim_imports(tyro_config)

    import torch

    from holosoma.utils.common import seeding

    seeding(42, torch_deterministic=False)

    env_target = tyro_config.env_class
    tyro_env_config = get_tyro_env_config(tyro_config)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = get_class(env_target)(tyro_env_config, device=device)

    done = False
    while not done:
        env.simulator.sim.step()
        done = env.step_visualize_motion(None)  # type: ignore[attr-defined]

    close_simulation_app(simulation_app)


def main() -> None:
    tyro_cfg = tyro.cli(AnnotatedExperimentConfig, config=TYRO_CONIFG)
    replay(tyro_cfg)


if __name__ == "__main__":
    main()
