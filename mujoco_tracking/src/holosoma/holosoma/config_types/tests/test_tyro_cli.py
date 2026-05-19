import tyro

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.perception import PerceptionConfig
from holosoma.utils.tyro_utils import TYRO_CONIFG


def test_experiment_config():
    assert isinstance(tyro.cli(ExperimentConfig, args=(), config=TYRO_CONIFG), ExperimentConfig)


def test_perception_object_geometry_mode_accepts_hyphen_and_underscore_flags():
    cases = (
        ["--object-geometry-mode", "primitive"],
        ["--object_geometry_mode", "primitive"],
    )

    for args in cases:
        cfg = tyro.cli(PerceptionConfig, args=args, config=TYRO_CONIFG)
        assert cfg.object_geometry_mode == "primitive"


def test_experiment_perception_object_geometry_mode_accepts_hyphen_and_underscore_flags():
    cases = (
        ["--perception.object-geometry-mode", "primitive"],
        ["--perception.object_geometry_mode", "primitive"],
    )

    for args in cases:
        cfg = tyro.cli(ExperimentConfig, args=args, config=TYRO_CONIFG)
        assert cfg.perception.object_geometry_mode == "primitive"
