import tyro

import holosoma.config_values.perception as perception_presets
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.perception import PerceptionConfig
from holosoma.utils.tyro_utils import TYRO_CONIFG


def test_experiment_config():
    assert isinstance(tyro.cli(ExperimentConfig, args=(), config=TYRO_CONIFG), ExperimentConfig)


def test_perception_object_geometry_mode_accepts_hyphen_and_underscore_flags():
    cases = (
        ["--object-geometry-mode", "mesh"],
        ["--object_geometry_mode", "mesh"],
    )

    for args in cases:
        cfg = tyro.cli(PerceptionConfig, args=args, config=TYRO_CONIFG)
        assert cfg.object_geometry_mode == "mesh"


def test_experiment_perception_object_geometry_mode_accepts_hyphen_and_underscore_flags():
    cases = (
        ["--perception.object-geometry-mode", "mesh"],
        ["--perception.object_geometry_mode", "mesh"],
    )

    for args in cases:
        cfg = tyro.cli(ExperimentConfig, args=args, config=TYRO_CONIFG)
        assert cfg.perception.object_geometry_mode == "mesh"


def test_mujoco_render_848x480_perception_preset():
    cfg = perception_presets.camera_depth_d435i_mujoco_render_848x480

    assert cfg.camera_source == "rendered"
    assert cfg.camera_width == 848
    assert cfg.camera_height == 480
    assert cfg.camera_warp_resize == (58, 87)
    assert cfg.camera_warp_crop_top == 16
    assert cfg.camera_warp_crop_left == 32
    assert cfg.camera_warp_crop_right == 32
    assert cfg.camera_warp_edge_noise is False
    assert cfg.camera_warp_enable_holes is False
    assert cfg.camera_apply_sensor_noise is False
