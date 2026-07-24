from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import numpy as np
import pytest


def _load_camera_class():
    """Load the helper without importing the optional full MuJoCo backend."""

    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "holosoma"
        / "holosoma"
        / "simulator"
        / "mujoco"
        / "perception_camera.py"
    )
    package_name = "holosoma.simulator.mujoco"
    scene_name = f"{package_name}.scene_manager"
    module_name = f"{package_name}._perception_camera_mode_test"
    saved = {name: sys.modules.get(name) for name in (package_name, scene_name, module_name)}
    fake_package = ModuleType(package_name)
    fake_package.__path__ = [str(module_path.parent)]
    fake_scene = ModuleType(scene_name)
    fake_scene.HOLOSOMA_PERCEPTION_CAMERA_NAME = "holosoma_perception_camera"
    try:
        sys.modules[package_name] = fake_package
        sys.modules[scene_name] = fake_scene
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.MuJoCoDepthCamera
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


MuJoCoDepthCamera = _load_camera_class()


class _FakeRenderer:
    def __init__(self) -> None:
        self.depth_enabled = False
        self.segmentation_enabled = False

    def enable_depth_rendering(self) -> None:
        self.depth_enabled = True

    def disable_depth_rendering(self) -> None:
        self.depth_enabled = False

    def enable_segmentation_rendering(self) -> None:
        self.segmentation_enabled = True

    def disable_segmentation_rendering(self) -> None:
        self.segmentation_enabled = False

    def render(self) -> np.ndarray:
        return np.zeros((2, 3, 3), dtype=np.uint8)


def _camera_with_fake_renderer() -> tuple[MuJoCoDepthCamera, _FakeRenderer]:
    camera = object.__new__(MuJoCoDepthCamera)
    renderer = _FakeRenderer()
    camera._renderer = renderer
    camera._camera_id = 0
    camera._capture_counter = 0
    camera._device = "cpu"
    camera._prepare_renderer = lambda *, depth: (renderer, object())
    camera._render_depth_with_clip = lambda _renderer: np.ones((2, 3), dtype=np.float32)
    camera._sanitize_depth_array = lambda value: value
    camera._sanitize_rgb_array = lambda value: value
    camera._orient_render_array = lambda value: value
    return camera, renderer


def test_depth_capture_restores_mode_after_debug_segmentation_pass() -> None:
    camera, renderer = _camera_with_fake_renderer()

    def debug_dump(_render_data, **_kwargs) -> None:
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()

    camera._maybe_dump_debug = debug_dump

    depth = camera.capture_depth()

    assert tuple(depth.shape) == (1, 2, 3)
    assert renderer.depth_enabled is True
    assert renderer.segmentation_enabled is False


def test_depth_capture_restores_mode_when_debug_dump_raises() -> None:
    camera, renderer = _camera_with_fake_renderer()

    def failing_debug_dump(_render_data, **_kwargs) -> None:
        renderer.disable_depth_rendering()
        renderer.enable_segmentation_rendering()
        raise RuntimeError("debug write failed")

    camera._maybe_dump_debug = failing_debug_dump

    with pytest.raises(RuntimeError, match="debug write failed"):
        camera.capture_depth()

    assert renderer.depth_enabled is True
    assert renderer.segmentation_enabled is False


def test_prepare_renderer_clears_stale_segmentation_before_depth() -> None:
    camera = object.__new__(MuJoCoDepthCamera)
    renderer = _FakeRenderer()
    renderer.segmentation_enabled = True
    camera._renderer = renderer
    camera._camera_id = 0
    camera._env_id = 0
    camera._use_user_gl_camera = False
    camera._warned_multi_env = False
    render_data = object()
    camera._env = SimpleNamespace(
        num_envs=1,
        simulator=SimpleNamespace(
            root_model=object(),
            backend=SimpleNamespace(get_render_data=lambda *, world_id: render_data),
        ),
    )
    camera._update_camera_pose = lambda _data: None
    camera._update_scene_with_active_camera = lambda _renderer, _data: None

    actual_renderer, actual_data = camera._prepare_renderer(depth=True)

    assert actual_renderer is renderer
    assert actual_data is render_data
    assert renderer.depth_enabled is True
    assert renderer.segmentation_enabled is False
