from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from holosoma.replay import _prepare_first_kinematic_trajectory

from holosoma.utils.visual_motion_transitions import (
    MAX_VISUAL_MOTION_TRANSITION_STEPS,
    configured_control_dt_s,
    configured_simulator_type,
    list_motion_source_clips,
    resolve_motion_transition_source_for_motion_path,
    resolve_visual_motion_transition_plan,
)


_GLOBAL_RUNTIME_SOURCE = {
    "version": 1,
    "source_clip_count": 30,
    "source_semantics": "global_multi_clip_runtime",
}
_NATIVE_SINGLE_SOURCE = {
    "version": 1,
    "source_clip_count": 1,
    "source_semantics": "single_clip_static",
}


def _motion_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        enable_default_pose_prepend=True,
        default_pose_prepend_duration_s=0.2,
        enable_default_pose_append=True,
        default_pose_append_duration_s=0.2,
    )


def test_direct_replay_restarts_and_activates_runtime_prepend() -> None:
    command = SimpleNamespace(
        time_steps=torch.tensor([7, 11], dtype=torch.long),
        _runtime_default_pose_prepend_enabled=True,
        _runtime_default_pose_prepend_active=torch.zeros(2, dtype=torch.bool),
    )

    def activate(env_ids: torch.Tensor) -> None:
        command._runtime_default_pose_prepend_active[env_ids] = True

    command._activate_runtime_default_pose_prepend = activate
    env = SimpleNamespace(
        num_envs=2,
        command_manager=SimpleNamespace(get_state=lambda name: command if name == "motion_command" else None),
    )

    assert _prepare_first_kinematic_trajectory(env, torch) is command
    assert command.time_steps.tolist() == [0, 0]
    assert command._runtime_default_pose_prepend_active.tolist() == [True, True]


def test_direct_replay_restarts_native_static_motion_without_runtime_activation() -> None:
    command = SimpleNamespace(
        time_steps=torch.tensor([5], dtype=torch.long),
        _runtime_default_pose_prepend_enabled=False,
    )
    command._activate_runtime_default_pose_prepend = lambda _env_ids: (_ for _ in ()).throw(
        AssertionError("static replay must not activate runtime prepend")
    )
    env = SimpleNamespace(
        num_envs=1,
        command_manager=SimpleNamespace(get_state=lambda _name: command),
    )

    _prepare_first_kinematic_trajectory(env, torch)
    assert command.time_steps.tolist() == [0]


def test_global_bank_view_plan_keeps_prepend_and_never_fabricates_append() -> None:
    plan = resolve_visual_motion_transition_plan(
        _motion_cfg(),
        fps=50.0,
        control_dt_s=0.02,
        source_clip_count=30,
        simulator_type="isaacsim",
    )

    assert plan.source_semantics == "global_multi_clip_runtime"
    assert plan.prepend_steps == 10
    assert plan.append_steps == 0


def test_selected_clip_from_global_bank_retains_original_source_semantics(tmp_path) -> None:
    (tmp_path / "clip_a.npz").touch()
    (tmp_path / "clip_b.NPZ").touch()
    clip_names = list_motion_source_clips(tmp_path)

    # A replay UI may select just clip_a, but classification happens from the
    # two-clip source before MotionLoader applies that selection.
    assert clip_names == ["clip_a", "clip_b"]
    plan = resolve_visual_motion_transition_plan(
        _motion_cfg(),
        fps=50.0,
        control_dt_s=0.02,
        source_clip_count=len(clip_names),
        simulator_type="isaacsim",
    )
    assert plan.source_semantics == "global_multi_clip_runtime"
    assert (plan.prepend_steps, plan.append_steps) == (10, 0)


def test_filtered_single_clip_view_uses_authenticated_global_source_semantics(tmp_path) -> None:
    motion_dir = tmp_path / "filtered_motion"
    motion_dir.mkdir()
    (motion_dir / "clip_a.npz").touch()
    (motion_dir / "_clip_object_urdf_map.json").write_text(
        json.dumps(
            {
                "motion_transition_source": _GLOBAL_RUNTIME_SOURCE,
                "clips": {"clip_a": {}},
            }
        ),
        encoding="utf-8",
    )

    transition_source = resolve_motion_transition_source_for_motion_path(motion_dir)
    assert transition_source == _GLOBAL_RUNTIME_SOURCE
    assert len(list_motion_source_clips(motion_dir)) == 1

    plan = resolve_visual_motion_transition_plan(
        _motion_cfg(),
        fps=50.0,
        control_dt_s=0.02,
        source_clip_count=1,
        motion_transition_source=transition_source,
        simulator_type="isaacsim",
    )

    assert plan.source_semantics == "global_multi_clip_runtime"
    assert (plan.prepend_steps, plan.append_steps) == (10, 0)


def test_native_single_clip_view_uses_authenticated_static_semantics(tmp_path) -> None:
    motion_dir = tmp_path / "native_motion"
    motion_dir.mkdir()
    (motion_dir / "clip_a.npz").touch()
    (motion_dir / "clip_object_urdf_map.json").write_text(
        json.dumps(
            {
                "motion_transition_source": _NATIVE_SINGLE_SOURCE,
                "clips": {"clip_a": {}},
            }
        ),
        encoding="utf-8",
    )

    transition_source = resolve_motion_transition_source_for_motion_path(motion_dir)
    plan = resolve_visual_motion_transition_plan(
        _motion_cfg(),
        fps=50.0,
        control_dt_s=0.02,
        source_clip_count=1,
        motion_transition_source=transition_source,
        simulator_type="isaacsim",
    )

    assert transition_source == _NATIVE_SINGLE_SOURCE
    assert plan.source_semantics == "single_clip_static"
    assert (plan.prepend_steps, plan.append_steps) == (10, 10)


def test_legacy_visual_source_without_provenance_falls_back_to_clip_count(tmp_path) -> None:
    motion_dir = tmp_path / "legacy_motion"
    motion_dir.mkdir()
    (motion_dir / "clip_a.npz").touch()
    (motion_dir / "clip_b.npz").touch()

    assert resolve_motion_transition_source_for_motion_path(motion_dir) is None
    plan = resolve_visual_motion_transition_plan(
        _motion_cfg(),
        fps=50.0,
        control_dt_s=0.02,
        source_clip_count=2,
        motion_transition_source=None,
        simulator_type="isaacsim",
    )

    assert plan.source_semantics == "global_multi_clip_runtime"


def test_visual_transition_source_requires_exact_schema(tmp_path) -> None:
    motion_dir = tmp_path / "malformed_motion"
    motion_dir.mkdir()
    (motion_dir / "clip_a.npz").touch()
    malformed = dict(_GLOBAL_RUNTIME_SOURCE, unbound_extra=True)
    (motion_dir / "_clip_object_urdf_map.json").write_text(
        json.dumps({"motion_transition_source": malformed, "clips": {"clip_a": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly"):
        resolve_motion_transition_source_for_motion_path(motion_dir)


def test_standalone_view_plan_keeps_both_static_splices() -> None:
    plan = resolve_visual_motion_transition_plan(
        _motion_cfg(),
        fps=50.0,
        control_dt_s=0.02,
        source_clip_count=1,
        simulator_type="isaacsim",
    )

    assert plan.source_semantics == "single_clip_static"
    assert (plan.prepend_steps, plan.append_steps) == (10, 10)


def test_visual_transition_plan_rejects_unbounded_frame_allocation() -> None:
    cfg = _motion_cfg()
    cfg.default_pose_prepend_duration_s = (MAX_VISUAL_MOTION_TRANSITION_STEPS + 1) / 50.0

    with pytest.raises(ValueError, match="safe maximum"):
        resolve_visual_motion_transition_plan(
            cfg,
            fps=50.0,
            control_dt_s=0.02,
            source_clip_count=2,
            simulator_type="isaacsim",
        )


def test_global_mujoco_view_plan_matches_disabled_training_runtime_prepend() -> None:
    plan = resolve_visual_motion_transition_plan(
        _motion_cfg(),
        fps=50.0,
        control_dt_s=0.02,
        source_clip_count=30,
        simulator_type="mujoco",
    )

    assert plan.source_semantics == "global_multi_clip_runtime"
    assert (plan.prepend_steps, plan.append_steps) == (0, 0)


def test_configured_simulator_type_requires_exact_target_name_agreement() -> None:
    simulator_cfg = SimpleNamespace(
        _target_="holosoma.simulator.isaacsim.isaacsim.IsaacSim",
        config=SimpleNamespace(
            name="isaacsim",
            sim=SimpleNamespace(fps=200, control_decimation=4),
        ),
    )
    assert configured_simulator_type(simulator_cfg) == "isaacsim"
    assert configured_control_dt_s(simulator_cfg) == 0.02

    simulator_cfg.config.name = "mujoco"
    with pytest.raises(ValueError, match="mismatch"):
        configured_simulator_type(simulator_cfg)


def test_hdf5_numpy_byte_clip_ids_decode_without_repr_prefix(tmp_path) -> None:
    h5py = pytest.importorskip("h5py")
    motion_path = tmp_path / "bank.h5"
    with h5py.File(motion_path, "w") as h5f:
        clips = h5f.create_group("clips")
        clips.create_dataset("clip_ids", data=[b"clip_a", b"clip_b"])

    assert list_motion_source_clips(motion_path) == ["clip_a", "clip_b"]


def test_visual_step_rounding_uses_control_dt_not_near_equal_motion_fps() -> None:
    cfg = _motion_cfg()
    cfg.default_pose_prepend_duration_s = 0.07

    plan = resolve_visual_motion_transition_plan(
        cfg,
        # This remains inside training's accepted timebase tolerance, but
        # duration * fps would fall just below the 3.5 half-even boundary.
        fps=49.999999,
        control_dt_s=0.02,
        source_clip_count=2,
        simulator_type="isaacsim",
    )

    assert plan.prepend_steps == round(0.07 / 0.02) == 4
