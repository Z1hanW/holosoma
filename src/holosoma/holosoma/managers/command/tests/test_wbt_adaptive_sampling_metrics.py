from __future__ import annotations

from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.managers.command.terms.wbt import (
    MotionLoader,
    MotionCommand,
    _convert_contact_interval_timebase,
    _contact_aware_carry_window_from_rel_z,
    _contact_aware_carry_window_from_peak_height,
    _compute_contact_stage_intervals,
    _kinematic_lift_window_from_rel_z,
    _probability_mass_on_intervals,
    _select_primary_contact_interval,
)
from holosoma.managers.observation.terms import wbt as wbt_obs_terms


def test_select_primary_contact_interval_unions_all_recognized_carry_regions():
    interval = _select_primary_contact_interval(
        {
            "left_palm": [40, 120],
            "right_wrist": [45, 135],
            "left_elbow": [20, 160],
        }
    )

    assert interval == (20, 160)


def test_contact_interval_timebase_converts_legacy_30hz_half_open_bounds_to_50hz():
    assert _convert_contact_interval_timebase(
        (40, 143),
        metadata={"fps": 30},
        motion_fps=50.0,
    ) == (67, 239)


def test_contact_interval_timebase_preserves_rollout_steps_without_fps_metadata():
    assert _convert_contact_interval_timebase(
        (20, 30),
        metadata={"clip_id": "box_10"},
        motion_fps=50.0,
    ) == (20, 30)


def test_motion_control_timebase_accepts_matching_motion_fps_and_control_dt():
    motion_command = object.__new__(MotionCommand)
    motion_command.motion = SimpleNamespace(fps=np.asarray([50.0], dtype=np.float32))
    motion_command._env = SimpleNamespace(dt=0.02)

    motion_command._validate_motion_control_timebase()


def test_motion_control_timebase_rejects_mismatched_motion_fps_and_control_dt():
    motion_command = object.__new__(MotionCommand)
    motion_command.motion = SimpleNamespace(fps=np.asarray([30.0], dtype=np.float32))
    motion_command._env = SimpleNamespace(dt=0.02)

    with pytest.raises(ValueError, match=r"Motion FPS must match.*motion\.fps=30\.0.*control_fps=50\.0"):
        motion_command._validate_motion_control_timebase()


def test_motion_loader_keeps_object_size_distinct_from_mesh_scale() -> None:
    with pytest.raises(ValueError, match="scale and size are not interchangeable"):
        MotionLoader._extract_object_size_np(
            {"object_scale": np.ones(3, dtype=np.float32)},
            2,
            source="synthetic.npz",
        )


def test_motion_loader_rejects_ambiguous_transposed_object_size() -> None:
    with pytest.raises(ValueError, match="Unsupported object-size shape"):
        MotionLoader._normalize_object_size_array(
            np.ones((3, 4), dtype=np.float32),
            4,
            source="synthetic.npz:object_size",
        )


@pytest.mark.parametrize("raw_scale", [[0.0], [-1.0, 1.0, 1.0], [float("nan")]])
def test_motion_loader_rejects_nonphysical_configured_object_size_scale(raw_scale) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        MotionLoader._normalize_object_size_scale(raw_scale)


def test_probability_mass_on_contact_stages_matches_uniform_lengths():
    bin_probabilities = torch.full((10,), 0.1, dtype=torch.float32)
    stage_intervals, after_t2_interval = _compute_contact_stage_intervals(
        t1=20,
        t2=80,
        sample_end_step=100.0,
    )

    stage_masses = _probability_mass_on_intervals(
        bin_probabilities,
        sample_end_step=100.0,
        intervals=stage_intervals,
    )
    after_t2_mass = _probability_mass_on_intervals(
        bin_probabilities,
        sample_end_step=100.0,
        intervals=[after_t2_interval],
    )[0]

    expected = torch.tensor([0.30, 0.06666667, 0.06666667, 0.06666667, 0.30], dtype=torch.float32)
    assert torch.allclose(stage_masses, expected, atol=1.0e-5)
    assert torch.isclose(after_t2_mass, torch.tensor(0.20, dtype=torch.float32), atol=1.0e-5)
    assert torch.isclose(stage_masses.sum() + after_t2_mass, torch.tensor(1.0, dtype=torch.float32), atol=1.0e-5)


def test_short_contact_window_collapses_middle_stages_without_overlap():
    bin_probabilities = torch.full((10,), 0.1, dtype=torch.float32)
    stage_intervals, after_t2_interval = _compute_contact_stage_intervals(
        t1=50,
        t2=70,
        sample_end_step=100.0,
    )

    stage_masses = _probability_mass_on_intervals(
        bin_probabilities,
        sample_end_step=100.0,
        intervals=stage_intervals,
    )
    after_t2_mass = _probability_mass_on_intervals(
        bin_probabilities,
        sample_end_step=100.0,
        intervals=[after_t2_interval],
    )[0]

    expected = torch.tensor([0.60, 0.0, 0.0, 0.0, 0.10], dtype=torch.float32)
    assert torch.allclose(stage_masses, expected, atol=1.0e-5)
    assert torch.isclose(after_t2_mass, torch.tensor(0.30, dtype=torch.float32), atol=1.0e-5)


def test_contact_aware_carry_window_turns_on_after_pickup_and_off_on_lowering():
    rel_z = torch.tensor([0.00, 0.00, 0.12, 0.15, 0.18, 0.18, 0.18, 0.16, 0.09, 0.08, 0.08], dtype=torch.float32)

    carry_start, carry_end = _contact_aware_carry_window_from_rel_z(
        rel_z,
        consecutive_steps=2,
        release_lead_steps=30,
    )

    assert carry_start == 2
    assert carry_end == 8


def test_contact_aware_carry_window_uses_contact_release_tail_when_object_stays_lifted():
    rel_z = torch.tensor([0.00, 0.00, 0.12, 0.15, 0.18, 0.18, 0.18, 0.18, 0.18, 0.18], dtype=torch.float32)

    carry_start, carry_end = _contact_aware_carry_window_from_rel_z(
        rel_z,
        contact_interval=(0, 35),
        consecutive_steps=2,
        release_lead_steps=30,
    )

    assert carry_start == 2
    assert carry_end == 5


def test_kinematic_lift_window_never_applies_contact_release_cap():
    rel_z = torch.tensor(
        [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )

    assert _kinematic_lift_window_from_rel_z(rel_z) == (2, 8)
    assert _contact_aware_carry_window_from_rel_z(
        rel_z,
        contact_interval=(0, 34),
        release_lead_steps=30,
    ) == (2, 4)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_kinematic_lift_window_rejects_nonfinite_motion(bad_value):
    rel_z = torch.tensor([0.0, bad_value, 0.2], dtype=torch.float32)

    with pytest.raises(ValueError, match="finite"):
        _kinematic_lift_window_from_rel_z(rel_z)


def test_kinematic_button_mode_ignores_sidecar_and_root_peak_window():
    motion_command = object.__new__(MotionCommand)
    motion_command.device = "cpu"
    motion_command.motion_cfg = SimpleNamespace(
        contact_aware_button_window_mode="kinematic_lift",
        contact_aware_carry_window_mode="peak_height",
    )
    rel_z = torch.tensor(
        [0.0, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float32,
    )
    body_pos = torch.zeros((rel_z.numel(), 1, 3), dtype=torch.float32)
    object_pos = torch.zeros((rel_z.numel(), 3), dtype=torch.float32)
    object_pos[:, 2] = rel_z
    motion_command.motion = SimpleNamespace(
        num_clips=1,
        has_object=True,
        clip_offsets=torch.tensor([0]),
        clip_lengths=torch.tensor([rel_z.numel()]),
        body_pos_w=body_pos,
        object_pos_w=object_pos,
    )
    motion_command._adaptive_sampling_contact_window_by_clip = torch.tensor([[10, 12]])
    motion_command._adaptive_sampling_contact_window_valid_by_clip = torch.tensor([True])
    root_window = motion_command._get_contact_aware_carry_window_by_clip()
    expected_root_window = _contact_aware_carry_window_from_peak_height(
        object_pos[:, 2],
        peak_height_alpha=0.91,
        smoothing_steps=5,
    )

    assert root_window.tolist() == [list(expected_root_window)]
    assert root_window.tolist() != [[2, 8]]
    assert motion_command._get_contact_aware_button_window_by_clip().tolist() == [[2, 8]]


@pytest.mark.parametrize(
    "method_name",
    [
        "_get_contact_aware_button_window_by_clip",
        "get_contact_aware_pickup_button",
        "get_contact_aware_drop_button",
    ],
)
def test_kinematic_button_training_rejects_motion_without_object_trajectory(
    method_name: str,
) -> None:
    motion_command = object.__new__(MotionCommand)
    motion_command.motion_cfg = SimpleNamespace(
        contact_aware_button_window_mode="kinematic_lift",
    )
    motion_command.motion = SimpleNamespace(has_object=False)

    with pytest.raises(ValueError, match="requires a motion with an object trajectory"):
        getattr(motion_command, method_name)()


def test_contact_aware_drop_button_turns_on_at_carry_end():
    motion_command = object.__new__(MotionCommand)
    motion_command.device = "cpu"
    motion_command.num_envs = 3
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.clip_ids = torch.tensor([0, 0, 0], dtype=torch.long)
    motion_command.time_steps = torch.tensor([1, 5, 8], dtype=torch.long)
    motion_command._get_contact_aware_carry_window_by_clip = MethodType(
        lambda self: torch.tensor([[2, 5]], dtype=torch.long),
        motion_command,
    )

    drop_button = motion_command.get_contact_aware_drop_button()

    assert drop_button.tolist() == [False, True, True]


def test_contact_aware_pickup_button_turns_off_at_carry_start():
    motion_command = object.__new__(MotionCommand)
    motion_command.device = "cpu"
    motion_command.num_envs = 3
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.clip_ids = torch.tensor([0, 0, 0], dtype=torch.long)
    motion_command.time_steps = torch.tensor([1, 2, 8], dtype=torch.long)
    motion_command._get_contact_aware_carry_window_by_clip = MethodType(
        lambda self: torch.tensor([[2, 5]], dtype=torch.long),
        motion_command,
    )

    pickup_button = motion_command.get_contact_aware_pickup_button()

    assert pickup_button.tolist() == [True, False, False]


def test_contact_aware_buttons_prefer_exported_t1_t2_per_clip():
    motion_command = object.__new__(MotionCommand)
    motion_command.device = "cpu"
    motion_command.num_envs = 4
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.clip_ids = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    motion_command.time_steps = torch.tensor([9, 20, 9, 20], dtype=torch.long)
    motion_command._adaptive_sampling_contact_window_by_clip = torch.tensor([[10, 20], [-1, -1]])
    motion_command._adaptive_sampling_contact_window_valid_by_clip = torch.tensor([True, False])
    motion_command._get_contact_aware_carry_window_by_clip = MethodType(
        lambda self: torch.tensor([[3, 30], [5, 15]], dtype=torch.long),
        motion_command,
    )

    pickup_button = motion_command.get_contact_aware_pickup_button()
    drop_button = motion_command.get_contact_aware_drop_button()

    # Clip 0 uses exported [t1=10, t2=20]; clip 1 falls back to [5, 15].
    assert pickup_button.tolist() == [True, False, False, False]
    assert drop_button.tolist() == [False, True, False, True]


def _contact_window_loader_stub(
    *,
    root: str,
    uniform_enabled: bool,
    observation_funcs: tuple[str, ...] = (),
    clip_ids: tuple[str, ...] = ("box_10",),
) -> MotionCommand:
    motion_command = object.__new__(MotionCommand)
    motion_command.device = "cpu"
    motion_command.use_adaptive_timesteps_sampler = False
    motion_command.motion = SimpleNamespace(
        num_clips=len(clip_ids),
        has_object=True,
        clip_ids=list(clip_ids),
    )
    motion_command.motion_cfg = SimpleNamespace(
        adaptive_sampling_contact_interval_root=root,
        uniform_t1_window_sampling_enabled=uniform_enabled,
    )
    terms = {
        f"term_{index}": SimpleNamespace(func=func)
        for index, func in enumerate(observation_funcs)
    }
    motion_command._env = SimpleNamespace(
        observation_manager=SimpleNamespace(
            cfg=SimpleNamespace(groups={"actor_obs": SimpleNamespace(terms=terms)})
        )
    )
    return motion_command


def test_uniform_t1_window_requires_configured_contact_root():
    motion_command = _contact_window_loader_stub(root="", uniform_enabled=True)

    with pytest.raises(ValueError, match="requires a non-empty"):
        motion_command._configure_adaptive_sampling_contact_interval_bank()


def test_uniform_t1_window_rejects_root_without_matching_windows(tmp_path):
    motion_command = _contact_window_loader_stub(root=str(tmp_path), uniform_enabled=True)

    with pytest.raises(RuntimeError, match="cannot be honored"):
        motion_command._configure_adaptive_sampling_contact_interval_bank()


def test_contact_observation_consumer_loads_windows_with_sampling_disabled(tmp_path):
    clip_dir = tmp_path / "0000_box_10"
    clip_dir.mkdir()
    (clip_dir / "contact_intervals.json").write_text(
        '{"left_wrist":[1,3]}',
        encoding="utf-8",
    )
    motion_command = _contact_window_loader_stub(
        root=str(tmp_path),
        uniform_enabled=False,
        observation_funcs=(
            "holosoma.managers.observation.terms.wbt:drop_button",
        ),
    )
    motion_command.motion.clip_lengths = torch.tensor([5])
    motion_command.motion.fps = 50.0
    motion_command.motion_cfg.contact_interval_runtime_prepend_compensation = False
    motion_command._runtime_default_pose_prepend_enabled = False
    motion_command._runtime_default_pose_prepend_steps = 0

    motion_command._configure_adaptive_sampling_contact_interval_bank()

    assert motion_command._adaptive_sampling_contact_window_valid_by_clip.tolist() == [True]
    assert motion_command._adaptive_sampling_contact_window_by_clip.tolist() == [[1, 3]]


def test_contact_observation_consumer_without_root_keeps_kinematic_fallback():
    motion_command = _contact_window_loader_stub(
        root="",
        uniform_enabled=False,
        observation_funcs=(
            "holosoma.managers.observation.terms.wbt:sparse_target_root_trajectory_command_contact_aware",
        ),
    )

    motion_command._configure_adaptive_sampling_contact_interval_bank()

    assert motion_command._adaptive_sampling_contact_interval_root is None
    assert motion_command._adaptive_sampling_contact_window_valid_by_clip.tolist() == [False]


def test_configured_contact_observation_root_rejects_zero_matching_windows(tmp_path):
    motion_command = _contact_window_loader_stub(
        root=str(tmp_path),
        uniform_enabled=False,
        observation_funcs=(
            "holosoma.managers.observation.terms.wbt:drop_button",
        ),
    )

    with pytest.raises(RuntimeError, match=r"only 0/1 clips.*box_10"):
        motion_command._configure_adaptive_sampling_contact_interval_bank()


def test_configured_contact_observation_root_rejects_partial_motion_bank(tmp_path):
    clip_dir = tmp_path / "0000_clip_a"
    clip_dir.mkdir()
    (clip_dir / "contact_intervals.json").write_text(
        '{"left_wrist":[1,3]}',
        encoding="utf-8",
    )
    motion_command = _contact_window_loader_stub(
        root=str(tmp_path),
        uniform_enabled=False,
        observation_funcs=(
            "holosoma.managers.observation.terms.wbt:pickup_button",
        ),
        clip_ids=("clip_a", "clip_b"),
    )
    motion_command.motion.clip_lengths = torch.tensor([5, 5])
    motion_command.motion.fps = 50.0
    motion_command.motion_cfg.contact_interval_runtime_prepend_compensation = False
    motion_command._runtime_default_pose_prepend_enabled = False
    motion_command._runtime_default_pose_prepend_steps = 0

    with pytest.raises(RuntimeError, match=r"only 1/2 clips.*clip_b"):
        motion_command._configure_adaptive_sampling_contact_interval_bank()


def test_configured_contact_root_is_not_loaded_without_a_consumer_or_sampler(tmp_path):
    motion_command = _contact_window_loader_stub(
        root=str(tmp_path),
        uniform_enabled=False,
        observation_funcs=("holosoma.managers.observation.terms.wbt:base_ang_vel",),
    )

    motion_command._configure_adaptive_sampling_contact_interval_bank()

    assert motion_command._adaptive_sampling_contact_interval_root is None
    assert motion_command._adaptive_sampling_contact_window_valid_by_clip.tolist() == [False]


def test_required_contact_interval_coverage_rejects_partial_motion_bank(tmp_path, monkeypatch):
    clip_dir = tmp_path / "0000_clip_a"
    clip_dir.mkdir()
    (clip_dir / "metadata.json").write_text('{"clip_id":"clip_a","fps":50}', encoding="utf-8")
    (clip_dir / "contact_intervals.json").write_text(
        '{"left_wrist":[1,3]}',
        encoding="utf-8",
    )
    motion_command = object.__new__(MotionCommand)
    motion_command.device = "cpu"
    motion_command.use_adaptive_timesteps_sampler = True
    motion_command.motion = SimpleNamespace(
        num_clips=2,
        has_object=True,
        clip_ids=["clip_a", "clip_b"],
        clip_lengths=torch.tensor([5, 5]),
        fps=50.0,
    )
    motion_command.motion_cfg = SimpleNamespace(
        adaptive_sampling_contact_interval_root=str(tmp_path),
        uniform_t1_window_sampling_enabled=False,
        contact_interval_runtime_prepend_compensation=False,
    )
    motion_command._runtime_default_pose_prepend_enabled = False
    motion_command._runtime_default_pose_prepend_steps = 0
    monkeypatch.setenv("HOLOSOMA_REQUIRE_CONTACT_INTERVAL_COVERAGE", "1")

    with pytest.raises(RuntimeError, match=r"only 1/2 clips.*clip_b"):
        motion_command._configure_adaptive_sampling_contact_interval_bank()


def test_contact_interval_bank_rejects_duplicate_exact_and_numbered_directories(tmp_path):
    for directory_name in ("box_10", "0000_box_10"):
        clip_dir = tmp_path / directory_name
        clip_dir.mkdir()
        (clip_dir / "contact_intervals.json").write_text(
            '{"left_wrist":[1,3]}',
            encoding="utf-8",
        )
    motion_command = _contact_window_loader_stub(root=str(tmp_path), uniform_enabled=True)
    motion_command.motion.clip_lengths = torch.tensor([5])
    motion_command.motion.fps = 50.0
    motion_command.motion_cfg.contact_interval_runtime_prepend_compensation = False
    motion_command._runtime_default_pose_prepend_enabled = False
    motion_command._runtime_default_pose_prepend_steps = 0

    with pytest.raises(RuntimeError, match="Multiple adaptive contact directories"):
        motion_command._configure_adaptive_sampling_contact_interval_bank()


def test_contact_interval_bank_skips_nonobject_metadata_for_inactive_directory(tmp_path):
    active_dir = tmp_path / "0000_box_10"
    active_dir.mkdir()
    (active_dir / "contact_intervals.json").write_text(
        '{"left_wrist":[1,3]}',
        encoding="utf-8",
    )
    inactive_dir = tmp_path / "unrelated_clip"
    inactive_dir.mkdir()
    (inactive_dir / "metadata.json").write_text("[]", encoding="utf-8")

    motion_command = _contact_window_loader_stub(root=str(tmp_path), uniform_enabled=True)
    motion_command.motion.clip_lengths = torch.tensor([5])
    motion_command.motion.fps = 50.0
    motion_command.motion_cfg.contact_interval_runtime_prepend_compensation = False
    motion_command._runtime_default_pose_prepend_enabled = False
    motion_command._runtime_default_pose_prepend_steps = 0

    motion_command._configure_adaptive_sampling_contact_interval_bank()

    assert motion_command._adaptive_sampling_contact_window_valid_by_clip.tolist() == [True]
    assert motion_command._adaptive_sampling_contact_window_by_clip.tolist() == [[1, 3]]


def test_contact_interval_bank_rejects_nonobject_metadata_for_active_directory(tmp_path):
    active_dir = tmp_path / "box_10"
    active_dir.mkdir()
    (active_dir / "metadata.json").write_text("[]", encoding="utf-8")

    motion_command = _contact_window_loader_stub(root=str(tmp_path), uniform_enabled=True)

    with pytest.raises(ValueError, match="active clip must be a JSON object"):
        motion_command._configure_adaptive_sampling_contact_interval_bank()


def test_contact_interval_loader_converts_runtime_prepend_rollout_steps_to_motion_time(tmp_path):
    clip_dir = tmp_path / "0000_box_10"
    clip_dir.mkdir()
    (clip_dir / "contact_intervals.json").write_text(
        '{"left_wrist": [20, 30], "right_wrist": [22, 28]}',
        encoding="utf-8",
    )
    motion_command = object.__new__(MotionCommand)
    motion_command.motion_cfg = SimpleNamespace(contact_interval_runtime_prepend_compensation=True)
    motion_command._runtime_default_pose_prepend_enabled = True
    motion_command._runtime_default_pose_prepend_steps = 10

    assert motion_command._load_adaptive_sampling_contact_window_from_dir(clip_dir) == (10, 20)


def test_contact_interval_loader_preserves_legacy_runtime_prepend_semantics(tmp_path):
    clip_dir = tmp_path / "0000_box_10"
    clip_dir.mkdir()
    (clip_dir / "contact_intervals.json").write_text(
        '{"left_wrist": [20, 30]}',
        encoding="utf-8",
    )
    motion_command = object.__new__(MotionCommand)
    motion_command.motion_cfg = SimpleNamespace(contact_interval_runtime_prepend_compensation=False)
    motion_command._runtime_default_pose_prepend_enabled = True
    motion_command._runtime_default_pose_prepend_steps = 10

    assert motion_command._load_adaptive_sampling_contact_window_from_dir(clip_dir) == (20, 30)


def test_contact_interval_loader_converts_metadata_fps_before_runtime_prepend(tmp_path):
    clip_dir = tmp_path / "0000_clip"
    clip_dir.mkdir()
    (clip_dir / "contact_intervals.json").write_text(
        '{"left_wrist": [40, 143]}',
        encoding="utf-8",
    )
    motion_command = object.__new__(MotionCommand)
    motion_command.motion = SimpleNamespace(fps=50.0)
    motion_command.motion_cfg = SimpleNamespace(contact_interval_runtime_prepend_compensation=True)
    motion_command._runtime_default_pose_prepend_enabled = True
    motion_command._runtime_default_pose_prepend_steps = 10

    assert motion_command._load_adaptive_sampling_contact_window_from_dir(
        clip_dir,
        metadata={"fps": 30},
    ) == (57, 229)


def test_drop_button_observation_uses_manual_override():
    motion_command = object.__new__(MotionCommand)
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.manual_control_enabled = True
    motion_command.manual_drop_button_override_enabled = True
    motion_command.manual_drop_button = torch.tensor([[0.0], [1.0], [1.0]], dtype=torch.float32)

    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_state=lambda name: motion_command),
    )

    drop_button = wbt_obs_terms.drop_button(env)

    assert drop_button.tolist() == [[0.0], [1.0], [1.0]]


def test_pickup_button_observation_uses_manual_override():
    motion_command = object.__new__(MotionCommand)
    motion_command.motion = SimpleNamespace(has_object=True)
    motion_command.manual_control_enabled = True
    motion_command.manual_pickup_button_override_enabled = True
    motion_command.manual_pickup_button = torch.tensor([[1.0], [0.0], [0.0]], dtype=torch.float32)

    env = SimpleNamespace(
        num_envs=3,
        device=torch.device("cpu"),
        command_manager=SimpleNamespace(get_state=lambda name: motion_command),
    )

    pickup_button = wbt_obs_terms.pickup_button(env)

    assert pickup_button.tolist() == [[1.0], [0.0], [0.0]]
