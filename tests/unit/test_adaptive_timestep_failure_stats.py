from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler, MotionCommand


class _RecordingAdaptiveSampler:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
        self.exposure_calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def update_current_bin_outcome_count(
        self,
        time_steps: torch.Tensor,
        *,
        clip_ids: torch.Tensor | None = None,
        failed: torch.Tensor | None = None,
        observed: torch.Tensor | None = None,
        _trusted_clip_ids: bool = False,
    ) -> None:
        del _trusted_clip_ids
        observed_mask = (
            torch.ones_like(time_steps, dtype=torch.bool)
            if observed is None
            else observed.to(dtype=torch.bool)
        )
        failed_mask = torch.ones_like(time_steps, dtype=torch.bool) if failed is None else failed
        self.calls.append(
            (
                time_steps[observed_mask].detach().clone(),
                None if clip_ids is None else clip_ids[observed_mask].detach().clone(),
                failed_mask[observed_mask].detach().clone(),
            )
        )

    def update_current_bin_exposure_count(
        self,
        time_steps: torch.Tensor,
        clip_ids: torch.Tensor | None = None,
        *,
        observed: torch.Tensor | None = None,
        _trusted_clip_ids: bool = False,
    ) -> None:
        del _trusted_clip_ids
        observed_mask = (
            torch.ones_like(time_steps, dtype=torch.bool)
            if observed is None
            else observed.to(dtype=torch.bool)
        )
        self.exposure_calls.append(
            (
                time_steps[observed_mask].detach().clone(),
                None if clip_ids is None else clip_ids[observed_mask].detach().clone(),
            )
        )


def _make_motion_command(terminated: torch.Tensor) -> tuple[MotionCommand, _RecordingAdaptiveSampler]:
    sampler = _RecordingAdaptiveSampler()
    command = object.__new__(MotionCommand)
    command.use_adaptive_timesteps_sampler = True
    command._env = SimpleNamespace(
        termination_manager=SimpleNamespace(terminated=terminated),
        episode_length_buf=torch.zeros(4, dtype=torch.long),
        _pending_episode_lengths=torch.ones(4, dtype=torch.long),
        is_evaluating=False,
    )
    command.time_steps = torch.tensor([4, 17, 29, 8], dtype=torch.long)
    command.clip_ids = torch.tensor([0, 2, 1, 3], dtype=torch.long)
    command.adaptive_timesteps_sampler = sampler
    return command, sampler


def test_adaptive_timestep_failure_stats_use_previous_clip_ids_before_resample() -> None:
    command, sampler = _make_motion_command(torch.tensor([False, True, False, True]))
    env_ids = torch.tensor([1, 3], dtype=torch.long)

    command._update_adaptive_timestep_failure_stats_before_resample(env_ids)
    command.clip_ids[env_ids] = torch.tensor([0, 0], dtype=torch.long)

    assert len(sampler.calls) == 1
    failed_steps, failed_clip_ids, failed = sampler.calls[0]
    assert torch.equal(failed_steps, torch.tensor([17, 8], dtype=torch.long))
    assert torch.equal(failed_clip_ids, torch.tensor([2, 3], dtype=torch.long))
    assert torch.equal(failed, torch.tensor([True, True]))


def test_adaptive_timestep_outcome_stats_record_nonfailure_exposure() -> None:
    command, sampler = _make_motion_command(torch.tensor([False, False, False, False]))

    command._update_adaptive_timestep_failure_stats_before_resample(torch.tensor([1, 3], dtype=torch.long))

    assert len(sampler.calls) == 1
    time_steps, clip_ids, failed = sampler.calls[0]
    assert torch.equal(time_steps, torch.tensor([17, 8]))
    assert torch.equal(clip_ids, torch.tensor([2, 3]))
    assert not torch.any(failed)


def test_adaptive_sampler_exact_fps_multiple_has_no_empty_tail_bin() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=30,
        clip_lengths=torch.tensor([62]),
        valid_start_counts=torch.tensor([60]),
    )

    assert sampler.num_bins_per_clip.tolist() == [2]
    assert sampler._bin_indices_for_clip(0).tolist() == ([0] * 30 + [1] * 30)


def test_adaptive_failure_bins_use_same_valid_start_coordinate_as_sampling() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=30,
        clip_lengths=torch.tensor([100]),
        valid_start_counts=torch.tensor([90]),
    )

    sampler.update_current_bin_failed_count(torch.tensor([29, 30, 59, 60, 89, 99]))

    # 99 is clamped to the last valid reset start (89); boundaries exactly
    # match the discrete ranges used by timestep_probabilities_for_clip().
    assert sampler.current_bin_failed_count[0].tolist() == [1.0, 2.0, 3.0]
    assert sampler.current_bin_exposure_count[0].tolist() == [1.0, 2.0, 3.0]


def test_adaptive_contact_target_composes_with_failure_density() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
        adaptive_uniform_ratio=0.1,
    )
    sampler.bin_failed_count[0] = torch.tensor([0.1, 0.9])
    sampler.bin_exposure_count[0] = torch.tensor([1.0, 1.0])

    base = sampler.timestep_probabilities_for_clip(0, exclude_zero=True)
    biased = sampler.timestep_probabilities_for_clip(
        0,
        exclude_zero=True,
        window=(2, 4),
        window_target_probability=0.6,
    )

    assert torch.isclose(biased[2:5].sum(), torch.tensor(0.6), atol=1.0e-6)
    # The contact reweight is multiplicative: failure-derived ratios within
    # the outside region remain intact instead of reverting to uniform.
    assert torch.isclose(biased[1] / biased[5], base[1] / base[5], atol=1.0e-6)


@pytest.mark.parametrize("target", [0.0, 0.6, 1.0])
def test_adaptive_contact_target_preserves_only_available_support(target: float) -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([6]),
        valid_start_counts=torch.tensor([4]),
    )

    probabilities = sampler.timestep_probabilities_for_clip(
        0,
        exclude_zero=True,
        window=(1, 3),
        window_target_probability=target,
    )

    assert torch.isfinite(probabilities).all()
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))
    assert probabilities[0].item() == 0.0
    assert torch.isclose(probabilities[1:].sum(), torch.tensor(1.0))


def test_adaptive_contact_target_cannot_create_missing_window_support() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([6]),
        valid_start_counts=torch.tensor([4]),
    )

    probabilities = sampler.timestep_probabilities_for_clip(
        0,
        exclude_zero=True,
        window=(0, 0),
        window_target_probability=1.0,
    )

    assert torch.isfinite(probabilities).all()
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))
    assert probabilities[0].item() == 0.0


@pytest.mark.parametrize("density", [True, 0.5, float("nan"), float("inf")])
def test_adaptive_contact_density_rejects_invalid_values(density: object) -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([6]),
        valid_start_counts=torch.tensor([4]),
    )

    with pytest.raises(ValueError, match="window_density_boost"):
        sampler.timestep_probabilities_for_clip(0, window_density_boost=density)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_adaptive_contact_target_rejects_invalid_values(target: object) -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([6]),
        valid_start_counts=torch.tensor([4]),
    )

    with pytest.raises(ValueError, match="window_target_probability"):
        sampler.timestep_probabilities_for_clip(0, window_target_probability=target)  # type: ignore[arg-type]


def test_adaptive_uniform_ratio_is_a_fixed_probability_mixture() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=1,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
        adaptive_uniform_ratio=0.1,
    )
    sampler.bin_failed_count[0, 0] = 10.0
    sampler.bin_exposure_count[0] = 10.0

    probabilities = sampler._sampling_probabilities_for_clip(0)

    assert torch.isclose(probabilities[0], torch.tensor(0.91), atol=1.0e-6)
    assert torch.allclose(probabilities[1:], torch.full((9,), 0.01), atol=1.0e-6)
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0), atol=1.0e-6)


@pytest.mark.parametrize("ratio", [-0.01, 1.01])
def test_adaptive_uniform_ratio_rejects_values_outside_unit_interval(ratio: float) -> None:
    with pytest.raises(ValueError, match="adaptive_uniform_ratio"):
        AdaptiveTimestepsSampler(
            None,
            "cpu",
            env_fps=1,
            clip_lengths=torch.tensor([4]),
            adaptive_uniform_ratio=ratio,
        )


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"env_fps": 0}, "env_fps"),
        ({"adaptive_kernel_size": 0}, "adaptive_kernel_size"),
        ({"adaptive_lambda": -0.1}, "adaptive_lambda"),
        ({"adaptive_lambda": float("nan")}, "adaptive_lambda"),
        ({"adaptive_alpha": -0.1}, "adaptive_alpha"),
        ({"adaptive_alpha": float("inf")}, "adaptive_alpha"),
    ],
)
def test_adaptive_sampler_rejects_invalid_hyperparameters(kwargs, expected) -> None:
    arguments = {
        "motion_time_step_total": None,
        "device": "cpu",
        "env_fps": 5,
        "clip_lengths": torch.tensor([12]),
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=expected):
        AdaptiveTimestepsSampler(**arguments)


@pytest.mark.parametrize(
    ("clip_lengths", "valid_start_counts", "expected"),
    [
        (torch.tensor([0]), None, "clip_lengths must be positive"),
        (torch.tensor([12.5]), None, "clip_lengths must use an integer dtype"),
        (torch.tensor([12]), torch.tensor([10.5]), "valid_start_counts must use an integer dtype"),
    ],
)
def test_adaptive_sampler_rejects_silently_truncated_geometry(
    clip_lengths,
    valid_start_counts,
    expected,
) -> None:
    with pytest.raises(ValueError, match=expected):
        AdaptiveTimestepsSampler(
            None,
            "cpu",
            env_fps=5,
            clip_lengths=clip_lengths,
            valid_start_counts=valid_start_counts,
        )


def test_adaptive_failure_rate_is_invariant_to_exposure_scale() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
        adaptive_uniform_ratio=0.2,
    )
    sampler.bin_failed_count[0] = torch.tensor([1.0, 3.0])
    sampler.bin_exposure_count[0] = torch.tensor([4.0, 4.0])
    probabilities = sampler._sampling_probabilities_for_clip(0)

    sampler.bin_failed_count.mul_(17.0)
    sampler.bin_exposure_count.mul_(17.0)

    assert torch.allclose(sampler._sampling_probabilities_for_clip(0), probabilities, atol=1.0e-7)


@pytest.mark.parametrize(
    ("failed", "exposure", "expected"),
    [
        (float("nan"), 1.0, "non-finite"),
        (-1.0, 1.0, "non-negative"),
        (2.0, 1.0, "exceeds exposure"),
    ],
)
def test_adaptive_sampling_fails_closed_on_corrupt_live_state(failed, exposure, expected) -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    sampler.bin_failed_count[0, 0] = failed
    sampler.bin_exposure_count[0, 0] = exposure

    with pytest.raises(RuntimeError, match=expected):
        sampler.timestep_probabilities_for_clip(0)


def test_motion_command_step_exposure_skips_envs_reset_before_step() -> None:
    command, sampler = _make_motion_command(torch.zeros(4, dtype=torch.bool))
    command._env.episode_length_buf = torch.tensor([0, 3, 0, 7], dtype=torch.long)

    command._record_adaptive_timestep_exposure_before_advance()

    assert len(sampler.exposure_calls) == 1
    time_steps, clip_ids = sampler.exposure_calls[0]
    assert torch.equal(time_steps, torch.tensor([17, 8]))
    assert torch.equal(clip_ids, torch.tensor([2, 3]))


def test_motion_end_is_not_failure_but_coincident_bad_tracking_is() -> None:
    command, _ = _make_motion_command(torch.tensor([True, True, False, False]))
    results = {
        "motion_ends": torch.tensor([True, True, False, False]),
        "bad_tracking": torch.tensor([False, True, False, False]),
    }
    command._env.termination_manager.cfg = SimpleNamespace(
        terms={
            "motion_ends": SimpleNamespace(is_timeout=False),
            "bad_tracking": SimpleNamespace(is_timeout=False),
        }
    )
    command._env.termination_manager.get_last_term_result = lambda name: results.get(name)

    failed = command._adaptive_failure_mask_for_env_ids(torch.tensor([0, 1]))

    assert torch.equal(failed, torch.tensor([False, True]))


def test_success_rate_counts_only_completed_nonfailure_outcomes() -> None:
    command, _ = _make_motion_command(torch.tensor([True, True, False, False]))
    command.multi_clip = True
    command.clip_weighting_strategy = "success_rate_adaptive"
    command._clip_success_counts = torch.zeros(4)
    command._clip_total_counts = torch.zeros(4)
    command._env.is_evaluating = False
    command._env._pending_episode_lengths = torch.tensor([20, 20, 0, 0])
    command.time_steps[:2] = 98
    command.clip_ids[:2] = torch.tensor([0, 1])
    command.motion = SimpleNamespace(clip_lengths=torch.tensor([100, 100, 100, 100]))
    command._refresh_adaptive_clip_weights = MethodType(lambda self: None, command)
    results = {
        "motion_ends": torch.tensor([True, True, False, False]),
        "bad_tracking": torch.tensor([False, True, False, False]),
    }
    command._env.termination_manager.cfg = SimpleNamespace(
        terms={
            "motion_ends": SimpleNamespace(is_timeout=False),
            "bad_tracking": SimpleNamespace(is_timeout=False),
        }
    )
    command._env.termination_manager.get_last_term_result = lambda name: results.get(name)

    command._update_clip_success_stats(torch.tensor([0, 1]))

    assert torch.equal(command._clip_total_counts[:2], torch.tensor([1.0, 1.0]))
    assert torch.equal(command._clip_success_counts[:2], torch.tensor([1.0, 0.0]))


def test_adaptive_sampler_checkpoint_round_trip_and_geometry_guard() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=30,
        clip_lengths=torch.tensor([62, 95]),
        valid_start_counts=torch.tensor([60, 90]),
    )
    sampler.current_bin_failed_count[0, 1] = 2.0
    sampler.bin_failed_count[1, 2] = 3.0
    sampler.current_bin_exposure_count[0, 1] = 5.0
    sampler.bin_exposure_count[1, 2] = 7.0
    state = sampler.state_dict()
    assert state["version"] == 3
    assert state["adaptive_kernel_size"] == 1
    assert state["adaptive_lambda"] == pytest.approx(0.8)
    assert state["adaptive_uniform_ratio"] == pytest.approx(0.1)
    assert state["adaptive_alpha"] == pytest.approx(0.001)

    restored = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=30,
        clip_lengths=torch.tensor([62, 95]),
        valid_start_counts=torch.tensor([60, 90]),
    )
    restored.load_state_dict(state)
    assert torch.equal(restored.current_bin_failed_count, sampler.current_bin_failed_count)
    assert torch.equal(restored.bin_failed_count, sampler.bin_failed_count)
    assert torch.equal(restored.current_bin_exposure_count, sampler.current_bin_exposure_count)
    assert torch.equal(restored.bin_exposure_count, sampler.bin_exposure_count)

    incompatible = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=30,
        clip_lengths=torch.tensor([62, 95]),
        valid_start_counts=torch.tensor([59, 90]),
    )
    with pytest.raises(ValueError, match="valid_start_counts"):
        incompatible.load_state_dict(state)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"adaptive_kernel_size": 3}, "adaptive_kernel_size"),
        ({"adaptive_lambda": 0.4}, "adaptive_lambda"),
        ({"adaptive_uniform_ratio": 0.25}, "adaptive_uniform_ratio"),
        ({"adaptive_alpha": 0.2}, "adaptive_alpha"),
    ],
)
def test_adaptive_sampler_checkpoint_rejects_hyperparameter_drift(override, expected) -> None:
    source = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    state = source.state_dict()
    restored = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
        **override,
    )

    with pytest.raises(ValueError, match=expected):
        restored.load_state_dict(state)


def test_legacy_adaptive_sampler_checkpoint_rejects_nondefault_runtime() -> None:
    source = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    legacy_state = source.state_dict()
    legacy_state["version"] = 2
    for key in (
        "adaptive_kernel_size",
        "adaptive_lambda",
        "adaptive_uniform_ratio",
        "adaptive_alpha",
    ):
        legacy_state.pop(key)
    restored = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
        adaptive_alpha=0.2,
    )

    with pytest.raises(ValueError, match="historical production defaults"):
        restored.load_state_dict(legacy_state)


def test_adaptive_sampler_loads_v1_failure_only_checkpoint() -> None:
    source = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    source.bin_failed_count[0] = torch.tensor([0.2, 0.8])
    legacy_state = source.state_dict()
    legacy_state["version"] = 1
    legacy_state.pop("current_bin_exposure_count")
    legacy_state.pop("bin_exposure_count")
    for key in (
        "adaptive_kernel_size",
        "adaptive_lambda",
        "adaptive_uniform_ratio",
        "adaptive_alpha",
    ):
        legacy_state.pop(key)

    restored = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    restored.load_state_dict(legacy_state)

    assert torch.equal(restored.bin_failed_count, source.bin_failed_count)
    assert torch.equal(restored.bin_exposure_count[0], torch.ones(2))


def test_adaptive_sampler_v1_migration_supports_failure_ema_above_one() -> None:
    source = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    source.bin_failed_count[0] = torch.tensor([2.0, 4.0])
    legacy_state = source.state_dict()
    legacy_state["version"] = 1
    for key in (
        "current_bin_exposure_count",
        "bin_exposure_count",
        "adaptive_kernel_size",
        "adaptive_lambda",
        "adaptive_uniform_ratio",
        "adaptive_alpha",
    ):
        legacy_state.pop(key)

    restored = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    restored.validate_state_dict(legacy_state)
    restored.load_state_dict(legacy_state)
    probabilities = restored.sampling_probabilities

    assert torch.equal(restored.bin_exposure_count[0], torch.tensor([4.0, 4.0]))
    assert torch.all(restored.bin_failed_count <= restored.bin_exposure_count)
    assert torch.isfinite(probabilities).all()
    assert probabilities.sum().item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("key", "invalid_value", "expected"),
    [
        ("current_bin_failed_count", float("nan"), "NaN or infinity"),
        ("bin_failed_count", float("inf"), "NaN or infinity"),
        ("current_bin_exposure_count", -1.0, "must be non-negative"),
        ("bin_exposure_count", -1.0, "must be non-negative"),
    ],
)
def test_adaptive_sampler_rejects_invalid_numeric_state_atomically(
    key: str,
    invalid_value: float,
    expected: str,
) -> None:
    source = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    state = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in source.state_dict().items()}
    state["current_bin_failed_count"][0, 0] = 1.0
    state["current_bin_exposure_count"][0, 0] = 1.0
    state[key][0, 1] = invalid_value

    restored = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    before = {
        name: getattr(restored, name).clone()
        for name in (
            "current_bin_failed_count",
            "bin_failed_count",
            "current_bin_exposure_count",
            "bin_exposure_count",
        )
    }

    with pytest.raises(ValueError, match=expected):
        restored.load_state_dict(state)

    for name, expected_value in before.items():
        assert torch.equal(getattr(restored, name), expected_value)


def test_adaptive_sampler_rejects_failure_greater_than_exposure() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    state = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in sampler.state_dict().items()}
    state["bin_failed_count"][0, 0] = 2.0
    state["bin_exposure_count"][0, 0] = 1.0

    with pytest.raises(ValueError, match="failure events must be a subset"):
        sampler.load_state_dict(state)


def test_adaptive_sampler_rejects_fractional_geometry_and_nonzero_padding() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([6, 12]),
        valid_start_counts=torch.tensor([5, 10]),
    )
    fractional = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in sampler.state_dict().items()}
    fractional["valid_start_counts"] = torch.tensor([5.9, 10.0])
    with pytest.raises(ValueError, match="valid_start_counts must use an integer dtype"):
        sampler.load_state_dict(fractional)

    padded = {name: value.clone() if isinstance(value, torch.Tensor) else value for name, value in sampler.state_dict().items()}
    assert sampler.num_bins_per_clip.tolist() == [1, 2]
    padded["bin_exposure_count"][0, 1] = 1.0
    with pytest.raises(ValueError, match="padded invalid bins"):
        sampler.load_state_dict(padded)


def test_motion_command_effective_distribution_uses_explicit_zero_mixture() -> None:
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.motion = SimpleNamespace(num_clips=1)
    command.motion_cfg = SimpleNamespace(
        uniform_t1_window_sampling_enabled=False,
        uniform_t1_window_density_boost=1.0,
        uniform_t1_window_target_sample_frac=None,
    )
    command._env = SimpleNamespace(is_evaluating=False)
    command.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=5,
        clip_lengths=torch.tensor([12]),
        valid_start_counts=torch.tensor([10]),
    )
    command._current_start_at_timestep_zero_prob = MethodType(lambda self: 0.25, command)

    probabilities = command._effective_adaptive_timestep_probabilities_for_clip(0)

    assert torch.isclose(probabilities[0], torch.tensor(0.25), atol=1.0e-7)
    assert torch.isclose(probabilities[1:].sum(), torch.tensor(0.75), atol=1.0e-7)


def _reset_curriculum_config(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "start_at_timestep_zero_prob": 0.2,
        "start_at_timestep_zero_prob_end": None,
        "start_at_timestep_zero_prob_start_iter": None,
        "start_at_timestep_zero_prob_end_iter": None,
        "freeze_at_timestep_zero_prob": 0.95,
        "freeze_at_timestep_zero_prob_end": None,
        "freeze_at_timestep_zero_prob_start_iter": None,
        "freeze_at_timestep_zero_prob_end_iter": None,
        "uniform_t1_window_sampling_enabled": False,
        "uniform_t1_window_half_width_steps": 50,
        "uniform_t1_window_density_boost": 1.0,
        "uniform_t1_window_target_sample_frac": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_reset_curriculum_config_accepts_defaults_and_complete_schedules() -> None:
    MotionCommand._validate_reset_sampling_curriculum_config(_reset_curriculum_config())
    MotionCommand._validate_reset_sampling_curriculum_config(
        _reset_curriculum_config(
            start_at_timestep_zero_prob_end=0.0,
            start_at_timestep_zero_prob_start_iter=100,
            start_at_timestep_zero_prob_end_iter=200,
            freeze_at_timestep_zero_prob_end=0.5,
            freeze_at_timestep_zero_prob_start_iter=0,
            freeze_at_timestep_zero_prob_end_iter=0,
            uniform_t1_window_sampling_enabled=True,
            uniform_t1_window_density_boost=7.0,
            uniform_t1_window_target_sample_frac=0.0,
        )
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start_at_timestep_zero_prob", True),
        ("start_at_timestep_zero_prob", -0.1),
        ("freeze_at_timestep_zero_prob", 1.1),
        ("freeze_at_timestep_zero_prob", float("nan")),
        ("freeze_at_timestep_zero_prob", float("inf")),
        ("uniform_t1_window_sampling_enabled", 1),
        ("uniform_t1_window_half_width_steps", True),
        ("uniform_t1_window_half_width_steps", 1.5),
        ("uniform_t1_window_half_width_steps", -1),
        ("uniform_t1_window_density_boost", True),
        ("uniform_t1_window_density_boost", 0.9),
        ("uniform_t1_window_density_boost", float("nan")),
        ("uniform_t1_window_density_boost", float("inf")),
        ("uniform_t1_window_target_sample_frac", True),
        ("uniform_t1_window_target_sample_frac", -0.1),
        ("uniform_t1_window_target_sample_frac", 1.1),
        ("uniform_t1_window_target_sample_frac", float("nan")),
        ("uniform_t1_window_target_sample_frac", float("inf")),
    ],
)
def test_reset_curriculum_config_rejects_invalid_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        MotionCommand._validate_reset_sampling_curriculum_config(
            _reset_curriculum_config(**{field: value})
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"start_at_timestep_zero_prob_end": 0.0},
        {"start_at_timestep_zero_prob_start_iter": 10},
        {
            "start_at_timestep_zero_prob_end": 0.0,
            "start_at_timestep_zero_prob_start_iter": 10,
        },
    ],
)
def test_reset_curriculum_config_rejects_partial_schedule(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="schedule must set"):
        MotionCommand._validate_reset_sampling_curriculum_config(
            _reset_curriculum_config(**overrides)
        )


@pytest.mark.parametrize(
    ("start_iter", "end_iter"),
    [(True, 10), (1.5, 10), (-1, 10), (11, 10)],
)
def test_reset_curriculum_config_rejects_invalid_iteration_bounds(
    start_iter: object,
    end_iter: object,
) -> None:
    with pytest.raises(ValueError, match="start_iter|end_iter"):
        MotionCommand._validate_reset_sampling_curriculum_config(
            _reset_curriculum_config(
                start_at_timestep_zero_prob_end=0.0,
                start_at_timestep_zero_prob_start_iter=start_iter,
                start_at_timestep_zero_prob_end_iter=end_iter,
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"uniform_t1_window_density_boost": 2.0},
        {"uniform_t1_window_target_sample_frac": 0.5},
    ],
)
def test_reset_curriculum_config_rejects_disabled_t1_settings(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="would be ignored"):
        MotionCommand._validate_reset_sampling_curriculum_config(
            _reset_curriculum_config(**overrides)
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "start_at_timestep_zero_prob": 0.4,
            "uniform_t1_window_sampling_enabled": True,
            "uniform_t1_window_target_sample_frac": 0.61,
        },
        {
            "start_at_timestep_zero_prob": 0.2,
            "start_at_timestep_zero_prob_end": 0.7,
            "start_at_timestep_zero_prob_start_iter": 10,
            "start_at_timestep_zero_prob_end_iter": 20,
            "uniform_t1_window_sampling_enabled": True,
            "uniform_t1_window_target_sample_frac": 0.31,
        },
    ],
)
def test_reset_curriculum_rejects_unrealizable_overall_t1_target(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="cannot be realized.*start-at-zero mixture"):
        MotionCommand._validate_reset_sampling_curriculum_config(
            _reset_curriculum_config(**overrides)
        )


def test_live_t1_target_never_silently_clips_to_nonzero_support() -> None:
    command = object.__new__(MotionCommand)
    command.motion_cfg = SimpleNamespace(
        uniform_t1_window_target_sample_frac=0.25,
    )
    command._current_start_at_timestep_zero_prob = MethodType(
        lambda self: 0.8,
        command,
    )

    with pytest.raises(RuntimeError, match="exceeds the live nonzero reset mass"):
        command._uniform_t1_window_conditional_target_probability()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_probability_clamp_rejects_nonfinite_values(value: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        MotionCommand._clamp01(value)


def test_uniform_t1_expected_probability_respects_support_constraint() -> None:
    command = object.__new__(MotionCommand)
    command.motion_cfg = SimpleNamespace(
        uniform_t1_window_target_sample_frac=0.0,
    )
    command._current_start_at_timestep_zero_prob = MethodType(lambda self: 0.0, command)

    probability = command._uniform_t1_window_probability(
        torch.tensor([3, 2]),
        torch.tensor([0, 4]),
    )

    assert torch.equal(probability, torch.tensor([1.0, 0.0]))
