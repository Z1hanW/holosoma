from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch

from holosoma.config_types.command import MotionConfig
from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler, MotionCommand


def _multi_clip_sampler() -> AdaptiveTimestepsSampler:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=4,
        clip_lengths=torch.tensor([9, 14, 5]),
        valid_start_counts=torch.tensor([7, 12, 3]),
        adaptive_kernel_size=3,
        adaptive_lambda=0.6,
        adaptive_uniform_ratio=0.2,
    )
    sampler.bin_exposure_count[:] = torch.tensor(
        [
            [2.0, 4.0, 0.0],
            [3.0, 5.0, 8.0],
            [7.0, 0.0, 0.0],
        ]
    )
    sampler.bin_failed_count[:] = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [0.0, 4.0, 2.0],
            [3.0, 0.0, 0.0],
        ]
    )
    return sampler


def test_motion_command_constructor_defers_device_telemetry_until_setup() -> None:
    cfg = SimpleNamespace(
        params={
            "motion_config": MotionConfig(
                motion_file="unused.npz",
                body_name_ref=[],
                body_names_to_track=[],
            )
        }
    )
    command = MotionCommand(cfg, SimpleNamespace())

    assert not hasattr(command, "device")
    assert command._uniform_t1_window_last_reset_available_frac == 0.0
    command.device = "cpu"
    command._initialize_uniform_t1_window_metric_state()
    values = (
        command._uniform_t1_window_last_reset_available_frac,
        command._uniform_t1_window_last_reset_sample_frac,
        command._uniform_t1_window_last_reset_expected_sample_frac,
        command._uniform_t1_window_last_reset_sample_frac_valid,
        command._uniform_t1_window_last_reset_expected_sample_frac_valid,
        command._uniform_t1_window_last_reset_mean_window_len,
    )
    assert all(isinstance(value, torch.Tensor) for value in values)
    assert all(value.shape == torch.Size([]) for value in values)
    assert all(value.device.type == "cpu" for value in values)


@pytest.mark.parametrize(
    ("window_density_boost", "window_target_probability"),
    [(2.3, None), (1.0, 0.6), (1.0, 0.0), (1.0, 1.0)],
)
def test_batched_timestep_probabilities_match_scalar_oracle(
    window_density_boost: float,
    window_target_probability: float | None,
) -> None:
    sampler = _multi_clip_sampler()
    clip_ids = torch.tensor([2, 0, 1, 0, 2, 1])
    windows = torch.tensor([[1, 2], [2, 4], [5, 8], [2, 4], [1, 2], [5, 8]])

    batched = sampler.timestep_probabilities_for_samples(
        clip_ids,
        exclude_zero=True,
        windows=windows,
        window_density_boost=window_density_boost,
        window_target_probability=window_target_probability,
    )

    for row, (clip_id, window) in enumerate(zip(clip_ids.tolist(), windows.tolist(), strict=True)):
        scalar = sampler.timestep_probabilities_for_clip(
            clip_id,
            exclude_zero=True,
            window=(window[0], window[1]),
            window_density_boost=window_density_boost,
            window_target_probability=window_target_probability,
        )
        assert torch.allclose(batched[row, : scalar.numel()], scalar, atol=1.0e-7)
        assert torch.count_nonzero(batched[row, scalar.numel() :]) == 0
        assert torch.isclose(batched[row].sum(), torch.tensor(1.0), atol=1.0e-7)


def test_batched_target_probability_has_exact_window_mass() -> None:
    sampler = _multi_clip_sampler()
    clip_ids = torch.tensor([0, 1, 2, 1])
    windows = torch.tensor([[2, 4], [5, 8], [1, 1], [5, 8]])

    probabilities = sampler.timestep_probabilities_for_samples(
        clip_ids,
        exclude_zero=True,
        windows=windows,
        window_target_probability=0.63,
    )

    for row, (lo, hi) in enumerate(windows.tolist()):
        assert torch.isclose(probabilities[row, lo : hi + 1].sum(), torch.tensor(0.63), atol=1.0e-6)


def test_batched_sampling_matches_expected_distribution() -> None:
    sampler = _multi_clip_sampler()
    samples_per_clip = 30_000
    clip_ids = torch.tensor([0, 1]).repeat_interleave(samples_per_clip)
    windows = torch.tensor([[2, 4], [5, 8]]).repeat_interleave(samples_per_clip, dim=0)
    expected = sampler.timestep_probabilities_for_samples(
        torch.tensor([0, 1]),
        exclude_zero=True,
        windows=torch.tensor([[2, 4], [5, 8]]),
        window_target_probability=0.55,
    )

    torch.manual_seed(12345)
    sampled = sampler.sample_time_steps(
        clip_ids,
        exclude_zero=True,
        windows=windows,
        window_target_probability=0.55,
    )

    for clip_id in range(2):
        clip_samples = sampled[clip_ids == clip_id]
        empirical = torch.bincount(
            clip_samples,
            minlength=sampler.max_valid_start_count,
        ).to(dtype=torch.float32) / samples_per_clip
        assert torch.max(torch.abs(empirical - expected[clip_id])) < 0.012


def test_batched_sampling_preserves_short_clip_and_invalid_window_fallbacks() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=3,
        clip_lengths=torch.tensor([1, 2, 7]),
        valid_start_counts=torch.tensor([1, 2, 7]),
    )
    clip_ids = torch.tensor([0, 1, 2])
    windows = torch.tensor([[1, 4], [1, 1], [9, 12]])
    window_valid = torch.tensor([True, True, True])

    probabilities = sampler.timestep_probabilities_for_samples(
        clip_ids,
        exclude_zero=True,
        windows=windows,
        window_valid=window_valid,
        window_target_probability=1.0,
    )

    # A one-step clip cannot exclude its only supported timestep.  A two-step
    # clip similarly preserves its only nonzero support even for a target that
    # cannot create mass elsewhere.
    assert torch.equal(probabilities[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert torch.equal(probabilities[1], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    scalar_fallback = sampler.timestep_probabilities_for_clip(2, exclude_zero=True)
    assert torch.allclose(probabilities[2], scalar_fallback)
    inactive_window = sampler.timestep_probabilities_for_samples(
        torch.tensor([2]),
        exclude_zero=True,
        windows=torch.tensor([[2, 4]]),
        window_valid=torch.tensor([False]),
        window_target_probability=1.0,
    )
    assert torch.allclose(inactive_window[0], scalar_fallback)
    assert probabilities[:, 0].tolist() == [1.0, 0.0, 0.0]


def test_batched_sampler_fails_closed_only_for_selected_corrupt_clip() -> None:
    sampler = _multi_clip_sampler()
    sampler.bin_failed_count[1, 0] = float("nan")

    probabilities = sampler.timestep_probabilities_for_samples(torch.tensor([0, 2]))
    assert torch.isfinite(probabilities).all()

    with pytest.raises(RuntimeError, match="non-finite"):
        sampler.timestep_probabilities_for_samples(torch.tensor([1]))


@pytest.mark.parametrize(
    "clip_ids",
    [torch.tensor([0.0, 1.0]), torch.tensor([True, False])],
)
def test_public_sampler_rejects_non_integer_clip_ids(clip_ids: torch.Tensor) -> None:
    sampler = _multi_clip_sampler()

    with pytest.raises(ValueError, match="integer dtype"):
        sampler.sample_time_steps(clip_ids)


@pytest.mark.parametrize("bad_step", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("update_kind", ["exposure", "outcome"])
def test_public_count_updates_reject_nonfinite_time_steps(
    bad_step: float,
    update_kind: str,
) -> None:
    sampler = _multi_clip_sampler()
    time_steps = torch.tensor([0.0, bad_step])
    clip_ids = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="time_steps must be finite"):
        if update_kind == "exposure":
            sampler.update_current_bin_exposure_count(time_steps, clip_ids=clip_ids)
        else:
            sampler.update_current_bin_outcome_count(time_steps, clip_ids=clip_ids)


def test_weighted_reset_outcomes_preserve_observed_failure_semantics() -> None:
    sampler = AdaptiveTimestepsSampler(
        None,
        "cpu",
        env_fps=2,
        clip_lengths=torch.tensor([6, 6]),
        valid_start_counts=torch.tensor([4, 4]),
    )

    sampler.update_current_bin_outcome_count(
        torch.tensor([0, 1, 2, 3]),
        clip_ids=torch.tensor([0, 0, 1, 1]),
        failed=torch.tensor([True, True, False, True]),
        observed=torch.tensor([True, False, True, False]),
        _trusted_clip_ids=True,
    )

    assert torch.equal(sampler.current_bin_exposure_count, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(sampler.current_bin_failed_count, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


@pytest.mark.parametrize(
    ("windows", "window_valid"),
    [
        (torch.tensor([[1, 2], [2, 3]]), torch.tensor([True, True])),
        (torch.tensor([[1, 2], [1, 2]]), torch.tensor([True, False])),
    ],
)
def test_batched_sampler_rejects_inconsistent_per_clip_windows(
    windows: torch.Tensor,
    window_valid: torch.Tensor,
) -> None:
    sampler = _multi_clip_sampler()

    with pytest.raises(ValueError, match="same contact window and validity"):
        sampler.timestep_probabilities_for_samples(
            torch.tensor([0, 0]),
            windows=windows,
            window_valid=window_valid,
        )


def test_sample_time_steps_uses_batched_path_without_scalar_probability_calls(monkeypatch) -> None:
    sampler = _multi_clip_sampler()

    def _forbidden_scalar_path(*_args, **_kwargs):
        raise AssertionError("reset sampling must not rebuild probabilities clip-by-clip")

    monkeypatch.setattr(sampler, "timestep_probabilities_for_clip", _forbidden_scalar_path)
    sampled = sampler.sample_time_steps(torch.tensor([0, 1, 2, 0, 1, 2]), exclude_zero=True)

    assert sampled.shape == (6,)
    assert torch.all(sampled >= 1)
    assert torch.all(sampled < sampler.valid_start_counts[torch.tensor([0, 1, 2, 0, 1, 2])])


def test_trusted_reset_sampling_has_no_tensor_scalar_readback(monkeypatch) -> None:
    sampler = _multi_clip_sampler()
    clip_ids = torch.tensor([0, 1, 2, 0, 1, 2])
    windows = torch.tensor([[2, 4], [5, 8], [1, 2], [2, 4], [5, 8], [1, 2]])

    def _forbidden_readback(*_args, **_kwargs):
        raise AssertionError("trusted reset sampling must not materialize device tensors on the host")

    monkeypatch.setattr(torch.Tensor, "item", _forbidden_readback)
    monkeypatch.setattr(torch.Tensor, "tolist", _forbidden_readback)
    sampled, probabilities = sampler._sample_time_steps_with_probabilities(
        clip_ids,
        exclude_zero=True,
        windows=windows,
        window_target_probability=0.55,
        _trusted_inputs=True,
    )

    assert sampled.shape == (6,)
    assert probabilities.shape == (6, sampler.max_valid_start_count)


def test_uniform_t1_adaptive_reset_telemetry_matches_scalar_distribution(monkeypatch) -> None:
    sampler = _multi_clip_sampler()
    command = object.__new__(MotionCommand)
    command.device = "cpu"
    command.motion_cfg = SimpleNamespace(
        uniform_t1_window_sampling_enabled=True,
        uniform_t1_window_half_width_steps=1,
        uniform_t1_window_density_boost=1.0,
        uniform_t1_window_target_sample_frac=0.30,
    )
    command._env = SimpleNamespace(is_evaluating=False)
    command.use_adaptive_timesteps_sampler = True
    command.adaptive_timesteps_sampler = sampler
    command.clip_ids = torch.tensor([0, 1, 0, 1])
    command.time_steps = torch.tensor([2, 4, 5, 7])
    command._adaptive_sampling_contact_window_by_clip = torch.tensor([[3, 6], [6, 10], [1, 2]])
    command._adaptive_sampling_contact_window_valid_by_clip = torch.tensor([True, True, True])
    command._current_start_at_timestep_zero_prob = MethodType(lambda self: 0.25, command)
    valid_starts = sampler.valid_start_counts[command.clip_ids]

    conditional_target = 0.30 / 0.75
    expected_by_clip = []
    for clip_id, window in ((0, (2, 4)), (1, (5, 7))):
        probabilities = sampler.timestep_probabilities_for_clip(
            clip_id,
            exclude_zero=True,
            window=window,
            window_target_probability=conditional_target,
        )
        expected_by_clip.append(float(probabilities[window[0] : window[1] + 1].sum()) * 0.75)

    reset_probabilities = sampler.timestep_probabilities_for_samples(
        command.clip_ids,
        exclude_zero=True,
        windows=torch.tensor([[2, 4], [5, 7], [2, 4], [5, 7]]),
        window_target_probability=conditional_target,
    )

    def _forbidden_scalar_path(*_args, **_kwargs):
        raise AssertionError("reset telemetry must not rebuild probabilities clip-by-clip")

    monkeypatch.setattr(sampler, "timestep_probabilities_for_clip", _forbidden_scalar_path)
    monkeypatch.setattr(sampler, "timestep_probabilities_for_samples", _forbidden_scalar_path)
    command._record_uniform_t1_window_reset_metrics(
        torch.arange(4),
        valid_starts,
        adaptive_reset_probabilities=reset_probabilities,
    )

    expected_per_env = torch.tensor(
        [expected_by_clip[0], expected_by_clip[1], expected_by_clip[0], expected_by_clip[1]]
    )
    torch.testing.assert_close(
        command._uniform_t1_window_last_reset_available_frac,
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        command._uniform_t1_window_last_reset_sample_frac,
        torch.tensor(0.5),
    )
    torch.testing.assert_close(
        command._uniform_t1_window_last_reset_expected_sample_frac,
        expected_per_env.mean(),
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        command._uniform_t1_window_last_reset_expected_sample_frac_valid,
        expected_per_env.mean(),
        atol=1.0e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        command._uniform_t1_window_last_reset_mean_window_len,
        torch.tensor(3.0),
    )
