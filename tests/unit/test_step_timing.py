from __future__ import annotations

from holosoma.utils.step_timing import StepTiming, compact_timing_summary, env_flag


def test_env_flag_parses_truthy_and_falsy(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_STEP_TIMING", "yes")
    assert env_flag("HOLOSOMA_STEP_TIMING") is True

    monkeypatch.setenv("HOLOSOMA_STEP_TIMING", "0")
    assert env_flag("HOLOSOMA_STEP_TIMING", default=True) is False

    monkeypatch.delenv("HOLOSOMA_STEP_TIMING")
    assert env_flag("HOLOSOMA_STEP_TIMING", default=True) is True


def test_step_timing_snapshot_and_reset():
    timing = StepTiming(enabled=True)
    timing.add("env_step_total", 0.001)
    timing.add("env_step_total", 0.003)

    snapshot = timing.snapshot(reset=False)
    assert snapshot["env_step_total"]["sum_ms"] == 4.0
    assert snapshot["env_step_total"]["mean_ms"] == 2.0
    assert snapshot["env_step_total"]["count"] == 2.0

    assert timing.snapshot(reset=True)
    assert timing.snapshot() == {}


def test_step_timing_from_env_requires_profile_flag(monkeypatch):
    monkeypatch.setenv("HOLOSOMA_STEP_TIMING", "1")
    monkeypatch.delenv("HOLOSOMA_STEP_TIMING_PROFILE", raising=False)
    assert StepTiming.from_env().enabled is False

    monkeypatch.setenv("HOLOSOMA_STEP_TIMING_PROFILE", "1")
    assert StepTiming.from_env().enabled is True


def test_compact_timing_summary_uses_preferred_order():
    summary = compact_timing_summary(
        {
            "physics": {"sum_ms": 5.0, "mean_ms": 1.0, "count": 5.0},
            "post/reward": {"sum_ms": 7.0, "mean_ms": 1.4, "count": 5.0},
        },
        ("post/reward", "physics"),
    )

    assert summary.startswith("post/reward:sum=7.00ms")
    assert "physics:sum=5.00ms" in summary
