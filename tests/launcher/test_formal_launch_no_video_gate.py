from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
LAUNCHERS = (
    ROOT / "batch_ne.sh",
    ROOT / "scripts" / "run_hmi_depth_stage1_ws8_formal.sh",
    ROOT / "scripts" / "run_hmi_depth_stage2_m8_ws8_formal.sh",
    ROOT / "scripts" / "formal_distill_7hvy_depth_ab_ws8_worker.sh",
    ROOT / "scripts" / "formal_distill_prism137_rollout_teacher_ws8_worker.sh",
    ROOT / "scripts" / "formal_prism137_teacher_ws32_worker.sh",
    ROOT / "scripts" / "formal_tuhu_t1_precontact_ws32_worker.sh",
)


def test_formal_launchers_do_not_depend_on_replay_or_video() -> None:
    forbidden = (
        "REPLAY_PREFLIGHT",
        "RULE90",
        "wandb_replay_preflight.py",
        "vis/replay",
    )
    for launcher in LAUNCHERS:
        text = launcher.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in text, f"{launcher.name} still contains {token}"


def test_changed_formal_launchers_have_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", *(str(path) for path in LAUNCHERS)], check=True)


def test_hmi_launchers_accept_direct_start_argument_shapes() -> None:
    stage1 = subprocess.run(
        ["bash", str(LAUNCHERS[1]), *("x" for _ in range(11))],
        check=False,
        capture_output=True,
        text=True,
    )
    assert stage1.returncode == 2
    assert "usage:" not in stage1.stderr
    assert "invalid expected commit" in stage1.stderr

    stage2 = subprocess.run(
        ["bash", str(LAUNCHERS[2]), *("x" for _ in range(15)), "object_xy"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert stage2.returncode == 2
    assert "usage:" not in stage2.stderr
    assert "invalid expected commit" in stage2.stderr


def test_direct_launch_keeps_non_video_integrity_gates() -> None:
    batch = (ROOT / "batch_ne.sh").read_text(encoding="utf-8")
    assert "ensure_local_source_snapshot" in batch
    assert "verify_python_runtimes_before_intent_parallel" in batch
    assert "preflight_external_as_asset_closures_parallel" in batch
    assert "preflight_selected_gpus_idle_parallel" in batch

    for launcher in LAUNCHERS[1:]:
        text = launcher.read_text(encoding="utf-8")
        assert "check_sha" in text
        assert "training.export-onnx=True" in text or "EXPORT_ONNX=True" in text
        assert "resume=must" not in text
        assert "RESUME=must" not in text
        assert "resume=never" in text or "RESUME=never" in text
