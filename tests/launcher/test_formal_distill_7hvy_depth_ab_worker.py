from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts" / "formal_distill_7hvy_depth_ab_ws8_worker.sh"


def test_worker_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WORKER)], check=True)


def test_worker_rejects_unknown_contact_profile_before_node_or_asset_checks() -> None:
    args = [
        "bash",
        str(WORKER),
        "canary",
        "spatial",
        "192.0.2.1",
        "/missing/source",
        "/missing/persist",
        "29999",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "0" * 40,
        "1" * 40,
        "sw_threshold_schedule",
        "not-a-contact-profile",
    ]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "unsupported contact profile: not-a-contact-profile" in result.stderr


def test_contact_profile_is_a_single_reward_weight_ablation() -> None:
    text = WORKER.read_text()
    assert "g1-29dof-wbt-w-object-generalist-tracking-no-contact" in text
    assert "g1-29dof-wbt-w-object-generalist-offline-contact-guidance" in text
    assert "contact-weight=10.0" in text
    assert "wrist-weight=5.0" in text
    assert "force-threshold=1.0" in text
    assert 'positive_contact_reward=${POSITIVE_CONTACT_REWARD_VALUE}' in text

