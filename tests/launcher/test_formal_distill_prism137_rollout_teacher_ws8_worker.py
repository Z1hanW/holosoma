from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts/formal_distill_prism137_rollout_teacher_ws8_worker.sh"


def test_worker_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WORKER)], check=True)
    assert "export HOME=" not in WORKER.read_text()


def test_worker_rejects_unknown_teacher_arm_before_node_checks() -> None:
    args = [
        "bash",
        str(WORKER),
        "canary",
        "unknown_teacher",
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
    ]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_worker_binds_corresponding_teacher_rollout_and_label_source() -> None:
    text = WORKER.read_text()
    assert "__9X_" not in text
    assert "__CH2_" not in text
    assert "teacher_9x40k" in text
    assert "teacher_ch228k" in text
    assert "9xkizjec_model40000_rollout137_precomputed_turn_forward_v1" in text
    assert "ch2ckwzw_model28000_rollout137_precomputed_turn_forward_v1" in text
    assert "_single_slot_motion_bank/by-source/${SINGLE_SLOT_VIEW_DIGEST}" in text
    assert "HOLOSOMA_RANK_LOCAL_MOTION_ROOT=${RANK_SHARD_DIR}" in text
    assert "REQUIRE_MOTION_GENERATOR_TEACHER_MATCH=1" in text
    assert "MOTION_GENERATOR_TEACHER_EXPECTED_SHA256=${TEACHER_SHA256}" in text
    assert "command bank does not bind the exact raw rollout digest" in text
    assert "command bank does not bind the exact raw rollout manifest bytes" in text
    assert "T1_VALID_WINDOW_CLIPS=134" in text
    assert "T1_VALID_WINDOW_CLIPS=133" in text
    assert "OMOMO_EXPECTED_TOTAL=137" in text


def test_worker_fixes_fair_student_and_no_contact_contract() -> None:
    text = WORKER.read_text()
    required = [
        "far_tracking_cnn_small",
        "STUDENT_ACTOR_HIDDEN_DIMS='[512,256,128]'",
        "g1-29dof-wbt-w-object-generalist-tracking-no-contact",
        "ENABLE_OFFLINE_CONTACT_GUIDANCE_VALUE=False",
        "--reward.terms.offline-contact-guidance.weight=0.0",
        "HOLOSOMA_ACTIVATE_OBJECT_CONTACT_SENSORS=0",
        "CONTACT_SIDECAR_MODE=full-sidecars",
        "ALLOW_PARTIAL_CONTACT_SIDECARS=1",
        'PYTHONPATH="${SOURCE_ROOT}/src/holosoma:${SOURCE_ROOT}/src/holosoma_inference:${SOURCE_ROOT}/src"',
        "--allow-missing-offline-contact-targets",
        '--expected-valid-runtime-windows "${T1_VALID_WINDOW_CLIPS}"',
        "precomputed_turn_then_forward",
        "contact-aware-button-window-mode=kinematic_lift",
        "contact-aware-carry-window-mode=peak_height",
        "CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True",
        "clip-weighting-strategy=uniform_clip",
        "use-adaptive-timesteps-sampler=False",
        "UNIFORM_T1_WINDOW_SAMPLING_ENABLED=True",
        "UNIFORM_T1_WINDOW_HALF_WIDTH_STEPS=50",
        "UNIFORM_T1_WINDOW_DENSITY_BOOST=7.0",
        "DAGGER_MATCH_STD_VALUE=True",
        "PPO_START=0.1 PPO_TARGET=0.9",
        "export EXPORT_ONNX=True",
    ]
    for value in required:
        assert value in text


def test_generic_distill_launcher_propagates_partial_target_contract() -> None:
    text = (ROOT / "distill_as_perception.sh").read_text()
    assert "CONTACT_PARTIAL_TARGET_ARGS=(--allow-missing-offline-contact-targets)" in text
    assert '"${CONTACT_PARTIAL_TARGET_ARGS[@]}"' in text


def test_worker_uses_nominal_d435_camera_and_sw_bad_tracking_thresholds() -> None:
    text = WORKER.read_text()
    assert "--perception.sensor-offset='[0.0576235,0.01753,0.42987]'" in text
    assert "BAD_REF_POS=1.0 BAD_REF_ORI=1.2" in text
    assert "BAD_BODY_POS=0.55 BAD_OBJECT_POS=0.65 BAD_OBJECT_ORI=1.2" in text
