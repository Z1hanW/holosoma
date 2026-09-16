from __future__ import annotations

from pathlib import Path
import subprocess
import json
import math

import pytest


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts" / "formal_prism137_teacher_ws32_worker.sh"


@pytest.mark.parametrize(
    "profile,pos,vel,pitch",
    [("baseline", 0.2, 0.35, 47.6), ("mgkt_joint_noise_47p6", 0.1, 0.0, 47.6),
     ("mgkt_joint_noise_37", 0.1, 0.0, 37.0),
     ("ch2_40k_joint_noise_47p6", 0.1, 0.0, 47.6)],
)
def test_ablation_profile_values(profile, pos, vel, pitch):
    source = WORKER.read_text()
    block = "case ${ABLATION_PROFILE} in" + source.split("case ${ABLATION_PROFILE} in", 1)[1].split("esac", 1)[0] + "esac"
    proc = subprocess.run(
        ["bash", "-c", "set -eu\nPOLICY_ARCH=command_student_large_mlp\n"
         + f"ABLATION_PROFILE={profile}\n" + block
         + '\nprintf "%s\\n" "$INITIAL_DOF_POS_NOISE" "$INITIAL_DOF_VEL_NOISE" "$CAMERA_PHYSICAL_PITCH_DEG" "$CAMERA_MOUNT_QUAT"'],
        check=True, text=True, capture_output=True,
    )
    values = proc.stdout.splitlines()
    assert list(map(float, values[:3])) == [pos, vel, pitch]
    quat = json.loads(values[3])
    assert quat == pytest.approx([0, math.sin(math.radians(pitch / 2)), 0, math.cos(math.radians(pitch / 2))])
    assert '--perception.camera-pitch-deg=0.0' in source
    assert '"ablation_profile": sys.argv[7]' in source
    assert '"initial_dof_pos_noise_rad": float(sys.argv[8])' in source


def test_ablation_rejects_unknown_profile_before_node_checks():
    args = ["canary", "command_student_large_mlp", "0", "192.0.2.1", "/missing/source",
            "/missing/persist", "192.0.2.2", "29999", "-", "-", "-", "-", "-", "-",
            "0" * 40, "1" * 40, "2" * 64, "3" * 64, "linear_startzero_0to1", "unknown"]
    proc = subprocess.run(["bash", str(WORKER), *args], text=True, capture_output=True)
    assert proc.returncode == 2
    assert "unsupported ablation profile" in proc.stderr


def test_worker_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WORKER)], check=True)


@pytest.mark.parametrize("producer", ["13k", "40k"])
def test_box23k_preserves_initializer_and_legacy_camera(producer):
    source = WORKER.read_text()
    block = "case ${ABLATION_PROFILE} in" + source.split("case ${ABLATION_PROFILE} in", 1)[1].split("esac", 1)[0] + "esac"
    proc = subprocess.run(
        ["bash", "-c", "set -eu\nPOLICY_ARCH=command_student_box23k\n"
         + f"ABLATION_PROFILE=box23k_corl_{producer}\n" + block
         + '\nprintf "%s\\n" "$POLICY_INIT_SHA" "$CAMERA_MOUNT_QUAT"'],
        check=True, text=True, capture_output=True,
    )
    digest, mount = proc.stdout.splitlines()
    assert digest == "e9de2954556f7f39c98cc5e90de2e28550dad4ba656c986280918c929af1256d"
    assert json.loads(mount) == [0.00644801, 0.23350163, 0.00644801, 0.97231365]
    init_block = source.split("PROVENANCE_INIT_ARGS=()", 1)[1].split('"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/validate_train_cli.py"', 1)[0]
    assert 'if [[ ${POLICY_ARCH} == command_student_box23k ]]' in init_block
    for flag in (
        "--perception.sensor-offset='[0.01,0.01,0.44]'",
        "--perception.camera-pitch-deg=10.0",
        "--perception.camera-warp-resize='[58,87]'",
        "--perception.camera-warp-latency-frame='[3,4]'",
        "--perception.camera-warp-buffer-len=6",
        "--perception.camera-warp-hole-reference-batch-size=4096",
        "--perception.camera-apply-sensor-noise=False",
        "--training.policy-init-actor-contract-migration=box_tracking_to_precomputed_kinematic_drop_exclusive_v1",
    ):
        assert flag in init_block


@pytest.mark.parametrize("arch,profile", [
    ("command_student_box23k", "baseline"),
    ("command_student_large_mlp", "box23k_corl_13k"),
])
def test_box23k_profile_cannot_leak_into_other_experiments(arch, profile):
    args = ["canary", arch, "0", "192.0.2.1", "/missing/source",
            "/missing/persist", "192.0.2.2", "29999", "-", "-", "-", "-", "-", "-",
            "0" * 40, "1" * 40, "2" * 64, "3" * 64, "linear_startzero_0to1", profile]
    proc = subprocess.run(["bash", str(WORKER), *args], text=True, capture_output=True)
    assert proc.returncode == 2
    assert "box23k" in proc.stderr
    assert "node-rank/IP mismatch" not in proc.stderr


def test_final_ch2_data_profile_fails_closed_without_git_bound_bank():
    args = ["canary", "command_student_large_mlp", "0", "192.0.2.1", "/missing/source",
            "/missing/persist", "192.0.2.2", "29999", "-", "-", "-", "-", "-", "-",
            "0" * 40, "1" * 40, "2" * 64, "3" * 64, "linear_startzero_0to1", "ch2_40k_joint_noise_47p6"]
    proc = subprocess.run(["bash", str(WORKER), *args], text=True, capture_output=True)
    assert proc.returncode == 2
    assert "invalid final40K rollout binding" in proc.stderr


def test_final_ch2_profile_binds_producer_and_canary_data():
    source = WORKER.read_text()
    assert 'parent["checkpoint"] == "model_40000"' in source
    assert 'parent["wandb_run"] == "zihanw22/carry-any/ch2ckwzw"' in source
    assert '"rollout_command_bank_digest": sys.argv[11]' in source
    assert '"rank_shard_digest": sys.argv[12]' in source
    assert 'rollout NPZ hash mismatch' in source


def test_worker_rejects_unknown_architecture_before_node_or_asset_checks() -> None:
    args = [
        "bash",
        str(WORKER),
        "canary",
        "gru",
        "0",
        "192.0.2.1",
        "/missing/source",
        "/missing/persist",
        "192.0.2.2",
        "29999",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "0" * 40,
        "1" * 40,
        "2" * 64,
        "3" * 64,
    ]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage:" in result.stderr


def test_worker_accepts_large_mlp_profile_before_node_or_asset_checks() -> None:
    args = [
        "bash",
        str(WORKER),
        "canary",
        "large_mlp",
        "0",
        "192.0.2.1",
        "/missing/source",
        "/missing/persist",
        "192.0.2.2",
        "29999",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "0" * 40,
        "1" * 40,
        "2" * 64,
        "3" * 64,
    ]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage:" not in result.stderr
    assert "node-rank/IP mismatch" in result.stderr


def test_worker_accepts_command_student_large_mlp_profile_before_node_checks() -> None:
    args = [
        "bash",
        str(WORKER),
        "canary",
        "command_student_large_mlp",
        "0",
        "192.0.2.1",
        "/missing/source",
        "/missing/persist",
        "192.0.2.2",
        "29999",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "0" * 40,
        "1" * 40,
        "2" * 64,
        "3" * 64,
    ]
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "usage:" not in result.stderr
    assert "node-rank/IP mismatch" in result.stderr


def test_worker_locks_formal_science_and_delivery_contract() -> None:
    text = WORKER.read_text()

    required_fragments = (
        "readonly NPROC=8 NNODES=4 WORLD_SIZE=32 ENVIRONMENTS_PER_RANK=2048",
        "--training.export-onnx=True",
        "readonly TARGET_ITERATIONS=40000 SAVE_INTERVAL=1000",
        "--algo.config.distill.enabled=False",
        "--algo.config.module-dict.actor.type=MLP",
        "--algo.config.module-dict.critic.type=MLP",
        "--algo.config.module-dict.actor.type=LSTM",
        "--algo.config.module-dict.critic.type=LSTM",
        "--algo.config.module-dict.actor.layer-config.lstm-hidden-dim=256",
        "--algo.config.module-dict.critic.layer-config.lstm-hidden-dim=256",
        "--algo.config.module-dict.actor.layer-config.hidden-dims='[512,256,128]'",
        "--algo.config.module-dict.actor.layer-config.hidden-dims='[2048,1024,512,256,128]'",
        "--algo.config.module-dict.critic.layer-config.hidden-dims='[512,256,128]'",
        "ch2ckwzw_model06000_rollout137_20260828",
        "688a4f1cdc170d4183190563a930aacc389fa5c6cf9768e7f95ad9d2e0d6dcc3",
        "449d15d287c20dd2d6f335144483aa9a706c0f40907e7d3d493192f261ecb3cb",
        "contact-aware-sparse-root-command-mode=tracking_error",
        "--reward.terms.offline-contact-guidance.weight=0.0",
        "HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=0",
        "HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK=0",
        "manifest[\"duplicated_to_fill_empty_ranks\"]",
        "set(manifest[\"clip_cover_counts\"].values()) != {1}",
        "--logger.resume=never",
        "verify_formal_git_checkout.py",
        "ch2ckwzw_model13000_rollout137_precomputed_turn_forward_v1",
        "contact-aware-sparse-root-command-mode=precomputed_turn_then_forward",
        "contact-aware-button-window-mode=kinematic_lift",
        "actor_obs_root_contact_aware",
        "actor_obs_drop_button",
        "actor_obs_proprio_with_actions_no_linvel",
        "HOLOSOMA_PERCEPTION_INJECT_INTO_POLICY_MODULES=True",
    )
    for fragment in required_fragments:
        assert fragment in text

    forbidden_fragments = (
        "SKIP_GIT_PULL=1",
        "WANDB_SKIP_UPLOAD=1",
        "SKIP_WANDB_UPLOAD=1",
        "HOLOSOMA_SHARD_OBJECT_ASSETS_BY_RANK=1",
        "HOLOSOMA_ALLOW_LEGACY_OBJECT_URDF_FALLBACK=1",
        "--algo.config.distill.enabled=True",
        "--training.export-onnx=False",
        "--algo.config.load-checkpoint",
        "--algo.config.init-from",
        "wandb_replay_preflight.py",
        "RULE90_",
    )
    for fragment in forbidden_fragments:
        assert fragment not in text


def test_only_architecture_differs_between_policy_branches() -> None:
    text = WORKER.read_text()
    branch = text.split("if [[ ${POLICY_ARCH} == lstm ]]; then", 1)[1].split("TRAIN_ARGS=(", 1)[0]
    assert "reward:" not in branch
    assert "randomization:" not in branch
    assert "command:" not in branch
    assert "perception:" not in branch
    assert "termination:" not in branch
    assert "motion-file" not in branch
    assert "learning-rate" not in branch
    assert "num-envs" not in branch
