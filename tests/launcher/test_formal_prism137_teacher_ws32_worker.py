from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "scripts" / "formal_prism137_teacher_ws32_worker.sh"


def test_worker_has_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(WORKER)], check=True)


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
        "--logger.resume=must",
        "wandb_replay_preflight.py\" verify",
        "verify_formal_git_checkout.py",
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
