"""Rollout must cover the supplied bank without inheriting training state."""

import json

import pytest

from scripts._rollout import prepare


def inputs(tmp_path):
    bank = tmp_path / "motion bank"
    bank.mkdir()
    (bank / "_clip_object_urdf_map.json").write_text(json.dumps({"clips": {"a": {}, "b": {}}}))
    for clip in ("a", "b"):
        (bank / f"{clip}.npz").touch()
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.touch()
    return bank, ["--checkpoint", str(checkpoint), "--motion-bank", str(bank),
                  "--output", str(tmp_path / "output"), "--check"]


def test_full_coverage_and_native_rollout_environment(tmp_path, monkeypatch):
    bank, argv = inputs(tmp_path)
    for key in ("WANDB_RUN_ID", "HOLOSOMA_TRAINING_PROVENANCE", "HOLOSOMA_EVAL_POLICY", "WORLD_SIZE"):
        monkeypatch.setenv(key, "stale")
    _, cli, env, command = prepare(argv)
    assert "--training.num-envs=2" in cli
    assert f"--command.setup-terms.motion-command.params.motion-config.motion-file={bank}" in cli
    assert "--max-rollout-steps" not in cli
    assert env["HOLOSOMA_EVAL_POLICY"] == "checkpoint_actor"
    assert env["HOLOSOMA_DISABLE_AUTO_RESET"] == "1"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert all(key not in env for key in ("WANDB_RUN_ID", "HOLOSOMA_TRAINING_PROVENANCE", "WORLD_SIZE"))
    assert command[1:3] == ["-m", "holosoma.export_teacher_box_contacts"]
    assert not (tmp_path / "output").exists()


def test_missing_clip_cannot_be_silently_filtered(tmp_path):
    bank, argv = inputs(tmp_path)
    (bank / "b.npz").unlink()
    with pytest.raises(SystemExit):
        prepare(argv)


def test_existing_output_is_preserved(tmp_path):
    _, argv = inputs(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    (output / "keep.txt").write_text("existing output")
    with pytest.raises(SystemExit):
        prepare(argv)
    assert (output / "keep.txt").read_text() == "existing output"
