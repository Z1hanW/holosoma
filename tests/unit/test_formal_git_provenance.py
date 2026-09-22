from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "scripts/compute_training_provenance.py"
    spec = importlib.util.spec_from_file_location("formal_git_provenance_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "repo"
    root.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()

    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    git("commit", "--allow-empty", "-m", "locked source")
    commit = git("rev-parse", "HEAD")
    payload = {
        "accepted": True, "source_root": str(root),
        "commit_sha": commit, "tree_sha": git("rev-parse", "HEAD^{tree}"),
        "fetched_ref_commit": commit, "remote_url": "https://example.invalid/repo",
        "remote_ref": "main", "declared_submodules": {},
        "tracked_diff_clean": True, "untracked_clean": True,
        "legacy_unmapped_gitlinks_inactive_and_empty": True,
    }
    proof = tmp_path / "proof.json"
    monkeypatch.setenv(module.FORMAL_GIT_VERIFICATION_PATH_ENV, str(proof))
    monkeypatch.setattr(module, "__file__", str(root / "scripts/compute_training_provenance.py"))

    def check():
        proof.write_text(json.dumps(payload))
        return module._formal_git_identity_from_env()

    return git, payload, check


def test_pinned_commit_can_equal_remote_tip(checkout):
    _, payload, check = checkout
    assert check()["git_commit_sha"] == payload["commit_sha"]


def test_pinned_commit_survives_remote_branch_advancing(checkout):
    git, payload, check = checkout
    git("commit", "--allow-empty", "-m", "later controller change")
    payload["fetched_ref_commit"] = git("rev-parse", "HEAD")
    git("checkout", "--detach", payload["commit_sha"])
    result = check()
    assert result["git_commit_sha"] == payload["commit_sha"]
    assert result["git_fetched_ref_commit"] != result["git_commit_sha"]


def test_unrelated_remote_history_rejected(checkout):
    git, payload, check = checkout
    git("checkout", "--orphan", "unrelated")
    git("commit", "--allow-empty", "-m", "unrelated root")
    payload["fetched_ref_commit"] = git("rev-parse", "HEAD")
    git("checkout", "--detach", payload["commit_sha"])
    with pytest.raises(ValueError, match="not reachable"):
        check()


def test_live_checkout_cannot_move_after_verification(checkout):
    git, _, check = checkout
    git("commit", "--allow-empty", "-m", "changed checkout")
    with pytest.raises(ValueError, match="live HEAD"):
        check()


@pytest.mark.parametrize("key,value", [
    ("source_root", "/another/checkout"), ("accepted", False),
    ("tree_sha", "a" * 40), ("fetched_ref_commit", "b" * 40),
])
def test_invalid_proof_rejected(checkout, key, value):
    _, payload, check = checkout
    payload[key] = value
    with pytest.raises(ValueError):
        check()
