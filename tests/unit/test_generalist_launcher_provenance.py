from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from holosoma.utils.training_provenance import disabled_teacher_sha256
from scripts.compute_training_provenance import _motion_manifest_digest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _generalist_bank(tmp_path: Path) -> tuple[Path, Path]:
    motion_dir = tmp_path / "motion"
    asset_dir = motion_dir / "assets"
    asset_dir.mkdir(parents=True)
    np.savez(
        motion_dir / "clip_a.npz",
        fps=np.asarray([50.0]),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
    )
    (asset_dir / "object.obj").write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        encoding="utf-8",
    )
    (asset_dir / "object.urdf").write_text(
        "<robot name='object'><link name='base'><visual><geometry>"
        "<mesh filename='object.obj'/></geometry></visual><collision><geometry>"
        "<mesh filename='object.obj'/></geometry></collision></link></robot>",
        encoding="utf-8",
    )
    object_map = motion_dir / "_clip_object_urdf_map.json"
    object_map.write_text(
        json.dumps(
            {
                "clips": {
                    "clip_a": {
                        "object_urdf_path": "assets/object.urdf",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return motion_dir, object_map


def _launcher_env(motion_dir: Path, object_map: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHON_BIN": sys.executable,
        "CUDA_VISIBLE_DEVICES": "0",
        "NPROC": "1",
        "NNODES": "1",
        "PER_GPU_ENVS": "2",
        "TORCH_DIST_BACKEND": "gloo",
        "DRY_RUN": "1",
        "ASSERT_NEW_DS_DATA": "0",
        "STRICT_DEFAULT_DS_BANK_VALIDATION": "0",
        "AUTO_PREP_DS_BANK": "0",
        "MOTION_DIR": str(motion_dir),
        "OBJECT_SPEC_PATH": str(object_map),
        "CONTACT_EXPORT_ROOT": "",
        "GENERALIST_CONTACT_REWARD_ENABLED": "0",
    }


def test_generalist_dry_run_exports_teacher_free_scientific_provenance(tmp_path):
    motion_dir, object_map = _generalist_bank(tmp_path)
    completed = subprocess.run(
        ["bash", "train_object_generalist_ds.sh", "pure-sd"],
        cwd=REPO_ROOT,
        env=_launcher_env(motion_dir, object_map),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    prefix = "[INFO] training_provenance="
    payloads = [
        json.loads(line.removeprefix(prefix))
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert len(payloads) == 1
    provenance = payloads[0]
    assert provenance["training_regime"] == "pure_rl"
    assert provenance["teacher_enabled"] is False
    assert provenance["teacher_sha256"] == disabled_teacher_sha256()
    assert provenance["policy_init_enabled"] is False
    assert provenance["training_resume_enabled"] is False
    assert provenance["contact_interval_runtime_prepend_compensation"] is False
    assert provenance["runtime_asset_manifest_phase"] == "pending"
    assert provenance["environment"]["execution_runtime"]["NNODES"] == 1
    assert "[INFO] Final train command:" in completed.stdout
    assert (
        "--command.setup-terms.motion-command.params.motion-config."
        "contact-interval-runtime-prepend-compensation=False"
    ) in completed.stdout


def test_generalist_contact_compensation_requires_bound_contact_inputs(tmp_path):
    motion_dir, object_map = _generalist_bank(tmp_path)
    environ = _launcher_env(motion_dir, object_map)
    environ["CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION"] = "True"

    completed = subprocess.run(
        ["bash", "train_object_generalist_ds.sh", "pure-sd"],
        cwd=REPO_ROOT,
        env=environ,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "CONTACT_INTERVAL_RUNTIME_PREPEND_COMPENSATION=True requires active offline "
        "contact guidance and CONTACT_EXPORT_ROOT"
    ) in completed.stderr


def test_generalist_multinode_provenance_records_launcher_nnodes(tmp_path):
    motion_dir, object_map = _generalist_bank(tmp_path)
    environ = _launcher_env(motion_dir, object_map)
    environ.update(
        {
            "NNODES": "2",
            "NODE_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29571",
        }
    )

    completed = subprocess.run(
        ["bash", "train_object_generalist_ds.sh", "pure-sd"],
        cwd=REPO_ROOT,
        env=environ,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    prefix = "[INFO] training_provenance="
    payloads = [
        json.loads(line.removeprefix(prefix))
        for line in completed.stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert len(payloads) == 1
    assert payloads[0]["environment"]["execution_runtime"]["NPROC"] == 1
    assert payloads[0]["environment"]["execution_runtime"]["NNODES"] == 2
    assert "--nnodes=2" in completed.stdout


def test_generalist_rank_shard_manifest_matches_worker_revalidation(tmp_path):
    motion_dir, object_map = _generalist_bank(tmp_path)
    shard_root = tmp_path / "rank-shards"
    prepared = subprocess.run(
        [
            sys.executable,
            "scripts/prepare_as_rank_shards.py",
            "--motion-dir",
            str(motion_dir),
            "--object-map",
            str(object_map),
            "--output-root",
            str(shard_root),
            "--world-size",
            "1",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert prepared.returncode == 0, prepared.stdout + prepared.stderr

    environ = _launcher_env(motion_dir, object_map)
    environ["HOLOSOMA_RANK_LOCAL_MOTION_ROOT"] = str(shard_root)
    launched = subprocess.run(
        ["bash", "train_object_generalist_ds.sh", "pure-sd"],
        cwd=REPO_ROOT,
        env=environ,
        text=True,
        capture_output=True,
        check=False,
    )
    assert launched.returncode == 0, launched.stdout + launched.stderr

    prefix = "[INFO] training_provenance="
    payloads = [
        line.removeprefix(prefix)
        for line in launched.stdout.splitlines()
        if line.startswith(prefix)
    ]
    assert len(payloads) == 1
    provenance = json.loads(payloads[0])
    worker_digest = _motion_manifest_digest(
        motion_dir,
        object_map,
        shard_root / "manifest.json",
    )
    assert provenance["motion_shard_manifest_sha256"] == worker_digest


@pytest.mark.parametrize(
    "override",
    [
        "--command.setup-terms.motion-command.params.motion-config.motion-file=/tmp/other",
        "--command.setup_terms.motion_command.params.motion_config.motion_file=/tmp/other",
        "command:g1_29dof_wbt_w_object",
    ],
)
def test_generalist_rejects_extra_arg_that_can_replace_hashed_command_input(
    tmp_path,
    override,
):
    motion_dir, object_map = _generalist_bank(tmp_path)
    completed = subprocess.run(
        [
            "bash",
            "train_object_generalist_ds.sh",
            "pure-sd",
            "test-sequence",
            override,
        ],
        cwd=REPO_ROOT,
        env=_launcher_env(motion_dir, object_map),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "is provenance-owned and cannot be overridden" in completed.stderr


def test_generalist_rejects_unhashed_adaptive_contact_root(tmp_path):
    motion_dir, object_map = _generalist_bank(tmp_path)
    environ = _launcher_env(motion_dir, object_map)
    environ["ADAPTIVE_SAMPLING_CONTACT_INTERVAL_ROOT"] = str(tmp_path / "other-contact")
    completed = subprocess.run(
        ["bash", "train_object_generalist_ds.sh", "pure-sd"],
        cwd=REPO_ROOT,
        env=environ,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "active data input but CONTACT_EXPORT_ROOT is unset" in completed.stderr
