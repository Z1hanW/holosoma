from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "train_object_generalist_ds.sh"


def _run_launcher(env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _early_contract_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHON_BIN": str(tmp_path / "must-not-run"),
        "EXPORT_ONNX": "True",
        "CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE": "tracking_error",
        "CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS": "",
        "CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG": "",
    }
    env.update(overrides)
    return env


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


def _dry_run_env(
    motion_dir: Path,
    object_map: Path,
    *,
    export_onnx: str,
    sparse_mode: str,
) -> dict[str, str]:
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
        "EXPORT_ONNX": export_onnx,
        "CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE": sparse_mode,
        "CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS": "000030",
        "CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG": "0.5",
    }


def test_t1_mode_with_default_onnx_fails_before_helpers_or_assets(tmp_path: Path) -> None:
    completed = _run_launcher(
        _early_contract_env(
            tmp_path,
            CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE="t1_aligned_segment",
        )
    )

    assert completed.returncode == 2
    assert (
        "CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=t1_aligned_segment is not implemented by inference "
        "and cannot be used with EXPORT_ONNX=True"
    ) in completed.stderr
    assert "Failed to detect available CUDA GPUs" not in completed.stderr


@pytest.mark.parametrize("value", ["maybe", "2", "none"])
def test_export_onnx_requires_a_strict_boolean(tmp_path: Path, value: str) -> None:
    completed = _run_launcher(_early_contract_env(tmp_path, EXPORT_ONNX=value))

    assert completed.returncode == 2
    assert f"EXPORT_ONNX must be a boolean. Got: {value}" in completed.stderr


@pytest.mark.parametrize("value", ["segment", "segment_30", "tracking-error", "TRACKING_ERROR"])
def test_sparse_root_mode_requires_a_canonical_value(tmp_path: Path, value: str) -> None:
    completed = _run_launcher(
        _early_contract_env(
            tmp_path,
            CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE=value,
        )
    )

    assert completed.returncode == 2
    assert (
        "CONTACT_AWARE_SPARSE_ROOT_COMMAND_MODE must be exactly tracking_error or "
        f"t1_aligned_segment. Got: {value}"
    ) in completed.stderr


@pytest.mark.parametrize(
    "value",
    ["0", "-1", "1.5", "1000001", "true", "9" * 80, "0" * 5000 + "1"],
)
def test_sparse_root_segment_steps_are_a_bounded_positive_integer(
    tmp_path: Path,
    value: str,
) -> None:
    completed = _run_launcher(
        _early_contract_env(
            tmp_path,
            CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS=value,
        )
    )

    assert completed.returncode == 2
    assert "CONTACT_AWARE_SPARSE_ROOT_SEGMENT_STEPS must be an integer in [1, 1000000]" in completed.stderr


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "180.1", "true", "1.2.3"])
def test_sparse_root_zero_yaw_is_finite_and_in_range(tmp_path: Path, value: str) -> None:
    completed = _run_launcher(
        _early_contract_env(
            tmp_path,
            CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG=value,
        )
    )

    assert completed.returncode == 2
    assert (
        "CONTACT_AWARE_SPARSE_ROOT_ZERO_YAW_THRESHOLD_DEG must be a finite number in [0, 180]"
        in completed.stderr
    )


@pytest.mark.parametrize(
    ("export_onnx", "sparse_mode"),
    [
        ("True", "tracking_error"),
        ("False", "t1_aligned_segment"),
    ],
)
def test_supported_export_and_sparse_mode_pairs_reach_one_explicit_training_cli(
    tmp_path: Path,
    export_onnx: str,
    sparse_mode: str,
) -> None:
    motion_dir, object_map = _generalist_bank(tmp_path)
    completed = _run_launcher(
        _dry_run_env(
            motion_dir,
            object_map,
            export_onnx=export_onnx,
            sparse_mode=sparse_mode,
        ),
        "pure-sd",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"[INFO] export_onnx={export_onnx}" in completed.stdout
    assert f"[INFO] contact_aware_sparse_root_command_mode={sparse_mode}" in completed.stdout
    assert completed.stdout.count(f"--training.export-onnx={export_onnx}") == 1
    assert completed.stdout.count(
        "--command.setup-terms.motion-command.params.motion-config."
        f"contact-aware-sparse-root-command-mode={sparse_mode}"
    ) == 1
    assert completed.stdout.count(
        "--command.setup-terms.motion-command.params.motion-config."
        "contact-aware-sparse-root-segment-steps=30"
    ) == 1
    assert completed.stdout.count(
        "--command.setup-terms.motion-command.params.motion-config."
        "contact-aware-sparse-root-zero-yaw-threshold-deg=0.5"
    ) == 1
