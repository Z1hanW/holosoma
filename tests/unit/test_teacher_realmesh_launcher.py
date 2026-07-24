from __future__ import annotations

import importlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "scripts" / "launch_teacher_as_realmesh_rollout.sh"
PREPARE = REPO_ROOT / "scripts" / "prepare_teacher_as_realmesh_rollout.py"
LOCK_HOLDER = REPO_ROOT / "scripts" / "hold_no_follow_lock.py"
INFER = REPO_ROOT / "infer_teacher_as_contacts.sh"
EXPORTER = REPO_ROOT / "src" / "holosoma" / "holosoma" / "export_teacher_box_contacts.py"


@pytest.fixture(scope="module")
def exporter_module():
    for source_root in (
        REPO_ROOT / "src" / "holosoma",
        REPO_ROOT / "src" / "holosoma_inference",
        REPO_ROOT / "src",
    ):
        if str(source_root) not in sys.path:
            sys.path.insert(0, str(source_root))
    return importlib.import_module("holosoma.export_teacher_box_contacts")


@pytest.fixture
def launcher_fixture(tmp_path: Path) -> tuple[Path, dict[str, str], dict[str, Path]]:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    shutil.copy2(LAUNCHER, scripts / LAUNCHER.name)
    shutil.copy2(PREPARE, scripts / PREPARE.name)
    shutil.copy2(LOCK_HOLDER, scripts / LOCK_HOLDER.name)
    shutil.copy2(INFER, repo / INFER.name)
    fixture_exporter = repo / "src" / "holosoma" / "holosoma" / EXPORTER.name
    fixture_exporter.parent.mkdir(parents=True)
    shutil.copy2(EXPORTER, fixture_exporter)

    bank = repo / "data" / "ds_as_data" / "source"
    bank.mkdir(parents=True)
    clips = {}
    for clip_id in ("box_a", "box_b"):
        (bank / f"{clip_id}.npz").write_bytes(clip_id.encode("utf-8"))
        asset_dir = bank / "objects" / clip_id
        asset_dir.mkdir(parents=True)
        (asset_dir / "model.obj").write_text("o box\n", encoding="utf-8")
        (asset_dir / "model.urdf").write_text(
            '<robot name="box"><link name="object"><visual><geometry>'
            '<mesh filename="model.obj"/></geometry></visual></link></robot>\n',
            encoding="utf-8",
        )
        clips[clip_id] = {
            "object_name": clip_id,
            "object_size": [1.0, 1.0, 1.0],
            "object_urdf_path": f"objects/{clip_id}/model.urdf",
            "object_mesh_path": f"objects/{clip_id}/model.obj",
        }
    (bank / "_clip_object_urdf_map.json").write_text(
        json.dumps({"clips": clips}) + "\n",
        encoding="utf-8",
    )

    fake_bin = repo / "fake-bin"
    fake_bin.mkdir()
    nvidia_smi = fake_bin / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "if [[ \"$*\" == *\"--query-gpu=index\"* ]]; then\n"
        "  printf '0\\n7\\n'\n"
        "  exit 0\n"
        "fi\n"
        "exit 2\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(nvidia_smi.stat().st_mode | stat.S_IXUSR)

    paths = {
        "shards": repo / "data" / "ds_as_data" / "_teacher_rollout_shards" / "dry-run",
        "output": repo / "outputs" / "teacher_as_contacts" / "dry-run",
        "logs": repo / "logs" / "runtime" / "dry-run",
        "tmp": repo / "tmp" / "dry-run",
    }
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "PYTHON_BIN": sys.executable,
        "DRY_RUN": "1",
        "GPU_LIST": "0,7",
        "NUM_SHARDS": "2",
        "SOURCE_AS_DATA_DIR": str(bank),
        "SOURCE_AS_OBJECT_MAP": str(bank / "_clip_object_urdf_map.json"),
        "SOURCE_EXPECTED_TOTAL": "2",
        "SHARD_ROOT": str(paths["shards"]),
        "OUTPUT_ROOT": str(paths["output"]),
        "LOG_ROOT": str(paths["logs"]),
        "TMP_ROOT": str(paths["tmp"]),
        "TARGET_BANK": str(repo / "data" / "ds_as_data" / "target"),
    }
    return scripts / LAUNCHER.name, env, paths


def _run(launcher: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(launcher)],
        cwd=launcher.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_dry_run_accepts_sparse_visible_gpu_indices_and_writes_nothing(
    launcher_fixture: tuple[Path, dict[str, str], dict[str, Path]],
) -> None:
    launcher, env, paths = launcher_fixture
    paths["shards"].mkdir(parents=True)
    sentinel = paths["shards"] / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")

    completed = _run(launcher, env)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "shard_00: gpu=0" in completed.stdout
    assert "shard_01: gpu=7" in completed.stdout
    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert list(paths["shards"].iterdir()) == [sentinel]
    assert not paths["output"].exists()
    assert not paths["logs"].exists()
    assert not paths["tmp"].exists()


def test_duplicate_gpu_index_is_rejected_before_preparation(
    launcher_fixture: tuple[Path, dict[str, str], dict[str, Path]],
) -> None:
    launcher, env, paths = launcher_fixture
    env["GPU_LIST"] = "0,0"

    completed = _run(launcher, env)

    assert completed.returncode == 2
    assert "duplicate GPU index 0" in completed.stderr
    assert not paths["shards"].exists()


def test_unavailable_gpu_error_preserves_separate_index_records(
    launcher_fixture: tuple[Path, dict[str, str], dict[str, Path]],
) -> None:
    launcher, env, paths = launcher_fixture
    env["GPU_LIST"] = "0,1"

    completed = _run(launcher, env)

    assert completed.returncode == 2
    assert "unavailable GPU index 1" in completed.stderr
    assert "available=0 7" in completed.stderr
    assert "available=07" not in completed.stderr
    assert not paths["shards"].exists()


def test_independent_shard_command_clears_inherited_distributed_topology() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "unset WORLD_SIZE RANK GROUP_RANK ROLE_RANK ROLE_WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT" in source
    assert 'export LOCAL_RANK=0' in source
    assert 'export CUDA_VISIBLE_DEVICES="${gpu}"' in source
    assert 'export HOLOSOMA_DEVICE="cuda:0"' in source
    assert 'export TEACHER_ROLLOUT_PREPARED_MANIFEST_SHA256="${PREPARED_MANIFEST_SHA256}"' in source
    assert 'export TEACHER_ROLLOUT_EXPECTED_CLIP_IDS_FILE="${shard_dir}/clip_ids.txt"' in source
    assert 'export TEACHER_ROLLOUT_SHARD_NAME="${shard_name}"' in source
    assert 'exec setsid "${cmd[@]}"' in source
    assert 'wait -n -p completed_pid "${!ACTIVE_SHARDS[@]}"' in source
    assert 'kill -TERM -- "-${pid}"' in source
    assert 'LAUNCH_LOCK_NAME="global.lock"' in source
    assert "realpath -m" not in source


def test_launch_lock_holder_rejects_symlinked_lock_file(tmp_path: Path) -> None:
    lock_root = tmp_path / "locks"
    lock_root.mkdir()
    outside = tmp_path / "outside.lock"
    outside.write_text("unchanged\n", encoding="utf-8")
    (lock_root / "scope.lock").symlink_to(outside)

    completed = subprocess.run(
        [
            sys.executable,
            str(LOCK_HOLDER),
            "--root",
            str(lock_root),
            "--name",
            "scope.lock",
            "--timeout-seconds",
            "0",
        ],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "no-follow launch lock" in completed.stderr
    assert outside.read_text(encoding="utf-8") == "unchanged\n"


def test_launch_lock_holder_rejects_symlinked_parent_component(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "lock-parent").symlink_to(outside, target_is_directory=True)

    completed = subprocess.run(
        [
            sys.executable,
            str(LOCK_HOLDER),
            "--root",
            str(tmp_path / "lock-parent" / "locks"),
            "--name",
            "global.lock",
            "--timeout-seconds",
            "0",
        ],
        input="",
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "no-follow lock root" in completed.stderr
    assert not (outside / "locks").exists()


def test_prepared_export_requires_fresh_output_namespace(
    tmp_path: Path,
    exporter_module,
) -> None:
    fresh = tmp_path / "fresh-shard-output"
    exporter_module._prepare_export_output_namespace(fresh, prepared_rollout=True)
    assert fresh.is_dir()

    sentinel = fresh / "old-contact-overlay.glb"
    sentinel.write_bytes(b"must not be deleted")
    with pytest.raises(RuntimeError, match="already exists"):
        exporter_module._prepare_export_output_namespace(fresh, prepared_rollout=True)
    assert sentinel.read_bytes() == b"must not be deleted"

    outside = tmp_path / "outside-output"
    outside.mkdir()
    outside_sentinel = outside / "sentinel.txt"
    outside_sentinel.write_text("unchanged\n", encoding="utf-8")
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="already exists"):
        exporter_module._prepare_export_output_namespace(output_alias, prepared_rollout=True)
    assert outside_sentinel.read_text(encoding="utf-8") == "unchanged\n"


def test_prepared_export_inputs_use_lexical_component_no_follow(
    tmp_path: Path,
    exporter_module,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    object_map = tmp_path / "object-map.json"
    object_map.write_text(
        json.dumps(
            {
                "clips": {
                    "box_clip": {
                        "object_name": "box",
                        "object_size": [1.0, 1.0, 1.0],
                        "object_urdf_path": "objects/box/model.urdf",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    clip_ids = tmp_path / "clip_ids.txt"
    clip_ids.write_text("box_clip\n", encoding="utf-8")
    monkeypatch.setenv("AS_OBJECT_MAP", str(object_map))
    monkeypatch.setenv("TEACHER_ROLLOUT_EXPECTED_CLIP_IDS_FILE", str(clip_ids))
    payload, loaded_path = exporter_module._load_exact_rollout_object_map(expected_clip_ids=["box_clip"])
    assert loaded_path == object_map
    assert set(payload["clips"]) == {"box_clip"}
    assert exporter_module._load_expected_rollout_clip_ids() == ["box_clip"]

    map_alias = tmp_path / "object-map-alias.json"
    map_alias.symlink_to(object_map)
    monkeypatch.setenv("AS_OBJECT_MAP", str(map_alias))
    with pytest.raises(RuntimeError, match="without following symlinks"):
        exporter_module._load_exact_rollout_object_map(expected_clip_ids=["box_clip"])

    clip_ids_alias = tmp_path / "clip-ids-alias.txt"
    clip_ids_alias.symlink_to(clip_ids)
    monkeypatch.setenv("TEACHER_ROLLOUT_EXPECTED_CLIP_IDS_FILE", str(clip_ids_alias))
    with pytest.raises(RuntimeError, match="without following symlinks"):
        exporter_module._load_expected_rollout_clip_ids()

    parent_target = tmp_path / "parent-target"
    parent_target.mkdir()
    (parent_target / "object-map.json").write_bytes(object_map.read_bytes())
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(parent_target, target_is_directory=True)
    monkeypatch.setenv("AS_OBJECT_MAP", str(parent_alias / "object-map.json"))
    with pytest.raises(RuntimeError, match="without following symlinks"):
        exporter_module._load_exact_rollout_object_map(expected_clip_ids=["box_clip"])


def _write_python_wrapper(path: Path, *, checkpoint: Path | None = None, prepare_gate: Path | None = None) -> None:
    conditions = []
    if checkpoint is not None:
        conditions.append(
            "if any(arg.endswith('resolve_exact_checkpoint.py') for arg in sys.argv[1:]):\n"
            f"    print({str(checkpoint)!r})\n"
            "    raise SystemExit(0)\n"
        )
    if prepare_gate is not None:
        conditions.append(
            "if any(arg.endswith('prepare_teacher_as_realmesh_rollout.py') for arg in sys.argv[1:]) "
            "and 'prepare-shards' in sys.argv and '--dry-run' not in sys.argv:\n"
            f"    Path({str(prepare_gate)!r}).write_text('ready\\n')\n"
            "    time.sleep(30)\n"
        )
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys, time\n"
        "from pathlib import Path\n"
        + "".join(conditions)
        + f"os.execv({sys.executable!r}, [{sys.executable!r}, *sys.argv[1:]])\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_same_scope_second_launcher_fails_fast_without_touching_first_input(
    launcher_fixture: tuple[Path, dict[str, str], dict[str, Path]],
) -> None:
    launcher, env, paths = launcher_fixture
    gate = launcher.parents[1] / "prepare-entered.txt"
    wrapper = launcher.parents[1] / "python-wrapper"
    _write_python_wrapper(wrapper, prepare_gate=gate)
    env.update({"DRY_RUN": "0", "PYTHON_BIN": str(wrapper), "TEACHER_ROLLOUT_LAUNCH_LOCK_TIMEOUT_S": "0"})

    first = subprocess.Popen(
        ["bash", str(launcher)],
        cwd=launcher.parents[1],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not gate.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert gate.exists(), first.communicate(timeout=1)
        before = sorted(path.relative_to(paths["shards"]) for path in paths["shards"].rglob("*"))

        second = subprocess.run(
            ["bash", str(launcher)],
            cwd=launcher.parents[1],
            env=env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert second.returncode == 2
        assert "owns this shard/output/target scope" in second.stderr
        after = sorted(path.relative_to(paths["shards"]) for path in paths["shards"].rglob("*"))
        assert after == before
    finally:
        os.killpg(first.pid, signal.SIGKILL)
        first.wait(timeout=5)


def test_global_launch_lock_rejects_partial_resource_overlap(
    launcher_fixture: tuple[Path, dict[str, str], dict[str, Path]],
) -> None:
    launcher, env, paths = launcher_fixture
    repo = launcher.parents[1]
    gate = repo / "prepare-entered.txt"
    wrapper = repo / "python-wrapper"
    _write_python_wrapper(wrapper, prepare_gate=gate)
    env.update({"DRY_RUN": "0", "PYTHON_BIN": str(wrapper), "TEACHER_ROLLOUT_LAUNCH_LOCK_TIMEOUT_S": "0"})

    first = subprocess.Popen(
        ["bash", str(launcher)],
        cwd=repo,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not gate.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert gate.exists(), first.communicate(timeout=1)

        # Only OUTPUT_ROOT overlaps.  A tuple-hash scope lock gave this launch a
        # different key and allowed concurrent writers into the same output.
        second_env = dict(env)
        second_shards = repo / "data" / "ds_as_data" / "_teacher_rollout_shards" / "other-run"
        second_env["SHARD_ROOT"] = str(second_shards)
        second_env["TARGET_BANK"] = str(repo / "data" / "ds_as_data" / "other-target")
        second = subprocess.run(
            ["bash", str(launcher)],
            cwd=repo,
            env=second_env,
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        assert second.returncode == 2
        assert "owns this shard/output/target scope" in second.stderr
        assert not second_shards.exists()
        assert paths["output"].exists()
    finally:
        os.killpg(first.pid, signal.SIGKILL)
        first.wait(timeout=5)


def test_launch_loop_error_cleans_already_started_process_group(
    launcher_fixture: tuple[Path, dict[str, str], dict[str, Path]],
) -> None:
    launcher, env, paths = launcher_fixture
    repo = launcher.parents[1]
    checkpoint = repo / "teacher.pt"
    checkpoint.write_bytes(b"fixture checkpoint")
    wrapper = repo / "python-wrapper"
    _write_python_wrapper(wrapper, checkpoint=checkpoint)
    fake_infer = repo / "infer_teacher_as_contacts.sh"
    fake_infer.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "printf '%s\\n' \"$$\" > \"${FAKE_SHARD_PID_FILE}\"\n"
        "exec sleep 30\n",
        encoding="utf-8",
    )
    fake_infer.chmod(fake_infer.stat().st_mode | stat.S_IXUSR)
    pid_file = repo / "fake-shard.pid"
    bad_log = paths["logs"] / "shard_01.log"
    bad_log.mkdir(parents=True)
    env.update(
        {
            "DRY_RUN": "0",
            "PYTHON_BIN": str(wrapper),
            "TEACHER_CHECKPOINT": str(checkpoint),
            "FAKE_SHARD_PID_FILE": str(pid_file),
            "LAUNCH_VISER": "0",
        }
    )

    completed = _run(launcher, env)

    assert completed.returncode != 0
    assert "Terminating" in completed.stderr
    if pid_file.exists():
        shard_pid = int(pid_file.read_text().strip())
        with pytest.raises(ProcessLookupError):
            os.kill(shard_pid, 0)
