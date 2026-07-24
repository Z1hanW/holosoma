from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "scripts" / "wandb_replay_preflight.py"
SPEC = importlib.util.spec_from_file_location("wandb_replay_preflight_v2_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_manifest(path: Path, payload: dict[str, Any]) -> str:
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _required_binding_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: inputs[key]
        for key in (
            "world_size",
            "motion_clip_id",
            "motion_npz_sha256",
            "object_map_sha256",
            "object_urdf_sha256",
            "object_mesh_sha256",
            "single_slot_source_digest",
            "single_slot_view_digest",
            "rank_shard_source_digest",
            "transition_digest",
        )
    }


def _v2_payload(video_path: Path) -> dict[str, Any]:
    video_path.write_bytes(b"strict-rule90-v2-mp4-fixture")
    video_sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
    snapshot_digest = _sha("source-manifest")
    snapshot_id = f"src-{snapshot_digest}"
    archive_sha = _sha("source-archive")
    entrypoint_sha = _sha("dual-entrypoint")
    motion_sha = _sha("motion")
    transition_sha = _sha("transition")
    captured_at = "2026-07-18T08:00:00Z"

    actor = {
        "ordered_groups": list(preflight._RULE90_V2_ACTOR_GROUPS),
        "input_dim": 95,
        "history_length": 1,
    }
    selector = {"algorithm": "all_carry_regions_union", "version": 2}
    boundaries = {
        "pickup_at_t1_minus_1": 1,
        "pickup_at_t1": 0,
        "drop_at_t2_minus_1": 0,
        "drop_at_t2": 1,
    }
    button = {
        "mode": "kinematic_lift",
        "algorithm": "object_root_rel_z_v1",
        "lift_height_threshold_m": 0.10,
        "lift_range_ratio": 0.35,
        "sustained_frames": 5,
        "source_semantics": "global_multi_clip_runtime",
        "motion_fps": 10.0,
        "source_motion_sha256": motion_sha,
        "motion_transition_contract_sha256": transition_sha,
        "source_window": {"frame_count": 100, "t1": 20, "t2": 70},
        "materialized_window": {"frame_count": 120, "t1": 30, "t2": 80},
        "effective_prepend_frames": 10,
        "effective_append_frames": 10,
        "boundary_values": boundaries,
    }
    rule90 = {
        "actor": actor,
        "contact_selector": selector,
        "button_window": button,
        "root_carry_mode": "peak_height",
    }
    overlay = {
        "burned_in": True,
        "fields": list(preflight._RULE90_V2_OVERLAY_FIELDS),
        "frame_value_source": "source_motion_frame",
        "index_value_source": "materialized_zero_based_index",
        "button_value_source": "materialized_kinematic_lift_window",
    }
    inputs = {
        "world_size": 8,
        "motion_clip_id": "unscale__any_ball_29",
        "motion_npz_sha256": motion_sha,
        "object_map_sha256": _sha("object-map"),
        "object_urdf_sha256": _sha("object-urdf"),
        "object_mesh_sha256": _sha("object-mesh"),
        "single_slot_source_digest": _sha("single-source"),
        "single_slot_view_digest": _sha("single-view"),
        "rank_shard_source_digest": _sha("rank-shards"),
        "transition_digest": transition_sha,
    }
    run = {
        "fresh": True,
        "entity": "entity",
        "project": "carry-any",
        "run_id": "fresh-v2-run",
        "name": "strict-v2-run",
        "rule90": rule90,
    }
    source = {
        "snapshot_id": snapshot_id,
        "archive_sha256": archive_sha,
        "source_manifest_sha256": snapshot_digest,
        "entrypoint": {
            "archive_member": "distill_as_dual_button_solid.sh",
            "sha256": entrypoint_sha,
        },
    }
    binding_payload = {
        "version": 2,
        "run": {
            "entity": run["entity"],
            "project": run["project"],
            "run_id": run["run_id"],
            "name": run["name"],
        },
        "source": source,
        "inputs": _required_binding_inputs(inputs),
        "rule90": rule90,
        "overlay": overlay,
        "captured_at_utc": captured_at,
    }
    binding_sha = preflight._canonical_json_sha256(
        binding_payload, role="test binding"
    )
    run["capture"] = {
        "fresh": True,
        "run_id": run["run_id"],
        "source_snapshot_id": snapshot_id,
        "source_archive_sha256": archive_sha,
        "entrypoint_archive_member": source["entrypoint"]["archive_member"],
        "entrypoint_sha256": entrypoint_sha,
        "video_sha256": video_sha,
        "captured_at_utc": captured_at,
        "semantic_binding_sha256": binding_sha,
    }
    return {
        "version": 2,
        "run": run,
        "source": source,
        "inputs": inputs,
        "video": {
            "path": str(video_path),
            "sha256": video_sha,
            "size_bytes": video_path.stat().st_size,
            "ffprobe": {
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "fps": 10.0,
                "frame_count": 120,
                "duration_s": 12.0,
                "rule90_v2_binding_sha256": binding_sha,
            },
            "overlay": overlay,
        },
        "visual_review": {
            "passed": True,
            "video_sha256": video_sha,
            "reviewer": "reviewer",
            "reviewed_at_utc": "2026-07-18T08:01:00Z",
            "overlay_verified": True,
            "run_id": run["run_id"],
            "source_snapshot_id": snapshot_id,
            "semantic_binding_sha256": binding_sha,
        },
    }


def _v1_payload(video_path: Path) -> dict[str, Any]:
    payload = _v2_payload(video_path)
    payload["version"] = 1
    payload["run"] = {
        key: payload["run"][key]
        for key in ("fresh", "entity", "project", "run_id", "name")
    }
    payload["source"].pop("entrypoint")
    payload["source"].pop("source_manifest_sha256")
    payload["video"].pop("overlay")
    payload["video"]["ffprobe"].pop("rule90_v2_binding_sha256")
    payload["visual_review"] = {
        key: payload["visual_review"][key]
        for key in ("passed", "video_sha256", "reviewer", "reviewed_at_utc")
    }
    return payload


def _args(
    manifest_path: Path,
    manifest_sha: str,
    payload: dict[str, Any],
    *,
    required_version: int | None,
) -> SimpleNamespace:
    source = payload["source"]
    entrypoint = source.get("entrypoint", {})
    return SimpleNamespace(
        manifest=manifest_path,
        expected_manifest_sha256=manifest_sha,
        expected_source_snapshot_id=source["snapshot_id"],
        required_manifest_version=required_version,
        expected_source_archive_sha256=source.get("archive_sha256"),
        expected_entrypoint_archive_member=entrypoint.get("archive_member"),
        expected_entrypoint_sha256=entrypoint.get("sha256"),
        expected_entity=payload["run"]["entity"],
        expected_project=payload["run"]["project"],
        expected_run_id=payload["run"]["run_id"],
        expected_run_name=payload["run"]["name"],
        expected_world_size=payload["inputs"]["world_size"],
        ffprobe="ffprobe",
        ffprobe_timeout_seconds=1.0,
    )


def _fake_probe(payload: dict[str, Any], *, binding: str | None = None) -> Any:
    declaration = payload["video"]["ffprobe"]
    if binding is None:
        binding = declaration.get("rule90_v2_binding_sha256")
    return preflight.ProbeResult(
        width=declaration["width"],
        height=declaration["height"],
        fps=declaration["fps"],
        frame_count=declaration["frame_count"],
        duration_s=declaration["duration_s"],
        codec_name=declaration["codec_name"],
        rule90_v2_binding_sha256=binding,
    )


def _validate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    *,
    required_version: int | None,
    actual_binding: str | None = None,
) -> Any:
    manifest_path = tmp_path / "manifest.json"
    manifest_sha = _write_manifest(manifest_path, payload)
    probe = _fake_probe(payload, binding=actual_binding)
    monkeypatch.setattr(preflight, "_probe_video", lambda *args, **kwargs: probe)
    return preflight._validate_manifest(
        _args(
            manifest_path,
            manifest_sha,
            payload,
            required_version=required_version,
        )
    )


def test_v1_remains_accepted_but_a_v2_controller_rejects_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _v1_payload(tmp_path / "only.mp4")
    validated = _validate(
        tmp_path, monkeypatch, payload, required_version=None, actual_binding=None
    )
    assert validated.schema_version == 1
    assert validated.metadata["replay_preflight/schema_version"] == 1
    assert "replay_preflight/rule90_v2_contract" not in validated.metadata
    assert _validate(
        tmp_path, monkeypatch, payload, required_version=1, actual_binding=None
    ).schema_version == 1

    with pytest.raises(
        preflight.PreflightError,
        match="controller-required version 2",
    ):
        _validate(tmp_path, monkeypatch, payload, required_version=2)


def test_v2_accepts_and_mirrors_the_complete_semantic_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _v2_payload(tmp_path / "only.mp4")
    validated = _validate(tmp_path, monkeypatch, payload, required_version=2)
    contract = validated.metadata["replay_preflight/rule90_v2_contract"]
    assert validated.schema_version == 2
    assert validated.metadata["replay_preflight/schema_version"] == 2
    assert contract["actor"]["input_dim"] == 95
    assert contract["actor"]["ordered_groups"] == preflight._RULE90_V2_ACTOR_GROUPS
    assert contract["contact_selector"] == {
        "algorithm": "all_carry_regions_union",
        "version": 2,
    }
    assert contract["button_window"]["source_window"] == {
        "frame_count": 100,
        "t1": 20,
        "t2": 70,
    }
    assert contract["button_window"]["materialized_window"] == {
        "frame_count": 120,
        "t1": 30,
        "t2": 80,
    }
    assert contract["overlay"]["fields"] == ["frame", "index", "pickup", "drop"]


Mutation = Callable[[dict[str, Any]], None]


def _set(path: tuple[str, ...], value: Any) -> Mutation:
    def mutate(payload: dict[str, Any]) -> None:
        target: dict[str, Any] = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        (_set(("run", "rule90", "actor", "input_dim"), True), "JSON integer 95"),
        (
            _set(
                ("run", "rule90", "actor", "ordered_groups"),
                list(reversed(preflight._RULE90_V2_ACTOR_GROUPS)),
            ),
            "exact ordered dual-button 95D",
        ),
        (_set(("run", "rule90", "contact_selector", "version"), True), "JSON integer 2"),
        (_set(("run", "rule90", "button_window", "mode"), "contact_interval"), "mode"),
        (_set(("run", "rule90", "button_window", "source_window", "t1"), 0), "t1"),
        (_set(("run", "rule90", "button_window", "source_window", "t2"), 20), "t1 < t2"),
        (
            _set(("run", "rule90", "button_window", "materialized_window", "t1"), 31),
            "materialized t1/t2",
        ),
        (
            _set(("run", "rule90", "button_window", "effective_prepend_frames"), True),
            "must be an integer",
        ),
        (
            _set(
                (
                    "run",
                    "rule90",
                    "button_window",
                    "boundary_values",
                    "pickup_at_t1_minus_1",
                ),
                True,
            ),
            "JSON integer 1",
        ),
        (_set(("run", "rule90", "root_carry_mode"), "contact_interval"), "peak_height"),
        (_set(("video", "overlay", "burned_in"), 1), "JSON boolean true"),
        (_set(("video", "overlay", "fields"), ["frame", "index"]), "burned-in fields"),
        (
            _set(("video", "rule90_v2_binding_sha256"), _sha("conflicting-binding")),
            "conflicting declarations",
        ),
        (_set(("visual_review", "overlay_verified"), 1), "JSON boolean true"),
        (
            _set(("source", "entrypoint", "archive_member"), "../dual.sh"),
            "exact root-level formal-dual",
        ),
        (
            _set(
                ("source", "entrypoint", "archive_member"),
                "distill_as_button_solid.sh",
            ),
            "exact root-level formal-dual",
        ),
        (_set(("run", "capture", "fresh"), 1), "JSON boolean true"),
    ],
)
def test_v2_semantics_fail_closed_with_strict_types_and_ranges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Mutation,
    error: str,
) -> None:
    payload = _v2_payload(tmp_path / "only.mp4")
    mutation(payload)
    with pytest.raises(preflight.PreflightError, match=error):
        _validate(tmp_path, monkeypatch, payload, required_version=2)


def test_v2_old_run_identity_cannot_relabel_the_same_bound_mp4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _v2_payload(tmp_path / "only.mp4")
    payload["run"]["run_id"] = "different-fresh-run"
    payload["run"]["capture"]["run_id"] = "different-fresh-run"
    payload["visual_review"]["run_id"] = "different-fresh-run"
    # The old semantic binding remains embedded in the exact same MP4.  A new
    # run identity therefore cannot pass by merely editing and re-hashing JSON.
    with pytest.raises(preflight.PreflightError, match="canonical v2 identity"):
        _validate(tmp_path, monkeypatch, payload, required_version=2)


def test_v2_requires_the_binding_in_the_actual_mp4_not_only_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _v2_payload(tmp_path / "only.mp4")
    with pytest.raises(preflight.PreflightError, match="MP4 Rule-90 v2 binding metadata"):
        _validate(
            tmp_path,
            monkeypatch,
            payload,
            required_version=2,
            actual_binding=_sha("old-video-binding"),
        )


def test_v2_controller_archive_and_entrypoint_expectations_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _v2_payload(tmp_path / "only.mp4")
    manifest_path = tmp_path / "manifest.json"
    manifest_sha = _write_manifest(manifest_path, payload)
    args = _args(manifest_path, manifest_sha, payload, required_version=2)
    args.expected_entrypoint_sha256 = _sha("different-entrypoint")
    monkeypatch.setattr(
        preflight,
        "_probe_video",
        lambda *args, **kwargs: _fake_probe(payload),
    )
    with pytest.raises(preflight.PreflightError, match="controller expectation"):
        preflight._validate_manifest(args)


class _FakeRemoteRun:
    def __init__(
        self,
        *,
        state: object = "finished",
        last_history_step: object = -1,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.state = state
        self.lastHistoryStep = last_history_step
        self._rows = [] if rows is None else rows

    def scan_history(self, *, page_size: int) -> Any:
        assert page_size == 1
        return iter(self._rows)


def test_v2_remote_fresh_summary_only_prebind_is_accepted() -> None:
    preflight._verify_rule90_v2_remote_prebind_only(
        _FakeRemoteRun(),
        summary={"vis/replay": {"_type": "video-file"}},
    )


@pytest.mark.parametrize("state", ["running", "failed", "crashed", "killed"])
def test_v2_remote_non_prebind_states_are_rejected(state: str) -> None:
    with pytest.raises(preflight.PreflightError, match="prebind state 'finished'"):
        preflight._verify_rule90_v2_remote_prebind_only(
            _FakeRemoteRun(state=state), summary={}
        )


def test_v2_remote_summary_step_is_rejected() -> None:
    with pytest.raises(preflight.PreflightError, match="summary contains _step"):
        preflight._verify_rule90_v2_remote_prebind_only(
            _FakeRemoteRun(), summary={"_step": 0}
        )


@pytest.mark.parametrize("last_history_step", [0, 1, True, "0", -2])
def test_v2_remote_last_history_step_is_rejected(last_history_step: object) -> None:
    with pytest.raises(preflight.PreflightError, match="lastHistoryStep"):
        preflight._verify_rule90_v2_remote_prebind_only(
            _FakeRemoteRun(last_history_step=last_history_step), summary={}
        )


def test_v2_remote_any_history_row_is_rejected() -> None:
    with pytest.raises(preflight.PreflightError, match="already contains a history row"):
        preflight._verify_rule90_v2_remote_prebind_only(
            _FakeRemoteRun(rows=[{"_step": 0}]), summary={}
        )


def test_v2_remote_unavailable_history_scan_fails_closed() -> None:
    remote = _FakeRemoteRun()
    remote.scan_history = None
    with pytest.raises(preflight.PreflightError, match="bounded history scan"):
        preflight._verify_rule90_v2_remote_prebind_only(remote, summary={})


@pytest.mark.parametrize("include_binding", [False, True])
def test_ffprobe_reads_the_binding_from_actual_format_comment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_binding: bool,
) -> None:
    binding_sha = _sha("media-binding")
    tags = (
        {"comment": f"holosoma_rule90_v2_binding_sha256={binding_sha}"}
        if include_binding
        else {}
    )
    ffprobe_payload = {
        "streams": [
            {
                "codec_name": "h264",
                "width": 1280,
                "height": 720,
                "avg_frame_rate": "10/1",
                "nb_read_frames": "120",
                "duration": "12.0",
            }
        ],
        "format": {"duration": "12.0", "size": "1", "tags": tags},
    }
    monkeypatch.setattr(preflight.shutil, "which", lambda _binary: "/fake/ffprobe")
    monkeypatch.setattr(
        preflight.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps(ffprobe_payload).encode("utf-8"),
            stderr=b"",
        ),
    )
    result = preflight._probe_video(
        tmp_path / "video.mp4", ffprobe_binary="ffprobe", timeout_s=1.0
    )
    assert result.rule90_v2_binding_sha256 == (
        binding_sha if include_binding else None
    )
