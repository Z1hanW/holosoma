from __future__ import annotations

import numpy as np
import pytest

from scripts.validate_contact_sidecars import infer_clip_id, validate_contact_root


@pytest.mark.parametrize(
    ("directory_name", "expected"),
    [
        ("0034_scaledown__any_box_1", "scaledown__any_box_1"),
        ("scaledown__any_box_1", "scaledown__any_box_1"),
        ("clip_with_many_parts", "clip_with_many_parts"),
    ],
)
def test_infer_clip_id_only_strips_numeric_directory_prefix(directory_name: str, expected: str):
    assert infer_clip_id(directory_name) == expected


def _make_bank(tmp_path, *, clip_id: str = "clip_a", directory_name: str | None = None):
    motion_dir = tmp_path / "motion"
    motion_dir.mkdir()
    np.savez(
        motion_dir / f"{clip_id}.npz",
        fps=np.asarray([50.0], dtype=np.float32),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
    )

    contact_root = tmp_path / "contacts"
    clip_dir = contact_root / "clips" / (directory_name or f"0000_{clip_id}")
    clip_dir.mkdir(parents=True)
    for side in ("left_wrist", "right_wrist"):
        np.save(clip_dir / f"{side}_contact_points.npy", np.zeros((1, 3), dtype=np.float32))
        np.save(clip_dir / f"{side}_contact_point_counts.npy", np.ones((1,), dtype=np.int32))
        np.save(clip_dir / f"{side}_contact_interval_steps.npy", np.asarray([0, 1], dtype=np.int32))
    return motion_dir, contact_root, clip_dir


def _rollout_payload(steps=2, *, clip_id: str = "clip_a"):
    bodies = 2
    payload = {
        "clip_id": np.asarray(clip_id),
        "tracked_body_names": np.asarray(["pelvis", "torso_link"]),
        "ref_body_name": np.asarray("torso_link"),
        "trajectory_length": np.asarray(steps, dtype=np.int32),
        "valid_steps": np.ones((steps,), dtype=np.bool_),
        "body_pos_local": np.zeros((steps, bodies, 3), dtype=np.float32),
        "body_quat_w": np.zeros((steps, bodies, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((steps, bodies, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((steps, bodies, 3), dtype=np.float32),
        "ref_pos_local": np.zeros((steps, 3), dtype=np.float32),
        "ref_quat_w": np.zeros((steps, 4), dtype=np.float32),
        "ref_lin_vel_w": np.zeros((steps, 3), dtype=np.float32),
        "ref_ang_vel_w": np.zeros((steps, 3), dtype=np.float32),
        "root_pos_local": np.zeros((steps, 3), dtype=np.float32),
        "root_quat_w": np.zeros((steps, 4), dtype=np.float32),
        "root_lin_vel_w": np.zeros((steps, 3), dtype=np.float32),
        "root_ang_vel_w": np.zeros((steps, 3), dtype=np.float32),
        "object_pos_local": np.zeros((steps, 3), dtype=np.float32),
        "object_quat_w": np.zeros((steps, 4), dtype=np.float32),
        "object_lin_vel_w": np.zeros((steps, 3), dtype=np.float32),
        "object_ang_vel_w": np.zeros((steps, 3), dtype=np.float32),
    }
    for key in ("body_quat_w", "ref_quat_w", "root_quat_w", "object_quat_w"):
        payload[key][..., 3] = 1.0
    return payload


def _validate(
    motion_dir,
    contact_root,
    *,
    motion_end_mode="continuing",
    runtime_prepend_compensation=False,
    offline_contact_region_names=None,
    require_offline_contact_targets=True,
    expected_valid_runtime_windows=None,
):
    return validate_contact_root(
        motion_dir,
        contact_root,
        expected_total=1,
        tracked_body_names=["pelvis", "torso_link"],
        ref_body_name="torso_link",
        motion_end_mode=motion_end_mode,
        runtime_prepend_compensation=runtime_prepend_compensation,
        runtime_prepend_duration_s=0.2,
        offline_contact_region_names=offline_contact_region_names,
        require_offline_contact_targets=require_offline_contact_targets,
        expected_valid_runtime_windows=expected_valid_runtime_windows,
    )


def test_contact_sidecar_contract_accepts_consistent_payload(tmp_path):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload())
    assert _validate(motion_dir, contact_root) == (contact_root / "clips").resolve()


def test_contact_sidecar_contract_prefers_exact_numeric_leading_clip_id(tmp_path):
    clip_id = "2024_box_10"
    motion_dir, contact_root, clip_dir = _make_bank(
        tmp_path,
        clip_id=clip_id,
        directory_name=clip_id,
    )
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload(clip_id=clip_id))

    assert _validate(motion_dir, contact_root) == (contact_root / "clips").resolve()


def test_offline_contact_preflight_accepts_secondary_pitch_targets_when_wrists_are_empty(tmp_path):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    for side in ("left_wrist", "right_wrist"):
        np.save(clip_dir / f"{side}_contact_points.npy", np.zeros((0, 3), dtype=np.float32))
        np.save(clip_dir / f"{side}_contact_point_counts.npy", np.zeros((0,), dtype=np.int32))
    for side in ("left_wrist_pitch", "right_wrist_pitch"):
        np.save(clip_dir / f"{side}_contact_points.npy", np.zeros((1, 3), dtype=np.float32))
        np.save(clip_dir / f"{side}_contact_point_counts.npy", np.ones((1,), dtype=np.int32))
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload())

    assert _validate(motion_dir, contact_root) == (contact_root / "clips").resolve()
    with pytest.raises(ValueError, match="no non-empty offline contact target"):
        _validate(
            motion_dir,
            contact_root,
            offline_contact_region_names=["left_wrist", "right_wrist"],
        )


def test_no_positive_contact_reward_profile_allows_empty_contact_targets(tmp_path, capsys):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    for side in ("left_wrist", "right_wrist"):
        np.save(clip_dir / f"{side}_contact_points.npy", np.zeros((0, 3), dtype=np.float32))
        np.save(clip_dir / f"{side}_contact_point_counts.npy", np.zeros((0,), dtype=np.int32))
        np.save(clip_dir / f"{side}_contact_interval_steps.npy", np.asarray([-1, -1], dtype=np.int32))
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload())

    assert _validate(
        motion_dir,
        contact_root,
        require_offline_contact_targets=False,
        expected_valid_runtime_windows=0,
    ) == (contact_root / "clips").resolve()
    stderr = capsys.readouterr().err
    assert "required=False" in stderr
    assert "contact_clip_coverage=0/1" in stderr

    with pytest.raises(ValueError, match="no non-empty offline contact target"):
        _validate(motion_dir, contact_root)


def test_runtime_window_coverage_can_be_bound_exactly(tmp_path):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload())

    assert _validate(
        motion_dir,
        contact_root,
        expected_valid_runtime_windows=1,
    ) == (contact_root / "clips").resolve()
    with pytest.raises(ValueError, match=r"expected 0/1, found 1/1"):
        _validate(
            motion_dir,
            contact_root,
            expected_valid_runtime_windows=0,
        )


def test_runtime_contact_preflight_uses_secondary_pitch_interval_when_wrists_are_empty(
    tmp_path, capsys
):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    np.savez(
        motion_dir / "clip_a.npz",
        fps=np.asarray([50.0], dtype=np.float32),
        body_pos_w=np.zeros((300, 1, 3), dtype=np.float32),
    )
    for side in ("left_wrist", "right_wrist"):
        np.save(clip_dir / f"{side}_contact_points.npy", np.zeros((0, 3), dtype=np.float32))
        np.save(clip_dir / f"{side}_contact_point_counts.npy", np.zeros((0,), dtype=np.int32))
        np.save(clip_dir / f"{side}_contact_interval_steps.npy", np.asarray([-1, -1], dtype=np.int32))
    for side, interval in (
        ("left_wrist_pitch", [32, 293]),
        ("right_wrist_pitch", [26, 293]),
    ):
        np.save(clip_dir / f"{side}_contact_points.npy", np.zeros((1, 3), dtype=np.float32))
        np.save(clip_dir / f"{side}_contact_point_counts.npy", np.ones((1,), dtype=np.int32))
        np.save(clip_dir / f"{side}_contact_interval_steps.npy", np.asarray(interval, dtype=np.int32))
    (clip_dir / "contact_intervals.json").write_text(
        '{"left_wrist":[-1,-1],"right_wrist":[-1,-1],'
        '"left_wrist_pitch":[32,293],"right_wrist_pitch":[26,293],"torso":[132,135]}',
        encoding="utf-8",
    )
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload(steps=300))

    assert _validate(
        motion_dir,
        contact_root,
        runtime_prepend_compensation=True,
    ) == (contact_root / "clips").resolve()
    assert "valid_windows=1/1" in capsys.readouterr().err


def test_offline_contact_preflight_matches_runtime_stable_contact_filter(tmp_path):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    (clip_dir / "metadata.json").write_text(
        '{"clip_id":"clip_a","stable_contact_success":false}',
        encoding="utf-8",
    )
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload())

    with pytest.raises(ValueError, match="no non-empty offline contact target"):
        _validate(motion_dir, contact_root)


def test_contact_sidecar_terminal_frame_contract_depends_on_motion_end_mode(tmp_path):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    payload = _rollout_payload()
    payload["valid_steps"] = np.asarray([True, False])
    np.savez(clip_dir / "teacher_rollout_reference.npz", **payload)

    assert _validate(motion_dir, contact_root, motion_end_mode="episodic") == (
        contact_root / "clips"
    ).resolve()
    with pytest.raises(ValueError, match=r"first_invalid_step=1.*motion_end_mode=continuing"):
        _validate(motion_dir, contact_root, motion_end_mode="continuing")


def test_contact_sidecar_raw_interval_uses_rollout_timebase(tmp_path):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.asarray([8, 11], dtype=np.int32))
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload())
    # Raw rollout-step intervals may exceed the teacher reference length because
    # runtime removes the explicit default-pose prepend before clip-length checks.
    assert _validate(motion_dir, contact_root, runtime_prepend_compensation=True) == (
        contact_root / "clips"
    ).resolve()


def test_contact_sidecar_preflight_applies_runtime_prepend_only_when_enabled(tmp_path):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    np.savez(
        motion_dir / "clip_a.npz",
        fps=np.asarray([50.0], dtype=np.float32),
        body_pos_w=np.zeros((2, 1, 3), dtype=np.float32),
    )
    np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.asarray([8, 11], dtype=np.int32))
    np.save(clip_dir / "right_wrist_contact_interval_steps.npy", np.asarray([8, 11], dtype=np.int32))
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload())
    assert _validate(motion_dir, contact_root, runtime_prepend_compensation=True) == (
        contact_root / "clips"
    ).resolve()

    np.save(clip_dir / "right_wrist_contact_interval_steps.npy", np.asarray([8, 13], dtype=np.int32))
    with pytest.raises(ValueError, match="runtime contact interval"):
        _validate(motion_dir, contact_root, runtime_prepend_compensation=True)


@pytest.mark.parametrize(
    ("file_name", "value", "message"),
    [
        ("left_wrist_contact_point_counts.npy", np.asarray([-1], dtype=np.int32), "positive integers"),
        ("left_wrist_contact_interval_steps.npy", np.asarray([0.0, 1.5]), "integer rollout-step"),
    ],
)
def test_contact_sidecar_contract_rejects_invalid_contact_metadata(
    tmp_path, file_name, value, message
):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    np.save(clip_dir / file_name, value)
    np.savez(clip_dir / "teacher_rollout_reference.npz", **_rollout_payload())
    with pytest.raises(ValueError, match=message):
        _validate(motion_dir, contact_root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(clip_id=np.asarray("other")), "embedded clip_id"),
        (lambda payload: payload.update(tracked_body_names=np.asarray(["torso_link", "pelvis"])), "tracked_body_names"),
        (lambda payload: payload.update(ref_body_name=np.asarray("pelvis")), "ref_body_name"),
        (lambda payload: payload.update(valid_steps=np.asarray([False, False])), "no valid rollout step"),
        (lambda payload: payload.update(body_pos_local=np.zeros((1, 2, 3), dtype=np.float32)), "body_pos_local shape"),
        (lambda payload: payload.pop("object_pos_local"), "missing object_pos_local"),
        (lambda payload: payload["root_quat_w"].fill(0.0), "unit quaternions"),
    ],
)
def test_contact_sidecar_contract_rejects_semantic_mismatch(tmp_path, mutation, message):
    motion_dir, contact_root, clip_dir = _make_bank(tmp_path)
    payload = _rollout_payload()
    mutation(payload)
    np.savez(clip_dir / "teacher_rollout_reference.npz", **payload)
    with pytest.raises(ValueError, match=message):
        _validate(motion_dir, contact_root)
