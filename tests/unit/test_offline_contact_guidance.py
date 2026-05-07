from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holosoma.config_types.reward import RewardTermCfg
import holosoma.managers.reward.terms.wbt as reward_wbt


class _DummyMotion:
    def __init__(self, clip_ids: list[str]):
        self.clip_ids = clip_ids
        self.num_clips = len(clip_ids)
        self.has_object = True


class _DummyMotionCommand:
    def __init__(
        self,
        *,
        clip_names: list[str],
        active_clip_indices: torch.Tensor,
        time_steps: torch.Tensor,
        body_names: list[str],
        body_positions_obj: torch.Tensor,
        body_force_magnitudes: torch.Tensor,
        pickup_steps_by_clip: torch.Tensor | None = None,
    ):
        self.motion = _DummyMotion(clip_names)
        self.clip_ids = active_clip_indices
        self.time_steps = time_steps
        self.num_envs = int(active_clip_indices.shape[0])
        self.device = str(active_clip_indices.device)
        self._body_names = list(body_names)
        self._body_positions_obj = body_positions_obj
        self._body_force_magnitudes = body_force_magnitudes
        self._body_name_to_index = {name: idx for idx, name in enumerate(self._body_names)}
        if pickup_steps_by_clip is None:
            pickup_steps_by_clip = torch.zeros((len(clip_names),), dtype=torch.long, device=active_clip_indices.device)
        self._pickup_steps_by_clip = pickup_steps_by_clip

    def _body_positions_in_object_frame(self, body_indices: torch.Tensor) -> torch.Tensor:
        return self._body_positions_obj.index_select(1, body_indices.to(dtype=torch.long))

    def get_body_object_contact_force_history(self, body_names: list[str]) -> torch.Tensor:
        indices = torch.tensor(
            [self._body_name_to_index[name] for name in body_names],
            device=self._body_positions_obj.device,
            dtype=torch.long,
        )
        magnitudes = self._body_force_magnitudes.index_select(1, indices)
        forces = torch.zeros(
            (self.num_envs, 1, len(body_names), 3),
            device=self._body_positions_obj.device,
            dtype=torch.float32,
        )
        forces[..., 0] = magnitudes.unsqueeze(1)
        return forces

    def _get_clip_pickup_steps_by_clip(self) -> torch.Tensor:
        return self._pickup_steps_by_clip


def _write_contact_clip(
    export_root: Path,
    *,
    clip_dir_name: str,
    clip_id: str,
    left_points: np.ndarray,
    right_points: np.ndarray,
    stable_contact_success: bool = True,
    write_metadata: bool = True,
    left_contact_interval: tuple[int, int] | None = None,
    right_contact_interval: tuple[int, int] | None = None,
) -> None:
    clip_dir = export_root / "clips" / clip_dir_name
    clip_dir.mkdir(parents=True, exist_ok=True)
    if write_metadata:
        metadata = {
            "clip_id": clip_id,
            "stable_contact_success": stable_contact_success,
        }
        (clip_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(clip_dir / "left_wrist_contact_points.npy", left_points.astype(np.float32))
    np.save(clip_dir / "right_wrist_contact_points.npy", right_points.astype(np.float32))
    for region_name in (
        "left_elbow",
        "right_elbow",
        "left_wrist_roll",
        "right_wrist_roll",
        "left_wrist_pitch",
        "right_wrist_pitch",
    ):
        np.save(clip_dir / f"{region_name}_contact_points.npy", np.zeros((0, 3), dtype=np.float32))
    np.save(clip_dir / "torso_contact_points.npy", np.zeros((0, 3), dtype=np.float32))
    if left_contact_interval is not None:
        np.save(clip_dir / "left_wrist_contact_interval_steps.npy", np.asarray(left_contact_interval, dtype=np.int32))
    if right_contact_interval is not None:
        np.save(clip_dir / "right_wrist_contact_interval_steps.npy", np.asarray(right_contact_interval, dtype=np.int32))


def _build_test_env(
    tmp_path: Path,
    *,
    left_force: float,
    right_force: float,
    time_steps: torch.Tensor | None = None,
    pickup_steps_by_clip: torch.Tensor | None = None,
    left_contact_interval: tuple[int, int] | None = None,
    right_contact_interval: tuple[int, int] | None = None,
):
    export_root = tmp_path / "teacher_box_contacts"
    _write_contact_clip(
        export_root,
        clip_dir_name="0000_box_10",
        clip_id="box_10",
        left_points=np.asarray([[0.20, 0.00, 0.00]], dtype=np.float32),
        right_points=np.asarray([[-0.20, 0.00, 0.00]], dtype=np.float32),
        left_contact_interval=left_contact_interval,
        right_contact_interval=right_contact_interval,
    )

    body_names = [
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_roll_link",
        "right_wrist_roll_link",
        "left_wrist_pitch_link",
        "right_wrist_pitch_link",
        "torso_link",
    ]
    body_positions_obj = torch.zeros((2, len(body_names), 3), dtype=torch.float32)
    body_positions_obj[0, body_names.index("left_wrist_yaw_link")] = torch.tensor([0.20, 0.00, 0.00])
    body_positions_obj[0, body_names.index("right_wrist_yaw_link")] = torch.tensor([-0.20, 0.00, 0.00])
    body_positions_obj[1, body_names.index("left_wrist_yaw_link")] = torch.tensor([0.55, 0.00, 0.00])
    body_positions_obj[1, body_names.index("right_wrist_yaw_link")] = torch.tensor([-0.55, 0.00, 0.00])

    body_force_magnitudes = torch.zeros((2, len(body_names)), dtype=torch.float32)
    body_force_magnitudes[0, body_names.index("left_wrist_yaw_link")] = left_force
    body_force_magnitudes[0, body_names.index("right_wrist_yaw_link")] = right_force
    body_force_magnitudes[1, body_names.index("left_wrist_yaw_link")] = left_force
    body_force_magnitudes[1, body_names.index("right_wrist_yaw_link")] = right_force

    motion_command = _DummyMotionCommand(
        clip_names=["box_10", "box_11"],
        active_clip_indices=torch.tensor([0, 1], dtype=torch.long),
        time_steps=torch.tensor([0, 0], dtype=torch.long) if time_steps is None else time_steps,
        body_names=body_names,
        body_positions_obj=body_positions_obj,
        body_force_magnitudes=body_force_magnitudes,
        pickup_steps_by_clip=pickup_steps_by_clip,
    )
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        simulator=SimpleNamespace(body_names=body_names),
        command_manager=SimpleNamespace(get_state=lambda name: motion_command if name == "motion_command" else None),
    )
    return env, export_root


def test_offline_contact_guidance_uses_clip_specific_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "force_sigma": 10.0,
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward.shape == (2,)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)
    assert reward[1].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_accepts_legacy_palm_region_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=40.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_palm", "right_palm"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "force_sigma": 10.0,
            },
            weight=1.0,
        ),
        env,
    )

    assert term.region_names == ["left_wrist", "right_wrist"]
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_soft_force_term_saturates_below_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=15.0, right_force=15.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "force_sigma": 10.0,
                "force_gate_mode": "soft",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(float(np.exp(-1.0)), rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_default_force_gate_requires_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=15.0, right_force=15.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "force_sigma": 10.0,
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_supports_outputs_clips_wrist_reach_without_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=0.0, right_force=0.0)
    clip_root = export_root / "clips"
    for child in list(clip_root.iterdir()):
        if child.is_dir():
            for file_path in child.iterdir():
                if file_path.name == "metadata.json":
                    file_path.unlink()

    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_binary_force_gate_requires_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(tmp_path, left_force=40.0, right_force=10.0)
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "force_threshold": 25.0,
                "use_force_term": True,
                "force_gate_mode": "binary",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.5, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_uses_contact_schedule_mask_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(
        tmp_path,
        left_force=40.0,
        right_force=40.0,
        time_steps=torch.tensor([0, 0], dtype=torch.long),
        left_contact_interval=(1, 3),
        right_contact_interval=(1, 3),
    )
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
                "use_contact_schedule": True,
                "contact_schedule_missing_mode": "always_on",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 1
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_falls_back_to_reference_pickup_gate_when_schedule_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(
        tmp_path,
        left_force=40.0,
        right_force=40.0,
        time_steps=torch.tensor([0, 0], dtype=torch.long),
        pickup_steps_by_clip=torch.tensor([2, 0], dtype=torch.long),
    )
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
                "use_contact_schedule": True,
                "contact_schedule_missing_mode": "after_pickup",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 2
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)


def test_offline_contact_guidance_relaxes_contact_interval_by_configured_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    env, export_root = _build_test_env(
        tmp_path,
        left_force=40.0,
        right_force=40.0,
        time_steps=torch.tensor([4, 4], dtype=torch.long),
        left_contact_interval=(10, 20),
        right_contact_interval=(10, 20),
    )
    motion_command = env.command_manager.get_state("motion_command")
    monkeypatch.setattr(reward_wbt, "_get_motion_command_and_assert_type", lambda _env: motion_command)
    term = reward_wbt.OfflineContactPointGuidance(
        RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:OfflineContactPointGuidance",
            params={
                "contact_export_root": str(export_root),
                "region_names": ["left_wrist", "right_wrist"],
                "position_sigma": 0.05,
                "use_force_term": False,
                "use_contact_schedule": True,
                "contact_schedule_relax_steps": 5,
                "contact_schedule_missing_mode": "always_on",
            },
            weight=1.0,
        ),
        env,
    )

    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 5
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 24
    reward = term(env)
    assert reward[0].item() == pytest.approx(1.0, rel=1e-5, abs=1e-5)

    motion_command.time_steps[0] = 25
    reward = term(env)
    assert reward[0].item() == pytest.approx(0.0, rel=1e-5, abs=1e-5)
