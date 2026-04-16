from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from holosoma.config_values import robot as robot_values
from holosoma.viser_motion_geometry import _load_motion_qpos


def test_load_motion_qpos_reads_qpos_directly(tmp_path: Path) -> None:
    robot_cfg = robot_values.DEFAULTS["g1_29dof"]
    joint_names = np.asarray(robot_cfg.dof_names, dtype=object)

    root_pos = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    root_quat_wxyz = np.asarray([[0.5, 0.1, 0.2, 0.3]], dtype=np.float32)
    object_pos = np.asarray([[4.0, 5.0, 6.0]], dtype=np.float32)
    object_quat_wxyz = np.asarray([[0.6, 0.7, 0.8, 0.9]], dtype=np.float32)
    joint_pos = np.zeros((1, len(joint_names)), dtype=np.float32)
    qpos = np.concatenate((root_pos, root_quat_wxyz, joint_pos, object_pos, object_quat_wxyz), axis=1)

    motion_path = tmp_path / "raw_motion.npz"
    np.savez_compressed(
        motion_path,
        qpos=qpos,
        joint_names=joint_names,
        fps=np.asarray(30, dtype=np.int32),
    )

    loaded_qpos, fps = _load_motion_qpos(motion_path, robot_cfg, list(robot_cfg.dof_names))

    assert fps == 30
    np.testing.assert_allclose(loaded_qpos, qpos)


def test_load_motion_qpos_requires_qpos_and_does_not_fallback(tmp_path: Path) -> None:
    robot_cfg = robot_values.DEFAULTS["g1_29dof"]
    joint_names = np.asarray(robot_cfg.dof_names, dtype=object)
    body_names = np.asarray([robot_cfg.body_names[0]], dtype=object)

    motion_path = tmp_path / "raw_motion_missing_qpos.npz"
    np.savez_compressed(
        motion_path,
        joint_names=joint_names,
        joint_pos=np.zeros((1, len(joint_names)), dtype=np.float32),
        body_names=body_names,
        body_pos_w=np.zeros((1, 1, 3), dtype=np.float32),
        body_quat_w=np.asarray([[[1.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
        fps=np.asarray(30, dtype=np.int32),
    )

    with pytest.raises(ValueError, match="missing qpos"):
        _load_motion_qpos(motion_path, robot_cfg, list(robot_cfg.dof_names))
