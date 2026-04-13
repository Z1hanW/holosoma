from __future__ import annotations

import numpy as np
import pytest

from holosoma.utils.object_pose_correction import compute_box_ground_contact_translation_xyz_wxyz_np


def test_compute_box_ground_contact_translation_moves_box_down_to_ground():
    pos_w = np.asarray([[0.0, 0.0, 0.15]], dtype=np.float32)
    quat_wxyz = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    extents_xyz = np.asarray([0.2, 0.2, 0.2], dtype=np.float32)

    translation = compute_box_ground_contact_translation_xyz_wxyz_np(pos_w, quat_wxyz, extents_xyz)

    assert translation == pytest.approx((0.0, 0.0, -0.05), abs=1.0e-6)


def test_compute_box_ground_contact_translation_lifts_box_up_to_ground():
    pos_w = np.asarray([[0.0, 0.0, 0.05]], dtype=np.float32)
    quat_wxyz = np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    extents_xyz = np.asarray([0.2, 0.2, 0.2], dtype=np.float32)

    translation = compute_box_ground_contact_translation_xyz_wxyz_np(pos_w, quat_wxyz, extents_xyz)

    assert translation == pytest.approx((0.0, 0.0, 0.05), abs=1.0e-6)
