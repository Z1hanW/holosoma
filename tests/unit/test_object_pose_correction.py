from __future__ import annotations

import numpy as np

from holosoma.utils.object_pose_correction import (
    apply_omomo_largebox_semantic_pose_correction_wxyz_np,
    remove_omomo_largebox_semantic_pose_correction_wxyz_np,
    uses_omomo_largebox_primitive_semantics,
)


def test_largebox_semantic_pose_correction_round_trips() -> None:
    raw_pos = np.array([[0.42, -0.15, 0.88]], dtype=np.float32)
    raw_quat_wxyz = np.array([[0.9659258, 0.0, 0.0, 0.25881904]], dtype=np.float32)

    corrected_pos, corrected_quat = apply_omomo_largebox_semantic_pose_correction_wxyz_np(raw_pos, raw_quat_wxyz)
    recovered_pos, recovered_quat = remove_omomo_largebox_semantic_pose_correction_wxyz_np(
        corrected_pos,
        corrected_quat,
    )

    np.testing.assert_allclose(recovered_pos, raw_pos, atol=1.0e-6)
    np.testing.assert_allclose(recovered_quat, raw_quat_wxyz, atol=1.0e-6)


def test_largebox_semantics_detection_uses_urdf_path() -> None:
    assert uses_omomo_largebox_primitive_semantics(
        object_urdf_path="holosoma/data/motions/g1_29dof/whole_body_tracking/objects_largebox.urdf"
    )
    assert not uses_omomo_largebox_primitive_semantics(object_urdf_path="objects_boxsmall.urdf")
