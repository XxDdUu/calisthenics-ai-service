import logging

import numpy as np
import pytest

from app.core.logging_context import set_request_id
from app.utils.pose_refinement import refine_wrists


def test_refine_wrists_logs_elbow_correction_and_preserves_knees(caplog):
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[5] = (100.0, 100.0, 0.95)
    keypoints[7] = (105.0, 110.0, 0.90)
    keypoints[9] = (100.0, 200.0, 0.40)
    keypoints[11] = (200.0, 100.0, 0.90)
    keypoints[13] = (330.0, 245.0, 0.88)
    original_knee = keypoints[13].copy()
    set_request_id("refinement-test")

    with caplog.at_level(logging.DEBUG, logger="app.utils.pose_refinement"):
        refined = refine_wrists(keypoints)

    np.testing.assert_allclose(refined[7, :2], (100.0, 152.0), atol=0.01)
    np.testing.assert_array_equal(refined[13], original_knee)
    assert any(
        "request_id=refinement-test" in record.message
        and "joint=left_elbow status=applied" in record.message
        and "raw=(105.00,110.00)" in record.message
        and "final=(100.00,152.00)" in record.message
        for record in caplog.records
    )


def test_refine_wrists_logs_when_elbow_correction_is_skipped(caplog):
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[5] = (100.0, 100.0, 0.95)
    keypoints[7] = (120.0, 120.0, 0.70)
    keypoints[9] = (100.0, 200.0, 0.40)
    set_request_id("refinement-skip-test")

    with caplog.at_level(logging.DEBUG, logger="app.utils.pose_refinement"):
        refined = refine_wrists(keypoints)

    np.testing.assert_array_equal(refined[7, :2], (120.0, 120.0))
    assert any(
        "request_id=refinement-skip-test" in record.message
        and "joint=left_elbow status=not_applied" in record.message
        and "reason=elbow_confidence_below_threshold" in record.message
        for record in caplog.records
    )
