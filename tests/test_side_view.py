import pytest
from app.biomechanics.metrics import BiomechanicsEvaluator, BiomechanicsProfile
from app.biomechanics.side_view import (
    SideViewKeypointProcessor,
    compute_midpoint,
)
from app.schemas.analysis import KeypointData, PoseKeypoints


def create_coco_keypoints(conf: float = 0.9):
    """Creates a standard set of 17 COCO keypoints for a horizontal athlete."""
    return {
        "nose": KeypointData(x=80, y=100, confidence=conf, name="nose", visible=True),
        "left_shoulder": KeypointData(x=100, y=95, confidence=conf, name="left_shoulder", visible=True),
        "right_shoulder": KeypointData(x=100, y=105, confidence=conf, name="right_shoulder", visible=True),
        "left_elbow": KeypointData(x=100, y=150, confidence=conf, name="left_elbow", visible=True),
        "right_elbow": KeypointData(x=100, y=150, confidence=conf, name="right_elbow", visible=True),
        "left_wrist": KeypointData(x=100, y=200, confidence=conf, name="left_wrist", visible=True),
        "right_wrist": KeypointData(x=100, y=200, confidence=conf, name="right_wrist", visible=True),
        "left_hip": KeypointData(x=250, y=95, confidence=conf, name="left_hip", visible=True),
        "right_hip": KeypointData(x=250, y=105, confidence=conf, name="right_hip", visible=True),
        "left_knee": KeypointData(x=380, y=95, confidence=conf, name="left_knee", visible=True),
        "right_knee": KeypointData(x=380, y=105, confidence=conf, name="right_knee", visible=True),
        "left_ankle": KeypointData(x=500, y=95, confidence=conf, name="left_ankle", visible=True),
        "right_ankle": KeypointData(x=500, y=105, confidence=conf, name="right_ankle", visible=True),
    }


def test_compute_midpoint_calculation():
    """Test midpoint calculation between left and right landmarks."""
    left = KeypointData(x=100.0, y=90.0, confidence=0.8, name="left_shoulder", visible=True)
    right = KeypointData(x=120.0, y=110.0, confidence=0.9, name="right_shoulder", visible=True)

    mid = compute_midpoint(left, right, "mid_shoulder", threshold=0.25)
    assert mid is not None
    assert mid.name == "mid_shoulder"
    assert mid.x == 110.0  # (100 + 120) / 2
    assert mid.y == 100.0  # (90 + 110) / 2
    assert mid.confidence == 0.8  # min(0.8, 0.9)
    assert mid.visible is True


def test_compute_midpoint_low_confidence():
    """
    Test occlusion-aware midpoint: when one landmark is below the confidence
    threshold, compute_midpoint snaps to the valid (high-confidence) side
    rather than returning None.  This prevents hallucinated occluded keypoints
    from corrupting the midpoint position.
    """
    left = KeypointData(x=100.0, y=90.0, confidence=0.15, name="left_shoulder", visible=True)
    right = KeypointData(x=120.0, y=110.0, confidence=0.9, name="right_shoulder", visible=True)

    mid = compute_midpoint(left, right, "mid_shoulder", threshold=0.25)

    # The low-confidence left side is invalid; mid should snap to the right keypoint
    assert mid is not None
    assert mid.name == "mid_shoulder"
    assert mid.x == right.x
    assert mid.y == right.y
    assert mid.confidence == pytest.approx(right.confidence, abs=0.001)
    assert mid.visible is True



def test_side_view_processor_derives_midpoints_and_preserves_originals():
    """
    Test side_view processing derives all midpoints and preserves original YOLO landmarks.
    """
    kps = create_coco_keypoints(conf=0.9)
    res = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)

    assert res.is_valid_side_view is True
    # Verify derived midpoints exist
    assert "mid_shoulder" in res.midpoints
    assert "mid_hip" in res.midpoints
    assert "mid_knee" in res.midpoints
    assert "mid_ankle" in res.midpoints

    assert res.midpoints["mid_shoulder"].x == 100.0
    assert res.midpoints["mid_shoulder"].y == 100.0  # (95 + 105) / 2
    assert res.midpoints["mid_hip"].x == 250.0
    assert res.midpoints["mid_hip"].y == 100.0

    # Verify original landmarks are preserved
    for orig in ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_knee", "right_knee"]:
        assert orig in res.keypoints


def test_side_view_branching_evaluator():
    """
    Test explicit branch:
    - side_view=True calculates midpoints
    - side_view=False does NOT calculate midpoints
    """
    kps = create_coco_keypoints(conf=0.9)
    pose_kp = PoseKeypoints(keypoints=kps, bounding_box=[0, 0, 500, 200], detection_confidence=0.9)

    # 1. side_view=True
    profile_side = BiomechanicsEvaluator.evaluate(pose_kp, side_view=True)
    assert profile_side is not None
    assert profile_side.side_view is True
    assert profile_side.midpoints is not None
    assert "mid_shoulder" in profile_side.midpoints
    assert "mid_hip" in profile_side.midpoints
    assert profile_side.side_view_features is not None

    # 2. side_view=False
    profile_std = BiomechanicsEvaluator.evaluate(pose_kp, side_view=False)
    assert profile_std is not None
    assert profile_std.side_view is False
    assert profile_std.midpoints is None
    assert profile_std.side_view_features is None


def test_side_view_unreliable_features_when_landmarks_missing():
    """
    Test that when landmarks are missing, features are marked unreliable
    rather than inventing fake numbers.
    """
    # Keypoints missing knees and ankles
    kps = {
        "left_shoulder": KeypointData(x=100, y=95, confidence=0.9, name="left_shoulder", visible=True),
        "right_shoulder": KeypointData(x=100, y=105, confidence=0.9, name="right_shoulder", visible=True),
        "left_wrist": KeypointData(x=100, y=200, confidence=0.9, name="left_wrist", visible=True),
        "right_wrist": KeypointData(x=100, y=200, confidence=0.9, name="right_wrist", visible=True),
        "left_hip": KeypointData(x=250, y=95, confidence=0.9, name="left_hip", visible=True),
        "right_hip": KeypointData(x=250, y=105, confidence=0.9, name="right_hip", visible=True),
    }
    processed = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)
    features = SideViewKeypointProcessor.extract_features(processed, threshold=0.25)

    assert "knee_extension" in features.unreliable_features
    assert features.knee_extension_deg is None
