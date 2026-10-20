import pytest
from app.biomechanics.alignment import compute_alignment_metrics, compute_center_of_mass_2d
from app.schemas.analysis import KeypointData


def test_compute_center_of_mass():
    kps = {
        "nose": KeypointData(x=50, y=50, confidence=0.9, name="nose", visible=True),
        "left_shoulder": KeypointData(x=100, y=100, confidence=0.9, name="left_shoulder", visible=True),
        "right_shoulder": KeypointData(x=100, y=100, confidence=0.9, name="right_shoulder", visible=True),
        "left_wrist": KeypointData(x=100, y=200, confidence=0.9, name="left_wrist", visible=True),
        "right_wrist": KeypointData(x=100, y=200, confidence=0.9, name="right_wrist", visible=True),
        "left_hip": KeypointData(x=200, y=100, confidence=0.9, name="left_hip", visible=True),
        "right_hip": KeypointData(x=200, y=100, confidence=0.9, name="right_hip", visible=True),
    }

    com_pt, wrist_x = compute_center_of_mass_2d(kps)
    assert com_pt is not None
    assert wrist_x == 100.0


def test_compute_alignment_metrics():
    kps = {
        "left_shoulder": KeypointData(x=100, y=100, confidence=0.9, name="left_shoulder", visible=True),
        "left_ear": KeypointData(x=100, y=80, confidence=0.9, name="left_ear", visible=True),
    }

    metrics = compute_alignment_metrics(keypoints=kps, hip_angle_deg=180.0, body_tilt_deg=0.0)
    assert metrics.body_straightness_score == 100.0
    assert metrics.horizontal_level_score == 100.0
    assert metrics.scapular_protraction_score >= 80.0
