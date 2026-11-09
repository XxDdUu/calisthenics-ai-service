import cv2
import logging
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pose.yolo_pose import YOLOPoseEstimator, COCO_KEYPOINTS
from app.schemas.analysis import LiveAnalysisStatus


@pytest.fixture
def client():
    return TestClient(app)


def test_reproducible_full_body_standing(client):
    """
    Test 1: Full-body person standing in front of camera.
    Must detect pose, extract all 17 COCO keypoints, and report high confidence.
    """
    image_path = ".venv/lib/python3.13/site-packages/ultralytics/assets/bus.jpg"
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/analysis/live/frame",
        files={"file": ("standing.jpg", img_bytes, "image/jpeg")},
        data={"exercise": "SQUAT", "session_id": "test-full-body", "frame_sequence": "1"}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["poseDetected"] is True
    assert data["confidence"] > 0.70
    assert len(data["landmarks"]) == 17

    landmark_names = {lm["name"] for lm in data["landmarks"]}
    for kp in COCO_KEYPOINTS:
        assert kp in landmark_names


def test_reproducible_upper_body_person(client):
    """
    Test 2: Upper-body person (e.g. lower body outside frame).
    Must NOT reject the person. poseDetected must be True even if lower body landmarks are missing.
    """
    image_path = ".venv/lib/python3.13/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg"
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/analysis/live/frame",
        files={"file": ("upper_body.jpg", img_bytes, "image/jpeg")},
        data={"exercise": "HANDSTAND", "session_id": "test-upper-body", "frame_sequence": "1"}
    )
    assert response.status_code == 200
    data = response.json()

    # Person MUST be detected even if only upper body is visible
    assert data["poseDetected"] is True
    assert data["confidence"] > 0.60
    assert len(data["landmarks"]) == 17

    # Upper body landmarks (nose, eyes, shoulders) should be visible
    visible_landmarks = {lm["name"] for lm in data["landmarks"] if lm["visible"]}
    assert "nose" in visible_landmarks or "left_shoulder" in visible_landmarks or "right_shoulder" in visible_landmarks

    # Lower body landmarks should be noted as missing or pose status indicates incomplete
    assert data["status"] in (LiveAnalysisStatus.INCOMPLETE_POSE, LiveAnalysisStatus.LOW_CONFIDENCE, LiveAnalysisStatus.VALID)


def test_reproducible_calisthenics_position(client):
    """
    Test 3: Athlete performing a calisthenics position (Planche).
    Must detect pose, extract keypoints, and compute biomechanical score.
    """
    image_path = "benchmark_data/real_images/Full_planche.jpg"
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = client.post(
        "/api/v1/analysis/live/frame",
        files={"file": ("planche.jpg", img_bytes, "image/jpeg")},
        data={"exercise": "FULL_PLANCHE", "session_id": "test-planche", "frame_sequence": "1"}
    )
    assert response.status_code == 200
    data = response.json()

    assert data["poseDetected"] is True
    assert data["confidence"] > 0.70
    assert data["status"] == LiveAnalysisStatus.VALID
    assert data["score"] is not None
    assert 0.0 <= data["score"] <= 100.0


def test_diagnostic_logging_distinguishes_no_person_reasons(caplog):
    """
    Test 4: Verify diagnostic logging explicitly distinguishes between no-person reasons:
    1. Invalid image input
    3. No bounding boxes detected
    """
    estimator = YOLOPoseEstimator()

    # Case 1: Invalid image input (empty)
    with caplog.at_level(logging.WARNING, logger="app.pose.yolo_pose"):
        success, pose = estimator.estimate_pose(np.array([]))
    assert success is False
    assert any("reason='1. Invalid image input'" in r.getMessage() for r in caplog.records)

    # Case 3: Blank black image (valid array, but no humans -> no bounding boxes)
    caplog.clear()
    black_img = np.zeros((320, 320, 3), dtype=np.uint8)
    with caplog.at_level(logging.INFO, logger="app.pose.yolo_pose"):
        success, pose = estimator.estimate_pose(black_img)
    assert success is False
    assert any("reason='3. No bounding boxes detected'" in r.getMessage() for r in caplog.records)
