from unittest.mock import MagicMock
import numpy as np
import pytest

from app.pose.yolo_pose import YOLOPoseEstimator


def test_estimate_pose_invalid_empty_image():
    """Test estimate_pose returns (False, None) when given empty or None image."""
    estimator = YOLOPoseEstimator()
    success, pose_kp = estimator.estimate_pose(None)
    assert success is False
    assert pose_kp is None

    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    success, pose_kp = estimator.estimate_pose(empty_img)
    assert success is False
    assert pose_kp is None


def test_estimate_pose_image_with_no_person():
    """Test estimate_pose gracefully returns (False, None) for black/blank image."""
    estimator = YOLOPoseEstimator()
    blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
    success, pose_kp = estimator.estimate_pose(blank_img)
    assert success is False
    assert pose_kp is None


def test_estimate_pose_model_inference_failure(monkeypatch):
    """Test estimate_pose handles unexpected model exception gracefully."""
    estimator = YOLOPoseEstimator()

    def mock_failing_call(*args, **kwargs):
        raise RuntimeError("Simulated inference failure")

    monkeypatch.setattr(estimator, "model", mock_failing_call)

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    success, pose_kp = estimator.estimate_pose(img)
    assert success is False
    assert pose_kp is None


def test_estimate_pose_valid_human_mocked():
    """Test estimate_pose successfully parses keypoints when YOLO returns valid human pose structure."""
    estimator = YOLOPoseEstimator()

    mock_conf = MagicMock()
    mock_conf.__len__.return_value = 1
    mock_conf.cpu.return_value.numpy.return_value = np.array([0.95])

    mock_xyxy = MagicMock()
    mock_xyxy.__len__.return_value = 1
    mock_xyxy.cpu.return_value.numpy.return_value = np.array([[10.0, 20.0, 100.0, 200.0]])

    mock_boxes = MagicMock()
    mock_boxes.__len__.return_value = 1
    mock_boxes.conf = mock_conf
    mock_boxes.xyxy = mock_xyxy

    mock_kp_data = np.ones((17, 3), dtype=np.float32) * 50.0
    mock_kp_data[:, 2] = 0.9

    mock_keypoint_item = MagicMock()
    mock_keypoint_item.cpu.return_value.numpy.return_value = mock_kp_data

    mock_keypoints = MagicMock()
    mock_keypoints.data = [mock_keypoint_item]
    mock_keypoints.__len__.return_value = 1

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_result.keypoints = mock_keypoints

    mock_yolo = MagicMock()
    mock_yolo.return_value = [mock_result]
    mock_yolo.task = "pose"

    estimator.model = mock_yolo

    test_img = np.ones((300, 300, 3), dtype=np.uint8) * 128
    success, pose_kp = estimator.estimate_pose(test_img)

    assert success is True
    assert pose_kp is not None
    assert pose_kp.detection_confidence == pytest.approx(0.95)
    assert len(pose_kp.keypoints) == 17
    assert "nose" in pose_kp.keypoints
    assert pose_kp.keypoints["nose"].visible is True
