import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.biomechanics.angles import (
    calculate_joint_angle_from_keypoints,
    compute_bilateral_symmetry_diff,
    compute_midpoint_coords,
)
from app.biomechanics.camera_quality import CameraQualityEvaluator
from app.biomechanics.confidence import (
    check_body_visibility,
    compute_average_keypoint_confidence,
    filter_confident_keypoints,
    is_valid_keypoint,
    normalize_keypoints_visibility,
)
from app.biomechanics.smoothing import OneEuroKeypointSmoother, TemporalKeypointSmoother
from app.exercises.registry import exercise_registry
from app.main import app
from app.pose.yolo_pose import COCO_KEYPOINTS, YOLOPoseEstimator, _resolve_device
from app.schemas.analysis import (
    CameraQualityAssessment,
    ExerciseType,
    IssueSeverity,
    KeypointData,
    MovementMetricItem,
    PoseKeypoints,
    RepBreakdownItem,
    StructuredIssueItem,
    VideoScoringResponse,
)
from app.services.video_pipeline import VideoScoringPipeline, validate_video_bytes

client = TestClient(app)

SAMPLE_VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmark_data", "videos", "planche_real_video.mp4"
)


def _kp(name: str, x: float, y: float, conf: float = 0.9, visible: bool = True) -> KeypointData:
    return KeypointData(x=x, y=y, confidence=conf, name=name, visible=visible)


def _create_dummy_mp4_bytes(
    num_frames: int = 30,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_path = f.name
    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tmp_path, fourcc, float(fps), (width, height))
        for i in range(num_frames):
            frame = np.full((height, width, 3), (i * 5) % 255, dtype=np.uint8)
            out.write(frame)
        out.release()
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _make_pose_keypoints_dict(
    view: str = "FRONT",
    conf: float = 0.9,
    include_lower_body: bool = True,
    offset_x: float = 0.0,
) -> dict[str, KeypointData]:
    if view == "FRONT":
        kps = {
            "nose": _kp("nose", 250 + offset_x, 60, conf),
            "left_eye": _kp("left_eye", 240 + offset_x, 55, conf),
            "right_eye": _kp("right_eye", 260 + offset_x, 55, conf),
            "left_ear": _kp("left_ear", 232 + offset_x, 58, conf),
            "right_ear": _kp("right_ear", 268 + offset_x, 58, conf),
            "left_shoulder": _kp("left_shoulder", 200 + offset_x, 100, conf),
            "right_shoulder": _kp("right_shoulder", 300 + offset_x, 100, conf),
            "left_elbow": _kp("left_elbow", 180 + offset_x, 160, conf),
            "right_elbow": _kp("right_elbow", 320 + offset_x, 160, conf),
            "left_wrist": _kp("left_wrist", 170 + offset_x, 220, conf),
            "right_wrist": _kp("right_wrist", 330 + offset_x, 220, conf),
            "left_hip": _kp("left_hip", 215 + offset_x, 240, conf),
            "right_hip": _kp("right_hip", 285 + offset_x, 240, conf),
        }
        if include_lower_body:
            kps.update({
                "left_knee": _kp("left_knee", 215 + offset_x, 340, conf),
                "right_knee": _kp("right_knee", 285 + offset_x, 340, conf),
                "left_ankle": _kp("left_ankle", 215 + offset_x, 440, conf),
                "right_ankle": _kp("right_ankle", 285 + offset_x, 440, conf),
            })
        return kps

    if view == "OBLIQUE":
        kps = {
            "nose": _kp("nose", 250 + offset_x, 60, conf),
            "left_shoulder": _kp("left_shoulder", 225 + offset_x, 100, conf),
            "right_shoulder": _kp("right_shoulder", 275 + offset_x, 102, conf),
            "left_elbow": _kp("left_elbow", 215 + offset_x, 160, conf),
            "right_elbow": _kp("right_elbow", 280 + offset_x, 162, conf),
            "left_wrist": _kp("left_wrist", 210 + offset_x, 220, conf),
            "right_wrist": _kp("right_wrist", 285 + offset_x, 222, conf),
            "left_hip": _kp("left_hip", 230 + offset_x, 240, conf),
            "right_hip": _kp("right_hip", 270 + offset_x, 242, conf),
        }
        if include_lower_body:
            kps.update({
                "left_knee": _kp("left_knee", 230 + offset_x, 340, conf),
                "right_knee": _kp("right_knee", 270 + offset_x, 342, conf),
                "left_ankle": _kp("left_ankle", 230 + offset_x, 440, conf),
                "right_ankle": _kp("right_ankle", 270 + offset_x, 442, conf),
            })
        return kps

    # SIDE view: near-side visible, far-side occluded or overlapping
    kps = {
        "nose": _kp("nose", 255 + offset_x, 60, conf),
        "left_shoulder": _kp("left_shoulder", 250 + offset_x, 100, conf),
        "right_shoulder": _kp("right_shoulder", 255 + offset_x, 102, 0.1, visible=False),
        "left_elbow": _kp("left_elbow", 250 + offset_x, 160, conf),
        "right_elbow": _kp("right_elbow", 255 + offset_x, 162, 0.1, visible=False),
        "left_wrist": _kp("left_wrist", 250 + offset_x, 220, conf),
        "right_wrist": _kp("right_wrist", 255 + offset_x, 222, 0.1, visible=False),
        "left_hip": _kp("left_hip", 250 + offset_x, 240, conf),
        "right_hip": _kp("right_hip", 255 + offset_x, 242, 0.1, visible=False),
    }
    if include_lower_body:
        kps.update({
            "left_knee": _kp("left_knee", 250 + offset_x, 340, conf),
            "right_knee": _kp("right_knee", 255 + offset_x, 342, 0.1, visible=False),
            "left_ankle": _kp("left_ankle", 250 + offset_x, 440, conf),
            "right_ankle": _kp("right_ankle", 255 + offset_x, 442, 0.1, visible=False),
        })
    return kps


# ---------------------------------------------------------------------------
# 1. Video Validation Tests
# ---------------------------------------------------------------------------

def test_validate_video_bytes_rejects_invalid_inputs():
    ok, msg = validate_video_bytes(b"")
    assert ok is False and "empty" in msg.lower()

    ok, msg = validate_video_bytes(b"short")
    assert ok is False

    # PNG disguised as video
    png_fake = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    ok, msg = validate_video_bytes(png_fake)
    assert ok is False

    # Random corrupt text payload > 32 bytes
    ok, msg = validate_video_bytes(b"corrupted non video binary data " * 4)
    assert ok is False


def test_pipeline_rejects_empty_corrupt_and_single_frame_videos():
    mock_est = MagicMock()
    pipeline = VideoScoringPipeline(pose_estimator=mock_est)

    # Empty bytes
    res_empty = pipeline.process_video(b"", exercise_name="SQUAT")
    assert res_empty.status == "INVALID_INPUT"
    assert res_empty.score is None
    assert res_empty.schemaVersion == "1.0"

    # Corrupt bytes
    res_corrupt = pipeline.process_video(b"garbage video data " * 10, exercise_name="PUSH_UP")
    assert res_corrupt.status == "INVALID_INPUT"
    assert res_corrupt.score is None

    # Single-frame video (must have >= 2 frames)
    one_frame_bytes = _create_dummy_mp4_bytes(num_frames=1, fps=30.0)
    res_short = pipeline.process_video(one_frame_bytes, exercise_name="PULL_UP")
    assert res_short.status == "INVALID_INPUT"
    assert res_short.score is None


def test_pipeline_extracts_valid_video_metadata():
    video_bytes = _create_dummy_mp4_bytes(num_frames=30, fps=30.0, width=640, height=480)
    ok, _ = validate_video_bytes(video_bytes)
    assert ok is True

    mock_est = MagicMock()
    mock_est.estimate_pose.return_value = (False, None)
    pipeline = VideoScoringPipeline(pose_estimator=mock_est)

    res = pipeline.process_video(video_bytes, exercise_name="SQUAT", sample_fps=10)
    assert res.status == "NO_PERSON_DETECTED"
    assert res.score == 0.0
    assert res.exercise == "SQUAT"
    assert res.video.width == 640
    assert res.video.height == 480
    assert res.video.fps == pytest.approx(30.0, abs=0.5)
    assert res.video.durationMs == pytest.approx(1000, abs=50)
    assert res.video.annotatedAvailable is False


# ---------------------------------------------------------------------------
# 2. Frame Sampling Rate & Timestamp Calculation
# ---------------------------------------------------------------------------

def test_frame_sampling_rate_and_timestamps():
    # 30 frames @ 30fps -> 1.0s; sample_fps=10 -> step=3 -> 10 sampled frames (0, 3, ..., 27)
    video_bytes = _create_dummy_mp4_bytes(num_frames=30, fps=30.0, width=320, height=240)
    kps = _make_pose_keypoints_dict(view="FRONT", conf=0.9)

    mock_est = MagicMock()
    mock_est.estimate_pose.return_value = (
        True,
        PoseKeypoints(keypoints=kps, bounding_box=[100, 40, 350, 460], detection_confidence=0.92),
    )

    pipeline = VideoScoringPipeline(pose_estimator=mock_est)
    captured_samples = []

    fake_analyzer = MagicMock()
    fake_analyzer.name = "SQUAT"
    fake_analyzer.camera_view_recommendation = "Side or 45-degree view."

    def _capture_seq(frame_samples, fps, side_view=False):
        captured_samples.extend(frame_samples)
        return {
            "status": "COMPLETED",
            "score": 88.0,
            "formAccuracy": 88.0,
            "stability": 90.0,
            "alignment": 85.0,
            "repCount": 1,
            "validReps": 1,
            "incompleteReps": 0,
            "reps": [],
            "movementMetrics": {},
            "issues": [],
            "recommendations": [],
            "feedback": ["Good rep!"],
            "metrics": {},
        }

    fake_analyzer.analyze_video_sequence = _capture_seq

    with patch.object(exercise_registry, "get_analyzer", return_value=fake_analyzer):
        res = pipeline.process_video(video_bytes, exercise_name="SQUAT", sample_fps=10)

    assert res.status == "COMPLETED"
    assert len(captured_samples) == 10
    assert [s["frame_index"] for s in captured_samples] == [0, 3, 6, 9, 12, 15, 18, 21, 24, 27]
    assert [s["timestamp_ms"] for s in captured_samples] == [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
    assert res.metrics["samplingFps"] == 10.0
    assert res.metrics["analyzedFrames"] == 10
    assert res.metrics["totalSampledFrames"] == 10
    assert res.metrics["schemaVersion"] == "1.0"


# ---------------------------------------------------------------------------
# 3. Landmark Normalization & Confidence Filtering
# ---------------------------------------------------------------------------

def test_landmark_normalization_clamps_to_unit_bounds():
    video_bytes = _create_dummy_mp4_bytes(num_frames=6, fps=10.0, width=200, height=100)
    kps = _make_pose_keypoints_dict(view="FRONT", conf=0.9)
    # Deliberately inject out-of-bounds coordinates
    kps["nose"] = _kp("nose", -15.0, 125.0, 0.95)
    kps["left_ankle"] = _kp("left_ankle", 250.0, -5.0, 0.85)

    mock_est = MagicMock()
    mock_est.estimate_pose.return_value = (
        True,
        PoseKeypoints(keypoints=kps, bounding_box=[0, 0, 200, 100], detection_confidence=0.9),
    )

    pipeline = VideoScoringPipeline(pose_estimator=mock_est)
    fake_analyzer = MagicMock()
    fake_analyzer.name = "PUSH_UP"
    fake_analyzer.camera_view_recommendation = "Side view."
    fake_analyzer.analyze_video_sequence.return_value = {
        "status": "COMPLETED",
        "score": 82.0,
        "formAccuracy": 82.0,
        "stability": 84.0,
        "alignment": 80.0,
        "repCount": 1,
        "validReps": 1,
        "incompleteReps": 0,
        "reps": [],
        "movementMetrics": {},
        "issues": [],
        "recommendations": [],
        "feedback": ["Clean push-ups."],
        "metrics": {},
    }

    with patch.object(exercise_registry, "get_analyzer", return_value=fake_analyzer):
        res = pipeline.process_video(video_bytes, exercise_name="PUSH_UP", sample_fps=10)

    assert len(res.keypoints.frames) == 6
    first_frame_kps = {kp.name: kp for kp in res.keypoints.frames[0].keypoints}
    assert len(first_frame_kps) == 17
    for name in COCO_KEYPOINTS:
        assert name in first_frame_kps
        assert 0.0 <= first_frame_kps[name].x <= 1.0
        assert 0.0 <= first_frame_kps[name].y <= 1.0
        assert 0.0 <= first_frame_kps[name].confidence <= 1.0

    assert first_frame_kps["nose"].x == 0.0
    assert first_frame_kps["nose"].y == 1.0
    assert first_frame_kps["left_ankle"].x == 1.0
    assert first_frame_kps["left_ankle"].y == 0.0


def test_confidence_filtering_and_visibility_helpers():
    kps = {
        "left_shoulder": _kp("left_shoulder", 100.0, 100.0, 0.90, visible=True),
        "right_shoulder": _kp("right_shoulder", 200.0, 100.0, 0.15, visible=True),
        "left_hip": _kp("left_hip", float("nan"), 200.0, 0.90, visible=True),
        "right_hip": _kp("right_hip", 200.0, 200.0, 0.85, visible=True),
    }

    assert is_valid_keypoint(kps["left_shoulder"], threshold=0.25) is True
    assert is_valid_keypoint(kps["right_shoulder"], threshold=0.25) is False
    assert is_valid_keypoint(kps["left_hip"], threshold=0.25) is False

    filtered = filter_confident_keypoints(kps, threshold=0.25)
    assert set(filtered.keys()) == {"left_shoulder", "right_hip"}

    normalized = normalize_keypoints_visibility(kps, threshold=0.25)
    assert normalized["right_shoulder"].visible is False
    assert normalized["left_hip"].visible is False
    assert normalized["left_shoulder"].visible is True

    avg_conf = compute_average_keypoint_confidence(kps, ["left_shoulder", "right_shoulder", "right_hip"])
    assert avg_conf == pytest.approx((0.90 + 0.15 + 0.85) / 3.0, abs=1e-3)

    vis = check_body_visibility(_make_pose_keypoints_dict(view="FRONT", include_lower_body=False), threshold=0.25)
    assert vis["full_body_visible"] is False
    assert "knees" in vis["missing_groups"]
    assert "ankles" in vis["missing_groups"]


# ---------------------------------------------------------------------------
# 4. Temporal Smoothing Behavior
# ---------------------------------------------------------------------------

def test_temporal_smoothing_handles_jitter_gaps_and_nan():
    smoother = TemporalKeypointSmoother(alpha=0.5, max_jump_distance_px=50.0)

    res1 = smoother.smooth({"left_knee": _kp("left_knee", 100.0, 200.0, 0.9)})
    assert res1["left_knee"].x == 100.0

    # Small jitter: 0.5*108 + 0.5*100 = 104.0
    res2 = smoother.smooth({"left_knee": _kp("left_knee", 108.0, 204.0, 0.8)})
    assert res2["left_knee"].x == pytest.approx(104.0, abs=0.01)
    assert res2["left_knee"].y == pytest.approx(202.0, abs=0.01)

    # Empty dict clears history cleanly
    assert smoother.smooth({}) == {}

    # After reset, new value should not blend with previous 104.0
    res3 = smoother.smooth({"left_knee": _kp("left_knee", 150.0, 250.0, 0.9)})
    assert res3["left_knee"].x == 150.0

    # Non-finite coordinates safely marked invisible
    res4 = smoother.smooth({"left_knee": _kp("left_knee", float("nan"), 250.0, 0.9)})
    assert res4["left_knee"].visible is False


def test_one_euro_smoother_suppresses_stationary_jitter_and_resets_on_jump():
    smoother = OneEuroKeypointSmoother(freq=30.0, mincutoff=0.5, beta=0.01, max_jump_distance_px=40.0)

    smoother.smooth({"left_elbow": _kp("left_elbow", 100.0, 100.0, 0.9)}, timestamp_sec=0.0)
    res_jitter = smoother.smooth({"left_elbow": _kp("left_elbow", 106.0, 100.0, 0.9)}, timestamp_sec=0.033)
    # Filtered position should dampen jitter significantly below raw 106.0
    assert 100.0 <= res_jitter["left_elbow"].x < 105.0

    # Large jump resets state and takes current confidence directly
    res_jump = smoother.smooth({"left_elbow": _kp("left_elbow", 250.0, 100.0, 0.55)}, timestamp_sec=0.066)
    assert res_jump["left_elbow"].x == pytest.approx(250.0, abs=0.1)
    assert res_jump["left_elbow"].confidence == pytest.approx(0.55, abs=0.01)


# ---------------------------------------------------------------------------
# 5. Joint Angle Calculations (angles.py)
# ---------------------------------------------------------------------------

def test_joint_angle_helpers_no_fabricated_values():
    kps = {
        "left_hip": _kp("left_hip", 100.0, 100.0, 0.9),
        "left_knee": _kp("left_knee", 100.0, 200.0, 0.9),
        "left_ankle": _kp("left_ankle", 200.0, 200.0, 0.9),
        "right_hip": _kp("right_hip", 150.0, 100.0, 0.15, visible=True),
        "right_knee": _kp("right_knee", 150.0, 200.0, 0.9),
        "right_ankle": _kp("right_ankle", 150.0, 300.0, 0.9),
    }

    left_knee = calculate_joint_angle_from_keypoints(kps, "left_hip", "left_knee", "left_ankle", min_confidence=0.25)
    assert left_knee == pytest.approx(90.0, abs=0.1)

    # Low-confidence hip must yield None rather than fabricating 180.0
    right_knee = calculate_joint_angle_from_keypoints(kps, "right_hip", "right_knee", "right_ankle", min_confidence=0.25)
    assert right_knee is None

    assert compute_bilateral_symmetry_diff(left_knee, right_knee) is None
    assert compute_bilateral_symmetry_diff(170.0, 155.5) == 14.5
    assert compute_midpoint_coords((10.0, 20.0), (30.0, 40.0)) == (20.0, 30.0)
    assert compute_midpoint_coords((10.0, 20.0), None) == (10.0, 20.0)
    assert compute_midpoint_coords(None, None) is None


# ---------------------------------------------------------------------------
# 6. Camera Quality Assessment
# ---------------------------------------------------------------------------

def test_camera_quality_good_side_front_oblique():
    side_kps = _make_pose_keypoints_dict(view="SIDE", conf=0.9, include_lower_body=True)
    side_samples = [
        {"pose_detected": True, "raw_keypoints": side_kps, "keypoints": side_kps}
        for _ in range(5)
    ]
    cq_side = CameraQualityEvaluator.evaluate(side_samples, exercise_name="SQUAT")
    assert cq_side.overallQuality == "GOOD"
    assert cq_side.viewAngle == "SIDE"
    assert cq_side.fullBodyVisibleRatio == 1.0
    assert cq_side.averageConfidence >= 0.85
    assert cq_side.missingLandmarks == []
    assert cq_side.score >= 85.0

    front_kps = _make_pose_keypoints_dict(view="FRONT", conf=0.9, include_lower_body=True)
    front_samples = [
        {"pose_detected": True, "raw_keypoints": front_kps, "keypoints": front_kps}
        for _ in range(5)
    ]
    cq_front = CameraQualityEvaluator.evaluate(front_samples, exercise_name="PUSH_UP")
    assert cq_front.overallQuality == "GOOD"
    assert cq_front.viewAngle == "FRONT"
    assert any("side-on" in w.lower() for w in cq_front.warnings)

    oblique_kps = _make_pose_keypoints_dict(view="OBLIQUE", conf=0.85, include_lower_body=True)
    oblique_samples = [
        {"pose_detected": True, "raw_keypoints": oblique_kps, "keypoints": oblique_kps}
        for _ in range(5)
    ]
    cq_oblique = CameraQualityEvaluator.evaluate(oblique_samples, exercise_name="SQUAT")
    assert cq_oblique.overallQuality == "GOOD"
    assert cq_oblique.viewAngle == "OBLIQUE"


def test_camera_quality_fair_and_poor_missing_landmarks():
    # FAIR: 4 of 5 frames full body with moderate confidence (~0.58)
    full_mod_kps = _make_pose_keypoints_dict(view="FRONT", conf=0.58, include_lower_body=True)
    partial_mod_kps = _make_pose_keypoints_dict(view="FRONT", conf=0.58, include_lower_body=False)
    fair_samples = [
        {"pose_detected": True, "raw_keypoints": full_mod_kps, "keypoints": full_mod_kps}
        for _ in range(4)
    ] + [{"pose_detected": True, "raw_keypoints": partial_mod_kps, "keypoints": partial_mod_kps}]
    cq_fair = CameraQualityEvaluator.evaluate(fair_samples)
    assert cq_fair.overallQuality == "FAIR"
    assert cq_fair.fullBodyVisibleRatio == pytest.approx(0.8, abs=0.01)

    # POOR: Lower body cropped out in all frames
    no_legs_kps = _make_pose_keypoints_dict(view="FRONT", conf=0.9, include_lower_body=False)
    poor_samples = [
        {"pose_detected": True, "raw_keypoints": no_legs_kps, "keypoints": no_legs_kps}
        for _ in range(5)
    ]
    cq_poor = CameraQualityEvaluator.evaluate(poor_samples)
    assert cq_poor.overallQuality == "POOR"
    assert cq_poor.fullBodyVisibleRatio == 0.0
    assert "left_knee" in cq_poor.missingLandmarks
    assert "left_ankle" in cq_poor.missingLandmarks
    assert len(cq_poor.warnings) >= 1

    # POOR: Empty / no detections
    cq_none = CameraQualityEvaluator.evaluate([{"pose_detected": False, "keypoints": {}}])
    assert cq_none.overallQuality == "POOR"
    assert cq_none.viewAngle == "UNKNOWN"
    assert cq_none.score == 0.0


# ---------------------------------------------------------------------------
# 7. Shared Contract Schemas & Invariants
# ---------------------------------------------------------------------------

def test_schema_contract_unavailable_metric_never_fabricates_values():
    item = MovementMetricItem(
        name="knee_valgus",
        unit="ratio",
        value=1.2,
        min=1.0,
        max=1.3,
        avg=1.15,
        score=88.0,
        status="UNAVAILABLE",
        confidence=0.1,
    )
    assert item.status == "UNAVAILABLE"
    assert item.value is None
    assert item.min is None
    assert item.max is None
    assert item.avg is None
    assert item.score is None

    assert ExerciseType.SQUAT.value == "squat"
    assert ExerciseType.PUSH_UP.value == "push_up"
    assert ExerciseType.PULL_UP.value == "pull_up"

    issue = StructuredIssueItem(
        code="SHALLOW_DEPTH",
        metric="squat_depth",
        severity="medium",
        title="Shallow Squat Depth",
        message="Squat deeper to reach parallel.",
        occurrences=2,
        affectedReps=[1, 3],
    )
    assert issue.severity == IssueSeverity.MEDIUM


# ---------------------------------------------------------------------------
# 8. Pipeline Integration: Sequence Analyzer, No Fabricated Scores, CPU Real Video
# ---------------------------------------------------------------------------

def test_pipeline_passes_all_sampled_frames_including_lost_tracking_and_populates_contract():
    video_bytes = _create_dummy_mp4_bytes(num_frames=8, fps=8.0, width=320, height=240)
    kps = _make_pose_keypoints_dict(view="SIDE", conf=0.9, include_lower_body=True)
    pose_ok = PoseKeypoints(keypoints=kps, bounding_box=[100, 40, 300, 460], detection_confidence=0.92)

    # Frame 3 drops out (pose_detected=False) while others succeed
    side_effects = [
        (True, pose_ok),
        (True, pose_ok),
        (True, pose_ok),
        (False, None),
        (True, pose_ok),
        (True, pose_ok),
        (True, pose_ok),
        (True, pose_ok),
    ]
    mock_est = MagicMock()
    mock_est.estimate_pose.side_effect = side_effects

    received_samples = []
    fake_analyzer = MagicMock()
    fake_analyzer.name = "SQUAT"
    fake_analyzer.camera_view_recommendation = "Side-on view."

    def _seq(frame_samples, fps, side_view=False):
        received_samples.extend(frame_samples)
        return {
            "status": "COMPLETED",
            "score": 86.5,
            "formAccuracy": 87.0,
            "stability": 85.0,
            "alignment": 88.0,
            "repCount": 2,
            "validReps": 1,
            "incompleteReps": 1,
            "reps": [
                {
                    "repNumber": 1,
                    "startFrame": 0,
                    "endFrame": 3,
                    "startTimestampMs": 0,
                    "endTimestampMs": 375,
                    "durationMs": 375,
                    "isValid": True,
                    "status": "VALID",
                    "formScore": 91.0,
                    "minAngleDeg": 88.0,
                    "maxAngleDeg": 175.0,
                    "romDeg": 87.0,
                    "depthClassification": "GOOD",
                    "keyMetrics": {"knee_angle": 88.0},
                    "issues": [],
                    "feedback": ["Good depth."],
                },
                {
                    "repNumber": 2,
                    "startFrame": 4,
                    "endFrame": 7,
                    "startTimestampMs": 500,
                    "endTimestampMs": 875,
                    "durationMs": 375,
                    "isValid": False,
                    "status": "INCOMPLETE",
                    "formScore": 72.0,
                    "minAngleDeg": 118.0,
                    "maxAngleDeg": 174.0,
                    "romDeg": 56.0,
                    "depthClassification": "MODERATE",
                    "keyMetrics": {"knee_angle": 118.0},
                    "issues": ["MODERATE_DEPTH"],
                    "feedback": ["Squat deeper."],
                },
            ],
            "movementMetrics": {
                "squat_depth": {
                    "unit": "deg",
                    "value": 88.0,
                    "min": 88.0,
                    "max": 175.0,
                    "avg": 122.0,
                    "targetRange": "<= 100 deg",
                    "score": 89.0,
                    "status": "VALID",
                    "confidence": 0.9,
                },
                "knee_valgus": {
                    "unit": "ratio",
                    "value": 1.0,
                    "status": "UNAVAILABLE",
                    "confidence": 0.0,
                },
            },
            "issues": [
                {
                    "code": "MODERATE_DEPTH",
                    "metric": "squat_depth",
                    "severity": "MEDIUM",
                    "title": "Depth Slightly High",
                    "message": "Rep 2 did not reach parallel.",
                    "recommendation": "Lower hips until thighs are parallel to floor.",
                    "measuredValue": 118.0,
                    "targetRange": "<= 100 deg",
                    "occurrences": 1,
                    "affectedReps": [2],
                }
            ],
            "recommendations": ["Lower hips until thighs are parallel to floor."],
            "feedback": ["1 of 2 reps reached full depth."],
            "metrics": {"avgKneeAngle": 122.0},
        }

    fake_analyzer.analyze_video_sequence = _seq

    pipeline = VideoScoringPipeline(pose_estimator=mock_est)
    with patch.object(exercise_registry, "get_analyzer", return_value=fake_analyzer):
        res = pipeline.process_video(video_bytes, exercise_name="SQUAT", sample_fps=8, side_view=True)

    assert len(received_samples) == 8
    assert received_samples[3]["pose_detected"] is False
    assert received_samples[3]["keypoints"] == {}
    assert "raw_keypoints" in received_samples[0]
    assert "keypoints" in received_samples[0]

    assert isinstance(res, VideoScoringResponse)
    assert res.schemaVersion == "1.0"
    assert res.status == "COMPLETED"
    assert res.repCount == 2
    assert res.validReps == 1
    assert res.incompleteReps == 1
    assert len(res.reps) == 2
    assert res.movementMetrics["squat_depth"].status == "VALID"
    assert res.movementMetrics["knee_valgus"].status == "UNAVAILABLE"
    assert res.movementMetrics["knee_valgus"].value is None
    assert res.movementMetrics["knee_valgus"].score is None
    assert len(res.issues) == 1
    assert res.issues[0].severity == IssueSeverity.MEDIUM
    assert isinstance(res.cameraQuality, CameraQualityAssessment)
    assert res.cameraQuality.viewAngle == "SIDE"
    assert res.metrics["repCount"] == 2
    assert res.metrics["schemaVersion"] == "1.0"


def test_pipeline_no_fabricated_scores_on_insufficient_pose_quality():
    """Verify legacy path returns INSUFFICIENT_POSE_QUALITY with None scores instead of 50.0 / 80.0."""
    video_bytes = _create_dummy_mp4_bytes(num_frames=6, fps=10.0, width=320, height=240)

    # Person detected, but only face landmarks are confident; shoulders/hips/wrists are below threshold
    face_only_kps = {
        name: _kp(name, 100.0, 100.0, conf=0.05 if name not in ("nose", "left_eye") else 0.9, visible=(name in ("nose", "left_eye")))
        for name in COCO_KEYPOINTS
    }
    pose_low = PoseKeypoints(keypoints=face_only_kps, bounding_box=[80, 60, 140, 140], detection_confidence=0.80)

    mock_est = MagicMock()
    mock_est.estimate_pose.return_value = (True, pose_low)

    pipeline = VideoScoringPipeline(pose_estimator=mock_est)
    res = pipeline.process_video(video_bytes, exercise_name="FULL_PLANCHE", sample_fps=10, side_view=True)

    assert res.status == "INSUFFICIENT_POSE_QUALITY"
    assert res.score is None
    assert res.formAccuracy is None
    assert res.stability is None
    assert res.alignment is None
    assert res.video.annotatedAvailable is False
    assert res.cameraQuality is not None
    assert res.cameraQuality.overallQuality == "POOR"


def test_cpu_fallback_and_cached_model_loading():
    with patch("torch.cuda.is_available", return_value=False):
        assert _resolve_device("cuda") == "cpu"
        assert _resolve_device("auto") == "cpu"
    assert _resolve_device("cpu") == "cpu"

    est1 = YOLOPoseEstimator(device="cpu")
    est2 = YOLOPoseEstimator(device="cpu")
    assert est1.device == "cpu"
    assert est1.model is not None
    assert est1.model is est2.model


def test_real_sample_video_cpu_integration():
    assert os.path.exists(SAMPLE_VIDEO_PATH), f"Missing sample video at {SAMPLE_VIDEO_PATH}"
    with open(SAMPLE_VIDEO_PATH, "rb") as f:
        video_bytes = f.read()

    pipeline = VideoScoringPipeline(pose_estimator=YOLOPoseEstimator(device="cpu"))
    res = pipeline.process_video(
        video_bytes=video_bytes,
        exercise_name="FULL_PLANCHE",
        sample_fps=5,
        side_view=True,
    )

    assert res.schemaVersion == "1.0"
    assert res.status == "COMPLETED"
    assert res.score is not None and res.score > 0.0
    assert res.cameraQuality is not None
    assert res.cameraQuality.overallQuality in ("GOOD", "FAIR", "POOR")
    assert res.metrics["schemaVersion"] == "1.0"
    assert res.video.annotatedAvailable is True


def test_api_exercises_and_video_score_validation():
    ex_resp = client.get("/api/v1/analysis/exercises")
    assert ex_resp.status_code == 200
    supported_ids = {item["id"] for item in ex_resp.json()["supported_exercises"]}
    assert {"planche", "squat", "push_up", "pull_up"}.issubset(supported_ids)

    # Empty upload -> HTTP 400
    empty_resp = client.post(
        "/api/v1/analysis/video/score?exercise=SQUAT",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert empty_resp.status_code == 400

    # Unsupported content type / extension -> HTTP 400
    bad_type_resp = client.post(
        "/api/v1/analysis/video/score?exercise=PUSH_UP",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert bad_type_resp.status_code == 400

    # Corrupt bytes with video/mp4 MIME -> structured INVALID_INPUT response
    corrupt_resp = client.post(
        "/api/v1/analysis/video/score?exercise=PULL_UP",
        files={"file": ("corrupt.mp4", b"corrupt binary payload " * 10, "video/mp4")},
    )
    assert corrupt_resp.status_code == 200
    assert corrupt_resp.json()["status"] == "INVALID_INPUT"
    assert corrupt_resp.json()["score"] is None


def test_cache_result_evicts_and_removes_old_annotated_file(tmp_path):
    from app.services import video_pipeline as vp_mod

    pipeline = VideoScoringPipeline(pose_estimator=MagicMock())
    dummy_resp = pipeline._create_failed_response(
        aid="dummy",
        exercise_name="SQUAT",
        exercise_id=None,
        status="NO_PERSON_DETECTED",
        message="none",
    )

    old_file = tmp_path / "annotated_old.mp4"
    old_file.write_bytes(b"old-video")
    new_file = tmp_path / "annotated_new.mp4"
    new_file.write_bytes(b"new-video")

    with patch.object(vp_mod, "MAX_CACHE_ENTRIES", 1):
        pipeline._cache_result("old-id", dummy_resp, str(old_file))
        assert old_file.exists()

        pipeline._cache_result("new-id", dummy_resp, str(new_file))
        assert not old_file.exists()
        assert new_file.exists()
        assert pipeline.get_cached_analysis("old-id") is None
        assert pipeline.get_annotated_video_path("old-id") is None
        assert pipeline.get_annotated_video_path("new-id") == str(new_file)
