"""
Tests verifying that the debug logging infrastructure emits the expected
structured log records at each pipeline stage without changing any API
behavior or algorithm output.
"""
import logging
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from app.schemas.analysis import KeypointData, PoseKeypoints, ProgressionStage
from app.biomechanics.side_view import (
    SideViewKeypointProcessor,
    resolve_side_view_chain,
)
from app.biomechanics.metrics import BiomechanicsEvaluator
from app.exercises.planche import PlancheAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _kp(name: str, x: float, y: float, conf: float = 0.85) -> KeypointData:
    return KeypointData(x=x, y=y, confidence=conf, name=name, visible=True)


def _low_kp(name: str, x: float, y: float) -> KeypointData:
    """Below default 0.25 threshold — should be flagged as rejected/missing."""
    return KeypointData(x=x, y=y, confidence=0.10, name=name, visible=True)


def make_side_view_kps():
    """Minimal valid side-view keypoint dict."""
    return {
        "left_shoulder":  _kp("left_shoulder",  100, 95),
        "right_shoulder": _kp("right_shoulder", 100, 105, conf=0.50),
        "left_elbow":     _kp("left_elbow",     100, 150),
        "right_elbow":    _kp("right_elbow",    100, 150, conf=0.40),
        "left_wrist":     _kp("left_wrist",     100, 200),
        "right_wrist":    _kp("right_wrist",    100, 200, conf=0.40),
        "left_hip":       _kp("left_hip",       250, 95),
        "right_hip":      _kp("right_hip",      250, 105, conf=0.50),
        "left_knee":      _kp("left_knee",      380, 95),
        "right_knee":     _kp("right_knee",     380, 105, conf=0.45),
        "left_ankle":     _kp("left_ankle",     500, 95),
        "right_ankle":    _kp("right_ankle",    500, 105, conf=0.40),
    }


# ---------------------------------------------------------------------------
# 1. resolve_side_view_chain emits DEBUG logs
# ---------------------------------------------------------------------------

def test_side_view_chain_emits_debug_logs(caplog):
    """resolve_side_view_chain must emit DEBUG logs for hip, knee, ankle stages."""
    kps = make_side_view_kps()
    with caplog.at_level(logging.DEBUG, logger="app.biomechanics.side_view"):
        resolve_side_view_chain(kps, threshold=0.25)

    messages = "\n".join(caplog.messages)
    assert "[SIDE_VIEW]" in messages, "Expected [SIDE_VIEW] prefix in logs"
    assert "hip_ref" in messages or "torso_len" in messages, \
        "Expected hip_ref or torso_len log in side-view chain resolution"
    # Knee selection should be logged
    assert "knee" in messages.lower()


def test_side_view_chain_warns_on_unresolved_knee(caplog):
    """WARNING must be emitted when both knee candidates are below threshold."""
    kps = make_side_view_kps()
    kps["left_knee"]  = _low_kp("left_knee",  380, 95)
    kps["right_knee"] = _low_kp("right_knee", 380, 105)

    with caplog.at_level(logging.WARNING, logger="app.biomechanics.side_view"):
        chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" not in chain
    warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("knee" in r.message.lower() for r in warning_msgs), \
        "Expected a WARNING log when knee is unresolved"


# ---------------------------------------------------------------------------
# 2. process_keypoints emits DEBUG logs for midpoints and missing
# ---------------------------------------------------------------------------

def test_process_keypoints_logs_midpoints_and_missing(caplog):
    """process_keypoints must log derived midpoints and missing landmark names."""
    kps = make_side_view_kps()
    # Remove both ankles so mid_ankle is missing
    kps["left_ankle"]  = _low_kp("left_ankle",  500, 95)
    kps["right_ankle"] = _low_kp("right_ankle", 500, 105)

    with caplog.at_level(logging.DEBUG, logger="app.biomechanics.side_view"):
        processed = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)

    messages = "\n".join(caplog.messages)
    assert "mid_ankle" in processed.missing_landmarks
    assert "[SIDE_VIEW]" in messages
    # Should log the missing list
    assert "missing" in messages.lower() or "mid_ankle" in messages


# ---------------------------------------------------------------------------
# 3. extract_features logs biomechanics angles
# ---------------------------------------------------------------------------

def test_extract_features_logs_angles(caplog):
    """extract_features must emit DEBUG logs for hip_extension and knee_extension."""
    kps = make_side_view_kps()
    processed = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)

    with caplog.at_level(logging.DEBUG, logger="app.biomechanics.side_view"):
        features = SideViewKeypointProcessor.extract_features(processed, threshold=0.25)

    messages = "\n".join(caplog.messages)
    assert "[BIOMECHANICS]" in messages, "Expected [BIOMECHANICS] prefix in extract_features logs"
    # hip_extension or knee_extension should appear
    assert "hip_extension" in messages or "elbow_extension" in messages


# ---------------------------------------------------------------------------
# 4. BiomechanicsEvaluator → side-view branch logs chain resolution
# ---------------------------------------------------------------------------

def test_evaluator_side_view_logs_chain(caplog):
    """Full evaluation in side_view=True must emit SIDE_VIEW logs."""
    kps = make_side_view_kps()
    pose_kp = PoseKeypoints(keypoints=kps, bounding_box=[0, 0, 500, 200], detection_confidence=0.9)

    with caplog.at_level(logging.DEBUG, logger="app.biomechanics.side_view"):
        profile = BiomechanicsEvaluator.evaluate(pose_kp, side_view=True)

    assert profile is not None
    messages = "\n".join(caplog.messages)
    assert "[SIDE_VIEW]" in messages


# ---------------------------------------------------------------------------
# 5. Existing API behavior unchanged — response structure preserved
# ---------------------------------------------------------------------------

def test_api_response_structure_unchanged():
    """
    Sanity-check that adding logging parameters does not break the existing
    analyze_image call path (no request_id provided → uses default None).
    """
    from app.services.analyzer import CalisthenicsAnalyzerService
    import cv2
    import numpy as np

    service = CalisthenicsAnalyzerService.__new__(CalisthenicsAnalyzerService)
    # Mock the pose estimator to avoid loading YOLO model
    mock_estimator = MagicMock()
    mock_estimator.estimate_pose.return_value = (False, None)
    service.pose_estimator = mock_estimator

    # Create a minimal valid JPEG image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    image_bytes = buf.tobytes()

    # Should NOT raise even without request_id; returns graceful failure response
    response = service.analyze_image(
        image_bytes=image_bytes,
        target_progression="FULL_PLANCHE",
        side_view=True
    )
    assert response.success is False  # No pose detected (mocked)
    assert response.pose_detected is False


def test_analyze_image_accepts_optional_request_id():
    """analyze_image must accept request_id kwarg without error."""
    from app.services.analyzer import CalisthenicsAnalyzerService
    import cv2
    import numpy as np

    service = CalisthenicsAnalyzerService.__new__(CalisthenicsAnalyzerService)
    mock_estimator = MagicMock()
    mock_estimator.estimate_pose.return_value = (False, None)
    service.pose_estimator = mock_estimator

    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)

    # Must not raise TypeError for unknown kwarg
    response = service.analyze_image(
        image_bytes=buf.tobytes(),
        target_progression="TUCK_PLANCHE",
        side_view=False,
        request_id="test-abc"
    )
    assert response is not None


# ---------------------------------------------------------------------------
# 6. draw_skeleton emits RENDER logs
# ---------------------------------------------------------------------------

def test_draw_skeleton_emits_render_logs(caplog):
    """draw_skeleton must emit [RENDER] DEBUG logs for the side-view chain."""
    from app.pose.yolo_pose import YOLOPoseEstimator
    from app.schemas.analysis import PlancheAnalysisResult, ExerciseType

    kps = make_side_view_kps()
    pose_kp = PoseKeypoints(keypoints=kps, bounding_box=[0, 0, 500, 200], detection_confidence=0.9)

    # Build a minimal PlancheAnalysisResult with side_view=True
    profile = BiomechanicsEvaluator.evaluate(pose_kp, side_view=True)
    assert profile is not None
    result = PlancheAnalyzer.analyze(profile, target_progression=ProgressionStage.FULL_PLANCHE)

    estimator = YOLOPoseEstimator.__new__(YOLOPoseEstimator)
    estimator.conf_threshold = 0.25
    estimator.keypoint_conf_threshold = 0.25
    estimator.model = None  # not needed for draw_skeleton

    dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)

    with caplog.at_level(logging.DEBUG, logger="app.pose.yolo_pose"):
        annotated = estimator.draw_skeleton(dummy_img, pose_kp, analysis=result)

    assert annotated is not None
    messages = "\n".join(caplog.messages)
    assert "[RENDER]" in messages, "Expected [RENDER] prefix in draw_skeleton logs"


# ---------------------------------------------------------------------------
# 7. YOLO estimate_pose emits YOLO logs (using mock to avoid loading model)
# ---------------------------------------------------------------------------

def test_estimate_pose_emits_yolo_logs_on_empty_input(caplog):
    """estimate_pose must emit a WARNING when given an empty image."""
    from app.pose.yolo_pose import YOLOPoseEstimator

    estimator = YOLOPoseEstimator.__new__(YOLOPoseEstimator)
    estimator.conf_threshold = 0.25
    estimator.keypoint_conf_threshold = 0.25
    estimator.model = MagicMock()

    with caplog.at_level(logging.WARNING, logger="app.pose.yolo_pose"):
        success, result = estimator.estimate_pose(np.array([]))  # empty array

    assert success is False
    assert result is None
    warning_msgs = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert warning_msgs, "Expected at least one WARNING for empty input"


# ---------------------------------------------------------------------------
# 8. No sensitive data in logs
# ---------------------------------------------------------------------------

def test_no_base64_or_large_array_in_logs(caplog):
    """Logs must not contain base64 payloads or raw numpy array dumps."""
    kps = make_side_view_kps()
    with caplog.at_level(logging.DEBUG, logger="app.biomechanics.side_view"):
        resolve_side_view_chain(kps, threshold=0.25)

    for record in caplog.records:
        msg = record.getMessage()
        # base64 strings are very long sequences of [A-Za-z0-9+/=]
        assert len(msg) < 1000, f"Log message suspiciously long ({len(msg)} chars): {msg[:200]}"
        assert "array([" not in msg, "numpy array dump found in log"


# ---------------------------------------------------------------------------
# 9. Full End-to-End Endpoint Pipeline Logs
# ---------------------------------------------------------------------------

def test_full_pipeline_draw_skeleton_logs(caplog):
    """
    Test POST /api/v1/analysis/image/draw-skeleton?progression=FULL_PLANCHE&side_view=true
    and verify the entire logging pipeline outputs structured logs from all stages.
    """
    import cv2
    from fastapi.testclient import TestClient
    from app.main import app
    from app.api.routes.analysis import analyzer_service

    kps = make_side_view_kps()
    pose_kp = PoseKeypoints(keypoints=kps, bounding_box=[0, 0, 500, 200], detection_confidence=0.92)

    # Mock estimate_pose on the singleton service
    with patch.object(analyzer_service.pose_estimator, "estimate_pose", return_value=(True, pose_kp)):
        dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
        _, buf = cv2.imencode(".jpg", dummy_img)

        client = TestClient(app)
        with caplog.at_level(logging.DEBUG):
            response = client.post(
                "/api/v1/analysis/image/draw-skeleton?progression=FULL_PLANCHE&side_view=true",
                files={"file": ("athlete_side.jpg", buf.tobytes(), "image/jpeg")}
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/jpeg"

        messages = "\n".join(caplog.messages)
        # 1. Request received
        assert "[DRAW_SKELETON]" in messages
        assert "athlete_side.jpg" in messages
        assert "progression=FULL_PLANCHE" in messages
        assert "side_view=True" in messages

        # 2. Analyzer layer
        assert "[ANALYZER]" in messages
        assert "image_decoded" in messages
        assert "pose_detected=True" in messages
        assert "form_score=" in messages

        # 3. Side-view processing
        assert "[SIDE_VIEW]" in messages
        assert "hip_ref" in messages
        assert "knee_selected" in messages
        assert "midpoints=" in messages

        # 4. Biomechanics
        assert "[BIOMECHANICS]" in messages
        assert "hip_extension=" in messages

        # 5. Render layer
        assert "[RENDER]" in messages
        assert "resolved_chain" in messages
        assert "bone" in messages

        # 6. Response completion
        assert "render_done" in messages
        assert "response_bytes=" in messages

        # 7. Joint traces across stages
        assert "[KEYPOINT_TRACE]" in messages
        assert "stage=side_view_candidate" in messages
        assert "stage=side_view_selection" in messages
        assert "stage=passed_to_biomechanics" in messages
        assert "stage=passed_to_rendering" in messages

        # 8. Segment vectors in biomechanics
        assert "segment=shoulder_to_hip" in messages
        assert "segment=hip_to_knee" in messages
        assert "segment=knee_to_ankle" in messages
        assert "length_px=" in messages

        # 9. Timing measurements across pipeline
        assert "[TIMING]" in messages
        assert "stage=image_decoding" in messages
        assert "stage=side_view_resolution" in messages
        assert "stage=biomechanics_analysis" in messages
        assert "stage=skeleton_rendering" in messages
        assert "stage=total_export" in messages


# ---------------------------------------------------------------------------
# 10. YOLO estimate_pose raw output and post-extraction trace logs
# ---------------------------------------------------------------------------

def test_yolo_estimate_pose_trace_logs(caplog):
    """
    estimate_pose must emit Ultralytics postprocessed outputs and post_extraction
    keypoint traces with original-image pixel coordinates and confidence.
    """
    from app.pose.yolo_pose import YOLOPoseEstimator

    estimator = YOLOPoseEstimator.__new__(YOLOPoseEstimator)
    estimator.conf_threshold = 0.25
    estimator.keypoint_conf_threshold = 0.25
    estimator.device = "cpu"

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
    mock_kp_data[:, 2] = 0.90

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

    with caplog.at_level(logging.DEBUG, logger="app.pose.yolo_pose"):
        success, pose_kp = estimator.estimate_pose(test_img)

    assert success is True
    assert pose_kp is not None

    messages = "\n".join(caplog.messages)
    # Check timings
    assert "[TIMING]" in messages
    assert "stage=yolo_inference" in messages
    assert "stage=keypoint_extraction" in messages

    # Check the output after Ultralytics has mapped keypoints to original-image pixels
    assert "[YOLO]" in messages
    assert "ultralytics_postprocessed_output" in messages
    assert "left_shoulder" in messages
    assert "right_knee" in messages
    assert "original_image=300x300" in messages

    # Check post-extraction traces
    assert "[KEYPOINT_TRACE]" in messages
    assert "stage=post_extraction" in messages


def test_model_preprocess_trace_logs_letterbox_geometry(caplog):
    from types import SimpleNamespace

    from app.pose.yolo_pose import YOLOPoseEstimator

    source = np.zeros((400, 200, 3), dtype=np.uint8)
    model_input = np.zeros((1, 3, 640, 640), dtype=np.float32)
    predictor = SimpleNamespace(
        preprocess=lambda images: model_input,
        batch=(None, [source]),
    )
    estimator = YOLOPoseEstimator.__new__(YOLOPoseEstimator)

    with caplog.at_level(logging.DEBUG, logger="app.pose.yolo_pose"):
        estimator._install_preprocess_trace(predictor)
        actual_input = predictor.preprocess([source])

    assert actual_input is model_input
    assert any(
        "stage=model_preprocess" in record.message
        and "original_shape=200x400" in record.message
        and "resized_shape=320x640" in record.message
        and "model_input_shape=640x640" in record.message
        and "scale=1.600000" in record.message
        and "padding_left=160" in record.message
        and "padding_top=0" in record.message
        for record in caplog.records
    )
    assert any(
        "stage=model_preprocess" in record.message
        and record.levelno == logging.INFO
        for record in caplog.records
    )


def test_coordinate_trace_logs_pre_and_post_scale_keypoints(caplog):
    from types import SimpleNamespace

    from app.core.logging_context import set_request_id
    from app.pose.yolo_pose import YOLOPoseEstimator

    source = np.zeros((100, 200, 3), dtype=np.uint8)
    pred = np.zeros((1, 6 + 17 * 3), dtype=np.float32)
    pred[0, 6 + 13 * 3:6 + 14 * 3] = (80.0, 120.0, 0.9)
    pred[0, 6 + 7 * 3:6 + 8 * 3] = (40.0, 60.0, 0.8)

    def construct_result(predictions, _img, _orig_img, _img_path):
        scaled = predictions[:, 6:].reshape((1, 17, 3)).copy()
        scaled[..., :2] *= 0.5
        return SimpleNamespace(keypoints=SimpleNamespace(data=scaled))

    predictor = SimpleNamespace(
        preprocess=lambda images: np.zeros((1, 3, 640, 640), dtype=np.float32),
        batch=(None, [source]),
        model=SimpleNamespace(kpt_shape=(17, 3)),
        construct_result=construct_result,
    )
    estimator = YOLOPoseEstimator.__new__(YOLOPoseEstimator)
    set_request_id("coordinate-trace-test")

    with caplog.at_level(logging.INFO, logger="app.pose.yolo_pose"):
        estimator._install_preprocess_trace(predictor)
        predictor.construct_result(pred, None, source, "test-image")

    assert any(
        "request_id=coordinate-trace-test" in record.message
        and "stage=model_output_before_scale" in record.message
        and "joint=left_knee" in record.message
        and "x=80.00 y=120.00 conf=0.9000" in record.message
        for record in caplog.records
    )
    assert any(
        "request_id=coordinate-trace-test" in record.message
        and "stage=ultralytics_scaled_to_original" in record.message
        and "joint=left_knee" in record.message
        and "x=40.00 y=60.00 conf=0.9000" in record.message
        for record in caplog.records
    )
