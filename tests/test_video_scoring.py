import io
import os
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.video_pipeline import video_scoring_pipeline, VideoScoringPipeline
from app.schemas.analysis import VideoScoringResponse

client = TestClient(app)

SAMPLE_VIDEO_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmark_data", "videos", "planche_real_video.mp4"
)


def _create_blank_video_bytes(duration_sec: int = 1, fps: int = 10, width: int = 320, height: int = 240) -> bytes:
    """Create a short dummy video file with black frames (no humans)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        tmp_name = f.name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_name, fourcc, float(fps), (width, height))
    blank = np.zeros((height, width, 3), dtype=np.uint8)
    for _ in range(duration_sec * fps):
        out.write(blank)
    out.release()

    with open(tmp_name, "rb") as f:
        data = f.read()
    if os.path.exists(tmp_name):
        os.remove(tmp_name)
    return data


def test_video_scoring_pipeline_real_video():
    """Test video scoring pipeline with actual planche exercise video."""
    assert os.path.exists(SAMPLE_VIDEO_PATH), f"Sample video not found at {SAMPLE_VIDEO_PATH}"
    with open(SAMPLE_VIDEO_PATH, "rb") as f:
        video_bytes = f.read()

    pipeline = VideoScoringPipeline()
    response = pipeline.process_video(
        video_bytes=video_bytes,
        exercise_name="FULL_PLANCHE",
        sample_fps=5,
        side_view=True
    )

    assert isinstance(response, VideoScoringResponse)
    assert response.status == "COMPLETED"
    assert response.score > 0
    assert response.formAccuracy > 0
    assert response.video.durationMs > 0
    assert response.video.annotatedAvailable is True
    assert len(response.keypoints.frames) > 0

    # Verify 17 COCO keypoints per frame
    first_frame = response.keypoints.frames[0]
    assert len(first_frame.keypoints) == 17
    for kp in first_frame.keypoints:
        assert 0.0 <= kp.x <= 1.0
        assert 0.0 <= kp.y <= 1.0
        assert 0.0 <= kp.confidence <= 1.0

    # Verify annotated MP4 was generated and is valid
    annotated_path = pipeline.get_annotated_video_path(response.analysisId)
    assert annotated_path is not None
    assert os.path.exists(annotated_path)
    assert os.path.getsize(annotated_path) > 0

    cap = cv2.VideoCapture(annotated_path)
    assert cap.isOpened()
    ann_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert ann_frames > 0
    cap.release()


def test_video_scoring_pipeline_no_person():
    """Test video scoring pipeline when video has no detectable humans."""
    blank_video = _create_blank_video_bytes(duration_sec=1, fps=10)
    pipeline = VideoScoringPipeline()
    response = pipeline.process_video(
        video_bytes=blank_video,
        exercise_name="PLANCHE",
        sample_fps=5
    )

    assert response.status == "NO_PERSON_DETECTED"
    assert response.score == 0.0
    assert len(response.keypoints.frames) == 0
    assert any("No human pose detected" in fb for fb in response.feedback)


def test_video_scoring_api_endpoints():
    """Test REST API endpoints for video scoring and keypoint retrieval."""
    assert os.path.exists(SAMPLE_VIDEO_PATH)
    with open(SAMPLE_VIDEO_PATH, "rb") as f:
        video_bytes = f.read()

    # 1. POST /api/v1/analysis/video/score
    files = {"file": ("test_planche.mp4", video_bytes, "video/mp4")}
    resp = client.post(
        "/api/v1/analysis/video/score?exercise=FULL_PLANCHE&sample_fps=5",
        files=files
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "COMPLETED"
    aid = data["analysisId"]
    assert aid is not None
    assert data["score"] > 0
    assert len(data["keypoints"]["frames"]) > 0

    # 2. GET /api/v1/analysis/video/{aid}/annotated
    ann_resp = client.get(f"/api/v1/analysis/video/{aid}/annotated")
    assert ann_resp.status_code == 200
    assert ann_resp.headers["content-type"] == "video/mp4"
    assert len(ann_resp.content) > 0

    # 3. GET /api/v1/analysis/video/{aid}/keypoints
    kp_resp = client.get(f"/api/v1/analysis/video/{aid}/keypoints")
    assert kp_resp.status_code == 200
    kp_data = kp_resp.json()
    assert kp_data["format"] == "COCO_17"
    assert len(kp_data["frames"]) > 0

    # 4. GET /api/v1/analysis/video/{aid}
    full_resp = client.get(f"/api/v1/analysis/video/{aid}")
    assert full_resp.status_code == 200
    assert full_resp.json()["analysisId"] == aid
