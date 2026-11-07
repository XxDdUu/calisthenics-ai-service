import base64
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.analysis import LiveAnalysisStatus


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sample_jpeg_bytes():
    """Create a minimal valid synthetic JPEG frame."""
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(img, (100, 100), 40, (255, 255, 255), -1)
    success, encoded = cv2.imencode(".jpg", img)
    assert success
    return encoded.tobytes()


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "model" in data


def test_list_exercises_includes_catalog(client):
    response = client.get("/api/v1/analysis/exercises")
    assert response.status_code == 200
    data = response.json()
    assert "supported_exercises" in data
    assert "all_catalog" in data
    ex_names = [e["exercise"] for e in data["all_catalog"]]
    assert "PLANCHE" in ex_names
    assert "FRONT_LEVER" in ex_names
    assert "HANDSTAND" in ex_names
    assert "PULL_UP" in ex_names
    assert "DIPS" in ex_names
    assert "MUSCLE_UP" in ex_names


def test_live_frame_multipart_empty_fails(client):
    response = client.post("/api/v1/analysis/live/frame", files={"file": ("test.jpg", b"", "image/jpeg")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_live_frame_multipart_invalid_content_type(client):
    response = client.post("/api/v1/analysis/live/frame", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
    assert "image" in response.json()["detail"].lower()


def test_live_frame_corrupt_payload(client):
    # Corrupt binary should not 500 crash, but gracefully return INVALID_INPUT status
    response = client.post(
        "/api/v1/analysis/live/frame",
        files={"file": ("corrupt.jpg", b"garbage bytes header not jpeg or png 1234567890", "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == LiveAnalysisStatus.INVALID_INPUT
    pose_det = data.get("pose_detected") if "pose_detected" in data else data.get("poseDetected")
    assert pose_det is False
    assert data["score"] is None


def test_live_frame_valid_synthetic_image(client, sample_jpeg_bytes):
    response = client.post(
        "/api/v1/analysis/live/frame",
        files={"file": ("frame.jpg", sample_jpeg_bytes, "image/jpeg")},
        data={"exercise": "FULL_PLANCHE", "draw_overlay": "true", "debug": "true"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["exercise"] == "FULL_PLANCHE"
    assert "timing" in data
    assert data["timing"]["total_ms"] >= 0.0


def test_live_frame_json_endpoint(client, sample_jpeg_bytes):
    b64_str = base64.b64encode(sample_jpeg_bytes).decode("utf-8")
    payload = {
        "image_base64": b64_str,
        "exercise": "FRONT_LEVER",
        "frame_sequence": 1,
        "timestamp_ms": 1000
    }
    response = client.post("/api/v1/analysis/live/frame-json", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["exercise"] == "FRONT_LEVER"
    frame_seq = data.get("frame_sequence") if "frame_sequence" in data else data.get("frameSequence")
    assert frame_seq == 1


def test_live_session_lifecycle(client):
    # 1. Start session
    init_res = client.post("/api/v1/analysis/live/session?exercise=HANDSTAND")
    assert init_res.status_code == 200
    session_data = init_res.json()
    session_id = session_data["session_id"]
    assert session_data["exercise"] == "HANDSTAND"

    # 2. Reset session
    reset_res = client.post(f"/api/v1/analysis/live/session/{session_id}/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["status"] == "RESET_SUCCESS"

    # 3. Terminate session
    del_res = client.delete(f"/api/v1/analysis/live/session/{session_id}")
    assert del_res.status_code == 200
    assert del_res.json()["status"] == "TERMINATED"

    # 4. Resetting nonexistent session returns 404
    missing_res = client.post(f"/api/v1/analysis/live/session/{session_id}/reset")
    assert missing_res.status_code == 404


def test_websocket_streaming(client, sample_jpeg_bytes):
    b64_str = base64.b64encode(sample_jpeg_bytes).decode("utf-8")
    with client.websocket_connect("/api/v1/analysis/live/ws") as ws:
        # 1. Ping / Pong
        ws.send_text('{"action": "ping"}')
        pong_msg = ws.receive_json()
        assert pong_msg["type"] == "pong"

        # 2. Configure exercise
        ws.send_text('{"action": "configure", "exercise": "FRONT_LEVER"}')
        conf_msg = ws.receive_json()
        assert conf_msg["status"] == "CONFIGURED"
        assert conf_msg["exercise"] == "FRONT_LEVER"

        # 3. Send frame
        frame_msg = {
            "action": "frame",
            "image_base64": b64_str,
            "exercise": "FRONT_LEVER",
            "frame_sequence": 1
        }
        ws.send_json(frame_msg)
        result = ws.receive_json()
        assert "status" in result
        assert result["exercise"] == "FRONT_LEVER"
        assert "timing" in result

        # 4. Reset
        ws.send_text('{"action": "reset"}')
        reset_res = ws.receive_json()
        assert reset_res["status"] == "RESET_SUCCESS"
