import io
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_dummy_jpeg() -> bytes:
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data


def test_exercises_endpoint():
    response = client.get("/api/v1/analysis/exercises")
    assert response.status_code == 200
    data = response.json()
    assert "supported_exercises" in data
    assert data["supported_exercises"][0]["id"] == "planche"


def test_invalid_image_upload():
    response = client.post(
        "/api/v1/analysis/image",
        files={"file": ("test.txt", b"invalid image content", "text/plain")}
    )
    assert response.status_code == 400


def test_export_draw_skeleton_invalid_image():
    response = client.post(
        "/api/v1/analysis/image/draw-skeleton",
        files={"file": ("test.txt", b"invalid image content", "text/plain")}
    )
    assert response.status_code == 400


def test_image_analysis_side_view_parameter():
    """Test image endpoint handles progression and side_view query params."""
    jpeg_bytes = create_dummy_jpeg()
    response = client.post(
        "/api/v1/analysis/image?progression=FULL_PLANCHE&side_view=true&debug=true",
        files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "side_view" in data
    assert data["side_view"] is True


def test_image_analysis_standard_mode():
    """Test image endpoint in standard mode (side_view=false)."""
    jpeg_bytes = create_dummy_jpeg()
    response = client.post(
        "/api/v1/analysis/image?progression=STRADDLE_PLANCHE&side_view=false",
        files={"file": ("test.jpg", jpeg_bytes, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "side_view" in data
    assert data["side_view"] is False
