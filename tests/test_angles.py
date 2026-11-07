import math
import pytest
from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_body_line_deviation,
    calculate_distance_2d,
    calculate_horizontal_angle,
    calculate_vertical_angle,
    compute_all_joint_angles,
    is_valid_coord,
    is_valid_point,
)
from app.schemas.analysis import KeypointData


def test_calculate_angle_2d_right_angle():
    p1 = (0.0, 1.0)
    vertex = (0.0, 0.0)
    p2 = (1.0, 0.0)
    angle = calculate_angle_2d(p1, vertex, p2)
    assert abs(angle - 90.0) < 1e-1


def test_calculate_angle_2d_straight_line():
    p1 = (-1.0, 0.0)
    vertex = (0.0, 0.0)
    p2 = (1.0, 0.0)
    angle = calculate_angle_2d(p1, vertex, p2)
    assert abs(angle - 180.0) < 1e-1


def test_straight_elbow():
    """Test locked straight elbow formed by shoulder, elbow, wrist in a line."""
    shoulder = (100.0, 100.0)
    elbow = (150.0, 100.0)
    wrist = (200.0, 100.0)
    angle = calculate_angle_2d(shoulder, elbow, wrist)
    assert angle >= 179.0


def test_bent_elbow():
    """Test bent elbow (e.g., 90 degree flexion)."""
    shoulder = (100.0, 100.0)
    elbow = (150.0, 100.0)
    wrist = (150.0, 150.0)
    angle = calculate_angle_2d(shoulder, elbow, wrist)
    assert abs(angle - 90.0) < 1e-1


def test_straight_body_line():
    """Test straight body line where shoulder, hip, ankle form 180 degrees."""
    shoulder = (100.0, 200.0)
    hip = (200.0, 200.0)
    ankle = (300.0, 200.0)
    angle, alignment = calculate_body_line_deviation(shoulder, hip, ankle)
    assert abs(angle - 180.0) < 1e-1
    assert alignment == "STRAIGHT"


def test_hip_sag():
    """Test sagging hips (dropped downward in image space, higher Y)."""
    shoulder = (100.0, 200.0)
    hip = (200.0, 240.0)  # Drooping down by 40px
    ankle = (300.0, 200.0)
    angle, alignment = calculate_body_line_deviation(shoulder, hip, ankle)
    assert angle < 170.0
    assert alignment == "SAG"


def test_hip_pike():
    """Test piked hips (flexed upward in image space, lower Y)."""
    shoulder = (100.0, 200.0)
    hip = (200.0, 160.0)  # Elevated up by 40px
    ankle = (300.0, 200.0)
    angle, alignment = calculate_body_line_deviation(shoulder, hip, ankle)
    assert angle < 170.0
    assert alignment == "PIKE"


def test_missing_landmark():
    """Ensure None or missing landmarks safely return default angles without crashing."""
    assert calculate_angle_2d(None, (0.0, 0.0), (1.0, 1.0)) == 180.0
    assert calculate_angle_2d((1.0, 1.0), None, (1.0, 1.0)) == 180.0
    assert calculate_angle_2d((1.0, 1.0), (0.0, 0.0), None) == 180.0
    assert calculate_distance_2d(None, (1.0, 1.0)) == 0.0


def test_low_confidence_landmark():
    """Verify that get_point filters out low-confidence/invisible keypoints."""
    from app.biomechanics.angles import get_point
    kp_hidden = KeypointData(x=100.0, y=100.0, confidence=0.1, name="wrist", visible=False)
    assert get_point(kp_hidden) is None

    kp_visible = KeypointData(x=100.0, y=100.0, confidence=0.8, name="wrist", visible=True)
    assert get_point(kp_visible) == (100.0, 100.0)


def test_degenerate_coordinates():
    """Ensure identical points, zero-length vectors, NaN, and Infinity never crash or yield NaN."""
    p_same = (100.0, 100.0)
    # Identical points (zero-length vector)
    assert calculate_angle_2d(p_same, p_same, (200.0, 200.0)) == 180.0
    assert calculate_angle_2d((200.0, 200.0), p_same, p_same) == 180.0

    # NaN coordinates
    nan_pt = (float("nan"), 100.0)
    assert calculate_angle_2d(nan_pt, (100.0, 100.0), (200.0, 200.0)) == 180.0
    assert calculate_vertical_angle(nan_pt, (100.0, 100.0)) == 90.0
    assert calculate_horizontal_angle(nan_pt, (100.0, 100.0)) == 90.0

    # Infinite coordinates
    inf_pt = (float("inf"), 100.0)
    assert calculate_angle_2d(inf_pt, (100.0, 100.0), (200.0, 200.0)) == 180.0

    # Helper coordinate checkers
    assert not is_valid_coord(float("nan"))
    assert not is_valid_coord(float("inf"))
    assert not is_valid_point((float("nan"), 0.0))
    assert is_valid_point((10.0, 20.0))


def test_calculate_vertical_angle():
    top = (0.0, 0.0)
    bottom = (0.0, 100.0)
    vert_angle = calculate_vertical_angle(top, bottom)
    assert abs(vert_angle - 0.0) < 1e-1

    top_tilted = (100.0, 0.0)
    vert_angle_tilted = calculate_vertical_angle(top_tilted, bottom)
    assert abs(vert_angle_tilted - 45.0) < 1e-1


def test_calculate_horizontal_angle():
    p1 = (0.0, 100.0)
    p2 = (100.0, 100.0)
    horiz_angle = calculate_horizontal_angle(p1, p2)
    assert abs(horiz_angle - 0.0) < 1e-1


def test_compute_all_joint_angles():
    kps = {
        "left_shoulder": KeypointData(x=100, y=100, confidence=0.9, name="left_shoulder", visible=True),
        "left_elbow": KeypointData(x=150, y=100, confidence=0.9, name="left_elbow", visible=True),
        "left_wrist": KeypointData(x=200, y=100, confidence=0.9, name="left_wrist", visible=True),
        "left_hip": KeypointData(x=300, y=100, confidence=0.9, name="left_hip", visible=True),
        "left_knee": KeypointData(x=400, y=100, confidence=0.9, name="left_knee", visible=True),
        "left_ankle": KeypointData(x=500, y=100, confidence=0.9, name="left_ankle", visible=True),
    }

    angles = compute_all_joint_angles(kps)
    assert angles.elbow_left_deg == 180.0
    assert angles.hip_angle_deg == 180.0
    assert angles.knee_left_deg == 180.0
    assert angles.primary_side == "left"
