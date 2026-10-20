import pytest
from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_horizontal_angle,
    calculate_vertical_angle,
    compute_all_joint_angles,
)
from app.schemas.analysis import KeypointData


def test_calculate_angle_2d_right_angle():
    p1 = (0.0, 1.0)
    vertex = (0.0, 0.0)
    p2 = (1.0, 0.0)
    angle = calculate_angle_2d(p1, vertex, p2)
    assert abs(angle - 90.0) < 1e-4


def test_calculate_angle_2d_straight_line():
    p1 = (-1.0, 0.0)
    vertex = (0.0, 0.0)
    p2 = (1.0, 0.0)
    angle = calculate_angle_2d(p1, vertex, p2)
    assert abs(angle - 180.0) < 1e-4


def test_calculate_vertical_angle():
    top = (0.0, 0.0)
    bottom = (0.0, 100.0)
    # Perfectly vertical line
    vert_angle = calculate_vertical_angle(top, bottom)
    assert abs(vert_angle - 0.0) < 1e-4

    # 45 degree tilt
    top_tilted = (100.0, 0.0)
    vert_angle_tilted = calculate_vertical_angle(top_tilted, bottom)
    assert abs(vert_angle_tilted - 45.0) < 1e-4


def test_calculate_horizontal_angle():
    p1 = (0.0, 100.0)
    p2 = (100.0, 100.0)
    # Perfectly horizontal line
    horiz_angle = calculate_horizontal_angle(p1, p2)
    assert abs(horiz_angle - 0.0) < 1e-4


def test_compute_all_joint_angles():
    # Construct a synthetic keypoints dictionary for a straight horizontal body
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
