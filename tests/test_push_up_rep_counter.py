import time
import pytest
from app.biomechanics.push_up_counter import (
    PushUpCounterConfig,
    PushUpRepCounter,
    PushUpState,
    extract_push_up_geometry,
)
from app.exercises.push_up import PushUpAnalyzer
from app.schemas.analysis import KeypointData, LiveAnalysisStatus
from app.services.live_session import LiveSession, LiveSessionManager


def test_case_1_normal_push_up():
    """
    Case 1: Normal Push-up
    165° → 150° → 130° → 105° → 85° → 78° → 92° → 120° → 145° → 165°
    Expected result: Rep count = 1, state = TOP, depth = GOOD.
    """
    counter = PushUpRepCounter()
    angles = [165.0, 150.0, 130.0, 105.0, 85.0, 78.0, 92.0, 120.0, 145.0, 165.0]

    res = None
    for a in angles:
        res = counter.process_angle(a, body_alignment=178.0)

    assert counter.rep_count == 1
    assert counter.state == PushUpState.TOP
    assert res is not None
    assert res.depth == "GOOD"
    assert counter.min_elbow_angle <= 90.0
    assert counter.last_completed_min_angle <= 90.0


def test_case_2_partial_push_up():
    """
    Case 2: Partial Push-up
    165° → 145° → 130° → 110° → 125° → 145° → 165°
    Expected result: Rep count = 0 (never reaches bottom <= 90°).
    """
    counter = PushUpRepCounter()
    angles = [165.0, 145.0, 130.0, 110.0, 125.0, 145.0, 165.0]

    for a in angles:
        counter.process_angle(a)

    assert counter.rep_count == 0
    assert counter.state == PushUpState.TOP


def test_case_3_top_position_hold():
    """
    Case 3: Top Position Hold (plank / resting at top)
    165° → 164° → 166° → 165° → 163° → 165°
    Expected result: Rep count = 0, state = TOP.
    """
    counter = PushUpRepCounter()
    angles = [165.0, 164.0, 166.0, 165.0, 163.0, 165.0]

    for a in angles:
        counter.process_angle(a)

    assert counter.rep_count == 0
    assert counter.state == PushUpState.TOP


def test_case_4_bottom_position_hold():
    """
    Case 4: Bottom Position Hold
    165° → 130° → 100° → 80° → 80° → 80° → 80° → 85° → 115° → 140° → 165°
    Expected result: Rep count = 1 only.
    """
    counter = PushUpRepCounter()
    angles = [165.0, 130.0, 100.0, 80.0, 80.0, 80.0, 80.0, 85.0, 115.0, 140.0, 165.0]

    for a in angles:
        counter.process_angle(a)

    assert counter.rep_count == 1
    assert counter.state == PushUpState.TOP


def test_case_5_multiple_push_ups():
    """
    Case 5: Multiple Push-ups
    3 full repetitions in succession.
    Expected result: Rep count = 3.
    """
    counter = PushUpRepCounter()
    single_rep = [165.0, 145.0, 120.0, 95.0, 82.0, 75.0, 85.0, 110.0, 135.0, 165.0]

    for _ in range(3):
        # Resting frames between reps
        counter.process_angle(165.0)
        counter.process_angle(165.0)
        counter.process_angle(165.0)
        counter.process_angle(165.0)
        for a in single_rep:
            counter.process_angle(a)

    assert counter.rep_count == 3
    assert counter.state == PushUpState.TOP


def test_case_6_noisy_keypoints():
    """
    Case 6: Noisy Keypoints around boundary angles
    Small jitter around 145° (descending trigger) without completing descent.
    Real descent with noise around 90°.
    """
    counter = PushUpRepCounter()
    counter.process_angle(165.0)

    # Jitter near 145° boundary (148 -> 143 -> 147 -> 144 -> 155)
    jitter = [148.0, 143.0, 147.0, 144.0, 155.0]
    for a in jitter:
        counter.process_angle(a)
    assert counter.rep_count == 0

    # Real descent to bottom with noise near 90°
    descent = [140.0, 120.0, 95.0, 88.0, 92.0, 85.0, 86.0]
    for a in descent:
        counter.process_angle(a)

    assert counter.state == PushUpState.BOTTOM
    assert counter.rep_count == 0

    # Ascend to top
    ascent = [105.0, 125.0, 145.0, 165.0]
    for a in ascent:
        counter.process_angle(a)

    assert counter.rep_count == 1
    assert counter.state == PushUpState.TOP


def test_case_7_tracking_loss():
    """
    Case 7: Tracking loss handling
    Elbow keypoint lost during descent.
    Does not crash or count false rep. Recovers and counts valid rep.
    """
    counter = PushUpRepCounter()

    # Descend
    counter.process_angle(165.0)
    counter.process_angle(135.0)
    assert counter.state == PushUpState.DESCENDING

    # Keypoint lost
    counter.process_angle(None)
    counter.process_angle(None)
    assert counter.is_tracking is False
    assert counter.rep_count == 0

    # Keypoint recovers
    counter.process_angle(130.0)
    assert counter.is_tracking is True

    # Complete rep
    for a in [95.0, 80.0, 95.0, 120.0, 145.0, 165.0]:
        counter.process_angle(a)

    assert counter.rep_count == 1


def test_case_8_incomplete_recovery():
    """
    Case 8: Incomplete Recovery
    User descends to 80° but only returns to 140° (top >= 155° not reached).
    Expected result: Rep count = 0 (remains in ASCENDING).
    """
    counter = PushUpRepCounter()
    angles = [165.0, 135.0, 95.0, 80.0, 105.0, 125.0, 140.0]

    for a in angles:
        counter.process_angle(a)

    assert counter.rep_count == 0
    assert counter.state == PushUpState.ASCENDING


def test_push_up_body_alignment_feedback():
    """Test body alignment feedback during push-up (detecting sagging hips)."""
    counter = PushUpRepCounter()
    counter.process_angle(165.0, body_alignment=150.0)
    assert "straight" in counter.feedback.lower() or "hips" in counter.feedback.lower()


def test_push_up_unilateral_arm_occlusion():
    """Test geometry extraction when one arm is occluded in side view."""
    def _kp(name: str, x: float, y: float, conf: float = 0.9, vis: bool = True):
        return KeypointData(name=name, x=x, y=y, confidence=conf, visible=vis)

    # Left arm visible, right arm occluded
    left_only = {
        "left_shoulder": _kp("left_shoulder", 100, 100),
        "left_elbow": _kp("left_elbow", 100, 150),
        "left_wrist": _kp("left_wrist", 100, 200),
        "left_hip": _kp("left_hip", 200, 100),
        "left_ankle": _kp("left_ankle", 300, 100),
    }

    geom = extract_push_up_geometry(left_only, side_view=True)
    assert geom["is_pose_valid"] is True
    assert geom["primary_elbow_angle"] is not None
    assert abs(geom["primary_elbow_angle"] - 180.0) < 1.0


def test_push_up_session_integration():
    """Test PushUpAnalyzer integration with LiveSession isolated counter."""
    manager = LiveSessionManager()
    session = manager.get_or_create("session_pu", "PUSH_UP")
    analyzer = PushUpAnalyzer()

    # Top position
    analyzer.update_dynamic_state(
        session=session,
        is_valid_frame=True,
        measurements={"elbow_angle": 165.0, "body_line_angle": 180.0},
        timestamp_sec=1.0
    )
    assert session.rep_count == 0
    assert session.rep_phase == "TOP"

    # Full rep sequence
    angles = [145.0, 120.0, 95.0, 80.0, 85.0, 110.0, 135.0, 165.0]
    t = 1.1
    for a in angles:
        analyzer.update_dynamic_state(
            session=session,
            is_valid_frame=True,
            measurements={"elbow_angle": a, "body_line_angle": 180.0},
            timestamp_sec=t
        )
        t += 0.1

    assert session.rep_count == 1
    assert session.rep_phase == "TOP"
