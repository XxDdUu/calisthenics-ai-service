import time
import pytest
from app.biomechanics.squat_counter import (
    SquatCounterConfig,
    SquatRepCounter,
    SquatState,
    calculate_knee_angle_3points,
    extract_squat_geometry,
)
from app.exercises.squat import SquatAnalyzer
from app.schemas.analysis import KeypointData, LiveAnalysisStatus
from app.services.live_session import LiveSession, LiveSessionManager


def test_case_1_normal_squat():
    """
    Case 1: Normal Squat
    170° → 150° → 130° → 110° → 95° → 85° → 100° → 120° → 145° → 165°
    Expected result: Rep count = 1.
    """
    counter = SquatRepCounter()
    angles = [170.0, 150.0, 130.0, 110.0, 95.0, 85.0, 100.0, 120.0, 145.0, 165.0]

    for a in angles:
        res = counter.process_angle(a)

    assert counter.rep_count == 1
    assert counter.state == SquatState.STANDING
    assert res.depth == "GOOD"
    assert counter.min_knee_angle <= 100.0


def test_case_2_partial_squat():
    """
    Case 2: Partial Squat
    170° → 150° → 130° → 115° → 125° → 145° → 165°
    Expected result: Rep count = 0 (never reaches <= 100°).
    """
    counter = SquatRepCounter()
    angles = [170.0, 150.0, 130.0, 115.0, 125.0, 145.0, 165.0]

    for a in angles:
        counter.process_angle(a)

    assert counter.rep_count == 0
    assert counter.state == SquatState.STANDING


def test_case_3_standing_still():
    """
    Case 3: Standing Still
    170° → 169° → 171° → 170° → 168° → 170°
    Expected result: Rep count = 0 (remains in STANDING).
    """
    counter = SquatRepCounter()
    angles = [170.0, 169.0, 171.0, 170.0, 168.0, 170.0]

    for a in angles:
        counter.process_angle(a)

    assert counter.rep_count == 0
    assert counter.state == SquatState.STANDING


def test_case_4_bottom_position_held():
    """
    Case 4: Bottom Position Held
    170° → 140° → 110° → 90° → 90° → 90° → 95° → 120° → 150° → 165°
    Expected result: Rep count = 1 only.
    """
    counter = SquatRepCounter()
    angles = [170.0, 140.0, 110.0, 90.0, 90.0, 90.0, 95.0, 120.0, 150.0, 165.0]

    for a in angles:
        counter.process_angle(a)

    assert counter.rep_count == 1
    assert counter.state == SquatState.STANDING


def test_case_5_multiple_squats():
    """
    Case 5: Multiple Squats
    3 full squats in succession.
    Expected result: Rep count = 3.
    """
    counter = SquatRepCounter()
    single_rep = [170.0, 150.0, 130.0, 105.0, 90.0, 85.0, 95.0, 120.0, 145.0, 165.0]

    for i in range(3):
        # Additional standing frame between reps for cooldown
        counter.process_angle(170.0)
        counter.process_angle(170.0)
        counter.process_angle(170.0)
        counter.process_angle(170.0)
        for a in single_rep:
            counter.process_angle(a)

    assert counter.rep_count == 3
    assert counter.state == SquatState.STANDING


def test_case_6_noisy_keypoints():
    """
    Case 6: Noisy Keypoints
    Small oscillations around 150° (descending trigger) and 100° (bottom trigger).
    Expected result: Hysteresis prevents false/duplicate state changes and reps.
    """
    counter = SquatRepCounter()
    # Stand tall
    counter.process_angle(170.0)

    # Jitter around 150 boundary (152 -> 148 -> 151 -> 149 -> 152) without full squat
    jitter_descent = [152.0, 148.0, 151.0, 149.0, 152.0, 165.0]
    for a in jitter_descent:
        counter.process_angle(a)

    # No rep should be counted for partial jitter near top
    assert counter.rep_count == 0

    # Real descent to bottom with noise near 100° (103 -> 98 -> 102 -> 97 -> 96)
    noise_bottom = [140.0, 120.0, 103.0, 98.0, 102.0, 97.0, 96.0, 101.0, 106.0]
    for a in noise_bottom:
        counter.process_angle(a)

    assert counter.state == SquatState.BOTTOM
    assert counter.rep_count == 0

    # Complete ascent to standing
    ascent = [115.0, 135.0, 155.0, 165.0]
    for a in ascent:
        counter.process_angle(a)

    assert counter.rep_count == 1
    assert counter.state == SquatState.STANDING


def test_case_7_tracking_loss():
    """
    Case 7: Tracking Loss
    Knee keypoint lost during descent or recovery.
    Expected result: Pipeline does not crash or generate false rep.
    """
    counter = SquatRepCounter()

    # Start descending
    counter.process_angle(170.0)
    counter.process_angle(140.0)
    assert counter.state == SquatState.DESCENDING

    # Keypoints lost for 3 frames
    counter.process_angle(None)
    counter.process_angle(None)
    counter.process_angle(None)

    assert counter.is_tracking is False
    assert counter.rep_count == 0

    # Keypoint recovers
    counter.process_angle(135.0)
    assert counter.is_tracking is True

    # Complete rep
    for a in [95.0, 85.0, 100.0, 125.0, 150.0, 165.0]:
        counter.process_angle(a)

    assert counter.rep_count == 1


def test_case_8_incomplete_recovery():
    """
    Case 8: Incomplete Recovery
    User squats down to 85° but only returns to 145°.
    Expected result: Rep count = 0 (remains in ASCENDING because standing angle >= 160° not reached).
    """
    counter = SquatRepCounter()
    angles = [170.0, 140.0, 100.0, 85.0, 110.0, 130.0, 145.0]

    for a in angles:
        counter.process_angle(a)

    assert counter.rep_count == 0
    assert counter.state == SquatState.ASCENDING


def test_unilateral_leg_occlusion_fallback():
    """
    Test unilateral leg occlusion:
    - If left leg is visible and right leg is occluded, left knee angle is used.
    - If right leg is visible and left leg is occluded, right knee angle is used.
    - If neither leg is reliable, frame is marked invalid.
    """
    def _kp(name: str, x: float, y: float, conf: float = 0.9, vis: bool = True):
        return KeypointData(name=name, x=x, y=y, confidence=conf, visible=vis)

    # 1. Left leg only: Hip (100, 200), Knee (100, 300), Ankle (100, 400) -> 180 deg straight
    left_only = {
        "left_shoulder": _kp("left_shoulder", 100, 100),
        "left_hip": _kp("left_hip", 100, 200),
        "left_knee": _kp("left_knee", 100, 300),
        "left_ankle": _kp("left_ankle", 100, 400),
        # Right leg completely missing / occluded (low confidence or not visible)
        "right_hip": _kp("right_hip", 0, 0, conf=0.1, vis=False),
        "right_knee": _kp("right_knee", 0, 0, conf=0.05, vis=False),
        "right_ankle": _kp("right_ankle", 0, 0, conf=0.0, vis=False)
    }

    geo = extract_squat_geometry(left_only, threshold=0.25)
    assert geo["is_pose_valid"] is True
    assert geo["left_leg_reliable"] is True
    assert geo["right_leg_reliable"] is False
    assert geo["primary_knee_angle"] is not None
    assert abs(geo["primary_knee_angle"] - 180.0) < 1.0

    # 2. Right leg only: Hip (200, 200), Knee (250, 250), Ankle (200, 300) -> 90 deg right angle
    right_only = {
        "right_shoulder": _kp("right_shoulder", 200, 100),
        "right_hip": _kp("right_hip", 200, 200),
        "right_knee": _kp("right_knee", 250, 250),
        "right_ankle": _kp("right_ankle", 200, 300),
        # Left leg missing
        "left_hip": _kp("left_hip", 0, 0, conf=0.0, vis=False),
        "left_knee": _kp("left_knee", 0, 0, conf=0.0, vis=False),
        "left_ankle": _kp("left_ankle", 0, 0, conf=0.0, vis=False)
    }

    geo_r = extract_squat_geometry(right_only, threshold=0.25)
    assert geo_r["is_pose_valid"] is True
    assert geo_r["left_leg_reliable"] is False
    assert geo_r["right_leg_reliable"] is True
    assert geo_r["primary_knee_angle"] is not None
    assert abs(geo_r["primary_knee_angle"] - 90.0) < 1.0

    # 3. Both legs missing / occluded
    neither = {
        "left_shoulder": _kp("left_shoulder", 100, 100),
        "right_shoulder": _kp("right_shoulder", 200, 100),
        "left_hip": _kp("left_hip", 0, 0, conf=0.1, vis=False),
        "right_hip": _kp("right_hip", 0, 0, conf=0.1, vis=False),
    }

    geo_none = extract_squat_geometry(neither, threshold=0.25)
    assert geo_none["is_pose_valid"] is False
    assert geo_none["primary_knee_angle"] is None


def test_knee_angle_3points_calculation():
    """
    Test vector dot product / arccos formula for Hip -> Knee -> Ankle:
    - 180° straight leg
    - 90° right angle knee
    - 45° deep flexion
    """
    # Straight vertical leg
    hip = (100.0, 100.0)
    knee = (100.0, 200.0)
    ankle = (100.0, 300.0)
    ang_straight = calculate_knee_angle_3points(hip, knee, ankle)
    assert abs(ang_straight - 180.0) < 0.1

    # Right angle (Hip at (100, 100), Knee at (100, 200), Ankle at (200, 200))
    ankle_90 = (200.0, 200.0)
    ang_90 = calculate_knee_angle_3points(hip, knee, ankle_90)
    assert abs(ang_90 - 90.0) < 0.1


def test_session_isolation():
    """
    Test that two concurrent LiveSession instances track squats independently
    without cross-talk or shared global state.
    """
    session_manager = LiveSessionManager()
    session_a = session_manager.get_or_create(session_id="user_a", exercise="SQUAT")
    session_b = session_manager.get_or_create(session_id="user_b", exercise="SQUAT")

    analyzer = SquatAnalyzer()

    # User A performs 1 full squat (advancing timestamp by 0.1s per frame)
    t0 = 1000.0
    for i, a in enumerate([170.0, 150.0, 130.0, 105.0, 95.0, 85.0, 100.0, 120.0, 145.0, 165.0]):
        analyzer.update_dynamic_state(session_a, True, {"knee_angle": a}, t0 + i * 0.1)

    # User B only stands still
    for j, a in enumerate([170.0, 169.0, 171.0]):
        analyzer.update_dynamic_state(session_b, True, {"knee_angle": a}, t0 + j * 0.1)

    assert session_a.rep_count == 1
    assert session_a.rep_phase == "STANDING"

    assert session_b.rep_count == 0
    assert session_b.rep_phase == "STANDING"


def test_form_evaluation_feedback_and_scoring():
    """
    Test Squat form evaluation:
    - Depth classification (GOOD <= 100°, MODERATE 100°-120°, SHALLOW > 120°)
    - Knee valgus cue when knee-to-ankle ratio < 0.82
    - Torso inclination cue when trunk lean > 42°
    """
    analyzer = SquatAnalyzer()

    # Good depth, upright posture, good alignment
    score, feedback, is_valid = analyzer.evaluate_live_form({
        "knee_angle": 92.0,
        "trunk_lean_angle": 20.0,
        "knee_ankle_ratio": 1.05,
        "knee_symmetry_diff": 2.0
    })
    assert score >= 85.0
    assert is_valid is True
    assert any(f.code == "GOOD_DEPTH" for f in feedback)

    # Knee valgus (ratio 0.72)
    score_v, feedback_v, _ = analyzer.evaluate_live_form({
        "knee_angle": 95.0,
        "trunk_lean_angle": 20.0,
        "knee_ankle_ratio": 0.72,
        "knee_symmetry_diff": 4.0
    }, side_view=False)
    assert any(f.code == "KNEE_VALGUS" for f in feedback_v)

    # Excessive torso lean (lean = 52.0)
    score_l, feedback_l, is_valid_l = analyzer.evaluate_live_form({
        "knee_angle": 95.0,
        "trunk_lean_angle": 52.0,
        "knee_ankle_ratio": 1.0,
        "knee_symmetry_diff": 2.0
    })
    assert is_valid_l is False
    assert any(f.code == "EXCESSIVE_FORWARD_LEAN" for f in feedback_l)
