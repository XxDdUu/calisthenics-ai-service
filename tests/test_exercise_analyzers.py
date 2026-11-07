import pytest
from app.exercises.dips import DipsAnalyzer
from app.exercises.front_lever import FrontLeverAnalyzer
from app.exercises.handstand import HandstandAnalyzer
from app.exercises.muscle_up import MuscleUpAnalyzer
from app.exercises.planche import PlancheAnalyzer
from app.exercises.pull_up import PullUpAnalyzer
from app.exercises.registry import exercise_registry
from app.schemas.analysis import FeedbackSeverity, KeypointData, LiveAnalysisStatus
from app.services.live_session import LiveSession


def test_registry_resolution():
    """Verify registry resolves aliases, casing, hyphens and unknown exercises properly."""
    assert isinstance(exercise_registry.get_analyzer("full_planche"), PlancheAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("STRADDLE-PLANCHE"), PlancheAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("FRONT_LEVER"), FrontLeverAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("tuck-front-lever"), FrontLeverAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("handstand"), HandstandAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("HSPU"), HandstandAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("pull_up"), PullUpAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("dips"), DipsAnalyzer)
    assert isinstance(exercise_registry.get_analyzer("muscle_up"), MuscleUpAnalyzer)
    # Default fallback
    assert isinstance(exercise_registry.get_analyzer("unknown_exercise_xyz"), PlancheAnalyzer)


def test_front_lever_full_good_form():
    analyzer = FrontLeverAnalyzer()
    measurements = {
        "elbow_angle": 175.0,
        "shoulder_angle": 75.0,
        "hip_angle": 174.0,
        "body_line_angle": 175.0,
        "knee_angle": 175.0,
        "horizontal_tilt": 6.0,
        "leg_spread_ratio": 1.0
    }
    score, feedback, is_valid = analyzer.evaluate_live_form(measurements, "FULL_FRONT_LEVER")
    assert score is not None and score >= 85.0
    assert is_valid is True
    assert any(fb.code == "GOOD_FORM" for fb in feedback)


def test_front_lever_hip_sag_and_bent_elbows():
    analyzer = FrontLeverAnalyzer()
    measurements = {
        "elbow_angle": 140.0,  # severely bent
        "shoulder_angle": 75.0,
        "hip_angle": 140.0,    # sagging hips
        "body_line_angle": 140.0,
        "knee_angle": 170.0,
        "horizontal_tilt": 10.0,
        "leg_spread_ratio": 1.0
    }
    score, feedback, is_valid = analyzer.evaluate_live_form(measurements, "FULL_FRONT_LEVER")
    assert score is not None and score < 60.0
    assert is_valid is False
    codes = [fb.code for fb in feedback]
    assert "EXTEND_ELBOWS" in codes
    assert "LIFT_HIPS" in codes


def test_front_lever_tuck_progression():
    analyzer = FrontLeverAnalyzer()
    # Good tuck: hips <= 100, knees <= 90
    measurements = {
        "elbow_angle": 172.0,
        "shoulder_angle": 75.0,
        "hip_angle": 85.0,
        "body_line_angle": 85.0,
        "knee_angle": 80.0,
        "horizontal_tilt": 8.0,
        "leg_spread_ratio": 1.0
    }
    score, feedback, is_valid = analyzer.evaluate_live_form(measurements, "TUCK_FRONT_LEVER")
    assert score is not None and score >= 85.0
    assert is_valid is True


def test_handstand_hold_evaluation():
    analyzer = HandstandAnalyzer()
    # Good vertical handstand: straight arms, open shoulders, hollow body
    measurements = {
        "elbow_angle": 175.0,
        "shoulder_angle": 172.0,
        "body_line_angle": 175.0,
        "vertical_tilt": 5.0
    }
    score, feedback, is_valid = analyzer.evaluate_live_form(measurements, "HANDSTAND")
    assert score is not None and score >= 90.0
    assert is_valid is True
    assert any(fb.code == "GOOD_FORM" for fb in feedback)


def test_handstand_banana_back_and_closed_shoulders():
    analyzer = HandstandAnalyzer()
    measurements = {
        "elbow_angle": 175.0,
        "shoulder_angle": 140.0,  # closed shoulders
        "body_line_angle": 145.0,  # arched back
        "vertical_tilt": 18.0
    }
    score, feedback, is_valid = analyzer.evaluate_live_form(measurements, "HANDSTAND")
    assert score is not None and score < 60.0
    assert is_valid is False
    codes = [fb.code for fb in feedback]
    assert "OPEN_SHOULDERS" in codes
    assert "TUCK_RIBS" in codes


def test_handstand_pushup_rep_cycle():
    analyzer = HandstandAnalyzer()
    session = LiveSession(session_id="test_hspu", exercise="HANDSTAND_PUSHUP")

    # Start at top lockout (elbow 175)
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 175.0}, 0.0)
    assert session.rep_phase == "TOP"

    # Descend (elbow 120 -> 80)
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 120.0}, 0.5)
    assert session.rep_phase == "DESCENDING"
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 85.0}, 1.0)
    assert session.rep_phase == "BOTTOM"

    # Push back up (elbow 130 -> 172)
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 130.0}, 1.5)
    assert session.rep_phase == "ASCENDING"
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 172.0}, 2.0)
    assert session.rep_phase == "TOP"
    assert session.rep_count == 1


def test_pull_up_rep_cycle_and_asymmetry():
    analyzer = PullUpAnalyzer()
    session = LiveSession(session_id="test_pullup", exercise="PULL_UP")

    # Test asymmetry warning
    score, feedback, is_valid = analyzer.evaluate_live_form(
        {"elbow_angle": 100.0, "elbow_symmetry_diff": 35.0}, "PULL_UP"
    )
    assert is_valid is False
    assert any(fb.code == "SYMMETRIC_PULL" for fb in feedback)

    # Test rep cycle: BOTTOM -> PULLING -> TOP -> LOWERING -> BOTTOM (rep=1)
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 170.0}, 0.0)
    assert session.rep_phase == "BOTTOM"

    analyzer.update_dynamic_state(session, True, {"elbow_angle": 120.0}, 0.5)
    assert session.rep_phase == "PULLING"

    analyzer.update_dynamic_state(session, True, {"elbow_angle": 55.0}, 1.0)
    assert session.rep_phase == "TOP"

    analyzer.update_dynamic_state(session, True, {"elbow_angle": 110.0}, 1.5)
    assert session.rep_phase == "LOWERING"

    analyzer.update_dynamic_state(session, True, {"elbow_angle": 165.0}, 2.0)
    assert session.rep_phase == "BOTTOM"
    assert session.rep_count == 1


def test_dips_rep_cycle():
    analyzer = DipsAnalyzer()
    session = LiveSession(session_id="test_dips", exercise="DIPS")

    # Start at top lockout (elbow 172)
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 172.0}, 0.0)
    assert session.rep_phase == "TOP"

    # Descend to 90 degrees
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 120.0}, 0.5)
    assert session.rep_phase == "DESCENDING"
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 85.0}, 1.0)
    assert session.rep_phase == "BOTTOM"

    # Ascend back to lockout
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 130.0}, 1.5)
    assert session.rep_phase == "ASCENDING"
    analyzer.update_dynamic_state(session, True, {"elbow_angle": 170.0}, 2.0)
    assert session.rep_phase == "TOP"
    assert session.rep_count == 1


def test_muscle_up_chicken_wing():
    analyzer = MuscleUpAnalyzer()
    measurements = {
        "elbow_angle": 110.0,
        "transition_asymmetry": 35.0,  # severe chicken wing
        "chest_above_bar": 1.0
    }
    score, feedback, is_valid = analyzer.evaluate_live_form(measurements, "MUSCLE_UP")
    assert is_valid is False
    assert any(fb.code == "TRANSITION_EVENLY" and fb.severity == FeedbackSeverity.ERROR for fb in feedback)
