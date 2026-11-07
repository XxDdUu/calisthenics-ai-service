import time
import pytest
from app.schemas.analysis import KeypointData
from app.services.live_session import LiveSession, LiveSessionManager


def test_session_isolation():
    manager = LiveSessionManager()
    s1 = manager.get_or_create(session_id="user_1", exercise="FULL_PLANCHE")
    s2 = manager.get_or_create(session_id="user_2", exercise="FRONT_LEVER")

    assert s1.session_id == "user_1"
    assert s2.session_id == "user_2"
    assert s1.exercise == "FULL_PLANCHE"
    assert s2.exercise == "FRONT_LEVER"

    s1.touch()
    assert s1.frame_count == 1
    assert s2.frame_count == 0


def test_session_reset():
    session = LiveSession(session_id="test_reset", exercise="FULL_PLANCHE")
    session.touch()
    session.current_hold_duration_sec = 5.2
    session.peak_score = 92.0
    session.rep_count = 3
    session.rep_phase = "TOP"

    session.reset()
    assert session.frame_count == 0
    assert session.current_hold_duration_sec == 0.0
    assert session.peak_score == 0.0
    assert session.rep_count == 0
    assert session.rep_phase == "START"


def test_temporal_smoothing_measurements():
    session = LiveSession(session_id="test_smooth", exercise="FULL_PLANCHE")
    session.measurement_alpha = 0.5  # 50% current, 50% previous

    # First frame: initial value
    m1 = session.smooth_measurements({"elbow_angle": 180.0})
    assert m1["elbow_angle"] == 180.0

    # Second frame with slight noise (170.0) -> smooth = 0.5 * 170 + 0.5 * 180 = 175.0
    m2 = session.smooth_measurements({"elbow_angle": 170.0})
    assert abs(m2["elbow_angle"] - 175.0) < 0.2

    # Third frame with large jump (teleportation/switch: 90.0 vs 175.0) -> resets to raw value
    m3 = session.smooth_measurements({"elbow_angle": 90.0})
    assert m3["elbow_angle"] == 90.0


def test_session_exercise_change_resets_history():
    manager = LiveSessionManager()
    s = manager.get_or_create("sess_reuse", "FULL_PLANCHE")
    s.peak_score = 85.0
    s.frame_count = 10

    # Changing exercise on same session ID resets historical state to prevent cross-exercise contamination
    s_new = manager.get_or_create("sess_reuse", "FRONT_LEVER")
    assert s_new.exercise == "FRONT_LEVER"
    assert s_new.frame_count == 0
    assert s_new.peak_score == 0.0


def test_session_eviction():
    manager = LiveSessionManager(max_idle_sec=0.1)
    s = manager.get_or_create("short_lived", "HANDSTAND")
    assert manager.get("short_lived") is not None

    time.sleep(0.15)
    # Getting or creating another session triggers cleanup
    manager.get_or_create("new_session", "DIPS")
    assert manager.get("short_lived") is None
