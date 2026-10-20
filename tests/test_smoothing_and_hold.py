import pytest
from app.biomechanics.hold_detection import HoldState, PlancheHoldDetector
from app.biomechanics.smoothing import TemporalKeypointSmoother
from app.schemas.analysis import KeypointData


def test_temporal_smoothing_ema():
    """Test EMA smoothing formula: alpha * curr + (1 - alpha) * prev."""
    smoother = TemporalKeypointSmoother(alpha=0.6)

    # Frame 1: x = 100
    f1 = {"left_wrist": KeypointData(x=100.0, y=200.0, confidence=0.8, name="left_wrist", visible=True)}
    res1 = smoother.smooth(f1)
    assert res1["left_wrist"].x == 100.0

    # Frame 2: x = 110 (jitter)
    # Expected: 0.6 * 110 + 0.4 * 100 = 66 + 40 = 106.0
    f2 = {"left_wrist": KeypointData(x=110.0, y=200.0, confidence=0.8, name="left_wrist", visible=True)}
    res2 = smoother.smooth(f2)
    assert abs(res2["left_wrist"].x - 106.0) < 1e-2


def test_temporal_smoothing_resets_on_large_jump():
    """Test smoother resets when landmark teleports (e.g. tracking switch)."""
    smoother = TemporalKeypointSmoother(alpha=0.6, max_jump_distance_px=50.0)

    f1 = {"left_wrist": KeypointData(x=100.0, y=200.0, confidence=0.8, name="left_wrist", visible=True)}
    smoother.smooth(f1)

    # Sudden jump of 200px
    f2 = {"left_wrist": KeypointData(x=300.0, y=200.0, confidence=0.8, name="left_wrist", visible=True)}
    res2 = smoother.smooth(f2)
    # Should not blend over 50px gap, adopts new position 300.0
    assert res2["left_wrist"].x == 300.0


def test_hold_detector_transitions_to_stable_hold():
    """
    Test state machine:
    POSE_DETECTED -> POTENTIAL_HOLD -> STABLE_HOLD -> HOLD_ENDED
    """
    fps = 10.0
    detector = PlancheHoldDetector(
        fps=fps,
        min_hold_duration_sec=0.5,  # 5 frames at 10 fps
        stability_variance_threshold=10.0,
        min_elbow_deg=160.0,
        min_lean_deg=10.0,
        max_tilt_deg=20.0
    )

    # Frame 0: Standing / relaxed
    state0 = detector.process_frame(
        frame_idx=0, timestamp_sec=0.0, pose_detected=True, is_valid_quality=True,
        elbow_angle=100.0, shoulder_lean=0.0, hip_angle=180.0, body_tilt=80.0, form_score=20.0
    )
    assert state0 == HoldState.POSE_DETECTED

    # Frames 1-4: Planche position assumed (potential hold, duration 0.1s to 0.4s)
    for i in range(1, 5):
        state = detector.process_frame(
            frame_idx=i, timestamp_sec=i / fps, pose_detected=True, is_valid_quality=True,
            elbow_angle=175.0, shoulder_lean=18.0, hip_angle=175.0, body_tilt=5.0, form_score=90.0
        )
        assert state == HoldState.POTENTIAL_HOLD

    # Frames 5-9: Held steady past min_hold_duration_sec (0.5s reached at frame 5: 5/10=0.5s)
    for i in range(5, 10):
        state = detector.process_frame(
            frame_idx=i, timestamp_sec=i / fps, pose_detected=True, is_valid_quality=True,
            elbow_angle=175.0, shoulder_lean=18.0, hip_angle=175.0, body_tilt=5.0, form_score=92.0
        )
        assert state == HoldState.STABLE_HOLD

    # Frame 10: Form breaks (arms collapse) -> HOLD_ENDED
    state_end = detector.process_frame(
        frame_idx=10, timestamp_sec=10 / fps, pose_detected=True, is_valid_quality=True,
        elbow_angle=120.0, shoulder_lean=5.0, hip_angle=140.0, body_tilt=40.0, form_score=30.0
    )
    assert state_end == HoldState.HOLD_ENDED

    summary = detector.get_summary(sample_interval=1)
    assert summary.is_valid_sustained_hold is True
    assert summary.stable_hold_duration_sec >= 0.5
    assert summary.average_score >= 90.0


def test_hold_detector_rejects_unstable_jittery_attempt():
    """Test that erratic, unstable movement does not transition to STABLE_HOLD."""
    fps = 10.0
    detector = PlancheHoldDetector(
        fps=fps,
        min_hold_duration_sec=0.5,
        stability_variance_threshold=5.0  # Tight variance
    )

    # Alternate wildly in hip angle and tilt
    for i in range(10):
        hip = 175.0 if i % 2 == 0 else 130.0
        tilt = 5.0 if i % 2 == 0 else 30.0
        state = detector.process_frame(
            frame_idx=i, timestamp_sec=i / fps, pose_detected=True, is_valid_quality=True,
            elbow_angle=170.0, shoulder_lean=15.0, hip_angle=hip, body_tilt=tilt, form_score=60.0
        )
        # Because of high variance, should remain in POTENTIAL_HOLD rather than STABLE_HOLD
        assert state in (HoldState.POTENTIAL_HOLD, HoldState.POSE_DETECTED)
