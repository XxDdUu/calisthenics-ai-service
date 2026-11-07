"""
Unit and integration tests for side-view Planche keypoint fixes (Cases A through F).
"""
import numpy as np
import pytest
from app.biomechanics.metrics import BiomechanicsEvaluator
from app.biomechanics.side_view import (
    SideViewKeypointProcessor,
    resolve_side_view_chain,
)
from app.exercises.planche import PlancheAnalyzer
from app.pose.yolo_pose import YOLOPoseEstimator
from app.schemas.analysis import KeypointData, PoseKeypoints, ProgressionStage


def create_base_side_keypoints():
    """Horizontal athlete facing right: shoulders at 100, hips at 250 (torso = 150px)."""
    return {
        "nose": KeypointData(x=60, y=100, confidence=0.85, name="nose", visible=True),
        "left_shoulder": KeypointData(x=100, y=95, confidence=0.88, name="left_shoulder", visible=True),
        "right_shoulder": KeypointData(x=100, y=105, confidence=0.40, name="right_shoulder", visible=True),
        "left_elbow": KeypointData(x=100, y=150, confidence=0.85, name="left_elbow", visible=True),
        "right_elbow": KeypointData(x=100, y=150, confidence=0.35, name="right_elbow", visible=True),
        "left_wrist": KeypointData(x=100, y=200, confidence=0.82, name="left_wrist", visible=True),
        "right_wrist": KeypointData(x=100, y=200, confidence=0.30, name="right_wrist", visible=True),
        "left_hip": KeypointData(x=250, y=95, confidence=0.90, name="left_hip", visible=True),
        "right_hip": KeypointData(x=250, y=105, confidence=0.45, name="right_hip", visible=True),
        "left_knee": KeypointData(x=380, y=95, confidence=0.85, name="left_knee", visible=True),
        "right_knee": KeypointData(x=380, y=105, confidence=0.40, name="right_knee", visible=True),
        "left_ankle": KeypointData(x=500, y=95, confidence=0.80, name="left_ankle", visible=True),
        "right_ankle": KeypointData(x=500, y=105, confidence=0.35, name="right_ankle", visible=True),
    }


def test_case_a_visible_ankle_below_threshold_no_silent_substitution():
    """
    Case A: Visible ankle is below threshold; opposite (occluded) ankle is hallucinated.
    Verify: Hip and knee are preserved; ankle is omitted rather than silently substituted.
    """
    kps = create_base_side_keypoints()
    # Left ankle below threshold
    kps["left_ankle"] = KeypointData(x=500, y=95, confidence=0.15, name="left_ankle", visible=True)
    # Right ankle hallucinated at bogus coordinates with moderate confidence
    kps["right_ankle"] = KeypointData(x=220, y=160, confidence=0.55, name="right_ankle", visible=True)

    chain = resolve_side_view_chain(kps, threshold=0.25)

    # Hip and knee must be resolved from visible left leg
    assert "hip" in chain
    assert chain["hip"].x == 250
    assert "knee" in chain
    assert chain["knee"].x == 380

    # Ankle must NOT be substituted with the opposite hallucinated right ankle
    assert "ankle" not in chain

    # Test processing and evaluator behavior
    processed = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)
    assert "mid_ankle" not in processed.midpoints
    assert "mid_ankle" in processed.missing_landmarks

    features = SideViewKeypointProcessor.extract_features(processed, threshold=0.25)
    # Hip extension is calculated; ankle-dependent knee extension is marked unreliable
    assert features.hip_extension_deg is not None
    assert "knee_extension" in features.unreliable_features
    assert features.knee_extension_deg is None


def test_case_b_competing_knees_similar_confidence_rejects_collapsed():
    """
    Case B: Both knees have similar confidence (delta_conf < 0.20).
    Visible knee is extended (ratio ~ 0.86), occluded knee is collapsed toward hip (ratio < 0.40).
    Verify: No midpoint averaging occurs; extended knee is selected.
    """
    kps = create_base_side_keypoints()
    # Left knee extended: dist from hip = 130px (130 / 150 = 0.867)
    kps["left_knee"] = KeypointData(x=380, y=95, confidence=0.75, name="left_knee", visible=True)
    # Right knee collapsed: dist from hip = 30px (30 / 150 = 0.20)
    kps["right_knee"] = KeypointData(x=280, y=105, confidence=0.70, name="right_knee", visible=True)
    # Confidence difference is 0.05 < 0.20

    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain
    # Must NOT be averaged to (380 + 280)/2 = 330
    assert chain["knee"].x == 380
    assert chain["knee"].y == 95

    processed = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)
    assert processed.midpoints["mid_knee"].x == 380


def test_case_c_both_knees_unreliable_marked_unresolved():
    """
    Case C: Both knees are below confidence threshold or anatomically collapsed.
    Verify: Knee is marked unresolved; downstream evaluator handles missing knee gracefully.
    """
    kps = create_base_side_keypoints()
    kps["left_knee"] = KeypointData(x=380, y=95, confidence=0.18, name="left_knee", visible=True)
    kps["right_knee"] = KeypointData(x=380, y=105, confidence=0.20, name="right_knee", visible=True)

    chain = resolve_side_view_chain(kps, threshold=0.25)
    assert "knee" not in chain

    processed = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)
    assert "mid_knee" not in processed.midpoints

    pose_kp = PoseKeypoints(keypoints=kps, bounding_box=[0, 0, 500, 200], detection_confidence=0.8)
    profile = BiomechanicsEvaluator.evaluate(pose_kp, side_view=True)

    # Downstream evaluator produces profile safely
    assert profile.side_view is True
    assert profile.side_view_features is not None
    assert profile.side_view_features.knee_extension_deg is None
    assert "knee_extension" in profile.side_view_features.unreliable_features
    assert profile.resolved_landmarks is not None
    assert "knee" not in profile.resolved_landmarks
    assert "mid_knee" in profile.missing_landmarks

    analyzer = PlancheAnalyzer()
    result = analyzer.analyze(profile, target_progression=ProgressionStage.FULL_PLANCHE)
    assert result is not None
    assert result.form_score >= 0.0


def test_case_d_renderer_evaluator_consistency_unresolved_joints():
    """
    Case D: Renderer/evaluator consistency.
    When ankle is unresolved, the rendered skeleton does not connect knee to ankle,
    and midpoints match evaluator's resolved landmarks.
    """
    kps = create_base_side_keypoints()
    kps["left_ankle"] = KeypointData(x=500, y=95, confidence=0.10, name="left_ankle", visible=True)
    kps["right_ankle"] = KeypointData(x=500, y=105, confidence=0.12, name="right_ankle", visible=True)

    pose_kp = PoseKeypoints(keypoints=kps, bounding_box=[0, 0, 500, 200], detection_confidence=0.85)
    profile = BiomechanicsEvaluator.evaluate(pose_kp, side_view=True)

    analyzer = PlancheAnalyzer()
    result = analyzer.analyze(profile, target_progression=ProgressionStage.FULL_PLANCHE)

    estimator = YOLOPoseEstimator()
    dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
    annotated = estimator.draw_skeleton(dummy_img, pose_kp, analysis=result)

    assert annotated is not None
    assert annotated.shape == (400, 600, 3)
    # Check debug info in result
    assert "resolved_landmarks" in result.debug_info
    assert "ankle" not in result.debug_info["resolved_landmarks"]
    assert "mid_ankle" not in result.debug_info.get("midpoints", {})


def test_case_e_standard_front_view_non_regression():
    """
    Case E: Non-side-view regression test.
    Verify standard evaluation (side_view=False) preserves full COCO connections and features.
    """
    kps = create_base_side_keypoints()
    pose_kp = PoseKeypoints(keypoints=kps, bounding_box=[0, 0, 500, 200], detection_confidence=0.9)

    profile_std = BiomechanicsEvaluator.evaluate(pose_kp, side_view=False)
    assert profile_std.side_view is False
    assert profile_std.midpoints is None
    assert profile_std.resolved_landmarks is None
    assert profile_std.joint_angles.primary_knee_deg is not None
    assert profile_std.joint_angles.hip_angle_deg is not None

    analyzer = PlancheAnalyzer()
    result = analyzer.analyze(profile_std, target_progression=ProgressionStage.FULL_PLANCHE)
    assert result.side_view is False
    assert result.form_score > 0

    estimator = YOLOPoseEstimator()
    dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
    annotated = estimator.draw_skeleton(dummy_img, pose_kp, analysis=result)
    assert annotated is not None


def test_case_f_temporal_stability_with_previous_chain():
    """
    Case F: Video temporal stability.
    Passing previous_chain prevents flip-flopping when opposite limb confidence fluctuates.
    """
    kps_f1 = create_base_side_keypoints()
    # Frame 1: Left leg has higher confidence (0.85 vs 0.40)
    chain_f1 = resolve_side_view_chain(kps_f1, threshold=0.25)
    assert chain_f1["knee"].x == 380
    assert chain_f1["knee"].y == 95

    # Frame 2: Right leg momentarily rises slightly in confidence (0.88 vs 0.85)
    kps_f2 = create_base_side_keypoints()
    kps_f2["left_knee"] = KeypointData(x=380, y=95, confidence=0.85, name="left_knee", visible=True)
    kps_f2["right_knee"] = KeypointData(x=375, y=105, confidence=0.88, name="right_knee", visible=True)

    # With previous_chain from Frame 1, temporal bias preserves the established left-side track
    chain_f2 = resolve_side_view_chain(kps_f2, threshold=0.25, previous_chain=chain_f1)
    assert chain_f2["knee"].y == 95  # Retained left knee
