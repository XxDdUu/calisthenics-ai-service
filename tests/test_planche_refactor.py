import pytest
from app.biomechanics.metrics import BiomechanicsEvaluator, BiomechanicsProfile
from app.exercises.planche import PlancheAnalyzer
from app.schemas.analysis import (
    AlignmentMetrics,
    AssessmentStatus,
    JointAngles,
    KeypointData,
    MetricStatus,
    PoseKeypoints,
    ProgressionStage,
)


def make_profile(
    elbow: float = 175.0,
    shoulder_lean: float = 20.0,
    hip_angle: float = 175.0,
    knee_angle: float = 175.0,
    body_tilt: float = 5.0,
    leg_spread_ratio: float = 0.3,
    side_view: bool = False,
    pose_quality_ok: bool = True
) -> BiomechanicsProfile:
    return BiomechanicsProfile(
        joint_angles=JointAngles(
            elbow_left_deg=elbow,
            elbow_right_deg=elbow,
            primary_elbow_deg=elbow,
            shoulder_lean_left_deg=shoulder_lean,
            shoulder_lean_right_deg=shoulder_lean,
            primary_shoulder_lean_deg=shoulder_lean,
            hip_angle_deg=hip_angle,
            knee_left_deg=knee_angle,
            knee_right_deg=knee_angle,
            primary_knee_deg=knee_angle,
            body_horizontal_tilt_deg=body_tilt,
            primary_side="left",
            leg_spread_ratio=leg_spread_ratio
        ),
        alignment=AlignmentMetrics(
            body_straightness_score=95.0,
            scapular_protraction_score=90.0,
            horizontal_level_score=95.0,
            center_of_mass_over_hands=True,
            center_of_mass_offset_px=10.0
        ),
        side_view=side_view,
        pose_quality_ok=pose_quality_ok
    )


def test_interpretable_component_scores():
    """Test that PlancheAnalyzer produces interpretable component scores."""
    profile = make_profile(elbow=175.0, shoulder_lean=18.0, hip_angle=170.0, knee_angle=175.0, body_tilt=4.0)
    result = PlancheAnalyzer.analyze(profile, target_progression=ProgressionStage.FULL_PLANCHE)

    assert result.component_scores is not None
    assert "elbow_extension" in result.component_scores
    assert "shoulder_position" in result.component_scores
    assert "hip_extension" in result.component_scores
    assert "knee_extension" in result.component_scores
    assert "body_line" in result.component_scores
    assert "overall" in result.component_scores

    assert result.component_scores["elbow_extension"] == 100.0
    assert result.component_scores["overall"] >= 90.0


def test_bent_elbow_reduces_component_and_overall_score():
    """
    Test that bent elbows directly decrease elbow component score and overall score.
    """
    straight_profile = make_profile(elbow=175.0)
    straight_result = PlancheAnalyzer.analyze(straight_profile, target_progression="FULL_PLANCHE")

    bent_profile = make_profile(elbow=150.0)
    bent_result = PlancheAnalyzer.analyze(bent_profile, target_progression="FULL_PLANCHE")

    assert bent_result.component_scores["elbow_extension"] < straight_result.component_scores["elbow_extension"]
    assert bent_result.form_score < straight_result.form_score
    assert any("elbow" in err.lower() for err in bent_result.form_errors)


def test_distinguish_insufficient_pose_quality_from_bad_form():
    """
    Test requirement 11:
    Poor pose quality returns INSUFFICIENT_POSE_QUALITY status instead of treating as bad form.
    """
    profile_poor_quality = make_profile(pose_quality_ok=False)
    result = PlancheAnalyzer.analyze(profile_poor_quality, target_progression="FULL_PLANCHE")

    assert result.assessment_status == AssessmentStatus.INSUFFICIENT_POSE_QUALITY
    assert result.is_valid_hold is False
    assert any(issue.metric == "pose_quality" for issue in result.structured_issues)


def test_side_view_analysis_flag_in_result():
    """Test side_view flag is correctly propagated in the result and metadata."""
    profile_side = make_profile(side_view=True)
    result = PlancheAnalyzer.analyze(profile_side, target_progression="FULL_PLANCHE", side_view=True)

    assert result.side_view is True
    assert result.debug_info["side_view"] is True

    profile_std = make_profile(side_view=False)
    result_std = PlancheAnalyzer.analyze(profile_std, target_progression="FULL_PLANCHE", side_view=False)
    assert result_std.side_view is False
