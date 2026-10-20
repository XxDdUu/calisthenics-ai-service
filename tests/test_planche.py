import pytest
from app.biomechanics.metrics import BiomechanicsProfile
from app.exercises.planche import PlancheAnalyzer
from app.schemas.analysis import (
    AlignmentMetrics,
    AssessmentStatus,
    JointAngles,
    MetricStatus,
    ProgressionStage,
)


def make_profile(
    elbow: float = 175.0,
    shoulder_lean: float = 20.0,
    hip_angle: float = 175.0,
    knee_angle: float = 175.0,
    body_tilt: float = 5.0,
    leg_spread_ratio: float = 0.3
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
        )
    )


def test_user_selected_adv_tuck_planche_valid():
    """Test user selects Advanced Tuck Planche with valid form (flat back ~110°, bent knees ~90°)."""
    profile = make_profile(hip_angle=110.0, knee_angle=90.0, leg_spread_ratio=0.5)
    result = PlancheAnalyzer.analyze(profile, target_progression=ProgressionStage.ADV_TUCK_PLANCHE)

    assert result.target_progression == ProgressionStage.ADV_TUCK_PLANCHE
    assert result.assessment_status == AssessmentStatus.VALID_ATTEMPT
    assert result.is_valid_hold is True
    assert result.form_score >= 85.0
    assert "knee_angle" in result.metrics
    assert result.metrics["knee_angle"].status in [MetricStatus.PASS, MetricStatus.GOOD]


def test_user_selected_adv_tuck_planche_knees_extending_failure():
    """Test user selects Advanced Tuck Planche, but knees extend to 137° (violates <= 135° rule)."""
    profile = make_profile(hip_angle=108.0, knee_angle=137.0, leg_spread_ratio=0.5)
    result = PlancheAnalyzer.analyze(profile, target_progression="ADV_TUCK_PLANCHE")

    assert result.target_progression == ProgressionStage.ADV_TUCK_PLANCHE
    assert "knee_angle" in result.metrics
    assert result.metrics["knee_angle"].status in [MetricStatus.NEEDS_IMPROVEMENT, MetricStatus.FAIL]

    # Explainable issue must be generated specifically for knee_angle
    knee_issues = [issue for issue in result.structured_issues if issue.metric == "knee_angle"]
    assert len(knee_issues) > 0
    assert "extending too far" in knee_issues[0].message.lower()


def test_user_selected_tuck_planche():
    """Test user selects Tuck Planche with knees tucked to chest (knee 80°, hip 90°)."""
    profile = make_profile(hip_angle=90.0, knee_angle=80.0, leg_spread_ratio=0.2)
    result = PlancheAnalyzer.analyze(profile, target_progression="Tuck Planche")

    assert result.target_progression == ProgressionStage.TUCK_PLANCHE
    assert result.is_valid_hold is True
    assert result.metrics["hip_angle"].status in [MetricStatus.PASS, MetricStatus.GOOD]
    assert result.metrics["knee_angle"].status in [MetricStatus.PASS, MetricStatus.GOOD]


def test_user_selected_straddle_planche():
    """Test user selects Straddle Planche with legs spread wide (ratio 1.85)."""
    profile = make_profile(hip_angle=170.0, knee_angle=175.0, leg_spread_ratio=1.85)
    result = PlancheAnalyzer.analyze(profile, target_progression="Straddle Planche")

    assert result.target_progression == ProgressionStage.STRADDLE_PLANCHE
    assert result.is_valid_hold is True
    assert result.metrics["leg_spread"].status in [MetricStatus.PASS, MetricStatus.GOOD]


def test_user_selected_full_planche_with_hip_145_range():
    """Test user selects Full Planche with hip angle 145° and legs together (ratio 0.35) -> evaluates as valid Full Planche."""
    profile = make_profile(hip_angle=145.0, knee_angle=170.0, leg_spread_ratio=0.35)
    result = PlancheAnalyzer.analyze(profile, target_progression="Full Planche")

    assert result.target_progression == ProgressionStage.FULL_PLANCHE
    assert result.is_valid_hold is True
    assert result.metrics["leg_spread"].status == MetricStatus.PASS


def test_incomplete_attempt_handling():
    """Test incomplete attempt with low forward lean (5°) and high body tilt (45°)."""
    profile = make_profile(shoulder_lean=5.0, body_tilt=45.0, hip_angle=170.0)
    result = PlancheAnalyzer.analyze(profile, target_progression="Full Planche")

    assert result.assessment_status == AssessmentStatus.INCOMPLETE_ATTEMPT
    assert result.is_valid_hold is False


def test_invalid_input_handling():
    """Test invalid input with None profile returns INVALID_INPUT status without misleading score."""
    result = PlancheAnalyzer.analyze(None, target_progression="ADV_TUCK_PLANCHE")

    assert result.assessment_status == AssessmentStatus.INVALID_INPUT
    assert result.is_valid_hold is False
    assert result.form_score == 0.0
    assert len(result.structured_issues) > 0
    assert result.structured_issues[0].metric == "pose_visibility"
