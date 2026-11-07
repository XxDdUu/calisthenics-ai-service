import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from app.biomechanics.metrics import BiomechanicsProfile
from app.exercises.config import (
    PLANCHE_CONFIG,
    PlancheProgressionConfig,
    get_planche_config,
)
from app.schemas.analysis import (
    AlignmentMetrics,
    AssessmentStatus,
    ExerciseType,
    FeedbackSeverity,
    IssueSeverity,
    JointAngles,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    MetricEvaluationDetail,
    MetricStatus,
    PlancheAnalysisResult,
    ProgressionStage,
    StructuredIssue,
    parse_progression_stage,
)
from app.exercises.base import BaseExerciseAnalyzer
from app.biomechanics.angles import compute_all_joint_angles, get_point

logger = logging.getLogger(__name__)


class PlancheAnalyzer(BaseExerciseAnalyzer):
    """
    Evaluates Planche form against a user-selected target progression stage.

    Deterministically assesses pose metrics, calculates scores, identifies form issues,
    and provides explainable coaching cues tailored specifically to the target progression.
    Supports both standard analysis and dedicated side-view midpoint analysis.
    """

    STRADDLE_LEG_SPREAD_THRESHOLD: float = float(os.getenv("STRADDLE_LEG_SPREAD_THRESHOLD", "1.25"))

    @classmethod
    def analyze(
        cls,
        profile: Optional[BiomechanicsProfile],
        target_progression: Optional[Union[str, ProgressionStage]] = None,
        side_view: Optional[bool] = None
    ) -> PlancheAnalysisResult:
        """
        Evaluates athlete pose biomechanics against the user's selected target progression.
        """
        target_stage = parse_progression_stage(target_progression)
        is_side_view = side_view if side_view is not None else (profile.side_view if profile else False)

        # 1. Handle Invalid Input (None profile)
        if profile is None or profile.joint_angles is None:
            return PlancheAnalysisResult(
                exercise_type=ExerciseType.PLANCHE,
                target_progression=target_stage,
                detected_progression=target_stage,
                side_view=is_side_view,
                assessment_status=AssessmentStatus.INVALID_INPUT,
                form_score=0.0,
                is_valid_hold=False,
                joint_angles=JointAngles(
                    elbow_left_deg=0.0,
                    elbow_right_deg=0.0,
                    primary_elbow_deg=0.0,
                    shoulder_lean_left_deg=0.0,
                    shoulder_lean_right_deg=0.0,
                    primary_shoulder_lean_deg=0.0,
                    hip_angle_deg=0.0,
                    knee_left_deg=0.0,
                    knee_right_deg=0.0,
                    primary_knee_deg=0.0,
                    body_horizontal_tilt_deg=0.0,
                    primary_side="left",
                    leg_spread_ratio=None
                ),
                alignment=AlignmentMetrics(
                    body_straightness_score=0.0,
                    scapular_protraction_score=0.0,
                    horizontal_level_score=0.0,
                    center_of_mass_over_hands=False,
                    center_of_mass_offset_px=0.0
                ),
                component_scores={
                    "body_line": 0.0,
                    "hip_extension": 0.0,
                    "knee_extension": 0.0,
                    "elbow_extension": 0.0,
                    "shoulder_position": 0.0,
                    "overall": 0.0
                },
                metrics={},
                structured_issues=[
                    StructuredIssue(
                        metric="pose_visibility",
                        severity=IssueSeverity.HIGH,
                        message="Insufficient or occluded keypoint data for form evaluation."
                    )
                ],
                form_errors=["Insufficient pose keypoint visibility."],
                positive_feedback=[],
                actionable_cues=["Ensure full body is visible in camera frame."],
                debug_info={"error": "INVALID_INPUT", "side_view": is_side_view}
            )

        # 2. Handle Poor Pose Quality (Distinguish bad form from bad pose data)
        if not profile.pose_quality_ok:
            missing_str = ", ".join(profile.missing_landmarks) if profile.missing_landmarks else "critical joints"
            return PlancheAnalysisResult(
                exercise_type=ExerciseType.PLANCHE,
                target_progression=target_stage,
                detected_progression=target_stage,
                side_view=is_side_view,
                assessment_status=AssessmentStatus.INSUFFICIENT_POSE_QUALITY,
                form_score=0.0,
                is_valid_hold=False,
                joint_angles=profile.joint_angles,
                alignment=profile.alignment,
                component_scores={
                    "body_line": 0.0,
                    "hip_extension": 0.0,
                    "knee_extension": 0.0,
                    "elbow_extension": 0.0,
                    "shoulder_position": 0.0,
                    "overall": 0.0
                },
                metrics={},
                structured_issues=[
                    StructuredIssue(
                        metric="pose_quality",
                        severity=IssueSeverity.HIGH,
                        message=f"Camera visibility too low on {missing_str} for accurate assessment."
                    )
                ],
                form_errors=[f"Low pose tracking confidence on {missing_str}."],
                positive_feedback=[],
                actionable_cues=["Position camera with clear line of sight, good lighting, and no occlusion."],
                debug_info={
                    "status": "INSUFFICIENT_POSE_QUALITY",
                    "missing_landmarks": profile.missing_landmarks,
                    "side_view": is_side_view
                }
            )

        angles = profile.joint_angles
        align = profile.alignment

        primary_elbow = angles.primary_elbow_deg
        primary_lean = angles.primary_shoulder_lean_deg
        hip_angle = angles.hip_angle_deg
        primary_knee = angles.primary_knee_deg
        body_tilt = angles.body_horizontal_tilt_deg
        leg_spread = angles.leg_spread_ratio

        # 3. Determine Input Assessment Status
        # Incomplete attempt if athlete lacks basic forward lean (< 8°) and has steep body tilt (> 40°),
        # or collapsed posture.
        if (primary_lean < 8.0 and body_tilt > 40.0) or (hip_angle < 85.0 and primary_knee < 60.0):
            assessment_status = AssessmentStatus.INCOMPLETE_ATTEMPT
        else:
            assessment_status = AssessmentStatus.VALID_ATTEMPT

        # 4. Progression-Specific Rule Evaluation using PLANCHE_CONFIG
        config = get_planche_config(target_stage)
        metric_evals, structured_issues, form_errors, positive_feedback, cues, component_scores = cls._evaluate_progression_rules(
            config=config,
            target_stage=target_stage,
            elbow_angle=primary_elbow,
            shoulder_lean=primary_lean,
            hip_angle=hip_angle,
            knee_angle=primary_knee,
            body_tilt=body_tilt,
            leg_spread_ratio=leg_spread,
            alignment=align,
            profile=profile
        )

        overall_score = component_scores.get("overall", 0.0)

        # 5. Valid Hold Criteria
        # Valid hold requires valid attempt, locked arms (>= 155°), forward lean (>= 10°), and overall score >= min_hold_score
        is_valid = (
            assessment_status == AssessmentStatus.VALID_ATTEMPT and
            primary_elbow >= config.elbow_extension.acceptable_min and
            primary_lean >= config.shoulder_position.acceptable_min and
            overall_score >= config.min_hold_score
        )

        # Debug metadata
        debug_data: Dict[str, Any] = {
            "side_view": is_side_view,
            "target_progression": target_stage.value,
            "component_scores": component_scores,
            "angles": {
                "primary_elbow": primary_elbow,
                "primary_lean": primary_lean,
                "hip_angle": hip_angle,
                "primary_knee": primary_knee,
                "body_tilt": body_tilt,
                "leg_spread_ratio": leg_spread
            }
        }
        if is_side_view:
            if profile.midpoints:
                debug_data["midpoints"] = {
                    k: {"x": v.x, "y": v.y, "confidence": v.confidence, "visible": v.visible}
                    for k, v in profile.midpoints.items()
                }
            if profile.resolved_landmarks:
                debug_data["resolved_landmarks"] = {
                    k: {"x": v.x, "y": v.y, "confidence": v.confidence, "visible": v.visible}
                    for k, v in profile.resolved_landmarks.items()
                }

        return PlancheAnalysisResult(
            exercise_type=ExerciseType.PLANCHE,
            target_progression=target_stage,
            detected_progression=target_stage,
            side_view=is_side_view,
            assessment_status=assessment_status,
            form_score=round(overall_score, 1),
            is_valid_hold=is_valid,
            joint_angles=angles,
            alignment=align,
            component_scores=component_scores,
            metrics=metric_evals,
            structured_issues=structured_issues,
            form_errors=form_errors,
            positive_feedback=positive_feedback,
            actionable_cues=cues,
            debug_info=debug_data
        )

    @classmethod
    def _evaluate_progression_rules(
        cls,
        config: PlancheProgressionConfig,
        target_stage: ProgressionStage,
        elbow_angle: float,
        shoulder_lean: float,
        hip_angle: float,
        knee_angle: float,
        body_tilt: float,
        leg_spread_ratio: Optional[float],
        alignment: AlignmentMetrics,
        profile: BiomechanicsProfile
    ) -> Tuple[Dict[str, MetricEvaluationDetail], List[StructuredIssue], List[str], List[str], List[str], Dict[str, float]]:
        """
        Evaluates measured biomechanical metrics deterministically against config for target_stage.
        """
        metrics: Dict[str, MetricEvaluationDetail] = {}
        issues: List[StructuredIssue] = []
        errors: List[str] = []
        positives: List[str] = []
        cues: List[str] = []

        # --- Rule 1: Elbow Extension ---
        elbow_rule = config.elbow_extension
        elbow_score, elbow_status = elbow_rule.evaluate_score(elbow_angle)

        if elbow_status == MetricStatus.PASS:
            positives.append("Excellent elbow lockout.")
        elif elbow_status == MetricStatus.NEEDS_IMPROVEMENT:
            errors.append(f"Slight elbow bend ({elbow_angle:.1f}° - target >= {elbow_rule.ideal_min:.0f}°)")
            cues.append("Focus on full elbow extension.")
            issues.append(StructuredIssue(
                metric="elbow_extension",
                severity=IssueSeverity.LOW,
                message=f"Slight elbow bend detected ({elbow_angle:.1f}°). Aim for full lockout >= {elbow_rule.ideal_min:.0f}°.",
                measured_value=elbow_angle,
                target_range=elbow_rule.target_range
            ))
        else:
            errors.append(f"Bent elbows ({elbow_angle:.1f}° - ideal >= {elbow_rule.ideal_min:.0f}°)")
            cues.append("Lock elbows completely; press hard through palms to maintain straight arms.")
            issues.append(StructuredIssue(
                metric="elbow_extension",
                severity=IssueSeverity.HIGH,
                message=f"Elbows are bent ({elbow_angle:.1f}°). Lock arms completely.",
                measured_value=elbow_angle,
                target_range=elbow_rule.target_range
            ))

        metrics["elbow_extension"] = MetricEvaluationDetail(
            name=elbow_rule.name,
            measured_value=elbow_angle,
            target_range=elbow_rule.target_range,
            score=round(elbow_score, 1),
            status=elbow_status
        )

        # --- Rule 2: Shoulder Forward Lean ---
        lean_rule = config.shoulder_position
        lean_score, lean_status = lean_rule.evaluate_score(shoulder_lean)

        if lean_status in (MetricStatus.PASS, MetricStatus.GOOD):
            positives.append("Solid forward shoulder lean.")
        elif lean_status == MetricStatus.NEEDS_IMPROVEMENT:
            errors.append(f"Insufficient forward lean ({shoulder_lean:.1f}° - target >= {lean_rule.ideal_min:.0f}°)")
            cues.append("Lean shoulders further forward past wrists to shift center of gravity.")
            issues.append(StructuredIssue(
                metric="shoulder_lean",
                severity=IssueSeverity.MEDIUM,
                message=f"Shoulder lean is slightly low ({shoulder_lean:.1f}°). Lean forward past wrists (target >= {lean_rule.ideal_min:.0f}°).",
                measured_value=shoulder_lean,
                target_range=lean_rule.target_range
            ))
        else:
            errors.append(f"Poor forward lean ({shoulder_lean:.1f}° - target >= {lean_rule.ideal_min:.0f}°)")
            cues.append("Significant forward lean required. Lean shoulders past hands.")
            issues.append(StructuredIssue(
                metric="shoulder_lean",
                severity=IssueSeverity.HIGH,
                message=f"Insufficient forward shoulder lean ({shoulder_lean:.1f}°). Lean shoulders past wrist base.",
                measured_value=shoulder_lean,
                target_range=lean_rule.target_range
            ))

        metrics["shoulder_lean"] = MetricEvaluationDetail(
            name=lean_rule.name,
            measured_value=shoulder_lean,
            target_range=lean_rule.target_range,
            score=round(lean_score, 1),
            status=lean_status
        )

        # --- Rule 3: Body Horizontal Tilt / Body Line ---
        tilt_rule = config.body_line
        tilt_score, tilt_status = tilt_rule.evaluate_score(body_tilt)

        if tilt_status in (MetricStatus.PASS, MetricStatus.GOOD):
            positives.append("Torso parallel to ground.")
        elif tilt_status == MetricStatus.NEEDS_IMPROVEMENT:
            errors.append(f"Torso inclined ({body_tilt:.1f}° off horizontal)")
            cues.append("Lower hips and legs so torso is parallel to ground.")
            issues.append(StructuredIssue(
                metric="body_tilt",
                severity=IssueSeverity.MEDIUM,
                message=f"Torso is inclined ({body_tilt:.1f}°). Bring body parallel to ground (target <= {tilt_rule.ideal_max:.0f}°).",
                measured_value=body_tilt,
                target_range=tilt_rule.target_range
            ))
        else:
            errors.append(f"High torso tilt ({body_tilt:.1f}°)")
            cues.append("Lower body to horizontal plane.")
            issues.append(StructuredIssue(
                metric="body_tilt",
                severity=IssueSeverity.HIGH,
                message=f"Torso is heavily inclined ({body_tilt:.1f}°).",
                measured_value=body_tilt,
                target_range=tilt_rule.target_range
            ))

        metrics["body_tilt"] = MetricEvaluationDetail(
            name=tilt_rule.name,
            measured_value=body_tilt,
            target_range=tilt_rule.target_range,
            score=round(tilt_score, 1),
            status=tilt_status
        )

        # --- Rule 4: Hip Extension / Angle ---
        hip_rule = config.hip_extension
        hip_score, hip_status = hip_rule.evaluate_score(hip_angle)

        if target_stage == ProgressionStage.TUCK_PLANCHE:
            if hip_status in (MetricStatus.PASS, MetricStatus.GOOD):
                positives.append("Compact hip tuck position.")
            else:
                errors.append(f"Hips opening ({hip_angle:.1f}° - target <= {hip_rule.ideal_max:.0f}°)")
                cues.append("Keep knees and hips tucked tightly to chest.")
                issues.append(StructuredIssue(
                    metric="hip_angle",
                    severity=IssueSeverity.MEDIUM if hip_score >= 60 else IssueSeverity.HIGH,
                    message=f"Hips are opening too much for Tuck Planche ({hip_angle:.1f}°). Keep hips tucked to chest.",
                    measured_value=hip_angle,
                    target_range=hip_rule.target_range
                ))

        elif target_stage == ProgressionStage.ADV_TUCK_PLANCHE:
            if hip_status in (MetricStatus.PASS, MetricStatus.GOOD):
                positives.append("Great 90° flat-back hip position.")
            elif hip_angle < (hip_rule.ideal_min or 90.0):
                errors.append(f"Hips too tucked ({hip_angle:.1f}° - target 90°–130°)")
                cues.append("Open hips outward to create a 90° angle with flat back.")
                issues.append(StructuredIssue(
                    metric="hip_angle",
                    severity=IssueSeverity.MEDIUM,
                    message=f"Hips are too tucked ({hip_angle:.1f}°). Open hips outward to 90°.",
                    measured_value=hip_angle,
                    target_range=hip_rule.target_range
                ))
            else:
                errors.append(f"Hips opening past 90° ({hip_angle:.1f}°)")
                cues.append("Maintain a 90° flat-back hip angle.")
                issues.append(StructuredIssue(
                    metric="hip_angle",
                    severity=IssueSeverity.MEDIUM if hip_score >= 60 else IssueSeverity.HIGH,
                    message=f"Hips are extending past 90° ({hip_angle:.1f}°). Maintain a 90° flat back position.",
                    measured_value=hip_angle,
                    target_range=hip_rule.target_range
                ))

        elif target_stage == ProgressionStage.STRADDLE_PLANCHE:
            if hip_status == MetricStatus.PASS:
                positives.append("Great straight hip extension.")
            elif hip_status == MetricStatus.GOOD:
                positives.append("Solid hip extension.")
            else:
                errors.append(f"Piking at hips ({hip_angle:.1f}° - target >= {hip_rule.acceptable_min:.0f}°)")
                cues.append("Squeeze glutes and extend hips into a flat line.")
                issues.append(StructuredIssue(
                    metric="hip_angle",
                    severity=IssueSeverity.HIGH,
                    message=f"Piking at hips ({hip_angle:.1f}°). Extend hips into a flat line.",
                    measured_value=hip_angle,
                    target_range=hip_rule.target_range
                ))

        else:  # FULL_PLANCHE
            if hip_status == MetricStatus.PASS:
                positives.append("Great straight body line.")
            elif hip_status == MetricStatus.GOOD:
                positives.append("Solid body extension.")
            else:
                errors.append(f"Piking at hips ({hip_angle:.1f}° - target >= {hip_rule.acceptable_min:.0f}°)")
                cues.append("Squeeze glutes and extend hips down into a flat, straight line.")
                issues.append(StructuredIssue(
                    metric="hip_angle",
                    severity=IssueSeverity.HIGH,
                    message=f"Piking at hips ({hip_angle:.1f}°). Extend hips fully into a straight line.",
                    measured_value=hip_angle,
                    target_range=hip_rule.target_range
                ))

        metrics["hip_angle"] = MetricEvaluationDetail(
            name=hip_rule.name,
            measured_value=hip_angle,
            target_range=hip_rule.target_range,
            score=round(hip_score, 1),
            status=hip_status
        )

        # --- Rule 5: Knee Extension / Tuck ---
        knee_rule = config.knee_extension
        knee_score, knee_status = knee_rule.evaluate_score(knee_angle)

        if target_stage == ProgressionStage.TUCK_PLANCHE:
            if knee_status in (MetricStatus.PASS, MetricStatus.GOOD):
                positives.append("Knees tucked into chest.")
            else:
                errors.append(f"Knees extending ({knee_angle:.1f}° - target <= {knee_rule.ideal_max:.0f}°)")
                cues.append("Maintain a compact knee tuck.")
                issues.append(StructuredIssue(
                    metric="knee_angle",
                    severity=IssueSeverity.MEDIUM if knee_score >= 60 else IssueSeverity.HIGH,
                    message=f"Knees are extending ({knee_angle:.1f}°). Pull knees tight to chest.",
                    measured_value=knee_angle,
                    target_range=knee_rule.target_range
                ))

        elif target_stage == ProgressionStage.ADV_TUCK_PLANCHE:
            if knee_status in (MetricStatus.PASS, MetricStatus.GOOD):
                positives.append("Bent knees ~90° for Advanced Tuck.")
            elif knee_angle > (knee_rule.ideal_max or 135.0):
                errors.append(f"Knees extending too far ({knee_angle:.1f}° - target <= {knee_rule.ideal_max:.0f}°)")
                cues.append("Keep knees bent ~90° for Advanced Tuck.")
                issues.append(StructuredIssue(
                    metric="knee_angle",
                    severity=IssueSeverity.MEDIUM,
                    message=f"Knees are extending too far ({knee_angle:.1f}°). Keep knees bent ~90°.",
                    measured_value=knee_angle,
                    target_range=knee_rule.target_range
                ))
            else:
                errors.append(f"Knees tucked too tight ({knee_angle:.1f}°)")
                cues.append("Open knee angle to ~90°.")
                issues.append(StructuredIssue(
                    metric="knee_angle",
                    severity=IssueSeverity.LOW,
                    message=f"Knees are tucked too tight ({knee_angle:.1f}°). Open knee angle to ~90°.",
                    measured_value=knee_angle,
                    target_range=knee_rule.target_range
                ))

        elif target_stage == ProgressionStage.STRADDLE_PLANCHE:
            if knee_status == MetricStatus.PASS:
                positives.append("Fully extended straight legs.")
            else:
                errors.append(f"Bent knees ({knee_angle:.1f}° - target >= {knee_rule.acceptable_min:.0f}°)")
                cues.append("Keep legs fully extended.")
                issues.append(StructuredIssue(
                    metric="knee_angle",
                    severity=IssueSeverity.HIGH,
                    message=f"Knees are bent ({knee_angle:.1f}°). Lock knees completely.",
                    measured_value=knee_angle,
                    target_range=knee_rule.target_range
                ))

        else:  # FULL_PLANCHE
            if knee_status == MetricStatus.PASS:
                positives.append("Fully extended straight legs.")
            else:
                errors.append(f"Bent knees ({knee_angle:.1f}° - target >= {knee_rule.acceptable_min:.0f}°)")
                cues.append("Lock knees completely for Full Planche.")
                issues.append(StructuredIssue(
                    metric="knee_angle",
                    severity=IssueSeverity.HIGH,
                    message=f"Knees are bent ({knee_angle:.1f}°). Lock knees completely.",
                    measured_value=knee_angle,
                    target_range=knee_rule.target_range
                ))

        metrics["knee_angle"] = MetricEvaluationDetail(
            name=knee_rule.name,
            measured_value=knee_angle,
            target_range=knee_rule.target_range,
            score=round(knee_score, 1),
            status=knee_status
        )

        # --- Rule 6: Leg Spread (Straddle vs Full) ---
        if target_stage == ProgressionStage.STRADDLE_PLANCHE:
            actual_ratio = leg_spread_ratio if leg_spread_ratio is not None else 1.5
            leg_target = ">= 1.25"
            if actual_ratio >= cls.STRADDLE_LEG_SPREAD_THRESHOLD:
                leg_score = 100.0
                leg_status = MetricStatus.PASS
                positives.append("Wide leg straddle.")
            else:
                leg_score = max(0.0, 100.0 - (cls.STRADDLE_LEG_SPREAD_THRESHOLD - actual_ratio) * 60.0)
                leg_status = MetricStatus.NEEDS_IMPROVEMENT if leg_score >= 60 else MetricStatus.FAIL
                errors.append(f"Legs not spread wide enough ({actual_ratio:.2f} - target >= 1.25)")
                cues.append("Spread legs wider apart for Straddle Planche.")
                issues.append(StructuredIssue(
                    metric="leg_spread",
                    severity=IssueSeverity.MEDIUM,
                    message=f"Legs are not spread wide enough for Straddle Planche ({actual_ratio:.2f}). Spread legs wider.",
                    measured_value=actual_ratio,
                    target_range=leg_target
                ))

            metrics["leg_spread"] = MetricEvaluationDetail(
                name="Leg Straddle Spread",
                measured_value=actual_ratio,
                target_range=leg_target,
                score=round(leg_score, 1),
                status=leg_status
            )

        elif target_stage == ProgressionStage.FULL_PLANCHE:
            actual_ratio = leg_spread_ratio if leg_spread_ratio is not None else 0.4
            leg_target = "< 1.25"
            if actual_ratio < cls.STRADDLE_LEG_SPREAD_THRESHOLD:
                leg_score = 100.0
                leg_status = MetricStatus.PASS
                positives.append("Legs together for Full Planche.")
            else:
                leg_score = max(0.0, 100.0 - (actual_ratio - cls.STRADDLE_LEG_SPREAD_THRESHOLD) * 50.0)
                leg_status = MetricStatus.NEEDS_IMPROVEMENT if leg_score >= 60 else MetricStatus.FAIL
                errors.append(f"Legs spread apart ({actual_ratio:.2f} - target < 1.25)")
                cues.append("Squeeze legs together for Full Planche.")
                issues.append(StructuredIssue(
                    metric="leg_spread",
                    severity=IssueSeverity.MEDIUM,
                    message=f"Legs are spread apart ({actual_ratio:.2f}). Squeeze legs together for Full Planche.",
                    measured_value=actual_ratio,
                    target_range=leg_target
                ))

            metrics["leg_spread"] = MetricEvaluationDetail(
                name="Leg Adduction (Legs Together)",
                measured_value=actual_ratio,
                target_range=leg_target,
                score=round(leg_score, 1),
                status=leg_status
            )

        # --- Rule 7: Scapular Protraction & Balance ---
        if alignment.scapular_protraction_score >= 70.0:
            positives.append("Good scapular protraction.")
        else:
            errors.append("Lacking scapular protraction")
            cues.append("Push shoulders down and round upper back (protract scapulae).")

        if not alignment.center_of_mass_over_hands:
            cues.append("Adjust forward lean so center of mass rests over hand base of support.")

        # --- Weighted Overall Score Calculation ---
        overall_score = (
            elbow_score * config.elbow_extension.weight +
            lean_score * config.shoulder_position.weight +
            hip_score * config.hip_extension.weight +
            knee_score * config.knee_extension.weight +
            tilt_score * config.body_line.weight
        )
        overall_score = max(0.0, min(100.0, overall_score))

        component_scores = {
            "elbow_extension": round(elbow_score, 1),
            "shoulder_position": round(lean_score, 1),
            "hip_extension": round(hip_score, 1),
            "knee_extension": round(knee_score, 1),
            "body_line": round(tilt_score, 1),
            "overall": round(overall_score, 1)
        }

        return metrics, issues, errors, positives, cues, component_scores

    @property
    def name(self) -> str:
        return "PLANCHE"

    @property
    def required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip"
        ]

    def extract_measurements(
        self,
        keypoints: Dict[str, KeypointData],
        side_view: bool = False
    ) -> Dict[str, float]:
        """Extract planche joint angles for live analysis."""
        angles = compute_all_joint_angles(keypoints)
        return {
            "elbow_angle": round(angles.primary_elbow_deg, 1),
            "shoulder_lean_angle": round(angles.primary_shoulder_lean_deg, 1),
            "hip_angle": round(angles.hip_angle_deg, 1),
            "knee_angle": round(angles.primary_knee_deg, 1),
            "body_line_angle": round(angles.body_horizontal_tilt_deg, 1),
            "leg_spread_ratio": round(angles.leg_spread_ratio, 2) if angles.leg_spread_ratio is not None else 1.0
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        """Evaluate planche live measurements against progression rules."""
        config = get_planche_config(progression or "FULL_PLANCHE")
        elbow = measurements.get("elbow_angle", 180.0)
        lean = measurements.get("shoulder_lean_angle", 0.0)
        hip = measurements.get("hip_angle", 180.0)
        knee = measurements.get("knee_angle", 180.0)
        tilt = measurements.get("body_line_angle", 0.0)
        spread = measurements.get("leg_spread_ratio", 1.0)

        feedback: List[LiveFeedbackItem] = []
        is_valid_frame = True

        elbow_score, elbow_status = config.elbow_extension.evaluate_score(elbow)
        lean_score, lean_status = config.shoulder_position.evaluate_score(lean)
        hip_score, hip_status = config.hip_extension.evaluate_score(hip)
        knee_score, knee_status = config.knee_extension.evaluate_score(knee)
        tilt_score, tilt_status = config.body_line.evaluate_score(tilt)

        # Check issues
        if elbow_status == MetricStatus.FAIL:
            feedback.append(LiveFeedbackItem(
                code="EXTEND_ELBOWS",
                message="Extend elbows fully (>= 170°).",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False
        elif elbow_status == MetricStatus.NEEDS_IMPROVEMENT:
            feedback.append(LiveFeedbackItem(
                code="EXTEND_ELBOWS",
                message="Straighten arms into full lockout.",
                severity=FeedbackSeverity.WARNING
            ))

        if lean_status == MetricStatus.FAIL:
            feedback.append(LiveFeedbackItem(
                code="LEAN_FORWARD",
                message="Lean further forward over your wrists.",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False
        elif lean_status == MetricStatus.NEEDS_IMPROVEMENT:
            feedback.append(LiveFeedbackItem(
                code="LEAN_FORWARD",
                message="Increase forward shoulder lean.",
                severity=FeedbackSeverity.WARNING
            ))

        if hip_status == MetricStatus.FAIL:
            feedback.append(LiveFeedbackItem(
                code="ALIGN_HIPS",
                message=f"Adjust hips for {config.name}.",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False
        elif hip_status == MetricStatus.NEEDS_IMPROVEMENT:
            feedback.append(LiveFeedbackItem(
                code="ALIGN_HIPS",
                message="Keep hips at proper progression angle.",
                severity=FeedbackSeverity.WARNING
            ))

        if tilt_status == MetricStatus.FAIL:
            feedback.append(LiveFeedbackItem(
                code="ALIGN_BODY",
                message="Body line is sagging or too high. Level out with the ground.",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False

        # Overall score
        overall_score = (
            elbow_score * config.elbow_extension.weight +
            lean_score * config.shoulder_position.weight +
            hip_score * config.hip_extension.weight +
            knee_score * config.knee_extension.weight +
            tilt_score * config.body_line.weight
        )
        overall_score = max(0.0, min(100.0, round(overall_score, 1)))

        if overall_score >= 80.0 and is_valid_frame and not feedback:
            feedback.append(LiveFeedbackItem(
                code="GOOD_FORM",
                message=f"Excellent {config.name} hold! Stay locked.",
                severity=FeedbackSeverity.INFO
            ))

        return overall_score, feedback, is_valid_frame

