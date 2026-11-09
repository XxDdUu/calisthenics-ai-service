import logging
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_body_line_deviation,
    calculate_horizontal_angle,
    calculate_leg_spread_ratio,
    get_point,
)
from app.exercises.base import BaseExerciseAnalyzer
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    LiveScoringMetric,
)

logger = logging.getLogger(__name__)


class FrontLeverAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Front Lever and its progression stages:
    - FULL_FRONT_LEVER
    - STRADDLE_FRONT_LEVER
    - ADV_TUCK_FRONT_LEVER
    - TUCK_FRONT_LEVER
    """

    @property
    def name(self) -> str:
        return "FRONT_LEVER"

    @property
    def required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee"
        ]

    @property
    def scoring_weights(self) -> Dict[str, float]:
        return {
            "elbow_extension": 0.25,
            "shoulder_angle": 0.25,
            "body_line": 0.20,
            "horizontal_tilt": 0.15,
            "stability": 0.15
        }

    @property
    def camera_view_recommendation(self) -> str:
        return "Position camera side-on at bar height, full horizontal body line visible."

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        elbow = measurements.get("elbow_angle", 180.0)
        shoulder = measurements.get("shoulder_angle", 45.0)
        body_line = measurements.get("body_line_angle", 180.0)
        tilt = measurements.get("horizontal_tilt", 0.0)

        elbow_score = max(0.0, min(100.0, 100.0 - max(0.0, 175.0 - elbow) * 3.0))
        shoulder_score = max(0.0, min(100.0, 100.0 - abs(45.0 - shoulder) * 1.5))
        line_score = max(0.0, min(100.0, 100.0 - abs(180.0 - body_line) * 2.5))
        tilt_score = max(0.0, min(100.0, 100.0 - tilt * 3.0))
        stability_score = 88.0

        return [
            LiveScoringMetric(
                name="elbow_extension",
                value=elbow,
                score=round(elbow_score, 1),
                weight=0.25,
                confidence=0.94,
                status="VALID"
            ),
            LiveScoringMetric(
                name="shoulder_angle",
                value=shoulder,
                score=round(shoulder_score, 1),
                weight=0.25,
                confidence=0.91,
                status="VALID"
            ),
            LiveScoringMetric(
                name="body_line",
                value=body_line,
                score=round(line_score, 1),
                weight=0.20,
                confidence=0.92,
                status="VALID"
            ),
            LiveScoringMetric(
                name="horizontal_tilt",
                value=tilt,
                score=round(tilt_score, 1),
                weight=0.15,
                confidence=0.90,
                status="VALID"
            ),
            LiveScoringMetric(
                name="stability",
                value=0.0,
                score=round(stability_score, 1),
                weight=0.15,
                confidence=0.88,
                status="VALID"
            )
        ]

    def validate_pose(
        self,
        keypoints: Dict[str, KeypointData],
        threshold: float = 0.25,
        side_view: bool = False
    ) -> Tuple[bool, LiveAnalysisStatus, List[str]]:
        if not keypoints:
            return False, LiveAnalysisStatus.NO_PERSON_DETECTED, ["all_keypoints"]

        # In side view or profile view, at least one arm and one torso side must be clear
        if side_view:
            has_sh = bool(keypoints.get("left_shoulder") or keypoints.get("right_shoulder"))
            has_el = bool(keypoints.get("left_elbow") or keypoints.get("right_elbow"))
            has_wr = bool(keypoints.get("left_wrist") or keypoints.get("right_wrist"))
            has_hip = bool(keypoints.get("left_hip") or keypoints.get("right_hip"))
            has_knee = bool(keypoints.get("left_knee") or keypoints.get("right_knee"))
            missing = []
            if not has_sh: missing.append("shoulders")
            if not has_el: missing.append("elbows")
            if not has_wr: missing.append("wrists")
            if not has_hip: missing.append("hips")
            if not has_knee: missing.append("knees")
            if missing:
                return False, LiveAnalysisStatus.INCOMPLETE_POSE, missing
            return True, LiveAnalysisStatus.VALID, []

        return super().validate_pose(keypoints, threshold, side_view)

    def extract_measurements(
        self,
        keypoints: Dict[str, KeypointData],
        side_view: bool = False
    ) -> Dict[str, float]:
        l_sh = get_point(keypoints.get("left_shoulder"))
        r_sh = get_point(keypoints.get("right_shoulder"))
        l_el = get_point(keypoints.get("left_elbow"))
        r_el = get_point(keypoints.get("right_elbow"))
        l_wr = get_point(keypoints.get("left_wrist"))
        r_wr = get_point(keypoints.get("right_wrist"))
        l_hip = get_point(keypoints.get("left_hip"))
        r_hip = get_point(keypoints.get("right_hip"))
        l_kn = get_point(keypoints.get("left_knee"))
        r_kn = get_point(keypoints.get("right_knee"))
        l_ak = get_point(keypoints.get("left_ankle"))
        r_ak = get_point(keypoints.get("right_ankle"))

        # Determine primary side
        l_conf = sum([keypoints[k].confidence for k in ["left_shoulder", "left_elbow", "left_wrist", "left_hip"] if k in keypoints])
        r_conf = sum([keypoints[k].confidence for k in ["right_shoulder", "right_elbow", "right_wrist", "right_hip"] if k in keypoints])
        side = "left" if l_conf >= r_conf else "right"

        # 1. Elbow Extension
        l_elbow = calculate_angle_2d(l_sh, l_el, l_wr) if (l_sh and l_el and l_wr) else 180.0
        r_elbow = calculate_angle_2d(r_sh, r_el, r_wr) if (r_sh and r_el and r_wr) else 180.0
        primary_elbow = l_elbow if side == "left" else r_elbow

        # 2. Shoulder Angle (Wrist - Shoulder - Hip)
        p_sh = l_sh if side == "left" and l_sh else r_sh or l_sh
        p_wr = l_wr if side == "left" and l_wr else r_wr or l_wr
        p_hip = l_hip if side == "left" and l_hip else r_hip or l_hip
        shoulder_angle = calculate_angle_2d(p_wr, p_sh, p_hip) if (p_wr and p_sh and p_hip) else 80.0

        # 3. Hip / Body line Angle & Deviation
        p_kn = l_kn if side == "left" and l_kn else r_kn or l_kn
        p_ak = l_ak if side == "left" and l_ak else r_ak or l_ak
        p_leg = p_ak or p_kn

        hip_angle = 180.0
        body_line_angle = 180.0
        if p_sh and p_hip and p_leg:
            hip_angle = calculate_angle_2d(p_sh, p_hip, p_leg)
            body_line_angle, _ = calculate_body_line_deviation(p_sh, p_hip, p_leg)

        # 4. Knee Angle
        l_knee = calculate_angle_2d(l_hip, l_kn, l_ak) if (l_hip and l_kn and l_ak) else 180.0
        r_knee = calculate_angle_2d(r_hip, r_kn, r_ak) if (r_hip and r_kn and r_ak) else 180.0
        primary_knee = l_knee if side == "left" else r_knee

        # 5. Horizontal Tilt (Shoulder to Hip angle relative to ground)
        horizontal_tilt = calculate_horizontal_angle(p_sh, p_hip) if (p_sh and p_hip) else 0.0

        # 6. Leg spread ratio
        spread_ratio = calculate_leg_spread_ratio(keypoints)

        return {
            "elbow_angle": round(primary_elbow, 1),
            "shoulder_angle": round(shoulder_angle, 1),
            "hip_angle": round(hip_angle, 1),
            "body_line_angle": round(body_line_angle, 1),
            "knee_angle": round(primary_knee, 1),
            "horizontal_tilt": round(horizontal_tilt, 1),
            "leg_spread_ratio": round(spread_ratio, 2) if spread_ratio is not None else 1.0
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        prog = (progression or "FULL_FRONT_LEVER").upper().replace(" ", "_").replace("-", "_")
        elbow_angle = measurements.get("elbow_angle", 180.0)
        hip_angle = measurements.get("hip_angle", 180.0)
        body_line_angle = measurements.get("body_line_angle", 180.0)
        knee_angle = measurements.get("knee_angle", 180.0)
        horiz_tilt = measurements.get("horizontal_tilt", 0.0)
        spread_ratio = measurements.get("leg_spread_ratio", 1.0)

        feedback: List[LiveFeedbackItem] = []
        is_valid_frame = True

        # --- Rule 1: Elbow Extension (Locked Arms >= 170°) ---
        if elbow_angle >= 170.0:
            elbow_score = 100.0
        elif elbow_angle >= 155.0:
            elbow_score = 65.0 + ((elbow_angle - 155.0) / 15.0) * 35.0
            feedback.append(LiveFeedbackItem(
                code="EXTEND_ELBOWS",
                message="Extend your elbows fully into lockout.",
                severity=FeedbackSeverity.WARNING
            ))
        else:
            elbow_score = max(0.0, 65.0 - (155.0 - elbow_angle) * 3.0)
            feedback.append(LiveFeedbackItem(
                code="EXTEND_ELBOWS",
                message="Arms are bent. Lock elbows completely.",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False

        # --- Rule 2: Body Tilt (Horizontal Alignment <= 15°) ---
        if horiz_tilt <= 15.0:
            tilt_score = 100.0
        elif horiz_tilt <= 25.0:
            tilt_score = max(60.0, 100.0 - (horiz_tilt - 15.0) * 4.0)
            feedback.append(LiveFeedbackItem(
                code="HORIZONTAL_BODY",
                message="Keep your body horizontal to the ground.",
                severity=FeedbackSeverity.WARNING
            ))
        else:
            tilt_score = max(0.0, 60.0 - (horiz_tilt - 25.0) * 3.0)
            feedback.append(LiveFeedbackItem(
                code="HORIZONTAL_BODY",
                message="Body is too vertical. Pull down harder to level out.",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False

        # --- Rule 3 & 4: Progression-Specific Hip & Knee Geometry ---
        if "TUCK" in prog and "ADV" not in prog and "ADVANCED" not in prog:
            # Tuck Front Lever: Knees tucked to chest (hip <= 100°, knee <= 90°)
            hip_score = 100.0 if hip_angle <= 100.0 else max(0.0, 100.0 - (hip_angle - 100.0) * 2.5)
            knee_score = 100.0 if knee_angle <= 90.0 else max(0.0, 100.0 - (knee_angle - 90.0) * 2.5)
            if hip_angle > 115.0:
                feedback.append(LiveFeedbackItem(
                    code="TUCK_HIPS",
                    message="Tuck knees closer to chest for Tuck Front Lever.",
                    severity=FeedbackSeverity.WARNING
                ))
        elif "ADV" in prog or "ADVANCED" in prog:
            # Advanced Tuck: 90° hips, flat back (hip 90°-130°, knee ~90°)
            if 85.0 <= hip_angle <= 130.0:
                hip_score = 100.0
            else:
                diff = min(abs(hip_angle - 85.0), abs(hip_angle - 130.0))
                hip_score = max(0.0, 100.0 - diff * 3.0)
                feedback.append(LiveFeedbackItem(
                    code="FLAT_BACK_TUCK",
                    message="Maintain a 90° hip angle with flat back.",
                    severity=FeedbackSeverity.WARNING
                ))
            knee_score = 100.0 if 70.0 <= knee_angle <= 120.0 else max(0.0, 100.0 - abs(knee_angle - 90.0) * 2.0)
        elif "STRADDLE" in prog:
            # Straddle Front Lever: Hip >= 160°, Knee >= 160°, leg spread >= 1.25
            hip_score = 100.0 if hip_angle >= 160.0 else max(0.0, 100.0 - (160.0 - hip_angle) * 3.0)
            knee_score = 100.0 if knee_angle >= 160.0 else max(0.0, 100.0 - (160.0 - knee_angle) * 3.0)
            if hip_angle < 155.0:
                feedback.append(LiveFeedbackItem(
                    code="LIFT_HIPS",
                    message="Hips are sagging. Squeeze glutes and lift hips horizontal.",
                    severity=FeedbackSeverity.WARNING
                ))
                is_valid_frame = False
            if spread_ratio < 1.20:
                feedback.append(LiveFeedbackItem(
                    code="SPREAD_LEGS",
                    message="Spread your legs wider for Straddle Front Lever.",
                    severity=FeedbackSeverity.WARNING
                ))
        else:
            # Full Front Lever: Straight line (hip >= 165°, knee >= 165°)
            if hip_angle >= 168.0:
                hip_score = 100.0
            elif hip_angle >= 150.0:
                hip_score = max(60.0, 100.0 - (168.0 - hip_angle) * 2.5)
                feedback.append(LiveFeedbackItem(
                    code="LIFT_HIPS",
                    message="Straighten body line. Avoid hip sag.",
                    severity=FeedbackSeverity.WARNING
                ))
            else:
                hip_score = max(0.0, 60.0 - (150.0 - hip_angle) * 3.5)
                feedback.append(LiveFeedbackItem(
                    code="LIFT_HIPS",
                    message="Severe hip sag. Raise hips to horizontal.",
                    severity=FeedbackSeverity.ERROR
                ))
                is_valid_frame = False

            if knee_angle >= 165.0:
                knee_score = 100.0
            else:
                knee_score = max(0.0, 100.0 - (165.0 - knee_angle) * 3.0)
                feedback.append(LiveFeedbackItem(
                    code="STRAIGHTEN_KNEES",
                    message="Lock your knees straight.",
                    severity=FeedbackSeverity.WARNING
                ))

        # Overall weighted score
        overall_score = (
            elbow_score * 0.30 +
            hip_score * 0.30 +
            tilt_score * 0.20 +
            knee_score * 0.20
        )
        overall_score = max(0.0, min(100.0, round(overall_score, 1)))

        if overall_score >= 80.0 and is_valid_frame and not feedback:
            feedback.append(LiveFeedbackItem(
                code="GOOD_FORM",
                message="Solid Front Lever hold! Maintain body tension.",
                severity=FeedbackSeverity.INFO
            ))

        return overall_score, feedback, is_valid_frame
