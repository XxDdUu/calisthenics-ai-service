import logging
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_vertical_angle,
    get_point,
)
from app.exercises.base import BaseExerciseAnalyzer
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
)

logger = logging.getLogger(__name__)


class HandstandAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Handstand and Handstand Push-up (HSPU).
    Evaluates vertical stack, shoulder openness, elbow lockout, and hollow body line.
    Supports dynamic rep counting for Handstand Push-up.
    """

    @property
    def name(self) -> str:
        return "HANDSTAND"

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

        # Primary side detection
        l_conf = sum([keypoints[k].confidence for k in ["left_shoulder", "left_elbow", "left_wrist"] if k in keypoints])
        r_conf = sum([keypoints[k].confidence for k in ["right_shoulder", "right_elbow", "right_wrist"] if k in keypoints])
        side = "left" if l_conf >= r_conf else "right"

        # 1. Elbow Extension (Shoulder - Elbow - Wrist)
        l_elbow = calculate_angle_2d(l_sh, l_el, l_wr) if (l_sh and l_el and l_wr) else 180.0
        r_elbow = calculate_angle_2d(r_sh, r_el, r_wr) if (r_sh and r_el and r_wr) else 180.0
        primary_elbow = l_elbow if side == "left" else r_elbow

        # 2. Shoulder Angle / Openness (Wrist - Shoulder - Hip)
        p_wr = l_wr if side == "left" and l_wr else r_wr or l_wr
        p_sh = l_sh if side == "left" and l_sh else r_sh or l_sh
        p_hip = l_hip if side == "left" and l_hip else r_hip or l_hip
        shoulder_angle = calculate_angle_2d(p_wr, p_sh, p_hip) if (p_wr and p_sh and p_hip) else 180.0

        # 3. Body Line / Hip Extension (Shoulder - Hip - Leg Endpoint)
        p_kn = l_kn if side == "left" and l_kn else r_kn or l_kn
        p_ak = l_ak if side == "left" and l_ak else r_ak or l_ak
        p_leg = p_ak or p_kn
        body_line_angle = calculate_angle_2d(p_sh, p_hip, p_leg) if (p_sh and p_hip and p_leg) else 180.0

        # 4. Vertical Alignment (Wrist to Ankle/Hip vertical plumline deviation)
        p_top = p_ak or p_kn or p_hip
        vertical_tilt = calculate_vertical_angle(p_top, p_wr) if (p_top and p_wr) else 0.0

        return {
            "elbow_angle": round(primary_elbow, 1),
            "shoulder_angle": round(shoulder_angle, 1),
            "body_line_angle": round(body_line_angle, 1),
            "vertical_tilt": round(vertical_tilt, 1)
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        is_hspu = bool(progression and ("PUSHUP" in progression.upper() or "HSPU" in progression.upper()))
        elbow_angle = measurements.get("elbow_angle", 180.0)
        shoulder_angle = measurements.get("shoulder_angle", 180.0)
        body_line = measurements.get("body_line_angle", 180.0)
        vertical_tilt = measurements.get("vertical_tilt", 0.0)

        feedback: List[LiveFeedbackItem] = []
        is_valid_frame = True

        # --- Rule 1: Shoulder Openness (Ideal 170°-180°) ---
        if shoulder_angle >= 165.0:
            sh_score = 100.0
        elif shoulder_angle >= 150.0:
            sh_score = max(60.0, 100.0 - (165.0 - shoulder_angle) * 2.5)
            feedback.append(LiveFeedbackItem(
                code="OPEN_SHOULDERS",
                message="Open your shoulders completely and push tall through traps.",
                severity=FeedbackSeverity.WARNING
            ))
        else:
            sh_score = max(0.0, 60.0 - (150.0 - shoulder_angle) * 3.0)
            feedback.append(LiveFeedbackItem(
                code="OPEN_SHOULDERS",
                message="Shoulders are closed. Push the floor away.",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False

        # --- Rule 2: Body Line / Hollow Form (Ideal 168°-180°) ---
        if body_line >= 168.0:
            body_score = 100.0
        elif body_line >= 155.0:
            body_score = max(65.0, 100.0 - (168.0 - body_line) * 2.5)
            feedback.append(LiveFeedbackItem(
                code="TUCK_RIBS",
                message="Engage core and squeeze ribs in to avoid arched back.",
                severity=FeedbackSeverity.WARNING
            ))
        else:
            body_score = max(0.0, 65.0 - (155.0 - body_line) * 3.5)
            feedback.append(LiveFeedbackItem(
                code="TUCK_RIBS",
                message="Excessive banana arch. Tuck pelvis and squeeze glutes.",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False

        # --- Rule 3: Vertical Stack Deviation (Ideal <= 10°) ---
        if vertical_tilt <= 10.0:
            tilt_score = 100.0
        elif vertical_tilt <= 20.0:
            tilt_score = max(65.0, 100.0 - (vertical_tilt - 10.0) * 3.5)
            feedback.append(LiveFeedbackItem(
                code="VERTICAL_STACK",
                message="Stack shoulders and hips directly above your hands.",
                severity=FeedbackSeverity.WARNING
            ))
        else:
            tilt_score = max(0.0, 65.0 - (vertical_tilt - 20.0) * 3.0)
            feedback.append(LiveFeedbackItem(
                code="VERTICAL_STACK",
                message="Body is drifting away from vertical.",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False

        # --- Rule 4: Elbow Extension ---
        if not is_hspu:
            # Static Handstand hold: Elbows must remain locked
            if elbow_angle >= 168.0:
                elbow_score = 100.0
            elif elbow_angle >= 155.0:
                elbow_score = max(60.0, 100.0 - (168.0 - elbow_angle) * 3.0)
                feedback.append(LiveFeedbackItem(
                    code="LOCK_ELBOWS",
                    message="Lock your elbows out.",
                    severity=FeedbackSeverity.WARNING
                ))
            else:
                elbow_score = max(0.0, 60.0 - (155.0 - elbow_angle) * 3.5)
                feedback.append(LiveFeedbackItem(
                    code="LOCK_ELBOWS",
                    message="Arms are bent. Lock elbows completely.",
                    severity=FeedbackSeverity.ERROR
                ))
                is_valid_frame = False

            overall_score = (
                sh_score * 0.35 +
                body_score * 0.30 +
                tilt_score * 0.20 +
                elbow_score * 0.15
            )
        else:
            # HSPU: Dynamic movement
            elbow_score = 100.0
            overall_score = (
                sh_score * 0.35 +
                body_score * 0.35 +
                tilt_score * 0.30
            )

        overall_score = max(0.0, min(100.0, round(overall_score, 1)))

        if overall_score >= 80.0 and is_valid_frame and not feedback:
            feedback.append(LiveFeedbackItem(
                code="GOOD_FORM",
                message="Clean handstand alignment! Stay tall.",
                severity=FeedbackSeverity.INFO
            ))

        return overall_score, feedback, is_valid_frame

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        is_hspu = bool("PUSHUP" in session.exercise.upper() or "HSPU" in session.exercise.upper())
        if not is_hspu:
            return super().update_dynamic_state(session, is_valid_frame, measurements, timestamp_sec)

        # Dynamic HSPU repetition counting state machine
        elbow = measurements.get("elbow_angle", 180.0)
        current_phase = getattr(session, "rep_phase", "START") or "START"

        if current_phase in ("START", "TOP"):
            if elbow <= 135.0:
                session.rep_phase = "DESCENDING"
            else:
                session.rep_phase = "TOP"
        elif current_phase == "DESCENDING":
            if elbow <= 95.0:
                session.rep_phase = "BOTTOM"
            elif elbow >= 160.0:
                session.rep_phase = "TOP"
        elif current_phase == "BOTTOM":
            if elbow >= 125.0:
                session.rep_phase = "ASCENDING"
        elif current_phase == "ASCENDING":
            if elbow >= 168.0:
                session.rep_count = getattr(session, "rep_count", 0) + 1
                session.rep_phase = "TOP"

        return None, session.rep_count, session.rep_phase
