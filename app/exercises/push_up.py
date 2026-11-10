import logging
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_body_line_deviation,
    get_point,
)
from app.biomechanics.phase_detector import ExercisePhaseDetector
from app.biomechanics.push_up_counter import PushUpRepCounter
from app.exercises.base import BaseExerciseAnalyzer
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    LiveScoringMetric,
)

logger = logging.getLogger(__name__)


class PushUpAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Push-ups.
    Evaluates:
    - Body alignment (plank line: shoulder-hip-ankle 180° ± 15°)
    - Range of motion (depth: elbow <= 90° at bottom, lockout: elbow >= 160° at top)
    - Elbow movement & flare
    - Shoulder positioning & scapular control
    - Repetition counting and phase detection
    """

    def __init__(self):
        self._fallback_counter = PushUpRepCounter()
        self.phase_detector = ExercisePhaseDetector(
            exercise_name="PUSH_UP",
            is_dynamic=True,
            min_rom_delta=40.0,
            min_rep_duration_sec=0.7
        )

    @property
    def name(self) -> str:
        return "PUSH_UP"

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def camera_view_recommendation(self) -> str:
        return "Position camera side-on at floor level or 45° angle, full body visible from hands to feet."

    @property
    def required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_ankle", "right_ankle"
        ]

    @property
    def scoring_weights(self) -> Dict[str, float]:
        return {
            "body_alignment": 0.25,
            "range_of_motion": 0.25,
            "elbow_movement": 0.20,
            "shoulder_positioning": 0.15,
            "movement_control": 0.15
        }

    def validate_pose(
        self,
        keypoints: Dict[str, KeypointData],
        threshold: float = 0.25,
        side_view: bool = False
    ) -> Tuple[bool, LiveAnalysisStatus, List[str]]:
        if not keypoints:
            return False, LiveAnalysisStatus.NO_PERSON_DETECTED, ["all_keypoints"]

        # Side view: at least one arm and one torso/leg chain must be visible
        has_sh = bool(keypoints.get("left_shoulder") or keypoints.get("right_shoulder"))
        has_el = bool(keypoints.get("left_elbow") or keypoints.get("right_elbow"))
        has_wr = bool(keypoints.get("left_wrist") or keypoints.get("right_wrist"))
        has_hip = bool(keypoints.get("left_hip") or keypoints.get("right_hip"))
        has_leg = bool(
            keypoints.get("left_ankle") or keypoints.get("right_ankle") or
            keypoints.get("left_knee") or keypoints.get("right_knee")
        )

        missing = []
        if not has_sh: missing.append("shoulder")
        if not has_el: missing.append("elbow")
        if not has_wr: missing.append("wrist")
        if not has_hip: missing.append("hip")
        if not has_leg: missing.append("legs/ankles")

        if missing:
            return False, LiveAnalysisStatus.INCOMPLETE_POSE, missing

        return True, LiveAnalysisStatus.VALID, []

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
        l_ak = get_point(keypoints.get("left_ankle"))
        r_ak = get_point(keypoints.get("right_ankle"))
        l_kn = get_point(keypoints.get("left_knee"))
        r_kn = get_point(keypoints.get("right_knee"))

        # Determine dominant visible side
        l_conf = sum([keypoints[k].confidence for k in ["left_shoulder", "left_elbow", "left_wrist"] if k in keypoints])
        r_conf = sum([keypoints[k].confidence for k in ["right_shoulder", "right_elbow", "right_wrist"] if k in keypoints])
        side = "left" if l_conf >= r_conf else "right"

        p_sh = l_sh if side == "left" and l_sh else r_sh or l_sh
        p_el = l_el if side == "left" and l_el else r_el or l_el
        p_wr = l_wr if side == "left" and l_wr else r_wr or l_wr
        p_hip = l_hip if side == "left" and l_hip else r_hip or l_hip
        p_ak = l_ak if side == "left" and l_ak else r_ak or l_ak or l_kn or r_kn

        # 1. Elbow angle (flexion/extension)
        l_elbow = calculate_angle_2d(l_sh, l_el, l_wr) if (l_sh and l_el and l_wr) else 180.0
        r_elbow = calculate_angle_2d(r_sh, r_el, r_wr) if (r_sh and r_el and r_wr) else 180.0
        primary_elbow = l_elbow if side == "left" else r_elbow
        elbow_diff = abs(l_elbow - r_elbow)

        # 2. Body alignment angle (Shoulder - Hip - Ankle)
        body_line, alignment_type = calculate_body_line_deviation(p_sh, p_hip, p_ak)

        # 3. Shoulder-to-wrist vertical alignment / forward lean
        dx_sh_wr = abs(p_sh[0] - p_wr[0]) if (p_sh and p_wr) else 0.0

        return {
            "elbow_angle": round(primary_elbow, 1),
            "elbow_left_angle": round(l_elbow, 1),
            "elbow_right_angle": round(r_elbow, 1),
            "elbow_symmetry_diff": round(elbow_diff, 1),
            "body_line_angle": round(body_line, 1),
            "hip_sag_indicator": 1.0 if alignment_type == "SAG" else 0.0,
            "hip_pike_indicator": 1.0 if alignment_type == "PIKE" else 0.0,
            "shoulder_offset_px": round(dx_sh_wr, 1)
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        elbow = measurements.get("elbow_angle", 180.0)
        body_line = measurements.get("body_line_angle", 180.0)
        is_sag = measurements.get("hip_sag_indicator", 0.0) > 0.5
        is_pike = measurements.get("hip_pike_indicator", 0.0) > 0.5
        elbow_diff = measurements.get("elbow_symmetry_diff", 0.0)

        feedback: List[LiveFeedbackItem] = []
        is_valid = True

        # Check Body Alignment
        if is_sag and body_line < 165.0:
            feedback.append(LiveFeedbackItem(
                code="HIP_SAG",
                message="Avoid letting your hips sag. Engage your core and glutes.",
                severity=FeedbackSeverity.WARNING
            ))
            is_valid = False
        elif is_pike and body_line < 165.0:
            feedback.append(LiveFeedbackItem(
                code="HIP_PIKE",
                message="Keep your hips aligned with your shoulders. Lower your hips.",
                severity=FeedbackSeverity.WARNING
            ))
            is_valid = False

        # Range of Motion checks
        if elbow > 105.0 and self.phase_detector.current_phase == "BOTTOM":
            feedback.append(LiveFeedbackItem(
                code="INSUFFICIENT_DEPTH",
                message="Lower your chest further for full range of motion.",
                severity=FeedbackSeverity.WARNING
            ))

        if elbow < 155.0 and self.phase_detector.current_phase in ("TOP", "START"):
            feedback.append(LiveFeedbackItem(
                code="INCOMPLETE_LOCKOUT",
                message="Extend your elbows fully at the top of each rep.",
                severity=FeedbackSeverity.INFO
            ))

        # Arm symmetry check
        if elbow_diff > 25.0:
            feedback.append(LiveFeedbackItem(
                code="UNEVEN_PRESS",
                message="Press evenly through both arms.",
                severity=FeedbackSeverity.WARNING
            ))

        # Calculate composite score
        alignment_score = max(0.0, 100.0 - abs(180.0 - body_line) * 2.0)
        symmetry_score = max(0.0, 100.0 - elbow_diff * 2.5)
        rom_score = 90.0
        if self.phase_detector.current_phase == "BOTTOM":
            rom_score = max(40.0, min(100.0, 100.0 - max(0.0, elbow - 90.0) * 2.0))

        overall_score = round(0.40 * alignment_score + 0.35 * rom_score + 0.25 * symmetry_score, 1)

        if not feedback:
            feedback.append(LiveFeedbackItem(
                code="GOOD_FORM",
                message="Great body line and depth. Keep pressing strong!",
                severity=FeedbackSeverity.INFO
            ))

        return overall_score, feedback, is_valid

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        elbow = measurements.get("elbow_angle")
        body_line = measurements.get("body_line_angle")
        elbow_diff = measurements.get("elbow_symmetry_diff")

        metrics: List[LiveScoringMetric] = []

        # 1. Body Alignment
        if body_line is not None:
            align_score = max(0.0, min(100.0, 100.0 - abs(180.0 - body_line) * 2.0))
            metrics.append(LiveScoringMetric(
                name="body_alignment",
                value=round(body_line, 1),
                score=round(align_score, 1),
                weight=0.35,
                confidence=0.95,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="body_alignment", value=0.0, score=0.0, weight=0.35, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 2. Range of Motion
        if elbow is not None:
            if self.phase_detector.current_phase == "BOTTOM":
                rom_score = max(20.0, min(100.0, 100.0 - max(0.0, elbow - 90.0) * 2.0))
            elif self.phase_detector.current_phase in ("TOP", "START"):
                rom_score = max(40.0, min(100.0, 100.0 - max(0.0, 165.0 - elbow) * 2.0))
            else:
                # Transitional angle progress score (full depth 90 -> 100, shallow -> proportional)
                rom_score = max(40.0, min(100.0, 100.0 - max(0.0, elbow - 95.0) * 0.6))
            metrics.append(LiveScoringMetric(
                name="range_of_motion",
                value=round(elbow, 1),
                score=round(rom_score, 1),
                weight=0.35,
                confidence=0.92,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="range_of_motion", value=0.0, score=0.0, weight=0.35, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 3. Elbow Positioning & Symmetry
        if elbow_diff is not None:
            elbow_score = max(0.0, min(100.0, 100.0 - elbow_diff * 2.0))
            metrics.append(LiveScoringMetric(
                name="elbow_movement",
                value=round(elbow_diff, 1),
                score=round(elbow_score, 1),
                weight=0.30,
                confidence=0.90 if not side_view else 0.70,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="elbow_movement", value=0.0, score=0.0, weight=0.30, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 4. Shoulder Positioning (not reliably measurable without scapular landmarks on single 2D frame)
        metrics.append(LiveScoringMetric(
            name="shoulder_positioning",
            value=0.0,
            score=0.0,
            weight=0.0,
            confidence=0.0,
            status="INSUFFICIENT_DATA"
        ))

        # 5. Movement Control / Tempo Consistency (evaluated at session level across completed reps)
        metrics.append(LiveScoringMetric(
            name="movement_control",
            value=0.0,
            score=0.0,
            weight=0.0,
            confidence=0.0,
            status="INSUFFICIENT_DATA"
        ))

        return metrics

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        if hasattr(session, "get_push_up_counter"):
            counter = session.get_push_up_counter()
        elif hasattr(session, "push_up_counter") and session.push_up_counter is not None:
            counter = session.push_up_counter
        else:
            counter = self._fallback_counter

        if not is_valid_frame:
            rep_res = counter.process_angle(None, timestamp_sec=timestamp_sec)
        else:
            elbow = measurements.get("elbow_angle", 180.0)
            body_line = measurements.get("body_line_angle", 180.0)
            rep_res = counter.process_angle(elbow, body_alignment=body_line, timestamp_sec=timestamp_sec)

        if hasattr(session, "rep_count"):
            session.rep_count = rep_res.rep_count
        if hasattr(session, "rep_phase"):
            session.rep_phase = rep_res.current_state

        return None, rep_res.rep_count, rep_res.current_state
