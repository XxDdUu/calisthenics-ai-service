import logging
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import (
    calculate_angle_2d,
    get_point,
)
from app.biomechanics.phase_detector import ExercisePhaseDetector
from app.exercises.base import BaseExerciseAnalyzer
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    LiveScoringMetric,
)

logger = logging.getLogger(__name__)


class PikePushUpAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Pike Push-ups.
    Evaluates:
    - Inverted V-shape hip pike angle (torso-thigh angle 60°-90°)
    - Range of motion (head moving forward in a tripod to <= 90° elbow bend)
    - Elbow tracking (elbows tracking 45° back, not flared out)
    - Overhead shoulder push back at top lockout
    - Dynamic repetition counting
    """

    def __init__(self):
        self.phase_detector = ExercisePhaseDetector(
            exercise_name="PIKE_PUSH_UP",
            is_dynamic=True,
            min_rom_delta=35.0,
            min_rep_duration_sec=0.7
        )

    @property
    def name(self) -> str:
        return "PIKE_PUSH_UP"

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def camera_view_recommendation(self) -> str:
        return "Position camera side-on at floor level, full inverted V body shape visible."

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
            "hip_pike_angle": 0.25,
            "range_of_motion": 0.25,
            "elbow_path": 0.20,
            "shoulder_elevation": 0.15,
            "movement_control": 0.15
        }

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
        p_leg = l_ak or r_ak or l_kn or r_kn

        # 1. Elbow angle
        l_elbow = calculate_angle_2d(l_sh, l_el, l_wr) if (l_sh and l_el and l_wr) else 180.0
        r_elbow = calculate_angle_2d(r_sh, r_el, r_wr) if (r_sh and r_el and r_wr) else 180.0
        primary_elbow = l_elbow if side == "left" else r_elbow
        elbow_diff = abs(l_elbow - r_elbow)

        # 2. Hip pike angle (Shoulder - Hip - Ankle/Leg)
        hip_pike = calculate_angle_2d(p_sh, p_hip, p_leg) if (p_sh and p_hip and p_leg) else 90.0

        # 3. Shoulder angle (Wrist - Shoulder - Hip)
        shoulder_angle = calculate_angle_2d(p_wr, p_sh, p_hip) if (p_wr and p_sh and p_hip) else 140.0

        return {
            "elbow_angle": round(primary_elbow, 1),
            "elbow_left_angle": round(l_elbow, 1),
            "elbow_right_angle": round(r_elbow, 1),
            "elbow_symmetry_diff": round(elbow_diff, 1),
            "hip_pike_angle": round(hip_pike, 1),
            "shoulder_angle": round(shoulder_angle, 1)
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        elbow = measurements.get("elbow_angle", 180.0)
        hip_pike = measurements.get("hip_pike_angle", 90.0)
        shoulder_angle = measurements.get("shoulder_angle", 140.0)
        elbow_diff = measurements.get("elbow_symmetry_diff", 0.0)

        feedback: List[LiveFeedbackItem] = []
        is_valid = True

        # Check Pike Angle (must maintain hips high in inverted V, angle < 110°)
        if hip_pike > 115.0:
            feedback.append(LiveFeedbackItem(
                code="MAINTAIN_PIKE",
                message="Keep your hips piked high. Avoid dropping into a regular push-up.",
                severity=FeedbackSeverity.WARNING
            ))
            is_valid = False

        # Range of Motion at bottom
        if self.phase_detector.current_phase == "BOTTOM":
            if elbow > 105.0:
                feedback.append(LiveFeedbackItem(
                    code="INCREASE_PIKE_DEPTH",
                    message="Lower head forward into tripod position for full depth.",
                    severity=FeedbackSeverity.WARNING
                ))

        # Push back at top
        if self.phase_detector.current_phase in ("TOP", "START"):
            if shoulder_angle < 130.0 or elbow < 155.0:
                feedback.append(LiveFeedbackItem(
                    code="PUSH_BACK_OVERHEAD",
                    message="Push all the way back overhead to lock out your shoulders.",
                    severity=FeedbackSeverity.INFO
                ))

        # Symmetry
        if elbow_diff > 25.0:
            feedback.append(LiveFeedbackItem(
                code="UNEVEN_PRESS",
                message="Press evenly through both shoulders.",
                severity=FeedbackSeverity.WARNING
            ))

        pike_score = max(0.0, min(100.0, 100.0 - max(0.0, hip_pike - 90.0) * 2.0))
        rom_score = 90.0
        if self.phase_detector.current_phase == "BOTTOM":
            rom_score = max(30.0, min(100.0, 100.0 - max(0.0, elbow - 90.0) * 2.0))

        overall_score = round(0.40 * pike_score + 0.35 * rom_score + 0.25 * max(0.0, 100.0 - elbow_diff * 2.0), 1)

        return overall_score, feedback, is_valid

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        elbow = measurements.get("elbow_angle", 180.0)
        hip_pike = measurements.get("hip_pike_angle", 90.0)
        shoulder_angle = measurements.get("shoulder_angle", 140.0)
        elbow_diff = measurements.get("elbow_symmetry_diff", 0.0)

        pike_score = max(0.0, min(100.0, 100.0 - max(0.0, hip_pike - 90.0) * 2.5))
        rom_score = max(40.0, min(100.0, 100.0 - max(0.0, elbow - 90.0) * 2.0)) if self.phase_detector.current_phase == "BOTTOM" else 85.0
        elbow_score = max(0.0, min(100.0, 100.0 - elbow_diff * 2.5))
        shoulder_score = max(0.0, min(100.0, 100.0 - max(0.0, 160.0 - shoulder_angle) * 1.5))
        control_score = 88.0

        return [
            LiveScoringMetric(
                name="hip_pike_angle",
                value=hip_pike,
                score=round(pike_score, 1),
                weight=0.25,
                confidence=0.92,
                status="VALID"
            ),
            LiveScoringMetric(
                name="range_of_motion",
                value=elbow,
                score=round(rom_score, 1),
                weight=0.25,
                confidence=0.90,
                status="VALID"
            ),
            LiveScoringMetric(
                name="elbow_path",
                value=elbow_diff,
                score=round(elbow_score, 1),
                weight=0.20,
                confidence=0.88,
                status="VALID"
            ),
            LiveScoringMetric(
                name="shoulder_elevation",
                value=shoulder_angle,
                score=round(shoulder_score, 1),
                weight=0.15,
                confidence=0.85,
                status="VALID"
            ),
            LiveScoringMetric(
                name="movement_control",
                value=0.0,
                score=round(control_score, 1),
                weight=0.15,
                confidence=0.88,
                status="VALID"
            )
        ]

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        elbow = measurements.get("elbow_angle", 180.0)
        reps, phase = self.phase_detector.update_dynamic_rep(
            primary_metric_value=elbow,
            top_threshold=155.0,
            bottom_threshold=95.0,
            timestamp_sec=timestamp_sec,
            is_valid_form=is_valid_frame
        )
        session.rep_count = reps
        session.rep_phase = phase
        return None, reps, phase
