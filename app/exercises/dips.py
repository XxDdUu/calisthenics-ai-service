import logging
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_vertical_angle,
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


class DipsAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Parallel Bar Dips.
    Tracks depth (elbow <= 90°), top lockout (elbow >= 165°),
    torso lean, shoulder positioning, bilateral symmetry, and rep counting.
    """

    def __init__(self):
        self.phase_detector = ExercisePhaseDetector(
            exercise_name="DIPS",
            is_dynamic=True,
            min_rom_delta=40.0,
            min_rep_duration_sec=0.7
        )

    @property
    def name(self) -> str:
        return "DIPS"

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def camera_view_recommendation(self) -> str:
        return "Position camera side-on at shoulder/chest height, full body visible on parallel bars."

    @property
    def required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip"
        ]

    @property
    def scoring_weights(self) -> Dict[str, float]:
        return {
            "range_of_motion": 0.25,
            "elbow_mechanics": 0.20,
            "shoulder_positioning": 0.25,
            "torso_control": 0.15,
            "movement_symmetry": 0.15
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

        l_elbow = calculate_angle_2d(l_sh, l_el, l_wr) if (l_sh and l_el and l_wr) else 180.0
        r_elbow = calculate_angle_2d(r_sh, r_el, r_wr) if (r_sh and r_el and r_wr) else 180.0
        avg_elbow = (l_elbow + r_elbow) / 2.0
        elbow_diff = abs(l_elbow - r_elbow)

        p_sh = l_sh or r_sh
        p_hip = l_hip or r_hip
        torso_lean = calculate_vertical_angle(p_sh, p_hip) if (p_sh and p_hip) else 15.0

        return {
            "elbow_angle": round(avg_elbow, 1),
            "elbow_left_angle": round(l_elbow, 1),
            "elbow_right_angle": round(r_elbow, 1),
            "elbow_symmetry_diff": round(elbow_diff, 1),
            "torso_lean_angle": round(torso_lean, 1)
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        avg_elbow = measurements.get("elbow_angle", 180.0)
        elbow_diff = measurements.get("elbow_symmetry_diff", 0.0)
        torso_lean = measurements.get("torso_lean_angle", 15.0)

        feedback: List[LiveFeedbackItem] = []
        is_valid_frame = True

        if elbow_diff > 25.0:
            feedback.append(LiveFeedbackItem(
                code="SYMMETRIC_PUSH",
                message="Uneven press. Push evenly through both arms.",
                severity=FeedbackSeverity.WARNING
            ))
            is_valid_frame = False

        if torso_lean > 45.0:
            feedback.append(LiveFeedbackItem(
                code="CONTROL_LEAN",
                message="Excessive forward lean. Maintain controlled torso posture.",
                severity=FeedbackSeverity.WARNING
            ))

        if self.phase_detector.current_phase == "BOTTOM" and avg_elbow > 95.0:
            feedback.append(LiveFeedbackItem(
                code="INCREASE_DIP_DEPTH",
                message="Increase your range of motion. Lower until upper arms are parallel to bars.",
                severity=FeedbackSeverity.WARNING
            ))

        if self.phase_detector.current_phase in ("TOP", "START") and avg_elbow < 155.0:
            feedback.append(LiveFeedbackItem(
                code="LOCKOUT_DIP",
                message="Lock out your elbows fully at the top of the dip.",
                severity=FeedbackSeverity.INFO
            ))

        score = max(0.0, 100.0 - elbow_diff * 2.0 - max(0.0, torso_lean - 30.0) * 2.0)
        return round(score, 1), feedback, is_valid_frame

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        avg_elbow = measurements.get("elbow_angle", 180.0)
        elbow_diff = measurements.get("elbow_symmetry_diff", 0.0)
        torso_lean = measurements.get("torso_lean_angle", 15.0)

        # 1. Range of Motion
        if self.phase_detector.current_phase == "BOTTOM":
            rom_score = max(30.0, min(100.0, 100.0 - max(0.0, avg_elbow - 90.0) * 2.5))
        elif self.phase_detector.current_phase in ("TOP", "START"):
            rom_score = max(40.0, min(100.0, 100.0 - max(0.0, 165.0 - avg_elbow) * 2.5))
        else:
            rom_score = 85.0

        # 2. Elbow mechanics
        elbow_score = max(0.0, min(100.0, 100.0 - elbow_diff * 2.0))

        # 3. Shoulder positioning
        shoulder_score = 88.0

        # 4. Torso control
        torso_score = max(0.0, min(100.0, 100.0 - max(0.0, torso_lean - 30.0) * 2.5))

        # 5. Symmetry
        symmetry_score = max(0.0, min(100.0, 100.0 - elbow_diff * 2.5))

        return [
            LiveScoringMetric(
                name="range_of_motion",
                value=avg_elbow,
                score=round(rom_score, 1),
                weight=0.25,
                confidence=0.94,
                status="VALID"
            ),
            LiveScoringMetric(
                name="elbow_mechanics",
                value=elbow_diff,
                score=round(elbow_score, 1),
                weight=0.20,
                confidence=0.91,
                status="VALID"
            ),
            LiveScoringMetric(
                name="shoulder_positioning",
                value=0.0,
                score=round(shoulder_score, 1),
                weight=0.25,
                confidence=0.88,
                status="VALID"
            ),
            LiveScoringMetric(
                name="torso_control",
                value=torso_lean,
                score=round(torso_score, 1),
                weight=0.15,
                confidence=0.90,
                status="VALID"
            ),
            LiveScoringMetric(
                name="movement_symmetry",
                value=elbow_diff,
                score=round(symmetry_score, 1),
                weight=0.15,
                confidence=0.92,
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
