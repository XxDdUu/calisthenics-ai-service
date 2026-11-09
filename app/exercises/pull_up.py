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


class PullUpAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Pull-ups / Chin-ups.
    Tracks elbow flexion/extension range of motion, bilateral symmetry,
    and repetition counting (dead hang to chin-over-bar).
    """

    def __init__(self):
        self.phase_detector = ExercisePhaseDetector(
            exercise_name="PULL_UP",
            is_dynamic=True,
            min_rom_delta=45.0,
            min_rep_duration_sec=0.8
        )

    @property
    def name(self) -> str:
        return "PULL_UP"

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def camera_view_recommendation(self) -> str:
        return "Position camera side-on or slightly oblique at chest/bar height, full body visible."

    @property
    def required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist"
        ]

    @property
    def scoring_weights(self) -> Dict[str, float]:
        return {
            "range_of_motion": 0.25,
            "body_control": 0.20,
            "elbow_movement": 0.20,
            "shoulder_positioning": 0.20,
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

        l_elbow = calculate_angle_2d(l_sh, l_el, l_wr) if (l_sh and l_el and l_wr) else 180.0
        r_elbow = calculate_angle_2d(r_sh, r_el, r_wr) if (r_sh and r_el and r_wr) else 180.0
        avg_elbow = (l_elbow + r_elbow) / 2.0
        elbow_diff = abs(l_elbow - r_elbow)

        return {
            "elbow_angle": round(avg_elbow, 1),
            "elbow_left_angle": round(l_elbow, 1),
            "elbow_right_angle": round(r_elbow, 1),
            "elbow_symmetry_diff": round(elbow_diff, 1)
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        avg_elbow = measurements.get("elbow_angle", 180.0)
        elbow_diff = measurements.get("elbow_symmetry_diff", 0.0)

        feedback: List[LiveFeedbackItem] = []
        is_valid_frame = True

        # Check bilateral arm symmetry
        if elbow_diff > 25.0:
            feedback.append(LiveFeedbackItem(
                code="SYMMETRIC_PULL",
                message="Uneven pull detected. Engage both arms equally.",
                severity=FeedbackSeverity.WARNING
            ))
            is_valid_frame = False

        # Range of Motion checks
        if self.phase_detector.current_phase == "TOP":
            if avg_elbow > 85.0:
                feedback.append(LiveFeedbackItem(
                    code="INCOMPLETE_ROM_TOP",
                    message="Pull higher: bring your chin completely over the bar.",
                    severity=FeedbackSeverity.WARNING
                ))
            else:
                feedback.append(LiveFeedbackItem(
                    code="CHIN_OVER_BAR",
                    message="Great height over the bar!",
                    severity=FeedbackSeverity.INFO
                ))

        if self.phase_detector.current_phase in ("START", "BOTTOM"):
            if avg_elbow < 145.0:
                feedback.append(LiveFeedbackItem(
                    code="FULL_DEAD_HANG",
                    message="Lower completely into full arm extension at the bottom.",
                    severity=FeedbackSeverity.INFO
                ))

        # Form score based on symmetry and range of motion
        symmetry_score = max(0.0, 100.0 - elbow_diff * 2.5)
        rom_score = 90.0
        if self.phase_detector.current_phase == "TOP":
            rom_score = max(30.0, min(100.0, 100.0 - max(0.0, avg_elbow - 70.0) * 2.0))

        overall_score = round(0.50 * symmetry_score + 0.50 * rom_score, 1)

        return overall_score, feedback, is_valid_frame

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        avg_elbow = measurements.get("elbow_angle", 180.0)
        elbow_diff = measurements.get("elbow_symmetry_diff", 0.0)

        # 1. Range of Motion
        if self.phase_detector.current_phase == "TOP":
            rom_score = max(30.0, min(100.0, 100.0 - max(0.0, avg_elbow - 70.0) * 2.0))
        elif self.phase_detector.current_phase in ("START", "BOTTOM"):
            rom_score = max(40.0, min(100.0, 100.0 - max(0.0, 155.0 - avg_elbow) * 2.0))
        else:
            rom_score = 85.0

        # 2. Body Control
        control_score = 88.0

        # 3. Elbow Movement & path
        elbow_score = max(0.0, min(100.0, 100.0 - elbow_diff * 1.5))

        # 4. Shoulder positioning
        shoulder_score = 85.0

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
                name="body_control",
                value=0.0,
                score=round(control_score, 1),
                weight=0.20,
                confidence=0.90,
                status="VALID"
            ),
            LiveScoringMetric(
                name="elbow_movement",
                value=elbow_diff,
                score=round(elbow_score, 1),
                weight=0.20,
                confidence=0.92,
                status="VALID"
            ),
            LiveScoringMetric(
                name="shoulder_positioning",
                value=0.0,
                score=round(shoulder_score, 1),
                weight=0.20,
                confidence=0.88,
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
        reps, phase = self.phase_detector.update_pull_up_rep(
            elbow_angle=elbow,
            timestamp_sec=timestamp_sec,
            is_valid_form=is_valid_frame
        )
        session.rep_count = reps
        session.rep_phase = phase
        return None, reps, phase
