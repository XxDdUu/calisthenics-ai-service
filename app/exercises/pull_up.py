import logging
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import (
    calculate_angle_2d,
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


class PullUpAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Pull-ups / Chin-ups.
    Tracks elbow flexion/extension range of motion, bilateral symmetry,
    and repetition counting (dead hang to chin-over-bar).
    """

    @property
    def name(self) -> str:
        return "PULL_UP"

    @property
    def required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist"
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

        # Form score based on symmetry and smooth control
        symmetry_score = max(0.0, 100.0 - elbow_diff * 2.5)
        overall_score = round(symmetry_score, 1)

        return overall_score, feedback, is_valid_frame

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        elbow = measurements.get("elbow_angle", 180.0)
        current_phase = getattr(session, "rep_phase", "START") or "START"

        if current_phase in ("START", "BOTTOM"):
            if elbow <= 140.0:
                session.rep_phase = "PULLING"
            else:
                session.rep_phase = "BOTTOM"
        elif current_phase == "PULLING":
            if elbow <= 65.0:
                session.rep_phase = "TOP"
            elif elbow >= 155.0:
                session.rep_phase = "BOTTOM"
        elif current_phase == "TOP":
            if elbow >= 90.0:
                session.rep_phase = "LOWERING"
        elif current_phase == "LOWERING":
            if elbow >= 160.0:
                session.rep_count = getattr(session, "rep_count", 0) + 1
                session.rep_phase = "BOTTOM"

        return None, session.rep_count, session.rep_phase
