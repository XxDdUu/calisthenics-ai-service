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


class MuscleUpAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Bar/Ring Muscle-Ups.
    Evaluates explosive pull phase, bilateral transition symmetry
    (detects chicken winging), and top dip lockout.
    """

    @property
    def name(self) -> str:
        return "MUSCLE_UP"

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

        l_elbow = calculate_angle_2d(l_sh, l_el, l_wr) if (l_sh and l_el and l_wr) else 180.0
        r_elbow = calculate_angle_2d(r_sh, r_el, r_wr) if (r_sh and r_el and r_wr) else 180.0
        avg_elbow = (l_elbow + r_elbow) / 2.0
        elbow_diff = abs(l_elbow - r_elbow)

        # Height comparison between shoulders and wrists:
        # Image Y grows downward, so sh_y < wr_y means shoulders are ABOVE wrists (over the bar!)
        sh_y = (l_sh[1] + r_sh[1]) / 2.0 if (l_sh and r_sh) else 0.0
        wr_y = (l_wr[1] + r_wr[1]) / 2.0 if (l_wr and r_wr) else 0.0
        chest_over_bar = bool(sh_y < wr_y)

        return {
            "elbow_angle": round(avg_elbow, 1),
            "elbow_left_angle": round(l_elbow, 1),
            "elbow_right_angle": round(r_elbow, 1),
            "transition_asymmetry": round(elbow_diff, 1),
            "chest_above_bar": 1.0 if chest_over_bar else 0.0
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        asymmetry = measurements.get("transition_asymmetry", 0.0)

        feedback: List[LiveFeedbackItem] = []
        is_valid_frame = True

        if asymmetry > 30.0:
            feedback.append(LiveFeedbackItem(
                code="TRANSITION_EVENLY",
                message="Chicken wing transition detected. Drive both elbows over together.",
                severity=FeedbackSeverity.ERROR
            ))
            is_valid_frame = False
        elif asymmetry > 18.0:
            feedback.append(LiveFeedbackItem(
                code="TRANSITION_EVENLY",
                message="Slight arm asymmetry during transition.",
                severity=FeedbackSeverity.WARNING
            ))

        score = max(0.0, 100.0 - asymmetry * 2.5)
        return round(score, 1), feedback, is_valid_frame

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        elbow = measurements.get("elbow_angle", 180.0)
        chest_above = measurements.get("chest_above_bar", 0.0) > 0.5
        current_phase = getattr(session, "rep_phase", "START") or "START"

        if current_phase in ("START", "HANG"):
            if elbow <= 135.0:
                session.rep_phase = "PULL"
            else:
                session.rep_phase = "HANG"
        elif current_phase == "PULL":
            if chest_above:
                session.rep_phase = "TRANSITION"
            elif elbow >= 160.0:
                session.rep_phase = "HANG"
        elif current_phase == "TRANSITION":
            if elbow >= 165.0:
                session.rep_count = getattr(session, "rep_count", 0) + 1
                session.rep_phase = "LOCKOUT"
        elif current_phase == "LOCKOUT":
            if not chest_above or elbow <= 120.0:
                session.rep_phase = "HANG"

        return None, session.rep_count, session.rep_phase
