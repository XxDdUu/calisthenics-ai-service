import abc
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import get_point, is_valid_point
from app.biomechanics.confidence import is_valid_keypoint
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    LiveScoringMetric,
)


class BaseExerciseAnalyzer(abc.ABC):
    """
    Abstract base class for all exercise analyzers.
    Defines required landmarks, biomechanical measurement extraction,
    deterministic form scoring, movement phase detection, and actionable coaching cues.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Primary exercise name (e.g., 'FULL_PLANCHE', 'FRONT_LEVER', 'PUSH_UP')."""
        pass

    @property
    @abc.abstractmethod
    def required_landmarks(self) -> List[str]:
        """List of essential COCO keypoints required to analyze this exercise."""
        pass

    @property
    def is_dynamic(self) -> bool:
        """True if exercise is repetition-based (Push-up, Squat); False for static holds (Planche)."""
        return False

    @property
    def scoring_weights(self) -> Dict[str, float]:
        """Configurable metric weights for overall score calculation."""
        return {}

    @property
    def camera_view_recommendation(self) -> str:
        """Coaching guidance for optimal camera placement."""
        return "Position camera side-on at torso height, 2.5m away with full body visible."

    def validate_pose(
        self,
        keypoints: Dict[str, KeypointData],
        threshold: float = 0.25,
        side_view: bool = False
    ) -> Tuple[bool, LiveAnalysisStatus, List[str]]:
        """
        Validate whether the detected keypoints meet visibility and confidence thresholds.
        Returns:
            (is_valid: bool, status: LiveAnalysisStatus, missing_landmarks: List[str])
        """
        if not keypoints:
            return False, LiveAnalysisStatus.NO_PERSON_DETECTED, ["all_keypoints"]

        missing = []
        low_confidence_count = 0

        for lm_name in self.required_landmarks:
            kp = keypoints.get(lm_name)
            if kp is None:
                missing.append(lm_name)
            elif not is_valid_keypoint(kp, threshold):
                missing.append(lm_name)
                if kp.confidence > 0.05:
                    low_confidence_count += 1

        if missing:
            # If multiple critical landmarks exist but are just below confidence threshold:
            if low_confidence_count >= len(missing) // 2 and low_confidence_count > 0:
                return False, LiveAnalysisStatus.LOW_CONFIDENCE, missing
            return False, LiveAnalysisStatus.INCOMPLETE_POSE, missing

        return True, LiveAnalysisStatus.VALID, []

    @abc.abstractmethod
    def extract_measurements(
        self,
        keypoints: Dict[str, KeypointData],
        side_view: bool = False
    ) -> Dict[str, float]:
        """
        Extract measurable joint angles and body line geometry.
        Returns a dictionary of named measurements (in degrees or ratios).
        """
        pass

    @abc.abstractmethod
    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        """
        Evaluate extracted measurements against biomechanical criteria.
        Returns:
            (form_score [0.0 - 100.0] or None, feedback_list, is_valid_frame)
        """
        pass

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        """
        Update hold duration or rep count in the session state.
        Default implementation handles isometric holds.
        Override in dynamic exercises for repetition counting.
        Returns:
            (hold_duration_sec, rep_count, current_phase)
        """
        # Default: Isometric hold tracking
        if is_valid_frame:
            if session.hold_start_time is None:
                session.hold_start_time = timestamp_sec
            hold_sec = max(0.0, timestamp_sec - session.hold_start_time)
            session.current_hold_duration_sec = hold_sec
            if hold_sec > session.peak_hold_duration_sec:
                session.peak_hold_duration_sec = hold_sec
            return round(hold_sec, 2), None, "HOLD"
        else:
            session.hold_start_time = None
            session.current_hold_duration_sec = 0.0
            return 0.0, None, "REST"

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        """
        Generate granular technique metrics with weights and status.
        Default fallback implementation wraps evaluate_live_form.
        """
        score, _, is_valid = self.evaluate_live_form(measurements, side_view=side_view)
        s_val = score if score is not None else 50.0
        return [
            LiveScoringMetric(
                name="form_accuracy",
                value=s_val,
                score=s_val,
                weight=1.0,
                confidence=0.9 if is_valid else 0.5,
                status="VALID" if is_valid else "LOW_CONFIDENCE"
            )
        ]
