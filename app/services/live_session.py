import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List, Optional
from app.biomechanics.feedback_engine import FeedbackEngine
from app.biomechanics.scoring_engine import ExerciseScoringEngine
from app.biomechanics.smoothing import TemporalKeypointSmoother
from app.biomechanics.squat_counter import SquatRepCounter
from app.core.config import settings
from app.pose.tracking import PersonTracker
from app.schemas.analysis import KeypointData

logger = logging.getLogger(__name__)


@dataclass
class LiveSession:
    """Session state isolating temporal smoothing and dynamic hold/rep tracking per user/device."""
    session_id: str
    exercise: str
    progression: Optional[str] = None
    side_view: bool = False
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)
    frame_count: int = 0

    # Keypoint EMA smoother
    keypoint_smoother: TemporalKeypointSmoother = field(default_factory=lambda: TemporalKeypointSmoother(alpha=settings.SMOOTHING_ALPHA))

    # Angle/measurement EMA smoother
    measurement_history: Dict[str, float] = field(default_factory=dict)
    measurement_alpha: float = 0.60

    # Hold state
    hold_start_time: Optional[float] = None
    current_hold_duration_sec: float = 0.0
    peak_hold_duration_sec: float = 0.0
    peak_score: float = 0.0

    # Dynamic repetition tracking
    rep_count: int = 0
    rep_phase: str = "START"
    squat_counter: Optional[SquatRepCounter] = None

    def get_squat_counter(self) -> SquatRepCounter:
        """Get or initialize the dedicated squat counter for this session."""
        if self.squat_counter is None:
            self.squat_counter = SquatRepCounter()
        return self.squat_counter

    # Multi-person tracker, Scoring engine, Feedback engine
    tracker: PersonTracker = field(default_factory=PersonTracker)
    scoring_engine: ExerciseScoringEngine = field(default_factory=ExerciseScoringEngine)
    feedback_engine: FeedbackEngine = field(default_factory=FeedbackEngine)

    # Session statistics
    score_history: List[float] = field(default_factory=list)
    lowest_score: float = 100.0
    average_score: float = 0.0

    def record_frame_score(
        self,
        score: Optional[float],
        hold_duration: Optional[float] = None,
        reps: Optional[int] = None
    ) -> None:
        if score is not None and score > 0:
            self.score_history.append(score)
            if score > self.peak_score:
                self.peak_score = score
            if score < self.lowest_score:
                self.lowest_score = score
            self.average_score = sum(self.score_history) / len(self.score_history)

        if hold_duration is not None and hold_duration > self.peak_hold_duration_sec:
            self.peak_hold_duration_sec = hold_duration

        if reps is not None:
            self.rep_count = reps

    def get_session_summary(self) -> Dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "exercise": self.exercise,
            "totalFrames": self.frame_count,
            "durationSec": round(time.time() - self.created_at, 1),
            "averageScore": round(self.average_score, 1) if self.score_history else 0.0,
            "peakScore": round(self.peak_score, 1),
            "lowestScore": round(self.lowest_score, 1) if self.score_history else 0.0,
            "peakHoldDurationSec": round(self.peak_hold_duration_sec, 1),
            "totalReps": self.rep_count
        }

    def touch(self) -> None:
        self.last_active_at = time.time()
        self.frame_count += 1

    def smooth_keypoints(self, keypoints: Dict[str, KeypointData]) -> Dict[str, KeypointData]:
        return self.keypoint_smoother.smooth(keypoints)

    def smooth_measurements(self, raw_measurements: Dict[str, float]) -> Dict[str, float]:
        """Apply EMA smoothing to joint angle measurements to suppress live jitter."""
        if not raw_measurements:
            return {}

        smoothed: Dict[str, float] = {}
        for key, val in raw_measurements.items():
            if not math.isfinite(val):
                smoothed[key] = val
                continue

            prev = self.measurement_history.get(key)
            if prev is not None and math.isfinite(prev):
                # If measurement difference is huge (> 60 deg), assume sudden change or posture switch
                if abs(val - prev) > 60.0:
                    smooth_val = val
                else:
                    smooth_val = self.measurement_alpha * val + (1.0 - self.measurement_alpha) * prev
            else:
                smooth_val = val

            smoothed[key] = round(float(smooth_val), 1)
            self.measurement_history[key] = smooth_val

        return smoothed

    def reset(self) -> None:
        """Reset temporal filters and performance trackers."""
        self.keypoint_smoother.reset()
        self.measurement_history.clear()
        self.hold_start_time = None
        self.current_hold_duration_sec = 0.0
        self.peak_hold_duration_sec = 0.0
        self.peak_score = 0.0
        self.rep_count = 0
        self.rep_phase = "START"
        self.frame_count = 0
        self.score_history.clear()
        self.lowest_score = 100.0
        self.average_score = 0.0
        self.tracker.reset()
        self.scoring_engine.reset()
        self.feedback_engine.reset()
        if self.squat_counter is not None:
            self.squat_counter.reset()
        self.last_active_at = time.time()
        logger.info("LiveSession %s reset.", self.session_id)


class LiveSessionManager:
    """Thread-safe manager for concurrent live camera analysis sessions."""

    def __init__(self, max_idle_sec: float = 300.0):
        self._sessions: Dict[str, LiveSession] = {}
        self._lock = Lock()
        self._max_idle_sec = max_idle_sec

    def get_or_create(
        self,
        session_id: Optional[str] = None,
        exercise: str = "FULL_PLANCHE",
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> LiveSession:
        with self._lock:
            self._cleanup_stale()

            sid = session_id or str(uuid.uuid4())[:12]
            session = self._sessions.get(sid)

            if session is None:
                session = LiveSession(
                    session_id=sid,
                    exercise=exercise,
                    progression=progression,
                    side_view=side_view
                )
                if exercise.upper() == "SQUAT":
                    session.squat_counter = SquatRepCounter()
                self._sessions[sid] = session
                logger.info("Created new LiveSession %s for exercise '%s'", sid, exercise)
            else:
                # If exercise changed, reset history to avoid smoothing across exercises
                if session.exercise.upper() != exercise.upper():
                    logger.info("LiveSession %s changed exercise from '%s' to '%s'. Resetting state.",
                                sid, session.exercise, exercise)
                    session.exercise = exercise
                    session.progression = progression
                    session.side_view = side_view
                    session.reset()
                    if exercise.upper() == "SQUAT":
                        session.squat_counter = SquatRepCounter()
                    else:
                        session.squat_counter = None
                session.side_view = side_view
                session.last_active_at = time.time()

            return session

    def get(self, session_id: str) -> Optional[LiveSession]:
        with self._lock:
            return self._sessions.get(session_id)

    def reset(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.reset()
                return True
            return False

    def remove(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Terminated LiveSession %s", session_id)
                return True
            return False

    def _cleanup_stale(self) -> None:
        """Evict sessions idle longer than max_idle_sec."""
        now = time.time()
        stale_ids = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_active_at) > self._max_idle_sec
        ]
        for sid in stale_ids:
            del self._sessions[sid]
            logger.info("Evicted stale LiveSession %s", sid)


# Global session manager instance
session_manager = LiveSessionManager()
