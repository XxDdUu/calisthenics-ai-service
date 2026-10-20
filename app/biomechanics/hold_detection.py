from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Optional
import numpy as np

from app.core.config import settings


class HoldState(str, Enum):
    NO_POSE = "NO_POSE"
    POSE_DETECTED = "POSE_DETECTED"
    POTENTIAL_HOLD = "POTENTIAL_HOLD"
    STABLE_HOLD = "STABLE_HOLD"
    HOLD_ENDED = "HOLD_ENDED"


@dataclass
class FrameMetricsSample:
    frame_idx: int
    timestamp_sec: float
    pose_detected: bool
    is_valid_quality: bool
    elbow_angle: float = 0.0
    shoulder_lean: float = 0.0
    hip_angle: float = 0.0
    body_tilt: float = 0.0
    form_score: float = 0.0
    hold_state: HoldState = HoldState.NO_POSE


@dataclass
class HoldDetectionSummary:
    total_hold_duration_sec: float
    stable_hold_duration_sec: float
    stable_frames_count: int
    potential_frames_count: int
    peak_score: float
    average_score: float
    aggregated_component_scores: dict
    is_valid_sustained_hold: bool


class PlancheHoldDetector:
    """
    State machine identifying sustained and stable Planche holds in temporal video streams:

    NO_POSE -> POSE_DETECTED -> POTENTIAL_HOLD -> STABLE_HOLD -> HOLD_ENDED

    Evaluates:
    - Pose confidence
    - Body-line stability
    - Joint-angle stability (temporal variance)
    - Minimum duration requirements
    """

    def __init__(
        self,
        fps: float = 30.0,
        min_hold_duration_sec: Optional[float] = None,
        stability_variance_threshold: Optional[float] = None,
        min_elbow_deg: float = 145.0,
        min_lean_deg: float = 8.0,
        max_tilt_deg: float = 35.0,
    ):
        self.fps = max(1.0, fps)
        self.min_hold_duration_sec = (
            min_hold_duration_sec if min_hold_duration_sec is not None else settings.MIN_HOLD_DURATION_SEC
        )
        self.stability_variance_threshold = (
            stability_variance_threshold if stability_variance_threshold is not None
            else settings.HOLD_STABILITY_VARIANCE_THRESHOLD
        )
        self.min_elbow_deg = min_elbow_deg
        self.min_lean_deg = min_lean_deg
        self.max_tilt_deg = max_tilt_deg

        self.current_state = HoldState.NO_POSE
        self.samples: List[FrameMetricsSample] = []
        self._consecutive_potential_count = 0
        self._consecutive_empty_count = 0

    def process_frame(
        self,
        frame_idx: int,
        timestamp_sec: float,
        pose_detected: bool,
        is_valid_quality: bool,
        elbow_angle: float,
        shoulder_lean: float,
        hip_angle: float,
        body_tilt: float,
        form_score: float
    ) -> HoldState:
        """
        Processes a single sampled video frame and updates the state machine.
        """
        if not pose_detected or not is_valid_quality:
            self._consecutive_empty_count += 1
            if self.current_state in (HoldState.POTENTIAL_HOLD, HoldState.STABLE_HOLD):
                if self._consecutive_empty_count >= 2:
                    self.current_state = HoldState.HOLD_ENDED
            else:
                self.current_state = HoldState.NO_POSE

            sample = FrameMetricsSample(
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                pose_detected=pose_detected,
                is_valid_quality=is_valid_quality,
                hold_state=self.current_state
            )
            self.samples.append(sample)
            return self.current_state

        self._consecutive_empty_count = 0

        # Check if posture resembles a planche hold attempt
        is_attempt = (
            elbow_angle >= self.min_elbow_deg and
            shoulder_lean >= self.min_lean_deg and
            body_tilt <= self.max_tilt_deg
        )

        if is_attempt:
            self._consecutive_potential_count += 1
            duration_in_attempt = self._consecutive_potential_count / self.fps

            # Check stability variance over attempt window (only frames within the current attempt)
            attempt_sample_count = self._consecutive_potential_count - 1
            max_window = int(self.fps * 2.0)
            window_size = min(attempt_sample_count, max_window)
            
            attempt_samples = self.samples[-window_size:] if window_size > 0 else []
            recent_elbows = [s.elbow_angle for s in attempt_samples if s.pose_detected] + [elbow_angle]
            recent_hips = [s.hip_angle for s in attempt_samples if s.pose_detected] + [hip_angle]
            recent_tilts = [s.body_tilt for s in attempt_samples if s.pose_detected] + [body_tilt]

            std_elbow = float(np.std(recent_elbows)) if len(recent_elbows) > 1 else 0.0
            std_hip = float(np.std(recent_hips)) if len(recent_hips) > 1 else 0.0
            std_tilt = float(np.std(recent_tilts)) if len(recent_tilts) > 1 else 0.0
            avg_variance = (std_elbow + std_hip + std_tilt) / 3.0

            if duration_in_attempt >= self.min_hold_duration_sec and avg_variance <= self.stability_variance_threshold:
                self.current_state = HoldState.STABLE_HOLD
            else:
                self.current_state = HoldState.POTENTIAL_HOLD
        else:
            if self.current_state == HoldState.STABLE_HOLD:
                self.current_state = HoldState.HOLD_ENDED
            else:
                self.current_state = HoldState.POSE_DETECTED
            self._consecutive_potential_count = 0

        sample = FrameMetricsSample(
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            pose_detected=True,
            is_valid_quality=True,
            elbow_angle=elbow_angle,
            shoulder_lean=shoulder_lean,
            hip_angle=hip_angle,
            body_tilt=body_tilt,
            form_score=form_score,
            hold_state=self.current_state
        )
        self.samples.append(sample)
        return self.current_state

    def get_summary(self, sample_interval: int = 1) -> HoldDetectionSummary:
        """
        Aggregates metrics across stable hold frames (or potential hold frames if no stable hold was achieved).
        Avoids single-frame scoring.
        """
        stable_samples = [s for s in self.samples if s.hold_state == HoldState.STABLE_HOLD]
        potential_samples = [s for s in self.samples if s.hold_state in (HoldState.POTENTIAL_HOLD, HoldState.STABLE_HOLD)]

        effective_step_sec = sample_interval / self.fps
        stable_duration = round(len(stable_samples) * effective_step_sec, 2)
        total_duration = round(len(potential_samples) * effective_step_sec, 2)

        # Final score aggregation: prefer stable frames
        if stable_samples:
            scores = [s.form_score for s in stable_samples]
            avg_score = round(float(np.mean(scores)), 1)
            peak_score = round(float(np.max(scores)), 1)
            is_valid = (stable_duration >= self.min_hold_duration_sec and avg_score >= 60.0)
        elif potential_samples:
            scores = [s.form_score for s in potential_samples]
            avg_score = round(float(np.mean(scores)), 1)
            peak_score = round(float(np.max(scores)), 1)
            is_valid = False
        else:
            avg_score = 0.0
            peak_score = 0.0
            is_valid = False

        return HoldDetectionSummary(
            total_hold_duration_sec=total_duration,
            stable_hold_duration_sec=stable_duration,
            stable_frames_count=len(stable_samples),
            potential_frames_count=len(potential_samples),
            peak_score=peak_score,
            average_score=avg_score,
            aggregated_component_scores={},
            is_valid_sustained_hold=is_valid
        )
