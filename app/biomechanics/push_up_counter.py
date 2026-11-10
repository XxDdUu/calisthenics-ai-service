import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_body_line_deviation,
    get_point,
)
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveFeedbackItem,
    LiveScoringMetric,
)

logger = logging.getLogger(__name__)


class PushUpState(str, Enum):
    TOP = "TOP"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"


@dataclass
class PushUpCounterConfig:
    top_angle: float = 155.0               # Elbow angle >= 155.0 to be in TOP / Lockout
    descending_angle: float = 145.0        # Elbow angle < 145.0 triggers DESCENDING
    bottom_angle: float = 90.0             # Elbow angle <= 90.0 triggers BOTTOM (valid depth)
    bottom_exit_angle: float = 98.0        # Elbow angle > 98.0 exits BOTTOM to ASCENDING (hysteresis)
    descending_abort_angle: float = 152.0  # Elbow angle >= 152.0 aborts partial descent without rep
    ema_alpha: float = 0.50                # Exponential moving average smoothing factor
    min_bottom_duration_sec: float = 0.15  # Minimum bottom hold duration (150 ms)
    min_rep_duration_sec: float = 0.50     # Minimum total duration of valid rep (500 ms)
    min_cooldown_sec: float = 0.30         # Minimum interval between completed reps (300 ms)
    max_lost_duration_sec: float = 2.0     # Time without valid tracking before resetting in-progress rep


@dataclass
class PushUpRepResult:
    exercise: str = "PUSH_UP"
    rep_count: int = 0
    current_state: str = "TOP"
    elbow_angle: float = 180.0
    left_elbow_angle: Optional[float] = None
    right_elbow_angle: Optional[float] = None
    average_elbow_angle: Optional[float] = None
    min_elbow_angle: float = 180.0
    body_alignment: float = 180.0
    depth: str = "GOOD"                    # GOOD, MODERATE, SHALLOW
    form_score: float = 100.0
    is_tracking: bool = True
    feedback: str = ""
    rep_just_completed: bool = False
    movement_duration_sec: float = 0.0


def extract_push_up_geometry(
    keypoints: Dict[str, KeypointData],
    threshold: float = 0.25,
    side_view: bool = False
) -> Dict[str, Any]:
    """
    Extract push-up biomechanical measurements from COCO 17 keypoints:
    - Left/Right Shoulder (5, 6)
    - Left/Right Elbow (7, 8)
    - Left/Right Wrist (9, 10)
    - Left/Right Hip (11, 12)
    - Left/Right Ankle (15, 16) / Knee (13, 14)

    Selects the most reliable side when the other side is occluded.
    Rejects unreliable frames instead of generating false measurements.
    """
    if not keypoints:
        return {
            "is_pose_valid": False,
            "primary_elbow_angle": None,
            "left_elbow_angle": None,
            "right_elbow_angle": None,
            "average_elbow_angle": None,
            "body_alignment": None,
            "hip_sag": False,
            "hip_pike": False,
            "left_arm_reliable": False,
            "right_arm_reliable": False,
            "missing": ["all_keypoints"]
        }

    def _is_reliable(name: str) -> bool:
        kp = keypoints.get(name)
        return kp is not None and kp.visible and kp.confidence >= threshold

    l_sh_ok = _is_reliable("left_shoulder")
    l_el_ok = _is_reliable("left_elbow")
    l_wr_ok = _is_reliable("left_wrist")
    left_arm_reliable = l_sh_ok and l_el_ok and l_wr_ok

    r_sh_ok = _is_reliable("right_shoulder")
    r_el_ok = _is_reliable("right_elbow")
    r_wr_ok = _is_reliable("right_wrist")
    right_arm_reliable = r_sh_ok and r_el_ok and r_wr_ok

    # Calculate elbow angles for reliable arms
    l_elbow_angle = None
    if left_arm_reliable:
        p_sh = get_point(keypoints.get("left_shoulder"))
        p_el = get_point(keypoints.get("left_elbow"))
        p_wr = get_point(keypoints.get("left_wrist"))
        if p_sh and p_el and p_wr:
            l_elbow_angle = calculate_angle_2d(p_sh, p_el, p_wr, default_angle=180.0)

    r_elbow_angle = None
    if right_arm_reliable:
        p_sh = get_point(keypoints.get("right_shoulder"))
        p_el = get_point(keypoints.get("right_elbow"))
        p_wr = get_point(keypoints.get("right_wrist"))
        if p_sh and p_el and p_wr:
            r_elbow_angle = calculate_angle_2d(p_sh, p_el, p_wr, default_angle=180.0)

    # Missing diagnostics
    missing = []
    if not (l_sh_ok or r_sh_ok): missing.append("shoulder")
    if not (l_el_ok or r_el_ok): missing.append("elbow")
    if not (l_wr_ok or r_wr_ok): missing.append("wrist")

    l_hip_ok = _is_reliable("left_hip")
    r_hip_ok = _is_reliable("right_hip")
    if not (l_hip_ok or r_hip_ok): missing.append("hip")

    l_ak_ok = _is_reliable("left_ankle") or _is_reliable("left_knee")
    r_ak_ok = _is_reliable("right_ankle") or _is_reliable("right_knee")
    if not (l_ak_ok or r_ak_ok): missing.append("ankle_or_knee")

    if not left_arm_reliable and not right_arm_reliable:
        return {
            "is_pose_valid": False,
            "primary_elbow_angle": None,
            "left_elbow_angle": l_elbow_angle,
            "right_elbow_angle": r_elbow_angle,
            "average_elbow_angle": None,
            "body_alignment": None,
            "hip_sag": False,
            "hip_pike": False,
            "left_arm_reliable": False,
            "right_arm_reliable": False,
            "missing": missing if missing else ["arms_unreliable"]
        }

    # Determine dominant/primary elbow angle and average
    if left_arm_reliable and right_arm_reliable:
        assert l_elbow_angle is not None and r_elbow_angle is not None
        avg_elbow = round((l_elbow_angle + r_elbow_angle) / 2.0, 1)
        if side_view:
            l_conf = sum(keypoints[k].confidence for k in ["left_shoulder", "left_elbow", "left_wrist"])
            r_conf = sum(keypoints[k].confidence for k in ["right_shoulder", "right_elbow", "right_wrist"])
            primary_elbow = l_elbow_angle if l_conf >= r_conf else r_elbow_angle
        else:
            primary_elbow = avg_elbow
    elif left_arm_reliable:
        assert l_elbow_angle is not None
        primary_elbow = l_elbow_angle
        avg_elbow = l_elbow_angle
    else:
        assert r_elbow_angle is not None
        primary_elbow = r_elbow_angle
        avg_elbow = r_elbow_angle

    # Body alignment (Shoulder - Hip - Ankle)
    # Pick the side with higher combined confidence
    l_side_ok = l_sh_ok and l_hip_ok and l_ak_ok
    r_side_ok = r_sh_ok and r_hip_ok and r_ak_ok

    p_sh = None
    p_hip = None
    p_ak = None

    if l_side_ok and r_side_ok:
        l_c = (keypoints["left_shoulder"].confidence + keypoints["left_hip"].confidence)
        r_c = (keypoints["right_shoulder"].confidence + keypoints["right_hip"].confidence)
        use_left = l_c >= r_c
        p_sh = get_point(keypoints["left_shoulder"] if use_left else keypoints["right_shoulder"])
        p_hip = get_point(keypoints["left_hip"] if use_left else keypoints["right_hip"])
        p_ak = get_point(keypoints.get("left_ankle" if use_left else "right_ankle") or keypoints.get("left_knee" if use_left else "right_knee"))
    elif l_side_ok:
        p_sh = get_point(keypoints["left_shoulder"])
        p_hip = get_point(keypoints["left_hip"])
        p_ak = get_point(keypoints.get("left_ankle") or keypoints.get("left_knee"))
    elif r_side_ok:
        p_sh = get_point(keypoints["right_shoulder"])
        p_hip = get_point(keypoints["right_hip"])
        p_ak = get_point(keypoints.get("right_ankle") or keypoints.get("right_knee"))
    else:
        p_sh = get_point(keypoints.get("left_shoulder") if l_sh_ok else keypoints.get("right_shoulder"))
        p_hip = get_point(keypoints.get("left_hip") if l_hip_ok else keypoints.get("right_hip"))
        p_ak = get_point(keypoints.get("left_ankle") or keypoints.get("right_ankle") or keypoints.get("left_knee") or keypoints.get("right_knee"))

    body_line = 180.0
    hip_sag = False
    hip_pike = False

    if p_sh and p_hip and p_ak:
        dev_angle, align_type = calculate_body_line_deviation(p_sh, p_hip, p_ak)
        body_line = dev_angle
        hip_sag = (align_type == "SAG")
        hip_pike = (align_type == "PIKE")

    return {
        "is_pose_valid": True,
        "primary_elbow_angle": primary_elbow,
        "left_elbow_angle": l_elbow_angle,
        "right_elbow_angle": r_elbow_angle,
        "average_elbow_angle": avg_elbow,
        "body_alignment": round(body_line, 1),
        "hip_sag": hip_sag,
        "hip_pike": hip_pike,
        "left_arm_reliable": left_arm_reliable,
        "right_arm_reliable": right_arm_reliable,
        "missing": []
    }


class PushUpRepCounter:
    """
    Dedicated, completely independent Push-up Rep Counter.
    Implements a strict 4-phase state machine:
    TOP -> DESCENDING -> BOTTOM -> ASCENDING -> TOP (REP COMPLETED)

    Features:
    - Biomechanical thresholds:
      * TOP: Elbow angle >= 155°
      * DESCENDING: Elbow angle < 145°
      * BOTTOM: Elbow angle <= 90° (full depth reached)
      * ASCENDING: Elbow angle > 98°
      * REP COMPLETED: Elbow returns to >= 155°
    - EMA temporal smoothing on elbow flexion angle.
    - Hysteresis to reject noise oscillations around state boundaries.
    - Minimum bottom duration check (prevents single-frame bounce false reps).
    - Minimum rep duration check (prevents detector glitches from completing a rep).
    - Minimum cooldown check between completed repetitions.
    - Body alignment monitoring (detects sagging or piking).
    - Automatic state recovery upon tracking loss.
    """

    def __init__(self, config: Optional[PushUpCounterConfig] = None):
        self.config = config or PushUpCounterConfig()
        self.state: PushUpState = PushUpState.TOP
        self.rep_count: int = 0
        self.current_elbow_angle: float = 180.0
        self.smoothed_elbow_angle: float = 180.0
        self.min_elbow_angle: float = 180.0
        self.body_alignment: float = 180.0
        self.depth_quality: str = "GOOD"
        self.is_tracking: bool = False
        self.rep_just_completed: bool = False
        self.feedback: str = "Ready. Begin your push-up."
        self.last_completed_min_angle: float = 180.0
        self.last_completed_depth: str = "GOOD"
        self.form_score: float = 100.0

        # Temporal timestamps
        self.rep_start_time: float = 0.0
        self.bottom_enter_time: float = 0.0
        self.last_rep_completed_time: float = 0.0
        self.last_valid_tracking_time: float = 0.0
        self.reached_bottom_depth: bool = False
        self.is_smoothed_initialized: bool = False
        self.consecutive_bottom_frames: int = 0
        self._simulated_clock: float = 0.0

        # Form metrics during rep
        self.rep_body_lines: List[float] = []

    def reset(self) -> None:
        """Reset rep counter and tracking state."""
        self.state = PushUpState.TOP
        self.rep_count = 0
        self.current_elbow_angle = 180.0
        self.smoothed_elbow_angle = 180.0
        self.min_elbow_angle = 180.0
        self.body_alignment = 180.0
        self.depth_quality = "GOOD"
        self.is_tracking = False
        self.rep_just_completed = False
        self.feedback = "Ready. Begin your push-up."
        self.last_completed_min_angle = 180.0
        self.last_completed_depth = "GOOD"
        self.form_score = 100.0
        self.rep_start_time = 0.0
        self.bottom_enter_time = 0.0
        self.last_rep_completed_time = 0.0
        self.last_valid_tracking_time = 0.0
        self.reached_bottom_depth = False
        self.is_smoothed_initialized = False
        self.consecutive_bottom_frames = 0
        self._simulated_clock = 0.0
        self.rep_body_lines.clear()

    def process_angle(
        self,
        angle: Optional[float],
        body_alignment: float = 180.0,
        timestamp_sec: Optional[float] = None
    ) -> PushUpRepResult:
        """
        Process a single elbow angle and body alignment measurement through Push-up State Machine.
        """
        self.rep_just_completed = False

        if timestamp_sec is not None:
            now = float(timestamp_sec)
        else:
            self._simulated_clock += 0.10
            now = self._simulated_clock

        # 1. Handle Tracking Loss (no confident elbow angle)
        if angle is None or not math.isfinite(angle):
            self.is_tracking = False
            if (
                self.last_valid_tracking_time > 0
                and (now - self.last_valid_tracking_time) > self.config.max_lost_duration_sec
            ):
                if self.state != PushUpState.TOP:
                    logger.debug("Push-up tracking lost for > %.1fs. Resetting state to TOP.", self.config.max_lost_duration_sec)
                    self.state = PushUpState.TOP
                    self.min_elbow_angle = 180.0
                    self.reached_bottom_depth = False
                    self.is_smoothed_initialized = False
                    self.consecutive_bottom_frames = 0
                    self.rep_body_lines.clear()

            return PushUpRepResult(
                exercise="PUSH_UP",
                rep_count=self.rep_count,
                current_state=self.state.value,
                elbow_angle=round(self.smoothed_elbow_angle, 1),
                left_elbow_angle=None,
                right_elbow_angle=None,
                average_elbow_angle=round(self.smoothed_elbow_angle, 1),
                min_elbow_angle=round(self.min_elbow_angle, 1),
                body_alignment=round(self.body_alignment, 1),
                depth=self.depth_quality,
                form_score=round(self.form_score, 1),
                is_tracking=False,
                feedback=self.feedback,
                rep_just_completed=False,
                movement_duration_sec=round(max(0.0, now - self.rep_start_time), 2) if self.state != PushUpState.TOP else 0.0
            )

        # 2. Tracking Active
        self.is_tracking = True
        self.last_valid_tracking_time = now
        self.current_elbow_angle = float(angle)
        self.body_alignment = float(body_alignment)

        if not self.is_smoothed_initialized or self.config.ema_alpha >= 1.0:
            self.smoothed_elbow_angle = self.current_elbow_angle
            self.is_smoothed_initialized = True
        else:
            if abs(self.current_elbow_angle - self.smoothed_elbow_angle) > 20.0:
                self.smoothed_elbow_angle = self.current_elbow_angle
            else:
                alpha = self.config.ema_alpha
                self.smoothed_elbow_angle = alpha * self.current_elbow_angle + (1.0 - alpha) * self.smoothed_elbow_angle

        e_angle = self.smoothed_elbow_angle

        # Check body alignment issues
        if self.body_alignment < 160.0:
            alignment_feedback = "Keep your body straight. Don't drop your hips."
        else:
            alignment_feedback = None

        # 3. State Machine Transitions
        if self.state == PushUpState.TOP:
            if e_angle < self.config.descending_angle:
                self.state = PushUpState.DESCENDING
                self.rep_start_time = now
                self.min_elbow_angle = e_angle
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self.rep_body_lines = [self.body_alignment]
                self.feedback = alignment_feedback or "Control your descent"
            else:
                self.min_elbow_angle = 180.0
                if e_angle < self.config.top_angle:
                    self.feedback = "Return to the top position"
                else:
                    self.feedback = alignment_feedback or "Ready. Begin your push-up."

        elif self.state == PushUpState.DESCENDING:
            self.min_elbow_angle = min(self.min_elbow_angle, e_angle)
            self.rep_body_lines.append(self.body_alignment)

            if e_angle <= self.config.bottom_angle or (self.current_elbow_angle <= self.config.bottom_angle and e_angle <= 96.0):
                self.state = PushUpState.BOTTOM
                self.bottom_enter_time = now
                self.reached_bottom_depth = True
                self.consecutive_bottom_frames = 1
                self.depth_quality = self._classify_depth(self.min_elbow_angle)
                self.feedback = alignment_feedback or "Good push-up depth"
            elif e_angle >= self.config.descending_abort_angle:
                # Aborted partial push-up
                self.state = PushUpState.TOP
                self.min_elbow_angle = 180.0
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self.rep_body_lines.clear()
                self.feedback = "Lower your body further"
            else:
                self.feedback = alignment_feedback or "Lower your body further"

        elif self.state == PushUpState.BOTTOM:
            self.min_elbow_angle = min(self.min_elbow_angle, e_angle)
            self.rep_body_lines.append(self.body_alignment)
            self.consecutive_bottom_frames += 1
            self.depth_quality = self._classify_depth(self.min_elbow_angle)

            if e_angle > self.config.bottom_exit_angle:
                bottom_duration = now - self.bottom_enter_time
                bottom_satisfied = (
                    bottom_duration >= self.config.min_bottom_duration_sec
                    or self.consecutive_bottom_frames >= 2
                    or self.config.min_bottom_duration_sec == 0.0
                )
                if bottom_satisfied:
                    self.reached_bottom_depth = True
                else:
                    self.reached_bottom_depth = False

                self.state = PushUpState.ASCENDING
                self.feedback = alignment_feedback or "Push up strong!"
            else:
                self.feedback = alignment_feedback or "Good push-up depth"

        elif self.state == PushUpState.ASCENDING:
            self.min_elbow_angle = min(self.min_elbow_angle, e_angle)
            self.rep_body_lines.append(self.body_alignment)

            is_top = (
                e_angle >= self.config.top_angle
                or (self.current_elbow_angle >= self.config.top_angle and e_angle >= 148.0)
            )

            if is_top:
                rep_duration = now - self.rep_start_time
                cooldown = now - self.last_rep_completed_time if self.last_rep_completed_time > 0 else 999.0

                is_valid_rep = (
                    self.reached_bottom_depth
                    and (rep_duration >= self.config.min_rep_duration_sec or self.config.min_rep_duration_sec == 0.0)
                    and (cooldown >= self.config.min_cooldown_sec or self.config.min_cooldown_sec == 0.0)
                )

                if is_valid_rep:
                    self.rep_count += 1
                    self.last_rep_completed_time = now
                    self.rep_just_completed = True
                    self.last_completed_min_angle = self.min_elbow_angle
                    self.last_completed_depth = self.depth_quality

                    # Calculate rep form score
                    avg_body_line = sum(self.rep_body_lines) / max(1, len(self.rep_body_lines))
                    align_score = max(50.0, min(100.0, 100.0 - abs(180.0 - avg_body_line) * 2.0))
                    depth_score = 100.0 if self.min_elbow_angle <= 90.0 else max(60.0, 100.0 - (self.min_elbow_angle - 90.0) * 2.0)
                    self.form_score = round(0.50 * align_score + 0.50 * depth_score, 1)

                    self.feedback = "Good push-up! Stand by for next rep."
                else:
                    self.feedback = "Return to the top position"

                self.state = PushUpState.TOP
                self.min_elbow_angle = self.last_completed_min_angle if is_valid_rep else 180.0
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self.rep_body_lines.clear()

            elif e_angle <= self.config.bottom_angle:
                self.state = PushUpState.BOTTOM
                self.bottom_enter_time = now
                self.consecutive_bottom_frames += 1

        self.depth_quality = self._classify_depth(self.min_elbow_angle if self.state != PushUpState.TOP else self.current_elbow_angle)

        if self.state == PushUpState.TOP and self.last_completed_min_angle < 180.0:
            reported_min = self.last_completed_min_angle
            reported_depth = self.last_completed_depth
        elif self.state != PushUpState.TOP:
            reported_min = self.min_elbow_angle
            reported_depth = self.depth_quality
        else:
            reported_min = self.current_elbow_angle
            reported_depth = self.depth_quality

        mov_duration = round(max(0.0, now - self.rep_start_time), 2) if self.state != PushUpState.TOP else 0.0

        return PushUpRepResult(
            exercise="PUSH_UP",
            rep_count=self.rep_count,
            current_state=self.state.value,
            elbow_angle=round(self.smoothed_elbow_angle, 1),
            left_elbow_angle=None,
            right_elbow_angle=None,
            average_elbow_angle=round(self.smoothed_elbow_angle, 1),
            min_elbow_angle=round(reported_min, 1),
            body_alignment=round(self.body_alignment, 1),
            depth=reported_depth,
            form_score=round(self.form_score, 1),
            is_tracking=self.is_tracking,
            feedback=self.feedback,
            rep_just_completed=self.rep_just_completed,
            movement_duration_sec=mov_duration
        )

    def _classify_depth(self, angle: float) -> str:
        """Classify push-up depth from elbow angle: GOOD (<=90), MODERATE (90-105), SHALLOW (>105)."""
        if angle <= self.config.bottom_angle:
            return "GOOD"
        elif angle <= 105.0:
            return "MODERATE"
        return "SHALLOW"
