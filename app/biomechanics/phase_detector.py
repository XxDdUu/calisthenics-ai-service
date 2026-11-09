import logging
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class DynamicPhase(str, Enum):
    START = "START"
    ECCENTRIC = "ECCENTRIC"
    BOTTOM = "BOTTOM"
    CONCENTRIC = "CONCENTRIC"
    TOP = "TOP"
    COMPLETED_REP = "COMPLETED_REP"


class StaticPhase(str, Enum):
    SETUP = "SETUP"
    HOLDING = "HOLDING"
    HOLD_LOST = "HOLD_LOST"
    COMPLETED = "COMPLETED"


class ExercisePhaseDetector:
    """
    Robust movement phase detector with hysteresis for calisthenics exercises.
    Supports both dynamic rep-based exercises and static isometric holds.
    Prevents false repetition counts from small oscillations or detector noise.
    """

    def __init__(
        self,
        exercise_name: str,
        is_dynamic: bool = True,
        min_rom_delta: float = 35.0,
        min_rep_duration_sec: float = 0.8
    ):
        self.exercise_name = exercise_name.upper()
        self.is_dynamic = is_dynamic
        self.min_rom_delta = min_rom_delta
        self.min_rep_duration_sec = min_rep_duration_sec

        # Dynamic state
        self.current_phase: str = DynamicPhase.START.value if is_dynamic else StaticPhase.SETUP.value
        self.rep_count: int = 0
        self.last_rep_end_sec: float = 0.0
        self.rep_start_sec: float = 0.0
        self.min_measured_value: float = 999.0
        self.max_measured_value: float = -999.0
        self.phase_enter_sec: float = 0.0

        # Static hold state
        self.hold_start_sec: Optional[float] = None
        self.total_hold_duration_sec: float = 0.0
        self.current_hold_duration_sec: float = 0.0
        self.lost_form_start_sec: Optional[float] = None

    def reset(self):
        """Reset phase and counter state."""
        self.current_phase = DynamicPhase.START.value if self.is_dynamic else StaticPhase.SETUP.value
        self.rep_count = 0
        self.last_rep_end_sec = 0.0
        self.rep_start_sec = 0.0
        self.min_measured_value = 999.0
        self.max_measured_value = -999.0
        self.phase_enter_sec = 0.0
        self.hold_start_sec = None
        self.total_hold_duration_sec = 0.0
        self.current_hold_duration_sec = 0.0
        self.lost_form_start_sec = None

    def update_dynamic_rep(
        self,
        primary_metric_value: float,
        top_threshold: float,
        bottom_threshold: float,
        timestamp_sec: float,
        is_valid_form: bool = True
    ) -> Tuple[int, str]:
        """
        Update state machine for dynamic exercises (Push-up, Squat, Dip, Pull-up, etc.).
        E.g. For Push-up:
            top_threshold = 155.0 (lockout angle)
            bottom_threshold = 95.0 (depth angle)
        """
        now = timestamp_sec if timestamp_sec is not None else time.time()
        val = primary_metric_value

        self.min_measured_value = min(self.min_measured_value, val)
        self.max_measured_value = max(self.max_measured_value, val)

        prev_phase = self.current_phase

        if self.current_phase == DynamicPhase.START.value:
            if val >= top_threshold - 5.0 and is_valid_form:
                self.current_phase = DynamicPhase.TOP.value
                self.rep_start_sec = now
                self.min_measured_value = val
                self.max_measured_value = val

        elif self.current_phase == DynamicPhase.TOP.value:
            # Beginning descent
            if val < top_threshold - 15.0:
                self.current_phase = DynamicPhase.ECCENTRIC.value
                self.phase_enter_sec = now

        elif self.current_phase == DynamicPhase.ECCENTRIC.value:
            # Reached inflection / depth
            if val <= bottom_threshold + 5.0:
                self.current_phase = DynamicPhase.BOTTOM.value
                self.phase_enter_sec = now
            # Aborted rep back to top without hitting bottom
            elif val >= top_threshold - 10.0:
                self.current_phase = DynamicPhase.TOP.value

        elif self.current_phase == DynamicPhase.BOTTOM.value:
            # Beginning ascent
            if val > bottom_threshold + 15.0:
                self.current_phase = DynamicPhase.CONCENTRIC.value
                self.phase_enter_sec = now

        elif self.current_phase == DynamicPhase.CONCENTRIC.value:
            # Return to lockout at top
            if val >= top_threshold - 10.0:
                # Validate range of motion requirement
                rom = self.max_measured_value - self.min_measured_value
                duration = now - self.rep_start_sec

                if rom >= self.min_rom_delta and duration >= self.min_rep_duration_sec:
                    self.rep_count += 1
                    self.current_phase = DynamicPhase.COMPLETED_REP.value
                    self.last_rep_end_sec = now
                else:
                    self.current_phase = DynamicPhase.TOP.value

                self.rep_start_sec = now
                self.min_measured_value = val
                self.max_measured_value = val

        elif self.current_phase == DynamicPhase.COMPLETED_REP.value:
            # Smooth transition back to TOP
            self.current_phase = DynamicPhase.TOP.value

        return self.rep_count, self.current_phase

    def update_pull_up_rep(
        self,
        elbow_angle: float,
        timestamp_sec: float,
        is_valid_form: bool = True
    ) -> Tuple[int, str]:
        """
        Pull-up state machine:
        START/BOTTOM (dead hang ~160°+) -> CONCENTRIC (pulling up) -> TOP (chin over bar ~75°-)
        -> ECCENTRIC (lowering down) -> BOTTOM (lockout hang) -> COMPLETED_REP
        """
        now = timestamp_sec if timestamp_sec is not None else time.time()
        val = elbow_angle

        self.min_measured_value = min(self.min_measured_value, val)
        self.max_measured_value = max(self.max_measured_value, val)

        if self.current_phase in (DynamicPhase.START.value, DynamicPhase.BOTTOM.value):
            if val >= 145.0:
                self.current_phase = DynamicPhase.BOTTOM.value
                self.rep_start_sec = now
                self.min_measured_value = val
                self.max_measured_value = val
            elif val < 135.0:
                self.current_phase = DynamicPhase.CONCENTRIC.value

        elif self.current_phase == DynamicPhase.CONCENTRIC.value:
            if val <= 85.0:
                self.current_phase = DynamicPhase.TOP.value
                self.phase_enter_sec = now
            elif val > 150.0:
                self.current_phase = DynamicPhase.BOTTOM.value

        elif self.current_phase == DynamicPhase.TOP.value:
            if val > 95.0:
                self.current_phase = DynamicPhase.ECCENTRIC.value

        elif self.current_phase == DynamicPhase.ECCENTRIC.value:
            if val >= 145.0:
                rom = self.max_measured_value - self.min_measured_value
                duration = now - self.rep_start_sec
                if rom >= 50.0 and duration >= self.min_rep_duration_sec:
                    self.rep_count += 1
                    self.current_phase = DynamicPhase.COMPLETED_REP.value
                else:
                    self.current_phase = DynamicPhase.BOTTOM.value
                self.min_measured_value = val
                self.max_measured_value = val

        elif self.current_phase == DynamicPhase.COMPLETED_REP.value:
            self.current_phase = DynamicPhase.BOTTOM.value

        return self.rep_count, self.current_phase

    def update_static_hold(
        self,
        is_valid_hold: bool,
        timestamp_sec: float
    ) -> Tuple[float, str]:
        """
        Static isometric hold state machine (Planche, Front Lever, Handstand, L-sit):
        SETUP -> HOLDING -> HOLD_LOST -> COMPLETED
        """
        now = timestamp_sec if timestamp_sec is not None else time.time()

        if is_valid_hold:
            self.lost_form_start_sec = None
            if self.hold_start_sec is None:
                self.hold_start_sec = now
                self.current_phase = StaticPhase.HOLDING.value

            self.current_hold_duration_sec = max(0.0, now - self.hold_start_sec)
            self.current_phase = StaticPhase.HOLDING.value
        else:
            if self.hold_start_sec is not None:
                if self.lost_form_start_sec is None:
                    self.lost_form_start_sec = now
                # Allow 0.5s grace period before officially marking hold as lost
                elif now - self.lost_form_start_sec > 0.5:
                    self.total_hold_duration_sec += self.current_hold_duration_sec
                    self.current_hold_duration_sec = 0.0
                    self.hold_start_sec = None
                    self.current_phase = StaticPhase.HOLD_LOST.value
            else:
                self.current_phase = StaticPhase.SETUP.value

        return round(self.current_hold_duration_sec, 2), self.current_phase
