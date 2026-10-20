from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union
from app.schemas.analysis import MetricStatus, ProgressionStage


@dataclass
class PlancheMetricRule:
    """Configurable rule and scoring parameters for a biomechanical feature."""
    name: str
    target_range: str
    ideal_min: Optional[float] = None
    ideal_max: Optional[float] = None
    acceptable_min: Optional[float] = None
    acceptable_max: Optional[float] = None
    weight: float = 0.20
    is_upper_bounded: bool = False  # True if smaller is better (e.g. body tilt <= 15)

    def evaluate_score(self, measured_value: float) -> Tuple[float, MetricStatus]:
        """
        Calculates an interpretable score [0.0, 100.0] and MetricStatus
        without magic numbers.
        """
        # Upper-bounded metric (smaller is better, e.g. body tilt <= 15 deg)
        if self.is_upper_bounded:
            ideal_limit = self.ideal_max if self.ideal_max is not None else 15.0
            acceptable_limit = self.acceptable_max if self.acceptable_max is not None else 25.0
            if measured_value <= ideal_limit:
                return 100.0, MetricStatus.GOOD
            elif measured_value <= acceptable_limit:
                # Interpolate from 100.0 down to 70.0
                ratio = (measured_value - ideal_limit) / max(1e-5, (acceptable_limit - ideal_limit))
                score = max(70.0, 100.0 - ratio * 30.0)
                return round(score, 1), MetricStatus.NEEDS_IMPROVEMENT
            else:
                # Drop from 70.0 down to 0.0
                over = measured_value - acceptable_limit
                score = max(0.0, 70.0 - over * 3.5)
                return round(score, 1), MetricStatus.FAIL

        # Range-bounded metric (e.g. Adv Tuck hip 90 - 130 deg)
        if self.ideal_min is not None and self.ideal_max is not None:
            if self.ideal_min <= measured_value <= self.ideal_max:
                return 100.0, MetricStatus.GOOD
            elif self.acceptable_min is not None and measured_value < self.ideal_min:
                if measured_value >= self.acceptable_min:
                    ratio = (self.ideal_min - measured_value) / max(1e-5, (self.ideal_min - self.acceptable_min))
                    score = max(60.0, 100.0 - ratio * 40.0)
                    return round(score, 1), MetricStatus.NEEDS_IMPROVEMENT
                else:
                    diff = self.acceptable_min - measured_value
                    score = max(0.0, 60.0 - diff * 3.0)
                    return round(score, 1), MetricStatus.FAIL
            elif self.acceptable_max is not None and measured_value > self.ideal_max:
                if measured_value <= self.acceptable_max:
                    ratio = (measured_value - self.ideal_max) / max(1e-5, (self.acceptable_max - self.ideal_max))
                    score = max(60.0, 100.0 - ratio * 40.0)
                    return round(score, 1), MetricStatus.NEEDS_IMPROVEMENT
                else:
                    diff = measured_value - self.acceptable_max
                    score = max(0.0, 60.0 - diff * 3.0)
                    return round(score, 1), MetricStatus.FAIL

        # Lower-bounded metric (higher is better, e.g. elbow extension >= 170 deg)
        ideal_limit = self.ideal_min if self.ideal_min is not None else 170.0
        acceptable_limit = self.acceptable_min if self.acceptable_min is not None else 155.0

        if measured_value >= ideal_limit:
            return 100.0, MetricStatus.PASS
        elif measured_value >= acceptable_limit:
            ratio = (ideal_limit - measured_value) / max(1e-5, (ideal_limit - acceptable_limit))
            score = max(65.0, 100.0 - ratio * 35.0)
            return round(score, 1), MetricStatus.NEEDS_IMPROVEMENT
        else:
            diff = acceptable_limit - measured_value
            score = max(0.0, 65.0 - diff * 3.0)
            return round(score, 1), MetricStatus.FAIL


@dataclass
class PlancheProgressionConfig:
    stage: ProgressionStage
    name: str
    elbow_extension: PlancheMetricRule
    shoulder_position: PlancheMetricRule
    hip_extension: PlancheMetricRule
    knee_extension: PlancheMetricRule
    body_line: PlancheMetricRule
    leg_spread: Optional[PlancheMetricRule] = None
    min_hold_score: float = 60.0
    min_elbow_for_hold: float = 150.0
    min_lean_for_hold: float = 8.0
    max_tilt_for_hold: float = 35.0


# Progression-specific configuration dictionary
PLANCHE_CONFIG: Dict[str, PlancheProgressionConfig] = {
    "FULL_PLANCHE": PlancheProgressionConfig(
        stage=ProgressionStage.FULL_PLANCHE,
        name="Full Planche",
        elbow_extension=PlancheMetricRule(
            name="Elbow Extension",
            target_range=">= 170.0°",
            ideal_min=170.0,
            acceptable_min=155.0,
            weight=0.25
        ),
        shoulder_position=PlancheMetricRule(
            name="Shoulder Forward Lean",
            target_range=">= 15.0°",
            ideal_min=15.0,
            acceptable_min=10.0,
            weight=0.20
        ),
        hip_extension=PlancheMetricRule(
            name="Hip Extension",
            target_range=">= 160.0°",
            ideal_min=160.0,
            acceptable_min=140.0,
            weight=0.20
        ),
        knee_extension=PlancheMetricRule(
            name="Knee Extension",
            target_range=">= 160.0°",
            ideal_min=160.0,
            acceptable_min=140.0,
            weight=0.15
        ),
        body_line=PlancheMetricRule(
            name="Body Line Tilt",
            target_range="<= 15.0°",
            ideal_max=15.0,
            acceptable_max=25.0,
            weight=0.20,
            is_upper_bounded=True
        ),
        leg_spread=PlancheMetricRule(
            name="Leg Adduction",
            target_range="< 1.25",
            ideal_max=1.0,
            acceptable_max=1.25,
            weight=0.0,
            is_upper_bounded=True
        ),
        min_hold_score=60.0
    ),
    "STRADDLE_PLANCHE": PlancheProgressionConfig(
        stage=ProgressionStage.STRADDLE_PLANCHE,
        name="Straddle Planche",
        elbow_extension=PlancheMetricRule(
            name="Elbow Extension",
            target_range=">= 170.0°",
            ideal_min=170.0,
            acceptable_min=155.0,
            weight=0.25
        ),
        shoulder_position=PlancheMetricRule(
            name="Shoulder Forward Lean",
            target_range=">= 15.0°",
            ideal_min=15.0,
            acceptable_min=10.0,
            weight=0.20
        ),
        hip_extension=PlancheMetricRule(
            name="Hip Extension",
            target_range=">= 160.0°",
            ideal_min=160.0,
            acceptable_min=140.0,
            weight=0.20
        ),
        knee_extension=PlancheMetricRule(
            name="Knee Extension",
            target_range=">= 160.0°",
            ideal_min=160.0,
            acceptable_min=140.0,
            weight=0.15
        ),
        body_line=PlancheMetricRule(
            name="Body Line Tilt",
            target_range="<= 15.0°",
            ideal_max=15.0,
            acceptable_max=25.0,
            weight=0.20,
            is_upper_bounded=True
        ),
        leg_spread=PlancheMetricRule(
            name="Leg Straddle Spread",
            target_range=">= 1.25",
            ideal_min=1.40,
            acceptable_min=1.25,
            weight=0.0
        ),
        min_hold_score=60.0
    ),
    "ADVANCED_TUCK_PLANCHE": PlancheProgressionConfig(
        stage=ProgressionStage.ADV_TUCK_PLANCHE,
        name="Advanced Tuck Planche",
        elbow_extension=PlancheMetricRule(
            name="Elbow Extension",
            target_range=">= 170.0°",
            ideal_min=170.0,
            acceptable_min=155.0,
            weight=0.25
        ),
        shoulder_position=PlancheMetricRule(
            name="Shoulder Forward Lean",
            target_range=">= 15.0°",
            ideal_min=15.0,
            acceptable_min=10.0,
            weight=0.20
        ),
        hip_extension=PlancheMetricRule(
            name="Hip Angle (90° Flat Back)",
            target_range="90.0° - 130.0°",
            ideal_min=90.0,
            ideal_max=130.0,
            acceptable_min=80.0,
            acceptable_max=140.0,
            weight=0.20
        ),
        knee_extension=PlancheMetricRule(
            name="Knee Angle (~90° Bent)",
            target_range="60.0° - 135.0°",
            ideal_min=60.0,
            ideal_max=135.0,
            acceptable_min=50.0,
            acceptable_max=145.0,
            weight=0.15
        ),
        body_line=PlancheMetricRule(
            name="Body Line Tilt",
            target_range="<= 15.0°",
            ideal_max=15.0,
            acceptable_max=25.0,
            weight=0.20,
            is_upper_bounded=True
        ),
        min_hold_score=60.0
    ),
    "TUCK_PLANCHE": PlancheProgressionConfig(
        stage=ProgressionStage.TUCK_PLANCHE,
        name="Tuck Planche",
        elbow_extension=PlancheMetricRule(
            name="Elbow Extension",
            target_range=">= 170.0°",
            ideal_min=170.0,
            acceptable_min=155.0,
            weight=0.25
        ),
        shoulder_position=PlancheMetricRule(
            name="Shoulder Forward Lean",
            target_range=">= 15.0°",
            ideal_min=15.0,
            acceptable_min=10.0,
            weight=0.20
        ),
        hip_extension=PlancheMetricRule(
            name="Hip Angle (Tucked to Chest)",
            target_range="<= 115.0°",
            ideal_max=115.0,
            acceptable_max=130.0,
            weight=0.20,
            is_upper_bounded=True
        ),
        knee_extension=PlancheMetricRule(
            name="Knee Angle (Tucked to Chest)",
            target_range="<= 105.0°",
            ideal_max=105.0,
            acceptable_max=120.0,
            weight=0.15,
            is_upper_bounded=True
        ),
        body_line=PlancheMetricRule(
            name="Body Line Tilt",
            target_range="<= 15.0°",
            ideal_max=15.0,
            acceptable_max=25.0,
            weight=0.20,
            is_upper_bounded=True
        ),
        min_hold_score=60.0
    ),
}

# Alias for ADV_TUCK_PLANCHE
PLANCHE_CONFIG["ADV_TUCK_PLANCHE"] = PLANCHE_CONFIG["ADVANCED_TUCK_PLANCHE"]


def get_planche_config(prog: Union[str, ProgressionStage]) -> PlancheProgressionConfig:
    """Retrieve progression config for a given stage or string."""
    if isinstance(prog, ProgressionStage):
        key = prog.name
    else:
        clean = str(prog).strip().upper().replace(" ", "_").replace("-", "_")
        if "ADV" in clean:
            key = "ADVANCED_TUCK_PLANCHE"
        elif "TUCK" in clean:
            key = "TUCK_PLANCHE"
        elif "STRADDLE" in clean:
            key = "STRADDLE_PLANCHE"
        else:
            key = "FULL_PLANCHE"

    if key in PLANCHE_CONFIG:
        return PLANCHE_CONFIG[key]
    return PLANCHE_CONFIG["FULL_PLANCHE"]
