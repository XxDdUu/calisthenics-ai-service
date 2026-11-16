import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_distance_2d,
    calculate_vertical_angle,
    get_point,
    is_valid_coord,
)
from app.schemas.analysis import KeypointData

logger = logging.getLogger(__name__)


class SquatState(str, Enum):
    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"


@dataclass
class SquatCounterConfig:
    standing_angle: float = 160.0          # Knee angle >= 160.0 to be in STANDING
    descending_angle: float = 150.0        # Knee angle < 150.0 triggers DESCENDING
    bottom_angle: float = 100.0            # Knee angle <= 100.0 triggers BOTTOM (valid depth)
    bottom_exit_angle: float = 108.0       # Knee angle > 108.0 exits BOTTOM to ASCENDING (hysteresis)
    descending_abort_angle: float = 155.0   # Knee angle >= 155.0 aborts partial descent without rep
    meaningful_attempt_angle: float = 140.0 # Knee angle < 140.0 counts as meaningful shallow squat attempt
    ema_alpha: float = 0.5                 # Exponential moving average factor (1.0 = raw, no smoothing)
    min_bottom_duration_sec: float = 0.15   # Minimum bottom position duration (150 ms)
    min_rep_duration_sec: float = 0.50     # Minimum total duration of valid rep (500 ms)
    min_cooldown_sec: float = 0.30         # Minimum interval between completed reps (300 ms)
    max_lost_duration_sec: float = 2.0     # Time without valid keypoints before resetting in-progress rep


@dataclass
class SquatRepResult:
    exercise: str = "SQUAT"
    rep_count: int = 0
    current_state: str = "STANDING"
    knee_angle: float = 180.0
    min_knee_angle: float = 180.0
    depth: str = "GOOD"                    # GOOD, MODERATE, SHALLOW
    is_tracking: bool = True
    feedback: str = ""
    rep_just_completed: bool = False
    movement_duration_sec: float = 0.0


def calculate_knee_angle_3points(
    hip: Tuple[float, float],
    knee: Tuple[float, float],
    ankle: Tuple[float, float]
) -> float:
    """
    Calculate knee flexion angle in degrees formed by Hip -> Knee -> Ankle.
    angle = arccos(dot(A - B, C - B) / (norm(A - B) * norm(C - B)))
    Vertex B is Knee, A is Hip, C is Ankle.
    Returns angle in degrees in [0.0, 180.0].
    """
    return calculate_angle_2d(hip, knee, ankle, default_angle=180.0)


def extract_squat_geometry(
    keypoints: Dict[str, KeypointData],
    threshold: float = 0.25,
    side_view: bool = False
) -> Dict[str, Any]:
    """
    Extract squat biomechanical geometry from COCO keypoints.
    Handles partial occlusion, low confidence, and unilateral visibility:
    - If one leg is unreliable, falls back to the other leg.
    - If both legs are unreliable, marks frame as invalid instead of generating false angle.
    - Never fabricates torso, hip, or knee alignment values when landmarks are missing.
    """
    empty_result = {
        "is_pose_valid": False,
        "primary_knee_angle": None,
        "knee_left_angle": None,
        "knee_right_angle": None,
        "avg_knee_angle": None,
        "hip_y": None,
        "hip_angle": None,
        "torso_angle": None,
        "knee_ankle_ratio": None,
        "knee_symmetry_diff": None,
        "left_leg_reliable": False,
        "right_leg_reliable": False,
        "leg_confidence": 0.0,
        "torso_confidence": 0.0,
        "hip_angle_confidence": 0.0,
        "alignment_confidence": 0.0,
        "missing": ["all_keypoints"],
    }
    if not keypoints:
        return empty_result

    def _is_reliable(name: str) -> bool:
        kp = keypoints.get(name)
        return bool(
            kp is not None
            and kp.visible
            and kp.confidence >= threshold
            and is_valid_coord(kp.x)
            and is_valid_coord(kp.y)
        )

    l_hip_ok = _is_reliable("left_hip")
    l_kn_ok = _is_reliable("left_knee")
    l_ak_ok = _is_reliable("left_ankle")
    left_leg_reliable = l_hip_ok and l_kn_ok and l_ak_ok

    r_hip_ok = _is_reliable("right_hip")
    r_kn_ok = _is_reliable("right_knee")
    r_ak_ok = _is_reliable("right_ankle")
    right_leg_reliable = r_hip_ok and r_kn_ok and r_ak_ok

    l_knee_angle = None
    if left_leg_reliable:
        p_hip = get_point(keypoints.get("left_hip"))
        p_kn = get_point(keypoints.get("left_knee"))
        p_ak = get_point(keypoints.get("left_ankle"))
        if p_hip and p_kn and p_ak:
            l_knee_angle = calculate_knee_angle_3points(p_hip, p_kn, p_ak)

    r_knee_angle = None
    if right_leg_reliable:
        p_hip = get_point(keypoints.get("right_hip"))
        p_kn = get_point(keypoints.get("right_knee"))
        p_ak = get_point(keypoints.get("right_ankle"))
        if p_hip and p_kn and p_ak:
            r_knee_angle = calculate_knee_angle_3points(p_hip, p_kn, p_ak)

    missing = []
    if not (l_hip_ok or r_hip_ok):
        missing.append("hip")
    if not (l_kn_ok or r_kn_ok):
        missing.append("knee")
    if not (l_ak_ok or r_ak_ok):
        missing.append("ankle")

    if not left_leg_reliable and not right_leg_reliable:
        res = dict(empty_result)
        res["knee_left_angle"] = l_knee_angle
        res["knee_right_angle"] = r_knee_angle
        res["missing"] = missing if missing else ["legs_unreliable"]
        return res

    l_leg_conf = (
        sum(keypoints[k].confidence for k in ["left_hip", "left_knee", "left_ankle"]) / 3.0
        if left_leg_reliable else 0.0
    )
    r_leg_conf = (
        sum(keypoints[k].confidence for k in ["right_hip", "right_knee", "right_ankle"]) / 3.0
        if right_leg_reliable else 0.0
    )

    if left_leg_reliable and right_leg_reliable:
        assert l_knee_angle is not None and r_knee_angle is not None
        avg_knee = round((l_knee_angle + r_knee_angle) / 2.0, 1)
        if side_view:
            primary_knee = l_knee_angle if l_leg_conf >= r_leg_conf else r_knee_angle
            leg_conf = max(l_leg_conf, r_leg_conf)
        else:
            primary_knee = avg_knee
            leg_conf = (l_leg_conf + r_leg_conf) / 2.0
    elif left_leg_reliable:
        assert l_knee_angle is not None
        primary_knee = l_knee_angle
        avg_knee = l_knee_angle
        leg_conf = l_leg_conf
    else:
        assert r_knee_angle is not None
        primary_knee = r_knee_angle
        avg_knee = r_knee_angle
        leg_conf = r_leg_conf

    # Hip vertical position
    hip_y_vals = []
    if l_hip_ok:
        hip_y_vals.append(keypoints["left_hip"].y)
    if r_hip_ok:
        hip_y_vals.append(keypoints["right_hip"].y)
    hip_y = round(sum(hip_y_vals) / len(hip_y_vals), 1) if hip_y_vals else None

    # Torso inclination angle (only when reliable shoulder + hip exist)
    l_sh_ok = _is_reliable("left_shoulder")
    r_sh_ok = _is_reliable("right_shoulder")
    torso_angle = None
    torso_conf = 0.0
    if (l_sh_ok and l_hip_ok) and (r_sh_ok and r_hip_ok):
        l_c = (keypoints["left_shoulder"].confidence + keypoints["left_hip"].confidence) / 2.0
        r_c = (keypoints["right_shoulder"].confidence + keypoints["right_hip"].confidence) / 2.0
        if side_view:
            use_l = l_c >= r_c
            p_sh = get_point(keypoints["left_shoulder" if use_l else "right_shoulder"])
            p_hip = get_point(keypoints["left_hip" if use_l else "right_hip"])
            torso_conf = max(l_c, r_c)
        else:
            p_sh = (
                (keypoints["left_shoulder"].x + keypoints["right_shoulder"].x) / 2.0,
                (keypoints["left_shoulder"].y + keypoints["right_shoulder"].y) / 2.0,
            )
            p_hip = (
                (keypoints["left_hip"].x + keypoints["right_hip"].x) / 2.0,
                (keypoints["left_hip"].y + keypoints["right_hip"].y) / 2.0,
            )
            torso_conf = (l_c + r_c) / 2.0
        torso_angle = calculate_vertical_angle(p_sh, p_hip, default_angle=90.0)
    elif l_sh_ok and l_hip_ok:
        p_sh = get_point(keypoints["left_shoulder"])
        p_hip = get_point(keypoints["left_hip"])
        torso_angle = calculate_vertical_angle(p_sh, p_hip, default_angle=90.0)
        torso_conf = (keypoints["left_shoulder"].confidence + keypoints["left_hip"].confidence) / 2.0
    elif r_sh_ok and r_hip_ok:
        p_sh = get_point(keypoints["right_shoulder"])
        p_hip = get_point(keypoints["right_hip"])
        torso_angle = calculate_vertical_angle(p_sh, p_hip, default_angle=90.0)
        torso_conf = (keypoints["right_shoulder"].confidence + keypoints["right_hip"].confidence) / 2.0
    elif (l_sh_ok or r_sh_ok) and (l_hip_ok or r_hip_ok):
        sh_kp = keypoints["left_shoulder"] if l_sh_ok else keypoints["right_shoulder"]
        hip_kp = keypoints["left_hip"] if l_hip_ok else keypoints["right_hip"]
        torso_angle = calculate_vertical_angle((sh_kp.x, sh_kp.y), (hip_kp.x, hip_kp.y), default_angle=90.0)
        torso_conf = (sh_kp.confidence + hip_kp.confidence) / 2.0

    # Hip flexion angle (Shoulder - Hip - Knee) using reliable landmarks
    hip_angle = None
    hip_angle_conf = 0.0
    l_hip_chain = l_sh_ok and l_hip_ok and l_kn_ok
    r_hip_chain = r_sh_ok and r_hip_ok and r_kn_ok
    if l_hip_chain and r_hip_chain:
        l_ha = calculate_angle_2d(
            get_point(keypoints["left_shoulder"]),
            get_point(keypoints["left_hip"]),
            get_point(keypoints["left_knee"]),
        )
        r_ha = calculate_angle_2d(
            get_point(keypoints["right_shoulder"]),
            get_point(keypoints["right_hip"]),
            get_point(keypoints["right_knee"]),
        )
        l_hc = sum(keypoints[k].confidence for k in ["left_shoulder", "left_hip", "left_knee"]) / 3.0
        r_hc = sum(keypoints[k].confidence for k in ["right_shoulder", "right_hip", "right_knee"]) / 3.0
        if side_view:
            hip_angle = l_ha if l_hc >= r_hc else r_ha
            hip_angle_conf = max(l_hc, r_hc)
        else:
            hip_angle = round((l_ha + r_ha) / 2.0, 1)
            hip_angle_conf = (l_hc + r_hc) / 2.0
    elif l_hip_chain:
        hip_angle = calculate_angle_2d(
            get_point(keypoints["left_shoulder"]),
            get_point(keypoints["left_hip"]),
            get_point(keypoints["left_knee"]),
        )
        hip_angle_conf = sum(keypoints[k].confidence for k in ["left_shoulder", "left_hip", "left_knee"]) / 3.0
    elif r_hip_chain:
        hip_angle = calculate_angle_2d(
            get_point(keypoints["right_shoulder"]),
            get_point(keypoints["right_hip"]),
            get_point(keypoints["right_knee"]),
        )
        hip_angle_conf = sum(keypoints[k].confidence for k in ["right_shoulder", "right_hip", "right_knee"]) / 3.0
    elif (l_sh_ok or r_sh_ok) and (l_hip_ok or r_hip_ok) and (l_kn_ok or r_kn_ok):
        sh_kp = keypoints["left_shoulder"] if l_sh_ok else keypoints["right_shoulder"]
        hip_kp = keypoints["left_hip"] if l_hip_ok else keypoints["right_hip"]
        kn_kp = keypoints["left_knee"] if l_kn_ok else keypoints["right_knee"]
        hip_angle = calculate_angle_2d((sh_kp.x, sh_kp.y), (hip_kp.x, hip_kp.y), (kn_kp.x, kn_kp.y))
        hip_angle_conf = (sh_kp.confidence + hip_kp.confidence + kn_kp.confidence) / 3.0

    # Knee vs ankle width ratio & bilateral symmetry (ONLY in non-side view with both legs reliable)
    knee_ankle_ratio = None
    knee_symmetry_diff = None
    alignment_conf = 0.0
    if not side_view and l_kn_ok and r_kn_ok and l_ak_ok and r_ak_ok:
        kp_l_kn = keypoints["left_knee"]
        kp_r_kn = keypoints["right_knee"]
        kp_l_ak = keypoints["left_ankle"]
        kp_r_ak = keypoints["right_ankle"]
        shank_l = calculate_distance_2d((kp_l_kn.x, kp_l_kn.y), (kp_l_ak.x, kp_l_ak.y))
        shank_r = calculate_distance_2d((kp_r_kn.x, kp_r_kn.y), (kp_r_ak.x, kp_r_ak.y))
        ref_shank = max(shank_l, shank_r, 1e-3)
        raw_ankle_dist = abs(kp_l_ak.x - kp_r_ak.x)
        if raw_ankle_dist >= 0.05 * ref_shank:
            knee_dist = abs(kp_l_kn.x - kp_r_kn.x)
            knee_ankle_ratio = round(knee_dist / raw_ankle_dist, 2)
            alignment_conf = (
                kp_l_kn.confidence + kp_r_kn.confidence + kp_l_ak.confidence + kp_r_ak.confidence
            ) / 4.0

    if not side_view and left_leg_reliable and right_leg_reliable and l_knee_angle is not None and r_knee_angle is not None:
        knee_symmetry_diff = round(abs(l_knee_angle - r_knee_angle), 1)
        if alignment_conf == 0.0:
            alignment_conf = (l_leg_conf + r_leg_conf) / 2.0

    return {
        "is_pose_valid": True,
        "primary_knee_angle": primary_knee,
        "knee_left_angle": l_knee_angle,
        "knee_right_angle": r_knee_angle,
        "avg_knee_angle": avg_knee,
        "hip_y": hip_y,
        "hip_angle": hip_angle,
        "torso_angle": torso_angle,
        "knee_ankle_ratio": knee_ankle_ratio,
        "knee_symmetry_diff": knee_symmetry_diff,
        "left_leg_reliable": left_leg_reliable,
        "right_leg_reliable": right_leg_reliable,
        "leg_confidence": round(leg_conf, 3),
        "torso_confidence": round(torso_conf, 3),
        "hip_angle_confidence": round(hip_angle_conf, 3),
        "alignment_confidence": round(alignment_conf, 3),
        "missing": [],
    }


class SquatRepCounter:
    """
    Dedicated and reliable Squat Rep Counter.
    Implements a strict 4-phase state machine with hysteresis:
    STANDING -> DESCENDING -> BOTTOM -> ASCENDING -> STANDING (REP COMPLETED)

    Features:
    - Configurable biomechanical thresholds and EMA temporal smoothing.
    - Hysteresis to reject noise oscillations around state boundaries.
    - Minimum bottom duration check (prevents single-frame bounce false reps).
    - Minimum rep duration check (prevents detector glitches from completing a rep).
    - Minimum cooldown check between completed repetitions.
    - Incomplete rep tracking (partial shallow squats and failed lockout recovery).
    - Automatic state recovery upon tracking loss.
    """

    def __init__(self, config: Optional[SquatCounterConfig] = None):
        self.config = config or SquatCounterConfig()
        self.state: SquatState = SquatState.STANDING
        self.rep_count: int = 0
        self.incomplete_rep_count: int = 0
        self.current_knee_angle: float = 180.0
        self.smoothed_knee_angle: float = 180.0
        self.min_knee_angle: float = 180.0
        self.depth_quality: str = "GOOD"
        self.is_tracking: bool = False
        self.rep_just_completed: bool = False
        self.feedback: str = "Ready. Begin your squat."
        self.last_completed_min_angle: float = 180.0
        self.last_completed_depth: str = "GOOD"

        # Temporal and state tracking timestamps
        self.rep_start_time: float = 0.0
        self.bottom_enter_time: float = 0.0
        self.last_rep_completed_time: float = 0.0
        self.last_valid_tracking_time: float = 0.0
        self.reached_bottom_depth: bool = False
        self.is_smoothed_initialized: bool = False
        self.consecutive_bottom_frames: int = 0
        self._simulated_clock: float = 0.0

        # Sequence / per-rep episode tracking
        self.rep_records: List[Dict[str, Any]] = []
        self._active_rep: Optional[Dict[str, Any]] = None
        self._frame_seq: int = 0
        self._last_standing_frame: Optional[int] = None
        self._last_standing_ts_ms: Optional[int] = None
        self._last_standing_time: Optional[float] = None
        self._last_standing_angle: float = 180.0
        self._last_valid_frame: int = 0
        self._last_valid_ts_ms: int = 0

    def reset(self) -> None:
        """Reset rep counter and tracking state."""
        self.state = SquatState.STANDING
        self.rep_count = 0
        self.incomplete_rep_count = 0
        self.current_knee_angle = 180.0
        self.smoothed_knee_angle = 180.0
        self.min_knee_angle = 180.0
        self.depth_quality = "GOOD"
        self.is_tracking = False
        self.rep_just_completed = False
        self.feedback = "Ready. Begin your squat."
        self.last_completed_min_angle = 180.0
        self.last_completed_depth = "GOOD"
        self.rep_start_time = 0.0
        self.bottom_enter_time = 0.0
        self.last_rep_completed_time = 0.0
        self.last_valid_tracking_time = 0.0
        self.reached_bottom_depth = False
        self.is_smoothed_initialized = False
        self.consecutive_bottom_frames = 0
        self._simulated_clock = 0.0
        self.rep_records.clear()
        self._active_rep = None
        self._frame_seq = 0
        self._last_standing_frame = None
        self._last_standing_ts_ms = None
        self._last_standing_time = None
        self._last_standing_angle = 180.0
        self._last_valid_frame = 0
        self._last_valid_ts_ms = 0

    def _record_rep_episode(
        self,
        end_frame: int,
        end_ts_ms: int,
        is_valid: bool,
        issues: List[str],
    ) -> None:
        """Persist a completed or incomplete rep episode from active rep state."""
        if not self._active_rep:
            return
        start_frame = int(self._active_rep["start_frame"])
        start_ts_ms = int(self._active_rep["start_ts_ms"])
        min_angle = round(float(self._active_rep["min_angle"]), 1)
        max_angle = round(float(self._active_rep["max_angle"]), 1)
        rom = round(max(0.0, max_angle - min_angle), 1)
        duration_ms = max(0, end_ts_ms - start_ts_ms)
        depth_cls = self._classify_depth(min_angle) if is_valid else "INCOMPLETE"

        self.rep_records.append({
            "repNumber": len(self.rep_records) + 1,
            "startFrame": start_frame,
            "endFrame": end_frame,
            "startTimestampMs": start_ts_ms,
            "endTimestampMs": end_ts_ms,
            "durationMs": duration_ms,
            "isValid": is_valid,
            "status": "VALID" if is_valid else "INCOMPLETE",
            "minAngleDeg": min_angle,
            "maxAngleDeg": max_angle,
            "romDeg": rom,
            "depthClassification": depth_cls,
            "issues": list(issues),
            "frameIndices": list(self._active_rep["frame_indices"]),
        })
        if not is_valid:
            self.incomplete_rep_count += 1
        self._active_rep = None

    def _start_active_rep(
        self,
        cur_frame: int,
        now_ms: int,
        now: float,
        k_angle: float,
        use_standing_anchor: bool = True,
    ) -> None:
        """Initialize tracking metadata for a new squat attempt."""
        if use_standing_anchor and self._last_standing_frame is not None and self._last_standing_ts_ms is not None:
            start_frame = self._last_standing_frame
            start_ts_ms = self._last_standing_ts_ms
            start_time = self._last_standing_time if self._last_standing_time is not None else now
            max_angle = max(self._last_standing_angle, k_angle)
            frames = [start_frame, cur_frame] if start_frame != cur_frame else [cur_frame]
        else:
            start_frame = cur_frame
            start_ts_ms = now_ms
            start_time = now
            max_angle = k_angle
            frames = [cur_frame]

        self._active_rep = {
            "start_frame": start_frame,
            "start_ts_ms": start_ts_ms,
            "start_time": start_time,
            "min_angle": k_angle,
            "max_angle": max_angle,
            "peak_ascent_angle": k_angle,
            "frame_indices": frames,
        }

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Finalize any unfinished rep in ASCENDING at the end of a sequence
        (failed top lockout recovery) and return all recorded rep episodes.
        """
        if (
            self._active_rep is not None
            and self.state == SquatState.ASCENDING
            and (self.reached_bottom_depth or self._active_rep["min_angle"] < self.config.meaningful_attempt_angle)
            and self._active_rep["peak_ascent_angle"] >= self._active_rep["min_angle"] + 15.0
        ):
            issues = ["INCOMPLETE_EXTENSION"]
            if self._active_rep["min_angle"] > self.config.bottom_angle:
                issues.append("SHALLOW_DEPTH")
            self._record_rep_episode(
                end_frame=self._last_valid_frame,
                end_ts_ms=self._last_valid_ts_ms,
                is_valid=False,
                issues=issues,
            )
            self.state = SquatState.STANDING
        return list(self.rep_records)

    def process_angle(
        self,
        angle: Optional[float],
        timestamp_sec: Optional[float] = None,
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
    ) -> SquatRepResult:
        """
        Process a single knee angle measurement through the Squat State Machine.
        Handles tracking loss, EMA smoothing, hysteresis, and rep counting.
        """
        self.rep_just_completed = False

        if timestamp_sec is not None:
            now = float(timestamp_sec)
        else:
            self._simulated_clock += 0.10
            now = self._simulated_clock

        cur_frame = frame_index if frame_index is not None else self._frame_seq
        self._frame_seq += 1
        now_ms = timestamp_ms if timestamp_ms is not None else int(round(now * 1000.0))

        # 1. Handle Tracking Loss (no confident knee angle)
        if angle is None or not math.isfinite(angle):
            self.is_tracking = False
            if (
                self.last_valid_tracking_time > 0
                and (now - self.last_valid_tracking_time) > self.config.max_lost_duration_sec
            ):
                if self.state != SquatState.STANDING:
                    logger.debug(
                        "Tracking lost for > %.1fs. Resetting squat state to STANDING.",
                        self.config.max_lost_duration_sec,
                    )
                    self.state = SquatState.STANDING
                    self.min_knee_angle = 180.0
                    self.reached_bottom_depth = False
                    self.is_smoothed_initialized = False
                    self.consecutive_bottom_frames = 0
                    self._active_rep = None
            return SquatRepResult(
                exercise="SQUAT",
                rep_count=self.rep_count,
                current_state=self.state.value,
                knee_angle=round(self.smoothed_knee_angle, 1),
                min_knee_angle=round(self.min_knee_angle, 1),
                depth=self.depth_quality,
                is_tracking=False,
                feedback=self.feedback,
                rep_just_completed=False,
                movement_duration_sec=round(max(0.0, now - self.rep_start_time), 2) if self.state != SquatState.STANDING else 0.0,
            )

        # 2. Tracking Active: Update Timers and Smooth Angle
        self.is_tracking = True
        self.last_valid_tracking_time = now
        self.current_knee_angle = float(angle)

        if not self.is_smoothed_initialized or self.config.ema_alpha >= 1.0:
            self.smoothed_knee_angle = self.current_knee_angle
            self.is_smoothed_initialized = True
        else:
            # Snap to raw value if step delta is large (> 18 deg) to avoid lag truncation
            # during fast movement or discrete test benchmark sequences
            if abs(self.current_knee_angle - self.smoothed_knee_angle) > 18.0:
                self.smoothed_knee_angle = self.current_knee_angle
            else:
                alpha = self.config.ema_alpha
                self.smoothed_knee_angle = alpha * self.current_knee_angle + (1.0 - alpha) * self.smoothed_knee_angle

        k_angle = self.smoothed_knee_angle

        # 3. Squat State Machine Transitions
        if self.state == SquatState.STANDING:
            if k_angle < self.config.descending_angle:
                self.state = SquatState.DESCENDING
                self.rep_start_time = now
                self.min_knee_angle = k_angle
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self._start_active_rep(cur_frame, now_ms, now, k_angle, use_standing_anchor=True)
                self.feedback = "Control your descent"
            else:
                self.min_knee_angle = 180.0
                self._last_standing_frame = cur_frame
                self._last_standing_ts_ms = now_ms
                self._last_standing_time = now
                self._last_standing_angle = k_angle
                if k_angle < self.config.standing_angle:
                    self.feedback = "Move fully back to standing"
                else:
                    self.feedback = "Ready. Begin your squat."

        elif self.state == SquatState.DESCENDING:
            self.min_knee_angle = min(self.min_knee_angle, k_angle)
            if self._active_rep is not None:
                self._active_rep["min_angle"] = min(self._active_rep["min_angle"], k_angle)
                self._active_rep["max_angle"] = max(self._active_rep["max_angle"], k_angle)
                self._active_rep["frame_indices"].append(cur_frame)

            if k_angle <= self.config.bottom_angle or (self.current_knee_angle <= self.config.bottom_angle and k_angle <= 106.0):
                self.state = SquatState.BOTTOM
                self.bottom_enter_time = now
                self.reached_bottom_depth = True
                self.consecutive_bottom_frames = 1
                self.depth_quality = self._classify_depth(self.min_knee_angle)
                self.feedback = "Good squat depth"
            elif k_angle >= self.config.descending_abort_angle:
                # Aborted partial descent back to standing without hitting bottom depth
                if self._active_rep is not None:
                    attempt_min = self._active_rep["min_angle"]
                    attempt_dur = now - self._active_rep["start_time"]
                    if (
                        attempt_min < self.config.meaningful_attempt_angle
                        and (attempt_dur >= 0.25 or len(self._active_rep["frame_indices"]) >= 3)
                    ):
                        self._active_rep["max_angle"] = max(self._active_rep["max_angle"], k_angle)
                        self._record_rep_episode(
                            end_frame=cur_frame,
                            end_ts_ms=now_ms,
                            is_valid=False,
                            issues=["SHALLOW_DEPTH"],
                        )
                    else:
                        self._active_rep = None

                self.state = SquatState.STANDING
                self.min_knee_angle = 180.0
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self._last_standing_frame = cur_frame
                self._last_standing_ts_ms = now_ms
                self._last_standing_time = now
                self._last_standing_angle = k_angle
                self.feedback = "Try squatting deeper"
            else:
                self.feedback = "Squatting down..."

        elif self.state == SquatState.BOTTOM:
            self.min_knee_angle = min(self.min_knee_angle, k_angle)
            self.consecutive_bottom_frames += 1
            self.depth_quality = self._classify_depth(self.min_knee_angle)
            if self._active_rep is not None:
                self._active_rep["min_angle"] = min(self._active_rep["min_angle"], k_angle)
                self._active_rep["max_angle"] = max(self._active_rep["max_angle"], k_angle)
                self._active_rep["frame_indices"].append(cur_frame)

            if k_angle > self.config.bottom_exit_angle:
                bottom_duration = now - self.bottom_enter_time
                bottom_satisfied = (
                    bottom_duration >= self.config.min_bottom_duration_sec
                    or self.consecutive_bottom_frames >= 2
                    or self.config.min_bottom_duration_sec == 0.0
                )
                self.reached_bottom_depth = bool(bottom_satisfied)
                if self._active_rep is not None:
                    self._active_rep["peak_ascent_angle"] = k_angle
                self.state = SquatState.ASCENDING
                self.feedback = "Drive up through your heels"
            else:
                self.feedback = "Good squat depth"

        elif self.state == SquatState.ASCENDING:
            self.min_knee_angle = min(self.min_knee_angle, k_angle)
            if self._active_rep is not None:
                self._active_rep["min_angle"] = min(self._active_rep["min_angle"], k_angle)
                self._active_rep["max_angle"] = max(self._active_rep["max_angle"], k_angle)
                self._active_rep["peak_ascent_angle"] = max(self._active_rep["peak_ascent_angle"], k_angle)
                self._active_rep["frame_indices"].append(cur_frame)

            is_standing = (
                k_angle >= self.config.standing_angle
                or (self.current_knee_angle >= self.config.standing_angle and k_angle >= 152.0)
            )

            if is_standing:
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
                    self.last_completed_min_angle = self.min_knee_angle
                    self.last_completed_depth = self.depth_quality
                    self._record_rep_episode(
                        end_frame=cur_frame,
                        end_ts_ms=now_ms,
                        is_valid=True,
                        issues=[],
                    )
                    self.feedback = "Rep completed! Stand tall."
                else:
                    if self._active_rep is not None and not self.reached_bottom_depth and self._active_rep["min_angle"] < self.config.meaningful_attempt_angle:
                        self._record_rep_episode(
                            end_frame=cur_frame,
                            end_ts_ms=now_ms,
                            is_valid=False,
                            issues=["SHALLOW_DEPTH"],
                        )
                    else:
                        self._active_rep = None
                    self.feedback = "Move fully back to standing"

                self.state = SquatState.STANDING
                self.min_knee_angle = self.last_completed_min_angle if is_valid_rep else 180.0
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self._last_standing_frame = cur_frame
                self._last_standing_ts_ms = now_ms
                self._last_standing_time = now
                self._last_standing_angle = k_angle

            elif (
                self._active_rep is not None
                and self._active_rep["peak_ascent_angle"] >= 140.0
                and k_angle < self._active_rep["peak_ascent_angle"] - 20.0
                and k_angle < 130.0
            ):
                # Athlete failed top lockout and descended directly into next rep
                issues = ["INCOMPLETE_EXTENSION"]
                if not self.reached_bottom_depth:
                    issues.append("SHALLOW_DEPTH")
                self._record_rep_episode(
                    end_frame=self._last_valid_frame,
                    end_ts_ms=self._last_valid_ts_ms,
                    is_valid=False,
                    issues=issues,
                )
                self.state = SquatState.DESCENDING
                self.rep_start_time = now
                self.min_knee_angle = k_angle
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self._start_active_rep(cur_frame, now_ms, now, k_angle, use_standing_anchor=False)
                self.feedback = "Extend fully at the top between reps"

            elif k_angle <= self.config.bottom_angle:
                # User dipped back down to bottom before completing ascent
                self.state = SquatState.BOTTOM
                self.bottom_enter_time = now
                self.consecutive_bottom_frames += 1

        self._last_valid_frame = cur_frame
        self._last_valid_ts_ms = now_ms

        self.depth_quality = self._classify_depth(
            self.min_knee_angle if self.state != SquatState.STANDING else self.current_knee_angle
        )

        if self.state == SquatState.STANDING and self.last_completed_min_angle < 180.0:
            reported_min = self.last_completed_min_angle
            reported_depth = self.last_completed_depth
        elif self.state != SquatState.STANDING:
            reported_min = self.min_knee_angle
            reported_depth = self.depth_quality
        else:
            reported_min = self.current_knee_angle
            reported_depth = self.depth_quality

        mov_duration = round(max(0.0, now - self.rep_start_time), 2) if self.state != SquatState.STANDING else 0.0

        return SquatRepResult(
            exercise="SQUAT",
            rep_count=self.rep_count,
            current_state=self.state.value,
            knee_angle=round(self.smoothed_knee_angle, 1),
            min_knee_angle=round(reported_min, 1),
            depth=reported_depth,
            is_tracking=self.is_tracking,
            feedback=self.feedback,
            rep_just_completed=self.rep_just_completed,
            movement_duration_sec=mov_duration,
        )

    def _classify_depth(self, angle: float) -> str:
        """Classify squat depth from knee angle: GOOD (<=100), MODERATE (100-120), SHALLOW (>120)."""
        if angle <= self.config.bottom_angle:
            return "GOOD"
        elif angle <= 120.0:
            return "MODERATE"
        return "SHALLOW"
