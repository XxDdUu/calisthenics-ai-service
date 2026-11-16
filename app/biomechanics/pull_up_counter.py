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


class PullUpState(str, Enum):
    DEAD_HANG = "DEAD_HANG"
    PULLING = "PULLING"
    TOP_POSITION = "TOP_POSITION"
    LOWERING = "LOWERING"


@dataclass
class PullUpCounterConfig:
    dead_hang_angle: float = 150.0             # Elbow angle >= 150.0 is DEAD_HANG
    pulling_enter_angle: float = 140.0         # Elbow angle < 140.0 enters PULLING
    top_enter_angle: float = 85.0              # Elbow angle <= 85.0 enters TOP_POSITION
    top_exit_angle: float = 95.0               # Elbow angle > 95.0 exits TOP_POSITION to LOWERING
    dead_hang_return_angle: float = 148.0      # Elbow angle >= 148.0 returns from LOWERING to DEAD_HANG
    pulling_abort_angle: float = 148.0         # Elbow angle >= 148.0 aborts partial pull back to DEAD_HANG
    incomplete_pull_threshold: float = 130.0   # Partial pull < 130.0 without reaching <= 90.0 -> INCOMPLETE_PULL
    full_extension_min_angle: float = 145.0    # Bottom dead-hang extension threshold before next pull
    ema_alpha: float = 0.50                    # Exponential moving average factor
    min_top_duration_sec: float = 0.12         # Minimum top hold duration (120 ms)
    min_rep_duration_sec: float = 0.50         # Minimum total duration of valid rep (500 ms)
    min_cooldown_sec: float = 0.30             # Minimum interval between completed reps (300 ms)
    max_lost_duration_sec: float = 2.0         # Time without valid tracking before resetting in-progress rep


@dataclass
class PullUpRepResult:
    exercise: str = "PULL_UP"
    rep_count: int = 0
    incomplete_rep_count: int = 0
    current_state: str = "DEAD_HANG"
    elbow_angle: float = 180.0
    min_elbow_angle: float = 180.0
    max_elbow_angle: float = 180.0
    depth: str = "GOOD"                        # GOOD, MODERATE, SHALLOW
    is_tracking: bool = True
    feedback: str = ""
    rep_just_completed: bool = False
    movement_duration_sec: float = 0.0


def extract_pull_up_geometry(
    keypoints: Dict[str, KeypointData],
    threshold: float = 0.25,
    side_view: bool = False,
) -> Dict[str, Any]:
    """
    Extract pull-up biomechanical measurements from COCO 17 keypoints.
    Handles side-view / unilateral arm occlusion, chin-to-bar vertical clearance,
    shoulder angle (when hips visible), and lower-body pendulum swing (when lower body visible).
    Never fabricates angles or clearance when landmarks are missing or low-confidence.
    """
    empty_result = {
        "is_pose_valid": False,
        "primary_elbow_angle": None,
        "left_elbow_angle": None,
        "right_elbow_angle": None,
        "elbow_symmetry_diff": None,
        "shoulder_angle": None,
        "chin_to_bar_clearance": None,
        "hip_swing_offset": None,
        "ankle_swing_offset": None,
        "torso_tilt": None,
        "lower_body_reliable": False,
        "left_arm_reliable": False,
        "right_arm_reliable": False,
        "arm_confidence": 0.0,
        "shoulder_angle_confidence": 0.0,
        "chin_to_bar_confidence": 0.0,
        "body_swing_confidence": 0.0,
        "symmetry_confidence": 0.0,
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

    def _pt(name: str) -> Optional[Tuple[float, float]]:
        if _is_reliable(name):
            return get_point(keypoints[name])
        return None

    l_sh_ok = _is_reliable("left_shoulder")
    l_el_ok = _is_reliable("left_elbow")
    l_wr_ok = _is_reliable("left_wrist")
    left_arm_reliable = l_sh_ok and l_el_ok and l_wr_ok

    r_sh_ok = _is_reliable("right_shoulder")
    r_el_ok = _is_reliable("right_elbow")
    r_wr_ok = _is_reliable("right_wrist")
    right_arm_reliable = r_sh_ok and r_el_ok and r_wr_ok

    l_elbow_angle = None
    if left_arm_reliable:
        l_elbow_angle = calculate_angle_2d(_pt("left_shoulder"), _pt("left_elbow"), _pt("left_wrist"))

    r_elbow_angle = None
    if right_arm_reliable:
        r_elbow_angle = calculate_angle_2d(_pt("right_shoulder"), _pt("right_elbow"), _pt("right_wrist"))

    missing = []
    if not (l_sh_ok or r_sh_ok):
        missing.append("shoulder")
    if not (l_el_ok or r_el_ok):
        missing.append("elbow")
    if not (l_wr_ok or r_wr_ok):
        missing.append("wrist")

    if not left_arm_reliable and not right_arm_reliable:
        res = dict(empty_result)
        res["left_elbow_angle"] = l_elbow_angle
        res["right_elbow_angle"] = r_elbow_angle
        res["missing"] = missing if missing else ["arms_unreliable"]
        return res

    l_arm_conf = (
        sum(keypoints[k].confidence for k in ["left_shoulder", "left_elbow", "left_wrist"]) / 3.0
        if left_arm_reliable else 0.0
    )
    r_arm_conf = (
        sum(keypoints[k].confidence for k in ["right_shoulder", "right_elbow", "right_wrist"]) / 3.0
        if right_arm_reliable else 0.0
    )

    if left_arm_reliable and right_arm_reliable:
        assert l_elbow_angle is not None and r_elbow_angle is not None
        avg_elbow = round((l_elbow_angle + r_elbow_angle) / 2.0, 1)
        if side_view:
            active_side = "left" if l_arm_conf >= r_arm_conf else "right"
            primary_elbow = l_elbow_angle if active_side == "left" else r_elbow_angle
            arm_conf = max(l_arm_conf, r_arm_conf)
        else:
            active_side = "left" if l_arm_conf >= r_arm_conf else "right"
            primary_elbow = avg_elbow
            arm_conf = (l_arm_conf + r_arm_conf) / 2.0
    elif left_arm_reliable:
        assert l_elbow_angle is not None
        active_side = "left"
        primary_elbow = l_elbow_angle
        arm_conf = l_arm_conf
    else:
        assert r_elbow_angle is not None
        active_side = "right"
        primary_elbow = r_elbow_angle
        arm_conf = r_arm_conf

    # Elbow symmetry (ONLY in non-side view with both arms reliable)
    elbow_symmetry_diff = None
    symmetry_conf = 0.0
    if not side_view and left_arm_reliable and right_arm_reliable and l_elbow_angle is not None and r_elbow_angle is not None:
        elbow_symmetry_diff = round(abs(l_elbow_angle - r_elbow_angle), 1)
        symmetry_conf = (l_arm_conf + r_arm_conf) / 2.0

    # Hip landmarks for shoulder angle & torso scale
    l_hip_ok = _is_reliable("left_hip")
    r_hip_ok = _is_reliable("right_hip")
    l_hip_pt = _pt("left_hip")
    r_hip_pt = _pt("right_hip")

    # Shoulder angle (Hip - Shoulder - Elbow)
    shoulder_angle = None
    shoulder_angle_conf = 0.0
    l_sh_chain = l_hip_ok and l_sh_ok and l_el_ok
    r_sh_chain = r_hip_ok and r_sh_ok and r_el_ok
    if l_sh_chain and r_sh_chain:
        l_sa = calculate_angle_2d(l_hip_pt, _pt("left_shoulder"), _pt("left_elbow"))
        r_sa = calculate_angle_2d(r_hip_pt, _pt("right_shoulder"), _pt("right_elbow"))
        l_sc = sum(keypoints[k].confidence for k in ["left_hip", "left_shoulder", "left_elbow"]) / 3.0
        r_sc = sum(keypoints[k].confidence for k in ["right_hip", "right_shoulder", "right_elbow"]) / 3.0
        if side_view:
            shoulder_angle = l_sa if l_sc >= r_sc else r_sa
            shoulder_angle_conf = max(l_sc, r_sc)
        else:
            shoulder_angle = round((l_sa + r_sa) / 2.0, 1)
            shoulder_angle_conf = (l_sc + r_sc) / 2.0
    elif l_sh_chain:
        shoulder_angle = calculate_angle_2d(l_hip_pt, _pt("left_shoulder"), _pt("left_elbow"))
        shoulder_angle_conf = sum(keypoints[k].confidence for k in ["left_hip", "left_shoulder", "left_elbow"]) / 3.0
    elif r_sh_chain:
        shoulder_angle = calculate_angle_2d(r_hip_pt, _pt("right_shoulder"), _pt("right_elbow"))
        shoulder_angle_conf = sum(keypoints[k].confidence for k in ["right_hip", "right_shoulder", "right_elbow"]) / 3.0
    elif (l_hip_ok or r_hip_ok) and (l_sh_ok or r_sh_ok) and (l_el_ok or r_el_ok):
        hip_k = keypoints["left_hip"] if l_hip_ok else keypoints["right_hip"]
        sh_k = keypoints["left_shoulder"] if l_sh_ok else keypoints["right_shoulder"]
        el_k = keypoints["left_elbow"] if l_el_ok else keypoints["right_elbow"]
        shoulder_angle = calculate_angle_2d((hip_k.x, hip_k.y), (sh_k.x, sh_k.y), (el_k.x, el_k.y))
        shoulder_angle_conf = (hip_k.confidence + sh_k.confidence + el_k.confidence) / 3.0

    # Derive reliable shoulder midpoint, hip midpoint, wrist midpoint, and torso/arm scale
    sh_pts = [p for p in (_pt("left_shoulder"), _pt("right_shoulder")) if p is not None]
    hip_pts = [p for p in (l_hip_pt, r_hip_pt) if p is not None]
    wr_pts = [p for p in (_pt("left_wrist"), _pt("right_wrist")) if p is not None]

    sh_mid = (sum(p[0] for p in sh_pts) / len(sh_pts), sum(p[1] for p in sh_pts) / len(sh_pts)) if sh_pts else None
    hip_mid = (sum(p[0] for p in hip_pts) / len(hip_pts), sum(p[1] for p in hip_pts) / len(hip_pts)) if hip_pts else None
    wr_mid = (sum(p[0] for p in wr_pts) / len(wr_pts), sum(p[1] for p in wr_pts) / len(wr_pts)) if wr_pts else None

    torso_len = calculate_distance_2d(sh_mid, hip_mid) if (sh_mid and hip_mid) else 0.0
    arm_scales = []
    for side in ("left", "right"):
        s_p, e_p, w_p = _pt(f"{side}_shoulder"), _pt(f"{side}_elbow"), _pt(f"{side}_wrist")
        if s_p and e_p and w_p:
            arm_scales.append(calculate_distance_2d(s_p, e_p) + calculate_distance_2d(e_p, w_p))
    arm_scale = sum(arm_scales) / len(arm_scales) if arm_scales else 0.0

    scale = torso_len if torso_len > 1e-3 else (arm_scale if arm_scale > 1e-3 else 0.0)

    # Chin-to-bar clearance: (wrist_y - nose_y) / torso_or_arm_scale
    # Positive means nose/chin passed above the wrist/bar line (image Y grows downward).
    chin_to_bar_clearance = None
    chin_conf = 0.0
    head_y: Optional[float] = None
    head_conf = 0.0
    if _is_reliable("nose"):
        nose_kp = keypoints["nose"]
        head_y = float(nose_kp.y)
        head_conf = float(nose_kp.confidence)
    else:
        fallback_heads = [
            keypoints[k] for k in ("left_eye", "right_eye", "left_ear", "right_ear")
            if _is_reliable(k)
        ]
        if fallback_heads:
            head_y = sum(float(kp.y) for kp in fallback_heads) / len(fallback_heads)
            head_conf = sum(float(kp.confidence) for kp in fallback_heads) / len(fallback_heads)

    wr_kps = [keypoints[k] for k in ("left_wrist", "right_wrist") if _is_reliable(k)]
    if head_y is not None and wr_kps and scale > 1e-3:
        wr_y = sum(float(kp.y) for kp in wr_kps) / len(wr_kps)
        wr_conf = sum(float(kp.confidence) for kp in wr_kps) / len(wr_kps)
        chin_to_bar_clearance = round((wr_y - head_y) / scale, 3)
        chin_conf = (head_conf + wr_conf) / 2.0

    # Lower-body pendulum swing & torso tilt (ONLY when hips/ankles are reliable)
    l_ak_ok = _is_reliable("left_ankle")
    r_ak_ok = _is_reliable("right_ankle")
    ak_pts = [p for p in (_pt("left_ankle"), _pt("right_ankle")) if p is not None]
    ak_mid = (sum(p[0] for p in ak_pts) / len(ak_pts), sum(p[1] for p in ak_pts) / len(ak_pts)) if ak_pts else None

    hip_swing_offset = None
    ankle_swing_offset = None
    torso_tilt = None
    lower_body_reliable = False
    swing_conf = 0.0

    if hip_mid is not None and sh_mid is not None and torso_len > 1e-3:
        torso_tilt = calculate_vertical_angle(sh_mid, hip_mid, default_angle=0.0)
        if wr_mid is not None:
            lower_body_reliable = True
            hip_swing_offset = round((hip_mid[0] - wr_mid[0]) / torso_len, 3)
            hip_confs = [float(keypoints[k].confidence) for k in ("left_hip", "right_hip") if _is_reliable(k)]
            wr_confs = [float(kp.confidence) for kp in wr_kps]
            swing_confs = hip_confs + wr_confs
            if ak_mid is not None and (l_ak_ok or r_ak_ok):
                ankle_swing_offset = round((ak_mid[0] - wr_mid[0]) / torso_len, 3)
                swing_confs.extend(float(keypoints[k].confidence) for k in ("left_ankle", "right_ankle") if _is_reliable(k))
            swing_conf = sum(swing_confs) / len(swing_confs) if swing_confs else 0.0

    return {
        "is_pose_valid": True,
        "primary_elbow_angle": primary_elbow,
        "left_elbow_angle": l_elbow_angle,
        "right_elbow_angle": r_elbow_angle,
        "elbow_symmetry_diff": elbow_symmetry_diff,
        "shoulder_angle": shoulder_angle,
        "chin_to_bar_clearance": chin_to_bar_clearance,
        "hip_swing_offset": hip_swing_offset,
        "ankle_swing_offset": ankle_swing_offset,
        "torso_tilt": torso_tilt,
        "lower_body_reliable": lower_body_reliable,
        "left_arm_reliable": left_arm_reliable,
        "right_arm_reliable": right_arm_reliable,
        "arm_confidence": round(arm_conf, 3),
        "shoulder_angle_confidence": round(shoulder_angle_conf, 3),
        "chin_to_bar_confidence": round(chin_conf, 3),
        "body_swing_confidence": round(swing_conf, 3),
        "symmetry_confidence": round(symmetry_conf, 3),
        "missing": [],
    }


class PullUpRepCounter:
    """
    Dedicated Pull-Up Repetition Counter with temporal state machine & hysteresis:
    DEAD_HANG -> PULLING -> TOP_POSITION -> LOWERING -> DEAD_HANG (REP COMPLETED)

    Features:
    - Hysteresis around dead hang (>= 150 / >= 148), pulling (< 140), and top (<= 85 / > 95).
    - Minimum top hold duration/frames to reject momentary noise spikes.
    - Minimum rep duration & cooldown guards to avoid double counting.
    - Incomplete rep tracking:
      * INCOMPLETE_PULL: partial pull reaching < 130° without reaching <= 90° top before returning to hang.
      * INCOMPLETE_EXTENSION: reversing into next rep or ending lowering without extending >= 145°/148°.
    - Automatic recovery upon tracking loss.
    """

    def __init__(self, config: Optional[PullUpCounterConfig] = None):
        self.config = config or PullUpCounterConfig()
        self.state: PullUpState = PullUpState.DEAD_HANG
        self.rep_count: int = 0
        self.incomplete_rep_count: int = 0
        self.current_elbow_angle: float = 180.0
        self.smoothed_elbow_angle: float = 180.0
        self.min_elbow_angle: float = 180.0
        self.max_elbow_angle: float = 180.0
        self.depth_quality: str = "GOOD"
        self.is_tracking: bool = False
        self.rep_just_completed: bool = False
        self.feedback: str = "Ready in dead hang. Begin your pull-up."
        self.last_completed_min_angle: float = 180.0
        self.last_completed_depth: str = "GOOD"

        # Temporal tracking
        self.rep_start_time: float = 0.0
        self.top_enter_time: float = 0.0
        self.last_rep_completed_time: float = 0.0
        self.last_valid_tracking_time: float = 0.0
        self.reached_top_position: bool = False
        self.is_smoothed_initialized: bool = False
        self.consecutive_top_frames: int = 0
        self._simulated_clock: float = 0.0

        # Sequence / per-rep episode tracking
        self.rep_records: List[Dict[str, Any]] = []
        self._active_rep: Optional[Dict[str, Any]] = None
        self._frame_seq: int = 0
        self._max_hang_angle_before_rep: Optional[float] = None
        self._last_hang_frame: Optional[int] = None
        self._last_hang_ts_ms: Optional[int] = None
        self._last_hang_time: Optional[float] = None
        self._last_hang_angle: float = 180.0
        self._last_valid_frame: int = 0
        self._last_valid_ts_ms: int = 0

    def reset(self) -> None:
        """Reset pull-up rep counter and tracking state."""
        self.state = PullUpState.DEAD_HANG
        self.rep_count = 0
        self.incomplete_rep_count = 0
        self.current_elbow_angle = 180.0
        self.smoothed_elbow_angle = 180.0
        self.min_elbow_angle = 180.0
        self.max_elbow_angle = 180.0
        self.depth_quality = "GOOD"
        self.is_tracking = False
        self.rep_just_completed = False
        self.feedback = "Ready in dead hang. Begin your pull-up."
        self.last_completed_min_angle = 180.0
        self.last_completed_depth = "GOOD"
        self.rep_start_time = 0.0
        self.top_enter_time = 0.0
        self.last_rep_completed_time = 0.0
        self.last_valid_tracking_time = 0.0
        self.reached_top_position = False
        self.is_smoothed_initialized = False
        self.consecutive_top_frames = 0
        self._simulated_clock = 0.0
        self.rep_records.clear()
        self._active_rep = None
        self._frame_seq = 0
        self._max_hang_angle_before_rep = None
        self._last_hang_frame = None
        self._last_hang_ts_ms = None
        self._last_hang_time = None
        self._last_hang_angle = 180.0
        self._last_valid_frame = 0
        self._last_valid_ts_ms = 0

    def _record_rep_episode(
        self,
        end_frame: int,
        end_ts_ms: int,
        is_valid: bool,
        issues: List[str],
    ) -> None:
        """Persist a completed or incomplete pull-up rep episode."""
        if not self._active_rep:
            return
        start_frame = int(self._active_rep["start_frame"])
        start_ts_ms = int(self._active_rep["start_ts_ms"])
        min_angle = round(float(self._active_rep["min_angle"]), 1)
        max_angle = round(float(self._active_rep["max_angle"]), 1)
        rom = round(max(0.0, max_angle - min_angle), 1)
        duration_ms = max(0, end_ts_ms - start_ts_ms)
        depth_cls = self._classify_top_height(min_angle) if is_valid else "INCOMPLETE"

        combined_issues = list(dict.fromkeys(list(self._active_rep.get("base_issues", [])) + list(issues)))

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
            "issues": combined_issues,
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
        e_angle: float,
        use_hang_anchor: bool = True,
        base_issues: Optional[List[str]] = None,
    ) -> None:
        """Initialize tracking metadata for a new pull-up attempt."""
        if use_hang_anchor and self._last_hang_frame is not None and self._last_hang_ts_ms is not None:
            start_frame = self._last_hang_frame
            start_ts_ms = self._last_hang_ts_ms
            start_time = self._last_hang_time if self._last_hang_time is not None else now
            max_angle = max(self._last_hang_angle, e_angle)
            frames = [start_frame, cur_frame] if start_frame != cur_frame else [cur_frame]
        else:
            start_frame = cur_frame
            start_ts_ms = now_ms
            start_time = now
            max_angle = e_angle
            frames = [cur_frame]

        self._active_rep = {
            "start_frame": start_frame,
            "start_ts_ms": start_ts_ms,
            "start_time": start_time,
            "min_angle": e_angle,
            "max_angle": max_angle,
            "peak_lowering_angle": e_angle,
            "frame_indices": frames,
            "base_issues": list(base_issues or []),
        }

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Finalize any unfinished rep in LOWERING at the end of a sequence
        (failed dead-hang extension) and return all recorded rep episodes.
        """
        if (
            self._active_rep is not None
            and self.state == PullUpState.LOWERING
            and self._active_rep["peak_lowering_angle"] >= self._active_rep["min_angle"] + 15.0
        ):
            issues = ["INCOMPLETE_EXTENSION"]
            if not self.reached_top_position and self._active_rep["min_angle"] > 90.0:
                issues.append("INCOMPLETE_PULL")
            self._record_rep_episode(
                end_frame=self._last_valid_frame,
                end_ts_ms=self._last_valid_ts_ms,
                is_valid=False,
                issues=issues,
            )
            self.state = PullUpState.DEAD_HANG
        return list(self.rep_records)

    def process_angle(
        self,
        angle: Optional[float],
        timestamp_sec: Optional[float] = None,
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
    ) -> PullUpRepResult:
        """
        Process a single elbow angle measurement through the Pull-up State Machine.
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

        # 1. Tracking Loss handling
        if angle is None or not math.isfinite(angle):
            self.is_tracking = False
            if (
                self.last_valid_tracking_time > 0
                and (now - self.last_valid_tracking_time) > self.config.max_lost_duration_sec
            ):
                if self.state != PullUpState.DEAD_HANG:
                    logger.debug(
                        "Pull-up tracking lost for > %.1fs. Resetting state to DEAD_HANG.",
                        self.config.max_lost_duration_sec,
                    )
                    self.state = PullUpState.DEAD_HANG
                    self.min_elbow_angle = 180.0
                    self.reached_top_position = False
                    self.is_smoothed_initialized = False
                    self.consecutive_top_frames = 0
                    self._active_rep = None

            return PullUpRepResult(
                exercise="PULL_UP",
                rep_count=self.rep_count,
                incomplete_rep_count=self.incomplete_rep_count,
                current_state=self.state.value,
                elbow_angle=round(self.smoothed_elbow_angle, 1),
                min_elbow_angle=round(self.min_elbow_angle, 1),
                max_elbow_angle=round(self.max_elbow_angle, 1),
                depth=self.depth_quality,
                is_tracking=False,
                feedback=self.feedback,
                rep_just_completed=False,
                movement_duration_sec=round(max(0.0, now - self.rep_start_time), 2) if self.state != PullUpState.DEAD_HANG else 0.0,
            )

        # 2. Tracking Active & EMA Smoothing
        self.is_tracking = True
        self.last_valid_tracking_time = now
        self.current_elbow_angle = float(angle)

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

        # 3. Pull-up State Machine Transitions
        if self.state == PullUpState.DEAD_HANG:
            if e_angle < self.config.pulling_enter_angle:
                base_issues: List[str] = []
                if (
                    self._max_hang_angle_before_rep is not None
                    and self._max_hang_angle_before_rep < self.config.full_extension_min_angle
                ):
                    base_issues.append("INCOMPLETE_EXTENSION")

                self.state = PullUpState.PULLING
                self.rep_start_time = now
                self.min_elbow_angle = e_angle
                self.max_elbow_angle = max(self._last_hang_angle, e_angle)
                self.reached_top_position = False
                self.consecutive_top_frames = 0
                self._start_active_rep(
                    cur_frame, now_ms, now, e_angle,
                    use_hang_anchor=True,
                    base_issues=base_issues,
                )
                self.feedback = "Pull strong toward the bar"
            else:
                self.min_elbow_angle = 180.0
                self.max_elbow_angle = max(self.max_elbow_angle, e_angle) if self._last_hang_frame is not None else e_angle
                self._max_hang_angle_before_rep = (
                    max(self._max_hang_angle_before_rep, e_angle)
                    if self._max_hang_angle_before_rep is not None else e_angle
                )
                self._last_hang_frame = cur_frame
                self._last_hang_ts_ms = now_ms
                self._last_hang_time = now
                self._last_hang_angle = e_angle
                if e_angle < self.config.dead_hang_angle:
                    self.feedback = "Extend arms fully into a dead hang"
                else:
                    self.feedback = "Good dead hang. Ready to pull."

        elif self.state == PullUpState.PULLING:
            self.min_elbow_angle = min(self.min_elbow_angle, e_angle)
            self.max_elbow_angle = max(self.max_elbow_angle, e_angle)
            if self._active_rep is not None:
                self._active_rep["min_angle"] = min(self._active_rep["min_angle"], e_angle)
                self._active_rep["max_angle"] = max(self._active_rep["max_angle"], e_angle)
                self._active_rep["frame_indices"].append(cur_frame)

            top_reached = (
                e_angle <= self.config.top_enter_angle
                or (self.current_elbow_angle <= self.config.top_enter_angle and e_angle <= 91.0)
            )

            if top_reached:
                self.state = PullUpState.TOP_POSITION
                self.top_enter_time = now
                self.reached_top_position = True
                self.consecutive_top_frames = 1
                self.depth_quality = self._classify_top_height(self.min_elbow_angle)
                self.feedback = "Great pull! Chin over bar."
            elif e_angle >= self.config.pulling_abort_angle:
                # Partial pull aborted back to dead hang without reaching top position
                if self._active_rep is not None:
                    attempt_min = self._active_rep["min_angle"]
                    attempt_dur = now - self._active_rep["start_time"]
                    if (
                        attempt_min < self.config.incomplete_pull_threshold
                        and attempt_min > 90.0
                        and (attempt_dur >= 0.25 or len(self._active_rep["frame_indices"]) >= 3)
                    ):
                        self._active_rep["max_angle"] = max(self._active_rep["max_angle"], e_angle)
                        self._record_rep_episode(
                            end_frame=cur_frame,
                            end_ts_ms=now_ms,
                            is_valid=False,
                            issues=["INCOMPLETE_PULL"],
                        )
                    elif attempt_min <= 90.0 and (attempt_dur >= 0.25 or len(self._active_rep["frame_indices"]) >= 3):
                        # Reached <= 90 via smoothing boundary without locking top
                        self._active_rep["max_angle"] = max(self._active_rep["max_angle"], e_angle)
                        self._record_rep_episode(
                            end_frame=cur_frame,
                            end_ts_ms=now_ms,
                            is_valid=False,
                            issues=["INCOMPLETE_PULL"],
                        )
                    else:
                        self._active_rep = None

                self.state = PullUpState.DEAD_HANG
                self.min_elbow_angle = 180.0
                self.reached_top_position = False
                self.consecutive_top_frames = 0
                self._max_hang_angle_before_rep = e_angle
                self._last_hang_frame = cur_frame
                self._last_hang_ts_ms = now_ms
                self._last_hang_time = now
                self._last_hang_angle = e_angle
                self.feedback = "Pull higher to bring chin over the bar"
            else:
                self.feedback = "Keep pulling..."

        elif self.state == PullUpState.TOP_POSITION:
            self.min_elbow_angle = min(self.min_elbow_angle, e_angle)
            self.max_elbow_angle = max(self.max_elbow_angle, e_angle)
            self.consecutive_top_frames += 1
            self.depth_quality = self._classify_top_height(self.min_elbow_angle)
            if self._active_rep is not None:
                self._active_rep["min_angle"] = min(self._active_rep["min_angle"], e_angle)
                self._active_rep["max_angle"] = max(self._active_rep["max_angle"], e_angle)
                self._active_rep["frame_indices"].append(cur_frame)

            if e_angle > self.config.top_exit_angle:
                top_duration = now - self.top_enter_time
                top_satisfied = (
                    top_duration >= self.config.min_top_duration_sec
                    or self.consecutive_top_frames >= 2
                    or self.config.min_top_duration_sec == 0.0
                )
                self.reached_top_position = bool(top_satisfied)
                if self._active_rep is not None:
                    self._active_rep["peak_lowering_angle"] = e_angle
                self.state = PullUpState.LOWERING
                self.feedback = "Lower under control"
            else:
                self.feedback = "Great pull! Chin over bar."

        elif self.state == PullUpState.LOWERING:
            self.min_elbow_angle = min(self.min_elbow_angle, e_angle)
            self.max_elbow_angle = max(self.max_elbow_angle, e_angle)
            if self._active_rep is not None:
                self._active_rep["min_angle"] = min(self._active_rep["min_angle"], e_angle)
                self._active_rep["max_angle"] = max(self._active_rep["max_angle"], e_angle)
                self._active_rep["peak_lowering_angle"] = max(self._active_rep["peak_lowering_angle"], e_angle)
                self._active_rep["frame_indices"].append(cur_frame)

            is_dead_hang = (
                e_angle >= self.config.dead_hang_return_angle
                or (self.current_elbow_angle >= self.config.dead_hang_return_angle and e_angle >= 142.0)
            )

            if is_dead_hang:
                rep_duration = now - self.rep_start_time
                cooldown = now - self.last_rep_completed_time if self.last_rep_completed_time > 0 else 999.0

                is_valid_rep = (
                    self.reached_top_position
                    and (rep_duration >= self.config.min_rep_duration_sec or self.config.min_rep_duration_sec == 0.0)
                    and (cooldown >= self.config.min_cooldown_sec or self.config.min_cooldown_sec == 0.0)
                )

                if is_valid_rep:
                    self.rep_count += 1
                    self.last_rep_completed_time = now
                    self.rep_just_completed = True
                    self.last_completed_min_angle = self.min_elbow_angle
                    self.last_completed_depth = self.depth_quality
                    self._record_rep_episode(
                        end_frame=cur_frame,
                        end_ts_ms=now_ms,
                        is_valid=True,
                        issues=[],
                    )
                    self.feedback = "Rep completed! Full dead hang."
                else:
                    if (
                        self._active_rep is not None
                        and not self.reached_top_position
                        and self._active_rep["min_angle"] < self.config.incomplete_pull_threshold
                    ):
                        self._record_rep_episode(
                            end_frame=cur_frame,
                            end_ts_ms=now_ms,
                            is_valid=False,
                            issues=["INCOMPLETE_PULL"],
                        )
                    else:
                        self._active_rep = None
                    self.feedback = "Lower completely to dead hang"

                self.state = PullUpState.DEAD_HANG
                self.min_elbow_angle = self.last_completed_min_angle if is_valid_rep else 180.0
                self.reached_top_position = False
                self.consecutive_top_frames = 0
                self._max_hang_angle_before_rep = e_angle
                self._last_hang_frame = cur_frame
                self._last_hang_ts_ms = now_ms
                self._last_hang_time = now
                self._last_hang_angle = e_angle

            elif (
                self._active_rep is not None
                and self._active_rep["peak_lowering_angle"] >= 120.0
                and e_angle < self._active_rep["peak_lowering_angle"] - 20.0
                and e_angle < 130.0
            ):
                # Athlete started pulling again before extending >= 145 deg into dead hang
                issues = ["INCOMPLETE_EXTENSION"]
                if not self.reached_top_position:
                    issues.append("INCOMPLETE_PULL")
                self._record_rep_episode(
                    end_frame=self._last_valid_frame,
                    end_ts_ms=self._last_valid_ts_ms,
                    is_valid=False,
                    issues=issues,
                )
                self.state = PullUpState.PULLING
                self.rep_start_time = now
                self.min_elbow_angle = e_angle
                self.reached_top_position = False
                self.consecutive_top_frames = 0
                self._max_hang_angle_before_rep = e_angle
                self._start_active_rep(
                    cur_frame, now_ms, now, e_angle,
                    use_hang_anchor=False,
                    base_issues=["INCOMPLETE_EXTENSION"],
                )
                self.feedback = "Extend arms fully into a dead hang between reps"

            elif e_angle <= self.config.top_enter_angle:
                self.state = PullUpState.TOP_POSITION
                self.top_enter_time = now
                self.consecutive_top_frames += 1

        self._last_valid_frame = cur_frame
        self._last_valid_ts_ms = now_ms

        self.depth_quality = self._classify_top_height(
            self.min_elbow_angle if self.state != PullUpState.DEAD_HANG else self.current_elbow_angle
        )

        if self.state == PullUpState.DEAD_HANG and self.last_completed_min_angle < 180.0:
            reported_min = self.last_completed_min_angle
            reported_depth = self.last_completed_depth
        elif self.state != PullUpState.DEAD_HANG:
            reported_min = self.min_elbow_angle
            reported_depth = self.depth_quality
        else:
            reported_min = self.current_elbow_angle
            reported_depth = self.depth_quality

        mov_duration = round(max(0.0, now - self.rep_start_time), 2) if self.state != PullUpState.DEAD_HANG else 0.0

        return PullUpRepResult(
            exercise="PULL_UP",
            rep_count=self.rep_count,
            incomplete_rep_count=self.incomplete_rep_count,
            current_state=self.state.value,
            elbow_angle=round(self.smoothed_elbow_angle, 1),
            min_elbow_angle=round(reported_min, 1),
            max_elbow_angle=round(self.max_elbow_angle, 1),
            depth=reported_depth,
            is_tracking=self.is_tracking,
            feedback=self.feedback,
            rep_just_completed=self.rep_just_completed,
            movement_duration_sec=mov_duration,
        )

    def _classify_top_height(self, angle: float) -> str:
        """Classify top flexion depth: GOOD (<=85), MODERATE (<=105), SHALLOW (>105)."""
        if angle <= self.config.top_enter_angle:
            return "GOOD"
        elif angle <= 105.0:
            return "MODERATE"
        return "SHALLOW"
