import logging
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_body_line_deviation,
    calculate_distance_2d,
    get_point,
    is_valid_coord,
)
from app.schemas.analysis import KeypointData

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
    descending_abort_angle: float = 150.0  # Elbow angle >= 150.0 aborts partial descent without rep
    meaningful_attempt_angle: float = 135.0  # Elbow angle < 135.0 marks meaningful push-up attempt
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
    Never fabricates body_alignment or symmetry values when landmarks are missing/unreliable.
    """
    empty_result = {
        "is_pose_valid": False,
        "primary_elbow_angle": None,
        "left_elbow_angle": None,
        "right_elbow_angle": None,
        "average_elbow_angle": None,
        "elbow_symmetry_diff": None,
        "body_alignment": None,
        "body_alignment_reliable": False,
        "alignment_type": None,
        "hip_sag": False,
        "hip_pike": False,
        "shoulder_offset_px": None,
        "shoulder_position_ratio": None,
        "left_arm_reliable": False,
        "right_arm_reliable": False,
        "arm_confidence": 0.0,
        "shoulder_confidence": 0.0,
        "body_alignment_confidence": 0.0,
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

    def _reliable_pt(name: str) -> Optional[Tuple[float, float]]:
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
        p_sh = _reliable_pt("left_shoulder")
        p_el = _reliable_pt("left_elbow")
        p_wr = _reliable_pt("left_wrist")
        if p_sh and p_el and p_wr:
            l_elbow_angle = calculate_angle_2d(p_sh, p_el, p_wr, default_angle=180.0)

    r_elbow_angle = None
    if right_arm_reliable:
        p_sh = _reliable_pt("right_shoulder")
        p_el = _reliable_pt("right_elbow")
        p_wr = _reliable_pt("right_wrist")
        if p_sh and p_el and p_wr:
            r_elbow_angle = calculate_angle_2d(p_sh, p_el, p_wr, default_angle=180.0)

    missing = []
    if not (l_sh_ok or r_sh_ok):
        missing.append("shoulder")
    if not (l_el_ok or r_el_ok):
        missing.append("elbow")
    if not (l_wr_ok or r_wr_ok):
        missing.append("wrist")

    l_hip_ok = _is_reliable("left_hip")
    r_hip_ok = _is_reliable("right_hip")
    if not (l_hip_ok or r_hip_ok):
        missing.append("hip")

    l_ak_ok = _is_reliable("left_ankle")
    r_ak_ok = _is_reliable("right_ankle")
    if not (l_ak_ok or r_ak_ok):
        missing.append("ankle")

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
        avg_elbow = l_elbow_angle
        arm_conf = l_arm_conf
    else:
        assert r_elbow_angle is not None
        active_side = "right"
        primary_elbow = r_elbow_angle
        avg_elbow = r_elbow_angle
        arm_conf = r_arm_conf

    # Elbow symmetry (ONLY in non-side view with both arms reliable)
    elbow_symmetry_diff = None
    symmetry_conf = 0.0
    if not side_view and left_arm_reliable and right_arm_reliable and l_elbow_angle is not None and r_elbow_angle is not None:
        elbow_symmetry_diff = round(abs(l_elbow_angle - r_elbow_angle), 1)
        symmetry_conf = (l_arm_conf + r_arm_conf) / 2.0

    # Shoulder stacking relative to wrist (normalized to upper arm length)
    shoulder_offset_px = None
    shoulder_position_ratio = None
    shoulder_conf = 0.0
    if left_arm_reliable and right_arm_reliable and not side_view:
        l_sh_pt = _reliable_pt("left_shoulder")
        l_el_pt = _reliable_pt("left_elbow")
        l_wr_pt = _reliable_pt("left_wrist")
        r_sh_pt = _reliable_pt("right_shoulder")
        r_el_pt = _reliable_pt("right_elbow")
        r_wr_pt = _reliable_pt("right_wrist")
        ratios = []
        offsets = []
        for s_pt, e_pt, w_pt in ((l_sh_pt, l_el_pt, l_wr_pt), (r_sh_pt, r_el_pt, r_wr_pt)):
            u_len = calculate_distance_2d(s_pt, e_pt)
            if s_pt and w_pt and u_len > 1e-3:
                off = abs(s_pt[0] - w_pt[0])
                offsets.append(off)
                ratios.append(off / u_len)
        if ratios:
            shoulder_offset_px = round(sum(offsets) / len(offsets), 1)
            shoulder_position_ratio = round(sum(ratios) / len(ratios), 2)
            shoulder_conf = arm_conf
    else:
        s_pt = _reliable_pt(f"{active_side}_shoulder")
        e_pt = _reliable_pt(f"{active_side}_elbow")
        w_pt = _reliable_pt(f"{active_side}_wrist")
        u_len = calculate_distance_2d(s_pt, e_pt)
        if s_pt and w_pt and u_len > 1e-3:
            off = abs(s_pt[0] - w_pt[0])
            shoulder_offset_px = round(off, 1)
            shoulder_position_ratio = round(off / u_len, 2)
            shoulder_conf = arm_conf

    # Body alignment (Shoulder - Hip - Ankle/Knee) using reliable landmarks only
    l_side_ok = l_sh_ok and l_hip_ok and l_ak_ok
    r_side_ok = r_sh_ok and r_hip_ok and r_ak_ok

    p_sh = None
    p_hip = None
    p_ak = None
    align_confs: List[float] = []

    def _pick_leg_pt(side: str) -> Optional[Tuple[float, float, float]]:
        ak_name = f"{side}_ankle"
        if _is_reliable(ak_name):
            kp = keypoints[ak_name]
            return float(kp.x), float(kp.y), float(kp.confidence)
        return None

    if l_side_ok and r_side_ok:
        l_c = keypoints["left_shoulder"].confidence + keypoints["left_hip"].confidence
        r_c = keypoints["right_shoulder"].confidence + keypoints["right_hip"].confidence
        chosen = "left" if l_c >= r_c else "right"
        sh_kp = keypoints[f"{chosen}_shoulder"]
        hip_kp = keypoints[f"{chosen}_hip"]
        leg_res = _pick_leg_pt(chosen)
        if leg_res is not None:
            p_sh = (float(sh_kp.x), float(sh_kp.y))
            p_hip = (float(hip_kp.x), float(hip_kp.y))
            p_ak = (leg_res[0], leg_res[1])
            align_confs = [float(sh_kp.confidence), float(hip_kp.confidence), leg_res[2]]
    elif l_side_ok:
        sh_kp = keypoints["left_shoulder"]
        hip_kp = keypoints["left_hip"]
        leg_res = _pick_leg_pt("left")
        if leg_res is not None:
            p_sh = (float(sh_kp.x), float(sh_kp.y))
            p_hip = (float(hip_kp.x), float(hip_kp.y))
            p_ak = (leg_res[0], leg_res[1])
            align_confs = [float(sh_kp.confidence), float(hip_kp.confidence), leg_res[2]]
    elif r_side_ok:
        sh_kp = keypoints["right_shoulder"]
        hip_kp = keypoints["right_hip"]
        leg_res = _pick_leg_pt("right")
        if leg_res is not None:
            p_sh = (float(sh_kp.x), float(sh_kp.y))
            p_hip = (float(hip_kp.x), float(hip_kp.y))
            p_ak = (leg_res[0], leg_res[1])
            align_confs = [float(sh_kp.confidence), float(hip_kp.confidence), leg_res[2]]
    elif (l_sh_ok or r_sh_ok) and (l_hip_ok or r_hip_ok) and (l_ak_ok or r_ak_ok):
        sh_kp = keypoints["left_shoulder"] if l_sh_ok else keypoints["right_shoulder"]
        hip_kp = keypoints["left_hip"] if l_hip_ok else keypoints["right_hip"]
        leg_res = _pick_leg_pt("left") or _pick_leg_pt("right")
        if leg_res is not None:
            p_sh = (float(sh_kp.x), float(sh_kp.y))
            p_hip = (float(hip_kp.x), float(hip_kp.y))
            p_ak = (leg_res[0], leg_res[1])
            align_confs = [float(sh_kp.confidence), float(hip_kp.confidence), leg_res[2]]

    body_line: Optional[float] = None
    body_alignment_reliable = False
    alignment_type: Optional[str] = None
    hip_sag = False
    hip_pike = False
    body_align_conf = 0.0

    if p_sh is not None and p_hip is not None and p_ak is not None:
        dev_angle, align_type = calculate_body_line_deviation(p_sh, p_hip, p_ak)
        body_line = round(dev_angle, 1)
        body_alignment_reliable = True
        alignment_type = align_type
        hip_sag = align_type == "SAG"
        hip_pike = align_type == "PIKE"
        body_align_conf = sum(align_confs) / len(align_confs) if align_confs else 0.0

    return {
        "is_pose_valid": True,
        "primary_elbow_angle": primary_elbow,
        "left_elbow_angle": l_elbow_angle,
        "right_elbow_angle": r_elbow_angle,
        "average_elbow_angle": avg_elbow,
        "elbow_symmetry_diff": elbow_symmetry_diff,
        "body_alignment": body_line,
        "body_alignment_reliable": body_alignment_reliable,
        "alignment_type": alignment_type,
        "hip_sag": hip_sag,
        "hip_pike": hip_pike,
        "shoulder_offset_px": shoulder_offset_px,
        "shoulder_position_ratio": shoulder_position_ratio,
        "left_arm_reliable": left_arm_reliable,
        "right_arm_reliable": right_arm_reliable,
        "arm_confidence": round(arm_conf, 3),
        "shoulder_confidence": round(shoulder_conf, 3),
        "body_alignment_confidence": round(body_align_conf, 3),
        "symmetry_confidence": round(symmetry_conf, 3),
        "missing": [],
    }


class PushUpRepCounter:
    """
    Dedicated, completely independent Push-up Rep Counter.
    Implements a strict 4-phase state machine:
    TOP -> DESCENDING -> BOTTOM -> ASCENDING -> TOP (REP COMPLETED)

    Features:
    - Biomechanical thresholds with hysteresis and EMA temporal smoothing.
    - Minimum bottom duration, rep duration, and cooldown guards.
    - Incomplete rep tracking (partial shallow descent and failed lockout).
    - Body alignment monitoring (detects sagging or piking when landmarks are available).
    - Automatic state recovery upon tracking loss.
    """

    def __init__(self, config: Optional[PushUpCounterConfig] = None):
        self.config = config or PushUpCounterConfig()
        self.state: PushUpState = PushUpState.TOP
        self.rep_count: int = 0
        self.incomplete_rep_count: int = 0
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

        # Form metrics & sequence episode tracking
        self.rep_body_lines: List[float] = []
        self.rep_records: List[Dict[str, Any]] = []
        self._active_rep: Optional[Dict[str, Any]] = None
        self._frame_seq: int = 0
        self._last_top_frame: Optional[int] = None
        self._last_top_ts_ms: Optional[int] = None
        self._last_top_time: Optional[float] = None
        self._last_top_angle: float = 180.0
        self._last_valid_frame: int = 0
        self._last_valid_ts_ms: int = 0

    def reset(self) -> None:
        """Reset rep counter and tracking state."""
        self.state = PushUpState.TOP
        self.rep_count = 0
        self.incomplete_rep_count = 0
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
        self.rep_records.clear()
        self._active_rep = None
        self._frame_seq = 0
        self._last_top_frame = None
        self._last_top_ts_ms = None
        self._last_top_time = None
        self._last_top_angle = 180.0
        self._last_valid_frame = 0
        self._last_valid_ts_ms = 0

    def _record_rep_episode(
        self,
        end_frame: int,
        end_ts_ms: int,
        is_valid: bool,
        issues: List[str],
    ) -> None:
        """Persist a completed or incomplete push-up rep episode."""
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
        e_angle: float,
        use_top_anchor: bool = True,
    ) -> None:
        """Initialize tracking metadata for a new push-up attempt."""
        if use_top_anchor and self._last_top_frame is not None and self._last_top_ts_ms is not None:
            start_frame = self._last_top_frame
            start_ts_ms = self._last_top_ts_ms
            start_time = self._last_top_time if self._last_top_time is not None else now
            max_angle = max(self._last_top_angle, e_angle)
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
            "peak_ascent_angle": e_angle,
            "frame_indices": frames,
        }

    def finalize(self) -> List[Dict[str, Any]]:
        """
        Finalize any unfinished rep in ASCENDING at the end of a sequence
        (failed top lockout) and return all recorded rep episodes.
        """
        if (
            self._active_rep is not None
            and self.state == PushUpState.ASCENDING
            and (self.reached_bottom_depth or self._active_rep["min_angle"] < self.config.meaningful_attempt_angle)
            and self._active_rep["peak_ascent_angle"] >= self._active_rep["min_angle"] + 15.0
        ):
            issues = ["INCOMPLETE_LOCKOUT"]
            if self._active_rep["min_angle"] > self.config.bottom_angle:
                issues.append("INSUFFICIENT_DEPTH")
            self._record_rep_episode(
                end_frame=self._last_valid_frame,
                end_ts_ms=self._last_valid_ts_ms,
                is_valid=False,
                issues=issues,
            )
            self.state = PushUpState.TOP
        return list(self.rep_records)

    def process_angle(
        self,
        angle: Optional[float],
        body_alignment: Optional[float] = 180.0,
        timestamp_sec: Optional[float] = None,
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
    ) -> PushUpRepResult:
        """
        Process a single elbow angle and optional body alignment measurement through Push-up State Machine.
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

        # 1. Handle Tracking Loss (no confident elbow angle)
        if angle is None or not math.isfinite(angle):
            self.is_tracking = False
            if (
                self.last_valid_tracking_time > 0
                and (now - self.last_valid_tracking_time) > self.config.max_lost_duration_sec
            ):
                if self.state != PushUpState.TOP:
                    logger.debug(
                        "Push-up tracking lost for > %.1fs. Resetting state to TOP.",
                        self.config.max_lost_duration_sec,
                    )
                    self.state = PushUpState.TOP
                    self.min_elbow_angle = 180.0
                    self.reached_bottom_depth = False
                    self.is_smoothed_initialized = False
                    self.consecutive_bottom_frames = 0
                    self.rep_body_lines.clear()
                    self._active_rep = None

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
                movement_duration_sec=round(max(0.0, now - self.rep_start_time), 2) if self.state != PushUpState.TOP else 0.0,
            )

        # 2. Tracking Active
        self.is_tracking = True
        self.last_valid_tracking_time = now
        self.current_elbow_angle = float(angle)
        has_valid_alignment = body_alignment is not None and math.isfinite(body_alignment)
        if has_valid_alignment:
            assert body_alignment is not None
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

        if has_valid_alignment and self.body_alignment < 160.0:
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
                self.rep_body_lines = [self.body_alignment] if has_valid_alignment else []
                self._start_active_rep(cur_frame, now_ms, now, e_angle, use_top_anchor=True)
                self.feedback = alignment_feedback or "Control your descent"
            else:
                self.min_elbow_angle = 180.0
                self._last_top_frame = cur_frame
                self._last_top_ts_ms = now_ms
                self._last_top_time = now
                self._last_top_angle = e_angle
                if e_angle < self.config.top_angle:
                    self.feedback = "Return to the top position"
                else:
                    self.feedback = alignment_feedback or "Ready. Begin your push-up."

        elif self.state == PushUpState.DESCENDING:
            self.min_elbow_angle = min(self.min_elbow_angle, e_angle)
            if has_valid_alignment:
                self.rep_body_lines.append(self.body_alignment)
            if self._active_rep is not None:
                self._active_rep["min_angle"] = min(self._active_rep["min_angle"], e_angle)
                self._active_rep["max_angle"] = max(self._active_rep["max_angle"], e_angle)
                self._active_rep["frame_indices"].append(cur_frame)

            if e_angle <= self.config.bottom_angle or (
                self.current_elbow_angle <= self.config.bottom_angle
                and e_angle <= self.config.bottom_angle + 6.0
            ):
                self.state = PushUpState.BOTTOM
                self.bottom_enter_time = now
                self.reached_bottom_depth = True
                self.consecutive_bottom_frames = 1
                self.depth_quality = self._classify_depth(self.min_elbow_angle)
                self.feedback = alignment_feedback or "Good push-up depth"
            elif e_angle >= self.config.descending_abort_angle:
                # Aborted partial push-up
                if self._active_rep is not None:
                    attempt_min = self._active_rep["min_angle"]
                    attempt_dur = now - self._active_rep["start_time"]
                    if (
                        attempt_min < self.config.meaningful_attempt_angle
                        and (attempt_dur >= 0.25 or len(self._active_rep["frame_indices"]) >= 3)
                    ):
                        self._active_rep["max_angle"] = max(self._active_rep["max_angle"], e_angle)
                        self._record_rep_episode(
                            end_frame=cur_frame,
                            end_ts_ms=now_ms,
                            is_valid=False,
                            issues=["INSUFFICIENT_DEPTH"],
                        )
                    else:
                        self._active_rep = None

                self.state = PushUpState.TOP
                self.min_elbow_angle = 180.0
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self.rep_body_lines.clear()
                self._last_top_frame = cur_frame
                self._last_top_ts_ms = now_ms
                self._last_top_time = now
                self._last_top_angle = e_angle
                self.feedback = "Lower your body further"
            else:
                self.feedback = alignment_feedback or "Lower your body further"

        elif self.state == PushUpState.BOTTOM:
            self.min_elbow_angle = min(self.min_elbow_angle, e_angle)
            if has_valid_alignment:
                self.rep_body_lines.append(self.body_alignment)
            self.consecutive_bottom_frames += 1
            self.depth_quality = self._classify_depth(self.min_elbow_angle)
            if self._active_rep is not None:
                self._active_rep["min_angle"] = min(self._active_rep["min_angle"], e_angle)
                self._active_rep["max_angle"] = max(self._active_rep["max_angle"], e_angle)
                self._active_rep["frame_indices"].append(cur_frame)

            if e_angle > self.config.bottom_exit_angle:
                bottom_duration = now - self.bottom_enter_time
                bottom_satisfied = (
                    bottom_duration >= self.config.min_bottom_duration_sec
                    or self.consecutive_bottom_frames >= 2
                    or self.config.min_bottom_duration_sec == 0.0
                )
                self.reached_bottom_depth = bool(bottom_satisfied)
                if self._active_rep is not None:
                    self._active_rep["peak_ascent_angle"] = e_angle
                self.state = PushUpState.ASCENDING
                self.feedback = alignment_feedback or "Push up strong!"
            else:
                self.feedback = alignment_feedback or "Good push-up depth"

        elif self.state == PushUpState.ASCENDING:
            self.min_elbow_angle = min(self.min_elbow_angle, e_angle)
            if has_valid_alignment:
                self.rep_body_lines.append(self.body_alignment)
            if self._active_rep is not None:
                self._active_rep["min_angle"] = min(self._active_rep["min_angle"], e_angle)
                self._active_rep["max_angle"] = max(self._active_rep["max_angle"], e_angle)
                self._active_rep["peak_ascent_angle"] = max(self._active_rep["peak_ascent_angle"], e_angle)
                self._active_rep["frame_indices"].append(cur_frame)

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

                    depth_score = (
                        100.0 if self.min_elbow_angle <= self.config.bottom_angle
                        else max(60.0, 100.0 - (self.min_elbow_angle - self.config.bottom_angle) * 2.0)
                    )
                    if self.rep_body_lines:
                        avg_body_line = sum(self.rep_body_lines) / len(self.rep_body_lines)
                        align_score = max(50.0, min(100.0, 100.0 - abs(180.0 - avg_body_line) * 2.0))
                        self.form_score = round(0.50 * align_score + 0.50 * depth_score, 1)
                    else:
                        self.form_score = round(depth_score, 1)

                    self._record_rep_episode(
                        end_frame=cur_frame,
                        end_ts_ms=now_ms,
                        is_valid=True,
                        issues=[],
                    )
                    self.feedback = "Good push-up! Stand by for next rep."
                else:
                    if (
                        self._active_rep is not None
                        and not self.reached_bottom_depth
                        and self._active_rep["min_angle"] < self.config.meaningful_attempt_angle
                    ):
                        self._record_rep_episode(
                            end_frame=cur_frame,
                            end_ts_ms=now_ms,
                            is_valid=False,
                            issues=["INSUFFICIENT_DEPTH"],
                        )
                    else:
                        self._active_rep = None
                    self.feedback = "Return to the top position"

                self.state = PushUpState.TOP
                self.min_elbow_angle = self.last_completed_min_angle if is_valid_rep else 180.0
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self.rep_body_lines.clear()
                self._last_top_frame = cur_frame
                self._last_top_ts_ms = now_ms
                self._last_top_time = now
                self._last_top_angle = e_angle

            elif (
                self._active_rep is not None
                and self._active_rep["peak_ascent_angle"] >= 135.0
                and e_angle < self._active_rep["peak_ascent_angle"] - 20.0
                and e_angle < 125.0
            ):
                # Athlete failed top lockout and descended directly into next rep
                issues = ["INCOMPLETE_LOCKOUT"]
                if not self.reached_bottom_depth:
                    issues.append("INSUFFICIENT_DEPTH")
                self._record_rep_episode(
                    end_frame=self._last_valid_frame,
                    end_ts_ms=self._last_valid_ts_ms,
                    is_valid=False,
                    issues=issues,
                )
                self.state = PushUpState.DESCENDING
                self.rep_start_time = now
                self.min_elbow_angle = e_angle
                self.reached_bottom_depth = False
                self.consecutive_bottom_frames = 0
                self.rep_body_lines = [self.body_alignment] if has_valid_alignment else []
                self._start_active_rep(cur_frame, now_ms, now, e_angle, use_top_anchor=False)
                self.feedback = "Lock out elbows fully between reps"

            elif e_angle <= self.config.bottom_angle:
                self.state = PushUpState.BOTTOM
                self.bottom_enter_time = now
                self.consecutive_bottom_frames += 1

        self._last_valid_frame = cur_frame
        self._last_valid_ts_ms = now_ms

        self.depth_quality = self._classify_depth(
            self.min_elbow_angle if self.state != PushUpState.TOP else self.current_elbow_angle
        )

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
            movement_duration_sec=mov_duration,
        )

    def _classify_depth(self, angle: float) -> str:
        """Classify push-up depth from elbow angle: GOOD (<=bottom), MODERATE (<=105), SHALLOW (>105)."""
        if angle <= self.config.bottom_angle:
            return "GOOD"
        elif angle <= 105.0:
            return "MODERATE"
        return "SHALLOW"
