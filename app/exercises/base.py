import abc
import math
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import get_point, is_valid_coord, is_valid_point
from app.biomechanics.confidence import is_valid_keypoint
from app.core.config import settings
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    LiveScoringMetric,
)


def make_unavailable_metric(
    name: str,
    unit: str,
    target_range: str,
) -> Dict[str, Any]:
    """Build a standard v1.0 movementMetrics entry when landmarks are unavailable."""
    return {
        "name": name,
        "unit": unit,
        "value": None,
        "min": None,
        "max": None,
        "avg": None,
        "targetRange": target_range,
        "score": None,
        "status": "UNAVAILABLE",
        "confidence": 0.0,
    }


def make_movement_metric(
    name: str,
    unit: str,
    value: Optional[float],
    min_val: Optional[float],
    max_val: Optional[float],
    avg_val: Optional[float],
    target_range: str,
    score: Optional[float],
    status: str,
    confidence: float,
    decimals: int = 1,
) -> Dict[str, Any]:
    """Build a standard v1.0 movementMetrics entry with safe rounding."""
    if status == "UNAVAILABLE" or value is None or score is None:
        return make_unavailable_metric(name=name, unit=unit, target_range=target_range)

    def _r(v: Optional[float]) -> Optional[float]:
        if v is None or not math.isfinite(v):
            return None
        return round(float(v), decimals)

    return {
        "name": name,
        "unit": unit,
        "value": _r(value),
        "min": _r(min_val),
        "max": _r(max_val),
        "avg": _r(avg_val),
        "targetRange": target_range,
        "score": round(max(0.0, min(100.0, float(score))), 1),
        "status": status,
        "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
    }


def make_biomechanics_issue(
    code: str,
    metric: str,
    severity: str,
    title: str,
    message: str,
    recommendation: str,
    measured_value: Optional[float] = None,
    target_range: Optional[str] = None,
    occurrences: int = 1,
    affected_reps: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Build a standard v1.0 biomechanical issue record."""
    return {
        "code": code,
        "metric": metric,
        "severity": severity,
        "title": title,
        "message": message,
        "recommendation": recommendation,
        "measuredValue": round(float(measured_value), 2) if measured_value is not None and math.isfinite(measured_value) else None,
        "targetRange": target_range,
        "occurrences": int(occurrences),
        "affectedReps": sorted(set(affected_reps or [])),
    }


def _coerce_keypoints(raw_kps: Any) -> Dict[str, KeypointData]:
    """Normalize dict[str, KeypointData | dict] into dict[str, KeypointData]."""
    if not isinstance(raw_kps, dict):
        return {}
    result: Dict[str, KeypointData] = {}
    for name, kp in raw_kps.items():
        if isinstance(kp, KeypointData):
            result[name] = kp
        elif isinstance(kp, dict):
            try:
                result[name] = KeypointData(
                    x=float(kp.get("x", 0.0)),
                    y=float(kp.get("y", 0.0)),
                    confidence=float(kp.get("confidence", 0.0)),
                    name=str(kp.get("name", name)),
                    visible=bool(kp.get("visible", False)),
                )
            except (TypeError, ValueError):
                continue
    return result


def parse_video_frame_samples(
    frame_samples: List[Dict[str, Any]],
    fps: float = 10.0,
) -> List[Dict[str, Any]]:
    """Normalize frame sample records into consistent fields for sequence analyzers."""
    safe_fps = float(fps) if fps and math.isfinite(fps) and fps > 0 else 10.0
    parsed: List[Dict[str, Any]] = []

    for idx, sample in enumerate(frame_samples or []):
        if not isinstance(sample, dict):
            continue
        frame_idx = int(sample.get("frame_index", idx))
        ts_sec = sample.get("timestamp_sec")
        ts_ms = sample.get("timestamp_ms")

        if ts_sec is None and ts_ms is not None:
            ts_sec = float(ts_ms) / 1000.0
        elif ts_ms is None and ts_sec is not None:
            ts_ms = int(round(float(ts_sec) * 1000.0))
        elif ts_sec is None and ts_ms is None:
            ts_sec = idx / safe_fps
            ts_ms = int(round(ts_sec * 1000.0))

        kps = _coerce_keypoints(sample.get("keypoints") or sample.get("raw_keypoints") or {})
        raw_kps = _coerce_keypoints(sample.get("raw_keypoints") or kps)
        pose_detected = bool(sample.get("pose_detected", bool(kps)))
        det_conf = float(sample.get("detection_confidence", 0.0) or 0.0)

        parsed.append({
            "frame_index": frame_idx,
            "timestamp_ms": int(ts_ms),
            "timestamp_sec": float(ts_sec),
            "keypoints": kps,
            "raw_keypoints": raw_kps,
            "pose_detected": pose_detected and bool(kps),
            "detection_confidence": det_conf,
        })

    return parsed


def is_reliable_kp(kp: Optional[KeypointData], threshold: Optional[float] = None) -> bool:
    """True when keypoint is visible, finite, and meets confidence threshold."""
    if kp is None:
        return False
    conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD
    return bool(
        kp.visible
        and kp.confidence >= conf_thresh
        and is_valid_coord(kp.x)
        and is_valid_coord(kp.y)
    )


def mean_reliable_confidence(
    keypoints: Dict[str, KeypointData],
    names: List[str],
    threshold: Optional[float] = None,
) -> float:
    """Average confidence across reliable keypoints from `names` (0.0 if none)."""
    confs = [
        float(keypoints[n].confidence)
        for n in names
        if n in keypoints and is_reliable_kp(keypoints[n], threshold)
    ]
    return sum(confs) / len(confs) if confs else 0.0


def compute_angle_smoothness_score(values: List[float]) -> Optional[float]:
    """
    Estimate movement trajectory smoothness from discrete second-order differences.
    Returns 0-100 score or None if fewer than 3 samples.
    """
    if len(values) < 3:
        return None
    jerks = [
        abs((values[i] - values[i - 1]) - (values[i - 1] - values[i - 2]))
        for i in range(2, len(values))
    ]
    if not jerks:
        return None
    mean_jerk = sum(jerks) / len(jerks)
    # Smooth human motion has low mean angle acceleration (< 6 deg/step); jittery > 15 deg/step
    return round(max(20.0, min(100.0, 100.0 - max(0.0, mean_jerk - 5.0) * 3.0)), 1)


def build_failed_sequence_result(
    exercise_name: str,
    status: str,
    total_frames: int,
    detected_frames: int,
    valid_pose_frames: int,
    metric_specs: List[Tuple[str, str, str, str]],
    missing_landmarks: Optional[List[str]] = None,
    side_view: bool = False,
) -> Dict[str, Any]:
    """
    Build standard v1.0 response for NO_PERSON_DETECTED or INSUFFICIENT_POSE_QUALITY.
    Never fabricates scores or metrics.
    """
    movement_metrics = {
        key: make_unavailable_metric(name=label, unit=unit, target_range=target_range)
        for key, label, unit, target_range in metric_specs
    }

    if status == "NO_PERSON_DETECTED":
        issue = make_biomechanics_issue(
            code="NO_PERSON_DETECTED",
            metric="pose_visibility",
            severity="HIGH",
            title="No Athlete Detected",
            message="No human pose could be detected in the uploaded video.",
            recommendation="Ensure the athlete is clearly visible in good lighting and centered in the camera frame.",
            occurrences=max(1, total_frames),
            affected_reps=[],
        )
        feedback = ["No human pose detected in the video. Ensure full body visibility and good lighting."]
    else:
        missing_str = ", ".join(missing_landmarks) if missing_landmarks else "required joints"
        issue = make_biomechanics_issue(
            code="INSUFFICIENT_POSE_QUALITY",
            metric="pose_visibility",
            severity="HIGH",
            title="Insufficient Pose Quality",
            message=f"Too few frames had reliable tracking ({valid_pose_frames}/{total_frames}); missing or low-confidence {missing_str}.",
            recommendation="Record from a clearer camera angle with full-body framing, brighter lighting, and minimal occlusion.",
            measured_value=round((valid_pose_frames / total_frames) * 100.0, 1) if total_frames > 0 else 0.0,
            target_range=">= 20% valid frames (>= 3 frames)",
            occurrences=max(1, total_frames - valid_pose_frames),
            affected_reps=[],
        )
        feedback = [
            f"Insufficient pose quality ({valid_pose_frames}/{total_frames} valid frames). "
            f"Ensure {missing_str} remain clearly visible."
        ]

    pose_ratio = round(valid_pose_frames / total_frames, 3) if total_frames > 0 else 0.0

    return {
        "status": status,
        "exercise": exercise_name,
        "score": None,
        "formAccuracy": None,
        "stability": None,
        "alignment": None,
        "repCount": 0,
        "validReps": 0,
        "incompleteReps": 0,
        "totalAttemptedReps": 0,
        "reps": [],
        "movementMetrics": movement_metrics,
        "issues": [issue],
        "feedback": feedback,
        "metrics": {
            "repCount": 0,
            "validReps": 0,
            "incompleteReps": 0,
            "totalAttemptedReps": 0,
            "analyzedFrames": total_frames,
            "detectedFrames": detected_frames,
            "validPoseFrames": valid_pose_frames,
            "poseQualityRatio": pose_ratio,
            "sideView": side_view,
        },
        "poseQuality": {
            "totalFrames": total_frames,
            "detectedFrames": detected_frames,
            "validPoseFrames": valid_pose_frames,
            "validPoseRatio": pose_ratio,
            "missingLandmarks": missing_landmarks or [],
        },
        "timeline": [],
    }


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

    def analyze_video_sequence(
        self,
        frame_samples: List[Dict[str, Any]],
        fps: float = 10.0,
        side_view: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze an ordered sequence of sampled video frames with pose keypoints.
        Returns a structured v1.0 biomechanical analysis dict, or None when not overridden.
        """
        return None
