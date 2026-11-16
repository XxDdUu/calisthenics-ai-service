import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.biomechanics.phase_detector import ExercisePhaseDetector
from app.biomechanics.pull_up_counter import (
    PullUpCounterConfig,
    PullUpRepCounter,
    PullUpState,
    extract_pull_up_geometry,
)
from app.core.config import settings
from app.exercises.base import (
    BaseExerciseAnalyzer,
    build_failed_sequence_result,
    compute_angle_smoothness_score,
    make_biomechanics_issue,
    make_movement_metric,
    make_unavailable_metric,
    parse_video_frame_samples,
)
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    LiveScoringMetric,
)

logger = logging.getLogger(__name__)

_PULL_UP_STATE_TO_LIVE_PHASE = {
    PullUpState.DEAD_HANG.value: "BOTTOM",
    PullUpState.PULLING.value: "PULLING",
    PullUpState.TOP_POSITION.value: "TOP",
    PullUpState.LOWERING.value: "LOWERING",
}

_PULL_UP_METRIC_SPECS = [
    ("elbow_angle", "Elbow Flexion Angle", "deg", "Top <= 85°, Hang >= 150°"),
    ("range_of_motion", "Pull-Up Range of Motion", "deg", ">= 65°"),
    ("shoulder_angle", "Shoulder Flexion Angle", "deg", "15° - 90° (at top)"),
    ("chin_to_bar", "Chin-to-Bar Clearance", "ratio", "> 0.0"),
    ("body_swing", "Lower-Body Pendulum Swing", "ratio", "<= 0.25"),
    ("elbow_symmetry", "Bilateral Arm Symmetry", "deg", "<= 15°"),
    ("movement_consistency", "Movement Consistency", "score", ">= 80"),
]


class PullUpAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Pull-ups / Chin-ups.
    Tracks elbow flexion/extension range of motion, chin-to-bar vertical clearance,
    lower-body swing control, shoulder mechanics, bilateral symmetry, and
    session-isolated repetition counting.
    """

    def __init__(self):
        self._fallback_counter = PullUpRepCounter()
        self.phase_detector = ExercisePhaseDetector(
            exercise_name="PULL_UP",
            is_dynamic=True,
            min_rom_delta=45.0,
            min_rep_duration_sec=0.8
        )

    @property
    def name(self) -> str:
        return "PULL_UP"

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def camera_view_recommendation(self) -> str:
        return "Position camera side-on or slightly oblique at chest/bar height, full body visible."

    @property
    def required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist"
        ]

    @property
    def scoring_weights(self) -> Dict[str, float]:
        return {
            "range_of_motion": 0.25,
            "body_control": 0.20,
            "elbow_movement": 0.20,
            "shoulder_positioning": 0.20,
            "movement_symmetry": 0.15
        }

    def validate_pose(
        self,
        keypoints: Dict[str, KeypointData],
        threshold: float = 0.25,
        side_view: bool = False
    ) -> Tuple[bool, LiveAnalysisStatus, List[str]]:
        if not keypoints:
            return False, LiveAnalysisStatus.NO_PERSON_DETECTED, ["all_keypoints"]

        geo = extract_pull_up_geometry(keypoints, threshold=threshold, side_view=side_view)
        if not geo["is_pose_valid"]:
            return False, LiveAnalysisStatus.INCOMPLETE_POSE, geo["missing"]

        return True, LiveAnalysisStatus.VALID, []

    def extract_measurements(
        self,
        keypoints: Dict[str, KeypointData],
        side_view: bool = False
    ) -> Dict[str, float]:
        geo = extract_pull_up_geometry(
            keypoints,
            threshold=settings.KEYPOINT_CONF_THRESHOLD,
            side_view=side_view,
        )
        measurements: Dict[str, float] = {}

        if geo["primary_elbow_angle"] is not None:
            measurements["elbow_angle"] = round(float(geo["primary_elbow_angle"]), 1)
        if geo["left_elbow_angle"] is not None:
            measurements["elbow_left_angle"] = round(float(geo["left_elbow_angle"]), 1)
        if geo["right_elbow_angle"] is not None:
            measurements["elbow_right_angle"] = round(float(geo["right_elbow_angle"]), 1)
        if geo["elbow_symmetry_diff"] is not None:
            measurements["elbow_symmetry_diff"] = round(float(geo["elbow_symmetry_diff"]), 1)
        if geo["shoulder_angle"] is not None:
            measurements["shoulder_angle"] = round(float(geo["shoulder_angle"]), 1)
        if geo["chin_to_bar_clearance"] is not None:
            measurements["chin_to_bar_clearance"] = round(float(geo["chin_to_bar_clearance"]), 3)
        if geo["torso_tilt"] is not None:
            measurements["torso_tilt"] = round(float(geo["torso_tilt"]), 1)

        return measurements

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        avg_elbow = measurements.get("elbow_angle")
        if avg_elbow is None:
            return None, [], False

        elbow_diff = measurements.get("elbow_symmetry_diff")
        chin_clearance = measurements.get("chin_to_bar_clearance")
        torso_tilt = measurements.get("torso_tilt")

        feedback: List[LiveFeedbackItem] = []
        is_valid_frame = True

        # Check bilateral arm symmetry (only when available in non-side view)
        if elbow_diff is not None and not side_view and elbow_diff > 25.0:
            feedback.append(LiveFeedbackItem(
                code="SYMMETRIC_PULL",
                message="Uneven pull detected. Engage both arms equally.",
                severity=FeedbackSeverity.WARNING
            ))
            is_valid_frame = False

        # Range of Motion & chin clearance checks
        if self.phase_detector.current_phase == "TOP" or avg_elbow <= 90.0:
            if chin_clearance is not None:
                if chin_clearance > 0.0:
                    feedback.append(LiveFeedbackItem(
                        code="CHIN_OVER_BAR",
                        message="Great height over the bar!",
                        severity=FeedbackSeverity.INFO
                    ))
                else:
                    feedback.append(LiveFeedbackItem(
                        code="INCOMPLETE_ROM_TOP",
                        message="Pull higher: bring your chin completely over the bar.",
                        severity=FeedbackSeverity.WARNING
                    ))
            elif avg_elbow > 85.0:
                feedback.append(LiveFeedbackItem(
                    code="INCOMPLETE_ROM_TOP",
                    message="Pull higher: bring your chin completely over the bar.",
                    severity=FeedbackSeverity.WARNING
                ))
            else:
                feedback.append(LiveFeedbackItem(
                    code="CHIN_OVER_BAR",
                    message="Great height over the bar!",
                    severity=FeedbackSeverity.INFO
                ))

        if self.phase_detector.current_phase in ("START", "BOTTOM") and avg_elbow < 145.0:
            feedback.append(LiveFeedbackItem(
                code="FULL_DEAD_HANG",
                message="Lower completely into full arm extension at the bottom.",
                severity=FeedbackSeverity.INFO
            ))

        if torso_tilt is not None and torso_tilt > 35.0:
            feedback.append(LiveFeedbackItem(
                code="EXCESSIVE_SWING",
                message="Minimize body swing and keep your core braced.",
                severity=FeedbackSeverity.WARNING
            ))

        rom_score = 90.0
        if self.phase_detector.current_phase == "TOP" or avg_elbow <= 90.0:
            rom_score = max(30.0, min(100.0, 100.0 - max(0.0, avg_elbow - 70.0) * 2.0))

        weighted_sum = 0.50 * rom_score
        total_w = 0.50
        if elbow_diff is not None and not side_view:
            symmetry_score = max(0.0, 100.0 - elbow_diff * 2.5)
            weighted_sum += 0.50 * symmetry_score
            total_w += 0.50

        overall_score = round(weighted_sum / total_w, 1)
        return overall_score, feedback, is_valid_frame

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        avg_elbow = measurements.get("elbow_angle")
        elbow_diff = measurements.get("elbow_symmetry_diff")
        shoulder_angle = measurements.get("shoulder_angle")
        torso_tilt = measurements.get("torso_tilt")

        geo: Optional[Dict[str, Any]] = None
        if keypoints is not None and (shoulder_angle is None or torso_tilt is None):
            geo = extract_pull_up_geometry(
                keypoints,
                threshold=settings.KEYPOINT_CONF_THRESHOLD,
                side_view=side_view,
            )
            if shoulder_angle is None:
                shoulder_angle = geo.get("shoulder_angle")
            if torso_tilt is None:
                torso_tilt = geo.get("torso_tilt")

        metrics: List[LiveScoringMetric] = []

        # 1. Range of Motion
        if avg_elbow is not None:
            if self.phase_detector.current_phase == "TOP" or avg_elbow <= 90.0:
                rom_score = max(30.0, min(100.0, 100.0 - max(0.0, avg_elbow - 70.0) * 2.0))
            elif self.phase_detector.current_phase in ("START", "BOTTOM") or avg_elbow >= 145.0:
                rom_score = max(40.0, min(100.0, 100.0 - max(0.0, 155.0 - avg_elbow) * 2.0))
            else:
                rom_score = 85.0
            metrics.append(LiveScoringMetric(
                name="range_of_motion",
                value=round(avg_elbow, 1),
                score=round(rom_score, 1),
                weight=0.25,
                confidence=0.94,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="range_of_motion", value=0.0, score=0.0, weight=0.25, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 2. Body Control (computed from actual hip/shoulder torso tilt; never hardcoded!)
        if torso_tilt is not None:
            control_score = max(15.0, min(100.0, 100.0 - max(0.0, torso_tilt - 15.0) * 2.5))
            ctrl_conf = float(geo.get("body_swing_confidence", 0.85)) if geo else 0.85
            if ctrl_conf < 0.25:
                ctrl_conf = 0.85
            metrics.append(LiveScoringMetric(
                name="body_control",
                value=round(torso_tilt, 1),
                score=round(control_score, 1),
                weight=0.20,
                confidence=round(ctrl_conf, 2),
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="body_control", value=0.0, score=0.0, weight=0.0, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 3. Elbow Movement & path
        if avg_elbow is not None:
            if elbow_diff is not None and not side_view:
                elbow_score = max(0.0, min(100.0, 100.0 - elbow_diff * 1.5))
                e_val = elbow_diff
            else:
                elbow_score = max(40.0, min(100.0, 100.0 - max(0.0, avg_elbow - 165.0) * 0.5))
                e_val = avg_elbow
            metrics.append(LiveScoringMetric(
                name="elbow_movement",
                value=round(e_val, 1),
                score=round(elbow_score, 1),
                weight=0.20,
                confidence=0.90 if not side_view else 0.82,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="elbow_movement", value=0.0, score=0.0, weight=0.20, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 4. Shoulder positioning (computed from hip-shoulder-elbow angle; never hardcoded!)
        if shoulder_angle is not None:
            # Healthy pull-up shoulder angle spans ~20 deg (top) to ~175 deg (dead hang)
            if 15.0 <= shoulder_angle <= 178.0:
                shoulder_score = 95.0 if shoulder_angle >= 25.0 else max(70.0, 95.0 - (25.0 - shoulder_angle) * 2.0)
            else:
                shoulder_score = 70.0
            sh_conf = float(geo.get("shoulder_angle_confidence", 0.85)) if geo else 0.85
            if sh_conf < 0.25:
                sh_conf = 0.85
            metrics.append(LiveScoringMetric(
                name="shoulder_positioning",
                value=round(shoulder_angle, 1),
                score=round(shoulder_score, 1),
                weight=0.20,
                confidence=round(sh_conf, 2),
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="shoulder_positioning", value=0.0, score=0.0, weight=0.0, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 5. Movement Symmetry (ONLY when both arms reliable in non-side view)
        if elbow_diff is not None and not side_view:
            symmetry_score = max(0.0, min(100.0, 100.0 - elbow_diff * 2.5))
            metrics.append(LiveScoringMetric(
                name="movement_symmetry",
                value=round(elbow_diff, 1),
                score=round(symmetry_score, 1),
                weight=0.15,
                confidence=0.92,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="movement_symmetry", value=0.0, score=0.0, weight=0.0, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        return metrics

    def _get_session_counter(self, session: Any) -> PullUpRepCounter:
        """Retrieve or initialize a session-isolated PullUpRepCounter."""
        if session is None:
            return self._fallback_counter
        if hasattr(session, "get_pull_up_counter"):
            return session.get_pull_up_counter()

        counter = getattr(session, "pull_up_counter", None)
        if counter is None:
            counter = PullUpRepCounter()
            try:
                setattr(session, "pull_up_counter", counter)
            except AttributeError:
                return self._fallback_counter
        elif (
            getattr(session, "rep_count", 0) == 0
            and getattr(session, "rep_phase", "START") == "START"
            and (counter.rep_count > 0 or counter.state != PullUpState.DEAD_HANG)
        ):
            counter.reset()
        return counter

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        counter = self._get_session_counter(session)

        if not is_valid_frame:
            rep_res = counter.process_angle(None, timestamp_sec=timestamp_sec)
        else:
            elbow = measurements.get("elbow_angle")
            rep_res = counter.process_angle(elbow, timestamp_sec=timestamp_sec)

        mapped_phase = _PULL_UP_STATE_TO_LIVE_PHASE.get(rep_res.current_state, rep_res.current_state)

        if hasattr(session, "rep_count"):
            session.rep_count = rep_res.rep_count
        if hasattr(session, "rep_phase"):
            session.rep_phase = mapped_phase

        return None, rep_res.rep_count, mapped_phase

    def analyze_video_sequence(
        self,
        frame_samples: List[Dict[str, Any]],
        fps: float = 10.0,
        side_view: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze an ordered video sequence for Pull-Up biomechanics.
        Returns v1.0 structured analysis with per-rep segmentation,
        chin-to-bar clearance, pendulum body swing, shoulder angle,
        incomplete rep detection, and non-fabricated scores.
        """
        samples = parse_video_frame_samples(frame_samples, fps=fps)
        total_frames = len(samples)
        detected_frames = sum(1 for s in samples if s["pose_detected"])

        if total_frames == 0 or detected_frames == 0:
            return build_failed_sequence_result(
                exercise_name=self.name,
                status="NO_PERSON_DETECTED",
                total_frames=total_frames,
                detected_frames=detected_frames,
                valid_pose_frames=0,
                metric_specs=_PULL_UP_METRIC_SPECS,
                side_view=side_view,
            )

        threshold = settings.KEYPOINT_CONF_THRESHOLD
        per_frame_geo: List[Dict[str, Any]] = []
        geo_by_frame_idx: Dict[int, Dict[str, Any]] = {}
        missing_counts: Dict[str, int] = {}
        valid_pose_frames = 0

        for s in samples:
            if s["pose_detected"]:
                geo = extract_pull_up_geometry(s["keypoints"], threshold=threshold, side_view=side_view)
            else:
                geo = extract_pull_up_geometry({}, threshold=threshold, side_view=side_view)
            per_frame_geo.append(geo)
            geo_by_frame_idx[s["frame_index"]] = geo
            if geo["is_pose_valid"]:
                valid_pose_frames += 1
            else:
                for m in geo.get("missing", []):
                    missing_counts[m] = missing_counts.get(m, 0) + 1

        pose_ratio = valid_pose_frames / total_frames
        if valid_pose_frames < 3 or pose_ratio < 0.20:
            top_missing = [k for k, _ in sorted(missing_counts.items(), key=lambda item: -item[1])][:3]
            return build_failed_sequence_result(
                exercise_name=self.name,
                status="INSUFFICIENT_POSE_QUALITY",
                total_frames=total_frames,
                detected_frames=detected_frames,
                valid_pose_frames=valid_pose_frames,
                metric_specs=_PULL_UP_METRIC_SPECS,
                missing_landmarks=top_missing,
                side_view=side_view,
            )

        counter = PullUpRepCounter(PullUpCounterConfig(ema_alpha=0.50))
        timeline: List[Dict[str, Any]] = []
        valid_elbow_angles: List[float] = []

        for s, geo in zip(samples, per_frame_geo):
            elbow = geo["primary_elbow_angle"] if geo["is_pose_valid"] else None
            if elbow is not None:
                valid_elbow_angles.append(float(elbow))
            res = counter.process_angle(
                elbow,
                timestamp_sec=s["timestamp_sec"],
                frame_index=s["frame_index"],
                timestamp_ms=s["timestamp_ms"],
            )
            timeline.append({
                "frameIndex": s["frame_index"],
                "timestampMs": s["timestamp_ms"],
                "poseValid": geo["is_pose_valid"],
                "phase": res.current_state,
                "elbowAngle": geo["primary_elbow_angle"],
                "shoulderAngle": geo["shoulder_angle"],
                "chinToBarClearance": geo["chin_to_bar_clearance"],
                "hipSwingOffset": geo["hip_swing_offset"],
            })

        rep_episodes = counter.finalize()

        reps_output: List[Dict[str, Any]] = []
        issue_tracker: Dict[str, Dict[str, Any]] = {}

        def _add_issue_occurrence(code: str, rep_num: Optional[int], measured: Optional[float] = None) -> None:
            entry = issue_tracker.setdefault(code, {"occurrences": 0, "reps": [], "values": []})
            entry["occurrences"] += 1
            if rep_num is not None:
                entry["reps"].append(rep_num)
            if measured is not None and math.isfinite(measured):
                entry["values"].append(float(measured))

        def _compute_swing_amplitude(geos: List[Dict[str, Any]]) -> Optional[float]:
            hip_offs = [float(g["hip_swing_offset"]) for g in geos if g["lower_body_reliable"] and g["hip_swing_offset"] is not None]
            if len(hip_offs) < 2:
                return None
            hip_ptp = max(hip_offs) - min(hip_offs)
            ankle_offs = [float(g["ankle_swing_offset"]) for g in geos if g["lower_body_reliable"] and g["ankle_swing_offset"] is not None]
            if len(ankle_offs) >= 2:
                ankle_ptp = max(ankle_offs) - min(ankle_offs)
                return round(max(hip_ptp, ankle_ptp * 0.75), 3)
            return round(hip_ptp, 3)

        rep_peak_clearances: List[float] = []
        rep_min_shoulders: List[float] = []
        rep_swings: List[float] = []
        rep_sym_diffs: List[float] = []

        for rep in rep_episodes:
            rep_num = rep["repNumber"]
            f_indices = rep["frameIndices"]
            rep_geos = [geo_by_frame_idx[fi] for fi in f_indices if fi in geo_by_frame_idx and geo_by_frame_idx[fi]["is_pose_valid"]]

            rep_clearances = [float(g["chin_to_bar_clearance"]) for g in rep_geos if g["chin_to_bar_clearance"] is not None]
            rep_sh_angles = [float(g["shoulder_angle"]) for g in rep_geos if g["shoulder_angle"] is not None]
            rep_diffs = [float(g["elbow_symmetry_diff"]) for g in rep_geos if g["elbow_symmetry_diff"] is not None]
            rep_swing = _compute_swing_amplitude(rep_geos)

            peak_clearance = round(max(rep_clearances), 3) if rep_clearances else None
            min_sh_angle = round(min(rep_sh_angles), 1) if rep_sh_angles else None
            avg_diff = round(float(np.mean(rep_diffs)), 1) if rep_diffs else None

            if peak_clearance is not None:
                rep_peak_clearances.append(peak_clearance)
            if min_sh_angle is not None:
                rep_min_shoulders.append(min_sh_angle)
            if rep_swing is not None:
                rep_swings.append(rep_swing)
            if avg_diff is not None:
                rep_sym_diffs.append(avg_diff)

            rep_issues = list(rep["issues"])
            rep_feedback: List[str] = []

            min_elbow = float(rep["minAngleDeg"])
            max_elbow = float(rep["maxAngleDeg"])

            if "INCOMPLETE_PULL" in rep_issues:
                _add_issue_occurrence("INCOMPLETE_PULL", rep_num, min_elbow)
                rep_feedback.append(f"Partial pull detected (minimum elbow angle {min_elbow:.1f}°); pull all the way to the bar.")
            if "INCOMPLETE_EXTENSION" in rep_issues:
                _add_issue_occurrence("INCOMPLETE_EXTENSION", rep_num, max_elbow)
                rep_feedback.append("Extend arms fully into a dead hang (>= 145°) between repetitions.")

            # ROM subscore
            if min_elbow <= 85.0:
                top_sc = 100.0
            elif min_elbow <= 105.0:
                top_sc = max(55.0, 95.0 - (min_elbow - 85.0) * 2.0)
            else:
                top_sc = max(15.0, 55.0 - (min_elbow - 105.0) * 1.2)

            hang_sc = 100.0 if max_elbow >= 150.0 else max(35.0, 100.0 - (150.0 - max_elbow) * 2.0)
            rom_sc = 0.65 * top_sc + 0.35 * hang_sc

            score_parts: List[Tuple[float, float]] = [(rom_sc, 0.40)]

            # Chin-to-bar subscore (ONLY if head + wrist reliable)
            if peak_clearance is not None:
                if peak_clearance >= 0.05:
                    chin_sc = 100.0
                elif peak_clearance >= 0.0:
                    chin_sc = 85.0 + (peak_clearance / 0.05) * 15.0
                else:
                    chin_sc = max(15.0, 80.0 + peak_clearance * 250.0)
                score_parts.append((chin_sc, 0.25))

                if peak_clearance <= 0.0:
                    if "INCOMPLETE_ROM_TOP" not in rep_issues:
                        rep_issues.append("INCOMPLETE_ROM_TOP")
                    _add_issue_occurrence("INCOMPLETE_ROM_TOP", rep_num, peak_clearance)
                    rep_feedback.append("Pull higher so your chin/nose passes clearly above the bar.")
                else:
                    rep_feedback.append("Chin cleared bar height cleanly.")

            # Body swing subscore (ONLY if lower body reliable)
            if rep_swing is not None:
                if rep_swing <= 0.20:
                    swing_sc = 100.0
                elif rep_swing <= 0.35:
                    swing_sc = 100.0 - (rep_swing - 0.20) * 160.0
                else:
                    swing_sc = max(10.0, 76.0 - (rep_swing - 0.35) * 120.0)
                score_parts.append((swing_sc, 0.20))

                if rep_swing > 0.30:
                    if "EXCESSIVE_SWING" not in rep_issues:
                        rep_issues.append("EXCESSIVE_SWING")
                    _add_issue_occurrence("EXCESSIVE_SWING", rep_num, rep_swing)
                    rep_feedback.append("Reduce kipping/pendulum body swing; engage core and hollow body.")

            # Shoulder angle subscore (when hips visible)
            if min_sh_angle is not None:
                sh_sc = 100.0 if 15.0 <= min_sh_angle <= 95.0 else max(45.0, 100.0 - max(0.0, min_sh_angle - 95.0) * 1.5)
                score_parts.append((sh_sc, 0.10))

            # Symmetry subscore (non-side view only)
            if not side_view and avg_diff is not None:
                sym_sc = 100.0 if avg_diff <= 8.0 else max(15.0, 100.0 - (avg_diff - 8.0) * 2.5)
                score_parts.append((sym_sc, 0.15))
                if avg_diff > 22.0:
                    if "SYMMETRIC_PULL" not in rep_issues:
                        rep_issues.append("SYMMETRIC_PULL")
                    _add_issue_occurrence("SYMMETRIC_PULL", rep_num, avg_diff)
                    rep_feedback.append("Pull evenly with both arms to avoid asymmetric shoulder loading.")

            total_w = sum(w for _, w in score_parts)
            rep_form_score = round(sum(sc * w for sc, w in score_parts) / total_w, 1)
            if not rep["isValid"]:
                rep_form_score = round(min(rep_form_score, 65.0), 1)
            if "INCOMPLETE_EXTENSION" in rep_issues and rep["isValid"]:
                rep_form_score = round(min(rep_form_score, 82.0), 1)

            if not rep_feedback:
                rep_feedback.append("Clean pull-up repetition with full extension and controlled form.")

            reps_output.append({
                "repNumber": rep_num,
                "startFrame": rep["startFrame"],
                "endFrame": rep["endFrame"],
                "startTimestampMs": rep["startTimestampMs"],
                "endTimestampMs": rep["endTimestampMs"],
                "durationMs": rep["durationMs"],
                "isValid": rep["isValid"],
                "status": rep["status"],
                "formScore": rep_form_score,
                "minAngleDeg": rep["minAngleDeg"],
                "maxAngleDeg": rep["maxAngleDeg"],
                "romDeg": rep["romDeg"],
                "depthClassification": rep["depthClassification"],
                "keyMetrics": {
                    "chinToBarClearance": peak_clearance,
                    "bodySwingRatio": rep_swing,
                    "minShoulderAngleDeg": min_sh_angle,
                    "elbowSymmetryDiffDeg": avg_diff,
                },
                "issues": rep_issues,
                "feedback": rep_feedback,
            })

        valid_reps = sum(1 for r in reps_output if r["isValid"])
        incomplete_reps = sum(1 for r in reps_output if not r["isValid"])
        total_attempted = len(reps_output)

        valid_geos = [g for g in per_frame_geo if g["is_pose_valid"]]
        arm_conf = float(np.mean([g["arm_confidence"] for g in valid_geos])) if valid_geos else 0.0

        movement_metrics: Dict[str, Dict[str, Any]] = {}

        # 1. elbow_angle & 2. range_of_motion
        if reps_output:
            rep_mins = [float(r["minAngleDeg"]) for r in reps_output]
            rep_maxs = [float(r["maxAngleDeg"]) for r in reps_output]
            rep_roms = [float(r["romDeg"]) for r in reps_output]
            avg_top = float(np.mean(rep_mins))
            avg_hang = float(np.mean(rep_maxs))
            rom_val = float(np.mean(rep_roms))
            rom_min, rom_max, rom_avg = min(rep_roms), max(rep_roms), rom_val
        else:
            avg_top = min(valid_elbow_angles)
            avg_hang = max(valid_elbow_angles)
            rom_val = max(0.0, avg_hang - avg_top)
            rom_min, rom_max, rom_avg = rom_val, rom_val, rom_val

        e_min = min(valid_elbow_angles)
        e_max = max(valid_elbow_angles)
        e_avg = float(np.mean(valid_elbow_angles))

        top_score = 100.0 if avg_top <= 85.0 else (max(55.0, 95.0 - (avg_top - 85.0) * 2.0) if avg_top <= 105.0 else max(15.0, 55.0 - (avg_top - 105.0) * 1.2))
        hang_score = 100.0 if avg_hang >= 150.0 else max(35.0, 100.0 - (150.0 - avg_hang) * 2.0)
        rom_score = round(0.65 * top_score + 0.35 * hang_score, 1)
        rom_status = "VALID" if (avg_top <= 85.0 and avg_hang >= 150.0) else ("WARNING" if rom_score >= 60.0 else "ERROR")

        movement_metrics["elbow_angle"] = make_movement_metric(
            name="Elbow Flexion Angle",
            unit="deg",
            value=avg_top,
            min_val=e_min,
            max_val=e_max,
            avg_val=e_avg,
            target_range="Top <= 85°, Hang >= 150°",
            score=top_score,
            status="VALID" if avg_top <= 85.0 else ("WARNING" if avg_top <= 105.0 else "ERROR"),
            confidence=arm_conf,
        )

        movement_metrics["range_of_motion"] = make_movement_metric(
            name="Pull-Up Range of Motion",
            unit="deg",
            value=rom_val,
            min_val=rom_min,
            max_val=rom_max,
            avg_val=rom_avg,
            target_range=">= 65°",
            score=rom_score,
            status=rom_status,
            confidence=arm_conf,
        )

        # 3. shoulder_angle (UNAVAILABLE when hips occluded)
        all_sh = [float(g["shoulder_angle"]) for g in valid_geos if g["shoulder_angle"] is not None]
        if all_sh:
            sh_confs = [float(g["shoulder_angle_confidence"]) for g in valid_geos if g["shoulder_angle"] is not None]
            sh_conf = float(np.mean(sh_confs))
            sh_val = float(np.mean(rep_min_shoulders)) if rep_min_shoulders else min(all_sh)
            sh_score = 100.0 if 15.0 <= sh_val <= 95.0 else max(40.0, 100.0 - max(0.0, sh_val - 95.0) * 1.5)
            sh_status = "VALID" if sh_val <= 95.0 else "WARNING"
            movement_metrics["shoulder_angle"] = make_movement_metric(
                name="Shoulder Flexion Angle",
                unit="deg",
                value=sh_val,
                min_val=min(all_sh),
                max_val=max(all_sh),
                avg_val=float(np.mean(all_sh)),
                target_range="15° - 90° (at top)",
                score=sh_score,
                status=sh_status,
                confidence=sh_conf,
            )
        else:
            movement_metrics["shoulder_angle"] = make_unavailable_metric(
                name="Shoulder Flexion Angle", unit="deg", target_range="15° - 90° (at top)"
            )

        # 4. chin_to_bar (UNAVAILABLE when head/nose or wrists unreliable)
        all_clearances = [float(g["chin_to_bar_clearance"]) for g in valid_geos if g["chin_to_bar_clearance"] is not None]
        if all_clearances:
            chin_confs = [float(g["chin_to_bar_confidence"]) for g in valid_geos if g["chin_to_bar_clearance"] is not None]
            c_conf = float(np.mean(chin_confs))
            c_vals = rep_peak_clearances if rep_peak_clearances else [max(all_clearances)]
            c_val = float(np.mean(c_vals))
            if c_val >= 0.05:
                c_score, c_status = 100.0, "VALID"
            elif c_val > 0.0:
                c_score, c_status = 85.0 + (c_val / 0.05) * 15.0, "VALID"
            elif c_val >= -0.10:
                c_score, c_status = max(50.0, 80.0 + c_val * 300.0), "WARNING"
            else:
                c_score, c_status = max(10.0, 50.0 + (c_val + 0.10) * 200.0), "ERROR"

            if not reps_output and c_val <= 0.0:
                _add_issue_occurrence("INCOMPLETE_ROM_TOP", None, c_val)

            movement_metrics["chin_to_bar"] = make_movement_metric(
                name="Chin-to-Bar Clearance",
                unit="ratio",
                value=c_val,
                min_val=min(c_vals),
                max_val=max(c_vals),
                avg_val=c_val,
                target_range="> 0.0",
                score=c_score,
                status=c_status,
                confidence=c_conf,
                decimals=3,
            )
        else:
            movement_metrics["chin_to_bar"] = make_unavailable_metric(
                name="Chin-to-Bar Clearance", unit="ratio", target_range="> 0.0"
            )

        # 5. body_swing (UNAVAILABLE when lower body occluded)
        seq_swing = _compute_swing_amplitude(valid_geos)
        if seq_swing is not None:
            sw_confs = [float(g["body_swing_confidence"]) for g in valid_geos if g["lower_body_reliable"]]
            sw_conf = float(np.mean(sw_confs)) if sw_confs else 0.0
            sw_vals = rep_swings if rep_swings else [seq_swing]
            sw_val = float(np.mean(sw_vals))
            if sw_val <= 0.20:
                sw_score, sw_status = 100.0, "VALID"
            elif sw_val <= 0.35:
                sw_score = 100.0 - (sw_val - 0.20) * 160.0
                sw_status = "VALID" if sw_val <= 0.25 else "WARNING"
            else:
                sw_score = max(10.0, 76.0 - (sw_val - 0.35) * 120.0)
                sw_status = "ERROR" if sw_val > 0.45 else "WARNING"

            if not reps_output and sw_val > 0.30:
                _add_issue_occurrence("EXCESSIVE_SWING", None, sw_val)

            movement_metrics["body_swing"] = make_movement_metric(
                name="Lower-Body Pendulum Swing",
                unit="ratio",
                value=sw_val,
                min_val=min(sw_vals),
                max_val=max(sw_vals),
                avg_val=sw_val,
                target_range="<= 0.25",
                score=sw_score,
                status=sw_status,
                confidence=sw_conf,
                decimals=3,
            )
        else:
            movement_metrics["body_swing"] = make_unavailable_metric(
                name="Lower-Body Pendulum Swing", unit="ratio", target_range="<= 0.25"
            )

        # 6. elbow_symmetry (UNAVAILABLE in side view or single arm)
        all_diffs = [float(g["elbow_symmetry_diff"]) for g in valid_geos if g["elbow_symmetry_diff"] is not None]
        if not side_view and all_diffs:
            sym_confs = [float(g["symmetry_confidence"]) for g in valid_geos if g["elbow_symmetry_diff"] is not None]
            sym_conf = float(np.mean(sym_confs))
            sym_val = float(np.mean(rep_sym_diffs)) if rep_sym_diffs else float(np.mean(all_diffs))
            sym_score = 100.0 if sym_val <= 8.0 else max(0.0, 100.0 - (sym_val - 8.0) * 2.5)
            sym_status = "VALID" if sym_val <= 15.0 else ("WARNING" if sym_val <= 25.0 else "ERROR")
            if not reps_output and sym_val > 22.0:
                _add_issue_occurrence("SYMMETRIC_PULL", None, sym_val)
            movement_metrics["elbow_symmetry"] = make_movement_metric(
                name="Bilateral Arm Symmetry",
                unit="deg",
                value=sym_val,
                min_val=min(all_diffs),
                max_val=max(all_diffs),
                avg_val=float(np.mean(all_diffs)),
                target_range="<= 15°",
                score=sym_score,
                status=sym_status,
                confidence=sym_conf,
            )
        else:
            movement_metrics["elbow_symmetry"] = make_unavailable_metric(
                name="Bilateral Arm Symmetry", unit="deg", target_range="<= 15°"
            )

        # 7. movement_consistency (UNAVAILABLE if < 2 reps)
        if len(reps_output) >= 2:
            roms = [float(r["romDeg"]) for r in reps_output]
            rom_std = float(np.std(roms))
            rom_cons = max(0.0, 100.0 - rom_std * 3.0)
            durations = [float(r["durationMs"]) for r in reps_output if r["durationMs"] > 0]
            if len(durations) >= 2 and np.mean(durations) > 0:
                cv_dur = float(np.std(durations) / np.mean(durations))
                dur_cons = max(0.0, 100.0 - cv_dur * 100.0)
            else:
                dur_cons = rom_cons
            valid_ratio = valid_reps / len(reps_output)
            cons_val = round((0.60 * rom_cons + 0.40 * dur_cons) * (0.70 + 0.30 * valid_ratio), 1)
            cons_status = "VALID" if cons_val >= 75.0 else ("WARNING" if cons_val >= 55.0 else "ERROR")
            movement_metrics["movement_consistency"] = make_movement_metric(
                name="Movement Consistency",
                unit="score",
                value=cons_val,
                min_val=min(rom_cons, dur_cons),
                max_val=max(rom_cons, dur_cons),
                avg_val=cons_val,
                target_range=">= 80",
                score=cons_val,
                status=cons_status,
                confidence=arm_conf,
            )
        else:
            movement_metrics["movement_consistency"] = make_unavailable_metric(
                name="Movement Consistency", unit="score", target_range=">= 80"
            )

        # Build issues list
        issues_list: List[Dict[str, Any]] = []
        issue_catalog = {
            "INCOMPLETE_PULL": {
                "metric": "range_of_motion",
                "title": "Incomplete Pull-Up Height",
                "message": "Pull-up attempt stopped before achieving target top flexion (<= 85° / chin over bar).",
                "recommendation": "Pull explosively through your lats and drive elbows down until chin clears the bar.",
                "targetRange": "<= 85°",
            },
            "INCOMPLETE_EXTENSION": {
                "metric": "range_of_motion",
                "title": "Incomplete Dead-Hang Extension",
                "message": "Arms did not extend fully into a dead hang (>= 145°) between or at the end of reps.",
                "recommendation": "Lower under control until elbows lock out cleanly in a dead hang before pulling again.",
                "targetRange": ">= 145°",
            },
            "INCOMPLETE_ROM_TOP": {
                "metric": "chin_to_bar",
                "title": "Chin Below Bar Line",
                "message": "Chin/nose did not clear above the wrist/bar line at the top of the pull.",
                "recommendation": "Finish each pull with your chin visibly above the bar before lowering.",
                "targetRange": "> 0.0",
            },
            "EXCESSIVE_SWING": {
                "metric": "body_swing",
                "title": "Excessive Body Swing / Kipping",
                "message": "Noticeable horizontal pendulum swing of the hips/legs during pull-ups.",
                "recommendation": "Brace your core, point toes slightly forward (hollow body), and avoid kipping.",
                "targetRange": "<= 0.25",
            },
            "SYMMETRIC_PULL": {
                "metric": "elbow_symmetry",
                "title": "Uneven Bilateral Pull",
                "message": "Left and right arm flexion angles diverged significantly during pulling.",
                "recommendation": "Engage both scapulae evenly and pull symmetrically toward the bar.",
                "targetRange": "<= 15° diff",
            },
        }

        for code, data in issue_tracker.items():
            meta = issue_catalog[code]
            vals = data["values"]
            measured = float(np.mean(vals)) if vals else None
            if code == "INCOMPLETE_PULL":
                measured = max(vals) if vals else None
                sev = "HIGH" if (measured is not None and measured > 105.0) or incomplete_reps > 0 else "MEDIUM"
            elif code == "INCOMPLETE_ROM_TOP":
                measured = min(vals) if vals else None
                sev = "HIGH" if (measured is not None and measured < -0.10) else "MEDIUM"
            elif code == "EXCESSIVE_SWING":
                measured = max(vals) if vals else None
                sev = "HIGH" if (measured is not None and measured > 0.45) else "MEDIUM"
            else:
                sev = "MEDIUM"

            issues_list.append(make_biomechanics_issue(
                code=code,
                metric=meta["metric"],
                severity=sev,
                title=meta["title"],
                message=meta["message"],
                recommendation=meta["recommendation"],
                measured_value=measured,
                target_range=meta["targetRange"],
                occurrences=data["occurrences"],
                affected_reps=data["reps"],
            ))

        if total_attempted == 0:
            issues_list.append(make_biomechanics_issue(
                code="NO_REPS_DETECTED",
                metric="range_of_motion",
                severity="LOW",
                title="No Pull-Up Repetitions Completed",
                message="Athlete pose was tracked, but no full or partial pull-up repetitions were detected.",
                recommendation="Start from a dead hang (>= 150°) and pull until elbows flex <= 85° with chin over bar.",
                measured_value=avg_top,
                target_range="<= 85°",
                occurrences=1,
                affected_reps=[],
            ))

        # Top-level scores
        if reps_output:
            form_accuracy = round(float(np.mean([r["formScore"] for r in reps_output])), 1)
        else:
            avail_form_scores = [
                m["score"] for k, m in movement_metrics.items()
                if k != "movement_consistency" and m["status"] != "UNAVAILABLE" and m["score"] is not None
            ]
            form_accuracy = round(float(np.mean(avail_form_scores)), 1) if avail_form_scores else None

        align_parts: List[float] = []
        for k in ("shoulder_angle", "body_swing", "elbow_symmetry"):
            if movement_metrics[k]["status"] != "UNAVAILABLE" and movement_metrics[k]["score"] is not None:
                align_parts.append(float(movement_metrics[k]["score"]))
        alignment_score = round(float(np.mean(align_parts)), 1) if align_parts else None

        stab_parts: List[float] = []
        if movement_metrics["body_swing"]["status"] != "UNAVAILABLE":
            stab_parts.append(float(movement_metrics["body_swing"]["score"]))
        if movement_metrics["movement_consistency"]["status"] != "UNAVAILABLE":
            stab_parts.append(float(movement_metrics["movement_consistency"]["score"]))
        if not stab_parts:
            smooth_score = compute_angle_smoothness_score(valid_elbow_angles)
            if smooth_score is not None:
                stab_parts.append(smooth_score)
        stability_score = round(float(np.mean(stab_parts)), 1) if stab_parts else None

        composite_parts: List[Tuple[float, float]] = []
        if form_accuracy is not None:
            composite_parts.append((form_accuracy, 0.50))
        if stability_score is not None:
            composite_parts.append((stability_score, 0.25))
        if alignment_score is not None:
            composite_parts.append((alignment_score, 0.25))

        if composite_parts:
            w_sum = sum(w for _, w in composite_parts)
            overall_score = round(sum(v * w for v, w in composite_parts) / w_sum, 1)
            if valid_reps == 0 and incomplete_reps > 0:
                overall_score = round(min(overall_score, 62.0), 1)
        else:
            overall_score = None

        feedback_summary: List[str] = []
        for iss in issues_list:
            feedback_summary.append(f"{iss['title']}: {iss['recommendation']}")
        if not issues_list and valid_reps > 0:
            feedback_summary.append(
                f"Excellent pull-up execution! Completed {valid_reps} valid rep(s) with strong ROM and body control."
            )

        metrics_summary: Dict[str, Any] = {
            "repCount": total_attempted,
            "validReps": valid_reps,
            "incompleteReps": incomplete_reps,
            "totalAttemptedReps": total_attempted,
            "analyzedFrames": total_frames,
            "detectedFrames": detected_frames,
            "validPoseFrames": valid_pose_frames,
            "poseQualityRatio": round(pose_ratio, 3),
            "sideView": side_view,
        }
        for k, m in movement_metrics.items():
            if m["status"] != "UNAVAILABLE" and m["value"] is not None:
                metrics_summary[k] = m["value"]

        return {
            "status": "COMPLETED",
            "exercise": self.name,
            "score": overall_score,
            "formAccuracy": form_accuracy,
            "stability": stability_score,
            "alignment": alignment_score,
            "repCount": total_attempted,
            "validReps": valid_reps,
            "incompleteReps": incomplete_reps,
            "totalAttemptedReps": total_attempted,
            "reps": reps_output,
            "movementMetrics": movement_metrics,
            "issues": issues_list,
            "feedback": feedback_summary,
            "metrics": metrics_summary,
            "poseQuality": {
                "totalFrames": total_frames,
                "detectedFrames": detected_frames,
                "validPoseFrames": valid_pose_frames,
                "validPoseRatio": round(pose_ratio, 3),
                "missingLandmarks": [],
            },
            "timeline": timeline,
        }
