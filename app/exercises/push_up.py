import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.biomechanics.phase_detector import ExercisePhaseDetector
from app.biomechanics.push_up_counter import (
    PushUpCounterConfig,
    PushUpRepCounter,
    extract_push_up_geometry,
)
from app.core.config import settings
from app.exercises.base import (
    BaseExerciseAnalyzer,
    build_failed_sequence_result,
    compute_angle_smoothness_score,
    is_reliable_kp,
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

_PUSH_UP_METRIC_SPECS = [
    ("range_of_motion", "Elbow Range of Motion", "deg", ">= 60° (Bottom <= 95°, Top >= 155°)"),
    ("shoulder_position", "Shoulder Stacking Over Wrists", "ratio", "<= 0.50"),
    ("body_alignment", "Body Plank Alignment", "deg", "165° - 180°"),
    ("hip_sag", "Hip Sag Rate", "%", "<= 10%"),
    ("hip_piking", "Hip Piking Rate", "%", "<= 10%"),
    ("elbow_symmetry", "Bilateral Elbow Symmetry", "deg", "<= 15°"),
    ("movement_consistency", "Movement Consistency", "score", ">= 80"),
]


class PushUpAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Push-ups.
    Evaluates:
    - Body alignment (plank line: shoulder-hip-ankle 180° ± 15°)
    - Range of motion (depth: elbow <= 90°-95° at bottom, lockout: elbow >= 155°-160° at top)
    - Elbow movement & symmetry (when bilateral arms visible in non-side view)
    - Shoulder positioning & stacking over wrists
    - Repetition counting and phase detection
    """

    def __init__(self):
        self._fallback_counter = PushUpRepCounter()
        self.phase_detector = ExercisePhaseDetector(
            exercise_name="PUSH_UP",
            is_dynamic=True,
            min_rom_delta=40.0,
            min_rep_duration_sec=0.7
        )

    @property
    def name(self) -> str:
        return "PUSH_UP"

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def camera_view_recommendation(self) -> str:
        return "Position camera side-on at floor level or 45° angle, full body visible from hands to feet."

    @property
    def required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_ankle", "right_ankle"
        ]

    @property
    def scoring_weights(self) -> Dict[str, float]:
        return {
            "body_alignment": 0.25,
            "range_of_motion": 0.25,
            "elbow_movement": 0.20,
            "shoulder_positioning": 0.15,
            "movement_control": 0.15
        }

    def validate_pose(
        self,
        keypoints: Dict[str, KeypointData],
        threshold: float = 0.25,
        side_view: bool = False
    ) -> Tuple[bool, LiveAnalysisStatus, List[str]]:
        if not keypoints:
            return False, LiveAnalysisStatus.NO_PERSON_DETECTED, ["all_keypoints"]

        geo = extract_push_up_geometry(keypoints, threshold=threshold, side_view=side_view)
        if not geo["is_pose_valid"]:
            return False, LiveAnalysisStatus.INCOMPLETE_POSE, geo["missing"]

        has_hip = (
            is_reliable_kp(keypoints.get("left_hip"), threshold)
            or is_reliable_kp(keypoints.get("right_hip"), threshold)
        )
        has_leg = (
            is_reliable_kp(keypoints.get("left_ankle"), threshold)
            or is_reliable_kp(keypoints.get("right_ankle"), threshold)
            or is_reliable_kp(keypoints.get("left_knee"), threshold)
            or is_reliable_kp(keypoints.get("right_knee"), threshold)
        )

        missing = []
        if not has_hip:
            missing.append("hip")
        if not has_leg:
            missing.append("legs/ankles")

        if missing:
            return False, LiveAnalysisStatus.INCOMPLETE_POSE, missing

        return True, LiveAnalysisStatus.VALID, []

    def extract_measurements(
        self,
        keypoints: Dict[str, KeypointData],
        side_view: bool = False
    ) -> Dict[str, float]:
        geo = extract_push_up_geometry(
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
        if geo["body_alignment"] is not None:
            measurements["body_line_angle"] = round(float(geo["body_alignment"]), 1)
            measurements["hip_sag_indicator"] = 1.0 if geo["hip_sag"] else 0.0
            measurements["hip_pike_indicator"] = 1.0 if geo["hip_pike"] else 0.0
        if geo["shoulder_offset_px"] is not None:
            measurements["shoulder_offset_px"] = round(float(geo["shoulder_offset_px"]), 1)
        if geo["shoulder_position_ratio"] is not None:
            measurements["shoulder_position_ratio"] = round(float(geo["shoulder_position_ratio"]), 2)

        return measurements

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        elbow = measurements.get("elbow_angle")
        if elbow is None:
            return None, [], False

        body_line = measurements.get("body_line_angle")
        is_sag = measurements.get("hip_sag_indicator", 0.0) > 0.5
        is_pike = measurements.get("hip_pike_indicator", 0.0) > 0.5
        elbow_diff = measurements.get("elbow_symmetry_diff")

        feedback: List[LiveFeedbackItem] = []
        is_valid = True

        # Check Body Alignment (only when body line landmarks were reliable)
        if body_line is not None:
            if is_sag and body_line < 165.0:
                feedback.append(LiveFeedbackItem(
                    code="HIP_SAG",
                    message="Avoid letting your hips sag. Engage your core and glutes.",
                    severity=FeedbackSeverity.WARNING
                ))
                is_valid = False
            elif is_pike and body_line < 165.0:
                feedback.append(LiveFeedbackItem(
                    code="HIP_PIKE",
                    message="Keep your hips aligned with your shoulders. Lower your hips.",
                    severity=FeedbackSeverity.WARNING
                ))
                is_valid = False

        # Range of Motion checks
        if elbow > 105.0 and self.phase_detector.current_phase == "BOTTOM":
            feedback.append(LiveFeedbackItem(
                code="INSUFFICIENT_DEPTH",
                message="Lower your chest further for full range of motion.",
                severity=FeedbackSeverity.WARNING
            ))

        if elbow < 155.0 and self.phase_detector.current_phase in ("TOP", "START"):
            feedback.append(LiveFeedbackItem(
                code="INCOMPLETE_LOCKOUT",
                message="Extend your elbows fully at the top of each rep.",
                severity=FeedbackSeverity.INFO
            ))

        # Arm symmetry check
        if elbow_diff is not None and not side_view and elbow_diff > 25.0:
            feedback.append(LiveFeedbackItem(
                code="UNEVEN_PRESS",
                message="Press evenly through both arms.",
                severity=FeedbackSeverity.WARNING
            ))

        # Calculate composite score normalized over available components
        rom_score = 90.0
        if self.phase_detector.current_phase == "BOTTOM":
            rom_score = max(40.0, min(100.0, 100.0 - max(0.0, elbow - 90.0) * 2.0))

        weighted_sum = 0.35 * rom_score
        total_weight = 0.35

        if body_line is not None:
            alignment_score = max(0.0, 100.0 - abs(180.0 - body_line) * 2.0)
            weighted_sum += 0.40 * alignment_score
            total_weight += 0.40

        if elbow_diff is not None and not side_view:
            symmetry_score = max(0.0, 100.0 - elbow_diff * 2.5)
            weighted_sum += 0.25 * symmetry_score
            total_weight += 0.25

        overall_score = round(weighted_sum / total_weight, 1)

        if not feedback:
            feedback.append(LiveFeedbackItem(
                code="GOOD_FORM",
                message="Great body line and depth. Keep pressing strong!",
                severity=FeedbackSeverity.INFO
            ))

        return overall_score, feedback, is_valid

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        elbow = measurements.get("elbow_angle")
        body_line = measurements.get("body_line_angle")
        elbow_diff = measurements.get("elbow_symmetry_diff")
        sh_ratio = measurements.get("shoulder_position_ratio")

        if sh_ratio is None and keypoints:
            geo = extract_push_up_geometry(
                keypoints,
                threshold=settings.KEYPOINT_CONF_THRESHOLD,
                side_view=side_view,
            )
            sh_ratio = geo.get("shoulder_position_ratio")

        metrics: List[LiveScoringMetric] = []

        # 1. Body Alignment
        if body_line is not None:
            align_score = max(0.0, min(100.0, 100.0 - abs(180.0 - body_line) * 2.0))
            metrics.append(LiveScoringMetric(
                name="body_alignment",
                value=round(body_line, 1),
                score=round(align_score, 1),
                weight=0.35,
                confidence=0.95,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="body_alignment", value=0.0, score=0.0, weight=0.35, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 2. Range of Motion
        if elbow is not None:
            if self.phase_detector.current_phase == "BOTTOM":
                rom_score = max(20.0, min(100.0, 100.0 - max(0.0, elbow - 90.0) * 2.0))
            elif self.phase_detector.current_phase in ("TOP", "START"):
                rom_score = max(40.0, min(100.0, 100.0 - max(0.0, 165.0 - elbow) * 2.0))
            else:
                rom_score = max(40.0, min(100.0, 100.0 - max(0.0, elbow - 95.0) * 0.6))
            metrics.append(LiveScoringMetric(
                name="range_of_motion",
                value=round(elbow, 1),
                score=round(rom_score, 1),
                weight=0.35,
                confidence=0.92,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="range_of_motion", value=0.0, score=0.0, weight=0.35, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 3. Elbow Positioning & Symmetry
        if elbow_diff is not None and not side_view:
            elbow_score = max(0.0, min(100.0, 100.0 - elbow_diff * 2.0))
            metrics.append(LiveScoringMetric(
                name="elbow_movement",
                value=round(elbow_diff, 1),
                score=round(elbow_score, 1),
                weight=0.30,
                confidence=0.90,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="elbow_movement", value=0.0, score=0.0, weight=0.0 if side_view else 0.30, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 4. Shoulder Positioning (shoulder horizontal stacking relative to wrist)
        if sh_ratio is not None:
            sh_score = 100.0 if sh_ratio <= 0.50 else max(20.0, 100.0 - (sh_ratio - 0.50) * 100.0)
            metrics.append(LiveScoringMetric(
                name="shoulder_positioning",
                value=round(sh_ratio, 2),
                score=round(sh_score, 1),
                weight=0.15,
                confidence=0.85,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="shoulder_positioning",
                value=0.0,
                score=0.0,
                weight=0.0,
                confidence=0.0,
                status="INSUFFICIENT_DATA"
            ))

        # 5. Movement Control / Tempo Consistency (evaluated at sequence level across completed reps)
        metrics.append(LiveScoringMetric(
            name="movement_control",
            value=0.0,
            score=0.0,
            weight=0.0,
            confidence=0.0,
            status="INSUFFICIENT_DATA"
        ))

        return metrics

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        if hasattr(session, "get_push_up_counter"):
            counter = session.get_push_up_counter()
        elif hasattr(session, "push_up_counter") and session.push_up_counter is not None:
            counter = session.push_up_counter
        else:
            counter = self._fallback_counter

        if not is_valid_frame:
            rep_res = counter.process_angle(None, timestamp_sec=timestamp_sec)
        else:
            elbow = measurements.get("elbow_angle")
            body_line = measurements.get("body_line_angle")
            rep_res = counter.process_angle(elbow, body_alignment=body_line, timestamp_sec=timestamp_sec)

        if hasattr(session, "rep_count"):
            session.rep_count = rep_res.rep_count
        if hasattr(session, "rep_phase"):
            session.rep_phase = rep_res.current_state

        return None, rep_res.rep_count, rep_res.current_state

    def analyze_video_sequence(
        self,
        frame_samples: List[Dict[str, Any]],
        fps: float = 10.0,
        side_view: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze an ordered video sequence for Push-Up biomechanics.
        Returns v1.0 structured analysis with per-rep segmentation,
        movementMetrics, hip sag/piking detection, and non-fabricated scores.
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
                metric_specs=_PUSH_UP_METRIC_SPECS,
                side_view=side_view,
            )

        threshold = settings.KEYPOINT_CONF_THRESHOLD
        per_frame_geo: List[Dict[str, Any]] = []
        geo_by_frame_idx: Dict[int, Dict[str, Any]] = {}
        missing_counts: Dict[str, int] = {}
        valid_pose_frames = 0

        for s in samples:
            if s["pose_detected"]:
                geo = extract_push_up_geometry(s["keypoints"], threshold=threshold, side_view=side_view)
            else:
                geo = extract_push_up_geometry({}, threshold=threshold, side_view=side_view)
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
                metric_specs=_PUSH_UP_METRIC_SPECS,
                missing_landmarks=top_missing,
                side_view=side_view,
            )

        counter = PushUpRepCounter(
            PushUpCounterConfig(
                bottom_angle=95.0,
                bottom_exit_angle=102.0,
                descending_abort_angle=150.0,
                meaningful_attempt_angle=135.0,
                ema_alpha=0.50,
            )
        )
        timeline: List[Dict[str, Any]] = []
        valid_elbow_angles: List[float] = []

        for s, geo in zip(samples, per_frame_geo):
            elbow = geo["primary_elbow_angle"] if geo["is_pose_valid"] else None
            if elbow is not None:
                valid_elbow_angles.append(float(elbow))
            res = counter.process_angle(
                elbow,
                body_alignment=geo["body_alignment"],
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
                "bodyAlignment": geo["body_alignment"],
                "hipSag": geo["hip_sag"],
                "hipPike": geo["hip_pike"],
                "shoulderPositionRatio": geo["shoulder_position_ratio"],
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

        rep_shoulder_ratios: List[float] = []
        rep_body_lines: List[float] = []
        rep_sym_diffs: List[float] = []

        for rep in rep_episodes:
            rep_num = rep["repNumber"]
            f_indices = rep["frameIndices"]
            rep_geos = [geo_by_frame_idx[fi] for fi in f_indices if fi in geo_by_frame_idx and geo_by_frame_idx[fi]["is_pose_valid"]]

            rep_sh_ratios = [float(g["shoulder_position_ratio"]) for g in rep_geos if g["shoulder_position_ratio"] is not None]
            rep_aligns = [float(g["body_alignment"]) for g in rep_geos if g["body_alignment"] is not None]
            rep_sags = [g for g in rep_geos if g["hip_sag"] and g["body_alignment"] is not None and g["body_alignment"] < 165.0]
            rep_pikes = [g for g in rep_geos if g["hip_pike"] and g["body_alignment"] is not None and g["body_alignment"] < 165.0]
            rep_diffs = [float(g["elbow_symmetry_diff"]) for g in rep_geos if g["elbow_symmetry_diff"] is not None]

            avg_sh_ratio = round(float(np.mean(rep_sh_ratios)), 2) if rep_sh_ratios else None
            avg_align = round(float(np.mean(rep_aligns)), 1) if rep_aligns else None
            min_align = round(min(rep_aligns), 1) if rep_aligns else None
            avg_diff = round(float(np.mean(rep_diffs)), 1) if rep_diffs else None

            if avg_sh_ratio is not None:
                rep_shoulder_ratios.append(avg_sh_ratio)
            if avg_align is not None:
                rep_body_lines.append(avg_align)
            if avg_diff is not None:
                rep_sym_diffs.append(avg_diff)

            rep_issues = list(rep["issues"])
            rep_feedback: List[str] = []

            min_elbow = float(rep["minAngleDeg"])
            max_elbow = float(rep["maxAngleDeg"])

            if "INSUFFICIENT_DEPTH" in rep_issues:
                _add_issue_occurrence("INSUFFICIENT_DEPTH", rep_num, min_elbow)
                rep_feedback.append(f"Lower your chest deeper: minimum elbow angle was {min_elbow:.1f}° (target <= 95°).")
            if "INCOMPLETE_LOCKOUT" in rep_issues:
                _add_issue_occurrence("INCOMPLETE_LOCKOUT", rep_num, max_elbow)
                rep_feedback.append(f"Extend elbows fully at top lockout (reached {max_elbow:.1f}°, target >= 155°).")

            # Bottom + Top ROM subscore
            if min_elbow <= 95.0:
                bottom_sc = min(100.0, 100.0 - max(0.0, min_elbow - 90.0) * 1.0)
            elif min_elbow <= 115.0:
                bottom_sc = max(45.0, 95.0 - (min_elbow - 95.0) * 2.0)
            else:
                bottom_sc = max(15.0, 55.0 - (min_elbow - 115.0) * 1.0)

            top_sc = 100.0 if max_elbow >= 155.0 else max(35.0, 100.0 - (155.0 - max_elbow) * 2.0)
            rom_sc = 0.65 * bottom_sc + 0.35 * top_sc

            score_parts: List[Tuple[float, float]] = [(rom_sc, 0.45)]

            # Body alignment subscore (only when reliable)
            if avg_align is not None and min_align is not None:
                align_sc = max(10.0, min(100.0, 100.0 - max(0.0, 175.0 - min_align) * 2.0))
                score_parts.append((align_sc, 0.35))

                sag_ratio = len(rep_sags) / len(rep_aligns)
                pike_ratio = len(rep_pikes) / len(rep_aligns)
                if rep_sags and (sag_ratio >= 0.20 or len(rep_sags) >= 2):
                    if "HIP_SAG" not in rep_issues:
                        rep_issues.append("HIP_SAG")
                    min_sag_ang = min(float(g["body_alignment"]) for g in rep_sags)
                    _add_issue_occurrence("HIP_SAG", rep_num, min_sag_ang)
                    rep_feedback.append("Hips sagged below plank line; brace core and squeeze glutes.")
                if rep_pikes and (pike_ratio >= 0.20 or len(rep_pikes) >= 2):
                    if "HIP_PIKE" not in rep_issues:
                        rep_issues.append("HIP_PIKE")
                    min_pike_ang = min(float(g["body_alignment"]) for g in rep_pikes)
                    _add_issue_occurrence("HIP_PIKE", rep_num, min_pike_ang)
                    rep_feedback.append("Hips piked upward; lower hips in line with shoulders and ankles.")

            # Shoulder stacking subscore
            if avg_sh_ratio is not None:
                sh_sc = 100.0 if avg_sh_ratio <= 0.50 else max(25.0, 100.0 - (avg_sh_ratio - 0.50) * 100.0)
                score_parts.append((sh_sc, 0.10))
                if avg_sh_ratio > 0.65:
                    if "SHOULDER_MISALIGNMENT" not in rep_issues:
                        rep_issues.append("SHOULDER_MISALIGNMENT")
                    _add_issue_occurrence("SHOULDER_MISALIGNMENT", rep_num, avg_sh_ratio)
                    rep_feedback.append("Keep shoulders stacked over wrists instead of shifting excessively.")

            # Symmetry subscore (non-side view only)
            if not side_view and avg_diff is not None:
                sym_sc = 100.0 if avg_diff <= 8.0 else max(15.0, 100.0 - (avg_diff - 8.0) * 2.5)
                score_parts.append((sym_sc, 0.10))
                if avg_diff > 22.0:
                    if "UNEVEN_PRESS" not in rep_issues:
                        rep_issues.append("UNEVEN_PRESS")
                    _add_issue_occurrence("UNEVEN_PRESS", rep_num, avg_diff)
                    rep_feedback.append("Press evenly through both arms to maintain symmetric lockout.")

            total_w = sum(w for _, w in score_parts)
            rep_form_score = round(sum(sc * w for sc, w in score_parts) / total_w, 1)
            if not rep["isValid"]:
                rep_form_score = round(min(rep_form_score, 65.0), 1)

            if not rep_feedback:
                rep_feedback.append("Strong push-up rep with solid depth and straight plank alignment.")

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
                    "bodyAlignmentDeg": avg_align,
                    "minBodyAlignmentDeg": min_align,
                    "shoulderPositionRatio": avg_sh_ratio,
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

        # 1. range_of_motion
        if reps_output:
            rom_vals = [float(r["romDeg"]) for r in reps_output]
            avg_bottom = float(np.mean([float(r["minAngleDeg"]) for r in reps_output]))
            avg_top = float(np.mean([float(r["maxAngleDeg"]) for r in reps_output]))
            r_val = float(np.mean(rom_vals))
            r_min, r_max, r_avg = min(rom_vals), max(rom_vals), r_val
        else:
            avg_bottom = min(valid_elbow_angles)
            avg_top = max(valid_elbow_angles)
            r_val = max(0.0, avg_top - avg_bottom)
            r_min, r_max, r_avg = r_val, r_val, r_val

        b_score = 100.0 if avg_bottom <= 95.0 else (max(45.0, 95.0 - (avg_bottom - 95.0) * 2.0) if avg_bottom <= 115.0 else max(15.0, 55.0 - (avg_bottom - 115.0) * 1.0))
        t_score = 100.0 if avg_top >= 155.0 else max(35.0, 100.0 - (155.0 - avg_top) * 2.0)
        rom_score = round(0.65 * b_score + 0.35 * t_score, 1)
        rom_status = "VALID" if (avg_bottom <= 95.0 and avg_top >= 155.0) else ("WARNING" if rom_score >= 60.0 else "ERROR")

        movement_metrics["range_of_motion"] = make_movement_metric(
            name="Elbow Range of Motion",
            unit="deg",
            value=r_val,
            min_val=r_min,
            max_val=r_max,
            avg_val=r_avg,
            target_range=">= 60° (Bottom <= 95°, Top >= 155°)",
            score=rom_score,
            status=rom_status,
            confidence=arm_conf,
        )

        # 2. shoulder_position
        all_sh_ratios = [float(g["shoulder_position_ratio"]) for g in valid_geos if g["shoulder_position_ratio"] is not None]
        if all_sh_ratios:
            sh_confs = [float(g["shoulder_confidence"]) for g in valid_geos if g["shoulder_position_ratio"] is not None]
            sh_conf = float(np.mean(sh_confs))
            sh_val = float(np.mean(rep_shoulder_ratios)) if rep_shoulder_ratios else float(np.mean(all_sh_ratios))
            if sh_val <= 0.50:
                sh_score, sh_status = 100.0, "VALID"
            elif sh_val <= 0.70:
                sh_score, sh_status = 100.0 - (sh_val - 0.50) * 100.0, "WARNING"
            else:
                sh_score = max(20.0, 80.0 - (sh_val - 0.70) * 120.0)
                sh_status = "ERROR" if sh_score < 50.0 else "WARNING"
            if not reps_output and sh_val > 0.65:
                _add_issue_occurrence("SHOULDER_MISALIGNMENT", None, sh_val)

            movement_metrics["shoulder_position"] = make_movement_metric(
                name="Shoulder Stacking Over Wrists",
                unit="ratio",
                value=sh_val,
                min_val=min(all_sh_ratios),
                max_val=max(all_sh_ratios),
                avg_val=float(np.mean(all_sh_ratios)),
                target_range="<= 0.50",
                score=sh_score,
                status=sh_status,
                confidence=sh_conf,
                decimals=2,
            )
        else:
            movement_metrics["shoulder_position"] = make_unavailable_metric(
                name="Shoulder Stacking Over Wrists", unit="ratio", target_range="<= 0.50"
            )

        # 3-5. body_alignment, hip_sag, hip_piking (UNAVAILABLE if hip/ankle unreliable)
        align_geos = [g for g in valid_geos if g["body_alignment_reliable"] and g["body_alignment"] is not None]
        if align_geos:
            all_aligns = [float(g["body_alignment"]) for g in align_geos]
            ba_conf = float(np.mean([float(g["body_alignment_confidence"]) for g in align_geos]))
            ba_val = float(np.mean(all_aligns))
            ba_score = max(0.0, min(100.0, 100.0 - max(0.0, 175.0 - ba_val) * 2.2))
            ba_status = "VALID" if ba_val >= 165.0 else ("WARNING" if ba_val >= 155.0 else "ERROR")

            movement_metrics["body_alignment"] = make_movement_metric(
                name="Body Plank Alignment",
                unit="deg",
                value=ba_val,
                min_val=min(all_aligns),
                max_val=max(all_aligns),
                avg_val=ba_val,
                target_range="165° - 180°",
                score=ba_score,
                status=ba_status,
                confidence=ba_conf,
            )

            sag_geos = [g for g in align_geos if g["hip_sag"] and float(g["body_alignment"]) < 165.0]
            sag_pct = round(100.0 * len(sag_geos) / len(align_geos), 1)
            sag_score = 100.0 if sag_pct <= 5.0 else max(0.0, 100.0 - sag_pct * 1.2)
            sag_status = "VALID" if sag_pct <= 10.0 else ("WARNING" if sag_pct <= 35.0 else "ERROR")
            if sag_geos and (sag_pct > 15.0 or len(sag_geos) >= 2) and "HIP_SAG" not in issue_tracker:
                _add_issue_occurrence("HIP_SAG", None, min(float(g["body_alignment"]) for g in sag_geos))

            movement_metrics["hip_sag"] = make_movement_metric(
                name="Hip Sag Rate",
                unit="%",
                value=sag_pct,
                min_val=0.0,
                max_val=sag_pct,
                avg_val=sag_pct,
                target_range="<= 10%",
                score=sag_score,
                status=sag_status,
                confidence=ba_conf,
            )

            pike_geos = [g for g in align_geos if g["hip_pike"] and float(g["body_alignment"]) < 165.0]
            pike_pct = round(100.0 * len(pike_geos) / len(align_geos), 1)
            pike_score = 100.0 if pike_pct <= 5.0 else max(0.0, 100.0 - pike_pct * 1.2)
            pike_status = "VALID" if pike_pct <= 10.0 else ("WARNING" if pike_pct <= 35.0 else "ERROR")
            if pike_geos and (pike_pct > 15.0 or len(pike_geos) >= 2) and "HIP_PIKE" not in issue_tracker:
                _add_issue_occurrence("HIP_PIKE", None, min(float(g["body_alignment"]) for g in pike_geos))

            movement_metrics["hip_piking"] = make_movement_metric(
                name="Hip Piking Rate",
                unit="%",
                value=pike_pct,
                min_val=0.0,
                max_val=pike_pct,
                avg_val=pike_pct,
                target_range="<= 10%",
                score=pike_score,
                status=pike_status,
                confidence=ba_conf,
            )
        else:
            movement_metrics["body_alignment"] = make_unavailable_metric(
                name="Body Plank Alignment", unit="deg", target_range="165° - 180°"
            )
            movement_metrics["hip_sag"] = make_unavailable_metric(
                name="Hip Sag Rate", unit="%", target_range="<= 10%"
            )
            movement_metrics["hip_piking"] = make_unavailable_metric(
                name="Hip Piking Rate", unit="%", target_range="<= 10%"
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
                _add_issue_occurrence("UNEVEN_PRESS", None, sym_val)
            movement_metrics["elbow_symmetry"] = make_movement_metric(
                name="Bilateral Elbow Symmetry",
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
                name="Bilateral Elbow Symmetry", unit="deg", target_range="<= 15°"
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
            "INSUFFICIENT_DEPTH": {
                "metric": "range_of_motion",
                "title": "Insufficient Push-Up Depth",
                "message": "Elbows did not bend to target bottom depth (<= 95°) on one or more attempts.",
                "recommendation": "Lower your chest closer to the floor until elbow angle reaches <= 95° before pressing up.",
                "targetRange": "<= 95°",
            },
            "INCOMPLETE_LOCKOUT": {
                "metric": "range_of_motion",
                "title": "Incomplete Top Lockout",
                "message": "Arms did not fully extend (>= 155°) at the top of the repetition.",
                "recommendation": "Press all the way through to full elbow extension at the top of every rep.",
                "targetRange": ">= 155°",
            },
            "HIP_SAG": {
                "metric": "hip_sag",
                "title": "Hip Sagging",
                "message": "Hips dropped below the shoulder-ankle plank line during push-ups.",
                "recommendation": "Brace your core, tuck your pelvis slightly, and squeeze your glutes to hold a rigid plank.",
                "targetRange": "165° - 180°",
            },
            "HIP_PIKE": {
                "metric": "hip_piking",
                "title": "Hip Piking",
                "message": "Hips rose above the straight shoulder-ankle plank line.",
                "recommendation": "Lower your hips in line with your shoulders and heels throughout the movement.",
                "targetRange": "165° - 180°",
            },
            "UNEVEN_PRESS": {
                "metric": "elbow_symmetry",
                "title": "Uneven Arm Press",
                "message": "Noticeable left-to-right asymmetry in elbow flexion during pressing.",
                "recommendation": "Distribute weight evenly across both hands and press symmetrically.",
                "targetRange": "<= 15° diff",
            },
            "SHOULDER_MISALIGNMENT": {
                "metric": "shoulder_position",
                "title": "Shoulder Stacking Drift",
                "message": "Shoulders shifted too far away from vertical alignment over the wrists.",
                "recommendation": "Set hands directly under your shoulders and maintain controlled scapular tracking.",
                "targetRange": "<= 0.50",
            },
        }

        for code, data in issue_tracker.items():
            meta = issue_catalog[code]
            vals = data["values"]
            measured = float(np.mean(vals)) if vals else None
            if code == "INSUFFICIENT_DEPTH":
                measured = max(vals) if vals else None
                sev = "HIGH" if (measured is not None and measured > 115.0) or incomplete_reps > 0 else "MEDIUM"
            elif code in ("HIP_SAG", "HIP_PIKE"):
                measured = min(vals) if vals else None
                sev = "HIGH" if (measured is not None and measured < 155.0) else "MEDIUM"
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
                title="No Push-Up Repetitions Completed",
                message="Athlete pose was tracked, but no full or partial push-up repetitions were detected.",
                recommendation="Lower until elbow angle reaches <= 95° and press back to >= 155°.",
                measured_value=avg_bottom,
                target_range="<= 95°",
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
        for k in ("body_alignment", "shoulder_position", "elbow_symmetry"):
            if movement_metrics[k]["status"] != "UNAVAILABLE" and movement_metrics[k]["score"] is not None:
                align_parts.append(float(movement_metrics[k]["score"]))
        alignment_score = round(float(np.mean(align_parts)), 1) if align_parts else None

        stab_parts: List[float] = []
        if movement_metrics["movement_consistency"]["status"] != "UNAVAILABLE":
            stab_parts.append(float(movement_metrics["movement_consistency"]["score"]))
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
                f"Excellent push-up execution! Completed {valid_reps} valid rep(s) with strong ROM and body line."
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
