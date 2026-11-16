import logging
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.biomechanics.squat_counter import (
    SquatCounterConfig,
    SquatRepCounter,
    extract_squat_geometry,
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
from app.biomechanics.angles import get_point
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    LiveScoringMetric,
)

logger = logging.getLogger(__name__)

_SQUAT_METRIC_SPECS = [
    ("squat_depth", "Squat Depth (Min Knee Angle)", "deg", "<= 100°"),
    ("hip_angle", "Hip Flexion Angle", "deg", "45° - 115°"),
    ("torso_inclination", "Torso Inclination", "deg", "<= 40°"),
    ("knee_alignment", "Knee Alignment (Valgus / Symmetry)", "ratio", ">= 0.85"),
    ("movement_consistency", "Movement Consistency", "score", ">= 80"),
]


class SquatAnalyzer(BaseExerciseAnalyzer):
    """
    Biomechanical analyzer for Bodyweight Squats.
    Evaluates:
    - Squat depth (GOOD: <= 100°, MODERATE: 100°-120°, SHALLOW: > 120°)
    - Knee alignment & valgus tracking (comparing knee distance to ankle distance)
    - Hip mechanics & torso inclination (excessive forward lean)
    - Complete lockout / extension at top
    - Dedicated session-isolated repetition counting with temporal stability & hysteresis
    """

    def __init__(self):
        # Fallback counter for standalone tests that don't pass a LiveSession
        self._fallback_counter = SquatRepCounter()

    @property
    def name(self) -> str:
        return "SQUAT"

    @property
    def is_dynamic(self) -> bool:
        return True

    @property
    def camera_view_recommendation(self) -> str:
        return "Position camera side-on or 45° angle at knee/hip height, full body visible."

    @property
    def required_landmarks(self) -> List[str]:
        return [
            "left_shoulder", "right_shoulder",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle"
        ]

    @property
    def scoring_weights(self) -> Dict[str, float]:
        return {
            "squat_depth": 0.35,
            "knee_alignment": 0.25,
            "torso_position": 0.25,
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

        geo = extract_squat_geometry(keypoints, threshold=threshold, side_view=side_view)
        if not geo["is_pose_valid"]:
            return False, LiveAnalysisStatus.INCOMPLETE_POSE, geo["missing"]

        has_sh = (
            is_reliable_kp(keypoints.get("left_shoulder"), threshold)
            or is_reliable_kp(keypoints.get("right_shoulder"), threshold)
        )
        if not has_sh:
            return False, LiveAnalysisStatus.INCOMPLETE_POSE, ["shoulder"]

        return True, LiveAnalysisStatus.VALID, []

    def extract_measurements(
        self,
        keypoints: Dict[str, KeypointData],
        side_view: bool = False
    ) -> Dict[str, float]:
        geo = extract_squat_geometry(
            keypoints,
            threshold=settings.KEYPOINT_CONF_THRESHOLD,
            side_view=side_view,
        )
        measurements: Dict[str, float] = {}

        if geo["primary_knee_angle"] is not None:
            measurements["knee_angle"] = round(float(geo["primary_knee_angle"]), 1)
        if geo["knee_left_angle"] is not None:
            measurements["knee_left_angle"] = round(float(geo["knee_left_angle"]), 1)
        if geo["knee_right_angle"] is not None:
            measurements["knee_right_angle"] = round(float(geo["knee_right_angle"]), 1)
        if geo["avg_knee_angle"] is not None:
            measurements["avg_knee_angle"] = round(float(geo["avg_knee_angle"]), 1)
        if geo["hip_y"] is not None:
            measurements["hip_y"] = round(float(geo["hip_y"]), 1)
        if geo["knee_symmetry_diff"] is not None:
            measurements["knee_symmetry_diff"] = round(float(geo["knee_symmetry_diff"]), 1)
        if geo["knee_ankle_ratio"] is not None:
            measurements["knee_ankle_ratio"] = round(float(geo["knee_ankle_ratio"]), 2)
        if geo["hip_angle"] is not None:
            measurements["hip_angle"] = round(float(geo["hip_angle"]), 1)
        if geo["torso_angle"] is not None:
            lean = round(float(geo["torso_angle"]), 1)
            measurements["torso_angle"] = lean
            measurements["trunk_lean_angle"] = lean

        return measurements

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        knee = measurements.get("knee_angle")
        if knee is None:
            return None, [], False

        trunk_lean = measurements.get("trunk_lean_angle", measurements.get("torso_angle"))
        knee_diff = measurements.get("knee_symmetry_diff")
        knee_ankle_ratio = measurements.get("knee_ankle_ratio")

        feedback: List[LiveFeedbackItem] = []
        is_valid = True

        # 1. Depth assessment
        if knee <= 100.0:
            feedback.append(LiveFeedbackItem(
                code="GOOD_DEPTH",
                message="Good squat depth!",
                severity=FeedbackSeverity.INFO
            ))
        elif 100.0 < knee <= 120.0:
            feedback.append(LiveFeedbackItem(
                code="MODERATE_DEPTH",
                message="Squat a bit deeper to reach parallel thighs.",
                severity=FeedbackSeverity.WARNING
            ))
        elif 120.0 < knee < 150.0:
            feedback.append(LiveFeedbackItem(
                code="SHALLOW_DEPTH",
                message="Squat deeper: hips should reach knee level.",
                severity=FeedbackSeverity.WARNING
            ))

        # 2. Knee alignment & valgus (only when bilateral metrics available)
        if knee_ankle_ratio is not None and knee_ankle_ratio < 0.82 and not side_view:
            feedback.append(LiveFeedbackItem(
                code="KNEE_VALGUS",
                message="Push your knees out in line with your toes.",
                severity=FeedbackSeverity.WARNING
            ))
        elif knee_diff is not None and knee_diff > 25.0:
            feedback.append(LiveFeedbackItem(
                code="UNEVEN_SQUAT",
                message="Distribute weight evenly between both legs.",
                severity=FeedbackSeverity.WARNING
            ))

        # 3. Torso inclination
        if trunk_lean is not None and trunk_lean > 42.0:
            feedback.append(LiveFeedbackItem(
                code="EXCESSIVE_FORWARD_LEAN",
                message="Keep your chest up and avoid excessive forward lean.",
                severity=FeedbackSeverity.WARNING
            ))
            is_valid = False

        # 4. Incomplete extension lockout
        if 150.0 <= knee < 160.0:
            feedback.append(LiveFeedbackItem(
                code="INCOMPLETE_EXTENSION",
                message="Stand tall and extend hips and knees fully at top.",
                severity=FeedbackSeverity.INFO
            ))

        # Form score normalized over available components
        if knee <= 100.0:
            depth_score = max(70.0, min(100.0, 100.0 - max(0.0, knee - 90.0) * 1.5))
        elif knee >= 155.0:
            depth_score = 95.0
        else:
            depth_score = max(40.0, 100.0 - (knee - 95.0) * 1.2)

        weighted_sum = 0.40 * depth_score
        total_weight = 0.40

        if trunk_lean is not None:
            posture_score = max(0.0, min(100.0, 100.0 - max(0.0, trunk_lean - 25.0) * 2.5))
            weighted_sum += 0.35 * posture_score
            total_weight += 0.35

        if knee_diff is not None or knee_ankle_ratio is not None:
            kd = knee_diff if knee_diff is not None else 0.0
            kar = knee_ankle_ratio if knee_ankle_ratio is not None else 1.0
            alignment_score = max(0.0, min(100.0, 100.0 - kd * 2.0 - max(0.0, 1.0 - kar) * 40.0))
            weighted_sum += 0.25 * alignment_score
            total_weight += 0.25

        overall_score = round(weighted_sum / total_weight, 1)

        return overall_score, feedback, is_valid

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        knee = measurements.get("knee_angle")
        trunk_lean = measurements.get("trunk_lean_angle", measurements.get("torso_angle"))
        knee_diff = measurements.get("knee_symmetry_diff")
        knee_ankle_ratio = measurements.get("knee_ankle_ratio")
        hip_angle = measurements.get("hip_angle")

        metrics: List[LiveScoringMetric] = []

        # 1. Squat Depth
        if knee is not None:
            if knee <= 100.0:
                depth_score = max(60.0, min(100.0, 100.0 - max(0.0, knee - 90.0) * 1.5))
            elif knee >= 155.0:
                depth_score = 95.0
            else:
                depth_score = max(40.0, 100.0 - (knee - 95.0) * 1.2)
            metrics.append(LiveScoringMetric(
                name="squat_depth",
                value=round(knee, 1),
                score=round(depth_score, 1),
                weight=0.35,
                confidence=0.95,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="squat_depth", value=0.0, score=0.0, weight=0.35, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 2. Knee Alignment & Valgus
        if not side_view and (knee_diff is not None or knee_ankle_ratio is not None):
            kd = knee_diff if knee_diff is not None else 0.0
            kar = knee_ankle_ratio if knee_ankle_ratio is not None else 1.0
            knee_score = max(0.0, min(100.0, 100.0 - kd * 2.0 - max(0.0, 1.0 - kar) * 40.0))
            metrics.append(LiveScoringMetric(
                name="knee_alignment",
                value=round(kd, 1),
                score=round(knee_score, 1),
                weight=0.25,
                confidence=0.90,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="knee_alignment", value=0.0, score=0.0, weight=0.25, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 3. Hip Positioning
        if hip_angle is not None:
            hip_score = max(35.0, min(100.0, 100.0 - max(0.0, (180.0 - hip_angle) - 115.0) * 1.5))
            metrics.append(LiveScoringMetric(
                name="hip_positioning",
                value=round(hip_angle, 1),
                score=round(hip_score, 1),
                weight=0.15,
                confidence=0.88,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="hip_positioning", value=0.0, score=0.0, weight=0.15, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 4. Torso Angle
        if trunk_lean is not None:
            trunk_score = max(0.0, min(100.0, 100.0 - max(0.0, trunk_lean - 25.0) * 2.5))
            metrics.append(LiveScoringMetric(
                name="torso_position",
                value=round(trunk_lean, 1),
                score=round(trunk_score, 1),
                weight=0.25,
                confidence=0.92,
                status="VALID"
            ))
        else:
            metrics.append(LiveScoringMetric(
                name="torso_position", value=0.0, score=0.0, weight=0.25, confidence=0.0, status="INSUFFICIENT_DATA"
            ))

        # 5. Balance (Center of Mass alignment over ankles when keypoints available)
        balance_metric = LiveScoringMetric(
            name="balance", value=0.0, score=0.0, weight=0.0, confidence=0.0, status="INSUFFICIENT_DATA"
        )
        if keypoints:
            l_hip = get_point(keypoints.get("left_hip")) if is_reliable_kp(keypoints.get("left_hip")) else None
            r_hip = get_point(keypoints.get("right_hip")) if is_reliable_kp(keypoints.get("right_hip")) else None
            l_ak = get_point(keypoints.get("left_ankle")) if is_reliable_kp(keypoints.get("left_ankle")) else None
            r_ak = get_point(keypoints.get("right_ankle")) if is_reliable_kp(keypoints.get("right_ankle")) else None
            l_sh = get_point(keypoints.get("left_shoulder")) if is_reliable_kp(keypoints.get("left_shoulder")) else None
            r_sh = get_point(keypoints.get("right_shoulder")) if is_reliable_kp(keypoints.get("right_shoulder")) else None
            if (l_hip or r_hip) and (l_ak or r_ak):
                hip_x = ((l_hip[0] + r_hip[0]) / 2.0) if (l_hip and r_hip) else (l_hip or r_hip)[0]
                hip_y = ((l_hip[1] + r_hip[1]) / 2.0) if (l_hip and r_hip) else (l_hip or r_hip)[1]
                ak_x = ((l_ak[0] + r_ak[0]) / 2.0) if (l_ak and r_ak) else (l_ak or r_ak)[0]
                ak_y = ((l_ak[1] + r_ak[1]) / 2.0) if (l_ak and r_ak) else (l_ak or r_ak)[1]
                leg_span = max(30.0, abs(ak_y - hip_y))
                if l_sh or r_sh:
                    sh_y = ((l_sh[1] + r_sh[1]) / 2.0) if (l_sh and r_sh) else (l_sh or r_sh)[1]
                    leg_span = max(leg_span, abs(ak_y - sh_y) * 0.5)
                sway_ratio = abs(hip_x - ak_x) / leg_span
                bal_score = max(20.0, min(100.0, 100.0 - max(0.0, sway_ratio - 0.18) * 140.0))
                balance_metric = LiveScoringMetric(
                    name="balance",
                    value=round(sway_ratio * 100.0, 1),
                    score=round(bal_score, 1),
                    weight=0.10,
                    confidence=0.80,
                    status="VALID"
                )
        metrics.append(balance_metric)

        return metrics

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        if hasattr(session, "get_squat_counter"):
            counter = session.get_squat_counter()
        elif hasattr(session, "squat_counter") and session.squat_counter is not None:
            counter = session.squat_counter
        else:
            counter = self._fallback_counter

        if not is_valid_frame:
            rep_res = counter.process_angle(None, timestamp_sec=timestamp_sec)
        else:
            knee = measurements.get("knee_angle")
            rep_res = counter.process_angle(knee, timestamp_sec=timestamp_sec)

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
        Analyze an ordered video sequence for Bodyweight Squat biomechanics.
        Returns v1.0 structured analysis with per-rep segmentation, movementMetrics,
        issues, and non-fabricated scores.
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
                metric_specs=_SQUAT_METRIC_SPECS,
                side_view=side_view,
            )

        threshold = settings.KEYPOINT_CONF_THRESHOLD
        per_frame_geo: List[Dict[str, Any]] = []
        geo_by_frame_idx: Dict[int, Dict[str, Any]] = {}
        missing_counts: Dict[str, int] = {}
        valid_pose_frames = 0

        for s in samples:
            if s["pose_detected"]:
                geo = extract_squat_geometry(s["keypoints"], threshold=threshold, side_view=side_view)
            else:
                geo = extract_squat_geometry({}, threshold=threshold, side_view=side_view)
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
                metric_specs=_SQUAT_METRIC_SPECS,
                missing_landmarks=top_missing,
                side_view=side_view,
            )

        # Pass through isolated SquatRepCounter
        counter = SquatRepCounter(SquatCounterConfig(ema_alpha=0.50))
        timeline: List[Dict[str, Any]] = []
        valid_knee_angles: List[float] = []

        for s, geo in zip(samples, per_frame_geo):
            knee = geo["primary_knee_angle"] if geo["is_pose_valid"] else None
            if knee is not None:
                valid_knee_angles.append(float(knee))
            res = counter.process_angle(
                knee,
                timestamp_sec=s["timestamp_sec"],
                frame_index=s["frame_index"],
                timestamp_ms=s["timestamp_ms"],
            )
            timeline.append({
                "frameIndex": s["frame_index"],
                "timestampMs": s["timestamp_ms"],
                "poseValid": geo["is_pose_valid"],
                "phase": res.current_state,
                "kneeAngle": geo["primary_knee_angle"],
                "hipAngle": geo["hip_angle"],
                "torsoInclination": geo["torso_angle"],
                "kneeAnkleRatio": geo["knee_ankle_ratio"],
            })

        rep_episodes = counter.finalize()

        # Enrich each rep with biomechanical metrics, formScore, issues, feedback
        reps_output: List[Dict[str, Any]] = []
        issue_tracker: Dict[str, Dict[str, Any]] = {}

        def _add_issue_occurrence(code: str, rep_num: Optional[int], measured: Optional[float] = None) -> None:
            entry = issue_tracker.setdefault(code, {"occurrences": 0, "reps": [], "values": []})
            entry["occurrences"] += 1
            if rep_num is not None:
                entry["reps"].append(rep_num)
            if measured is not None and math.isfinite(measured):
                entry["values"].append(float(measured))

        rep_min_hips: List[float] = []
        rep_peak_leans: List[float] = []
        rep_min_ratios: List[float] = []
        rep_sym_diffs: List[float] = []

        for rep in rep_episodes:
            rep_num = rep["repNumber"]
            f_indices = rep["frameIndices"]
            rep_geos = [geo_by_frame_idx[fi] for fi in f_indices if fi in geo_by_frame_idx and geo_by_frame_idx[fi]["is_pose_valid"]]

            rep_hips = [float(g["hip_angle"]) for g in rep_geos if g["hip_angle"] is not None]
            rep_leans = [float(g["torso_angle"]) for g in rep_geos if g["torso_angle"] is not None]
            # Evaluate knee valgus/symmetry primarily during active squat frames (< 150 deg)
            active_geos = [g for g in rep_geos if g["primary_knee_angle"] is not None and g["primary_knee_angle"] < 150.0] or rep_geos
            rep_ratios = [float(g["knee_ankle_ratio"]) for g in active_geos if g["knee_ankle_ratio"] is not None]
            rep_diffs = [float(g["knee_symmetry_diff"]) for g in active_geos if g["knee_symmetry_diff"] is not None]

            min_hip = round(min(rep_hips), 1) if rep_hips else None
            peak_lean = round(max(rep_leans), 1) if rep_leans else None
            min_ratio = round(min(rep_ratios), 2) if rep_ratios else None
            avg_diff = round(float(np.mean(rep_diffs)), 1) if rep_diffs else None

            if min_hip is not None:
                rep_min_hips.append(min_hip)
            if peak_lean is not None:
                rep_peak_leans.append(peak_lean)
            if min_ratio is not None:
                rep_min_ratios.append(min_ratio)
            if avg_diff is not None:
                rep_sym_diffs.append(avg_diff)

            rep_issues = list(rep["issues"])
            rep_feedback: List[str] = []

            min_knee = float(rep["minAngleDeg"])
            if "SHALLOW_DEPTH" in rep_issues:
                _add_issue_occurrence("SHALLOW_DEPTH", rep_num, min_knee)
                rep_feedback.append(f"Squat deeper: minimum knee angle was {min_knee:.1f}° (target <= 100°).")
            if "INCOMPLETE_EXTENSION" in rep_issues:
                _add_issue_occurrence("INCOMPLETE_EXTENSION", rep_num, rep["maxAngleDeg"])
                rep_feedback.append("Stand tall and lock out hips and knees fully at the top.")

            # Depth subscore
            if rep["isValid"]:
                depth_score = min(100.0, max(80.0, 100.0 - max(0.0, min_knee - 90.0) * 1.5))
            elif min_knee <= 120.0:
                depth_score = max(40.0, 75.0 - (min_knee - 100.0) * 1.5)
            else:
                depth_score = max(15.0, 45.0 - (min_knee - 120.0) * 0.9)

            score_parts: List[Tuple[float, float]] = [(depth_score, 0.50)]

            # Torso inclination subscore
            if peak_lean is not None:
                lean_score = min(100.0, max(15.0, 100.0 - max(0.0, peak_lean - 30.0) * 2.5))
                score_parts.append((lean_score, 0.25))
                if peak_lean > 42.0:
                    if "EXCESSIVE_FORWARD_LEAN" not in rep_issues:
                        rep_issues.append("EXCESSIVE_FORWARD_LEAN")
                    _add_issue_occurrence("EXCESSIVE_FORWARD_LEAN", rep_num, peak_lean)
                    rep_feedback.append(f"Reduce forward torso lean ({peak_lean:.1f}° > 40°); keep chest proud.")

            # Knee alignment subscore (only non-side view with bilateral landmarks)
            if not side_view and (min_ratio is not None or avg_diff is not None):
                valgus_sc = 100.0 if (min_ratio is None or min_ratio >= 0.85) else max(0.0, 100.0 - (0.85 - min_ratio) * 200.0)
                sym_sc = 100.0 if (avg_diff is None or avg_diff <= 10.0) else max(0.0, 100.0 - (avg_diff - 10.0) * 2.5)
                align_sc = 0.70 * valgus_sc + 0.30 * sym_sc if (min_ratio is not None and avg_diff is not None) else (valgus_sc if min_ratio is not None else sym_sc)
                score_parts.append((align_sc, 0.25))

                if min_ratio is not None and min_ratio < 0.82:
                    if "KNEE_VALGUS" not in rep_issues:
                        rep_issues.append("KNEE_VALGUS")
                    _add_issue_occurrence("KNEE_VALGUS", rep_num, min_ratio)
                    rep_feedback.append("Knees caved inward; drive knees outward over toes.")
                if avg_diff is not None and avg_diff > 22.0:
                    if "UNEVEN_SQUAT" not in rep_issues:
                        rep_issues.append("UNEVEN_SQUAT")
                    _add_issue_occurrence("UNEVEN_SQUAT", rep_num, avg_diff)
                    rep_feedback.append("Uneven left/right knee depth; balance weight across both legs.")

            total_w = sum(w for _, w in score_parts)
            rep_form_score = round(sum(sc * w for sc, w in score_parts) / total_w, 1)
            if not rep["isValid"]:
                rep_form_score = round(min(rep_form_score, 65.0), 1)

            if not rep_feedback:
                rep_feedback.append("Solid repetition with good depth and controlled alignment.")

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
                    "hipAngleDeg": min_hip,
                    "torsoInclinationDeg": peak_lean,
                    "kneeAnkleRatio": min_ratio,
                    "kneeSymmetryDiffDeg": avg_diff,
                },
                "issues": rep_issues,
                "feedback": rep_feedback,
            })

        valid_reps = sum(1 for r in reps_output if r["isValid"])
        incomplete_reps = sum(1 for r in reps_output if not r["isValid"])
        total_attempted = len(reps_output)

        # Build sequence-level movementMetrics (never fabricate unavailable ones)
        valid_geos = [g for g in per_frame_geo if g["is_pose_valid"]]
        leg_conf = float(np.mean([g["leg_confidence"] for g in valid_geos])) if valid_geos else 0.0

        movement_metrics: Dict[str, Dict[str, Any]] = {}

        # 1. squat_depth
        if reps_output:
            depth_vals = [float(r["minAngleDeg"]) for r in reps_output]
            d_val = float(np.mean(depth_vals))
            d_min, d_max, d_avg = min(depth_vals), max(depth_vals), d_val
        else:
            d_min = min(valid_knee_angles)
            d_max = max(valid_knee_angles)
            d_avg = float(np.mean(valid_knee_angles))
            d_val = d_min

        if d_val <= 100.0:
            depth_metric_score = min(100.0, max(85.0, 100.0 - max(0.0, d_val - 90.0) * 1.5))
            depth_status = "VALID"
        elif d_val <= 120.0:
            depth_metric_score = max(55.0, 85.0 - (d_val - 100.0) * 1.5)
            depth_status = "WARNING"
        else:
            depth_metric_score = max(0.0, 55.0 - (d_val - 120.0) * 1.1)
            depth_status = "ERROR"

        movement_metrics["squat_depth"] = make_movement_metric(
            name="Squat Depth (Min Knee Angle)",
            unit="deg",
            value=d_val,
            min_val=d_min,
            max_val=d_max,
            avg_val=d_avg,
            target_range="<= 100°",
            score=depth_metric_score,
            status=depth_status,
            confidence=leg_conf,
        )

        # 2. hip_angle (UNAVAILABLE if shoulder/hip/knee unreliable)
        all_hips = [float(g["hip_angle"]) for g in valid_geos if g["hip_angle"] is not None]
        if all_hips:
            hip_confs = [float(g["hip_angle_confidence"]) for g in valid_geos if g["hip_angle"] is not None]
            h_conf = float(np.mean(hip_confs))
            h_vals = rep_min_hips if rep_min_hips else all_hips
            h_val = float(np.mean(h_vals)) if rep_min_hips else min(all_hips)
            if 45.0 <= h_val <= 115.0:
                h_score, h_status = 100.0, "VALID"
            elif h_val < 45.0:
                h_score = max(30.0, 100.0 - (45.0 - h_val) * 2.5)
                h_status = "ERROR" if h_val < 30.0 else "WARNING"
            else:
                h_score = max(35.0, 100.0 - (h_val - 115.0) * 1.5)
                h_status = "ERROR" if h_val > 135.0 else "WARNING"
            movement_metrics["hip_angle"] = make_movement_metric(
                name="Hip Flexion Angle",
                unit="deg",
                value=h_val,
                min_val=min(all_hips),
                max_val=max(all_hips),
                avg_val=float(np.mean(all_hips)),
                target_range="45° - 115°",
                score=h_score,
                status=h_status,
                confidence=h_conf,
            )
        else:
            movement_metrics["hip_angle"] = make_unavailable_metric(
                name="Hip Flexion Angle", unit="deg", target_range="45° - 115°"
            )

        # 3. torso_inclination (UNAVAILABLE if shoulder/hip unreliable)
        all_leans = [float(g["torso_angle"]) for g in valid_geos if g["torso_angle"] is not None]
        if all_leans:
            lean_confs = [float(g["torso_confidence"]) for g in valid_geos if g["torso_angle"] is not None]
            t_conf = float(np.mean(lean_confs))
            t_val = float(np.mean(rep_peak_leans)) if rep_peak_leans else max(all_leans)
            if t_val <= 35.0:
                t_score, t_status = 100.0, "VALID"
            elif t_val <= 40.0:
                t_score, t_status = 90.0 - (t_val - 35.0) * 2.0, "VALID"
            elif t_val <= 52.0:
                t_score, t_status = 80.0 - (t_val - 40.0) * 2.5, "WARNING"
            else:
                t_score, t_status = max(10.0, 50.0 - (t_val - 52.0) * 2.0), "ERROR"

            if not reps_output and t_val > 42.0:
                _add_issue_occurrence("EXCESSIVE_FORWARD_LEAN", None, t_val)

            movement_metrics["torso_inclination"] = make_movement_metric(
                name="Torso Inclination",
                unit="deg",
                value=t_val,
                min_val=min(all_leans),
                max_val=max(all_leans),
                avg_val=float(np.mean(all_leans)),
                target_range="<= 40°",
                score=t_score,
                status=t_status,
                confidence=t_conf,
            )
        else:
            movement_metrics["torso_inclination"] = make_unavailable_metric(
                name="Torso Inclination", unit="deg", target_range="<= 40°"
            )

        # 4. knee_alignment (UNAVAILABLE in strict side-view or when bilateral knees/ankles unreliable)
        active_valid_geos = [g for g in valid_geos if g["primary_knee_angle"] is not None and g["primary_knee_angle"] < 150.0] or valid_geos
        all_ratios = [float(g["knee_ankle_ratio"]) for g in active_valid_geos if g["knee_ankle_ratio"] is not None]
        all_diffs = [float(g["knee_symmetry_diff"]) for g in active_valid_geos if g["knee_symmetry_diff"] is not None]

        if not side_view and all_ratios:
            align_confs = [float(g["alignment_confidence"]) for g in active_valid_geos if g["knee_ankle_ratio"] is not None]
            ka_conf = float(np.mean(align_confs))
            ka_val = float(np.mean(rep_min_ratios)) if rep_min_ratios else min(all_ratios)
            avg_diff_seq = float(np.mean(rep_sym_diffs)) if rep_sym_diffs else (float(np.mean(all_diffs)) if all_diffs else 0.0)

            valgus_score = 100.0 if ka_val >= 0.85 else max(0.0, 100.0 - (0.85 - ka_val) * 200.0)
            sym_score = 100.0 if avg_diff_seq <= 10.0 else max(0.0, 100.0 - (avg_diff_seq - 10.0) * 2.5)
            ka_score = round(0.70 * valgus_score + 0.30 * sym_score, 1)

            if ka_val < 0.72 or avg_diff_seq > 25.0:
                ka_status = "ERROR"
            elif ka_val < 0.85 or avg_diff_seq > 18.0:
                ka_status = "WARNING"
            else:
                ka_status = "VALID"

            if not reps_output and ka_val < 0.82:
                _add_issue_occurrence("KNEE_VALGUS", None, ka_val)
            if not reps_output and avg_diff_seq > 22.0:
                _add_issue_occurrence("UNEVEN_SQUAT", None, avg_diff_seq)

            movement_metrics["knee_alignment"] = make_movement_metric(
                name="Knee Alignment (Valgus / Symmetry)",
                unit="ratio",
                value=ka_val,
                min_val=min(all_ratios),
                max_val=max(all_ratios),
                avg_val=float(np.mean(all_ratios)),
                target_range=">= 0.85",
                score=ka_score,
                status=ka_status,
                confidence=ka_conf,
                decimals=2,
            )
        else:
            movement_metrics["knee_alignment"] = make_unavailable_metric(
                name="Knee Alignment (Valgus / Symmetry)", unit="ratio", target_range=">= 0.85"
            )

        # 5. movement_consistency (UNAVAILABLE if < 2 reps)
        if len(reps_output) >= 2:
            depths = [float(r["minAngleDeg"]) for r in reps_output]
            depth_std = float(np.std(depths))
            depth_consistency = max(0.0, 100.0 - depth_std * 3.0)

            durations = [float(r["durationMs"]) for r in reps_output if r["durationMs"] > 0]
            if len(durations) >= 2 and np.mean(durations) > 0:
                cv_dur = float(np.std(durations) / np.mean(durations))
                dur_consistency = max(0.0, 100.0 - cv_dur * 100.0)
            else:
                dur_consistency = depth_consistency

            valid_ratio = valid_reps / len(reps_output)
            cons_val = round((0.60 * depth_consistency + 0.40 * dur_consistency) * (0.70 + 0.30 * valid_ratio), 1)
            cons_status = "VALID" if cons_val >= 75.0 else ("WARNING" if cons_val >= 55.0 else "ERROR")
            movement_metrics["movement_consistency"] = make_movement_metric(
                name="Movement Consistency",
                unit="score",
                value=cons_val,
                min_val=min(depth_consistency, dur_consistency),
                max_val=max(depth_consistency, dur_consistency),
                avg_val=cons_val,
                target_range=">= 80",
                score=cons_val,
                status=cons_status,
                confidence=leg_conf,
            )
        else:
            movement_metrics["movement_consistency"] = make_unavailable_metric(
                name="Movement Consistency", unit="score", target_range=">= 80"
            )

        # Build issues list
        issues_list: List[Dict[str, Any]] = []
        issue_catalog = {
            "SHALLOW_DEPTH": {
                "metric": "squat_depth",
                "title": "Shallow Squat Depth",
                "message": "Squat did not reach parallel thigh depth (knee angle <= 100°).",
                "recommendation": "Lower hips until thighs are at least parallel to the floor while keeping heels planted.",
                "targetRange": "<= 100°",
            },
            "INCOMPLETE_EXTENSION": {
                "metric": "squat_depth",
                "title": "Incomplete Top Lockout",
                "message": "Knees and hips did not extend fully back to standing position (>= 160°).",
                "recommendation": "Stand tall and squeeze your glutes to finish full extension at the top of each rep.",
                "targetRange": ">= 160°",
            },
            "EXCESSIVE_FORWARD_LEAN": {
                "metric": "torso_inclination",
                "title": "Excessive Forward Lean",
                "message": "Torso tilted too far forward during the descent/bottom of the squat.",
                "recommendation": "Brace your core, keep your chest upright, and work on ankle dorsiflexion mobility.",
                "targetRange": "<= 40°",
            },
            "KNEE_VALGUS": {
                "metric": "knee_alignment",
                "title": "Knee Valgus (Inward Collapse)",
                "message": "Knees collapsed inward relative to ankle width during the squat.",
                "recommendation": "Actively push your knees outward in line with your toes and strengthen hip abductors.",
                "targetRange": ">= 0.85",
            },
            "UNEVEN_SQUAT": {
                "metric": "knee_alignment",
                "title": "Asymmetric Squat Mechanics",
                "message": "Noticeable left-to-right asymmetry in knee flexion during the squat.",
                "recommendation": "Distribute load evenly across both feet and control descent speed.",
                "targetRange": "<= 15° diff",
            },
        }

        for code, data in issue_tracker.items():
            meta = issue_catalog[code]
            vals = data["values"]
            measured = float(np.mean(vals)) if vals else None
            if code == "SHALLOW_DEPTH":
                measured = max(vals) if vals else None
                sev = "HIGH" if (measured is not None and measured > 120.0) or incomplete_reps > 0 else "MEDIUM"
            elif code == "EXCESSIVE_FORWARD_LEAN":
                measured = max(vals) if vals else None
                sev = "HIGH" if (measured is not None and measured > 52.0) else "MEDIUM"
            elif code == "KNEE_VALGUS":
                measured = min(vals) if vals else None
                sev = "HIGH" if (measured is not None and measured < 0.72) else "MEDIUM"
            elif code == "INCOMPLETE_EXTENSION":
                sev = "MEDIUM"
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
                metric="squat_depth",
                severity="LOW",
                title="No Squat Repetitions Completed",
                message="Athlete pose was tracked, but no full or partial squat repetitions were detected.",
                recommendation="Perform controlled squat repetitions lowering until knee angle reaches <= 100° and returning to stand.",
                measured_value=d_min,
                target_range="<= 100°",
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
        if movement_metrics["torso_inclination"]["status"] != "UNAVAILABLE":
            align_parts.append(float(movement_metrics["torso_inclination"]["score"]))
        if movement_metrics["knee_alignment"]["status"] != "UNAVAILABLE":
            align_parts.append(float(movement_metrics["knee_alignment"]["score"]))
        alignment_score = round(float(np.mean(align_parts)), 1) if align_parts else None

        stab_parts: List[float] = []
        if movement_metrics["movement_consistency"]["status"] != "UNAVAILABLE":
            stab_parts.append(float(movement_metrics["movement_consistency"]["score"]))
        smooth_score = compute_angle_smoothness_score(valid_knee_angles)
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
                f"Excellent squat execution! Completed {valid_reps} valid rep(s) with strong depth and alignment."
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
