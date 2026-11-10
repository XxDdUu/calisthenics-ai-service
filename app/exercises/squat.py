import logging
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_vertical_angle,
    get_point,
)
from app.biomechanics.squat_counter import (
    SquatRepCounter,
    SquatState,
    extract_squat_geometry,
)
from app.exercises.base import BaseExerciseAnalyzer
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    LiveScoringMetric,
)

logger = logging.getLogger(__name__)


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

        # Also verify at least one shoulder or upper body landmark is visible for posture
        has_sh = bool(keypoints.get("left_shoulder") or keypoints.get("right_shoulder"))
        if not has_sh:
            return False, LiveAnalysisStatus.INCOMPLETE_POSE, ["shoulder"]

        return True, LiveAnalysisStatus.VALID, []

    def extract_measurements(
        self,
        keypoints: Dict[str, KeypointData],
        side_view: bool = False
    ) -> Dict[str, float]:
        geo = extract_squat_geometry(keypoints, side_view=side_view)
        primary_knee = geo["primary_knee_angle"] if geo["primary_knee_angle"] is not None else 180.0
        l_knee = geo["knee_left_angle"] if geo["knee_left_angle"] is not None else primary_knee
        r_knee = geo["knee_right_angle"] if geo["knee_right_angle"] is not None else primary_knee
        avg_knee = geo["avg_knee_angle"] if geo["avg_knee_angle"] is not None else primary_knee
        hip_y = geo["hip_y"] if geo["hip_y"] is not None else 0.0
        trunk_lean = geo["torso_angle"] if geo["torso_angle"] is not None else 15.0
        knee_ankle_ratio = geo["knee_ankle_ratio"] if geo["knee_ankle_ratio"] is not None else 1.0

        # Knee symmetry diff
        knee_diff = abs(l_knee - r_knee) if (geo["knee_left_angle"] is not None and geo["knee_right_angle"] is not None) else 0.0

        # Hip flexion angle (Shoulder - Hip - Knee)
        l_sh = get_point(keypoints.get("left_shoulder"))
        r_sh = get_point(keypoints.get("right_shoulder"))
        l_hip = get_point(keypoints.get("left_hip"))
        r_hip = get_point(keypoints.get("right_hip"))
        l_kn = get_point(keypoints.get("left_knee"))
        r_kn = get_point(keypoints.get("right_knee"))

        p_sh = l_sh or r_sh
        p_hip = l_hip or r_hip
        p_kn = l_kn or r_kn
        hip_angle = calculate_angle_2d(p_sh, p_hip, p_kn) if (p_sh and p_hip and p_kn) else 180.0

        return {
            "knee_angle": round(primary_knee, 1),
            "knee_left_angle": round(l_knee, 1),
            "knee_right_angle": round(r_knee, 1),
            "avg_knee_angle": round(avg_knee, 1),
            "hip_y": round(hip_y, 1),
            "knee_symmetry_diff": round(knee_diff, 1),
            "knee_ankle_ratio": round(knee_ankle_ratio, 2),
            "hip_angle": round(hip_angle, 1),
            "torso_angle": round(trunk_lean, 1),
            "trunk_lean_angle": round(trunk_lean, 1)
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        knee = measurements.get("knee_angle", 180.0)
        trunk_lean = measurements.get("trunk_lean_angle", measurements.get("torso_angle", 15.0))
        knee_diff = measurements.get("knee_symmetry_diff", 0.0)
        knee_ankle_ratio = measurements.get("knee_ankle_ratio", 1.0)

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

        # 2. Knee alignment & valgus
        if knee_ankle_ratio < 0.82 and not side_view:
            feedback.append(LiveFeedbackItem(
                code="KNEE_VALGUS",
                message="Push your knees out in line with your toes.",
                severity=FeedbackSeverity.WARNING
            ))
        elif knee_diff > 25.0:
            feedback.append(LiveFeedbackItem(
                code="UNEVEN_SQUAT",
                message="Distribute weight evenly between both legs.",
                severity=FeedbackSeverity.WARNING
            ))

        # 3. Torso inclination
        if trunk_lean > 42.0:
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

        # Form score
        if knee <= 100.0:
            depth_score = max(70.0, min(100.0, 100.0 - max(0.0, knee - 90.0) * 1.5))
        elif knee >= 155.0:
            depth_score = 95.0
        else:
            depth_score = max(40.0, 100.0 - (knee - 95.0) * 1.2)

        posture_score = max(0.0, min(100.0, 100.0 - max(0.0, trunk_lean - 25.0) * 2.5))
        alignment_score = max(0.0, min(100.0, 100.0 - knee_diff * 2.0 - max(0.0, 1.0 - knee_ankle_ratio) * 40.0))

        overall_score = round(0.40 * depth_score + 0.35 * posture_score + 0.25 * alignment_score, 1)

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
        knee_ankle_ratio = measurements.get("knee_ankle_ratio", 1.0)
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
        if knee_diff is not None:
            knee_score = max(0.0, min(100.0, 100.0 - knee_diff * 2.0 - max(0.0, 1.0 - knee_ankle_ratio) * 40.0))
            metrics.append(LiveScoringMetric(
                name="knee_alignment",
                value=round(knee_diff, 1),
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
            # At squat bottom or lockout, hip flexion tracks proportionally with knee flexion without excessive folding
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
            l_hip, r_hip = get_point(keypoints.get("left_hip")), get_point(keypoints.get("right_hip"))
            l_ak, r_ak = get_point(keypoints.get("left_ankle")), get_point(keypoints.get("right_ankle"))
            l_sh, r_sh = get_point(keypoints.get("left_shoulder")), get_point(keypoints.get("right_shoulder"))
            if (l_hip or r_hip) and (l_ak or r_ak):
                hip_x = ((l_hip[0] + r_hip[0]) / 2.0) if (l_hip and r_hip) else (l_hip or r_hip)[0]
                hip_y = ((l_hip[1] + r_hip[1]) / 2.0) if (l_hip and r_hip) else (l_hip or r_hip)[1]
                ak_x = ((l_ak[0] + r_ak[0]) / 2.0) if (l_ak and r_ak) else (l_ak or r_ak)[0]
                ak_y = ((l_ak[1] + r_ak[1]) / 2.0) if (l_ak and r_ak) else (l_ak or r_ak)[1]
                leg_span = max(30.0, abs(ak_y - hip_y))
                if (l_sh or r_sh):
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
        # Retrieve isolated session counter, or fallback if session object is mock/lacks it
        if hasattr(session, "get_squat_counter"):
            counter = session.get_squat_counter()
        elif hasattr(session, "squat_counter") and session.squat_counter is not None:
            counter = session.squat_counter
        else:
            counter = self._fallback_counter

        if not is_valid_frame:
            rep_res = counter.process_angle(None, timestamp_sec=timestamp_sec)
        else:
            knee = measurements.get("knee_angle", 180.0)
            rep_res = counter.process_angle(knee, timestamp_sec=timestamp_sec)

        if hasattr(session, "rep_count"):
            session.rep_count = rep_res.rep_count
        if hasattr(session, "rep_phase"):
            session.rep_phase = rep_res.current_state

        return None, rep_res.rep_count, rep_res.current_state
