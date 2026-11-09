import logging
from typing import Any, Dict, List, Optional, Tuple
from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_vertical_angle,
    get_point,
)
from app.biomechanics.phase_detector import ExercisePhaseDetector
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
    - Squat depth (thigh parallel / hip crease below knee: knee angle <= 90°-95°)
    - Knee alignment & tracking
    - Hip mechanics & flexion
    - Trunk uprightness & spinal positioning
    - Repetition counting and phase detection
    """

    def __init__(self):
        self.phase_detector = ExercisePhaseDetector(
            exercise_name="SQUAT",
            is_dynamic=True,
            min_rom_delta=45.0,
            min_rep_duration_sec=0.8
        )

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
            "squat_depth": 0.25,
            "knee_alignment": 0.20,
            "hip_mechanics": 0.20,
            "trunk_positioning": 0.20,
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

        has_hip = bool(keypoints.get("left_hip") or keypoints.get("right_hip"))
        has_knee = bool(keypoints.get("left_knee") or keypoints.get("right_knee"))
        has_ankle = bool(keypoints.get("left_ankle") or keypoints.get("right_ankle"))
        has_sh = bool(keypoints.get("left_shoulder") or keypoints.get("right_shoulder"))

        missing = []
        if not has_sh: missing.append("shoulder")
        if not has_hip: missing.append("hip")
        if not has_knee: missing.append("knee")
        if not has_ankle: missing.append("ankle")

        if missing:
            return False, LiveAnalysisStatus.INCOMPLETE_POSE, missing

        return True, LiveAnalysisStatus.VALID, []

    def extract_measurements(
        self,
        keypoints: Dict[str, KeypointData],
        side_view: bool = False
    ) -> Dict[str, float]:
        l_sh = get_point(keypoints.get("left_shoulder"))
        r_sh = get_point(keypoints.get("right_shoulder"))
        l_hip = get_point(keypoints.get("left_hip"))
        r_hip = get_point(keypoints.get("right_hip"))
        l_kn = get_point(keypoints.get("left_knee"))
        r_kn = get_point(keypoints.get("right_knee"))
        l_ak = get_point(keypoints.get("left_ankle"))
        r_ak = get_point(keypoints.get("right_ankle"))

        # Determine dominant visible side
        l_conf = sum([keypoints[k].confidence for k in ["left_hip", "left_knee", "left_ankle"] if k in keypoints])
        r_conf = sum([keypoints[k].confidence for k in ["right_hip", "right_knee", "right_ankle"] if k in keypoints])
        side = "left" if l_conf >= r_conf else "right"

        p_sh = l_sh if side == "left" and l_sh else r_sh or l_sh
        p_hip = l_hip if side == "left" and l_hip else r_hip or l_hip
        p_kn = l_kn if side == "left" and l_kn else r_kn or l_kn
        p_ak = l_ak if side == "left" and l_ak else r_ak or l_ak

        # 1. Knee flexion angle (Hip - Knee - Ankle)
        l_knee = calculate_angle_2d(l_hip, l_kn, l_ak) if (l_hip and l_kn and l_ak) else 180.0
        r_knee = calculate_angle_2d(r_hip, r_kn, r_ak) if (r_hip and r_kn and r_ak) else 180.0
        primary_knee = l_knee if side == "left" else r_knee
        knee_diff = abs(l_knee - r_knee)

        # 2. Hip flexion angle (Shoulder - Hip - Knee)
        hip_angle = calculate_angle_2d(p_sh, p_hip, p_kn) if (p_sh and p_hip and p_kn) else 180.0

        # 3. Trunk inclination angle from vertical (Shoulder to Hip)
        trunk_lean = calculate_vertical_angle(p_sh, p_hip) if (p_sh and p_hip) else 15.0

        return {
            "knee_angle": round(primary_knee, 1),
            "knee_left_angle": round(l_knee, 1),
            "knee_right_angle": round(r_knee, 1),
            "knee_symmetry_diff": round(knee_diff, 1),
            "hip_angle": round(hip_angle, 1),
            "trunk_lean_angle": round(trunk_lean, 1)
        }

    def evaluate_live_form(
        self,
        measurements: Dict[str, float],
        progression: Optional[str] = None,
        side_view: bool = False
    ) -> Tuple[Optional[float], List[LiveFeedbackItem], bool]:
        knee = measurements.get("knee_angle", 180.0)
        hip = measurements.get("hip_angle", 180.0)
        trunk_lean = measurements.get("trunk_lean_angle", 15.0)
        knee_diff = measurements.get("knee_symmetry_diff", 0.0)

        feedback: List[LiveFeedbackItem] = []
        is_valid = True

        # Check Depth
        if self.phase_detector.current_phase == "BOTTOM":
            if knee > 105.0:
                feedback.append(LiveFeedbackItem(
                    code="INCREASE_DEPTH",
                    message="Increase your squat depth. Aim for thighs parallel to ground.",
                    severity=FeedbackSeverity.WARNING
                ))
            elif knee <= 90.0:
                feedback.append(LiveFeedbackItem(
                    code="EXCELLENT_DEPTH",
                    message="Great squat depth!",
                    severity=FeedbackSeverity.INFO
                ))

        # Check Trunk posture
        if trunk_lean > 45.0:
            feedback.append(LiveFeedbackItem(
                code="EXCESSIVE_FORWARD_LEAN",
                message="Keep your torso more upright and chest proud.",
                severity=FeedbackSeverity.WARNING
            ))
            is_valid = False

        # Check Lockout at top
        if self.phase_detector.current_phase in ("TOP", "START"):
            if knee < 160.0 or hip < 160.0:
                feedback.append(LiveFeedbackItem(
                    code="INCOMPLETE_EXTENSION",
                    message="Stand tall and extend hips fully at the top.",
                    severity=FeedbackSeverity.INFO
                ))

        # Check symmetry
        if knee_diff > 25.0:
            feedback.append(LiveFeedbackItem(
                code="UNEVEN_SQUAT",
                message="Maintain consistent knee alignment and weight distribution.",
                severity=FeedbackSeverity.WARNING
            ))

        # Form score
        depth_score = 90.0
        if self.phase_detector.current_phase == "BOTTOM":
            depth_score = max(30.0, min(100.0, 100.0 - max(0.0, knee - 90.0) * 2.0))

        posture_score = max(0.0, min(100.0, 100.0 - max(0.0, trunk_lean - 30.0) * 2.5))
        symmetry_score = max(0.0, min(100.0, 100.0 - knee_diff * 2.0))

        overall_score = round(0.40 * depth_score + 0.35 * posture_score + 0.25 * symmetry_score, 1)

        return overall_score, feedback, is_valid

    def evaluate_scoring_metrics(
        self,
        measurements: Dict[str, float],
        keypoints: Optional[Dict[str, KeypointData]] = None,
        side_view: bool = False
    ) -> List[LiveScoringMetric]:
        knee = measurements.get("knee_angle", 180.0)
        hip = measurements.get("hip_angle", 180.0)
        trunk_lean = measurements.get("trunk_lean_angle", 15.0)
        knee_diff = measurements.get("knee_symmetry_diff", 0.0)

        # 1. Depth
        if knee <= 100.0 or self.phase_detector.current_phase == "BOTTOM":
            depth_score = max(20.0, min(100.0, 100.0 - max(0.0, knee - 90.0) * 2.0))
        elif knee >= 155.0 or self.phase_detector.current_phase in ("TOP", "START"):
            depth_score = max(50.0, min(100.0, 100.0 - max(0.0, 165.0 - knee) * 2.0))
        else:
            depth_score = 85.0

        # 2. Knee alignment & symmetry
        knee_score = max(0.0, min(100.0, 100.0 - knee_diff * 2.5))

        # 3. Hip mechanics
        hip_score = max(0.0, min(100.0, 100.0 - max(0.0, hip - 120.0) * 0.5))

        # 4. Trunk positioning
        trunk_score = max(0.0, min(100.0, 100.0 - max(0.0, trunk_lean - 30.0) * 2.5))

        # 5. Movement control
        control_score = 88.0

        return [
            LiveScoringMetric(
                name="squat_depth",
                value=knee,
                score=round(depth_score, 1),
                weight=0.25,
                confidence=0.95,
                status="VALID"
            ),
            LiveScoringMetric(
                name="knee_alignment",
                value=knee_diff,
                score=round(knee_score, 1),
                weight=0.20,
                confidence=0.90,
                status="VALID"
            ),
            LiveScoringMetric(
                name="hip_mechanics",
                value=hip,
                score=round(hip_score, 1),
                weight=0.20,
                confidence=0.92,
                status="VALID"
            ),
            LiveScoringMetric(
                name="trunk_positioning",
                value=trunk_lean,
                score=round(trunk_score, 1),
                weight=0.20,
                confidence=0.93,
                status="VALID"
            ),
            LiveScoringMetric(
                name="movement_control",
                value=0.0,
                score=round(control_score, 1),
                weight=0.15,
                confidence=0.88,
                status="VALID"
            )
        ]

    def update_dynamic_state(
        self,
        session: Any,
        is_valid_frame: bool,
        measurements: Dict[str, float],
        timestamp_sec: float
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        knee = measurements.get("knee_angle", 180.0)
        reps, phase = self.phase_detector.update_dynamic_rep(
            primary_metric_value=knee,
            top_threshold=155.0,
            bottom_threshold=95.0,
            timestamp_sec=timestamp_sec,
            is_valid_form=is_valid_frame
        )
        session.rep_count = reps
        session.rep_phase = phase
        return None, reps, phase
