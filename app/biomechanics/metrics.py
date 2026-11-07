import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.biomechanics.alignment import compute_alignment_metrics
from app.biomechanics.angles import calculate_leg_spread_ratio, compute_all_joint_angles
from app.biomechanics.confidence import check_pose_quality_for_planche
from app.biomechanics.side_view import (
    SideViewBiomechanicalFeatures,
    SideViewKeypointProcessor,
)
from app.core.logging_context import get_request_id
from app.schemas.analysis import AlignmentMetrics, JointAngles, KeypointData, PoseKeypoints

logger = logging.getLogger(__name__)


class BiomechanicsProfile(BaseModel):
    joint_angles: JointAngles
    alignment: AlignmentMetrics
    side_view: bool = False
    side_view_features: Optional[SideViewBiomechanicalFeatures] = None
    midpoints: Optional[Dict[str, KeypointData]] = None
    resolved_landmarks: Optional[Dict[str, KeypointData]] = None
    pose_quality_ok: bool = True
    missing_landmarks: List[str] = Field(default_factory=list)


class BiomechanicsEvaluator:
    """Evaluates raw pose keypoints into unified biomechanics metrics."""

    @staticmethod
    def evaluate(
        pose_kp: PoseKeypoints,
        side_view: bool = False
    ) -> Optional[BiomechanicsProfile]:
        if pose_kp is None or not pose_kp.keypoints:
            return None

        rid = get_request_id()
        _t_eval_start = time.perf_counter()
        keypoints = pose_kp.keypoints

        # 1. Pose Quality & Visibility Check
        quality_ok, status_str, missing_parts = check_pose_quality_for_planche(
            keypoints=keypoints,
            side_view=side_view
        )

        # 2. Branch explicitly on side_view mode
        if side_view:
            # --- SIDE-VIEW PROCESSING PIPELINE ---
            # Derives mid_shoulder, mid_hip, mid_knee, mid_ankle while preserving original landmarks
            processed_side = SideViewKeypointProcessor.process_keypoints(keypoints)
            side_features = SideViewKeypointProcessor.extract_features(processed_side)

            # Map side-view features to unified JointAngles schema
            primary_elbow = (
                side_features.elbow_extension_deg
                if side_features.elbow_extension_deg is not None
                else 180.0
            )
            primary_lean = (
                side_features.shoulder_lean_deg
                if side_features.shoulder_lean_deg is not None
                else 0.0
            )
            hip_angle = (
                side_features.hip_extension_deg
                if side_features.hip_extension_deg is not None
                else (side_features.body_line_deg if side_features.body_line_deg is not None else 180.0)
            )
            primary_knee = (
                side_features.knee_extension_deg
                if side_features.knee_extension_deg is not None
                else 180.0
            )
            body_tilt = (
                side_features.body_tilt_deg
                if side_features.body_tilt_deg is not None
                else 0.0
            )

            # Profile views do not provide reliable bilateral separation. A
            # resolved single leg is a joined-leg neutral value, not a spread.
            leg_spread = 0.0

            joint_angles = JointAngles(
                elbow_left_deg=side_features.left_elbow_deg or primary_elbow,
                elbow_right_deg=side_features.right_elbow_deg or primary_elbow,
                primary_elbow_deg=round(primary_elbow, 1),
                shoulder_lean_left_deg=round(primary_lean, 1),
                shoulder_lean_right_deg=round(primary_lean, 1),
                primary_shoulder_lean_deg=round(primary_lean, 1),
                hip_angle_deg=round(hip_angle, 1),
                knee_left_deg=side_features.left_knee_deg or primary_knee,
                knee_right_deg=side_features.right_knee_deg or primary_knee,
                primary_knee_deg=round(primary_knee, 1),
                body_horizontal_tilt_deg=round(body_tilt, 1),
                primary_side=side_features.primary_side,
                leg_spread_ratio=leg_spread
            )

            alignment = compute_alignment_metrics(
                keypoints=processed_side.keypoints,
                hip_angle_deg=joint_angles.hip_angle_deg,
                body_tilt_deg=joint_angles.body_horizontal_tilt_deg
            )

            profile = BiomechanicsProfile(
                joint_angles=joint_angles,
                alignment=alignment,
                side_view=True,
                side_view_features=side_features,
                midpoints=processed_side.midpoints,
                resolved_landmarks=processed_side.resolved_chain,
                pose_quality_ok=quality_ok and processed_side.is_valid_side_view,
                missing_landmarks=missing_parts + processed_side.missing_landmarks
            )
            t_eval_ms = (time.perf_counter() - _t_eval_start) * 1000.0
            logger.debug("[TIMING] request_id=%s stage=biomechanics_evaluation duration_ms=%.2f side_view=True", rid, t_eval_ms)
            return profile

        else:
            # --- STANDARD PROCESSING PIPELINE ---
            # ABSOLUTELY NO MIDPOINTS COMPUTED. Operates using existing individual landmarks.
            joint_angles = compute_all_joint_angles(keypoints)
            alignment = compute_alignment_metrics(
                keypoints=keypoints,
                hip_angle_deg=joint_angles.hip_angle_deg,
                body_tilt_deg=joint_angles.body_horizontal_tilt_deg
            )

            profile = BiomechanicsProfile(
                joint_angles=joint_angles,
                alignment=alignment,
                side_view=False,
                side_view_features=None,
                midpoints=None,
                pose_quality_ok=quality_ok,
                missing_landmarks=missing_parts
            )
            t_eval_ms = (time.perf_counter() - _t_eval_start) * 1000.0
            logger.debug("[TIMING] request_id=%s stage=biomechanics_evaluation duration_ms=%.2f side_view=False", rid, t_eval_ms)
            return profile
