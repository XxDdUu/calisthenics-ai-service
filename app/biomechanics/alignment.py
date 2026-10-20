import math
from typing import Dict, Optional, Tuple
import numpy as np

from app.schemas.analysis import AlignmentMetrics, KeypointData
from app.biomechanics.angles import get_point, calculate_horizontal_angle


def compute_center_of_mass_2d(keypoints: Dict[str, KeypointData]) -> Tuple[Optional[Tuple[float, float]], float]:
    """
    Estimate 2D Center of Mass (CoM) from 17 COCO keypoints using standard biomechanical mass distributions:
    - Head & Neck: 8% (Nose/Ear)
    - Torso & Hips: 46% (Shoulders & Hips midpoint)
    - Arms: 10% (Elbows & Wrists)
    - Legs: 36% (Knees & Ankles)
    
    Returns (CoM point (x, y), wrist_base_x)
    """
    segments = []

    # Head
    head_pts = [get_point(keypoints.get(k)) for k in ["nose", "left_ear", "right_ear"] if get_point(keypoints.get(k))]
    if head_pts:
        hx = sum(p[0] for p in head_pts) / len(head_pts)
        hy = sum(p[1] for p in head_pts) / len(head_pts)
        segments.append(((hx, hy), 0.08))

    # Torso
    sh_pts = [get_point(keypoints.get(k)) for k in ["left_shoulder", "right_shoulder"] if get_point(keypoints.get(k))]
    hip_pts = [get_point(keypoints.get(k)) for k in ["left_hip", "right_hip"] if get_point(keypoints.get(k))]
    
    if sh_pts and hip_pts:
        sh_mid = (sum(p[0] for p in sh_pts) / len(sh_pts), sum(p[1] for p in sh_pts) / len(sh_pts))
        hip_mid = (sum(p[0] for p in hip_pts) / len(hip_pts), sum(p[1] for p in hip_pts) / len(hip_pts))
        torso_mid = ((sh_mid[0] + hip_mid[0]) / 2.0, (sh_mid[1] + hip_mid[1]) / 2.0)
        segments.append((torso_mid, 0.46))

    # Arms
    arm_pts = [get_point(keypoints.get(k)) for k in ["left_elbow", "right_elbow", "left_wrist", "right_wrist"] if get_point(keypoints.get(k))]
    if arm_pts:
        ax = sum(p[0] for p in arm_pts) / len(arm_pts)
        ay = sum(p[1] for p in arm_pts) / len(arm_pts)
        segments.append(((ax, ay), 0.10))

    # Legs
    leg_pts = [get_point(keypoints.get(k)) for k in ["left_knee", "right_knee", "left_ankle", "right_ankle"] if get_point(keypoints.get(k))]
    if leg_pts:
        lx = sum(p[0] for p in leg_pts) / len(leg_pts)
        ly = sum(p[1] for p in leg_pts) / len(leg_pts)
        segments.append(((lx, ly), 0.36))

    if not segments:
        return None, 0.0

    total_weight = sum(w for _, w in segments)
    com_x = sum(pt[0] * w for pt, w in segments) / total_weight
    com_y = sum(pt[1] * w for pt, w in segments) / total_weight

    # Wrist base of support X center
    wrist_pts = [get_point(keypoints.get(k)) for k in ["left_wrist", "right_wrist"] if get_point(keypoints.get(k))]
    wrist_x = sum(p[0] for p in wrist_pts) / len(wrist_pts) if wrist_pts else com_x

    return (com_x, com_y), wrist_x


def compute_alignment_metrics(keypoints: Dict[str, KeypointData], hip_angle_deg: float, body_tilt_deg: float) -> AlignmentMetrics:
    """Computes body straightness, scapular protraction, horizontal level, and CoM balance metrics."""
    
    # 1. Body Straightness Score
    # Ideal straight planche line is 180 degrees.
    straightness_err = abs(180.0 - hip_angle_deg)
    body_straightness_score = max(0.0, min(100.0, 100.0 - (straightness_err * 2.2)))

    # 2. Horizontal Level Score
    # Ideal torso tilt is 0-10 degrees relative to ground horizontal.
    tilt_err = abs(body_tilt_deg)
    horizontal_level_score = max(0.0, min(100.0, 100.0 - (tilt_err * 3.5)))

    # 3. Scapular Protraction Score
    # Derived from shoulder position relative to ears and spine elevation.
    l_sh = get_point(keypoints.get("left_shoulder"))
    r_sh = get_point(keypoints.get("right_shoulder"))
    l_ear = get_point(keypoints.get("left_ear"))
    r_ear = get_point(keypoints.get("right_ear"))

    protraction_score = 80.0  # default baseline
    if (l_sh or r_sh) and (l_ear or r_ear):
        sh_y = (l_sh[1] if l_sh else r_sh[1]) if not (l_sh and r_sh) else (l_sh[1] + r_sh[1]) / 2.0
        ear_y = (l_ear[1] if l_ear else r_ear[1]) if not (l_ear and r_ear) else (l_ear[1] + r_ear[1]) / 2.0
        # If shoulders are depressed (below ears in image space, i.e. sh_y > ear_y), protraction is strong
        delta_y = sh_y - ear_y
        if delta_y > 10:
            protraction_score = min(100.0, 85.0 + (delta_y * 0.5))
        else:
            protraction_score = max(40.0, 75.0 - abs(delta_y) * 1.5)

    # 4. Center of Mass Balance
    com_pt, wrist_x = compute_center_of_mass_2d(keypoints)
    offset_px = 0.0
    com_over_hands = True

    if com_pt is not None:
        offset_px = abs(com_pt[0] - wrist_x)
        # Check if CoM is within reasonable pixel margin of support base
        com_over_hands = offset_px <= 45.0

    return AlignmentMetrics(
        body_straightness_score=round(body_straightness_score, 1),
        scapular_protraction_score=round(protraction_score, 1),
        horizontal_level_score=round(horizontal_level_score, 1),
        center_of_mass_over_hands=com_over_hands,
        center_of_mass_offset_px=round(offset_px, 1)
    )
