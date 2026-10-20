import math
from typing import Dict, Tuple, Optional
import numpy as np

from app.schemas.analysis import KeypointData, JointAngles


def calculate_angle_2d(
    p1: Tuple[float, float],
    vertex: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:
    """
    Calculate 2D interior angle in degrees formed by three points (p1 - vertex - p2).
    Returns angle in range [0.0, 180.0].
    """
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])

    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 < 1e-6 or mag2 < 1e-6:
        return 180.0

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)


def calculate_vertical_angle(
    top_pt: Tuple[float, float],
    bottom_pt: Tuple[float, float]
) -> float:
    """
    Calculate deviation angle in degrees relative to vertical plumline.
    0 degrees means perfectly vertical.
    Positive value indicates tilt.
    """
    dx = top_pt[0] - bottom_pt[0]
    dy = abs(top_pt[1] - bottom_pt[1])
    if dy < 1e-6:
        return 90.0
    angle_rad = math.atan2(abs(dx), dy)
    return math.degrees(angle_rad)


def calculate_horizontal_angle(
    p1: Tuple[float, float],
    p2: Tuple[float, float]
) -> float:
    """
    Calculate angle in degrees relative to ground horizontal axis.
    0 degrees means perfectly horizontal.
    """
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dx < 1e-6:
        return 90.0
    angle_rad = math.atan2(dy, dx)
    return math.degrees(angle_rad)


def get_point(kp: Optional[KeypointData]) -> Optional[Tuple[float, float]]:
    """Helper to extract (x, y) tuple if keypoint is visible."""
    if kp and kp.visible:
        return (kp.x, kp.y)
    return None


def calculate_leg_spread_ratio(keypoints: Dict[str, KeypointData]) -> Optional[float]:
    """
    Calculate normalized leg spread ratio:
    Ratio of 2D distance between feet/ankles (or knees) to 2D hip width.

    Leg spread ratio < ~1.25 indicates legs close together (Full Planche).
    Leg spread ratio >= ~1.25 indicates legs spread apart (Straddle Planche).
    Returns None if keypoints are insufficient.
    """
    l_hip = get_point(keypoints.get("left_hip"))
    r_hip = get_point(keypoints.get("right_hip"))

    if not l_hip or not r_hip:
        return None

    hip_width = math.hypot(l_hip[0] - r_hip[0], l_hip[1] - r_hip[1])

    # If hips overlap in 2D (< 10px), e.g. in pure profile view, use shoulder width scale
    if hip_width < 10.0:
        l_sh = get_point(keypoints.get("left_shoulder"))
        r_sh = get_point(keypoints.get("right_shoulder"))
        if l_sh and r_sh:
            sh_width = math.hypot(l_sh[0] - r_sh[0], l_sh[1] - r_sh[1])
            hip_width = max(sh_width * 0.75, 10.0)
        else:
            hip_width = 30.0

    l_ankle = get_point(keypoints.get("left_ankle"))
    r_ankle = get_point(keypoints.get("right_ankle"))

    if l_ankle and r_ankle:
        ankle_dist = math.hypot(l_ankle[0] - r_ankle[0], l_ankle[1] - r_ankle[1])
        return round(ankle_dist / hip_width, 2)

    # Fallback to knee separation if ankles are missing / occluded
    l_knee = get_point(keypoints.get("left_knee"))
    r_knee = get_point(keypoints.get("right_knee"))
    if l_knee and r_knee:
        knee_dist = math.hypot(l_knee[0] - r_knee[0], l_knee[1] - r_knee[1])
        return round((knee_dist * 1.5) / hip_width, 2)

    return None


def compute_all_joint_angles(keypoints: Dict[str, KeypointData]) -> JointAngles:
    """
    Computes all relevant joint angles for Planche biomechanics from keypoint dictionary.
    Handles side detection and occlusion gracefully.
    """
    l_shoulder = get_point(keypoints.get("left_shoulder"))
    r_shoulder = get_point(keypoints.get("right_shoulder"))
    l_elbow = get_point(keypoints.get("left_elbow"))
    r_elbow = get_point(keypoints.get("right_elbow"))
    l_wrist = get_point(keypoints.get("left_wrist"))
    r_wrist = get_point(keypoints.get("right_wrist"))
    l_hip = get_point(keypoints.get("left_hip"))
    r_hip = get_point(keypoints.get("right_hip"))
    l_knee = get_point(keypoints.get("left_knee"))
    r_knee = get_point(keypoints.get("right_knee"))
    l_ankle = get_point(keypoints.get("left_ankle"))
    r_ankle = get_point(keypoints.get("right_ankle"))

    # Determine primary side based on confidence sums
    left_conf_sum = sum([
        keypoints[k].confidence for k in ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee"]
        if k in keypoints
    ])
    right_conf_sum = sum([
        keypoints[k].confidence for k in ["right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee"]
        if k in keypoints
    ])
    primary_side = "left" if left_conf_sum >= right_conf_sum else "right"

    # 1. Elbow Angles (Shoulder - Elbow - Wrist)
    elbow_left = calculate_angle_2d(l_shoulder, l_elbow, l_wrist) if (l_shoulder and l_elbow and l_wrist) else 180.0
    elbow_right = calculate_angle_2d(r_shoulder, r_elbow, r_wrist) if (r_shoulder and r_elbow and r_wrist) else 180.0
    primary_elbow = elbow_left if primary_side == "left" else elbow_right

    # 2. Shoulder Lean Angles (Shoulder forward lean over Wrist base)
    # Measured by vertical angle from Shoulder to Wrist
    shoulder_lean_left = calculate_vertical_angle(l_shoulder, l_wrist) if (l_shoulder and l_wrist) else 0.0
    shoulder_lean_right = calculate_vertical_angle(r_shoulder, r_wrist) if (r_shoulder and r_wrist) else 0.0
    primary_shoulder_lean = shoulder_lean_left if primary_side == "left" else shoulder_lean_right

    # 3. Hip Angle (Shoulder - Hip - Knee or Ankle)
    p_shoulder = l_shoulder if primary_side == "left" and l_shoulder else r_shoulder or l_shoulder
    p_hip = l_hip if primary_side == "left" and l_hip else r_hip or l_hip
    p_knee = l_knee if primary_side == "left" and l_knee else r_knee or l_knee
    p_ankle = l_ankle if primary_side == "left" and l_ankle else r_ankle or l_ankle

    # Use ankle if visible, otherwise knee for body line extension
    p_leg_endpoint = p_knee if p_knee else p_ankle
    if p_shoulder and p_hip and p_leg_endpoint:
        hip_angle = calculate_angle_2d(p_shoulder, p_hip, p_leg_endpoint)
    else:
        hip_angle = 180.0

    # 4. Knee Angles (Hip - Knee - Ankle)
    knee_left = calculate_angle_2d(l_hip, l_knee, l_ankle) if (l_hip and l_knee and l_ankle) else 180.0
    knee_right = calculate_angle_2d(r_hip, r_knee, r_ankle) if (r_hip and r_knee and r_ankle) else 180.0
    primary_knee = knee_left if primary_side == "left" else knee_right

    # 5. Body Horizontal Tilt (Shoulder to Hip horizontal angle relative to ground)
    if p_shoulder and p_hip:
        body_tilt = calculate_horizontal_angle(p_shoulder, p_hip)
    else:
        body_tilt = 0.0

    # 6. Normalized Leg Spread Ratio
    leg_spread_ratio = calculate_leg_spread_ratio(keypoints)

    return JointAngles(
        elbow_left_deg=round(elbow_left, 1),
        elbow_right_deg=round(elbow_right, 1),
        primary_elbow_deg=round(primary_elbow, 1),
        shoulder_lean_left_deg=round(shoulder_lean_left, 1),
        shoulder_lean_right_deg=round(shoulder_lean_right, 1),
        primary_shoulder_lean_deg=round(primary_shoulder_lean, 1),
        hip_angle_deg=round(hip_angle, 1),
        knee_left_deg=round(knee_left, 1),
        knee_right_deg=round(knee_right, 1),
        primary_knee_deg=round(primary_knee, 1),
        body_horizontal_tilt_deg=round(body_tilt, 1),
        primary_side=primary_side,
        leg_spread_ratio=leg_spread_ratio
    )
