import math
from typing import Any, Dict, Optional, Tuple
import numpy as np

from app.schemas.analysis import JointAngles, KeypointData


def is_valid_coord(val: Any) -> bool:
    """Check if a numeric coordinate value is finite and valid (not NaN or Inf)."""
    if val is None:
        return False
    try:
        f = float(val)
        return math.isfinite(f)
    except (TypeError, ValueError):
        return False


def is_valid_point(pt: Optional[Tuple[Any, ...]]) -> bool:
    """Check if a point tuple has at least 2 finite coordinates."""
    if pt is None or len(pt) < 2:
        return False
    return is_valid_coord(pt[0]) and is_valid_coord(pt[1])


def calculate_angle_2d(
    p1: Optional[Tuple[float, float]],
    vertex: Optional[Tuple[float, float]],
    p2: Optional[Tuple[float, float]],
    default_angle: float = 180.0
) -> float:
    """
    Calculate 2D interior angle in degrees formed by three points (p1 - vertex - p2).
    Returns angle in range [0.0, 180.0].
    Safely rejects degenerate vectors, NaN, Infinity, and missing coordinates.
    """
    if not is_valid_point(p1) or not is_valid_point(vertex) or not is_valid_point(p2):
        return default_angle

    assert p1 is not None and vertex is not None and p2 is not None

    v1 = (float(p1[0]) - float(vertex[0]), float(p1[1]) - float(vertex[1]))
    v2 = (float(p2[0]) - float(vertex[0]), float(p2[1]) - float(vertex[1]))

    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])

    if mag1 < 1e-6 or mag2 < 1e-6 or not math.isfinite(mag1) or not math.isfinite(mag2):
        return default_angle

    dot = v1[0] * v2[0] + v1[1] * v2[1]
    if not math.isfinite(dot):
        return default_angle

    denom = mag1 * mag2
    if denom < 1e-12:
        return default_angle

    cos_angle = max(-1.0, min(1.0, dot / denom))
    angle_rad = math.acos(cos_angle)
    deg = math.degrees(angle_rad)
    return round(float(deg), 1)


def calculate_vertical_angle(
    top_pt: Optional[Tuple[float, float]],
    bottom_pt: Optional[Tuple[float, float]],
    default_angle: float = 90.0
) -> float:
    """
    Calculate deviation angle in degrees relative to vertical plumline.
    0 degrees means perfectly vertical.
    Positive value indicates tilt.
    """
    if not is_valid_point(top_pt) or not is_valid_point(bottom_pt):
        return default_angle

    assert top_pt is not None and bottom_pt is not None
    dx = float(top_pt[0]) - float(bottom_pt[0])
    dy = abs(float(top_pt[1]) - float(bottom_pt[1]))
    if dy < 1e-6 or not math.isfinite(dy):
        return default_angle
    angle_rad = math.atan2(abs(dx), dy)
    deg = math.degrees(angle_rad)
    return round(float(deg), 1)


def calculate_horizontal_angle(
    p1: Optional[Tuple[float, float]],
    p2: Optional[Tuple[float, float]],
    default_angle: float = 90.0
) -> float:
    """
    Calculate angle in degrees relative to ground horizontal axis.
    0 degrees means perfectly horizontal.
    """
    if not is_valid_point(p1) or not is_valid_point(p2):
        return default_angle

    assert p1 is not None and p2 is not None
    dx = abs(float(p2[0]) - float(p1[0]))
    dy = abs(float(p2[1]) - float(p1[1]))
    if dx < 1e-6 or not math.isfinite(dx):
        return default_angle
    angle_rad = math.atan2(dy, dx)
    deg = math.degrees(angle_rad)
    return round(float(deg), 1)


def calculate_distance_2d(
    p1: Optional[Tuple[float, float]],
    p2: Optional[Tuple[float, float]]
) -> float:
    """Calculate Euclidean distance between two points, handling invalid values safely."""
    if not is_valid_point(p1) or not is_valid_point(p2):
        return 0.0
    assert p1 is not None and p2 is not None
    dx = float(p1[0]) - float(p2[0])
    dy = float(p1[1]) - float(p2[1])
    dist = math.hypot(dx, dy)
    return float(dist) if math.isfinite(dist) else 0.0


def calculate_body_line_deviation(
    shoulder: Optional[Tuple[float, float]],
    hip: Optional[Tuple[float, float]],
    ankle_or_knee: Optional[Tuple[float, float]]
) -> Tuple[float, str]:
    """
    Calculate the body line angle at the hips and classify alignment:
    - angle: 3-point angle (shoulder - hip - leg_endpoint) [0, 180]
    - alignment:
      - 'STRAIGHT' if angle >= 170.0
      - 'PIKE' if hips are flexed upward relative to the shoulder-leg axis
      - 'SAG' if hips are dropped downward relative to the shoulder-leg axis
    """
    if not is_valid_point(shoulder) or not is_valid_point(hip) or not is_valid_point(ankle_or_knee):
        return 180.0, "UNKNOWN"

    assert shoulder is not None and hip is not None and ankle_or_knee is not None

    angle = calculate_angle_2d(shoulder, hip, ankle_or_knee)

    if angle >= 170.0:
        return angle, "STRAIGHT"

    # Cross product (2D determinant) of vector (shoulder -> leg) and vector (shoulder -> hip)
    # in image space (y grows downward):
    # dx_line = leg.x - shoulder.x
    # dy_line = leg.y - shoulder.y
    # hip deviation = (hip.x - shoulder.x) * dy_line - (hip.y - shoulder.y) * dx_line
    dx = ankle_or_knee[0] - shoulder[0]
    dy = ankle_or_knee[1] - shoulder[1]
    
    # Midpoint of shoulder and leg
    mid_x = (shoulder[0] + ankle_or_knee[0]) / 2.0
    mid_y = (shoulder[1] + ankle_or_knee[1]) / 2.0
    
    # In horizontal holds (Planche/Front Lever), if hip.y > mid_y, hips are drooping down (SAG)
    # If hip.y < mid_y, hips are elevated above body line (PIKE)
    if hip[1] > mid_y + 10.0:
        return angle, "SAG"
    elif hip[1] < mid_y - 10.0:
        return angle, "PIKE"
    else:
        # Cross product sign check for arbitrary orientations
        cross = (hip[0] - shoulder[0]) * dy - (hip[1] - shoulder[1]) * dx
        return angle, "PIKE" if cross > 0 else "SAG"


def get_point(
    kp: Optional[KeypointData],
    min_confidence: Optional[float] = None,
) -> Optional[Tuple[float, float]]:
    """Helper to extract (x, y) tuple if keypoint is visible, confident, and coordinates are finite."""
    if kp is None or not kp.visible:
        return None
    if not (is_valid_coord(kp.x) and is_valid_coord(kp.y) and is_valid_coord(kp.confidence)):
        return None
    if min_confidence is not None and float(kp.confidence) < float(min_confidence):
        return None
    return (float(kp.x), float(kp.y))


def compute_midpoint_coords(
    p1: Optional[Tuple[float, float]],
    p2: Optional[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """Compute 2D midpoint between two valid points, or return the single valid point if one is missing."""
    p1_ok = is_valid_point(p1)
    p2_ok = is_valid_point(p2)
    if p1_ok and p2_ok:
        assert p1 is not None and p2 is not None
        return ((float(p1[0]) + float(p2[0])) / 2.0, (float(p1[1]) + float(p2[1])) / 2.0)
    if p1_ok:
        assert p1 is not None
        return (float(p1[0]), float(p1[1]))
    if p2_ok:
        assert p2 is not None
        return (float(p2[0]), float(p2[1]))
    return None


def calculate_joint_angle_from_keypoints(
    keypoints: Dict[str, KeypointData],
    name_a: str,
    name_vertex: str,
    name_b: str,
    min_confidence: Optional[float] = None,
) -> Optional[float]:
    """
    Calculate 2D interior joint angle from named keypoints.
    Returns None when any required landmark is missing, below confidence threshold, or degenerate.
    """
    if not keypoints:
        return None
    pt_a = get_point(keypoints.get(name_a), min_confidence=min_confidence)
    pt_v = get_point(keypoints.get(name_vertex), min_confidence=min_confidence)
    pt_b = get_point(keypoints.get(name_b), min_confidence=min_confidence)
    if pt_a is None or pt_v is None or pt_b is None:
        return None

    mag1 = math.hypot(pt_a[0] - pt_v[0], pt_a[1] - pt_v[1])
    mag2 = math.hypot(pt_b[0] - pt_v[0], pt_b[1] - pt_v[1])
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None
    return calculate_angle_2d(pt_a, pt_v, pt_b)


def compute_bilateral_symmetry_diff(
    left_angle: Optional[float],
    right_angle: Optional[float],
) -> Optional[float]:
    """Compute absolute bilateral angle difference in degrees, or None if either side is unavailable."""
    if not is_valid_coord(left_angle) or not is_valid_coord(right_angle):
        return None
    return round(abs(float(left_angle) - float(right_angle)), 1)


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
