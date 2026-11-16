import math
from typing import Any, Dict, List, Optional, Tuple
from app.core.config import settings
from app.schemas.analysis import KeypointData

BODY_LANDMARK_GROUPS: Tuple[Tuple[str, str, str], ...] = (
    ("shoulders", "left_shoulder", "right_shoulder"),
    ("elbows", "left_elbow", "right_elbow"),
    ("wrists", "left_wrist", "right_wrist"),
    ("hips", "left_hip", "right_hip"),
    ("knees", "left_knee", "right_knee"),
    ("ankles", "left_ankle", "right_ankle"),
)


def is_valid_keypoint(
    kp: Optional[KeypointData],
    threshold: Optional[float] = None
) -> bool:
    """
    Check if a keypoint exists, is visible, has finite coordinates, and meets the confidence threshold.
    """
    if kp is None:
        return False
    if not (math.isfinite(kp.x) and math.isfinite(kp.y) and math.isfinite(kp.confidence)):
        return False
    conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD
    return bool(kp.visible and kp.confidence >= conf_thresh)


def filter_confident_keypoints(
    keypoints: Optional[Dict[str, KeypointData]],
    threshold: Optional[float] = None
) -> Dict[str, KeypointData]:
    """Return a filtered dictionary containing only keypoints that pass confidence and finiteness checks."""
    if not keypoints:
        return {}
    conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD
    return {
        name: kp
        for name, kp in keypoints.items()
        if is_valid_keypoint(kp, threshold=conf_thresh)
    }


def normalize_keypoints_visibility(
    keypoints: Optional[Dict[str, KeypointData]],
    threshold: Optional[float] = None
) -> Dict[str, KeypointData]:
    """Return a copy of keypoints with each landmark's `visible` flag synchronized to `threshold`."""
    if not keypoints:
        return {}
    conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD
    normalized: Dict[str, KeypointData] = {}
    for name, kp in keypoints.items():
        if kp is None:
            continue
        finite_ok = math.isfinite(kp.x) and math.isfinite(kp.y) and math.isfinite(kp.confidence)
        conf = max(0.0, min(1.0, float(kp.confidence))) if finite_ok else 0.0
        normalized[name] = KeypointData(
            x=float(kp.x) if finite_ok else 0.0,
            y=float(kp.y) if finite_ok else 0.0,
            confidence=conf,
            name=kp.name or name,
            visible=bool(finite_ok and conf >= conf_thresh),
        )
    return normalized


def compute_average_keypoint_confidence(
    keypoints: Optional[Dict[str, KeypointData]],
    landmark_names: Optional[List[str]] = None
) -> float:
    """Compute average finite keypoint confidence across selected landmarks (or all landmarks if omitted)."""
    if not keypoints:
        return 0.0
    names = landmark_names if landmark_names is not None else list(keypoints.keys())
    confs: List[float] = []
    for name in names:
        kp = keypoints.get(name)
        if kp is not None and math.isfinite(kp.confidence):
            confs.append(max(0.0, min(1.0, float(kp.confidence))))
    if not confs:
        return 0.0
    return round(sum(confs) / len(confs), 4)


def check_body_visibility(
    keypoints: Optional[Dict[str, KeypointData]],
    threshold: Optional[float] = None
) -> Dict[str, Any]:
    """Evaluate per-joint-group visibility for upper, core, and lower body."""
    if not keypoints:
        return {
            "full_body_visible": False,
            "visible_group_count": 0,
            "total_groups": len(BODY_LANDMARK_GROUPS),
            "missing_groups": [g[0] for g in BODY_LANDMARK_GROUPS],
            "missing_landmarks": [lm for _, l, r in BODY_LANDMARK_GROUPS for lm in (l, r)],
        }

    conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD
    visible_groups: List[str] = []
    missing_groups: List[str] = []
    missing_landmarks: List[str] = []

    for group_name, left_name, right_name in BODY_LANDMARK_GROUPS:
        l_ok = is_valid_keypoint(keypoints.get(left_name), conf_thresh)
        r_ok = is_valid_keypoint(keypoints.get(right_name), conf_thresh)
        if l_ok or r_ok:
            visible_groups.append(group_name)
        else:
            missing_groups.append(group_name)
            missing_landmarks.extend([left_name, right_name])

    full_body_visible = all(
        grp in visible_groups
        for grp in ("shoulders", "hips", "knees", "ankles")
    )
    return {
        "full_body_visible": full_body_visible,
        "visible_group_count": len(visible_groups),
        "total_groups": len(BODY_LANDMARK_GROUPS),
        "missing_groups": missing_groups,
        "missing_landmarks": missing_landmarks,
    }


def validate_keypoint_set(
    keypoints: Dict[str, KeypointData],
    required_names: List[str],
    threshold: Optional[float] = None
) -> Tuple[bool, List[str]]:
    """
    Validates that a list of keypoints are all present and valid.
    Returns (all_valid, list_of_missing_or_unreliable_names).
    """
    missing = []
    for name in required_names:
        kp = keypoints.get(name)
        if not is_valid_keypoint(kp, threshold):
            missing.append(name)
    return len(missing) == 0, missing


def check_pose_quality_for_planche(
    keypoints: Optional[Dict[str, KeypointData]],
    side_view: bool = False,
    threshold: Optional[float] = None
) -> Tuple[bool, str, List[str]]:
    """
    Evaluates whether the pose keypoints have sufficient quality to perform
    reliable planche biomechanical evaluation.

    Distinguishes:
    - High quality: returns (True, "OK", [])
    - Insufficient quality: returns (False, "INSUFFICIENT_POSE_QUALITY", missing_keypoints)
    """
    if not keypoints:
        return False, "NO_POSE_DETECTED", ["no_keypoints"]

    conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD

    # In side-view mode, we derive midpoints between left and right landmarks.
    # At minimum, at least one arm (shoulder+elbow+wrist) and core (shoulder+hip) must be visible.
    if side_view:
        # Check core side-view points
        has_l_sh = is_valid_keypoint(keypoints.get("left_shoulder"), conf_thresh)
        has_r_sh = is_valid_keypoint(keypoints.get("right_shoulder"), conf_thresh)
        has_l_hip = is_valid_keypoint(keypoints.get("left_hip"), conf_thresh)
        has_r_hip = is_valid_keypoint(keypoints.get("right_hip"), conf_thresh)
        has_l_wrist = is_valid_keypoint(keypoints.get("left_wrist"), conf_thresh)
        has_r_wrist = is_valid_keypoint(keypoints.get("right_wrist"), conf_thresh)

        missing = []
        if not (has_l_sh or has_r_sh):
            missing.append("shoulders")
        if not (has_l_hip or has_r_hip):
            missing.append("hips")
        if not (has_l_wrist or has_r_wrist):
            missing.append("wrists")

        if missing:
            return False, "INSUFFICIENT_POSE_QUALITY", missing

        return True, "OK", []

    else:
        # Standard mode: check if at least one complete side or body axis is visible
        has_left_side = all([
            is_valid_keypoint(keypoints.get("left_shoulder"), conf_thresh),
            is_valid_keypoint(keypoints.get("left_hip"), conf_thresh),
            is_valid_keypoint(keypoints.get("left_wrist"), conf_thresh),
        ])
        has_right_side = all([
            is_valid_keypoint(keypoints.get("right_shoulder"), conf_thresh),
            is_valid_keypoint(keypoints.get("right_hip"), conf_thresh),
            is_valid_keypoint(keypoints.get("right_wrist"), conf_thresh),
        ])

        if not (has_left_side or has_right_side):
            missing = []
            if not (is_valid_keypoint(keypoints.get("left_shoulder"), conf_thresh) or is_valid_keypoint(keypoints.get("right_shoulder"), conf_thresh)):
                missing.append("shoulders")
            if not (is_valid_keypoint(keypoints.get("left_hip"), conf_thresh) or is_valid_keypoint(keypoints.get("right_hip"), conf_thresh)):
                missing.append("hips")
            if not (is_valid_keypoint(keypoints.get("left_wrist"), conf_thresh) or is_valid_keypoint(keypoints.get("right_wrist"), conf_thresh)):
                missing.append("wrists")
            return False, "INSUFFICIENT_POSE_QUALITY", missing

        return True, "OK", []
