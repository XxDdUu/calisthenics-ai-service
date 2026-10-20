from typing import Dict, List, Optional, Set, Tuple
from app.core.config import settings
from app.schemas.analysis import KeypointData


def is_valid_keypoint(
    kp: Optional[KeypointData],
    threshold: Optional[float] = None
) -> bool:
    """
    Check if a keypoint exists, is visible, and meets the confidence threshold.
    """
    if kp is None:
        return False
    conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD
    return bool(kp.visible and kp.confidence >= conf_thresh)


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
