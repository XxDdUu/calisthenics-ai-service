import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.biomechanics.angles import (
    calculate_angle_2d,
    calculate_horizontal_angle,
    calculate_vertical_angle,
    get_point,
)
from app.biomechanics.confidence import is_valid_keypoint
from app.core.config import settings
from app.schemas.analysis import KeypointData


_MIDPOINT_CONF_DELTA_THRESHOLD = 0.20
_MIDPOINT_SPATIAL_THRESHOLD_PX = 80.0


def compute_midpoint(
    kp1: Optional[KeypointData],
    kp2: Optional[KeypointData],
    name: str,
    threshold: Optional[float] = None
) -> Optional[KeypointData]:
    """
    Computes the midpoint between two landmarks (e.g. left and right shoulder).

    Occlusion-aware snapping:
    - If both keypoints are valid, check for potential YOLO hallucination on the
      occluded side.
    - When the confidence delta exceeds ``_MIDPOINT_CONF_DELTA_THRESHOLD`` (0.20)
      **or** the Euclidean distance between them exceeds
      ``_MIDPOINT_SPATIAL_THRESHOLD_PX`` (80 px), snap the "midpoint" to the
      higher-confidence keypoint instead of averaging.  This prevents corrupted
      positions (e.g. hallucinated occluded knee near the torso) from pulling the
      midpoint off the true body axis.
    - If only one keypoint is valid, that keypoint is used directly (same snap).
    - If neither is valid, returns None.
    """
    conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD

    kp1_valid = is_valid_keypoint(kp1, conf_thresh)
    kp2_valid = is_valid_keypoint(kp2, conf_thresh)

    # If neither is valid, return nothing
    if not kp1_valid and not kp2_valid:
        return None

    # If only one side is valid, use it directly
    if kp1_valid and not kp2_valid:
        assert kp1 is not None
        return KeypointData(x=round(kp1.x, 2), y=round(kp1.y, 2),
                            confidence=round(kp1.confidence, 3), name=name, visible=True)
    if kp2_valid and not kp1_valid:
        assert kp2 is not None
        return KeypointData(x=round(kp2.x, 2), y=round(kp2.y, 2),
                            confidence=round(kp2.confidence, 3), name=name, visible=True)

    assert kp1 is not None and kp2 is not None

    conf_delta = abs(kp1.confidence - kp2.confidence)
    distance = math.sqrt((kp1.x - kp2.x) ** 2 + (kp1.y - kp2.y) ** 2)

    # Snap to the higher-confidence side when occlusion is suspected
    if conf_delta > _MIDPOINT_CONF_DELTA_THRESHOLD or distance > _MIDPOINT_SPATIAL_THRESHOLD_PX:
        dominant = kp1 if kp1.confidence >= kp2.confidence else kp2
        return KeypointData(
            x=round(dominant.x, 2),
            y=round(dominant.y, 2),
            confidence=round(dominant.confidence, 3),
            name=name,
            visible=True,
        )

    mid_x = (kp1.x + kp2.x) / 2.0
    mid_y = (kp1.y + kp2.y) / 2.0
    mid_conf = min(kp1.confidence, kp2.confidence)

    return KeypointData(
        x=round(mid_x, 2),
        y=round(mid_y, 2),
        confidence=round(mid_conf, 3),
        name=name,
        visible=True
    )



def resolve_side_view_chain(keypoints: Dict[str, KeypointData], threshold: float = 0.25) -> Dict[str, KeypointData]:
    """Select independent best arm and leg landmarks for a profile view."""
    def valid(name: str, extremity: bool = False) -> Optional[KeypointData]:
        kp = keypoints.get(name)
        return kp if kp is not None and kp.confidence >= threshold else None

    def copy_as(name: str, kp: KeypointData) -> KeypointData:
        return KeypointData(x=kp.x, y=kp.y, confidence=kp.confidence, name=name, visible=True)

    chain: Dict[str, KeypointData] = {}
    # Support arm: the lower (ground-nearer) valid wrist is preferred; confidence
    # breaks ties. Select the complete arm from that side so its joints remain collinear.
    arms = []
    for side in ("left", "right"):
        sh, el, wr = valid(f"{side}_shoulder"), valid(f"{side}_elbow"), valid(f"{side}_wrist")
        if sh and el and wr:
            arms.append((wr.y, wr.confidence, side, (sh, el, wr)))
    if arms:
        if len(arms) == 2 and abs(arms[0][0] - arms[1][0]) < 15.0 and abs(arms[0][1] - arms[1][1]) < 0.20:
            points = tuple(KeypointData(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0,
                                       confidence=min(a.confidence, b.confidence), name=j, visible=True)
                           for j, a, b in zip(("shoulder", "elbow", "wrist"), arms[0][3], arms[1][3]))
        else:
            _, _, _, points = max(arms, key=lambda item: (item[0], item[1]))
        for joint, kp in zip(("shoulder", "elbow", "wrist"), points):
            chain[joint] = copy_as(joint, kp)

    # Occluded leg: choose the side with a valid ankle and the longer hip->ankle
    # reach, rather than averaging a hallucinated opposite leg.
    legs = []
    for side in ("left", "right"):
        hip, knee, ankle = valid(f"{side}_hip"), valid(f"{side}_knee"), valid(f"{side}_ankle")
        if hip and knee and ankle:
            reach = math.hypot(ankle.x - hip.x, ankle.y - hip.y)
            legs.append((reach, ankle.confidence, side, (hip, knee, ankle)))
    if legs:
        if len(legs) == 2 and abs(legs[0][0] - legs[1][0]) < 15.0 and abs(legs[0][1] - legs[1][1]) < 0.20:
            points = tuple(KeypointData(x=(a.x + b.x) / 2.0, y=(a.y + b.y) / 2.0,
                                       confidence=min(a.confidence, b.confidence), name=j, visible=True)
                           for j, a, b in zip(("hip", "knee", "ankle"), legs[0][3], legs[1][3]))
        else:
            _, _, _, points = max(legs, key=lambda item: (item[0], item[1]))
        for joint, kp in zip(("hip", "knee", "ankle"), points):
            chain[joint] = copy_as(joint, kp)
    return chain


@dataclass
class SideViewProcessingResult:
    """Contains combined original + derived midpoint keypoints, plus quality status."""
    keypoints: Dict[str, KeypointData]
    midpoints: Dict[str, KeypointData]
    is_valid_side_view: bool
    missing_landmarks: List[str] = field(default_factory=list)


from pydantic import BaseModel, Field


class SideViewBiomechanicalFeatures(BaseModel):
    """Biomechanical features extracted specifically from side-view midpoint landmarks."""
    body_line_deg: Optional[float] = None          # Shoulder - Hip - Knee/Ankle alignment
    hip_extension_deg: Optional[float] = None       # mid_shoulder - mid_hip - mid_knee
    knee_extension_deg: Optional[float] = None      # mid_hip - mid_knee - mid_ankle
    elbow_extension_deg: Optional[float] = None     # shoulder - elbow - wrist (active arm)
    shoulder_lean_deg: Optional[float] = None       # mid_shoulder vertical lean over wrists
    body_tilt_deg: Optional[float] = None           # mid_shoulder - mid_hip horizontal tilt
    
    # Original landmark features preserved (especially for straddle / bilateral checks)
    left_knee_deg: Optional[float] = None
    right_knee_deg: Optional[float] = None
    left_elbow_deg: Optional[float] = None
    right_elbow_deg: Optional[float] = None
    primary_side: str = "left"

    unreliable_features: List[str] = Field(default_factory=list)
    feature_confidence: Dict[str, float] = Field(default_factory=dict)


class SideViewKeypointProcessor:
    """
    Dedicated processing layer for Side-View Planche analysis.

    Responsibilities:
    1. Validate required YOLO landmarks.
    2. Derive midpoint landmarks (mid_shoulder, mid_hip, mid_knee, mid_ankle).
    3. Preserve all original YOLO landmarks alongside derived midpoints.
    4. Calculate interpretable side-view biomechanical features.
    """

    @classmethod
    def process_keypoints(
        cls,
        keypoints: Dict[str, KeypointData],
        threshold: Optional[float] = None
    ) -> SideViewProcessingResult:
        """
        Derives midpoint landmarks while preserving original YOLO landmarks.
        """
        conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD

        # Copy original landmarks to preserve them completely
        combined_keypoints = dict(keypoints)
        midpoints: Dict[str, KeypointData] = {}
        missing: List[str] = []

        # Resolve arm and leg independently; this avoids mixing a good ankle
        # with an occluded knee from the opposite side.
        chain = resolve_side_view_chain(keypoints, threshold=min(conf_thresh, 0.25))
        for joint, kp in chain.items():
            combined_keypoints[f"side_{joint}"] = kp

        pairs = [
            ("left_shoulder", "right_shoulder", "mid_shoulder"),
            ("left_hip", "right_hip", "mid_hip"),
            ("left_knee", "right_knee", "mid_knee"),
            ("left_ankle", "right_ankle", "mid_ankle"),
        ]
        for left_name, right_name, mid_name in pairs:
            joint = mid_name[4:]
            mid = chain.get(joint)
            if mid is None:
                mid = compute_midpoint(keypoints.get(left_name), keypoints.get(right_name), name=mid_name, threshold=conf_thresh)
            else:
                mid = KeypointData(x=mid.x, y=mid.y, confidence=mid.confidence, name=mid_name, visible=True)
            if mid is not None:
                combined_keypoints[mid_name] = mid
                midpoints[mid_name] = mid
            else:
                missing.append(mid_name)

        # Essential midpoints for side-view are shoulder and hip
        is_valid = ("mid_shoulder" in midpoints and "mid_hip" in midpoints)

        return SideViewProcessingResult(
            keypoints=combined_keypoints,
            midpoints=midpoints,
            is_valid_side_view=is_valid,
            missing_landmarks=missing
        )

    @classmethod
    def extract_features(
        cls,
        processed_result: SideViewProcessingResult,
        threshold: Optional[float] = None
    ) -> SideViewBiomechanicalFeatures:
        """
        Extracts interpretable biomechanical features from side-view keypoints.
        Does not invent fake values when landmarks are missing or unreliable.
        """
        conf_thresh = threshold if threshold is not None else settings.KEYPOINT_CONF_THRESHOLD
        kps = processed_result.keypoints
        midpoints = processed_result.midpoints

        unreliable: List[str] = []
        feature_conf: Dict[str, float] = {}

        mid_sh = midpoints.get("mid_shoulder")
        mid_hip = midpoints.get("mid_hip")
        mid_knee = midpoints.get("mid_knee")
        mid_ankle = midpoints.get("mid_ankle")

        # 1. Body Tilt (mid_shoulder to mid_hip horizontal tilt relative to ground)
        body_tilt: Optional[float] = None
        if mid_sh and mid_hip and mid_sh.visible and mid_hip.visible:
            body_tilt = calculate_horizontal_angle((mid_sh.x, mid_sh.y), (mid_hip.x, mid_hip.y))
            feature_conf["body_tilt"] = min(mid_sh.confidence, mid_hip.confidence)
        else:
            unreliable.append("body_tilt")

        # 2. Hip Extension (mid_shoulder - mid_hip - mid_knee)
        hip_ext: Optional[float] = None
        if mid_sh and mid_hip and mid_knee and mid_sh.visible and mid_hip.visible and mid_knee.visible:
            hip_ext = calculate_angle_2d((mid_sh.x, mid_sh.y), (mid_hip.x, mid_hip.y), (mid_knee.x, mid_knee.y))
            feature_conf["hip_extension"] = min(mid_sh.confidence, mid_hip.confidence, mid_knee.confidence)
        elif mid_sh and mid_hip and mid_ankle and mid_sh.visible and mid_hip.visible and mid_ankle.visible:
            # Fallback to ankle if knees are locked straight
            hip_ext = calculate_angle_2d((mid_sh.x, mid_sh.y), (mid_hip.x, mid_hip.y), (mid_ankle.x, mid_ankle.y))
            feature_conf["hip_extension"] = min(mid_sh.confidence, mid_hip.confidence, mid_ankle.confidence)
        else:
            unreliable.append("hip_extension")

        # 3. Knee Extension (mid_hip - mid_knee - mid_ankle)
        knee_ext: Optional[float] = None
        if mid_hip and mid_knee and mid_ankle and mid_hip.visible and mid_knee.visible and mid_ankle.visible:
            knee_ext = calculate_angle_2d((mid_hip.x, mid_hip.y), (mid_knee.x, mid_knee.y), (mid_ankle.x, mid_ankle.y))
            feature_conf["knee_extension"] = min(mid_hip.confidence, mid_knee.confidence, mid_ankle.confidence)
        else:
            unreliable.append("knee_extension")

        # 4. Body Line Alignment (overall linearity shoulder - hip - leg endpoint)
        body_line: Optional[float] = None
        leg_endpoint = mid_ankle if (mid_ankle and mid_ankle.visible) else (mid_knee if (mid_knee and mid_knee.visible) else None)
        if mid_sh and mid_hip and leg_endpoint and mid_sh.visible and mid_hip.visible:
            body_line = calculate_angle_2d((mid_sh.x, mid_sh.y), (mid_hip.x, mid_hip.y), (leg_endpoint.x, leg_endpoint.y))
            feature_conf["body_line"] = min(mid_sh.confidence, mid_hip.confidence, leg_endpoint.confidence)
        else:
            unreliable.append("body_line")

        # 5. Elbow Extension from original landmarks (active/visible arm)
        resolved_sh = get_point(kps.get("side_shoulder"))
        resolved_el = get_point(kps.get("side_elbow"))
        resolved_wr = get_point(kps.get("side_wrist"))
        l_sh = get_point(kps.get("left_shoulder"))
        l_el = get_point(kps.get("left_elbow"))
        l_wr = get_point(kps.get("left_wrist"))
        r_sh = get_point(kps.get("right_shoulder"))
        r_el = get_point(kps.get("right_elbow"))
        r_wr = get_point(kps.get("right_wrist"))

        l_el_angle = calculate_angle_2d(l_sh, l_el, l_wr) if (l_sh and l_el and l_wr) else None
        r_el_angle = calculate_angle_2d(r_sh, r_el, r_wr) if (r_sh and r_el and r_wr) else None
        resolved_el_angle = calculate_angle_2d(resolved_sh, resolved_el, resolved_wr) if (resolved_sh and resolved_el and resolved_wr) else None

        # Determine primary arm based on landmark confidence
        l_conf = sum([kps[k].confidence for k in ["left_shoulder", "left_elbow", "left_wrist"] if k in kps and is_valid_keypoint(kps[k], conf_thresh)])
        r_conf = sum([kps[k].confidence for k in ["right_shoulder", "right_elbow", "right_wrist"] if k in kps and is_valid_keypoint(kps[k], conf_thresh)])
        primary_side = "left" if l_conf >= r_conf else "right"

        elbow_ext: Optional[float] = resolved_el_angle
        if elbow_ext is not None:
            feature_conf["elbow_extension"] = min(1.0, max(0.0, kps["side_elbow"].confidence))
        elif primary_side == "left" and l_el_angle is not None:
            elbow_ext = l_el_angle
            feature_conf["elbow_extension"] = l_conf / 3.0
        elif primary_side == "right" and r_el_angle is not None:
            elbow_ext = r_el_angle
            feature_conf["elbow_extension"] = r_conf / 3.0
        elif l_el_angle is not None:
            elbow_ext = l_el_angle
            feature_conf["elbow_extension"] = l_conf / 3.0
        elif r_el_angle is not None:
            elbow_ext = r_el_angle
            feature_conf["elbow_extension"] = r_conf / 3.0
        else:
            unreliable.append("elbow_extension")

        # 6. Shoulder Position (Forward Lean over Wrists)
        # Normalized forward lean angle in degrees relative to vertical plumline
        active_wrist = resolved_wr or (l_wr if primary_side == "left" and l_wr else (r_wr if r_wr else l_wr))
        shoulder_lean: Optional[float] = None
        if mid_sh and active_wrist:
            shoulder_lean = calculate_vertical_angle((mid_sh.x, mid_sh.y), active_wrist)
            feature_conf["shoulder_position"] = mid_sh.confidence
        elif (l_sh or r_sh) and active_wrist:
            sh_pt = l_sh if l_sh else r_sh
            shoulder_lean = calculate_vertical_angle(sh_pt, active_wrist)
            feature_conf["shoulder_position"] = 0.5
        else:
            unreliable.append("shoulder_position")

        # 7. Bilateral Knee Angles from original landmarks (preserved for Straddle)
        l_hip = get_point(kps.get("left_hip"))
        l_kn = get_point(kps.get("left_knee"))
        l_ak = get_point(kps.get("left_ankle"))
        r_hip = get_point(kps.get("right_hip"))
        r_kn = get_point(kps.get("right_knee"))
        r_ak = get_point(kps.get("right_ankle"))

        l_knee_angle = calculate_angle_2d(l_hip, l_kn, l_ak) if (l_hip and l_kn and l_ak) else None
        r_knee_angle = calculate_angle_2d(r_hip, r_kn, r_ak) if (r_hip and r_kn and r_ak) else None

        return SideViewBiomechanicalFeatures(
            body_line_deg=round(body_line, 1) if body_line is not None else None,
            hip_extension_deg=round(hip_ext, 1) if hip_ext is not None else None,
            knee_extension_deg=round(knee_ext, 1) if knee_ext is not None else None,
            elbow_extension_deg=round(elbow_ext, 1) if elbow_ext is not None else None,
            shoulder_lean_deg=round(shoulder_lean, 1) if shoulder_lean is not None else None,
            body_tilt_deg=round(body_tilt, 1) if body_tilt is not None else None,
            left_knee_deg=round(l_knee_angle, 1) if l_knee_angle is not None else None,
            right_knee_deg=round(r_knee_angle, 1) if r_knee_angle is not None else None,
            left_elbow_deg=round(l_el_angle, 1) if l_el_angle is not None else None,
            right_elbow_deg=round(r_el_angle, 1) if r_el_angle is not None else None,
            primary_side=primary_side,
            unreliable_features=unreliable,
            feature_confidence=feature_conf
        )
