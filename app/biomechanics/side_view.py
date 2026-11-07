import logging
import math
import time
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
from app.core.logging_context import get_request_id
from app.schemas.analysis import KeypointData

logger = logging.getLogger(__name__)


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
    rid = get_request_id()
    if conf_delta > _MIDPOINT_CONF_DELTA_THRESHOLD or distance > _MIDPOINT_SPATIAL_THRESHOLD_PX:
        dominant = kp1 if kp1.confidence >= kp2.confidence else kp2
        logger.debug(
            "[KEYPOINT_TRACE] request_id=%s stage=midpoint_snap name=%s snap_to=%s conf_delta=%.3f dist_px=%.1f x=%.2f y=%.2f conf=%.3f",
            rid, name, dominant.name, conf_delta, distance, dominant.x, dominant.y, dominant.confidence
        )
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



def resolve_side_view_chain(
    keypoints: Dict[str, KeypointData],
    threshold: float = 0.25,
    previous_chain: Optional[Dict[str, KeypointData]] = None,
) -> Dict[str, KeypointData]:
    """
    Select anatomically plausible, profile-view landmarks for the visible arm and leg.

    Landmarks are assessed individually without all-or-nothing gating across joints:
    - Arm joints (shoulder, elbow, wrist) are resolved based on ground-support kinematics.
    - Leg joints (hip, knee, ankle) are resolved independently using visibility, confidence,
      anatomical plausibility, segment length consistency (thigh/shank), and temporal stability.
    - Hallucinated occluded joints (e.g., knee pulled toward pelvis) are rejected rather than averaged.
    - If a joint cannot be reliably resolved, it is marked unavailable (omitted from chain).
    """
    def valid(name: str) -> Optional[KeypointData]:
        kp = keypoints.get(name)
        if kp is not None and kp.visible and kp.confidence >= threshold:
            return kp
        return None

    def copy_as(name: str, kp: KeypointData) -> KeypointData:
        return KeypointData(x=kp.x, y=kp.y, confidence=kp.confidence, name=name, visible=True)

    chain: Dict[str, KeypointData] = {}
    rid = get_request_id()

    # Log candidates for bilateral joints of interest
    for j_name in (
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle"
    ):
        kp_cand = keypoints.get(j_name)
        log = logger.info if j_name in (
            "left_knee", "right_knee", "left_elbow", "right_elbow"
        ) else logger.debug
        if kp_cand is not None:
            log(
                "[KEYPOINT_TRACE] request_id=%s stage=side_view_candidate joint=%s x=%.2f y=%.2f conf=%.4f visible=%s",
                rid, j_name, kp_cand.x, kp_cand.y, kp_cand.confidence, kp_cand.visible
            )
        else:
            log(
                "[KEYPOINT_TRACE] request_id=%s stage=side_view_candidate joint=%s status=missing",
                rid, j_name
            )

    # ---------------------------------------------------------
    # 1. Arm Resolution (Support arm in Planche)
    # ---------------------------------------------------------
    arms = []
    for side in ("left", "right"):
        sh, el, wr = valid(f"{side}_shoulder"), valid(f"{side}_elbow"), valid(f"{side}_wrist")
        if sh and el and wr:
            arms.append((wr.y, wr.confidence, side, (sh, el, wr)))
        elif wr and (sh or el):
            arms.append((wr.y, wr.confidence * 0.85, side, (sh, el, wr)))

    primary_arm_side = "left"
    arm_selection_reason = "fallback_left"
    if arms:
        if len(arms) == 2 and abs(arms[0][0] - arms[1][0]) < 15.0 and abs(arms[0][1] - arms[1][1]) < 0.20:
            pts = tuple(
                KeypointData(
                    x=(a.x + b.x) / 2.0 if (a and b) else (a.x if a else b.x),
                    y=(a.y + b.y) / 2.0 if (a and b) else (a.y if a else b.y),
                    confidence=min(a.confidence, b.confidence) if (a and b) else (a.confidence if a else b.confidence),
                    name=j, visible=True
                )
                for j, a, b in zip(("shoulder", "elbow", "wrist"), arms[0][3], arms[1][3])
                if (a or b)
            )
            primary_arm_side = arms[0][2]
            arm_selection_reason = "bilateral_average"
        else:
            best_arm = max(arms, key=lambda item: (item[0], item[1]))
            primary_arm_side = best_arm[2]
            pts = best_arm[3]
            arm_selection_reason = f"max_wrist_y_conf_{best_arm[2]}"

        for joint, kp in zip(("shoulder", "elbow", "wrist"), pts):
            if kp is not None:
                chain[joint] = copy_as(joint, kp)
    else:
        for joint in ("shoulder", "elbow", "wrist"):
            cand = valid(f"left_{joint}") or valid(f"right_{joint}")
            if cand:
                chain[joint] = copy_as(joint, cand)
                arm_selection_reason = "first_valid_joint_cand"

    for joint in ("shoulder", "elbow", "wrist"):
        selected = chain.get(joint)
        log = logger.info if joint in ("elbow", "knee") else logger.debug
        if selected is not None:
            log(
                "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=%s selected_source=%s "
                "x=%.2f y=%.2f conf=%.4f reason=%s",
                rid, joint,
                "bilateral_average" if arm_selection_reason == "bilateral_average" else primary_arm_side,
                selected.x, selected.y, selected.confidence, arm_selection_reason
            )
        else:
            log(
                "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=%s "
                "status=unresolved reason=no_valid_arm_candidate",
                rid, joint
            )

    # ---------------------------------------------------------
    # 2. Reference Torso & Hip Resolution
    # ---------------------------------------------------------
    sh_ref = chain.get("shoulder") or valid("left_shoulder") or valid("right_shoulder")
    hip_l = valid("left_hip")
    hip_r = valid("right_hip")

    # Resolve hip using the same occlusion-aware snapping used by compute_midpoint.
    # This guarantees that hip_ref (used for torso_len and knee evaluation) and
    # mid_hip (computed later in process_keypoints) agree on the same coordinate
    # whenever only one side is reliable.
    if hip_l and hip_r:
        conf_delta_hip = abs(hip_l.confidence - hip_r.confidence)
        dist_hip = math.hypot(hip_l.x - hip_r.x, hip_l.y - hip_r.y)
        if conf_delta_hip > _MIDPOINT_CONF_DELTA_THRESHOLD:
            # One side dominates — snap to it (mirrors compute_midpoint snapping)
            chosen_hip = hip_l if hip_l.confidence > hip_r.confidence else hip_r
            hip_source = "left" if hip_l.confidence > hip_r.confidence else "right"
            hip_reason = f"confidence_delta_{conf_delta_hip:.3f}_exceeds_threshold"
        elif dist_hip < 40.0:
            # Hips overlap (true side view) — prefer same side as arm to stay consistent
            chosen_hip = hip_l if primary_arm_side == "left" else hip_r
            hip_source = "left" if primary_arm_side == "left" else "right"
            hip_reason = f"hips_overlap_dist_{dist_hip:.1f}px_prefer_arm_side_{primary_arm_side}"
        else:
            chosen_hip = hip_l if hip_l.confidence >= hip_r.confidence else hip_r
            hip_source = "left" if hip_l.confidence >= hip_r.confidence else "right"
            hip_reason = "highest_confidence"
        chain["hip"] = copy_as("hip", chosen_hip)
        logger.debug(
            "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=hip selected_source=%s x=%.2f y=%.2f conf=%.4f reason=%s",
            rid, hip_source, chosen_hip.x, chosen_hip.y, chosen_hip.confidence, hip_reason
        )
    elif hip_l:
        chain["hip"] = copy_as("hip", hip_l)
        logger.debug(
            "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=hip selected_source=left x=%.2f y=%.2f conf=%.4f reason=only_left_valid",
            rid, hip_l.x, hip_l.y, hip_l.confidence
        )
    elif hip_r:
        chain["hip"] = copy_as("hip", hip_r)
        logger.debug(
            "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=hip selected_source=right x=%.2f y=%.2f conf=%.4f reason=only_right_valid",
            rid, hip_r.x, hip_r.y, hip_r.confidence
        )
    else:
        logger.debug(
            "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=hip status=unresolved reason=neither_hip_valid",
            rid
        )
        chain["hip"] = copy_as("hip", hip_r)

    hip_ref = chain.get("hip")

    if sh_ref and hip_ref:
        torso_len = math.hypot(hip_ref.x - sh_ref.x, hip_ref.y - sh_ref.y)
    else:
        torso_len = 100.0
    torso_len = max(torso_len, 20.0)

    left_leg_conf = sum([keypoints[k].confidence for k in ("left_hip", "left_knee", "left_ankle") if k in keypoints and keypoints[k].visible])
    right_leg_conf = sum([keypoints[k].confidence for k in ("right_hip", "right_knee", "right_ankle") if k in keypoints and keypoints[k].visible])

    prev_leg_side = None
    if previous_chain and "knee" in previous_chain and hip_ref:
        prev_k = previous_chain["knee"]
        d_l = math.hypot(keypoints["left_knee"].x - prev_k.x, keypoints["left_knee"].y - prev_k.y) if "left_knee" in keypoints else 999.0
        d_r = math.hypot(keypoints["right_knee"].x - prev_k.x, keypoints["right_knee"].y - prev_k.y) if "right_knee" in keypoints else 999.0
        if d_l < 30.0 and d_l < d_r:
            prev_leg_side = "left"
        elif d_r < 30.0 and d_r < d_l:
            prev_leg_side = "right"

    if prev_leg_side is not None:
        primary_leg_side = prev_leg_side
    elif abs(left_leg_conf - right_leg_conf) > 0.30:
        primary_leg_side = "left" if left_leg_conf > right_leg_conf else "right"
    else:
        primary_leg_side = primary_arm_side

    logger.debug("[SIDE_VIEW] hip_ref source=%s x=%s y=%s conf=%s",
                 "chain" if hip_ref else "none",
                 f"{hip_ref.x:.1f}" if hip_ref else "n/a",
                 f"{hip_ref.y:.1f}" if hip_ref else "n/a",
                 f"{hip_ref.confidence:.3f}" if hip_ref else "n/a")
    logger.debug("[SIDE_VIEW] request_id=%s torso_len=%.1f primary_leg_side=%s",
                 rid, torso_len, primary_leg_side)

    # ---------------------------------------------------------
    # 3. Joint Knee–Ankle Evaluation
    #
    # We evaluate each (side_knee, side_ankle) pair together so that a
    # high-confidence but segment-implausible knee does not discard a
    # viable ankle from the OTHER side whose knee is anatomically sound.
    #
    # Scoring of each complete candidate:
    #   score = knee_quality * ankle_quality_weight
    # where ankle_quality_weight is:
    #   1.0   if the same-side ankle is plausible (full chain available)
    #   0.80  if the same-side ankle is missing but knee alone is plausible
    #   0.0   if the knee segment itself is implausible
    # ---------------------------------------------------------
    def evaluate_knee(knee_kp: Optional[KeypointData], hip_kp: Optional[KeypointData]) -> Tuple[bool, float]:
        """Assess anatomical plausibility and segment consistency of knee candidate."""
        if knee_kp is None or hip_kp is None:
            return False, 0.0
        thigh_len = math.hypot(knee_kp.x - hip_kp.x, knee_kp.y - hip_kp.y)
        # Check for collapsed knee hallucination (occluded knee pulled toward torso/pelvis)
        # Normal femur length is 0.5x - 1.4x torso length. Under 0.40x indicates collapsed joint.
        if thigh_len < 0.40 * torso_len or thigh_len > 1.60 * torso_len:
            return False, 0.0

        expected_len = 0.85 * torso_len
        ratio_penalty = abs(thigh_len - expected_len) / expected_len
        quality = float(knee_kp.confidence) * max(0.2, 1.0 - 0.4 * ratio_penalty)
        return True, quality

    def evaluate_ankle(ankle_kp: Optional[KeypointData], knee_kp: Optional[KeypointData], hip_kp: Optional[KeypointData]) -> Tuple[bool, float]:
        """Assess anatomical plausibility and segment consistency of ankle candidate."""
        if ankle_kp is None:
            return False, 0.0
        if knee_kp is not None:
            shank_len = math.hypot(ankle_kp.x - knee_kp.x, ankle_kp.y - knee_kp.y)
            if shank_len < 0.35 * torso_len or shank_len > 1.60 * torso_len:
                return False, 0.0
            expected_len = 0.85 * torso_len
            ratio_penalty = abs(shank_len - expected_len) / expected_len
            quality = float(ankle_kp.confidence) * max(0.2, 1.0 - 0.4 * ratio_penalty)
            return True, quality
        elif hip_kp is not None:
            leg_len = math.hypot(ankle_kp.x - hip_kp.x, ankle_kp.y - hip_kp.y)
            if leg_len < 0.70 * torso_len or leg_len > 2.40 * torso_len:
                return False, 0.0
            return True, float(ankle_kp.confidence)
        return False, 0.0

    l_knee, r_knee = valid("left_knee"), valid("right_knee")
    l_ankle_raw, r_ankle_raw = valid("left_ankle"), valid("right_ankle")

    # Evaluate each knee candidate individually against hip_ref
    l_k_ok, l_k_score = evaluate_knee(l_knee, hip_ref)
    r_k_ok, r_k_score = evaluate_knee(r_knee, hip_ref)

    if l_knee:
        logger.debug("[SIDE_VIEW] left_knee_candidate x=%.1f y=%.1f conf=%.3f knee_ok=%s knee_score=%.3f",
                     l_knee.x, l_knee.y, l_knee.confidence, l_k_ok, l_k_score)
    else:
        logger.debug("[SIDE_VIEW] left_knee_candidate not_available threshold=%.2f", threshold)
    if r_knee:
        logger.debug("[SIDE_VIEW] right_knee_candidate x=%.1f y=%.1f conf=%.3f knee_ok=%s knee_score=%.3f",
                     r_knee.x, r_knee.y, r_knee.confidence, r_k_ok, r_k_score)
    else:
        logger.debug("[SIDE_VIEW] right_knee_candidate not_available threshold=%.2f", threshold)

    # Evaluate same-side ankle for each valid knee to produce a joint score
    def _joint_score(knee_ok: bool, k_score: float, knee_kp: Optional[KeypointData], ankle_kp: Optional[KeypointData]) -> float:
        """Combined knee+ankle score. Missing ankle degrades but does not disqualify."""
        if not knee_ok:
            return 0.0
        a_ok, a_score = evaluate_ankle(ankle_kp, knee_kp, hip_ref)
        if a_ok:
            return k_score * (0.5 + 0.5 * min(1.0, a_score / max(k_score, 0.01)))
        else:
            # Ankle missing or implausible: knee alone can still be selected
            # but with a 20 % penalty to prefer full chains when both sides compete
            return k_score * 0.80

    l_joint_score = _joint_score(l_k_ok, l_k_score, l_knee, l_ankle_raw)
    r_joint_score = _joint_score(r_k_ok, r_k_score, r_knee, r_ankle_raw)

    # Apply primary-leg-side prior (small +0.05 bias towards the expected visible side)
    if l_k_ok and primary_leg_side == "left":
        l_joint_score += 0.05
    if r_k_ok and primary_leg_side == "right":
        r_joint_score += 0.05

    selected_knee: Optional[KeypointData] = None
    selected_knee_side: str = primary_leg_side
    selected_ankle: Optional[KeypointData] = None

    if l_k_ok and r_k_ok:
        assert l_knee is not None and r_knee is not None
        # Overlapping knees in profile view — pick by joint score
        if math.hypot(l_knee.x - r_knee.x, l_knee.y - r_knee.y) < 25.0:
            if l_joint_score >= r_joint_score:
                selected_knee, selected_knee_side = l_knee, "left"
            else:
                selected_knee, selected_knee_side = r_knee, "right"
        else:
            # Both knees pass the hip-distance plausibility gate; choose by joint
            # score which accounts for ankle segment quality — no blind averaging.
            if l_joint_score >= r_joint_score:
                selected_knee, selected_knee_side = l_knee, "left"
            else:
                selected_knee, selected_knee_side = r_knee, "right"
    elif l_k_ok:
        selected_knee, selected_knee_side = l_knee, "left"
    elif r_k_ok:
        selected_knee, selected_knee_side = r_knee, "right"
    # else: both candidates are unreliable (Case C) — selected_knee stays None

    if selected_knee is not None:
        chain["knee"] = copy_as("knee", selected_knee)
        logger.info(
            "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=knee selected_source=%s x=%.2f y=%.2f conf=%.4f reason=joint_score_l=%.3f_r=%.3f_leg_side=%s",
            rid, selected_knee_side, selected_knee.x, selected_knee.y, selected_knee.confidence,
            l_joint_score, r_joint_score, primary_leg_side
        )
        logger.debug("[SIDE_VIEW] knee_selected side=%s x=%.1f y=%.1f conf=%.3f joint_score_l=%.3f joint_score_r=%.3f",
                     selected_knee_side, selected_knee.x, selected_knee.y, selected_knee.confidence,
                     l_joint_score, r_joint_score)
    else:
        logger.info(
            "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=knee status=unresolved reason=no_plausible_knee_l_ok=%s_r_ok=%s",
            rid, l_k_ok, r_k_ok
        )
        logger.warning("[SIDE_VIEW] knee_unresolved l_ok=%s r_ok=%s l_score=%.3f r_score=%.3f",
                       l_k_ok, r_k_ok, l_joint_score, r_joint_score)

    # ---------------------------------------------------------
    # 4. Ankle Resolution (after knee is settled)
    #
    # Only the same-side ankle is tried. The opposite ankle is NOT
    # substituted unless both knees were plausible AND overlapping
    # (bilateral alignment confirmed), per Case A requirement.
    # ---------------------------------------------------------
    l_ankle, r_ankle = l_ankle_raw, r_ankle_raw

    if l_ankle_raw:
        logger.debug("[SIDE_VIEW] left_ankle_candidate x=%.1f y=%.1f conf=%.3f",
                     l_ankle_raw.x, l_ankle_raw.y, l_ankle_raw.confidence)
    else:
        logger.debug("[SIDE_VIEW] left_ankle_candidate not_available threshold=%.2f", threshold)
    if r_ankle_raw:
        logger.debug("[SIDE_VIEW] right_ankle_candidate x=%.1f y=%.1f conf=%.3f",
                     r_ankle_raw.x, r_ankle_raw.y, r_ankle_raw.confidence)
    else:
        logger.debug("[SIDE_VIEW] right_ankle_candidate not_available threshold=%.2f", threshold)

    if "knee" in chain:
        ref_knee = chain["knee"]
        same_ankle = l_ankle if selected_knee_side == "left" else r_ankle

        same_a_ok, _ = evaluate_ankle(same_ankle, ref_knee, hip_ref)

        if same_a_ok and same_ankle is not None:
            chain["ankle"] = copy_as("ankle", same_ankle)
            logger.debug(
                "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=ankle selected_source=%s x=%.2f y=%.2f conf=%.4f reason=same_side_plausible",
                rid, selected_knee_side, same_ankle.x, same_ankle.y, same_ankle.confidence
            )
            logger.debug("[SIDE_VIEW] ankle_selected side=%s x=%.1f y=%.1f conf=%.3f",
                         selected_knee_side, same_ankle.x, same_ankle.y, same_ankle.confidence)
        else:
            # Case A: same-side ankle below threshold or implausible.
            # Do NOT silently substitute the opposite occluded/hallucinated ankle.
            # Hip and knee are retained; ankle is left unavailable.
            logger.debug(
                "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=ankle status=unresolved reason=same_side_%s_ankle_invalid_or_implausible",
                rid, selected_knee_side
            )
            logger.debug("[SIDE_VIEW] ankle_unresolved knee_available=%s same_a_ok=%s", True, same_a_ok)
    else:
        # knee unavailable → no unanchored ankle is fabricated
        logger.debug(
            "[KEYPOINT_TRACE] request_id=%s stage=side_view_selection joint=ankle status=unresolved reason=knee_unresolved",
            rid
        )
        logger.debug("[SIDE_VIEW] ankle_unresolved knee_available=%s", False)

    logger.debug("[SIDE_VIEW] chain_resolved joints=%s", list(chain.keys()))
    return chain


@dataclass
class SideViewProcessingResult:
    """Contains combined original + derived midpoint keypoints, plus quality status and resolved chain."""
    keypoints: Dict[str, KeypointData]
    midpoints: Dict[str, KeypointData]
    is_valid_side_view: bool
    missing_landmarks: List[str] = field(default_factory=list)
    resolved_chain: Dict[str, KeypointData] = field(default_factory=dict)


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
        rid = get_request_id()
        _t_res_start = time.perf_counter()

        # Copy original landmarks to preserve them completely
        combined_keypoints = dict(keypoints)
        midpoints: Dict[str, KeypointData] = {}
        missing: List[str] = []

        # Resolve arm and leg independently; this avoids mixing a good ankle
        # with an occluded knee from the opposite side.
        chain = resolve_side_view_chain(keypoints, threshold=min(conf_thresh, 0.25))
        for joint, kp in chain.items():
            combined_keypoints[f"side_{joint}"] = kp

        logger.debug("[SIDE_VIEW] request_id=%s chain_joints=%s", rid, list(chain.keys()))

        # 1. Torso center landmarks: mid_shoulder and mid_hip
        #
        # mid_shoulder: always derived from bilateral average/snap (both shoulders
        # are generally visible from the side).
        mid_sh = compute_midpoint(
            keypoints.get("left_shoulder"), keypoints.get("right_shoulder"),
            name="mid_shoulder", threshold=conf_thresh
        )
        if mid_sh is None and "shoulder" in chain:
            mid_sh = KeypointData(
                x=chain["shoulder"].x, y=chain["shoulder"].y,
                confidence=chain["shoulder"].confidence, name="mid_shoulder", visible=True
            )
        if mid_sh is not None:
            combined_keypoints["mid_shoulder"] = mid_sh
            midpoints["mid_shoulder"] = mid_sh
        else:
            missing.append("mid_shoulder")

        # mid_hip: use the SAME resolved hip from resolve_side_view_chain so that
        # mid_hip and hip_ref (used inside resolve_side_view_chain for torso length
        # and knee evaluation) are always identical.  Falling back to compute_midpoint
        # is kept only when the chain produced no hip at all.
        if "hip" in chain:
            mid_hip: Optional[KeypointData] = KeypointData(
                x=chain["hip"].x, y=chain["hip"].y,
                confidence=chain["hip"].confidence, name="mid_hip", visible=True
            )
        else:
            mid_hip = compute_midpoint(
                keypoints.get("left_hip"), keypoints.get("right_hip"),
                name="mid_hip", threshold=conf_thresh
            )
        if mid_hip is not None:
            combined_keypoints["mid_hip"] = mid_hip
            midpoints["mid_hip"] = mid_hip
        else:
            missing.append("mid_hip")

        # 2. Side-view knees and ankles: use ONLY the resolved landmark from resolve_side_view_chain.
        # DO NOT blindly average left and right knee or ankle predictions in side-view mode!
        # If the joint was not reliably resolved, mark it unresolved/unavailable.
        if "knee" in chain:
            mid_knee = KeypointData(x=chain["knee"].x, y=chain["knee"].y, confidence=chain["knee"].confidence, name="mid_knee", visible=True)
            combined_keypoints["mid_knee"] = mid_knee
            midpoints["mid_knee"] = mid_knee
        else:
            missing.append("mid_knee")

        if "ankle" in chain:
            mid_ankle = KeypointData(x=chain["ankle"].x, y=chain["ankle"].y, confidence=chain["ankle"].confidence, name="mid_ankle", visible=True)
            combined_keypoints["mid_ankle"] = mid_ankle
            midpoints["mid_ankle"] = mid_ankle
        else:
            missing.append("mid_ankle")

        # Essential midpoints for side-view are shoulder and hip
        is_valid = ("mid_shoulder" in midpoints and "mid_hip" in midpoints)

        t_res_ms = (time.perf_counter() - _t_res_start) * 1000.0
        logger.debug("[TIMING] request_id=%s stage=side_view_resolution duration_ms=%.2f", rid, t_res_ms)

        for m_name in ("mid_shoulder", "mid_hip", "mid_knee", "mid_ankle"):
            mkp = midpoints.get(m_name)
            if mkp is not None:
                log = logger.info if m_name == "mid_knee" else logger.debug
                log(
                    "[KEYPOINT_TRACE] request_id=%s stage=passed_to_biomechanics joint=%s x=%.2f y=%.2f conf=%.4f visible=%s",
                    rid, m_name, mkp.x, mkp.y, mkp.confidence, mkp.visible
                )
            else:
                log = logger.info if m_name == "mid_knee" else logger.debug
                log(
                    "[KEYPOINT_TRACE] request_id=%s stage=passed_to_biomechanics joint=%s status=missing",
                    rid, m_name
                )

        logger.debug("[SIDE_VIEW] request_id=%s midpoints=%s missing=%s",
                     rid,
                     {k: f"({v.x:.1f},{v.y:.1f})" for k, v in midpoints.items()},
                     missing)

        return SideViewProcessingResult(
            keypoints=combined_keypoints,
            midpoints=midpoints,
            is_valid_side_view=is_valid,
            missing_landmarks=missing,
            resolved_chain=chain,
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
        rid = get_request_id()
        _t_bio_start = time.perf_counter()
        kps = processed_result.keypoints
        midpoints = processed_result.midpoints

        unreliable: List[str] = []
        feature_conf: Dict[str, float] = {}

        mid_sh = midpoints.get("mid_shoulder")
        mid_hip = midpoints.get("mid_hip")
        mid_knee = midpoints.get("mid_knee")
        mid_ankle = midpoints.get("mid_ankle")

        def _log_segment(seg_name: str, p1: Optional[KeypointData], p2: Optional[KeypointData]):
            if p1 and p2 and p1.visible and p2.visible:
                dx = p2.x - p1.x
                dy = p2.y - p1.y
                length_px = math.hypot(dx, dy)
                logger.debug(
                    "[BIOMECHANICS] request_id=%s segment=%s start=(%.2f,%.2f) end=(%.2f,%.2f) vector=(dx=%.2f,dy=%.2f) length_px=%.2f",
                    rid, seg_name, p1.x, p1.y, p2.x, p2.y, dx, dy, length_px
                )
            else:
                missing_str = []
                if not p1 or not p1.visible:
                    missing_str.append("start")
                if not p2 or not p2.visible:
                    missing_str.append("end")
                logger.debug(
                    "[BIOMECHANICS] request_id=%s segment=%s status=missing missing_endpoint=%s",
                    rid, seg_name, "+".join(missing_str)
                )

        _log_segment("shoulder_to_hip", mid_sh, mid_hip)
        _log_segment("hip_to_knee", mid_hip, mid_knee)
        _log_segment("knee_to_ankle", mid_knee, mid_ankle)

        logger.debug("[BIOMECHANICS] request_id=%s extracting features mid_sh=%s mid_hip=%s mid_knee=%s mid_ankle=%s",
                     rid,
                     f"({mid_sh.x:.1f},{mid_sh.y:.1f})" if mid_sh else "none",
                     f"({mid_hip.x:.1f},{mid_hip.y:.1f})" if mid_hip else "none",
                     f"({mid_knee.x:.1f},{mid_knee.y:.1f})" if mid_knee else "none",
                     f"({mid_ankle.x:.1f},{mid_ankle.y:.1f})" if mid_ankle else "none")

        # 1. Body Tilt (mid_shoulder to mid_hip horizontal tilt relative to ground)
        body_tilt: Optional[float] = None
        if mid_sh and mid_hip and mid_sh.visible and mid_hip.visible:
            body_tilt = calculate_horizontal_angle((mid_sh.x, mid_sh.y), (mid_hip.x, mid_hip.y))
            feature_conf["body_tilt"] = min(mid_sh.confidence, mid_hip.confidence)
            logger.debug("[BIOMECHANICS] body_tilt=%.1f°", body_tilt)
        else:
            unreliable.append("body_tilt")
            logger.debug("[BIOMECHANICS] body_tilt=unreliable")

        # 2. Hip Extension (mid_shoulder - mid_hip - mid_knee)
        hip_ext: Optional[float] = None
        if mid_sh and mid_hip and mid_knee and mid_sh.visible and mid_hip.visible and mid_knee.visible:
            hip_ext = calculate_angle_2d((mid_sh.x, mid_sh.y), (mid_hip.x, mid_hip.y), (mid_knee.x, mid_knee.y))
            feature_conf["hip_extension"] = min(mid_sh.confidence, mid_hip.confidence, mid_knee.confidence)
            logger.debug("[BIOMECHANICS] hip_extension=%.1f°", hip_ext)
        elif mid_sh and mid_hip and mid_ankle and mid_sh.visible and mid_hip.visible and mid_ankle.visible:
            # Fallback to ankle if knees are locked straight
            hip_ext = calculate_angle_2d((mid_sh.x, mid_sh.y), (mid_hip.x, mid_hip.y), (mid_ankle.x, mid_ankle.y))
            feature_conf["hip_extension"] = min(mid_sh.confidence, mid_hip.confidence, mid_ankle.confidence)
            logger.debug("[BIOMECHANICS] hip_extension=%.1f° (fallback ankle)", hip_ext)
        else:
            unreliable.append("hip_extension")
            logger.debug("[BIOMECHANICS] hip_extension=unreliable")

        # 3. Knee Extension (mid_hip - mid_knee - mid_ankle)
        knee_ext: Optional[float] = None
        if mid_hip and mid_knee and mid_ankle and mid_hip.visible and mid_knee.visible and mid_ankle.visible:
            knee_ext = calculate_angle_2d((mid_hip.x, mid_hip.y), (mid_knee.x, mid_knee.y), (mid_ankle.x, mid_ankle.y))
            feature_conf["knee_extension"] = min(mid_hip.confidence, mid_knee.confidence, mid_ankle.confidence)
            logger.debug("[BIOMECHANICS] knee_extension=%.1f°", knee_ext)
        else:
            unreliable.append("knee_extension")
            logger.debug("[BIOMECHANICS] knee_extension=unreliable")

        # 4. Body Line Alignment (overall linearity shoulder - hip - leg endpoint)
        body_line: Optional[float] = None
        leg_endpoint = mid_ankle if (mid_ankle and mid_ankle.visible) else (mid_knee if (mid_knee and mid_knee.visible) else None)
        if mid_sh and mid_hip and leg_endpoint and mid_sh.visible and mid_hip.visible:
            body_line = calculate_angle_2d((mid_sh.x, mid_sh.y), (mid_hip.x, mid_hip.y), (leg_endpoint.x, leg_endpoint.y))
            feature_conf["body_line"] = min(mid_sh.confidence, mid_hip.confidence, leg_endpoint.confidence)
            logger.debug("[BIOMECHANICS] body_line=%.1f°", body_line)
        else:
            unreliable.append("body_line")
            logger.debug("[BIOMECHANICS] body_line=unreliable")

        # 5. Elbow Extension from original landmarks (active/visible arm)
        resolved_sh = get_point(kps.get("side_shoulder"))
        resolved_el = get_point(kps.get("side_elbow"))
        resolved_wr = get_point(kps.get("side_wrist"))
        if "side_elbow" in kps:
            side_elbow = kps["side_elbow"]
            logger.info(
                "[KEYPOINT_TRACE] request_id=%s stage=biomechanics_input joint=side_elbow "
                "x=%.2f y=%.2f conf=%.4f",
                rid, side_elbow.x, side_elbow.y, side_elbow.confidence
            )
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

        if elbow_ext is not None:
            logger.debug("[BIOMECHANICS] elbow_extension=%.1f°", elbow_ext)
        else:
            logger.debug("[BIOMECHANICS] elbow_extension=unreliable")

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

        t_bio_ms = (time.perf_counter() - _t_bio_start) * 1000.0
        logger.debug("[TIMING] request_id=%s stage=biomechanics_analysis duration_ms=%.2f", rid, t_bio_ms)

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
