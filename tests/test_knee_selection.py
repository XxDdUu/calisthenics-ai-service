"""
Regression tests for the joint knee–ankle selection algorithm in
``app/biomechanics/side_view.resolve_side_view_chain``.

Covers:
  1. Correct knee detection (standard case).
  2. High-confidence but misplaced / implausible knee.
  3. Occluded knee hallucinated near the pelvis (collapsed).
  4. Two overlapping knees.
  5. One valid and one invalid knee candidate.
  6. Knee candidate whose corresponding ankle produces an implausible shank
     — the OTHER knee's valid ankle should win.
  7. Missing ankles.
  8. Inconsistent resolved-hip vs mid_hip coordinates after the fix.
  9. Different image resolutions (small and large scale).
"""

import math
import pytest
from app.biomechanics.side_view import (
    SideViewKeypointProcessor,
    resolve_side_view_chain,
    compute_midpoint,
)
from app.biomechanics.metrics import BiomechanicsEvaluator
from app.schemas.analysis import KeypointData, PoseKeypoints


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _kp(name: str, x: float, y: float, conf: float = 0.85) -> KeypointData:
    return KeypointData(x=x, y=y, confidence=conf, name=name, visible=True)


def make_horizontal_athlete(
    *,
    torso_len: float = 150.0,
    conf: float = 0.85,
    left_knee_offset: float = 130.0,   # dist from hip along +x
    right_knee_offset: float = 130.0,
    left_ankle_offset: float = 130.0,  # dist from knee along +x
    right_ankle_offset: float = 130.0,
    left_knee_conf: float | None = None,
    right_knee_conf: float | None = None,
    left_ankle_conf: float | None = None,
    right_ankle_conf: float | None = None,
    hip_y_spread: float = 10.0,        # |left_hip.y - right_hip.y|
    shoulder_x: float = 100.0,
) -> dict:
    """
    Build a minimal side-view keypoint dict for a horizontal planche athlete.

    The athlete's body runs along the X axis.  left_* joints are the "near"
    (visible) side; right_* joints are the "far" (potentially occluded) side.
    """
    lkc = left_knee_conf if left_knee_conf is not None else conf
    rkc = right_knee_conf if right_knee_conf is not None else conf
    lac = left_ankle_conf if left_ankle_conf is not None else conf
    rac = right_ankle_conf if right_ankle_conf is not None else conf

    # Hips at x = shoulder_x + torso_len; y split by hip_y_spread
    hip_x = shoulder_x + torso_len
    half_spread = hip_y_spread / 2.0

    left_knee_x  = hip_x + left_knee_offset
    right_knee_x = hip_x + right_knee_offset
    left_ankle_x  = left_knee_x  + left_ankle_offset
    right_ankle_x = right_knee_x + right_ankle_offset

    return {
        # Arm (both visible for clarity)
        "left_shoulder":  _kp("left_shoulder",  shoulder_x,  95.0, conf),
        "right_shoulder": _kp("right_shoulder", shoulder_x, 105.0, conf),
        "left_elbow":     _kp("left_elbow",     shoulder_x,  150.0, conf),
        "right_elbow":    _kp("right_elbow",    shoulder_x,  150.0, conf * 0.60),
        "left_wrist":     _kp("left_wrist",     shoulder_x,  200.0, conf),
        "right_wrist":    _kp("right_wrist",    shoulder_x,  200.0, conf * 0.60),
        # Hips
        "left_hip":  _kp("left_hip",  hip_x, 100.0 - half_spread, conf),
        "right_hip": _kp("right_hip", hip_x, 100.0 + half_spread, conf * 0.55),
        # Knees
        "left_knee":  _kp("left_knee",  left_knee_x,  100.0 - half_spread, lkc),
        "right_knee": _kp("right_knee", right_knee_x, 100.0 + half_spread, rkc),
        # Ankles
        "left_ankle":  _kp("left_ankle",  left_ankle_x,  100.0 - half_spread, lac),
        "right_ankle": _kp("right_ankle", right_ankle_x, 100.0 + half_spread, rac),
    }


# ---------------------------------------------------------------------------
# 1. Correct knee detection — standard case
# ---------------------------------------------------------------------------

def test_correct_knee_detection_standard():
    """
    Visible left knee at the expected femur length; right knee occluded but
    plausible.  Left should be selected (higher total confidence).
    """
    kps = make_horizontal_athlete(
        left_knee_conf=0.88, right_knee_conf=0.40,
        left_ankle_conf=0.85, right_ankle_conf=0.35,
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain, "Knee must be resolved when a valid candidate exists"
    # Left knee (high-confidence visible side) should win
    assert chain["knee"].y < 100.0, "Selected knee should be from the left (near) side"
    assert "ankle" in chain, "Ankle must follow the selected knee"


# ---------------------------------------------------------------------------
# 2. High-confidence but misplaced knee (thigh too long)
# ---------------------------------------------------------------------------

def test_high_confidence_misplaced_knee_rejected():
    """
    Left knee has very high confidence but lies too far from the hip
    (thigh > 1.60 × torso_len).  Right knee is plausible.
    """
    kps = make_horizontal_athlete(
        torso_len=150.0,
        left_knee_offset=260.0,   # 260/150 = 1.73 × torso_len — too long
        right_knee_offset=130.0,  # 130/150 = 0.87 × torso_len — plausible
        left_knee_conf=0.95,
        right_knee_conf=0.70,
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain, "Right knee should be selected as left is implausible"
    # Right knee sits at ~half_spread below y=100; check it's the "right" candidate
    assert chain["knee"].y > 100.0, "Selected knee should be from the right (plausible) side"


# ---------------------------------------------------------------------------
# 3. Occluded knee hallucinated near the pelvis (collapsed)
# ---------------------------------------------------------------------------

def test_collapsed_occluded_knee_hallucination_rejected():
    """
    Right knee is hallucinated near the pelvis: hip-to-knee distance ≈ 25 px
    which is 0.17 × torso_len (< 0.40 threshold) → must be rejected.
    """
    kps = make_horizontal_athlete(
        torso_len=150.0,
        left_knee_offset=130.0,
        right_knee_offset=25.0,   # collapsed — hallucinated near pelvis
        left_knee_conf=0.80,
        right_knee_conf=0.72,     # moderate confidence: would win on confidence alone
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain
    # Collapsed right knee must NOT be selected even with moderate confidence
    assert chain["knee"].y < 100.0, "Left (valid) knee must be chosen, not the collapsed right"


# ---------------------------------------------------------------------------
# 4. Two overlapping knees (both within 25 px of each other)
# ---------------------------------------------------------------------------

def test_overlapping_knees_selects_by_joint_score():
    """
    In a true side view the two knees overlap almost perfectly.
    Selection should use joint score (knee + ankle quality), not flip-flop.
    """
    kps = make_horizontal_athlete(
        torso_len=150.0,
        left_knee_offset=130.0,
        right_knee_offset=130.0,
        hip_y_spread=4.0,          # near-perfect profile view
        left_knee_conf=0.88,
        right_knee_conf=0.82,
        left_ankle_conf=0.85,
        right_ankle_conf=0.80,
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain, "A knee must be resolved when both are plausible"
    # Exactly ONE knee is selected — no averaging
    # Confidence-weighted score should favour left (higher conf)
    assert chain["knee"].y <= 100.0, "Left (higher-conf) knee expected when overlap"


# ---------------------------------------------------------------------------
# 5. One valid and one invalid (too short thigh) knee
# ---------------------------------------------------------------------------

def test_one_valid_one_too_short_knee():
    """
    Left knee is at the right distance; right knee thigh = 0.30 × torso_len
    (< 0.40 threshold).  Only left should appear in the chain.
    """
    kps = make_horizontal_athlete(
        torso_len=150.0,
        left_knee_offset=130.0,
        right_knee_offset=45.0,    # 45/150 = 0.30 — below the 0.40 floor
        left_knee_conf=0.80,
        right_knee_conf=0.85,      # right has higher confidence but implausible thigh
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain
    assert chain["knee"].y < 100.0, "Left knee (plausible) must win over high-conf implausible right"


# ---------------------------------------------------------------------------
# 6. Knee candidate whose same-side ankle is implausible — other side wins
# ---------------------------------------------------------------------------

def test_knee_with_implausible_ankle_loses_to_other_side():
    """
    Left knee is slightly better on the thigh metric but its left ankle
    is placed too far from it (shank > 1.60 × torso).  Right knee has a
    slightly lower thigh score but a valid, plausible right ankle.
    The joint (knee + ankle) score should favour the right side.
    """
    torso = 150.0
    hip_x = 100.0 + torso
    kps = make_horizontal_athlete(
        torso_len=torso,
        left_knee_offset=130.0,     # thigh = 0.87t — ideal
        right_knee_offset=115.0,    # thigh = 0.77t — still plausible
        left_ankle_offset=310.0,    # shank = 310/150 = 2.07t — too long!
        right_ankle_offset=120.0,   # shank = 120/150 = 0.80t — plausible
        left_knee_conf=0.88,
        right_knee_conf=0.82,
        left_ankle_conf=0.85,
        right_ankle_conf=0.80,
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain
    # Right side has valid ankle chain; left ankle is implausible
    # The joint score should select right knee + right ankle
    assert chain["knee"].y > 100.0, "Right knee (valid ankle chain) should win"
    assert "ankle" in chain, "Right ankle should be present"
    assert chain["ankle"].y > 100.0, "Ankle should be from the right side"


# ---------------------------------------------------------------------------
# 7. Missing ankles — knee must still be resolved
# ---------------------------------------------------------------------------

def test_knee_resolved_when_both_ankles_missing():
    """
    Both ankles are below the confidence threshold.  The knee with the
    better thigh score should still be selected; ankle should be absent.
    """
    kps = make_horizontal_athlete(
        left_knee_conf=0.82,
        right_knee_conf=0.40,
        left_ankle_conf=0.10,   # below threshold
        right_ankle_conf=0.12,  # below threshold
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain, "Knee must be resolved even without a valid ankle"
    assert "ankle" not in chain, "No ankle should appear when both are below threshold"

    # Downstream processor should gracefully record ankle as missing
    processed = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)
    assert "mid_knee" in processed.midpoints
    assert "mid_ankle" not in processed.midpoints
    assert "mid_ankle" in processed.missing_landmarks


def test_knee_resolved_when_one_ankle_missing():
    """
    Same-side ankle is missing; opposite ankle is also missing.
    Knee still resolved; ankle absent.
    """
    kps = make_horizontal_athlete(
        left_knee_conf=0.85,
        right_knee_conf=0.40,
        left_ankle_conf=0.15,   # below threshold — same side as selected knee
        right_ankle_conf=0.15,  # below threshold
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain
    assert "ankle" not in chain


# ---------------------------------------------------------------------------
# 8. Inconsistent hip: resolved chain hip == mid_hip after fix
# ---------------------------------------------------------------------------

def test_mid_hip_equals_resolved_chain_hip():
    """
    After the fix, mid_hip in process_keypoints must equal chain['hip']
    from resolve_side_view_chain.  They must not diverge even when left
    and right hips have different confidences that trigger occlusion snapping.
    """
    kps = make_horizontal_athlete(
        conf=0.9,
        hip_y_spread=10.0,  # non-trivial offset between L and R hips
    )
    # Give left hip higher confidence so the chain snaps to it
    kps["left_hip"] = _kp("left_hip", kps["left_hip"].x, kps["left_hip"].y, conf=0.90)
    kps["right_hip"] = _kp("right_hip", kps["right_hip"].x, kps["right_hip"].y, conf=0.45)

    chain = resolve_side_view_chain(kps, threshold=0.25)
    processed = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)

    assert "hip" in chain
    assert "mid_hip" in processed.midpoints

    chain_hip = chain["hip"]
    mid_hip = processed.midpoints["mid_hip"]

    assert chain_hip.x == mid_hip.x, (
        f"chain hip x ({chain_hip.x}) must equal mid_hip x ({mid_hip.x})"
    )
    assert chain_hip.y == mid_hip.y, (
        f"chain hip y ({chain_hip.y}) must equal mid_hip y ({mid_hip.y})"
    )


def test_mid_hip_equals_resolved_chain_hip_symmetric_case():
    """
    When both hips are symmetric (equal confidence, small distance), the
    resolved chain picks the primary-arm side. mid_hip must match.
    """
    kps = make_horizontal_athlete(conf=0.9, hip_y_spread=10.0)
    # Both hips identical confidence — chain falls back to arm-side (left)
    chain = resolve_side_view_chain(kps, threshold=0.25)
    processed = SideViewKeypointProcessor.process_keypoints(kps, threshold=0.25)

    assert chain["hip"].x == pytest.approx(processed.midpoints["mid_hip"].x)
    assert chain["hip"].y == pytest.approx(processed.midpoints["mid_hip"].y)


# ---------------------------------------------------------------------------
# 9. Different image resolutions
# ---------------------------------------------------------------------------

def test_small_resolution_knee_selection():
    """
    320×240 equivalent (torso ~50 px).  Geometric thresholds are body-normalised
    so the algorithm must still work at low resolution.
    """
    kps = make_horizontal_athlete(
        torso_len=50.0,
        shoulder_x=40.0,
        left_knee_offset=43.0,    # 43/50 = 0.86 — plausible
        right_knee_offset=8.0,    # 8/50  = 0.16 — collapsed hallucination
        left_ankle_offset=43.0,
        right_ankle_offset=43.0,
        left_knee_conf=0.80,
        right_knee_conf=0.75,
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain
    # Left knee is plausible, right is collapsed
    assert chain["knee"].y < 100.0, "Left (valid) knee should be selected at low resolution"


def test_large_resolution_knee_selection():
    """
    4K equivalent (torso ~600 px).  Same normalised ratios — result must match
    the standard resolution behaviour.
    """
    kps = make_horizontal_athlete(
        torso_len=600.0,
        shoulder_x=400.0,
        left_knee_offset=520.0,    # 520/600 = 0.87 — plausible
        right_knee_offset=80.0,    # 80/600  = 0.13 — collapsed
        left_ankle_offset=520.0,
        right_ankle_offset=520.0,
        left_knee_conf=0.80,
        right_knee_conf=0.75,
    )
    chain = resolve_side_view_chain(kps, threshold=0.25)

    assert "knee" in chain
    assert chain["knee"].y < 100.0, "Left knee should be selected at large resolution"


# ---------------------------------------------------------------------------
# Extra: verify front-view behavior is not regressed
# ---------------------------------------------------------------------------

def test_front_view_evaluation_not_affected():
    """
    Calling BiomechanicsEvaluator with side_view=False must not use
    resolve_side_view_chain at all, and must return resolved_landmarks=None.
    """
    kps = make_horizontal_athlete()
    pose_kp = PoseKeypoints(keypoints=kps, bounding_box=[0, 0, 800, 400], detection_confidence=0.9)
    profile = BiomechanicsEvaluator.evaluate(pose_kp, side_view=False)

    assert profile is not None
    assert profile.side_view is False
    assert profile.resolved_landmarks is None
    assert profile.midpoints is None
    assert profile.joint_angles.primary_knee_deg is not None
