import math
import logging
from typing import Dict, List, Optional, Tuple, Any
from app.schemas.analysis import KeypointData

logger = logging.getLogger(__name__)


def compute_bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Boxes format: [x1, y1, x2, y2]
    """
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    box_a_area = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    box_b_area = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])

    union_area = box_a_area + box_b_area - inter_area
    if union_area <= 1e-6:
        return 0.0
    return float(inter_area / union_area)


def compute_derived_landmarks(
    keypoints: Dict[str, KeypointData],
    conf_threshold: float = 0.25
) -> Dict[str, KeypointData]:
    """
    Calculate derived landmarks with propagated confidence:
    - shoulder_center
    - hip_center
    - wrist_center
    - ankle_center
    - body_center (midpoint of shoulder_center and hip_center)
    """
    derived: Dict[str, KeypointData] = {}

    def _make_center(
        name: str,
        k1_name: str,
        k2_name: str
    ) -> Optional[KeypointData]:
        k1 = keypoints.get(k1_name)
        k2 = keypoints.get(k2_name)

        has_k1 = k1 is not None and k1.visible
        has_k2 = k2 is not None and k2.visible

        if has_k1 and has_k2:
            assert k1 is not None and k2 is not None
            cx = (k1.x + k2.x) / 2.0
            cy = (k1.y + k2.y) / 2.0
            c_conf = (k1.confidence + k2.confidence) / 2.0
            return KeypointData(
                x=round(cx, 2),
                y=round(cy, 2),
                confidence=round(c_conf, 3),
                name=name,
                visible=c_conf >= conf_threshold
            )
        elif has_k1:
            assert k1 is not None
            return KeypointData(
                x=round(k1.x, 2),
                y=round(k1.y, 2),
                confidence=round(k1.confidence * 0.85, 3),
                name=name,
                visible=k1.confidence * 0.85 >= conf_threshold
            )
        elif has_k2:
            assert k2 is not None
            return KeypointData(
                x=round(k2.x, 2),
                y=round(k2.y, 2),
                confidence=round(k2.confidence * 0.85, 3),
                name=name,
                visible=k2.confidence * 0.85 >= conf_threshold
            )
        return None

    # 1. Shoulder center
    sc = _make_center("shoulder_center", "left_shoulder", "right_shoulder")
    if sc:
        derived["shoulder_center"] = sc

    # 2. Hip center
    hc = _make_center("hip_center", "left_hip", "right_hip")
    if hc:
        derived["hip_center"] = hc

    # 3. Wrist center
    wc = _make_center("wrist_center", "left_wrist", "right_wrist")
    if wc:
        derived["wrist_center"] = wc

    # 4. Ankle center
    ac = _make_center("ankle_center", "left_ankle", "right_ankle")
    if ac:
        derived["ankle_center"] = ac

    # 5. Body center
    if "shoulder_center" in derived and "hip_center" in derived:
        s_pt = derived["shoulder_center"]
        h_pt = derived["hip_center"]
        bc_x = (s_pt.x + h_pt.x) / 2.0
        bc_y = (s_pt.y + h_pt.y) / 2.0
        bc_conf = (s_pt.confidence + h_pt.confidence) / 2.0
        derived["body_center"] = KeypointData(
            x=round(bc_x, 2),
            y=round(bc_y, 2),
            confidence=round(bc_conf, 3),
            name="body_center",
            visible=bc_conf >= conf_threshold
        )

    return derived


class DetectedPersonCandidate:
    """Represents a detected person candidate in a single frame."""

    def __init__(
        self,
        box: List[float],
        confidence: float,
        keypoints: Dict[str, KeypointData]
    ):
        self.box = box
        self.confidence = confidence
        self.keypoints = keypoints
        self.width = max(1.0, box[2] - box[0])
        self.height = max(1.0, box[3] - box[1])
        self.area = self.width * self.height
        self.centroid = ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)
        self.diagonal = math.hypot(self.width, self.height)

        # Count visible core landmarks (shoulders, hips, knees, elbows)
        core_names = [
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_hip", "right_hip",
            "left_knee", "right_knee"
        ]
        visible_core = sum(1 for k in core_names if k in keypoints and keypoints[k].visible)
        self.core_completeness = visible_core / len(core_names)


class PersonTracker:
    """
    Consistent athlete tracker across frames.
    - Preserves athlete identity across multi-person scenes.
    - Avoids erratic switching to spectators or background people.
    - Handles temporary occlusion up to max_lost_frames.
    - Reacquires the athlete upon reappearance.
    """

    def __init__(self, max_lost_frames: int = 15):
        self.max_lost_frames = max_lost_frames
        self.track_id: int = 1
        self.is_tracking: bool = False
        self.last_box: Optional[List[float]] = None
        self.last_centroid: Optional[Tuple[float, float]] = None
        self.last_area: Optional[float] = None
        self.last_diagonal: Optional[float] = None
        self.last_keypoints: Optional[Dict[str, KeypointData]] = None
        self.consecutive_lost_frames: int = 0
        self.frames_tracked: int = 0

    def reset(self):
        """Reset tracking state."""
        self.is_tracking = False
        self.last_box = None
        self.last_centroid = None
        self.last_area = None
        self.last_diagonal = None
        self.last_keypoints = None
        self.consecutive_lost_frames = 0
        self.frames_tracked = 0

    def select_athlete(
        self,
        candidates: List[DetectedPersonCandidate]
    ) -> Optional[Tuple[DetectedPersonCandidate, bool]]:
        """
        Select the target athlete among detected candidates.
        Returns:
            (selected_candidate, is_interpolated_or_reacquired: bool) or None if no valid person.
        """
        if not candidates:
            if self.is_tracking:
                self.consecutive_lost_frames += 1
                if self.consecutive_lost_frames <= self.max_lost_frames and self.last_keypoints and self.last_box:
                    # Coast previous position with slightly reduced confidence (occlusion handling)
                    coasted_kps: Dict[str, KeypointData] = {}
                    decay = max(0.2, 1.0 - (self.consecutive_lost_frames * 0.05))
                    for name, kp in self.last_keypoints.items():
                        coasted_kps[name] = KeypointData(
                            x=kp.x,
                            y=kp.y,
                            confidence=round(kp.confidence * decay, 3),
                            name=name,
                            visible=bool(kp.confidence * decay >= 0.25)
                        )
                    coasted_cand = DetectedPersonCandidate(
                        box=list(self.last_box),
                        confidence=round(0.5 * decay, 3),
                        keypoints=coasted_kps
                    )
                    return coasted_cand, True
                else:
                    self.reset()
            return None

        # Case 1: Not yet tracking or lost beyond recovery -> Initial acquisition
        if not self.is_tracking or self.last_box is None:
            # Score candidates based on confidence, bounding box area (athlete in foreground), and core landmark completeness
            best_cand = max(
                candidates,
                key=lambda c: (c.confidence * 0.35 + c.core_completeness * 0.35 + min(1.0, c.area / 100000.0) * 0.30)
            )
            self._update_track(best_cand)
            return best_cand, False

        # Case 2: Actively tracking -> Match candidate closest to current track
        best_match_score = -1.0
        best_match_cand: Optional[DetectedPersonCandidate] = None

        assert self.last_box is not None and self.last_centroid is not None and self.last_diagonal is not None

        for cand in candidates:
            iou = compute_bbox_iou(self.last_box, cand.box)
            dist = math.hypot(cand.centroid[0] - self.last_centroid[0], cand.centroid[1] - self.last_centroid[1])
            norm_dist = dist / max(1.0, self.last_diagonal)
            proximity = max(0.0, 1.0 - norm_dist)

            area_ratio = min(cand.area, self.last_area or cand.area) / max(cand.area, self.last_area or cand.area, 1.0)

            # Composite match score [0.0 - 1.0]
            match_score = 0.45 * iou + 0.35 * proximity + 0.20 * area_ratio

            if match_score > best_match_score:
                best_match_score = match_score
                best_match_cand = cand

        # Threshold for continuing the track
        # Accept if either IoU > 0.15, or proximity is high (> 0.65), or overall match_score >= 0.30
        if best_match_cand is not None and best_match_score >= 0.25:
            reacquired = self.consecutive_lost_frames > 0
            self._update_track(best_match_cand)
            return best_match_cand, reacquired

        # No candidate matches the tracked athlete
        self.consecutive_lost_frames += 1
        if self.consecutive_lost_frames <= self.max_lost_frames and self.last_keypoints and self.last_box:
            decay = max(0.2, 1.0 - (self.consecutive_lost_frames * 0.05))
            coasted_kps = {
                name: KeypointData(
                    x=kp.x,
                    y=kp.y,
                    confidence=round(kp.confidence * decay, 3),
                    name=name,
                    visible=bool(kp.confidence * decay >= 0.25)
                )
                for name, kp in self.last_keypoints.items()
            }
            coasted_cand = DetectedPersonCandidate(
                box=list(self.last_box),
                confidence=round(0.5 * decay, 3),
                keypoints=coasted_kps
            )
            return coasted_cand, True

        # Lost tracking permanently; reacquire best new person candidate
        best_cand = max(
            candidates,
            key=lambda c: (c.confidence * 0.35 + c.core_completeness * 0.35 + min(1.0, c.area / 100000.0) * 0.30)
        )
        self._update_track(best_cand)
        return best_cand, True

    def _update_track(self, cand: DetectedPersonCandidate):
        self.is_tracking = True
        self.last_box = cand.box
        self.last_centroid = cand.centroid
        self.last_area = cand.area
        self.last_diagonal = cand.diagonal
        self.last_keypoints = cand.keypoints
        self.consecutive_lost_frames = 0
        self.frames_tracked += 1
