import logging
import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from app.utils.pose_refinement import refine_wrists
from app.biomechanics.side_view import resolve_side_view_chain

from app.core.config import settings
from app.schemas.analysis import KeypointData, PlancheAnalysisResult, PoseKeypoints

logger = logging.getLogger(__name__)

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Keypoints to suppress in side-view (face area is occluded / irrelevant)
FACE_KEYPOINTS = {"nose", "left_eye", "right_eye", "left_ear", "right_ear"}

# Side-view: only draw the dominant (visible) limb chain
LEFT_SIDE_CONNECTIONS = [
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_shoulder", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
]

RIGHT_SIDE_CONNECTIONS = [
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_shoulder", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Limb keypoints used for dominant-side detection
_LEFT_LIMB_KPS = ["left_shoulder", "left_elbow", "left_wrist", "left_hip", "left_knee", "left_ankle"]
_RIGHT_LIMB_KPS = ["right_shoulder", "right_elbow", "right_wrist", "right_hip", "right_knee", "right_ankle"]



def _resolve_model_path(model_path: str) -> str:
    """Resolve model path to exist locally or fall back to model filename for Ultralytics download."""
    if not model_path:
        model_path = settings.MODEL_PATH

    if os.path.isabs(model_path) and os.path.exists(model_path):
        return model_path

    if os.path.exists(model_path):
        return os.path.abspath(model_path)

    # Check in models/ directory relative to project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    models_dir_file = os.path.join(project_root, "models", os.path.basename(model_path))
    if os.path.exists(models_dir_file):
        return models_dir_file

    return model_path


class YOLOPoseEstimator:
    """Wrapper around Ultralytics YOLO-Pose model for keypoint extraction."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self.device = device or settings.DEVICE
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD
        self.keypoint_conf_threshold = settings.KEYPOINT_CONF_THRESHOLD
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize YOLO model and verify it is a pose estimation model."""
        resolved_path = _resolve_model_path(self.model_path)
        logger.info(f"Loading YOLO-Pose model from '{resolved_path}' on device '{self.device}'...")
        try:
            from ultralytics import YOLO
            model = YOLO(resolved_path)
            task = getattr(model, "task", None)
            if task != "pose":
                err_msg = (
                    f"Loaded YOLO model from '{resolved_path}' has task '{task}', expected 'pose'. "
                    f"Verify that MODEL_PATH points to a YOLO pose model (e.g., yolov8n-pose.pt)."
                )
                logger.error(err_msg)
                self.model = None
                raise ValueError(err_msg)

            self.model = model
            logger.info(f"Successfully loaded YOLO-Pose model (task='{task}') from '{resolved_path}'")
        except Exception as e:
            logger.error(f"Failed to load YOLO-Pose model from '{resolved_path}': {e}", exc_info=True)
            self.model = None
            raise RuntimeError(f"YOLO-Pose model initialization failed for '{resolved_path}': {str(e)}") from e

    def estimate_pose(self, image_np: np.ndarray) -> Tuple[bool, Optional[PoseKeypoints]]:
        """
        Estimate pose keypoints from an RGB/BGR numpy image array.
        Returns (success: bool, PoseKeypoints or None).
        """
        if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
            logger.warning("Empty or invalid image array provided to estimate_pose.")
            return False, None

        if self.model is None:
            logger.error("YOLO model is not initialized.")
            return False, None

        try:
            results = self.model(
                image_np,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )

            if results is None or len(results) == 0:
                logger.info("YOLO inference returned no results object.")
                return False, None

            res = results[0]
            boxes = getattr(res, "boxes", None)
            keypoints = getattr(res, "keypoints", None)

            num_boxes = len(boxes) if boxes is not None else 0
            has_keypoints = (
                "yes"
                if keypoints is not None and getattr(keypoints, "data", None) is not None and len(keypoints.data) > 0
                else "no"
            )
            logger.debug(f"YOLO inference: boxes={num_boxes}, keypoints={has_keypoints}")

            if boxes is None or len(boxes) == 0:
                logger.info("YOLO inference: no bounding boxes detected.")
                return False, None

            if keypoints is None or getattr(keypoints, "data", None) is None or len(keypoints.data) == 0:
                logger.info("YOLO inference: boxes detected, but keypoints data is empty.")
                return False, None

            conf_tensor = getattr(boxes, "conf", None)
            if conf_tensor is None or len(conf_tensor) == 0:
                logger.warning("Bounding boxes missing confidence tensor.")
                return False, None

            confs = conf_tensor.cpu().numpy()
            best_idx = int(np.argmax(confs))
            best_conf = float(confs[best_idx])

            kp_data_tensor = keypoints.data
            if best_idx >= len(kp_data_tensor):
                logger.warning(f"Selected index {best_idx} out of bounds for keypoints data of length {len(kp_data_tensor)}.")
                return False, None

            kp_array = kp_data_tensor[best_idx].cpu().numpy()
            kp_array = refine_wrists(kp_array)
            xyxy_tensor = getattr(boxes, "xyxy", None)
            if xyxy_tensor is None or best_idx >= len(xyxy_tensor):
                logger.warning("Bounding boxes missing xyxy coordinates.")
                return False, None

            box_tensor = xyxy_tensor[best_idx].cpu().numpy()

            logger.debug(
                f"Selected person index={best_idx} with confidence={best_conf:.4f}. Keypoint shape={kp_array.shape}"
            )

            if kp_array.ndim != 2 or kp_array.shape[0] < 17:
                logger.warning(f"Keypoint tensor shape {kp_array.shape} invalid, expected at least (17, 3).")
                return False, None

            keypoints_dict: Dict[str, KeypointData] = {}
            for idx, kp_name in enumerate(COCO_KEYPOINTS):
                x = float(kp_array[idx][0])
                y = float(kp_array[idx][1])
                conf = float(kp_array[idx][2]) if kp_array.shape[1] >= 3 else best_conf
                visible = conf >= self.keypoint_conf_threshold
                keypoints_dict[kp_name] = KeypointData(
                    x=x, y=y, confidence=conf, name=kp_name, visible=visible
                )

            pose_kp = PoseKeypoints(
                keypoints=keypoints_dict,
                bounding_box=[float(box_tensor[0]), float(box_tensor[1]), float(box_tensor[2]), float(box_tensor[3])],
                detection_confidence=best_conf
            )
            return True, pose_kp

        except Exception as e:
            logger.exception("Unexpected exception during YOLO pose estimation inference.")
            return False, None

    def _get_dominant_side(self, kps: Dict[str, "KeypointData"]) -> str:
        """Return 'left' or 'right' based on total limb keypoint confidence."""
        l_conf = sum(kps[k].confidence for k in _LEFT_LIMB_KPS if k in kps and kps[k].visible)
        r_conf = sum(kps[k].confidence for k in _RIGHT_LIMB_KPS if k in kps and kps[k].visible)
        return "left" if l_conf >= r_conf else "right"

    def _snap_midpoint_px(
        self,
        kps: Dict[str, "KeypointData"],
        left_name: str,
        right_name: str,
    ) -> Optional[Tuple[int, int]]:
        """
        Compute the on-screen midpoint coordinate applying the same occlusion-aware
        snapping logic as compute_midpoint() so the rendered body line matches the
        biomechanics calculation.
        """
        import math as _math
        kp_l = kps.get(left_name)
        kp_r = kps.get(right_name)
        l_ok = kp_l is not None and kp_l.visible
        r_ok = kp_r is not None and kp_r.visible

        if not l_ok and not r_ok:
            return None
        if l_ok and not r_ok:
            assert kp_l is not None
            return (int(kp_l.x), int(kp_l.y))
        if r_ok and not l_ok:
            assert kp_r is not None
            return (int(kp_r.x), int(kp_r.y))

        assert kp_l is not None and kp_r is not None
        conf_delta = abs(kp_l.confidence - kp_r.confidence)
        dist = _math.sqrt((kp_l.x - kp_r.x) ** 2 + (kp_l.y - kp_r.y) ** 2)

        if conf_delta > 0.20 or dist > 80.0:
            dominant = kp_l if kp_l.confidence >= kp_r.confidence else kp_r
            return (int(dominant.x), int(dominant.y))

        return (int((kp_l.x + kp_r.x) / 2), int((kp_l.y + kp_r.y) / 2))

    def draw_skeleton(
        self,
        image_np: np.ndarray,
        pose_kp: PoseKeypoints,
        form_score: Optional[float] = None,
        progression: Optional[str] = None,
        errors: Optional[List[str]] = None,
        analysis: Optional[PlancheAnalysisResult] = None,
    ) -> np.ndarray:
        """Draw pose and evaluator values as a debug panel.

        ``analysis`` is preferred. Legacy scalar arguments remain supported for
        callers that only need the skeleton/HUD. This renderer never computes
        biomechanics; it only formats values already present in the result.

        Side-view mode:
        - Only draws the dominant limb chain (no cross-body connections).
        - Face keypoints (nose, eyes, ears) are always suppressed.
        - Midpoint body-line uses occlusion-aware snapping (same as compute_midpoint).
        """
        annotated = image_np.copy()
        if pose_kp is None or not pose_kp.keypoints:
            return annotated

        kps = pose_kp.keypoints
        is_side_view = analysis is not None and getattr(analysis, "side_view", False)

        if is_side_view:
            # Resolve arm and leg independently. A side-view skeleton must not
            # discard a good leg merely because the opposite arm wins confidence.
            chain = resolve_side_view_chain(kps, threshold=0.25)
            for a, b in (("shoulder", "elbow"), ("elbow", "wrist"),
                         ("hip", "knee"), ("knee", "ankle")):
                kp1, kp2 = chain.get(a), chain.get(b)
                if kp1 and kp2 and kp1.visible and kp2.visible:
                    cv2.line(annotated, (int(kp1.x), int(kp1.y)), (int(kp2.x), int(kp2.y)),
                             (0, 255, 127), 3, cv2.LINE_AA)
            for kp in chain.values():
                center = (int(kp.x), int(kp.y))
                cv2.circle(annotated, center, 6, (0, 215, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, center, 8, (255, 255, 255), 1, cv2.LINE_AA)

            # The resolved chain is also the rendered body axis, so metrics and
            # annotations cannot disagree about which ankle/knee was selected.
            mid_pts = {}
            for name in ("shoulder", "hip", "knee", "ankle"):
                kp = chain.get(name)
                if kp:
                    mid_pts[f"mid_{name}"] = (int(kp.x), int(kp.y))
            for a, b in (("mid_shoulder", "mid_hip"), ("mid_hip", "mid_knee"),
                         ("mid_knee", "mid_ankle")):
                if a in mid_pts and b in mid_pts:
                    cv2.line(annotated, mid_pts[a], mid_pts[b], (255, 255, 0), 4, cv2.LINE_AA)
            for name, pt in mid_pts.items():
                cv2.circle(annotated, pt, 8, (255, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(annotated, pt, 11, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, name.replace("mid_", ""), (pt[0] + 12, pt[1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1, cv2.LINE_AA)

        else:
            # Standard mode: draw all SKELETON_CONNECTIONS, suppress face keypoints circles
            for kp1_name, kp2_name in SKELETON_CONNECTIONS:
                kp1 = kps.get(kp1_name)
                kp2 = kps.get(kp2_name)
                if kp1 and kp2 and kp1.visible and kp2.visible:
                    cv2.line(
                        annotated, (int(kp1.x), int(kp1.y)), (int(kp2.x), int(kp2.y)),
                        (0, 255, 127), 3, cv2.LINE_AA
                    )

            for kp_name, kp in kps.items():
                if kp.visible and kp_name not in FACE_KEYPOINTS:
                    center = (int(kp.x), int(kp.y))
                    cv2.circle(annotated, center, 6, (0, 215, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, center, 8, (255, 255, 255), 1, cv2.LINE_AA)

        if analysis is not None:
            draw_analysis_panel(annotated, _analysis_panel_data(analysis))
        elif form_score is not None or progression is not None:
            draw_analysis_panel(
                annotated, _legacy_analysis_data(form_score, progression, errors),
                show_debug_sections=False
            )
        return annotated



_STATUS_COLORS = {
    "PASS": (0, 230, 118),
    "GOOD": (0, 230, 118),
    "NEEDS_IMPROVEMENT": (0, 191, 255),
    "FAIL": (50, 50, 255),
}
_STATUS_LABELS = {
    "PASS": "OK",
    "GOOD": "OK",
    "NEEDS_IMPROVEMENT": "Warning",
    "FAIL": "Error",
}


def _legacy_analysis_data(form_score, progression, errors) -> dict:
    return {"form_score": form_score, "progression": progression, "issues": errors or [], "metrics": {}}


def _analysis_panel_data(analysis: PlancheAnalysisResult) -> dict:
    return {
        "form_score": analysis.form_score,
        "progression": analysis.detected_progression.value,
        "issues": analysis.form_errors,
        "metrics": analysis.metrics,
        "alignment": analysis.alignment,
        "side_view": getattr(analysis, "side_view", False),
    }


def _metric_value_text(metric_key: str, value: float) -> str:
    """Format an existing evaluator value without changing its meaning."""
    return f"{value:.2f}" if metric_key == "leg_spread" else f"{value:.1f}°"


def _alignment_value_text(metric_key: str, value) -> str:
    if metric_key == "center_of_mass_over_hands":
        return "Yes" if value else "No"
    if metric_key == "center_of_mass_offset_px":
        return f"{value:.1f} px"
    return f"{value:.1f}%"


def _metric_name(metric_key: str, detail) -> str:
    names = {
        "elbow_extension": "Elbow Extension",
        "shoulder_lean": "Shoulder Lean",
        "body_tilt": "Body Tilt",
        "hip_angle": "Hip Angle / Body Line",
        "knee_angle": "Knee Extension / Tuck",
        "leg_spread": "Leg Spread Ratio",
    }
    return names.get(metric_key, detail.name)


def _text_size(text: str, scale: float, thickness: int) -> tuple[int, int]:
    return cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]


def _draw_text(image: np.ndarray, text: str, origin: tuple[int, int], scale: float, color, thickness: int = 1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_section_title(image: np.ndarray, title: str, x: int, y: int, width: int) -> int:
    _draw_text(image, title, (x, y), 0.48, (148, 163, 184), 1)
    cv2.line(image, (x, y + 7), (x + width, y + 7), (71, 85, 105), 1, cv2.LINE_AA)
    return y + 25


def draw_progression(image: np.ndarray, analysis, x: int, y: int, width: int) -> int:
    progression = analysis.get("progression")
    score = analysis.get("form_score")
    side_view = analysis.get("side_view", False)
    if progression:
        _draw_text(image, "Progression", (x, y), 0.53, (226, 232, 240), 1)
        _draw_text(image, str(progression), (x + int(width * 0.40), y), 0.53, (255, 255, 255), 1)
        y += 22
    mode_text = "Side-View" if side_view else "Standard"
    _draw_text(image, "Mode", (x, y), 0.48, (148, 163, 184), 1)
    _draw_text(image, mode_text, (x + int(width * 0.40), y), 0.48, (203, 213, 225), 1)
    y += 25
    if score is not None:
        color = (0, 230, 118) if score >= 80 else (0, 191, 255) if score >= 60 else (50, 50, 255)
        _draw_text(image, "Form Score", (x, y), 0.53, (226, 232, 240), 1)
        _draw_text(image, f"{score:.1f}%", (x + int(width * 0.40), y), 0.53, color, 2)
        y += 25
    return y


def draw_metrics(image: np.ndarray, analysis, x: int, y: int, width: int) -> int:
    metrics = analysis.get("metrics") or {}
    if not metrics:
        return y
    y = _draw_section_title(image, "BIOMECHANICS", x, y, width)
    for key, detail in metrics.items():
        label = _metric_name(key, detail)
        value = _metric_value_text(key, detail.measured_value)
        _draw_text(image, label, (x, y), 0.43, (226, 232, 240), 1)
        value_width = _text_size(value, 0.43, 1)[0]
        _draw_text(image, value, (x + width - value_width, y), 0.43, (255, 255, 255), 1)
        y += 21

    alignment = analysis.get("alignment")
    alignment_rows = (
        ("body_straightness_score", "Body Straightness"),
        ("scapular_protraction_score", "Scapular Protraction"),
        ("horizontal_level_score", "Horizontal Level"),
        ("center_of_mass_over_hands", "CoM Over Hands"),
        ("center_of_mass_offset_px", "CoM Offset"),
    )
    if alignment is not None:
        y = _draw_section_title(image, "ALIGNMENT", x, y + 2, width)
        for key, label in alignment_rows:
            value = getattr(alignment, key, None)
            if value is None:
                continue
            value_text = _alignment_value_text(key, value)
            _draw_text(image, label, (x, y), 0.43, (226, 232, 240), 1)
            value_width = _text_size(value_text, 0.43, 1)[0]
            _draw_text(image, value_text, (x + width - value_width, y), 0.43, (255, 255, 255), 1)
            y += 21
    return y + 4


def draw_biomechanics_form(image: np.ndarray, analysis, x: int, y: int, width: int) -> int:
    metrics = analysis.get("metrics") or {}
    if not metrics:
        return y
    y = _draw_section_title(image, "BIOMECHANICS & FORM", x, y, width)
    for key, detail in metrics.items():
        status = getattr(detail.status, "value", str(detail.status))
        color = _STATUS_COLORS.get(status, (255, 255, 255))
        icon = "OK" if status in ("PASS", "GOOD") else "!" if status == "NEEDS_IMPROVEMENT" else "X"
        label = _metric_name(key, detail)
        value = _metric_value_text(key, detail.measured_value)
        status_label = _STATUS_LABELS.get(status, status.title())
        _draw_text(image, icon, (x, y), 0.43, color, 2)
        _draw_text(image, label, (x + 18, y), 0.40, (226, 232, 240), 1)
        sw = _text_size(status_label, 0.40, 1)[0]; vw = _text_size(value, 0.40, 1)[0]
        _draw_text(image, value, (x + width - sw - vw - 12, y), 0.40, color, 1)
        _draw_text(image, status_label, (x + width - sw, y), 0.40, color, 1)
        y += 21
    return y + 4


def draw_form_checks(image: np.ndarray, analysis, x: int, y: int, width: int) -> int:
    metrics = analysis.get("metrics") or {}
    if not metrics:
        return y
    y = _draw_section_title(image, "FORM CHECK", x, y, width)
    for key, detail in metrics.items():
        status = getattr(detail.status, "value", str(detail.status))
        color = _STATUS_COLORS.get(status, (255, 255, 255))
        status_label = _STATUS_LABELS.get(status, status.title())
        label = _metric_name(key, detail)
        value = _metric_value_text(key, detail.measured_value)
        icon = "OK" if status in ("PASS", "GOOD") else "!" if status == "NEEDS_IMPROVEMENT" else "X"
        _draw_text(image, icon, (x, y), 0.52, color, 2)
        _draw_text(image, label, (x + 18, y), 0.40, (226, 232, 240), 1)
        value_width = _text_size(value, 0.40, 1)[0]
        status_width = _text_size(status_label, 0.40, 1)[0]
        _draw_text(image, value, (x + width - value_width - status_width - 18, y), 0.40, (203, 213, 225), 1)
        _draw_text(image, status_label, (x + width - status_width, y), 0.40, color, 1)
        y += 21
    return y + 4


def draw_issues(image: np.ndarray, analysis, x: int, y: int, width: int) -> int:
    issues = (analysis.get("issues") or [])[:3]
    if not issues:
        return y
    y = _draw_section_title(image, "ISSUES", x, y, width)
    for issue in issues:
        text = f"- {issue}"
        max_chars = max(20, int(width / 8))
        if len(text) > max_chars:
            text = text[:max_chars - 1] + "…"
        _draw_text(image, text, (x, y), 0.39, (96, 165, 250), 1)
        y += 20
    return y


def draw_analysis_panel(image: np.ndarray, analysis, show_debug_sections: bool = True) -> np.ndarray:
    """Render a compact, overflow-safe panel from existing analysis/evaluator values.

    Panel width is capped at 28 % of the frame width (down from 38 %) and all
    font scales / line spacings have been reduced and the redundant FORM CHECK
    section is merged into BIOMECHANICS & FORM so the panel remains within frame.
    Panel height is computed from the actual number of rows that will be rendered,
    not a fixed estimate.
    """
    h, w = image.shape[:2]
    margin = max(10, int(min(w, h) * 0.025))

    # Narrower panel — leaves more of the athlete visible
    panel_width = min(max(240, int(w * 0.28)), w - 2 * margin)
    x = margin + 10
    inner_width = panel_width - 20

    metrics = analysis.get("metrics") or {}
    alignment = analysis.get("alignment")
    issues = (analysis.get("issues") or [])[:3]

    # Pre-calculate content height so the panel fits exactly
    ROW_H = 21           # matches the renderer's row spacing
    SECTION_GAP = 27     # title/divider plus the following section offset
    HEADER_H = 22        # title "FORM ANALYSIS"
    PROG_H = 22 if analysis.get("progression") else 0
    MODE_H = 25          # Mode row
    SCORE_H = 25 if analysis.get("form_score") is not None else 0

    content_h = HEADER_H + PROG_H + MODE_H + SCORE_H + 4  # header block

    if show_debug_sections and metrics:
        content_h += SECTION_GAP + len(metrics) * ROW_H  # BIOMECHANICS & FORM
        if alignment is not None:
            # count non-None alignment fields
            alignment_rows = (
                "body_straightness_score", "scapular_protraction_score",
                "horizontal_level_score", "center_of_mass_over_hands",
                "center_of_mass_offset_px",
            )
            n_align = sum(1 for k in alignment_rows if getattr(alignment, k, None) is not None)
            content_h += SECTION_GAP + n_align * ROW_H  # ALIGNMENT

    if issues:
        content_h += SECTION_GAP + len(issues) * ROW_H  # ISSUES

    panel_height = min(h - 2 * margin, content_h + 12)  # bounded by image margins

    panel_x1, panel_y1 = margin, margin
    panel_x2 = min(w - margin, panel_x1 + panel_width)
    panel_y2 = min(h - margin, panel_y1 + panel_height)

    overlay = image.copy()
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    cv2.rectangle(image, (panel_x1, panel_y1), (panel_x2, panel_y2), (71, 85, 105), 1)

    y = panel_y1 + 18
    _draw_text(image, "FORM ANALYSIS", (x, y), 0.50, (255, 255, 255), 2)
    y += 22
    y = draw_progression(image, analysis, x, y, inner_width)
    if show_debug_sections:
        y = draw_biomechanics_form(image, analysis, x, y + 4, inner_width)
    draw_issues(image, analysis, x, y + 2, inner_width)
    return image

