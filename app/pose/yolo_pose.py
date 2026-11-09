import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from app.utils.pose_refinement import refine_wrists
from app.biomechanics.side_view import resolve_side_view_chain
from app.pose.tracking import (
    DetectedPersonCandidate,
    PersonTracker,
    compute_derived_landmarks,
)

from app.core.config import settings
from app.core.logging_context import get_request_id
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
        self.resolved_model_path = _resolve_model_path(self.model_path)
        self.device = device or settings.DEVICE
        self.conf_threshold = settings.CONFIDENCE_THRESHOLD
        self.keypoint_conf_threshold = settings.KEYPOINT_CONF_THRESHOLD
        self.model = None
        self._init_model()

    def _init_model(self):
        """Initialize YOLO model and verify it is a pose estimation model."""
        resolved_path = self.resolved_model_path
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

            add_callback = getattr(model, "add_callback", None)
            if callable(add_callback):
                add_callback("on_predict_start", self._install_preprocess_trace)

            self.model = model
            logger.info(f"Successfully loaded YOLO-Pose model (task='{task}') from '{resolved_path}'")
        except Exception as e:
            logger.error(f"Failed to load YOLO-Pose model from '{resolved_path}': {e}", exc_info=True)
            self.model = None
            raise RuntimeError(f"YOLO-Pose model initialization failed for '{resolved_path}': {str(e)}") from e

    def _install_preprocess_trace(self, predictor) -> None:
        """Log the actual letterboxed tensor geometry without changing its pixels."""
        if getattr(predictor, "_pose_preprocess_trace_installed", False):
            return

        preprocess = predictor.preprocess

        def traced_preprocess(images):
            model_input = preprocess(images)
            batch = getattr(predictor, "batch", None)
            original_images = batch[1] if batch is not None and len(batch) > 1 else ()
            input_h, input_w = model_input.shape[-2:]
            rid = get_request_id()

            for image_index, original in enumerate(original_images):
                original_h, original_w = original.shape[:2]
                scale = min(input_h / original_h, input_w / original_w)
                resized_w = round(original_w * scale)
                resized_h = round(original_h * scale)
                pad_x = input_w - resized_w
                pad_y = input_h - resized_h
                pad_left = round(pad_x / 2.0 - 0.1)
                pad_top = round(pad_y / 2.0 - 0.1)
                logger.info(
                    "[YOLO] request_id=%s stage=model_preprocess image_index=%d "
                    "original_shape=%dx%d resized_shape=%dx%d model_input_shape=%dx%d "
                    "scale=%.6f padding_left=%d padding_top=%d padding_right=%d padding_bottom=%d",
                    rid, image_index, original_w, original_h, resized_w, resized_h,
                    input_w, input_h, scale, pad_left, pad_top,
                    pad_x - pad_left, pad_y - pad_top
                )
            return model_input

        predictor.preprocess = traced_preprocess
        predictor._pose_preprocess_trace_installed = True

        construct_result = getattr(predictor, "construct_result", None)
        keypoint_shape = getattr(getattr(predictor, "model", None), "kpt_shape", None)
        if callable(construct_result) and keypoint_shape and not getattr(predictor, "_pose_coordinate_trace_installed", False):
            def traced_construct_result(pred, img, orig_img, img_path, *args, **kwargs):
                raw_keypoints = pred[:, 6:].reshape((pred.shape[0], *keypoint_shape))
                for person_index, person_keypoints in enumerate(raw_keypoints):
                    for keypoint_index in (7, 8, 13, 14):
                        point = person_keypoints[keypoint_index]
                        confidence = float(point[2]) if point.shape[0] >= 3 else None
                        logger.info(
                            "[KEYPOINT_TRACE] request_id=%s stage=model_output_before_scale "
                            "person=%d joint=%s idx=%d x=%.2f y=%.2f conf=%s "
                            "coord_space=model_input_pixels",
                            get_request_id(), person_index, COCO_KEYPOINTS[keypoint_index],
                            keypoint_index, float(point[0]), float(point[1]),
                            f"{confidence:.4f}" if confidence is not None else "unavailable"
                        )

                result = construct_result(pred, img, orig_img, img_path, *args, **kwargs)
                scaled_keypoints = result.keypoints.data
                for person_index, person_keypoints in enumerate(scaled_keypoints):
                    for keypoint_index in (7, 8, 13, 14):
                        point = person_keypoints[keypoint_index]
                        confidence = float(point[2]) if point.shape[0] >= 3 else None
                        logger.info(
                            "[KEYPOINT_TRACE] request_id=%s stage=ultralytics_scaled_to_original "
                            "person=%d joint=%s idx=%d x=%.2f y=%.2f conf=%s "
                            "coord_space=original_image_pixels",
                            get_request_id(), person_index, COCO_KEYPOINTS[keypoint_index],
                            keypoint_index, float(point[0]), float(point[1]),
                            f"{confidence:.4f}" if confidence is not None else "unavailable"
                        )
                return result

            predictor.construct_result = traced_construct_result
            predictor._pose_coordinate_trace_installed = True

    def estimate_pose(
        self,
        image_np: np.ndarray,
        tracker: Optional[PersonTracker] = None
    ) -> Tuple[bool, Optional[PoseKeypoints]]:
        """
        Estimate pose keypoints from an RGB/BGR numpy image array.
        Returns (success: bool, PoseKeypoints or None).
        """
        rid = get_request_id()

        if image_np is None or not isinstance(image_np, np.ndarray) or image_np.size == 0:
            logger.warning(
                "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='1. Invalid image input' "
                "Empty or invalid image array provided to estimate_pose.",
                rid
            )
            return False, None

        if self.model is None:
            logger.error(
                "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='2. YOLO inference failure' "
                "YOLO model is not initialized.",
                rid
            )
            return False, None

        img_h, img_w = image_np.shape[:2]
        img_dtype = str(image_np.dtype)
        img_shape = tuple(image_np.shape)

        try:
            _t0 = time.perf_counter()
            results = self.model(
                image_np,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )
            t_inference_ms = (time.perf_counter() - _t0) * 1000.0
            logger.debug("[TIMING] request_id=%s stage=yolo_inference duration_ms=%.2f", rid, t_inference_ms)
            logger.debug("[YOLO] request_id=%s inference_ms=%.1f image_shape=%dx%d", rid, t_inference_ms, img_w, img_h)

            if results is None or len(results) == 0:
                logger.info(
                    "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='3. No bounding boxes detected' "
                    "input_w=%d input_h=%d shape=%s dtype=%s inference_ms=%.1f boxes=0",
                    rid, img_w, img_h, img_shape, img_dtype, t_inference_ms
                )
                return False, None

            res = results[0]
            boxes = getattr(res, "boxes", None)
            keypoints = getattr(res, "keypoints", None)
            predictor = getattr(self.model, "predictor", None)
            predictor_args = getattr(predictor, "args", None)
            result_orig_shape = getattr(res, "orig_shape", None)
            keypoint_orig_shape = getattr(keypoints, "orig_shape", None)
            keypoint_output_shape = getattr(getattr(keypoints, "data", None), "shape", None)
            logger.info(
                "[YOLO] request_id=%s model=%s weights=%s source_shape=%s result_orig_shape=%s "
                "keypoint_orig_shape=%s keypoint_output_shape=%s imgsz=%s rect=%s stride=%s "
                "coordinate_space=original_image_pixels",
                rid,
                type(self.model).__name__,
                getattr(self, "resolved_model_path", getattr(self, "model_path", "unknown")),
                tuple(image_np.shape),
                result_orig_shape,
                keypoint_orig_shape,
                tuple(keypoint_output_shape) if keypoint_output_shape is not None else None,
                getattr(predictor_args, "imgsz", None),
                getattr(predictor_args, "rect", None),
                getattr(getattr(predictor, "model", None), "stride", None),
            )

            num_boxes = len(boxes) if boxes is not None else 0
            has_keypoints = (
                "yes"
                if keypoints is not None and getattr(keypoints, "data", None) is not None and len(keypoints.data) > 0
                else "no"
            )
            logger.debug("[YOLO] request_id=%s inference boxes=%d keypoints=%s", rid, num_boxes, has_keypoints)

            if boxes is None or len(boxes) == 0:
                logger.info(
                    "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='3. No bounding boxes detected' "
                    "input_w=%d input_h=%d shape=%s dtype=%s inference_ms=%.1f boxes=0",
                    rid, img_w, img_h, img_shape, img_dtype, t_inference_ms
                )
                return False, None

            if keypoints is None or getattr(keypoints, "data", None) is None or len(keypoints.data) == 0:
                logger.info(
                    "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='4. Bounding box detected but keypoints missing' "
                    "input_w=%d input_h=%d shape=%s dtype=%s inference_ms=%.1f boxes=%d",
                    rid, img_w, img_h, img_shape, img_dtype, t_inference_ms, num_boxes
                )
                return False, None

            conf_tensor = getattr(boxes, "conf", None)
            if conf_tensor is None or len(conf_tensor) == 0:
                logger.warning(
                    "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='3. No bounding boxes detected' "
                    "bounding boxes missing confidence tensor.",
                    rid
                )
                return False, None

            _t_kp_start = time.perf_counter()
            confs = conf_tensor.cpu().numpy()
            xyxy_tensor = getattr(boxes, "xyxy", None)
            kp_data_tensor = keypoints.data

            if xyxy_tensor is None or len(xyxy_tensor) == 0:
                logger.warning(
                    "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='3. No bounding boxes detected' "
                    "bounding boxes missing xyxy coordinates.",
                    rid
                )
                return False, None

            candidates = []
            num_people = min(len(confs), len(xyxy_tensor), len(kp_data_tensor))

            try:
                xyxy_np = xyxy_tensor.cpu().numpy() if hasattr(xyxy_tensor, "cpu") else np.array(xyxy_tensor)
            except Exception:
                xyxy_np = None

            for p_idx in range(num_people):
                p_conf = float(confs[p_idx])
                if xyxy_np is not None and len(xyxy_np) > p_idx:
                    box_item = xyxy_np[p_idx]
                else:
                    item_ref = xyxy_tensor[p_idx]
                    box_item = item_ref.cpu().numpy() if hasattr(item_ref, "cpu") else np.array(item_ref)
                p_box = [float(c) for c in np.ravel(box_item)[:4]]
                if len(p_box) < 4:
                    p_box = [0.0, 0.0, 100.0, 100.0]

                kp_elem = kp_data_tensor[p_idx]
                p_kp_raw = kp_elem.cpu().numpy() if hasattr(kp_elem, "cpu") else np.array(kp_elem)
                
                if p_kp_raw.shape[0] >= 17:
                    for idx, kp_name in enumerate(COCO_KEYPOINTS):
                        rx, ry = float(p_kp_raw[idx][0]), float(p_kp_raw[idx][1])
                        rc = float(p_kp_raw[idx][2]) if p_kp_raw.shape[1] >= 3 else p_conf
                        if kp_name in (
                            "left_shoulder", "right_shoulder",
                            "left_hip", "right_hip",
                            "left_knee", "right_knee",
                            "left_ankle", "right_ankle",
                            "left_elbow", "right_elbow",
                            "left_wrist", "right_wrist"
                        ):
                            log = logger.info if kp_name in (
                                "left_knee", "right_knee", "left_elbow", "right_elbow"
                            ) else logger.debug
                            log(
                                "[YOLO] request_id=%s stage=ultralytics_postprocessed_output idx=%d name=%s "
                                "x=%.2f y=%.2f conf=%.4f visible=%s (original_image=%dx%d)",
                                rid, idx, kp_name, rx, ry, rc, rc >= self.keypoint_conf_threshold, img_w, img_h
                            )

                p_kp_arr = refine_wrists(p_kp_raw)

                if p_kp_arr.ndim != 2 or p_kp_arr.shape[0] < 17:
                    continue

                for idx, kp_name in enumerate(COCO_KEYPOINTS):
                    if kp_name in (
                        "left_shoulder", "right_shoulder",
                        "left_hip", "right_hip",
                        "left_knee", "right_knee",
                        "left_ankle", "right_ankle",
                        "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist"
                    ):
                        jx, jy = float(p_kp_arr[idx][0]), float(p_kp_arr[idx][1])
                        jc = float(p_kp_arr[idx][2]) if p_kp_arr.shape[1] >= 3 else p_conf
                        rx, ry = float(p_kp_raw[idx][0]), float(p_kp_raw[idx][1])
                        delta_str = f" delta=(dx={jx-rx:+.2f}, dy={jy-ry:+.2f})" if (abs(jx-rx) > 0.01 or abs(jy-ry) > 0.01) else ""
                        log = logger.info if kp_name in (
                            "left_knee", "right_knee", "left_elbow", "right_elbow"
                        ) else logger.debug
                        log(
                            "[KEYPOINT_TRACE] request_id=%s stage=post_extraction joint=%s idx=%d "
                            "x=%.2f y=%.2f conf=%.4f coord_space=original_image_pixels%s",
                            rid, kp_name, idx, jx, jy, jc, delta_str
                        )

                p_kps = {}
                for idx, kp_name in enumerate(COCO_KEYPOINTS):
                    x = float(p_kp_arr[idx][0])
                    y = float(p_kp_arr[idx][1])
                    c = float(p_kp_arr[idx][2]) if p_kp_arr.shape[1] >= 3 else p_conf
                    p_kps[kp_name] = KeypointData(
                        x=x, y=y, confidence=c, name=kp_name,
                        visible=c >= self.keypoint_conf_threshold
                    )

                candidates.append(DetectedPersonCandidate(
                    box=p_box,
                    confidence=p_conf,
                    keypoints=p_kps
                ))

            if not candidates:
                logger.warning(
                    "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON "
                    "reason='5. Keypoints detected but rejected by application-level filtering' "
                    "input_w=%d input_h=%d shape=%s dtype=%s inference_ms=%.1f boxes=%d",
                    rid, img_w, img_h, img_shape, img_dtype, t_inference_ms, num_boxes
                )
                return False, None

            if tracker is not None:
                track_res = tracker.select_athlete(candidates)
                if track_res is None:
                    logger.warning(
                        "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON "
                        "reason='5. Keypoints detected but rejected by application-level filtering' "
                        "input_w=%d input_h=%d tracker_rejected=True",
                        rid, img_w, img_h
                    )
                    return False, None
                selected_cand, _ = track_res
            else:
                best_cand_idx = int(np.argmax(confs[:len(candidates)]))
                selected_cand = candidates[best_cand_idx]

            keypoints_dict = dict(selected_cand.keypoints)
            box_tensor = selected_cand.box
            best_conf = selected_cand.confidence

            # Compute Derived Landmarks
            derived = compute_derived_landmarks(keypoints_dict, conf_threshold=self.keypoint_conf_threshold)

            pose_kp = PoseKeypoints(
                keypoints=keypoints_dict,
                bounding_box=[float(box_tensor[0]), float(box_tensor[1]), float(box_tensor[2]), float(box_tensor[3])],
                detection_confidence=best_conf,
                derived_landmarks=derived
            )
            t_kp_extract_ms = (time.perf_counter() - _t_kp_start) * 1000.0
            logger.debug("[TIMING] request_id=%s stage=keypoint_extraction duration_ms=%.2f", rid, t_kp_extract_ms)

            kp_confs = [round(float(kp.confidence), 3) for kp in keypoints_dict.values()]
            visible_kps_count = sum(1 for kp in keypoints_dict.values() if kp.visible)
            confs_summary = [round(float(c), 3) for c in confs[:num_people]]
            logger.info(
                "[POSE_DIAGNOSTIC] request_id=%s decision=PERSON_DETECTED "
                "input_w=%d input_h=%d shape=%s dtype=%s inference_ms=%.1f "
                "num_boxes=%d person_confs=%s selected_conf=%.3f "
                "extracted_keypoints=%d visible_keypoints=%d "
                "kp_confs_min=%.3f kp_confs_avg=%.3f kp_confs_max=%.3f",
                rid, img_w, img_h, img_shape, img_dtype, t_inference_ms,
                num_boxes, confs_summary, best_conf,
                len(keypoints_dict), visible_kps_count,
                min(kp_confs) if kp_confs else 0.0,
                sum(kp_confs) / len(kp_confs) if kp_confs else 0.0,
                max(kp_confs) if kp_confs else 0.0
            )

            return True, pose_kp

        except Exception as e:
            logger.exception(
                "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='2. YOLO inference failure' error=%s",
                rid, str(e)
            )
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
        _t_render_start = time.perf_counter()
        rid = get_request_id()
        annotated = image_np.copy()
        if pose_kp is None or not pose_kp.keypoints:
            return annotated

        kps = pose_kp.keypoints
        is_side_view = analysis is not None and getattr(analysis, "side_view", False)
        h_img, w_img = annotated.shape[:2]
        logger.debug("[RENDER] request_id=%s draw_skeleton image=%dx%d is_side_view=%s", rid, w_img, h_img, is_side_view)

        if is_side_view:
            # Synchronize skeleton rendering with evaluator inputs.
            # 1. First preference: use resolved_landmarks from analysis.debug_info
            chain: Dict[str, KeypointData] = {}
            if analysis is not None and analysis.debug_info and "resolved_landmarks" in analysis.debug_info:
                for k, v in analysis.debug_info["resolved_landmarks"].items():
                    chain[k] = KeypointData(
                        x=v["x"], y=v["y"], confidence=v["confidence"], name=k, visible=v.get("visible", True)
                    )
            # 2. Second preference: check if kps contains resolved_{joint} or side_{joint}
            if not chain:
                for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
                    if f"resolved_{joint}" in kps:
                        chain[joint] = kps[f"resolved_{joint}"]
                    elif f"side_{joint}" in kps:
                        chain[joint] = kps[f"side_{joint}"]
            # 3. Fallback for legacy calls without analysis / resolved markers
            if not chain:
                chain = resolve_side_view_chain(kps, threshold=0.25)

            logger.debug("[RENDER] request_id=%s resolved_chain joints=%s", rid, list(chain.keys()))
            for jname in ("shoulder", "hip", "knee", "ankle", "elbow", "wrist"):
                if jname in chain:
                    jkp = chain[jname]
                    is_clipped = not (0 <= jkp.x < w_img and 0 <= jkp.y < h_img)
                    log = logger.info if jname in ("knee", "elbow") else logger.debug
                    log(
                        "[KEYPOINT_TRACE] request_id=%s stage=passed_to_rendering joint=%s x=%.2f y=%.2f conf=%.4f visible=%s clipped=%s",
                        rid, jname, jkp.x, jkp.y, jkp.confidence, jkp.visible, is_clipped
                    )
                else:
                    log = logger.info if jname in ("knee", "elbow") else logger.debug
                    log(
                        "[KEYPOINT_TRACE] request_id=%s stage=passed_to_rendering joint=%s status=unresolved",
                        rid, jname
                    )

            for a, b in (("shoulder", "elbow"), ("elbow", "wrist"),
                         ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle")):
                kp1, kp2 = chain.get(a), chain.get(b)
                if kp1 and kp2 and kp1.visible and kp2.visible:
                    is_clipped = not (0 <= kp1.x < w_img and 0 <= kp1.y < h_img and 0 <= kp2.x < w_img and 0 <= kp2.y < h_img)
                    if is_clipped:
                        logger.warning(
                            "[RENDER] request_id=%s bone_clipped %s->%s coords=(%.1f,%.1f)->(%.1f,%.1f) bounds=%dx%d",
                            rid, a, b, kp1.x, kp1.y, kp2.x, kp2.y, w_img, h_img
                        )
                    logger.debug("[RENDER] request_id=%s bone %s->%s (%.1f,%.1f)->(%.1f,%.1f)", rid, a, b, kp1.x, kp1.y, kp2.x, kp2.y)
                    cv2.line(annotated, (int(kp1.x), int(kp1.y)), (int(kp2.x), int(kp2.y)),
                             (0, 255, 127), 3, cv2.LINE_AA)
                else:
                    logger.debug("[RENDER] request_id=%s bone_skipped %s->%s kp1=%s kp2=%s",
                                 rid, a, b,
                                 f"({kp1.x:.0f},{kp1.y:.0f}) vis={kp1.visible}" if kp1 else "missing",
                                 f"({kp2.x:.0f},{kp2.y:.0f}) vis={kp2.visible}" if kp2 else "missing")
            for kp in chain.values():
                if kp and kp.visible:
                    center = (int(kp.x), int(kp.y))
                    cv2.circle(annotated, center, 6, (0, 215, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated, center, 8, (255, 255, 255), 1, cv2.LINE_AA)

            # Midpoints / body-line: synchronize with evaluator midpoints or resolved chain
            mid_pts = {}
            if analysis is not None and analysis.debug_info and "midpoints" in analysis.debug_info:
                for name, pt_data in analysis.debug_info["midpoints"].items():
                    if pt_data.get("visible", True):
                        mid_pts[name] = (int(pt_data["x"]), int(pt_data["y"]))
            else:
                for name in ("shoulder", "hip", "knee", "ankle"):
                    mname = f"mid_{name}"
                    if mname in kps and kps[mname].visible:
                        mid_pts[mname] = (int(kps[mname].x), int(kps[mname].y))
                    elif name in chain and chain[name].visible:
                        mid_pts[mname] = (int(chain[name].x), int(chain[name].y))

            logger.debug("[RENDER] request_id=%s midpoints drawn=%s", rid, list(mid_pts.keys()))
            if "mid_knee" in mid_pts:
                knee_source = kps.get("mid_knee") or chain.get("knee")
                logger.info(
                    "[KEYPOINT_TRACE] request_id=%s stage=rendered_midpoint joint=mid_knee "
                    "x=%d y=%d conf=%s",
                    rid, mid_pts["mid_knee"][0], mid_pts["mid_knee"][1],
                    f"{knee_source.confidence:.4f}" if knee_source is not None else "unavailable"
                )
            for a, b in (("mid_shoulder", "mid_hip"), ("mid_hip", "mid_knee"),
                         ("mid_knee", "mid_ankle")):
                if a in mid_pts and b in mid_pts:
                    logger.debug("[RENDER] request_id=%s body_line %s->%s %s->%s", rid, a, b, mid_pts[a], mid_pts[b])
                    cv2.line(annotated, mid_pts[a], mid_pts[b], (255, 255, 0), 4, cv2.LINE_AA)
                else:
                    logger.debug("[RENDER] request_id=%s body_line_skipped %s->%s a_in=%s b_in=%s", rid, a, b, a in mid_pts, b in mid_pts)
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
        t_render_ms = (time.perf_counter() - _t_render_start) * 1000.0
        logger.debug("[TIMING] request_id=%s stage=skeleton_rendering duration_ms=%.2f", rid, t_render_ms)
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
