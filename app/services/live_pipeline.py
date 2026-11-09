import base64
import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from app.core.config import settings
from app.core.logging_context import get_request_id, set_request_id
from app.exercises.registry import exercise_registry
from app.pose.yolo_pose import (
    COCO_KEYPOINTS,
    SKELETON_CONNECTIONS,
    YOLOPoseEstimator,
)
from app.schemas.analysis import (
    FeedbackSeverity,
    KeypointData,
    LandmarkItem,
    LiveAnalysisResponse,
    LiveAnalysisStatus,
    LiveFeedbackItem,
    LiveTiming,
    PoseKeypoints,
)
from app.services.live_session import LiveSession, session_manager

logger = logging.getLogger(__name__)


def detect_image_format(data: bytes) -> str:
    """Detect image encoding format from magic bytes."""
    if not data or len(data) < 4:
        return "UNKNOWN"
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return "WEBP"
    if data.startswith(b"BM"):
        return "BMP"
    return "UNKNOWN"


class LiveAnalysisPipeline:
    """
    Production real-time frame processing pipeline:
    Camera Frame -> Validation -> Preprocessing -> YOLOv8 Pose ->
    Landmark Extraction -> Validation -> Exercise-Specific Analysis ->
    Temporal Smoothing -> Form Evaluation & Feedback -> Response.
    """

    def __init__(self, pose_estimator: Optional[YOLOPoseEstimator] = None):
        self.pose_estimator = pose_estimator or YOLOPoseEstimator()

    def process_frame(
        self,
        image_bytes: bytes,
        exercise: str = "FULL_PLANCHE",
        session_id: Optional[str] = None,
        frame_sequence: int = 0,
        timestamp_ms: int = 0,
        side_view: bool = False,
        draw_overlay: bool = False,
        debug: bool = False,
        request_id: Optional[str] = None
    ) -> LiveAnalysisResponse:
        t_pipeline_start = time.perf_counter()
        rid = request_id or get_request_id()
        set_request_id(rid)

        # 1. Frame Validation
        format_name = detect_image_format(image_bytes)
        if not image_bytes or len(image_bytes) < 32:
            t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            logger.warning(
                "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='1. Invalid image input' "
                "format=%s bytes_len=%d",
                rid, format_name, len(image_bytes) if image_bytes else 0
            )
            return LiveAnalysisResponse(
                status=LiveAnalysisStatus.INVALID_INPUT,
                exercise=exercise,
                timestamp=timestamp_ms,
                frame_sequence=frame_sequence,
                pose_detected=False,
                confidence=0.0,
                score=None,
                feedback=[
                    LiveFeedbackItem(
                        code="EMPTY_FRAME",
                        message="Received empty or truncated image payload.",
                        severity=FeedbackSeverity.ERROR
                    )
                ],
                timing=LiveTiming(total_ms=round(t_total_ms, 2)),
                session_id=session_id
            )

        # 2. Image Preprocessing & Decoding
        t_pre_start = time.perf_counter()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        t_pre_ms = (time.perf_counter() - t_pre_start) * 1000.0

        if image_np is None or image_np.size == 0 or image_np.shape[0] < 32 or image_np.shape[1] < 32:
            t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            logger.warning(
                "[POSE_DIAGNOSTIC] request_id=%s decision=NO_PERSON reason='1. Invalid image input' "
                "format=%s bytes_len=%d decode_success=%s",
                rid, format_name, len(image_bytes), image_np is not None
            )
            return LiveAnalysisResponse(
                status=LiveAnalysisStatus.INVALID_INPUT,
                exercise=exercise,
                timestamp=timestamp_ms,
                frame_sequence=frame_sequence,
                pose_detected=False,
                confidence=0.0,
                score=None,
                feedback=[
                    LiveFeedbackItem(
                        code="DECODE_FAILED",
                        message="Could not decode frame as a valid image (JPEG/PNG expected).",
                        severity=FeedbackSeverity.ERROR
                    )
                ],
                timing=LiveTiming(
                    preprocessing_ms=round(t_pre_ms, 2),
                    total_ms=round(t_total_ms, 2)
                ),
                session_id=session_id
            )

        img_h, img_w = image_np.shape[:2]
        logger.info(
            "[FRAME_INFERENCE_REQUEST] request_id=%s exercise=%s format=%s input_w=%d input_h=%d "
            "shape=%s dtype=%s bytes=%d",
            rid, exercise, format_name, img_w, img_h, tuple(image_np.shape), str(image_np.dtype), len(image_bytes)
        )

        # Retrieve or initialize live session early for tracking & temporal smoothing
        session = session_manager.get_or_create(
            session_id=session_id,
            exercise=exercise,
            side_view=side_view
        )
        session.touch()

        # 3. YOLOv8 Pose Estimation (with athlete tracking persistence)
        t_infer_start = time.perf_counter()
        success, pose_kp = self.pose_estimator.estimate_pose(image_np, tracker=session.tracker)
        t_infer_ms = (time.perf_counter() - t_infer_start) * 1000.0

        if not success or pose_kp is None or not pose_kp.keypoints:
            t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            logger.info(
                "[FRAME_INFERENCE_SUMMARY] request_id=%s exercise=%s pose_detected=False "
                "status=NO_PERSON_DETECTED inference_ms=%.1f total_ms=%.1f",
                rid, exercise, t_infer_ms, t_total_ms
            )
            return LiveAnalysisResponse(
                status=LiveAnalysisStatus.NO_PERSON_DETECTED,
                exercise=exercise,
                timestamp=timestamp_ms,
                frame_sequence=frame_sequence,
                pose_detected=False,
                confidence=0.0,
                score=None,
                feedback=[
                    LiveFeedbackItem(
                        code="NO_PERSON_DETECTED",
                        message="No athlete detected in frame. Step into full view of camera.",
                        severity=FeedbackSeverity.WARNING
                    )
                ],
                timing=LiveTiming(
                    preprocessing_ms=round(t_pre_ms, 2),
                    inference_ms=round(t_infer_ms, 2),
                    analysis_ms=0.0,
                    total_ms=round(t_total_ms, 2)
                ),
                session_id=session.session_id,
                session_summary=session.get_session_summary()
            )

        # 4. Pose Landmark Extraction & Coordinate Mapping
        t_analysis_start = time.perf_counter()
        raw_keypoints = pose_kp.keypoints
        landmarks_list: List[LandmarkItem] = []

        for name, kp in raw_keypoints.items():
            x_norm = max(0.0, min(1.0, kp.x / img_w)) if img_w > 0 else 0.0
            y_norm = max(0.0, min(1.0, kp.y / img_h)) if img_h > 0 else 0.0
            landmarks_list.append(
                LandmarkItem(
                    name=name,
                    x=round(kp.x, 1),
                    y=round(kp.y, 1),
                    x_norm=round(x_norm, 4),
                    y_norm=round(y_norm, 4),
                    confidence=round(kp.confidence, 3),
                    visible=bool(kp.visible)
                )
            )

        # 5. Exercise Analyzer Resolution & Landmark Validation
        analyzer = exercise_registry.get_analyzer(exercise)
        is_pose_valid, pose_status, missing_lms = analyzer.validate_pose(
            keypoints=raw_keypoints,
            threshold=settings.KEYPOINT_CONF_THRESHOLD,
            side_view=side_view
        )

        # If pose is incomplete or low confidence:
        if not is_pose_valid:
            t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0
            missing_clean = [m.replace("left_", "").replace("right_", "") for m in missing_lms]
            distinct_missing = list(dict.fromkeys(missing_clean))

            feedback = [
                LiveFeedbackItem(
                    code="INCOMPLETE_POSE" if pose_status == LiveAnalysisStatus.INCOMPLETE_POSE else "LOW_CONFIDENCE",
                    message=f"Ensure your full body is visible: missing or occluded {', '.join(distinct_missing)}.",
                    severity=FeedbackSeverity.WARNING
                )
            ]

            logger.info(
                "[FRAME_INFERENCE_SUMMARY] request_id=%s exercise=%s pose_detected=True "
                "status=%s detection_confidence=%.3f landmarks_count=%d missing_landmarks=%s "
                "inference_ms=%.1f total_ms=%.1f",
                rid, exercise, pose_status.value if hasattr(pose_status, "value") else str(pose_status),
                pose_kp.detection_confidence, len(landmarks_list), missing_lms, t_infer_ms, t_total_ms
            )

            # Inform session counter of tracking loss if squat
            now_sec = (timestamp_ms / 1000.0) if timestamp_ms > 0 else time.time()
            if session.squat_counter is not None:
                session.squat_counter.process_angle(None, timestamp_sec=now_sec)

            return LiveAnalysisResponse(
                status=pose_status,
                exercise=exercise,
                timestamp=timestamp_ms,
                frame_sequence=frame_sequence,
                pose_detected=True,
                confidence=round(pose_kp.detection_confidence, 3),
                score=None,  # NEVER fabricate a score for incomplete poses
                feedback=feedback,
                missing_landmarks=missing_lms,
                landmarks=landmarks_list,
                current_state=session.squat_counter.state.value if session.squat_counter else None,
                knee_angle=round(session.squat_counter.smoothed_knee_angle, 1) if session.squat_counter else None,
                min_knee_angle=round(session.squat_counter.min_knee_angle, 1) if session.squat_counter else None,
                depth=session.squat_counter.depth_quality if session.squat_counter else None,
                is_tracking=False,
                rep_count=session.squat_counter.rep_count if session.squat_counter else None,
                rep_phase=session.squat_counter.state.value if session.squat_counter else None,
                phase=session.squat_counter.state.value if session.squat_counter else None,
                timing=LiveTiming(
                    preprocessing_ms=round(t_pre_ms, 2),
                    inference_ms=round(t_infer_ms, 2),
                    analysis_ms=round((time.perf_counter() - t_analysis_start) * 1000.0, 2),
                    total_ms=round(t_total_ms, 2)
                ),
                session_id=session.session_id,
                session_summary=session.get_session_summary()
            )

        # 6. Apply Temporal Smoothing
        # Apply EMA temporal filter to keypoints
        smoothed_kps = session.smooth_keypoints(raw_keypoints)

        # 7. Exercise Biomechanical Analysis
        raw_measurements = analyzer.extract_measurements(raw_keypoints, side_view=side_view)
        smoothed_measurements = analyzer.extract_measurements(smoothed_kps, side_view=side_view)
        # Further smooth angle measurements through session EMA
        final_measurements = session.smooth_measurements(smoothed_measurements)

        # Sanitize measurements (no NaN or Inf in JSON)
        clean_measurements = {
            k: (v if math.isfinite(v) else 0.0)
            for k, v in final_measurements.items()
        }

        # 8. Form Evaluation, Scoring, and Feedback
        legacy_score, feedback_items, is_valid_frame = analyzer.evaluate_live_form(
            measurements=clean_measurements,
            progression=exercise,
            side_view=side_view
        )

        # Evaluate detailed scoring metrics per exercise
        scoring_metrics = analyzer.evaluate_scoring_metrics(
            measurements=clean_measurements,
            keypoints=smoothed_kps,
            side_view=side_view
        )

        # Calculate weighted scoring with dynamic normalization and reliability
        overall_score, score_reliability, _, _ = session.scoring_engine.evaluate_score(
            metrics=scoring_metrics,
            keypoints=smoothed_kps,
            required_landmarks=analyzer.required_landmarks
        )

        if overall_score is None:
            overall_score = legacy_score

        # Feedback Engine processing (cooldown & severity priority)
        now_sec = (timestamp_ms / 1000.0) if timestamp_ms > 0 else time.time()
        processed_feedback = session.feedback_engine.process_feedback(
            feedback_items,
            timestamp_sec=now_sec
        )

        # 9. Dynamic State Update (Hold duration or Repetition count)
        hold_sec, rep_count, rep_phase = analyzer.update_dynamic_state(
            session=session,
            is_valid_frame=is_valid_frame,
            measurements=clean_measurements,
            timestamp_sec=now_sec
        )

        # Record performance history
        session.record_frame_score(overall_score, hold_sec, rep_count)

        t_analysis_ms = (time.perf_counter() - t_analysis_start) * 1000.0

        # 10. Optional Annotated Skeleton Overlay
        annotated_b64 = None
        if draw_overlay:
            annotated_b64 = self._render_annotated_frame(
                image_np=image_np,
                keypoints=smoothed_kps,
                score=overall_score,
                exercise=exercise,
                feedback=processed_feedback
            )

        t_total_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        logger.info(
            "[FRAME_INFERENCE_SUMMARY] request_id=%s exercise=%s pose_detected=True "
            "status=VALID detection_confidence=%.3f score=%s landmarks_count=%d "
            "inference_ms=%.1f analysis_ms=%.1f total_ms=%.1f",
            rid, exercise, pose_kp.detection_confidence,
            f"{overall_score:.1f}" if overall_score is not None else "None",
            len(landmarks_list), t_infer_ms, t_analysis_ms, t_total_ms
        )

        sq_counter = session.squat_counter
        current_state = sq_counter.state.value if sq_counter else rep_phase
        knee_angle = round(sq_counter.smoothed_knee_angle, 1) if sq_counter else clean_measurements.get("knee_angle")
        min_knee_angle = round(sq_counter.min_knee_angle, 1) if sq_counter else clean_measurements.get("knee_angle")
        depth = sq_counter.depth_quality if sq_counter else None
        is_tracking = sq_counter.is_tracking if sq_counter else True

        return LiveAnalysisResponse(
            status=LiveAnalysisStatus.VALID,
            exercise=exercise,
            timestamp=timestamp_ms,
            frame_sequence=frame_sequence,
            pose_detected=True,
            confidence=round(pose_kp.detection_confidence, 3),
            score=round(overall_score, 1) if overall_score is not None else None,
            current_score=round(overall_score, 1) if overall_score is not None else None,
            score_confidence=round(score_reliability, 2),
            is_valid_hold=is_valid_frame,
            hold_duration_sec=hold_sec,
            hold_duration_ms=int(hold_sec * 1000) if hold_sec is not None else None,
            rep_count=rep_count,
            rep_phase=rep_phase,
            phase=rep_phase,
            current_state=current_state,
            knee_angle=knee_angle,
            min_knee_angle=min_knee_angle,
            depth=depth,
            is_tracking=is_tracking,
            measurements=clean_measurements,
            scoring_metrics=scoring_metrics,
            raw_measurements=raw_measurements if debug else None,
            feedback=processed_feedback,
            missing_landmarks=[],
            landmarks=landmarks_list,
            annotated_frame_base64=annotated_b64,
            timing=LiveTiming(
                preprocessing_ms=round(t_pre_ms, 2),
                inference_ms=round(t_infer_ms, 2),
                analysis_ms=round(t_analysis_ms, 2),
                total_ms=round(t_total_ms, 2)
            ),
            session_id=session.session_id,
            session_summary=session.get_session_summary()
        )

    def _render_annotated_frame(
        self,
        image_np: np.ndarray,
        keypoints: Dict[str, KeypointData],
        score: Optional[float],
        exercise: str,
        feedback: List[LiveFeedbackItem]
    ) -> Optional[str]:
        """Render minimal, real-time pose skeleton and form overlay."""
        try:
            overlay = image_np.copy()
            # Draw skeleton connections
            for p1_name, p2_name in SKELETON_CONNECTIONS:
                kp1 = keypoints.get(p1_name)
                kp2 = keypoints.get(p2_name)
                if kp1 and kp2 and kp1.visible and kp2.visible:
                    pt1 = (int(kp1.x), int(kp1.y))
                    pt2 = (int(kp2.x), int(kp2.y))
                    cv2.line(overlay, pt1, pt2, (0, 255, 128), 2, cv2.LINE_AA)

            # Draw keypoint circles
            for name, kp in keypoints.items():
                if kp.visible:
                    cv2.circle(overlay, (int(kp.x), int(kp.y)), 4, (0, 165, 255), -1, cv2.LINE_AA)

            # Top HUD banner
            h, w = overlay.shape[:2]
            cv2.rectangle(overlay, (10, 10), (min(320, w - 10), 75), (20, 20, 20), -1)
            cv2.rectangle(overlay, (10, 10), (min(320, w - 10), 75), (80, 80, 80), 1)

            cv2.putText(overlay, f"{exercise.replace('_', ' ')}", (20, 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            score_str = f"Score: {score:.1f}" if score is not None else "Score: --"
            score_color = (0, 255, 0) if (score and score >= 80) else (0, 200, 255) if (score and score >= 60) else (0, 0, 255)
            cv2.putText(overlay, score_str, (20, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, score_color, 2, cv2.LINE_AA)

            # Draw feedback message if present
            if feedback:
                first_fb = feedback[0]
                fb_color = (0, 0, 255) if first_fb.severity == FeedbackSeverity.ERROR else (0, 200, 255)
                cv2.putText(overlay, first_fb.message[:45], (20, h - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.50, fb_color, 1, cv2.LINE_AA)

            # Encode to JPEG
            success, encoded_jpg = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if success:
                return base64.b64encode(encoded_jpg.tobytes()).decode("utf-8")
        except Exception as e:
            logger.warning("Error rendering live annotated frame: %s", e)
        return None


# Global pipeline instance
live_pipeline = LiveAnalysisPipeline()
