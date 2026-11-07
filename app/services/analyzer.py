import base64
import logging
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from app.biomechanics.hold_detection import HoldState, PlancheHoldDetector
from app.biomechanics.metrics import BiomechanicsEvaluator
from app.biomechanics.smoothing import TemporalKeypointSmoother
from app.core.config import settings
from app.core.logging_context import get_request_id, set_request_id
from app.exercises.planche import PlancheAnalyzer
from app.pose.yolo_pose import YOLOPoseEstimator
from app.schemas.analysis import (
    ImageAnalysisResponse,
    PlancheAnalysisResult,
    PoseKeypoints,
    ProgressionStage,
    StabilityMetrics,
    VideoAnalysisResponse,
    VideoFrameAnalysis,
    VideoMetadata,
    parse_progression_stage,
)

logger = logging.getLogger(__name__)


class CalisthenicsAnalyzerService:
    """Service orchestrating pose estimation, biomechanics calculation, temporal smoothing, and video stream processing."""

    def __init__(self, pose_estimator: Optional[YOLOPoseEstimator] = None):
        self.pose_estimator = pose_estimator or YOLOPoseEstimator()

    def analyze_image(
        self,
        image_bytes: bytes,
        target_progression: Optional[Union[str, ProgressionStage]] = None,
        side_view: bool = False,
        draw_overlay: bool = True,
        debug: bool = False,
        request_id: Optional[str] = None
    ) -> ImageAnalysisResponse:
        """
        Analyzes a single image for user-selected Planche progression form evaluation.
        Explicitly branches between side-view midpoint processing and standard processing.
        """
        start_time = time.perf_counter()
        target_stage = parse_progression_stage(target_progression)
        rid = request_id or get_request_id()
        set_request_id(rid)
        logger.info(
            "[ANALYZER] request_id=%s progression=%s side_view=%s",
            rid, target_stage.value if target_stage else "none", side_view
        )

        # 1. Decode image bytes to numpy array
        _t_decode_start = time.perf_counter()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000.0
        logger.debug("[TIMING] request_id=%s stage=image_decoding duration_ms=%.2f", rid, t_decode_ms)

        if image_np is not None:
            h, w = image_np.shape[:2]
            ch = image_np.shape[2] if image_np.ndim == 3 else 1
            logger.info("[ANALYZER] request_id=%s image_decoded w=%d h=%d channels=%d dtype=%s",
                         rid, w, h, ch, image_np.dtype)
        else:
            logger.warning("[ANALYZER] request_id=%s image_decode_failed bytes=%d", rid, len(image_bytes))

        if image_np is None:
            return ImageAnalysisResponse(
                success=False,
                message="Invalid or corrupt image format.",
                processing_time_ms=0.0,
                pose_detected=False,
                side_view=side_view
            )

        # 2. Perform Pose Estimation
        success, pose_kp = self.pose_estimator.estimate_pose(image_np)
        logger.debug("[ANALYZER] request_id=%s pose_detected=%s detection_conf=%s",
                     rid, success,
                     f"{pose_kp.detection_confidence:.3f}" if (success and pose_kp) else "n/a")

        if not success or pose_kp is None:
            proc_time = (time.perf_counter() - start_time) * 1000.0
            return ImageAnalysisResponse(
                success=False,
                message="No human pose detected in image.",
                processing_time_ms=round(proc_time, 2),
                pose_detected=False,
                side_view=side_view
            )

        # 3. Evaluate Biomechanics (explicit side_view branch)
        profile = BiomechanicsEvaluator.evaluate(pose_kp, side_view=side_view)
        if profile is not None:
            missing = profile.missing_landmarks or []
            logger.debug(
                "[ANALYZER] request_id=%s biomechanics_ok side_view=%s missing_landmarks=%s",
                rid, profile.side_view, missing if missing else "none"
            )
        else:
            logger.warning("[ANALYZER] request_id=%s biomechanics_profile=None", rid)
        if profile is None:
            proc_time = (time.perf_counter() - start_time) * 1000.0
            return ImageAnalysisResponse(
                success=False,
                message="Could not evaluate biomechanics from keypoints.",
                processing_time_ms=round(proc_time, 2),
                pose_detected=True,
                side_view=side_view,
                pose_keypoints=pose_kp
            )

        # 4. Evaluate Form Against User-Selected Progression
        planche_result: PlancheAnalysisResult = PlancheAnalyzer.analyze(
            profile=profile,
            target_progression=target_stage,
            side_view=side_view
        )
        logger.info(
            "[ANALYZER] request_id=%s form_score=%.1f progression=%s is_valid_hold=%s",
            rid, planche_result.form_score,
            planche_result.target_progression.value, planche_result.is_valid_hold
        )

        # 5. Render Annotated Overlay Image
        annotated_b64 = None
        if draw_overlay:
            # In side-view mode, pass updated keypoints with midpoints and resolved chain
            render_pose = pose_kp
            if side_view:
                combined_dict = dict(pose_kp.keypoints)
                if profile.midpoints:
                    combined_dict.update(profile.midpoints)
                if profile.resolved_landmarks:
                    for k, v in profile.resolved_landmarks.items():
                        combined_dict[f"resolved_{k}"] = v
                        combined_dict[f"side_{k}"] = v
                render_pose = PoseKeypoints(
                    keypoints=combined_dict,
                    bounding_box=pose_kp.bounding_box,
                    detection_confidence=pose_kp.detection_confidence
                )

            annotated = self.pose_estimator.draw_skeleton(
                image_np=image_np,
                pose_kp=render_pose,
                form_score=planche_result.form_score,
                progression=planche_result.target_progression.value,
                errors=planche_result.form_errors,
                analysis=planche_result
            )
            success_enc, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if success_enc:
                annotated_b64 = base64.b64encode(buffer).decode("utf-8")

        proc_time = (time.perf_counter() - start_time) * 1000.0
        mode_label = "side-view" if side_view else "standard"
        return ImageAnalysisResponse(
            success=True,
            message=f"Successfully evaluated form ({mode_label}) for target progression: {target_stage.value}.",
            processing_time_ms=round(proc_time, 2),
            pose_detected=True,
            side_view=side_view,
            pose_keypoints=pose_kp,
            planche_analysis=planche_result,
            annotated_image_base64=annotated_b64
        )

    def export_annotated_image(
        self,
        image_bytes: bytes,
        target_progression: Optional[Union[str, ProgressionStage]] = None,
        side_view: bool = False,
        request_id: Optional[str] = None
    ) -> Tuple[bytes, ImageAnalysisResponse]:
        """
        Runs pose estimation & form evaluation on image_bytes against target_progression,
        draws skeleton overlay onto image, and returns raw JPEG bytes of annotated image.
        """
        _t_export_start = time.perf_counter()
        rid = request_id or get_request_id()
        set_request_id(rid)

        _t_decode_start = time.perf_counter()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        t_decode_ms = (time.perf_counter() - _t_decode_start) * 1000.0
        logger.debug("[TIMING] request_id=%s stage=image_decoding duration_ms=%.2f", rid, t_decode_ms)

        if image_np is None:
            logger.warning("[DRAW_SKELETON] request_id=%s image_decode_failed bytes=%d", rid, len(image_bytes))
            raise ValueError("Invalid or corrupt image format.")

        h, w = image_np.shape[:2]
        logger.debug("[DRAW_SKELETON] request_id=%s image_decoded w=%d h=%d", rid, w, h)

        response = self.analyze_image(
            image_bytes=image_bytes,
            target_progression=target_progression,
            side_view=side_view,
            draw_overlay=True,
            request_id=request_id
        )

        if not response.pose_detected or response.pose_keypoints is None:
            _, buffer = cv2.imencode(".jpg", image_np)
            t_export_ms = (time.perf_counter() - _t_export_start) * 1000.0
            logger.debug("[TIMING] request_id=%s stage=total_export duration_ms=%.2f", rid, t_export_ms)
            return buffer.tobytes(), response

        planche_result = response.planche_analysis
        render_pose = response.pose_keypoints
        if side_view and planche_result and planche_result.debug_info:
            from app.schemas.analysis import KeypointData
            combined_dict = dict(render_pose.keypoints)
            if "midpoints" in planche_result.debug_info:
                midpoints_raw = planche_result.debug_info["midpoints"]
                for m_name, m_val in midpoints_raw.items():
                    combined_dict[m_name] = KeypointData(
                        x=m_val["x"],
                        y=m_val["y"],
                        confidence=m_val["confidence"],
                        name=m_name,
                        visible=m_val.get("visible", True)
                    )
            if "resolved_landmarks" in planche_result.debug_info:
                resolved_raw = planche_result.debug_info["resolved_landmarks"]
                for r_name, r_val in resolved_raw.items():
                    kp = KeypointData(
                        x=r_val["x"],
                        y=r_val["y"],
                        confidence=r_val["confidence"],
                        name=r_name,
                        visible=r_val.get("visible", True)
                    )
                    combined_dict[f"resolved_{r_name}"] = kp
                    combined_dict[f"side_{r_name}"] = kp
            render_pose = PoseKeypoints(
                keypoints=combined_dict,
                bounding_box=render_pose.bounding_box,
                detection_confidence=render_pose.detection_confidence
            )

        annotated = self.pose_estimator.draw_skeleton(
            image_np=image_np,
            pose_kp=render_pose,
            form_score=planche_result.form_score if planche_result else None,
            progression=planche_result.target_progression.value if planche_result else None,
            errors=planche_result.form_errors if planche_result else None,
            analysis=planche_result
        )
        success_enc, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not success_enc:
            raise ValueError("Failed to encode annotated image.")

        t_export_ms = (time.perf_counter() - _t_export_start) * 1000.0
        logger.debug("[TIMING] request_id=%s stage=total_export duration_ms=%.2f", rid, t_export_ms)
        logger.info("[DRAW_SKELETON] request_id=%s render_done output_bytes=%d", rid, len(buffer.tobytes()))
        return buffer.tobytes(), response

    def analyze_video(
        self,
        video_bytes: bytes,
        target_progression: Optional[Union[str, ProgressionStage]] = None,
        side_view: bool = False,
        target_sample_fps: int = 5,
        debug: bool = False
    ) -> VideoAnalysisResponse:
        """
        Analyzes a video file frame-by-frame against the user's target progression.
        Applies temporal keypoint smoothing (EMA) and multi-frame stable hold detection.
        Extracts original video metadata (resolution, fps, frames) before sampling.
        Populates per-frame biomechanical angles and stability metrics.
        """
        start_time = time.perf_counter()
        target_stage = parse_progression_stage(target_progression)
        file_size_bytes = len(video_bytes)

        # Write video bytes to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        try:
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return VideoAnalysisResponse(
                    success=False,
                    message="Failed to open video file.",
                    processing_time_ms=0.0,
                    total_frames=0,
                    fps=0.0,
                    duration_sec=0.0,
                    total_hold_duration_sec=0.0,
                    valid_hold_duration_sec=0.0,
                    peak_form_score=0.0,
                    average_form_score=0.0,
                    primary_progression=target_stage,
                    side_view=side_view
                )

            # --- Extract original video metadata BEFORE sampling ---
            original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            # Use original fps/frames for duration — never calculated from sampled frames
            duration_sec = total_frames / original_fps if original_fps > 0 else 0.0
            duration_ms = int(duration_sec * 1000)

            sample_interval = max(1, int(original_fps / target_sample_fps))
            effective_sample_fps = round(original_fps / sample_interval, 2)

            # Initialize Temporal Smoother and Hold Detector
            smoother = TemporalKeypointSmoother(alpha=settings.SMOOTHING_ALPHA)
            detector = PlancheHoldDetector(
                fps=effective_sample_fps,
                min_hold_duration_sec=settings.MIN_HOLD_DURATION_SEC,
                stability_variance_threshold=settings.HOLD_STABILITY_VARIANCE_THRESHOLD
            )

            frame_idx = 0
            analyzed_frames_count = 0
            timeline: List[VideoFrameAnalysis] = []
            stable_component_scores: List[Dict[str, float]] = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    analyzed_frames_count += 1
                    timestamp_sec = frame_idx / original_fps
                    success_pose, pose_kp = self.pose_estimator.estimate_pose(frame)

                    if success_pose and pose_kp is not None:
                        # 1. Temporal Smoothing before biomechanical feature calculation
                        smoothed_kps = smoother.smooth(pose_kp.keypoints)
                        smoothed_pose_kp = PoseKeypoints(
                            keypoints=smoothed_kps,
                            bounding_box=pose_kp.bounding_box,
                            detection_confidence=pose_kp.detection_confidence
                        )

                        # 2. Biomechanics Evaluation
                        profile = BiomechanicsEvaluator.evaluate(smoothed_pose_kp, side_view=side_view)

                        if profile and profile.pose_quality_ok:
                            # 3. Form Evaluation
                            res = PlancheAnalyzer.analyze(
                                profile=profile,
                                target_progression=target_stage,
                                side_view=side_view
                            )

                            # 4. State Machine Hold Detection
                            hold_state = detector.process_frame(
                                frame_idx=frame_idx,
                                timestamp_sec=timestamp_sec,
                                pose_detected=True,
                                is_valid_quality=True,
                                elbow_angle=res.joint_angles.primary_elbow_deg,
                                shoulder_lean=res.joint_angles.primary_shoulder_lean_deg,
                                hip_angle=res.joint_angles.hip_angle_deg,
                                body_tilt=res.joint_angles.body_horizontal_tilt_deg,
                                form_score=res.form_score
                            )

                            if hold_state == HoldState.STABLE_HOLD:
                                stable_component_scores.append(res.component_scores)

                            timeline.append(VideoFrameAnalysis(
                                frame_index=frame_idx,
                                timestamp_sec=round(timestamp_sec, 2),
                                pose_detected=True,
                                form_score=res.form_score,
                                is_valid_hold=res.is_valid_hold,
                                hold_state=hold_state.value,
                                detected_progression=res.target_progression.value,
                                key_issues=res.form_errors,
                                # Per-frame biomechanical angles
                                elbow_angle=round(res.joint_angles.primary_elbow_deg, 1),
                                shoulder_angle=round(res.joint_angles.primary_shoulder_lean_deg, 1),
                                hip_alignment=round(res.joint_angles.hip_angle_deg, 1),
                                body_line_deviation=round(res.joint_angles.body_horizontal_tilt_deg, 1),
                                shoulder_lean=round(res.joint_angles.primary_shoulder_lean_deg, 2),
                                knee_angle=round(res.joint_angles.primary_knee_deg, 1),
                                pose_confidence=round(smoothed_pose_kp.detection_confidence, 3),
                            ))
                        else:
                            hold_state = detector.process_frame(
                                frame_idx=frame_idx,
                                timestamp_sec=timestamp_sec,
                                pose_detected=True,
                                is_valid_quality=False,
                                elbow_angle=0.0,
                                shoulder_lean=0.0,
                                hip_angle=0.0,
                                body_tilt=0.0,
                                form_score=0.0
                            )
                            timeline.append(VideoFrameAnalysis(
                                frame_index=frame_idx,
                                timestamp_sec=round(timestamp_sec, 2),
                                pose_detected=True,
                                form_score=0.0,
                                is_valid_hold=False,
                                hold_state=hold_state.value,
                                detected_progression=target_stage.value,
                                key_issues=["Insufficient pose quality / occluded joints"],
                                pose_confidence=round(pose_kp.detection_confidence, 3) if pose_kp else None,
                            ))
                    else:
                        hold_state = detector.process_frame(
                            frame_idx=frame_idx,
                            timestamp_sec=timestamp_sec,
                            pose_detected=False,
                            is_valid_quality=False,
                            elbow_angle=0.0,
                            shoulder_lean=0.0,
                            hip_angle=0.0,
                            body_tilt=0.0,
                            form_score=0.0
                        )
                        timeline.append(VideoFrameAnalysis(
                            frame_index=frame_idx,
                            timestamp_sec=round(timestamp_sec, 2),
                            pose_detected=False,
                            form_score=0.0,
                            is_valid_hold=False,
                            hold_state=hold_state.value,
                            detected_progression=target_stage.value,
                            key_issues=["No pose detected"],
                        ))

                frame_idx += 1

            cap.release()

            # 5. Hold Detection Aggregation
            summary = detector.get_summary(sample_interval=1)

            # Compute aggregated component scores
            aggregated_components: Dict[str, float] = {}
            if stable_component_scores:
                for k in stable_component_scores[0].keys():
                    vals = [c[k] for c in stable_component_scores if k in c]
                    aggregated_components[k] = round(float(np.mean(vals)), 1)
            elif timeline:
                attempt_scores = [f.form_score for f in timeline if f.pose_detected and f.form_score > 0]
                if attempt_scores:
                    aggregated_components["overall"] = round(float(np.mean(attempt_scores)), 1)

            # 6. Compute Stability Metrics from timeline frames with valid pose
            stability: Optional[StabilityMetrics] = None
            valid_frames = [f for f in timeline if f.pose_detected and f.body_line_deviation is not None]
            if valid_frames:
                body_devs = [f.body_line_deviation for f in valid_frames if f.body_line_deviation is not None]
                shoulder_leans = [f.shoulder_angle for f in valid_frames if f.shoulder_angle is not None]
                hip_alignments = [f.hip_alignment for f in valid_frames if f.hip_alignment is not None]
                confidences = [f.pose_confidence for f in valid_frames if f.pose_confidence is not None]

                avg_dev = float(np.mean(body_devs)) if body_devs else 0.0
                max_dev = float(np.max(np.abs(body_devs))) if body_devs else 0.0
                shoulder_var = float(np.std(shoulder_leans)) if len(shoulder_leans) > 1 else 0.0
                hip_var = float(np.std(hip_alignments)) if len(hip_alignments) > 1 else 0.0
                avg_conf = float(np.mean(confidences)) if confidences else 0.0

                # Stability score: penalise high deviation and high variation
                # Perfect stability = 100; deduct points proportionally
                dev_penalty = min(avg_dev * 3.0, 40.0)   # up to 40 pts for body deviation
                var_penalty = min(shoulder_var * 1.5, 30.0)  # up to 30 pts for shoulder variation
                conf_bonus = avg_conf * 30.0               # up to 30 pts for high confidence
                stability_score = max(0.0, min(100.0, conf_bonus + 70.0 - dev_penalty - var_penalty))

                stability = StabilityMetrics(
                    average_body_line_deviation=round(avg_dev, 2),
                    max_body_line_deviation=round(max_dev, 2),
                    shoulder_position_variation=round(shoulder_var, 2),
                    hip_position_variation=round(hip_var, 2),
                    average_pose_confidence=round(avg_conf, 3),
                    stability_score=round(stability_score, 1),
                )

            # 7. Build Video Metadata object
            video_metadata = VideoMetadata(
                duration_ms=duration_ms,
                width=video_width,
                height=video_height,
                original_fps=round(original_fps, 2),
                total_frames=total_frames,
                analyzed_frames=analyzed_frames_count,
                sampling_fps=effective_sample_fps,
                file_size_bytes=file_size_bytes,
            )

            # Compile overall coaching feedback
            summary_feedback: List[str] = []
            key_recommendations: List[str] = []

            if summary.stable_hold_duration_sec >= 2.0:
                summary_feedback.append(
                    f"Solid {target_stage.value} hold! Maintained stable form for {summary.stable_hold_duration_sec}s."
                )
            elif summary.stable_hold_duration_sec > 0:
                summary_feedback.append(
                    f"Brief {target_stage.value} hold detected ({summary.stable_hold_duration_sec}s). Work on endurance to sustain the hold for >= 2.0s."
                )
            elif summary.total_hold_duration_sec > 0:
                summary_feedback.append(
                    f"Planche attempt detected ({summary.total_hold_duration_sec}s), but form did not stabilize into a valid hold for {target_stage.value}."
                )
            else:
                summary_feedback.append(f"No sustained {target_stage.value} hold detected in video.")

            # Recommendations based on aggregated component deficits
            if aggregated_components.get("elbow_extension", 100.0) < 70.0:
                key_recommendations.append("Work on straight-arm strength with planche leans and locked-elbow pseudo planche pushups.")
            if aggregated_components.get("hip_extension", 100.0) < 70.0:
                key_recommendations.append("Practice reverse hyperextensions and hollow body holds to reinforce hip extension and prevent piking.")
            if aggregated_components.get("shoulder_position", 100.0) < 70.0:
                key_recommendations.append("Increase forward shoulder lean over wrists to properly counterbalance bodyweight.")

            if not key_recommendations and summary.average_score < 70.0 and summary.total_hold_duration_sec > 0:
                key_recommendations.append("Focus on foundational planche strength and progressive overload.")

            proc_time = (time.perf_counter() - start_time) * 1000.0
            return VideoAnalysisResponse(
                success=True,
                message=f"Successfully analyzed video sequence ({'side-view' if side_view else 'standard'}) for target progression: {target_stage.value}.",
                processing_time_ms=round(proc_time, 2),
                total_frames=total_frames,
                fps=round(original_fps, 2),
                duration_sec=round(duration_sec, 2),
                video_metadata=video_metadata,
                total_hold_duration_sec=summary.total_hold_duration_sec,
                valid_hold_duration_sec=summary.stable_hold_duration_sec,
                peak_form_score=summary.peak_score,
                average_form_score=summary.average_score,
                primary_progression=target_stage,
                side_view=side_view,
                component_scores=aggregated_components if aggregated_components else None,
                stability=stability,
                timeline=timeline,
                summary_feedback=summary_feedback,
                key_recommendations=key_recommendations,
                debug_info={
                    "side_view": side_view,
                    "target_progression": target_stage.value,
                    "stable_frames_count": summary.stable_frames_count,
                    "potential_frames_count": summary.potential_frames_count,
                    "aggregated_components": aggregated_components
                } if debug else None
            )

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
