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
from app.exercises.planche import PlancheAnalyzer
from app.pose.yolo_pose import YOLOPoseEstimator
from app.schemas.analysis import (
    ImageAnalysisResponse,
    PlancheAnalysisResult,
    PoseKeypoints,
    ProgressionStage,
    VideoAnalysisResponse,
    VideoFrameAnalysis,
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
        debug: bool = False
    ) -> ImageAnalysisResponse:
        """
        Analyzes a single image for user-selected Planche progression form evaluation.
        Explicitly branches between side-view midpoint processing and standard processing.
        """
        start_time = time.perf_counter()
        target_stage = parse_progression_stage(target_progression)

        # 1. Decode image bytes to numpy array
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

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

        # 5. Render Annotated Overlay Image
        annotated_b64 = None
        if draw_overlay:
            # In side-view mode, pass updated keypoints with midpoints if available
            render_pose = pose_kp
            if side_view and profile.midpoints:
                combined_dict = dict(pose_kp.keypoints)
                combined_dict.update(profile.midpoints)
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
        side_view: bool = False
    ) -> Tuple[bytes, ImageAnalysisResponse]:
        """
        Runs pose estimation & form evaluation on image_bytes against target_progression,
        draws skeleton overlay onto image, and returns raw JPEG bytes of annotated image.
        """
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image_np is None:
            raise ValueError("Invalid or corrupt image format.")

        response = self.analyze_image(
            image_bytes=image_bytes,
            target_progression=target_progression,
            side_view=side_view,
            draw_overlay=True
        )

        if not response.pose_detected or response.pose_keypoints is None:
            _, buffer = cv2.imencode(".jpg", image_np)
            return buffer.tobytes(), response

        planche_result = response.planche_analysis
        render_pose = response.pose_keypoints
        if side_view and planche_result and planche_result.debug_info and "midpoints" in planche_result.debug_info:
            midpoints_raw = planche_result.debug_info["midpoints"]
            from app.schemas.analysis import KeypointData
            combined_dict = dict(render_pose.keypoints)
            for m_name, m_val in midpoints_raw.items():
                combined_dict[m_name] = KeypointData(
                    x=m_val["x"], y=m_val["y"], confidence=m_val["confidence"], name=m_name, visible=True
                )
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
        """
        start_time = time.perf_counter()
        target_stage = parse_progression_stage(target_progression)

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

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration_sec = total_frames / fps if fps > 0 else 0.0

            sample_interval = max(1, int(fps / target_sample_fps))
            effective_sample_fps = fps / sample_interval

            # Initialize Temporal Smoother and Hold Detector
            smoother = TemporalKeypointSmoother(alpha=settings.SMOOTHING_ALPHA)
            detector = PlancheHoldDetector(
                fps=effective_sample_fps,
                min_hold_duration_sec=settings.MIN_HOLD_DURATION_SEC,
                stability_variance_threshold=settings.HOLD_STABILITY_VARIANCE_THRESHOLD
            )

            frame_idx = 0
            timeline: List[VideoFrameAnalysis] = []
            stable_component_scores: List[Dict[str, float]] = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_interval == 0:
                    timestamp_sec = frame_idx / fps
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
                                key_issues=res.form_errors
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
                                key_issues=["Insufficient pose quality / occluded joints"]
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
                            key_issues=["No pose detected"]
                        ))

                frame_idx += 1

            cap.release()

            # 5. Hold Detection Aggregation (No single frame scoring)
            summary = detector.get_summary(sample_interval=1)

            # Compute aggregated component scores
            aggregated_components: Dict[str, float] = {}
            if stable_component_scores:
                for k in stable_component_scores[0].keys():
                    vals = [c[k] for c in stable_component_scores if k in c]
                    aggregated_components[k] = round(float(np.mean(vals)), 1)
            elif timeline:
                # Fallback to attempt frames if no stable hold
                attempt_scores = [f.form_score for f in timeline if f.pose_detected and f.form_score > 0]
                if attempt_scores:
                    aggregated_components["overall"] = round(float(np.mean(attempt_scores)), 1)

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
                fps=round(fps, 2),
                duration_sec=round(duration_sec, 2),
                total_hold_duration_sec=summary.total_hold_duration_sec,
                valid_hold_duration_sec=summary.stable_hold_duration_sec,
                peak_form_score=summary.peak_score,
                average_form_score=summary.average_score,
                primary_progression=target_stage,
                side_view=side_view,
                component_scores=aggregated_components if aggregated_components else None,
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
