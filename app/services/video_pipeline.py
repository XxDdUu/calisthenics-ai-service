import logging
import math
import os
import shutil
import subprocess
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import settings
from app.core.logging_context import get_request_id, set_request_id
from app.biomechanics.feedback_engine import FeedbackEngine
from app.biomechanics.scoring_engine import ExerciseScoringEngine
from app.exercises.registry import exercise_registry
from app.pose.tracking import PersonTracker
from app.pose.yolo_pose import COCO_KEYPOINTS, YOLOPoseEstimator
from app.schemas.analysis import (
    KeypointData,
    PoseKeypoints,
    VideoKeypointFrame,
    VideoKeypointItem,
    VideoKeypointsData,
    VideoOutputMetadata,
    VideoScoringResponse,
)

logger = logging.getLogger(__name__)

# Complete COCO-17 skeleton bone connections
COCO_SKELETON_BONES = [
    # Head & Face
    ("nose", "left_eye"),
    ("nose", "right_eye"),
    ("left_eye", "left_ear"),
    ("right_eye", "right_ear"),
    # Upper Body
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    # Torso
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    # Lower Body
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Color palette: BGR format (High contrast on dark and light backgrounds)
BONE_COLOR = (0, 255, 128)        # Neon Spring Green
JOINT_FILL_COLOR = (0, 215, 255)   # Amber Gold
JOINT_BORDER_COLOR = (255, 255, 255) # Pure White


class VideoScoringPipeline:
    """
    End-to-end video analysis pipeline:
    - Frame extraction with timestamp preservation
    - Configurable frame sampling for efficient inference
    - YOLOv8 Pose estimation & 17 COCO keypoint normalization
    - Biomechanical form scoring & feedback generation
    - Full-video skeleton interpolation & high-contrast rendering
    - H.264 MP4 encoding with audio preservation (via FFmpeg/OpenCV)
    """

    def __init__(self, pose_estimator: Optional[YOLOPoseEstimator] = None):
        self.pose_estimator = pose_estimator or YOLOPoseEstimator()
        self.temp_dir = settings.TEMP_DIR
        os.makedirs(self.temp_dir, exist_ok=True)
        # In-memory storage for recently generated analyses
        self._analysis_cache: Dict[str, VideoScoringResponse] = {}
        self._video_path_cache: Dict[str, str] = {}

    def get_cached_analysis(self, analysis_id: str) -> Optional[VideoScoringResponse]:
        return self._analysis_cache.get(analysis_id)

    def get_annotated_video_path(self, analysis_id: str) -> Optional[str]:
        path = self._video_path_cache.get(analysis_id)
        if path and os.path.exists(path):
            return path
        # Fallback check disk
        fallback = os.path.join(self.temp_dir, f"annotated_{analysis_id}.mp4")
        if os.path.exists(fallback):
            return fallback
        return None

    def process_video(
        self,
        video_bytes: bytes,
        exercise_name: Optional[str] = None,
        exercise_id: Optional[Any] = None,
        sample_fps: int = 10,
        confidence_threshold: float = 0.25,
        side_view: bool = False,
        analysis_id: Optional[str] = None,
    ) -> VideoScoringResponse:
        """Process video and generate annotated MP4 + scoring results."""
        start_time = time.perf_counter()
        aid = analysis_id or str(uuid.uuid4())
        rid = aid[:8]
        set_request_id(rid)

        logger.info(
            "[VIDEO_PIPELINE] Starting video analysis aid=%s exercise=%s sample_fps=%d bytes=%d",
            aid, exercise_name, sample_fps, len(video_bytes)
        )

        input_path = os.path.join(self.temp_dir, f"input_{aid}.mp4")
        raw_annotated_path = os.path.join(self.temp_dir, f"raw_annotated_{aid}.mp4")
        final_annotated_path = os.path.join(self.temp_dir, f"annotated_{aid}.mp4")

        with open(input_path, "wb") as f:
            f.write(video_bytes)

        try:
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error("[VIDEO_PIPELINE] aid=%s failed to open video file", aid)
                return self._create_failed_response(
                    aid=aid,
                    exercise_name=exercise_name or "PLANCHE",
                    exercise_id=exercise_id,
                    status="INVALID_INPUT",
                    message="Failed to decode video file format."
                )

            original_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = total_frames / original_fps if original_fps > 0 else 0.0
            duration_ms = int(duration_sec * 1000)

            logger.info(
                "[VIDEO_PIPELINE] aid=%s video_meta: %dx%d @ %.2f fps, frames=%d, duration=%.2fs",
                aid, width, height, original_fps, total_frames, duration_sec
            )

            if total_frames <= 0 or width <= 0 or height <= 0:
                cap.release()
                return self._create_failed_response(
                    aid=aid,
                    exercise_name=exercise_name or "PLANCHE",
                    exercise_id=exercise_id,
                    status="INVALID_INPUT",
                    message="Video has invalid dimensions or zero frames."
                )

            # Sampling step
            step = max(1, int(round(original_fps / max(1, sample_fps))))

            # Pass 1: Sampled frames inference & keypoint extraction
            sampled_poses: Dict[int, Dict[str, VideoKeypointItem]] = {}
            keypoint_frames: List[VideoKeypointFrame] = []
            frame_scores: List[float] = []
            frame_issues: List[str] = []
            frame_positive: List[str] = []
            measurements_history: Dict[str, List[float]] = {}

            analyzer = exercise_registry.get_analyzer(exercise_name)
            tracker = PersonTracker()
            scoring_engine = ExerciseScoringEngine()
            feedback_engine = FeedbackEngine()
            logger.info("[VIDEO_PIPELINE] aid=%s resolved analyzer=%s", aid, analyzer.name)

            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % step == 0:
                    timestamp_ms = int((frame_idx / original_fps) * 1000)
                    now_sec = timestamp_ms / 1000.0
                    success_pose, pose_kp = self.pose_estimator.estimate_pose(frame, tracker=tracker)

                    if success_pose and pose_kp is not None and pose_kp.keypoints:
                        # Extract 17 COCO keypoints (normalized 0-1)
                        norm_kps: Dict[str, VideoKeypointItem] = {}
                        keypoint_items: List[VideoKeypointItem] = []

                        for name in COCO_KEYPOINTS:
                            kp_data = pose_kp.keypoints.get(name)
                            if kp_data is not None:
                                norm_x = max(0.0, min(1.0, kp_data.x / width))
                                norm_y = max(0.0, min(1.0, kp_data.y / height))
                                conf = float(kp_data.confidence)
                            else:
                                norm_x, norm_y, conf = 0.0, 0.0, 0.0

                            item = VideoKeypointItem(
                                name=name,
                                x=round(norm_x, 4),
                                y=round(norm_y, 4),
                                confidence=round(conf, 4)
                            )
                            norm_kps[name] = item
                            keypoint_items.append(item)

                        sampled_poses[frame_idx] = norm_kps
                        keypoint_frames.append(VideoKeypointFrame(
                            frameIndex=frame_idx,
                            timestampMs=timestamp_ms,
                            keypoints=keypoint_items
                        ))

                        # Evaluate biomechanics for exercise
                        measurements = analyzer.extract_measurements(pose_kp.keypoints, side_view=side_view)
                        for m_key, m_val in measurements.items():
                            measurements_history.setdefault(m_key, []).append(m_val)

                        f_score, feedback_items, is_valid = analyzer.evaluate_live_form(
                            measurements=measurements,
                            side_view=side_view
                        )

                        # Detailed exercise scoring metrics
                        scoring_metrics = analyzer.evaluate_scoring_metrics(
                            measurements=measurements,
                            keypoints=pose_kp.keypoints,
                            side_view=side_view
                        )
                        calculated_score, _, _, _ = scoring_engine.evaluate_score(
                            metrics=scoring_metrics,
                            keypoints=pose_kp.keypoints,
                            required_landmarks=analyzer.required_landmarks
                        )

                        chosen_score = calculated_score if calculated_score is not None else f_score
                        if chosen_score is not None:
                            frame_scores.append(chosen_score)

                        # Process feedback with deduplication
                        processed_fb = feedback_engine.process_feedback(feedback_items, timestamp_sec=now_sec)
                        for fb in processed_fb:
                            if fb.severity.name in ("WARNING", "ERROR"):
                                frame_issues.append(fb.message)
                            elif fb.severity.name == "INFO":
                                frame_positive.append(fb.message)

                frame_idx += 1

            cap.release()

            # Pass 1 Evaluation summary
            total_sampled = len(keypoint_frames)
            logger.info("[VIDEO_PIPELINE] aid=%s pass1_done sampled_frames=%d with_pose=%d", aid, frame_idx // step, total_sampled)

            if total_sampled == 0:
                logger.warning("[VIDEO_PIPELINE] aid=%s no person detected in any sampled frame", aid)
                return self._create_failed_response(
                    aid=aid,
                    exercise_name=analyzer.name,
                    exercise_id=exercise_id,
                    status="NO_PERSON_DETECTED",
                    message="No human pose detected in the video.",
                    width=width,
                    height=height,
                    fps=original_fps,
                    duration_ms=duration_ms
                )

            # Compute Biomechanical Metrics & Form Scores
            form_accuracy = float(np.mean(frame_scores)) if frame_scores else 50.0

            # Stability: based on variance of measurements
            stability_score = 80.0
            if measurements_history:
                stds = [float(np.std(vals)) for vals in measurements_history.values() if len(vals) > 1]
                avg_std = float(np.mean(stds)) if stds else 0.0
                stability_score = max(20.0, min(100.0, 100.0 - (avg_std * 2.5)))

            # Alignment: based on torso / body line straightness
            alignment_score = 80.0
            if "body_line_angle" in measurements_history:
                body_lines = measurements_history["body_line_angle"]
                avg_line = float(np.mean(body_lines))
                # deviation from 180° straight line
                dev = abs(180.0 - avg_line)
                alignment_score = max(20.0, min(100.0, 100.0 - (dev * 2.0)))
            elif "horizontal_tilt" in measurements_history:
                tilts = measurements_history["horizontal_tilt"]
                avg_tilt = float(np.mean(tilts))
                alignment_score = max(20.0, min(100.0, 100.0 - (avg_tilt * 2.0)))

            # Overall Score: weighted composite
            overall_score = round(
                0.50 * form_accuracy + 0.25 * stability_score + 0.25 * alignment_score,
                1
            )
            overall_score = max(0.0, min(100.0, overall_score))

            # Compile top feedback
            unique_issues = list(dict.fromkeys(frame_issues))[:4]
            unique_positives = list(dict.fromkeys(frame_positive))[:2]
            feedback_list = []
            if unique_issues:
                feedback_list.extend(unique_issues)
            else:
                feedback_list.append("Excellent execution! Form remained consistent and controlled.")
            if unique_positives:
                feedback_list.extend(unique_positives)

            # Average exercise metrics
            avg_metrics: Dict[str, Any] = {
                "formAccuracy": round(form_accuracy, 1),
                "stability": round(stability_score, 1),
                "alignment": round(alignment_score, 1),
                "analyzedFrames": total_sampled,
                "samplingFps": sample_fps,
            }
            for k, v in measurements_history.items():
                if v:
                    avg_metrics[k] = round(float(np.mean(v)), 1)

            # Pass 2: Render skeleton directly onto frames with smooth interpolation
            self._render_annotated_video(
                input_path=input_path,
                raw_annotated_path=raw_annotated_path,
                final_annotated_path=final_annotated_path,
                sampled_poses=sampled_poses,
                width=width,
                height=height,
                original_fps=original_fps,
                total_frames=total_frames,
                sample_step=step,
                conf_threshold=confidence_threshold
            )

            proc_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.info(
                "[VIDEO_PIPELINE] aid=%s completed successfully score=%.1f proc_time=%.2fms",
                aid, overall_score, proc_time_ms
            )

            response = VideoScoringResponse(
                analysisId=aid,
                status="COMPLETED",
                exerciseId=exercise_id,
                exercise=analyzer.name,
                score=overall_score,
                formAccuracy=round(form_accuracy, 1),
                stability=round(stability_score, 1),
                alignment=round(alignment_score, 1),
                metrics=avg_metrics,
                feedback=feedback_list,
                video=VideoOutputMetadata(
                    originalAvailable=True,
                    annotatedAvailable=True,
                    originalObjectKey=f"orig_{aid}.mp4",
                    annotatedObjectKey=f"annotated_{aid}.mp4",
                    durationMs=duration_ms,
                    width=width,
                    height=height,
                    fps=round(original_fps, 2)
                ),
                keypoints=VideoKeypointsData(
                    format="COCO_17",
                    coordinateSystem="NORMALIZED",
                    frames=keypoint_frames
                )
            )

            # Cache response and file paths
            self._analysis_cache[aid] = response
            self._video_path_cache[aid] = final_annotated_path

            return response

        finally:
            # Clean up raw temporary files
            if os.path.exists(input_path):
                try:
                    os.remove(input_path)
                except OSError:
                    pass
            if os.path.exists(raw_annotated_path):
                try:
                    os.remove(raw_annotated_path)
                except OSError:
                    pass

    def _render_annotated_video(
        self,
        input_path: str,
        raw_annotated_path: str,
        final_annotated_path: str,
        sampled_poses: Dict[int, Dict[str, VideoKeypointItem]],
        width: int,
        height: int,
        original_fps: float,
        total_frames: int,
        sample_step: int,
        conf_threshold: float
    ) -> None:
        """Render skeleton overlay on every frame with interpolation and encode to MP4."""
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(raw_annotated_path, fourcc, original_fps, (width, height))

        sorted_sampled_indices = sorted(sampled_poses.keys())
        radius = max(4, int(min(width, height) * 0.007))
        thickness = max(2, int(min(width, height) * 0.004))

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Find active pose for this frame (interpolated if between sampled frames)
            active_pose = self._interpolate_pose_for_frame(
                frame_idx=frame_idx,
                sampled_indices=sorted_sampled_indices,
                sampled_poses=sampled_poses,
                max_gap=sample_step * 3
            )

            if active_pose is not None:
                self._draw_skeleton_on_frame(
                    frame=frame,
                    keypoints=active_pose,
                    width=width,
                    height=height,
                    conf_threshold=conf_threshold,
                    radius=radius,
                    thickness=thickness
                )

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()

        # Transcode to H.264 (yuv420p) with AAC audio using FFmpeg
        self._transcode_to_h264_with_audio(
            raw_annotated_path=raw_annotated_path,
            original_input_path=input_path,
            output_path=final_annotated_path
        )

    def _interpolate_pose_for_frame(
        self,
        frame_idx: int,
        sampled_indices: List[int],
        sampled_poses: Dict[int, Dict[str, VideoKeypointItem]],
        max_gap: int
    ) -> Optional[Dict[str, VideoKeypointItem]]:
        """Interpolate keypoint positions between adjacent sampled frames."""
        if frame_idx in sampled_poses:
            return sampled_poses[frame_idx]

        if not sampled_indices:
            return None

        # Binary search or scan for surrounding sampled frames
        prev_idx = None
        next_idx = None
        for idx in sampled_indices:
            if idx <= frame_idx:
                prev_idx = idx
            elif idx > frame_idx:
                next_idx = idx
                break

        if prev_idx is not None and next_idx is not None:
            if (next_idx - prev_idx) <= max_gap:
                alpha = (frame_idx - prev_idx) / float(next_idx - prev_idx)
                pose_a = sampled_poses[prev_idx]
                pose_b = sampled_poses[next_idx]

                interpolated: Dict[str, VideoKeypointItem] = {}
                for name in COCO_KEYPOINTS:
                    ka = pose_a.get(name)
                    kb = pose_b.get(name)
                    if ka and kb and ka.confidence > 0.1 and kb.confidence > 0.1:
                        interp_x = (1.0 - alpha) * ka.x + alpha * kb.x
                        interp_y = (1.0 - alpha) * ka.y + alpha * kb.y
                        interp_c = (1.0 - alpha) * ka.confidence + alpha * kb.confidence
                        interpolated[name] = VideoKeypointItem(
                            name=name,
                            x=interp_x,
                            y=interp_y,
                            confidence=interp_c
                        )
                    elif ka and ka.confidence > 0.1:
                        interpolated[name] = ka
                    elif kb and kb.confidence > 0.1:
                        interpolated[name] = kb
                return interpolated

        if prev_idx is not None and (frame_idx - prev_idx) <= max_gap:
            return sampled_poses[prev_idx]
        if next_idx is not None and (next_idx - frame_idx) <= max_gap:
            return sampled_poses[next_idx]

        return None

    def _draw_skeleton_on_frame(
        self,
        frame: np.ndarray,
        keypoints: Dict[str, VideoKeypointItem],
        width: int,
        height: int,
        conf_threshold: float,
        radius: int,
        thickness: int
    ) -> None:
        """Render high contrast bones and joint circles on the frame."""
        # 1. Draw Bones
        for joint_a, joint_b in COCO_SKELETON_BONES:
            kp_a = keypoints.get(joint_a)
            kp_b = keypoints.get(joint_b)
            if (
                kp_a is not None
                and kp_b is not None
                and kp_a.confidence >= conf_threshold
                and kp_b.confidence >= conf_threshold
            ):
                pt_a = (int(round(kp_a.x * width)), int(round(kp_a.y * height)))
                pt_b = (int(round(kp_b.x * width)), int(round(kp_b.y * height)))

                # Draw subtle dark outline for maximum contrast on bright backgrounds
                cv2.line(frame, pt_a, pt_b, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                # Draw neon bone line
                cv2.line(frame, pt_a, pt_b, BONE_COLOR, thickness, cv2.LINE_AA)

        # 2. Draw Joints
        for joint_name, kp in keypoints.items():
            if kp is not None and kp.confidence >= conf_threshold:
                center = (int(round(kp.x * width)), int(round(kp.y * height)))
                # Outer black border for high contrast
                cv2.circle(frame, center, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
                # Inner white ring
                cv2.circle(frame, center, radius + 1, JOINT_BORDER_COLOR, 1, cv2.LINE_AA)
                # Filled joint marker
                cv2.circle(frame, center, radius, JOINT_FILL_COLOR, -1, cv2.LINE_AA)

    def _transcode_to_h264_with_audio(
        self,
        raw_annotated_path: str,
        original_input_path: str,
        output_path: str
    ) -> None:
        """Transcode raw OpenCV video to H.264 (yuv420p) + AAC audio with faststart."""
        ffmpeg_bin = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        if not os.path.exists(ffmpeg_bin):
            logger.warning("[VIDEO_PIPELINE] FFmpeg not found, falling back to raw OpenCV video")
            shutil.copyfile(raw_annotated_path, output_path)
            return

        # Attempt 1: Mux annotated video with original audio
        cmd_with_audio = [
            ffmpeg_bin, "-y",
            "-i", raw_annotated_path,
            "-i", original_input_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "22",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-movflags", "+faststart",
            output_path
        ]

        try:
            result = subprocess.run(
                cmd_with_audio,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=90
            )
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.debug("[VIDEO_PIPELINE] FFmpeg transcode with audio succeeded")
                return
        except Exception as e:
            logger.warning("[VIDEO_PIPELINE] FFmpeg transcode with audio failed: %s, retrying video-only", e)

        # Attempt 2: Video-only transcode without audio
        cmd_video_only = [
            ffmpeg_bin, "-y",
            "-i", raw_annotated_path,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "22",
            "-movflags", "+faststart",
            output_path
        ]

        try:
            result = subprocess.run(
                cmd_video_only,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=90
            )
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.debug("[VIDEO_PIPELINE] FFmpeg video-only transcode succeeded")
                return
        except Exception as e:
            logger.warning("[VIDEO_PIPELINE] FFmpeg video-only transcode failed: %s", e)

        # Fallback: Copy raw video
        shutil.copyfile(raw_annotated_path, output_path)

    def _create_failed_response(
        self,
        aid: str,
        exercise_name: str,
        exercise_id: Optional[Any],
        status: str,
        message: str,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        duration_ms: int = 0
    ) -> VideoScoringResponse:
        return VideoScoringResponse(
            analysisId=aid,
            status=status,
            exerciseId=exercise_id,
            exercise=exercise_name,
            score=0.0,
            formAccuracy=0.0,
            stability=0.0,
            alignment=0.0,
            metrics={},
            feedback=[message],
            video=VideoOutputMetadata(
                originalAvailable=True,
                annotatedAvailable=False,
                originalObjectKey=f"orig_{aid}.mp4",
                annotatedObjectKey=None,
                durationMs=duration_ms,
                width=width,
                height=height,
                fps=fps
            ),
            keypoints=VideoKeypointsData(
                format="COCO_17",
                coordinateSystem="NORMALIZED",
                frames=[]
            )
        )


# Singleton pipeline instance
video_scoring_pipeline = VideoScoringPipeline()
