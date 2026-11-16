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
from app.biomechanics.camera_quality import CameraQualityEvaluator
from app.biomechanics.confidence import check_pose_quality_for_planche
from app.biomechanics.feedback_engine import FeedbackEngine
from app.biomechanics.scoring_engine import ExerciseScoringEngine
from app.biomechanics.smoothing import TemporalKeypointSmoother
from app.exercises.registry import exercise_registry
from app.pose.tracking import PersonTracker
from app.pose.yolo_pose import COCO_KEYPOINTS, YOLOPoseEstimator
from app.schemas.analysis import (
    CameraQualityAssessment,
    IssueSeverity,
    KeypointData,
    MovementMetricItem,
    PoseKeypoints,
    RepBreakdownItem,
    StructuredIssueItem,
    VideoKeypointFrame,
    VideoKeypointItem,
    VideoKeypointsData,
    VideoOutputMetadata,
    VideoScoringResponse,
)

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0"
MAX_VIDEO_DURATION_SEC = 600.0
MAX_VIDEO_FRAMES = 36000
MAX_CACHE_ENTRIES = 200

_NON_VIDEO_MAGIC_PREFIXES = (
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"GIF87a",
    b"GIF89a",
    b"%PDF",
    b"PK\x03\x04",
    b"<!DOCTYPE",
    b"<html",
    b"<?xml",
)


def validate_video_bytes(
    video_bytes: Optional[bytes],
    max_size_mb: Optional[int] = None,
) -> Tuple[bool, str]:
    """
    Validate raw uploaded video bytes for emptiness, size limits, and container magic headers.
    Returns (is_valid, error_message).
    """
    if not video_bytes or not isinstance(video_bytes, (bytes, bytearray)):
        return False, "Empty video file uploaded."

    size = len(video_bytes)
    if size < 32:
        return False, "Video payload is too small or truncated to be a valid video."

    limit_mb = max_size_mb if max_size_mb is not None else settings.MAX_UPLOAD_SIZE_MB
    max_bytes = max(1, int(limit_mb)) * 1024 * 1024
    if size > max_bytes:
        return False, f"Video file exceeds maximum allowed size of {limit_mb} MB."

    head = bytes(video_bytes[:32])
    head_lower = head.lstrip().lower()
    for bad_prefix in _NON_VIDEO_MAGIC_PREFIXES:
        if head.startswith(bad_prefix) or head_lower.startswith(bad_prefix.lower()):
            return False, "Uploaded file is not a valid video format."

    # Check common video container magic headers
    is_mp4_mov = len(head) >= 12 and head[4:8] in (
        b"ftyp", b"moov", b"mdat", b"free", b"wide", b"skip", b"pnot"
    )
    is_avi = head[:4] == b"RIFF" and len(head) >= 12 and head[8:12] == b"AVI "
    is_mkv_webm = head[:4] == b"\x1a\x45\xdf\xa3"
    is_flv = head[:3] == b"FLV"
    is_asf_wmv = head[:4] == b"\x30\x26\xb2\x75"
    is_ogg = head[:4] == b"OggS"
    is_mpeg_annex_b = head[:4] in (
        b"\x00\x00\x00\x01", b"\x00\x00\x01\xba", b"\x00\x00\x01\xb3"
    ) or head[:3] == b"\x00\x00\x01"
    is_mpeg_ts = len(video_bytes) >= 376 and video_bytes[0] == 0x47 and video_bytes[188] == 0x47

    if not any((is_mp4_mov, is_avi, is_mkv_webm, is_flv, is_asf_wmv, is_ogg, is_mpeg_annex_b, is_mpeg_ts)):
        return False, "Unsupported or corrupted video container format."

    return True, ""


def _round_optional(val: Optional[float], digits: int = 1) -> Optional[float]:
    if val is None:
        return None
    f = float(val)
    if not math.isfinite(f):
        return None
    return round(f, digits)

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

        analyzer = exercise_registry.get_analyzer(exercise_name)
        resolved_exercise_name = analyzer.name

        logger.info(
            "[VIDEO_PIPELINE] Starting video analysis aid=%s exercise=%s resolved=%s sample_fps=%d bytes=%d",
            aid, exercise_name, resolved_exercise_name, sample_fps, len(video_bytes) if video_bytes else 0
        )

        # 0. Raw bytes & container header validation
        bytes_ok, validation_err = validate_video_bytes(video_bytes)
        if not bytes_ok:
            logger.warning("[VIDEO_PIPELINE] aid=%s invalid video bytes: %s", aid, validation_err)
            return self._create_failed_response(
                aid=aid,
                exercise_name=resolved_exercise_name,
                exercise_id=exercise_id,
                status="INVALID_INPUT",
                message=validation_err,
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
                    exercise_name=resolved_exercise_name,
                    exercise_id=exercise_id,
                    status="INVALID_INPUT",
                    message="Failed to decode video file format.",
                )

            raw_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if raw_fps is None or not math.isfinite(raw_fps) or raw_fps < 0.1 or raw_fps > 240.0:
                cap.release()
                logger.warning("[VIDEO_PIPELINE] aid=%s invalid fps=%.4f", aid, raw_fps or 0.0)
                return self._create_failed_response(
                    aid=aid,
                    exercise_name=resolved_exercise_name,
                    exercise_id=exercise_id,
                    status="INVALID_INPUT",
                    message="Video has an invalid or unsupported frame rate.",
                    width=max(0, width),
                    height=max(0, height),
                )

            original_fps = float(raw_fps)
            duration_sec = total_frames / original_fps if original_fps > 0 else 0.0
            duration_ms = int(round(duration_sec * 1000.0))

            logger.info(
                "[VIDEO_PIPELINE] aid=%s video_meta: %dx%d @ %.2f fps, frames=%d, duration=%.2fs",
                aid, width, height, original_fps, total_frames, duration_sec
            )

            if total_frames < 2 or width <= 0 or height <= 0:
                cap.release()
                return self._create_failed_response(
                    aid=aid,
                    exercise_name=resolved_exercise_name,
                    exercise_id=exercise_id,
                    status="INVALID_INPUT",
                    message="Video must have valid dimensions and at least 2 frames.",
                    width=max(0, width),
                    height=max(0, height),
                    fps=round(original_fps, 2),
                    duration_ms=max(0, duration_ms),
                )

            if duration_sec > MAX_VIDEO_DURATION_SEC or total_frames > MAX_VIDEO_FRAMES:
                cap.release()
                return self._create_failed_response(
                    aid=aid,
                    exercise_name=resolved_exercise_name,
                    exercise_id=exercise_id,
                    status="INVALID_INPUT",
                    message=f"Video exceeds maximum supported duration ({int(MAX_VIDEO_DURATION_SEC)}s) or frame count.",
                    width=width,
                    height=height,
                    fps=round(original_fps, 2),
                    duration_ms=duration_ms,
                )

            # Sampling configuration
            clamped_sample_fps = max(1, min(60, int(sample_fps)))
            step = max(1, int(round(original_fps / clamped_sample_fps)))
            effective_sample_fps = round(original_fps / step, 2)

            # Pass 1: Efficient frame sampling with grab() skipping, pose inference, smoothing, and normalization
            sampled_poses: Dict[int, Dict[str, VideoKeypointItem]] = {}
            keypoint_frames: List[VideoKeypointFrame] = []
            frame_samples: List[Dict[str, Any]] = []

            tracker = PersonTracker()
            smoother = TemporalKeypointSmoother(alpha=settings.SMOOTHING_ALPHA)
            logger.info(
                "[VIDEO_PIPELINE] aid=%s resolved analyzer=%s step=%d effective_sample_fps=%.2f",
                aid, resolved_exercise_name, step, effective_sample_fps
            )

            frame_idx = 0
            decoded_frames_seen = 0
            while cap.isOpened():
                if frame_idx % step != 0:
                    if not cap.grab():
                        break
                    decoded_frames_seen += 1
                    frame_idx += 1
                    continue

                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                decoded_frames_seen += 1

                timestamp_ms = int(round((frame_idx / original_fps) * 1000.0))
                now_sec = frame_idx / original_fps
                success_pose, pose_kp = self.pose_estimator.estimate_pose(frame, tracker=tracker)

                if success_pose and pose_kp is not None and pose_kp.keypoints:
                    # Preserve all keypoints while synchronizing visibility with caller confidence_threshold
                    raw_kps: Dict[str, KeypointData] = {}
                    for kp_name, kp_val in pose_kp.keypoints.items():
                        finite_ok = (
                            math.isfinite(kp_val.x)
                            and math.isfinite(kp_val.y)
                            and math.isfinite(kp_val.confidence)
                        )
                        conf_val = max(0.0, min(1.0, float(kp_val.confidence))) if finite_ok else 0.0
                        raw_kps[kp_name] = KeypointData(
                            x=float(kp_val.x) if finite_ok else 0.0,
                            y=float(kp_val.y) if finite_ok else 0.0,
                            confidence=conf_val,
                            name=kp_val.name or kp_name,
                            visible=bool(finite_ok and kp_val.visible and conf_val >= confidence_threshold),
                        )

                    smoothed_kps = smoother.smooth(raw_kps)

                    # Extract 17 COCO keypoints (normalized 0-1) from smoothed keypoints
                    norm_kps: Dict[str, VideoKeypointItem] = {}
                    keypoint_items: List[VideoKeypointItem] = []

                    for name in COCO_KEYPOINTS:
                        kp_data = smoothed_kps.get(name)
                        if (
                            kp_data is not None
                            and math.isfinite(kp_data.x)
                            and math.isfinite(kp_data.y)
                            and math.isfinite(kp_data.confidence)
                        ):
                            norm_x = max(0.0, min(1.0, kp_data.x / width))
                            norm_y = max(0.0, min(1.0, kp_data.y / height))
                            conf = max(0.0, min(1.0, float(kp_data.confidence)))
                        else:
                            norm_x, norm_y, conf = 0.0, 0.0, 0.0

                        item = VideoKeypointItem(
                            name=name,
                            x=round(norm_x, 4),
                            y=round(norm_y, 4),
                            confidence=round(conf, 4),
                        )
                        norm_kps[name] = item
                        keypoint_items.append(item)

                    sampled_poses[frame_idx] = norm_kps
                    keypoint_frames.append(VideoKeypointFrame(
                        frameIndex=frame_idx,
                        timestampMs=timestamp_ms,
                        keypoints=keypoint_items,
                    ))

                    frame_samples.append({
                        "frame_index": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "timestamp_sec": now_sec,
                        "keypoints": smoothed_kps,
                        "raw_keypoints": raw_kps,
                        "pose_detected": True,
                        "detection_confidence": float(pose_kp.detection_confidence or 0.0),
                    })
                else:
                    smoother.reset()
                    frame_samples.append({
                        "frame_index": frame_idx,
                        "timestamp_ms": timestamp_ms,
                        "timestamp_sec": now_sec,
                        "keypoints": {},
                        "raw_keypoints": {},
                        "pose_detected": False,
                        "detection_confidence": 0.0,
                    })

                frame_idx += 1

            cap.release()

            if decoded_frames_seen < 2 or not frame_samples:
                logger.warning("[VIDEO_PIPELINE] aid=%s insufficient decodable frames (%d)", aid, decoded_frames_seen)
                return self._create_failed_response(
                    aid=aid,
                    exercise_name=resolved_exercise_name,
                    exercise_id=exercise_id,
                    status="INVALID_INPUT",
                    message="Failed to decode sufficient frames from video.",
                    width=width,
                    height=height,
                    fps=round(original_fps, 2),
                    duration_ms=duration_ms,
                )

            # Evaluate camera framing and pose quality across all sampled frames
            camera_quality = CameraQualityEvaluator.evaluate(
                frame_samples=frame_samples,
                exercise_name=resolved_exercise_name,
                conf_threshold=confidence_threshold,
            )

            total_sampled_with_pose = len(keypoint_frames)
            logger.info(
                "[VIDEO_PIPELINE] aid=%s pass1_done sampled_frames=%d with_pose=%d cam_quality=%s (%.1f)",
                aid, len(frame_samples), total_sampled_with_pose, camera_quality.overallQuality, camera_quality.score
            )

            if total_sampled_with_pose == 0:
                logger.warning("[VIDEO_PIPELINE] aid=%s no person detected in any sampled frame", aid)
                return self._create_failed_response(
                    aid=aid,
                    exercise_name=resolved_exercise_name,
                    exercise_id=exercise_id,
                    status="NO_PERSON_DETECTED",
                    message="No human pose detected in the video.",
                    width=width,
                    height=height,
                    fps=round(original_fps, 2),
                    duration_ms=duration_ms,
                    camera_quality=camera_quality,
                    recommendations=[analyzer.camera_view_recommendation, *camera_quality.warnings],
                    metrics={
                        "schemaVersion": SCHEMA_VERSION,
                        "repCount": 0,
                        "validReps": 0,
                        "incompleteReps": 0,
                        "analyzedFrames": 0,
                        "totalSampledFrames": len(frame_samples),
                        "samplingFps": effective_sample_fps,
                    },
                )

            # Delegate sequence analysis if analyzer supports the shared video sequence interface
            seq_res: Optional[Dict[str, Any]] = None
            analyze_seq_fn = getattr(analyzer, "analyze_video_sequence", None)
            if callable(analyze_seq_fn):
                try:
                    seq_res = analyze_seq_fn(
                        frame_samples=frame_samples,
                        fps=effective_sample_fps,
                        side_view=side_view,
                    )
                except Exception as exc:
                    logger.exception("[VIDEO_PIPELINE] aid=%s analyze_video_sequence failed: %s", aid, exc)
                    return self._create_failed_response(
                        aid=aid,
                        exercise_name=resolved_exercise_name,
                        exercise_id=exercise_id,
                        status="FAILED",
                        message=f"Exercise sequence analysis failed: {str(exc)}",
                        width=width,
                        height=height,
                        fps=round(original_fps, 2),
                        duration_ms=duration_ms,
                        camera_quality=camera_quality,
                        keypoints=VideoKeypointsData(
                            format="COCO_17",
                            coordinateSystem="NORMALIZED",
                            frames=keypoint_frames,
                        ),
                    )

            if seq_res is not None:
                response = self._build_response_from_sequence_result(
                    aid=aid,
                    exercise_id=exercise_id,
                    default_exercise=resolved_exercise_name,
                    seq_res=seq_res,
                    camera_quality=camera_quality,
                    keypoint_frames=keypoint_frames,
                    total_sampled_frames=len(frame_samples),
                    effective_sample_fps=effective_sample_fps,
                    width=width,
                    height=height,
                    original_fps=original_fps,
                    duration_ms=duration_ms,
                )
            else:
                response = self._run_legacy_fallback_analysis(
                    aid=aid,
                    exercise_id=exercise_id,
                    analyzer=analyzer,
                    frame_samples=frame_samples,
                    camera_quality=camera_quality,
                    keypoint_frames=keypoint_frames,
                    effective_sample_fps=effective_sample_fps,
                    confidence_threshold=confidence_threshold,
                    side_view=side_view,
                    width=width,
                    height=height,
                    original_fps=original_fps,
                    duration_ms=duration_ms,
                )

            # Pass 2: Render skeleton only for COMPLETED analyses
            annotated_ok = False
            if response.status == "COMPLETED" and sampled_poses:
                annotated_ok = self._render_annotated_video(
                    input_path=input_path,
                    raw_annotated_path=raw_annotated_path,
                    final_annotated_path=final_annotated_path,
                    sampled_poses=sampled_poses,
                    width=width,
                    height=height,
                    original_fps=original_fps,
                    total_frames=total_frames,
                    sample_step=step,
                    conf_threshold=confidence_threshold,
                )

            response.video.annotatedAvailable = annotated_ok
            response.video.annotatedObjectKey = f"annotated_{aid}.mp4" if annotated_ok else None

            proc_time_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
            logger.info(
                "[VIDEO_PIPELINE] aid=%s status=%s score=%s reps=%d annotated=%s proc_time=%.2fms",
                aid, response.status, response.score, response.repCount, annotated_ok, proc_time_ms
            )

            self._cache_result(aid, response, final_annotated_path if annotated_ok else None)
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

    def _cache_result(
        self,
        aid: str,
        response: VideoScoringResponse,
        annotated_path: Optional[str],
    ) -> None:
        """Store recent analysis results and video paths with bounded cache capacity."""
        while len(self._analysis_cache) >= MAX_CACHE_ENTRIES:
            oldest_key = next(iter(self._analysis_cache))
            self._analysis_cache.pop(oldest_key, None)
            old_video_path = self._video_path_cache.pop(oldest_key, None)
            if old_video_path and old_video_path != annotated_path and os.path.exists(old_video_path):
                try:
                    os.remove(old_video_path)
                except OSError:
                    pass
        self._analysis_cache[aid] = response
        if annotated_path:
            self._video_path_cache[aid] = annotated_path

    def _build_response_from_sequence_result(
        self,
        aid: str,
        exercise_id: Optional[Any],
        default_exercise: str,
        seq_res: Dict[str, Any],
        camera_quality: CameraQualityAssessment,
        keypoint_frames: List[VideoKeypointFrame],
        total_sampled_frames: int,
        effective_sample_fps: float,
        width: int,
        height: int,
        original_fps: float,
        duration_ms: int,
    ) -> VideoScoringResponse:
        """Assemble a VideoScoringResponse directly from an analyzer's analyze_video_sequence output."""
        resp_status = str(seq_res.get("status", "COMPLETED")).upper()
        exercise_label = str(seq_res.get("exercise") or default_exercise)

        score = _round_optional(seq_res.get("score"))
        form_accuracy = _round_optional(seq_res.get("formAccuracy"))
        stability = _round_optional(seq_res.get("stability"))
        alignment = _round_optional(seq_res.get("alignment"))

        rep_count = int(seq_res.get("repCount", 0) or 0)
        valid_reps = int(seq_res.get("validReps", 0) or 0)
        incomplete_reps = int(seq_res.get("incompleteReps", 0) or 0)

        reps: List[RepBreakdownItem] = []
        for r in seq_res.get("reps") or []:
            if isinstance(r, RepBreakdownItem):
                reps.append(r)
            elif isinstance(r, dict):
                reps.append(RepBreakdownItem.model_validate(r))

        movement_metrics: Dict[str, MovementMetricItem] = {}
        raw_mm = seq_res.get("movementMetrics") or {}
        for m_key, m_val in raw_mm.items():
            if isinstance(m_val, MovementMetricItem):
                movement_metrics[m_key] = m_val
            elif isinstance(m_val, dict):
                item_payload = {"name": m_key, **m_val}
                movement_metrics[m_key] = MovementMetricItem.model_validate(item_payload)

        issues: List[StructuredIssueItem] = []
        for iss in seq_res.get("issues") or []:
            if isinstance(iss, StructuredIssueItem):
                issues.append(iss)
            elif isinstance(iss, dict):
                issues.append(StructuredIssueItem.model_validate(iss))

        recommendations = list(seq_res.get("recommendations") or [])
        feedback = list(seq_res.get("feedback") or [])

        if resp_status == "INSUFFICIENT_POSE_QUALITY":
            # Never fabricate scores when pose quality is insufficient
            score = None
            form_accuracy = None
            stability = None
            alignment = None
            if not feedback:
                feedback = list(camera_quality.warnings) or [
                    "Insufficient pose quality for reliable movement analysis."
                ]
            for warn in camera_quality.warnings:
                if warn not in recommendations:
                    recommendations.append(warn)

        metrics_dict: Dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "repCount": rep_count,
            "validReps": valid_reps,
            "incompleteReps": incomplete_reps,
            "analyzedFrames": len(keypoint_frames),
            "totalSampledFrames": total_sampled_frames,
            "samplingFps": effective_sample_fps,
        }
        if form_accuracy is not None:
            metrics_dict["formAccuracy"] = form_accuracy
        if stability is not None:
            metrics_dict["stability"] = stability
        if alignment is not None:
            metrics_dict["alignment"] = alignment
        for k, v in (seq_res.get("metrics") or {}).items():
            metrics_dict[k] = v

        return VideoScoringResponse(
            schemaVersion=SCHEMA_VERSION,
            analysisId=aid,
            status=resp_status,
            exerciseId=exercise_id,
            exercise=exercise_label,
            score=score,
            formAccuracy=form_accuracy,
            stability=stability,
            alignment=alignment,
            repCount=rep_count,
            validReps=valid_reps,
            incompleteReps=incomplete_reps,
            reps=reps,
            movementMetrics=movement_metrics,
            issues=issues,
            recommendations=recommendations,
            cameraQuality=camera_quality,
            metrics=metrics_dict,
            feedback=feedback,
            video=VideoOutputMetadata(
                originalAvailable=True,
                annotatedAvailable=False,
                originalObjectKey=f"orig_{aid}.mp4",
                annotatedObjectKey=None,
                durationMs=duration_ms,
                width=width,
                height=height,
                fps=round(original_fps, 2),
            ),
            keypoints=VideoKeypointsData(
                format="COCO_17",
                coordinateSystem="NORMALIZED",
                frames=keypoint_frames,
            ),
        )

    def _run_legacy_fallback_analysis(
        self,
        aid: str,
        exercise_id: Optional[Any],
        analyzer: Any,
        frame_samples: List[Dict[str, Any]],
        camera_quality: CameraQualityAssessment,
        keypoint_frames: List[VideoKeypointFrame],
        effective_sample_fps: float,
        confidence_threshold: float,
        side_view: bool,
        width: int,
        height: int,
        original_fps: float,
        duration_ms: int,
    ) -> VideoScoringResponse:
        """
        Fallback sequence evaluation for isometric/legacy analyzers that do not implement
        `analyze_video_sequence`. Rejects low-quality frames instead of fabricating scores.
        """
        scoring_engine = ExerciseScoringEngine()
        feedback_engine = FeedbackEngine()
        frame_scores: List[float] = []
        frame_issues: List[str] = []
        frame_positive: List[str] = []
        measurements_history: Dict[str, List[float]] = {}

        for sample in frame_samples:
            if not sample.get("pose_detected"):
                continue

            kps = sample["keypoints"]
            if analyzer.name == "PLANCHE":
                pose_ok, _, _ = check_pose_quality_for_planche(
                    kps, side_view=side_view, threshold=confidence_threshold
                )
            else:
                pose_ok, _, _ = analyzer.validate_pose(
                    kps, threshold=confidence_threshold, side_view=side_view
                )

            if not pose_ok:
                continue

            measurements = analyzer.extract_measurements(kps, side_view=side_view)
            for m_key, m_val in measurements.items():
                if m_val is not None and math.isfinite(m_val):
                    measurements_history.setdefault(m_key, []).append(float(m_val))

            f_score, feedback_items, _ = analyzer.evaluate_live_form(
                measurements=measurements,
                side_view=side_view,
            )
            scoring_metrics = analyzer.evaluate_scoring_metrics(
                measurements=measurements,
                keypoints=kps,
                side_view=side_view,
            )
            calculated_score, _, _, _ = scoring_engine.evaluate_score(
                metrics=scoring_metrics,
                keypoints=kps,
                required_landmarks=analyzer.required_landmarks,
            )

            chosen_score = calculated_score if calculated_score is not None else f_score
            if chosen_score is not None and math.isfinite(chosen_score):
                frame_scores.append(float(chosen_score))

            processed_fb = feedback_engine.process_feedback(
                feedback_items, timestamp_sec=sample["timestamp_sec"]
            )
            for fb in processed_fb:
                if fb.severity.name in ("WARNING", "ERROR"):
                    frame_issues.append(fb.message)
                elif fb.severity.name == "INFO":
                    frame_positive.append(fb.message)

        if not frame_scores or not measurements_history:
            logger.warning(
                "[VIDEO_PIPELINE] aid=%s insufficient pose quality for legacy analyzer=%s",
                aid, analyzer.name
            )
            recommendations = [analyzer.camera_view_recommendation]
            for w in camera_quality.warnings:
                if w not in recommendations:
                    recommendations.append(w)
            feedback_msgs = (
                list(camera_quality.warnings)
                if camera_quality.warnings
                else ["Pose quality was insufficient for reliable biomechanical evaluation."]
            )
            return self._create_failed_response(
                aid=aid,
                exercise_name=analyzer.name,
                exercise_id=exercise_id,
                status="INSUFFICIENT_POSE_QUALITY",
                message=feedback_msgs[0],
                width=width,
                height=height,
                fps=round(original_fps, 2),
                duration_ms=duration_ms,
                camera_quality=camera_quality,
                recommendations=recommendations,
                feedback=feedback_msgs,
                metrics={
                    "schemaVersion": SCHEMA_VERSION,
                    "repCount": 0,
                    "validReps": 0,
                    "incompleteReps": 0,
                    "analyzedFrames": len(keypoint_frames),
                    "totalSampledFrames": len(frame_samples),
                    "samplingFps": effective_sample_fps,
                },
                keypoints=VideoKeypointsData(
                    format="COCO_17",
                    coordinateSystem="NORMALIZED",
                    frames=keypoint_frames,
                ),
            )

        form_accuracy = round(float(np.mean(frame_scores)), 1)

        stds = [float(np.std(vals)) for vals in measurements_history.values() if len(vals) > 1]
        stability_score: Optional[float] = None
        if stds:
            avg_std = float(np.mean(stds))
            stability_score = round(max(20.0, min(100.0, 100.0 - (avg_std * 2.5))), 1)

        alignment_score: Optional[float] = None
        if "body_line_angle" in measurements_history:
            body_lines = measurements_history["body_line_angle"]
            avg_line = float(np.mean(body_lines))
            dev = abs(180.0 - avg_line)
            alignment_score = round(max(20.0, min(100.0, 100.0 - (dev * 2.0))), 1)
        elif "horizontal_tilt" in measurements_history:
            tilts = measurements_history["horizontal_tilt"]
            avg_tilt = float(np.mean(tilts))
            alignment_score = round(max(20.0, min(100.0, 100.0 - (avg_tilt * 2.0))), 1)

        weighted_sum = 0.50 * form_accuracy
        total_w = 0.50
        if stability_score is not None:
            weighted_sum += 0.25 * stability_score
            total_w += 0.25
        if alignment_score is not None:
            weighted_sum += 0.25 * alignment_score
            total_w += 0.25
        overall_score = round(max(0.0, min(100.0, weighted_sum / total_w)), 1)

        unique_issues = list(dict.fromkeys(frame_issues))[:4]
        unique_positives = list(dict.fromkeys(frame_positive))[:2]
        feedback_list: List[str] = []
        if unique_issues:
            feedback_list.extend(unique_issues)
        else:
            feedback_list.append("Excellent execution! Form remained consistent and controlled.")
        if unique_positives:
            feedback_list.extend(unique_positives)

        avg_metrics: Dict[str, Any] = {
            "schemaVersion": SCHEMA_VERSION,
            "repCount": 0,
            "validReps": 0,
            "incompleteReps": 0,
            "formAccuracy": form_accuracy,
            "analyzedFrames": len(keypoint_frames),
            "totalSampledFrames": len(frame_samples),
            "samplingFps": effective_sample_fps,
        }
        if stability_score is not None:
            avg_metrics["stability"] = stability_score
        if alignment_score is not None:
            avg_metrics["alignment"] = alignment_score

        movement_metrics: Dict[str, MovementMetricItem] = {}
        for k, vals in measurements_history.items():
            if vals:
                avg_v = round(float(np.mean(vals)), 1)
                avg_metrics[k] = avg_v
                movement_metrics[k] = MovementMetricItem(
                    name=k,
                    unit="ratio" if "ratio" in k else "deg" if "angle" in k or "tilt" in k else "",
                    value=avg_v,
                    min=round(float(np.min(vals)), 1),
                    max=round(float(np.max(vals)), 1),
                    avg=avg_v,
                    score=None,
                    status="VALID",
                    confidence=camera_quality.averageConfidence,
                )

        recommendations = list(camera_quality.warnings)
        if unique_issues and analyzer.camera_view_recommendation not in recommendations:
            recommendations.append(analyzer.camera_view_recommendation)

        return VideoScoringResponse(
            schemaVersion=SCHEMA_VERSION,
            analysisId=aid,
            status="COMPLETED",
            exerciseId=exercise_id,
            exercise=analyzer.name,
            score=overall_score,
            formAccuracy=form_accuracy,
            stability=stability_score,
            alignment=alignment_score,
            repCount=0,
            validReps=0,
            incompleteReps=0,
            reps=[],
            movementMetrics=movement_metrics,
            issues=[],
            recommendations=recommendations,
            cameraQuality=camera_quality,
            metrics=avg_metrics,
            feedback=feedback_list,
            video=VideoOutputMetadata(
                originalAvailable=True,
                annotatedAvailable=False,
                originalObjectKey=f"orig_{aid}.mp4",
                annotatedObjectKey=None,
                durationMs=duration_ms,
                width=width,
                height=height,
                fps=round(original_fps, 2),
            ),
            keypoints=VideoKeypointsData(
                format="COCO_17",
                coordinateSystem="NORMALIZED",
                frames=keypoint_frames,
            ),
        )

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
    ) -> bool:
        """Render skeleton overlay on every frame with interpolation and encode to MP4."""
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(raw_annotated_path, fourcc, original_fps, (width, height))
        if not out.isOpened():
            cap.release()
            return False

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

        if frame_idx == 0 or not os.path.exists(raw_annotated_path) or os.path.getsize(raw_annotated_path) == 0:
            return False

        # Transcode to H.264 (yuv420p) with AAC audio using FFmpeg
        self._transcode_to_h264_with_audio(
            raw_annotated_path=raw_annotated_path,
            original_input_path=input_path,
            output_path=final_annotated_path
        )
        return os.path.exists(final_annotated_path) and os.path.getsize(final_annotated_path) > 0

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
        duration_ms: int = 0,
        camera_quality: Optional[CameraQualityAssessment] = None,
        recommendations: Optional[List[str]] = None,
        feedback: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        keypoints: Optional[VideoKeypointsData] = None,
    ) -> VideoScoringResponse:
        # Preserve 0.0 on NO_PERSON_DETECTED for backward compatibility; keep None on insufficient quality / errors
        is_no_person = status == "NO_PERSON_DETECTED"
        default_score = 0.0 if is_no_person else None

        resp_metrics = dict(metrics) if metrics is not None else {"schemaVersion": SCHEMA_VERSION}
        resp_metrics.setdefault("schemaVersion", SCHEMA_VERSION)

        return VideoScoringResponse(
            schemaVersion=SCHEMA_VERSION,
            analysisId=aid,
            status=status,
            exerciseId=exercise_id,
            exercise=exercise_name,
            score=default_score,
            formAccuracy=default_score,
            stability=default_score,
            alignment=default_score,
            repCount=0,
            validReps=0,
            incompleteReps=0,
            reps=[],
            movementMetrics={},
            issues=[],
            recommendations=list(recommendations or []),
            cameraQuality=camera_quality,
            metrics=resp_metrics,
            feedback=list(feedback) if feedback else [message],
            video=VideoOutputMetadata(
                originalAvailable=(status != "INVALID_INPUT"),
                annotatedAvailable=False,
                originalObjectKey=f"orig_{aid}.mp4" if status != "INVALID_INPUT" else None,
                annotatedObjectKey=None,
                durationMs=duration_ms,
                width=width,
                height=height,
                fps=fps,
            ),
            keypoints=keypoints
            or VideoKeypointsData(
                format="COCO_17",
                coordinateSystem="NORMALIZED",
                frames=[],
            ),
        )


# Singleton pipeline instance
video_scoring_pipeline = VideoScoringPipeline()
