"""
Contract-driven QA and E2E integration tests for Calisthenics AI Video Analysis v1.0.

Covers:
1. Synthetic 17-COCO keypoint sequence generators for SQUAT, PUSH_UP, and PULL_UP.
2. Biomechanical detection accuracy, rep counting, incomplete reps, boundary jitter
   hysteresis, and occlusion-driven UNAVAILABLE metric invariants (never fabricated).
3. Low-confidence / occluded landmark handling (INSUFFICIENT_POSE_QUALITY).
4. Malformed, empty, and zero-frame video handling via VideoScoringPipeline & FastAPI TestClient.
5. Full VideoScoringResponse v1.0 schema compliance.
"""

import math
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.pose.yolo_pose import COCO_KEYPOINTS
from app.schemas.analysis import (
    CameraQualityAssessment,
    KeypointData,
    MovementMetricItem,
    PoseKeypoints,
    RepBreakdownItem,
    StructuredIssueItem,
    VideoScoringResponse,
)
from app.services.video_pipeline import VideoScoringPipeline, video_scoring_pipeline

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
DEFAULT_FPS = 10.0

client = TestClient(app)


# ==============================================================================
# Synthetic Video & Pose Estimator Fixtures
# ==============================================================================


class SequencePoseEstimator:
    """Deterministic pose estimator returning pre-generated COCO-17 PoseKeypoints per frame."""

    def __init__(self, frame_keypoints: List[Optional[Dict[str, KeypointData]]]):
        self.frame_keypoints = frame_keypoints
        self.calls = 0

    def estimate_pose(self, image_np: np.ndarray, tracker=None) -> Tuple[bool, Optional[PoseKeypoints]]:
        if not self.frame_keypoints:
            self.calls += 1
            return False, None

        idx = min(self.calls, len(self.frame_keypoints) - 1)
        self.calls += 1
        kps = self.frame_keypoints[idx]
        if not kps:
            return False, None

        visible_pts = [kp for kp in kps.values() if kp.visible and kp.confidence > 0.0]
        if visible_pts:
            xs = [kp.x for kp in visible_pts]
            ys = [kp.y for kp in visible_pts]
            bbox = [
                max(0.0, min(xs) - 20.0),
                max(0.0, min(ys) - 20.0),
                min(float(FRAME_WIDTH), max(xs) + 20.0),
                min(float(FRAME_HEIGHT), max(ys) + 20.0),
            ]
            det_conf = round(sum(kp.confidence for kp in visible_pts) / len(visible_pts), 3)
        else:
            all_xs = [kp.x for kp in kps.values()]
            all_ys = [kp.y for kp in kps.values()]
            bbox = [
                max(0.0, min(all_xs) - 20.0),
                max(0.0, min(all_ys) - 20.0),
                min(float(FRAME_WIDTH), max(all_xs) + 20.0),
                min(float(FRAME_HEIGHT), max(all_ys) + 20.0),
            ]
            det_conf = round(sum(kp.confidence for kp in kps.values()) / max(1, len(kps)), 3)

        return True, PoseKeypoints(
            keypoints=kps,
            bounding_box=bbox,
            detection_confidence=det_conf,
        )


def create_synthetic_mp4_bytes(
    num_frames: int,
    fps: float = DEFAULT_FPS,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
) -> bytes:
    """Create a valid synthetic MP4 video with `num_frames` frames."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, float(fps), (width, height))
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(num_frames):
            # Subtle changing pixel keeps video stream non-trivial
            frame[:, :] = ((i * 3) % 40, 15, 25)
            writer.write(frame)
        writer.release()

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def create_zero_frame_mp4_bytes(fps: float = DEFAULT_FPS, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT) -> bytes:
    """Create a video container with zero encoded frames."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, float(fps), (width, height))
        writer.release()

        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _clamp_point(x: float, y: float) -> Tuple[float, float]:
    return (
        max(5.0, min(float(FRAME_WIDTH - 5), float(x))),
        max(5.0, min(float(FRAME_HEIGHT - 5), float(y))),
    )


def _build_coco17_keypoints(
    coords: Dict[str, Tuple[float, float]],
    default_conf: float = 0.95,
    overrides: Optional[Dict[str, Tuple[float, float, float, bool]]] = None,
) -> Dict[str, KeypointData]:
    """Assemble a full 17-landmark COCO dictionary with optional per-joint occlusion overrides."""
    overrides = overrides or {}
    result: Dict[str, KeypointData] = {}

    for name in COCO_KEYPOINTS:
        if name in overrides:
            ox, oy, oconf, ovis = overrides[name]
            cx, cy = _clamp_point(ox, oy)
            result[name] = KeypointData(x=cx, y=cy, confidence=oconf, name=name, visible=ovis)
        else:
            px, py = _clamp_point(*coords.get(name, (FRAME_WIDTH / 2.0, FRAME_HEIGHT / 2.0)))
            result[name] = KeypointData(
                x=px,
                y=py,
                confidence=default_conf,
                name=name,
                visible=default_conf >= 0.25,
            )
    return result


def _rotate_unit(ux: float, uy: float, angle_deg: float) -> Tuple[float, float]:
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    return (ux * c - uy * s, ux * s + uy * c)


def _interpolate_series(waypoints: List[Tuple[float, int]]) -> List[float]:
    """Build smooth linear interpolation across `(target_value, num_steps)` segments."""
    if not waypoints:
        return []
    values: List[float] = []
    prev_val, first_steps = waypoints[0]
    values.extend([prev_val] * max(1, first_steps))
    for target_val, steps in waypoints[1:]:
        steps = max(1, steps)
        for s in range(1, steps + 1):
            alpha = s / float(steps)
            values.append(prev_val + (target_val - prev_val) * alpha)
        prev_val = target_val
    return values


# ==============================================================================
# 17-COCO Synthetic Pose Sequence Generators
# ==============================================================================


def generate_squat_frame(
    knee_angle_deg: float,
    torso_lean_deg: float = 15.0,
    knee_ankle_ratio: float = 0.96,
    conf: float = 0.95,
    side_view: bool = False,
    occlude_right_leg: bool = False,
) -> Dict[str, KeypointData]:
    """Generate a geometrically exact 17-COCO keypoint frame for a Squat."""
    ankle_y = 425.0
    knee_y = 325.0
    thigh_len = 100.0
    torso_len = 105.0

    if side_view:
        l_ankle_x, r_ankle_x = 320.0, 324.0
        l_knee_x, r_knee_x = 335.0, 338.0
        hip_sep = 4.0
        sh_sep = 6.0
    else:
        ankle_span = 120.0
        l_ankle_x = 320.0 - ankle_span / 2.0
        r_ankle_x = 320.0 + ankle_span / 2.0
        knee_span = ankle_span * knee_ankle_ratio
        l_knee_x = 320.0 - knee_span / 2.0
        r_knee_x = 320.0 + knee_span / 2.0
        hip_sep = 70.0
        sh_sep = 96.0

    def _hip_from_knee_ankle(kx: float, ky: float, ax: float, ay: float) -> Tuple[float, float]:
        dx, dy = ax - kx, ay - ky
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        # Rotate shin vector clockwise in image space by -knee_angle_deg so hip goes backward/upward
        vx, vy = _rotate_unit(ux, uy, -knee_angle_deg)
        return (kx + thigh_len * vx, ky + thigh_len * vy)

    l_hip = _hip_from_knee_ankle(l_knee_x, knee_y, l_ankle_x, ankle_y)
    r_hip = _hip_from_knee_ankle(r_knee_x, knee_y, r_ankle_x, ankle_y)
    if not side_view:
        hip_mid_x = (l_hip[0] + r_hip[0]) / 2.0
        hip_mid_y = (l_hip[1] + r_hip[1]) / 2.0
        l_hip = (hip_mid_x - hip_sep / 2.0, hip_mid_y)
        r_hip = (hip_mid_x + hip_sep / 2.0, hip_mid_y)
        # Re-project knees slightly if needed? Keep exact angle on primary side by aligning hip x around midpoint:
        l_hip = _hip_from_knee_ankle(l_knee_x, knee_y, l_ankle_x, ankle_y)
        r_hip = _hip_from_knee_ankle(r_knee_x, knee_y, r_ankle_x, ankle_y)

    hip_mid_x = (l_hip[0] + r_hip[0]) / 2.0
    hip_mid_y = (l_hip[1] + r_hip[1]) / 2.0

    lean_rad = math.radians(torso_lean_deg)
    sh_mid_x = hip_mid_x + torso_len * math.sin(lean_rad)
    sh_mid_y = hip_mid_y - torso_len * math.cos(lean_rad)

    l_sh = (sh_mid_x - sh_sep / 2.0, sh_mid_y)
    r_sh = (sh_mid_x + sh_sep / 2.0, sh_mid_y)

    coords = {
        "nose": (sh_mid_x + 4.0, sh_mid_y - 34.0),
        "left_eye": (sh_mid_x - 6.0, sh_mid_y - 40.0),
        "right_eye": (sh_mid_x + 12.0, sh_mid_y - 40.0),
        "left_ear": (sh_mid_x - 14.0, sh_mid_y - 34.0),
        "right_ear": (sh_mid_x + 18.0, sh_mid_y - 34.0),
        "left_shoulder": l_sh,
        "right_shoulder": r_sh,
        "left_elbow": (l_sh[0] + 15.0, l_sh[1] + 45.0),
        "right_elbow": (r_sh[0] + 15.0, r_sh[1] + 45.0),
        "left_wrist": (l_sh[0] + 45.0, l_sh[1] + 45.0),
        "right_wrist": (r_sh[0] + 45.0, r_sh[1] + 45.0),
        "left_hip": l_hip,
        "right_hip": r_hip,
        "left_knee": (l_knee_x, knee_y),
        "right_knee": (r_knee_x, knee_y),
        "left_ankle": (l_ankle_x, ankle_y),
        "right_ankle": (r_ankle_x, ankle_y),
    }

    overrides = {}
    if occlude_right_leg:
        overrides["right_knee"] = (r_knee_x, knee_y, 0.05, False)
        overrides["right_ankle"] = (r_ankle_x, ankle_y, 0.05, False)

    return _build_coco17_keypoints(coords, default_conf=conf, overrides=overrides)


def generate_squat_sequence(
    rep_min_angles: List[float],
    torso_lean_deg: float = 15.0,
    knee_ankle_ratio: float = 0.96,
    conf: float = 0.95,
    side_view: bool = False,
    occlude_right_leg: bool = False,
) -> List[Dict[str, KeypointData]]:
    """Build a multi-rep squat keypoint sequence at 10 fps."""
    waypoints: List[Tuple[float, int]] = [(175.0, 4)]
    for min_ang in rep_min_angles:
        waypoints.extend([
            (min_ang, 6),   # 0.6s descent
            (min_ang, 4),   # 0.4s bottom hold
            (175.0, 6),     # 0.6s ascent
            (175.0, 4),     # 0.4s standing cooldown
        ])
    angles = _interpolate_series(waypoints)
    return [
        generate_squat_frame(
            knee_angle_deg=ang,
            torso_lean_deg=torso_lean_deg,
            knee_ankle_ratio=knee_ankle_ratio if ang < 140.0 else 0.96,
            conf=conf,
            side_view=side_view,
            occlude_right_leg=occlude_right_leg,
        )
        for ang in angles
    ]


def generate_push_up_frame(
    elbow_angle_deg: float,
    body_line_deg: float = 180.0,
    alignment_mode: str = "STRAIGHT",
    conf: float = 0.95,
    occlude_ankles: bool = False,
) -> Dict[str, KeypointData]:
    """Generate a geometrically exact 17-COCO keypoint frame for a Push-up (side profile)."""
    l_wrist = (445.0, 385.0)
    r_wrist = (451.0, 389.0)
    l_elbow = (445.0, 320.0)
    r_elbow = (451.0, 324.0)
    upper_arm_len = 70.0

    # u points downward from elbow to wrist: (0, +1). Rotate by -elbow_angle_deg -> forward/upward.
    rad = math.radians(elbow_angle_deg)
    sh_dx = upper_arm_len * math.sin(rad)
    sh_dy = upper_arm_len * math.cos(rad)

    l_sh = (l_elbow[0] + sh_dx, l_elbow[1] + sh_dy)
    r_sh = (r_elbow[0] + sh_dx, r_elbow[1] + sh_dy)
    sh_mid = ((l_sh[0] + r_sh[0]) / 2.0, (l_sh[1] + r_sh[1]) / 2.0)

    l_ankle = (140.0, 385.0)
    r_ankle = (146.0, 389.0)
    ak_mid = ((l_ankle[0] + r_ankle[0]) / 2.0, (l_ankle[1] + r_ankle[1]) / 2.0)

    # Body line midpoint & normal for SAG / PIKE
    mid_x = (sh_mid[0] + ak_mid[0]) / 2.0
    mid_y = (sh_mid[1] + ak_mid[1]) / 2.0
    line_dx = ak_mid[0] - sh_mid[0]
    line_dy = ak_mid[1] - sh_mid[1]
    line_len = max(1.0, math.hypot(line_dx, line_dy))
    half_len = line_len / 2.0

    # Normal pointing downward (+y in screen coords) since line_dx < 0 -> -line_dx > 0
    nx = line_dy / line_len
    ny = -line_dx / line_len

    clamped_line = max(90.0, min(180.0, float(body_line_deg)))
    if alignment_mode == "STRAIGHT" or clamped_line >= 179.5:
        offset = 0.0
    else:
        offset = half_len / math.tan(math.radians(clamped_line / 2.0))
        if alignment_mode == "PIKE":
            offset = -offset

    hip_mid = (mid_x + offset * nx, mid_y + offset * ny)
    l_hip = (hip_mid[0] - 3.0, hip_mid[1] - 2.0)
    r_hip = (hip_mid[0] + 3.0, hip_mid[1] + 2.0)

    l_knee = ((l_hip[0] + l_ankle[0]) / 2.0, (l_hip[1] + l_ankle[1]) / 2.0)
    r_knee = ((r_hip[0] + r_ankle[0]) / 2.0, (r_hip[1] + r_ankle[1]) / 2.0)

    coords = {
        "nose": (sh_mid[0] + 26.0, sh_mid[1] - 18.0),
        "left_eye": (sh_mid[0] + 22.0, sh_mid[1] - 24.0),
        "right_eye": (sh_mid[0] + 24.0, sh_mid[1] - 22.0),
        "left_ear": (sh_mid[0] + 14.0, sh_mid[1] - 22.0),
        "right_ear": (sh_mid[0] + 16.0, sh_mid[1] - 20.0),
        "left_shoulder": l_sh,
        "right_shoulder": r_sh,
        "left_elbow": l_elbow,
        "right_elbow": r_elbow,
        "left_wrist": l_wrist,
        "right_wrist": r_wrist,
        "left_hip": l_hip,
        "right_hip": r_hip,
        "left_knee": l_knee,
        "right_knee": r_knee,
        "left_ankle": l_ankle,
        "right_ankle": r_ankle,
    }

    overrides = {}
    if occlude_ankles:
        overrides["left_ankle"] = (l_ankle[0], l_ankle[1], 0.05, False)
        overrides["right_ankle"] = (r_ankle[0], r_ankle[1], 0.05, False)

    return _build_coco17_keypoints(coords, default_conf=conf, overrides=overrides)


def generate_push_up_sequence(
    rep_min_angles: List[float],
    body_line_deg: float = 180.0,
    alignment_mode: str = "STRAIGHT",
    conf: float = 0.95,
    occlude_ankles: bool = False,
) -> List[Dict[str, KeypointData]]:
    """Build a multi-rep push-up keypoint sequence at 10 fps."""
    waypoints: List[Tuple[float, int]] = [(175.0, 4)]
    for min_ang in rep_min_angles:
        waypoints.extend([
            (min_ang, 6),
            (min_ang, 4),
            (175.0, 6),
            (175.0, 4),
        ])
    angles = _interpolate_series(waypoints)
    return [
        generate_push_up_frame(
            elbow_angle_deg=ang,
            body_line_deg=body_line_deg,
            alignment_mode=alignment_mode,
            conf=conf,
            occlude_ankles=occlude_ankles,
        )
        for ang in angles
    ]


def generate_pull_up_frame(
    elbow_angle_deg: float,
    nose_above_wrists: bool = True,
    hip_offset_x: float = 0.0,
    conf: float = 0.95,
    occlude_hips_legs: bool = False,
    occlude_nose: bool = False,
) -> Dict[str, KeypointData]:
    """Generate a geometrically exact 17-COCO keypoint frame for a Pull-up."""
    l_wrist = (250.0, 100.0)
    r_wrist = (390.0, 100.0)
    upper_arm_len = 65.0

    # Elbow moves slightly outward as athlete pulls up
    flex_ratio = max(0.0, min(1.0, (175.0 - elbow_angle_deg) / 120.0))
    l_elbow = (250.0 - 12.0 * flex_ratio, 165.0)
    r_elbow = (390.0 + 12.0 * flex_ratio, 165.0)

    def _shoulder_from_elbow_wrist(
        ex: float, ey: float, wx: float, wy: float, sign: float
    ) -> Tuple[float, float]:
        dx, dy = wx - ex, wy - ey
        norm = math.hypot(dx, dy)
        ux, uy = dx / norm, dy / norm
        vx, vy = _rotate_unit(ux, uy, sign * elbow_angle_deg)
        return (ex + upper_arm_len * vx, ey + upper_arm_len * vy)

    # Rotate wrist->elbow upward vector inward toward body midline
    l_sh = _shoulder_from_elbow_wrist(l_elbow[0], l_elbow[1], l_wrist[0], l_wrist[1], +1.0)
    r_sh = _shoulder_from_elbow_wrist(r_elbow[0], r_elbow[1], r_wrist[0], r_wrist[1], -1.0)

    sh_mid_x = (l_sh[0] + r_sh[0]) / 2.0
    sh_mid_y = (l_sh[1] + r_sh[1]) / 2.0

    # Position nose clearly above bar (y < 100) at top when nose_above_wrists=True
    if elbow_angle_deg <= 85.0 and nose_above_wrists:
        nose_y = 76.0
    else:
        nose_y = max(112.0, sh_mid_y - 32.0)

    hip_mid_x = 320.0 + hip_offset_x
    hip_y = sh_mid_y + 115.0
    knee_y = hip_y + 85.0
    ankle_y = knee_y + 85.0

    coords = {
        "nose": (sh_mid_x, nose_y),
        "left_eye": (sh_mid_x - 8.0, nose_y - 5.0),
        "right_eye": (sh_mid_x + 8.0, nose_y - 5.0),
        "left_ear": (sh_mid_x - 16.0, nose_y - 2.0),
        "right_ear": (sh_mid_x + 16.0, nose_y - 2.0),
        "left_shoulder": l_sh,
        "right_shoulder": r_sh,
        "left_elbow": l_elbow,
        "right_elbow": r_elbow,
        "left_wrist": l_wrist,
        "right_wrist": r_wrist,
        "left_hip": (hip_mid_x - 32.0, hip_y),
        "right_hip": (hip_mid_x + 32.0, hip_y),
        "left_knee": (hip_mid_x - 30.0, knee_y),
        "right_knee": (hip_mid_x + 30.0, knee_y),
        "left_ankle": (hip_mid_x - 28.0, ankle_y),
        "right_ankle": (hip_mid_x + 28.0, ankle_y),
    }

    overrides = {}
    if occlude_hips_legs:
        for lm in ("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"):
            px, py = coords[lm]
            overrides[lm] = (px, py, 0.05, False)
    if occlude_nose:
        for lm in ("nose", "left_eye", "right_eye", "left_ear", "right_ear"):
            px, py = coords[lm]
            overrides[lm] = (px, py, 0.05, False)

    return _build_coco17_keypoints(coords, default_conf=conf, overrides=overrides)


def generate_pull_up_sequence(
    rep_min_angles: List[float],
    nose_above_wrists: bool = True,
    kip_amplitude_px: float = 0.0,
    conf: float = 0.95,
    occlude_hips_legs: bool = False,
    occlude_nose: bool = False,
) -> List[Dict[str, KeypointData]]:
    """Build a multi-rep pull-up keypoint sequence at 10 fps."""
    waypoints: List[Tuple[float, int]] = [(175.0, 4)]
    for min_ang in rep_min_angles:
        waypoints.extend([
            (min_ang, 6),
            (min_ang, 4),
            (175.0, 6),
            (175.0, 4),
        ])
    angles = _interpolate_series(waypoints)
    frames: List[Dict[str, KeypointData]] = []
    for idx, ang in enumerate(angles):
        hip_dx = kip_amplitude_px * math.sin(idx * 0.9) if kip_amplitude_px > 0 else 0.0
        frames.append(
            generate_pull_up_frame(
                elbow_angle_deg=ang,
                nose_above_wrists=nose_above_wrists,
                hip_offset_x=hip_dx,
                conf=conf,
                occlude_hips_legs=occlude_hips_legs,
                occlude_nose=occlude_nose,
            )
        )
    return frames


def _run_pipeline_with_sequence(
    sequence: List[Optional[Dict[str, KeypointData]]],
    exercise: str,
    side_view: bool = False,
    fps: float = DEFAULT_FPS,
    sample_fps: int = 10,
) -> VideoScoringResponse:
    video_bytes = create_synthetic_mp4_bytes(num_frames=len(sequence), fps=fps)
    estimator = SequencePoseEstimator(sequence)
    pipeline = VideoScoringPipeline(pose_estimator=estimator)
    return pipeline.process_video(
        video_bytes=video_bytes,
        exercise_name=exercise,
        sample_fps=sample_fps,
        side_view=side_view,
    )


def _assert_unavailable_metric_invariant(response: VideoScoringResponse, metric_key: str) -> None:
    assert metric_key in response.movementMetrics, f"Missing expected metric '{metric_key}' in {list(response.movementMetrics.keys())}"
    metric = response.movementMetrics[metric_key]
    assert metric.status == "UNAVAILABLE", f"Expected {metric_key} UNAVAILABLE, got {metric}"
    assert metric.value is None, f"UNAVAILABLE metric {metric_key} leaked fabricated value={metric.value}"
    assert metric.min is None
    assert metric.max is None
    assert metric.avg is None
    assert metric.score is None, f"UNAVAILABLE metric {metric_key} leaked fabricated score={metric.score}"


def _assert_no_fabricated_metrics(response: VideoScoringResponse) -> None:
    for key, metric in response.movementMetrics.items():
        if metric.status == "UNAVAILABLE":
            assert metric.value is None, f"{key} UNAVAILABLE has non-None value={metric.value}"
            assert metric.score is None, f"{key} UNAVAILABLE has non-None score={metric.score}"
            assert metric.min is None
            assert metric.max is None
            assert metric.avg is None


def _assert_valid_v1_schema(response: VideoScoringResponse, expected_exercise: str) -> None:
    assert response.schemaVersion == "1.0"
    assert response.exercise == expected_exercise
    assert response.analysisId
    assert response.repCount == response.validReps + response.incompleteReps
    assert len(response.reps) == response.repCount
    assert isinstance(response.movementMetrics, dict)
    assert isinstance(response.issues, list)
    assert isinstance(response.recommendations, list)
    assert response.cameraQuality is not None
    assert response.cameraQuality.overallQuality in ("GOOD", "FAIR", "POOR")
    for issue in response.issues:
        assert issue.severity.value in ("LOW", "MEDIUM", "HIGH")
    _assert_no_fabricated_metrics(response)


# ==============================================================================
# SQUAT QA & Contract Tests
# ==============================================================================


def test_squat_clean_valid_reps_e2e():
    seq = generate_squat_sequence(rep_min_angles=[85.0, 88.0, 82.0], torso_lean_deg=15.0, knee_ankle_ratio=0.98)
    resp = _run_pipeline_with_sequence(seq, exercise="SQUAT", side_view=False)

    _assert_valid_v1_schema(resp, "SQUAT")
    assert resp.status == "COMPLETED"
    assert resp.repCount == 3
    assert resp.validReps == 3
    assert resp.incompleteReps == 0
    assert all(r.isValid for r in resp.reps)
    assert resp.score is not None and resp.score >= 75.0


def test_squat_mixed_valid_and_incomplete_reps():
    # 2 deep reps (85 deg) + 1 shallow rep (125 deg)
    seq = generate_squat_sequence(rep_min_angles=[85.0, 125.0, 86.0], torso_lean_deg=15.0)
    resp = _run_pipeline_with_sequence(seq, exercise="SQUAT", side_view=False)

    _assert_valid_v1_schema(resp, "SQUAT")
    assert resp.status == "COMPLETED"
    assert resp.validReps == 2
    assert resp.incompleteReps >= 1
    assert resp.repCount == resp.validReps + resp.incompleteReps
    assert any(not r.isValid for r in resp.reps)
    assert any(r.isValid for r in resp.reps)


def test_squat_knee_valgus_detected():
    seq = generate_squat_sequence(rep_min_angles=[88.0, 88.0], knee_ankle_ratio=0.45, torso_lean_deg=15.0)
    resp = _run_pipeline_with_sequence(seq, exercise="SQUAT", side_view=False)

    _assert_valid_v1_schema(resp, "SQUAT")
    assert resp.status == "COMPLETED"
    issue_text = " ".join(f"{i.code} {i.metric} {i.title} {i.message}".lower() for i in resp.issues)
    assert "valgus" in issue_text or "knee_alignment" in issue_text or "knee" in issue_text


def test_squat_excessive_torso_lean_detected():
    seq = generate_squat_sequence(rep_min_angles=[88.0, 88.0], torso_lean_deg=55.0, knee_ankle_ratio=0.96)
    resp = _run_pipeline_with_sequence(seq, exercise="SQUAT", side_view=False)

    _assert_valid_v1_schema(resp, "SQUAT")
    assert resp.status == "COMPLETED"
    issue_text = " ".join(f"{i.code} {i.metric} {i.title} {i.message}".lower() for i in resp.issues)
    assert "lean" in issue_text or "torso" in issue_text


def test_squat_boundary_jitter_no_double_counting():
    # First: pure standing/top jitter around 155 deg -> 0 reps
    top_jitter_angles = [165.0, 156.0, 153.0, 157.0, 152.0, 156.0, 154.0, 158.0, 165.0] * 2
    top_jitter_seq = [generate_squat_frame(a) for a in top_jitter_angles]
    top_resp = _run_pipeline_with_sequence(top_jitter_seq, exercise="SQUAT", side_view=True)
    assert top_resp.repCount == 0

    # Second: 1 real rep with jitter around bottom threshold 100 deg -> strictly 1 rep, never 2+
    bottom_jitter_angles = (
        [175.0] * 4
        + [150.0, 130.0, 110.0, 101.0, 98.0, 102.0, 99.0, 101.0, 97.0, 102.0]
        + [120.0, 145.0, 175.0, 175.0, 175.0, 175.0]
    )
    bottom_jitter_seq = [generate_squat_frame(a) for a in bottom_jitter_angles]
    bottom_resp = _run_pipeline_with_sequence(bottom_jitter_seq, exercise="SQUAT", side_view=True)
    assert bottom_resp.validReps == 1
    assert bottom_resp.repCount == 1


def test_squat_side_view_occluded_contralateral_knee_ankle_unavailable():
    seq = generate_squat_sequence(
        rep_min_angles=[86.0, 88.0],
        side_view=True,
        occlude_right_leg=True,
    )
    resp = _run_pipeline_with_sequence(seq, exercise="SQUAT", side_view=True)

    _assert_valid_v1_schema(resp, "SQUAT")
    assert resp.status == "COMPLETED"
    _assert_unavailable_metric_invariant(resp, "knee_alignment")


# ==============================================================================
# PUSH_UP QA & Contract Tests
# ==============================================================================


def test_push_up_clean_valid_reps_e2e():
    seq = generate_push_up_sequence(rep_min_angles=[85.0, 84.0], body_line_deg=180.0, alignment_mode="STRAIGHT")
    resp = _run_pipeline_with_sequence(seq, exercise="PUSH_UP", side_view=True)

    _assert_valid_v1_schema(resp, "PUSH_UP")
    assert resp.status == "COMPLETED"
    assert resp.repCount == 2
    assert resp.validReps == 2
    assert resp.incompleteReps == 0
    assert resp.score is not None and resp.score >= 75.0


def test_push_up_shallow_incomplete_reps():
    seq = generate_push_up_sequence(rep_min_angles=[120.0, 118.0], body_line_deg=180.0, alignment_mode="STRAIGHT")
    resp = _run_pipeline_with_sequence(seq, exercise="PUSH_UP", side_view=True)

    _assert_valid_v1_schema(resp, "PUSH_UP")
    assert resp.status == "COMPLETED"
    assert resp.validReps == 0
    assert resp.incompleteReps >= 1


def test_push_up_hip_sag_and_hip_piking():
    sag_seq = generate_push_up_sequence(rep_min_angles=[85.0, 86.0], body_line_deg=142.0, alignment_mode="SAG")
    sag_resp = _run_pipeline_with_sequence(sag_seq, exercise="PUSH_UP", side_view=True)
    _assert_valid_v1_schema(sag_resp, "PUSH_UP")
    sag_text = " ".join(f"{i.code} {i.metric} {i.title} {i.message}".lower() for i in sag_resp.issues)
    assert "sag" in sag_text or "body_alignment" in sag_text or "hip" in sag_text

    pike_seq = generate_push_up_sequence(rep_min_angles=[85.0, 86.0], body_line_deg=142.0, alignment_mode="PIKE")
    pike_resp = _run_pipeline_with_sequence(pike_seq, exercise="PUSH_UP", side_view=True)
    _assert_valid_v1_schema(pike_resp, "PUSH_UP")
    pike_text = " ".join(f"{i.code} {i.metric} {i.title} {i.message}".lower() for i in pike_resp.issues)
    assert "pike" in pike_text or "body_alignment" in pike_text or "hip" in pike_text


def test_push_up_boundary_jitter_no_double_counting():
    # Top boundary jitter around 155 deg -> 0 reps
    top_jitter_angles = [165.0, 156.0, 153.0, 157.0, 152.0, 156.0, 154.0, 165.0] * 2
    top_resp = _run_pipeline_with_sequence(
        [generate_push_up_frame(a) for a in top_jitter_angles],
        exercise="PUSH_UP",
        side_view=True,
    )
    assert top_resp.repCount == 0

    # 1 real rep with bottom jitter around 90 deg -> strictly 1 rep
    bottom_jitter_angles = (
        [175.0] * 4
        + [145.0, 120.0, 95.0, 91.0, 88.0, 92.0, 89.0, 91.0, 87.0, 92.0]
        + [120.0, 150.0, 175.0, 175.0, 175.0, 175.0]
    )
    bottom_resp = _run_pipeline_with_sequence(
        [generate_push_up_frame(a) for a in bottom_jitter_angles],
        exercise="PUSH_UP",
        side_view=True,
    )
    assert bottom_resp.validReps == 1
    assert bottom_resp.repCount == 1


def test_push_up_occluded_ankles_body_alignment_unavailable():
    seq = generate_push_up_sequence(
        rep_min_angles=[85.0, 86.0],
        occlude_ankles=True,
    )
    resp = _run_pipeline_with_sequence(seq, exercise="PUSH_UP", side_view=True)

    _assert_valid_v1_schema(resp, "PUSH_UP")
    assert resp.status == "COMPLETED"
    _assert_unavailable_metric_invariant(resp, "body_alignment")


# ==============================================================================
# PULL_UP QA & Contract Tests
# ==============================================================================


def test_pull_up_clean_valid_reps_nose_above_wrists():
    seq = generate_pull_up_sequence(rep_min_angles=[55.0, 58.0], nose_above_wrists=True, kip_amplitude_px=0.0)
    resp = _run_pipeline_with_sequence(seq, exercise="PULL_UP", side_view=False)

    _assert_valid_v1_schema(resp, "PULL_UP")
    assert resp.status == "COMPLETED"
    assert resp.repCount == 2
    assert resp.validReps == 2
    assert resp.incompleteReps == 0
    assert resp.score is not None and resp.score >= 75.0


def test_pull_up_partial_incomplete_pull():
    seq = generate_pull_up_sequence(rep_min_angles=[112.0, 115.0], nose_above_wrists=False)
    resp = _run_pipeline_with_sequence(seq, exercise="PULL_UP", side_view=False)

    _assert_valid_v1_schema(resp, "PULL_UP")
    assert resp.status == "COMPLETED"
    assert resp.validReps == 0
    assert resp.incompleteReps >= 1


def test_pull_up_horizontal_kipping_body_swing():
    seq = generate_pull_up_sequence(rep_min_angles=[58.0, 60.0], nose_above_wrists=True, kip_amplitude_px=95.0)
    resp = _run_pipeline_with_sequence(seq, exercise="PULL_UP", side_view=False)

    _assert_valid_v1_schema(resp, "PULL_UP")
    assert resp.status == "COMPLETED"
    issue_text = " ".join(f"{i.code} {i.metric} {i.title} {i.message}".lower() for i in resp.issues)
    assert "swing" in issue_text or "kip" in issue_text or "body_swing" in issue_text


def test_pull_up_occluded_hips_and_nose_unavailable():
    hips_occ_seq = generate_pull_up_sequence(rep_min_angles=[58.0, 60.0], occlude_hips_legs=True)
    hips_resp = _run_pipeline_with_sequence(hips_occ_seq, exercise="PULL_UP", side_view=False)
    _assert_valid_v1_schema(hips_resp, "PULL_UP")
    assert hips_resp.status == "COMPLETED"
    _assert_unavailable_metric_invariant(hips_resp, "body_swing")

    nose_occ_seq = generate_pull_up_sequence(rep_min_angles=[58.0, 60.0], occlude_nose=True)
    nose_resp = _run_pipeline_with_sequence(nose_occ_seq, exercise="PULL_UP", side_view=False)
    _assert_valid_v1_schema(nose_resp, "PULL_UP")
    assert nose_resp.status == "COMPLETED"
    _assert_unavailable_metric_invariant(nose_resp, "chin_to_bar")


# ==============================================================================
# Low-Confidence / Occluded Landmarks (No Fabricated Numbers)
# ==============================================================================


@pytest.mark.parametrize("exercise", ["SQUAT", "PUSH_UP", "PULL_UP"])
def test_low_confidence_sequence_never_fabricates_scores(exercise: str):
    if exercise == "SQUAT":
        seq = [generate_squat_frame(90.0, conf=0.08) for _ in range(15)]
    elif exercise == "PUSH_UP":
        seq = [generate_push_up_frame(90.0, conf=0.08) for _ in range(15)]
    else:
        seq = [generate_pull_up_frame(65.0, conf=0.08) for _ in range(15)]

    resp = _run_pipeline_with_sequence(seq, exercise=exercise, side_view=False)

    assert resp.schemaVersion == "1.0"
    assert resp.status in ("INSUFFICIENT_POSE_QUALITY", "NO_PERSON_DETECTED")
    assert resp.score is None, f"Expected None score for low-confidence pose, got {resp.score}"
    assert resp.formAccuracy is None
    assert resp.stability is None
    assert resp.alignment is None
    _assert_no_fabricated_metrics(resp)


# ==============================================================================
# Malformed, Empty, and Zero-Frame Video Tests (Pipeline + FastAPI)
# ==============================================================================


def test_malformed_empty_zero_frame_via_pipeline():
    pipeline = VideoScoringPipeline(pose_estimator=SequencePoseEstimator([]))

    empty_resp = pipeline.process_video(video_bytes=b"", exercise_name="SQUAT")
    assert empty_resp.schemaVersion == "1.0"
    assert empty_resp.status in ("INVALID_INPUT", "FAILED")
    assert empty_resp.score is None or empty_resp.score == 0.0

    corrupt_resp = pipeline.process_video(video_bytes=b"definitely-not-a-valid-mp4-stream", exercise_name="PUSH_UP")
    assert corrupt_resp.schemaVersion == "1.0"
    assert corrupt_resp.status in ("INVALID_INPUT", "FAILED")
    assert corrupt_resp.score is None or corrupt_resp.score == 0.0

    zero_frame_bytes = create_zero_frame_mp4_bytes()
    zero_resp = pipeline.process_video(video_bytes=zero_frame_bytes, exercise_name="PULL_UP")
    assert zero_resp.schemaVersion == "1.0"
    assert zero_resp.status in ("INVALID_INPUT", "FAILED", "NO_PERSON_DETECTED")
    assert zero_resp.score is None or zero_resp.score == 0.0


def test_malformed_empty_zero_frame_via_fastapi_testclient():
    # Empty file -> 400 Bad Request
    empty_resp = client.post(
        "/api/v1/analysis/video/score?exercise=SQUAT",
        files={"file": ("empty.mp4", b"", "video/mp4")},
    )
    assert empty_resp.status_code == 400

    # Unsupported content type -> 400 Bad Request
    bad_type_resp = client.post(
        "/api/v1/analysis/video/score?exercise=SQUAT",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert bad_type_resp.status_code == 400

    # Corrupt MP4 bytes -> graceful non-500 failure payload or 400/422
    corrupt_resp = client.post(
        "/api/v1/analysis/video/score?exercise=PUSH_UP",
        files={"file": ("corrupt.mp4", b"not_an_mp4_header_123456", "video/mp4")},
    )
    assert corrupt_resp.status_code in (200, 400, 422)
    if corrupt_resp.status_code == 200:
        payload = corrupt_resp.json()
        assert payload["status"] in ("INVALID_INPUT", "FAILED")
        assert payload["schemaVersion"] == "1.0"

    # Zero-frame MP4 -> graceful non-500 failure
    zero_bytes = create_zero_frame_mp4_bytes()
    zero_resp = client.post(
        "/api/v1/analysis/video/score?exercise=PULL_UP",
        files={"file": ("zero.mp4", zero_bytes, "video/mp4")},
    )
    assert zero_resp.status_code in (200, 400, 422)
    if zero_resp.status_code == 200:
        payload = zero_resp.json()
        assert payload["status"] in ("INVALID_INPUT", "FAILED", "NO_PERSON_DETECTED")
        assert payload["schemaVersion"] == "1.0"


# ==============================================================================
# Full VideoScoringResponse v1.0 Schema Validation via FastAPI TestClient
# ==============================================================================


def test_full_v1_schema_via_fastapi_testclient(monkeypatch):
    seq = generate_squat_sequence(rep_min_angles=[86.0, 122.0], torso_lean_deg=48.0, knee_ankle_ratio=0.95)
    mp4_bytes = create_synthetic_mp4_bytes(num_frames=len(seq), fps=DEFAULT_FPS)
    fake_estimator = SequencePoseEstimator(seq)

    monkeypatch.setattr(video_scoring_pipeline, "pose_estimator", fake_estimator)

    resp = client.post(
        "/api/v1/analysis/video/score?exercise=SQUAT&sample_fps=10&side_view=false",
        files={"file": ("qa_squat.mp4", mp4_bytes, "video/mp4")},
    )
    assert resp.status_code == 200
    raw = resp.json()

    expected_keys = {
        "schemaVersion",
        "analysisId",
        "status",
        "exercise",
        "score",
        "repCount",
        "validReps",
        "incompleteReps",
        "reps",
        "movementMetrics",
        "issues",
        "recommendations",
        "cameraQuality",
        "video",
        "keypoints",
    }
    assert expected_keys.issubset(raw.keys())
    assert raw["schemaVersion"] == "1.0"

    parsed = VideoScoringResponse.model_validate(raw)
    _assert_valid_v1_schema(parsed, "SQUAT")
    assert parsed.status == "COMPLETED"
    assert parsed.repCount >= 1
    assert isinstance(parsed.cameraQuality, CameraQualityAssessment)
    for r in parsed.reps:
        assert isinstance(r, RepBreakdownItem)
        assert r.startTimestampMs <= r.endTimestampMs
        assert r.durationMs >= 0
    for m in parsed.movementMetrics.values():
        assert isinstance(m, MovementMetricItem)
    for issue in parsed.issues:
        assert isinstance(issue, StructuredIssueItem)
    assert len(parsed.keypoints.frames) == len(seq)
    for frame in parsed.keypoints.frames:
        assert len(frame.keypoints) == 17
        for kp in frame.keypoints:
            assert 0.0 <= kp.x <= 1.0
            assert 0.0 <= kp.y <= 1.0
            assert 0.0 <= kp.confidence <= 1.0
