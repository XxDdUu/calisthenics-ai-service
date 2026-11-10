from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class ExerciseType(str, Enum):
    PLANCHE = "planche"
    TUCK_PLANCHE = "tuck_planche"
    ADVANCED_TUCK_PLANCHE = "adv_tuck_planche"
    STRADDLE_PLANCHE = "straddle_planche"
    FULL_PLANCHE = "full_planche"
    FRONT_LEVER = "front_lever"
    FULL_FRONT_LEVER = "full_front_lever"
    STRADDLE_FRONT_LEVER = "straddle_front_lever"
    ADV_TUCK_FRONT_LEVER = "adv_tuck_front_lever"
    TUCK_FRONT_LEVER = "tuck_front_lever"
    HANDSTAND = "handstand"
    HANDSTAND_PUSHUP = "handstand_pushup"
    PULL_UP = "pull_up"
    DIPS = "dips"
    MUSCLE_UP = "muscle_up"


class ProgressionStage(str, Enum):
    TUCK_PLANCHE = "Tuck Planche"
    ADV_TUCK_PLANCHE = "Advanced Tuck Planche"
    STRADDLE_PLANCHE = "Straddle Planche"
    FULL_PLANCHE = "Full Planche"
    INCOMPLETE_ATTEMPT = "Incomplete / Poor Form Hold"
    NO_PLANCHE_DETECTED = "No Planche Detected"


class AssessmentStatus(str, Enum):
    VALID_ATTEMPT = "VALID_ATTEMPT"
    INCOMPLETE_ATTEMPT = "INCOMPLETE_ATTEMPT"
    INVALID_INPUT = "INVALID_INPUT"
    INSUFFICIENT_POSE_QUALITY = "INSUFFICIENT_POSE_QUALITY"
    NO_POSE_DETECTED = "NO_POSE_DETECTED"
    NO_VALID_HOLD = "NO_VALID_HOLD"
    VIDEO_TOO_SHORT = "VIDEO_TOO_SHORT"
    ANALYSIS_SUCCESS = "ANALYSIS_SUCCESS"


class MetricStatus(str, Enum):
    PASS = "PASS"
    GOOD = "GOOD"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    FAIL = "FAIL"


class IssueSeverity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


def parse_progression_stage(prog: Optional[Union[str, ProgressionStage]]) -> ProgressionStage:
    """Safely parse user-supplied progression strings into ProgressionStage enum."""
    if not prog:
        return ProgressionStage.FULL_PLANCHE

    if isinstance(prog, ProgressionStage):
        return prog

    clean = str(prog).strip().lower().replace(" ", "_").replace("-", "_")

    if "tuck" in clean and "adv" not in clean and "advanced" not in clean:
        return ProgressionStage.TUCK_PLANCHE
    elif "adv" in clean or "advanced" in clean:
        return ProgressionStage.ADV_TUCK_PLANCHE
    elif "straddle" in clean:
        return ProgressionStage.STRADDLE_PLANCHE
    elif "full" in clean:
        return ProgressionStage.FULL_PLANCHE

    for p in ProgressionStage:
        if p.value.lower() == str(prog).strip().lower() or p.name.lower() == clean:
            return p

    return ProgressionStage.FULL_PLANCHE


class KeypointData(BaseModel):
    x: float = Field(..., description="X coordinate in pixels")
    y: float = Field(..., description="Y coordinate in pixels")
    confidence: float = Field(..., description="Confidence score [0.0, 1.0]")
    name: str = Field(..., description="COCO keypoint name")
    visible: bool = Field(..., description="Whether keypoint is reliably detected")


class PoseKeypoints(BaseModel):
    keypoints: Dict[str, KeypointData] = Field(..., description="Dictionary of mapped keypoints")
    bounding_box: List[float] = Field(..., description="[x1, y1, x2, y2] bounding box")
    detection_confidence: float = Field(..., description="Overall detection confidence")
    derived_landmarks: Dict[str, KeypointData] = Field(default_factory=dict, description="Derived center landmarks (shoulder, hip, wrist, ankle, body)")


class JointAngles(BaseModel):
    elbow_left_deg: float = Field(..., description="Left elbow extension angle in degrees")
    elbow_right_deg: float = Field(..., description="Right elbow extension angle in degrees")
    primary_elbow_deg: float = Field(..., description="Active side elbow extension angle")

    shoulder_lean_left_deg: float = Field(..., description="Left shoulder forward lean angle from vertical")
    shoulder_lean_right_deg: float = Field(..., description="Right shoulder forward lean angle from vertical")
    primary_shoulder_lean_deg: float = Field(..., description="Active side forward lean angle")

    hip_angle_deg: float = Field(..., description="Hip angle (Shoulder-Hip-Knee/Ankle)")
    knee_left_deg: float = Field(..., description="Left knee bend angle")
    knee_right_deg: float = Field(..., description="Right knee bend angle")
    primary_knee_deg: float = Field(..., description="Active side knee angle")

    body_horizontal_tilt_deg: float = Field(..., description="Angle of main body line relative to ground")
    primary_side: str = Field("left", description="Detected primary view side ('left' or 'right')")
    leg_spread_ratio: Optional[float] = Field(None, description="Normalized leg spread ratio (ankle distance / hip width)")


class AlignmentMetrics(BaseModel):
    body_straightness_score: float = Field(..., description="Score 0-100 for torso line straightness")
    scapular_protraction_score: float = Field(..., description="Score 0-100 indicating scapular protraction level")
    horizontal_level_score: float = Field(..., description="Score 0-100 for torso ground-parallel alignment")
    center_of_mass_over_hands: bool = Field(..., description="True if Center of Mass is balanced over support base")
    center_of_mass_offset_px: float = Field(..., description="Horizontal pixel distance of CoM from wrist base")


class MetricEvaluationDetail(BaseModel):
    name: str = Field(..., description="Human readable metric name")
    measured_value: float = Field(..., description="Measured numerical value")
    target_range: str = Field(..., description="Acceptable target range string")
    score: float = Field(..., description="Individual metric score [0.0, 100.0]")
    status: MetricStatus = Field(..., description="Evaluation status for this metric")


class StructuredIssue(BaseModel):
    metric: str = Field(..., description="Key identifier of failed metric")
    severity: IssueSeverity = Field(..., description="Issue severity level")
    message: str = Field(..., description="Detailed explanation of the form issue")
    measured_value: Optional[float] = Field(None, description="Measured value")
    target_range: Optional[str] = Field(None, description="Target range")


class PlancheAnalysisResult(BaseModel):
    exercise_type: ExerciseType = Field(ExerciseType.PLANCHE, description="Evaluated exercise type")
    target_progression: ProgressionStage = Field(..., description="Target progression stage selected for evaluation")
    detected_progression: ProgressionStage = Field(..., description="Alias for target progression")
    side_view: bool = Field(False, description="Whether side-view analysis pipeline was used")
    assessment_status: AssessmentStatus = Field(AssessmentStatus.VALID_ATTEMPT, description="Assessment input status")
    form_score: float = Field(..., description="Overall form score from 0.0 to 100.0")
    is_valid_hold: bool = Field(..., description="True if form meets minimal standards for valid hold")

    joint_angles: JointAngles = Field(..., description="Calculated biomechanical joint angles")
    alignment: AlignmentMetrics = Field(..., description="Body alignment & position metrics")

    component_scores: Dict[str, float] = Field(default_factory=dict, description="Interpretable component scores")
    metrics: Dict[str, MetricEvaluationDetail] = Field(default_factory=dict, description="Per-metric evaluation details")
    structured_issues: List[StructuredIssue] = Field(default_factory=list, description="Structured issues with severity & messages")

    form_errors: List[str] = Field(default_factory=list, description="List of detected form errors")
    positive_feedback: List[str] = Field(default_factory=list, description="List of correct technique points")
    actionable_cues: List[str] = Field(default_factory=list, description="Coaching tips and corrections")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Detailed debug metrics when debug mode is enabled")


class ImageAnalysisResponse(BaseModel):
    success: bool = Field(..., description="Analysis status")
    message: str = Field(..., description="Summary status message")
    processing_time_ms: float = Field(..., description="Execution time in milliseconds")
    pose_detected: bool = Field(..., description="Whether a person pose was detected")
    side_view: bool = Field(False, description="Whether image was analyzed in side-view mode")
    pose_keypoints: Optional[PoseKeypoints] = Field(None, description="Raw pose keypoints if detected")
    planche_analysis: Optional[PlancheAnalysisResult] = Field(None, description="Detailed Planche biomechanics evaluation")
    annotated_image_base64: Optional[str] = Field(None, description="Base64 encoded JPEG with pose overlay")


class VideoMetadata(BaseModel):
    """Original video file metadata extracted before analysis."""
    duration_ms: int = Field(..., description="Original video duration in milliseconds")
    width: int = Field(..., description="Original video frame width in pixels")
    height: int = Field(..., description="Original video frame height in pixels")
    original_fps: float = Field(..., description="Original video frames per second (not sampling rate)")
    total_frames: int = Field(..., description="Total frame count in original video")
    analyzed_frames: int = Field(..., description="Number of frames actually sampled and analyzed")
    sampling_fps: float = Field(..., description="Actual analysis sampling rate used (frames per second)")
    file_size_bytes: int = Field(..., description="Original uploaded video file size in bytes")


class VideoFrameAnalysis(BaseModel):
    frame_index: int = Field(..., description="Frame index in original video")
    timestamp_sec: float = Field(..., description="Timestamp in seconds")
    pose_detected: bool = Field(..., description="Pose detected in frame")
    form_score: float = Field(..., description="Form score for this frame")
    is_valid_hold: bool = Field(..., description="Valid hold criteria met")
    hold_state: Optional[str] = Field(None, description="Hold state from detector")
    detected_progression: str = Field(..., description="Target progression level in this frame")
    key_issues: List[str] = Field(default_factory=list, description="Errors detected in frame")
    # Per-frame biomechanical angles (None when pose not detected or landmark missing)
    elbow_angle: Optional[float] = Field(None, description="Primary elbow extension angle in degrees")
    shoulder_angle: Optional[float] = Field(None, description="Primary shoulder forward lean angle in degrees")
    hip_alignment: Optional[float] = Field(None, description="Hip angle (Shoulder-Hip-Knee) in degrees")
    body_line_deviation: Optional[float] = Field(None, description="Body horizontal tilt deviation from ground in degrees")
    shoulder_lean: Optional[float] = Field(None, description="Normalized shoulder horizontal displacement over wrists")
    knee_angle: Optional[float] = Field(None, description="Primary knee extension angle in degrees")
    pose_confidence: Optional[float] = Field(None, description="Overall pose detection confidence [0.0, 1.0]")


class StabilityMetrics(BaseModel):
    """Stability statistics computed across all analyzed frames with valid pose."""
    average_body_line_deviation: float = Field(..., description="Mean body-line tilt deviation in degrees")
    max_body_line_deviation: float = Field(..., description="Maximum body-line tilt deviation in degrees")
    shoulder_position_variation: float = Field(..., description="Std dev of shoulder lean values across hold frames")
    hip_position_variation: float = Field(..., description="Std dev of hip alignment values across hold frames")
    average_pose_confidence: float = Field(..., description="Mean landmark detection confidence across analyzed frames")
    stability_score: float = Field(..., description="Computed stability score 0–100 (higher = more stable)")


class VideoAnalysisResponse(BaseModel):
    success: bool = Field(..., description="Analysis status")
    message: str = Field(..., description="Summary status message")
    processing_time_ms: float = Field(..., description="Total execution time in milliseconds")
    # Legacy fields (kept for backwards compatibility)
    total_frames: int = Field(..., description="Total video frames in original video")
    fps: float = Field(..., description="Original video frames per second")
    duration_sec: float = Field(..., description="Total video duration in seconds")
    # Rich video metadata
    video_metadata: Optional[VideoMetadata] = Field(None, description="Detailed original video file metadata")

    total_hold_duration_sec: float = Field(..., description="Total time planche position was attempted")
    valid_hold_duration_sec: float = Field(..., description="Total time planche was held with valid form (score >= 70)")

    peak_form_score: float = Field(..., description="Highest frame form score achieved")
    average_form_score: float = Field(..., description="Average form score during hold")
    primary_progression: ProgressionStage = Field(..., description="Target progression stage evaluated across video")
    side_view: bool = Field(False, description="Whether video was analyzed in side-view mode")
    component_scores: Optional[Dict[str, float]] = Field(None, description="Aggregated component scores across hold frames")
    stability: Optional[StabilityMetrics] = Field(None, description="Body stability metrics across analyzed frames")

    timeline: List[VideoFrameAnalysis] = Field(default_factory=list, description="Sampled frame metrics with per-frame biomechanics")
    summary_feedback: List[str] = Field(default_factory=list, description="Overall technique strengths and weaknesses")
    key_recommendations: List[str] = Field(default_factory=list, description="Actionable training recommendations")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Video debug metadata if requested")


class RealtimeStreamResponse(BaseModel):
    type: str = Field("frame_result", description="Message payload type")
    timestamp_ms: float = Field(..., description="Timestamp of processed frame in milliseconds")
    processing_time_ms: float = Field(..., description="Frame processing latency in milliseconds")
    pose_detected: bool = Field(..., description="True if pose keypoints detected")
    form_score: float = Field(0.0, description="Real-time form score [0.0 - 100.0]")
    is_valid_hold: bool = Field(False, description="True if form meets valid hold standards")
    current_hold_duration_sec: float = Field(0.0, description="Active continuous hold duration in seconds")
    peak_form_score: float = Field(0.0, description="Peak form score recorded in current live camera session")
    detected_progression: str = Field("None", description="Target planche stage evaluated")

    joint_angles: Optional[JointAngles] = Field(None, description="Calculated joint angles")
    alignment: Optional[AlignmentMetrics] = Field(None, description="Alignment metrics")
    form_errors: List[str] = Field(default_factory=list, description="Real-time detected form errors")
    actionable_cues: List[str] = Field(default_factory=list, description="Instant coaching cues")
    annotated_frame_base64: Optional[str] = Field(None, description="Base64 encoded JPEG overlay frame")


class LiveAnalysisStatus(str, Enum):
    VALID = "VALID"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INCOMPLETE_POSE = "INCOMPLETE_POSE"
    NO_PERSON_DETECTED = "NO_PERSON_DETECTED"
    INVALID_INPUT = "INVALID_INPUT"


class FeedbackSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LiveFeedbackItem(BaseModel):
    code: str = Field(..., description="Stable feedback identifier code")
    message: str = Field(..., description="Human-readable coaching cue")
    severity: FeedbackSeverity = Field(FeedbackSeverity.WARNING, description="Severity: INFO, WARNING, ERROR")


class LandmarkItem(BaseModel):
    name: str = Field(..., description="COCO joint landmark name")
    x: float = Field(..., description="Original image pixel X coordinate")
    y: float = Field(..., description="Original image pixel Y coordinate")
    x_norm: float = Field(..., description="Normalized [0.0, 1.0] X coordinate")
    y_norm: float = Field(..., description="Normalized [0.0, 1.0] Y coordinate")
    confidence: float = Field(..., description="Detection confidence score [0.0, 1.0]")
    visible: bool = Field(..., description="True if keypoint confidence meets threshold")


class LiveScoringMetric(BaseModel):
    name: str = Field(..., description="Biomechanical metric name (e.g., body_alignment, range_of_motion)")
    value: float = Field(..., description="Raw biomechanical measurement value (degrees/ratio)")
    score: float = Field(..., description="Form score 0-100 for this metric")
    weight: float = Field(..., description="Weight [0.0, 1.0] in overall exercise evaluation")
    confidence: float = Field(1.0, description="Confidence/reliability of landmark detection [0.0, 1.0]")
    status: str = Field("VALID", description="Metric evaluation status: VALID, LOW_CONFIDENCE, INSUFFICIENT_DATA")


class LiveTiming(BaseModel):
    preprocessing_ms: float = Field(0.0, description="Preprocessing latency in ms")
    inference_ms: float = Field(0.0, description="YOLO pose inference latency in ms")
    analysis_ms: float = Field(0.0, description="Biomechanical analysis latency in ms")
    total_ms: float = Field(0.0, description="Total pipeline latency in ms")


class LiveAnalysisResponse(BaseModel):
    status: LiveAnalysisStatus = Field(..., description="Analysis status: VALID, LOW_CONFIDENCE, INCOMPLETE_POSE, NO_PERSON_DETECTED, INVALID_INPUT")
    exercise: str = Field(..., description="Evaluated exercise identifier")
    timestamp: int = Field(0, alias="timestampMs", description="Capture timestamp in milliseconds")
    frame_sequence: int = Field(0, alias="frameSequence", description="Monotonically increasing frame sequence number")
    pose_detected: bool = Field(False, alias="poseDetected", description="True if human pose is detected")
    confidence: float = Field(0.0, description="Overall pose detection confidence [0.0, 1.0]")
    score: Optional[float] = Field(None, description="Biomechanical form score [0.0, 100.0]")
    current_score: Optional[float] = Field(None, alias="currentScore", description="Alias for score")
    score_confidence: Optional[float] = Field(None, alias="scoreConfidence", description="Reliability score [0.0, 1.0]")
    is_valid_hold: bool = Field(False, alias="isValidHold", description="True if form meets valid hold standards")
    hold_duration_sec: Optional[float] = Field(None, alias="holdDurationSec", description="Continuous valid hold duration in seconds")
    hold_duration_ms: Optional[int] = Field(None, alias="holdDurationMs", description="Hold duration in milliseconds")
    rep_count: Optional[int] = Field(None, alias="repCount", description="Repetition count for dynamic exercises")
    rep_phase: Optional[str] = Field(None, alias="repPhase", description="Current repetition phase (e.g., START, INFLECTION, LOCKOUT)")
    phase: Optional[str] = Field(None, alias="phase", description="Current movement phase")
    current_state: Optional[str] = Field(None, alias="currentState", description="Current squat or push-up state")
    knee_angle: Optional[float] = Field(None, alias="kneeAngle", description="Primary knee flexion angle in degrees")
    min_knee_angle: Optional[float] = Field(None, alias="minKneeAngle", description="Minimum knee flexion angle reached in current rep")
    elbow_angle: Optional[float] = Field(None, alias="elbowAngle", description="Primary elbow angle in degrees")
    left_elbow_angle: Optional[float] = Field(None, alias="leftElbowAngle", description="Left elbow angle in degrees")
    right_elbow_angle: Optional[float] = Field(None, alias="rightElbowAngle", description="Right elbow angle in degrees")
    average_elbow_angle: Optional[float] = Field(None, alias="averageElbowAngle", description="Average elbow angle in degrees")
    min_elbow_angle: Optional[float] = Field(None, alias="minElbowAngle", description="Minimum elbow angle reached in current rep")
    body_alignment: Optional[float] = Field(None, alias="bodyAlignment", description="Body alignment (shoulder-hip-ankle) in degrees")
    form_score: Optional[float] = Field(None, alias="formScore", description="Exercise form score")
    depth: Optional[str] = Field(None, alias="depth", description="Depth classification: GOOD, MODERATE, SHALLOW")
    is_tracking: Optional[bool] = Field(None, alias="isTracking", description="Whether athlete joints are currently reliably tracked")
    measurements: Dict[str, float] = Field(default_factory=dict, description="Joint angles and alignment measurements")
    scoring_metrics: List[LiveScoringMetric] = Field(default_factory=list, alias="metrics", description="Detailed scoring metrics breakdown")
    raw_measurements: Optional[Dict[str, float]] = Field(None, alias="rawMeasurements", description="Raw unsmoothed measurements before temporal filter")
    feedback: List[LiveFeedbackItem] = Field(default_factory=list, description="List of actionable coaching cues")
    missing_landmarks: List[str] = Field(default_factory=list, alias="missingLandmarks", description="Missing or unreliable landmark names")
    landmarks: List[LandmarkItem] = Field(default_factory=list, description="Detected keypoints with pixel and normalized coordinates")
    annotated_frame_base64: Optional[str] = Field(None, alias="annotatedFrameBase64", description="Optional base64 JPEG image with overlay skeleton")
    timing: Optional[LiveTiming] = Field(None, description="Latency breakdown in milliseconds")
    session_id: Optional[str] = Field(None, alias="sessionId", description="Associated analysis session ID")
    session_summary: Optional[Dict[str, Any]] = Field(None, alias="sessionSummary", description="End of session performance summary")

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }


class LiveFrameRequest(BaseModel):
    image_base64: Optional[str] = Field(None, alias="imageBase64", description="Base64 encoded JPEG or PNG image frame")
    exercise: str = Field("FULL_PLANCHE", description="Exercise code (e.g., FULL_PLANCHE, FRONT_LEVER, HANDSTAND)")
    session_id: Optional[str] = Field(None, alias="sessionId", description="Optional session ID for temporal smoothing")
    frame_sequence: int = Field(0, alias="frameSequence", description="Frame sequence counter")
    timestamp_ms: int = Field(0, alias="timestampMs", description="Frame timestamp in ms")
    side_view: bool = Field(False, alias="sideView", description="True if camera is positioned side-on")
    draw_overlay: bool = Field(False, alias="drawOverlay", description="Whether to include annotated overlay image in response")
    debug: bool = Field(False, description="Whether to include raw measurements and debug diagnostics")

    model_config = {
        "populate_by_name": True
    }


class LiveSessionInfo(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    exercise: str = Field(..., description="Exercise associated with session")
    created_at: float = Field(..., description="Session start epoch timestamp")
    frame_count: int = Field(0, description="Total frames processed in session")
    peak_score: float = Field(0.0, description="Peak form score recorded")
    current_hold_duration_sec: float = Field(0.0, description="Current hold duration in seconds")
    rep_count: int = Field(0, description="Current completed rep count")

    model_config = {
        "populate_by_name": True
    }


# ==============================================================================
# End-to-End AI Video Scoring & Keypoints Visualization Schemas
# ==============================================================================

class VideoKeypointItem(BaseModel):
    name: str = Field(..., description="COCO joint landmark name (e.g. left_shoulder)")
    x: float = Field(..., description="Normalized [0.0, 1.0] X coordinate relative to video width")
    y: float = Field(..., description="Normalized [0.0, 1.0] Y coordinate relative to video height")
    confidence: float = Field(..., description="Detection confidence score [0.0, 1.0]")


class VideoKeypointFrame(BaseModel):
    frameIndex: int = Field(..., description="Original video frame index")
    timestampMs: int = Field(..., description="Frame timestamp in milliseconds")
    keypoints: List[VideoKeypointItem] = Field(default_factory=list, description="17 COCO keypoints")


class VideoKeypointsData(BaseModel):
    format: str = Field("COCO_17", description="Pose model landmark topology")
    coordinateSystem: str = Field("NORMALIZED", description="Normalized 0.0-1.0 coordinates")
    frames: List[VideoKeypointFrame] = Field(default_factory=list, description="All analyzed video frames keypoints")


class VideoOutputMetadata(BaseModel):
    originalAvailable: bool = Field(True, description="Original video availability")
    annotatedAvailable: bool = Field(True, description="Annotated skeleton video availability")
    originalObjectKey: Optional[str] = Field(None, description="MinIO object key for original video")
    annotatedObjectKey: Optional[str] = Field(None, description="MinIO object key for annotated video")
    durationMs: int = Field(..., description="Original video duration in milliseconds")
    width: int = Field(..., description="Video frame width in pixels")
    height: int = Field(..., description="Video frame height in pixels")
    fps: float = Field(..., description="Video frame rate (frames per second)")


class VideoScoringResponse(BaseModel):
    analysisId: str = Field(..., description="Unique UUID for this analysis session")
    status: str = Field(..., description="Status: COMPLETED, FAILED, NO_PERSON_DETECTED, INSUFFICIENT_POSE_QUALITY")
    exerciseId: Optional[Any] = Field(None, description="Exercise identifier (e.g., ID or code)")
    exercise: str = Field("PLANCHE", description="Normalized exercise name")
    score: float = Field(..., description="Overall movement quality score [0.0, 100.0]")
    formAccuracy: float = Field(0.0, description="Form accuracy score [0.0, 100.0]")
    stability: float = Field(0.0, description="Stability score [0.0, 100.0]")
    alignment: float = Field(0.0, description="Body alignment score [0.0, 100.0]")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Exercise-specific metrics and angles")
    feedback: List[str] = Field(default_factory=list, description="Actionable mistakes and coaching suggestions")
    video: VideoOutputMetadata = Field(..., description="Video metadata and availability")
    keypoints: VideoKeypointsData = Field(..., description="Per-frame 17 COCO keypoint timeline")


