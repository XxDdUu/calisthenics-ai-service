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
    HANDSTAND = "handstand"


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


class VideoFrameAnalysis(BaseModel):
    frame_index: int = Field(..., description="Frame index")
    timestamp_sec: float = Field(..., description="Timestamp in seconds")
    pose_detected: bool = Field(..., description="Pose detected in frame")
    form_score: float = Field(..., description="Form score for this frame")
    is_valid_hold: bool = Field(..., description="Valid hold criteria met")
    hold_state: Optional[str] = Field(None, description="Hold state from detector")
    detected_progression: str = Field(..., description="Target progression level in this frame")
    key_issues: List[str] = Field(default_factory=list, description="Errors detected in frame")


class VideoAnalysisResponse(BaseModel):
    success: bool = Field(..., description="Analysis status")
    message: str = Field(..., description="Summary status message")
    processing_time_ms: float = Field(..., description="Total execution time in milliseconds")
    total_frames: int = Field(..., description="Total video frames processed")
    fps: float = Field(..., description="Video frames per second")
    duration_sec: float = Field(..., description="Total video duration in seconds")

    total_hold_duration_sec: float = Field(..., description="Total time planche position was attempted")
    valid_hold_duration_sec: float = Field(..., description="Total time planche was held with valid form (score >= 70)")

    peak_form_score: float = Field(..., description="Highest frame form score achieved")
    average_form_score: float = Field(..., description="Average form score during hold")
    primary_progression: ProgressionStage = Field(..., description="Target progression stage evaluated across video")
    side_view: bool = Field(False, description="Whether video was analyzed in side-view mode")
    component_scores: Optional[Dict[str, float]] = Field(None, description="Aggregated component scores across hold frames")

    timeline: List[VideoFrameAnalysis] = Field(default_factory=list, description="Sampled frame metrics")
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
