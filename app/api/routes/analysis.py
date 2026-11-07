import logging
import uuid
from typing import Any, Dict, Optional
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status

from app.schemas.analysis import (
    ImageAnalysisResponse,
    VideoAnalysisResponse,
)
from app.services.analyzer import CalisthenicsAnalyzerService
from app.core.logging_context import set_request_id, get_request_id
from app.exercises.registry import exercise_registry

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["Analysis"])

# Service singleton / dependency
analyzer_service = CalisthenicsAnalyzerService()


@router.post(
    "/image",
    response_model=ImageAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze single image for user-selected Planche progression form evaluation",
    description="Uploads an image file, evaluates biomechanics against user-selected progression (Tuck, Advanced Tuck, Straddle, Full Planche), with optional side-view midpoint processing."
)
async def analyze_image(
    file: UploadFile = File(..., description="JPEG or PNG image file"),
    progression: Optional[str] = Query(
        None,
        description="User-selected progression: 'FULL_PLANCHE', 'STRADDLE_PLANCHE', 'ADVANCED_TUCK_PLANCHE', 'TUCK_PLANCHE'"
    ),
    target_progression: Optional[str] = Query(
        None,
        description="Alias for progression parameter"
    ),
    side_view: bool = Query(
        False,
        description="User-selected flag indicating whether camera is side-on (activates midpoint body-line pipeline)"
    ),
    draw_overlay: bool = Query(True, description="Whether to include Base64 annotated image overlay in response"),
    debug: bool = Query(False, description="Whether to return detailed debug keypoints and feature calculations")
) -> ImageAnalysisResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image."
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file uploaded."
            )

        chosen_progression = progression or target_progression
        response = analyzer_service.analyze_image(
            image_bytes=content,
            target_progression=chosen_progression,
            side_view=side_view,
            draw_overlay=draw_overlay,
            debug=debug
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during image analysis endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process image analysis: {str(e)}"
        )


@router.post(
    "/image/draw-skeleton",
    response_class=Response,
    status_code=status.HTTP_200_OK,
    summary="Export image with drawn pose skeleton overlay for user-selected progression",
    description="Uploads an image file and directly returns the output JPEG image file with drawn pose skeleton, keypoints, and form metrics overlay."
)
async def export_draw_skeleton(
    file: UploadFile = File(..., description="JPEG or PNG image file"),
    progression: Optional[str] = Query(
        None,
        description="User-selected progression: 'FULL_PLANCHE', 'STRADDLE_PLANCHE', 'ADVANCED_TUCK_PLANCHE', 'TUCK_PLANCHE'"
    ),
    target_progression: Optional[str] = Query(
        None,
        description="Alias for progression parameter"
    ),
    side_view: bool = Query(
        False,
        description="User-selected flag indicating whether camera is side-on"
    )
) -> Response:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a valid image."
        )

    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    try:
        content = await file.read()
        logger.debug("[DRAW_SKELETON] request_id=%s upload_bytes=%d", request_id, len(content))

        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty image file uploaded."
            )

        chosen_progression = progression or target_progression
        logger.info(
            "[DRAW_SKELETON] request_id=%s progression=%s side_view=%s filename=%s content_type=%s",
            request_id, chosen_progression, side_view,
            getattr(file, "filename", "unknown"), file.content_type
        )
        jpeg_bytes, _ = analyzer_service.export_annotated_image(
            image_bytes=content,
            target_progression=chosen_progression,
            side_view=side_view,
            request_id=request_id
        )
        logger.info("[DRAW_SKELETON] request_id=%s response_bytes=%d status=ok", request_id, len(jpeg_bytes))
        return Response(
            content=jpeg_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": "inline; filename=annotated_skeleton.jpg"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("[DRAW_SKELETON] request_id=%s error=%s", request_id, e, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export annotated skeleton image: {str(e)}"
        )


@router.post(
    "/video",
    response_model=VideoAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze video file against user-selected Planche progression",
    description="Uploads a video file (MP4/MOV/AVI), evaluates frame-by-frame form metrics with temporal smoothing and hold detection."
)
async def analyze_video(
    file: UploadFile = File(..., description="Video file (MP4, MOV, AVI)"),
    progression: Optional[str] = Query(
        None,
        description="User-selected progression: 'FULL_PLANCHE', 'STRADDLE_PLANCHE', 'ADVANCED_TUCK_PLANCHE', 'TUCK_PLANCHE'"
    ),
    target_progression: Optional[str] = Query(
        None,
        description="Alias for progression parameter"
    ),
    side_view: bool = Query(
        False,
        description="User-selected flag indicating whether camera is side-on"
    ),
    sample_fps: int = Query(5, ge=1, le=30, description="Frames per second to sample for analysis"),
    debug: bool = Query(False, description="Whether to include debug metadata in response")
) -> VideoAnalysisResponse:
    if not file.content_type or not (file.content_type.startswith("video/") or file.filename.endswith((".mp4", ".mov", ".avi", ".mkv"))):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File provided is not a supported video format."
        )

    try:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Empty video file uploaded."
            )

        chosen_progression = progression or target_progression
        response = analyzer_service.analyze_video(
            video_bytes=content,
            target_progression=chosen_progression,
            side_view=side_view,
            target_sample_fps=sample_fps,
            debug=debug
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during video analysis endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process video analysis: {str(e)}"
        )


@router.get(
    "/exercises",
    response_model=Dict[str, Any],
    summary="List supported exercises and biomechanical standards",
    description="Returns reference parameters, ideal angle thresholds, and description of supported skill analysis."
)
async def list_exercises() -> Dict[str, Any]:
    return {
        "supported_exercises": [
            {
                "id": "planche",
                "name": "Planche",
                "progressions": [
                    {
                        "id": "tuck_planche",
                        "name": "Tuck Planche",
                        "target_criteria": {
                            "elbow_extension": "Target >= 170° (fully extended straight arms)",
                            "shoulder_forward_lean": "Target >= 15° forward lean past wrists base",
                            "hip_angle": "Target <= 115° (knees & hips tucked to chest)",
                            "knee_angle": "Target <= 105° (compact knee tuck)",
                            "torso_inclination": "Target <= 15° relative to ground horizontal"
                        }
                    },
                    {
                        "id": "adv_tuck_planche",
                        "name": "Advanced Tuck Planche",
                        "target_criteria": {
                            "elbow_extension": "Target >= 170°",
                            "shoulder_forward_lean": "Target >= 15°",
                            "hip_angle": "Target 90°–130° (hips untucked, 90° flat back angle)",
                            "knee_angle": "Target 60°–135° (knees bent ~90°)",
                            "torso_inclination": "Target <= 15°"
                        }
                    },
                    {
                        "id": "straddle_planche",
                        "name": "Straddle Planche",
                        "target_criteria": {
                            "elbow_extension": "Target >= 170°",
                            "shoulder_forward_lean": "Target >= 15°",
                            "hip_angle": "Target >= 140° (extended hips)",
                            "knee_angle": "Target >= 140° (straight legs)",
                            "leg_spread": "Target ratio >= 1.25 (legs spread wide apart)",
                            "torso_inclination": "Target <= 15°"
                        }
                    },
                    {
                        "id": "full_planche",
                        "name": "Full Planche",
                        "target_criteria": {
                            "elbow_extension": "Target >= 170°",
                            "shoulder_forward_lean": "Target >= 15°",
                            "hip_angle": "Target >= 140° (straight body line)",
                            "knee_angle": "Target >= 140° (straight legs)",
                            "leg_spread": "Target ratio < 1.25 (legs together)",
                            "torso_inclination": "Target <= 15°"
                        }
                    }
                ]
            },
            {
                "id": "front_lever",
                "name": "Front Lever",
                "progressions": [
                    {"id": "tuck_front_lever", "name": "Tuck Front Lever", "target_criteria": {"elbow_extension": ">= 170°", "hip_angle": "<= 100°"}},
                    {"id": "adv_tuck_front_lever", "name": "Advanced Tuck Front Lever", "target_criteria": {"elbow_extension": ">= 170°", "hip_angle": "90°–130°"}},
                    {"id": "straddle_front_lever", "name": "Straddle Front Lever", "target_criteria": {"elbow_extension": ">= 170°", "hip_angle": ">= 160°", "leg_spread": ">= 1.25"}},
                    {"id": "full_front_lever", "name": "Full Front Lever", "target_criteria": {"elbow_extension": ">= 170°", "hip_angle": ">= 165°", "knee_angle": ">= 165°"}}
                ]
            },
            {
                "id": "handstand",
                "name": "Handstand & Handstand Push-up",
                "progressions": [
                    {"id": "handstand", "name": "Handstand Hold", "target_criteria": {"elbow_extension": ">= 170°", "shoulder_angle": ">= 165°", "body_line": ">= 168°"}},
                    {"id": "handstand_pushup", "name": "Handstand Push-up", "target_criteria": {"depth_elbow": "<= 95°", "lockout_elbow": ">= 168°"}}
                ]
            },
            {
                "id": "pull_up",
                "name": "Pull-Up",
                "progressions": [
                    {"id": "pull_up", "name": "Strict Pull-Up", "target_criteria": {"dead_hang_elbow": ">= 160°", "top_chin_clearance_elbow": "<= 65°", "symmetry": "<= 15° difference"}}
                ]
            },
            {
                "id": "dips",
                "name": "Parallel Bar Dips",
                "progressions": [
                    {"id": "dips", "name": "Parallel Bar Dips", "target_criteria": {"depth_elbow": "<= 90°", "top_lockout_elbow": ">= 165°"}}
                ]
            },
            {
                "id": "muscle_up",
                "name": "Muscle-Up",
                "progressions": [
                    {"id": "muscle_up", "name": "Bar Muscle-Up", "target_criteria": {"transition_symmetry": "Simultaneous elbow transition (no chicken wing)", "dip_lockout": ">= 165°"}}
                ]
            }
        ],
        "all_catalog": exercise_registry.list_supported_exercises()
    }
