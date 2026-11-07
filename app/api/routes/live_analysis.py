import asyncio
import base64
import json
import logging
import time
import uuid
from typing import Any, Dict, Optional
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from starlette.concurrency import run_in_threadpool

from app.core.logging_context import get_request_id, set_request_id
from app.exercises.registry import exercise_registry
from app.schemas.analysis import (
    LiveAnalysisResponse,
    LiveAnalysisStatus,
    LiveFrameRequest,
    LiveSessionInfo,
)
from app.services.live_pipeline import live_pipeline
from app.services.live_session import session_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis/live", tags=["Live Analysis"])


@router.post(
    "/frame",
    response_model=LiveAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Process single live camera frame",
    description="Submits a camera frame for real-time biomechanical analysis, form scoring, and coaching feedback. Supports multipart image file upload or JSON payload."
)
async def analyze_live_frame(
    file: Optional[UploadFile] = File(None, description="Camera frame image (JPEG or PNG)"),
    exercise: str = Form("FULL_PLANCHE", description="Exercise identifier (e.g. FULL_PLANCHE, FRONT_LEVER, HANDSTAND)"),
    session_id: Optional[str] = Form(None, description="Optional session ID for temporal smoothing"),
    frame_sequence: int = Form(0, description="Sequential frame counter"),
    timestamp_ms: int = Form(0, description="Frame timestamp in milliseconds"),
    side_view: bool = Form(False, description="Flag indicating side-on camera view"),
    draw_overlay: bool = Form(False, description="Whether to include annotated JPEG overlay in response"),
    debug: bool = Form(False, description="Whether to include raw diagnostic measurements")
) -> LiveAnalysisResponse:
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)

    if file is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image file provided in multipart payload."
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file content type '{file.content_type}'. Image file required."
        )

    image_bytes = await file.read()
    if not image_bytes or len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded image frame is empty."
        )

    # Execute CPU-bound inference in thread pool to avoid blocking async event loop
    response = await run_in_threadpool(
        live_pipeline.process_frame,
        image_bytes=image_bytes,
        exercise=exercise,
        session_id=session_id,
        frame_sequence=frame_sequence,
        timestamp_ms=timestamp_ms,
        side_view=side_view,
        draw_overlay=draw_overlay,
        debug=debug,
        request_id=request_id
    )

    return response


@router.post(
    "/frame-json",
    response_model=LiveAnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Process live camera frame via JSON Base64 payload",
    description="Allows sending a base64-encoded frame in a standard JSON body."
)
async def analyze_live_frame_json(
    request: LiveFrameRequest
) -> LiveAnalysisResponse:
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)

    if not request.image_base64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Field 'image_base64' is required in JSON payload."
        )

    try:
        # Strip potential data URL prefix
        b64_str = request.image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_str)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 encoded image string: {str(e)}"
        )

    response = await run_in_threadpool(
        live_pipeline.process_frame,
        image_bytes=image_bytes,
        exercise=request.exercise,
        session_id=request.session_id,
        frame_sequence=request.frame_sequence,
        timestamp_ms=request.timestamp_ms,
        side_view=request.side_view,
        draw_overlay=request.draw_overlay,
        debug=request.debug,
        request_id=request_id
    )

    return response


# Alias endpoint: POST /api/v1/analysis/live -> analyze_live_frame
@router.post(
    "",
    response_model=LiveAnalysisResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False
)
async def analyze_live_alias(
    file: Optional[UploadFile] = File(None),
    exercise: str = Form("FULL_PLANCHE"),
    session_id: Optional[str] = Form(None),
    frame_sequence: int = Form(0),
    timestamp_ms: int = Form(0),
    side_view: bool = Form(False),
    draw_overlay: bool = Form(False),
    debug: bool = Form(False)
) -> LiveAnalysisResponse:
    return await analyze_live_frame(
        file=file,
        exercise=exercise,
        session_id=session_id,
        frame_sequence=frame_sequence,
        timestamp_ms=timestamp_ms,
        side_view=side_view,
        draw_overlay=draw_overlay,
        debug=debug
    )


@router.post(
    "/session",
    response_model=LiveSessionInfo,
    status_code=status.HTTP_200_OK,
    summary="Initialize a live camera analysis session",
    description="Registers an active session for temporal smoothing and tracking."
)
async def init_session(
    exercise: str = Query("FULL_PLANCHE", description="Exercise code"),
    session_id: Optional[str] = Query(None, description="Optional custom session ID"),
    side_view: bool = Query(False, description="Camera perspective")
) -> LiveSessionInfo:
    session = session_manager.get_or_create(
        session_id=session_id,
        exercise=exercise,
        side_view=side_view
    )
    return LiveSessionInfo(
        session_id=session.session_id,
        exercise=session.exercise,
        created_at=session.created_at,
        frame_count=session.frame_count,
        peak_score=session.peak_score,
        current_hold_duration_sec=session.current_hold_duration_sec,
        rep_count=session.rep_count
    )


@router.post(
    "/session/{session_id}/reset",
    status_code=status.HTTP_200_OK,
    summary="Reset live session state",
    description="Clears temporal smoothing history, hold timer, and repetition counts."
)
async def reset_session(session_id: str) -> Dict[str, Any]:
    success = session_manager.reset(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found."
        )
    return {"session_id": session_id, "status": "RESET_SUCCESS"}


@router.delete(
    "/session/{session_id}",
    status_code=status.HTTP_200_OK,
    summary="Terminate live session",
    description="Releases all session-specific temporal smoothing and memory resources."
)
async def terminate_session(session_id: str) -> Dict[str, Any]:
    success = session_manager.remove(session_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found."
        )
    return {"session_id": session_id, "status": "TERMINATED"}


@router.websocket("/ws")
async def websocket_live_stream(websocket: WebSocket):
    """
    WebSocket endpoint for bidirectional real-time camera streaming.
    Streams camera frames and receives instantaneous form evaluation, angles, and cues.
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())[:12]
    current_exercise = "FULL_PLANCHE"
    side_view = False
    logger.info("WebSocket client connected. Assigned session_id=%s", session_id)

    try:
        while True:
            # Handle incoming message: either text (JSON) or binary (raw image frame bytes)
            message = await websocket.receive()
            msg_type = message.get("type")

            if msg_type == "websocket.disconnect":
                break

            if "text" in message:
                text_data = message["text"]
                try:
                    data = json.loads(text_data)
                except json.JSONDecodeError:
                    await websocket.send_json({"status": "ERROR", "message": "Invalid JSON format."})
                    continue

                action = data.get("action") or data.get("type", "frame")

                if action in ("init", "configure"):
                    current_exercise = data.get("exercise", current_exercise)
                    side_view = bool(data.get("side_view", side_view))
                    session_manager.get_or_create(session_id, current_exercise, side_view=side_view)
                    await websocket.send_json({
                        "status": "CONFIGURED",
                        "session_id": session_id,
                        "exercise": current_exercise
                    })

                elif action == "reset":
                    session_manager.reset(session_id)
                    await websocket.send_json({
                        "status": "RESET_SUCCESS",
                        "session_id": session_id
                    })

                elif action == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": time.time()})

                elif action == "frame":
                    b64_image = data.get("image_base64") or data.get("frame")
                    if not b64_image:
                        await websocket.send_json({"status": "ERROR", "message": "Missing image_base64 in frame."})
                        continue

                    try:
                        if "," in b64_image:
                            b64_image = b64_image.split(",", 1)[1]
                        img_bytes = base64.b64decode(b64_image)
                    except Exception as e:
                        await websocket.send_json({"status": "ERROR", "message": f"Base64 decode failed: {e}"})
                        continue

                    ex = data.get("exercise", current_exercise)
                    seq = int(data.get("frame_sequence", 0))
                    ts = int(data.get("timestamp_ms", int(time.time() * 1000)))
                    sv = bool(data.get("side_view", side_view))
                    overlay = bool(data.get("draw_overlay", False))
                    dbg = bool(data.get("debug", False))

                    response = await run_in_threadpool(
                        live_pipeline.process_frame,
                        image_bytes=img_bytes,
                        exercise=ex,
                        session_id=session_id,
                        frame_sequence=seq,
                        timestamp_ms=ts,
                        side_view=sv,
                        draw_overlay=overlay,
                        debug=dbg
                    )
                    await websocket.send_text(response.model_dump_json(by_alias=False))

            elif "bytes" in message:
                img_bytes = message["bytes"]
                if img_bytes:
                    response = await run_in_threadpool(
                        live_pipeline.process_frame,
                        image_bytes=img_bytes,
                        exercise=current_exercise,
                        session_id=session_id,
                        frame_sequence=0,
                        timestamp_ms=int(time.time() * 1000),
                        side_view=side_view,
                        draw_overlay=False,
                        debug=False
                    )
                    await websocket.send_text(response.model_dump_json(by_alias=False))

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected normally: session_id=%s", session_id)
    except Exception as e:
        logger.warning("WebSocket exception for session_id=%s: %s", session_id, e)
    finally:
        session_manager.remove(session_id)
