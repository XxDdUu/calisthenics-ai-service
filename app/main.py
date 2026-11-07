import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.analysis import router as analysis_router
from app.api.routes.live_analysis import router as live_analysis_router
from app.core.config import settings

log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle context manager."""
    logger.info(f"Starting {settings.PROJECT_NAME} v{settings.VERSION} on device '{settings.DEVICE}'...")
    try:
        from app.services.live_pipeline import live_pipeline
        # Warm-up model on startup so first live frame does not incur initialization delay
        if live_pipeline.pose_estimator.model is not None:
            logger.info("Pre-warming YOLO-Pose inference engine...")
            import numpy as np
            dummy_frame = np.zeros((256, 256, 3), dtype=np.uint8)
            live_pipeline.pose_estimator.estimate_pose(dummy_frame)
            logger.info("YOLO-Pose model warm-up completed successfully.")
    except Exception as e:
        logger.error(f"Failed during model warm-up: {e}", exc_info=True)

    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI service for Calisthenics Pose Estimation, Biomechanics Evaluation, and Form Scoring powered by Ultralytics YOLO-Pose and OpenCV.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Set up CORS
if settings.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Register API Routers
app.include_router(analysis_router, prefix=settings.API_V1_STR)
app.include_router(live_analysis_router, prefix=settings.API_V1_STR)


@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "device": settings.DEVICE,
        "model": settings.MODEL_PATH
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": f"Internal server error: {str(exc)}"}
    )
