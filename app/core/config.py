import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Calisthenics AI Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Model Configuration
    MODEL_PATH: str = os.getenv("MODEL_PATH", "yolov8n-pose.pt")
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    KEYPOINT_CONF_THRESHOLD: float = float(os.getenv("KEYPOINT_CONF_THRESHOLD", "0.25"))
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    # Biomechanics & Hold Detection Configuration
    SMOOTHING_ALPHA: float = float(os.getenv("SMOOTHING_ALPHA", "0.65"))
    MIN_HOLD_DURATION_SEC: float = float(os.getenv("MIN_HOLD_DURATION_SEC", "0.5"))
    HOLD_STABILITY_VARIANCE_THRESHOLD: float = float(os.getenv("HOLD_STABILITY_VARIANCE_THRESHOLD", "15.0"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Storage & Outputs
    TEMP_DIR: str = os.getenv("TEMP_DIR", "/tmp/calisthenics_ai")
    MAX_UPLOAD_SIZE_MB: int = 50
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )


settings = Settings()
