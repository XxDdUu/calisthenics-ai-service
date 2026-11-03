FROM python:3.13-slim

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Optimize uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./

# Install runtime dependencies
RUN uv sync --frozen --no-dev

# Copy application source code
COPY . /app/

# Create models directory
RUN mkdir -p /app/models

EXPOSE 8000

ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=yolov8n-pose.pt

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]