# Calisthenics AI Service

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLO-Pose](https://img.shields.io/badge/Ultralytics-YOLO--Pose-orange.svg)](https://docs.ultralytics.com/)

A production-ready FastAPI microservice for computer vision pose estimation, biomechanical angle tracking, and exercise form analysis for calisthenics skills (with real-time Live Analysis and dedicated analyzers for **Planche**, **Front Lever**, **Handstand / HSPU**, **Pull-Up**, **Parallel Bar Dips**, and **Muscle-Up**).

---

## 📁 Repository Architecture

```
calisthenics-ai-service/
├── app/
│   ├── main.py                   # FastAPI application initialization & routes registration
│   ├── api/
│   │   └── routes/
│   │       ├── analysis.py       # REST API endpoints for image/video pose analysis
│   │       └── live_analysis.py  # Real-time Live Analysis REST & WebSocket endpoints
│   ├── core/
│   │   ├── config.py             # Pydantic environment configuration & settings
│   │   └── logging_context.py    # Request ID tracking context manager
│   ├── schemas/
│   │   └── analysis.py           # Pydantic request/response models & live schemas
│   ├── pose/
│   │   └── yolo_pose.py          # Ultralytics YOLO-Pose keypoint detector & skeleton drawing
│   ├── biomechanics/
│   │   ├── angles.py             # 2D joint angle vector geometry calculations
│   │   ├── alignment.py          # Body line, center of mass, & scapular alignment metrics
│   │   ├── confidence.py         # Keypoint confidence & pose visibility validation
│   │   ├── smoothing.py          # Temporal EMA filter suppressing live detector jitter
│   │   ├── hold_detection.py     # Isometric hold tracking state machine
│   │   ├── side_view.py          # Profile view occlusion handling and midpoint derivation
│   │   └── metrics.py            # Biomechanics profile evaluator orchestrator
│   ├── exercises/
│   │   ├── base.py               # Abstract BaseExerciseAnalyzer
│   │   ├── planche.py            # Planche biomechanics, form scoring & coaching cues
│   │   ├── front_lever.py        # Front Lever biomechanics & progression evaluation
│   │   ├── handstand.py          # Handstand & HSPU alignment and repetition counting
│   │   ├── pull_up.py            # Pull-up ROM & bilateral arm symmetry tracking
│   │   ├── dips.py               # Parallel Bar Dips depth & lockout rep analysis
│   │   ├── muscle_up.py          # Muscle-up transition & chicken-wing detection
│   │   └── registry.py           # Central exercise analyzer registry
│   └── services/
│       ├── analyzer.py           # High-level image and video processing pipeline
│       ├── live_pipeline.py      # Production Live Analysis frame processing pipeline
│       └── live_session.py       # Thread-safe session manager & temporal smoothing state
├── tests/                        # Comprehensive PyTest unit, API & integration tests
│   ├── test_angles.py
│   ├── test_exercise_analyzers.py
│   ├── test_live_session.py
│   ├── test_live_api.py
│   ├── test_alignment.py
│   ├── test_planche.py
│   └── test_api.py
├── models/                       # Directory for YOLO-Pose model weights
├── requirements.txt              # Dependency specifications
├── Dockerfile                    # Containerization configuration
└── README.md
```

---

## ⚡ Tech Stack

- **Python 3.11+**
- **FastAPI** (REST API, OpenAPI/Swagger docs)
- **Ultralytics YOLO-Pose** (YOLOv8 pose estimation)
- **OpenCV** (Image & video processing, skeleton visualization)
- **NumPy** (Biomechanical vector mathematics & matrix operations)
- **Pydantic v2** (Type validation & schema serialization)
- **PyTorch** (Deep learning inference backend)

---

## 🤸 Biomechanical Analysis Methodology

### Planche Key Indicators

1. **Elbow Lockout Angle ($\ge 170^\circ$)**:
   - Measures joint extension angle $(Shoulder \rightarrow Elbow \rightarrow Wrist)$.
   - Bent arms ($< 160^\circ$) trigger form penalties and elbow strain warnings.

2. **Shoulder Forward Lean ($\ge 15^\circ$)**:
   - Calculates the angular deviation of the shoulder relative to the wrist base of support.
   - Forward lean is required to balance the athlete's Center of Mass (CoM).

3. **Body Line Extension ($175^\circ - 180^\circ$)**:
   - Measures joint angle $(Shoulder \rightarrow Hip \rightarrow Knee/Ankle)$.
   - Detects **Piking** (hips raised, $< 155^\circ$) and **Sagging/Banana form** ($> 190^\circ$).

4. **Progression Stage Classification**:
   - **Tuck Planche**: Knees tucked to chest ($Hip < 115^\circ, Knee < 105^\circ$).
   - **Advanced Tuck Planche**: Flat back, hips un-tucked ($Hip \approx 90^\circ - 130^\circ$).
   - **Straddle Planche**: Legs extended, wide straddle.
   - **Full Planche**: Straight body line ($Hip \ge 160^\circ, Knee \ge 160^\circ$), torso ground-parallel.

---

## 🚀 Getting Started

### 1. Local Setup

```bash
# Clone repository
git clone https://github.com/XxDdUu/calisthenics-ai-service.git
cd calisthenics-ai-service

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Local Dev Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Access Swagger UI interactive docs at: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

```bash
# Build docker image
docker build -t calisthenics-ai-service:latest .

# Run container
docker run -p 8000:8000 calisthenics-ai-service:latest
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📡 API Reference

### 1. Health Check
`GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "Calisthenics AI Service",
  "version": "1.0.0",
  "device": "cpu",
  "model": "yolov8n-pose.pt"
}
```

### 2. Live Camera Analysis (Frame-by-Frame REST)
`POST /api/v1/analysis/live/frame`

- **Content-Type**: `multipart/form-data`
- **Form Data**:
  - `file`: Camera frame JPEG/PNG
  - `exercise`: e.g. `FULL_PLANCHE`, `FRONT_LEVER`, `HANDSTAND`, `PULL_UP`, `DIPS`, `MUSCLE_UP`
  - `session_id`: Optional session ID for temporal smoothing & hold tracking
  - `draw_overlay`: `false` (or `true` for base64 annotated image)
  - `debug`: `false`

Also available via JSON Base64: `POST /api/v1/analysis/live/frame-json` with body:
```json
{
  "imageBase64": "...",
  "exercise": "FULL_PLANCHE",
  "sessionId": "optional-session-id",
  "frameSequence": 1,
  "timestampMs": 1700000000000
}
```

### 3. Live WebSocket Streaming
`WebSocket /api/v1/analysis/live/ws`

Bidirectional streaming for continuous real-time camera processing.
- Connect: `ws://<HOST>:8000/api/v1/analysis/live/ws`
- Send frame message:
```json
{
  "action": "frame",
  "image_base64": "...",
  "exercise": "FRONT_LEVER",
  "frame_sequence": 1
}
```
- Immediate Response:
```json
{
  "status": "VALID",
  "exercise": "FRONT_LEVER",
  "timestamp": 1700000000000,
  "frameSequence": 1,
  "poseDetected": true,
  "confidence": 0.94,
  "score": 88.5,
  "isValidHold": true,
  "holdDurationSec": 2.4,
  "measurements": {
    "elbow_angle": 174.0,
    "shoulder_angle": 76.0,
    "body_line_angle": 175.0,
    "knee_angle": 176.0,
    "horizontal_tilt": 8.0
  },
  "feedback": [
    {
      "code": "GOOD_FORM",
      "message": "Solid Front Lever hold! Maintain body tension.",
      "severity": "INFO"
    }
  ],
  "timing": {
    "preprocessing_ms": 1.1,
    "inference_ms": 28.4,
    "analysis_ms": 0.5,
    "total_ms": 30.0
  }
}
```

### 4. Single Image Pose & Form Analysis
`POST /api/v1/analysis/image`

- **Content-Type**: `multipart/form-data`
- **Query Params**: `draw_overlay=true`
- **Form Data**: `file` (Image file)

### 5. Video Hold Duration & Timeline Analysis
`POST /api/v1/analysis/video`

- **Content-Type**: `multipart/form-data`
- **Query Params**: `sample_fps=5`
- **Form Data**: `file` (Video file MP4/MOV)

---

## 📄 License

Apache 2.0 License.
