# Calisthenics AI Service

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![YOLO-Pose](https://img.shields.io/badge/Ultralytics-YOLO--Pose-orange.svg)](https://docs.ultralytics.com/)

A production-ready FastAPI microservice for computer vision pose estimation, biomechanical angle tracking, and exercise form analysis for calisthenics skills (with dedicated support for **Planche** progressions: Tuck, Advanced Tuck, Straddle, and Full Planche).

---

## 📁 Repository Architecture

```
calisthenics-ai-service/
├── app/
│   ├── main.py                   # FastAPI application initialization & routes registration
│   ├── api/
│   │   └── routes/
│   │       └── analysis.py       # REST API endpoints for image/video pose analysis
│   ├── core/
│   │   └── config.py             # Pydantic environment configuration & settings
│   ├── schemas/
│   │   └── analysis.py           # Pydantic request/response models & schemas
│   ├── pose/
│   │   └── yolo_pose.py          # Ultralytics YOLO-Pose keypoint detector & skeleton drawing
│   ├── biomechanics/
│   │   ├── angles.py             # 2D/3D joint angle vector geometry calculations
│   │   ├── alignment.py          # Body line, center of mass, & scapular alignment metrics
│   │   └── metrics.py            # Biomechanics profile evaluator orchestrator
│   ├── exercises/
│   │   └── planche.py            # Planche biomechanics, form scoring & coaching cues
│   └── services/
│       └── analyzer.py           # High-level image and video processing pipeline
├── tests/                        # Comprehensive PyTest unit & integration tests
│   ├── test_angles.py
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

### 2. Single Image Pose & Form Analysis
`POST /api/v1/analysis/image`

- **Content-Type**: `multipart/form-data`
- **Query Params**: `draw_overlay=true`
- **Form Data**: `file` (Image file)

### 3. Video Hold Duration & Timeline Analysis
`POST /api/v1/analysis/video`

- **Content-Type**: `multipart/form-data`
- **Query Params**: `sample_fps=5`
- **Form Data**: `file` (Video file MP4/MOV)

---

## 📄 License

Apache 2.0 License.
