"""
Benchmark script comparing YOLOv8n-pose vs YOLOv8m-pose on inference latency,
confidence distributions, and side-view keypoint accuracy.
"""
import os
import time
import cv2
import numpy as np
from ultralytics import YOLO

from app.pose.yolo_pose import YOLOPoseEstimator
from app.biomechanics.side_view import SideViewKeypointProcessor, resolve_side_view_chain

def run_benchmark():
    test_img_path = "scratch/output_annotated.jpg"
    if not os.path.exists(test_img_path):
        # Fall back to asset image
        test_img_path = ".venv/lib/python3.13/site-packages/ultralytics/assets/bus.jpg"

    img = cv2.imread(test_img_path)
    h, w = img.shape[:2]
    print(f"=== BENCHMARKING POSE MODELS ON {test_img_path} ({w}x{h}) ===")

    models_to_test = [
        ("YOLOv8n-pose (Nano)", "models/yolov8n-pose.pt"),
        ("YOLOv8m-pose (Medium)", "models/yolov8m-pose.pt"),
    ]

    results = {}

    for name, path in models_to_test:
        if not os.path.exists(path):
            print(f"Model file not found: {path}")
            continue

        file_size_mb = os.path.getsize(path) / (1024 * 1024)
        estimator = YOLOPoseEstimator(model_path=path, device="cpu")
        model = estimator.model
        param_count = sum(p.numel() for p in model.parameters())

        # Warmup
        for _ in range(5):
            estimator.estimate_pose(img)

        # Latency benchmark (30 runs)
        latencies = []
        for _ in range(30):
            t0 = time.perf_counter()
            success, pose_kp = estimator.estimate_pose(img)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_lat = np.mean(latencies)
        median_lat = np.median(latencies)
        p95_lat = np.percentile(latencies, 95)

        # Evaluate keypoints
        success, pose_kp = estimator.estimate_pose(img)
        kp_dict = pose_kp.keypoints if (success and pose_kp) else {}

        confidences = {k: v.confidence for k, v in kp_dict.items()}
        avg_conf = np.mean(list(confidences.values())) if confidences else 0.0

        leg_joints = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
        leg_confs = {k: confidences.get(k, 0.0) for k in leg_joints}

        chain = resolve_side_view_chain(kp_dict, threshold=0.25) if kp_dict else {}
        proc = SideViewKeypointProcessor.process_keypoints(kp_dict, threshold=0.25) if kp_dict else None

        results[name] = {
            "file_size_mb": file_size_mb,
            "params": param_count,
            "mean_latency_ms": mean_lat,
            "median_latency_ms": median_lat,
            "p95_latency_ms": p95_lat,
            "avg_confidence": avg_conf,
            "leg_confs": leg_confs,
            "resolved_joints": list(chain.keys()),
            "missing_midpoints": proc.missing_landmarks if proc else [],
        }

        print(f"\n--- {name} ---")
        print(f"File Size: {file_size_mb:.2f} MB | Parameters: {param_count:,}")
        print(f"Latency (CPU): Mean={mean_lat:.2f}ms, Median={median_lat:.2f}ms, P95={p95_lat:.2f}ms")
        print(f"Average Keypoint Confidence: {avg_conf:.3f}")
        print(f"Leg Keypoint Confidences: {leg_confs}")
        print(f"Resolved Side-View Chain Joints: {list(chain.keys())}")
        if proc:
            print(f"Derived Midpoints: {list(proc.midpoints.keys())}")
            print(f"Missing Midpoints: {proc.missing_landmarks}")

    return results

if __name__ == "__main__":
    run_benchmark()
