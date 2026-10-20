import urllib.request
import json
import requests

def test_api_with_real_image():
    # Download standard ultralytics sample image containing real people
    image_url = "https://ultralytics.com/images/bus.jpg"
    print(f"Downloading sample image with real humans from {image_url}...")
    req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        image_bytes = response.read()

    files = {"file": ("bus.jpg", image_bytes, "image/jpeg")}
    url = "http://localhost:8000/api/v1/analysis/image?draw_overlay=true"
    
    print(f"Sending test image request to {url}...")
    resp = requests.post(url, files=files)
    print("Status Code:", resp.status_code)
    data = resp.json()
    
    print("\n--- RESULTS ---")
    print("Success:", data.get("success"))
    print("Message:", data.get("message"))
    print("Pose Detected:", data.get("pose_detected"))
    print("Pose Keypoints Present:", data.get("pose_keypoints") is not None)
    if data.get("pose_keypoints"):
        kp = data["pose_keypoints"]
        print("Detection Confidence:", kp.get("detection_confidence"))
        print("Bounding Box:", kp.get("bounding_box"))
        print("Number of keypoints:", len(kp.get("keypoints", {})))
        print("Sample keypoint (nose):", kp.get("keypoints", {}).get("nose"))
    print("Annotated Image B64 Present:", data.get("annotated_image_base64") is not None)
    return data

if __name__ == "__main__":
    test_api_with_real_image()
