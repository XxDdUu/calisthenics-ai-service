import urllib.request
import requests

def test_export_draw_skeleton():
    image_url = "https://ultralytics.com/images/bus.jpg"
    print(f"Downloading sample image from {image_url}...")
    req = urllib.request.Request(image_url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        image_bytes = response.read()

    files = {"file": ("bus.jpg", image_bytes, "image/jpeg")}
    url = "http://localhost:8000/api/v1/analysis/image/draw-skeleton"

    print(f"Sending request to {url}...")
    resp = requests.post(url, files=files)
    print("Status Code:", resp.status_code)
    print("Content-Type:", resp.headers.get("Content-Type"))
    print("Content-Length (bytes):", len(resp.content))

    if resp.status_code == 200 and resp.headers.get("Content-Type") == "image/jpeg":
        with open("/app/scratch/output_annotated.jpg", "wb") as f:
            f.write(resp.content)
        print("Successfully exported annotated skeleton image to /app/scratch/output_annotated.jpg!")

if __name__ == "__main__":
    test_export_draw_skeleton()
