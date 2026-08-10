import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def create_dummy_image_bytes(color: str = "red", size: tuple = (224, 224), format: str = "JPEG") -> bytes:
    """Helper to generate dummy image byte data for testing."""
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["docs_url"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info_endpoint():
    response = client.get("/api/v1/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "SmartParkingCNN"
    assert "occupied" in data["class_names"]
    assert "vacant" in data["class_names"]


def test_predict_single_image():
    img_bytes = create_dummy_image_bytes(color="blue")
    files = {"file": ("test_spot.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/v1/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "prediction" in data
    assert data["prediction"]["class_name"] in ["occupied", "vacant"]
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0
    assert data["inference_time_ms"] > 0


def test_predict_batch_images():
    img1 = create_dummy_image_bytes(color="red")
    img2 = create_dummy_image_bytes(color="green")

    files = [
        ("files", ("spot1.jpg", img1, "image/jpeg")),
        ("files", ("spot2.jpg", img2, "image/jpeg"))
    ]
    response = client.post("/api/v1/predict/batch", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_images"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["class_name"] in ["occupied", "vacant"]
    assert data["predictions"][1]["class_name"] in ["occupied", "vacant"]


def test_predict_invalid_file_type():
    files = {"file": ("test.txt", b"this is not an image", "text/plain")}
    response = client.post("/api/v1/predict", files=files)
    assert response.status_code in [400, 415]
