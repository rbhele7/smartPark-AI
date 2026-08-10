import io
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def create_dummy_parking_lot_bytes(size: tuple = (640, 480)) -> bytes:
    """Helper to generate dummy parking lot image bytes."""
    img = Image.new("RGB", size, color="gray")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_parking_lot_roi_prediction():
    img_bytes = create_dummy_parking_lot_bytes()
    files = {"file": ("parking_lot.jpg", img_bytes, "image/jpeg")}
    response = client.post("/api/v1/predict/parking-lot", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_spots"] > 0
    assert "occupied_count" in data
    assert "vacant_count" in data
    assert "occupancy_rate" in data
    assert len(data["predictions"]) == data["total_spots"]
    assert data["annotated_image_base64"] is not None
    assert "X-Process-Time-Ms" in response.headers


def test_analytics_endpoint():
    response = client.get("/api/v1/analytics")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_requests"] >= 0
    assert "avg_inference_time_ms" in data
