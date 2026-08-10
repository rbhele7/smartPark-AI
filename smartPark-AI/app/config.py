import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Model configuration
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "parking_cnn.keras"))
FALLBACK_MODEL_PATH = str(BASE_DIR / "models" / "parking_cnn_final.keras")
CLASS_NAMES_PATH = os.getenv("CLASS_NAMES_PATH", str(BASE_DIR / "models" / "class_names.json"))

IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# API configuration
API_TITLE = "SmartPark AI Backend API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "High-performance real-time parking spot occupancy detection & analytics API"
API_PREFIX = "/api/v1"

# Allowed file formats
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/pjpeg"}
