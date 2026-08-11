import io
import base64
from typing import Tuple, List, Optional
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from fastapi import HTTPException, UploadFile, status
from app.config import ALLOWED_EXTENSIONS, ALLOWED_MIMETYPES, IMG_SIZE


def validate_image_file(file: UploadFile, content: Optional[bytes] = None) -> bytes:
    """Validate that uploaded file is a valid image and return image bytes."""
    filename = file.filename or ""
    ext = ("." + filename.split(".")[-1].lower()) if "." in filename else ""
    
    # Check mimetype or extension
    if file.content_type and file.content_type.lower() not in ALLOWED_MIMETYPES and ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Allowed types: JPEG, PNG, WEBP, BMP."
        )

    if content is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty image file content."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    try:
        image = Image.open(io.BytesIO(content))
        image.verify()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content could not be parsed as a valid image."
        )

    return content


def preprocess_image_bytes(content: bytes, target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Read image bytes, handle EXIF rotation, convert to RGB, resize, and return numpy array shape (1, H, W, 3)."""
    raw_img = Image.open(io.BytesIO(content))
    image = ImageOps.exif_transpose(raw_img).convert("RGB")
    image = image.resize(target_size, Image.Resampling.BILINEAR)
    img_array = np.array(image, dtype=np.float32)
    # Add batch dimension -> (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)


def preprocess_batch_image_bytes(content_list: List[bytes], target_size: Tuple[int, int] = IMG_SIZE) -> np.ndarray:
    """Convert list of image bytes into a single batch numpy array shape (N, H, W, 3) with EXIF transposition."""
    batch_list = []
    for content in content_list:
        raw_img = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(raw_img).convert("RGB")
        image = image.resize(target_size, Image.Resampling.BILINEAR)
        img_array = np.array(image, dtype=np.float32)
        batch_list.append(img_array)
    return np.array(batch_list, dtype=np.float32)



def draw_parking_annotations(
    image_bytes: bytes,
    predictions: List[dict]
) -> str:
    """Draw bounding boxes and class labels on image, returning base64 JPEG string."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)

    for item in predictions:
        bbox = item.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        label = item.get("class_name", "unknown")
        confidence = item.get("confidence", 0.0)
        spot_id = item.get("spot_id", "")

        color = (34, 197, 94) if label == "vacant" else (239, 68, 68)  # Green vs Red

        # Draw box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=3)

        # Draw label background & text
        text = f"{spot_id}: {label.upper()} ({confidence:.0%})"
        text_bbox = draw.textbbox((x, max(0, y - 18)), text)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x, max(0, y - 18)), text, fill=(255, 255, 255))

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
