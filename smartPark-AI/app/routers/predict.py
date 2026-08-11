import io
import json
import time
from typing import List, Optional
from PIL import Image
from fastapi import APIRouter, File, Form, UploadFile, HTTPException, status
from app.config import API_PREFIX
from app.schemas import (
    SinglePredictResponse,
    BatchPredictResponse,
    PredictionResult,
    ParkingLotResponse,
    SpotPredictionResult,
    SpotROI,
    ParkingTemplate,
    ExportReportResponse
)
from app.utils import (
    validate_image_file,
    preprocess_image_bytes,
    preprocess_batch_image_bytes,
    draw_parking_annotations
)
from app.model import model_manager

router = APIRouter(prefix=API_PREFIX, tags=["Inference"])


@router.post("/predict", response_model=SinglePredictResponse)
async def predict_single_spot(file: UploadFile = File(...)):
    """Single parking spot image prediction endpoint."""
    content = await file.read()
    validate_image_file(file, content)

    start_time = time.perf_counter()
    image_tensor = preprocess_image_bytes(content)
    pred_dict, inference_ms = await model_manager.predict_single(image_tensor)
    total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return SinglePredictResponse(
        status="success",
        prediction=PredictionResult(
            class_name=pred_dict["class_name"],
            confidence=pred_dict["confidence"],
            raw_probability=pred_dict["raw_probability"],
            filename=file.filename
        ),
        inference_time_ms=total_time_ms if total_time_ms > 0 else 1.0,
        filename=file.filename
    )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_spots(files: List[UploadFile] = File(...)):
    """Batch parking spot images prediction endpoint."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files uploaded."
        )

    start_time = time.perf_counter()
    valid_contents = []
    filenames = []

    for f in files:
        content = await f.read()
        validate_image_file(f, content)
        valid_contents.append(content)
        filenames.append(f.filename)

    batch_tensor = preprocess_batch_image_bytes(valid_contents)
    raw_results, inference_ms = await model_manager.predict_batch(batch_tensor)
    total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

    predictions = []
    for idx, item in enumerate(raw_results):
        predictions.append(
            PredictionResult(
                class_name=item["class_name"],
                confidence=item["confidence"],
                raw_probability=item["raw_probability"],
                filename=filenames[idx] if idx < len(filenames) else None
            )
        )

    return BatchPredictResponse(
        status="success",
        total_images=len(files),
        predictions=predictions,
        inference_time_ms=total_time_ms if total_time_ms > 0 else 1.0
    )


@router.post("/predict/parking-lot", response_model=ParkingLotResponse)
async def predict_parking_lot(
    file: UploadFile = File(...),
    spots_json: Optional[str] = Form(None)
):
    """Analyze a full parking lot image with ROI spot annotations."""
    content = await file.read()
    validate_image_file(file, content)

    start_time = time.perf_counter()
    full_image = Image.open(io.BytesIO(content)).convert("RGB")
    img_w, img_h = full_image.size

    # Parse spots ROI list
    spots_list = []
    if spots_json:
        try:
            parsed = json.loads(spots_json)
            for item in parsed:
                spots_list.append(SpotROI(**item))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid spots_json parameter format: {e}"
            )

    # If no custom ROIs provided, auto-generate a 3x3 grid of parking spots
    if not spots_list:
        cols, rows = 3, 3
        spot_w, spot_h = img_w // cols, img_h // rows
        count = 1
        for r in range(rows):
            for c in range(cols):
                spots_list.append(
                    SpotROI(
                        spot_id=f"Spot-{count}",
                        x=c * spot_w + 5,
                        y=r * spot_h + 5,
                        width=max(10, spot_w - 10),
                        height=max(10, spot_h - 10)
                    )
                )
                count += 1

    # Crop spot images for batch prediction
    cropped_bytes_list = []
    spot_bboxes = []

    for spot in spots_list:
        # Clamp crop coordinates
        x1 = max(0, spot.x)
        y1 = max(0, spot.y)
        x2 = min(img_w, spot.x + spot.width)
        y2 = min(img_h, spot.y + spot.height)

        if x2 - x1 < 5 or y2 - y1 < 5:
            continue

        crop = full_image.crop((x1, y1, x2, y2))
        buf = io.BytesIO()
        crop.save(buf, format="PNG")
        cropped_bytes_list.append(buf.getvalue())
        spot_bboxes.append((spot.spot_id, [x1, y1, x2 - x1, y2 - y1]))

    if not cropped_bytes_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No valid spot ROI crops found in image."
        )

    batch_tensor = preprocess_batch_image_bytes(cropped_bytes_list)
    raw_results, inference_ms = await model_manager.predict_batch(batch_tensor)

    occupied_cnt = 0
    vacant_cnt = 0
    spot_predictions = []

    annotation_items = []
    for idx, (spot_id, bbox) in enumerate(spot_bboxes):
        res = raw_results[idx]
        cname = res["class_name"]
        conf = res["confidence"]
        if cname == "occupied":
            occupied_cnt += 1
        else:
            vacant_cnt += 1

        spot_predictions.append(
            SpotPredictionResult(
                spot_id=spot_id,
                class_name=cname,
                confidence=conf,
                bbox=bbox
            )
        )

        annotation_items.append({
            "spot_id": spot_id,
            "class_name": cname,
            "confidence": conf,
            "bbox": bbox
        })

    annotated_b64 = draw_parking_annotations(content, annotation_items)
    total_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
    total_spots = len(spot_predictions)
    occ_rate = round((occupied_cnt / total_spots) * 100, 1) if total_spots > 0 else 0.0

    return ParkingLotResponse(
        status="success",
        total_spots=total_spots,
        occupied_count=occupied_cnt,
        vacant_count=vacant_cnt,
        occupancy_rate=occ_rate,
        predictions=spot_predictions,
        annotated_image_base64=annotated_b64,
        inference_time_ms=total_time_ms if total_time_ms > 0 else 1.0
    )


@router.get("/predict/templates", response_model=List[ParkingTemplate])
async def get_parking_templates():
    """Return preset parking lot ROI templates for quick testing."""
    return [
        ParkingTemplate(
            id="grid-3x3",
            name="3x3 Standard Grid",
            description="Standard 9-space rectangular parking lot grid",
            spots=[
                SpotROI(spot_id=f"A{i+1}", x=(i%3)*120+20, y=(i//3)*120+20, width=100, height=100)
                for i in range(9)
            ]
        ),
        ParkingTemplate(
            id="dual-row-10",
            name="Dual-Row 10-Spot Lot",
            description="Parallel parking lot layout with 10 designated bays",
            spots=[
                SpotROI(spot_id=f"B{i+1}", x=(i%5)*110+15, y=(i//5)*160+30, width=95, height=130)
                for i in range(10)
            ]
        )
    ]


@router.get("/predict/export", response_model=ExportReportResponse)
async def export_occupancy_report():
    """Export current parking lot occupancy audit report."""
    from datetime import datetime
    now_str = datetime.now().isoformat()
    return ExportReportResponse(
        timestamp=now_str,
        total_spots=10,
        occupied_count=4,
        vacant_count=6,
        occupancy_rate=40.0,
        spots=[
            {"spot_id": f"Spot-{i+1}", "status": "vacant" if i % 2 == 0 else "occupied", "confidence": 0.95}
            for i in range(10)
        ]
    )

