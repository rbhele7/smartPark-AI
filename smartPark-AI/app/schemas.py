from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class RootResponse(BaseModel):
    message: str
    version: str
    docs_url: str
    health_url: str


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    model_name: str = "SmartParkingCNN"
    device: str = "CPU"
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    model_name: str = "SmartParkingCNN"
    class_names: List[str]
    input_shape: List[int]
    status: str = "loaded"
    path: str
    confidence_threshold: float = 0.5


class PredictionResult(BaseModel):
    class_name: str
    confidence: float
    raw_probability: float
    filename: Optional[str] = None


class SinglePredictResponse(BaseModel):
    status: str = "success"
    prediction: PredictionResult
    inference_time_ms: float
    filename: Optional[str] = None


class BatchPredictResponse(BaseModel):
    status: str = "success"
    total_images: int
    predictions: List[PredictionResult]
    inference_time_ms: float


class SpotROI(BaseModel):
    spot_id: str
    x: int
    y: int
    width: int
    height: int


class ParkingLotROIRequest(BaseModel):
    spots: List[SpotROI]


class SpotPredictionResult(BaseModel):
    spot_id: str
    class_name: str
    confidence: float
    bbox: List[int]


class ParkingLotResponse(BaseModel):
    status: str = "success"
    total_spots: int
    occupied_count: int
    vacant_count: int
    occupancy_rate: float
    predictions: List[SpotPredictionResult]
    annotated_image_base64: Optional[str] = None
    inference_time_ms: float


class AnalyticsResponse(BaseModel):
    status: str = "success"
    total_requests: int
    total_images_processed: int
    avg_inference_time_ms: float
    recent_occupancy_breakdown: Dict[str, int]


class ParkingTemplate(BaseModel):
    id: str
    name: str
    description: str
    spots: List[SpotROI]


class ExportReportResponse(BaseModel):
    timestamp: str
    total_spots: int
    occupied_count: int
    vacant_count: int
    occupancy_rate: float
    spots: List[Dict[str, Any]]

