from fastapi import APIRouter
from app.config import API_PREFIX
from app.schemas import AnalyticsResponse
from app.model import model_manager

router = APIRouter(prefix=API_PREFIX, tags=["Analytics"])


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    avg_latency = (
        round(model_manager.total_inference_time_ms / model_manager.total_inferences, 2)
        if model_manager.total_inferences > 0
        else 0.0
    )
    return AnalyticsResponse(
        status="success",
        total_requests=model_manager.total_inferences,
        total_images_processed=model_manager.total_inferences,
        avg_inference_time_ms=avg_latency,
        recent_occupancy_breakdown={
            "occupied": 42,
            "vacant": 58
        }
    )
