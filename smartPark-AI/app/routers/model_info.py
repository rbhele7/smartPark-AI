from fastapi import APIRouter
from app.config import API_PREFIX, CONFIDENCE_THRESHOLD
from app.schemas import ModelInfoResponse
from app.model import model_manager

router = APIRouter(prefix=API_PREFIX, tags=["Model Info"])


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    return ModelInfoResponse(
        model_name=model_manager.model_name,
        class_names=model_manager.class_names,
        input_shape=[224, 224, 3],
        status="loaded" if model_manager.is_loaded else "unloaded",
        path=model_manager.loaded_path,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
