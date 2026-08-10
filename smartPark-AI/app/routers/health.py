import time
from fastapi import APIRouter
from app.schemas import RootResponse, HealthResponse
from app.model import model_manager

router = APIRouter(tags=["Health & System"])
SERVER_START_TIME = time.time()


@router.get("/", response_model=RootResponse)
async def root():
    return RootResponse(
        message="Welcome to SmartPark AI API - Smart Parking Occupancy Detection Backend",
        version="1.0.0",
        docs_url="/docs",
        health_url="/health"
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    uptime = time.time() - SERVER_START_TIME
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.is_loaded,
        model_name=model_manager.model_name,
        device="CPU",
        uptime_seconds=round(uptime, 2)
    )
