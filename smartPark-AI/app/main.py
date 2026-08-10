import time
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

from app.config import API_TITLE, API_VERSION, API_DESCRIPTION
from app.model import model_manager
from app.routers import health, model_info, predict, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to pre-load model on startup and clean up on shutdown."""
    print("==================================================")
    print("Starting SmartPark AI FastAPI Server...")
    print("==================================================")
    # Pre-load model and run warm-up prediction
    model_manager.load()
    yield
    print("Shutting down SmartPark AI FastAPI Server.")


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware for interactive web dashboard and frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add latency header to every API response."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# Include Routers
app.include_router(health.router)
app.include_router(model_info.router)
app.include_router(predict.router)
app.include_router(analytics.router)

# Static files for interactive dashboard
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/dashboard", response_class=HTMLResponse, tags=["Dashboard"])
async def dashboard():
    """Serve real-time interactive AI parking dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h1>SmartPark AI Dashboard coming soon!</h1>")
