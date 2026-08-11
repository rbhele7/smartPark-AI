# Production Dockerfile for SmartPark AI FastAPI Backend & Dashboard
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080 \
    HOST=0.0.0.0 \
    KERAS_HOME=/tmp/.keras \
    MPLCONFIGDIR=/tmp/.matplotlib

# Install system dependencies required by OpenCV and numerical libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first to leverage layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and pre-trained models
COPY app/ ./app/
COPY models/ ./models/
COPY run.py .

# Expose backend port
EXPOSE 8080

# Health check specification for container orchestration and Cloud Run
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Launch FastAPI production server
CMD ["python", "run.py"]
