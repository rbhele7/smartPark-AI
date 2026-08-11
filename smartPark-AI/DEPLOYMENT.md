# 🚀 SmartPark AI - Deployment Guide

This guide provides step-by-step instructions for deploying **SmartPark AI** across local, containerized, and serverless cloud production environments.

---

## 🏗️ Architecture Overview

SmartPark AI consists of three complementary runtime components:

1. **FastAPI Backend REST API**: High-performance asynchronous inference engine serving `/api/v1/predict`, `/api/v1/predict/batch`, `/api/v1/predict/parking-lot`, `/health`, and `/api/v1/analytics`.
2. **Built-in HTML5/JS Web Dashboard**: Served directly from the backend at `/dashboard`.
3. **Streamlit Interactive UI**: Multi-tab administrative analytics app hosted in `app/streamlit_app.py`.

---

## ⚙️ Environment Variables

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `PORT` | `8080` | Port for the FastAPI HTTP server |
| `HOST` | `0.0.0.0` | Network binding host address |
| `MODEL_PATH` | `models/parking_cnn.keras` | Absolute or relative path to pre-trained Keras model |
| `CONFIDENCE_THRESHOLD` | `0.5` | Threshold score for vacant vs occupied classification |
| `API_BASE` | `http://127.0.0.1:8080` | Base URL used by Streamlit app to communicate with FastAPI backend |
| `KERAS_HOME` | `/tmp/.keras` | Writable directory for Keras cache (vital for serverless containers) |
| `MPLCONFIGDIR` | `/tmp/.matplotlib` | Writable directory for Matplotlib cache |

---

## 🐳 Option 1: Local Docker Deployment

### 1. Build Docker Image
```bash
docker build -t smartpark-ai:latest .
```

### 2. Run Container
```bash
docker run -d -p 8080:8080 --name smartpark-service smartpark-ai:latest
```

### 3. Verify Deployment
- API Documentation: [http://localhost:8080/docs](http://localhost:8080/docs)
- Interactive Dashboard: [http://localhost:8080/dashboard](http://localhost:8080/dashboard)
- Health Check: `curl http://localhost:8080/health`

---

## 🐙 Option 2: Docker Compose (Backend + Streamlit Dashboard)

To spin up both the FastAPI backend and the Streamlit UI dashboard together:

```bash
docker-compose up --build -d
```

- **FastAPI Backend & Web Dashboard**: [http://localhost:8080](http://localhost:8080)
- **Streamlit Analytics App**: [http://localhost:8501](http://localhost:8501)

To stop services:
```bash
docker-compose down
```

---

## ☁️ Option 3: Google Cloud Run (Recommended for Serverless)

Google Cloud Run automatically scales containerized applications to zero when idle.

### Step 1: Set Google Cloud Project & Region
```bash
gcloud config set project YOUR_PROJECT_ID
gcloud config set run/region us-central1
```

### Step 2: Deploy directly from Source Code
```bash
gcloud run deploy smartpark-ai \
  --source . \
  --platform managed \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --cpu 2
```

Once deployed, Cloud Run will output your live service URL (e.g. `https://smartpark-ai-xyz-uc.a.run.app`).

---

## 💜 Option 4: Render / Railway Deployment

### Render (1-Click Deployment via Infrastructure as Code)
This repository includes a `render.yaml` blueprint:
1. Connect your GitHub repository to [Render](https://render.com).
2. Click **New +** -> **Blueprint**.
3. Select your `smartPark-AI` repository.
4. Render will automatically launch both `smartpark-api` and `smartpark-dashboard`.

### Railway / Heroku
Using the included `Procfile`:
```bash
railway up
```

---

## 🎈 Option 5: Streamlit Community Cloud

To host the Streamlit UI dashboard on Streamlit Cloud:
1. Push your repository to GitHub.
2. Log into [Streamlit Cloud](https://streamlit.io/cloud).
3. Click **New app** and point to `app/streamlit_app.py`.
4. In **Advanced Settings**, add the environment variable `API_BASE` pointing to your deployed FastAPI backend URL.

---

## 🧪 Option 6: Python Virtual Environment Deployment (Manual)

```bash
# 1. Clone repository
git clone https://github.com/rbhele7/smartPark-AI.git
cd smartPark-AI

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start production FastAPI server
python run.py
```

---

## 🔍 Health & Observability Endpoints

| Endpoint | Method | Purpose |
| :--- | :--- | :--- |
| `/health` | GET | Liveness probe & model status |
| `/api/v1/model/info` | GET | Model metadata, tensor shape, & class mapping |
| `/api/v1/analytics` | GET | Real-time aggregate inference metrics & request counts |
| `/dashboard` | GET | Real-time HTML5/JS Occupancy Monitoring Web UI |
| `/docs` | GET | Interactive OpenAPI (Swagger) Documentation |
