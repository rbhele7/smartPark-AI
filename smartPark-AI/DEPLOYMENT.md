# 🚀 SmartPark AI - Cloud & Serverless Deployment Guide

This guide provides comprehensive, step-by-step instructions for deploying **SmartPark AI** across serverless, containerized, and cloud production environments including **Vercel** and **Render**.

---

## 🏗️ Architecture Overview

SmartPark AI consists of three complementary runtime components:

1. **FastAPI Backend REST API**: High-performance asynchronous inference engine serving `/api/v1/predict`, `/api/v1/predict/batch`, `/api/v1/predict/parking-lot`, `/health`, and `/api/v1/analytics`.
2. **Built-in HTML5/JS Web Dashboard**: Served directly from the backend at `/dashboard` (and root `/`).
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

## ⚡ Option 1: Deploy on Vercel (FastAPI Backend + Dashboard)

Vercel hosts the FastAPI backend server and embedded web dashboard using `@vercel/python` Serverless Functions.

### Method 1: Vercel Web Dashboard (Recommended)
1. Go to [Vercel Dashboard](https://vercel.com/new) and log in.
2. Select **Import Repository** and select `rbhele7/smartPark-AI`.
3. Keep default settings (Vercel automatically detects `vercel.json` and `api/index.py`).
4. (Optional) In **Environment Variables**, add:
   - `CONFIDENCE_THRESHOLD`: `0.5`
5. Click **Deploy**. Vercel will output your live URL (e.g., `https://smartpark-ai.vercel.app`).

### Method 2: Vercel CLI Deployment
```bash
# Install Vercel CLI
npm install -g vercel

# Authenticate & Deploy
vercel login
vercel --prod
```

### Accessing Vercel Deployment Endpoints:
- **Interactive Dashboard**: `https://<your-vercel-app>.vercel.app/` or `/dashboard`
- **OpenAPI Swagger Docs**: `https://<your-vercel-app>.vercel.app/docs`
- **Health Check**: `https://<your-vercel-app>.vercel.app/health`

---

## 💜 Option 2: Deploy on Render (FastAPI + Streamlit Analytics)

Render deploys both the FastAPI API and the Streamlit Analytics dashboard simultaneously using Infrastructure-as-Code (`render.yaml`).

### Step-by-Step Render Blueprint Deployment:
1. Log into your [Render Account](https://dashboard.render.com).
2. Click **New +** -> **Blueprint**.
3. Connect your GitHub repository `https://github.com/rbhele7/smartPark-AI`.
4. Render will parse `render.yaml` and create two web services:
   - `smartpark-api`: FastAPI backend Web Service
   - `smartpark-dashboard`: Streamlit analytics Web Service
5. Click **Apply**. Render will build and deploy both services automatically!

---

## 🐳 Option 3: Local Docker & Docker Compose Deployment

### 1. Run single Docker container
```bash
docker build -t smartpark-ai:latest .
docker run -d -p 8080:8080 --name smartpark-service smartpark-ai:latest
```

### 2. Multi-Container Stack (FastAPI + Streamlit) via Docker Compose
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

## ☁️ Option 4: Google Cloud Run

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

---

## 🎈 Option 5: Streamlit Community Cloud

To host the Streamlit UI dashboard on Streamlit Cloud:
1. Push your repository to GitHub.
2. Log into [Streamlit Cloud](https://streamlit.io/cloud).
3. Click **New app** and point to `app/streamlit_app.py`.
4. In **Advanced Settings**, add the environment variable `API_BASE` pointing to your deployed FastAPI backend URL (e.g., your Vercel or Render backend URL).

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
| `/` | GET | Redirects to `/dashboard` |
| `/health` | GET | Liveness probe & model status |
| `/api/v1/model/info` | GET | Model metadata, tensor shape, & class mapping |
| `/api/v1/analytics` | GET | Real-time aggregate inference metrics & request counts |
| `/dashboard` | GET | Real-time HTML5/JS Occupancy Monitoring Web UI |
| `/docs` | GET | Interactive OpenAPI (Swagger) Documentation |
