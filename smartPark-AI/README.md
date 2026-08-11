# 🅿️ SmartPark AI - Real-Time Parking Occupancy Detection & Analytics

[![Vercel](https://img.shields.io/badge/Deploy%20with-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Frbhele7%2FsmartPark-AI)
[![Render](https://img.shields.io/badge/Deploy%20to-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/deploy?repo=https://github.com/rbhele7/smartPark-AI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

SmartPark AI is an end-to-end deep learning system powered by custom Convolutional Neural Networks (CNN) and FastAPI designed for high-precision, low-latency parking spot occupancy classification and analytics.

---

## ✨ Features

- **🚀 FastAPI High-Performance Backend**: Asynchronous inferencing supporting single-image, multi-image batching, and full parking lot ROI segmentation grid prediction.
- **📊 Real-Time Interactive Dashboards**:
  - Embedded web dashboard served at `/dashboard` (or root `/`).
  - Rich administrative analytics dashboard powered by Streamlit (`app/streamlit_app.py`).
- **🧠 CNN Occupancy Model**: Pre-trained Keras model for binary parking spot occupancy detection (`vacant` vs `occupied`).
- **☁️ Cloud & Serverless Deployable**: Native support for **Vercel** serverless functions and **Render** 1-Click Blueprints.
- **🐳 Containerized**: Pre-configured Docker, Docker Compose, Google Cloud Run, Heroku/Railway Procfile, and CI/CD GitHub Actions pipeline.
- **⚡ Health Monitoring & Analytics**: Endpoints for latency metrics, request counts, and model info.

---

## ☁️ Cloud Deployment Options

### ⚡ Option 1: Deploy on Vercel (FastAPI Backend + Dashboard)

SmartPark AI is pre-configured with `api/index.py` and `vercel.json` for instant deployment on Vercel Serverless Functions.

#### Method A: Via Vercel Web Dashboard (1-Click)
1. Import your repository `https://github.com/rbhele7/smartPark-AI` in the [Vercel Dashboard](https://vercel.com/new).
2. Framework Preset: Select **Other**.
3. Environment Variables (Optional):
   - `CONFIDENCE_THRESHOLD`: `0.5`
4. Click **Deploy**. Vercel will automatically build the serverless functions and serve `/` and `/dashboard`.

#### Method B: Via Vercel CLI
```bash
npm i -g vercel
vercel login
vercel --prod
```

---

### 💜 Option 2: Deploy on Render (FastAPI + Streamlit Analytics)

SmartPark AI includes a `render.yaml` Infrastructure-as-Code Blueprint that launches both the FastAPI API service (`smartpark-api`) and the Streamlit UI dashboard (`smartpark-dashboard`).

1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Blueprint**.
3. Connect your repository `https://github.com/rbhele7/smartPark-AI`.
4. Render will automatically launch both services:
   - `smartpark-api`: FastAPI backend at `https://smartpark-api.onrender.com`
   - `smartpark-dashboard`: Streamlit UI at `https://smartpark-dashboard.onrender.com`

---

## 🛠️ Quickstart (Local)

### 1. Clone & Install
```bash
git clone https://github.com/rbhele7/smartPark-AI.git
cd smartPark-AI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch FastAPI Server
```bash
python run.py
```
- Interactive Web Dashboard: [http://localhost:8080/dashboard](http://localhost:8080/dashboard)
- API OpenAPI Documentation: [http://localhost:8080/docs](http://localhost:8080/docs)
- Health Check: [http://localhost:8080/health](http://localhost:8080/health)

### 3. Launch Streamlit Analytics App (Optional)
```bash
API_BASE=http://localhost:8080 streamlit run app/streamlit_app.py
```
- Streamlit Analytics UI: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker & Container Deployment

### Run Single Container
```bash
docker build -t smartpark-ai .
docker run -p 8080:8080 smartpark-ai
```

### Run Multi-Container Stack (FastAPI + Streamlit) via Docker Compose
```bash
docker-compose up --build -d
```

---

## 🧪 Running Tests

```bash
python3 -m pytest tests/ -v
```

---

## 📁 Repository Structure

```
smartPark-AI/
├── api/
│   └── index.py             # Vercel Serverless Function entry point
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Environment configuration
│   ├── model.py             # Keras CNN Singleton model manager
│   ├── streamlit_app.py     # Streamlit UI dashboard
│   ├── routers/             # API routes (predict, health, analytics, model_info)
│   └── static/              # Dashboard frontend static assets
├── models/                  # Saved Keras model artifacts
├── tests/                   # Automated pytest suite
├── vercel.json              # Vercel Serverless Function deployment configuration
├── render.yaml              # Render Blueprint IaC configuration
├── Dockerfile               # Production Dockerfile for FastAPI
├── Dockerfile.streamlit     # Dedicated Dockerfile for Streamlit
├── docker-compose.yml       # Docker Compose multi-service orchestration
├── Procfile                 # Process manifest for Railway/Heroku
├── .github/workflows/       # CI/CD GitHub Actions workflow
├── DEPLOYMENT.md            # Detailed Cloud & Serverless Deployment Guide
└── run.py                   # Production launcher script
```