# 🅿️ SmartPark AI - Real-Time Parking Occupancy Detection & Analytics

[![Vercel Frontend](https://img.shields.io/badge/Deploy%20Frontend-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com/new/clone?repository-url=https%3A%2F%2Fgithub.com%2Frbhele7%2FsmartPark-AI)
[![Render Backend](https://img.shields.io/badge/Deploy%20Backend-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://render.com/deploy?repo=https://github.com/rbhele7/smartPark-AI)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

SmartPark AI is an end-to-end deep learning system powered by custom Convolutional Neural Networks (CNN) and FastAPI designed for high-precision, low-latency parking spot occupancy classification and analytics.

---

## 🏗️ Architecture: Decoupled Cloud Deployment

- **🌐 Frontend Dashboard (Vercel)**: Lightning-fast static HTML5/JS Occupancy Monitoring Dashboard deployed on **Vercel Edge Network**.
- **⚙️ AI Inference Engine (Render)**: High-performance FastAPI Python Web Service serving deep-learning inference endpoints (`/api/v1/predict`, `/health`, `/api/v1/analytics`).

---

## 🚀 Quick Deployment Guide

### 1️⃣ Deploy Backend on Render
1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Blueprint**.
3. Connect your repository `https://github.com/rbhele7/smartPark-AI`.
4. Render will launch the **`smartpark-api`** service. Note down your Render service URL (e.g. `https://smartpark-api.onrender.com`).

### 2️⃣ Deploy Frontend on Vercel
1. Import repository `https://github.com/rbhele7/smartPark-AI` in the [Vercel Dashboard](https://vercel.com/new).
2. Framework Preset: **Other** / **Static Site** (Vercel automatically detects `public/index.html`).
3. Click **Deploy**. Vercel will host your live dashboard at `https://smartpark-ai.vercel.app`.
4. Open the deployed Vercel site. The UI automatically connects to `https://smartpark-api.onrender.com`. You can also click **API URL** in the top header to enter your custom Render backend URL!

---

## 🛠️ Local Development Quickstart

### 1. Clone & Install
```bash
git clone https://github.com/rbhele7/smartPark-AI.git
cd smartPark-AI

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Launch FastAPI Server
```bash
python run.py
```
- Interactive Web Dashboard: [http://localhost:8080/dashboard](http://localhost:8080/dashboard)
- OpenAPI Documentation: [http://localhost:8080/docs](http://localhost:8080/docs)
- Health Check: [http://localhost:8080/health](http://localhost:8080/health)

### 3. Launch Streamlit Analytics UI (Optional)
```bash
API_BASE=http://localhost:8080 streamlit run app/streamlit_app.py
```

---

## 🐳 Docker Deployment

```bash
# Single container
docker build -t smartpark-ai .
docker run -p 8080:8080 smartpark-ai

# Multi-container stack (FastAPI + Streamlit)
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
├── public/
│   └── index.html           # Vercel Frontend static web dashboard
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Environment configuration
│   ├── model.py             # Keras CNN Singleton model manager
│   ├── streamlit_app.py     # Streamlit analytics UI dashboard
│   ├── routers/             # API routes (predict, health, analytics, model_info)
│   └── static/              # Backend embedded dashboard static assets
├── models/                  # Saved Keras CNN model artifacts
├── tests/                   # Automated pytest test suite
├── vercel.json              # Vercel Frontend static deployment configuration
├── render.yaml              # Render Backend IaC Blueprint configuration
├── Dockerfile               # Production Dockerfile for FastAPI backend
├── Dockerfile.streamlit     # Dedicated Dockerfile for Streamlit
├── docker-compose.yml       # Docker Compose multi-service orchestration
├── DEPLOYMENT.md            # Comprehensive Cloud Deployment Guide
└── run.py                   # Production FastAPI server launcher
```