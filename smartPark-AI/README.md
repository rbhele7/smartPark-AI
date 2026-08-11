# 🅿️ SmartPark AI - Real-Time Parking Occupancy Detection & Analytics

SmartPark AI is an end-to-end deep learning system powered by custom Convolutional Neural Networks (CNN) and FastAPI designed for high-precision, low-latency parking spot occupancy classification and analytics.

---

## ✨ Features

- **🚀 FastAPI High-Performance Backend**: Asynchronous inferencing supporting single-image, multi-image batching, and full parking lot ROI segmentation grid prediction.
- **📊 Real-Time Interactive Dashboards**:
  - Embedded web dashboard served at `/dashboard`.
  - Rich administrative analytics dashboard powered by Streamlit (`app/streamlit_app.py`).
- **🧠 CNN Occupancy Model**: Pre-trained Keras model for binary parking spot occupancy detection (`vacant` vs `occupied`).
- **🐳 Production Ready & Deployable**: Pre-configured Docker, Docker Compose, Google Cloud Run, Render Blueprint, Heroku/Railway Procfile, and CI/CD GitHub Actions pipeline.
- **⚡ Health Monitoring & Analytics**: Endpoints for latency metrics, request counts, and model info.

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
- API Docs: [http://localhost:8080/docs](http://localhost:8080/docs)
- Embedded Dashboard: [http://localhost:8080/dashboard](http://localhost:8080/dashboard)

### 3. Launch Streamlit Analytics App (Optional)
```bash
API_BASE=http://localhost:8080 streamlit run app/streamlit_app.py
```
- Streamlit UI: [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker Deployment

### Run single Docker container
```bash
docker build -t smartpark-ai .
docker run -p 8080:8080 smartpark-ai
```

### Run Multi-Container Stack (FastAPI + Streamlit) via Docker Compose
```bash
docker-compose up --build -d
```

---

## ☁️ Cloud Deployment Options

For step-by-step deployment instructions on:
- **Google Cloud Run**
- **Render / Railway (1-Click Blueprint)**
- **Streamlit Community Cloud**
- **AWS App Runner / ECS**

Please see the comprehensive **[DEPLOYMENT.md](file:///Users/rbhele/ParkAI/smartPark-AI/DEPLOYMENT.md)** guide.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📁 Repository Structure

```
smartPark-AI/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Environment configuration
│   ├── model.py             # Keras CNN Singleton model manager
│   ├── streamlit_app.py     # Streamlit UI dashboard
│   ├── routers/             # API routes (predict, health, analytics, model_info)
│   └── static/              # Dashboard frontend static assets
├── models/                  # Saved Keras model artifacts
├── tests/                   # Automated pytest suite
├── Dockerfile               # Production Dockerfile for FastAPI
├── Dockerfile.streamlit     # Dedicated Dockerfile for Streamlit
├── docker-compose.yml       # Docker Compose multi-service orchestration
├── render.yaml              # Render Blueprint IaC configuration
├── Procfile                 # Process manifest for Render/Railway/Heroku
├── .github/workflows/       # CI/CD GitHub Actions workflow
├── DEPLOYMENT.md            # Comprehensive Cloud Deployment Guide
└── run.py                   # Production launcher script
```