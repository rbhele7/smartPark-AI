# 🚀 SmartPark AI - Decoupled Cloud Deployment Guide (Vercel + Render)

This guide provides step-by-step instructions for deploying **SmartPark AI** with the **Frontend on Vercel** and the **AI Inference Backend on Render**.

---

## 🏗️ Architecture Overview

```
 ┌──────────────────────────────────────┐
 │     Vercel Edge Network              │
 │  (Frontend HTML5/JS Dashboard)       │
 │  https://smartpark-ai.vercel.app     │
 └──────────────────┬───────────────────┘
                    │ REST API (CORS enabled)
                    ▼
 ┌──────────────────────────────────────┐
 │          Render Cloud                │
 │  (FastAPI Deep Learning Backend)     │
 │  https://smartpark-api.onrender.com  │
 └──────────────────────────────────────┘
```

- **Frontend (Vercel)**: Serves static assets (`public/index.html`) on Vercel Edge CDN for global low latency.
- **Backend (Render)**: Executes Python 3.12 + FastAPI + Keras CNN model inference (`smartpark-api`).

---

## ⚙️ Step 1: Deploy Backend on Render

1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** -> **Blueprint**.
3. Select your `rbhele7/smartPark-AI` GitHub repository.
4. Render will parse `render.yaml` and provision:
   - `smartpark-api`: FastAPI backend web service.
5. Click **Apply**.
6. Once deployed, copy your live Render API URL:
   `https://smartpark-api.onrender.com`

### Verify Backend Deployment:
- Health check: `curl https://smartpark-api.onrender.com/health`
- OpenAPI Swagger docs: `https://smartpark-api.onrender.com/docs`

---

## ⚡ Step 2: Deploy Frontend on Vercel

1. Log into your [Vercel Dashboard](https://vercel.com/new).
2. Select **Import Repository** and select `rbhele7/smartPark-AI`.
3. Keep default settings (Vercel automatically detects `vercel.json` and serves `public/index.html`).
4. Click **Deploy**. Vercel will launch your site at:
   `https://smartpark-ai.vercel.app`

### Dynamic API URL Configuration:
- By default, the frontend connects to `https://smartpark-api.onrender.com`.
- If your Render URL is different, open the deployed Vercel URL, click the **API URL** button in the header, enter your custom Render backend URL, and click OK.

---

## 🧪 Local Testing & Verification

```bash
# 1. Clone repository
git clone https://github.com/rbhele7/smartPark-AI.git
cd smartPark-AI

# 2. Virtual environment & dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Run automated tests
python3 -m pytest tests/ -v

# 4. Launch local backend
python run.py
```
- Open [http://localhost:8080](http://localhost:8080) to test the dashboard locally.
