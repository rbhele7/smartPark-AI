import streamlit as st
import requests
import io
import json
from PIL import Image

st.set_page_config(
    page_title="SmartPark AI - Real-Time Parking Occupancy",
    page_icon="🅿️",
    layout="wide"
)

st.title("🅿️ SmartPark AI - Parking Occupancy Detection")
st.subheader("FastAPI Powered Real-Time CNN Occupancy Analytics")

API_BASE = "http://127.0.0.1:8001"

# Sidebar System Health
st.sidebar.header("System Status")
try:
    health_res = requests.get(f"{API_BASE}/health", timeout=2)
    if health_res.status_code == 200:
        st.sidebar.success("Backend Server: ACTIVE")
        data = health_res.json()
        st.sidebar.json(data)
    else:
        st.sidebar.warning("Backend Offline")
except Exception:
    st.sidebar.error("Cannot connect to FastAPI backend at port 8001")

tab1, tab2, tab3 = st.tabs(["Single Spot Predictor", "Batch Analyzer", "Full Parking Lot ROI"])

with tab1:
    st.header("Single Spot Prediction")
    uploaded_file = st.file_uploader("Upload Parking Spot Image", type=["jpg", "jpeg", "png"], key="single")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Spot", width=300)
        
        if st.button("Predict Spot Occupancy", type="primary"):
            buf = io.BytesIO()
            image.save(buf, format="JPEG")
            res = requests.post(f"{API_BASE}/api/v1/predict", files={"file": (uploaded_file.name, buf.getvalue(), "image/jpeg")})
            if res.status_code == 200:
                result = res.json()
                pred = result["prediction"]
                cname = pred["class_name"].upper()
                conf = pred["confidence"]
                lat = result["inference_time_ms"]
                
                if cname == "VACANT":
                    st.success(f"STATUS: {cname} | Confidence: {conf:.1%} | Latency: {lat} ms")
                else:
                    st.error(f"STATUS: {cname} | Confidence: {conf:.1%} | Latency: {lat} ms")
            else:
                st.error("Prediction failed")

with tab2:
    st.header("Batch Spot Processing")
    uploaded_files = st.file_uploader("Upload Multiple Spot Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch")
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} images.")
        if st.button("Analyze Batch", type="primary"):
            files_payload = []
            for f in uploaded_files:
                buf = io.BytesIO()
                Image.open(f).save(buf, format="JPEG")
                files_payload.append(("files", (f.name, buf.getvalue(), "image/jpeg")))
            
            res = requests.post(f"{API_BASE}/api/v1/predict/batch", files=files_payload)
            if res.status_code == 200:
                data = res.json()
                st.write(f"Total: {data['total_images']} images processed in {data['inference_time_ms']} ms")
                st.json(data["predictions"])

with tab3:
    st.header("Full Parking Lot ROI Grid Analysis")
    lot_file = st.file_uploader("Upload Full Parking Lot Camera Overview", type=["jpg", "jpeg", "png"], key="lot")
    if lot_file is not None:
        if st.button("Generate Occupancy Map", type="primary"):
            buf = io.BytesIO()
            Image.open(lot_file).save(buf, format="JPEG")
            res = requests.post(f"{API_BASE}/api/v1/predict/parking-lot", files={"file": (lot_file.name, buf.getvalue(), "image/jpeg")})
            if res.status_code == 200:
                data = res.json()
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Spots", data["total_spots"])
                col2.metric("Occupied", data["occupied_count"])
                col3.metric("Vacant", data["vacant_count"])
                
                import base64
                img_data = base64.b64decode(data["annotated_image_base64"])
                st.image(Image.open(io.BytesIO(img_data)), caption="Annotated Parking Lot Occupancy Map")
