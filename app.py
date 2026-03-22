import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
import torch
import torchvision.models as models
from PIL import Image
import numpy as np

# Show title and description.
st.title("📄 Veggies Detection")
st.write(
    "Upload an image below "
    "To use this app, you need to "
)
# ================= Streamlit UI =================


st.title("Vegetable Classification App :)")
st.write(
    "Upload an image below "
    "To use this app, you need to "
    
'''
st.sidebar.title("Settings")
model_type = st.sidebar.selectbox(
    "Choose Model Configuration",
    ("YOLO (Solo)", "MobileNetV2 (Solo)","EfficientNetV2 (Solo)","ResNet50 (Solo)",)
)
'''

# Load YOLO model
model = YOLO("best.pt")

# Upload image
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    st.info("Running Classification...")

        veg_count = class_names.count("Carrot_Healthy")
        st.write(f"Vegetable detected: **{veg_count}**")
