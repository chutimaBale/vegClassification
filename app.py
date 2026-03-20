import streamlit as st
import tensorflow as tf
from ultralytics import YOLO
import torch
import torchvision.models as models
from PIL import Image
import numpy as np


# ================= FUNCTION =================

@st.cache_resource
def load_classification_models(model_type):
    models_dict = {}

    if model_type == "ResNet50 (Solo)":
        models_dict = {
            "carrot": tf.keras.models.load_model("resnet/carrot_resnet50_model.h5"),
            "cucumber": tf.keras.models.load_model("resnet/cucumber_resnet50_model.h5"),
            "tomato": tf.keras.models.load_model("resnet/tomato_resnet50_model.h5"),
        }

    elif model_type == "MobileNetV3 (Solo)":
        models_dict = {
            "carrot": tf.keras.models.load_model("mobilenet/carrot_mobilenetv2.h5"),
            "cucumber": tf.keras.models.load_model("mobilenet/cucumber_mobilenetv2.h5"),
            "tomato": tf.keras.models.load_model("mobilenet/tomato_mobilenetv2.h5"),
        }

    elif model_type == "EfficientNetV2 (Solo)":
        models_dict = {
            "carrot": tf.keras.models.load_model("efficientnet/carrot_efficientnetv2.h5"),
            "cucumber": tf.keras.models.load_model("efficientnet/cucumber_efficientnetv2.h5"),
            "tomato": tf.keras.models.load_model("efficientnet/tomato_efficientnetv2.h5"),
        }

    return models_dict

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image 

# ================= Streamlit UI =================


st.title("Vegetable Classification App :)")
st.sidebar.title("Settings")
model_type = st.sidebar.selectbox(
    "Choose Model Configuration",
    ("YOLO (Solo)", "MobileNetV2 (Solo)","EfficientNetV2 (Solo)","ResNet50 (Solo)",)
)

# Load YOLO model
# model = YOLO("detect/train/best.pt")

# Upload image
uploaded_image = st.file_uploader("Upload an image (jpg, png)", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

    image = Image.open(uploaded_image).convert("RGB")
    image_np = np.array(image)

    st.info("Running Classification...")

    # ================= YOLO =================
    if model_type == "YOLO (Solo)":
        model = YOLO("best.pt")

        results = model.predict(image_np, conf=0.4)
        result_image = results[0].plot()

        st.image(result_image, caption="Detection Result", use_container_width=True)

        boxes = results[0].boxes
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [model.names[i] for i in class_ids]

        veg_count = class_names.count("Vegetable")
        st.write(f"Vegetable detected: **{veg_count}**")

    # ================= CLASSIFICATION MODELS =================
    else:
        models_dict = load_classification_models(model_type)
        processed_img = preprocess_image(image)

        predictions = {}

        for name, model in models_dict.items():
            pred = model.predict(processed_img)[0][0]  # assuming binary output
            predictions[name] = float(pred)

        # Get best prediction
        best_class = max(predictions, key=predictions.get)
        confidence = predictions[best_class]

        st.write("### Prediction Results:")
        for k, v in predictions.items():
            st.write(f"{k}: {v:.4f}")

        st.success(f"Predicted: **{best_class}** (confidence: {confidence:.4f})")
