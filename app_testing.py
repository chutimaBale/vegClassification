import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================
# CONFIG
# =========================
IMG_SIZE = 224

# 🔥 เปลี่ยน path โมเดลตรงนี้
MODEL_PATH = "model/vegetable_condition_EfficientNetV2_final.h5"

# 🔥 เปลี่ยนชื่อ class ให้ตรง dataset
CLASS_NAMES = ['carrot_Healthy', 'carrot_Rotten', 'tomato_Healthy', 'tomato_Rotten', 'cucumber_Healthy', 'cucumber_Rotten']


# =========================
# LOAD MODEL (cache)
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()


# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image)

    # 🔥 IMPORTANT: ใช้ preprocess_input ตามโมเดล
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
    img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# =========================
# UI
# =========================
st.title("🥦 Vegetable Classification App")
st.write("Upload an image to classify vegetable condition")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    st.image(image, caption='Uploaded Image', use_column_width=True)

    st.write("🔍 Predicting...")

    processed_image = preprocess_image(image)

    predictions = model.predict(processed_image)
    confidence = np.max(predictions)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")

    # แสดง probability ทุก class
    st.subheader("Class Probabilities")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{CLASS_NAMES[i]}: {prob:.4f}")
