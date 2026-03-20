import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

# Set the title of the Streamlit app
st.title("Basic Object Detection App")

# Create a file uploader in the sidebar
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
"""
if uploaded_file is not None:
    # Read the image
    image_bytes = uploaded_file.getvalue()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # Display the original image in the main area
    st.subheader("Original Image")
    st.image(image, channels="BGR", caption="Original Image", use_column_width=True)

    # Load the pre-trained YOLO model (using st.cache_resource to load once)
    @st.cache_resource
    def load_model():
        # You can use any model Ultralytics supports, e.g., 'yolov8n.pt'
        model = YOLO("best.pt")
        return model
    
    model = load_model()

    # Perform object detection when a button is clicked
    if st.sidebar.button("Detect Objects"):
        st.subheader("Detected Objects")
        
        # Perform detection
        results = model(image)
        
        # Render the results on the image
        # The 'plot()' method draws bounding boxes and labels
        detected_image = results[0].plot()

        # Display the image with detections
        st.image(detected_image, channels="BGR", caption="Image with Detections", use_column_width=True)
else:
    st.sidebar.info("Upload an image to get started.")
"""
