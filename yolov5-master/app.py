import streamlit as st
import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image

# Load YOLOv5 model
@st.cache_resource
def load_model(weights_path="yolov5s.pt"):
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path)
    return model

# Function to detect objects
def detect_objects(image, model):
    results = model(image)
    return results

# Streamlit app
st.title("Object Detection App")
st.write("Capture an image using your webcam and detect objects using YOLOv5.")

# Camera input
camera = st.camera_input("Take a photo")

if camera:
    # Convert camera input to PIL Image
    img = Image.open(camera)
    
    # Convert to BGR for OpenCV processing
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Display the original image
    st.image(img, caption="Captured Image", use_column_width=True)

    # Load YOLOv5 model
    model = load_model()

    # Perform object detection
    results = detect_objects(img_bgr, model)

    # Render the results
    detected_img = np.squeeze(results.render())
    detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)

    # Display the detected image
    st.image(detected_img_rgb, caption="Detected Objects", use_column_width=True)

    # Display detection details
    st.write("Detection Results:")
    st.write(results.pandas().xyxy[0])
