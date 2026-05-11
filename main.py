import torch
import ultralytics.nn.tasks
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

import streamlit as st
from pathlib import Path
import PIL
import tempfile
import cv2
import numpy as np
import settings
import helper

# Set page title and icon
st.set_page_config(
    page_title="Threat Detection App"
)

# Title and description
st.title("Threat Detection")
st.write("Welcome to the Threat Detection webapp. It detects threats such as a Fire Arm or a Sharp Object that could be used a weapon")

# Instructions
st.header("Instructions:")
st.write("1. Use the sidebar to select the model confidence level and the type of source (image or video).")
st.write("2. Upload an image or video for object detection.")
st.write("3. Click the 'Detect Threat' button to perform object detection.")
st.write("4. View the detected weapons in the output section.")

# Sidebar
st.sidebar.header("Model Configuration")

# Model Options
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 50)) / 100

model_path = Path(settings.DETECTION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

source_img = None

# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DETECTED_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Threat'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_user_video(confidence, model)

else:
    st.error("Please select a valid source type!")
