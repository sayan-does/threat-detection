import streamlit as st
from pathlib import Path
import PIL
import tempfile
import cv2
import numpy as np
import settings
import helper

st.set_page_config(page_title="Threat Detection App")

st.title("Threat Detection")
st.write("Welcome to the Threat Detection webapp. It detects threats such as a Fire Arm or a Sharp Object that could be used a weapon")

st.header("Instructions:")
st.write("1. Use the sidebar to select the model confidence level and the type of source (image or video).")
st.write("2. Upload an image or video for object detection.")
st.write("3. Click the 'Detect Threat' button to perform object detection.")
st.write("4. View the detected weapons in the output section.")

st.sidebar.header("Model Configuration")

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 50)) / 100

model_path = Path(settings.DETECTION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio("Select Source", settings.SOURCES_LIST)

source_img = None

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
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
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Threat'):
                res = model.predict(uploaded_image, conf=confidence)
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
