import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Webcam Face Detection")
st.write("Use your webcam to capture an image and detect faces.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img_file_buffer = st.camera_input("Take a picture")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img_cv, (x, y), (x + w, y + h), (0, 255, 0), 2)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption=f"Detected faces: {len(faces)}")
else:
    st.info("Click 'Take a picture' above to capture your face using the webcam.")
