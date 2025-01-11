import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


model = tf.keras.models.load_model(r"D:\mini project 1\model.keras")  

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

st.title("Deepfake Face Swap Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:

    img = preprocess_image(uploaded_file)
    
 
    prediction = model.predict(img)
    label = "Deepfake Detected" if prediction[0][0] >= 0.7 else "No Deepfake Detected"
    
  
    st.write(f"Prediction: {label}")
    
    
    uploaded_img = Image.open(uploaded_file)
    st.image(uploaded_img, caption="Uploaded Image", use_column_width=True)
