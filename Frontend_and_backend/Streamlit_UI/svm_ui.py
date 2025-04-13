import sys
import os
import joblib
import base64
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import streamlit as st

# Dynamically add Backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.predict import extract_features_from_uploaded_file

# Load SVM models and preprocessors using joblib
svm_color = joblib.load("Backend/Models/svm_color_hist.pkl")
color_scaler = joblib.load("Backend/Models/svm_color_scaler.pkl")
label_encoder_color = joblib.load("Backend/Models/svm_label_encoder.pkl")

svm_hog = joblib.load("Backend/Models/svm_hog_model.pkl")
scaler_hog = joblib.load("Backend/Models/svm_hog_scaler.pkl")
label_encoder_hog = joblib.load("Backend/Models/svm_hog_label_encoder.pkl")

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_centered_image(image, title):
    image = image.resize((300, 300))  # Resize to 300x300
    img_b64 = image_to_base64(image)
    st.markdown(
        f"""
        <div style='text-align: center; margin-bottom: 40px;'>
            <h5>{title}</h5>
            <img src='data:image/png;base64,{img_b64}' width='300' style='border-radius: 12px;'/>
        </div>
        """,
        unsafe_allow_html=True
    )

def run():
    st.title("SVM Image Classification")
    st.header("Upload or Link an Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or enter an image URL")

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        display_centered_image(image, "Uploaded Image")

    elif image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            display_centered_image(image, "Image from URL")
        except:
            st.error("Failed to load image from the provided URL.")

    # Center the Predict button with vertical spacing from image
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
        if st.button("Predict"):
            if image is None:
                st.error("Please upload an image or enter a valid URL.")
            else:
                try:
                    img_byte_arr = BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    fake_uploaded_file = BytesIO(img_byte_arr.getvalue())

                    color_feat, hog_feat = extract_features_from_uploaded_file(fake_uploaded_file)

                    color_feat_scaled = color_scaler.transform(color_feat)
                    color_pred_encoded = svm_color.predict(color_feat_scaled)[0]
                    color_pred = label_encoder_color.inverse_transform([color_pred_encoded])[0]

                    hog_scaled = scaler_hog.transform(hog_feat)
                    hog_pred_encoded = svm_hog.predict(hog_scaled)[0]
                    hog_pred = label_encoder_hog.inverse_transform([hog_pred_encoded])[0]

                    st.success("Prediction successful!")

                    st.markdown(
                        f"""
                        <div style='text-align: left; font-size: 18px; margin-top: 30px;'>
                            <h4 style='margin-bottom: 15px;'>Predicted Class Labels</h4>
                            <p><strong>Using Color Histogram:</strong> {color_pred}</p>
                            <p><strong>Using HOG Features:</strong> {hog_pred}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")





