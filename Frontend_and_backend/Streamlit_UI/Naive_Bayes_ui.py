import sys
import os
import pickle
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import streamlit as st
import base64

# Add backend path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.predict import extract_features_from_uploaded_file

# Load Naive Bayes models
with open("Backend/Models/nb_model.pkl", "rb") as f:
    nb_color_model = pickle.load(f)

with open("Backend/Models/nb_scaler.pkl", "rb") as f:
    color_scaler = pickle.load(f)

with open("Backend/Models/nb_label_encoder.pkl", "rb") as f:
    color_encoder = pickle.load(f)

with open("Backend/Models/nb_hog_model.pkl", "rb") as f:
    nb_hog_model = pickle.load(f)

with open("Backend/Models/nb_hog_scaler.pkl", "rb") as f:
    hog_scaler = pickle.load(f)

with open("Backend/Models/nb_hog_label_encoder.pkl", "rb") as f:
    hog_encoder = pickle.load(f)

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def display_centered_image(image, title):
    image = image.resize((300, 300))
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

    st.markdown(
        f"""
        <div style='text-align: center; margin-bottom: 30px;'>
            <h5>{title}</h5>
            <img src='data:image/png;base64,{img_b64}' width='300' style='border-radius: 12px;'/>
        </div>
        """,
        unsafe_allow_html=True
    )

def run():
    st.title("Naive Bayes Image Classification")
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

    # Button Styling
    st.markdown(
        """
        <style>
        div.stButton > button {
            color: black !important;
            background-color: white !important;
            font-size: 18px !important;
            font-weight: bold !important;
            width: 150px !important;
            border: 2px solid black !important;
            border-radius: 8px !important;
        }
        div.stButton > button:hover {
            background-color: #f0f0f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Center the Predict button with spacing
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
                    color_pred_encoded = nb_color_model.predict(color_feat_scaled)[0]
                    color_pred_label = color_encoder.inverse_transform([color_pred_encoded])[0]

                    hog_feat_scaled = hog_scaler.transform(hog_feat)
                    hog_pred_encoded = nb_hog_model.predict(hog_feat_scaled)[0]
                    hog_pred_label = hog_encoder.inverse_transform([hog_pred_encoded])[0]

                    st.success("Prediction successful!")

                    st.markdown(
                        f"""
                        <div style='text-align: left; font-size: 18px; margin-top: 30px;'>
                            <h4 style='margin-bottom: 15px;'>Predicted Class Labels</h4>
                            <p><strong>Using Color Histogram:</strong> {color_pred_label}</p>
                            <p><strong>Using HOG Features:</strong> {hog_pred_label}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")



