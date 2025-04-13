import sys
import os
import pickle
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import streamlit as st
import base64

# Dynamically add Backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.predict import extract_features_from_uploaded_file

# Load models
with open("Backend/Models/knn_color_hist.pkl", "rb") as f:
    knn_color = pickle.load(f)
with open("Backend/Models/knn_hog_scaler.pkl", "rb") as f:
    scaler_hog = pickle.load(f)
with open("Backend/Models/knn_hog_pca.pkl", "rb") as f:
    pca_hog = pickle.load(f)
with open("Backend/Models/knn_hog.pkl", "rb") as f:
    knn_hog = pickle.load(f)

def run():
    st.title("KNN Image Classification")
    st.header("Upload or Link an Image")
    
    # Image uploader and URL input
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or enter an image URL")
    image = None
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Centering the image using HTML
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src="data:image/png;base64,{image_to_base64(image)}" width="300" style="border-radius: 12px;">
            </div>
            """, unsafe_allow_html=True)
    elif image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            # Centering the image using HTML
            st.markdown(
                f"""
                <div style='text-align: center;'>
                    <img src="data:image/png;base64,{image_to_base64(image)}" width="300" style="border-radius: 12px;">
                </div>
                """, unsafe_allow_html=True)
        except:
            st.error("Failed to load image from the provided URL.")
    
    # Add a gap between the image and the Predict button
    st.markdown("<br><br>", unsafe_allow_html=True)  # Adding gap between the image and the button
    
    # Layout for prediction button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Predict"):
            if image is None:
                st.error("Please upload an image or enter a valid URL.")
            else:
                try:
                    img_byte_arr = BytesIO()
                    image.save(img_byte_arr, format='PNG')
                    fake_uploaded_file = BytesIO(img_byte_arr.getvalue())
                    # Extract features
                    color_feat, hog_feat = extract_features_from_uploaded_file(fake_uploaded_file)
                    # ---- Prediction using color histogram ----
                    color_pred = knn_color.predict(color_feat)[0]
                    hog_scaled = scaler_hog.transform(hog_feat)
                    hog_reduced = pca_hog.transform(hog_scaled)
                    hog_pred = knn_hog.predict(hog_reduced)[0]

                    # Show Results
                    st.success("Prediction successful!")
                    st.subheader("Predicted Class Labels")
                    st.write(f"Using Color Histogram: **{color_pred}**")
                    st.write(f"Using HOG + PCA: **{hog_pred}**")
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()




            










