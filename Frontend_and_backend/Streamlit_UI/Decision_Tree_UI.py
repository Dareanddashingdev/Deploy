import sys
import os
import pickle
from io import BytesIO
import requests
from PIL import Image
import numpy as np
import streamlit as st
import base64

# Add Backend directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Backend.predict import extract_features_from_uploaded_file

# Load models and encoders
with open("Backend/Models/decision_tree_color.pkl", "rb") as f:
    dt_color_model = pickle.load(f)
with open("Backend/Models/dt_color_label_encoder.pkl", "rb") as f:
    dt_color_encoder = pickle.load(f)
with open("Backend/Models/hog_decision_tree.pkl", "rb") as f:
    dt_hog_model = pickle.load(f)
with open("Backend/Models/hog_imputer.pkl", "rb") as f:
    hog_imputer = pickle.load(f)
with open("Backend/Models/hog_label_encoder.pkl", "rb") as f:
    hog_label_encoder = pickle.load(f)

def render_centered_image(image):
    buf = BytesIO()
    image.resize((300, 300)).save(buf, format="PNG")
    base64_img = base64.b64encode(buf.getvalue()).decode()
    img_html = f'''
        <div style="display: flex; justify-content: center; margin-top: 20px; margin-bottom: 40px;">
            <img src="data:image/png;base64,{base64_img}" width="300" height="300"
                 style="border: 2px solid black; border-radius: 10px; object-fit: contain;" />
        </div>
    '''
    st.markdown(img_html, unsafe_allow_html=True)

def run():
    st.title("Decision Tree Image Classification")
    st.header("Upload or Link an Image")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    image_url = st.text_input("Or enter an image URL")

    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        render_centered_image(image)

    elif image_url:
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            render_centered_image(image)
        except:
            st.error("Failed to load image from the provided URL.")

    # Button styling
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
            margin-top: 20px;
        }
        div.stButton > button:hover {
            background-color: #f0f0f0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

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

                    # --- Predict using color histogram ---
                    color_pred_encoded = dt_color_model.predict(color_feat)[0]
                    color_pred = dt_color_encoder.inverse_transform([color_pred_encoded])[0]

                    # --- Predict using HOG ---
                    hog_imputed = hog_imputer.transform(hog_feat)
                    hog_pred_encoded = dt_hog_model.predict(hog_imputed)[0]
                    hog_pred = hog_label_encoder.inverse_transform([hog_pred_encoded])[0]

                    # Show Results
                    st.success("Prediction successful!")
                    st.subheader("Predicted Class Labels")
                    st.write(f"Using Color Histogram: **{color_pred}**")
                    st.write(f"Using HOG Features: **{hog_pred}**")

                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")








