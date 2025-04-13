import streamlit as st
from streamlit_option_menu import option_menu
import base64
import os
from PIL import Image

# Import model UI modules
import svm_ui
import knn_ui
import Naive_Bayes_ui
import Decision_Tree_UI
import Logistic_Regression_ui

# Page setup
st.set_page_config(page_title="PRML Course Project", layout="wide", page_icon="🍍")

# --- Convert file to Base64 ---
def get_base64(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        return ""

# --- Get Absolute Paths ---
base_dir = os.path.dirname(os.path.abspath(__file__))
gif_path = os.path.join(base_dir, "UI_background.gif")
image_path = os.path.join(base_dir, "UI_sidebar_image.png")

# --- Sidebar Toggle Switch ---
with st.sidebar:
    enable_gif = st.toggle("Enable GIF Background", value=True)

# --- Apply Background GIF ---
if enable_gif:
    base64_gif = get_base64(gif_path)
    gif_css = f"""
    <style>
    .stApp {{
        position: relative;
        background-image: url("data:image/gif;base64,{base64_gif}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.6);
        z-index: 0;
    }}
    .stApp > * {{
        position: relative;
        z-index: 1;
    }}
    h1, h2, h3, h4, h5, h6, p, li, label {{
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.3);
    }}
    </style>
    """
    st.markdown(gif_css, unsafe_allow_html=True)

# --- Sidebar Background Image ---
base64_image = get_base64(image_path)
sidebar_css = f"""
<style>
[data-testid="stSidebar"] {{
    background-image: url("data:image/png;base64,{base64_image}");
    background-size: cover;
    background-repeat: no-repeat;
}}
</style>
"""
st.markdown(sidebar_css, unsafe_allow_html=True)

# --- Sidebar Menu ---
with st.sidebar:
    selected = option_menu(
        menu_title="SELECT MODEL",
        options=[
            "HOME",
            "SVM",
            "KNN",
            "NAIVE-BAYES",
            "DECISION-TREE",
            "LOGISTIC-REGRESSION"
        ],
        menu_icon="cast",
        default_index=0
    )

# --- Main App Routing ---
if selected == "SVM":
    st.subheader("Support Vector Machine")
    svm_ui.run()

elif selected == "KNN":
    st.subheader("K-Nearest Neighbours")
    knn_ui.run()

elif selected == "NAIVE-BAYES":
    st.subheader("Naive Bayes")
    Naive_Bayes_ui.run()

elif selected == "DECISION-TREE":
    st.subheader("Decision Tree Classifier")
    Decision_Tree_UI.run()

elif selected == "LOGISTIC-REGRESSION":
    st.subheader("Logistic Regression")
    Logistic_Regression_ui.run()

# --- Homepage Content ---
else:
    st.markdown("""
    <div style='text-align: center; white-space: nowrap; overflow: hidden;'>
        <h1>Pattern Recognition and Machine Learning Course Project (CSL2050)</h1>
        <h2>🍎🍌🍇 Fruits Classification 🍎🍌🍇</h2>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; line-height: 1.6;">
        <span><a href="https://github.com/Chandana2609">Jadala Chandana<sup>1</sup></a>&nbsp;&nbsp;</span>
        <span><a href="https://github.com/b23cm1024">Meejuru Lakshmi Sowmya<sup>1</sup></a>&nbsp;&nbsp;</span>
        <span><a href="https://github.com/navyasripenmetsa">Penmetsa Navyasri<sup>1</sup></a>&nbsp;&nbsp;</span>
        <span><a href="https://github.com/harrypotteris">Gattu Charitha<sup>1</sup></a>&nbsp;&nbsp;</span>
        <span><a href="https://github.com/Saichandana-123">Tumma Sai Chandana<sup>1</sup></a></span>
        <br>
        <span style="font-size: 14px;"><sup>1</sup> Indian Institute of Technology Jodhpur</span>
        <br><br>
        <span style="font-size: 16px;">
            | <a href="#">Abstract</a> | <a href="#">Dataset</a> | <a href="#">Techniques Used</a> | <a href="#">Short Talk</a> |
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📝 Abstract")
    st.markdown("""
    This project presents a structured approach to fruit classification using traditional machine learning techniques. 
    Leveraging the well-known Fruits-360 dataset, we focused on extracting meaningful visual features—specifically, Colour Histograms 
    and Histogram of Oriented Gradients (HOG). These features capture crucial aspects like color distribution, shape, and texture from fruit images.

    We trained and evaluated multiple classifiers including K-Nearest Neighbors (KNN), Decision Trees, Naive Bayes, Logistic Regression, 
    and Support Vector Machines (SVM) to assess how well each model performs with different feature sets. Our experiments revealed that 
    feature selection significantly impacts classification accuracy.

    Overall, the project highlights how conventional machine learning methods can effectively address image classification problems 
    in practical scenarios.
    """)

    st.markdown("### 📊 Dataset")
    st.markdown("""
    The dataset used is the Fruits 360 Dataset, a comprehensive collection of labeled, high-resolution images of various fruit categories.
    It contains 141 fruit categories with a total of 94,110 images. Fruits are photographed from multiple angles, enhancing model robustness and generalization.
    """)

    # Load and display sample images
    try:
        apple_img = Image.open(os.path.join(base_dir, "dataset_apple.jpg"))
        banana_img = Image.open(os.path.join(base_dir, "dataset_banana.jpg"))
        cherry_img = Image.open(os.path.join(base_dir, "dataset_cherry.jpg"))
        strawberry_img = Image.open(os.path.join(base_dir, "dataset_strawberry.jpg"))

        col1, col2 = st.columns(2)
        with col1:
            st.image(apple_img, width=200, caption="Apple")
        with col2:
            st.image(banana_img, width=200, caption="Banana")

        col3, col4 = st.columns(2)
        with col3:
            st.image(cherry_img, width=200, caption="Cherry")
        with col4:
            st.image(strawberry_img, width=200, caption="Strawberry")
    except Exception as e:
        st.warning(f"⚠️ Could not load one or more dataset images: {e}")


    st.markdown("### 🤖 Feature Extraction")
    st.markdown("""
- **Colour Histogram**: Captures the distribution of colours in an image, useful for object recognition and image retrieval.
- **HOG (Histogram of Oriented Gradients)**: Describes object shape and appearance by counting occurrences of gradient orientation.
- **Key Features**: Both techniques help in converting image data into numerical features for model training.
""")

    st.markdown("### 🤖 Techniques Used")
    with st.expander("🔹 SVM (Support Vector Machines)"):
        st.write("""
        SVM is a supervised learning algorithm used for classification and regression. It works by finding the optimal hyperplane that separates the data points of different classes with the maximum margin. 
        - **Strengths**: Effective in high-dimensional spaces, robust to overfitting, especially in cases where the number of dimensions exceeds the number of samples.
        - **Kernel Trick**: Allows SVM to perform well with non-linearly separable data by transforming it into higher dimensions using kernel functions (e.g., RBF, polynomial).
        """)
    with st.expander("🔹 KNN(K-Nearest Neighbors)"):
        st.write("""
        KNN is a simple, non-parametric, instance-based learning algorithm. It classifies a new sample based on the majority label of its ‘k’ closest neighbors in the feature space.
        - **Strengths**: Easy to understand and implement, works well with smaller datasets and low-dimensional spaces.
        - **Challenges**: Computationally expensive with large datasets; performance degrades in high dimensions due to the curse of dimensionality.
        """)

    with st.expander("🔹 Naive Bayes"):
        st.write("""
        Naive Bayes is a probabilistic classifier based on Bayes’ theorem, assuming independence between features.
        - **Strengths**: Simple, fast, performs well on text classification tasks like spam detection.
        - **Variants**: Includes Gaussian, Multinomial, and Bernoulli Naive Bayes for different types of data distributions.
        """)
    with st.expander("🔹 Decision Trees"):
        st.write("""
        A Decision Tree is a flowchart-like tree structure where internal nodes represent feature tests, branches represent outcomes, and leaf nodes represent class labels.
        - **Strengths**: Highly interpretable, easy to visualize, handles both numerical and categorical data.
        - **Weakness**: Can overfit if not pruned properly or limited by depth.
        """)
    with st.expander("🔹 Logistic Regression"):
        st.write("""
    Logistic Regression is a statistical model that uses a logistic (sigmoid) function to model the probability of a binary outcome.
    - **Strengths**: Efficient, interpretable, and works well for linearly separable data.
    - **Extension**: Can be extended to multiclass classification using strategies like one-vs-rest.
    """)
        st.divider()

    st.markdown("### 🎤 Short Talk")
    st.video("https://youtu.be/z53GDs0PmNU?si=tJpYUTZ6EQhgib-Y")

    st.markdown("### 👩‍💻 OUR TEAM")

    cols = st.columns(5)
    team = [
        ("saichandhana.jpg", "Sai Chandana", "https://github.com/Saichandana-123"),
        ("chandhana.jpg", "Jadala Chandana", "https://github.com/Chandana2609"),
        ("charitha.jpg", "Charitha", "https://github.com/harrypotteris"),
        ("navya.jpg", "Navya", "https://github.com/navyasripenmetsa"),
        ("sowmya.jpg", "Sowmya", "https://github.com/b23cm1024"), 
    ]

    for col, (img, name, link) in zip(cols, team):
        img_path = os.path.join(base_dir, img)
        if os.path.exists(img_path):
            col.image(img_path, width=120)
        else:
            col.markdown("🚫 Image not found")
        col.markdown(f"**{name}**")
        col.markdown(f"[💻 GitHub]({link})")

    st.divider()
    st.markdown("🔗 **GitHub Link:** [CSL2050 Project Repository](https://github.com/navyasripenmetsa/CSL2050_PRML_Major_Course_Project)")


