import cv2
import numpy as np
import joblib
import os
from .utils.colour_histogram import extract_color_histogram_from_image
from .utils.hog_features import extract_hog_features_from_image

# === MODEL PATH SETUP ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# KNN Models
color_knn_path = os.path.join(BASE_DIR, "Models", "knn_color_hist.pkl")
hog_knn_path = os.path.join(BASE_DIR, "Models", "knn_hog.pkl")
hog_knn_scaler_path = os.path.join(BASE_DIR, "Models", "knn_hog_scaler.pkl")
hog_knn_pca_path = os.path.join(BASE_DIR, "Models", "knn_hog_pca.pkl")

# SVM Models
svm_color_model_path = os.path.join(BASE_DIR, "Models", "svm_color_hist.pkl")
svm_color_scaler_path = os.path.join(BASE_DIR, "Models", "svm_color_scaler.pkl")
svm_color_encoder_path = os.path.join(BASE_DIR, "Models", "svm_label_encoder.pkl")

svm_hog_model_path = os.path.join(BASE_DIR, "Models", "svm_hog_model.pkl")
svm_hog_scaler_path = os.path.join(BASE_DIR, "Models", "svm_hog_scaler.pkl")
svm_hog_encoder_path = os.path.join(BASE_DIR, "Models", "svm_hog_label_encoder.pkl")

# Logistic Regression Models
logreg_poly_path = os.path.join(BASE_DIR, "Models", "logreg_poly_features.pkl")
logreg_scaler_path = os.path.join(BASE_DIR, "Models", "logreg_scaler.pkl")
logreg_model_path = os.path.join(BASE_DIR, "Models", "logreg_poly_model.pkl")
logreg_label_encoder_path = os.path.join(BASE_DIR, "Models", "logreg_label_encoder.pkl")

lr_hog_scaler_path = os.path.join(BASE_DIR, "Models", "lr_hog_scaler.pkl")
lr_hog_model_path = os.path.join(BASE_DIR, "Models", "lr_hog_model.pkl")
lr_hog_label_encoder_path = os.path.join(BASE_DIR, "Models", "lr_hog_label_encoder.pkl")

# Naive Bayes Models
nb_model_path = os.path.join(BASE_DIR, "Models", "nb_model.pkl")
nb_scaler_path = os.path.join(BASE_DIR, "Models", "nb_scaler.pkl")
nb_label_encoder_path = os.path.join(BASE_DIR, "Models", "nb_label_encoder.pkl")

nb_hog_model_path = os.path.join(BASE_DIR, "Models", "nb_hog_model.pkl")
nb_hog_scaler_path = os.path.join(BASE_DIR, "Models", "nb_hog_scaler.pkl")
nb_hog_label_encoder_path = os.path.join(BASE_DIR, "Models", "nb_hog_label_encoder.pkl")

# Decision Tree Models
dt_color_model_path = os.path.join(BASE_DIR, "Models", "decision_tree_color.pkl")
dt_color_encoder_path = os.path.join(BASE_DIR, "Models", "dt_color_label_encoder.pkl")

dt_hog_model_path = os.path.join(BASE_DIR, "Models", "hog_decision_tree.pkl")
dt_hog_imputer_path = os.path.join(BASE_DIR, "Models", "hog_imputer.pkl")
dt_hog_encoder_path = os.path.join(BASE_DIR, "Models", "hog_label_encoder.pkl")

# === LOAD MODELS ===
# KNN
color_knn_model = joblib.load(color_knn_path)
hog_knn_model = joblib.load(hog_knn_path)
hog_knn_scaler = joblib.load(hog_knn_scaler_path)
hog_knn_pca = joblib.load(hog_knn_pca_path)

# SVM
svm_color_model = joblib.load(svm_color_model_path)
svm_color_scaler = joblib.load(svm_color_scaler_path)
svm_color_encoder = joblib.load(svm_color_encoder_path)

svm_hog_model = joblib.load(svm_hog_model_path)
svm_hog_scaler = joblib.load(svm_hog_scaler_path)
svm_hog_encoder = joblib.load(svm_hog_encoder_path)

# Logistic Regression
logreg_poly = joblib.load(logreg_poly_path)
logreg_scaler = joblib.load(logreg_scaler_path)
logreg_model = joblib.load(logreg_model_path)
logreg_encoder = joblib.load(logreg_label_encoder_path)

lr_hog_scaler = joblib.load(lr_hog_scaler_path)
lr_hog_model = joblib.load(lr_hog_model_path)
lr_hog_encoder = joblib.load(lr_hog_label_encoder_path)

# Naive Bayes
nb_model = joblib.load(nb_model_path)
nb_scaler = joblib.load(nb_scaler_path)
nb_encoder = joblib.load(nb_label_encoder_path)

nb_hog_model = joblib.load(nb_hog_model_path)
nb_hog_scaler = joblib.load(nb_hog_scaler_path)
nb_hog_encoder = joblib.load(nb_hog_label_encoder_path)

# Decision Tree
dt_color_model = joblib.load(dt_color_model_path)
dt_color_encoder = joblib.load(dt_color_encoder_path)

dt_hog_model = joblib.load(dt_hog_model_path)
dt_hog_imputer = joblib.load(dt_hog_imputer_path)
dt_hog_encoder = joblib.load(dt_hog_encoder_path)

# === FEATURE EXTRACTION ===
def extract_features_from_uploaded_file(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Uploaded image could not be decoded. Please check the image format or re-upload.")

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    color_features = extract_color_histogram_from_image(rgb_image).reshape(1, -1)
    hog_features = extract_hog_features_from_image(image).reshape(1, -1)

    print(f"[INFO] Extracted {color_features.shape[1]} color histogram features.")
    print(f"[INFO] Extracted {hog_features.shape[1]} HOG features.")

    return color_features, hog_features

# === PREDICTION ===
def predict_class(uploaded_file):
    color_features, hog_features = extract_features_from_uploaded_file(uploaded_file)

    # --- KNN ---
    knn_color_pred = color_knn_model.predict(color_features)[0]
    hog_scaled_knn = hog_knn_scaler.transform(hog_features)
    hog_pca_features = hog_knn_pca.transform(hog_scaled_knn)
    knn_hog_pred = hog_knn_model.predict(hog_pca_features)[0]

    # --- SVM ---
    color_scaled = svm_color_scaler.transform(color_features)
    svm_color_pred_encoded = svm_color_model.predict(color_scaled)[0]
    svm_color_pred = svm_color_encoder.inverse_transform([svm_color_pred_encoded])[0]

    hog_scaled = svm_hog_scaler.transform(hog_features)
    svm_hog_pred_encoded = svm_hog_model.predict(hog_scaled)[0]
    svm_hog_pred = svm_hog_encoder.inverse_transform([svm_hog_pred_encoded])[0]

    # --- Logistic Regression ---
    color_poly = logreg_poly.transform(color_features)
    color_scaled_logreg = logreg_scaler.transform(color_poly)
    logreg_pred_encoded = logreg_model.predict(color_scaled_logreg)[0]
    logreg_color_pred = logreg_encoder.inverse_transform([logreg_pred_encoded])[0]

    hog_scaled_lr = lr_hog_scaler.transform(hog_features)
    logreg_hog_pred_encoded = lr_hog_model.predict(hog_scaled_lr)[0]
    logreg_hog_pred = lr_hog_encoder.inverse_transform([logreg_hog_pred_encoded])[0]

    # --- Naive Bayes ---
    nb_color_scaled = nb_scaler.transform(color_features)
    nb_color_encoded = nb_model.predict(nb_color_scaled)[0]
    nb_color_pred = nb_encoder.inverse_transform([nb_color_encoded])[0]

    nb_hog_scaled = nb_hog_scaler.transform(hog_features)
    nb_hog_encoded = nb_hog_model.predict(nb_hog_scaled)[0]
    nb_hog_pred = nb_hog_encoder.inverse_transform([nb_hog_encoded])[0]

    # --- Decision Tree ---
    dt_color_encoded = dt_color_model.predict(color_features)[0]
    dt_color_pred = dt_color_encoder.inverse_transform([dt_color_encoded])[0]

    hog_imputed = dt_hog_imputer.transform(hog_features)
    dt_hog_encoded = dt_hog_model.predict(hog_imputed)[0]
    dt_hog_pred = dt_hog_encoder.inverse_transform([dt_hog_encoded])[0]

    return {
        "knn": {
            "color": knn_color_pred,
            "hog": knn_hog_pred
        },
        "svm": {
            "color": svm_color_pred,
            "hog": svm_hog_pred
        },
        "logistic_regression": {
            "color": logreg_color_pred,
            "hog": logreg_hog_pred
        },
        "naive_bayes": {
            "color": nb_color_pred,
            "hog": nb_hog_pred
        },
        "decision_tree": {
            "color": dt_color_pred,
            "hog": dt_hog_pred
        }
    }











