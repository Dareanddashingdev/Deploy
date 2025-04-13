import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
import pickle
import os
# Load datasets
train_df = pd.read_csv('Datasets/colour_Histogram_Training.csv')
test_df = pd.read_csv('Datasets/colour_Histogram_Testing.csv')
# Prepare features and target
X_train = train_df.drop(columns=['filename', 'class'])
y_train = train_df['class']
X_test = test_df.drop(columns=['filename', 'class'])
y_test = test_df['class']
# Encode target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Generate polynomial features (interaction-only)
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)
# Train logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_poly, y_train_encoded)
# Predict and evaluate
y_pred_poly = model.predict(X_test_poly)
accuracy_poly = accuracy_score(y_test_encoded, y_pred_poly)
print("Accuracy with Polynomial Features:", accuracy_poly)
# --- Save Model & Preprocessors ---
os.makedirs("Models", exist_ok=True)
with open("Models/logreg_poly_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("Models/logreg_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open("Models/logreg_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("Models/logreg_poly_features.pkl", "wb") as f:
    pickle.dump(poly, f)

     