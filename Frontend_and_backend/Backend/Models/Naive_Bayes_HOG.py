import pandas as pd
import pickle
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
# Load datasets
train_df = pd.read_csv("Datasets/hog_features_train.csv")
test_df = pd.read_csv("Datasets/fruits_hog_features_test.csv")
# Set label column
label_col = "Class"
# Drop rows with missing labels
train_df = train_df.dropna(subset=[label_col])
test_df = test_df.dropna(subset=[label_col])
# Features and labels
X_train = train_df.drop(columns=[label_col])
y_train = train_df[label_col]
X_test = test_df.drop(columns=[label_col])
y_test = test_df[label_col]
# Label encode classes
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
# Normalize features (important for GaussianNB)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Try a few values of var_smoothing manually (start with 1e-9)
model = GaussianNB(var_smoothing=1e-9)
model.fit(X_train_scaled, y_train)
# Predict
y_pred = model.predict(X_test_scaled)
# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc * 100:.2f}%")
# Save model, scaler, and label encoder into Models folder
os.makedirs("Models", exist_ok=True)
with open("Models/nb_hog_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("Models/nb_hog_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("Models/nb_hog_label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
