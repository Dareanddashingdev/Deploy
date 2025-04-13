import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Ensure 'Models' folder exists
os.makedirs("Models", exist_ok=True)
# 1. LOAD THE DATASETS
train_data = pd.read_csv("Datasets/hog_features_train.csv")
test_data = pd.read_csv("Datasets/fruits_hog_features_test.csv")
# 2. SEPARATE FEATURES AND LABELS
X_train = train_data.drop(columns=['Class'])
y_train = train_data['Class']
X_test = test_data.drop(columns=['Class'])
y_test = test_data['Class']
# 3. ENCODE STRING LABELS
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# Save label encoder
with open("Models/lr_hog_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
# 4. FEATURE SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Save scaler
with open("Models/lr_hog_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
# 5. TRAIN THE LOGISTIC REGRESSION MODEL
model = LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced', C=0.1)
model.fit(X_train_scaled, y_train_encoded)
# Save trained model
with open("Models/lr_hog_model.pkl", "wb") as f:
    pickle.dump(model, f)
# 6. MAKE PREDICTIONS AND EVALUATE
y_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test_encoded, y_test_pred)
print(f"✅ Test Accuracy: {test_accuracy:.4f}")
print("✅ Files saved:")
print("- Models/lr_hog_model.pkl")
print("- Models/lr_hog_scaler.pkl")
print("- Models/lr_hog_label_encoder.pkl")


