import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# === Load Dataset ===
train_df = pd.read_csv('Datasets/colour_Histogram_Training.csv')
test_df = pd.read_csv('Datasets/colour_Histogram_Testing.csv')

X_train = train_df.drop(columns=['filename', 'class'])
y_train = train_df['class']
X_test = test_df.drop(columns=['filename', 'class'])
y_test = test_df['class']

# === Encode Labels ===
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# === Standardize Features ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train SVM Model ===
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_scaled, y_train_encoded)

# === Predict and Evaluate ===
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')
f1 = f1_score(y_test_encoded, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')

# === Save Models ===
os.makedirs("Models", exist_ok=True)
joblib.dump(svm_model, "Models/svm_color_hist.pkl")
joblib.dump(scaler, "Models/svm_color_scaler.pkl")
joblib.dump(label_encoder, "Models/svm_label_encoder.pkl")

print("✅ Models saved successfully!")


