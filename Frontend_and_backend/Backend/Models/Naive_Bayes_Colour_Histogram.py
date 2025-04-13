# Imports
import pandas as pd
import pickle
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Load data - only histogram files now
hist_train = pd.read_csv('Datasets/colour_Histogram_Training.csv')
hist_test = pd.read_csv('Datasets/colour_Histogram_Testing.csv')
# Separate features and target (no merging with metadata needed)
X_train = hist_train.drop(columns=['filename', 'class'])  # Keep only histogram features
y_train = hist_train['class']
X_test = hist_test.drop(columns=['filename', 'class'])
y_test = hist_test['class']
# Encode target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Naive Bayes model
model = GaussianNB()
model.fit(X_train_scaled, y_train_encoded)
# Predictions
y_pred = model.predict(X_test_scaled)
# Evaluation
print("Accuracy:", accuracy_score(y_test_encoded, y_pred))
print("\nClassification Report:\n", classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_encoded, y_pred))
# Ensure Models directory exists
os.makedirs("Models", exist_ok=True)
# Save model, scaler, and label encoder in Models folder
with open('Models/nb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('Models/nb_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('Models/nb_label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


     