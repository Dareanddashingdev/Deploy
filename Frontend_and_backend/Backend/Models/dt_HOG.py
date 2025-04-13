import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from tqdm import tqdm
import time
import pickle
import os

# === Load Data ===
train_data = pd.read_csv('Datasets/hog_features_train.csv')
test_data = pd.read_csv('Datasets/fruits_hog_features_test.csv')

# === Separate features and labels ===
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# === Filter test labels not present in training set ===
mask = y_test.isin(y_train.unique())
X_test = X_test[mask]
y_test = y_test[mask]

# === Encode class labels ===
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# === Handle missing values with imputer ===
imputer = SimpleImputer(strategy='mean')
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# === Simulate progress bar for training ===
print("Training Decision Tree Classifier...")
for _ in tqdm(range(100), desc="Training Progress", ncols=75):
    time.sleep(0.005)

# === Train the Decision Tree Classifier ===
best_model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42
)
best_model.fit(X_train_imp, y_train_enc)

# === Predict and Evaluate ===
y_pred = best_model.predict(X_test_imp)
accuracy = accuracy_score(y_test_enc, y_pred)
print(f"\nBest Decision Tree Accuracy: {accuracy:.4f}")

# === Save the model, label encoder, and imputer ===
os.makedirs("Models", exist_ok=True)

with open("Models/hog_decision_tree.pkl", "wb") as f_model:
    pickle.dump(best_model, f_model)

with open("Models/hog_label_encoder.pkl", "wb") as f_encoder:
    pickle.dump(le, f_encoder)

with open("Models/hog_imputer.pkl", "wb") as f_imputer:
    pickle.dump(imputer, f_imputer)

print("\nModel, Label Encoder, and Imputer saved in 'Models/' directory.")

