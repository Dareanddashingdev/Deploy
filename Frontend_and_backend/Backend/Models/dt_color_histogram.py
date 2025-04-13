import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# === Paths ===
train_path = "Datasets/colour_Histogram_Training.csv"
test_path = "Datasets/colour_Histogram_Testing.csv"

# === Output model paths ===
model_dir = "Models"
os.makedirs(model_dir, exist_ok=True)
dt_model_path = os.path.join(model_dir, "decision_tree_color.pkl")
dt_label_encoder_path = os.path.join(model_dir, "dt_color_label_encoder.pkl")

# === Load Data ===
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

# Drop 'filename' column
df_train = df_train.drop(columns=["filename"])
df_test = df_test.drop(columns=["filename"])

# Encode class labels
encoder = LabelEncoder()
df_train["class"] = encoder.fit_transform(df_train["class"])
df_test["class"] = encoder.transform(df_test["class"])

# Split features and labels
X_train = df_train.drop(columns=["class"])
y_train = df_train["class"]
X_test = df_test.drop(columns=["class"])
y_test = df_test["class"]

# Train Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Predict & Accuracy
y_pred = dt_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# === Save model and encoder ===
with open(dt_model_path, "wb") as model_file:
    pickle.dump(dt_model, model_file)
with open(dt_label_encoder_path, "wb") as encoder_file:
    pickle.dump(encoder, encoder_file)

print("[INFO] Model and encoder saved successfully.")



