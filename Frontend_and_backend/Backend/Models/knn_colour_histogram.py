import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pickle
# Load data
train_df = pd.read_csv("Datasets/colour_Histogram_Training.csv")
test_df = pd.read_csv("Datasets/colour_Histogram_Testing.csv")
# Split features and labels
X_train = train_df.drop(columns=["filename", "class"])
y_train = train_df["class"]
X_test = test_df.drop(columns=["filename", "class"])
y_test = test_df["class"]
# Train KNN (k=2)
knn_model = KNeighborsClassifier(n_neighbors=2)
knn_model.fit(X_train, y_train)
# Accuracy for clarity
accuracy = knn_model.score(X_test, y_test)
print(f"✅ Color Histogram KNN Accuracy (k=2): {accuracy:.4f}")
# Save as .pkl
with open("knn_color_hist.pkl", "wb") as f:
    pickle.dump(knn_model, f)
print("✅ Model saved as knn_color_hist.pkl")
