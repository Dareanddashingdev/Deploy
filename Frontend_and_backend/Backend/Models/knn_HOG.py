import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle
# Load data
train_df = pd.read_csv("Datasets/hog_features_train.csv")
test_df = pd.read_csv("Datasets/fruits_hog_features_test.csv")
X_train = train_df.drop(columns=["Class"])
y_train = train_df["Class"]
X_test = test_df.drop(columns=["Class"])
y_test = test_df["Class"]
# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# PCA
pca = PCA(n_components=0.95)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
# KNN
knn_model = KNeighborsClassifier(n_neighbors=2)
knn_model.fit(X_train_pca, y_train)
accuracy = knn_model.score(X_test_pca, y_test)
print(f"✅ HOG KNN Accuracy (k=2): {accuracy:.4f}")
# Save model
with open("knn_hog.pkl", "wb") as f:
    pickle.dump(knn_model, f)
# Save scaler and PCA
with open("hog_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("hog_pca.pkl", "wb") as f:
    pickle.dump(pca, f)
print("✅ Model + Scaler + PCA saved")
