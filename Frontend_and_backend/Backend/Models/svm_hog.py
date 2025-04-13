import pandas as pd
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm

# Create tqdm progress bar
progress = tqdm(total=10, desc="SVM Training Progress", ncols=100)

# Step 1: Load data
train_df = pd.read_csv('Datasets/hog_features_train.csv')
test_df = pd.read_csv('Datasets/fruits_hog_features_test.csv')
progress.update(1)

# Step 2: Ensure 'Class' column exists
if 'Class' not in train_df.columns or 'Class' not in test_df.columns:
    raise ValueError("Column 'Class' not found in dataset!")
progress.update(1)

# Step 3: Convert 'Class' to string to ensure correct encoding
train_df['Class'] = train_df['Class'].astype(str)
test_df['Class'] = test_df['Class'].astype(str)
progress.update(1)

# Step 4: Filter test set to only include known classes
known_classes = set(train_df['Class'].unique())
test_df = test_df[test_df['Class'].isin(known_classes)]
progress.update(1)

# Step 5: Separate features and labels
X_train = train_df.drop(columns=['Class'])
y_train = train_df['Class']
X_test = test_df.drop(columns=['Class'])
y_test = test_df['Class']
progress.update(1)

# Step 6: Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
progress.update(1)

# Step 7: Impute missing values
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
progress.update(1)

# Step 8: Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)
progress.update(1)

# Step 9: Train and evaluate SVM
svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train_scaled, y_train_encoded)
y_pred = svm_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred, average='weighted')
recall = recall_score(y_test_encoded, y_pred, average='weighted')
f1 = f1_score(y_test_encoded, y_pred, average='weighted')
progress.update(1)

# Step 10: Save the model, scaler, and label encoder
joblib.dump(svm_model, 'svm_hog_model.pkl')
joblib.dump(scaler, 'svm_hog_scaler.pkl')
joblib.dump(label_encoder, 'svm_label_encoder.pkl')
progress.update(1)
progress.close()
# Print evaluation
print("\n✅ SVM Evaluation Metrics:")
print(f'Accuracy:  {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall:    {recall:.4f}')
print(f'F1-score:  {f1:.4f}')
print("✅ Model, scaler, and label encoder saved as .pkl files.")

