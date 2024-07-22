import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, classification_report

# Load data
data = pd.read_csv('home_credit_data.csv')

# Define feature columns and target column
feature_columns = ['credit_score', 'income', 'loan_amount', 'property_value', 'debt', 'savings', 'monthly_expenses', 'crime_rate', 'unemployment_rate', 'financial_health_score']
target_column = 'default'

# Define numeric features for scaling
numeric_features = ['income', 'loan_amount', 'property_value', 'debt', 'savings', 'monthly_expenses']

# Handle missing values
data.ffill(inplace=True)  # Use ffill directly

# Split data into features (X) and target (y)
X = data[feature_columns]
y = data[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and testing data
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Check the class distribution
print("Class distribution in y_train before resampling:")
print(y_train.value_counts())
print("\nClass distribution in y_test before resampling:")
print(y_test.value_counts())

# Find the minimum number of samples in any class in y_train
min_samples = y_train.value_counts().min()

# Initialize SMOTE with adjusted k_neighbors
smote = SMOTE(sampling_strategy='auto', k_neighbors=min(min_samples-1, 5), random_state=42)

# Resample the training set
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Verify the new class distribution
print("\nClass distribution in y_train after resampling:")
print(y_train_res.value_counts())

# Train an XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train_res, y_train_res)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Evaluate the model using various metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, zero_division=1)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"ROC AUC: {roc_auc}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))
