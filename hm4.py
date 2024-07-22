import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import datetime

# Load data
data = pd.read_csv('home_credit_data.csv')

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Normalize/standardize data
scaler = StandardScaler()
numeric_features = ['income', 'loan_amount', 'property_value']
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Check for and handle outliers (example using Z-score)
data = data[(np.abs(stats.zscore(data[numeric_features])) < 3).all(axis=1)]

# Traditional features
data['LTV_ratio'] = data['loan_amount'] / data['property_value']
data['DTI_ratio'] = data['debt'] / data['income']

# Behavioral features
data['savings_rate'] = data['savings'] / data['income']
data['monthly_expense'] = data['monthly_expenses'] / data['income']

# Geospatial features (assuming geospatial data is already merged)
data['location_risk'] = data['crime_rate'] + data['unemployment_rate']

# Psychometric features (assuming psychometric data is already available)
# data['risk_tolerance'], data['financial_literacy'] should be precomputed

# Dynamic features
# Assume real-time financial health indicators are part of the dataset
data['financial_health'] = data['financial_health_score']

# Split data into train and test sets
X = data.drop(['default'], axis=1)
y = data['default']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# XGBoost
xgb_model = xgb.XGBClassifier()
xgb_model.fit(X_train, y_train)

# Model evaluation
models = [log_reg, rf, xgb_model]
model_names = ['Logistic Regression', 'Random Forest', 'XGBoost']

for model, name in zip(models, model_names):
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    if len(set(y_train)) > 1:
        auc = roc_auc_score(y_train, y_pred_train)
    else:
        print("ROC AUC score cannot be computed due to only one class present in y_true.")
    print(f'{name} Accuracy: {accuracy_score(y_test, y_pred)}')
    print(f'{name} Precision: {precision_score(y_test, y_pred)}')
    print(f'{name} Recall: {recall_score(y_test, y_pred)}')
    print(f'{name} F1 Score: {f1_score(y_test, y_pred)}')
    print(f'{name} ROC-AUC: {roc_auc_score(y_test, y_pred)}')
    print('-' * 30)

# Example of backtesting with cross-validation
cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='roc_auc')
print(f'Cross-validated ROC-AUC scores: {cv_scores}')
print(f'Mean ROC-AUC: {np.mean(cv_scores)}')

# Save the model
joblib.dump(xgb_model, 'xgb_model.pkl')

# Load the model (in production)
loaded_model = joblib.load('xgb_model.pkl')

# Example monitoring script
def monitor_model_performance():
    # Load real-time data
    real_time_data = pd.read_csv('real_time_data.csv')
    real_time_data[numeric_features] = scaler.transform(real_time_data[numeric_features])

    # Predict
    predictions = loaded_model.predict(real_time_data.drop(['default'], axis=1))

    # Log performance
    accuracy = accuracy_score(real_time_data['default'], predictions)
    print(f'{datetime.datetime.now()} - Accuracy: {accuracy}')

# Schedule this function to run periodically

# Example feedback incorporation and model update
def update_model_with_feedback(new_data):
    # Preprocess new data
    new_data[numeric_features] = scaler.transform(new_data[numeric_features])

    # Update the model
    X_new = new_data.drop(['default'], axis=1)
    y_new = new_data['default']
    xgb_model.fit(X_new, y_new, xgb_model.get_params())

    # Save the updated model
    joblib.dump(xgb_model, 'xgb_model_updated.pkl')
