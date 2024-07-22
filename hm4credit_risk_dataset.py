import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import joblib
import datetime

# Load data
data = pd.read_csv('credit_risk_dataset.csv')

# Handle missing values
data = data.ffill()  # Forward fill missing values

# Check for and handle outliers (example using Z-score)
numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
                    'loan_percent_income', 'cb_person_cred_hist_length']
data = data[(np.abs(stats.zscore(data[numeric_features])) < 3).all(axis=1)]

# Define the initial lists of features
scalable_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
                     'loan_percent_income', 'cb_person_cred_hist_length', 'LTV_ratio', 'DTI_ratio',
                     'savings_rate', 'monthly_expense', 'location_risk', 'financial_health']

categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']

# Feature Engineering (Create columns only if required columns exist)
if 'loan_amnt' in data.columns and 'property_value' in data.columns:
    data['LTV_ratio'] = data['loan_amnt'] / data['property_value']
if 'debt' in data.columns and 'person_income' in data.columns:
    data['DTI_ratio'] = data['debt'] / data['person_income']
if 'savings' in data.columns and 'person_income' in data.columns:
    data['savings_rate'] = data['savings'] / data['person_income']
if 'monthly_expenses' in data.columns and 'person_income' in data.columns:
    data['monthly_expense'] = data['monthly_expenses'] / data['person_income']
if 'crime_rate' in data.columns and 'unemployment_rate' in data.columns:
    data['location_risk'] = data['crime_rate'] + data['unemployment_rate']
if 'financial_health_score' in data.columns:
    data['financial_health'] = data['financial_health_score']

# Function to filter out missing columns
def filter_missing_columns(features, df):
    return [feature for feature in features if feature in df.columns]

# Filter the features lists
scalable_features = filter_missing_columns(scalable_features, data)
categorical_features = filter_missing_columns(categorical_features, data)

# Check if 'loan_status' column exists
if 'loan_status' in data.columns:
    # Encode 'loan_status' to binary values: 1 for 'default' and 0 for 'paid'
    data['loan_status'] = data['loan_status'].apply(lambda x: 1 if x == 'default' else 0)
    X = data.drop(['loan_status'], axis=1)
    y = data['loan_status']
else:
    raise KeyError("'loan_status' column not found in the dataset")

# Preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, scalable_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply preprocessing
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

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
    real_time_data = preprocessor.transform(real_time_data)

    # Predict
    predictions = loaded_model.predict(real_time_data)

    # Log performance
    accuracy = accuracy_score(real_time_data['loan_status'], predictions)
    print(f'{datetime.datetime.now()} - Accuracy: {accuracy}')

# Example feedback incorporation and model update
def update_model_with_feedback(new_data):
    # Preprocess new data
    new_data = preprocessor.transform(new_data)

    # Update the model
    X_new = new_data.drop(['loan_status'], axis=1)
    y_new = new_data['loan_status']
    xgb_model.fit(X_new, y_new, xgb_model.get_params())

    # Save the updated model
    joblib.dump(xgb_model, 'xgb_model_updated.pkl')
