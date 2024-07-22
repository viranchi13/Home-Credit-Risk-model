import pandas as pd

data = {
    'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'credit_score': [700, 650, 720, 600, 750, 680, 710, 630, 690, 760],
    'income': [5000, 4500, 6000, 3000, 7000, 4800, 5500, 4000, 5200, 7500],
    'loan_amount': [200000, 180000, 210000, 150000, 250000, 190000, 220000, 170000, 200000, 260000],
    'property_value': [250000, 220000, 270000, 200000, 300000, 230000, 275000, 210000, 240000, 310000],
    'debt': [10000, 12000, 5000, 15000, 8000, 11000, 7000, 13000, 9000, 6000],
    'savings': [15000, 10000, 20000, 5000, 25000, 8000, 18000, 7000, 16000, 30000],
    'monthly_expenses': [2000, 2500, 1800, 2200, 1500, 2400, 2100, 2600, 2000, 1400],
    'crime_rate': [0.02, 0.03, 0.01, 0.04, 0.01, 0.03, 0.02, 0.04, 0.02, 0.01],
    'unemployment_rate': [0.05, 0.06, 0.04, 0.07, 0.03, 0.06, 0.05, 0.07, 0.05, 0.03],
    'financial_health_score': [80, 75, 85, 60, 90, 70, 78, 65, 75, 95],
    'default': [0, 1, 0, 1, 0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)
df.to_csv('home_credit_data.csv', index=False)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('home_credit_data.csv')

# Define feature columns and target column
feature_columns = ['credit_score', 'income', 'loan_amount', 'property_value', 'debt', 'savings', 'monthly_expenses', 'crime_rate', 'unemployment_rate', 'financial_health_score']
target_column = 'default'

# Define numeric features for scaling
numeric_features = ['income', 'loan_amount', 'property_value', 'debt', 'savings', 'monthly_expenses']

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

# Verify the size of the splits
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")
