import pandas as pd
import numpy as np

# Create a dictionary with sample data
data = {
    'loan_amnt': np.random.randint(1000, 40000, 1000),
    'term': np.random.choice(['36 months', '60 months'], 1000),
    'int_rate': np.round(np.random.uniform(5.0, 25.0, 1000), 2),
    'installment': np.round(np.random.uniform(50, 1000, 1000), 2),
    'grade': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], 1000),
    'emp_length': np.random.choice(['< 1 year', '1 year', '2 years', '3 years', '4 years', '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'], 1000),
    'home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], 1000),
    'annual_inc': np.round(np.random.uniform(20000, 200000, 1000), 2),
    'loan_status': np.random.choice(['Fully Paid', 'Charged Off', 'Default'], 1000),
    'purpose': np.random.choice(['credit_card', 'debt_consolidation', 'home_improvement', 'major_purchase', 'small_business'], 1000),
    'dti': np.round(np.random.uniform(0.0, 30.0, 1000), 2),
    'delinq_2yrs': np.random.randint(0, 10, 1000),
    'fico_range_low': np.random.randint(600, 850, 1000),
    'fico_range_high': np.random.randint(600, 850, 1000),
    'inq_last_6mths': np.random.randint(0, 10, 1000),
    'revol_bal': np.random.randint(0, 50000, 1000),
    'revol_util': np.round(np.random.uniform(0.0, 100.0, 1000), 2),
    'total_acc': np.random.randint(5, 50, 1000),
    'longest_credit_length': np.random.randint(1, 50, 1000)
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv('lending_club_data.csv', index=False)

print("Sample Lending Club dataset generated and saved to 'lending_club_data.csv'.")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('home_credit_data.csv')

# Define feature columns and target column
feature_columns = ['credit_score', 'income', 'loan_amount', 'property_value', 'debt', 'savings', 'monthly_expenses', 'crime_rate', 'unemployment_rate', 'financial_health_score']
target_column = 'default'

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
