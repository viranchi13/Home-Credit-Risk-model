# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
# Assuming the Lending Club dataset is stored in a CSV file named 'lending_club_data.csv'
df = pd.read_csv('C:/Users/viran/PycharmProjects/Credit risk/pythonProject3/lending_club_data.csv')


# Display the first few rows of the dataset
print(df.head())

# Preprocessing the data
# Removing any rows with missing values
df.dropna(inplace=True)

# Feature selection
# Assuming 'loan_status' is the target variable and all other columns are features
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Encoding categorical features
# Assuming there are categorical features that need to be encoded
X = pd.get_dummies(X, drop_first=True)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Optional: Displaying the coefficients of the logistic regression model
coefficients = pd.DataFrame(model.coef_, columns=X.columns)
print('Coefficients:')
print(coefficients)
