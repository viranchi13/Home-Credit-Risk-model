import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    roc_curve
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Credit Risk Modeling GUI")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data.head())

    # Data Preprocessing Options
    if st.checkbox("Handle Missing Values"):
        data = data.ffill()
        data = data.dropna(subset=['loan_status'])

    if st.checkbox("Handle Outliers"):
        numeric_features = ['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
                            'loan_percent_income', 'cb_person_cred_hist_length']
        data = data[(np.abs(stats.zscore(data[numeric_features])) < 3).all(axis=1)]

    # Feature Engineering Options
    if st.checkbox("Add Feature: LTV Ratio") and 'loan_amnt' in data.columns and 'property_value' in data.columns:
        data['LTV_ratio'] = data['loan_amnt'] / data['property_value']

    if st.checkbox("Add Feature: DTI Ratio") and 'debt' in data.columns and 'person_income' in data.columns:
        data['DTI_ratio'] = data['debt'] / data['person_income']

    if st.checkbox("Add Feature: Savings Rate") and 'savings' in data.columns and 'person_income' in data.columns:
        data['savings_rate'] = data['savings'] / data['person_income']

    if st.checkbox(
            "Add Feature: Monthly Expense") and 'monthly_expenses' in data.columns and 'person_income' in data.columns:
        data['monthly_expense'] = data['monthly_expenses'] / data['person_income']

    if st.checkbox(
            "Add Feature: Location Risk") and 'crime_rate' in data.columns and 'unemployment_rate' in data.columns:
        data['location_risk'] = data['crime_rate'] + data['unemployment_rate']

    if st.checkbox("Add Feature: Financial Health") and 'financial_health_score' in data.columns:
        data['financial_health'] = data['financial_health_score']

    # Encoding categorical variables
    label_encoder = LabelEncoder()
    data['person_home_ownership'] = label_encoder.fit_transform(data['person_home_ownership'])
    data['loan_intent'] = label_encoder.fit_transform(data['loan_intent'])
    data['loan_grade'] = label_encoder.fit_transform(data['loan_grade'])
    data['cb_person_default_on_file'] = label_encoder.fit_transform(data['cb_person_default_on_file'])

    # Separate features and target
    X = data.drop('loan_status', axis=1)
    y = data['loan_status']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Preprocessing pipelines
    scalable_features = list(
        set(X.columns) & set(['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate',
                              'loan_percent_income', 'cb_person_cred_hist_length', 'LTV_ratio', 'DTI_ratio',
                              'savings_rate', 'monthly_expense', 'location_risk', 'financial_health']))
    categorical_features = list(
        set(X.columns) & set(['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']))

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

    # Apply preprocessing
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    # Model Training
    if st.button("Train Models"):
        log_reg = LogisticRegression()
        log_reg.fit(X_train, y_train)

        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)

        xgb_model = xgb.XGBClassifier()
        xgb_model.fit(X_train, y_train)

        # Meta-model
        train_preds_large = np.column_stack([
            log_reg.predict(X_train),
            rf.predict(X_train),
            xgb_model.predict(X_train)
        ])

        test_preds_large = np.column_stack([
            log_reg.predict(X_test),
            rf.predict(X_test),
            xgb_model.predict(X_test)
        ])

        small_model = joblib.load(r'path_to_your_file\xgb_model.pkl')
        X_train_small = X_train[:, :small_model.n_features_in_]
        X_test_small = X_test[:, :small_model.n_features_in_]

        train_preds_small = small_model.predict(X_train_small).reshape(-1, 1)
        test_preds_small = small_model.predict(X_test_small).reshape(-1, 1)

        train_preds_combined = np.hstack([train_preds_large, train_preds_small])
        test_preds_combined = np.hstack([test_preds_large, test_preds_small])

        meta_model = LogisticRegression()
        meta_model.fit(train_preds_combined, y_train)

        y_pred_meta = meta_model.predict(test_preds_combined)

        st.write(f'Meta-model Accuracy: {accuracy_score(y_test, y_pred_meta)}')
        st.write(f'Meta-model Precision: {precision_score(y_test, y_pred_meta)}')
        st.write(f'Meta-model Recall: {recall_score(y_test, y_pred_meta)}')
        st.write(f'Meta-model F1 Score: {f1_score(y_test, y_pred_meta)}')
        st.write(f'Meta-model ROC-AUC: {roc_auc_score(y_test, y_pred_meta)}')

        # Plotting confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_meta)
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix - Meta Model')
        st.pyplot(fig)

        # Plotting ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_meta)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='orange', label='ROC curve')
        ax.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve - Meta Model')
        ax.legend()
        st.pyplot(fig)

        # Save the meta-model
        joblib.dump(meta_model, 'meta_model.pkl')
        st.write("Meta-model saved successfully.")

    # Load the meta-model
    if st.button("Load Meta-model"):
        loaded_meta_model = joblib.load('meta_model.pkl')
        st.write("Meta-model loaded successfully.")
