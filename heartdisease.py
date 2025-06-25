import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from xgboost import XGBClassifier

# Load and preprocess dataset
@st.cache_data
def load_models():
    doc = pd.read_csv('C:/Users/ASUS/OneDrive/Desktop/Flask/heart.csv')
    original_data = doc.copy()

    # One-hot encoding
    doc_encoded = pd.get_dummies(doc)
    X = doc_encoded.drop('HeartDisease', axis=1)
    y = doc_encoded['HeartDisease']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=150),
        'SVM': SVC(kernel='linear', probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150, max_depth=4, learning_rate=0.1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=150),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150)
    }

    trained_models = {}
    accuracy_scores = {}

    for name, clf in models.items():
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        trained_models[name] = clf
        accuracy_scores[name] = {'Accuracy': acc, 'Recall': rec, 'F1-Score': f1}

    # Hard Voting Ensemble
    hard_ensemble = VotingClassifier(
        estimators=[(name, trained_models[name]) for name in models.keys()],
        voting='hard'
    )
    hard_ensemble.fit(X_train_scaled, y_train)
    y_pred_hard = hard_ensemble.predict(X_test_scaled)
    accuracy_scores['Hard Ensemble'] = {
        'Accuracy': accuracy_score(y_test, y_pred_hard),
        'Recall': recall_score(y_test, y_pred_hard),
        'F1-Score': f1_score(y_test, y_pred_hard)
    }
    trained_models['Hard Ensemble'] = hard_ensemble

    # Soft Voting Ensemble
    soft_ensemble = VotingClassifier(
        estimators=[(name, trained_models[name]) for name in ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost']],
        voting='soft'
    )
    soft_ensemble.fit(X_train_scaled, y_train)
    y_pred_soft = soft_ensemble.predict(X_test_scaled)
    accuracy_scores['Soft Ensemble'] = {
        'Accuracy': accuracy_score(y_test, y_pred_soft),
        'Recall': recall_score(y_test, y_pred_soft),
        'F1-Score': f1_score(y_test, y_pred_soft)
    }
    trained_models['Soft Ensemble'] = soft_ensemble

    return trained_models, scaler, original_data, accuracy_scores, X.columns

# Load models and metadata
models, scaler, raw_data, accuracy_scores, feature_columns = load_models()

# UI setup
st.title(" Heart Disease Prediction App")
st.subheader("Enter Patient Details")

# Collect user input
input_dict = {}
for col in raw_data.columns:
    if col == 'HeartDisease':
        continue
    if raw_data[col].dtype == 'object':
        input_dict[col] = st.selectbox(f"{col}", raw_data[col].unique())
    else:
        input_dict[col] = st.number_input(f"{col}", value=float(raw_data[col].mean()))

# --- Feature Importance Section ---
st.subheader(" Feature Importance / Coefficients")

for model_name, clf in models.items():
    st.markdown(f"**{model_name}**")
    try:
        if model_name in ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'XGBoost']:
            importance = clf.feature_importances_
            imp_df = pd.DataFrame({'Feature': feature_columns, 'Importance': importance})
            imp_df = imp_df.sort_values(by='Importance', ascending=False)
            st.dataframe(imp_df)

        elif model_name in ['Logistic Regression', 'SVM']:
            if hasattr(clf, 'coef_'):
                importance = np.abs(clf.coef_[0])
                imp_df = pd.DataFrame({'Feature': feature_columns, 'Coefficient Magnitude': importance})
                imp_df = imp_df.sort_values(by='Coefficient Magnitude', ascending=False)
                st.dataframe(imp_df)
            else:
                st.warning("Coefficients not available (non-linear kernel).")

        elif model_name in ['Hard Ensemble', 'Soft Ensemble']:
            st.info("Feature importance not available for ensemble models.")

    except Exception as e:
        st.error(f"Error extracting importance for {model_name}: {str(e)}")

# --- Prediction Section ---
if st.button("Predict"):
    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)

    # Align input with training feature columns
    for col in feature_columns:
        if col not in input_encoded:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_columns]

    input_scaled = scaler.transform(input_encoded)

    st.subheader(" Prediction Results")
    for model_name, clf in models.items():
        prediction = clf.predict(input_scaled)[0]
        if prediction == 1:
            st.error(f"{model_name}: Likely to have heart disease")
        else:
            st.success(f"{model_name}: Not likely to have heart disease")

    # Metrics table
    st.subheader(" Model Performance Comparison")
    metrics_df = pd.DataFrame(accuracy_scores).T  # Transpose to get metrics in rows
    metrics_df = metrics_df.sort_values(by='Accuracy', ascending=False)
    # Convert all metric values to percentages
    metrics_df_percent = metrics_df.copy() * 100

# Format and display with two decimals
    st.dataframe(metrics_df_percent.style.format("{:.2f}"))


    # Bar chart for accuracy
    st.subheader(" Accuracy Comparison (Bar Chart)")
    st.bar_chart(metrics_df_percent['Accuracy'])

