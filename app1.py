import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Agent Burnout Risk Classification App", layout="wide")

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def preprocess_data(df):
    if 'Risk Indicator' not in df.columns:
        st.error("The dataset must contain a 'Risk Indicator' column.")
        return None, None

    X = df.drop(['Risk Indicator', 'Agent ID', 'Attrition Flag'], axis=1)
    y = df['Risk Indicator']
    X = pd.get_dummies(X)

    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    st.pyplot(plt)

    # ROC Curve (for multi-class, we'll use one-vs-rest)
    n_classes = len(model.classes_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == model.classes_[i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label=f'ROC curve (AUC = {roc_auc[i]:.2f}) for {model.classes_[i]}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    st.pyplot(plt)

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    st.pyplot(plt)

def visualize_data(df):
    st.subheader("Data Visualization")

    # Correlation heatmap
    corr = df.drop(['Agent ID', 'Risk Indicator'], axis=1).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Risk Indicator', data=df)
    plt.title('Distribution of Risk Indicator')
    st.pyplot(plt)

    # Feature distributions
    num_cols = ['Average of AHT (seconds)', 'Average of Attendance', 'Average of CSAT (%)']
    for col in num_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue='Risk Indicator', kde=True)
        plt.title(f'Distribution of {col} by Risk Indicator')
        st.pyplot(plt)

def data_upload_page():
    st.subheader("Data Upload")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.write("Data loaded successfully!")
            st.write(df.head())
            st.session_state['df'] = df
            visualize_data(df)

def model_training_page():
    st.subheader("Model Training")
    if 'df' not in st.session_state:
        st.warning("Please upload data first.")
        return

    df = st.session_state['df']
    X, y = preprocess_data(df)

    if X is not None and y is not None:
        if st.button("Train Model"):
            model, X_test, y_test = train_model(X, y)
            st.write("Model trained successfully!")
            st.session_state['model'] = model
            st.session_state['X'] = X
            evaluate_model(model, X_test, y_test)

def predictions_page():
    st.subheader("Make Predictions")
    if 'model' not in st.session_state or 'X' not in st.session_state:
        st.warning("Please train the model first.")
        return

    model = st.session_state['model']
    X = st.session_state['X']

    input_data = {}
    for column in X.columns:
        input_data[column] = st.number_input(f"Enter {column}", value=0.0)

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0]

        st.write(f"Predicted Risk Indicator: {prediction[0]}")
        st.write(f"Probability of Risk Levels:")
        for risk_level, prob in zip(model.classes_, probability):
            st.write(f"{risk_level}: {prob:.2f}")

def main():
    st.title("Agent Burnout Risk Classification App")

    st.write("""
    This app predicts the burnout risk for agents based on various features.
    Upload your data, train the model, and make predictions!
    """)

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Upload", "Model Training", "Predictions"])

    if page == "Data Upload":
        data_upload_page()
    elif page == "Model Training":
        model_training_page()
    elif page == "Predictions":
        predictions_page()

if __name__ == "__main__":
    main()
