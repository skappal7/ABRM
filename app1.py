import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import BytesIO
from PIL import Image

# Set page config
st.set_page_config(page_title="Agent Burnout Risk Classification App", layout="wide")

# Function to load image from URL
def load_image_from_url(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))

# Background image URL
bg_image_url = "https://www.flatworldsolutions.com/call-center/images/what-are-the-top-10-qualities-a-call-center-agent-should-possess.jpg"

# Logo URL
logo_url = "https://humach.com/wp-content/uploads/2023/01/HuMach_logo-bold.png"

# Function to create rounded rectangle with shadow
def rounded_rectangle(color, text, value):
    return f"""
    <div style="
        background-color: {color};
        padding: 20px;
        border-radius: 10px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
        margin: 10px;
        text-align: center;
    ">
        <h3 style="color: white;">{text}</h3>
        <h2 style="color: white;">{value}</h2>
    </div>
    """

# Login function
def login():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{bg_image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            height: 100vh;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown(
            """
            <div style="
                background-color: rgba(255,255,255,0.7);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
            ">
            <h2 style="text-align: center;">Agent Burnout Risk Management Application</h2>
            """, 
            unsafe_allow_html=True
        )
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == "humach" and password == "password":
                st.session_state['logged_in'] = True
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 0.8em;'>Please login using provided credentials</p>", unsafe_allow_html=True)

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
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
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
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right")
    st.pyplot(plt)

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importances', fontsize=16)
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    st.pyplot(plt)

def visualize_data(df):
    st.subheader("Data Visualization")

    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(rounded_rectangle("#1f77b4", "Agents Count", len(df['Agent ID'].unique())), unsafe_allow_html=True)
    with col2:
        risk_percentages = df['Risk Indicator'].value_counts(normalize=True) * 100
        st.markdown(rounded_rectangle("#ff7f0e", "High Risk %", f"{risk_percentages.get('High Risk', 0):.1f}%"), unsafe_allow_html=True)
    with col3:
        st.markdown(rounded_rectangle("#2ca02c", "Avg AHT", f"{df['Average of AHT (seconds)'].mean():.0f}s"), unsafe_allow_html=True)
    with col4:
        st.markdown(rounded_rectangle("#d62728", "Avg CSAT", f"{df['Average of CSAT (%)'].mean():.1f}%"), unsafe_allow_html=True)
    with col5:
        st.markdown(rounded_rectangle("#9467bd", "Avg Attendance", f"{df['Average of Attendance'].mean():.1f}%"), unsafe_allow_html=True)

    # Correlation heatmap
    corr = df.drop(['Agent ID', 'Risk Indicator'], axis=1).corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap', fontsize=16)
    st.pyplot(plt)

    # Distribution of target variable
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Risk Indicator', data=df, palette='viridis')
    plt.title('Distribution of Risk Indicator', fontsize=16)
    plt.xlabel('Risk Indicator', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    st.pyplot(plt)

    # Feature distributions
    num_cols = ['Average of AHT (seconds)', 'Average of Attendance', 'Average of CSAT (%)']
    for col in num_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, hue='Risk Indicator', kde=True, palette='viridis')
        plt.title(f'Distribution of {col} by Risk Indicator', fontsize=16)
        plt.xlabel(col, fontsize=12)
        plt.ylabel('Count', fontsize=12)
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
    
    prediction_method = st.radio("Choose prediction method:", ["Manual Input", "File Upload"])
    
    if prediction_method == "Manual Input":
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
    
    else:  # File Upload
        uploaded_file = st.file_uploader("Choose a CSV or Excel file for predictions", type=["csv", "xlsx"])
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.write("Data loaded successfully!")
                st.write(df.head())
                
                if 'model' not in st.session_state:
                    st.warning("Please train the model first.")
                    return
                
                model = st.session_state['model']
                
                # Prepare the data for prediction
                X_pred = df.drop(['Agent ID'], axis=1, errors='ignore')
                X_pred = pd.get_dummies(X_pred)
                
                # Align the columns with the training data
                X_train_columns = st.session_state['X'].columns
                X_pred = X_pred.reindex(columns=X_train_columns, fill_value=0)
                
                # Make predictions
                predictions = model.predict(X_pred)
                probabilities = model.predict_proba(X_pred)
                
                # Add predictions to the dataframe
                df['Predicted Risk Indicator'] = predictions
                for i, class_name in enumerate(model.classes_):
                    df[f'Probability_{class_name}'] = probabilities[:, i]
                
                st.write("Predictions:")
                st.write(df)
                
                # Download predictions
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download predictions as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

def main():
    # Add logo
    logo = load_image_from_url(logo_url)
    st.sidebar.image(logo, width=200)

    if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
        login()
    else:
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
