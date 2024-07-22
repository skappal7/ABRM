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
def rounded_rectangle(color, value):
    return f"""
    <div style="
        background-color: {color};
        padding: 10px;
        border-radius: 10px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.2);
        margin: 10px;
        text-align: center;
        height: 60px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    ">
        <p style="color: white; margin: 0; font-size: 20px; font-weight: bold;">{value}</p>
    </div>
    """

# Login function (remains the same)

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

def preprocess_data(df, selected_features):
    if 'Risk Indicator' not in df.columns:
        st.error("The dataset must contain a 'Risk Indicator' column.")
        return None, None

    X = df[selected_features]
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
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    st.pyplot(fig)

    # ROC Curve
    n_classes = len(model.classes_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test == model.classes_[i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots(figsize=(8, 6))
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], lw=2, label=f'ROC curve (AUC = {roc_auc[i]:.2f}) for {model.classes_[i]}')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    ax.legend(loc="lower right")
    st.pyplot(fig)

    # Feature Importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(10), ax=ax)
    ax.set_title('Top 10 Feature Importances', fontsize=14)
    ax.set_xlabel('Importance', fontsize=10)
    ax.set_ylabel('Feature', fontsize=10)
    st.pyplot(fig)

def visualize_data(df):
    st.subheader("Data Visualization")

    # Key Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    metrics = [
        ("Agents Count", len(df['Agent ID'].unique()), col1),
        ("High Risk %", f"{df['Risk Indicator'].value_counts(normalize=True).get('High Risk', 0)*100:.1f}%", col2),
        ("Avg AHT", f"{df['Average of AHT (seconds)'].mean():.0f}s", col3),
        ("Avg CSAT", f"{df['Average of CSAT (%)'].mean():.1f}%", col4),
        ("Avg Attendance", f"{df['Average of Attendance'].mean():.1f}%", col5)
    ]

    for i, (name, value, col) in enumerate(metrics):
        with col:
            st.markdown(f"<h5 style='text-align: center;'>{name}</h6>", unsafe_allow_html=True)
            st.markdown(rounded_rectangle(f"rgba(0, 100, 200, {0.5 + i*0.1})", value), unsafe_allow_html=True)

    # Charts
    chart_types = ["Correlation Heatmap", "Risk Distribution", "Feature Distributions"]
    selected_chart = st.selectbox("Select chart to view", chart_types)

    if selected_chart == "Correlation Heatmap":
        corr = df.drop(['Agent ID', 'Risk Indicator'], axis=1).corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=14)
        st.pyplot(fig)

    elif selected_chart == "Risk Distribution":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.countplot(x='Risk Indicator', data=df, palette='viridis', ax=ax)
        ax.set_title('Distribution of Risk Indicator', fontsize=14)
        ax.set_xlabel('Risk Indicator', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        st.pyplot(fig)

    elif selected_chart == "Feature Distributions":
        num_cols = ['Average of AHT (seconds)', 'Average of Attendance', 'Average of CSAT (%)']
        selected_feature = st.selectbox("Select feature", num_cols)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data=df, x=selected_feature, hue='Risk Indicator', kde=True, palette='viridis', ax=ax)
        ax.set_title(f'Distribution of {selected_feature} by Risk Indicator', fontsize=14)
        ax.set_xlabel(selected_feature, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        st.pyplot(fig)

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
    
    # Allow user to select features
    features = df.columns.tolist()
    features.remove('Risk Indicator')
    features.remove('Agent ID')
    features.remove('Attrition Flag')
    selected_features = st.multiselect("Select features for training", features, default=features)

    X, y = preprocess_data(df, selected_features)

    if X is not None and y is not None:
        if st.button("Train Model"):
            model, X_test, y_test = train_model(X, y)
            st.write("Model trained successfully!")
            st.session_state['model'] = model
            st.session_state['X'] = X
            evaluate_model(model, X_test, y_test)

# predictions_page function remains the same
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
