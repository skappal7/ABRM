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

# Rest of the functions (load_data, preprocess_data, train_model, evaluate_model, visualize_data, data_upload_page, model_training_page) remain the same

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
