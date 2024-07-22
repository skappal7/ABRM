import os
import pandas as pd
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

FILE_PATH = 'ABRMData.csv'
MODEL_NAMES = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'Gradient Boosting']

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Convert necessary columns to numeric types
    data['Average of AHT (seconds)'] = pd.to_numeric(data['Average of AHT (seconds)'], errors='coerce')
    data['Average of Attendance'] = pd.to_numeric(data['Average of Attendance'], errors='coerce')
    data['Average of CSAT (%)'] = pd.to_numeric(data['Average of CSAT (%)'], errors='coerce')
    data['Attrition Flag'] = pd.to_numeric(data['Attrition Flag'], errors='coerce')
    
    data.fillna(data.mean(), inplace=True)
    
    # Convert 'Risk Indicator' to categorical
    data['Risk Indicator'] = data['Risk Indicator'].astype('category')
    
    # One-hot encode the 'Risk Indicator' column
    encoder = OneHotEncoder()
    risk_indicator_encoded = encoder.fit_transform(data[['Risk Indicator']]).toarray()
    risk_indicator_df = pd.DataFrame(risk_indicator_encoded, columns=encoder.get_feature_names_out(['Risk Indicator']))
    
    # Concatenate the one-hot encoded columns back to the original dataframe
    data = pd.concat([data, risk_indicator_df], axis=1)
    
    # Drop the original 'Risk Indicator' and any other non-feature columns
    X = data.drop(columns=['Agent ID', 'Risk Indicator'])
    y = data['Risk Indicator']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_models(X, y):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier()
    }
    
    for model_name, model in models.items():
        model.fit(X, y)
        joblib.dump(model, f'{model_name}.joblib')

    return models

def main():
    st.title("Agent Burnout Prediction")
    
    # Navigation sidebar
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Introduction", "Data Visualization", "Data Upload", "Data Preparation", "Model Training", "Prediction", "Database"]
    )

    if page == "Introduction":
        st.markdown("""
        # Agent Burnout Prediction

        Burnout risk is a significant issue in workplaces. This application helps predict the risk of burnout for agents based on historical data and various metrics.
        """)

    elif page == "Data Visualization":
        st.markdown("""
        ## Data Visualization

        Visualize the dataset with scatter plots and histograms.
        """)
        if os.path.exists(FILE_PATH):
            data = pd.read_csv(FILE_PATH)
            st.write(data.head())
            if st.button('Show Scatter Plot'):
                try:
                    st.scatter_chart(data, x='Average of AHT (seconds)', y='Average of Attendance')
                except Exception as e:
                    st.error(f"Error creating scatter plot: {e}")

    elif page == "Data Upload":
        st.markdown("""
        ## Data Upload

        Upload your dataset here.
        """)
        uploaded_file = st.file_uploader("Choose a file", type=['csv'])
        if uploaded_file is not None:
            file_path = os.path.join(os.getcwd(), FILE_PATH)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.success("File uploaded successfully")
            st.session_state.file_uploaded = True

    elif page == "Data Preparation":
        st.markdown("""
        ## Data Preparation

        Preprocess and prepare your data for model training.
        """)
        if 'file_uploaded' in st.session_state and st.session_state.file_uploaded:
            if st.button("Prepare Data"):
                try:
                    X, y, scaler = preprocess_data(FILE_PATH)
                    joblib.dump(scaler, 'scaler.joblib')
                    st.success("Data prepared successfully.")
                except Exception as e:
                    st.error(f"Error during data preparation: {e}")

    elif page == "Model Training":
        st.markdown("""
        ## Model Training

        Train machine learning models and evaluate their performance.
        """)
        if 'file_uploaded' in st.session_state and st.session_state.file_uploaded:
            if st.button("Train Models"):
                try:
                    X, y, scaler = preprocess_data(FILE_PATH)
                    models = train_models(X, y)
                    for model_name in MODEL_NAMES:
                        model = joblib.load(f'{model_name}.joblib')
                        y_pred = model.predict(X)
                        accuracy = accuracy_score(y, y_pred)
                        report = classification_report(y, y_pred, target_names=['Low Risk', 'Medium Risk', 'High Risk'])
                        st.write(f"{model_name} model accuracy: {accuracy}")
                        st.text(report)
                except Exception as e:
                    st.error(f"Error during model training: {e}")

    elif page == "Prediction":
        st.markdown("""
        ## Prediction

        Input new data to predict burnout risk.
        """)
        aht = st.text_input("Average of AHT (seconds)")
        attendance = st.text_input("Average of Attendance")
        csat = st.text_input("Average of CSAT (%)")
        attrition = st.text_input("Attrition Flag")
        
        if st.button("Predict"):
            try:
                input_data = {
                    "Average of AHT (seconds)": aht,
                    "Average of Attendance": attendance,
                    "Average of CSAT (%)": csat,
                    "Attrition Flag": attrition
                }
                input_df = pd.DataFrame([input_data])
                scaler = joblib.load('scaler.joblib')
                input_scaled = scaler.transform(input_df)
                predictions = {}
                for model_name in MODEL_NAMES:
                    model = joblib.load(f'{model_name}.joblib')
                    prediction = model.predict(input_scaled)[0]
                    predictions[model_name] = ['Low Risk', 'Medium Risk', 'High Risk'][prediction]
                st.write("Predictions:")
                for model_name, prediction in predictions.items():
                    st.write(f"{model_name}: {prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    elif page == "Database":
        st.markdown("""
        ## Database

        View and download predictions and probabilities.
        """)
        # Database related functionalities can be implemented here

if __name__ == "__main__":
    main()
