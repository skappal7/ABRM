import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    # Load the data
    data = pd.read_excel(file_path)

    # Create derived features
    data['High AHT Indicator'] = (data['Average of AHT (seconds)'] > data['Average of AHT (seconds)'].mean()).astype(int)
    data['Low Attendance Indicator'] = (data['Average of Attendance'] < data['Average of Attendance'].mean()).astype(int)
    data['Performance Score'] = (data['Average of CSAT (%)'] * data['Average of Attendance']) / 100

    # Encode the target variable
    data['Risk Indicator'] = data['Risk Indicator'].map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2})

    # Define features and target
    X = data.drop(columns=['Agent ID', 'Risk Indicator'])
    y = data['Risk Indicator']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
