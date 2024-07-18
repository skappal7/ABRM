import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    data.fillna(data.mean(), inplace=True)

    data['High AHT Indicator'] = (data['Average of AHT (seconds)'] > data['Average of AHT (seconds)'].mean()).astype(int)
    data['Low Attendance Indicator'] = (data['Average of Attendance'] < data['Average of Attendance'].mean()).astype(int)
    data['Performance Score'] = (data['Average of CSAT (%)'] * data['Average of Attendance']) / 100

    data['Risk Indicator'] = data['Risk Indicator'].map({'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2})

    X = data.drop(columns=['Agent ID', 'Risk Indicator'])
    y = data['Risk Indicator']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
