import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    
    # Convert necessary columns to numeric types
    data['Average of AHT (seconds)'] = pd.to_numeric(data['Average of AHT (seconds)'], errors='coerce')
    data['Average of Attendance'] = pd.to_numeric(data['Average of Attendance'], errors='coerce')
    data['Average of CSAT (%)'] = pd.to_numeric(data['Average of CSAT (%)'], errors='coerce')
    data['Attrition Flag'] = pd.to_numeric(data['Attrition Flag'], errors='coerce')
    
    data.fillna(data.mean(), inplace=True)
    
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
